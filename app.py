import os
import shutil
import uuid
from pathlib import Path
from typing import Optional
import aiohttp
import asyncio

import torch
from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, HttpUrl
from omegaconf import OmegaConf
from diffusers import AutoencoderKL, DDIMScheduler
from accelerate.utils import set_seed

from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from latentsync.whisper.audio2feature import Audio2Feature

app = FastAPI(title="Audio-Driven Avatar Generation API")

# Configuration
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
CONFIG_PATH = "configs/unet/stage2.yaml"
INFERENCE_CKPT_PATH = "checkpoints/latentsync_unet.pt"
INFERENCE_STEPS = 20
GUIDANCE_SCALE = 1.0
SEED = 1247

# Create directories if they don't exist
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Load config once at startup
config = OmegaConf.load(CONFIG_PATH)

# Request model for URL-based inputs
class GenerationRequest(BaseModel):
    video_url: str  # Changed from HttpUrl to str
    audio_url: str  # Changed from HttpUrl to str
    inference_steps: Optional[int] = INFERENCE_STEPS
    guidance_scale: Optional[float] = GUIDANCE_SCALE
    seed: Optional[int] = SEED

# Initialize model components at startup
@app.on_event("startup")
async def startup_event():
    global pipeline
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Using CPU instead.")
        device = "cpu"
    else:
        device = "cuda"
        
    # Check if the GPU supports float16
    is_fp16_supported = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
    dtype = torch.float16 if is_fp16_supported else torch.float32
    
    # Initialize scheduler
    scheduler = DDIMScheduler.from_pretrained("configs")
    
    # Select whisper model based on config
    if config.model.cross_attention_dim == 768:
        whisper_model_path = "checkpoints/whisper/small.pt"
    elif config.model.cross_attention_dim == 384:
        whisper_model_path = "checkpoints/whisper/tiny.pt"
    else:
        raise ValueError("cross_attention_dim must be 768 or 384")
    
    # Initialize audio encoder
    audio_encoder = Audio2Feature(
        model_path=whisper_model_path,
        device=device,
        num_frames=config.data.num_frames,
        audio_feat_length=config.data.audio_feat_length,
    )
    
    # Initialize VAE
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dtype)
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0
    
    # Initialize UNet
    denoising_unet, *_ = UNet3DConditionModel.from_pretrained(
        OmegaConf.to_container(config.model),
        INFERENCE_CKPT_PATH,
        device="cpu",
    )
    denoising_unet = denoising_unet.to(dtype=dtype)
    
    # Initialize pipeline
    pipeline = LipsyncPipeline(
        vae=vae,
        audio_encoder=audio_encoder,
        denoising_unet=denoising_unet,
        scheduler=scheduler,
    ).to(device)
    
    print("Model initialized successfully!")

def cleanup_files(file_paths):
    """Delete temporary files after processing"""
    for file_path in file_paths:
        if os.path.exists(file_path):
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)
            else:
                os.remove(file_path)
    print(f"Cleaned up temporary files: {file_paths}")

async def download_file(url, destination):
    """Download a file from a URL to a destination path"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    raise HTTPException(
                        status_code=response.status,
                        detail=f"Failed to download file from {url}. Status: {response.status}"
                    )
                
                with open(destination, 'wb') as f:
                    while True:
                        chunk = await response.content.read(1024)
                        if not chunk:
                            break
                        f.write(chunk)
                
                return destination
    except Exception as e:
        raise Exception(f"Failed to download file: {str(e)}")

@app.post("/generate/")
async def generate_avatar(
    request: GenerationRequest,
    background_tasks: BackgroundTasks
):
    """
    Generate an audio-driven avatar video from input video and audio URLs
    
    - **video_url**: URL to the input video file
    - **audio_url**: URL to the input audio file
    - **inference_steps**: Number of inference steps (default: 20)
    - **guidance_scale**: Guidance scale (default: 1.0)
    - **seed**: Random seed (default: 1247)
    
    Returns:
        FileResponse containing the generated video
    """
    # Generate a unique ID for this request
    request_id = str(uuid.uuid4())
    
    # Create unique directories for this request
    request_upload_dir = UPLOAD_DIR / request_id
    request_output_dir = OUTPUT_DIR / request_id
    
    request_upload_dir.mkdir(exist_ok=True)
    request_output_dir.mkdir(exist_ok=True)
    
    # Define file paths
    video_ext = os.path.splitext(request.video_url.split('/')[-1])[1] or ".mp4"
    audio_ext = os.path.splitext(request.audio_url.split('/')[-1])[1] or ".wav"
    
    video_path = request_upload_dir / f"input_video{video_ext}"
    audio_path = request_upload_dir / f"input_audio{audio_ext}"
    video_out_path = request_output_dir / "output_video.mp4"
    
    try:
        # Download files from URLs
        await download_file(request.video_url, video_path)
        await download_file(request.audio_url, audio_path)
    except Exception as e:
        # Clean up any partially downloaded files
        background_tasks.add_task(cleanup_files, [request_upload_dir, request_output_dir])
        raise HTTPException(status_code=500, detail=f"Error downloading files: {str(e)}")
    
    # Check if files exist
    if not os.path.exists(video_path):
        raise HTTPException(status_code=400, detail=f"Video file could not be downloaded or accessed")
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=400, detail=f"Audio file could not be downloaded or accessed")
    
    # Set seed
    if request.seed != -1:
        set_seed(request.seed)
    else:
        torch.seed()
    
    # Get data type
    is_fp16_supported = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
    dtype = torch.float16 if is_fp16_supported else torch.float32
    
    try:
        # Run inference
        pipeline(
            video_path=str(video_path),
            audio_path=str(audio_path),
            video_out_path=str(video_out_path),
            video_mask_path=str(video_out_path).replace(".mp4", "_mask.mp4"),
            num_frames=config.data.num_frames,
            num_inference_steps=request.inference_steps,
            guidance_scale=request.guidance_scale,
            weight_dtype=dtype,
            width=config.data.resolution,
            height=config.data.resolution,
            mask_image_path=config.data.mask_image_path,
        )
    except Exception as e:
        # Schedule cleanup of temporary files
        background_tasks.add_task(cleanup_files, [request_upload_dir, request_output_dir])
        raise HTTPException(status_code=500, detail=f"Error during inference: {str(e)}")
    
    # Check if output file exists
    if not os.path.exists(video_out_path):
        # Schedule cleanup of temporary files
        background_tasks.add_task(cleanup_files, [request_upload_dir, request_output_dir])
        raise HTTPException(status_code=500, detail="Failed to generate output video")
    
    # Schedule cleanup of upload files (but not output files yet)
    background_tasks.add_task(cleanup_files, [request_upload_dir])
    
    # Return the video file
    return FileResponse(
        path=video_out_path,
        filename="generated_avatar.mp4",
        media_type="video/mp4",
        background=background_tasks.add_task(cleanup_files, [request_output_dir])
    )

@app.post("/generate-from-files/")
async def generate_avatar_from_files(
    background_tasks: BackgroundTasks,
    video_file: UploadFile = File(...),
    audio_file: UploadFile = File(...),
    inference_steps: Optional[int] = INFERENCE_STEPS,
    guidance_scale: Optional[float] = GUIDANCE_SCALE,
    seed: Optional[int] = SEED
):
    """Legacy endpoint supporting direct file uploads"""
    # Generate a unique ID for this request
    request_id = str(uuid.uuid4())
    
    # Create unique directories for this request
    request_upload_dir = UPLOAD_DIR / request_id
    request_output_dir = OUTPUT_DIR / request_id
    
    request_upload_dir.mkdir(exist_ok=True)
    request_output_dir.mkdir(exist_ok=True)
    
    # Save uploaded files
    video_path = request_upload_dir / f"input_video{os.path.splitext(video_file.filename)[1]}"
    audio_path = request_upload_dir / f"input_audio{os.path.splitext(audio_file.filename)[1]}"
    video_out_path = request_output_dir / "output_video.mp4"
    
    # Save the uploaded files
    try:
        with open(video_path, "wb") as f:
            shutil.copyfileobj(video_file.file, f)
        video_file.file.close()
        
        with open(audio_path, "wb") as f:
            shutil.copyfileobj(audio_file.file, f)
        audio_file.file.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving uploaded files: {str(e)}")
    
    # Rest of the implementation is the same as in the original code
    # Check if files exist
    if not os.path.exists(video_path):
        raise HTTPException(status_code=400, detail=f"Video file could not be saved or accessed")
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=400, detail=f"Audio file could not be saved or accessed")
    
    # Set seed
    if seed != -1:
        set_seed(seed)
    else:
        torch.seed()
    
    # Get data type
    is_fp16_supported = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
    dtype = torch.float16 if is_fp16_supported else torch.float32
    
    try:
        # Run inference
        pipeline(
            video_path=str(video_path),
            audio_path=str(audio_path),
            video_out_path=str(video_out_path),
            video_mask_path=str(video_out_path).replace(".mp4", "_mask.mp4"),
            num_frames=config.data.num_frames,
            num_inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            weight_dtype=dtype,
            width=config.data.resolution,
            height=config.data.resolution,
            mask_image_path=config.data.mask_image_path,
        )
    except Exception as e:
        # Schedule cleanup of temporary files
        background_tasks.add_task(cleanup_files, [request_upload_dir, request_output_dir])
        raise HTTPException(status_code=500, detail=f"Error during inference: {str(e)}")
    
    # Check if output file exists
    if not os.path.exists(video_out_path):
        # Schedule cleanup of temporary files
        background_tasks.add_task(cleanup_files, [request_upload_dir, request_output_dir])
        raise HTTPException(status_code=500, detail="Failed to generate output video")
    
    # Schedule cleanup of upload files (but not output files yet)
    background_tasks.add_task(cleanup_files, [request_upload_dir])
    
    # Return the video file
    return FileResponse(
        path=video_out_path,
        filename="generated_avatar.mp4",
        media_type="video/mp4",
        background=background_tasks.add_task(cleanup_files, [request_output_dir])
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7990)
