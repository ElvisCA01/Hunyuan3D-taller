from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import torch
from PIL import Image
import time
import os
from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen.pipelines import Hunyuan3DDiTFlowMatchingPipeline
from io import BytesIO
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Load the model at API startup
try:
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        'tencent/Hunyuan3D-2',
        subfolder='hunyuan3d-dit-v2-0',
        variant='fp16'
    )
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    # Continue without failing - we'll handle the error when an API call is made

@app.post("/generate-3d/")
async def generate_3d(file: UploadFile = File(...)):
    # Validate file content type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read file bytes
        file_bytes = await file.read()
        
        if not file_bytes:
            raise HTTPException(status_code=400, detail="File is empty")
        
        # Log file information for debugging
        logger.info(f"Processing file: {file.filename}, size: {len(file_bytes)} bytes, content type: {file.content_type}")
        
        try:
            # Open image with PIL
            image = Image.open(BytesIO(file_bytes))
            logger.info(f"Image opened successfully: {image.format}, mode: {image.mode}, size: {image.size}")
            
            # Convert to RGBA
            image = image.convert("RGBA")
            
        except Exception as e:
            logger.error(f"Failed to open image: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")
        
        # Remove background if necessary
        try:
            rembg = BackgroundRemover()
            image = rembg(image)
            logger.info("Background removed successfully")
        except Exception as e:
            logger.error(f"Failed to remove background: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Background removal failed: {str(e)}")
        
        # Generate 3D model
        try:
            start_time = time.time()
            
            # Check if pipeline is available
            if 'pipeline' not in globals() or pipeline is None:
                raise HTTPException(status_code=500, detail="3D generation model not available")
            
            mesh = pipeline(
                image=image,
                num_inference_steps=50,
                octree_resolution=380,
                num_chunks=20000,
                generator=torch.manual_seed(12345),
                output_type='trimesh'
            )[0]
            
            elapsed = time.time() - start_time
            logger.info(f"Generation completed in {elapsed:.2f} seconds")
            
        except Exception as e:
            logger.error(f"3D generation failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"3D generation failed: {str(e)}")
        
        # Create unique filename
        output_path = f"output_{int(time.time())}.glb"
        
        # Export mesh to GLB file
        try:
            mesh.export(output_path)
            logger.info(f"Mesh exported to {output_path}")
        except Exception as e:
            logger.error(f"Failed to export mesh: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to export 3D model: {str(e)}")
        
        # Return the GLB file
        return FileResponse(
            path=output_path, 
            media_type="model/gltf-binary", 
            filename="result.glb",
            background=None  # Run in the main thread to prevent issues
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log and return any other errors
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")