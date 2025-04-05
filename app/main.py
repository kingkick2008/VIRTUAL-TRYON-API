from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import os
import uuid
import datetime
from app.viton_service import VITONService
from app.config import Settings

app = FastAPI(title="Virtual Try-On API")

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize settings and service
settings = Settings()
viton_service = VITONService()

# Function to remove file after response is sent
def remove_file(path: str):
    if os.path.exists(path):
        os.remove(path)

@app.post("/try-on/")
async def virtual_try_on(
    background_tasks: BackgroundTasks,
    person_image: UploadFile = File(...),
    cloth_image: UploadFile = File(...),
):
    """
    Process virtual try-on request with person and clothing images
    """
    output_path = None
    try:
        # Validate file sizes
        if (await person_image.read()).__len__() > settings.MAX_IMAGE_SIZE:
            raise HTTPException(status_code=400, detail="Person image too large")
        if (await cloth_image.read()).__len__() > settings.MAX_IMAGE_SIZE:
            raise HTTPException(status_code=400, detail="Cloth image too large")
            
        # Reset file streams
        await person_image.seek(0)
        await cloth_image.seek(0)
        
        # Read images
        person_img = Image.open(io.BytesIO(await person_image.read())).convert('RGB')
        cloth_img = Image.open(io.BytesIO(await cloth_image.read())).convert('RGB')
        
        # Process images through VITON-HD
        result = viton_service.process_images(person_img, cloth_img)
        
        # Ensure temp directory exists
        os.makedirs(settings.TEMP_DIR, exist_ok=True)
        
        # Create a permanent results directory for inspection
        results_dir = os.path.join(os.getcwd(), "results")
        os.makedirs(results_dir, exist_ok=True)
        
        # Save result with a unique filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"result_{timestamp}_{uuid.uuid4()}.jpg"
        
        # Save a copy in the temp directory (for response)
        output_path = os.path.join(settings.TEMP_DIR, filename)
        result.save(output_path)
        
        # Save a permanent copy for inspection
        permanent_path = os.path.join(results_dir, filename)
        result.save(permanent_path)
        
        # Schedule file removal for temp file after response is sent
        background_tasks.add_task(remove_file, output_path)
        
        return FileResponse(output_path, media_type="image/jpeg")
        
    except Exception as e:
        # Clean up file if an error occurs
        if output_path and os.path.exists(output_path):
            os.remove(output_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": viton_service.is_ready()
    }