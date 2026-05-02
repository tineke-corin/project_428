from fastapi import APIRouter, UploadFile, File, HTTPException, Response
import cv2
import numpy as np
import time
import logging
from app.services.image_context import ImageContext
from app.services.detect_faces import process_faces
from app.services.detect_vlps import process_plates

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/anonymize")
async def anonymize_image(file: UploadFile = File(...)):
    # Validate file extension
    extension = file.filename.split(".")[-1].lower()
    if extension not in ["png", "jpg", "jpeg"]:
        raise HTTPException(status_code=422, detail="Invalid file extension. Only png, jpg, jpeg are allowed.")

    # Read image and decode
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=422, detail="Invalid image data.")
    
    # Create ImageContext
    ctx = ImageContext(img)
    
    # Anonymize
    start_time = time.perf_counter()
    process_faces(ctx)
    process_plates(ctx)
    duration = time.perf_counter() - start_time

    # Log timing info    
    logger.info(f"Anonymization took {duration:.4f} seconds")
    
    # Convert back to bytes
    out_ext = f".{extension}"
    image_bytes = ctx.to_bytes(extension=out_ext)
    media_type = "image/png" if extension == "png" else "image/jpeg"
    
    return Response(content=image_bytes, media_type=media_type)
