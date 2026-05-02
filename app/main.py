import uvicorn
from fastapi import FastAPI
from app.api.v1.images import router as images_router

app = FastAPI()

app.include_router(images_router, prefix="/api/v1/images", tags=["images"])

def start():
    """Launched with `poetry run start` at root level"""
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
