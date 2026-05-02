import uvicorn
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return { "message": "OH HAI" }

@app.post("/")
async def post_image():
    return { "message": "You posted!" }

def start():
    """Launched with `poetry run start` at root level"""
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
