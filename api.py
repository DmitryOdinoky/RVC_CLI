from fastapi import FastAPI, Request, HTTPException
import subprocess
import time
import gdown
import zipfile
import os
import re

app = FastAPI()

# Helper function to execute commands
def execute_command(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        return {"output": result.stdout, "error": result.stderr}
    except Exception as e:
        return {"error": str(e)}

# Function to download Google Drive folder and extract .pth files
def download_and_extract_gdrive_folder(gdrive_url, extract_to):
    # Extract folder ID from Google Drive URL
    match = re.search(r'/folders/([a-zA-Z0-9_-]+)', gdrive_url)
    if not match:
        raise HTTPException(status_code=400, detail="Invalid Google Drive URL")
    folder_id = match.group(1)

    # Create a temporary zip file path
    zip_path = '/app/data/temp.zip'

    # Download the Google Drive folder as a zip file
    gdown.download(f'https://drive.google.com/uc?id={folder_id}&export=download', zip_path, quiet=False)

    # Extract only .pth files to the specified directory
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file_info in zip_ref.infolist():
            if file_info.filename.endswith('.pth'):
                file_info.filename = os.path.basename(file_info.filename)
                zip_ref.extract(file_info, extract_to)

    # Remove the temporary zip file
    os.remove(zip_path)

    return {"message": "Files extracted successfully"}

# FastAPI endpoint to handle Google Drive download and extraction
@app.post("/download_gdrive")
async def download_gdrive(request: Request):
    data = await request.json()
    gdrive_url = data.get('url')
    if not gdrive_url:
        raise HTTPException(status_code=400, detail="URL is required")
    
    try:
        result = download_and_extract_gdrive_folder(gdrive_url, '/app/data/RVC_CLI/logs/weights')
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Other endpoints...

# Infer
@app.post("/infer")
async def infer(request: Request):
    command = ["python", "rvc.py", "infer"] + await request.json()
    return execute_command(command)

# Batch Infer
@app.post("/batch_infer")
async def batch_infer(request: Request):
    command = ["python", "rvc.py", "batch_infer"] + await request.json()
    return execute_command(command)

# TTS
@app.post("/tts")
async def tts(request: Request):
    command = ["python", "rvc.py", "tts"] + await request.json()
    return execute_command(command)

# Preprocess
@app.post("/preprocess")
async def preprocess(request: Request):
    command = ["python", "rvc.py", "preprocess"] + await request.json()
    return execute_command(command)

# Extract
@app.post("/extract")
async def extract(request: Request):
    command = ["python", "rvc.py", "extract"] + await request.json()
    return execute_command(command)

# Train
@app.post("/train")
async def train(request: Request):
    command = ["python", "rvc.py", "train"] + await request.json()
    return execute_command(command)

# Index
@app.post("/index")
async def index(request: Request):
    command = ["python", "rvc.py", "index"] + await request.json()
    return execute_command(command)

# Model Information
@app.post("/model_information")
async def model_information(request: Request):
    command = ["python", "rvc.py", "model_information"] + await request.json()
    return execute_command(command)

# Model Fusion
@app.post("/model_fusion")
async def model_fusion(request: Request):
    command = ["python", "rvc.py", "model_fusion"] + await request.json()
    return execute_command(command)

# Download
@app.post("/download")
async def download(request: Request):
    command = ["python", "rvc.py", "download"] + await request.json()
    return execute_command(command)

# Ping endpoint to check latency
@app.get("/ping")
async def ping():
    start_time = time.time()
    end_time = time.time()
    latency = end_time - start_time
    return {"ping": "pong", "latency": latency}
