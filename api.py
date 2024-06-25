from fastapi import FastAPI, Request, HTTPException
import subprocess
import time
import os
import zipfile
import gdown

app = FastAPI()

# Helper function to execute commands
def execute_command(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        return {"output": result.stdout, "error": result.stderr}
    except Exception as e:
        return {"error": str(e)}

# Helper function to download and extract files from Google Drive using gdown
def download_and_extract_gdrive(gdrive_id, target_dir):
    url = f"https://drive.google.com/uc?id={gdrive_id}"
    zip_path = os.path.join(target_dir, "temp.zip")
    
    gdown.download(url, output=zip_path, quiet=False)

    # Extract .pth files from the downloaded zip
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file in zip_ref.namelist():
            if file.endswith('.pth'):
                zip_ref.extract(file, target_dir)
    
    # Remove the downloaded zip file after extraction
    os.remove(zip_path)

# Endpoint to handle Google Drive downloads
@app.post("/download_gdrive")
async def download_gdrive(request: Request):
    data = await request.json()
    gdrive_id = data.get("id")

    if not gdrive_id:
        raise HTTPException(status_code=400, detail="Google Drive ID not provided")

    target_dir = "/app/data/RVC_CLI/logs/weights"

    # Ensure the target directory exists
    os.makedirs(target_dir, exist_ok=True)

    try:
        download_and_extract_gdrive(gdrive_id, target_dir)
        return {"status": "success", "message": "Files downloaded and extracted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Other endpoints...
@app.post("/infer")
async def infer(request: Request):
    command = ["python", "rvc.py", "infer"] + await request.json()
    return execute_command(command)

@app.post("/batch_infer")
async def batch_infer(request: Request):
    command = ["python", "rvc.py", "batch_infer"] + await request.json()
    return execute_command(command)

@app.post("/tts")
async def tts(request: Request):
    command = ["python", "rvc.py", "tts"] + await request.json()
    return execute_command(command)

@app.post("/preprocess")
async def preprocess(request: Request):
    command = ["python", "rvc.py", "preprocess"] + await request.json()
    return execute_command(command)

@app.post("/extract")
async def extract(request: Request):
    command = ["python", "rvc.py", "extract"] + await request.json()
    return execute_command(command)

@app.post("/train")
async def train(request: Request):
    command = ["python", "rvc.py", "train"] + await request.json()
    return execute_command(command)

@app.post("/index")
async def index(request: Request):
    command = ["python", "rvc.py", "index"] + await request.json()
    return execute_command(command)

@app.post("/model_information")
async def model_information(request: Request):
    command = ["python", "rvc.py", "model_information"] + await request.json()
    return execute_command(command)

@app.post("/model_fusion")
async def model_fusion(request: Request):
    command = ["python", "rvc.py", "model_fusion"] + await request.json()
    return execute_command(command)

@app.post("/download")
async def download(request: Request):
    command = ["python", "rvc.py", "download"] + await request.json()
    return execute_command(command)

@app.get("/ping")
async def ping():
    start_time = time.time()
    end_time = time.time()
    latency = end_time - start_time
    return {"ping": "pong", "latency": latency}
