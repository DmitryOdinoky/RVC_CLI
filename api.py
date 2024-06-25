from fastapi import FastAPI, Request, HTTPException
import subprocess
import time
import os
import zipfile
import gdown
import shutil

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
    zip_path = os.path.join(target_dir, f"{gdrive_id}.zip")
    
    gdown.download(url, output=zip_path, quiet=False)

    # Extract .pth files from the downloaded zip
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file in zip_ref.namelist():
            if file.endswith('.pth'):
                zip_ref.extract(file, target_dir)
    
    # Remove the downloaded zip file after extraction
    os.remove(zip_path)

# Combined function to download and extract .wav files from Google Drive using gdown
def download_extract_dataset(gdrive_id, target_dir):
    url = f"https://drive.google.com/uc?id={gdrive_id}"
    zip_path = os.path.join(target_dir, f"{gdrive_id}.zip")
    
    gdown.download(url, output=zip_path, quiet=False)

    # Extract all .wav files from the downloaded zip
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_filename_without_extension = os.path.splitext(os.path.basename(zip_path))[0]
       
        
        # Ensure the target directory is empty before extraction
        if os.path.exists(target_dir):
            for root, dirs, files in os.walk(target_dir):
                for file in files:
                    os.remove(os.path.join(root, file))
                for dir in dirs:
                    os.rmdir(os.path.join(root, dir))
        else:
            os.makedirs(target_dir, exist_ok=True)
        
        # Extract all .wav files from the zip
        for file in zip_ref.namelist():
            if file.endswith('.wav'):
                zip_ref.extract(file, target_dir)
    
    # Remove the downloaded zip file after extraction
    os.remove(zip_path)

# Endpoint to handle removing dataset directory
@app.delete("/remove_dataset")
async def remove_dataset():
    target_dir = "/app/data/dataset"

    if os.path.exists(target_dir):
        try:
            shutil.rmtree(target_dir)
            return {"status": "success", "message": "Dataset directory removed successfully"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    else:
        raise HTTPException(status_code=404, detail="Dataset directory not found")

# Endpoint to handle removing a specific subfolder within the dataset directory
@app.delete("/remove_dataset_subfolder")
async def remove_dataset_subfolder(subfolder_name: str):
    target_dir = "/app/data/dataset"
    subfolder_path = os.path.join(target_dir, subfolder_name)

    if os.path.exists(subfolder_path):
        try:
            shutil.rmtree(subfolder_path)
            return {"status": "success", "message": f"Subfolder '{subfolder_name}' removed successfully"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    else:
        raise HTTPException(status_code=404, detail=f"Subfolder '{subfolder_name}' not found in dataset directory")


# Endpoint to handle downloading and extracting zip file containing .wav files from Google Drive
@app.post("/download_extract_dataset")
async def download_extract_dataset_endpoint(request: Request):
    data = await request.json()
    gdrive_id = data.get("id")

    if not gdrive_id:
        raise HTTPException(status_code=400, detail="Google Drive ID not provided")

    target_dir = "/app/data/dataset"

    # Ensure the target directory exists
    os.makedirs(target_dir, exist_ok=True)

    try:
        download_extract_dataset(gdrive_id, target_dir)
        return {"status": "success", "message": "Files downloaded and extracted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to handle Google Drive downloads
@app.post("/download_gdrive_pths")
async def download_gdrive_pths(request: Request):
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
    data = await request.json()
    command = ["python", "rvc.py", "preprocess"]
    for key, value in data.items():
        command.append(f"--{key}")
        command.append(str(value))
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
