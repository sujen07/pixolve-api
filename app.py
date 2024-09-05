from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import io
from jose import JWTError, jwt
from mangum import Mangum
import numpy as np
import onnxruntime as ort
import os
from PIL import Image
from typing import Dict
import zipfile
import cluster
import scoring
import time
import merge
from rate_limiter import rate_limiter


load_dotenv()

app = FastAPI()

origins = [
    "http://localhost:3000",
    "https://testing.pixolve.app",
    "https://www.pixolve.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

handler = Mangum(app)
security = HTTPBearer()

CLERK_JWT_PUBLIC_KEY = """
-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA34FvFwT6JAsy0aNLAMm8
IoZtC7HfrRLer7psvQS6+GoZ1xE/ElZacGX8Vjd5PcrknCXI8nNneD6QwiSr1et9
/VuKMPPoIiwKM22wqXmfn6CnqpdpUAx8ObBR7xTf+MbsSZb/AQoty2C2hww4Ivfa
tTLJivZVsUiH/TVu7024x40Qt8GGyyptg9jIUOFRonvy3qx95av/zZwipAZ8H53y
GgPUQAxcKfplQaro6jps5pFB/CzoAWdl+lzAqTxS35kY3zT55iYPG9V8j8YdS4+b
YRgpwf00KS1cnYNiKx+g8D3bWX3lJrCekwDsClZ+aLsfsSAkKIEUx2zzS0xCMkOO
SQIDAQAB
-----END PUBLIC KEY-----
"""

model_path = './model.onnx'

def load_model(model_path):
    session = ort.InferenceSession(model_path)
    return session

def prepare_image(img):
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0).transpose(0, 3, 1, 2)
    return arr

def run_inference(session, input_tensor):
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: input_tensor})
    return output[0]

def image_to_bytes(arr):
    image = Image.fromarray(arr)
    byte_arr = io.BytesIO()
    image.save(byte_arr, format='PNG')
    byte_arr.seek(0)
    return byte_arr

def score_each_cluster(clusters):
    for cluster in clusters:
        if cluster == -1:
            scores = {filename:0 for filename in clusters[cluster]}
        else:
            scores = scoring.main(clusters[cluster])
        clusters[cluster] = scores
    return clusters


def verify_jwt(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict:
    if os.getenv("ENVIRONMENT") == "development":
        return {"sub": "dev_user"}
    else:
        try:
            token = credentials.credentials
            payload = jwt.decode(token, CLERK_JWT_PUBLIC_KEY, algorithms=["RS256"])
            return payload
        except JWTError:
            raise HTTPException(status_code=401, detail="Invalid token")

def rate_limit(payload: Dict = Depends(verify_jwt)):
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID not found in token")
    rate_limiter.check_rate_limit(user_id)
    return payload

@app.get("/")
def read_root():
   return {"Welcome to": "My first FastAPI depolyment using Docker image"}

@app.post("/enhance")
async def predict(
    file: UploadFile = File(...),
    payload: Dict = Depends(rate_limit)
):
    try:        
        img = Image.open(file.file).convert('RGB')
        input_tensor = prepare_image(img)
        
        model_session = load_model(model_path)
        output_tensor = run_inference(model_session, input_tensor)
        del model_session

        arr = np.clip(output_tensor.squeeze(0), 0, 1).transpose(1, 2, 0)
        arr = (arr * 255).astype(np.uint8)
        output_image = image_to_bytes(arr)

        return StreamingResponse(output_image, media_type="image/png")
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/cluster")
async def cluster_post(
    file: UploadFile = File(...),
    payload: Dict = Depends(rate_limit)
):
    
    file_content = await file.read()
    
    # Create a BytesIO object
    zip_bytes = io.BytesIO(file_content)
    
    # Create a temporary directory to extract files
    temp_dir = "temp_extracted"
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        with zipfile.ZipFile(zip_bytes, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        
        clusters = cluster.main(temp_dir)
        clusters = score_each_cluster(clusters)

        return clusters

    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid ZIP file")

    finally:
        # Clean up: remove the temporary directory and its contents
        for root, dirs, files in os.walk(temp_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(temp_dir)

@app.post("/merge")
async def cluster_post(
    file: UploadFile = File(...),
    payload: Dict = Depends(rate_limit)
):
    
    file_content = await file.read()
    
    # Create a BytesIO object
    zip_bytes = io.BytesIO(file_content)
    
    # Create a temporary directory to extract files
    temp_dir = "temp_extracted"
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        with zipfile.ZipFile(zip_bytes, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        
        final_image = merge.main(temp_dir)
        if not isinstance(final_image, np.ndarray):
            raise HTTPException(status_code=500, detail="Images are not similar enough to merge!")

        output_image = image_to_bytes(final_image)
        return StreamingResponse(output_image, media_type="image/png")

    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid ZIP file")

    finally:
        # Clean up: remove the temporary directory and its contents
        for root, dirs, files in os.walk(temp_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(temp_dir)