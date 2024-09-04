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
import uvicorn


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
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAw1Z2+EdvvKSei8hD1y5X
rmaXD2+udtdK9JrGveW4g0Qn0C3Qw7b8Co0elHnj2HG/MnMzLJiOIrgSxoZnZtpS
B5rEEJwZCx6p6NeDVtJsPACtTe01L6KPYpFAn0RaBe+gBdPHbAzpxUj8eEtwIWWz
wYiTU3HlSHAFt8hii1Yn42pzuXWM+QYmF/8PzEinkVWrmrZY3usus0asqIv/EtH0
aZUxt6Yhi3us1F5DsY0ZAerxC40tpkGPpPTsoNdpRcVlw51BDFJX+7t528GSgBse
PYGP6Ewvu2G9Z/Wwk96U2sIW+k2K+E3i0xmnaSA7wKpmJN6+EPZkYlNNAS2/v2ds
pQIDAQAB
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

if __name__ == "__main__":
   uvicorn.run(app, host="0.0.0.0", port=8000)