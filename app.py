from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import io
from jose import JWTError, jwt
from mangum import Mangum
import numpy as np
import onnxruntime as ort
from PIL import Image
from typing import Dict

app = FastAPI()
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

def image_to_bytes(image_tensor):
    arr = np.clip(image_tensor.squeeze(0), 0, 1).transpose(1, 2, 0)
    arr = (arr * 255).astype(np.uint8)
    image = Image.fromarray(arr)
    byte_arr = io.BytesIO()
    image.save(byte_arr, format='PNG')
    byte_arr.seek(0)
    return byte_arr

def verify_jwt(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict:
    try:
        token = credentials.credentials
        payload = jwt.decode(token, CLERK_JWT_PUBLIC_KEY, algorithms=["RS256"])
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    payload: Dict = Depends(verify_jwt)
):
    try:        
        img = Image.open(file.file).convert('RGB')
        input_tensor = prepare_image(img)
        
        # Load model, run inference, and unload model
        model_session = load_model(model_path)
        output_tensor = run_inference(model_session, input_tensor)
        del model_session

        output_image = image_to_bytes(output_tensor)
        return StreamingResponse(output_image, media_type="image/png")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
