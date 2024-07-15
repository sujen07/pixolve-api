from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import onnxruntime as ort
from PIL import Image
import io
import numpy as np
from mangum import Mangum

app = FastAPI()
handler = Mangum(app)


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

model_path = './model.onnx'

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img = Image.open(file.file).convert('RGB')
        input_tensor = prepare_image(img)
        
        # Load model, run inference, and unload model
        model_session = load_model(model_path)
        output_tensor = run_inference(model_session, input_tensor)
        del model_session

        output_image = image_to_bytes(output_tensor)
        return StreamingResponse(output_image, media_type="image/png")
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
