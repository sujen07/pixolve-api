import requests
import bz2
import os
import dlib

def get_raw_github_content_url(github_url):
    parts = github_url.split('/')
    if 'blob' in parts:
        parts[parts.index('blob')] = 'raw'
    return '/'.join(parts)

def download_file(url, filename):
    if os.path.exists(filename):
        os.remove(filename)
        print(f"Removed existing {filename}")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(filename, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    print(f"Downloaded {filename}")

def decompress_bz2(file_path):
    if os.path.exists(file_path[:-4]):
        os.remove(file_path[:-4])
        print(f"Removed existing {file_path[:-4]}")
    
    with open(file_path, 'rb') as file:
        data = bz2.decompress(file.read())
    with open(file_path[:-4], 'wb') as file:
        file.write(data)
    print(f"Decompressed {file_path}")

# URLs of the files to download
github_url = "https://github.com/onnx/models/blob/main/validated/vision/classification/resnet/model/resnet50-v2-7.onnx"
raw_url = get_raw_github_content_url(github_url)

urls = {
    'dlib_face_recognition_resnet_model_v1.dat.bz2': 'http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2',
    'shape_predictor_68_face_landmarks.dat.bz2': 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2',
    'resnet50.onnx': raw_url
}

for filename, url in urls.items():
    download_file(url, filename)
    if filename.endswith('.bz2'):
        decompress_bz2(filename)

# Test if the model loads correctly (you can comment this part out if not needed)
try:
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    print("Model loaded successfully.")
except RuntimeError as e:
    print(f"Failed to load the model: {e}")
