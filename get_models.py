import requests
import bz2
import os

def download_file(url, filename):
    # Check if the file already exists and delete it
    if os.path.exists(filename + '.bz2'):
        os.remove(filename + '.bz2')
        print(f"Removed existing {filename}.bz2")
    
    # Download the file
    response = requests.get(url)
    with open(filename + '.bz2', 'wb') as file:
        file.write(response.content)
    print(f"Downloaded {filename}.bz2")

def decompress_bz2(file_path):
    # Check if the decompressed file already exists and delete it
    if os.path.exists(file_path[:-4]):
        os.remove(file_path[:-4])
        print(f"Removed existing {file_path[:-4]}")
    
    # Decompress the file
    with open(file_path, 'rb') as file:
        data = bz2.decompress(file.read())
    with open(file_path[:-4], 'wb') as file:
        file.write(data)
    print(f"Decompressed {file_path}")

# URLs of the files to download
urls = {
    'dlib_face_recognition_resnet_model_v1.dat': 'http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2',
    'shape_predictor_68_face_landmarks.dat': 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
}

for filename, url in urls.items():
    download_file(url, filename)
    decompress_bz2(filename + '.bz2')
