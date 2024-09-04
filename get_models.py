import os
import requests

# Google Drive file IDs
file_ids = {
    'shape_predictor_68_face_landmarks.dat': '1ezlZygd4SQpQq-N4VgCk7rXA8Sp9L3p0',
    'dlib_face_recognition_resnet_model_v1.dat': '1eM58gYNu1xRmf5bBySQwYO-sWPwc0AE7'
}

def download_file(filename, file_id):
    url = f'https://drive.google.com/uc?export=download&id={file_id}'
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded and saved {filename}")
    else:
        print(f"Failed to download {filename}. Status code: {response.status_code}")

def check_and_replace_files(file_ids):
    for filename, file_id in file_ids.items():
        if os.path.exists(filename):
            print(f"{filename} exists. Replacing...")
            download_file(filename, file_id)
        else:
            print(f"{filename} does not exist. Downloading...")
            download_file(filename, file_id)

if __name__ == "__main__":
    check_and_replace_files(file_ids)

