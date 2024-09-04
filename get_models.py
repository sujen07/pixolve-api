import os
import requests

# URL of the directory containing the files
base_url = 'https://filebrowser-production-a7c5.up.railway.app/files/'

# List of filenames to check and download
filenames = ['shape_predictor_68_face_landmarks.dat', 'dlib_face_recognition_resnet_model_v1.dat']  # replace with your filenames

def download_file(filename):
    url = base_url + filename
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded and replaced {filename}")
    else:
        print(f"Failed to download {filename}. Status code: {response.status_code}")

def check_and_replace_files(filenames):
    for filename in filenames:
        if os.path.exists(filename):
            print(f"{filename} exists. Replacing...")
            download_file(filename)
        else:
            print(f"{filename} does not exist. Downloading...")
            download_file(filename)

if __name__ == "__main__":
    check_and_replace_files(filenames)
