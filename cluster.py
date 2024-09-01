import os
import numpy as np
import onnxruntime as ort
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial.distance import pdist, squareform
#from sklearn.manifold import TSNE
#import matplotlib.pyplot as plt
#from scipy.spatial.distance import euclidean

def load_model():
    onnx_model_path = 'resnet50.onnx'
    ort_session = ort.InferenceSession(onnx_model_path)
    return ort_session

def preprocess_image(image):
    image_np = np.array(image).astype(np.float32)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_np = (image_np / 255.0 - mean) / std
    image_np = np.transpose(image_np, (2, 0, 1))
    image_np = np.expand_dims(image_np, axis=0).astype(np.float32)
    return image_np

def load_and_preprocess_image(file_path, target_size=(224, 224)):
    try:
        with Image.open(file_path) as img:
            img = img.convert('RGB')
            img = img.resize(target_size, Image.LANCZOS)
            image_np = preprocess_image(img)
            return file_path, image_np
    except Exception as e:
        print(f"Error loading image {file_path}: {e}")
        return None

def extract_features(ort_session, images):
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    features = []
    for img in images:
        ort_inputs = {input_name: img}
        ort_outs = ort_session.run([output_name], ort_inputs)
        features.append(ort_outs[0].flatten())
    return np.array(features)

def get_images_from_folder(folder_path, num_threads=4):
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(load_and_preprocess_image, image_files))

    valid_results = [r for r in results if r is not None]
    image_paths, images = zip(*valid_results) if valid_results else ([], [])

    return list(image_paths), list(images)

def adaptive_dbscan(features, min_samples=2):
    # Calculate pairwise distances
    distances = pdist(features)
    
    # Calculate the average distance to the k-nearest neighbors
    k = min(min_samples, len(features) - 1)
    sorted_distances = np.sort(squareform(distances), axis=1)
    eps = np.mean(sorted_distances[:, k])
    print(eps)
    
    # Run DBSCAN with the adaptive eps
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(features)
    
    return clusters


def main(folder_path):
    ort_session = load_model()
    scaler = StandardScaler()

    paths, images = get_images_from_folder(folder_path)
    if len(images) == 0:
        raise Exception("No images found in folder")

    features = extract_features(ort_session, images)
    #features = scaler.fit_transform(features)

    # Use adaptive DBSCAN
    clusters = adaptive_dbscan(features)

    # Group images by clusters
    cluster_images = {int(i): [] for i in set(clusters)}
    for idx, cluster in enumerate(clusters):
        cluster_images[cluster].append(paths[idx])


    
    return cluster_images

if __name__ == "__main__":
    folder_path = 'test'
    cluster_imgs = main(folder_path)
    print(cluster_imgs)