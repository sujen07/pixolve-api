import os
import numpy as np
import onnxruntime as ort
from PIL import Image
import time
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ThreadPoolExecutor
#from sklearn.manifold import TSNE
#import matplotlib.pyplot as plt
#from scipy.spatial.distance import euclidean

# Constants
EPS = 20


def load_model():
    onnx_model_path = 'resnet50.onnx'
    ort_session = ort.InferenceSession(onnx_model_path)
    return ort_session

def preprocess_image(image):
    image = image.resize((224, 224))
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
            image_np = np.array(img).astype(np.float32)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            
            image_np = (image_np / 255.0 - mean) / std
            image_np = np.transpose(image_np, (2, 0, 1)) 
            image_np = np.expand_dims(image_np, axis=0).astype(np.float32)
            return file_path, image_np
    except Exception as e:
        print(f"Error loading image {file_path}: {e}")
        return None


def extract_features(ort_session, images):
    """Extract features from a list of images using the given model."""
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    features = []
    for img in images:

        ort_inputs = {input_name: img}
        ort_outs = ort_session.run([output_name], ort_inputs)
        features.append(ort_outs[0].flatten())
    return np.array(features)

def get_images_from_folder(folder_path, num_threads=4):
    """Load images from the specified folder using multi-threading."""
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(load_and_preprocess_image, image_files))
    
    # Filter out None results (failed loads) and separate paths and images
    valid_results = [r for r in results if r is not None]
    image_paths, images = zip(*valid_results) if valid_results else ([], [])
    
    return list(image_paths), list(images)

"""
def plot_clusters(features_tsne, clusters):
    # Plot the TSNE clusters.
    plt.scatter(features_tsne[:,0], features_tsne[:,1], c=clusters)
    plt.colorbar()
    plt.savefig('clusters.png')

def calculate_image_distance(image_path1, image_path2, scaler):
    # Calculate the Euclidean distance between two images.
    model = load_model()

    img1 = Image.open(image_path1).convert('RGB')
    img2 = Image.open(image_path2).convert('RGB')
    img1 = preprocess_image(img1)
    img2 = preprocess_image(img2)

    features1 = extract_features(model, [img1])
    features2 = extract_features(model, [img2])
    features1 = scaler.transform(features1)[0]
    features2 = scaler.transform(features2)[0]

    distance = euclidean(features1, features2)
    return distance
"""


def main(folder_path):
    scaler = StandardScaler()
    # Load model
    ort_session = load_model()

    # Get images
    start_time = time.time()
    paths, images = get_images_from_folder(folder_path)
    if len(images) == 0:
        raise Exception("No Images in folder found")

    end_time = time.time()
    print('getting images time: ', (end_time - start_time))

    # Extract features
    start_time = time.time()
    features = extract_features(ort_session, images)
    end_time = time.time()
    print('resnet time: ', (end_time - start_time))
    features = scaler.fit_transform(features)


    # DBSCAN clustering
    start_time = time.time()
    dbscan = DBSCAN(eps=EPS, min_samples=1)
    clusters = dbscan.fit_predict(features)
    end_time = time.time()
    print('clustering time: ', (end_time - start_time))

    # Evaluate clusters with scatterplot
    """
    tsne = TSNE(n_components=2, random_state=0)
    features_tsne = tsne.fit_transform(features)
    plot_clusters(features_tsne, clusters)
    """

    # Group images by clusters
    start_time = time.time()
    cluster_images = {i: [] for i in range(-1, max(clusters) + 1)}
    for idx, cluster in enumerate(clusters):
        cluster_images[cluster].append(paths[idx])
    end_time = time.time()
    print('make cluster object time: ', (end_time - start_time))
    return cluster_images



if __name__ == "__main__":
     folder_path = 'album_images'
     cluster_imgs = main(folder_path)
     print(cluster_imgs)
