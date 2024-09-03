import scoring
import cluster
import cv2
import os
import dlib
from collections import defaultdict
import numpy as np
import face_composite
import pdb

detector = dlib.get_frontal_face_detector()
recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
face_distance_threshold = 0.6


def load_images_from_folder(folder, files):
    images = []
    for filename in files:
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append((filename, img))
    return images


def is_same_face(known_faces, face_encoding):
    for idx, known_face in enumerate(known_faces):
        distance = np.linalg.norm(known_face - face_encoding)
        if distance < face_distance_threshold:
            return idx
    return -1

def filter_main_subjects(image, faces):
    height, width = image.shape[:2]
    image_area = height * width
    
    filtered_faces = []
    face_areas = []
    
    # Calculate face areas and store them
    for face in faces:
        face_width = face.right() - face.left()
        face_height = face.bottom() - face.top()
        face_area = face_width * face_height
        face_areas.append(face_area)
    
    if len(faces) == 0:
        return []
    
    # Calculate statistics
    max_area = max(face_areas)
    avg_area = sum(face_areas) / len(face_areas)
    
    for i, face in enumerate(faces):
        face_area = face_areas[i]
        
        # Criteria for keeping a face:
        # 1. If it's the largest face
        # 2. If it's at least 50% of the largest face's size
        # 3. If it's larger than the average face size
        # 4. If it takes up more than 1% of the image (adjust as needed)
        if (face_area == max_area or 
            face_area >= 0.5 * max_area or 
            face_area > avg_area or 
            face_area / image_area > 0.1):
            filtered_faces.append(face)
    
    return filtered_faces

def clip_to_image(x, y, w, h, image_width, image_height):
    """
    Clip the face bounding box coordinates to be within the image boundaries.
    
    Parameters:
    x, y: Top-left corner coordinates of the face bounding box
    w, h: Width and height of the face bounding box
    image_width, image_height: Dimensions of the main image
    
    Returns:
    Tuple of clipped (x1, y1, x2, y2) coordinates
    """
    x1 = max(0, min(x, image_width - 1))
    y1 = max(0, min(y, image_height - 1))
    x2 = max(0, min(x + w, image_width))
    y2 = max(0, min(y + h, image_height))
    return x1, y1, x2, y2

def create_composite_score(scores):
    max_scores = {
        'eyes_score' : 0,
        'smile_score' : 0,
        'blur_score' : 0,
        'brightness_score' : 0,
        'contrast_score' : 0
    }
    for score in scores:
        for type in score:
            max_scores[type] = max(max_scores[type], score[type])
    
    best_score = [0,0]
    for i, score in enumerate(scores):
        composite_score = 0
        for type in max_scores:
            score[type] /= max(max_scores[type], 0.0001)
            composite_score += score[type] * scoring.COEF_MAPPING[type]
        scores[i] = composite_score
        if composite_score > best_score[1]:
            best_score[1] = composite_score
            best_score[0] = i
    return best_score[0]


def create_face_dicts(images):
    # Dictionary to store the face encodings, scores, and locations
    face_dict = defaultdict(lambda: {"scores": [], "locations": []})
    unique_faces = []

    # Process each image
    for filename, image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for faster processing
        faces = detector(gray, 1)  # Detect faces in the image
        faces = filter_main_subjects(image, faces)
        image_height, image_width = image.shape[:2]

        for face in faces:
            print('file: ', filename)
            print(face)
            shape = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")(gray, face)
            face_encoding = np.array(recognizer.compute_face_descriptor(image, shape))

            # Check if this face is already known
            match_index = is_same_face(unique_faces, face_encoding)
            if match_index == -1:
                # New face
                unique_faces.append(face_encoding)
                match_index = len(unique_faces) - 1

            # Use the match index as the key
            face_key = f"face_{match_index}"

            x1, y1 = face.left(), face.top()
            x2, y2 = face.right(), face.bottom()

            x1, y1, x2, y2 = clip_to_image(x1, y1, x2 - x1, y2 - y1, image_width, image_height)

            face_image = image[y1:y2, x1:x2]

            # Get all the scores for the face
            brightness_score, contrast_score = scoring.lighting_score(face_image)
            scores = {
                'eyes_score' : scoring.eyes_open_score(image, [face], gray),
                'smile_score' : scoring.smile_score(image, [face], gray),
                'blur_score' : scoring.blur_score(face_image),
                'brightness_score' : brightness_score,
                'contrast_score' : contrast_score
            }


            # Score the face and store in dictionary
            face_dict[face_key]["scores"].append(scores)
            face_dict[face_key]["locations"].append((filename, (face.left(), face.top(), face.right(), face.bottom())))


    for face in face_dict:
        scores = face_dict[face]['scores']
        best_face = create_composite_score(scores)
        face_dict[face]['best_face_ind'] = best_face
        #del face_dict[face]['scores']

    return face_dict



def main(folder_path):
    cluster_imgs = cluster.main(folder_path, min_samples=2)
    if len(cluster_imgs) > 1:
        return None
    
    del cluster_imgs

    files = os.listdir(folder_path)

    # Load images
    images = load_images_from_folder(folder_path, files)
    face_dict = create_face_dicts(images)

    best_score = float('-inf')
    for i, image in enumerate(images):
        filename,img = image
        score = 0
        score += scoring.blur_score(img)
        b_score, c_score = scoring.lighting_score(img)
        score += b_score + c_score
        if score > best_score:
            best_score = score
            target_image = img
            target_ind = i
            
    

    #pdb.set_trace()
    for face in face_dict:
        if len(face_dict[face]['locations']) != len(images):
            continue
        locations = face_dict[face]['locations'][target_ind][1]
        best_ind = face_dict[face]['best_face_ind']
        x1, y1, x2, y2 = face_dict[face]['locations'][best_ind][1]
        img = images[best_ind][1]
        source_face = img[y1:y2, x1:x2]
        target_image = face_composite.merge_faces(source_face, target_image, locations)

    target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
    return target_image


if __name__ == '__main__':
    target_image = main('test')
    cv2.imwrite('test1.jpg', cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB))