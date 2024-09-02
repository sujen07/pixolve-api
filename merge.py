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

        for face in faces:
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
    cluster_imgs = cluster.main(folder_path, min_samples=len(os.listdir(folder_path)))
    if len(cluster_imgs) > 1:
        raise Exception("Images are not similar enough to merge!")
    
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
        print(face)
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