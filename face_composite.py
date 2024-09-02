import cv2
import numpy as np
import dlib
from skimage.exposure import match_histograms

# Load the face detector and the landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def get_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if not faces:
        return None
    landmarks = predictor(gray, faces[0])
    return np.array([(p.x, p.y) for p in landmarks.parts()])

def create_focus_mask(landmarks, feature_indices, image_shape):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    feature_points = landmarks[feature_indices]
    hull = cv2.convexHull(np.array(feature_points))
    cv2.fillConvexPoly(mask, hull, 255)
    return mask

def extract_feature(image, landmarks, feature_indices):
    mask = create_focus_mask(landmarks, feature_indices, image.shape)
    feature = cv2.bitwise_and(image, image, mask=mask)
    return feature, mask

def calculate_feature_center(landmarks, feature_indices):
    feature_points = landmarks[feature_indices]
    x_coords, y_coords = zip(*feature_points)
    center_x = int(np.mean(x_coords))
    center_y = int(np.mean(y_coords))
    return (center_x, center_y)

def merge_faces(source_image, target_image_full, face_location):
    left, top, right, bottom = face_location
    target_image = target_image_full[top:bottom, left:right]

    # Get landmarks
    source_landmarks = get_landmarks(source_image)
    target_landmarks = get_landmarks(target_image)

    if source_landmarks is None or target_landmarks is None:
        print("Could not detect landmarks in one of the images.")
        return target_image_full

    # Features to blend
    all_features = list(range(17,68))

    # Extract and align features
    source_feature, source_mask = extract_feature(source_image, source_landmarks, all_features)
    target_feature, target_mask = extract_feature(target_image, target_landmarks, all_features)

    # Align features
    source_points = source_landmarks[all_features]
    target_points = target_landmarks[all_features]
    M, _ = cv2.estimateAffinePartial2D(source_points, target_points, method=cv2.RANSAC)

    if M is not None:
        aligned_feature = cv2.warpAffine(source_feature, M, (target_image.shape[1], target_image.shape[0]))

        # Calculate the center of the feature for seamless cloning
        feature_center = calculate_feature_center(target_landmarks, all_features)
        center = (feature_center[0] + left, feature_center[1] + top)

        # Blend images using seamless cloning
        target_image_full = cv2.seamlessClone(aligned_feature, target_image_full, target_mask, center, cv2.NORMAL_CLONE)


    return target_image_full
