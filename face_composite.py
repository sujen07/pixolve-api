import cv2
import numpy as np
import dlib
from skimage.exposure import match_histograms
import pdb

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



def calculate_scaling_factor(source_landmarks, target_landmarks, feature_indices):
    source_feature_points = source_landmarks[feature_indices]
    target_feature_points = target_landmarks[feature_indices]

    # Calculate the distance between the outermost points (e.g., between the eyes)
    source_width = np.linalg.norm(source_feature_points[0] - source_feature_points[-1])
    target_width = np.linalg.norm(target_feature_points[0] - target_feature_points[-1])

    # Scaling factor based on the width of the feature
    scaling_factor = target_width / source_width
    return scaling_factor



def merge_faces(source_image, target_image_full, face_location, debug=False):
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

    # Extract features
    source_feature, source_mask = extract_feature(source_image, source_landmarks, all_features)
    target_feature, target_mask = extract_feature(target_image, target_landmarks, all_features)

    # Align features and masks
    source_points = source_landmarks[all_features]
    target_points = target_landmarks[all_features]

    M, _ = cv2.estimateAffinePartial2D(source_points, target_points, method=cv2.RANSAC)

    if M is not None:
        # Align the source feature and mask
        aligned_feature = cv2.warpAffine(source_feature, M, (target_image.shape[1], target_image.shape[0]))
        aligned_mask = cv2.warpAffine(source_mask, M, (target_image.shape[1], target_image.shape[0]), flags=cv2.INTER_NEAREST)

        # Calculate scaling factor based on the aligned mask and target mask
        aligned_mask_area = np.sum(aligned_mask > 0)
        target_mask_area = np.sum(target_mask > 0)
        scaling_factor = np.sqrt(target_mask_area / aligned_mask_area)

        # Scale the aligned mask
        kernel = np.ones((3,3), np.uint8)
        dilated_mask = cv2.dilate(aligned_mask, kernel, iterations=int(scaling_factor) - 1)
        eroded_mask = cv2.erode(dilated_mask, kernel, iterations=int(scaling_factor) - 1)
        scaled_mask = cv2.resize(eroded_mask, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_LINEAR)

        # Ensure the mask fits within the target image
        h, w = target_image.shape[:2]
        scaled_mask = scaled_mask[:h, :w]

        # Ensure that masks and images have the same size and type
        if aligned_feature.shape[:2] != scaled_mask.shape:
            scaled_mask = cv2.resize(scaled_mask, (aligned_feature.shape[1], aligned_feature.shape[0]))

        if aligned_feature.shape[:2] != target_mask.shape:
            target_mask = cv2.resize(target_mask, (aligned_feature.shape[1], aligned_feature.shape[0]))

        # Ensure the mask is of type np.uint8
        if scaled_mask.dtype != np.uint8:
            scaled_mask = scaled_mask.astype(np.uint8)

        # Apply color correction only to the masked region
        aligned_feature_masked = cv2.bitwise_and(aligned_feature, aligned_feature, mask=scaled_mask)
        target_feature_masked = cv2.bitwise_and(target_feature, target_feature, mask=target_mask)

        # Color correction using histogram matching on masked regions
        corrected_feature_masked = match_histograms(aligned_feature_masked, target_feature_masked)

        # Combine the corrected masked region with the original aligned feature
        corrected_feature = aligned_feature.copy()
        corrected_feature[scaled_mask > 0] = corrected_feature_masked[scaled_mask > 0]

        # Ensure that corrected feature has 3 channels
        if corrected_feature.shape[2] != 3:
            corrected_feature = cv2.cvtColor(corrected_feature, cv2.COLOR_GRAY2BGR)

        # Ensure the mask is a single channel
        if len(scaled_mask.shape) == 3:
            scaled_mask = cv2.cvtColor(scaled_mask, cv2.COLOR_BGR2GRAY)

        # Calculate the center of the feature for seamless cloning
        feature_center = calculate_feature_center(target_landmarks, all_features)
        center = (feature_center[0] + left, feature_center[1] + top)

        # Blend images using seamless cloning
        target_image_full = cv2.seamlessClone(corrected_feature, target_image_full, scaled_mask, center, cv2.NORMAL_CLONE)

    return target_image_full
