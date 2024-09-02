import cv2
import numpy as np
import dlib

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

def create_focus_mask(landmarks, image_shape):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    indices = {
        'whole_face': list(range(17,60)) + list(range(0,6)) + list(range(11, 17))
    }
    for region_points in indices.values():
        hull = cv2.convexHull(landmarks[region_points])
        cv2.fillConvexPoly(mask, hull, 255)
    return mask

def merge_faces(source_image, target_image_full, face_location):
    left, top, right, bottom = face_location
    target_image = target_image_full[top:bottom, left:right]

    # Get landmarks
    source_landmarks = get_landmarks(source_image)
    target_landmarks = get_landmarks(target_image)

    if source_landmarks is None or target_landmarks is None:
        print("Could not detect landmarks in one of the images.")
        return target_image_full

    # Use only the main facial features for alignment
    selected_points = list(range(36, 48)) + list(range(30, 36))  # Eyes + Nose

    source_points = source_landmarks[selected_points]
    target_points = target_landmarks[selected_points]

    # Compute and apply the affine transform
    M, _ = cv2.estimateAffinePartial2D(source_points, target_points, method=cv2.RANSAC)
    if M is None:
        print("Similarity transform could not be computed.")
        return target_image_full

    aligned_image = cv2.warpAffine(source_image, M, (target_image.shape[1], target_image.shape[0]))

    # Clip the aligned face to the target face region
    aligned_image = cv2.bitwise_and(aligned_image, aligned_image, mask=create_focus_mask(target_landmarks, target_image.shape))

    # Color correction
    mean_source = cv2.mean(aligned_image)[:3]
    mean_target = cv2.mean(target_image)[:3]
    corrected_image = np.clip(aligned_image + (np.array(mean_target) - np.array(mean_source)), 0, 255).astype(np.uint8)

    # Create and apply focus mask
    source_mask = create_focus_mask(get_landmarks(corrected_image), corrected_image.shape)
    source_mask = cv2.GaussianBlur(source_mask / 255.0, (7, 7), 0)
    source_mask = np.repeat(source_mask[:, :, np.newaxis], 3, axis=2)

    # Blend images
    composite_img = (source_mask * corrected_image + (1 - source_mask) * target_image).astype(np.uint8)

    # Insert composite into full target image
    final_img = target_image_full.copy()
    final_img[top:bottom, left:right] = composite_img

    return final_img