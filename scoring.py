import cv2
import dlib
from scipy.spatial import distance
import numpy as np
#import matplotlib.pyplot as plt
#import pdb

# Eye Aspect Ratio calculation
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Constant
COEF_MAPPING = {'eyes_score': 0.5,
                'smile_score': 0.2,
                'blur_score': 0.2,
                'brightness_score': 0.05,
                'contrast_score': 0.05,}
EYE_THRESHOLD = 0.2

# Load images from folder
def load_images_from_folder(filenames):
    images = []
    for file in filenames:
        img = cv2.imread(file)
        img = cv2.resize(img, (0,0), fx=0.8, fy=0.8)
        if img is not None:
            images.append((file, img))
    return images



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
            face_area / image_area > 0.01):
            filtered_faces.append(face)
    
    return filtered_faces


def normalize_array(arr):
    mean = np.mean(arr)
    std = np.std(arr)
    
    if std == 0:
        return np.zeros_like(arr)  # Avoid division by zero
    
    return (arr - mean) / std

def eye_aspect_ratio(eye):
    A = distance.euclidean((eye[1].x, eye[1].y), (eye[5].x, eye[5].y))
    B = distance.euclidean((eye[2].x, eye[2].y), (eye[4].x, eye[4].y))
    C = distance.euclidean((eye[0].x, eye[0].y), (eye[3].x, eye[3].y))
    ear = (A + B) / (2.0 * C)
    return ear

def mean(scores):
    return np.mean(np.array(scores))

def eyes_open_score(image, faces, gray):
    if len(faces) == 0:
        return 0  # No face detected
    scores = []
    for face in faces:
        shape = predictor(gray, face)
        leftEye = [shape.part(i) for i in range(36, 42)]
        rightEye = [shape.part(i) for i in range(42, 48)]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        scores.append(ear)
    
    open_eyes = sum(1 for score in scores if score > EYE_THRESHOLD)
    total_faces = len(scores)
    open_eye_percentage = open_eyes / total_faces
    
    penalized_scores = [max(0, score - EYE_THRESHOLD) for score in scores]
    average_ear = sum(penalized_scores) / total_faces
    
    # More aggressive penalty
    penalty_factor = (open_eye_percentage ** 2)
    final_score = (average_ear * penalty_factor)
    
    return final_score

# Smile detection
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

def smile_score(image, faces, gray):

    if len(faces) == 0:
        return 0

    comp_score = []

    for face in faces:
        # Assuming the first detected face is the target
        landmarks = predictor(gray, face)

        # Extract coordinates of the mouth region
        mouth_points = []
        for i in range(48, 68):  # 48-67 are mouth points in the 68 point model
            mouth_points.append((landmarks.part(i).x, landmarks.part(i).y))
        
        mouth_points = np.array(mouth_points)
        
        # Calculate the width and height of the mouth
        mouth_width = np.linalg.norm(mouth_points[6] - mouth_points[0])  # Distance between corners of the mouth
        mouth_height = np.linalg.norm(mouth_points[3] - mouth_points[9])  # Distance between top and bottom of the mouth
        
        # Calculate smile intensity based on width and height
        smile_intensity = mouth_width * mouth_height
        
        # Analyze curvature: higher curvature implies a better smile
        upper_lip_curve = np.polyfit(mouth_points[:7, 0], mouth_points[:7, 1], 2)  # Upper lip
        lower_lip_curve = np.polyfit(mouth_points[7:, 0], mouth_points[7:, 1], 2)  # Lower lip
        
        upper_lip_curvature = upper_lip_curve[0]
        lower_lip_curvature = lower_lip_curve[0]
        
        curvature_score = (upper_lip_curvature - lower_lip_curvature) * -1  # Inverse curvature
        
        # Combine the features into a final smile score
        raw_score = smile_intensity + curvature_score

        comp_score.append(raw_score)
    
    comp_score = sum(comp_score) / len(comp_score)
    return comp_score

# Blur detection
def blur_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var

# Lighting evaluation
def lighting_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = gray.mean() / 255.0
    contrast = gray.std() / 255.0
    return mean_brightness, contrast

# Code to plot faces
"""
def plot_faces(image, faces):
    
    # Create a copy of the image to draw on
    img_with_boxes = image.copy()
    
    # Draw bounding boxes around detected faces
    for face in faces:
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Display the image with bounding boxes
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
"""



def main(img_filenames):
    if len(img_filenames) <= 1:
        return {file: 10 for file in img_filenames}
    images = load_images_from_folder(img_filenames)

    # Initialize lists to hold scores
    scores = []
    max_scores = {'eyes_score': float('-inf'),
                'smile_score': float('-inf'),
                'blur_score': float('-inf'),
                'brightness_score': float('-inf'),
                'contrast_score': float('-inf'),}
    all_faces = {}
    max_faces = 0
    for filename,img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        faces = filter_main_subjects(img,faces)
        all_faces[filename] = ((faces, gray))
        if len(faces) >= max_faces:
            max_faces = len(faces)

    

    # Calculate scores
    for filename, img in images:
        e_score = 1
        s_score = 1

        faces, gray = all_faces[filename]
        #plot_faces(img,faces)

        if len(faces) > 0:
            e_score = eyes_open_score(img, faces, gray)
            s_score = smile_score(img, faces, gray)


        b_score = blur_score(img)
        brightness_score, contrast_score = lighting_score(img)


        if len(faces) < max_faces:
            e_score -= len(faces) / max_faces
            s_score -= len(faces) / max_faces

        scores.append(
            {
                'eyes_score': max(e_score, 0),
                'smile_score': max(s_score, 0),
                'blur_score': b_score,
                'brightness_score': brightness_score,
                'contrast_score': contrast_score,
            }
        )
        for score_type in max_scores:
            if scores[-1][score_type] > max_scores[score_type]:
                max_scores[score_type] = scores[-1][score_type]
            
    composite_scores = {}
    for i, img_scores in enumerate(scores):
        comp_score = 0
        for score_type in max_scores:
            img_scores[score_type] = img_scores[score_type] / max(max_scores[score_type], 0.001)
            comp_score += img_scores[score_type] * COEF_MAPPING[score_type]
        composite_scores[img_filenames[i]] = round(comp_score*10)

    # Evaluation CODE
    """
    # Select the best image
    best_image_index = np.argmax(composite_scores)
    best_image_filename = img_filenames[best_image_index]
    best_image = images[best_image_index][1]

    # Printing out all scoring info and picking best
    for i, image_tuple in enumerate(images):
        filename, _ = image_tuple
        print(f"Image: {filename}")
        print(f"  Eyes Open Score: {scores[i]['eyes_score']}")
        print(f"  Smile Score: {scores[i]['smile_score']}")
        print(f"  Blur Score: {scores[i]['blur_score']}")
        print(f"  Lighting Score (Brightness, Contrast): {scores[i]['brightness_score'], scores[i]['contrast_score']}")
        print(f"  Composite Score: {composite_scores[filename]}")
        print("")

    print(f"The best image is {best_image_filename} with a composite score of {composite_scores[best_image_filename]}")

    # Save or display the best image
    cv2.imwrite('best_image.jpg', best_image)
    cv2.imshow('Best Image', best_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """

    return composite_scores



if __name__=='__main__':
    img_filenames = ['album_images/63.jpg', 'album_images/62.jpg', 'album_images/61.jpg', 'album_images/65.jpg']
    file = main(img_filenames)
