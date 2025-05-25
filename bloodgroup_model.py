# import os
# import cv2
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, accuracy_score
# from keras.applications.vgg16 import VGG16, preprocess_input
# from keras.preprocessing.image import load_img, img_to_array
# from keras.models import Model

# # Initialize feature extractors
# sift = cv2.SIFT_create()
# orb = cv2.ORB_create()

# # Load VGG16 model
# base_model = VGG16(weights='imagenet', include_top=True)
# vgg_model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

# def process_image(img_path):
#     """Process image for VGG16"""
#     img = load_img(img_path, target_size=(224, 224))
#     img = img_to_array(img)
#     img = preprocess_input(img)
#     return img

# def extract_vgg_features(img_array):
#     """Extract VGG16 features"""
#     img_array = np.expand_dims(img_array, axis=0)
#     return vgg_model.predict(img_array, verbose=0).flatten()

# def extract_sift_features(img_path):
#     """Extract SIFT features with fixed-size vector"""
#     img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#     img = cv2.resize(img, (224, 224))
#     _, desc = sift.detectAndCompute(img, None)
#     if desc is not None:
#         return np.mean(desc, axis=0)
#     return np.zeros(128)

# def extract_orb_features(img_path):
#     """Extract ORB features with fixed-size vector"""
#     img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#     img = cv2.resize(img, (224, 224))
#     _, desc = orb.detectAndCompute(img, None)
#     if desc is not None:
#         return np.mean(desc, axis=0)
#     return np.zeros(32)

# # Load dataset
# dataset_path = r'C:\Users\gopal\OneDrive\Desktop\Major project\NIR DATASET'
# X = []
# y = []

# # First loop: Only process directories
# for blood_group in os.listdir(dataset_path):
#     group_path = os.path.join(dataset_path, blood_group)
    
#     # Skip if it's not a directory
#     if not os.path.isdir(group_path):
#         continue
    
#     # Second loop: Process image files
#     for img_file in os.listdir(group_path):
#         img_path = os.path.join(group_path, img_file)
        
#         # Skip non-image files and directories
#         if not os.path.isfile(img_path):
#             continue
#         if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
#             continue
            
#         X.append(img_path)
#         y.append(blood_group)


# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, 
#     test_size=0.2, 
#     random_state=42,
#     stratify=y
# )
# # Feature extraction
# def extract_features(img_paths):
#     features = []
#     for path in img_paths:
#         # Extract VGG features
#         vgg_img = process_image(path)
#         vgg_feat = extract_vgg_features(vgg_img)
        
#         # Extract handcrafted features
#         sift_feat = extract_sift_features(path)
#         orb_feat = extract_orb_features(path)
        
#         # Combine features
#         combined = np.concatenate([vgg_feat, sift_feat, orb_feat])
#         features.append(combined)
#     return np.array(features)

# X_train_feat = extract_features(X_train)
# X_test_feat = extract_features(X_test)

# # Encode labels
# le = LabelEncoder()
# y_train_enc = le.fit_transform(y_train)
# y_test_enc = le.transform(y_test)

# # Initialize and train classifier
# clf = RandomForestClassifier(n_estimators=100, random_state=42)
# clf.fit(X_train_feat, y_train_enc)


# # Evaluate
# y_pred = clf.predict(X_test_feat)
# print(f"Accuracy: {accuracy_score(y_test_enc, y_pred):.2f}")
# print(classification_report(
#     y_test_enc, 
#     y_pred, 
#     target_names=le.classes_,
#     labels=np.unique(y_test_enc),
#     zero_division=0
# ))

# # Save model
# import joblib
# joblib.dump(clf, 'blood_group_classifier.pkl')
# joblib.dump(le, 'label_encoder.pkl')
import os
import cv2
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Model
import joblib

# Initialize feature extractors
sift = cv2.SIFT_create()
orb = cv2.ORB_create()

# Load VGG16 model
base_model = VGG16(weights='imagenet', include_top=True)
vgg_model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

# Load pre-trained classifier and label encoder
clf = joblib.load('blood_group_classifier.pkl')
le = joblib.load('label_encoder.pkl')

def process_image(img_path):
    """Process image for VGG16"""
    img = load_img(img_path, target_size=(224, 224))
    img = img_to_array(img)
    img = preprocess_input(img)
    return img

def extract_vgg_features(img_array):
    """Extract VGG16 features"""
    img_array = np.expand_dims(img_array, axis=0)
    return vgg_model.predict(img_array, verbose=0).flatten()

def extract_sift_features(img_path):
    """Extract SIFT features with fixed-size vector"""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224, 224))
    _, desc = sift.detectAndCompute(img, None)
    if desc is not None:
        return np.mean(desc, axis=0)
    return np.zeros(128)

def extract_orb_features(img_path):
    """Extract ORB features with fixed-size vector"""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224, 224))
    _, desc = orb.detectAndCompute(img, None)
    if desc is not None:
        return np.mean(desc, axis=0)
    return np.zeros(32)

def predict_blood_group(img_path):
    """Predict blood group for a single image"""
    # Extract features
    vgg_img = process_image(img_path)
    vgg_feat = extract_vgg_features(vgg_img)
    sift_feat = extract_sift_features(img_path)
    orb_feat = extract_orb_features(img_path)
    
    # Combine features
    combined = np.concatenate([vgg_feat, sift_feat, orb_feat])
    combined = combined.reshape(1, -1)
    
    # Predict
    pred = clf.predict(combined)
    blood_group = le.inverse_transform(pred)[0]
    return blood_group