# import os
# import cv2
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.svm import SVC  # Replaced RandomForestClassifier with SVM
# from sklearn.metrics import mean_absolute_error, classification_report
# from keras.applications.vgg16 import VGG16, preprocess_input
# from keras.models import Model
# from skimage.feature import local_binary_pattern
# import joblib  # For saving and loading models

# # Step 1: Load the dataset
# def load_dataset(csv_path, image_folder):
#     print("Load dataset from CSV and image folder.")
#     df = pd.read_csv(csv_path)
#     image_paths = []
#     ages = []
#     genders = []

#     for index, row in df.iterrows():
#         image_name = row['imageName']
#         image_path = os.path.join(image_folder, image_name)
#         if os.path.exists(image_path):
#             image_paths.append(image_path)
#             ages.append(row['age'])
#             genders.append(row['gender'])
#         else:
#             print(f"Warning: Image {image_name} not found in folder.")

#     return image_paths, np.array(ages), np.array(genders)

# # Step 2: Extract VGG16 features
# def extract_vgg_features(image_paths):
#     print("Extract VGG16 features from images.")
#     # Load VGG16 model
#     base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
#     model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)

#     features = []
#     for path in image_paths:
#         img = cv2.imread(path)
#         img = cv2.resize(img, (224, 224))
#         img = preprocess_input(img)
#         img = np.expand_dims(img, axis=0)
#         feature = model.predict(img, verbose=0)
#         features.append(feature.flatten())
#     return np.array(features)

# # Step 3: Extract LBP features
# def extract_lbp_features(image_paths):
#     print("Extract LBP features from images.")
#     features = []
#     for path in image_paths:
#         img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#         img = cv2.resize(img, (224, 224))
#         lbp = local_binary_pattern(img, P=8, R=1, method="uniform")
#         hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
#         hist = hist.astype("float")
#         hist /= (hist.sum() + 1e-6)  # Normalize histogram
#         features.append(hist)
#     return np.array(features)

# # Step 4: Combine features and train models
# def train_models(image_paths, ages, genders):
#     print("Train models for age and gender prediction.")
#     # Extract features
#     vgg_features = extract_vgg_features(image_paths)
#     lbp_features = extract_lbp_features(image_paths)
#     combined_features = np.hstack([vgg_features, lbp_features])

#     # Split dataset
#     X_train, X_test, y_age_train, y_age_test, y_gender_train, y_gender_test = train_test_split(
#         combined_features, ages, genders, test_size=0.2, random_state=42)

#     # Train age prediction model (regression)
#     age_model = RandomForestRegressor(n_estimators=100, random_state=42)
#     age_model.fit(X_train, y_age_train)
#     age_pred = age_model.predict(X_test)
#     print("Age Prediction Results:")
#     print(f"Mean Absolute Error: {mean_absolute_error(y_age_test, age_pred)}")

#     # Train gender prediction model (classification) using SVM
#     gender_model = SVC(kernel='rbf', probability=True, random_state=42)  # SVM with RBF kernel
#     gender_model.fit(X_train, y_gender_train)
#     gender_pred = gender_model.predict(X_test)
#     print("Gender Prediction Results:")
#     print(classification_report(y_gender_test, gender_pred))

#     # Save models
#     joblib.dump(age_model, 'age_model.pkl')
#     joblib.dump(gender_model, 'gender_model.pkl')
#     print("Models saved to disk.")

# # Step 5: Predict age and gender for new images
# def predict_age_gender(image_path, age_model_path='age_model.pkl', gender_model_path='gender_model.pkl'):
#     print("Predict age and gender for a new image.")
#     # Load models
#     age_model = joblib.load(age_model_path)
#     gender_model = joblib.load(gender_model_path)

#     # Extract features for the new image
#     vgg_features = extract_vgg_features([image_path])
#     lbp_features = extract_lbp_features([image_path])
#     combined_features = np.hstack([vgg_features, lbp_features])

#     # Predict age and gender
#     age_pred = age_model.predict(combined_features)
#     gender_pred = gender_model.predict(combined_features)

#     return age_pred[0], gender_pred[0]

# # Main script
# if __name__ == "__main__":
#     # Paths
#     csv_path = "C:/Users/gopal/OneDrive/Desktop/Major project/HandInfo.csv"
#     image_folder = "C:/Users/gopal/OneDrive/Desktop/Major project/Processed_NIR_Images"

#     # Load dataset
#     image_paths, ages, genders = load_dataset(csv_path, image_folder)

#     # Train models
#     train_models(image_paths, ages, genders)

#     # Predict age and gender for a new image
#     new_image_path = "C:/Users/gopal/OneDrive/Desktop/Major project/Processed_NIR_Images/Hand_0011587.jpg"
#     age_pred, gender_pred = predict_age_gender(new_image_path)
#     print(f"Predicted Age: {age_pred}, Predicted Gender: {gender_pred}")

import cv2
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from skimage.feature import local_binary_pattern
import joblib

# Load pre-trained models
age_model = joblib.load('age_model.pkl')
gender_model = joblib.load('gender_model.pkl')

# Load VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
vgg_model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)

def extract_vgg_features(image_path):
    """Extract VGG16 features for a single image"""
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    feature = vgg_model.predict(img, verbose=0)
    return feature.flatten()

def extract_lbp_features(image_path):
    """Extract LBP features for a single image"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224, 224))
    lbp = local_binary_pattern(img, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)  # Normalize histogram
    return hist

def predict_age_gender(image_path):
    """Predict age and gender for a single image"""
    # Extract features
    vgg_features = extract_vgg_features(image_path)
    lbp_features = extract_lbp_features(image_path)
    combined_features = np.hstack([vgg_features, lbp_features]).reshape(1, -1)
    
    # Predict
    age_pred = age_model.predict(combined_features)[0]
    gender_pred = gender_model.predict(combined_features)[0]
    
    return age_pred, gender_pred