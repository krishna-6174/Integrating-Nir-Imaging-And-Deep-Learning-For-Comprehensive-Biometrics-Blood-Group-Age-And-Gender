import tkinter as tk
from tkinter import ttk, filedialog
from PIL import ImageTk, Image
import cv2
import numpy as np
import joblib
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from skimage.feature import local_binary_pattern
import warnings
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVC  # Better classifier for gender prediction
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, classification_report
from sklearn.preprocessing import LabelEncoder

# Suppress TensorFlow warnings
warnings.filterwarnings('ignore')

# Load all models and encoders
age_model = joblib.load('age_model.pkl')
gender_model = joblib.load('gender_model.pkl')  # Replace with improved model
blood_group_model = joblib.load('blood_group_classifier.pkl')
le = joblib.load('label_encoder.pkl')

# Initialize VGG16 models for different feature extraction
# For age/gender prediction
base_model_ag = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
vgg_model_ag = Model(inputs=base_model_ag.input, outputs=base_model_ag.get_layer('block5_pool').output)

# For blood group prediction
base_model_bg = VGG16(weights='imagenet', include_top=True)
vgg_model_bg = Model(inputs=base_model_bg.input, outputs=base_model_bg.get_layer('fc1').output)

# Initialize feature detectors
sift = cv2.SIFT_create()
orb = cv2.ORB_create()

class PredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Bio-Metric Prediction System")
        self.root.geometry("800x600")
        
        # Initialize image path
        self.image_path = None
        
        # Create GUI elements
        self.create_widgets()
    
    def create_widgets(self):
        # Image display
        self.img_label = ttk.Label(self.root)
        self.img_label.pack(pady=10)
        
        # Upload button
        self.upload_btn = ttk.Button(self.root, text="Upload Image", command=self.upload_image)
        self.upload_btn.pack(pady=5)
        
        # Submit button
        self.submit_btn = ttk.Button(self.root, text="Predict", command=self.predict, state=tk.DISABLED)
        self.submit_btn.pack(pady=5)
        
        # Result display
        self.result_label = ttk.Label(self.root, text="", font=('Helvetica', 14))
        self.result_label.pack(pady=20)
    
    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            self.image_path = file_path
            self.show_image()
            self.submit_btn.config(state=tk.NORMAL)
    
    def show_image(self):
        img = Image.open(self.image_path)
        img.thumbnail((300, 300))
        img_tk = ImageTk.PhotoImage(img)
        self.img_label.config(image=img_tk)
        self.img_label.image = img_tk
    
    def predict(self):
        if self.image_path:
            # Process image for age and gender prediction
            age, gender = self.predict_age_gender()
            
            # Process image for blood group prediction
            blood_group = self.predict_blood_group()
            
            # Display results
            result_text = f"Predicted Age: {age}\n" \
                          f"Predicted Gender: {gender}\n" \
                          f"Predicted Blood Group: {blood_group}"
            self.result_label.config(text=result_text)
    
    # Feature extraction functions
    def extract_vgg_ag_features(self, img):
        """Extract VGG16 features for age and gender prediction."""
        img = cv2.resize(img, (224, 224))  # Resize to 224x224
        img = preprocess_input(img)  # Preprocess for VGG16
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        feature = vgg_model_ag.predict(img, verbose=0)
        return feature.flatten()
    
    def extract_lbp_features(self, img):
        """Extract LBP features for age and gender prediction."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (224, 224))  # Resize to 224x224
        lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")  # Use same LBP parameters
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))  # Use same bins
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)  # Normalize
        return hist
    
    def extract_vgg_bg_features(self, img):
        """Extract VGG16 features for blood group prediction."""
        img = cv2.resize(img, (224, 224))
        img = preprocess_input(img)
        img = np.expand_dims(img, axis=0)
        feature = vgg_model_bg.predict(img, verbose=0)
        return feature.flatten()
    
    def extract_sift_features(self, img):
        """Extract SIFT features for blood group prediction."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, desc = sift.detectAndCompute(gray, None)
        return np.mean(desc, axis=0) if desc is not None else np.zeros(128)
    
    def extract_orb_features(self, img):
        """Extract ORB features for blood group prediction."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, desc = orb.detectAndCompute(gray, None)
        return np.mean(desc, axis=0) if desc is not None else np.zeros(32)
    
    def predict_age_gender(self):
        """Predict age and gender for the uploaded image."""
        # Load models
        age_model = joblib.load('age_model.pkl')
        gender_model = joblib.load('gender_model.pkl')

        # Extract features for the new image
        vgg_features = self.extract_vgg_ag_features(cv2.imread(self.image_path))
        lbp_features = self.extract_lbp_features(cv2.imread(self.image_path))
        combined_features = np.hstack([vgg_features, lbp_features])

        # Predict age and gender
        age_pred = age_model.predict([combined_features])[0]
        gender_pred = gender_model.predict([combined_features])[0]

        # Convert age to range
        age_range = self.get_age_range(age_pred)

        # Map gender prediction to label
        gender_label = "Male" if gender_pred == 1 else "Female"

        return age_range, gender_label
    
    def get_age_range(self, age):
        """Convert exact age to a range of 4 numbers (e.g., 20-24)."""
        lower = int(age // 5) * 5  # Round down to nearest multiple of 5
        upper = lower + 4
        return f"{lower}-{upper}"
    
    def predict_blood_group(self):
        """Predict blood group for the uploaded image."""
        # Read and process image
        img = cv2.imread(self.image_path)
        
        # Extract features
        vgg_features = self.extract_vgg_bg_features(img)
        sift_features = self.extract_sift_features(img)
        orb_features = self.extract_orb_features(img)
        combined = np.hstack([vgg_features, sift_features, orb_features])
        
        # Predict
        encoded = blood_group_model.predict([combined])[0]
        return le.inverse_transform([encoded])[0]

if __name__ == "__main__":
    root = tk.Tk()
    app = PredictionApp(root)
    root.mainloop()