import os
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import logging
from bloodgroup_model import predict_blood_group
from age_gender_model import predict_age_gender

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Add project directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure upload folder
UPLOAD_FOLDER = 'Uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def predict():
    logger.debug("Received prediction request")
    
    if 'image' not in request.files:
        logger.error("No image provided in request")
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        logger.error("No image selected")
        return jsonify({'error': 'No image selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(filepath)
            logger.debug(f"Image saved to {filepath}")
            
            # Predict blood group
            logger.debug("Predicting blood group")
            blood_group = predict_blood_group(filepath)
            
            # Predict age and gender
            logger.debug("Predicting age and gender")
            age, gender = predict_age_gender(filepath)
            
            # Clean up
            os.remove(filepath)
            logger.debug("Temporary image file removed")
            
            return jsonify({
                'blood_group': blood_group,
                'age': float(age),
                'gender': str(gender)
            })
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': str(e)}), 500
    
    logger.error("Invalid file type")
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/')
def index():
    return jsonify({'message': 'Welcome to the NIR Hand Analysis API. Use /predict for image predictions.'})

@app.route('/favicon.ico')
def favicon():
    return '', 204

if __name__ == '__main__':
    logger.info("Starting Flask server")
    app.run(debug=True, port=5000)