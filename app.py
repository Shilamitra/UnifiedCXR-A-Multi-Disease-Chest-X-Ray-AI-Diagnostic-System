from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import logging
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global variables for models
pneumonia_model = None
tb_model = None
cardiomegaly_model = None

def create_exact_tb_model():
    """Create the exact VGG16-based TB model architecture based on the layer names found"""
    logger.info("🔧 Creating exact VGG16-based TB model...")
    
    # Load VGG16 base (without top layers)
    vgg_base = tf.keras.applications.VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    vgg_base.trainable = False
    
    model = tf.keras.Sequential([
        vgg_base,
        tf.keras.layers.Flatten(name='flatten_1'),
        tf.keras.layers.Dense(256, activation='relu', name='dense_2'),
        tf.keras.layers.Dropout(0.5, name='dropout_1'),
        tf.keras.layers.Dense(1, activation='sigmoid', name='dense_3')
    ])
    
    return model

def load_tb_model_exact():
    """Load TB model with exact VGG16-based architecture"""
    global tb_model
    
    try:
        logger.info("🔄 Loading TB Model with VGG16 architecture...")
        tb_model = create_exact_tb_model()
        tb_weights_path = 'models/tb_detector.h5'
        if os.path.exists(tb_weights_path):
            tb_model.load_weights(tb_weights_path, by_name=True, skip_mismatch=True)
            tb_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            logger.info("✅ TB Model loaded successfully with VGG16 architecture!")
            return True
        else:
            logger.warning(f"⚠️ TB weights not found at {tb_weights_path}")
            tb_model = None
            return False
        
    except Exception as e:
        logger.error(f"❌ VGG16 architecture load failed: {str(e)}")
        # fallback: functional API
        try:
            logger.info("🔄 Trying functional API approach for TB model...")
            vgg_base = tf.keras.applications.VGG16(
                weights='imagenet',
                include_top=False,
                input_shape=(224, 224, 3)
            )
            vgg_base.trainable = False
            x = vgg_base.output
            x = tf.keras.layers.Flatten(name='flatten_1')(x)
            x = tf.keras.layers.Dense(256, activation='relu', name='dense_2')(x)
            x = tf.keras.layers.Dropout(0.5, name='dropout_1')(x)
            predictions = tf.keras.layers.Dense(1, activation='sigmoid', name='dense_3')(x)
            tb_model = tf.keras.Model(inputs=vgg_base.input, outputs=predictions)
            tb_weights_path = 'models/tb_detector.h5'
            if os.path.exists(tb_weights_path):
                tb_model.load_weights(tb_weights_path)
                tb_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                logger.info("✅ TB Model loaded with functional API!")
                return True
            else:
                logger.warning(f"⚠️ TB weights not found at {tb_weights_path}")
                tb_model = None
                return False
        except Exception as e2:
            logger.error(f"❌ Functional API also failed: {str(e2)}")
            tb_model = None
            return False

def load_cardiomegaly_model():
    """Load EfficientNet cardiomegaly model"""
    global cardiomegaly_model

    try:
        logger.info("🔄 Loading Cardiomegaly Model...")
        model_path = 'models/efficientnet_cardiomegaly.h5'
        if not os.path.exists(model_path):
            logger.warning(f"⚠️ Cardiomegaly model file missing at {model_path}")
            return False

        # load_model will restore the full model if saved as .h5
        cardiomegaly_model = tf.keras.models.load_model(model_path)
        # optional: compile if needed; some TF versions require compile() before predict() for custom losses/metrics
        try:
            cardiomegaly_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        except Exception:
            # not critical; prediction usually works without compiling
            pass

        logger.info("✅ Cardiomegaly Model loaded successfully!")
        return True

    except Exception as e:
        logger.error(f"❌ Cardiomegaly model load failed: {str(e)}")
        cardiomegaly_model = None
        return False

def load_models():
    """Load all available models (pneumonia, TB, cardiomegaly). Returns True if at least one model loads."""
    global pneumonia_model, tb_model, cardiomegaly_model
    
    logger.info("🔄 Starting model loading...")
    
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    
    # Track successes
    any_loaded = False

    # Pneumonia model
    pneumonia_path = os.path.join(models_dir, 'vgg_unfrozen.h5')
    if os.path.exists(pneumonia_path):
        try:
            pneumonia_model = tf.keras.models.load_model(pneumonia_path)
            logger.info("✅ Pneumonia Model loaded successfully!")
            any_loaded = True
        except Exception as e:
            logger.error(f"❌ Pneumonia model failed to load: {str(e)}")
            pneumonia_model = None
    else:
        logger.warning(f"⚠️ Pneumonia model not found at {pneumonia_path}")

    # TB model
    tb_loaded = load_tb_model_exact()
    if tb_loaded:
        any_loaded = True

    # Cardiomegaly model
    cardio_loaded = load_cardiomegaly_model()
    if cardio_loaded:
        any_loaded = True

    if any_loaded:
        logger.info("✅ One or more models loaded successfully!")
    else:
        logger.error("❌ No models loaded. Check model files in the models/ directory.")
    
    return any_loaded

def preprocess_image(image_file, target_size=(224, 224)):
    """Preprocess uploaded image file"""
    try:
        if hasattr(image_file, 'seek'):
            try:
                image_file.seek(0)
            except Exception:
                pass
        
        if isinstance(image_file, bytes):
            image = Image.open(io.BytesIO(image_file)).convert('RGB')
        else:
            image = Image.open(image_file).convert('RGB')
        
        image = image.resize(target_size)
        img_array = np.array(image).astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        logger.error(f"❌ Error preprocessing image: {str(e)}")
        return None

def predict_pneumonia(image_file):
    """Predict pneumonia"""
    try:
        if pneumonia_model is None:
            return {
                "disease": "Pneumonia",
                "result": "Model Not Loaded", 
                "confidence": 0.0,
                "has_disease": False,
                "status": "error"
            }
        
        processed_image = preprocess_image(image_file, target_size=(128, 128))
        if processed_image is None:
            return {"error": "Image preprocessing failed", "status": "error"}
        
        prediction = pneumonia_model.predict(processed_image, verbose=0)
        if len(prediction.shape) > 1 and prediction.shape[1] == 2:
            confidence = float(prediction[0][1])
        else:
            confidence = float(prediction[0][0])
            
        has_pneumonia = confidence > 0.5
        result = "Pneumonia Detected" if has_pneumonia else "Normal"
        
        return {
            "disease": "Pneumonia",
            "result": result,
            "confidence": round(confidence, 4),
            "has_disease": has_pneumonia,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"❌ Pneumonia prediction error: {str(e)}")
        return {
            "disease": "Pneumonia",
            "result": "Prediction Failed",
            "confidence": 0.0,
            "has_disease": False,
            "status": "error"
        }

def predict_tuberculosis(image_file):
    """Predict tuberculosis"""
    try:
        if tb_model is None:
            return {
                "disease": "Tuberculosis", 
                "result": "Model Not Loaded",
                "confidence": 0.0,
                "has_disease": False,
                "status": "error"
            }
        
        processed_image = preprocess_image(image_file, target_size=(224, 224))
        if processed_image is None:
            return {"error": "Image preprocessing failed", "status": "error"}
        
        prediction = tb_model.predict(processed_image, verbose=0)
        confidence = float(prediction[0][0])
        has_tb = confidence > 0.5
        result = "Tuberculosis Detected" if has_tb else "Normal"
        
        return {
            "disease": "Tuberculosis",
            "result": result,
            "confidence": round(confidence, 4),
            "has_disease": has_tb,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"❌ TB prediction error: {str(e)}")
        return {
            "disease": "Tuberculosis",
            "result": "Prediction Failed", 
            "confidence": 0.0,
            "has_disease": False,
            "status": "error"
        }

def predict_cardiomegaly(image_file):
    """Predict cardiomegaly using EfficientNet model"""
    try:
        if cardiomegaly_model is None:
            return {
                "disease": "Cardiomegaly",
                "result": "Model Not Loaded",
                "confidence": 0.0,
                "has_disease": False,
                "status": "error"
            }

        # EfficientNet usually expects 224x224 (adjust if your model expects different)
        processed_image = preprocess_image(image_file, target_size=(224, 224))
        if processed_image is None:
            return {"error": "Image preprocessing failed", "status": "error"}

        prediction = cardiomegaly_model.predict(processed_image, verbose=0)
        # Handle shape differences (binary or single-output)
        try:
            confidence = float(prediction[0][0])
        except Exception:
            # if model outputs probabilities in [0,1] as a single scalar
            confidence = float(np.squeeze(prediction))

        has_cardio = confidence > 0.5
        result = "Cardiomegaly Detected" if has_cardio else "Normal"

        return {
            "disease": "Cardiomegaly",
            "result": result,
            "confidence": round(confidence, 4),
            "has_disease": has_cardio,
            "status": "success"
        }

    except Exception as e:
        logger.error(f"❌ Cardiomegaly prediction error: {str(e)}")
        return {
            "disease": "Cardiomegaly",
            "result": "Prediction Failed",
            "confidence": 0.0,
            "has_disease": False,
            "status": "error"
        }

@app.route('/')
def home():
    return jsonify({"message": "unifiedCXR API - Medical AI Assistant"})

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded', 'status': 'error'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected', 'status': 'error'}), 400
        
        # Validate file type
        allowed_extensions = {'jpg', 'jpeg', 'png'}
        file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        if file_ext not in allowed_extensions:
            return jsonify({
                'error': 'Invalid file type. Please use JPG, JPEG, or PNG.',
                'status': 'error'
            }), 400
        
        logger.info(f"📊 Processing image: {file.filename}")
        file_data = file.read()
        
        # Run predictions
        pneumonia_result = predict_pneumonia(file_data)
        tb_result = predict_tuberculosis(file_data)
        cardiomegaly_result = predict_cardiomegaly(file_data)
        
        results = {
            'pneumonia': pneumonia_result,
            'tuberculosis': tb_result,
            'cardiomegaly': cardiomegaly_result,
            'status': 'success'
        }
        
        logger.info("✅ Prediction completed")
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"❌ Prediction error: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}', 'status': 'error'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'pneumonia_model_loaded': pneumonia_model is not None,
        'tb_model_loaded': tb_model is not None,
        'cardiomegaly_model_loaded': cardiomegaly_model is not None,
        'message': 'Medical AI Assistant Ready - models may vary'
    })

# Initialize models
logger.info("🚀 Starting unifiedCXR Medical AI Server...")
load_models()

logger.info("🔍 Screening for: Pneumonia, Tuberculosis, Cardiomegaly") 
logger.info("🌐 Server starting on: http://localhost:5500")

if __name__ == '__main__':
    # Use threaded=True to allow concurrent requests (basic responsiveness)
    app.run(host='0.0.0.0', port=5500, debug=True, threaded=True)
