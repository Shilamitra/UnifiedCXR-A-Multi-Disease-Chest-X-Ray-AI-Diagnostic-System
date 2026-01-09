# test_models.py
import tensorflow as tf
import numpy as np

def test_model(model_path, model_name):
    try:
        print(f"🧪 Testing {model_name}...")
        model = tf.keras.models.load_model(model_path)
        print(f"✅ {model_name} loaded successfully!")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
        
        # Test prediction with dummy data
        if model.input_shape[1] == 224:
            dummy_input = np.random.random((1, 224, 224, 3))
        elif model.input_shape[1] == 128:
            dummy_input = np.random.random((1, 128, 128, 3))
        else:
            dummy_input = np.random.random((1, 256, 256, 3))
            
        prediction = model.predict(dummy_input)
        print(f"   Prediction test: {prediction}")
        return True
        
    except Exception as e:
        print(f"❌ {model_name} failed: {e}")
        return False

# Test both models
test_model('models/vgg_unfrozen.h5', 'Pneumonia Model')
test_model('models/tb_detector.h5', 'TB Model')