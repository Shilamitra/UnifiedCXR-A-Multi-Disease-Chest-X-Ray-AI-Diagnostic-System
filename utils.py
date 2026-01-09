import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import logging
import os

logger = logging.getLogger(__name__)

def load_custom_model(model_path):
    """
    Load a TensorFlow/Keras model with error handling and support for custom objects
    
    Args:
        model_path (str): Path to the .h5 model file
        
    Returns:
        model: Loaded Keras model or None if failed
    """
    try:
        # Try loading with different approaches
        
        # Approach 1: Standard load
        try:
            model = tf.keras.models.load_model(model_path)
            logger.info(f"Model loaded successfully with standard method: {model_path}")
            return model
        except Exception as e1:
            logger.warning(f"Standard load failed for {model_path}: {str(e1)}")
            
        # Approach 2: Load with custom objects (if you have custom layers)
        try:
            # If you have custom layers, define them here:
            custom_objects = {
                # 'CustomLayer': CustomLayer,  # Uncomment and replace with your custom layers
            }
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
            logger.info(f"Model loaded successfully with custom objects: {model_path}")
            return model
        except Exception as e2:
            logger.warning(f"Load with custom objects failed for {model_path}: {str(e2)}")
            
        # Approach 3: Load weights only (if architecture is known)
        # This requires you to know the model architecture
        # model = create_model_architecture()
        # model.load_weights(model_path)
        
        logger.error(f"All loading methods failed for {model_path}")
        return None
        
    except Exception as e:
        logger.error(f"Critical error loading model {model_path}: {str(e)}")
        return None

def preprocess_for_model(pil_image, target_size=(224, 224)):
    """
    Preprocess image for model prediction
    
    Args:
        pil_image (PIL.Image): Input image
        target_size (tuple): Target size (width, height)
        
    Returns:
        numpy.array: Preprocessed image array ready for model prediction
    """
    try:
        # Resize image to target size
        resized_image = pil_image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        image_array = np.array(resized_image)
        
        # Ensure 3 channels (RGB)
        if len(image_array.shape) == 2:  # Grayscale
            image_array = np.stack([image_array] * 3, axis=-1)
        elif image_array.shape[2] == 4:  # RGBA
            image_array = image_array[:, :, :3]
        
        # Normalize pixel values to [0, 1]
        image_array = image_array.astype(np.float32) / 255.0
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise e

def get_model_input_size(model):
    """
    Extract input size from model architecture
    
    Args:
        model: Keras model
        
    Returns:
        tuple: (height, width) of model input
    """
    try:
        input_shape = model.input_shape
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        
        # Assuming shape is (batch, height, width, channels)
        if len(input_shape) == 4:
            return input_shape[1], input_shape[2]  # (height, width)
        else:
            return (224, 224)  # Default size
    except:
        return (224, 224)  # Default size

def validate_image_file(file_path):
    """
    Validate if file is a valid image
    
    Args:
        file_path (str): Path to image file
        
    Returns:
        bool: True if valid image, False otherwise
    """
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except Exception as e:
        logger.error(f"Invalid image file {file_path}: {str(e)}")
        return False

def create_grad_cam_heatmap(model, image_array, layer_name=None):
    """
    Generate Grad-CAM heatmap for model interpretability
    (Advanced feature - can be implemented later)
    
    Args:
        model: Keras model
        image_array: Preprocessed image array
        layer_name: Name of the layer to use for Grad-CAM
        
    Returns:
        numpy.array: Heatmap array
    """
    # This is a placeholder for Grad-CAM implementation
    # You can implement this based on your specific model architecture
    
    logger.info("Grad-CAM not implemented yet")
    return None

def save_uploaded_image(file, save_dir='uploads'):
    """
    Save uploaded image to disk (optional - for debugging)
    
    Args:
        file: Flask file object
        save_dir (str): Directory to save images
        
    Returns:
        str: Path to saved image
    """
    try:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        filename = f"{np.random.randint(10000, 99999)}_{file.filename}"
        file_path = os.path.join(save_dir, filename)
        
        file.save(file_path)
        logger.info(f"Uploaded image saved to: {file_path}")
        return file_path
        
    except Exception as e:
        logger.error(f"Error saving uploaded image: {str(e)}")
        return None

def cleanup_old_files(directory='uploads', max_age_hours=24):
    """
    Clean up old uploaded files (optional - for maintenance)
    
    Args:
        directory (str): Directory to clean
        max_age_hours (int): Maximum age of files in hours
    """
    try:
        if not os.path.exists(directory):
            return
            
        import time
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                file_age = current_time - os.path.getctime(file_path)
                if file_age > max_age_seconds:
                    os.remove(file_path)
                    logger.info(f"Removed old file: {file_path}")
                    
    except Exception as e:
        logger.error(f"Error cleaning up files: {str(e)}")