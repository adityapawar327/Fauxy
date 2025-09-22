import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import asyncio
import os
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class AIFaceDetector:
    """
    High-performance AI face detector using CNN architecture
    Trained on the Kaggle dataset for detecting AI-generated faces
    """
    
    def __init__(self):
        self.model = None
        self.model_loaded = False
        self.total_analyzed = 0
        self.input_size = (224, 224)
        
    async def load_model(self):
        """Load the pre-trained model"""
        try:
            model_path = "model/ai_face_detector.h5"
            weights_path = "model/model_weights.h5"
            
            # Try to load the trained model
            if os.path.exists(model_path):
                logger.info(f"Loading trained model from {model_path}")
                self.model = tf.keras.models.load_model(model_path)
                self.model_loaded = True
                logger.info("✅ Trained AI Face Detector model loaded successfully")
            elif os.path.exists(weights_path):
                logger.info("Loading model weights...")
                self.model = self._create_demo_model()
                self.model.load_weights(weights_path)
                self.model_loaded = True
                logger.info("✅ Model weights loaded successfully")
            else:
                logger.warning("⚠️ No trained model found, using demo model")
                logger.info("To train a real model, run: python scripts/train_model.py")
                self.model = self._create_demo_model()
                self.model_loaded = True
                logger.info("✅ Demo model loaded (for testing purposes)")
                
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            # Fallback to demo model
            try:
                self.model = self._create_demo_model()
                self.model_loaded = True
                logger.info("✅ Fallback demo model loaded")
            except Exception as fallback_error:
                logger.error(f"❌ Even demo model failed: {fallback_error}")
                raise
    
    def _create_demo_model(self):
        """
        Create a demo CNN model architecture
        In production, replace this with your trained model
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Preprocess image for model input"""
        # Resize image
        image = image.resize(self.input_size)
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Normalize pixel values
        img_array = img_array.astype(np.float32) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def _analyze_artifacts(self, image: Image.Image, prediction_prob: float) -> Dict[str, str]:
        """
        Analyze image for AI generation artifacts
        Enhanced analysis based on actual image properties and model prediction
        """
        img_array = np.array(image)
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Calculate image statistics
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        
        # Calculate gradient magnitude (edge detection)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        edge_density = np.mean(gradient_magnitude)
        
        # Analyze color distribution
        color_std = np.std(img_array, axis=(0, 1))
        color_uniformity = np.mean(color_std)
        
        # Frequency domain analysis
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        freq_energy = np.mean(magnitude_spectrum)
        
        # Determine artifacts based on analysis and model prediction
        if prediction_prob > 0.7:  # High AI probability
            artifacts = "High" if edge_density < 15 or color_uniformity < 20 else "Medium"
            consistency = "Low" if std_intensity < 30 else "Medium"
            texture = "Synthetic" if freq_energy < 8 else "Processed"
        elif prediction_prob > 0.3:  # Medium AI probability
            artifacts = "Medium" if edge_density < 20 else "Low"
            consistency = "Medium"
            texture = "Processed" if freq_energy < 10 else "Natural"
        else:  # Low AI probability (likely real)
            artifacts = "Low"
            consistency = "High" if std_intensity > 40 else "Medium"
            texture = "Natural"
        
        # Symmetry analysis (simplified)
        height, width = gray.shape
        left_half = gray[:, :width//2]
        right_half = np.fliplr(gray[:, width//2:])
        
        if left_half.shape == right_half.shape:
            symmetry_score = np.corrcoef(left_half.flatten(), right_half.flatten())[0, 1]
            if symmetry_score > 0.8:
                symmetry = "High" if prediction_prob > 0.5 else "Normal"
            elif symmetry_score > 0.6:
                symmetry = "Normal"
            else:
                symmetry = "Low"
        else:
            symmetry = "Normal"
        
        return {
            "artifacts": artifacts,
            "consistency": consistency,
            "texture": texture,
            "symmetry": symmetry
        }
    
    async def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Predict if an image contains an AI-generated face
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")
        
        try:
            # Preprocess image
            processed_image = self._preprocess_image(image)
            
            # Make prediction
            prediction_prob = self.model.predict(processed_image, verbose=0)[0][0]
            
            # Determine if AI generated (threshold = 0.5)
            is_ai_generated = prediction_prob > 0.5
            confidence = float(prediction_prob if is_ai_generated else 1 - prediction_prob)
            
            # Ensure confidence is reasonable (between 0.5 and 1.0)
            confidence = max(0.5, min(1.0, confidence))
            
            prediction_label = "AI Generated" if is_ai_generated else "Real/Authentic"
            
            # Analyze artifacts based on actual image analysis
            artifacts = self._analyze_artifacts(image, prediction_prob)
            
            # Update statistics
            self.total_analyzed += 1
            
            result = {
                "prediction": prediction_label,
                "confidence": confidence,
                "raw_prediction": float(prediction_prob),
                **artifacts
            }
            
            logger.info(f"Prediction: {prediction_label}, Confidence: {confidence:.3f}, Raw: {prediction_prob:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if not self.model_loaded:
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "input_shape": self.input_size,
            "total_parameters": self.model.count_params() if self.model else 0,
            "architecture": "CNN",
            "version": "2.1"
        }