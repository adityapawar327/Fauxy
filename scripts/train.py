#!/usr/bin/env python3
"""
Training script for AI Face Detector
Uses the Kaggle dataset to train a CNN model for detecting AI-generated faces
"""

import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from datetime import datetime
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set memory growth for GPU (if available)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"✅ GPU acceleration enabled: {len(gpus)} GPU(s) found")
    except RuntimeError as e:
        logger.warning(f"⚠️  GPU setup warning: {e}")
else:
    logger.info("🖥️  Using CPU for training")

class AIFaceTrainer:
    """Simple AI Face Detector Trainer"""
    
    def __init__(self, data_dir="dataset/organized", model_dir="model"):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Training parameters
        self.img_size = (224, 224)
        self.batch_size = 32
        self.epochs = 15
        self.learning_rate = 0.001
        
        # Model components
        self.model = None
        self.history = None
        
        logger.info(f"🏗️  Trainer initialized")
        logger.info(f"   Data directory: {self.data_dir}")
        logger.info(f"   Model directory: {self.model_dir}")
    
    def create_data_generators(self):
        """Create data generators for training and validation"""
        logger.info("🔄 Creating data generators...")
        
        # Check if organized dataset exists
        train_dir = self.data_dir / "train"
        val_dir = self.data_dir / "val"
        
        if not train_dir.exists() or not val_dir.exists():
            logger.error(f"❌ Dataset not found at {self.data_dir}")
            logger.info("💡 Please run: python download_dataset.py")
            return None, None
        
        # Data augmentation for training
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1,
            fill_mode='nearest'
        )
        
        # Only rescaling for validation
        val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            classes=['real', 'fake'],  # real=0, fake=1
            shuffle=True
        )
        
        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            classes=['real', 'fake'],  # real=0, fake=1
            shuffle=False
        )
        
        logger.info(f"✅ Data generators created:")
        logger.info(f"   Training samples: {train_generator.samples}")
        logger.info(f"   Validation samples: {val_generator.samples}")
        
        return train_generator, val_generator
    
    def create_model(self):
        """Create CNN model for AI face detection"""
        logger.info("🏗️  Building CNN model...")
        
        model = tf.keras.Sequential([
            # First block
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*self.img_size, 3)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Dropout(0.25),
            
            # Second block
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Dropout(0.25),
            
            # Third block
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Dropout(0.25),
            
            # Fourth block
            tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Dropout(0.25),
            
            # Dense layers
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        logger.info(f"✅ Model created with {model.count_params():,} parameters")
        return model
    
    def train(self):
        """Main training function"""
        logger.info("🚀 Starting training process...")
        
        # Create data generators
        train_gen, val_gen = self.create_data_generators()
        if train_gen is None:
            return None, None
        
        # Create model
        model = self.create_model()
        
        # Setup callbacks
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                self.model_dir / "best_model.h5",
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            )
        ]
        
        # Train model
        logger.info(f"🎯 Training for {self.epochs} epochs...")
        
        steps_per_epoch = max(1, train_gen.samples // self.batch_size)
        validation_steps = max(1, val_gen.samples // self.batch_size)
        
        history = model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=self.epochs,
            validation_data=val_gen,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate and save
        self._evaluate_model(model, val_gen)
        self._save_model(model)
        self._plot_training_history(history)
        
        logger.info("✅ Training completed successfully!")
        return model, history
    
    def _evaluate_model(self, model, val_generator):
        """Evaluate model performance"""
        logger.info("📊 Evaluating model...")
        
        # Get predictions
        val_generator.reset()
        predictions = model.predict(val_generator, verbose=1)
        y_pred = (predictions > 0.5).astype(int)
        y_true = val_generator.classes
        
        # Classification report
        unique_classes = np.unique(y_true)
        if len(unique_classes) == 1:
            # Handle single class case
            class_name = 'Fake' if unique_classes[0] == 1 else 'Real'
            accuracy = np.mean(y_pred == y_true)
            report = f"Single class dataset detected: {class_name}\nAccuracy: {accuracy:.4f}"
            logger.info(f"\n📈 Classification Report:\n{report}")
        else:
            report = classification_report(y_true, y_pred, target_names=['Real', 'Fake'])
            logger.info(f"\n📈 Classification Report:\n{report}")
        
        # Save report
        with open(self.model_dir / "classification_report.txt", "w") as f:
            f.write(report)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(self.model_dir / "confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("✅ Model evaluation completed")
    
    def _save_model(self, model):
        """Save the trained model"""
        logger.info("💾 Saving model...")
        
        # Load best model if it exists
        best_model_path = self.model_dir / "best_model.h5"
        if best_model_path.exists():
            model = tf.keras.models.load_model(best_model_path)
        
        # Save full model
        model_path = self.model_dir / "ai_face_detector.h5"
        model.save(model_path)
        
        # Save weights only
        weights_path = self.model_dir / "model_weights.h5"
        model.save_weights(weights_path)
        
        logger.info(f"✅ Model saved:")
        logger.info(f"   Full model: {model_path}")
        logger.info(f"   Weights: {weights_path}")
    
    def _plot_training_history(self, history):
        """Plot training history"""
        logger.info("📊 Plotting training history...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.model_dir / "training_history.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("✅ Training history plots saved")

def main():
    """Main training function"""
    print("🚀 AI Face Detector Training")
    print("=" * 50)
    
    # Check if dataset exists
    data_dir = Path("dataset/organized")
    if not data_dir.exists() or not any(data_dir.iterdir()):
        print("❌ Dataset not found!")
        print("\n📥 Please download the dataset first:")
        print("1. Run: python download_dataset.py")
        print("2. Or manually download from:")
        print("   https://www.kaggle.com/datasets/shahzaibshazoo/detect-ai-generated-faces-high-quality-dataset")
        print("3. Extract to the 'dataset/' folder")
        return
    
    # Initialize trainer
    trainer = AIFaceTrainer()
    
    # Create model directory
    trainer.model_dir.mkdir(exist_ok=True)
    
    print(f"📂 Data directory: {data_dir.absolute()}")
    print(f"📂 Model directory: {trainer.model_dir.absolute()}")
    
    # Start training
    try:
        model, history = trainer.train()
        
        if model is not None:
            print("\n🎉 Training completed successfully!")
            print("\n📊 Model Performance Summary:")
            print("- Check 'model/confusion_matrix.png' for detailed results")
            print("- Check 'model/training_history.png' for training curves")
            print("- Check 'model/roc_curve.png' for ROC analysis")
            print("\n🚀 Ready to use! Start the application with:")
            print("  python main.py")
        else:
            print("❌ Training failed. Please check the error messages above.")
            
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()