#!/usr/bin/env python3
"""
Training script for balanced AI Face Detector
Uses both real and AI-generated faces for better classification
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

class BalancedAIFaceTrainer:
    """Balanced AI Face Detector Trainer with real and fake faces"""
    
    def __init__(self, data_dir="dataset/balanced", model_dir="backend/model"):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Training parameters
        self.img_size = (224, 224)
        self.batch_size = 32
        self.epochs = 20
        self.learning_rate = 0.0001
        
        # Model components
        self.model = None
        self.history = None
        
        logger.info(f"🏗️  Balanced trainer initialized")
        logger.info(f"   Data directory: {self.data_dir}")
        logger.info(f"   Model directory: {self.model_dir}")
    
    def create_data_generators(self):
        """Create data generators for training and validation"""
        logger.info("🔄 Creating balanced data generators...")
        
        # Check if balanced dataset exists
        train_dir = self.data_dir / "train"
        val_dir = self.data_dir / "val"
        
        if not train_dir.exists() or not val_dir.exists():
            logger.error(f"❌ Balanced dataset not found at {self.data_dir}")
            logger.info("💡 Please run: python download_balanced_dataset.py")
            return None, None
        
        # Check class balance
        for split_name, split_dir in [("train", train_dir), ("val", val_dir)]:
            real_count = len(list((split_dir / "real").glob("*.jpg")))
            fake_count = len(list((split_dir / "fake").glob("*.jpg")))
            logger.info(f"   {split_name}: {real_count} real, {fake_count} fake images")
        
        # Data augmentation for training (more conservative for faces)
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=10,
            width_shift_range=0.05,
            height_shift_range=0.05,
            horizontal_flip=True,
            zoom_range=0.05,
            brightness_range=[0.9, 1.1],
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
            shuffle=True,
            seed=42
        )
        
        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            classes=['real', 'fake'],  # real=0, fake=1
            shuffle=False,
            seed=42
        )
        
        logger.info(f"✅ Balanced data generators created:")
        logger.info(f"   Training samples: {train_generator.samples}")
        logger.info(f"   Validation samples: {val_generator.samples}")
        logger.info(f"   Class indices: {train_generator.class_indices}")
        
        return train_generator, val_generator
    
    def create_model(self):
        """Create improved CNN model for balanced AI face detection"""
        logger.info("🏗️  Building improved CNN model...")
        
        model = tf.keras.Sequential([
            # Input layer
            tf.keras.layers.Input(shape=(*self.img_size, 3)),
            
            # First block
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Dropout(0.25),
            
            # Second block
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Dropout(0.25),
            
            # Third block
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Dropout(0.25),
            
            # Fourth block
            tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Dropout(0.25),
            
            # Global average pooling instead of flatten
            tf.keras.layers.GlobalAveragePooling2D(),
            
            # Dense layers
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile model with lower learning rate for better convergence
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self.model = model
        logger.info(f"✅ Improved model created with {model.count_params():,} parameters")
        return model
    
    def train(self):
        """Main training function for balanced dataset"""
        logger.info("🚀 Starting balanced training process...")
        
        # Create data generators
        train_gen, val_gen = self.create_data_generators()
        if train_gen is None:
            return None, None
        
        # Create model
        model = self.create_model()
        
        # Setup callbacks
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                self.model_dir / "best_balanced_model.h5",
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=7,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train model
        logger.info(f"🎯 Training for up to {self.epochs} epochs...")
        
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
        
        logger.info("✅ Balanced training completed successfully!")
        return model, history
    
    def _evaluate_model(self, model, val_generator):
        """Evaluate model performance on balanced dataset"""
        logger.info("📊 Evaluating balanced model...")
        
        # Get predictions
        val_generator.reset()
        predictions = model.predict(val_generator, verbose=1)
        y_pred = (predictions > 0.5).astype(int)
        y_true = val_generator.classes
        
        # Classification report
        unique_classes = np.unique(y_true)
        if len(unique_classes) > 1:
            report = classification_report(y_true, y_pred, target_names=['Real', 'Fake'])
            logger.info(f"\n📈 Classification Report:\n{report}")
        else:
            class_name = 'Fake' if unique_classes[0] == 1 else 'Real'
            accuracy = np.mean(y_pred == y_true)
            report = f"Single class dataset: {class_name}\nAccuracy: {accuracy:.4f}"
            logger.info(f"\n📈 Classification Report:\n{report}")
        
        # Save report
        with open(self.model_dir / "balanced_classification_report.txt", "w") as f:
            f.write(report)
        
        # Confusion matrix
        if len(unique_classes) > 1:
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
            plt.title('Confusion Matrix - Balanced Model')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(self.model_dir / "balanced_confusion_matrix.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info("✅ Balanced model evaluation completed")
    
    def _save_model(self, model):
        """Save the trained balanced model"""
        logger.info("💾 Saving balanced model...")
        
        # Load best model if it exists
        best_model_path = self.model_dir / "best_balanced_model.h5"
        if best_model_path.exists():
            model = tf.keras.models.load_model(best_model_path)
        
        # Save as the main model (overwrite previous)
        model_path = self.model_dir / "ai_face_detector.h5"
        model.save(model_path)
        
        # Save weights
        weights_path = self.model_dir / "model_weights.h5"
        model.save_weights(weights_path)
        
        # Save backup
        backup_path = self.model_dir / "balanced_ai_face_detector.h5"
        model.save(backup_path)
        
        logger.info(f"✅ Balanced model saved:")
        logger.info(f"   Main model: {model_path}")
        logger.info(f"   Backup: {backup_path}")
        logger.info(f"   Weights: {weights_path}")
    
    def _plot_training_history(self, history):
        """Plot training history for balanced model"""
        logger.info("📊 Plotting balanced training history...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
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
        
        # Precision
        if 'precision' in history.history:
            ax3.plot(history.history['precision'], label='Training Precision')
            ax3.plot(history.history['val_precision'], label='Validation Precision')
            ax3.set_title('Model Precision')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Precision')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Recall
        if 'recall' in history.history:
            ax4.plot(history.history['recall'], label='Training Recall')
            ax4.plot(history.history['val_recall'], label='Validation Recall')
            ax4.set_title('Model Recall')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Recall')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.model_dir / "balanced_training_history.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("✅ Balanced training history plots saved")

def main():
    """Main training function for balanced dataset"""
    print("🚀 Balanced AI Face Detector Training")
    print("=" * 60)
    
    # Check if balanced dataset exists
    data_dir = Path("dataset/balanced")
    if not data_dir.exists():
        print("❌ Balanced dataset not found!")
        print("\n📥 Please download the balanced dataset first:")
        print("   python download_balanced_dataset.py")
        return
    
    # Check dataset structure
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    
    if not (train_dir.exists() and val_dir.exists()):
        print("❌ Dataset structure incomplete!")
        print("   Expected: dataset/balanced/train/ and dataset/balanced/val/")
        return
    
    # Initialize trainer
    trainer = BalancedAIFaceTrainer()
    
    print(f"📂 Data directory: {data_dir.absolute()}")
    print(f"📂 Model directory: {trainer.model_dir.absolute()}")
    
    # Start training
    try:
        model, history = trainer.train()
        
        if model is not None:
            print("\n🎉 Balanced training completed successfully!")
            print("\n📊 Model Performance Summary:")
            print("- Check 'model/balanced_confusion_matrix.png' for detailed results")
            print("- Check 'model/balanced_training_history.png' for training curves")
            print("- Model saved as 'model/ai_face_detector.h5'")
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