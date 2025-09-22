#!/usr/bin/env python3
"""
Training script for AI Face Detector
Uses the Kaggle dataset to train a CNN model
"""

import tensorflow as tf
import numpy as np
import os
from pathlib import Path
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class AIFaceTrainer:
    """Trainer class for AI face detection model"""
    
    def __init__(self, data_dir="data", model_dir="model"):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.img_size = (224, 224)
        self.batch_size = 32
        self.epochs = 50
        
    def load_dataset(self):
        """Load and preprocess the Kaggle dataset"""
        print("Loading Kaggle dataset...")
        
        images = []
        labels = []
        
        # The Kaggle dataset structure
        # Real faces are in: data/real/
        # AI faces are in: data/fake/ or data/ai/
        
        dataset_paths = [
            (self.data_dir / "real", 0, "Real"),
            (self.data_dir / "fake", 1, "AI Generated"),
            (self.data_dir / "ai", 1, "AI Generated"),
            (self.data_dir / "artificial", 1, "AI Generated"),
        ]
        
        for folder_path, label, label_name in dataset_paths:
            if folder_path.exists():
                print(f"Loading {label_name} images from {folder_path}")
                count = 0
                
                # Support multiple image formats
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                    for img_path in folder_path.glob(ext):
                        try:
                            img = cv2.imread(str(img_path))
                            if img is not None:
                                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                img = cv2.resize(img, self.img_size)
                                
                                # Basic image quality check
                                if img.mean() > 10:  # Avoid completely black images
                                    images.append(img)
                                    labels.append(label)
                                    count += 1
                        except Exception as e:
                            print(f"Error loading {img_path}: {e}")
                            continue
                
                print(f"Loaded {count} {label_name} images")
        
        if len(images) == 0:
            print("❌ No images found! Please check dataset structure:")
            print("Expected structure:")
            print("data/")
            print("├── real/     (real face images)")
            print("└── fake/     (AI-generated face images)")
            print("\nOr run: python scripts/setup.py to download the dataset")
            return None, None
        
        # Convert to numpy arrays
        X = np.array(images, dtype=np.float32) / 255.0
        y = np.array(labels)
        
        print(f"\n📊 Dataset Summary:")
        print(f"Total images: {len(X)}")
        print(f"Real faces: {np.sum(y == 0)} ({np.sum(y == 0)/len(y)*100:.1f}%)")
        print(f"AI faces: {np.sum(y == 1)} ({np.sum(y == 1)/len(y)*100:.1f}%)")
        print(f"Image shape: {X[0].shape}")
        
        return X, y
    
    def create_model(self):
        """Create an advanced CNN model for AI face detection"""
        
        # Use transfer learning with EfficientNetB0 as base
        base_model = tf.keras.applications.EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.img_size, 3)
        )
        
        # Freeze base model initially
        base_model.trainable = False
        
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model, base_model
    
    def create_simple_cnn(self):
        """Create a simpler CNN model if transfer learning fails"""
        model = tf.keras.Sequential([
            # First convolutional block
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*self.img_size, 3)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Dropout(0.25),
            
            # Second convolutional block
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Dropout(0.25),
            
            # Third convolutional block
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Dropout(0.25),
            
            # Fourth convolutional block
            tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Dropout(0.25),
            
            # Dense layers
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def train(self):
        """Train the model with the Kaggle dataset"""
        # Load dataset
        X, y = self.load_dataset()
        
        if X is None:
            return None, None
        
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        print(f"\n📊 Data Split:")
        print(f"Training set: {len(X_train)} images")
        print(f"Validation set: {len(X_val)} images")
        print(f"Test set: {len(X_test)} images")
        
        # Try transfer learning first, fallback to simple CNN
        try:
            print("\n🚀 Creating model with transfer learning (EfficientNetB0)...")
            model, base_model = self.create_model()
            use_transfer_learning = True
        except Exception as e:
            print(f"Transfer learning failed: {e}")
            print("🔄 Falling back to simple CNN...")
            model = self.create_simple_cnn()
            base_model = None
            use_transfer_learning = False
        
        model.summary()
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                str(self.model_dir / "best_model.h5"),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7,
                verbose=1
            ),
            tf.keras.callbacks.CSVLogger(
                str(self.model_dir / "training_log.csv")
            )
        ]
        
        # Data augmentation for AI face detection
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.15,
            height_shift_range=0.15,
            horizontal_flip=True,
            zoom_range=0.15,
            shear_range=0.1,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )
        
        # Phase 1: Train with frozen base (if using transfer learning)
        if use_transfer_learning:
            print("\n🎯 Phase 1: Training with frozen base model...")
            history1 = model.fit(
                datagen.flow(X_train, y_train, batch_size=self.batch_size),
                epochs=20,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1
            )
            
            # Phase 2: Fine-tune with unfrozen base
            print("\n🔥 Phase 2: Fine-tuning with unfrozen base model...")
            base_model.trainable = True
            
            # Use lower learning rate for fine-tuning
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss='binary_crossentropy',
                metrics=['accuracy', 'precision', 'recall']
            )
            
            history2 = model.fit(
                datagen.flow(X_train, y_train, batch_size=self.batch_size),
                epochs=self.epochs - 20,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1
            )
            
            # Combine histories
            history = self.combine_histories(history1, history2)
        else:
            # Train simple CNN
            print("\n🎯 Training simple CNN model...")
            history = model.fit(
                datagen.flow(X_train, y_train, batch_size=self.batch_size),
                epochs=self.epochs,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1
            )
        
        # Load best model for evaluation
        model.load_weights(str(self.model_dir / "best_model.h5"))
        
        # Evaluate on test set
        print("\n📊 Evaluating on test set...")
        test_results = model.evaluate(X_test, y_test, verbose=0)
        test_loss, test_acc = test_results[0], test_results[1]
        test_precision, test_recall = test_results[2], test_results[3]
        
        print(f"\n🎉 Final Test Results:")
        print(f"Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        print(f"Precision: {test_precision:.4f}")
        print(f"Recall: {test_recall:.4f}")
        f1_score = 2 * (test_precision * test_recall) / (test_precision + test_recall) if (test_precision + test_recall) > 0 else 0
        print(f"F1-Score: {f1_score:.4f}")
        
        # Generate predictions for detailed analysis
        print("\n🔍 Generating detailed analysis...")
        y_pred_prob = model.predict(X_test, verbose=0)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        
        # Classification report
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Real', 'AI Generated']))
        
        # Save detailed results
        self.save_evaluation_results(y_test, y_pred, y_pred_prob)
        
        # Plot training history
        self.plot_training_history(history)
        
        # Save final model in multiple formats
        model.save(str(self.model_dir / "ai_face_detector.h5"))
        model.save_weights(str(self.model_dir / "model_weights.h5"))
        
        # Save model architecture
        with open(self.model_dir / "model_architecture.json", "w") as f:
            f.write(model.to_json())
        
        print(f"\n💾 Model saved to {self.model_dir}")
        print("Files created:")
        print("- ai_face_detector.h5 (complete model)")
        print("- model_weights.h5 (weights only)")
        print("- model_architecture.json (architecture)")
        print("- training_log.csv (training metrics)")
        print("- confusion_matrix.png (evaluation plot)")
        print("- training_history.png (training plots)")
        
        return model, history
    
    def combine_histories(self, hist1, hist2):
        """Combine two training histories"""
        combined = {}
        for key in hist1.history.keys():
            combined[key] = hist1.history[key] + hist2.history[key]
        
        class CombinedHistory:
            def __init__(self, history_dict):
                self.history = history_dict
        
        return CombinedHistory(combined)
    
    def save_evaluation_results(self, y_true, y_pred, y_pred_prob):
        """Save detailed evaluation results"""
        from sklearn.metrics import confusion_matrix, roc_curve, auc
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Real', 'AI Generated'],
                   yticklabels=['Real', 'AI Generated'])
        plt.title('Confusion Matrix - AI Face Detection')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(self.model_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - AI Face Detection')
        plt.legend(loc="lower right")
        plt.savefig(self.model_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 ROC AUC Score: {roc_auc:.4f}")
    
    def plot_training_history(self, history):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(history.history['accuracy'], label='Training')
        axes[0, 0].plot(history.history['val_accuracy'], label='Validation')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        
        # Loss
        axes[0, 1].plot(history.history['loss'], label='Training')
        axes[0, 1].plot(history.history['val_loss'], label='Validation')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        
        # Precision
        axes[1, 0].plot(history.history['precision'], label='Training')
        axes[1, 0].plot(history.history['val_precision'], label='Validation')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        
        # Recall
        axes[1, 1].plot(history.history['recall'], label='Training')
        axes[1, 1].plot(history.history['val_recall'], label='Validation')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(self.model_dir / 'training_history.png')
        plt.close()

def main():
    """Main training function"""
    print("🚀 Starting AI Face Detector Training...")
    
    trainer = AIFaceTrainer()
    
    # Create model directory
    trainer.model_dir.mkdir(exist_ok=True)
    
    # Train model
    model, history = trainer.train()
    
    print("\n🎉 Training complete!")
    print("Model files saved in the 'model' directory")

if __name__ == "__main__":
    main()