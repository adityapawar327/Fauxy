#!/usr/bin/env python3
"""
Test script to verify installation
"""

import sys
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported"""
    packages = [
        ('tensorflow', 'TensorFlow'),
        ('cv2', 'OpenCV'),
        ('PIL', 'Pillow'),
        ('numpy', 'NumPy'),
        ('sklearn', 'Scikit-learn'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn'),
        ('fastapi', 'FastAPI'),
        ('uvicorn', 'Uvicorn'),
        ('pydantic', 'Pydantic')
    ]
    
    print("🧪 Testing package imports...")
    print("=" * 40)
    
    failed = []
    
    for package, name in packages:
        try:
            __import__(package)
            print(f"✅ {name}")
        except ImportError as e:
            print(f"❌ {name}: {e}")
            failed.append(name)
    
    # Test Kaggle separately (it tries to authenticate on import)
    try:
        import kaggle
        print(f"✅ Kaggle API")
    except Exception as e:
        if "kaggle.json" in str(e):
            print(f"⚠️ Kaggle API: Not configured (run setup.py first)")
        else:
            print(f"❌ Kaggle API: {e}")
            failed.append("Kaggle API")
    
    if failed:
        print(f"\n⚠️ Failed imports: {', '.join(failed)}")
        print("Run: python install.py")
        return False
    else:
        print("\n🎉 All packages imported successfully!")
        return True

def test_tensorflow():
    """Test TensorFlow functionality"""
    try:
        import tensorflow as tf
        
        # Get version
        try:
            version = tf.__version__
        except AttributeError:
            version = tf.version.VERSION
        
        print(f"\n🧠 TensorFlow version: {version}")
        
        # Test GPU availability
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                print(f"🚀 GPU available: {len(gpus)} device(s)")
                for i, gpu in enumerate(gpus):
                    print(f"   GPU {i}: {gpu.name}")
            else:
                print("💻 No GPU detected, will use CPU")
        except Exception as e:
            print(f"⚠️ GPU check failed: {e}")
        
        # Test basic operation
        x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        y = tf.constant([[1.0, 1.0], [0.0, 1.0]])
        z = tf.matmul(x, y)
        print(f"✅ TensorFlow basic operations working")
        
        return True
    except Exception as e:
        print(f"❌ TensorFlow test failed: {e}")
        return False

def test_opencv():
    """Test OpenCV functionality"""
    try:
        import cv2
        import numpy as np
        
        print(f"\n📷 OpenCV version: {cv2.__version__}")
        
        # Test basic image operations
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print("✅ OpenCV basic operations working")
        
        return True
    except Exception as e:
        print(f"❌ OpenCV test failed: {e}")
        return False

def test_model_structure():
    """Test if model can be created"""
    try:
        import tensorflow as tf
        from tensorflow import keras
        
        print(f"\n🏗️ Testing model creation...")
        
        # Simple test model
        model = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Flatten(),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print("✅ Model creation successful")
        print(f"   Parameters: {model.count_params():,}")
        
        return True
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 AI Face Detector - Installation Test")
    print("=" * 50)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
    
    # Test TensorFlow
    if not test_tensorflow():
        all_passed = False
    
    # Test OpenCV
    if not test_opencv():
        all_passed = False
    
    # Test model creation
    if not test_model_structure():
        all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! Installation is working correctly.")
        print("\n📋 Ready for next steps:")
        print("1. python scripts/setup.py  # Download dataset")
        print("2. python train.py          # Train model")
        print("3. python main.py           # Start backend")
        print("4. npm run dev              # Start frontend")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("Try running: python install.py")

if __name__ == "__main__":
    main()