#!/usr/bin/env python3
"""
Test script to verify the AI Face Detector setup
"""

import os
import sys
from pathlib import Path
import requests
import time

def test_kaggle_setup():
    """Test if Kaggle API is configured"""
    print("🔑 Testing Kaggle API setup...")
    
    try:
        import kaggle
        # Try to authenticate
        kaggle.api.authenticate()
        print("✅ Kaggle API configured correctly")
        return True
    except Exception as e:
        print(f"❌ Kaggle API error: {e}")
        print("💡 Make sure kaggle.json is in ~/.kaggle/ directory")
        return False

def test_dependencies():
    """Test if all Python dependencies are installed"""
    print("📦 Testing Python dependencies...")
    
    # Package name mapping: pip_name -> import_name
    packages = {
        'tensorflow': 'tensorflow',
        'fastapi': 'fastapi', 
        'uvicorn': 'uvicorn',
        'pillow': 'PIL',
        'opencv-python': 'cv2',
        'numpy': 'numpy',
        'scikit-learn': 'sklearn',
        'kaggle': 'kaggle'
    }
    
    missing = []
    for pip_name, import_name in packages.items():
        try:
            __import__(import_name)
            print(f"✅ {pip_name}")
        except ImportError:
            print(f"❌ {pip_name}")
            missing.append(pip_name)
    
    if missing:
        print(f"\n💡 Install missing packages: pip install {' '.join(missing)}")
        return False
    
    print("✅ All Python dependencies installed")
    return True

def test_dataset():
    """Test if dataset is downloaded and organized"""
    print("📊 Testing dataset...")
    
    dataset_path = Path("dataset/organized")
    if not dataset_path.exists():
        print("❌ Dataset not found")
        print("💡 Run: python download_dataset.py")
        return False
    
    train_real = dataset_path / "train" / "real"
    train_fake = dataset_path / "train" / "fake"
    val_real = dataset_path / "val" / "real"
    val_fake = dataset_path / "val" / "fake"
    
    for path in [train_real, train_fake, val_real, val_fake]:
        if not path.exists():
            print(f"❌ Missing: {path}")
            return False
        
        count = len(list(path.glob("*")))
        print(f"✅ {path.name}: {count} images")
    
    print("✅ Dataset organized correctly")
    return True

def test_model():
    """Test if model is trained and available"""
    print("🤖 Testing AI model...")
    
    model_path = Path("backend/model/ai_face_detector.h5")
    weights_path = Path("backend/model/model_weights.h5")
    
    if model_path.exists():
        print("✅ Full model found")
        return True
    elif weights_path.exists():
        print("✅ Model weights found")
        return True
    else:
        print("❌ No trained model found")
        print("💡 Run: python train.py")
        return False

def test_frontend():
    """Test if frontend dependencies are installed"""
    print("🌐 Testing frontend setup...")
    
    if not Path("frontend/node_modules").exists():
        print("❌ Node modules not installed")
        print("💡 Run: cd frontend && npm install")
        return False
    
    if not Path("frontend/package.json").exists():
        print("❌ package.json not found")
        return False
    
    print("✅ Frontend dependencies installed")
    return True

def test_api_connection():
    """Test if the API is running and responding"""
    print("🔌 Testing API connection...")
    
    try:
        response = requests.get("http://localhost:8000/api/health", timeout=5)
        if response.status_code == 200:
            print("✅ API is running and healthy")
            return True
        else:
            print(f"❌ API returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ API not running")
        print("💡 Start with: python main.py")
        return False
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 AI Face Detector - Setup Test")
    print("=" * 50)
    
    tests = [
        ("Kaggle API", test_kaggle_setup),
        ("Dependencies", test_dependencies),
        ("Dataset", test_dataset),
        ("AI Model", test_model),
        ("Frontend", test_frontend),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        result = test_func()
        results.append((test_name, result))
    
    # Test API only if other tests pass
    if all(result for _, result in results):
        print(f"\n🔌 API Connection:")
        api_result = test_api_connection()
        results.append(("API Connection", api_result))
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Your setup is ready.")
        print("\n🚀 To start the app:")
        print("   1. python main.py (in one terminal)")
        print("   2. npm run dev (in another terminal)")
        print("   3. Open http://localhost:3000")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        
        # Provide specific guidance
        if not any(name == "Dataset" and result for name, result in results):
            print("\n💡 To fix dataset issues:")
            print("   python download_dataset.py")
        
        if not any(name == "AI Model" and result for name, result in results):
            print("\n💡 To fix model issues:")
            print("   python train.py")

if __name__ == "__main__":
    main()