#!/usr/bin/env python3
"""
AI Face Detector - Main Launcher
Run this file to start the application
"""

import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Check if all requirements are installed"""
    try:
        import tensorflow
        import fastapi
        import uvicorn
        import PIL
        import cv2
        import numpy
        import sklearn
        import kaggle
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("💡 Install with: pip install -r backend/requirements.txt")
        return False

def start_backend():
    """Start the FastAPI backend server"""
    print("🚀 Starting AI Face Detector Backend...")
    os.chdir("backend")
    subprocess.run([sys.executable, "main.py"])

def main():
    """Main launcher function"""
    print("🤖 AI Face Detector")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not Path("backend").exists():
        print("❌ Backend folder not found!")
        print("💡 Make sure you're running this from the project root")
        return
    
    # Check requirements
    if not check_requirements():
        return
    
    print("✅ All dependencies found")
    print("🌐 Starting backend server...")
    print("📝 Frontend: Open another terminal and run 'npm run dev' in the frontend folder")
    print("🔗 API will be available at: http://localhost:8000")
    print("🔗 Frontend will be available at: http://localhost:3000")
    print("\n" + "=" * 40)
    
    try:
        start_backend()
    except KeyboardInterrupt:
        print("\n⏹️  Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")

if __name__ == "__main__":
    main()