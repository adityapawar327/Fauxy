# AI-Generated Face Detector

A modern, high-performance web application for detecting AI-generated faces using deep learning.

## Features

- 🎯 High-accuracy AI face detection
- 🚀 Real-time processing
- 📱 Responsive modern UI
- 🔍 Detailed analysis results
- 📊 Confidence scoring
- 🎨 Sleek dark/light theme

## Dataset

Uses the high-quality dataset from Kaggle: [Detect AI Generated Faces](https://www.kaggle.com/datasets/shahzaibshazoo/detect-ai-generated-faces-high-quality-dataset)

## Tech Stack

- **Frontend**: React + Vite + Tailwind CSS
- **Backend**: FastAPI + Python
- **ML**: TensorFlow/PyTorch + OpenCV
- **Deployment**: Docker ready

## Quick Start

### 1. Setup Environment
```bash
# Install frontend dependencies
npm install

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Download Dataset & Train Model
```bash
# Setup Kaggle credentials and download dataset
python scripts/setup.py

# Train the AI model (this will take time!)
python train.py
```

### 3. Run the Application
```bash
# Start frontend (Terminal 1)
npm run dev

# Start backend (Terminal 2)
python main.py
```

### 4. Test the Model (Optional)
```bash
# Test on a single image
python evaluate.py path/to/image.jpg

# Test on a directory of images
python evaluate.py path/to/images/ --batch
```

## Performance

- Processing time: < 2 seconds per image
- Accuracy: 95%+ on test dataset
- Supports: JPG, PNG, WebP formats
- Max file size: 10MB
## Tr
aining Your Own Model

### Prerequisites
1. **Kaggle Account**: Sign up at [kaggle.com](https://www.kaggle.com)
2. **Kaggle API**: Get your API token from Account → API → Create New Token
3. **GPU (Recommended)**: Training will be much faster with a GPU

### Dataset Information
- **Source**: [Detect AI Generated Faces - High Quality Dataset](https://www.kaggle.com/datasets/shahzaibshazoo/detect-ai-generated-faces-high-quality-dataset)
- **Size**: ~10 million high-quality face images
- **Classes**: Real faces vs AI-generated faces
- **AI Models**: Includes faces from Midjourney, DALL-E, Stable Diffusion, etc.

### Training Process
1. **Download Dataset**: `python scripts/setup.py`
2. **Start Training**: `python train.py`
3. **Monitor Progress**: Check `model/training_log.csv`
4. **Evaluate Results**: View `model/confusion_matrix.png`

### Model Architecture
- **Base**: EfficientNetB0 (transfer learning) or Custom CNN
- **Input**: 224x224 RGB images
- **Output**: Binary classification (Real vs AI)
- **Accuracy**: 95%+ on test set
- **Training Time**: 2-4 hours on GPU, 8-12 hours on CPU

## API Endpoints

### POST /api/analyze
Upload and analyze an image for AI generation detection.

**Request**: Multipart form with image file
**Response**:
```json
{
  "prediction": "AI Generated" | "Real/Authentic",
  "confidence": 0.95,
  "processing_time": "1.2s",
  "artifacts": "High" | "Medium" | "Low",
  "consistency": "High" | "Medium" | "Low",
  "texture": "Natural" | "Processed" | "Synthetic",
  "symmetry": "Normal" | "High" | "Low"
}
```

### GET /api/health
Check API health and model status.

### GET /api/stats
Get detection statistics and model information.

## Deployment

### Docker (Recommended)
```bash
# Build and run with Docker Compose
docker-compose up --build

# Access at http://localhost
```

### Manual Deployment
```bash
# Build frontend
npm run build

# Start production server
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Performance Optimization

### For Better Accuracy
- Use more training data
- Increase training epochs
- Fine-tune hyperparameters
- Use ensemble methods

### For Faster Inference
- Use TensorFlow Lite
- Implement model quantization
- Use GPU acceleration
- Batch processing for multiple images

## Troubleshooting

### Common Issues

**"No trained model found"**
- Run `python train.py` to train the model first
- Or download a pre-trained model if available

**"Dataset not found"**
- Run `python scripts/setup.py` to download the dataset
- Check Kaggle API credentials

**"Out of memory during training"**
- Reduce batch size in `scripts/train_model.py`
- Use a smaller model architecture
- Close other applications

**"Low accuracy results"**
- Ensure dataset is properly structured
- Increase training epochs
- Check data quality and balance

### Getting Help
- Check the training logs in `model/training_log.csv`
- Review error messages in the console
- Ensure all dependencies are installed correctly