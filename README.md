# Face and Emotion Detection Project

This project uses OpenCV and TensorFlow (or PyTorch) to perform real-time face and emotion detection using your MacBook's webcam. The emotions detected include Happy, Sad, Angry, and more.

## Setup

### 1. Clone the Repository
```
git clone https://github.com/bkhedayat/emotion-detection.git
```

### 2. Set Up a Virtual Environment
It’s recommended to use a virtual environment to manage dependencies:
```
python -m venv venv
source venv/bin/activate  # On macOS/Linux
```

### 3. Install Dependencies
Install the required Python packages listed in requirements.txt:

```bash
pip install -r requirements.txt
```

### 4. Download the Pre-trained Emotion Detection Model

Download a pre-trained emotion detection model (e.g., from this repository or train your own) and place it in the models/ directory. Ensure it is named emotion_detection_model.h5.

### 5. Run the Script
To start the face and emotion detection using your webcam, run:
```
python scripts/detect.py
```