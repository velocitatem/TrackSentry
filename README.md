# YOLO for Trains


YOLO for Trains is a machine learning project that leverages the YOLO11n model from Ultralytics to detect railway sleepers and clips in train track images. The project utilizes a labeled dataset of train tracks and offers an easy-to-use Gradio interface for running inferences on input images.


## Table of Contents
- [About the Project](#about-the-project)
- [Model Information](#model-information)
- [Results](#results)
- [Getting Started](#getting-started)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)



## About the Project
The "YOLO for Trains" project aims to automate the detection of critical components on railway tracks using state-of-the-art object detection models. The lightweight YOLO11n model is trained on a custom dataset of train track images using Ultralytics' platform, providing a robust and efficient solution for real-time applications and edge deployment.

### Key Features
- Efficient detection of sleepers and clips on railway tracks.
- Trained using YOLO11n model with a focus on speed and low computational requirements.
- Gradio interface for simple and fast model inference.



## Model Information
YOLO11n is a "Nano" variant of the YOLO family developed by Ultralytics, with approximately 2.6 million parameters. The model was trained for 80 epochs, with a recommended ideal stopping point around 50 epochs based on observed performance. The model's architecture is designed for high performance in environments with limited resources.

### Training Configuration
- **Model:** YOLO11n
- **Dataset:** Custom labeled train track images
- **Epochs:** 80 (suggested optimal 50)
- **Training Platform:** Ultralytics



## Results
The model achieved a high mAP50 (mean Average Precision) close to 1, indicating excellent detection performance. The training logs and further insights into model performance can be found [here](https://gist.github.com/velocitatem/70f6531f517a4889c477d0338603821b).



## Getting Started
These instructions will help you set up and run the "YOLO for Trains" project on your local machine.

### Prerequisites
Make sure you have the following installed:
- Python 3.10 or newer
- Ultralytics library
- Gradio
- PyTorch

### Installation
```bash
pip install ultralytics gradio torch gdown gitpython setuptools==70.0.0 torchviz
```

### Model Download
The final model checkpoint is available on Hugging Face:
```bash
wget https://huggingface.co/velocitatem/railway-image-processing/resolve/main/model_-%208%20march%202025%2023_19.pt
```



## Usage
To run the Gradio interface and test the model, execute the following:

```python
from ultralytics import YOLO
import gradio as gr

model = YOLO('model_- 8 march 2025 23_19.pt')

def predict(image):
    results = model(image)
    return results[0].plot()

interface = gr.Interface(fn=predict, inputs=gr.Image(type='numpy'), outputs=gr.Image(type='pil'))
interface.launch()
```

Visit the provided local or public URL to upload images and see detection results.


## Acknowledgments
- [Ultralytics](https://ultralytics.com) for the YOLO11n model
- [Gradio](https://gradio.app) for the easy-to-use web interface
- [Hugging Face](https://huggingface.co) for model hosting
