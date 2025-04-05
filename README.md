# Virtual Try-On API with VITON-HD

This project implements a virtual try-on API using the VITON-HD framework. It allows users to upload a person image and a cloth image, and the API generates a new image with the person virtually trying on the cloth. The API is built using FastAPI and leverages the VITON-HD model for high-resolution virtual try-on.

## Project Structure
VIRTUAL-TRYON-API/
│
├── app/
│   ├── init.py
│   ├── config.py          # Configuration settings
│   ├── main.py            # FastAPI application
│   └── viton_service.py   # Core VITON-HD service logic
├── VITON-HD/
│   ├── checkpoints/
│   │   ├── alias_final.pth  # Checkpoint for ALIASGenerator
│   │   ├── gmm_final.pth    # Checkpoint for GMM
│   │   └── seg_final.pth    # Checkpoint for SegGenerator
│   ├── datasets.py          # Dataset utilities
│   ├── networks.py          # Model architectures
│   ├── utils.py             # Utility functions
│   ├── test.py              # Reference test script
│   └── README.md            # VITON-HD documentation
├── debug/                   # Directory for debugging outputs
├── results/                 # Directory for storing final results
├── temp/                    # Directory for temporary files
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation


## Features

- **Virtual Try-On**: Upload a person image and a cloth image to generate a try-on result.
- **High-Resolution Output**: Uses VITON-HD for high-quality results (1024x768 resolution).
- **FastAPI Backend**: Provides a RESTful API for easy integration.
- **Debugging Support**: Saves intermediate outputs (e.g., cloth mask, pose map) for debugging.

## Prerequisites

- Python 3.8 or higher
- A GPU with CUDA support (recommended for faster inference)
- Pre-trained VITON-HD checkpoints (`alias_final.pth`, `gmm_final.pth`, `seg_final.pth`)

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/kingkick2008/VIRTUAL-TRYON-API.git
cd VIRTUAL-TRYON-API

```


### 2. Create a Virtual Environment
```bash
python -m venv virtual-tryon-api
source virtual-tryon-api/bin/activate  # On Windows: virtual-tryon-api\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

The requirements.txt file includes:


fastapi==0.95.0
uvicorn==0.21.1
python-multipart==0.0.9
torch==2.0.0
torchvision==0.15.0
opencv-python==4.7.0.72
pillow==9.5.0
numpy==1.24.3
torchgeometry==0.1.2


### 4 Usage
1. Run the API
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
The API will be available at http://localhost:8000.

2. Test the API

Health Check
Check if the API is running and the models are loaded:


```bash
curl http://localhost:8000/health
```
Virtual Try-On
Upload a person image and a cloth image to perform virtual try-on:

Using curl
```bash
curl -X POST "http://localhost:8000/try-on/" \
  -F "person_image=@path/to/person_image.jpg" \
  -F "cloth_image=@path/to/cloth_image.jpg" \
  --output result.jpg

```


The result will be saved as result.jpg. A permanent copy of the result is also saved in the results/ directory with a timestamped filename.

### Debugging
Intermediate outputs (e.g., cloth mask, pose map, agnostic image) are saved in the debug/ directory during inference. These can be used to debug issues with the pipeline, such as incorrect preprocessing or model outputs.

### Known Issues and Limitations
1. Placeholder Preprocessing:
The current implementation uses placeholder methods for pose estimation (_generate_pose_map) and human parsing (_generate_parse_map) in viton_service.py, which leads to poor try-on results.
Proposed Fix: Integrate OpenPose for pose estimation and SCHP for human parsing (see "Contributing" section).
2. Performance:
Inference can be slow on CPU or low-end GPUs. Consider using mixed precision or optimizing the model for faster inference.
3. Input Validation:
The API lacks robust input validation (e.g., checking image formats, sizes, or content). This can lead to errors if invalid images are uploaded.


### Contributing
Contributions are welcome! Here are some areas where the project can be improved:

1. Improve Preprocessing:
Integrate OpenPose or openpifpaf for accurate pose estimation in the _generate_pose_map method.
Integrate SCHP for human parsing in the _generate_parse_map method.
Enhance cloth mask generation in the _generate_cloth_mask method using a pre-trained cloth segmentation model.
2. Optimize Performance:
Implement mixed precision training using torch.cuda.amp for faster inference.
Explore model quantization or pruning to reduce inference time.
3. Add Input Validation:
Add checks for image formats, sizes, and content (e.g., ensure the person image contains a person, the cloth image is a clean garment).
4. Improve Error Handling:
Enhance error messages to provide more context (e.g., "Failed to process image due to invalid format" instead of a generic 500 error).


To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix:
``` bash
git checkout -b feature/your-feature-name
``` 
3. Make your changes and commit them:
``` bash
git commit -m "Add your feature description"
```
4. Push your changes to your fork
``` bash
git push origin feature/your-feature-name
```
5. Create a pull request with a detailed description of your changes.


### License
This project is licensed under Creative Commons BY-NC 4.0. See the LICENSE file in the VITON-HD directory for details:
https://github.com/shadow2496/VITON-HD

