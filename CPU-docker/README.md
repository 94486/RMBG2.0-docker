# Background Removal Tool

A web-based tool that leverages the RMBG-2.0 model to remove backgrounds from images with high precision. Built with Gradio for an intuitive user interface and optimized for Apple Silicon.

## Key Features

- 🖼️ High-quality background removal using RMBG-2.0
- 📱 User-friendly web interface
- 🔗 Support for both file upload and image URLs
- 👀 Side-by-side comparison view
- ⚡ M1/M2 Mac optimization with MPS acceleration
- 🌐 Local web server for easy access

## Tech Stack

- Python 3.10
- PyTorch
- Gradio
- Transformers
- RMBG-2.0 model

## 💻 Installation
1. Clone the repository:
```
git clone https://github.com/almensu/RMBG-2.0-webui.git
cd RMBG-2.0-webui
```
2. Create and activate virtual environment
```
conda create -n RMBG-2.0-webui python=3.10
conda activate RMBG-2.0-webui
```
3. Install dependencies:
```
pip install -r requirements.txt
```
4. Running the App
```
python webui.py
```

## Usage
1. On first run, the script will download the RMBG-2.0 model (approximately 1GB)
2. Open your browser and navigate to the local server (typically http://localhost:7860)
3. Upload an image or paste an image URL
4. Wait for processing
5. View and download the result with transparent background

Note: The initial model download may take a few minutes depending on your internet connection. The model will be cached locally for subsequent runs.

## credits
RMBG-2.0: https://huggingface.co/briaai/RMBG-2.0

## License
MIT
