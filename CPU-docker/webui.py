import os
# Set environment variable for MPS fallback to CPU
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import gradio as gr
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
import requests
from io import BytesIO

def load_model():
    """
    Load model and set device
    """
    model = AutoModelForImageSegmentation.from_pretrained("models", trust_remote_code=True)
    # Use MPS if available (M1/M2 Mac), otherwise CPU
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model.to(device)
    model.eval()
    return model, device

def process_image(input_image, model, device):
    """
    Process image and remove background
    Args:
        input_image: Input PIL Image
        model: Pretrained model
        device: Computing device
    Returns:
        PIL Image with transparent background
    """
    if input_image is None:
        return None
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Convert to tensor and move to device
    img_tensor = transform(input_image).unsqueeze(0).to(device)
    
    # Predict segmentation mask
    with torch.no_grad():
        pred = model(img_tensor)[-1].sigmoid().cpu()
    
    # Convert mask to PIL Image
    mask = transforms.ToPILImage()(pred[0].squeeze())
    mask = mask.resize(input_image.size)
    
    # Create new image with transparent background
    result = Image.new('RGBA', input_image.size, (0, 0, 0, 0))
    # Convert input image to RGBA mode
    input_image = input_image.convert('RGBA')
    # Merge images using mask
    result.paste(input_image, (0, 0), mask=mask)
    
    return result

def remove_bg(image=None, url=None):
    """
    Main function for background removal
    Args:
        image: Uploaded image
        url: Image URL
    """
    if url:
        try:
            response = requests.get(url)
            input_image = Image.open(BytesIO(response.content)).convert('RGB')
        except:
            return None
    elif image is not None:
        input_image = Image.fromarray(image).convert('RGB')
    else:
        return None
    
    model, device = load_model()
    return process_image(input_image, model, device)

# Create Gradio interface
demo = gr.Interface(
    fn=remove_bg,
    inputs=[
        gr.Image(label="Upload an image", type="numpy"),
        gr.Textbox(label="Image URL")
    ],
    outputs=gr.Image(label="Output", type="pil"),
    title="RMBG-2.0 Background Removal",
    description="Upload an image or provide URL to remove background",
    examples=[["models/giraffe.jpg", None]],
    cache_examples=False
)

if __name__ == "__main__":
    # Launch server
    demo.launch(server_name="0.0.0.0")