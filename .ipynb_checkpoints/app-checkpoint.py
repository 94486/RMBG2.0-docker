import os
import gradio as gr
from gradio_imageslider import ImageSlider
from loadimg import load_img
import spaces
from transformers import AutoModelForImageSegmentation
import torch
from torchvision import transforms

model = AutoModelForImageSegmentation.from_pretrained(
    "models", trust_remote_code=True
)
torch.set_float32_matmul_precision(['high', 'highest'][0])
model.to("cuda")
model.eval()
# Data settings
transform_image = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

output_folder = 'output_images'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def fn(image):
    im = load_img(image, output_type="pil")
    im = im.convert("RGB")
    origin = im.copy()
    image = process(im)    
    image_path = os.path.join(output_folder, "no_bg_image.png")
    image.save(image_path)
    return (image, origin), image_path

@spaces.GPU
def process(image):
    image_size = image.size
    input_images = transform_image(image).unsqueeze(0).to("cuda")
    # Prediction
    with torch.no_grad():
        preds = model(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image_size)
    image.putalpha(mask)
    return image
  
def process_file(f):
    name_path = f.rsplit(".",1)[0]+".png"
    im = load_img(f, output_type="pil")
    im = im.convert("RGB")
    transparent = process(im)
    transparent.save(name_path)
    return name_path

#slider1 = ImageSlider(label="RMBG-2.0", type="pil")
#slider2 = ImageSlider(label="RMBG-2.0", type="pil")
#image = gr.Image(label="Upload an image")
image2 = gr.Image(label="Upload an image",type="filepath")
text = gr.Textbox(label="Paste an image URL")
png_file = gr.File(label="output png file")
chameleon = load_img("models/giraffe.jpg", output_type="pil")
url = "http://xxx.jpg 或 /gemini/code/RMBG-2.0/models/giraffe.jpg"
'''tab1 = gr.Interface(
    fn, inputs=image, outputs=[slider1, gr.Image(label="output png file")], examples=[chameleon], api_name="image"
)
tab2 = gr.Interface(fn, inputs=text, outputs=[slider2, gr.Image(label="output png file")], examples=[url], api_name="text")'''
tab1 = gr.Interface(
    process_file, inputs=image2, outputs=[gr.Image(label="output png file")], examples=[chameleon], api_name="image"
)

tab2 = gr.Interface(process_file, inputs=text, outputs=[gr.Image(label="output png file")], examples=[url], api_name="text")
tab3 = gr.Interface(process_file, inputs=image2, outputs=png_file, examples=["giraffe.jpg"], api_name="png")


demo = gr.TabbedInterface(
    [tab1, tab2], ["input image", "input url"], title="RMBG-2.0 for background removal"
)

if __name__ == "__main__":
    demo.launch(show_error=True,server_name='0.0.0.0')