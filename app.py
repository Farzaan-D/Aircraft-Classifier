import gradio as gr
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from ultralytics import YOLO
import cv2
from models import HierarchicalNet
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import os

# Device setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# Load YOLOv8 for airplane detection (class 4 in COCO is airplane)
# It will download yolov8n.pt on first run
yolo_model = YOLO("yolov8n.pt")

# Model configuration
MODEL_NAME = 'convnext_tiny'
NUM_VARIANTS = 100
NUM_FAMILIES = 70
NUM_MAKERS = 30
WEIGHTS_PATH = 'checkpoints/best_model.pth'

# Initialize model
classifier = HierarchicalNet(MODEL_NAME, NUM_VARIANTS, NUM_FAMILIES, NUM_MAKERS, pretrained=False)
if os.path.exists(WEIGHTS_PATH):
    classifier.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device, weights_only=True))
    print("Loaded custom weights.")
else:
    print("WARNING: No trained weights found at checkpoints/best_model.pth. Using untrained model.")
classifier = classifier.to(device)
classifier.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# For dummy labels in the UI since we don't have the dataset loaded here
# In a real app, we would load the mapping from a JSON or txt file.
def load_class_names():
    # Attempt to load from dataset folder if it exists
    labels = {'variant': [], 'family': [], 'maker': []}
    base_dir = "./data/fgvc-aircraft-2013b/data"
    
    if os.path.exists(base_dir):
        for level in ['variant', 'family', 'manufacturer']:
            key = 'maker' if level == 'manufacturer' else level
            filepath = os.path.join(base_dir, f"images_{level}_train.txt")
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    for line in f:
                        parts = line.strip().split(' ', 1)
                        if len(parts) == 2 and parts[1] not in labels[key]:
                            labels[key].append(parts[1])
                            
    return labels

class_names = load_class_names()

def predict(image_pil):
    if image_pil is None:
        return None, "Please upload an image.", None
        
    # 1. Object Detection with YOLO
    results = yolo_model(image_pil)
    
    # Find the best airplane bounding box (COCO airplane class = 4)
    best_box = None
    max_conf = 0
    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) == 4 and box.conf[0] > max_conf:
                max_conf = box.conf[0]
                best_box = box.xyxy[0].cpu().numpy()
                
    # If YOLO didn't find an airplane, use the whole image
    if best_box is not None:
        x1, y1, x2, y2 = map(int, best_box)
        crop_pil = image_pil.crop((x1, y1, x2, y2))
    else:
        crop_pil = image_pil
        
    # 2. Classification
    input_tensor = transform(crop_pil).unsqueeze(0).to(device)
    
    with torch.no_grad():
        out_v, out_f, out_m = classifier(input_tensor)
        
        prob_v = F.softmax(out_v, dim=1)[0]
        prob_f = F.softmax(out_f, dim=1)[0]
        prob_m = F.softmax(out_m, dim=1)[0]
        
        pred_v = torch.argmax(prob_v).item()
        pred_f = torch.argmax(prob_f).item()
        pred_m = torch.argmax(prob_m).item()
        
    # Formatting output text
    variant_str = class_names['variant'][pred_v] if class_names['variant'] else f"Variant ID {pred_v}"
    family_str = class_names['family'][pred_f] if class_names['family'] else f"Family ID {pred_f}"
    maker_str = class_names['maker'][pred_m] if class_names['maker'] else f"Maker ID {pred_m}"
    
    result_text = (
        f"**Maker:** {maker_str} (Conf: {prob_m[pred_m]:.2f})\n"
        f"**Family:** {family_str} (Conf: {prob_f[pred_f]:.2f})\n"
        f"**Variant:** {variant_str} (Conf: {prob_v[pred_v]:.2f})"
    )
    
    if best_box is None:
        result_text = "WARNING: No airplane detected by YOLO. Classified full image.\n\n" + result_text
        
    # 3. Grad-CAM Explainability
    # For ConvNeXt, the target layer is usually the last normalization or convolution block.
    # We will try to get the last block of the features
    target_layers = [classifier.backbone.stages[-1].blocks[-1]]
    
    cam = GradCAM(model=classifier.backbone, target_layers=target_layers)
    
    # Create an input tensor with requires_grad for GradCAM
    input_tensor_cam = transform(crop_pil).unsqueeze(0)
    
    # Generate heatmap
    grayscale_cam = cam(input_tensor=input_tensor_cam, targets=None)
    grayscale_cam = grayscale_cam[0, :]
    
    # Overlay on original cropped image
    crop_np = np.array(crop_pil.resize((224, 224))) / 255.0
    cam_image = show_cam_on_image(crop_np, grayscale_cam, use_rgb=True)
    
    return crop_pil, result_text, Image.fromarray(cam_image)

# Gradio Interface
with gr.Blocks(title="Aircraft Classifier AI") as demo:
    gr.Markdown("# Aircraft Classifier (Hierarchical multi-task learning + YOLOv8 + Grad-CAM)")
    gr.Markdown("Upload an image of an airplane. The system will detect the plane, crop it, classify its Maker, Family, and Variant, and highlight the important visual features.")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Input Image")
            submit_btn = gr.Button("Classify")
            
        with gr.Column():
            output_crop = gr.Image(type="pil", label="YOLOv8 Detected Airplane")
            output_text = gr.Markdown(label="Classification Results")
            output_cam = gr.Image(type="pil", label="Grad-CAM Heatmap (What the model looked at)")
            
    submit_btn.click(fn=predict, inputs=input_image, outputs=[output_crop, output_text, output_cam])
    
if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
