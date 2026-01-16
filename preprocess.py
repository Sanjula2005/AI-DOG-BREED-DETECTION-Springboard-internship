import torch
from torchvision import transforms
from PIL import Image

def load_image(path_or_bytes):
    img = Image.open(path_or_bytes).convert("RGB")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    img = transform(img).unsqueeze(0)
    return img
