import base64
import json
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from io import BytesIO
from model import Model

class Inference:
    def __init__(self, model_dir='pytorch_model.pth'):
        self.model = Model(num_classes=3)
        self.model.load_state_dict(torch.load(model_dir))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_image_from_bytes(self, image_bytes):
        image = Image.open(BytesIO(image_bytes))
        image = image.convert('RGB')
        return image

    def preprocess_image(self, image):
        return self.transform(image).unsqueeze(0)

    def infer_single_image(self, image_bytes):
        image = self.load_image_from_bytes(image_bytes)
        preprocessed_image = self.preprocess_image(image).to(self.device)
        with torch.no_grad():
            outputs = self.model(preprocessed_image)
            rotation_pred, l2_pred, r2_pred = outputs[:, 0], outputs[:, 1], outputs[:, 2]
        return rotation_pred.item(), l2_pred.item(), r2_pred.item()

    def inference(self, inputfile):
        with open(inputfile, "rb") as f:
            image_bytes = f.read()
        rotation, l2, r2 = self.infer_single_image(image_bytes)
        return {
            "rot": rotation,
            "l2": l2,
            "r2": r2
        }
