import argparse
import base64
import json
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from io import BytesIO
from model import Model

def load_image_from_bytes(image_bytes):
    image = Image.open(image_bytes)
    image = image.convert('RGB')
    return image

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def infer_single_image(image_bytes, model):
    image = load_image_from_bytes(image_bytes)
    preprocessed_image = preprocess_image(image)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(preprocessed_image.to(device))
        rotation_pred, l2_pred, r2_pred = outputs[:, 0], outputs[:, 1], outputs[:, 2]

    return rotation_pred.item(), l2_pred.item(), r2_pred.item()

def main():
    parser = argparse.ArgumentParser(description='Infer image with trained model')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    parser.add_argument('--save_to_file', action='store_true', help='Save predictions to a JSON file')
    parser.add_argument('--load_file', type=str, help='Path to the model file')
    args = parser.parse_args()

    if args.load_file:
        model = Model(num_classes=3)
        model.load_state_dict(torch.load(args.load_file))
    else:
        model = Model(num_classes=3)
        model_path = 'pytorch_model.pth'
        model.load_state_dict(torch.load(model_path))

    image_bytes = args.image_path
    rotation, l2, r2 = infer_single_image(image_bytes, model)

    print("Predicted Rotation:", rotation)
    print("Predicted L2 Value:", l2)
    print("Predicted R2 Value:", r2)

    if args.save_to_file:
        predictions = {
            'rotation': rotation,
            'l2': l2,
            'r2': r2
        }
        with open('predictions.json', 'w') as f:
            json.dump(predictions, f)

if __name__ == "__main__":
    main()