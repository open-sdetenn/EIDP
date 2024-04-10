import os
import json
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from io import BytesIO
import base64

class Dataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_list = os.listdir(data_dir)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.data_dir, file_name)
        
        with open(file_path, 'r') as f:
            frame_data_list = json.load(f)

        frame_data = np.random.choice(frame_data_list)

        image = Image.open(BytesIO(base64.b64decode(frame_data["base64_image"])))
        image = image.convert('RGB')

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = transform(image)

        rotation = float(frame_data["rotation_angle"])
        l2 = float(frame_data["L2_value"])
        r2 = float(frame_data["R2_value"])

        return image, rotation, l2, r2
