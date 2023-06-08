import torch
import numpy as np
from torchvision.transforms import ToTensor
from torchvision.transforms import ToPILImage
from PIL import Image
import requests
from . import model_v0
import cv2

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class HairColor:
    def __init__(self, load_checkpoint=False, checkpoint_path="hair_segmentation_29"):
        print(f"loading from {checkpoint_path}")
        self.model = model_v0.MobileHairNetV2()
        self.model.load_state_dict(torch.load(checkpoint_path))
        self.model.to(DEVICE)

    def load_image(self, file_path):
        # Open the image file with PIL
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))

        return image

    def load_checkpoint_file(self):
        file_url = "https://drive.google.com/uc?export=download&id=1-EIWKrO7kiuNuf4Ku2oXqSHfR3-hbJcH"
        save_path = "checkpoint200.pt"  # Replace with the desired file name

        response = requests.get(file_url)
        if response.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(response.content)
            print("File downloaded successfully.")
        else:
            print("Unable to download the file.")
    def save_images(self, tensor, file_paths):
        # Convert the tensor to a PIL Image and save it
        for i in range(tensor.shape[0]):
            img = ToPILImage()(tensor[i])
            img.save(file_paths[i])

    def selective_mask_t(self, image_src, mask, channels=[]):
        # mask = mask[:, torch.tensor(channels).long()]
        # mask = torch.sgn(torch.sum(mask, dim=1)).to(dtype=image_src.dtype).unsqueeze(-1)
        return mask[:, :, np.newaxis] * image_src
    def hair_color(self, file_path, rgb):
        # array_img = []
        # for m in images:
        r, g, b = rgb
        print(f'going to segment image {file_path} and color {rgb}')
        image = self.load_image(file_path)
        image_norm = image / 255
        image_tensor = ToTensor()(image_norm)
        print("loaded image")
        image_tensor = image_tensor.unsqueeze(0).float()
        image_tensor = image_tensor.to(DEVICE)
        print(f"starting inference")
        mask_out = self.model(image_tensor)
        mask_out = mask_out.squeeze(0)
        mask_out = torch.cat([mask_out, mask_out, mask_out], dim=0)
        mask_out = mask_out.detach().to('cpu').numpy()
        mask_out = mask_out * 255
        mask_out = mask_out.astype(np.uint8)
        mask_out = mask_out.transpose(1, 2, 0)
        lab_img = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab_img)
        B, G, R = cv2.split(image)
        B[mask_out[:, :, 0] >= 180] = b
        G[mask_out[:, :, 0] >= 180] = g
        R[mask_out[:, :, 0] >= 180] = r

        colored = cv2.merge([B, G, R])
        L1, A1, B1 = cv2.split(cv2.cvtColor(colored, cv2.COLOR_BGR2LAB))
        processed = cv2.merge([L, A1, B1])
        processed = cv2.cvtColor(processed, cv2.COLOR_LAB2BGR)
        # back to RGB
        processed = processed[..., ::-1].copy()
        print(f"returned processed")

        return processed