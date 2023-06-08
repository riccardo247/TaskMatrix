import torch
import numpy as np
import torch.nn as nn

from torchvision.transforms import ToTensor
from torchvision.transforms import ToPILImage

import requests
import cv2

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class Fire(nn.Module):
    def __init__(self, inplanes: int, squeeze_planes: int, expand1x1_planes: int, expand3x3_planes: int) -> None:
        super().__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat(
            [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        )


class SqueezeNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(96, 16, 64, 64),
            Fire(128, 16, 64, 64),
            Fire(128, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(256, 32, 128, 128),
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(512, 64, 256, 256),
            nn.Conv2d(512, 256, 3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 512),
            nn.ReLU(),
            nn.Linear(512, 136)

        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):

                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return x


class LipsColor:
    def __init__(self, load_checkpoint=False, checkpoint_path="face_landmarks49"):
        print(f"loading from {checkpoint_path}")
        self.model = SqueezeNet()
        self.model.load_state_dict(torch.load(checkpoint_path))
        self.model.to(DEVICE)

    def load_image(self, file_path):
        # Open the image file with PIL
        image = cv2.imread(file_path, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (128, 128))

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

    def lips_color(self, file_path, rgb):
        # array_img = []
        # for m in images:
        r, g, b = rgb
        print(f'going to lips color image {file_path} and color {rgb}')
        image = self.load_image(file_path)
        image_norm = image/255
        image_tensor = torch.from_numpy(image_norm)
        print("loaded image")
        image_tensor = image_tensor.unsqueeze(0).permute([0, 3, 1, 2]).float()
        image_tensor = image_tensor.to(DEVICE)
        print(f"starting inference")
        #use landmarks model on image
        self.model.eval()
        y_pred = self.model(image_tensor)
        y_pred = (y_pred * 128).detach().to('cpu').numpy()
        marks = np.round(y_pred).astype(np.int32)
        marks = marks.reshape(68, 2)
        #get lips mask
        poly = []
        for a, (x, y) in enumerate(marks):

            if (a > 47 and a < 60):
                poly.append((x, y))
            if a == 48:
                x1 = int(x)
                y1 = int(y)
            elif a == 52:
                x3 = int(x)
                y3 = int(y)
            elif a == 54:
                x2 = int(x)
                y2 = int(y)

            elif a == 58:
                x4 = int(x)
                y4 = int(y)
        mask = np.zeros_like(image[..., ::-1])
        pts = np.array(poly)
        pts = pts.astype(np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], (255, 255, 255))
        #apply color
        B, G, R = cv2.split(image)
        L1, A1, B1 = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2LAB))
        B[mask[:, :, 0] == 255] = b
        G[mask[:, :, 0] == 255] = g
        R[mask[:, :, 0] == 255] = r
        mod = cv2.merge([B, G, R])
        L2, A2, B2 = cv2.split(cv2.cvtColor(mod, cv2.COLOR_BGR2LAB))
        processed = cv2.merge([L1, A2, B2])
        processed = cv2.cvtColor(processed, cv2.COLOR_LAB2BGR)
        #back to RGB
        processed = processed[..., ::-1].copy()
        print(f"returned processed image")

        return processed