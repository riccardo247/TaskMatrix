import numpy as np
import torch
from torch import nn
from torchvision.transforms import ToTensor
from torchvision.transforms import ToPILImage

from PIL import Image
import torchvision.transforms as transforms

import requests
import cv2

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


## Construct Residual Block##
class ResidualBlock(nn.Module):
    """Residual Block."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True))

    def forward(self, x):
        return x + self.main(x)

## Construct the generator ##
class Generator(nn.Module):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, conv_dim=64, repeat_num=6):
        super(Generator, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-Sampling
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-Sampling
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        out = self.main(x)
        return out
class MakeupTransfer:
    def __init__(self, load_checkpoint=False, checkpoint_paths=("checkpoint/40_1300_G_A.pth",
                                                                "checkpoint/40_1300_G_B.pth")):
        print(f"loading from {checkpoint_paths}")
        self.modelA = Generator()
        self.modelB = Generator()
        self.modelA.load_state_dict(torch.load(checkpoint_paths[0]))
        self.modelB.load_state_dict(torch.load(checkpoint_paths[1]))
        self.modelA.to(DEVICE)
        self.modelB.to(DEVICE)

    def load_image(self, file_path):
        # Open the image file with PIL
        image = Image.open(file_path).convert('RGB')
        # image = cv2.imread(file_path)
        # Convert the PIL image to a PyTorch tensor
        # tensor = ToTensor()(image)
        desired_size = (256, 256)
        image = Image.fromarray(cv2.resize(np.array(image), (256, 256)))
        transform = transforms.Compose([
            transforms.Resize(desired_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        image = transform(image)
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

    def denorm(self, x):
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def makeup(self, file_path, command):
        # array_img = []
        # for m in images:
        print(f'going to {command} image {file_path}')
        image = self.load_image(file_path)

        print("loaded image")
        image = image.unsqueeze(0).float()
        image = image.to(DEVICE)
        print(f"starting inference")
        if "remove" in command:
            image_out = self.modelB(image)
        else:
            image_out = self.modelA(image)

        image_out = self.denorm(image_out)
        print(f"returned image")
        print(f"max: {np.max(image_out)}")
        return image_out
