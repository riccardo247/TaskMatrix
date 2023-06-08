import torch
import numpy as np
from torchvision.transforms import ToTensor
from torchvision.transforms import ToPILImage
from PIL import Image
import requests
from . import model_v0

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class HairSegmentation:
    def __init__(self, load_checkpoint=False, checkpoint_path="hair_segmentation_20epochs.pt"):
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


    def load_images(self, file_paths):
        # Create a list to store the image tensors
        tensors = []

        # Loop over the file paths
        for file_path in file_paths:
            # Open the image file
            image = Image.open(file_path)

            # Convert the image to a tensor and resize it
            # tensor = transform(image)
            tensor = ToTensor()(image)
            # Add the tensor to the list
            tensors.append(tensor)

        # Stack the tensors along a new dimension
        tensors = torch.stack(tensors)

        return tensors

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
    def hair_segment(self, file_path):
        # array_img = []
        # for m in images:
        print(f'going to segment image {file_path}')
        image = self.load_image(file_path)
        print("loaded image")
        image = image.unsqueeze(0)
        # array_img.append(image)
        # array_img = torch.stack(array_img)
        image = image.to(DEVICE)
        print(f"starting inference")
        mask_out = self.model(image)
        print(f"returned mask")

        return mask_out