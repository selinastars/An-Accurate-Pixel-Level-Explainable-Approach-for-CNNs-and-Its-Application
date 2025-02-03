import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import cv2
import os
from PIL import Image
import math
from torchvision import transforms

# Define the Grad-CAM++ class
class GradCAMPlusPlus:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.feature_grad = None
        self.feature_map = None
        self.gradient = None
        self.hook()

    def hook(self):
        def feature_hook(module, input, output):
            self.feature_map = output.detach()

        def gradient_hook(module, grad_input, grad_output):
            self.gradient = grad_output[0].detach()

        target_layer = self.target_layer
        assert target_layer is not None, 'Please provide a valid target layer'

        target_layer.register_forward_hook(feature_hook)
        target_layer.register_full_backward_hook(gradient_hook)

    def __call__(self, input_image, target_class=None):
        self.model.zero_grad()

        # Forward propagation
        input_image.requires_grad_()
        output = self.model(input_image)
        if target_class is None:
            target_class = torch.argmax(output)

        # Backward propagation, compute class-specific feature gradients
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # Compute Grad-CAM++ weights
        weights = torch.mean(self.gradient, dim=(2, 3), keepdim=True)
        activations = F.relu(self.feature_map)
        grad_cam = torch.mean(weights * activations, dim=1, keepdim=True)
        grad_cam = F.relu(grad_cam)

        # Multiply the Grad-CAM++ results with the original image and normalize the result
        grad_cam = nn.functional.interpolate(grad_cam, size=(input_image.size(2), input_image.size(3)), mode='bilinear',
                                             align_corners=False)
        grad_cam = (grad_cam - grad_cam.min()) / (grad_cam.max() - grad_cam.min() + 1e-8)

        return grad_cam.squeeze().cpu().numpy()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=False)

# Load model weights
model_weight_path = "..."
model.load_state_dict(torch.load(model_weight_path, map_location=device))
model.eval()

# Define a sorting function
def sort_func(file_name):
    return int(''.join(filter(str.isdigit, file_name)))

# Folder path
input_folder = "..."

# Specify the tampering images save address
output_folder_path_increase = "..."
output_folder_path_decrease = "..."

ratio_increase = 1.01
ratio_decrease = 0.99

# Retrieve the names of image files in the folder and sort them based on the numeric part
image_files = sorted(os.listdir(input_folder), key=sort_func)

# Create an output folder
os.makedirs(output_folder_path_increase, exist_ok=True)
os.makedirs(output_folder_path_decrease, exist_ok=True)

# Perform initialization operations on the images
data_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

# Iterate through image files
for file_name in image_files:
    input_file_path = os.path.join(input_folder, file_name)
    output_file_path_increase = os.path.join(output_folder_path_increase, file_name)
    output_file_path_decrease = os.path.join(output_folder_path_decrease, file_name)

    image = Image.open(input_file_path)
    image = image.resize((224, 224))
    actual_image_matrix = np.array(image)

    actual_image_R_channel_increase = actual_image_matrix[:, :, 0].copy()
    actual_image_G_channel_increase = actual_image_matrix[:, :, 1].copy()
    actual_image_B_channel_increase = actual_image_matrix[:, :, 2].copy()

    actual_image_R_channel_decrease = actual_image_matrix[:, :, 0].copy()
    actual_image_G_channel_decrease = actual_image_matrix[:, :, 1].copy()
    actual_image_B_channel_decrease = actual_image_matrix[:, :, 2].copy()

    img = Image.open(input_file_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)
    img_tensor = data_transform(img)
    input_tensor = torch.unsqueeze(img_tensor, dim=0)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0].numpy()

    target_category = [np.argmax(probabilities)]
    print(file_name)
    print(target_category)

    image = cv2.imread(input_file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = image.transpose((2, 0, 1))
    image = image / 255.0
    image = torch.from_numpy(image).float().unsqueeze(0)

    # Create a Grad-CAM++ object and visualize the results for the predicted class
    grad_cam_plus_plus = GradCAMPlusPlus(model, model.layer4[-1])
    output = model(image)
    _, predicted_class = torch.max(output, 1)
    grayscale_cam = grad_cam_plus_plus(image, target_class=target_category)

    for i in range(224):
        for j in range(224):
            if grayscale_cam[i][j] >= 0.5:

                if (math.ceil(actual_image_R_channel_increase[i][j] * ratio_increase)) > 255:
                    actual_image_R_channel_increase[i][j] = 255
                elif (math.ceil(actual_image_R_channel_increase[i][j] * ratio_increase)) <= 255:
                    actual_image_R_channel_increase[i][j] = math.ceil(
                        actual_image_R_channel_increase[i][j] * ratio_increase)

                if (math.ceil(actual_image_G_channel_increase[i][j] * ratio_increase)) > 255:
                    actual_image_G_channel_increase[i][j] = 255
                elif (math.ceil(actual_image_G_channel_increase[i][j] * ratio_increase)) <= 255:
                    actual_image_G_channel_increase[i][j] = math.ceil(
                        actual_image_G_channel_increase[i][j] * ratio_increase)

                if (math.ceil(actual_image_B_channel_increase[i][j] * ratio_increase)) > 255:
                    actual_image_B_channel_increase[i][j] = 255
                elif (math.ceil(actual_image_B_channel_increase[i][j] * ratio_increase)) <= 255:
                    actual_image_B_channel_increase[i][j] = math.ceil(
                        actual_image_B_channel_increase[i][j] * ratio_increase)

                actual_image_R_channel_decrease[i][j] = int(actual_image_R_channel_decrease[i][j] * ratio_decrease)
                actual_image_G_channel_decrease[i][j] = int(actual_image_G_channel_decrease[i][j] * ratio_decrease)
                actual_image_B_channel_decrease[i][j] = int(actual_image_B_channel_decrease[i][j] * ratio_decrease)

    image_rgb = np.stack([actual_image_R_channel_increase, actual_image_G_channel_increase, actual_image_B_channel_increase], axis=-1)
    image_rgb = image_rgb.astype(np.uint8)
    image_pil = Image.fromarray(image_rgb)
    image_pil.save(output_file_path_increase)

    image_rgb = np.stack([actual_image_R_channel_decrease, actual_image_G_channel_decrease, actual_image_B_channel_decrease], axis=-1)
    image_rgb = image_rgb.astype(np.uint8)
    image_pil = Image.fromarray(image_rgb)
    image_pil.save(output_file_path_decrease)
