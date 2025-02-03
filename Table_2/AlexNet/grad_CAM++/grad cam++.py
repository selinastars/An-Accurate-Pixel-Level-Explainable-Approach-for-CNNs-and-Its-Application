import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from PIL import Image
from torchvision import transforms
from torchvision.models import alexnet

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

# Load the AlexNet model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = alexnet(num_classes=1000)

# Load model weights
model_weight_path = "..."
model.load_state_dict(torch.load(model_weight_path, map_location=device))
model.eval()

# Define a sorting function
def sort_func(file_name):
    return int(''.join(filter(str.isdigit, file_name)))

# Import actual image
actual_image_folder_absolute_path = "..."

# Specify the saliency map save address
output_folder_absolute_path_saliency_map = "..."

# Retrieve the names of image files in the folder and sort them based on the numeric part
image_files = sorted(os.listdir(actual_image_folder_absolute_path), key=sort_func)

# Create an output folder
os.makedirs(output_folder_absolute_path_saliency_map, exist_ok=True)

# Perform initialization operations on the images
data_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

# Iterate through image files
for file_name in image_files:
    image_path = os.path.join(actual_image_folder_absolute_path, file_name)
    input_file_path = image_path

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

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = image.transpose((2, 0, 1))
    image = image / 255.0
    image = torch.from_numpy(image).float().unsqueeze(0)

    # Create a Grad-CAM++ object and visualize the results for the predicted class
    grad_cam_plus_plus = GradCAMPlusPlus(model, model.features[-1])
    output = model(image)
    _, predicted_class = torch.max(output, 1)
    grad_cam = grad_cam_plus_plus(image, target_class=target_category)


    # Visualize the results
    heatmap = cv2.applyColorMap(np.uint8(255 * grad_cam), cv2.COLORMAP_JET)
    heatmap = heatmap / 255.0
    result = heatmap + (image.squeeze().permute(1, 2, 0)).detach().numpy()
    result = result / np.max(result)

    # Construct the output path
    output_path = os.path.join(output_folder_absolute_path_saliency_map, f'{os.path.splitext(file_name)[0]}.png')
    cv2.imwrite(output_path, result*255)

    # Display the saved image
    saved_image = cv2.imread(output_path)
    saved_image = cv2.cvtColor(saved_image, cv2.COLOR_BGR2RGB)

    plt.imshow(saved_image)
    plt.axis('off')
    plt.show()
