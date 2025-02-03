import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import json
import re
import numpy as np
import time
from torchvision.models import alexnet


def sort_func(file_name):
    return int(''.join(filter(str.isdigit, file_name)))


start_time = time.time()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

AlexNet = alexnet(num_classes=1000).to(device)

# Load model weights
weights_path = "..."
assert os.path.exists(weights_path), "file: '{}' does not exist.".format(weights_path)
AlexNet.load_state_dict(torch.load(weights_path, map_location=device))

# Import the original image of the pixel weight matrix to be calculated
image_folder_path = "..."

image_files = sorted([f for f in os.listdir(image_folder_path) if f.endswith('.png')], key=sort_func)

# Read class_indict
json_path = "..."
assert os.path.exists(json_path), "file: '{}' does not exist.".format(json_path)

with open(json_path, "r") as f:
    class_indict = json.load(f)

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

for image_file in image_files:
    image_path = os.path.join(image_folder_path, image_file)

    image = Image.open(image_path)
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0).to(device)

    AlexNet.eval()
    with torch.no_grad():
        output = AlexNet(input_batch)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top5_prob, top5_indices = torch.topk(output[0], 3)

    print(f"\nResults for image: {image_file}")

    index = top5_indices[0].item()

    img = Image.open(image_path)
    plt.imshow(img)
    img = preprocess(img)
    img = torch.unsqueeze(img, dim=0).to(device)  # Ensure the image is on the same device as the model

    AlexNet.eval()

    img.requires_grad_()
    output = AlexNet(img)

    # Target category index number
    pred_score = output[0, index]
    pred_score.backward()
    gradients = img.grad

    pixel_weight_matrix_R = gradients[0, 0, :, :].cpu().detach().numpy()
    pixel_weight_matrix_G = gradients[0, 1, :, :].cpu().detach().numpy()
    pixel_weight_matrix_B = gradients[0, 2, :, :].cpu().detach().numpy()

    match = re.search(r'(\d+).png', image_path)
    extracted_number = match.group(1)
    print("image number：")
    print(extracted_number)
    print("index：")
    print(index)

    image_name = extracted_number

    np.savetxt(f"...\\{image_name}.txt",
               pixel_weight_matrix_R.reshape(-1, pixel_weight_matrix_R.shape[-1]), fmt='%.10f', delimiter='\t')
    np.savetxt(f"...\\{image_name}.txt",
               pixel_weight_matrix_G.reshape(-1, pixel_weight_matrix_G.shape[-1]), fmt='%.10f', delimiter='\t')
    np.savetxt(f"...\\{image_name}.txt",
               pixel_weight_matrix_B.reshape(-1, pixel_weight_matrix_B.shape[-1]), fmt='%.10f', delimiter='\t')

end_time = time.time()
print(f"Total processing time: {end_time - start_time} seconds")