"""
The main function of this code file is to evaluate the scores of experimental images generated based on VGGNet that aim to decrease classification values.
First, you need to import the VGGNet model using `from guid_Xmage.experiment_5.VGGNet.mode_VGG import VGG`.
Second, you need to set the parameter `weights_absolute_path` to the absolute path of the weight file of the officially trained VGGNet model.
Third, you need to set the parameter `json_absolute_path` to the absolute path of the index file containing the 1000 classification labels of the model.
Fourth, you need to set the parameter `actual_image_folder_absolute_path` to the absolute path of the folder storing the original images.
Fifth, you need to set the parameter `tampered_image_absolute_path` to the absolute path of the folder storing the tampered images.
Finally, execute this code file to obtain the scores and other related information.
"""


import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import json
import glob
from torchvision.models import vgg19

def sort_func(file_name):
    return int(''.join(filter(str.isdigit, file_name)))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Instantiate the VGGNet model and move it to the device
vgg = vgg19(num_classes=1000).to(device)

# Load the model weights
weights_absolute_path = "..."
assert os.path.exists(weights_absolute_path), "File '{}' does not exist.".format(weights_absolute_path)
vgg.load_state_dict(torch.load(weights_absolute_path, map_location=device))

# Read class_indict
json_absolute_path = "..."
assert os.path.exists(json_absolute_path), "File '{}' does not exist.".format(json_absolute_path)

with open(json_absolute_path, "r") as f:
    class_indict = json.load(f)

# Read image folder
actual_image_folder_absolute_path = "..."
image_paths_actual = sorted(glob.glob(os.path.join(actual_image_folder_absolute_path, "*.png")), key=sort_func)

# Read tampered image folder
tampered_image_absolute_path = "..."
tampered_images = sorted(glob.glob(os.path.join(tampered_image_absolute_path, "*.png")), key=sort_func)

score = 0

for img_path_actual, img_path_tampere in zip(image_paths_actual, tampered_images):
    index = sort_func(os.path.basename(img_path_actual))

    image_actual = Image.open(img_path_actual)
    preprocess_actual = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor_actual = preprocess_actual(image_actual)
    input_batch_actual = input_tensor_actual.unsqueeze(0).to(device)

    vgg.eval()
    with torch.no_grad():
        output_actual = vgg(input_batch_actual)

    # Get the index of the class with the highest probability
    top1_prob, top1_index = torch.topk(output_actual[0], 1)
    class_index = top1_index.item()

    # Get the probability for the class index
    classification_value_actual_image = output_actual[0][class_index].item()
    classification_probability_actual_image = torch.nn.functional.softmax(output_actual[0], dim=0)[class_index].item()

    print(f"image {index}：")
    print(f"Probability prediction for the actual image: class index {class_index}, Class name {class_indict[str(class_index)]}, Classification probability {classification_probability_actual_image}, Classification value {classification_value_actual_image}")

    image_tampere = Image.open(img_path_tampere)
    preprocess_tampere = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor_tampere = preprocess_tampere(image_tampere)
    input_batch_tampere = input_tensor_tampere.unsqueeze(0).to(device)

    vgg.eval()
    with torch.no_grad():
        output_tampere = vgg(input_batch_tampere)

    # Get the probability for the class index
    classification_value_tampere_image = output_tampere[0][class_index].item()
    classification_probability_tampere_image = torch.nn.functional.softmax(output_tampere[0], dim=0)[class_index].item()

    # Output the results
    print(f"Tampered image probability prediction: class index {class_index}, Class name {class_indict[str(class_index)]}, Classification probability {classification_probability_tampere_image}, Classification value {classification_value_tampere_image}")
    print("=" * 50)

    if classification_value_tampere_image - classification_value_actual_image < 0:
        print(classification_value_tampere_image - classification_value_actual_image)
        score = score + 1

print("score：")
print(score)