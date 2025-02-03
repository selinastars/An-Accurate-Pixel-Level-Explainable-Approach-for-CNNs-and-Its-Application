import os
import json
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import vgg19
import numpy as np

def sort_func(file_name):
    return int(''.join(filter(str.isdigit, file_name)))

def read_matrix(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        matrix = [list(map(float, line.split())) for line in lines]
    return np.array(matrix)

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Import the absolute path of the folder storing the original images
    folder_path = r"..."

    # Read class_indict
    json_path = "..."
    assert os.path.exists(json_path), "file: '{}' does not exist.".format(json_path)
    with open(json_path, "r") as f:
        class_indict = json.load(f)

    model = vgg19(num_classes=1000).to(device)

    # Load model weights
    weights_path = "..."
    assert os.path.exists(weights_path), "file: '{}' does not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path))

    model.eval()

    file_list = os.listdir(folder_path)
    file_list = sorted(file_list, key=sort_func)

    false_image = []
    num = 0
    for file_name in file_list:
        num = num + 1
        file_path = os.path.join(folder_path, file_name)

        img = Image.open(file_path)

        img = data_transform(img)
        img = torch.unsqueeze(img, dim=0)

        # Get the image number from the file name
        image_number = os.path.splitext(file_name)[0]

        # Construct matrix file paths based on the image number
        pixel_weight_matrix_R_path  = os.path.join(r"...", f"{image_number}.txt")
        pixel_weight_matrix_G_path  = os.path.join(r"...", f"{image_number}.txt")
        pixel_weight_matrix_B_path  = os.path.join(r"...", f"{image_number}.txt")

        print("image number：")
        print(image_number)

        # Read pixel_weight_matrix
        pixel_weight_matrix_R  = read_matrix(pixel_weight_matrix_R_path)
        pixel_weight_matrix_G = read_matrix(pixel_weight_matrix_G_path)
        pixel_weight_matrix_B = read_matrix(pixel_weight_matrix_B_path)

        # # Read BIAS
        bias_path = os.path.join(r"...", f"{image_number}.txt")
        bias = read_matrix(bias_path)

        img = img.to(device)

        with torch.no_grad():
            output = torch.squeeze(model(img)).cpu()
            predict = torch.softmax(output, dim=0)
            class_id = torch.argmax(predict).item()
            class_name = class_indict[str(class_id)]

        top_prob, top_index = torch.max(predict, 0)

        class_index = top_index.item()
        classification_probability = top_prob.item()
        classification_value = output[class_index].item()

        print("File: {}  Top 1: index: {}  class: {:10}   Classification probability: {:.10f}  Classification value: {:.10f}".format(file_name,
                                                                                                class_index,
                                                                                                class_name,
                                                                                                classification_probability,
                                                                                                classification_value
                                                                                               ))
        # Image processing
        image_array = np.array(img.squeeze().cpu())

        image_red_channel = image_array[0, :, :]
        image_green_channel = image_array[1, :, :]
        image_blue_channel = image_array[2, :, :]

        # # Calculate the pixel classification contribution matrix
        pixel_classification_contribution_matrix_R = image_red_channel * pixel_weight_matrix_R
        pixel_classification_contribution_matrix_G = image_green_channel * pixel_weight_matrix_G
        pixel_classification_contribution_matrix_B = image_blue_channel * pixel_weight_matrix_B

        # Sum all elements in the resulting matrices
        pixel_classification_contribution = np.sum(pixel_classification_contribution_matrix_R) + np.sum(pixel_classification_contribution_matrix_G) + np.sum(pixel_classification_contribution_matrix_B) + bias

        print("Sum of all elements in the resulting matrices: ", pixel_classification_contribution)

        if abs(pixel_classification_contribution - classification_value) > 0.1:
            print(abs(pixel_classification_contribution - classification_value))
            false_image.append(image_number)
            print(image_number)

    print("The number of errors is：")
    print(len(false_image))
    print("The incorrect sequence number is：")
    print(false_image)
    print("The number of processed images is：")
    print(num)

if __name__ == '__main__':
    main()
