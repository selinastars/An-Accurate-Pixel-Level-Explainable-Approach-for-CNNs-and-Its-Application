"""
The main function of this code file is to calculate the bias of multiple images.
Step 1, you need to set "weights_path" to the absolute path of the model weight file.
Step 2, you need to set "image_folder_path" to the absolute path of the folder storing the original images to be calculated.
Step 3, you need to set "json_path" to the absolute path of the index file containing information about the 1000 classes.
Step 4, you need to set the absolute path to the folder where the bias files will be saved in "np.savetxt".
"""


import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import json
import re
import time
from model_AlexNet import AlexNet

def maxpool_specify_trajectory(input, map):
    map_shape = map.shape
    input = input.view(input.shape[0], input.shape[1], -1)
    map = map.view(map.shape[0], map.shape[1], -1)
    result = torch.gather(input, dim=2, index=map)
    result = result.view(map_shape)
    return result

def relu_specify_trajectory(input, map):
    return input * map


start_time = time.time()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

alexnet = AlexNet(num_classes=1000).to(device)

# Load model weights
weights_path = "..."
assert os.path.exists(weights_path), "file: '{}' none.".format(weights_path)
alexnet.load_state_dict(torch.load(weights_path, map_location=device))

# Import the original image of the bias to be calculated
image_folder_path = "..."

image_files = sorted([f for f in os.listdir(image_folder_path) if f.endswith('.png')])

for image_file in image_files:
    image_path = os.path.join(image_folder_path, image_file)

    b_weight = torch.zeros((1, 1000), dtype=torch.float32)
    x_weight = torch.zeros((1, 3, 224, 224, 1000), dtype=torch.float32)
    relu_map = []
    maxpool_map = []
    modules = []

    # Open the image and preprocess it
    image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0).to(device)

    alexnet.eval()
    with torch.no_grad():
        # features
        for name, module in alexnet.features.named_children():
            modules.append(module)

            if isinstance(module, nn.ReLU):
                relumap = (input_batch > 0)
                relu_map.append(relumap.clone())

            if isinstance(module, nn.MaxPool2d):
                module.return_indices = True
                input_batch, indices = module(input_batch)
                maxpool_map.append(indices.clone())
            else:
                input_batch = module(input_batch)

        avgpool_layer = alexnet.avgpool
        input_batch = avgpool_layer(input_batch)
        modules.append(avgpool_layer)

        for name, module in alexnet.classifier.named_children():

            if isinstance(module, nn.Dropout):
                continue

            if isinstance(module, nn.ReLU):
                relumap = (input_batch >= 0)
                relu_map.append(relumap.clone())

            if isinstance(module, nn.Linear):
                input_batch = input_batch.view(input_batch.size(0), -1)
            input_batch = module(input_batch)
            modules.append(module)

        sum = x_weight.shape[0] * x_weight.shape[1] * x_weight.shape[2] * x_weight.shape[3]
        cnt = 0

        for i in range(1):
            for j in range(1):
                for k in range(1):
                    for l in range(1):
                        if cnt % 500 == 0:
                            print("!!!")
                            print(str(cnt) + '/' + str(sum))
                            print("Time:{}".format(int((time.time() - start_time) / 60)) + " mins")
                            print("!!!")
                        cnt += 1
                        img = torch.zeros((1, 3, 224, 224))
                        img[i, j, k, l] = 0
                        img = img.to(device)
                        # layer1-conv2d
                        img = modules[0](img)
                        # layer2-relu
                        img = relu_specify_trajectory(img, relu_map[0])
                        # layer3-maxpool2d
                        img = maxpool_specify_trajectory(img, maxpool_map[0])
                        # layer4-conv2d
                        img = modules[3](img)
                        # layer5-relu
                        img = relu_specify_trajectory(img, relu_map[1])
                        # layer6-maxpool2d
                        img = maxpool_specify_trajectory(img, maxpool_map[1])
                        # layer7-conv2d
                        img = modules[6](img)
                        # layer8-relu
                        img = relu_specify_trajectory(img, relu_map[2])
                        # layer9-conv2d
                        img = modules[8](img)
                        # layer10-relu
                        img = relu_specify_trajectory(img, relu_map[3])
                        # layer11-conv2d
                        img = modules[10](img)
                        # layer12-relu
                        img = relu_specify_trajectory(img, relu_map[4])
                        # layer13-maxpool2d
                        img = maxpool_specify_trajectory(img, maxpool_map[2])
                        # layer14-adaptiveAvgPool2d
                        img = modules[13](img)
                        img = img.view(img.size(0), -1)
                        # layer15-linear
                        img = modules[14](img)
                        # layer16-relu
                        img = relu_specify_trajectory(img, relu_map[5])
                        # layer17-linear
                        img = modules[16](img)
                        # layer18-relu
                        img = relu_specify_trajectory(img, relu_map[6])
                        # layer19-linear
                        img = modules[18](img)
                        b_weight[i] = img


    torch.save(b_weight, "b_weight.pt")

    # The following is the first step in determining the probability of the image
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    alexnet = AlexNet(num_classes=1000).to(device)

    assert os.path.exists(weights_path), "file: '{}' none.".format(weights_path)
    alexnet.load_state_dict(torch.load(weights_path, map_location=device))

    # Read class_indict
    json_path = "..."
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    mage = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0).to(device)

    alexnet.eval()
    with torch.no_grad():
        output = alexnet(input_batch)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    # Obtain the top five categories with the highest classification probability
    top5_prob, top5_indices = torch.topk(output[0], 5)

    print(f"\nResults for image: {image_file}")
    for i in range(5):
        class_index = top5_indices[i].item()
        class_prob = top5_prob[i].item()
        class_prob_full = probabilities[class_index].item()
    index = top5_indices[0]

    b_weight = torch.load("b_weight.pt")

    match = re.search(r'(\d+).png', image_path)
    extracted_number = match.group(1)
    print("Image Number：")
    print(extracted_number)
    print("Category index number：")
    print(index)

    image_name = extracted_number

    bias = b_weight[0, index]
    print("bias：")
    print(bias)

    # Save bias values as text files
    with open(f"...\\{image_name}.txt", 'w') as file:
        file.write(str(bias.item()))