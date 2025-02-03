import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import json
import re
import time
import torchvision.models as models


def maxpool_specify_trajectory(input, map):
    map_shape = map.shape
    input = input.view(input.shape[0], input.shape[1], -1)
    map = map.view(map.shape[0], map.shape[1], -1)
    result = torch.gather(input, dim=2, index=map)
    result = result.view(map_shape)
    return result

def relu_specify_trajectory(input, map):
    return input * map

def sort_func(file_name):
    return int(''.join(filter(str.isdigit, file_name)))

start_time = time.time()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

resnet = models.resnet18(pretrained=False).to(device)

# Load model weights
weights_path = "..."
assert os.path.exists(weights_path), "file: '{}' none.".format(weights_path)
resnet.load_state_dict(torch.load(weights_path, map_location=device))

first_relu_found = False

block_1_0 = resnet.layer1[0]
block_1_1 = resnet.layer1[1]

block_2_0 = resnet.layer2[0]
block_2_1 = resnet.layer2[1]

block_3_0 = resnet.layer3[0]
block_3_1 = resnet.layer3[1]

block_4_0 = resnet.layer4[0]
block_4_1 = resnet.layer4[1]

# Import the actual image of the bias to be calculated
image_folder_path = "..."
image_files = sorted([f for f in os.listdir(image_folder_path) if f.endswith('.png')], key=sort_func)
relu = nn.ReLU(inplace=True)

for image_file in image_files:
    image_path = os.path.join(image_folder_path, image_file)

    b_weight = torch.zeros((1, 1000), dtype=torch.float32)
    relu_map = []
    maxpool_map = []
    modules = []
    first_relu_found = False
    block_1_0 = resnet.layer1[0]
    block_1_1 = resnet.layer1[1]

    block_2_0 = resnet.layer2[0]
    block_2_1 = resnet.layer2[1]

    block_3_0 = resnet.layer3[0]
    block_3_1 = resnet.layer3[1]

    block_4_0 = resnet.layer4[0]
    block_4_1 = resnet.layer4[1]

    image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0).to(device)

    resnet.eval()
    with torch.no_grad():

        for name, module in resnet.named_children():
            modules.append(module)
            if isinstance(module, nn.ReLU) and not first_relu_found:
                relumap = (input_batch > 0)
                relu_map.append(relumap.clone())
                first_relu_found = True
                input_batch = module(input_batch)

            elif isinstance(module, nn.MaxPool2d):
                module.return_indices = True
                input_batch, indices = module(input_batch)
                maxpool_map.append(indices.clone())
                break
            else:
                input_batch = module(input_batch)

        actual_value = input_batch

        for name, module in block_1_0.named_children():
            modules.append(module)
            if isinstance(module, nn.ReLU):
                relumap = (input_batch > 0)
                relu_map.append(relumap.clone())
                input_batch = module(input_batch)
            else:
                input_batch = module(input_batch)

        actual_value = input_batch + actual_value
        modules.append(nn.ReLU(inplace=True))
        relumap = (actual_value > 0)
        relu_map.append(relumap.clone())
        actual_value = relu(actual_value)

        input_batch = actual_value
        for name, module in block_1_1.named_children():
            modules.append(module)
            if isinstance(module, nn.ReLU):
                relumap = (input_batch > 0)
                relu_map.append(relumap.clone())
                input_batch = module(input_batch)
            else:
                input_batch = module(input_batch)

        actual_value = input_batch + actual_value

        modules.append(nn.ReLU(inplace=True))
        relumap = (actual_value > 0)
        relu_map.append(relumap.clone())
        actual_value = relu(actual_value)
        ##############################################################################################
        input_batch = actual_value
        for name, module in block_2_0.named_children():
            modules.append(module)
            if isinstance(module, nn.ReLU):
                relumap = (input_batch > 0)
                relu_map.append(relumap.clone())
                input_batch = module(input_batch)
            if isinstance(module, nn.Sequential) and len(module) == 2:
                downsample_conv, downsample_bn = module
                actual_value = downsample_bn(downsample_conv(actual_value))
            else:
                input_batch = module(input_batch)
        actual_value = input_batch + actual_value

        modules.append(nn.ReLU(inplace=True))
        relumap = (actual_value > 0)
        relu_map.append(relumap.clone())
        actual_value = relu(actual_value)

        input_batch = actual_value
        for name, module in block_2_1.named_children():
            modules.append(module)
            if isinstance(module, nn.ReLU):
                relumap = (input_batch > 0)
                relu_map.append(relumap.clone())
                input_batch = module(input_batch)
            else:
                input_batch = module(input_batch)

        actual_value = actual_value + input_batch

        modules.append(nn.ReLU(inplace=True))
        relumap = (actual_value > 0)
        relu_map.append(relumap.clone())
        actual_value = relu(actual_value)

        ##############################################################################################
        input_batch = actual_value
        for name, module in block_3_0.named_children():
            modules.append(module)
            if isinstance(module, nn.ReLU):
                relumap = (input_batch > 0)
                relu_map.append(relumap.clone())
                input_batch = module(input_batch)
            if isinstance(module, nn.Sequential) and len(module) == 2:
                downsample_conv, downsample_bn = module
                actual_value = downsample_bn(downsample_conv(actual_value))
            else:
                input_batch = module(input_batch)
        actual_value = input_batch + actual_value

        modules.append(nn.ReLU(inplace=True))
        relumap = (actual_value > 0)
        relu_map.append(relumap.clone())
        actual_value = relu(actual_value)

        input_batch = actual_value
        for name, module in block_3_1.named_children():
            modules.append(module)
            if isinstance(module, nn.ReLU):
                relumap = (input_batch > 0)
                relu_map.append(relumap.clone())
                input_batch = module(input_batch)
            else:
                input_batch = module(input_batch)

        actual_value = actual_value + input_batch

        modules.append(nn.ReLU(inplace=True))
        relumap = (actual_value > 0)
        relu_map.append(relumap.clone())
        actual_value = relu(actual_value)
        ##############################################################################################
        input_batch = actual_value
        for name, module in block_4_0.named_children():
            modules.append(module)
            if isinstance(module, nn.ReLU):
                relumap = (input_batch > 0)
                relu_map.append(relumap.clone())
                input_batch = module(input_batch)
            if isinstance(module, nn.Sequential) and len(module) == 2:
                # continue
                downsample_conv, downsample_bn = module
                actual_value = downsample_bn(downsample_conv(actual_value))
            else:
                input_batch = module(input_batch)
        actual_value = input_batch + actual_value

        modules.append(nn.ReLU(inplace=True))
        relumap = (actual_value > 0)
        relu_map.append(relumap.clone())
        actual_value = relu(actual_value)

        input_batch = actual_value
        for name, module in block_4_1.named_children():
            modules.append(module)
            if isinstance(module, nn.ReLU):
                relumap = (input_batch > 0)
                relu_map.append(relumap.clone())
                input_batch = module(input_batch)
            else:
                input_batch = module(input_batch)

        actual_value = actual_value + input_batch

        modules.append(nn.ReLU(inplace=True))
        relumap = (actual_value > 0)
        relu_map.append(relumap.clone())
        actual_value = relu(actual_value)

        avgpool_layer = resnet.avgpool
        actual_value = avgpool_layer(actual_value)
        modules.append(avgpool_layer)

        actual_value = actual_value.view(actual_value.size(0), -1)
        fc = resnet.fc
        actual_value = fc(actual_value)
        modules.append(fc)

        cnt = 0
        for i in range(1):
            for j in range(1):
                for k in range(1):
                    for l in range(1):
                        if cnt % 500 == 0:
                            print("!!!")
                            print(str(cnt) + '/' + str(1))
                            print("!!!")
                        cnt += 1
                        img = torch.zeros((1, 3, 224, 224))
                        img[i, j, k, l] = 0
                        img = img.to(device)
                        img = modules[0](img)
                        img = modules[1](img)
                        img = relu_specify_trajectory(img, relu_map[0])
                        img = maxpool_specify_trajectory(img, maxpool_map[0])

                        ####################################################
                        temp = img
                        img = modules[4](img)
                        img = modules[5](img)
                        img = relu_specify_trajectory(img, relu_map[1])
                        img = modules[7](img)
                        img = modules[8](img)
                        img = img + temp
                        img = relu_specify_trajectory(img, relu_map[2])

                        temp = img
                        img = modules[10](img)
                        img = modules[11](img)
                        img = relu_specify_trajectory(img, relu_map[3])
                        img = modules[13](img)
                        img = modules[14](img)
                        img = img + temp
                        img = relu_specify_trajectory(img, relu_map[4])
                        ####################################################
                        temp = img
                        img = modules[16](img)
                        img = modules[17](img)
                        img = relu_specify_trajectory(img, relu_map[5])
                        img = modules[19](img)
                        img = modules[20](img)
                        downsample_conv, downsample_bn = modules[21]
                        temp = downsample_bn(downsample_conv(temp))
                        img = img + temp
                        img = relu_specify_trajectory(img, relu_map[6])

                        temp = img
                        img = modules[23](img)
                        img = modules[24](img)
                        img = relu_specify_trajectory(img, relu_map[7])
                        img = modules[26](img)
                        img = modules[27](img)
                        img = img + temp
                        img = relu_specify_trajectory(img, relu_map[8])
                        ####################################################
                        temp = img
                        img = modules[29](img)
                        img = modules[30](img)
                        img = relu_specify_trajectory(img, relu_map[9])
                        img = modules[32](img)
                        img = modules[33](img)
                        downsample_conv, downsample_bn = modules[34]
                        temp = downsample_bn(downsample_conv(temp))
                        img = img + temp
                        img = relu_specify_trajectory(img, relu_map[10])

                        temp = img
                        img = modules[36](img)
                        img = modules[37](img)
                        img = relu_specify_trajectory(img, relu_map[11])
                        img = modules[39](img)
                        img = modules[40](img)
                        img = img + temp
                        img = relu_specify_trajectory(img, relu_map[12])
                        ####################################################
                        temp = img
                        img = modules[42](img)
                        img = modules[43](img)
                        img = relu_specify_trajectory(img, relu_map[13])
                        img = modules[45](img)
                        img = modules[46](img)
                        downsample_conv, downsample_bn = modules[47]
                        temp = downsample_bn(downsample_conv(temp))
                        img = img + temp
                        img = relu_specify_trajectory(img, relu_map[14])

                        temp = img
                        img = modules[49](img)
                        img = modules[50](img)
                        img = relu_specify_trajectory(img, relu_map[15])
                        img = modules[52](img)
                        img = modules[53](img)
                        img = img + temp
                        img = relu_specify_trajectory(img, relu_map[16])
                        ###################################################
                        img = modules[55](img)
                        img = img.view(img.size(0), -1)
                        img = modules[56](img)

                        b_weight[i] = img

    print(b_weight.shape)
    torch.save(b_weight, "b_weight.pt")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    resnet = models.resnet18(pretrained=False).to(device)

    # Load model weights
    weights_path = "..."
    assert os.path.exists(weights_path), "file: '{}' none.".format(weights_path)
    resnet.load_state_dict(torch.load(weights_path, map_location=device))

    # Read class_indict
    json_path = "..."
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0).to(device)

    resnet.eval()
    with torch.no_grad():
        output = resnet(input_batch)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top5_prob, top5_indices = torch.topk(output[0], 3)

    print(f"\nResults for image: {image_file}")

    index = top5_indices[0]

    b_weight = torch.load("b_weight.pt")

    match = re.search(r'(\d+).png', image_path)
    extracted_number = match.group(1)
    print("image number：")
    print(extracted_number)
    print("index：")
    print(index)

    image_name = extracted_number

    biad = b_weight[0, index]
    print("bias：")
    print(biad)

    with open(f"...\\{image_name}.txt", 'w') as file:
        file.write(str(biad.item()))




