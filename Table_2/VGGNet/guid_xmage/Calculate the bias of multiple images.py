import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import json
import re
import time
from torchvision.models import vgg19

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

vggnet = vgg19(num_classes=1000).to(device)

# Load model weights
weights_path = "..."
assert os.path.exists(weights_path), "file: '{}' none.".format(weights_path)
vggnet.load_state_dict(torch.load(weights_path, map_location=device))

# Import the original image of the bias to be calculated
image_folder_path = "..."

# Import the original image of the bias to be calculated
image_files = sorted([f for f in os.listdir(image_folder_path) if f.endswith('.png')])

for image_file in image_files:
    image_path = os.path.join(image_folder_path, image_file)

    b_weight = torch.zeros((1, 1000), dtype=torch.float32)
    relu_map = []
    maxpool_map = []
    modules = []

    image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0).to(device)

    vggnet.eval()
    with torch.no_grad():
        for name, module in vggnet.features.named_children():
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

        avgpool_layer = vggnet.avgpool
        input_batch = avgpool_layer(input_batch)
        modules.append(avgpool_layer)

        for name, module in vggnet.classifier.named_children():
            if isinstance(module, nn.Dropout):
                continue
            if isinstance(module, nn.ReLU):
                relumap = (input_batch >= 0)
                relu_map.append(relumap.clone())
            if isinstance(module, nn.Linear):
                input_batch = input_batch.view(input_batch.size(0), -1)
            input_batch = module(input_batch)
            modules.append(module)

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

                        # layer0-conv2d
                        img = modules[0](img)
                        # layer1-relu
                        img = relu_specify_trajectory(img, relu_map[0])
                        # layer2-conv2d
                        img = modules[2](img)
                        # layer3-relu
                        img = relu_specify_trajectory(img, relu_map[1])
                        # layer4-maxpool2d
                        img = maxpool_specify_trajectory(img, maxpool_map[0])

                        # layer5-conv2d
                        img = modules[5](img)
                        # layer6-relu
                        img = relu_specify_trajectory(img, relu_map[2])
                        # layer7-conv2d
                        img = modules[7](img)
                        # layer8-relu
                        img = relu_specify_trajectory(img, relu_map[3])
                        # layer9-maxpool2d
                        img = maxpool_specify_trajectory(img, maxpool_map[1])

                        # layer10-conv2d
                        img = modules[10](img)
                        # layer11-relu
                        img = relu_specify_trajectory(img, relu_map[4])
                        # layer12-conv2d
                        img = modules[12](img)
                        # layer13-relu
                        img = relu_specify_trajectory(img, relu_map[5])
                        # layer14-conv2d
                        img = modules[14](img)
                        # layer15-relu
                        img = relu_specify_trajectory(img, relu_map[6])
                        # layer16-conv2d
                        img = modules[16](img)
                        # layer17-relu
                        img = relu_specify_trajectory(img, relu_map[7])
                        # layer18-maxpool2d
                        img = maxpool_specify_trajectory(img, maxpool_map[2])

                        # layer19-conv2d
                        img = modules[19](img)
                        # layer20-relu
                        img = relu_specify_trajectory(img, relu_map[8])
                        # layer21-conv2d
                        img = modules[21](img)
                        # layer22-relu
                        img = relu_specify_trajectory(img, relu_map[9])
                        # layer23-conv2d
                        img = modules[23](img)
                        # layer24-relu
                        img = relu_specify_trajectory(img, relu_map[10])
                        # layer25-conv2d
                        img = modules[25](img)
                        # layer26-relu
                        img = relu_specify_trajectory(img, relu_map[11])
                        # layer27-maxpool2d
                        img = maxpool_specify_trajectory(img, maxpool_map[3])

                        # layer28-conv2d
                        img = modules[28](img)
                        # layer29-relu
                        img = relu_specify_trajectory(img, relu_map[12])
                        # layer30-conv2d
                        img = modules[30](img)
                        # layer31-relu
                        img = relu_specify_trajectory(img, relu_map[13])
                        # layer32-conv2d
                        img = modules[32](img)
                        # layer33-relu
                        img = relu_specify_trajectory(img, relu_map[14])
                        # layer34-conv2d
                        img = modules[34](img)
                        # layer35-relu
                        img = relu_specify_trajectory(img, relu_map[15])
                        # layer36-maxpool2d
                        img = maxpool_specify_trajectory(img, maxpool_map[4])

                        # layer37-adaptiveAvgPool2d
                        img = modules[37](img)

                        img = img.view(img.size(0), -1)

                        # layer38-linear
                        img = modules[38](img)
                        # layer39-relu
                        img = relu_specify_trajectory(img, relu_map[16])
                        # layer40-linear
                        img = modules[40](img)
                        # layer41-relu
                        img = relu_specify_trajectory(img, relu_map[17])
                        # layer42-linear
                        img = modules[42](img)

                        b_weight[i] = img

    print(b_weight.shape)
    torch.save(b_weight, "b_weight.pt")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    vggnet = vgg19(num_classes=1000).to(device)

    # Load model weights
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    vggnet.load_state_dict(torch.load(weights_path, map_location=device))

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

    vggnet.eval()
    with torch.no_grad():
        output = vggnet(input_batch)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top5_prob, top5_indices = torch.topk(output[0], 3)

    print(f"\nResults for image: {image_file}")
    indiex = top5_indices[0]
    b_weight = torch.load("b_weight.pt")

    match = re.search(r'(\d+).png', image_path)
    extracted_number = match.group(1)
    print("image number：")
    print(extracted_number)
    print("index：")
    print(indiex)

    image_name = extracted_number

    bias = b_weight[0, indiex]
    print("bias：")
    print(bias)

    with open(f"...\\{image_name}.txt", 'w') as file:
        file.write(str(bias.item()))