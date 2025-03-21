import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from utils import GradCAM, show_cam_on_image
from torchvision.models import vgg19


def sort_func(file_name):
    return int(''.join(filter(str.isdigit, file_name)))

def save_cam_to_txt(file_path, cam):
    np.savetxt(file_path, cam, fmt='%f')

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = vgg19(num_classes=1000)

    # Load model weights
    model_weight_path = "..."
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    target_layers = [model.features[-1]]

    # Perform initialization operations on the images
    data_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    # Import actual image
    input_folder_path = "..."

    # Specify the saliency map save address
    output_folder_path = "..."
    # Specify image start reading number
    starting_image_number = 1

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    file_list = os.listdir(input_folder_path)
    file_list = sorted(file_list, key=sort_func)

    for file_name in file_list:
        print(file_name)
        image_number = int(''.join(filter(str.isdigit, file_name)))

        if image_number < starting_image_number:
            continue

        print(file_name)
        input_file_path = os.path.join(input_folder_path, file_name)
        output_file_path = os.path.join(output_folder_path, file_name)

        img = Image.open(input_file_path).convert('RGB')
        img = np.array(img, dtype=np.uint8)

        img_tensor = data_transform(img)
        input_tensor = torch.unsqueeze(img_tensor, dim=0)

        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)[0].numpy()

        target_category = [np.argmax(probabilities)]
        print(target_category)

        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)

        grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
        grayscale_cam = grayscale_cam[0, :]

        visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255., grayscale_cam, use_rgb=True)

        image = Image.fromarray((visualization).astype(np.uint8))
        image.save(output_file_path)

if __name__ == '__main__':
    main()
