import argparse
import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import math
from pytorch_grad_cam import ScoreCAM
import torchvision.models as models


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--aug-smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument('--eigen-smooth', action='store_true',
                        help='Reduce noise by taking the first principle component'
                             'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='scorecam',
                        choices=['scorecam'],
                        help='CAM method')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    device = torch.device('cuda' if args.use_cuda else 'cpu')
    model = models.resnet18(pretrained=False).to(device)

    # Load model weights
    model_weight_absolute_path = torch.load("...", map_location=device)
    model.load_state_dict(model_weight_absolute_path, strict=False)

    target_layers = [model.layer4[-1]]

    actual_image_folder_absolute_path = r"..."

    # Specify the tampering images save address
    output_folder_path_increase = "..."
    output_folder_path_decrease = "..."

    ratio_increase = 1.01
    ratio_decrease = 0.99

    # Create an output folder
    os.makedirs(output_folder_path_increase, exist_ok=True)
    os.makedirs(output_folder_path_decrease, exist_ok=True)

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image_files = sorted([filename for filename in os.listdir(actual_image_folder_absolute_path) if filename.endswith(".png")],
                         key=lambda x: int(os.path.splitext(x)[0]))

    for file_name in image_files:

        input_file_path = os.path.join(actual_image_folder_absolute_path, file_name)
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

        image = Image.open(input_file_path).convert('RGB')
        rgb_img = np.array(image)
        rgb_img = rgb_img.astype(np.float32) / 255.0

        input_tensor = preprocess(rgb_img).to(device)
        input_tensor = input_tensor.unsqueeze(0)

        cam_algorithm = ScoreCAM
        with cam_algorithm(model=model, target_layers=target_layers) as cam:
            cam.batch_size = 4
            grayscale_cam = cam(input_tensor=input_tensor,aug_smooth=args.aug_smooth,eigen_smooth=args.eigen_smooth)
            grayscale_cam = grayscale_cam[0, :]

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

                        actual_image_R_channel_decrease[i][j] = int(
                            actual_image_R_channel_decrease[i][j] * ratio_decrease)
                        actual_image_G_channel_decrease[i][j] = int(
                            actual_image_G_channel_decrease[i][j] * ratio_decrease)
                        actual_image_B_channel_decrease[i][j] = int(
                            actual_image_B_channel_decrease[i][j] * ratio_decrease)

            image_rgb = np.stack([actual_image_R_channel_increase, actual_image_G_channel_increase, actual_image_B_channel_increase], axis=-1)
            image_rgb = image_rgb.astype(np.uint8)
            image_pil = Image.fromarray(image_rgb)
            image_pil.save(output_file_path_increase)

            image_rgb = np.stack([actual_image_R_channel_decrease, actual_image_G_channel_decrease, actual_image_B_channel_decrease], axis=-1)
            image_rgb = image_rgb.astype(np.uint8)
            image_pil = Image.fromarray(image_rgb)
            image_pil.save(output_file_path_decrease)