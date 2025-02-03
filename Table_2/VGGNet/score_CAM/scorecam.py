import argparse
import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from pytorch_grad_cam import ScoreCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import vgg19


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
    model = vgg19().to(device)

    # Load model weights
    model_weight_absolute_path = torch.load(
        "...",
        map_location=device)
    model.load_state_dict(model_weight_absolute_path, strict=False)

    target_layers = [model.features[-1]]

    # Import actual image
    actual_image_folder_absolute_path = r"..."

    # Specify the saliency map save address
    output_folder_absolute_path_saliency_map = r"..."

    if not os.path.exists(output_folder_absolute_path_saliency_map):
        os.makedirs(output_folder_absolute_path_saliency_map)

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image_files = sorted([filename for filename in os.listdir(actual_image_folder_absolute_path) if filename.endswith(".png")],
                         key=lambda x: int(os.path.splitext(x)[0]))

    for filename in image_files:
        img_path = os.path.join(actual_image_folder_absolute_path, filename)
        image = Image.open(img_path).convert('RGB')

        rgb_img = np.array(image)
        rgb_img = rgb_img.astype(np.float32) / 255.0

        input_tensor = preprocess(rgb_img).to(device)
        input_tensor = input_tensor.unsqueeze(0)

        cam_algorithm = ScoreCAM
        with cam_algorithm(model=model,
                           target_layers=target_layers) as cam:
            cam.batch_size = 4
            grayscale_cam = cam(input_tensor=input_tensor,
                                aug_smooth=args.aug_smooth,
                                eigen_smooth=args.eigen_smooth)

            grayscale_cam = grayscale_cam[0, :]

            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

        cam_output_path = os.path.join(output_folder_absolute_path_saliency_map, filename)
        cv2.imwrite(cam_output_path, cam_image)
