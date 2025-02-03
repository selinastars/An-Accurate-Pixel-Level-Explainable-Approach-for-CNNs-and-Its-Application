from PIL import Image
import numpy as np
import os
import math

# Read the matrix from the document
def read_matrix(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        matrix = [list(map(float, line.split())) for line in lines]
    return np.array(matrix)

# Sort in ascending order
def sort_func(file_name):
    return int(''.join(filter(str.isdigit, file_name)))


# Import the absolute path of the folder storing the actual images
actual_image_folder_absolute_path = r"..."

# Set the absolute path of the folder to save the generated manipulated images
tamper_image_folder_absolute_path = r"..."

# Read the pixel weight matrix of the RGB three-channel of the actual image
pixel_weight_matrix_R_path = r"..."
pixel_weight_matrix_G_path = r"..."
pixel_weight_matrix_B_path = r"..."

file_list_images = os.listdir(actual_image_folder_absolute_path)
image_files = sorted(file_list_images, key=sort_func)

ratio = 0.01

for image_file in image_files:
    if image_file.endswith('.png'):
        image_path = os.path.join(actual_image_folder_absolute_path, image_file)
        image = Image.open(image_path)
        image = image.resize((224, 224))
        actual_image_matrix = np.array(image)
        actual_image_R_channel = actual_image_matrix[:, :, 0]
        actual_image_G_channel = actual_image_matrix[:, :, 1]
        actual_image_B_channel = actual_image_matrix[:, :, 2]

        image_number = ''.join(filter(str.isdigit, image_file))
        matrix_file_path_R = os.path.join(pixel_weight_matrix_R_path, f"{image_number}.txt")
        matrix_file_path_G = os.path.join(pixel_weight_matrix_G_path, f"{image_number}.txt")
        matrix_file_path_B = os.path.join(pixel_weight_matrix_B_path, f"{image_number}.txt")
        pixel_weight_matrix_R = read_matrix(matrix_file_path_R)
        pixel_weight_matrix_G = read_matrix(matrix_file_path_G)
        pixel_weight_matrix_B = read_matrix(matrix_file_path_B)

        tamper_matrix_R = actual_image_R_channel.copy()
        tamper_matrix_G = actual_image_G_channel.copy()
        tamper_matrix_B = actual_image_B_channel.copy()

        preprocessing_actual_image_channel_R = ((actual_image_R_channel / 255) - 0.485) / 0.229
        preprocessing_actual_image_channel_G = ((actual_image_G_channel / 255) - 0.456) / 0.224
        preprocessing_actual_image_channel_B = ((actual_image_B_channel / 255) - 0.406) / 0.225

        preprocessing_actual_image_channel_R_increase = preprocessing_actual_image_channel_R.copy()
        preprocessing_actual_image_channel_G_increase = preprocessing_actual_image_channel_G.copy()
        preprocessing_actual_image_channel_B_increase = preprocessing_actual_image_channel_B.copy()

        addend = 0.01
        for i in range(224):
            for j in range(224):
                addend = abs(preprocessing_actual_image_channel_R[i][j] * ratio)
                preprocessing_actual_image_channel_R_increase[i][j] = preprocessing_actual_image_channel_R_increase[i][j] + addend

                addend = abs(preprocessing_actual_image_channel_G[i][j] * ratio)
                preprocessing_actual_image_channel_G_increase[i][j] = preprocessing_actual_image_channel_G_increase[i][j] + addend

                addend = abs(preprocessing_actual_image_channel_B[i][j] * ratio)
                preprocessing_actual_image_channel_B_increase[i][j] = preprocessing_actual_image_channel_B_increase[i][j] + addend


        preprocessing_actual_image_channel_R_decrease = preprocessing_actual_image_channel_R.copy()
        preprocessing_actual_image_channel_G_decrease = preprocessing_actual_image_channel_G.copy()
        preprocessing_actual_image_channel_B_decrease = preprocessing_actual_image_channel_B.copy()

        addend = 0
        for i in range(224):
            for j in range(224):
                addend = abs(preprocessing_actual_image_channel_R[i][j] * ratio)
                preprocessing_actual_image_channel_R_decrease[i][j] = preprocessing_actual_image_channel_R_decrease[i][j] - addend

                addend = abs(preprocessing_actual_image_channel_G[i][j] * ratio)
                preprocessing_actual_image_channel_G_decrease[i][j] = preprocessing_actual_image_channel_G_decrease[i][j] - addend

                addend = abs(preprocessing_actual_image_channel_B[i][j] * ratio)
                preprocessing_actual_image_channel_B_decrease[i][j] = preprocessing_actual_image_channel_B_decrease[i][j] - addend

        for i in range(224):
            for j in range(224):

                if preprocessing_actual_image_channel_R[i][j] >= 0 and pixel_weight_matrix_R[i][j] >= 0:
                    if (preprocessing_actual_image_channel_R_decrease[i][j] * 0.229 + 0.485) < 0:
                        tamper_matrix_R[i][j] = 0
                    elif (preprocessing_actual_image_channel_R_decrease[i][j] * 0.229 + 0.485) >= 0:
                        tamper_matrix_R[i][j] = int((preprocessing_actual_image_channel_R_decrease[i][j] * 0.229 + 0.485) * 255)

                if preprocessing_actual_image_channel_G[i][j] >= 0 and pixel_weight_matrix_G[i][j] >= 0:
                    if (preprocessing_actual_image_channel_G_decrease[i][j] * 0.224 + 0.456) < 0:
                        tamper_matrix_G[i][j] = 0
                    elif (preprocessing_actual_image_channel_G_decrease[i][j] * 0.224 + 0.456) >= 0:
                        tamper_matrix_G[i][j] = int((preprocessing_actual_image_channel_G_decrease[i][j] * 0.224 + 0.456) * 255)

                if preprocessing_actual_image_channel_B[i][j] >= 0 and pixel_weight_matrix_B[i][j] >= 0:
                    if (preprocessing_actual_image_channel_B_decrease[i][j] * 0.225 + 0.406) < 0:
                        tamper_matrix_B[i][j] = 0
                    elif (preprocessing_actual_image_channel_B_decrease[i][j] * 0.225 + 0.406) >= 0:
                        tamper_matrix_B[i][j] = int((preprocessing_actual_image_channel_B_decrease[i][j] * 0.225 + 0.406) * 255)


                if preprocessing_actual_image_channel_R[i][j] <= 0 and pixel_weight_matrix_R[i][j] <= 0:
                    if (preprocessing_actual_image_channel_R_increase[i][j] * 0.229 + 0.485) > 1:
                        tamper_matrix_R[i][j] = 255
                    elif (preprocessing_actual_image_channel_R_increase[i][j] * 0.229 + 0.485) <= 1:
                        tamper_matrix_R[i][j] = math.ceil((preprocessing_actual_image_channel_R_increase[i][j] * 0.229 + 0.485) * 255)

                if preprocessing_actual_image_channel_G[i][j] <= 0 and pixel_weight_matrix_G[i][j] <= 0:
                    if (preprocessing_actual_image_channel_G_increase[i][j] * 0.224 + 0.456) > 1:
                        tamper_matrix_G[i][j] = 255
                    elif (preprocessing_actual_image_channel_G_increase[i][j] * 0.224 + 0.456) <= 1:
                        tamper_matrix_G[i][j] = math.ceil((preprocessing_actual_image_channel_G_increase[i][j] * 0.224 + 0.456) * 255)

                if preprocessing_actual_image_channel_B[i][j] <= 0 and pixel_weight_matrix_B[i][j] <= 0:
                    if (preprocessing_actual_image_channel_B_increase[i][j] * 0.225 + 0.406) > 1:
                        tamper_matrix_B[i][j] = 255
                    elif (preprocessing_actual_image_channel_B_increase[i][j] * 0.225 + 0.406) <= 1:
                        tamper_matrix_B[i][j] = math.ceil((preprocessing_actual_image_channel_B_increase[i][j] * 0.225 + 0.406) * 255)


                if preprocessing_actual_image_channel_R[i][j] <= 0 and pixel_weight_matrix_R[i][j] >= 0:
                    if (preprocessing_actual_image_channel_R_decrease[i][j] * 0.229 + 0.485) < 0:
                        tamper_matrix_R[i][j] = 0
                    elif (preprocessing_actual_image_channel_R_decrease[i][j] * 0.229 + 0.485) >= 0:
                        tamper_matrix_R[i][j] = int((preprocessing_actual_image_channel_R_decrease[i][j] * 0.229 + 0.485) * 255)

                if preprocessing_actual_image_channel_G[i][j] <= 0 and pixel_weight_matrix_G[i][j] >= 0:
                    if (preprocessing_actual_image_channel_G_decrease[i][j] * 0.224 + 0.456) < 0:
                        tamper_matrix_G[i][j] = 0
                    elif (preprocessing_actual_image_channel_G_decrease[i][j] * 0.224 + 0.456) >= 0:
                        tamper_matrix_G[i][j] = int((preprocessing_actual_image_channel_G_decrease[i][j] * 0.224 + 0.456) * 255)

                if preprocessing_actual_image_channel_B[i][j] <= 0 and pixel_weight_matrix_B[i][j] >= 0:
                    if (preprocessing_actual_image_channel_B_decrease[i][j] * 0.225 + 0.406) < 0:
                        tamper_matrix_B[i][j] = 0
                    elif (preprocessing_actual_image_channel_B_decrease[i][j] * 0.225 + 0.406) >= 0:
                        tamper_matrix_B[i][j] = int((preprocessing_actual_image_channel_B_decrease[i][j] * 0.225 + 0.406) * 255)


                if preprocessing_actual_image_channel_R[i][j] >= 0 and pixel_weight_matrix_R[i][j] <= 0:
                    if (preprocessing_actual_image_channel_R_increase[i][j] * 0.229 + 0.485) > 1:
                        tamper_matrix_R[i][j] = 255
                    elif (preprocessing_actual_image_channel_R_increase[i][j] * 0.229 + 0.485) <= 1:
                        tamper_matrix_R[i][j] = math.ceil((preprocessing_actual_image_channel_R_increase[i][j] * 0.229 + 0.485) * 255)

                if preprocessing_actual_image_channel_G[i][j] >= 0 and pixel_weight_matrix_G[i][j] <= 0:
                    if (preprocessing_actual_image_channel_G_increase[i][j] * 0.224 + 0.456) > 1:
                        tamper_matrix_G[i][j] = 255
                    elif (preprocessing_actual_image_channel_G_increase[i][j] * 0.224 + 0.456) <= 1:
                        tamper_matrix_G[i][j] = math.ceil((preprocessing_actual_image_channel_G_increase[i][j] * 0.224 + 0.456) * 255)

                if preprocessing_actual_image_channel_B[i][j] >= 0 and pixel_weight_matrix_B[i][j] <= 0:
                    if (preprocessing_actual_image_channel_B_increase[i][j] * 0.225 + 0.406) > 1:
                        tamper_matrix_B[i][j] = 255
                    elif (preprocessing_actual_image_channel_B_increase[i][j] * 0.225 + 0.406) <= 1:
                        tamper_matrix_B[i][j] = math.ceil((preprocessing_actual_image_channel_B_increase[i][j] * 0.225 + 0.406) * 255)


        image_rgb = np.stack([tamper_matrix_R, tamper_matrix_G, tamper_matrix_B], axis=-1)
        image_rgb = image_rgb.astype(np.uint8)
        image_pil = Image.fromarray(image_rgb)
        os.makedirs(tamper_image_folder_absolute_path, exist_ok=True)
        image_pil.save(os.path.join(tamper_image_folder_absolute_path, image_file))
