import os
import numpy as np
import string
import captcha_image

# Constant
CAPTCHA_DIR = "captchas"
SAVE_DIR = "dataset"
FILE_EXTENSION = ".png"
IMAGE_SET_NAME = "image_set.npy"
LABEL_SET_NAME = "label_set.npy"
IMAGE_SIZE = (120, 100)
CHAR_COUNT = 4
CHARSET = string.ascii_lowercase # from 'a' to 'z'
NUM_CLASSES = len(CHARSET) # 26 classes

os.makedirs(SAVE_DIR, exist_ok=True)

def preprocess_image(image_path):
    """process images to numpy array"""
    image_array = captcha_image.normalize_image_to_np_array(image_path, IMAGE_SIZE)
    image_array = image_array.reshape(1, IMAGE_SIZE[1], IMAGE_SIZE[0])  # (channels, height, width)
    return image_array

def encode_label(text):
    """encode label to one-hot vector (4, 26)"""
    label_matrix = np.zeros((CHAR_COUNT, NUM_CLASSES))

    for i, char in enumerate(text):
        char_index = CHARSET.index(char)
        label_matrix[i, char_index] = 1

    return label_matrix

def load_all_data():
    """load images and labels"""
    image_data_list = []
    label_data_list = []
    for folder in os.listdir(CAPTCHA_DIR):
        folder_path = os.path.join(CAPTCHA_DIR, folder)
        
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                if file_name.endswith(FILE_EXTENSION):
                    image_path = os.path.join(folder_path, file_name)
                    image_data = preprocess_image(image_path)
                    label_data = encode_label(folder)  # the name of folder is the answer
                    image_data_list.append(image_data)
                    label_data_list.append(label_data)

    return image_data_list, label_data_list

image_data_list, label_data_list = load_all_data()

image_set = np.array(image_data_list)
label_set = np.array(label_data_list)

np.save(os.path.join(SAVE_DIR, IMAGE_SET_NAME), image_set)
np.save(os.path.join(SAVE_DIR, LABEL_SET_NAME), label_set)

print(f"{len(image_set)} data is saved")
