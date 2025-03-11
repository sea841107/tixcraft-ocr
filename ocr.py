import os
import numpy as np
import string
import captcha_image
import onnxruntime as ort

# Constant
MODEL_PATH = "ocr_v1.0.onnx"
TEST_IMAGE_PATH = "test.png"
TEST_IMAGE_FOLDER = "captchas/test"
IMAGE_SIZE = (120, 100)
CHARS = string.ascii_lowercase # from 'a' to 'z'

def predict_captcha(image_path):
    # import
    image_array = captcha_image.normalize_image_to_np_array(image_path, IMAGE_SIZE)
    image_array = image_array.reshape(1, 1, IMAGE_SIZE[1], IMAGE_SIZE[0])  # (batch_size, channels, height, width)
    image_array = image_array.astype(np.float32)

    # predict
    ort_session = ort.InferenceSession(MODEL_PATH)
    output = ort_session.run(None, {"input.1": image_array})

    # decode
    predicted_indices = np.argmax(output[0], axis=2).flatten()
    predicted_text = ''.join(CHARS[i] for i in predicted_indices)

    return predicted_text

total = 0
correct = 0

# test
for folder in os.listdir(TEST_IMAGE_FOLDER):
    folder_path = os.path.join(TEST_IMAGE_FOLDER, folder)
    
    if os.path.isdir(folder_path):
        for file_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, file_name)
            predicted_text = predict_captcha(image_path)
            
            total += 1

            # folder's name is the answer
            if predicted_text == folder:
                correct += 1
            else:
                print(f"Wrong | Answer: {folder} / Predict: {predicted_text} / File: {file_name}")
                
print(f"Total: {total} / Correct: {correct} / Acc: {correct / total:.4f}")

# predicted_text = predict_captcha(TEST_IMAGE_PATH)
# print(f"Result: {predicted_text}")