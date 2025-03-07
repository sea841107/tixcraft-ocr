import torch
import string
import captcha_image
from cnn.normal import CaptchaCNN as CNN

# Constant
MODEL_PATH = "ocr_model.pth"
TEST_IMAGE_PATH = "test.png"
IMAGE_SIZE = (120, 100)
CHARS = string.ascii_lowercase # from 'a' to 'z'
    
# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()

def predict_captcha(image_path):
    # import
    image_array = captcha_image.normalize_image_to_np_array(image_path, IMAGE_SIZE)
    image_array = image_array.reshape(1, 1, IMAGE_SIZE[1], IMAGE_SIZE[0])  # (batch_size, channels, height, width)

    # predict
    image = torch.tensor(image_array, dtype=torch.float32).to(device)
    with torch.no_grad():
        output = model(image)

    # decode
    predicted_indices = output.argmax(dim=2).cpu().numpy().flatten()  # find the target characters (1, 4, 26)
    predicted_text = ''.join(CHARS[i] for i in predicted_indices)

    return predicted_text

# test
predicted_text = predict_captcha(TEST_IMAGE_PATH)
print(f"Result: {predicted_text}")