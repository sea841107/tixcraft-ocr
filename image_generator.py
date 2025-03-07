import itertools
import string
import os
import random
from PIL import Image, ImageDraw, ImageFont

# Constant
FONT_PATH = "font/SpicyRice-Regular.ttf"
IMAGE_SIZE = (120, 100)
CHAR_COUNT = 4
CHAR_HEIGHT_RANGE = (50, 60)
ROTATE_ANGLE = (-5, 5)
CHAR_SPACING = (-5, 0)
ROTATE_PADDING = 10 # for more space when rotating
SAVE_DIR = "captchas"

# all combinations with "abcd" ~ "zyxw" (26 * 25 * 24* 23)
all_combinations = [''.join(chars) for chars in itertools.permutations(string.ascii_lowercase, CHAR_COUNT)]

def generate_captcha(text):
    img = Image.new("1", IMAGE_SIZE)  # only black & white
    ImageDraw.Draw(img)

    # initial x
    x_offset = -ROTATE_PADDING // 2  

    last_char_spacing = 0

    # generate every char
    for char in text:
        char_img = generate_character_image(char, FONT_PATH)

        # ensure the images's width
        if x_offset + char_img.width > IMAGE_SIZE[0]:
            return

        # fix y position
        y_offset = (IMAGE_SIZE[1] - char_img.height) // 2

        img.paste(char_img, (x_offset, y_offset), char_img)

        # fix x position
        x_offset += char_img.width - ROTATE_PADDING
        last_char_spacing = random.randint(*CHAR_SPACING)
        x_offset += last_char_spacing

    # ensure directory
    captcha_dir = os.path.join(SAVE_DIR, text)
    os.makedirs(captcha_dir, exist_ok=True)

    # find the max index of existing files
    existing_files = [f for f in os.listdir(captcha_dir) if f.endswith(".png")]
    next_index = len(existing_files)

    # save
    save_path = os.path.join(captcha_dir, f"{next_index}.png")
    img.save(save_path)

    print(f"Generated CAPTCHA: {text} -> {save_path}")

def generate_character_image(char, font_path):
    char_height = random.randint(*CHAR_HEIGHT_RANGE)
    char_font = ImageFont.truetype(font_path, char_height)

    char_w, char_h = char_font.getmask(char).size

    # draw char image
    char_img = Image.new("1", (char_w + ROTATE_PADDING, IMAGE_SIZE[1]))
    char_draw = ImageDraw.Draw(char_img)

    # put the char to center
    char_draw.text((ROTATE_PADDING, char_h // 3), char, font=char_font, fill="white")

    # rotate
    char_img = char_img.rotate(random.randint(*ROTATE_ANGLE), resample=Image.BICUBIC)

    return char_img

for text in all_combinations:
    generate_captcha(text)