import imageio
from PIL import Image, ImageDraw, ImageFont
from typing import Iterable
import numpy as np


# Function to add padding at the top of the image for text
def add_padding(image, padding_size):
    new_image = Image.new('RGB', (image.width, image.height + padding_size), (0, 0, 0))  # Black background
    new_image.paste(image, (0, padding_size))
    return new_image


# Function to add text above the image
def add_text(image, text):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()  # You can choose a different font if desired
    text_width = draw.textlength(text, font)
    image_width, image_height = image.size
    text_x = (image_width - text_width) / 2
    text_y = 0  # Adjust this value to position text above the image
    draw.text((text_x, text_y), text, font=font, fill="white")
    return image


# Function to concatenate images side by side
def concatenate_images(left_image, right_image):
    dst = Image.new('RGB', (left_image.width + right_image.width, left_image.height))
    dst.paste(left_image, (0, 0))
    dst.paste(right_image, (left_image.width, 0))
    return dst


def generate_movie(ground_truth_imgs:Iterable[np.ndarray], predicted_imgs:Iterable[np.ndarray], savename:str):
    # List to store frames for the video
    frames = []

    for gt_array, pred_array in zip(ground_truth_imgs, predicted_imgs):
        # Open and resize images
        gt_image = Image.fromarray(gt_array).resize((256, 256))
        pred_image = Image.fromarray(pred_array).resize((256, 256))
        
        # Add text
        gt_image_with_text = add_text(gt_image.copy(), 'Ground Truth')
        pred_image_with_text = add_text(pred_image.copy(), 'Predicted')
        
        # Concatenate images
        concatenated_image = concatenate_images(gt_image_with_text, pred_image_with_text)
        
        # Convert PIL image to numpy array
        frames.append(np.array(concatenated_image))

    # Save frames as a video
    imageio.mimwrite(savename, frames, fps=10)  # Adjust fps as needed