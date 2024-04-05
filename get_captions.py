from PIL import Image
import os
import json
import time
from transformers import BitsAndBytesConfig, pipeline
import torch


# Define constants
IMAGES_FOLDER = '/mnt/data1/nick/depth/fill50k/images'
IMAGES_FOLDER_2 = '/mnt/data1/nick/depth/fill50k/conditioning_images'
OUTPUT_JSON_FILE = 'captions.json'
BATCH_SIZE = 1000

# Define prompt and model for image captioning
prompt = "USER: <image>\nDescribe the interior image. Be detailed, describe a style, a color, and furniture fabric. Use only one but detailed sentence. It must begin with room type description, then always describe the general style and after that describe the all furniture items and their arrangement in the room and color. \n ASSISTANT:"
quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
pipe = pipeline("image-to-text", model="llava-hf/llava-1.5-7b-hf", model_kwargs={"quantization_config": quantization_config})

def generate_caption(image_path):
    """Generate caption for the given image."""
    image = Image.open(image_path)
    outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 100, "temperature":0.4, "do_sample":True})
    text = outputs[0]["generated_text"]
    return text[text.find('ASSISTANT:')+len('ASSISTANT:')+1:]

def filter_image_files(images_folder, other_folder):
    """Filter image files based on existence of corresponding segmentation files."""
    image_files = sorted(os.listdir(images_folder))
    other_files = set(os.listdir(other_folder))
    filtered_files = [filename for filename in image_files if f"{os.path.splitext(filename)[0]}segmentation.png" in other_files]
    return filtered_files

def process_images(images_folder, images_folder2, output_json_file, batch_size):
    """Process images and generate captions."""
    image_files = sorted(filter_image_files(images_folder, images_folder2))
    captions_dict = {}
    start_time = time.time()
    for i, image_file in enumerate(image_files, start=1):
        image_path = os.path.join(images_folder, image_file)
        caption = generate_caption(image_path)
        captions_dict[image_file] = caption
        if i % batch_size == 0 or i == len(image_files):
            with open(output_json_file, 'a') as json_file:
                json.dump(captions_dict, json_file, indent=4)
            print(f"Processed {i}/{len(image_files)} images. Results dumped to {output_json_file}. Elapsed time: {time.time() - start_time:.2f} seconds")
            captions_dict = {}
            start_time = time.time()
    print("All images processed and results dumped to", output_json_file)

if __name__ == "__main__":
    process_images(IMAGES_FOLDER, IMAGES_FOLDER_2, OUTPUT_JSON_FILE, BATCH_SIZE)