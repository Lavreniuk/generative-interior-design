import os
import json

def clean_caption(caption):
    """Clean caption by removing trailing punctuation and excess characters."""
    caption = caption.strip().strip('"')
    if caption.endswith(","):
        caption = caption[:-1]
    last_period_index = caption.rfind('.')
    last_comma_index = caption.rfind(',')
    last_index = max(last_period_index, last_comma_index)
    if last_index != -1:
        caption = caption[:last_index]
    return caption

def process_caption_line(line, images_folder, conditioning_images_folder, parent_folder):
    """Process a line from the captions file."""
    if "{" in line or "}" in line:
        return None
    filename, caption = line.split(':', 1)
    filename, caption = filename.strip().strip('"'), clean_caption(caption)
    image_path = os.path.join(images_folder, filename)
    conditioning_image_path = os.path.join(conditioning_images_folder, filename.replace(".jpg", "segmentation.png"))
    if os.path.exists(image_path) and os.path.exists(conditioning_image_path):
        image_path = image_path.replace(parent_folder, '').lstrip('/')
        conditioning_image_path = conditioning_image_path.replace(parent_folder, '').lstrip('/')
        return {"text": caption, "image": image_path, "conditioning_image": conditioning_image_path}
    return None

def process_captions_file(captions_file, images_folder, conditioning_images_folder, output_file):
    """Process the captions file and write JSON objects to the output file."""
    parent_folder = os.path.dirname(images_folder)
    with open(output_file, 'w') as writer, open(captions_file, 'r') as f:
        for line in f:
            json_object = process_caption_line(line, images_folder, conditioning_images_folder, parent_folder)
            if json_object:
                writer.write(json.dumps(json_object) + '\n')

def main():
    # Define paths
    images_folder = "/mnt/data1/nick/depth/fill50k/images"
    conditioning_images_folder = "/mnt/data1/nick/depth/fill50k/conditioning_images"
    captions_file = "/mnt/data1/nick/depth/fill50k/bnb-dataset/get_captions/captions.json"
    output_file = "/mnt/data1/nick/depth/fill50k/train.jsonl"
    # Process captions file
    process_captions_file(captions_file, images_folder, conditioning_images_folder, output_file)

if __name__ == "__main__":
    main()