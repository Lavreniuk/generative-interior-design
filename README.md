# StableDesign for generative interior design
<a href='https://huggingface.co/spaces/MykolaL/StableDesign'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a>  
[Mykola Lavreniuk](https://www.linkedin.com/in/mykola-lavreniuk/), [Bartosz Ludwiczuk](https://www.linkedin.com/in/bartosz-ludwiczuk-a677a760)

## Dataset creation

These script are extended version of code from [bnb-dataset](https://github.com/airbert-vln/bnb-dataset/tree/main)

1.  Download listing from airbnb
```bash
python search_listings_with_price.py --location data/cites_world.txt 
```
2. Download image-metadata from each listing
```bash
python download_listings.py --listings data/listings --output data/merlin --with_photo --num_splits 1 --start 0
```
3. Create a single TSV file
```bash
python extract_photo_metadata.py --merlin data/merlin/ --output data/bnb-dataset-raw-2ndpart.tsv
```
4. Download images
Currently, in the same script images are downsampled to have size 768x. Note: For final run, we downloaded ~300k images.
```bash
python download_images.py --csv_file data/bnb-dataset-raw-2ndpart.tsv --output /media/blcv/drive_2TB/genai/bnb/data/images_price --correspondance /tmp/cache-download-images/ --num_parts 1 --num_splits 4 --num_procs 4 --start 0
 ```
5. Extract features for interior/outdoor detection
```bash
python detect_room.py --output data/places365/detect-25koffers.tsv --images /media/blcv/drive_2TB/genai/bnb/data/images_price
```
6. Extract TSV with all indoor images
```bash
python extract_indoor.py --output data/bnb-dataset-indoor-25koffers.tsv --detection data/places365/25offers
 ```
7. Create a directory only containing indoor images


### Creating empty rooms
run `get_empty_room.py` to remove furniture from collected images (set correct path inside the script).
For each inout image 3 images would be saved:
- cleaned RGB image
- segmentation mask of clean image
- depth estimation for clean image


## Preparing the dataset
run *get_captions.py* to retrieve captions for the images using <b>llava-1.5</b>. Use original images, with furniture.  
run *prepare_train_jsonl.py* to generate train.jsonl for subsequent training

<b>Folder Structure:</b>  
fill50k/  
├── images/  
├── conditioning_images/  
├── diffusers/  
├── fill50k.py  
├── image_0.jpg  
├── image_1.jpg  
├── train.jsonl  
├── train_controlnet.py (copy from diffusers)  
└── train_text_to_image_lora.py (copy from diffusers)

## Training
<b>ControlNet</b> for segmentation (as `conditioning_images` use segmentation images)
```
rm -rf ~/.cache/huggingface/datasets/
accelerate config
CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision="bf16" train_controlnet.py  --checkpointing_steps=20000 --validation_steps=10000 --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5"  --output_dir=model --resolution=512  --learning_rate=1e-5  --validation_image "./image_0.jpg" "./image_1.jpg"  --validation_prompt "A Bauhaus-inspired living room with a sleek black leather sofa, a tubular steel coffee table exemplifying modernist design, and a geometric patterned rug adding a touch of artistic flair." "A glamorous master bedroom in Hollywood Regency style, boasting a plush tufted headboard, mirrored furniture reflecting elegance, luxurious fabrics in rich textures, and opulent gold accents for a touch of luxury."  --train_batch_size=4 --dataset_name=fill50k.py --controlnet_model_name_or_path "BertChristiaens/controlnet-seg-room" --report_to wandb --gradient_accumulation_steps=1 --mixed_precision="bf16" --num_train_epochs=10
```

<b>ControlNet</b> for depth (as `conditioning_images` use depth-estimation images)
```
rm -rf ~/.cache/huggingface/datasets/
accelerate config
CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision="bf16" train_controlnet.py  --checkpointing_steps=20000 --validation_steps=10000 --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5"  --output_dir=model --resolution=512  --learning_rate=1e-5  --validation_image "./image_0.jpg" "./image_1.jpg"  --validation_prompt "A Bauhaus-inspired living room with a sleek black leather sofa, a tubular steel coffee table exemplifying modernist design, and a geometric patterned rug adding a touch of artistic flair." "A glamorous master bedroom in Hollywood Regency style, boasting a plush tufted headboard, mirrored furniture reflecting elegance, luxurious fabrics in rich textures, and opulent gold accents for a touch of luxury."  --train_batch_size=4 --dataset_name=fill50k.py --controlnet_model_name_or_path "lllyasviel/sd-controlnet-depth " --report_to wandb --gradient_accumulation_steps=1 --mixed_precision="bf16" --num_train_epochs=10
```

<b>Lora</b>
```
CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision="bf16" train_text_to_image_lora.py  --checkpointing_steps=20000 --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5"  --output_dir=model_lora --resolution=512  --learning_rate=1e-4 --validation_prompt "A Bauhaus-inspired living room with a sleek black leather sofa, a tubular steel coffee table exemplifying modernist design, and a geometric patterned rug adding a touch of artistic flair."  --train_batch_size=4 --dataset_name=fill50k.py --random_flip --gradient_accumulation_steps=1 --mixed_precision="bf16" --num_train_epochs=10 --rank=64 --report_to wandb
```
