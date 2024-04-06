# StableDesign for generative interior design
<a href='https://huggingface.co/spaces/MykolaL/StableDesign'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a>  
[Mykola Lavreniuk](https://www.linkedin.com/in/mykola-lavreniuk/), [Bartosz Ludwiczuk](https://www.linkedin.com/in/bartosz-ludwiczuk-a677a760)

## Preparing the dataset
run *get_captions.py* to retrieve captions for the images using <b>llava-1.5</b>  
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
<b>ControlNet</b>
```
rm -rf ~/.cache/huggingface/datasets/
accelerate config
CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision="bf16" train_controlnet.py  --checkpointing_steps=20000 --validation_steps=10000 --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5"  --output_dir=model --resolution=512  --learning_rate=1e-5  --validation_image "./image_0.jpg" "./image_1.jpg"  --validation_prompt "A Bauhaus-inspired living room with a sleek black leather sofa, a tubular steel coffee table exemplifying modernist design, and a geometric patterned rug adding a touch of artistic flair." "A glamorous master bedroom in Hollywood Regency style, boasting a plush tufted headboard, mirrored furniture reflecting elegance, luxurious fabrics in rich textures, and opulent gold accents for a touch of luxury."  --train_batch_size=4 --dataset_name=fill50k.py --controlnet_model_name_or_path "BertChristiaens/controlnet-seg-room" --report_to wandb --gradient_accumulation_steps=1 --mixed_precision="bf16" --num_train_epochs=10
```

<b>Lora</b>
```
CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision="bf16" train_text_to_image_lora.py  --checkpointing_steps=20000 --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5"  --output_dir=model_lora --resolution=512  --learning_rate=1e-4 --validation_prompt "A Bauhaus-inspired living room with a sleek black leather sofa, a tubular steel coffee table exemplifying modernist design, and a geometric patterned rug adding a touch of artistic flair."  --train_batch_size=4 --dataset_name=fill50k.py --random_flip --gradient_accumulation_steps=1 --mixed_precision="bf16" --num_train_epochs=10 --rank=64 --report_to wandb
```
