import argparse
from marker.convert import convert_single_pdf
from marker.models import load_all_models
import torch
import os
import json
from tqdm import tqdm

def gen_markdown(dir_path, model_lst):
    for file_name in tqdm(os.listdir(dir_path), desc=dir_path):
        file_name = file_name.replace(".pdf", "")
        save_path = f"ocr/{file_name}"

        print(f"Markdown save path: {save_path}")
        
        if not os.path.exists(save_path):
            fpath = f"{dir_path}/{file_name}.pdf"
            full_text, images, out_meta = convert_single_pdf(fpath, model_lst, batch_multiplier=5)
            
            os.mkdir(save_path)
            with open(f"{save_path}/markdown.md", 'w', encoding='utf-8') as file:
                file.write(full_text)
            with open(f"{save_path}/meta_data.json", 'w', encoding='utf-8') as file:
                json.dump(out_meta, file)
            for _, key in enumerate(images):
                images[key].save(f"{save_path}/{key}", format="PNG")
        else:
            continue
    
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--dir_path", type=str, default='')

    args = parser.parse_args()

    model_lst = load_all_models(device=args.device, dtype=torch.float16)

    gen_markdown(args.dir_path, model_lst)