import pandas as pd
from tqdm import tqdm
from os.path import join, exists
import logging
import json
import torch
import argparse
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel, CLIPImageProcessor, CLIPTokenizer

import sys
import os

sys.path.append(os.getcwd())
from src.dataset.dataset_class import TextDataset, ImageDatasetFromFile
from pathlib import Path
from src.tools.logger import setup_logger


logger = logging.getLogger(__name__)


def arg_parser():
    parser = argparse.ArgumentParser(description="retrieval evaluation")
    parser.add_argument(
        "--model_path", type=str, default="openai/clip-vit-large-patch14"
    )
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument(
        "--target_root",
        type=str,
        default="./ucmcaptions_img_txt_pairs_test.csv",
    )
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--recall_k", type=list, default=[1, 5, 10])
    parser.add_argument("--target_column", type=str, default="title")
    parser.add_argument("--image_column", type=str, default="filepath")

    return parser


def eval(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(
        args.model_path,
        device_map=device,
        torch_dtype=torch.float16,
    )
    transform = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    df = pd.read_csv(args.target_root)

    # unique images
    dataset_image = ImageDatasetFromFile(
        data_root=args.data_root,
        image_file=args.target_root,
        file_type="csv",
        transform=transform,
        image_column=args.image_column,
    )

    dataloader = DataLoader(
        dataset_image, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    all_image_features = []
    all_image_paths = []
    with torch.no_grad():
        for batch in tqdm(dataloader, unit_scale=args.batch_size):
            images, img_paths = batch
            images = images.to(device=device)
            image_features = model.get_image_features(images)
            all_image_features.append(image_features.cpu())
            all_image_paths.extend(img_paths)
    all_image_features = torch.cat(all_image_features)
    # unique texts
    dataset_text = TextDataset(
        target_root=args.target_root,
        target_type="csv",
        tokenizer=tokenizer,
        target_column=args.target_column,
    )

    dataloader = DataLoader(
        dataset_text, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    all_text_features = []
    all_texts = []
    with torch.no_grad():
        for batch in tqdm(dataloader, unit_scale=args.batch_size):
            original_texts = batch["text"]

            texts = tokenizer(
                [text for text in original_texts],
                max_length=tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(device)
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            input_ids = input_ids.to(device=device)
            attention_mask = attention_mask.to(device=device)

            text_features = model.get_text_features(
                input_ids=input_ids, attention_mask=attention_mask
            )
            all_text_features.append(text_features.cpu())
            all_texts.extend(original_texts)
    all_text_features = torch.cat(all_text_features)

    text_indices = {x: i for i, x in enumerate(all_texts)}
    img_indices = {x: i for i, x in enumerate(all_image_paths)}

    df[args.image_column] = df[args.image_column].apply(
        lambda x: join(args.data_root, x)
    )
    # ground truth
    img_path2text = {}
    text2img_path = {}
    for i in tqdm(df.index):
        text = df.loc[i, args.target_column]
        img_path = df.loc[i, args.image_column]
        text_id = text_indices[text]
        img_id = img_indices[img_path]
        if img_path not in img_path2text:
            img_path2text[img_path] = set()
        img_path2text[img_path].add(text_id)
        if text not in text2img_path:
            text2img_path[text] = set()
        text2img_path[text].add(img_id)

        res = {"text2img_R@" + str(k): 0 for k in args.recall_k}
        res.update({"img2text_R@" + str(k): 0 for k in args.recall_k})

        # text to image

        logit_scale = 100
    for i in tqdm(range(len(all_texts))):
        text_feature = all_text_features[i]
        logits = logit_scale * text_feature @ all_image_features.t()
        ranking = torch.argsort(logits, descending=True).cpu().numpy()

        for k in args.recall_k:
            intersec = set(ranking[:k]) & set(text2img_path[all_texts[i]])
            if intersec:
                res["text2img_R@" + str(k)] += 1
    for k in args.recall_k:
        res["text2img_R@" + str(k)] /= len(all_texts)

    # image to text
    logit_scale = 100
    for i in tqdm(range(len(all_image_paths))):
        image_feature = all_image_features[i]
        logits = logit_scale * image_feature @ all_text_features.t()
        ranking = torch.argsort(logits, descending=True).cpu().numpy()
        for k in args.recall_k:
            intersec = set(ranking[:k]) & img_path2text[all_image_paths[i]]
            if intersec:
                res["img2text_R@" + str(k)] += 1
    for k in args.recall_k:
        res["img2text_R@" + str(k)] /= len(all_image_paths)

    return res


if __name__ == "__main__":
    now = datetime.now()
    parser = arg_parser()
    args = parser.parse_args()

    setup_logger(__name__, args.output_dir, rank=0)
    logger.info(f"Start processing")

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    image_root = Path(args.target_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    res = eval(args)

    logger.info("model_path:" + args.model_path)
    logger.info("data_root:" + args.data_root)
    logger.info("target_root:" + args.target_root)
    logger.info(f"Finish processing {args.model_path}")
    dict_str = json.dumps(res, indent=4, ensure_ascii=False)
    logger.info(dict_str)

    current_time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    output_path = output_dir / (
        args.model_path.replace("/", "_") + "_" + current_time_str + ".txt"
    )
    with open(output_path, "w") as f:
        f.write("model_path:" + args.model_path + "\n")
        f.write("data_root:" + args.data_root + "\n")
        f.write("target_root:" + args.target_root + "\n")
        json.dump(res, f, indent=4)
