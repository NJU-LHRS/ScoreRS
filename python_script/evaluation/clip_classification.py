import argparse
import logging
from functools import partial
from pathlib import Path

import torch
from PIL import Image
from src.dataset.prompt_class import CLIP_TEMPLATE
from src.evaluation.clip_cls import accuracy, zero_shot_classifier
from src.tools.logger import setup_logger
from torch.nn import functional as F
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizerFast

Image.MAX_IMAGE_PIXELS = None

logger = logging.getLogger(__name__)


def hack_hf_image_transform(image_processor: CLIPImageProcessor, image: Image.Image):
    image = image_processor(image, return_tensors="pt")["pixel_values"].squeeze()
    return image


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, default="openai/clip-vit-large-patch14"
    )
    parser.add_argument("--image_root", type=str, default="data/images")
    parser.add_argument("--output_path", type=str, default="/output")
    return parser.parse_args()


def main(args):
    device = torch.device("cuda")
    model = CLIPModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
    ).to(device)
    image_process = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-large-patch14")

    logger.info(f"Loading image folder from {args.image_root}")
    eval_loader = {}
    for val_data_path in Path(args.image_root).glob("*_Image"):
        dataset = ImageFolder(
            val_data_path,
            transform=partial(hack_hf_image_transform, image_process),
        )
        classes_name = dataset.classes

        data_name = val_data_path.name.split("_")[0]
        eval_loader[data_name] = {
            "dataloader": torch.utils.data.DataLoader(
                dataset,
                batch_size=128,
                shuffle=False,
                sampler=None,
                pin_memory=False,
                num_workers=8,
            ),
            "classes_name": classes_name,
        }

    model.eval()
    logger.info(f"Evaluating {len(eval_loader)} datasets")
    for data_name, data in eval_loader.items():
        logger.info(f"Evaluating {data_name}")
        templates = CLIP_TEMPLATE[data_name.lower()]
        classifier = zero_shot_classifier(
            model,
            data["classes_name"],
            templates,
            tokenizer,
            device,
        )

        top1, top5, n = 0, 0, 0
        for image, label in tqdm(data["dataloader"], desc="Evaluating"):
            image = image.to(device)
            label = label.to(device)

            with torch.autocast(
                device_type=device.type,
                enabled=False,
                dtype=torch.float16,
            ):
                image_features = model.get_image_features(pixel_values=image)
                image_features = F.normalize(image_features, dim=-1)
                logits = 100 * image_features @ classifier

            acc1, acc5 = accuracy(logits, label, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += image.size(0)

        top1 = top1 / n
        top5 = top5 / n

        logger.info(
            f"Dataset {data_name}: Top-1 accuracy: {top1:.4f}, Top-5 accuracy: {top5:.4f}"
        )


if __name__ == "__main__":
    args = arg_parser()
    args.output_path = Path(args.output_path)
    if not args.output_path.exists():
        args.output_path.mkdir(parents=True, exist_ok=True)

    setup_logger(__name__, str(args.output_path), rank=0)
    logger.info(f"Starting clip classification with model {args.model_path}")
    main(args)
