import argparse
import json
from pathlib import Path
from src.tools.logger import setup_logger
from src.evaluation.lm_prediction_call import (
    LHRSLM,
    VHM,
    GeoChatLM,
    LMDeployLM,
    SkysenseGPTLM,
    VLLMLM,
)
import torch
from tqdm import tqdm
import numpy as np
from src.model.qwen_reward import Qwen2Reward
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from collections import defaultdict
from qwen_vl_utils import process_vision_info
import re
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

logger = setup_logger(__name__)


MODEL_TYPE_MAP = {
    "lhrs": LHRSLM,
    "vhm": VHM,
    "skysensegpt": SkysenseGPTLM,
    "geochat": GeoChatLM,
    "lmdeploy": LMDeployLM,
    "vllm": VLLMLM,
}

LHRS_TYPE_MAP = {
    "1": "identity",
    "2": "color",
    "3": "orientation",
    "4": "shape",
    "5": "quantity",
    "6": "area",
    "7": "distance",
    "8": "resolution",
    "9": "modality",
    "10": "location",
    "11": "reasoning",
}


def convt_qa(conversations, model):
    values = [conversation["value"] for conversation in conversations]
    query = values[0]
    answer = values[1]

    vg_prefix = getattr(model, "vg_prefix", "")
    vg_suffix = getattr(model, "vg_suffix", "")
    query = vg_prefix + " " + query + " " + vg_suffix
    return query, answer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--reward_model_path", type=str, required=True)
    parser.add_argument("--reward_model_type", type=str, default="Qwen2Reward")
    parser.add_argument("--eval_image_root", type=str, required=True)
    parser.add_argument("--eval_target_file", type=str, required=True)
    parser.add_argument("--sample_num", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--policy_model_cuda_id", type=int, default=0)
    parser.add_argument("--reward_model_cuda_id", type=int, default=1)
    parser.add_argument("--reasoning_config", type=str, default=None)
    args = parser.parse_args()
    return args


def build_reward_model(path, args):
    if args.reward_model_type == "Qwen2Reward":
        reward_model = Qwen2Reward.from_pretrained(path)
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    elif args.reward_model_type == "majority_voting":
        return None, None
    else:
        reward_model = CLIPModel.from_pretrained(path)
        processor = CLIPProcessor.from_pretrained(path)
        processor.tokenizer.truncation_side = "left"
    reward_model.to(f"cuda:{args.reward_model_cuda_id}")
    reward_model.eval()
    for name, param in reward_model.named_parameters():
        param.requires_grad = False
    return reward_model, processor


def extract_answer_bbox(answer):
    pattern = r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]"
    matches = re.findall(pattern, answer)

    coords = [[float(x) for x in match] for match in matches]
    return coords


def intersection_geo(box1, box2):
    x_min1, y_min1, x_max1, y_max1 = box1
    x_min2, y_min2, x_max2, y_max2 = box2

    x_min_int = max(x_min1, x_min2)
    y_min_int = max(y_min1, y_min2)
    x_max_int = min(x_max1, x_max2)
    y_max_int = min(y_max1, y_max2)

    return x_min_int, y_min_int, x_max_int, y_max_int


def calculate_area(box):
    x_min1, y_min1, x_max1, y_max1 = box
    area_box1 = (x_max1 - x_min1) * (y_max1 - y_min1)
    return area_box1


def calculate_iou(box1, box2):
    x_min1, y_min1, x_max1, y_max1 = box1
    x_min2, y_min2, x_max2, y_max2 = box2
    x_min_int, y_min_int, x_max_int, y_max_int = intersection_geo(box1, box2)

    if x_min_int >= x_max_int or y_min_int >= y_max_int:
        return 0.0

    area_int = (x_max_int - x_min_int) * (y_max_int - y_min_int)

    area_box1 = (x_max1 - x_min1) * (y_max1 - y_min1)
    area_box2 = (x_max2 - x_min2) * (y_max2 - y_min2)
    iou = area_int / (area_box1 + area_box2 - area_int)
    return iou


def main(args):
    jsonl_path = Path(args.output_path) / f"{args.model_name}_bon.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    json_path = Path(args.output_path) / f"{args.model_name}_bon.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)

    policy_model = MODEL_TYPE_MAP[args.model_type](
        model_path=args.model_path,
        model_name=args.model_name,
        temperature=args.temperature,
        top_p=args.top_p,
        beam_size=1,
        do_sample=True,
        use_cache=True,
        dtype="float16",
        device=f"cuda:{args.policy_model_cuda_id}",
        max_new_tokens=50,
        reasoning_config=args.reasoning_config,
    )
    reward_model, processor = build_reward_model(args.reward_model_path, args)

    eval_data = json.load(open(args.eval_target_file, "r"))
    eval_data = eval_data[:1000]

    final_dict = defaultdict(list)
    image_root = Path(args.eval_image_root)
    for idx, anns in tqdm(enumerate(eval_data), total=len(eval_data)):
        image_name = anns["image"]
        image_name = image_root / image_name
        w, h = Image.open(image_name).size
        question, answer = convt_qa(anns["conversations"], policy_model)
        question = question.replace("<image>\n", "")
        outputs = policy_model.generate_n(
            question, image_name, n_samples=args.sample_num
        )

        if args.reasoning_config is not None:
            outputs = [
                output.split("<answer>")[1].split("</answer>")[0].strip()
                for output in outputs
            ]

        new_outputs = []
        pased_result = []
        for output in outputs:
            parsed_output = policy_model.extract_bbox(output)
            if parsed_output is not None:
                pased_result.append(parsed_output)

                parsed_str = ""
                for idx, box in enumerate(parsed_output):
                    if not isinstance(box, list) or len(box) < 4:
                        parsed_str = output
                        break
                    else:
                        parsed_str += f"[{box[0]},{box[1]},{box[2]},{box[3]}]"
                        if idx != len(parsed_output) - 1:
                            parsed_str += ","
                new_outputs.append(parsed_str)
            else:
                pased_result.append(None)
                new_outputs.append(output)
        outputs = new_outputs

        if args.reward_model_type == "Qwen2Reward":
            messages = []
            for output in outputs:
                message = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": question},
                            {
                                "type": "image",
                                "image": f"file://{image_name}",
                            },
                        ],
                    },
                    {"role": "assistant", "content": output},
                ]
                messages.append(message)

            if len(messages) > 3:
                # single gpu only support 5 samples, otherwise will out of memory
                total_samples = len(messages)
                wrap_messages = []
                for i in range(0, total_samples, 3):
                    wrap_messages.append(messages[i : i + 3])
            else:
                wrap_messages = [messages]

            all_scores = []
            for sub_messages in wrap_messages:
                reward_texts = [
                    processor.apply_chat_template(
                        message, tokenize=False, add_generation_prompt=False
                    )
                    for message in sub_messages
                ]
                image_inputs, video_inputs = process_vision_info(sub_messages)

                reward_inputs = processor(
                    text=reward_texts,
                    images=image_inputs,
                    videos=video_inputs,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=16384,
                )
                reward_inputs = reward_inputs.to(f"cuda:{args.reward_model_cuda_id}")

                with torch.inference_mode() and torch.autocast(
                    device_type="cuda", dtype=torch.float16
                ):
                    reward_scores = (
                        reward_model(**reward_inputs)
                        .values.detach()
                        .cpu()
                        .numpy()
                        .tolist()
                    )
                all_scores.extend(reward_scores)
        elif args.reward_model_type == "majority_voting":
            # get the output that appears most frequently
            from collections import Counter

            output_counts = Counter(outputs)
            most_common_output = output_counts.most_common(1)
            if most_common_output:
                chose_answer = most_common_output[0][0]
                chose_bbox = pased_result[outputs.index(chose_answer)]
        else:
            messages = []
            for output in outputs:
                message = question + "\n" + output
                messages.append(message)

            wrap_messages = [messages]

            all_scores = []
            for sub_messages in wrap_messages:
                image = Image.open(image_name)
                inputs = processor(
                    images=image,
                    text=sub_messages,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                inputs = inputs.to(f"cuda:{args.reward_model_cuda_id}")
                with torch.inference_mode() and torch.autocast(
                    device_type="cuda", dtype=torch.float16
                ):
                    reward_scores = (
                        reward_model(**inputs)
                        .logits_per_image.softmax(dim=-1)
                        .cpu()
                        .numpy()
                        .tolist()
                    )
                all_scores.extend(reward_scores)

        if args.reward_model_type != "majority_voting":
            max_score_index = np.argmax(all_scores)
            chose_answer = outputs[max_score_index]
            chose_bbox = pased_result[max_score_index]

        result_dict = {
            "filename": str(image_name.name),
            "query": question,
            "answer": answer,
            "pred": chose_answer,
        }

        # write to jsonl
        with open(jsonl_path, "a") as f:
            f.write(json.dumps(result_dict) + "\n")

        prediction = chose_answer.strip()
        answer = answer.strip()

        answer_bbox = extract_answer_bbox(answer)
        pred_bbox_ori = chose_bbox

        if answer_bbox is not None and pred_bbox_ori is not None:
            for answer, pred in zip(answer_bbox, pred_bbox_ori):
                if answer and pred and len(pred) > 0 and len(answer) > 0:
                    if policy_model.bbox_normalize_bound is not None:
                        pred_bbox = [
                            float(pred[0] * w / policy_model.bbox_normalize_bound),
                            float(pred[1] * h / policy_model.bbox_normalize_bound),
                            float(pred[2] * w / policy_model.bbox_normalize_bound),
                            float(pred[3] * h / policy_model.bbox_normalize_bound),
                        ]
                    else:
                        pred_bbox = pred

                    try:
                        iou = calculate_iou(answer, pred_bbox)
                    except Exception as e:
                        iou = 0
                    if iou >= 0.5:
                        score = 1
                    else:
                        score = 0
                else:
                    pred = None
                    score = 0

                final_dict["score"].append(score)
                final_dict["image_name"].append(image_name)
                final_dict["question"].append(question)
                final_dict["answer"].append(answer)
                final_dict["prediction"].append(prediction)
                final_dict["answer_bbox"].append(str(answer_bbox))
                final_dict["pred_bbox"].append(str(pred_bbox_ori))
                final_dict["iou"].append(iou)
        else:
            for answer in answer_bbox:
                final_dict["score"].append(0)
                final_dict["image_name"].append(image_name)
                final_dict["question"].append(question)
                final_dict["answer"].append(answer)
                final_dict["prediction"].append(prediction)
                final_dict["answer_bbox"].append(str(answer_bbox))
                final_dict["iou"].append(0)

    avg_score = sum(final_dict["score"]) / len(final_dict["score"])
    perf_dict = {"accuracy": avg_score}
    logger.info(f"accuracy: {avg_score}")

    json.dump(perf_dict, open(json_path, "w"))


if __name__ == "__main__":
    args = get_args()

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    setup_logger(__name__, output_path, rank=0)
    main(args)
