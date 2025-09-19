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
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.padding_side = "right"
        reward_model.config.pad_token_id = processor.tokenizer.pad_token_id
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


def main(args):
    if args.model_type == "lmdeploy":
        model_name = Path(args.model_path).stem
    else:
        model_name = args.model_type
    jsonl_path = Path(args.output_path) / f"{model_name}_bon.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    json_path = Path(args.output_path) / f"{model_name}_bon.json"
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

    ret = {}
    final_dict = defaultdict(list)
    type_level_score = defaultdict(list)
    image_root = Path(args.eval_image_root)
    for idx, anns in tqdm(enumerate(eval_data), total=len(eval_data)):
        image_name = anns["image"]
        image_name = image_root / image_name
        types = anns["type"]
        question, answer = convt_qa(anns["conversations"], policy_model)
        question = question.replace("<image>\n", "")
        outputs = policy_model.generate_n(
            question, image_name, n_samples=args.sample_num
        )

        if args.reasoning_config is not None:
            new_outputs = []
            for output in outputs:
                try:
                    new_outputs.append(
                        output.split("<answer>")[1].split("</answer>")[0].strip()
                    )
                except:
                    new_outputs.append(output)
            outputs = new_outputs

        if args.reward_model_type == "Qwen2Reward":
            messages = []
            for output in outputs:
                if "." in output:
                    output = output.split(".")[0]
                output = output.replace(" ", "").strip()
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

            if len(messages) > 2:
                # single gpu only support 2 samples, otherwise will out of memory
                total_samples = len(messages)
                wrap_messages = []
                for i in range(0, total_samples, 2):
                    wrap_messages.append(messages[i : i + 2])
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
        else:
            messages = []
            for output in outputs:
                if "." in output:
                    output = output.split(".")[0]
                output = output.replace(" ", "").strip()
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
            # get the max score index
            max_score_index = np.argmax(all_scores)
            chose_answer = outputs[max_score_index]

        result_dict = {
            "filename": str(image_name.name),
            "query": question,
            "answer": answer,
            "pred": chose_answer,
            "type": types,
        }

        # write to jsonl
        with open(jsonl_path, "a") as f:
            f.write(json.dumps(result_dict) + "\n")

        if "." in chose_answer:
            prediction = chose_answer.split(".")[0]
        else:
            prediction = chose_answer
        prediction = prediction.replace(" ", "").replace(".", "").lower()
        answer = answer.replace(" ", "").replace(".", "").lower()

        for type in types:
            type_level_score[type].append(prediction in answer)

        final_dict["score"].append(prediction in answer)
        final_dict["image_name"].append(image_name)
        final_dict["question"].append(question)
        final_dict["answer"].append(answer)
        final_dict["prediction"].append(prediction)
        final_dict["type"].append(types)

    for type in type_level_score:
        type_level_score[type] = sum(type_level_score[type]) / len(
            type_level_score[type]
        )

    avg_score = sum(final_dict["score"]) / len(final_dict["score"])
    perf_dict = {"accuracy": avg_score}
    for type in type_level_score:
        perf_dict[f"accuracy_{LHRS_TYPE_MAP[type]}"] = type_level_score[type]

    json.dump(perf_dict, open(json_path, "w"))


if __name__ == "__main__":
    args = get_args()

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    setup_logger(__name__, output_path, rank=0)
    main(args)
