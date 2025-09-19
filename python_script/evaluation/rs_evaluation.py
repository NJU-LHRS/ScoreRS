import argparse
import csv
import datetime
import json
import logging
import os
import pickle
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from PIL import Image
from src.evaluation.lm_prediction_call import (
    LHRSLM,
    VHM,
    GeoChatLM,
    LMDeployLM,
    SkysenseGPTLM,
)
from src.tools.logger import setup_logger
from tqdm import tqdm


Image.MAX_IMAGE_PIXELS = None
logger = logging.getLogger("RSEval")


BENCH_DATASETS = {
    # cls
    "cls_aid": ("cls_AID.json", "cls"),
    "cls_METER_ML": ("cls_METER_ML.json", "cls"),
    "cls_NWPU_RESISC45": ("cls_NWPU_RESISC45.json", "cls"),
    "cls_SIRI": ("cls_SIRI_WHU.json", "cls"),
    "cls_WHU_RS19": ("cls_WHU_RS19.json", "cls"),
    # rsvqa
    "vqa_HR-comp": ("RSVQA_HR-comp_RSVQA.json", "vqa"),
    "vqa_HR-pre": ("RSVQA_HR-presence_RSVQA.json", "vqa"),
    "vqa_HR-counting": ("RSVQA_HR-Counting_RSVQA.json", "vqa"),
    "vqa_LR-comp": ("RSVQA_LR-comp_RSVQA.json", "vqa"),
    "vqa_LR-pre": ("RSVQA_LR-presence_RSVQA.json", "vqa"),
    "vqa_LR-rural": ("RSVQA_LR-rural_urban_RSVQA.json", "vqa"),
    "vqa_LR-counting": ("RSVQA_LR-Counting_RSVQA.json", "vqa"),
    # vg
    "rs_vg": ("VG_DOIR_RSVG_test.json", "bbox"),
    # LHRS-Bench
    "lhrs_bench": ("LHRS-Bench.json", "lhrsbench"),
}

MODEL_TYPE_MAP = {
    "lhrs": LHRSLM,
    "vhm": VHM,
    "skysensegpt": SkysenseGPTLM,
    "geochat": GeoChatLM,
    "lmdeploy": LMDeployLM,
    "lmdeploy_reasoning": LMDeployLM,
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


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="all")
    parser.add_argument(
        "--data_root",
        type=str,
        default="data",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument(
        "--model_type",
        type=str,
        default="lhrs",
        choices=[
            "lhrs",
            "vhm",
            "skysensegpt",
            "geochat",
            "lmdeploy",
        ],
    )
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--beam_size", type=int, default=1)
    parser.add_argument("--do_sample", type=bool, default=False)
    parser.add_argument("--use_cache", type=bool, default=True)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--force_inference", type=bool, default=False)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="limit the number of data to evaluate",
    )
    parser.add_argument("--reasoning_config", type=str, default=None)
    return parser.parse_args()


def eval_results_vqa(args, result_json_file, save_metric_json):

    with open(result_json_file, "r") as f:
        result_lines = f.readlines()

    ret = {}
    final_dict = defaultdict(list)

    for img_i, line in enumerate(result_lines):
        result_dict = json.loads(line)
        raw_prediction = str(result_dict["pred"])
        answer = str(result_dict["answer"])

        prediction = raw_prediction
        if "<think>" in prediction:
            try:
                prediction = prediction.split("<think>")[1].split("</think>")[1]
            except:
                prediction = prediction

        if "<answer>" in prediction:
            try:
                prediction = prediction.split("<answer>")[1].split("</answer>")[0]
            except:
                prediction = prediction

        prediction = prediction.replace(" ", "")
        answer = answer.replace(" ", "")

        while " " in prediction:
            prediction = prediction.replace(" ", "")

        if "." in prediction:
            prediction = prediction.split(".")[0]

        if "," in prediction:
            prediction = prediction.split(",")[0]

        while " " in answer:
            answer = answer.replace(" ", "")

        prediction = prediction.lower()
        answer = answer.lower()

        final_dict["score"].append(prediction in answer)

        final_dict["filename"].append(str(os.path.basename(result_dict["filename"])))
        final_dict["size"].append(str(result_dict["size"]))
        final_dict["query"].append(str(result_dict["query"]))
        final_dict["answer"].append(str(result_dict["answer"]))
        final_dict["prediction"].append(str(prediction))
        final_dict["raw_prediction"].append(str(result_dict["pred"]))

    avg_score = sum(final_dict["score"]) / len(final_dict["score"])
    perf_dict = {
        "accuracy": avg_score,
    }
    ret.update(perf_dict)
    ret.update(final_dict)

    ret = pd.DataFrame({x: ret[x] for x in ret})
    ret.to_json(save_metric_json, orient="records", indent=4)
    logger.info(f"accuracy: {avg_score}")
    return perf_dict


def eval_results_bbox(
    args, model, result_json_file, save_metric_json, iou_threshold=0.5
):
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

    with open(result_json_file, "r") as f:
        result_lines = f.readlines()

    AREA_LEVEL = (32**2, 96**2, float("inf"))
    LEVEL_NAME = ("S", "M", "L")
    level_count = np.zeros(len(AREA_LEVEL))
    level_hit_count = np.zeros(len(AREA_LEVEL))

    ret = {}
    final_dict = defaultdict(list)
    for img_i, line in enumerate(result_lines):
        result_dict = json.loads(line)
        h, w = result_dict["size"]

        raw_prediction = str(result_dict["pred"])

        prediction = raw_prediction
        if "<think>" in prediction:
            try:
                prediction = prediction.split("<think>")[1].split("</think>")[1]
            except:
                prediction = prediction

        if "<answer>" in prediction:    
            try:
                prediction = prediction.split("<answer>")[1].split("</answer>")[0]
            except:
                prediction = prediction

        answer_bbox = extract_answer_bbox(str(result_dict["answer"]))
        pred_bbox_ori = model.extract_bbox(prediction)

        l = 0

        if answer_bbox is not None and pred_bbox_ori is not None:

            for answer, pred in zip(answer_bbox, pred_bbox_ori):
                while calculate_area(answer) > AREA_LEVEL[l]:
                    l += 1
                level_count[l] += 1

                if answer and pred and len(pred) > 0 and len(answer) > 0:
                    if model.bbox_normalize_bound is not None:
                        pred_bbox = [
                            float(pred[0] * w / model.bbox_normalize_bound),
                            float(pred[1] * h / model.bbox_normalize_bound),
                            float(pred[2] * w / model.bbox_normalize_bound),
                            float(pred[3] * h / model.bbox_normalize_bound),
                        ]
                    else:
                        pred_bbox = pred
                    iou = calculate_iou(answer, pred_bbox)

                    if img_i % 100 == 0:
                        logger.info(
                            f"[{img_i}/{len(result_lines)}] answer_bbox:{answer}, pred_bbox:{pred_bbox}, iou:{iou}"
                        )

                    if iou >= iou_threshold:
                        level_hit_count[l] += 1
                else:
                    pred = None

                final_dict["filename"].append(
                    str(os.path.basename(result_dict["filename"]))
                )
                final_dict["size"].append(str(result_dict["size"]))
                final_dict["query"].append(str(result_dict["query"]))
                final_dict["answer"].append(str(result_dict["answer"]))
                final_dict["pred"].append(str(result_dict["pred"]))
                final_dict["answer_bbox"].append(str(answer))
                final_dict["pred_bbox"].append(str(pred))
                final_dict["pred_bbox_ori"].append(str(pred_bbox_ori))
                final_dict["iou"].append(iou)
        else:
            for answer in answer_bbox:
                while calculate_area(answer) > AREA_LEVEL[l]:
                    l += 1
                level_count[l] += 1

                final_dict["filename"].append(
                    str(os.path.basename(result_dict["filename"]))
                )
                final_dict["size"].append(str(result_dict["size"]))
                final_dict["query"].append(str(result_dict["query"]))
                final_dict["answer"].append(str(result_dict["answer"]))
                final_dict["pred"].append(str(result_dict["pred"]))
                final_dict["answer_bbox"].append(str(answer))
                final_dict["pred_bbox"].append(str(pred_bbox_ori))
                final_dict["pred_bbox_ori"].append(str(pred_bbox_ori))
                final_dict["iou"].append(0)

    precision = np.sum(level_hit_count) / sum(level_count)
    perf_dict = {"precision": precision}
    ret.update({"precision": precision})
    level_precision = level_hit_count / level_count
    for i, name in enumerate(LEVEL_NAME):
        ret.update({f"precision_{name}_{level_count[i]}": level_precision[i]})
    ret.update(final_dict)

    ret = pd.DataFrame({x: ret[x] for x in ret})
    ret.to_json(save_metric_json, orient="records", indent=4)

    logger.info(f"precision: {precision}")
    for i, name in enumerate(LEVEL_NAME):
        logger.info(f"precision_{name}: {level_precision[i]}")
    return perf_dict


def eval_results_lhrsbench(args, result_json_file, save_metric_json):
    with open(result_json_file, "r") as f:
        result_lines = f.readlines()

    ret = {}
    final_dict = defaultdict(list)
    type_level_score = defaultdict(list)
    for img_i, line in enumerate(result_lines):
        result_dict = json.loads(line)
        raw_prediction = str(result_dict["pred"])
        prediction = raw_prediction

        if "<think>" in prediction:
            try:
                prediction = prediction.split("<think>")[1].split("</think>")[1]
            except:
                prediction = prediction

        if "<answer>" in prediction:
            try:
                prediction = prediction.split("<answer>")[1].split("</answer>")[0]
            except:
                prediction = prediction

        answer = str(result_dict["answer"])

        if "." in prediction:
            prediction = prediction.split(".")[0]

        prediction = prediction.replace(" ", "").replace(".", "").lower()
        answer = answer.replace(" ", "").replace(".", "").lower()

        types = result_dict["type"]
        for type in types:
            type_level_score[type].append(prediction in answer)

        final_dict["score"].append(prediction in answer)
        final_dict["filename"].append(str(os.path.basename(result_dict["filename"])))
        final_dict["size"].append(str(result_dict["size"]))
        final_dict["query"].append(str(result_dict["query"]))
        final_dict["answer"].append(str(result_dict["answer"]))
        final_dict["prediction"].append(str(prediction))
        final_dict["raw_prediction"].append(str(result_dict["pred"]))
        final_dict["type"].append(result_dict["type"])

    for type in type_level_score:
        type_level_score[type] = sum(type_level_score[type]) / len(
            type_level_score[type]
        )

    avg_score = sum(final_dict["score"]) / len(final_dict["score"])
    perf_dict = {"accuracy": avg_score}
    for type in type_level_score:
        perf_dict[f"accuracy_{LHRS_TYPE_MAP[type]}"] = type_level_score[type]

    ret.update(perf_dict)
    ret.update(final_dict)

    ret = pd.DataFrame({x: ret[x] for x in ret})
    ret.to_json(save_metric_json, orient="records", indent=4)

    logger.info(f"accuracy: {avg_score}")
    for type in type_level_score:
        logger.info(f"accuracy_{LHRS_TYPE_MAP[type]}: {type_level_score[type]}")
    return perf_dict


def convt_qa(conversations, task_type, model):
    values = [conversation["value"] for conversation in conversations]
    query = values[0]
    answer = values[1]

    classification_prefix = getattr(model, "cls_prefix", "")
    vqa_prefix = getattr(model, "vqa_prefix", "")
    vqa_suffix = getattr(model, "vqa_suffix", "")
    vg_prefix = getattr(model, "vg_prefix", "")
    vg_suffix = getattr(model, "vg_suffix", "")
    if "cls" in task_type:
        query = classification_prefix + " " + query
    elif "vqa" in task_type:
        query = vqa_prefix + " " + query + " " + vqa_suffix
    elif "bbox" in task_type:
        query = vg_prefix + " " + query + " " + vg_suffix

    return query, answer


def infer_single(model, anns_json_path, anns, task_type):
    fn = anns["image"]
    if "image_path" not in anns.keys():
        json_pathlib_obj = Path(anns_json_path)
        fn_full = json_pathlib_obj.parent / json_pathlib_obj.stem / fn
    elif Path(anns["image_path"]).is_absolute():
        fn_full = Path(anns["image_path"]) / fn
    else:
        dataset_base = Path(anns_json_path).parent
        fn_full = dataset_base / anns["image_path"] / fn

    if "conversations" in anns.keys():
        question, answer = convt_qa(anns["conversations"], task_type, model)
    else:
        question, answer = convt_qa(anns["conversation"], task_type, model)

    question = question.replace("<image>\n", "")
    if "size" not in anns.keys():
        image = Image.open(fn_full).convert("RGB")
        anns["size"] = image.size

    result_dict = {
        "filename": str(fn_full),
        "size": anns["size"],
        "query": question,
        "answer": answer,
    }

    if "type" in anns.keys():
        result_dict["type"] = anns["type"]

    outputs = model.generate(prompt=question, image_files=fn_full)
    if isinstance(outputs, list):
        outputs = outputs[0]
    result_dict["pred"] = outputs
    return result_dict


def infer_model(
    args, model, anns_json_path, task_type, save_json, local_rank, world_size
):
    with open(anns_json_path, "r") as f:
        anns_dict = json.load(f)

    if args.limit is not None:
        anns_dict = anns_dict[: args.limit]

    chunk_size = len(anns_dict) // world_size
    sub_lists = [
        anns_dict[i : i + chunk_size] for i in range(0, len(anns_dict), chunk_size)
    ]
    if len(anns_dict) % world_size != 0:
        sub_lists[-2] = sub_lists[-2] + sub_lists[-1]

    sub_anns_dict = sub_lists[local_rank]

    final_results = []
    for idx, anns in tqdm(enumerate(sub_anns_dict), total=len(sub_anns_dict)):
        result_dict = infer_single(model, anns_json_path, anns, task_type)

        final_results.append(json.dumps(result_dict))

    if world_size > 1:
        model.accelerator.wait_for_everyone()
        if local_rank == 0:
            gathered_objects = (
                model.accelerator.gather(final_results).cpu().detach().numpy().tolist()
            )

            final_results = []
            for sublist in gathered_objects:
                final_results.extend(sublist)

    if local_rank == 0:
        with open(save_json, "w") as f:
            f.write("\n".join(final_results))


def eval_task(args, model, task_index, local_rank, world_size):
    anns_json, task_type = BENCH_DATASETS[task_index]

    anns_json_path = Path(args.data_root) / anns_json

    test_name = Path(anns_json_path).stem

    if args.model_type == "lmdeploy":
        model_name = Path(args.model_path).stem
    else:
        model_name = args.model_type
    save_json = Path(args.output_dir) / f"{test_name}_{model_name}_eval.jsonl"
    save_metric_json = Path(args.output_dir) / f"{test_name}_{model_name}_eval.json"

    if not save_json.exists() or args.force_inference:
        infer_model(
            args,
            model,
            anns_json_path,
            task_type,
            save_json,
            local_rank,
            world_size,
        )
    else:
        # check integrity
        with open(anns_json_path, "r") as f:
            anns_dict = json.load(f)
        with open(save_json, "r") as f:
            save_dict = f.readlines()
        if len(anns_dict) != len(save_dict):
            infer_model(
                args,
                model,
                anns_json_path,
                task_type,
                save_json,
                local_rank,
                world_size,
            )

    if local_rank == 0:
        if task_type == "bbox":
            return eval_results_bbox(args, model, save_json, save_metric_json)
        elif task_type in ("cls", "vqa"):
            return eval_results_vqa(args, save_json, save_metric_json)
        elif task_type == "lhrsbench":
            return eval_results_lhrsbench(args, save_json, save_metric_json)
    return None


if __name__ == "__main__":
    args = arg_parser()
    output_dir = Path(args.output_dir)

    kwargs_handler = InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=60000))
    accelerator = Accelerator(kwargs_handlers=[kwargs_handler])

    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)

    setup_logger("RSEval", args.output_dir, rank=accelerator.local_process_index)
    logger.info(f"Args: {args}")

    device = accelerator.device
    extra_kwargs = {}
    if args.reasoning_config is not None:
        extra_kwargs["reasoning_config"] = args.reasoning_config
    model = MODEL_TYPE_MAP[args.model_type](
        model_path=args.model_path,
        temperature=args.temperature,
        top_p=args.top_p,
        beam_size=args.beam_size,
        do_sample=args.do_sample,
        use_cache=args.use_cache,
        dtype=args.dtype,
        device=device,
        max_new_tokens=args.max_new_tokens,
        **extra_kwargs,
    )

    model.accelerator = accelerator
    model.rank = accelerator.local_process_index
    model.world_size = accelerator.num_processes

    total_metrics = {}
    if args.task == "all":
        for task_index in BENCH_DATASETS.keys():
            if accelerator.is_main_process:
                logger.info(f"===================={task_index}======================")
                metrics = eval_task(
                    args,
                    model,
                    task_index,
                    accelerator.local_process_index,
                    accelerator.num_processes,
                )
                total_metrics.update({task_index: metrics})
    else:
        metrics = eval_task(
            args,
            model,
            args.task,
            accelerator.local_process_index,
            accelerator.num_processes,
        )
        total_metrics.update({args.task: metrics})
