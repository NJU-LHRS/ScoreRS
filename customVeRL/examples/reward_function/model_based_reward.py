import re
from typing import Dict, List
from PIL import Image
import numpy as np
import requests
import base64
from io import BytesIO

from mathruler.grader import extract_boxed_content, grade_answer


BOX_START = "<|box_start|>"
BOX_END = "<|box_end|>"
BOX_PATTERN = re.compile(
    r"\((\d+(?:\.\d+)?),(\d+(?:\.\d+)?)\),\((\d+(?:\.\d+)?),(\d+(?:\.\d+)?)\)"
)
BRACKET_BOX_PATTERN = re.compile(
    r"\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]"
)


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


def ensure_equal_normalized(box1, box2):
    box1_x1 = box1[0]
    box2_x1 = box2[0]

    if box1_x1 < 1 and box2_x1 > 1:
        box1 = [coord * 1000 for coord in box1]
    elif box1_x1 > 1 and box2_x1 < 1:
        box2 = [coord * 1000 for coord in box2]

    return box1, box2


def calculate_iou(box1, box2):
    box1, box2 = ensure_equal_normalized(box1, box2)
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


def get_model_score(
    text1: str,
    text2: str,
    problem: str,
    images: List[Image.Image],
    reward_server_url: str,
    reward_model_name: str,
) -> float:
    predict_match = re.search(r"<answer>(.*?)</answer>", text1)
    predict_answer = predict_match.group(1).strip() if predict_match else text1.strip()
    ground_truth = text2

    problem = problem.replace("<image>", "")
    if isinstance(images, list):
        image = images[0]
    else:
        image = images

    base64_image_data = base64.b64encode(image).decode("utf-8")
    payload = {
        "model": reward_model_name,
        "problem_description": problem,
        "completion1": predict_answer,
        "completion2": ground_truth,
    }

    if base64_image_data:
        payload["image_data"] = f"data:image/png;base64,{base64_image_data}"

    success_flag = False

    for _ in range(3):
        # try 3 times
        response = requests.post(reward_server_url, json=payload)
        status_code = response.status_code
        if status_code == 200:
            success_flag = True
            break

    if success_flag:
        response_json = response.json()
        scores = response_json["scores"]
        predict_score = scores[0]["score"]
        ground_truth_score = scores[1]["score"]

        if predict_score > ground_truth_score:
            diff = predict_score - ground_truth_score
            reward = 1.0 - np.exp(-diff * 0.2)
        else:
            reward = 0.0
    else:
        reward = 0

    return reward


def format_reward(predict_str: str) -> float:
    pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    format_match = re.fullmatch(pattern, predict_str)
    return 1.0 if format_match else 0.0


def iou_reward(predict_str: str, ground_truth: str) -> float:
    predict_match = re.search(r"<answer>(.*?)</answer>", predict_str)
    predict_answer = (
        predict_match.group(1).strip() if predict_match else predict_str.strip()
    )

    reward = 0.0
    parse_success = False

    if BOX_START in ground_truth and BOX_END in ground_truth:
        answer_coords = BOX_PATTERN.findall(ground_truth)
        answer_coords = [list(map(float, coord)) for coord in answer_coords]
    else:
        answer_coords = BRACKET_BOX_PATTERN.findall(ground_truth)
        answer_coords = [list(map(float, coord)) for coord in answer_coords]

    predict_coords = BOX_PATTERN.findall(predict_answer)
    if len(predict_coords) == 0:
        predict_coords = BRACKET_BOX_PATTERN.findall(predict_answer)
        if len(predict_coords) == 0:
            parse_success = False
        else:
            predict_coords = [list(map(float, coord)) for coord in predict_coords]
            parse_success = True

    if parse_success:
        if len(predict_coords) != len(answer_coords):
            reward = 0.0
        else:
            reward_for_per_correct_box = 0.5 / len(answer_coords)
            for pred_coord, ans_coord in zip(predict_coords, answer_coords):
                iou = calculate_iou(pred_coord, ans_coord)
                if iou > 0.5:
                    reward += reward_for_per_correct_box

                if iou > 0.8:
                    reward += 0.5
                elif iou > 0.7:
                    reward += 0.2
                elif iou > 0.6:
                    reward += 0.1
    else:
        reward = 0.0

    return reward


def math_reward(predict_str: str, ground_truth: str) -> float:
    predict_match = re.search(r"<answer>(.*?)</answer>", predict_str)
    predict_answer = (
        predict_match.group(1).strip() if predict_match else predict_str.strip()
    )

    gt_answer = extract_boxed_content(ground_truth)
    if gt_answer == "None":
        gt_answer = ground_truth

    predict_answer = extract_boxed_content(predict_answer)
    if predict_answer == "None":
        predict_answer = predict_str

    return 1.0 if grade_answer(predict_answer, gt_answer) else 0.0


def em_reward(predict_str: str, ground_truth: str) -> float:
    predict_match = re.search(r"<answer>(.*?)</answer>", predict_str)
    predict_answer = (
        predict_match.group(1).strip() if predict_match else predict_str.strip()
    )

    processed_predict = predict_answer.lower().replace(" ", "").replace(".", "")
    while " " in processed_predict:
        processed_predict = processed_predict.replace(" ", "")

    processed_solution = ground_truth.lower().replace(" ", "").replace(".", "")
    while " " in processed_solution:
        processed_solution = processed_solution.replace(" ", "")

    if processed_predict == processed_solution:
        reward = 1.0
    else:
        reward = 0.0

    return reward


def compute_score(
    predict_str: str,
    ground_truth: str,
    problem: str = "",
    images: List[Image.Image] = [],
    reward_type: str = "em",
    reward_model_name: str = "",
    reward_server_url: str = "",
    format_weight: float = 0.3,
) -> Dict[str, float]:
    format_score = format_reward(predict_str)
    if format_score == 0.0:
        return {"overall": 0.0, "format": format_score, "accuracy": 0.0}
    else:
        if reward_type == "em":
            accuracy_score = em_reward(predict_str, ground_truth)
        elif reward_type == "iou":
            accuracy_score = iou_reward(predict_str, ground_truth)
        elif reward_type == "open_ended":
            accuracy_score = get_model_score(
                predict_str,
                ground_truth,
                problem,
                images,
                reward_server_url,
                reward_model_name,
            )
        elif reward_type == "math":
            accuracy_score = math_reward(predict_str, ground_truth)

    # The model score itself serves as a continuous reward
    overall_score = format_weight * format_score + (1 - format_weight) * accuracy_score
    return {
        "overall": overall_score,
        "format_score": format_score,
        "accuracy_score": accuracy_score,
    }
