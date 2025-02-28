import base64
import json
import os
from abc import ABC, abstractmethod
from glob import glob
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset
from transformers import BaseImageProcessor

Image.MAX_IMAGE_PIXELS = None
DEFAULT_IMAGE_TOKEN = "<image>"


class BasedDataset(Dataset, ABC):
    def __init__(self, data_root, transform=None):
        if not isinstance(data_root, Path):
            data_root = Path(data_root)

        self.data_root = data_root
        self.transform = transform
        self.img_list = []
        self._load_data()
        self._post_process()

    @abstractmethod
    def __getitem__(self, index):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def _load_data(self):
        pass

    def _post_process(self):
        pass

    def _get_img(self, img_path):
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            if isinstance(self.transform, BaseImageProcessor):
                img = self.transform(img, return_tensors="pt")["pixel_values"].squeeze()
            else:
                img = self.transform(img)

        return img


class ImageDataset(BasedDataset):
    def __init__(self, data_root, transform=None, walk_root=False, suffix="*.jpg"):
        self.walk_root = walk_root
        self.suffix = suffix
        super().__init__(data_root, transform)

    def __getitem__(self, index):
        img_path = self.img_list[index]
        img = self._get_img(img_path)
        return img, str(img_path)

    def __len__(self):
        return len(self.img_list)

    def _load_data(self):
        if self.walk_root:
            if isinstance(self.suffix, str):
                self.img_list = list(self.data_root.rglob(self.suffix))
            else:
                assert isinstance(self.suffix, list)
                self.img_list = []
                for suffix in self.suffix:
                    self.img_list.extend(list(self.data_root.rglob(suffix)))
        else:
            if isinstance(self.suffix, str):
                self.img_list = list(self.data_root.glob(self.suffix))
            else:
                assert isinstance(self.suffix, list)
                self.img_list = []
                for suffix in self.suffix:
                    self.img_list.extend(list(self.data_root.glob(suffix)))


class ImageDatasetFromFile(BasedDataset):
    def __init__(
        self, data_root, image_file, file_type="txt", image_column=None, transform=None
    ):
        self.image_file = image_file
        self.file_type = file_type
        self.image_column = image_column
        super().__init__(data_root, transform)

    def __len__(self):
        return len(self.img_list)

    def _load_data(self):
        if self.file_type == "txt":
            with open(self.image_file, "r") as f:
                self.img_list = [Path(line.strip()) for line in f.readlines()]
        elif self.file_type == "csv":
            df = pd.read_csv(self.image_file)
            df_image = df.groupby(self.image_column).count().reset_index()
            if self.data_root is not None:
                df_image[self.image_column] = df_image[self.image_column].apply(
                    lambda x: os.path.join(self.data_root, x)
                )
            self.img_list = df_image[self.image_column].tolist()
        elif self.file_type == "dataframe":
            if self.data_root is not None:
                df = self.image_file
                df[self.image_column] = df[self.image_column].apply(
                    lambda x: os.path.join(self.data_root, x)
                )
            self.img_list = df[self.image_column].tolist()

    def _post_process(self):
        new_img_list = []
        for image_path in self.img_list:
            if os.path.exists(image_path):
                new_img_list.append(image_path)
        self.img_list = new_img_list

    def __getitem__(self, index):

        img = self._get_img(self.img_list[index])
        return img, str(self.img_list[index])


class TextDataset(Dataset):
    def __init__(
        self,
        target_root,
        target_type="csv",
        tokenizer=None,
        target_column="title_multi_objects",
    ):
        self.target_root = target_root
        self.tokenizer = tokenizer
        self.target_type = target_type
        self.target_column = target_column
        self._load_data()

    def _load_data(self):
        if self.target_type == "csv":
            df = pd.read_csv(self.target_root)
            self.target_root = df.groupby(self.target_column).count().reset_index()
        elif self.target_type == "dataframe":
            self.target_root = self.target_root
        else:
            raise ValueError(f"Invalid target type: {self.target_type}")

    def _get_target(self, index):
        if self.target_type == "csv" or self.target_type == "dataframe":
            return self.target_root.iloc[index][self.target_column]
        else:
            raise ValueError(f"Invalid target type: {self.target_type}")

    def __len__(self):
        return len(self.target_root)

    def __getitem__(self, index):

        text = self._get_target(index)
        text_target = self.tokenizer(
            text, return_tensors="pt", padding="max_length", truncation=True
        )
        return dict(
            text=text,
            input_ids=text_target.input_ids.squeeze(),
            attention_mask=text_target.attention_mask.squeeze(),
        )


class ImageTextDataset(BasedDataset):
    def __init__(
        self,
        data_root,
        target_root,
        target_type="csv",
        transform=None,
        tokenizer=None,
        target_column="title_multi_objects",
        image_column="image_name",
        filter_record=False,
        save_top_percent=0.3,
        filter_column="clip_score",
    ):
        self.target_root = target_root
        self.tokenizer = tokenizer
        self.target_type = target_type
        self.target_column = target_column
        self.image_column = image_column
        self.filter_record = filter_record
        self.save_top_percent = save_top_percent
        self.filter_column = filter_column
        super().__init__(data_root, transform)

    def _load_data(self):
        if self.target_type == "csv":
            self.target_root = pd.read_csv(self.target_root)

        else:
            raise ValueError(f"Invalid target type: {self.target_type}")

    def _post_process(self):
        # check whether each of the target image exists
        for index, row in self.target_root.iterrows():
            img_path = self.data_root / row[self.image_column]
            if not img_path.exists():
                self.target_root = self.target_root.drop(index)

        if self.filter_record:
            self.target_root = self.target_root.sort_values(
                by=self.filter_column, ascending=False
            )
            self.target_root = self.target_root.head(
                int(len(self.target_root) * self.save_top_percent)
            )

    def _get_target(self, index):
        if self.target_type == "csv":
            return self.target_root.iloc[index][self.target_column]
        else:
            raise ValueError(f"Invalid target type: {self.target_type}")

    def __len__(self):
        return len(self.target_root)

    def __getitem__(self, index):
        img_path = self.target_root.iloc[index][self.image_column]
        img_path = self.data_root / img_path
        img = self._get_img(img_path)
        text_target = self._get_target(index)
        text_target = self.tokenizer(
            text_target, return_tensors="pt", padding="max_length", truncation=True
        )
        return dict(
            pixel_values=img,
            input_ids=text_target.input_ids.squeeze(),
            attention_mask=text_target.attention_mask.squeeze(),
        )


class Qwen2RewardDataset(BasedDataset):
    def __init__(self, data_root, image_root, transform=None):
        if not isinstance(image_root, Path) and image_root is not None:
            image_root = Path(image_root)

        self.image_root = image_root
        super().__init__(data_root, transform)

    def _build_message(self, question, answer, image_path=None, image_url=None):
        if image_path is not None:
            message = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": str(image_path),
                        },
                        {
                            "type": "text",
                            "text": question,
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": answer,
                        }
                    ],
                },
            ]
        elif image_url is not None:
            message = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": image_url,
                        },
                        {
                            "type": "text",
                            "text": question,
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": answer,
                        }
                    ],
                },
            ]
        else:
            message = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": question,
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": answer,
                        }
                    ],
                },
            ]
        return message

    def build_message(self, item, imagefolder=None, image_column=None):
        column_names = item.keys()
        if "question" not in column_names and "text" in column_names:
            text_message = item["text"]
            text_message = json.loads(text_message)
        else:
            text_message = item

        if "question" in column_names:
            question = text_message["question"]
        else:
            question = ""
        chosen_answer = text_message["chosen"]
        rejected_answer = text_message["rejected"]
        if not isinstance(rejected_answer, list):
            rejected_answer = [rejected_answer]

        if imagefolder is not None:
            image_name = (
                item["image_name"] if "image_name" in item else item["image_path"]
            )
            image_path = imagefolder / image_name

            if not image_path.exists():
                return None

            image_path = str(image_path)
            image_url = None
        elif image_column is not None:
            image_path = item[image_column]
            if isinstance(image_path, dict):
                image_path = image_path["bytes"]

            try:
                image_url = base64.b64encode(image_path).decode("utf-8")
                image_url = f"data:image/jpeg;base64,{image_url}"
                image_path = None
            except:
                path = Path(item[image_column]["path"]).name
                image_path = self.image_root / "CoCo" / path

                if not image_path.exists():
                    return None

                image_path = str(image_path)
                image_url = None
        else:
            image_path = None
            image_url = None

        ans_message = self._build_message(
            question, chosen_answer, image_path, image_url
        )

        for rej_answer_i in rejected_answer:
            rej_message = self._build_message(question, rej_answer_i, image_path)
            self.data_list.append([ans_message, rej_message])

    def load_from_parquet(self, parquet_path):
        df = pd.read_parquet(parquet_path)
        for index, row in df.iterrows():
            self.build_message(row, image_column="image")

    def load_text_dataset(self, data):
        for item in data:
            self.build_message(item)

    def _load_data(self):
        self.data_list = []

        for json_path in self.data_root.glob("*.json"):
            with open(json_path, "r") as f:
                data = json.load(f)

            dataset_name = json_path.stem

            if self.image_root is not None:
                imagefolder = self.image_root / dataset_name
            else:
                imagefolder = None

            if imagefolder is not None and not imagefolder.exists():
                # pure text dataset
                self.load_text_dataset(data)
                continue

            for item in data:
                self.build_message(item, imagefolder)

        for parquet_path in self.data_root.glob("*.parquet"):
            self.load_from_parquet(parquet_path)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        """
        Return:
            inputs: dict
                  input_ids: torch.Tensor
                  attention_mask: torch.Tensor
                  pixel_values: torch.Tensor
                  pixel_values_videos: torch.Tensor
                  image_grid_thw: torch.Tensor
                  video_grid_thw
        """
        pair = self.data_list[index]

        texts = [
            self.transform.apply_chat_template(
                message, tokenizer=False, add_generation_prompt=False
            )
            for message in pair
        ]

        image_inputs, video_inputs = process_vision_info(pair)
        inputs = self.transform(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        return inputs


class ShareGPTDataset2Qwen2RewardDataset(BasedDataset):
    def _load_data(self):
        self.messages = []
        all_data = json.load(open(self.data_root))
        for item in all_data:
            self.build_message(item)

    def build_message(self, item):
        images = item["images"]
        conversations = item["messages"]
        message = []

        image_idx = 0
        for idx, conversation in enumerate(conversations):
            if conversation["from"] == "human":
                content = conversation["value"]
                if DEFAULT_IMAGE_TOKEN in content:
                    content = content.replace(DEFAULT_IMAGE_TOKEN, "")
                    message.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": content},
                                {"type": "image", "image": str(images[image_idx])},
                            ],
                        }
                    )
                    image_idx += 1
                else:
                    message.append(
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": content}],
                        }
                    )

            if conversation["from"] == "gpt":
                message.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": conversation["value"]}],
                    }
                )

        self.messages.append(message)

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, index):
        message = self.messages[index]
        text = self.transform.apply_chat_template(
            message, tokenizer=False, add_generation_prompt=False
        )
        image_inputs, video_inputs = process_vision_info(message)
        inputs = self.transform(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs["index"] = torch.tensor(index)
        return inputs


class Qwen2ImageTextDataset(Qwen2RewardDataset):
    def __init__(
        self,
        data_root,
        image_root,
        image_column,
        target_column,
        transform,
        target_type="csv",
    ):
        self.image_column = image_column
        self.target_column = target_column
        self.target_type = target_type
        super().__init__(data_root, image_root, transform)

    def _load_data(self):
        if self.target_type == "csv":
            self.target_root = pd.read_csv(self.data_root)
        elif self.target_type == "json":
            self.target_root = json.load(open(self.data_root))
        else:
            raise ValueError(f"Invalid target type: {self.target_type}")

    def _post_process(self):
        # check whether each of the target image exists
        if self.target_type == "csv":
            for index, row in self.target_root.iterrows():
                img_path = self.image_root / row[self.image_column]
                if not img_path.exists():
                    self.target_root = self.target_root.drop(index)
        elif self.target_type == "json":
            new_data_list = []
            for item in self.target_root:
                images = item["images"]

                all_exist = True
                for image in images:
                    if not Path(image).exists():
                        all_exist = False
                        break
                if all_exist:
                    new_data_list.append(item)
            self.target_root = new_data_list
        else:
            raise ValueError(f"Invalid target type: {self.target_type}")

    def _get_target(self, index):
        if self.target_type == "csv":
            return self.target_root.iloc[index][self.target_column]
        elif self.target_type == "json":
            return self.target_root[index]["messages"]
        else:
            raise ValueError(f"Invalid target type: {self.target_type}")

    def _build_message_from_conversations(self, img_path, text_target):
        current_image_index = 0
        conversations_length = len(text_target)
        messages = []
        for i in range(0, conversations_length, 2):
            question = text_target[i]["value"]
            answer = text_target[i + 1]["value"]

            if DEFAULT_IMAGE_TOKEN in question:
                question = question.replace(DEFAULT_IMAGE_TOKEN, "")
                question_message = {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": img_path[current_image_index],
                        },
                        {
                            "type": "text",
                            "text": question,
                        },
                    ],
                }
                messages.append(question_message)
                current_image_index += 1

                answer_message = {
                    "role": "assistant",
                    "content": [{"type": "text", "text": answer}],
                }
                messages.append(answer_message)
            else:
                messages.append(
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": question}],
                    }
                )
                messages.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": answer}],
                    }
                )

        assert current_image_index == len(
            img_path
        ), "The number of images does not match the number of conversations"
        assert (
            len(messages) == conversations_length
        ), "The number of messages does not match the number of conversations"
        return messages

    def __len__(self):
        return len(self.target_root)

    def __getitem__(self, index):
        if self.target_type == "csv":
            img_path = self.target_root.iloc[index][self.image_column]
            img_path = self.image_root / img_path
        elif self.target_type == "json":
            img_path = self.target_root[index]["images"]

        text_target = self._get_target(index)

        if self.target_type == "csv":
            message = self._build_message(
                "",
                text_target,
                image_path=img_path,
            )
        elif self.target_type == "json":
            message = self._build_message_from_conversations(
                img_path,
                text_target,
            )

        text = self.transform.apply_chat_template(
            message, tokenizer=False, add_generation_prompt=False
        )

        image_inputs, video_inputs = process_vision_info(message)
        inputs = self.transform(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs["index"] = torch.tensor(index)

        return inputs


class RSVQA(BasedDataset):
    """Base RSVQA dataset"""

    splits = ["train", "val", "test"]
    prefix = ""

    def __init__(
        self,
        data_root: str = "",
        image_root: str = None,
        split: str = "train",
        return_image: bool = False,
    ):
        assert split in self.splits
        self.split = split
        self.image_root = Path(image_root)
        self.return_image = return_image
        super().__init__(data_root)

    def _load_data(self):
        paths = glob(str(self.image_root / "*.tif"))
        paths = sorted(paths, key=sort)
        with open(
            self.data_root / f"{self.prefix}_split_{self.split}_questions.json"
        ) as f:
            questions = json.load(f)["questions"]
        with open(
            self.data_root / f"{self.prefix}_split_{self.split}_answers.json"
        ) as f:
            answers = json.load(f)["answers"]
        with open(
            self.data_root / f"{self.prefix}_split_{self.split}_images.json"
        ) as f:
            images = json.load(f)["images"]
        ids = [x["id"] for x in images if x["active"]]

        self.ids = ids
        self.paths = paths
        self.images = images
        self.questions = questions
        self.answers = answers

    def _post_process(self):
        neglect_question_type = ["count", "area"]

        new_questions = []
        new_ids = []
        for id in self.ids:
            questions_ids = self.images[id]["questions_ids"]
            valid_questions_ids = [
                i
                for i in questions_ids
                if self.questions[i]["type"].lower() not in neglect_question_type
            ]
            new_questions.extend(valid_questions_ids)
            new_ids.extend([id] * len(valid_questions_ids))

        self.questions_ids = new_questions
        self.ids = new_ids

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Dict:
        """Returns a dict containing x, questions, answers, q/a category
        x: (3, h, w)
        questions: List[torch.Tensor]
        answers: List[str]
        types: List[str]
        """
        id = self.ids[idx]
        image_path = self.image_root / f"{id}.tif"
        if self.return_image:
            x = self._get_img(image_path)

        questions = self.questions[self.questions_ids[idx]]
        answers = self.answers[questions["answers_ids"][0]]["answer"]
        types = questions["type"]
        questions = questions["question"]
        if "?" not in questions:
            questions += "?"
        questions += " Answer in a single word. Answer:"

        if self.return_image:
            output = dict(
                image=x,
                image_path=image_path,
                question=questions,
                answer=answers,
                type=types,
                questions_idx=self.questions_ids[idx],
            )
        else:
            output = dict(
                image_path=image_path,
                question=questions,
                answer=answers,
                type=types,
                questions_idx=self.questions_ids[idx],
            )

        return output


class RSVQALR(RSVQA):
    prefix = "LR"

    def __init__(self, data_root: str = ".data/RSVQA_LR", *args, **kwargs):
        super().__init__(data_root, *args, **kwargs)


class RSVQAHR(RSVQA):
    prefix = "USGS"

    def __init__(self, data_root: str = ".data/RSVQA_HR", *args, **kwargs):
        super().__init__(data_root, *args, **kwargs)


class EarthVQA(BasedDataset):
    def __init__(
        self,
        data_root: str = ".data/EarthVQA",
        image_root: str = ".data/EarthVQA/images",
        return_image: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(data_root, *args, **kwargs)
        self.image_root = Path(image_root)
        self.return_image = return_image

    def _load_data(self):
        self.data_list = json.load(open(self.data_root))

    def _post_process(self):
        flatten_image_question_answer_dict = []
        for image_name, image_data in self.data_list.items():
            for item in image_data:
                item["image_path"] = image_name
                flatten_image_question_answer_dict.append(item)
        self.data_list = flatten_image_question_answer_dict

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        image_name = self.data_list[idx]["image_path"]
        image_path = self.image_root / image_name
        if self.return_image:
            image = self._get_img(image_path)

        question = self.data_list[idx]["Question"]
        answer = self.data_list[idx]["Answer"]
        type = self.data_list[idx]["Type"]

        if self.return_image:
            return dict(
                image=image,
                image_path=image_path,
                question=question,
                answer=answer,
                type=type,
            )
        else:
            return dict(
                image_path=image_path,
                question=question,
                answer=answer,
                type=type,
            )


def sort(x):
    x = os.path.basename(x)
    x = os.path.splitext(x)[0]
    return int(x)
