import hashlib
import json
import logging
import os
import re
import random
from abc import ABC, abstractmethod
from typing import List

import torch
from lmdeploy import (
    GenerationConfig,
    TurbomindEngineConfig,
    pipeline,
    ChatTemplateConfig,
)
from lmdeploy.vl import load_image
from PIL import Image
from src.evaluation.eval_utils import Collator
from src.model.geochat import load_pretrained_model
from src.model.lhrsbot_llama import LHRSBotLlamaForCausalLM
from src.model.llava_constant import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from src.model.llava_mm_utils import process_images, tokenizer_image_token
from src.model.skysensegpt import (
    load_pretrained_model as load_skysensegpt_pretrained_model,
)
from src.model.vhm import VHM as VHM_Loader
from src.preprocess.conversation_template import conv_templates
from tqdm import tqdm
from transformers import AutoTokenizer
from .vllm_template_factory import (
    model_example_map,
    load_llm,
)
from qwen_vl_utils import process_vision_info
from PIL import Image
from vllm import SamplingParams

eval_logger = logging.getLogger("RSEval")

Image.MAX_IMAGE_PIXELS = None

STR_TO_TYPE = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
}


class CacheHook:
    def __init__(self, cachinglm) -> None:
        if cachinglm is None:
            self.dbdict = None
            return

        self.dbdict = cachinglm.dbdict

    def add_partial(self, attr, req, res) -> None:
        if self.dbdict is None:
            return
        hsh = hash_args(attr, req)
        self.dbdict[hsh] = res


class LM(ABC):
    def __init__(
        self,
        model_path: str,
        temperature: float = 1.0,
        top_p: float = 1.0,
        beam_size: int = 1,
        do_sample: bool = False,
        use_cache: bool = True,
        device: str = "cuda",
        dtype: str = "float16",
        max_new_tokens: int = 256,
    ):
        self.model_path = model_path
        self.temperature = temperature
        self.top_p = top_p
        self.beam_size = beam_size
        self.do_sample = do_sample
        self.use_cache = use_cache
        self.device = device
        self.dtype = STR_TO_TYPE[dtype]
        self.max_new_tokens = max_new_tokens
        self.cache_hook = CacheHook(None)
        self.load_everything()

    @abstractmethod
    def load_everything(self):
        pass

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        pass


class VLLMLM(LM):
    def __init__(self, *args, **kwargs):
        self.reasoning_config = kwargs.pop("reasoning_config", None)
        self.model_name = kwargs.pop("model_name", None)
        super().__init__(*args, **kwargs)

    def load_everything(self):
        self.model, self.processor = load_llm(
            self.model_name, self.model_path, self.device
        )
        self.sampling_param = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=16384,
        )

        if (
            self.model_name == "qwen2-vl"
            or self.model_name == "qwen2-vl-chat"
            or self.model_name == "qwen2-vl-r1"
        ):
            if "Qwen2.5-VL" in self.model_path:
                self.vg_prefix = "Output the bounding box of this object in json format: "
                self.vg_suffix = ""
                self.bbox_normalize_bound = None
                self.extract_bbox = self._extract_bbox_qwen2_5vl
            else:
                self.vg_prefix = "Output the bounding box of the following object in the image. <|object_ref_start|>"
                self.vg_suffix = "<|object_ref_end|>"
                self.bbox_normalize_bound = 1000
                self.extract_bbox = self._extract_bbox_qwen2vl
        elif self.model_name == "internvl":
            if "InternVL3" in self.model_path:
                self.vg_prefix = "Please provide the bounding box coordinate of the region this sentence describes: <ref>"
                self.vg_suffix = "</ref>"
                self.bbox_normalize_bound = 1000
                self.extract_bbox = self._extract_bbox_internvl3
            else:
                self.vg_prefix = "Please provide the bounding box coordinate of the region this sentence describes: <ref>"
                self.vg_suffix = "</ref>"
                self.bbox_normalize_bound = 1000
                self.extract_bbox = self._extract_bbox_internvl
        else:
            config_path = os.path.join(self.model_path, "config.json")
            model_config = json.load(open(config_path, "r"))
            if (
                "Qwen2-VL" in model_config["_name_or_path"]
                or "qwen2_vl" in model_config["_name_or_path"]
                or "qwen2_vl" in model_config["model_type"]
            ):
                self.vg_prefix = "Output the bounding box of the following object in the image. <|object_ref_start|>"
                self.vg_suffix = "<|object_ref_end|>"
                self.bbox_normalize_bound = 1000
                if self.reasoning_config:
                    self.bbox_normalize_bound = 1000
                self.extract_bbox = self._extract_bbox_qwen2vl
            elif self.model_name == "internvl":
                self.vg_prefix = "Please provide the bounding box coordinate of the region this sentence describes: <ref>"
                self.vg_suffix = "</ref>"
                self.bbox_normalize_bound = 1000
                self.extract_bbox = self._extract_bbox_internvl
            else:
                raise ValueError(
                    f"Unsupported model type: {model_config['model_type']}"
                )

        self.vqa_suffix = ""

    def generate(self, prompt: str, image_files: str, **kwargs) -> str:
        questions, stop_token_ids = model_example_map[self.model_name](
            prompt, self.processor
        )

        self.sampling_param.stop_token_ids = stop_token_ids
        inputs = [
            {
                "prompt": questions,
                "multi_modal_data": {"image": str(image_files)},
            }
        ]
        outputs = self.model.generate(inputs, sampling_params=self.sampling_param)
        outputs = [output.outputs[0].text for output in outputs]
        return outputs

    def generate_n(
        self, prompt: str, image_files: str, n_samples: int, **kwargs
    ) -> str:
        questions, stop_token_ids = model_example_map[self.model_name](
            prompt, self.processor
        )

        self.sampling_param.stop_token_ids = stop_token_ids
        image_data = Image.open(str(image_files)).convert("RGB")

        inputs = [
            {
                "prompt": questions,
                "multi_modal_data": {"image": image_data},
            }
            for _ in range(n_samples)
        ]

        if len(inputs) > 3:
            # single gpu only support 3 samples, otherwise will out of memory
            total_samples = len(inputs)
            wrap_inputs = []
            for i in range(0, total_samples, 3):
                wrap_inputs.append(inputs[i : i + 3])
        else:
            wrap_inputs = [inputs]

        outputs = []
        for input in wrap_inputs:
            outputs.extend(
                self.model.generate(input, sampling_params=self.sampling_param)
            )
        outputs = [output.outputs[0].text for output in outputs]
        return outputs

    def _extract_bbox_internvl3(self, text):
        # Use regex to extract content inside <box> tags
        box_pattern = re.compile(r'<box>(.*?)</box>', re.DOTALL)
        box_match = box_pattern.search(text)
        
        if box_match:
            box_content = box_match.group(1)
            # Extract coordinates using pattern that matches numeric arrays
            coords_pattern = re.compile(r'\[\[(.*?)\]\]')
            coords_match = coords_pattern.search(box_content)
            
            if coords_match:
                # Extract the numbers from the coordinate string
                coord_values = coords_match.group(1)
                # Split by comma and convert to floats
                try:
                    coords = [float(x.strip()) for x in coord_values.split(',')]
                    if len(coords) == 4:  # Ensure we have 4 coordinates (x1, y1, x2, y2)
                        return [coords]
                except ValueError:
                    pass
        
        # Fallback to the regular InternVL extraction method
        pattern = re.compile(r"\[*\[(.*?),(.*?),(.*?),(.*?)\]\]*")
        coords = pattern.findall(text)
        if len(coords) == 0:
            return None
        
        try:        
            results = [list(map(float, coord)) for coord in coords]
        except Exception as e:
            print(f"Error extracting bounding boxes: {e}")
            results = None
        return results

    def _extract_bbox_qwen2_5vl(self, text):
        # Pattern for json format with bbox_2d
        json_pattern = re.compile(r'```json\n(.*?)\n```', re.DOTALL)
        json_match = json_pattern.search(text)
        
        if json_match:
            try:
                json_data = json.loads(json_match.group(1))
                if isinstance(json_data, list):
                    bboxes = []
                    for item in json_data:
                        if "bbox_2d" in item:
                            bboxes.append(item["bbox_2d"])
                else:
                    if "bbox_2d" in json_data:
                        bboxes = [json_data["bbox_2d"]]
                    else:
                        bboxes = None
                if bboxes:
                    return bboxes
            except json.JSONDecodeError:
                pass
        
        # Fallback to pattern matching similar to qwen2vl format
        pattern = re.compile(
            r"\((\d+(?:\.\d+)?),(\d+(?:\.\d+)?)\),\((\d+(?:\.\d+)?),(\d+(?:\.\d+)?)\)"
        )
        reasoning_pattern = re.compile(
            r"\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]"
        )
        coords = pattern.findall(text)
        if len(coords) == 0:
            reasoning_coords = reasoning_pattern.findall(text)
            if len(reasoning_coords) == 0:
                return None
            return [list(map(float, coord)) for coord in reasoning_coords]
        
        return [list(map(float, coord)) for coord in coords]

    def _extract_bbox_internvl(self, text):
        pattern = re.compile(r"\[*\[(.*?),(.*?),(.*?),(.*?)\]\]*")
        coords = pattern.findall(text)
        if len(coords) == 0:
            return None
        return [list(map(float, coord)) for coord in coords]

    def _extract_bbox_qwen2vl(self, text):
        pattern = re.compile(
            r"\((\d+(?:\.\d+)?),(\d+(?:\.\d+)?)\),\((\d+(?:\.\d+)?),(\d+(?:\.\d+)?)\)"
        )
        reasoning_pattern = re.compile(
            r"\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]"
        )
        coords = pattern.findall(text)
        if len(coords) == 0:
            if self.reasoning_config:
                reasoning_coords = reasoning_pattern.findall(text)
                if len(reasoning_coords) == 0:
                    return None
                return [list(map(float, coord)) for coord in reasoning_coords]
            return None
        # Convert each coordinate tuple from strings to floats
        return [list(map(float, coord)) for coord in coords]


class LMDeployLM(LM):
    def __init__(self, *args, **kwargs):
        self.reasoning_config = kwargs.pop("reasoning_config", None)
        super().__init__(*args, **kwargs)

    def load_everything(self):
        self.model = pipeline(
            self.model_path,
            backend_config=TurbomindEngineConfig(session_len=8192),
            chat_template_config=(
                ChatTemplateConfig.from_json(self.reasoning_config)
                if self.reasoning_config
                else None
            ),
        )
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=self.do_sample,
        )

        if "Qwen2-VL" in self.model_path or "Qwen2VL" in self.model_path:
            self.vg_prefix = "Output the bounding box of the following object in the image. <|object_ref_start|>"
            self.vg_suffix = "<|object_ref_end|>"
            self.bbox_normalize_bound = 1000
            self.extract_bbox = self._extract_bbox_qwen2vl
        elif "Qwen2.5-VL" in self.model_path:
            self.vg_prefix = "Output the bounding box of the following object in the image. <|object_ref_start|>"
            self.vg_suffix = "<|object_ref_end|>"
            self.extract_bbox = self._extract_bbox_qwen2_5vl
            self.bbox_normalize_bound = None
        elif "InternVL3" in self.model_path:
            self.vg_prefix = "Please provide the bounding box coordinate of the region this sentence describes: <ref>"
            self.vg_suffix = "</ref>"
            self.bbox_normalize_bound = 1000
            self.extract_bbox = self._extract_bbox_internvl3
        elif "InternVL" in self.model_path:
            self.vg_prefix = "Please provide the bounding box coordinate of the region this sentence describes: <ref>"
            self.vg_suffix = "</ref>"
            self.bbox_normalize_bound = 1000
            self.extract_bbox = self._extract_bbox_internvl
        elif "llava" not in self.model_path:
            config_path = os.path.join(self.model_path, "config.json")
            model_config = json.load(open(config_path, "r"))
            if (
                "Qwen2-VL" in model_config["_name_or_path"]
                or "qwen2_vl" in model_config["_name_or_path"]
                or "qwen2_vl" in model_config["model_type"]
            ):
                self.vg_prefix = "Output the bounding box of the following object in the image. <|object_ref_start|>"
                self.vg_suffix = "<|object_ref_end|>"
                self.bbox_normalize_bound = 1000
                if self.reasoning_config:
                    self.bbox_normalize_bound = 1000
                self.extract_bbox = self._extract_bbox_qwen2vl
            elif "Qwen2.5-VL" in model_config["_name_or_path"]:
                self.vg_prefix = "Output the bounding box of the following object in the image. <|object_ref_start|>"
                self.vg_suffix = "<|object_ref_end|>"
                self.bbox_normalize_bound = 1000
                self.extract_bbox = self._extract_bbox_qwen2_5vl
            elif "InternVL" in model_config["_name_or_path"]:
                self.vg_prefix = "Please provide the bounding box coordinate of the region this sentence describes: <ref>"
                self.vg_suffix = "</ref>"
                self.bbox_normalize_bound = 1000
                self.extract_bbox = self._extract_bbox_internvl
            else:
                raise ValueError(
                    f"Unsupported model type: {model_config['model_type']}"
                )

        self.vqa_suffix = ""
    
    def _extract_bbox_internvl3(self, text):
        # Use regex to extract content inside <box> tags
        box_pattern = re.compile(r'<box>(.*?)</box>', re.DOTALL)
        box_match = box_pattern.search(text)
        
        if box_match:
            box_content = box_match.group(1)
            # Extract coordinates using pattern that matches numeric arrays
            coords_pattern = re.compile(r'\[\[(.*?)\]\]')
            coords_match = coords_pattern.search(box_content)
            
            if coords_match:
                # Extract the numbers from the coordinate string
                coord_values = coords_match.group(1)
                # Split by comma and convert to floats
                try:
                    coords = [float(x.strip()) for x in coord_values.split(',')]
                    if len(coords) == 4:  # Ensure we have 4 coordinates (x1, y1, x2, y2)
                        return [coords]
                except ValueError:
                    pass
        
        # Fallback to the regular InternVL extraction method
        pattern = re.compile(r"\[*\[(.*?),(.*?),(.*?),(.*?)\]\]*")
        coords = pattern.findall(text)
        if len(coords) == 0:
            return None
        
        try:        
            results = [list(map(float, coord)) for coord in coords]
        except Exception as e:
            print(f"Error extracting bounding boxes: {e}")
            results = None
        return results

    def _extract_bbox_qwen2_5vl(self, text):
        # Pattern for json format with bbox_2d
        json_pattern = re.compile(r'```json\n(.*?)\n```', re.DOTALL)
        json_match = json_pattern.search(text)
        
        if json_match:
            try:
                json_data = json.loads(json_match.group(1))
                bboxes = []
                for item in json_data:
                    if "bbox_2d" in item:
                        bboxes.append(item["bbox_2d"])
                if bboxes:
                    return bboxes
            except json.JSONDecodeError:
                pass
        
        # Fallback to pattern matching similar to qwen2vl format
        pattern = re.compile(
            r"\((\d+(?:\.\d+)?),(\d+(?:\.\d+)?)\),\((\d+(?:\.\d+)?),(\d+(?:\.\d+)?)\)"
        )
        reasoning_pattern = re.compile(
            r"\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]"
        )
        coords = pattern.findall(text)
        if len(coords) == 0:
            reasoning_coords = reasoning_pattern.findall(text)
            if len(reasoning_coords) == 0:
                return None
            return [list(map(float, coord)) for coord in reasoning_coords]
        
        return [list(map(float, coord)) for coord in coords]

    def _extract_bbox_internvl(self, text):
        pattern = re.compile(r"\[*\[(.*?),(.*?),(.*?),(.*?)\]\]*")
        coords = pattern.findall(text)
        if len(coords) == 0:
            return None
        return [list(map(float, coord)) for coord in coords]

    def _extract_bbox_qwen2vl(self, text):
        pattern = re.compile(
            r"\((\d+(?:\.\d+)?),(\d+(?:\.\d+)?)\),\((\d+(?:\.\d+)?),(\d+(?:\.\d+)?)\)"
        )
        reasoning_pattern = re.compile(
            r"\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]"
        )
        coords = pattern.findall(text)
        if len(coords) == 0:
            if self.reasoning_config:
                reasoning_coords = reasoning_pattern.findall(text)
                if len(reasoning_coords) == 0:
                    return None
                return [list(map(float, coord)) for coord in reasoning_coords]
            return None
        # Convert each coordinate tuple from strings to floats
        return [list(map(float, coord)) for coord in coords]

    def generate(self, prompt: List[str], image_files: List[str], **kwargs) -> str:
        if not isinstance(prompt, list):
            prompt = [prompt]
        if not isinstance(image_files, list):
            image_files = [image_files]

        input_message = []
        for text_message, image_file in zip(prompt, image_files):
            if not isinstance(image_file, str):
                image_file = str(image_file)

            if not isinstance(image_file, Image.Image):
                input_message.append((text_message, load_image(image_file)))
            else:
                input_message.append((text_message, image_file))

        response = self.model(
            input_message,
            generation_config=self.generation_config,
        )

        response = [res.text for res in response]
        return response

    def generate_n(
        self, prompt: List[str], image_files: List[str], n_samples: int, **kwargs
    ) -> str:
        if not isinstance(prompt, list):
            prompt = [prompt]
        if not isinstance(image_files, list):
            image_files = [image_files]

        input_message = []
        for text_message, image_file in zip(prompt, image_files):
            if not isinstance(image_file, str):
                image_file = str(image_file)

            if not isinstance(image_file, Image.Image):
                input_message.append((text_message, load_image(image_file)))
            else:
                input_message.append((text_message, image_file))

        n_responses = []
        for i in range(n_samples):
            self.generation_config.random_seed = random.randint(0, 1000000)
            response = self.model(
                input_message,
                generation_config=self.generation_config,
            )
            n_responses.extend(response)
        return [res.text for res in n_responses]

    def generate_until(self, requests):
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = (
            len(requests) // self.batch_size
            if len(requests) % self.batch_size == 0
            else len(requests) // self.batch_size + 1
        )
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            batched_visuals = [
                doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id
            ]
            flatten_visuals = self.flatten(batched_visuals)
            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]

            # Update values from gen_kwargs if present
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(
                        f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}"
                    )
            assert (
                self.batch_size_per_gpu == 1
            ), "Do not support batch_size_per_gpu > 1 for now"

            question_input = []
            for context in contexts:
                question_input.append(context)

            # Apply chat template
            if self.accelerator.is_main_process and doc_id[0] % 100 == 0:
                eval_logger.debug(
                    f"Prompt for doc ID {doc_id[0]}:\n\n{question_input[0]}\n"
                )

            if "max_new_tokens" not in gen_kwargs:
                self.max_new_tokens = gen_kwargs["max_new_tokens"]
            if "temperature" not in gen_kwargs:
                self.temperature = gen_kwargs["temperature"]
            if "top_p" not in gen_kwargs:
                self.top_p = gen_kwargs["top_p"]
            if "num_beams" not in gen_kwargs:
                self.beam_size = gen_kwargs["num_beams"]

            try:
                text_outputs = self.generate(
                    question_input, flatten_visuals, **gen_kwargs
                )
            except Exception as e:
                eval_logger.error(f"Error {e} in generating")
                cont = ""

            if self.accelerator.is_main_process and doc_id[0] % 100 == 0:
                eval_logger.debug(
                    f"Generated text for doc ID {doc_id[0]}:\n\n{text_outputs}\n"
                )

            res.extend(text_outputs)
            self.cache_hook.add_partial(
                "generate_until", (context, gen_kwargs), text_outputs
            )
            pbar.update(1)
        # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res


class LLaVALM(LM):
    def generate_n(
        self, prompt: List[str], image_files: List[str], n_samples: int, **kwargs
    ) -> str:
        n_responses = []
        for i in range(n_samples):
            response = self.generate(prompt, image_files, **kwargs)
            n_responses.extend(response)
        return n_responses

    def generate(self, prompt: List[str], image_files: List[str], **kwargs) -> str:
        if not isinstance(prompt, list):
            prompt = [prompt]
        if not isinstance(image_files, list):
            image_files = [image_files]

        input_batch = []
        image_folder = []

        for idx in range(len(prompt)):
            current_prompt = prompt[idx]
            current_image_file = image_files[idx]
            if getattr(self.model.config, "mm_use_im_start_end", False):
                current_prompt = (
                    DEFAULT_IM_START_TOKEN
                    + DEFAULT_IMAGE_TOKEN
                    + DEFAULT_IM_END_TOKEN
                    + "\n"
                    + current_prompt
                )
            else:
                current_prompt = DEFAULT_IMAGE_TOKEN + "\n" + current_prompt

            conv = conv_templates[self.template_name].copy()
            conv.append_message(conv.roles[0], current_prompt)
            conv.append_message(conv.roles[1], None)
            current_prompt = conv.get_prompt()

            input_ids = (
                tokenizer_image_token(
                    current_prompt,
                    self.tokenizer,
                    IMAGE_TOKEN_INDEX,
                    return_tensors="pt",
                )
                .unsqueeze(0)
                .cuda()
            )
            input_batch.append(input_ids)

            if not isinstance(current_image_file, Image.Image):
                image = Image.open(current_image_file)
            else:
                image = current_image_file

            image_folder.append(image)

        max_length = max(tensor.size(1) for tensor in input_batch)

        final_input_list = [
            torch.cat(
                (
                    torch.zeros(
                        (1, max_length - tensor.size(1)),
                        dtype=tensor.dtype,
                        device=tensor.get_device(),
                    ),
                    tensor,
                ),
                dim=1,
            )
            for tensor in input_batch
        ]
        final_input_tensors = torch.cat(final_input_list, dim=0).to(self.device)
        image_tensor_batch = process_images(
            image_folder, self.image_processor, self.image_aspect_ratio
        )

        with torch.inference_mode() and torch.autocast(
            device_type="cuda", dtype=self.dtype
        ):
            output_ids = self.model.generate(
                final_input_tensors,
                images=image_tensor_batch.to(device=self.device, dtype=self.dtype),
                do_sample=self.do_sample,
                temperature=self.temperature,
                top_p=self.top_p,
                num_beams=self.beam_size,
                max_new_tokens=self.max_new_tokens,
                use_cache=self.use_cache,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        if input_ids[0][0] == output_ids[0][0]:
            input_token_len = final_input_tensors.shape[1]
            outputs = self.tokenizer.batch_decode(
                output_ids[:, input_token_len:], skip_special_tokens=True
            )
        else:
            outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        new_outputs = []
        for output in outputs:
            output = output.strip()
            if "</s>" in output:
                output = output.split("</s>")[0]
                output = output.strip()
            if "<|eot_id|>" in output:
                output = output.split("<|eot_id|>")[0]
                output = output.strip()
            new_outputs.append(output)
        return new_outputs

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests):
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = (
            len(requests) // self.batch_size
            if len(requests) % self.batch_size == 0
            else len(requests) // self.batch_size + 1
        )
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            batched_visuals = [
                doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id
            ]
            flatten_visuals = self.flatten(batched_visuals)
            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]

            # Update values from gen_kwargs if present
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(
                        f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}"
                    )
            assert (
                self.batch_size_per_gpu == 1
            ), "Do not support batch_size_per_gpu > 1 for now"

            question_input = []
            for visual, context in zip(batched_visuals, contexts):
                if DEFAULT_IMAGE_TOKEN not in context:
                    """
                    Three senarios:
                    1. No image, and there for, no image token should be added.
                    2. image token is already specified in the context, so we don't need to add it.
                    3. image token is not specified in the context and there is image inputs, so we need to add it. In this case, we add the image token at the beginning of the context and add a new line.
                    """
                    image_tokens = (
                        [DEFAULT_IMAGE_TOKEN] * len(visual)
                        if isinstance(visual, list)
                        else [DEFAULT_IMAGE_TOKEN]
                    )
                    image_tokens = " ".join(image_tokens)
                    question = image_tokens + "\n" + context
                else:
                    question = context
                question_input.append(question)

            if len(flatten_visuals) == 0:
                for context in contexts:
                    question_input.append(context)

            # Apply chat template
            if self.accelerator.is_main_process and doc_id[0] % 100 == 0:
                eval_logger.debug(
                    f"Prompt for doc ID {doc_id[0]}:\n\n{question_input[0]}\n"
                )

            if "max_new_tokens" not in gen_kwargs:
                self.max_new_tokens = gen_kwargs["max_new_tokens"]
            if "temperature" not in gen_kwargs:
                self.temperature = gen_kwargs["temperature"]
            if "top_p" not in gen_kwargs:
                self.top_p = gen_kwargs["top_p"]
            if "num_beams" not in gen_kwargs:
                self.beam_size = gen_kwargs["num_beams"]

            try:
                text_outputs = self.generate(
                    question_input, flatten_visuals, **gen_kwargs
                )
            except Exception as e:
                eval_logger.error(f"Error {e} in generating")
                cont = ""

            if self.accelerator.is_main_process and doc_id[0] % 100 == 0:
                eval_logger.debug(
                    f"Generated text for doc ID {doc_id[0]}:\n\n{text_outputs}\n"
                )

            res.extend(text_outputs)
            self.cache_hook.add_partial(
                "generate_until", (context, gen_kwargs), text_outputs
            )
            pbar.update(1)
        # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res


class GeoChatLM(LLaVALM):
    def load_everything(self):
        model_name = "geochat"
        model_base = None
        self.tokenizer, self.model, self.image_processor, self.coontext_len = (
            load_pretrained_model(self.model_path, model_base, model_name)
        )
        self.model.to(self.device)
        self.template_name = "llava_v1"

        self.bbox_normalize_bound = 100
        self.image_aspect_ratio = "pad"

        self.vg_prefix = "{grounding}"

    def extract_bbox(self, text):
        pattern = re.compile(r"{<([\d.]+)><([\d.]+)><([\d.]+)><([\d.]+)>\|<\d+>}")
        coords = pattern.findall(text)
        if len(coords) == 0:
            return None
        # remove the last rotation coordinate
        coords = [coord for coord in coords]
        return [list(map(float, coord)) for coord in coords]


class SkysenseGPTLM(GeoChatLM):
    def load_everything(self):
        model_name = "geochat"  # since SkysenseGPT is exactly the same as GeoChat except the data that is used to train it
        model_base = None
        self.tokenizer, self.model, self.image_processor, self.coontext_len = (
            load_skysensegpt_pretrained_model(self.model_path, model_base, model_name)
        )
        self.model.to(self.device)
        self.template_name = "llava_v1"

        self.bbox_normalize_bound = 100
        self.image_aspect_ratio = "pad"

        # self.vg_prefix = "[grounding]"
        self.vg_prefix = "[detection]"

    def extract_bbox(self, text):
        pattern = re.compile(r"\(?{<([\d.]+)><([\d.]+)><([\d.]+)><([\d.]+)>\|<\d+>}\)?")
        coords = pattern.findall(text)
        if len(coords) == 0:
            return None
        return [list(map(float, coord)) for coord in coords]


class VHM(LLaVALM):
    def load_everything(self):
        self.model_wrapper = VHM_Loader("vhm", "FitzPC/vhm_7B")
        self.tokenizer = self.model_wrapper.tokenizer
        self.image_processor = self.model_wrapper.image_processor
        self.model = self.model_wrapper.model
        self.model.to(self.device)
        self.template_name = "v1"

        self.cls_prefix = "{CLS}"
        self.vqa_prefix = "{VQA}"
        self.vg_prefix = "{VG}"

        self.bbox_normalize_bound = 1000
        self.image_aspect_ratio = None

    def extract_bbox(self, text):
        pattern = re.compile(r"\[\[(\d+)[,\s]+(\d+)[,\s]+(\d+)[,\s]+(\d+)\]\]")
        answer_numbers = pattern.findall(text)
        return [list(map(float, number)) for number in answer_numbers]


class LHRSLM(LLaVALM):
    def load_everything(self):
        self.model = LHRSBotLlamaForCausalLM.from_pretrained(
            "NousResearch/Meta-Llama-3-8B-Instruct", torch_dtype=torch.float16
        ).to(self.device)
        self.model.custom_load_state_dict(self.model_path, strict=False)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "NousResearch/Meta-Llama-3-8B-Instruct"
        )
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.image_processor = self.model.get_image_processor()
        self.template_name = "llama3"

        self.bbox_normalize_bound = 1
        self.image_aspect_ratio = None
        self.box_start_idientifier = "<bbox>"
        self.box_end_idientifier = "</bbox>"

        self.vg_prefix = "[DET]"
        self.cls_prefix = "[CLS]"
        self.vqa_prefix = "[CONSIZE]"

    def extract_bbox(self, text):
        pattern = re.compile(r"<bbox>\[([\d.,]+)\]</bbox>")
        coords = pattern.findall(text)
        if len(coords) == 0:
            return None
        return [list(map(float, coord.split(","))) for coord in coords]


def hash_args(attr, args):
    dat = json.dumps([attr] + list(args))
    return hashlib.sha256(dat.encode("utf-8")).hexdigest()
