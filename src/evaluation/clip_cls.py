import logging
from typing import List

import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm

from ..custom_trainer.PumpkinTrainer.hook.eval_hook import EvalHook
from ..custom_trainer.PumpkinTrainer.hook.logger_hook import LoggerHook
from ..dataset.prompt_class import CLIP_TEMPLATE
from ..custom_trainer.PumpkinTrainer.utils import is_main_process, barrier

logger = logging.getLogger("train")


def zero_shot_classifier(
    model,
    classnames,
    templates,
    tokenizer,
    device,
):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template(classname) for template in templates]  # format with class
            texts = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)  # tokenize
            class_embeddings = model.get_text_features(**texts)
            class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [
        float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        for k in topk
    ]


class CLIPClsEvalHook(EvalHook):
    func_mapping = {
        "perplexity": "perplexity_eval_func",
        "clip_cls": "clip_cls_eval_func",
    }

    def __init__(self, period: int, evaluators: List[str], tokenizer):
        super().__init__(period, evaluators)
        self.tokenizer = tokenizer

    def clip_cls_eval_func(self):
        logger.info("Start Zero-shot CLIP classification")
        self.trainer.model.eval()
        for dataset_name in self.trainer.eval_data_loader:
            logger.info(f"Evaluating {dataset_name}")
            templates = CLIP_TEMPLATE[dataset_name.lower()]
            classifier = zero_shot_classifier(
                self.trainer.model,
                self.trainer.eval_data_loader[dataset_name]["classes_name"],
                templates,
                self.tokenizer,
                self.trainer.device,
            )

            top1, top5, n = 0, 0, 0
            for tup in tqdm(
                self.trainer.eval_data_loader[dataset_name]["dataloader"],
                desc="Evaluating",
            ):
                image, label = tup
                image = image.to(self.trainer.device)
                label = label.to(self.trainer.device)

                with (
                    torch.autocast(
                        device_type=self.trainer.autocast_type,
                        enabled=self.trainer._enable_amp,
                        dtype=self.trainer.dtype,
                    )
                    and torch.inference_mode()
                ):
                    image_features = self.trainer.model.get_image_features(
                        pixel_values=image
                    )
                    image_features = F.normalize(image_features, dim=-1)
                    logits = 100 * image_features @ classifier

                acc1, acc5 = accuracy(logits, label, topk=(1, 5))
                top1 += acc1
                top5 += acc5
                n += image.size(0)

            top1 = top1 / n
            top5 = top5 / n

            if is_main_process():
                logger.info(
                    f"Dataset: {dataset_name}, Top-1 accuracy: {top1:.4f}, Top-5 accuracy: {top5:.4f}"
                )
                for hook in self.trainer._hooks:
                    if isinstance(hook, LoggerHook):
                        hook._tb_writer.add_scalar(f"eval/clip_cls_{dataset_name}_top1", top1)
                        hook._tb_writer.add_scalar(f"eval/clip_cls_{dataset_name}_top5", top5)
                        if hook.wandb:
                            wandb.log(
                                {
                                    f"eval/clip_cls_{dataset_name}_top1": top1,
                                    f"eval/clip_cls_{dataset_name}_top5": top5,
                                }
                            )
                        break
        barrier()   



class EpochCLIPClsEvalHook(CLIPClsEvalHook):
    def after_epoch(self):
        if self.every_n_epochs(self._period) or self.is_last_epoch():
            self._eval_func()


class IterCLIPClsEvalHook(CLIPClsEvalHook):
    def after_iter(self):
        if self.every_n_iters(self._period) or self.is_last_iter():
            self._eval_func()
