# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Word-level Knowledge Distillation Trainer.

Student learns teacher's per-token distribution via forward KL on SFT data.
Based on SFTTrainer with an additional teacher model for KD loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any
from accelerate import logging
from datasets import Dataset, IterableDataset
from transformers import (
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
)
from transformers.utils import is_peft_available

from ..trainer.sft_trainer import SFTTrainer
from ..trainer.sft_config import SFTConfig
from ..trainer.utils import disable_dropout_in_model, empty_cache
from ..models import prepare_deepspeed

if is_peft_available():
    from peft import PeftConfig

logger = logging.get_logger(__name__)


class KDTrainer(SFTTrainer):
    """
    Knowledge Distillation Trainer that extends SFTTrainer.
    
    The student model is trained with a combination of:
    - Standard SFT loss (cross-entropy with ground truth labels)
    - Forward KL divergence loss (student learns teacher's per-token distribution)
    
    loss = alpha * kd_loss + (1 - alpha) * sft_loss
    
    Args:
        model: Student model to train.
        teacher_model: Teacher model (frozen) for providing soft targets.
        kd_alpha: Weight for KD loss vs SFT loss. Default 0.5.
        kd_temperature: Temperature for softening distributions. Default 1.0.
        disable_teacher_dropout: Whether to disable dropout in teacher. Default True.
        All other args are passed to SFTTrainer.
    """

    def __init__(
        self,
        model: str | PreTrainedModel,
        teacher_model: str | PreTrainedModel,
        args: SFTConfig | None = None,
        kd_alpha: float = 0.5,
        kd_temperature: float = 1.0,
        disable_teacher_dropout: bool = True,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset | IterableDataset | dict[str, Dataset | IterableDataset] | None = None,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None),
        peft_config: "PeftConfig | None" = None,
        **kwargs,
    ):
        # Initialize the SFTTrainer (handles student model, dataset, etc.)
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
            **kwargs,
        )

        # Load teacher model if string
        if isinstance(teacher_model, str):
            teacher_model = AutoModelForCausalLM.from_pretrained(
                teacher_model,
                torch_dtype=self.model.dtype if hasattr(self.model, 'dtype') else torch.bfloat16,
                attn_implementation=getattr(args, 'attn_implementation', None),
            )

        # Disable dropout in teacher
        if disable_teacher_dropout:
            disable_dropout_in_model(teacher_model)

        # Prepare teacher model (DeepSpeed or accelerate)
        if self.is_deepspeed_enabled:
            self.teacher_model = prepare_deepspeed(teacher_model, self.accelerator)
        else:
            self.teacher_model = self.accelerator.prepare_model(teacher_model, evaluation_mode=True)

        self.kd_alpha = kd_alpha
        self.kd_temperature = kd_temperature

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ):
        """
        Compute combined SFT + KD loss.
        
        loss = alpha * kd_loss + (1 - alpha) * sft_loss
        """
        mode = "train" if self.model.training else "eval"

        # Get labels before they might be modified
        labels = inputs.get("labels")
        if labels is None:
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

        # Compute student forward pass (via parent's compute_loss for SFT loss + metrics)
        inputs["use_cache"] = False
        sft_loss, student_outputs = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )

        # Compute teacher forward pass
        self.teacher_model.eval()
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=inputs.get("input_ids"),
                attention_mask=inputs.get("attention_mask"),
                use_cache=False,
            )

        # Compute word-level KD loss (forward KL)
        student_logits = student_outputs.logits
        teacher_logits = teacher_outputs.logits

        # Shift for autoregressive: predict next token
        shift_student_logits = student_logits[..., :-1, :].contiguous()
        shift_teacher_logits = teacher_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Mask: only compute KD loss where we have valid labels
        mask = (shift_labels != -100)

        if mask.any():
            # Apply temperature
            student_log_probs = F.log_softmax(shift_student_logits / self.kd_temperature, dim=-1)
            teacher_probs = F.softmax(shift_teacher_logits / self.kd_temperature, dim=-1)

            # Forward KL: KL(teacher || student) = sum(teacher * (log(teacher) - log(student)))
            kd_loss_per_token = F.kl_div(
                student_log_probs, teacher_probs, reduction="none", log_target=False
            ).sum(dim=-1)  # (batch_size, seq_len)

            # Apply mask and average
            kd_loss = (kd_loss_per_token * mask).sum() / mask.sum()

            # Scale by temperature^2 (standard KD practice)
            kd_loss = kd_loss * (self.kd_temperature ** 2)
        else:
            kd_loss = torch.tensor(0.0, device=sft_loss.device)

        # Combined loss
        loss = self.kd_alpha * kd_loss + (1 - self.kd_alpha) * sft_loss

        # Log metrics
        self._metrics[mode]["kd_loss"].append(self.accelerator.gather_for_metrics(kd_loss.detach()).mean().item())
        self._metrics[mode]["sft_loss"].append(self.accelerator.gather_for_metrics(sft_loss.detach()).mean().item())

        # Clean up
        empty_cache()

        if return_outputs:
            return (loss, student_outputs)
        return loss
