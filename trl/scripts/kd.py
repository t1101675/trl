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
Word-level Knowledge Distillation script.

Student learns teacher's per-token distribution via forward KL on SFT data.

Example:
```
torchrun --nproc_per_node 8 trl/scripts/kd.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --teacher_model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --dataset_name dataset/llm_rl/OpenR1-Math-220k \
    --kd_alpha 0.5 \
    --kd_temperature 1.0 \
    --learning_rate 5e-6 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --output_dir results/kd/qwen2.5-1.5B-from-7B \
    --deepspeed configs/ds_zero1_bf16.json
```
"""

import os
import time
from dataclasses import dataclass, field

from accelerate import logging
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint

from trl import (
    DatasetMixtureConfig,
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    TrlParser,
    get_dataset,
    get_peft_config,
)
from trl.experimental.kd_trainer import KDTrainer

logger = logging.get_logger(__name__)


class SaveStep0CallBack(TrainerCallback):
    def __init__(self, trainer, trial=None):
        self.trainer = trainer
        self.trial = trial

    def on_train_begin(self, args, state, control, **kwargs):
        self.trainer._save_checkpoint(self.trainer.model_wrapped, self.trial)


class SaveTimeLimitCallBack(TrainerCallback):
    def __init__(self, trainer, start_time, time_limit, trial=None):
        self.trainer = trainer
        self.trial = trial
        self.start_time = start_time
        self.time_limit = time_limit

    def on_step_end(self, args, state, control, **kwargs):
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        if elapsed_time > self.time_limit:
            control.should_save = True
            self.start_time = current_time


@dataclass
class TeacherArguments:
    teacher_model_name_or_path: str = field(
        default=None, metadata={"help": "Path to the teacher model."}
    )
    teacher_model_revision: str = field(
        default="main", metadata={"help": "Teacher model revision."}
    )
    teacher_trust_remote_code: bool = field(
        default=False, metadata={"help": "Whether to trust remote code for teacher."}
    )
    teacher_attn_implementation: str = field(
        default=None, metadata={"help": "Attention implementation for teacher."}
    )
    teacher_dtype: str = field(
        default=None, metadata={"help": "Data type for teacher model."}
    )


@dataclass
class KDArguments:
    kd_alpha: float = field(
        default=0.5,
        metadata={"help": "Weight for KD loss. loss = alpha * kd_loss + (1 - alpha) * sft_loss"},
    )
    kd_temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature for softening distributions in KD."},
    )
    time_limit: int = field(
        default=110 * 60,
        metadata={"help": "Time limit in seconds before forcing a checkpoint save."},
    )


def main(
    teacher_args: TeacherArguments,
    kd_args: KDArguments,
    script_args: ScriptArguments,
    training_args: SFTConfig,
    model_args: ModelConfig,
    dataset_args: DatasetMixtureConfig,
):
    start_time = time.time()
    if os.environ.get("ACCELERATE_GRADIENT_ACCUMULATION_STEPS") == "auto":
        os.environ["ACCELERATE_GRADIENT_ACCUMULATION_STEPS"] = str(training_args.gradient_accumulation_steps)

    # Load student model
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        dtype=model_args.dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)

    # Load teacher model
    teacher_model_kwargs = dict(
        revision=teacher_args.teacher_model_revision,
        trust_remote_code=teacher_args.teacher_trust_remote_code,
        attn_implementation=teacher_args.teacher_attn_implementation,
        dtype=teacher_args.teacher_dtype,
    )
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_args.teacher_model_name_or_path, **teacher_model_kwargs
    )

    # Load dataset
    if dataset_args.datasets and script_args.dataset_name:
        logger.warning(
            "Both `datasets` and `dataset_name` are provided. `datasets` will be used."
        )
        dataset = get_dataset(dataset_args)
    elif dataset_args.datasets and not script_args.dataset_name:
        dataset = get_dataset(dataset_args)
    elif not dataset_args.datasets and script_args.dataset_name:
        dataset = load_dataset(
            script_args.dataset_name, name=script_args.dataset_config, streaming=script_args.dataset_streaming
        )
    else:
        raise ValueError("Either `datasets` or `dataset_name` must be provided.")

    # Initialize KD trainer
    trainer = KDTrainer(
        model=model,
        teacher_model=teacher_model,
        args=training_args,
        kd_alpha=kd_args.kd_alpha,
        kd_temperature=kd_args.kd_temperature,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
    )

    trainer.add_callback(SaveStep0CallBack(trainer))
    trainer.add_callback(SaveTimeLimitCallBack(trainer, start_time, kd_args.time_limit))

    try:
        resume_from_checkpoint = eval(training_args.resume_from_checkpoint)
    except Exception:
        pass
    if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
        resume_from_checkpoint = get_last_checkpoint(training_args.output_dir)

    if resume_from_checkpoint is not None:
        trainer.accelerator.print(f"Resuming training from checkpoint: {resume_from_checkpoint}")

    # Train
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    trainer.accelerator.print("✅ Training completed.")

    # Save
    trainer.save_model(training_args.output_dir)
    trainer.accelerator.print(f"💾 Model saved to {training_args.output_dir}.")


def make_parser(subparsers=None):
    dataclass_types = (
        TeacherArguments, KDArguments, ScriptArguments, SFTConfig,
        ModelConfig, DatasetMixtureConfig,
    )
    if subparsers is not None:
        parser = subparsers.add_parser("kd", help="Run word-level KD training", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    (
        teacher_args, kd_args, script_args, training_args,
        model_args, dataset_args, additional_args,
    ) = parser.parse_args_and_config(return_remaining_strings=True)
    print(additional_args)
    main(teacher_args, kd_args, script_args, training_args, model_args, dataset_args)
