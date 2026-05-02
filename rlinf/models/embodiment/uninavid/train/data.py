# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ruff: noqa: C408, C416, UP006

import copy
import importlib
import json
import math
import os
import pickle
import random
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import transformers
from omegaconf import ListConfig
from PIL import Image
from torch.utils.data import Dataset

from rlinf.models.embodiment.uninavid.constants import (
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
)
from rlinf.models.embodiment.uninavid.train.preprocess import (
    preprocess,
    preprocess_multimodal,
    preprocess_multimodal_movie,
)

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def _resolve_data_path(data_paths):
    if isinstance(data_paths, (list, tuple, ListConfig)):
        return str(data_paths[0])
    return str(data_paths)


def _load_decord_for_raw_video():
    try:
        decord = importlib.import_module("decord")
    except (ImportError, ModuleNotFoundError) as exc:
        raise ModuleNotFoundError(
            "decord is required for Uni-NaVid raw video loading. "
            "Install decord or use precomputed .pkl video features."
        ) from exc
    return decord.VideoReader, decord.cpu


def duplicate_with_probability(lst, n):
    result = []
    for index, num in enumerate(lst):
        result.append(num)
        if random.random() < n or (index == len(lst) - 1 and random.random() < 2 * n):
            result.append(num)
    return result


def random_color_jitter(
    video,
    brightness_range=(0.8, 1.2),
    contrast_range=(0.8, 1.2),
    saturation_range=(0.8, 1.2),
    hue_range=(-0.1, 0.1),
    prob=0.05,
):
    def adjust_brightness(image, factor):
        return np.clip(image * factor, 0, 255).astype(np.uint8)

    def adjust_contrast(image, factor):
        mean = np.mean(image, axis=(0, 1), keepdims=True)
        return np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)

    def adjust_saturation(image, factor):
        grayscale = np.mean(image, axis=2, keepdims=True)
        return np.clip((image - grayscale) * factor + grayscale, 0, 255).astype(
            np.uint8
        )

    n = video.shape[0]

    augmented_video = np.copy(video)

    for i in range(n):
        if np.random.rand() < prob:
            brightness_factor = np.random.uniform(*brightness_range)
            augmented_video[i] = adjust_brightness(
                augmented_video[i], brightness_factor
            )

        if np.random.rand() < prob:
            contrast_factor = np.random.uniform(*contrast_range)
            augmented_video[i] = adjust_contrast(augmented_video[i], contrast_factor)

        if np.random.rand() < prob:
            saturation_factor = np.random.uniform(*saturation_range)
            augmented_video[i] = adjust_saturation(
                augmented_video[i], saturation_factor
            )

    return augmented_video


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    video_folder: Optional[str] = field(default=None)
    video_fps: Optional[int] = field(default=1)
    video_token: Optional[int] = field(default=2)
    image_aspect_ratio: str = "square"
    image_grid_pinpoints: Optional[str] = field(default=None)
    input_prompt: Optional[str] = field(default=None)
    refine_prompt: Optional[bool] = field(default=False)


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
    ):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(
                sum(len(conv["value"].split()) for conv in sample["conversations"])
                + img_tokens
            )
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(
                len(conv["value"].split()) for conv in sample["conversations"]
            )
            cur_len = (
                cur_len if ("image" in sample) or ("video" in sample) else -cur_len
            )
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        attempt, max_attempt = 0, 10
        while attempt < max_attempt:
            try:
                sources = self.list_data_dict[i]
                suffix = None
                if isinstance(i, int):
                    sources = [sources]
                assert len(sources) == 1, (
                    "Don't know why it is wrapped to a list"
                )  # FIXME
                if "image" in sources[0]:
                    image_file = self.list_data_dict[i]["image"]
                    image_folder = self.data_args.image_folder
                    processor = self.data_args.image_processor

                    # convert image type for OCR VQA dataset
                    if image_file is not None:
                        if "ocr" in image_file:
                            if not os.path.exists(
                                os.path.join(image_folder, image_file)
                            ):
                                image_file = image_file.replace(".jpg", ".png")

                        # convert image for VG dataset
                        # elif 'VG_100K' in image_file:
                        # image_file = image_file.replace('VG_100K_2', 'images')
                        # image_file = image_file.replace('VG_100K', 'images')

                    image = Image.open(os.path.join(image_folder, image_file)).convert(
                        "RGB"
                    )
                    if self.data_args.image_aspect_ratio == "pad":

                        def expand2square(pil_img, background_color):
                            width, height = pil_img.size
                            if width == height:
                                return pil_img
                            elif width > height:
                                result = Image.new(
                                    pil_img.mode, (width, width), background_color
                                )
                                result.paste(pil_img, (0, (width - height) // 2))
                                return result
                            else:
                                result = Image.new(
                                    pil_img.mode, (height, height), background_color
                                )
                                result.paste(pil_img, ((height - width) // 2, 0))
                                return result

                        image = expand2square(
                            image, tuple(int(x * 255) for x in processor.image_mean)
                        )
                        image = processor.preprocess(image, return_tensors="pt")[
                            "pixel_values"
                        ][0]
                    else:
                        image = processor.preprocess(image, return_tensors="pt")[
                            "pixel_values"
                        ][0]
                    sources = preprocess_multimodal(
                        copy.deepcopy([e["conversations"] for e in sources]),
                        self.data_args,
                    )
                elif "video" in sources[0]:
                    video_file = self.list_data_dict[i]["video"]
                    video_folder = self.data_args.video_folder
                    video_file = os.path.join(video_folder, video_file)
                    suffix = video_file.split(".")[-1]
                    if not os.path.exists(video_file):
                        print("File {} not exist!".format(video_file))

                    if suffix == "pkl":
                        video_info = pickle.load(open(video_file, "rb"))
                        image = torch.from_numpy(video_info["feats"][:, 1:])
                        input_prompt = video_info["inputs"].replace("...", "")
                        # replace the default image token with multiple tokens
                        input_prompt = input_prompt.replace(
                            DEFAULT_IMAGE_TOKEN,
                            DEFAULT_IMAGE_TOKEN * self.data_args.video_token,
                        )
                        sources, query_prompt = preprocess_multimodal_movie(
                            copy.deepcopy([e["conversations"] for e in sources]),
                            self.data_args,
                            input_prompt,
                        )
                    else:
                        VideoReader, cpu = _load_decord_for_raw_video()
                        vr = VideoReader(video_file, ctx=cpu(0))
                        sample_fps = round(vr.get_avg_fps() / self.data_args.video_fps)
                        frame_idx = [i for i in range(0, len(vr), sample_fps)]
                        video = vr.get_batch(frame_idx).asnumpy()
                        if (
                            "NAV_ID" in self.list_data_dict[i]["id"]
                            and len(frame_idx) > 1
                        ):  # TODO: temp fix for nav
                            assert len(video) > 1

                            last_frame_index = len(video) - 1
                            max_drop_frames = math.ceil(0.1 * (len(video) - 1))

                            num_frames_to_sample = (
                                len(video) - 1 - random.randint(0, max_drop_frames)
                            )
                            assert num_frames_to_sample >= 0
                            sampled_frame_indices = sorted(
                                random.sample(
                                    range(len(video) - 1), num_frames_to_sample
                                )
                            )

                            sampled_frame_indices.append(last_frame_index)

                            sampled_frame_indices = duplicate_with_probability(
                                sampled_frame_indices, 0.03
                            )

                            video = video[sampled_frame_indices]
                            video = random_color_jitter(video)

                        processor = self.data_args.image_processor
                        image = processor.preprocess(video, return_tensors="pt")[
                            "pixel_values"
                        ]
                        sources = preprocess_multimodal(
                            copy.deepcopy([e["conversations"] for e in sources]),
                            self.data_args,
                        )
                else:
                    sources = copy.deepcopy([e["conversations"] for e in sources])

                break
            except (ImportError, ModuleNotFoundError) as exc:
                if "decord" in str(exc) and "raw video loading" in str(exc):
                    raise
                attempt += 1
                print(f"Error in loading {i}, retrying...")
                i = random.randint(0, len(self.list_data_dict) - 1)
            except Exception:
                attempt += 1
                print(f"Error in loading {i}, retrying...")
                i = random.randint(0, len(self.list_data_dict) - 1)

        has_image = ("image" in self.list_data_dict[i]) or (
            "video" in self.list_data_dict[i]
        )
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=has_image,
            prompt=self.data_args.input_prompt,
            refine_prompt=self.data_args.refine_prompt,
            video_or_not="video" in self.list_data_dict[i],
        )

        if "prompt" in data_dict:
            prompt = data_dict["prompt"]
        else:
            prompt = None

        if suffix == "pkl":
            prompt = [query_prompt]

        if isinstance(i, int):
            data_dict = dict(
                input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0]
            )

        # image exist in the data
        if "image" in self.list_data_dict[i]:
            data_dict["image"] = image
        elif "video" in self.list_data_dict[i]:
            data_dict["image"] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict["image"] = torch.zeros(3, crop_size["height"], crop_size["width"])

        # prompt exist in the data
        if prompt is not None:
            data_dict["prompt"] = prompt

        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if "image" in instances[0]:
            images = [instance["image"] for instance in instances]
            if (
                all(x is not None and x.shape == images[0].shape for x in images)
                and len(images) > 1
            ):
                batch["images"] = torch.stack(images)
            else:
                batch["images"] = images

        if "prompt" in instances[0]:
            batch["prompts"] = [instance["prompt"] for instance in instances]

        return batch


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(
        tokenizer=tokenizer, data_path=data_args.data_path, data_args=data_args
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


def build_uninavid_sft_dataloader(
    cfg, world_size: int, rank: int, data_paths, eval_dataset: bool = False
):
    import torch.distributed as dist
    from torch.utils.data import DataLoader, DistributedSampler
    from transformers import AutoTokenizer, CLIPImageProcessor

    data_path = _resolve_data_path(data_paths)
    model_cfg = cfg.actor.model
    data_cfg = cfg.data

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.model_path,
        model_max_length=int(model_cfg.get("model_max_length", 2048)),
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    image_processor = CLIPImageProcessor.from_pretrained(model_cfg.image_processor)

    from rlinf.models.embodiment.uninavid import conversation as conversation_lib

    version = str(model_cfg.get("version", "imgsp_v1"))
    conversation_lib.default_conversation = conversation_lib.conv_templates.get(
        version,
        conversation_lib.conv_templates["vicuna_v1"],
    )

    data_args = DataArguments(
        data_path=str(data_path),
        lazy_preprocess=bool(data_cfg.get("lazy_preprocess", True)),
        is_multimodal=True,
        image_folder=str(data_cfg.image_folder),
        video_folder=str(data_cfg.video_folder),
        video_fps=int(model_cfg.get("video_fps", data_cfg.get("video_fps", 1))),
        image_aspect_ratio=str(model_cfg.get("image_aspect_ratio", "pad")),
    )
    data_args.mm_use_im_start_end = bool(model_cfg.get("mm_use_im_start_end", False))
    data_args.image_processor = image_processor

    dataset = LazySupervisedDataset(
        tokenizer=tokenizer,
        data_path=str(data_path),
        data_args=data_args,
    )

    if dist.is_available() and dist.is_initialized():
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=bool(data_cfg.get("shuffle", True)),
            seed=int(data_cfg.get("seed", cfg.actor.get("seed", 0))),
            drop_last=True,
        )
    else:
        sampler = None

    batch_size = (
        cfg.actor.eval_batch_size if eval_dataset else cfg.actor.micro_batch_size
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None and bool(data_cfg.get("shuffle", True))),
        num_workers=int(data_cfg.get("num_workers", 4)),
        drop_last=True,
        collate_fn=DataCollatorForSupervisedDataset(tokenizer=tokenizer),
    )
    return data_loader, {"dataset_name": "uninavid", "num_samples": len(dataset)}
