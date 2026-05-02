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
# ruff: noqa: C408, UP006, RUF013

from __future__ import annotations

import copy
import random
from typing import TYPE_CHECKING, Dict, Sequence

import torch
import transformers

from rlinf.models.embodiment.uninavid import conversation as conversation_lib
from rlinf.models.embodiment.uninavid.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IAMGE_SEPARATOR,
    IGNORE_INDEX,
    IMAGE_END_TOKEN,
    IMAGE_START_TOKEN,
    NAVIGATION_IDENTIFIER,
    NAVIGATION_SPECIAL_TOKEN,
    VIDEO_END_SPECIAL_TOKEN,
    VIDEO_START_SPECIAL_TOKEN,
)
from rlinf.models.embodiment.uninavid.mm_utils import tokenizer_image_token

if TYPE_CHECKING:
    from rlinf.models.embodiment.uninavid.train.data import DataArguments


def _tokenize_fn(
    strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx + 2 : cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = "unknown"
        sentence["value"] = (
            BEGIN_SIGNAL + from_str + ": " + sentence["value"] + END_SIGNAL
        )
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments,
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence["value"]:
                sentence["value"] = (
                    sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
                )
                sentence["value"] = DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]
                sentence["value"] = sentence["value"].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence["value"] = sentence["value"].replace(
                        DEFAULT_IMAGE_TOKEN,
                        "<Image>" + DEFAULT_IMAGE_TOKEN + "</Image>",
                    )
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = (
                    DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                )
            sentence["value"] = sentence["value"].replace(
                DEFAULT_IMAGE_TOKEN, replace_token
            )

    return sources


def preprocess_multimodal_movie(
    sources: Sequence[str], data_args: DataArguments, video_inputs: str
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence["value"]:
                prompt = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
            replace_token = video_inputs
            if data_args.mm_use_im_start_end:
                replace_token = (
                    DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                )
            sentence["value"] = sentence["value"].replace(
                DEFAULT_IMAGE_TOKEN, replace_token
            )

    return sources, prompt


def preprocess_llama_2(
    sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack(
            [
                tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
                for prompt in conversations
            ],
            dim=0,
        )
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack(
            [
                tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
                for prompt in conversations
            ],
            dim=0,
        )
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_imgsp_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    img_token: str = "<image>",
    refine_prompt: bool = False,
    video_or_not: bool = False,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    guided_prompt = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        img_in_text = False
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"

            # add guided prompt
            if role == conv.roles[0]:
                guided_sent = (
                    sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").replace("\n", "")
                )
                if refine_prompt:
                    # only keep the useful part of the prompt
                    if "\n" in guided_sent:
                        for _sent in guided_sent.split("\n"):
                            if "?" in _sent:
                                guided_sent = _sent
                                break
                guided_prompt.append(guided_sent)
            # check if image token in text
            if img_token in sentence["value"]:
                img_in_text = True
            # add image token to all sentence if multimoal input
            if (
                role == conv.roles[0]
                and img_in_text
                and img_token not in sentence["value"]
            ):
                # randomly add image token to the beginning or end of the sentence
                if random.randint(0, 1) == 0:
                    img_conv = img_token + "\n" + sentence["value"]
                else:
                    img_conv = sentence["value"] + "\n" + img_token

                conv.append_message(role, img_conv)
            else:
                conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    image_start_special_token = tokenizer(
        IMAGE_START_TOKEN, return_tensors="pt"
    ).input_ids[0][1:]
    image_end_special_token = tokenizer(IMAGE_END_TOKEN, return_tensors="pt").input_ids[
        0
    ][1:]
    video_start_special_token = tokenizer(
        VIDEO_START_SPECIAL_TOKEN, return_tensors="pt"
    ).input_ids[0][1:]
    video_end_special_token = tokenizer(
        VIDEO_END_SPECIAL_TOKEN, return_tensors="pt"
    ).input_ids[0][1:]
    navigation_special_token = tokenizer(
        NAVIGATION_SPECIAL_TOKEN, return_tensors="pt"
    ).input_ids[0][1:]
    image_seperator = tokenizer(IAMGE_SEPARATOR, return_tensors="pt").input_ids[0][1:]

    # Tokenize conversations
    if has_image:
        # input_ids_ = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
        new_list_all = []
        for prompt in conversations:
            token_prompt = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
            indices_to_replace = torch.where(token_prompt == -200)[0]
            new_list = []
            while indices_to_replace.numel() > 0:
                idx = indices_to_replace[0]
                if video_or_not:
                    if NAVIGATION_IDENTIFIER in prompt:  # navigation data
                        new_list.append(token_prompt[:idx])
                        new_list.append(video_start_special_token)
                        new_list.append(image_seperator)
                        new_list.append(token_prompt[idx : idx + 1])
                        new_list.append(video_end_special_token)
                        new_list.append(image_start_special_token)
                        new_list.append(image_end_special_token)
                        new_list.append(navigation_special_token)
                        token_prompt = token_prompt[idx + 1 :]
                    else:  # video
                        new_list.append(token_prompt[:idx])
                        new_list.append(video_start_special_token)
                        new_list.append(image_seperator)
                        new_list.append(token_prompt[idx : idx + 1])
                        new_list.append(video_end_special_token)
                        token_prompt = token_prompt[idx + 1 :]

                else:  # image only
                    new_list.append(token_prompt[:idx])
                    new_list.append(image_start_special_token)
                    new_list.append(token_prompt[idx : idx + 1])
                    new_list.append(image_end_special_token)
                    token_prompt = token_prompt[idx + 1 :]
                indices_to_replace = torch.where(token_prompt == -200)[0]
            if token_prompt.numel() > 0:
                new_list.append(token_prompt)
            new_list_all.append(torch.cat(new_list, dim=0))
        input_ids = torch.stack(new_list_all, dim=0)

    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                # video
                if NAVIGATION_IDENTIFIER not in conversation and video_or_not:
                    round_len = len(tokenizer_image_token(rou, tokenizer)) + 3
                    instruction_len = (
                        len(tokenizer_image_token(parts[0], tokenizer)) + 3 - 2
                    )
                # image
                elif NAVIGATION_IDENTIFIER not in conversation and not video_or_not:
                    round_len = len(tokenizer_image_token(rou, tokenizer)) + 2
                    instruction_len = (
                        len(tokenizer_image_token(parts[0], tokenizer)) + 2 - 2
                    )

                # navigation video
                elif NAVIGATION_IDENTIFIER in conversation and video_or_not:
                    round_len = len(tokenizer_image_token(rou, tokenizer)) + 6
                    instruction_len = (
                        len(tokenizer_image_token(parts[0], tokenizer)) + 6 - 2
                    )

                else:
                    print(
                        f"converstation:{conversation}, video_or_not:{video_or_not}, identifier: {NAVIGATION_IDENTIFIER in conversation}"
                    )
                    raise ValueError

            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                # target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}. converstation:{conversation}, target:{target}, input_id:{input_ids} video_or_not:{video_or_not}, identifier: {NAVIGATION_IDENTIFIER in conversation}  "
                    f" (ignored)"
                )
    return dict(
        input_ids=input_ids,
        labels=targets,
        prompt=guided_prompt,
    )


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = torch.stack(
        [
            tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
            for prompt in conversations
        ],
        dim=0,
    )
    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])]  # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(
                conv.sep.join(rounds[conv_idx : conv_idx + 2])
            )  # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer_image_token(rou, tokenizer)) + len(
                tokenizer_image_token(conv.sep, tokenizer)
            )
            instruction_len = len(tokenizer_image_token(parts[0], tokenizer))
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]["value"]
        source[0]["value"] = DEFAULT_IMAGE_TOKEN
        conversation = (
            source[0]["value"]
            + source[1]["value"]
            + conversation_lib.default_conversation.sep
        )
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [
        tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        for prompt in conversations
    ]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]["value"], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess_plain_guided(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    prompt: str = None,
    video_or_not: bool = False,
) -> Dict:
    # add end signal and concatenate together
    guided_prompt = []
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]["value"]
        guided_prompt.append(
            source[0]["value"].replace(DEFAULT_IMAGE_TOKEN, "").replace("\n", "")
        )
        source[0]["value"] = DEFAULT_IMAGE_TOKEN
        conversation = (
            source[0]["value"]
            + source[1]["value"]
            + conversation_lib.default_conversation.sep
        )
        conversations.append(conversation)
    # tokenize conversations
    # input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    image_start_special_token = tokenizer(
        IMAGE_START_TOKEN, return_tensors="pt"
    ).input_ids[0][1:]
    image_end_special_token = tokenizer(IMAGE_END_TOKEN, return_tensors="pt").input_ids[
        0
    ][1:]
    video_start_special_token = tokenizer(
        VIDEO_START_SPECIAL_TOKEN, return_tensors="pt"
    ).input_ids[0][1:]
    video_end_special_token = tokenizer(
        VIDEO_END_SPECIAL_TOKEN, return_tensors="pt"
    ).input_ids[0][1:]
    image_seperator = tokenizer(IAMGE_SEPARATOR, return_tensors="pt").input_ids[0][1:]
    new_list_all = []
    for prompt in conversations:
        token_prompt = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        indices_to_replace = torch.where(token_prompt == -200)[0]
        new_list = []
        while indices_to_replace.numel() > 0:
            idx = indices_to_replace[0]
            if video_or_not:
                new_list.append(token_prompt[:idx])
                new_list.append(video_start_special_token)
                new_list.append(image_seperator)
                new_list.append(token_prompt[idx : idx + 1])
                new_list.append(video_end_special_token)
                token_prompt = token_prompt[idx + 1 :]
            else:
                new_list.append(token_prompt[:idx])
                new_list.append(image_start_special_token)
                new_list.append(token_prompt[idx : idx + 1])
                new_list.append(image_end_special_token)
                token_prompt = token_prompt[idx + 1 :]
            indices_to_replace = torch.where(token_prompt == -200)[0]
        if token_prompt.numel() > 0:
            new_list.append(token_prompt)
        new_list_all.append(torch.cat(new_list, dim=0))
    input_ids = torch.stack(new_list_all, dim=0)

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        # tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        if video_or_not:
            tokenized_len = (
                len(tokenizer_image_token(source[0]["value"], tokenizer)) + 3
            )
        else:
            tokenized_len = (
                len(tokenizer_image_token(source[0]["value"], tokenizer)) + 2
            )
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets, prompt=guided_prompt)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    prompt: str = None,
    refine_prompt: bool = False,
    video_or_not: bool = False,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.version.startswith("plain_guided"):
        return preprocess_plain_guided(
            sources, tokenizer, prompt=prompt, video_or_not=video_or_not
        )
    elif (
        conversation_lib.default_conversation.sep_style
        == conversation_lib.SeparatorStyle.PLAIN
    ):
        return preprocess_plain(sources, tokenizer)
    if (
        conversation_lib.default_conversation.sep_style
        == conversation_lib.SeparatorStyle.LLAMA_2
    ):
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    elif conversation_lib.default_conversation.version.startswith("imgsp"):
        return preprocess_imgsp_v1(
            sources,
            tokenizer,
            has_image=has_image,
            refine_prompt=refine_prompt,
            video_or_not=video_or_not,
        )
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)

    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [
            tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
            for prompt in conversations
        ]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn(
                [header] + [s["value"] for s in source], tokenizer
            )["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)
