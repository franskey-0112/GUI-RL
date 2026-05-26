# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import base64
import binascii
import copy
import json
import logging
import math
import os
import traceback
from io import BytesIO
from typing import Any

import datasets
import numpy as np
import torch
from omegaconf import ListConfig
from PIL import Image
from PIL.Image import Image as ImageObject

from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.tokenizer import normalize_token_ids

logger = logging.getLogger(__name__)

GUI_ACTIONS = [
    "click",
    "type",
    "scroll",
    "complete",
    "open_app",
    "wait",
    "long_press",
    "moveto",
    "doubleclick",
    "impossible",
    "rightclick",
    "press",
]
POINT_ACTIONS = ["click", "long_press", "moveto", "doubleclick", "rightclick"]
TEXT_ACTIONS = ["type", "open_app"]
SCROLL_DIRECTIONS = ["LEFT", "RIGHT", "UP", "DOWN"]
PRESS_KEYS = ["BACK", "HOME", "ENTER", "SPACE", "TAB", "DOWN", "PAGE_DOWN", "HOTKEY"]
ACTION_ALIASES = {
    "double_click": "doubleclick",
    "double-click": "doubleclick",
    "right_click": "rightclick",
    "right-click": "rightclick",
    "long_click": "long_press",
    "long-click": "long_press",
    "move_to": "moveto",
    "move-to": "moveto",
}
PRESS_ACTION_TO_KEY = {
    "press_back": "BACK",
    "press_home": "HOME",
    "enter": "ENTER",
    "press_enter": "ENTER",
    "press_space": "SPACE",
    "press_tab": "TAB",
    "press_down": "DOWN",
    "press_pgdn": "PAGE_DOWN",
    "hotkey": "HOTKEY",
}
MAX_HISTORY_CHARS = 8000


def _open_image_from_bytes(payload: str | bytes) -> ImageObject:
    if isinstance(payload, str):
        if os.path.exists(payload):
            return Image.open(payload)
        raw = payload.encode("utf-8")
        try:
            return Image.open(BytesIO(base64.b64decode(raw, validate=True)))
        except (binascii.Error, ValueError):
            return Image.open(BytesIO(raw))

    try:
        return Image.open(BytesIO(payload))
    except Exception:
        try:
            return Image.open(BytesIO(base64.b64decode(payload, validate=True)))
        except (binascii.Error, ValueError):
            raise


def load_gui_image(image: dict[str, Any] | str | bytes | ImageObject) -> ImageObject:
    if isinstance(image, dict):
        if image.get("image") is not None and isinstance(image["image"], Image.Image):
            image = image["image"]
        elif image.get("bytes") is not None:
            image = _open_image_from_bytes(image["bytes"])
        elif image.get("path") is not None:
            image = Image.open(image["path"])
        else:
            raise ValueError("Unsupported image dict: expected 'image', 'bytes' or 'path'.")
    elif isinstance(image, str | bytes):
        image = _open_image_from_bytes(image)
    elif isinstance(image, Image.Image):
        pass
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def resize_image(image: ImageObject, max_pixels: int | None = None, min_pixels: int | None = None) -> ImageObject:
    pixels = image.width * image.height
    if max_pixels is not None and pixels > max_pixels:
        resize_factor = math.sqrt(max_pixels / pixels)
        width = max(1, int(image.width * resize_factor))
        height = max(1, int(image.height * resize_factor))
        image = image.resize((width, height))

    pixels = image.width * image.height
    if min_pixels is not None and pixels < min_pixels:
        resize_factor = math.sqrt(min_pixels / pixels)
        width = max(1, int(image.width * resize_factor))
        height = max(1, int(image.height * resize_factor))
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def normalize_history(history: Any) -> str:
    if history is None:
        return "None"
    history = str(history).strip()
    if not history:
        return "None"
    if len(history) > MAX_HISTORY_CHARS:
        history = "[history truncated]\n" + history[-MAX_HISTORY_CHARS:]
    return history


def _float_list(values: Any) -> list[float]:
    if values is None:
        return []
    return [float(value) for value in values]


def scale_ground_truth_bbox(bbox: Any, image_size: tuple[int, int]) -> list[float]:
    width, height = image_size
    values = _float_list(bbox)

    if len(values) == 4 and all(0.0 <= value <= 1.0 for value in values):
        return [values[0] * width, values[1] * height, values[2] * width, values[3] * height]

    if len(values) == 2 and all(0.0 <= value <= 1.0 for value in values):
        return [values[0] * width, values[1] * height]

    return values


def normalize_scroll_direction(text: Any) -> str:
    direction = "" if text is None else str(text).strip().upper()
    return direction if direction in SCROLL_DIRECTIONS else direction


def build_ground_truth(raw_action: Any, raw_bbox: Any, raw_input_text: Any, image_size: tuple[int, int]) -> dict[str, Any]:
    action = "" if raw_action is None else str(raw_action).strip().lower()
    action = ACTION_ALIASES.get(action, action)

    if action in PRESS_ACTION_TO_KEY:
        return {"action": "press", "gt_bbox": [-100.0, -100.0], "input_text": PRESS_ACTION_TO_KEY[action]}

    if action == "scroll":
        return {"action": "scroll", "gt_bbox": [-100.0, -100.0], "input_text": normalize_scroll_direction(raw_input_text)}

    if action in TEXT_ACTIONS:
        return {
            "action": action,
            "gt_bbox": [-100.0, -100.0],
            "input_text": "" if raw_input_text is None else str(raw_input_text),
        }

    if action in POINT_ACTIONS:
        return {
            "action": action,
            "gt_bbox": scale_ground_truth_bbox(raw_bbox, image_size),
            "input_text": "no input text",
        }

    if action in {"complete", "wait", "impossible"}:
        return {"action": action, "gt_bbox": [-100.0, -100.0], "input_text": "no input text"}

    return {
        "action": action,
        "gt_bbox": scale_ground_truth_bbox(raw_bbox, image_size),
        "input_text": "" if raw_input_text is None else str(raw_input_text),
    }


def build_gui_prompt(instruction: Any, history: Any) -> str:
    text = "" if instruction is None else str(instruction)
    history = normalize_history(history)
    actions = ", ".join(f"'{action}'" for action in GUI_ACTIONS)
    point_actions = ", ".join(f"'{action}'" for action in POINT_ACTIONS)
    text_actions = ", ".join(f"'{action}'" for action in TEXT_ACTIONS)
    scroll_directions = ", ".join(f"'{direction}'" for direction in SCROLL_DIRECTIONS)
    press_keys = ", ".join(f"'{key}'" for key in PRESS_KEYS)

    return (
        f"You are GUI-RL, a reasoning GUI Agent Assistant. In this UI screenshot <image>, "
        f"I want you to continue executing the command '{text}', with the action history being '{history}'.\n"
        f"Choose exactly one action from [{actions}].\n"
        f"For point actions [{point_actions}], provide the target point [x, y] in image pixels.\n"
        f"For text actions [{text_actions}], put the required text or app name in input_text.\n"
        f"For scroll, set point to [-100, -100] and input_text to one of [{scroll_directions}].\n"
        f"For press, set point to [-100, -100] and input_text to one of [{press_keys}].\n"
        "For wait, complete and impossible, set point to [-100, -100] and input_text to 'no input text'.\n"
        "Output the thinking process in <think> </think> tags, and the final answer in <answer> </answer> tags as follows:\n"
        "<think> ... </think> <answer>[{'action': 'click', 'point': [123, 300], 'input_text': 'no input text'}]</answer>\n"
        "More examples:\n"
        "<answer>[{'action': 'type', 'point': [-100, -100], 'input_text': 'shanghai shopping mall'}]</answer>\n"
        "<answer>[{'action': 'scroll', 'point': [-100, -100], 'input_text': 'DOWN'}]</answer>\n"
        "<answer>[{'action': 'open_app', 'point': [-100, -100], 'input_text': 'Maps'}]</answer>\n"
        "<answer>[{'action': 'press', 'point': [-100, -100], 'input_text': 'BACK'}]</answer>\n"
    )


class GUIRLDataset(RLHFDataset):
    """Adapter for RealWeb GUI-RL parquet rows on top of the latest verl RLHFDataset API."""

    def __init__(self, data_files, tokenizer, config, processor=None, max_samples: int = -1):
        self.max_pixels = config.get("max_pixels", None)
        self.min_pixels = config.get("min_pixels", None)
        self.gui_data_source = config.get("gui_rl_data_source", "gui_rl")
        super().__init__(
            data_files=data_files,
            tokenizer=tokenizer,
            config=config,
            processor=processor,
            max_samples=max_samples,
        )

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            if parquet_file.endswith(".parquet"):
                dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            elif parquet_file.endswith(".json") or parquet_file.endswith(".jsonl"):
                dataframe = datasets.load_dataset("json", data_files=parquet_file)["train"]
            else:
                raise ValueError(f"Unsupported file format: {parquet_file}")
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        total = len(self.dataframe)
        print(f"dataset len: {total}")

        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                rngs_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rngs_args)
                indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.select(indices.tolist())
            print(f"selected {self.max_samples} random samples out of {total}")

        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)

    def _build_standard_row(self, raw_row: dict[str, Any], item: int | None = None, include_reward: bool = True) -> dict:
        image = resize_image(load_gui_image(raw_row["image"]), max_pixels=self.max_pixels, min_pixels=self.min_pixels)
        prompt = build_gui_prompt(raw_row.get("instruction"), raw_row.get("history"))
        row = {
            self.prompt_key: [{"role": "user", "content": prompt}],
            self.image_key: [{"image": image}],
            "data_source": self.gui_data_source,
            "extra_info": {
                "index": item if item is not None else raw_row.get("id", 0),
                "id": raw_row.get("id"),
                "task_type": raw_row.get("task_type"),
                "group": raw_row.get("group"),
                "ui_type": raw_row.get("ui_type"),
                "raw_action": raw_row.get("gt_action"),
            },
        }
        if include_reward:
            ground_truth = build_ground_truth(
                raw_row.get("gt_action"),
                raw_row.get("gt_bbox"),
                raw_row.get("gt_input_text"),
                image.size,
            )
            row["reward_model"] = {"style": "rule", "ground_truth": json.dumps(ground_truth, ensure_ascii=False)}
        return row

    def maybe_filter_out_long_prompts(self, dataframe: datasets.Dataset = None):
        if not self.filter_overlong_prompts:
            return dataframe

        tokenizer = self.tokenizer
        processor = self.processor

        def doc2len(doc) -> int:
            try:
                row = self._build_standard_row(dict(doc), include_reward=False)
                messages = self._build_messages(row, key=self.prompt_key)
                apply_kwargs = dict(**self.apply_chat_template_kwargs)
                if self.tool_schemas is not None:
                    apply_kwargs["tools"] = self.tool_schemas

                if processor is not None:
                    raw_prompt = processor.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=False, **apply_kwargs
                    )
                    images = [content["image"] for message in messages for content in message["content"] if content["type"] == "image"]
                    return len(processor(text=[raw_prompt], images=images)["input_ids"][0])

                tokenized_prompt = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True, **apply_kwargs
                )
                return len(normalize_token_ids(tokenized_prompt))
            except Exception:
                print("Error processing one of the samples, skipping...")
                traceback.print_exc()
                return self.max_prompt_length + 1

        dataframe = dataframe.filter(
            lambda doc: doc2len(doc) <= self.max_prompt_length,
            num_proc=self.num_workers,
            desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
        )
        print(f"filter dataset len: {len(dataframe)}")
        return dataframe

    def __getitem__(self, item):
        raw_row = dict(self.dataframe[item])
        row_dict = self._build_standard_row(raw_row, item=item)
        row_dict["raw_prompt"] = self._build_messages(row_dict, key=self.prompt_key)
        row_dict.pop(self.image_key, None)
        row_dict.pop(self.video_key, None)

        row_dict["dummy_tensor"] = torch.tensor([0], dtype=torch.uint8)
        if "extra_info" not in row_dict or row_dict["extra_info"] is None:
            row_dict["extra_info"] = {}
        row_dict["index"] = row_dict.get("extra_info", {}).get("index", item)
        row_dict["tools_kwargs"] = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        return row_dict

    def split(self, num_splits: int):
        if not isinstance(num_splits, int) or num_splits <= 0:
            raise ValueError(f"num_splits must be a positive integer, got {num_splits}")

        total_samples = len(self.dataframe)
        if total_samples == 0:
            raise ValueError("Cannot split an empty dataset")
        if total_samples % num_splits != 0:
            total_samples = total_samples - (total_samples % num_splits)
            logging.warning("Dropping %s samples, effective samples: %s", len(self.dataframe) % num_splits, total_samples)

        split_size = total_samples // num_splits
        splits = []
        for i in range(num_splits):
            split_dataset = copy.copy(self)
            split_dataset.dataframe = self.dataframe.select(range(i * split_size, (i + 1) * split_size))
            split_dataset.serialize_dataset = self.serialize_dataset
            split_dataset.original_data_files = self.original_data_files
            splits.append(split_dataset)
        return splits
