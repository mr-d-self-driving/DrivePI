import numpy as np
import torch

torch.backends.cuda.matmul.allow_tf32 = True

import os

import copy
import warnings
from datetime import timedelta
from typing import List, Optional, Tuple, Union

from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from tqdm import tqdm

import torch
import torch.nn as nn
import json
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoProcessor, AutoConfig
import argparse

from emova.model.builder import load_pretrained_model
from emova.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from emova.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, \
    DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from emova.conversation import conv_templates, SeparatorStyle
from emova.registry_utils import Config
import torch.distributed as dist
from loguru import logger as eval_logger

default_system_prompt = 'You are a helpful assistant.'


def parse_args():
    parser = argparse.ArgumentParser(description="Process images and questions with a pretrained model.")

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to the pretrained model directory or name. For example: "/path/to/model"'
    )
    parser.add_argument(
        '--torch_dtype',
        type=str,
        default='fp32',
        help='data type for the modell'
    )

    parser.add_argument(
        '--system_prompt',
        type=str,
        default=None,
        help='System prompt to set the model\'s role and tone.'
    )

    parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help='Path to save the model responses as a JSON file.'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=2,
        help='Number of samples to process in each batch.'
    )

    parser.add_argument(
        '--max_new_tokens',
        type=int,
        default=1000,
        help='Maximum new tokens during inference.'
    )

    parser.add_argument(
        '--num_beams',
        type=int,
        default=1,
        help='Num beam in beam search.'
    )

    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='Maximum new tokens during inference.'
    )

    parser.add_argument(
        '--do_sample',
        action='store_true',
        help='do sampling during inference.',
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
    )

    parser.add_argument(
        '--device_map',
        type=str,
        default=None,
    )

    parser.add_argument(
        '--local_rank',
        type=int,
        default=0,
    )

    # 新增参数：只测试projector
    parser.add_argument(
        '--projector_only',
        action='store_true',
        help='Only test the projector output without language model inference.'
    )

    # 新增参数：projector权重文件路径
    parser.add_argument(
        '--projector_weights_path',
        type=str,
        default=None,
        help='Path to the trained projector weights file (mm_projector.bin). If not specified, will look for mm_projector.bin in the model output directory.'
    )

    args = parser.parse_args()
    return args


def read_config(file):
    # solve config loading conflict when multi-processes
    import time
    while True:
        config = Config.fromfile(file)
        if len(config) == 0:
            time.sleep(0.1)
            continue
        break
    return config


if torch.__version__ > "2.1.2":
    best_fit_attn_implementation = "sdpa"
else:
    try:
        from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func

        best_fit_attn_implementation = "flash_attention_2"
    except Exception as e:
        best_fit_attn_implementation = "eager"


class ProjectorOnlyChatbot:
    """按照训练时的初始化方法初始化模型，然后只给projector加载训练好的权重"""

    def __init__(
            self,
            config: str,
            device: Optional[str] = "cuda",
            device_map="",
            dtype='fp16',
            projector_weights_path: Optional[str] = None,
            **kwargs,
    ) -> None:
        super().__init__()

        self.log_step = 10
        self._device = torch.device(device)
        self.device_map = device_map
        self.torch_dtype = dtype = torch.float16 if dtype == 'fp16' else torch.bfloat16

        # 加载配置
        config = read_config(config)
        model_args = config.model_args

        self.output_dir = config.training_args.output_dir
        # 按照训练时的初始化方法初始化完整模型
        from emova.model.language_model.builder import build_language_model

        # 构建语言模型（按照训练时的初始化方法）
        self._model = build_language_model(
            model_args.language_model,
            low_cpu_mem_usage=True,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map
        )

        # 初始化vision modules（按照训练时的方法）
        self._model.model.initialize_vision_modules(model_args)

        # 设置模型到设备
        if self.device_map != "auto":
            self._model.to(device=self._device, dtype=self.torch_dtype)

        # 获取vision tower和projector
        self.vision_tower = self._model.get_vision_tower()
        self.mm_projector = self._model.get_mm_projector()

        # 获取image processor
        self._image_processor = self.vision_tower.image_processor

        # 获取tokenizer（从语言模型）
        self._tokenizer = self._model.get_tokenizer() if hasattr(self._model, 'get_tokenizer') else None
        if self._tokenizer is None:
            # 如果没有get_tokenizer方法，尝试从配置中获取
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_args.language_model.pretrained_model_name_or_path,
                use_fast=False
            )

        # 设置其他必要的属性
        self._config = AutoConfig.from_pretrained(config.training_args.output_dir)
        self._model.config = self._config

        self._max_length = getattr(self._config, "max_sequence_length", 2048)
        self.conv_template = model_args.version
        self.use_cache = True
        self.truncate_context = False

        print(
            f"Model initialized with training method. Vision tower: {self.vision_tower.hidden_size} -> Projector: {self._model.config.hidden_size}")

        # 加载训练好的projector权重
        # 使用默认路径（模型输出目录下的mm_projector.bin）
        model_path = config.training_args.output_dir
        model_path = os.path.expanduser(model_path)
        mm_projector_path = os.path.join(model_path, 'mm_projector.bin')

        if os.path.exists(mm_projector_path):
            print(f"Loading trained projector weights from {mm_projector_path}")
            mm_projector_weights = torch.load(mm_projector_path, map_location='cpu')

            # 提取projector部分的权重
            def get_projector_weights(weights, keyword='mm_projector'):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            projector_state = get_projector_weights(mm_projector_weights, 'mm_projector')
            msg = self.mm_projector.load_state_dict(projector_state, strict=False)
            print(f"Projector weights loaded. Missing keys: {msg.missing_keys}, Unexpected keys: {msg.unexpected_keys}")
        else:
            print(f"No projector weights found at {mm_projector_path}, using randomly initialized projector weights")

    @property
    def device(self):
        return self._device

    @property
    def image_processor(self):
        return self._image_processor

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        return self._model

    @property
    def config(self):
        return self._config

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens):
        try:
            return self.tokenizer.decode(tokens)
        except:
            return self.tokenizer.decode([tokens])

    def chat(self, inputs) -> List[str]:
        """与Chatbot相同的chat方法，支持完整的QA功能"""
        image_sizes = None

        if isinstance(inputs[0]['image'], str) and inputs[0]['image'].endswith('npy'):
            batched_visuals = [np.load(item['image']) for item in inputs]  # [B, N]
        elif isinstance(inputs[0]['image'], str):
            batched_visuals = [Image.open(item['image']) for item in inputs]  # [B, N]
        else:
            batched_visuals = [[Image.open(img) for img in item['image']] for item in inputs]  # [B, N]

        contexts = ['<image>\n' + item['question'] if '<image>' not in item['question'] else item['question'] for item
                    in inputs]  # [B, N]

        gen_kwargs = dict()

        # Set default values for until and max_new_tokens
        until = [self.tok_decode(self.eot_token_id)]

        # Update values from gen_kwargs if present
        if "until" in gen_kwargs:
            until = gen_kwargs.pop("until")
            if isinstance(until, str):
                until = [until]
            elif not isinstance(until, list):
                raise ValueError(
                    f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}")

        if "image_aspect_ratio" in gen_kwargs.keys() and "image_aspect_ratio" not in self._config.__dict__:
            # here we should pop it out of gen_kwargs so that it doesn't get passed to the model for next step of generation
            self._config.image_aspect_ratio = gen_kwargs.pop("image_aspect_ratio")
            eval_logger.info(f"Setting image aspect ratio: {self._config.image_aspect_ratio}")
        eval_logger.info(f"Setting image aspect ratio: {self._config.image_aspect_ratio}")

        # encode, pad, and truncate contexts for this batch
        if batched_visuals:
            if isinstance(batched_visuals[0], list):
                image_tensor = []
                image_sizes = []
                for visuals in batched_visuals:
                    one_image_tensor, one_image_sizes = process_images(visuals, self._image_processor, self._config)
                    image_tensor.append(one_image_tensor)
                    image_sizes.append(torch.as_tensor(one_image_sizes))
                image_tensor = torch.concat(image_tensor)
                image_sizes = torch.concat(image_sizes)
            else:
                image_tensor, image_sizes = process_images(batched_visuals, self._image_processor, self._config)

            if type(image_tensor) is list:
                image_tensor = [_image.to(dtype=self.torch_dtype, device=self.device) for _image in image_tensor]
            else:
                image_tensor = image_tensor.to(dtype=self.torch_dtype, device=self.device)
        else:
            image_tensor = None

        question_input = []

        for visual, context in zip(batched_visuals, contexts):
            if image_tensor is not None and len(image_tensor) != 0 and DEFAULT_IMAGE_TOKEN not in context:
                """
                Three senarios:
                1. No image, and there for, no image token should be added.
                2. image token is already specified in the context, so we don't need to add it.
                3. image token is not specified in the context and there is image inputs, so we need to add it. In this case, we add the image token at the beginning of the context and add a new line.
                """
                image_tokens = [DEFAULT_IMAGE_TOKEN] * len(visual) if isinstance(visual, list) else [
                    DEFAULT_IMAGE_TOKEN]
                image_tokens = " ".join(image_tokens)
                question = image_tokens + "\n" + context
            else:
                question = context

            # This is much safer for llama3, as we now have some object type in it
            if "llama_3" in self.conv_template or "llama3" in self.conv_template:
                conv = copy.deepcopy(conv_templates[self.conv_template])
            else:
                conv = conv_templates[self.conv_template].copy()
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()
            question_input.append(prompt_question)

        # The above for loop has bugs. When there is no visuals, e.g. pure text,
        # there will be no for loop execute resulting in an empty question_input (because no visuals)
        # Scenario 1 won't even be execute
        if len(batched_visuals) == 0:
            for context in contexts:
                question = context
                conv = conv_templates[self.conv_template].copy()
                conv.append_message(conv.roles[0], question)
                conv.append_message(conv.roles[1], None)
                prompt_question = conv.get_prompt()
                question_input.append(prompt_question)

        # preconfigure gen_kwargs with defaults
        if image_sizes is not None:
            gen_kwargs["image_sizes"] = image_sizes
        else:
            if isinstance(batched_visuals[0], list):
                image_sizes = []
                for visual in batched_visuals:
                    image_sizes.extend([v.size for v in visual])
                gen_kwargs["image_sizes"] = image_sizes
            else:
                gen_kwargs["image_sizes"] = [batched_visuals[idx].size for idx in range(len(batched_visuals))]
        if "max_new_tokens" not in gen_kwargs:
            gen_kwargs["max_new_tokens"] = 1024
        if "temperature" not in gen_kwargs:
            gen_kwargs["temperature"] = 0
        if "top_p" not in gen_kwargs:
            gen_kwargs["top_p"] = None
        if "num_beams" not in gen_kwargs:
            gen_kwargs["num_beams"] = 1

        input_ids_list = [tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt") for
                          prompt in question_input]
        pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        input_ids = self.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_ids).to(self.device)
        attention_masks = input_ids.ne(pad_token_ids).to(self.device)

        extra_kwargs = dict()
        cont = self.model.generate(
            input_ids,
            attention_mask=attention_masks,
            pad_token_id=pad_token_ids,
            images=image_tensor,
            image_sizes=gen_kwargs["image_sizes"],
            do_sample=True if gen_kwargs["temperature"] > 0 else False,
            temperature=gen_kwargs["temperature"],
            top_p=gen_kwargs["top_p"],
            num_beams=gen_kwargs["num_beams"],
            max_new_tokens=gen_kwargs["max_new_tokens"],
            use_cache=self.use_cache,
            **extra_kwargs,
        )

        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
        # reorder this group of results back to original unsorted form
        return text_outputs

    def process_images_projector_only(self, inputs):
        """只处理图像并返回projector的输出（用于调试）"""
        batched_visuals = [Image.open(item['image']) for item in inputs]

        # 处理图像
        image_tensor, image_sizes = process_images(batched_visuals, self._image_processor, self.vision_tower.config)
        if type(image_tensor) is list:
            image_tensor = [_image.to(dtype=self.torch_dtype, device=self.device) for _image in image_tensor]
        else:
            image_tensor = image_tensor.to(dtype=self.torch_dtype, device=self.device)

        # 通过vision tower获取特征
        with torch.no_grad():
            vision_features = self.vision_tower(image_tensor, image_sizes=image_sizes)

            # 通过projector处理特征
            projected_features = self.mm_projector(vision_features)

        # 转换为numpy并返回
        results = []
        for i in range(projected_features.shape[0]):
            feature_dict = {
                'projected_features': projected_features[i].cpu().numpy().tolist(),
                'feature_shape': list(projected_features[i].shape),
                'original_image_size': image_sizes[i] if image_sizes else None
            }
            results.append(feature_dict)

        return results


class Chatbot(object):
    def __init__(
            self,
            config: str = "Emova-ollm/emova-7b",
            truncation: Optional[bool] = True,
            device: Optional[str] = "cuda",
            batch_size: Optional[Union[int, str]] = 1,
            attn_implementation=best_fit_attn_implementation,
            device_map="",
            dtype='fp16',
            use_cache=True,
            tie_weights=True,
            truncate_context=False,  # whether to truncate the context in generation, set it False for LLaVA-1.6
            inference_max_num_slices=None,
            inference_max_pixels=None,
            inference_max_length=None,
            customized_config=None,
            log_step=10,
            **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        self.log_step = log_step

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
            print("DDP inference the model.")
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
            print("Using device_map: auto to parallelly inference the model.")
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
            print("Single process inference the model.")

        # config
        config = read_config(config)
        self.output_dir = config.training_args.output_dir
        model_base = None
        if config.training_args.get('lora_enable', False):
            model_base = config.model_args.language_model.pretrained_model_name_or_path

        model_path = config.training_args.output_dir
        model_path = os.path.expanduser(model_path)
        model_name = get_model_name_from_path(model_path)

        conv_template = config.model_args.version

        self.torch_dtype = dtype = torch.float16 if dtype == 'fp16' else torch.bfloat16

        llava_model_args = {
            # "multimodal": True,
        }

        if customized_config is not None:
            llava_model_args["customized_config"] = customized_config
        if attn_implementation is not None:
            llava_model_args["attn_implementation"] = attn_implementation
        if "use_flash_attention_2" in kwargs:
            llava_model_args["use_flash_attention_2"] = kwargs["use_flash_attention_2"]

        self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(
            model_path, model_base, model_name, device=self.device, device_map=self.device_map,
            config=config, torch_dtype=dtype, **llava_model_args)

        if inference_max_num_slices:
            from emova.utils import find_possible_grids
            base_size = self._model.get_vision_tower().config.image_size
            grids = find_possible_grids(inference_max_num_slices)
            eval_logger.info(f"Inference Time reset the max num slices to {inference_max_num_slices}")
            eval_logger.info(f"Inference Time Grids: {grids}")
            self._model.config.image_grid_pinpoints = [[g[0] * base_size, g[1] * base_size] for g in grids]

        if inference_max_length:
            self._max_length = self._model.config.tokenizer_model_max_length = self._tokenizer.model_max_length = inference_max_length
            eval_logger.info(f"Inference Time reset the max context length to {inference_max_length}")

        if inference_max_pixels:
            self._image_processor.max_pixels = inference_max_pixels
            eval_logger.info(f"Inference Time reset the max pixels to {inference_max_num_slices}")

        self._config = self._model.config

        self.model.eval()
        if tie_weights:
            self.model.tie_weights()

        self.truncation = truncation
        self.batch_size_per_gpu = int(batch_size)
        self.conv_template = conv_template
        self.use_cache = use_cache
        self.truncate_context = truncate_context
        assert self.batch_size_per_gpu == 1, "Emova currently does not support batched generation."

        if accelerator.num_processes > 1 and device_map == "":
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU,
                                                    DistributedType.MULTI_NPU,  # TONOTE
                                                    DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config before using the model
            # Also, you have to select zero stage 0 (equivalent to DDP) in order to make the prepare model works
            # I tried to set different parameters in the kwargs to let default zero 2 stage works, but it didn't work.
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info(
                    "Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")

            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        elif accelerator.num_processes == 1 and device_map == "auto":
            eval_logger.info(f"Using {accelerator.num_processes} devices with tensor parallelism")
            self._rank = 0
            self._word_size = 1
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        """ """
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens):
        try:
            return self.tokenizer.decode(tokens)
        except:
            return self.tokenizer.decode([tokens])

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def chat(self, inputs) -> List[str]:
        image_sizes = None

        assert len(inputs) == 1

        if isinstance(inputs[0]['image'], str) and inputs[0]['image'].endswith('npy'):
            batched_visuals = [np.load(item['image']) for item in inputs]  # [B, N]
        elif isinstance(inputs[0]['image'], str):
            batched_visuals = [Image.open(item['image']) for item in inputs]  # [B, N]
        else:
            batched_visuals = [[Image.open(img) for img in item['image']] for item in inputs]  # [B, N]

        contexts = [item['question'] for item in inputs]  # [B, N]

        gen_kwargs = dict()

        # Set default values for until and max_new_tokens
        until = [self.tok_decode(self.eot_token_id)]

        # Update values from gen_kwargs if present
        if "until" in gen_kwargs:
            until = gen_kwargs.pop("until")
            if isinstance(until, str):
                until = [until]
            elif not isinstance(until, list):
                raise ValueError(
                    f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}")

        if "image_aspect_ratio" in gen_kwargs.keys() and "image_aspect_ratio" not in self._config.__dict__:
            # here we should pop it out of gen_kwargs so that it doesn't get passed to the model for next step of generation
            self._config.image_aspect_ratio = gen_kwargs.pop("image_aspect_ratio")
            eval_logger.info(f"Setting image aspect ratio: {self._config.image_aspect_ratio}")

        # encode, pad, and truncate contexts for this batch
        if batched_visuals:
            if isinstance(batched_visuals[0], list):
                image_tensor = []
                image_sizes = []
                for visuals in batched_visuals:
                    one_image_tensor, one_image_sizes = process_images(visuals, self._image_processor, self._config)
                    image_tensor.append(one_image_tensor)
                    image_sizes.append(one_image_sizes)

                image_tensor = torch.concat(image_tensor)
                image_sizes = torch.concat(image_sizes)
            else:
                image_tensor, image_sizes = process_images(batched_visuals, self._image_processor, self._config)

            if type(image_tensor) is list:
                image_tensor = [_image.to(dtype=self.torch_dtype, device=self.device) for _image in image_tensor]
            else:
                image_tensor = image_tensor.to(dtype=self.torch_dtype, device=self.device)
        else:
            image_tensor = None

        question_input = []

        for visual, context in zip(batched_visuals, contexts):
            if image_tensor is not None and len(image_tensor) != 0 and DEFAULT_IMAGE_TOKEN not in context:
                """
                Three senarios:
                1. No image, and there for, no image token should be added.
                2. image token is already specified in the context, so we don't need to add it.
                3. image token is not specified in the context and there is image inputs, so we need to add it. In this case, we add the image token at the beginning of the context and add a new line.
                """
                image_tokens = [DEFAULT_IMAGE_TOKEN] * len(visual) if isinstance(visual, list) else [
                    DEFAULT_IMAGE_TOKEN]
                image_tokens = " ".join(image_tokens)
                question = image_tokens + "\n" + context
            else:
                question = context

            # This is much safer for llama3, as we now have some object type in it
            if "llama_3" in self.conv_template or "llama3" in self.conv_template:
                conv = copy.deepcopy(conv_templates[self.conv_template])
            else:
                conv = conv_templates[self.conv_template].copy()
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()
            question_input.append(prompt_question)

        # The above for loop has bugs. When there is no visuals, e.g. pure text,
        # there will be no for loop execute resulting in an empty question_input (because no visuals)
        # Scenario 1 won't even be execute
        if len(batched_visuals) == 0:
            for context in contexts:
                question = context
                conv = conv_templates[self.conv_template].copy()
                conv.append_message(conv.roles[0], question)
                conv.append_message(conv.roles[1], None)
                prompt_question = conv.get_prompt()
                question_input.append(prompt_question)

        # input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
        # preconfigure gen_kwargs with defaults
        if image_sizes is not None:
            gen_kwargs["image_sizes"] = image_sizes
        else:
            if isinstance(batched_visuals[0], list):
                image_sizes = []
                for visual in batched_visuals:
                    image_sizes.extend([v.size for v in visual])
                gen_kwargs["image_sizes"] = image_sizes
            else:
                gen_kwargs["image_sizes"] = [batched_visuals[idx].size for idx in range(len(batched_visuals))]
        if "max_new_tokens" not in gen_kwargs:
            gen_kwargs["max_new_tokens"] = 1024
        if "temperature" not in gen_kwargs:
            gen_kwargs["temperature"] = 0
        if "top_p" not in gen_kwargs:
            gen_kwargs["top_p"] = None
        if "num_beams" not in gen_kwargs:
            gen_kwargs["num_beams"] = 1

        input_ids_list = [tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt") for
                          prompt in question_input]
        pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        input_ids = self.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_ids).to(self.device)
        attention_masks = input_ids.ne(pad_token_ids).to(self.device)
        # These steps are not in LLaVA's original code, but are necessary for generation to work
        # TODO: pay attention to this major generation step...

        extra_kwargs = dict()
        cont = self.model.generate(
            input_ids,
            attention_mask=attention_masks,
            pad_token_id=pad_token_ids,
            images=image_tensor,
            image_sizes=gen_kwargs["image_sizes"],
            do_sample=True if gen_kwargs["temperature"] > 0 else False,
            temperature=gen_kwargs["temperature"],
            top_p=gen_kwargs["top_p"],
            num_beams=gen_kwargs["num_beams"],
            max_new_tokens=gen_kwargs["max_new_tokens"],
            use_cache=self.use_cache,
            **extra_kwargs,
        )

        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
        # reorder this group of results back to original unsorted form
        return text_outputs


def gather_dicts(data):
    # 获取当前进程的 rank 和 world size
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # 初始化一个空列表，用于存储所有进程的数据
    gathered_data = [None for _ in range(world_size)]

    # 使用 all_gather_object 收集数据
    dist.all_gather_object(gathered_data, data)

    # 返回收集到的数据
    return gathered_data


if __name__ == '__main__':
    args = parse_args()

    accelerator = Accelerator()
    rank = accelerator.local_process_index
    world_size = accelerator.num_processes

    # 根据参数选择使用哪种模式
    if args.projector_only:
        print("Running in projector-only mode...")
        chatbot = ProjectorOnlyChatbot(
            config=args.config,
            device=args.device,
            device_map=args.device_map,
            dtype=args.torch_dtype,
            projector_weights_path=args.projector_weights_path
        )
    else:
        print("Running in full model mode...")
        chatbot = Chatbot(config=args.config)

    batch_size = args.batch_size
    task_file_root = '/data2/rhhuang/captions/'
    is_train_format = True
    # for task_file in ['nusceneqa_val_llava.json']:
    #     data = json.load(open(os.path.join(task_file_root, task_file)))
    #
    #     # 切分数据，每个进程只处理自己负责的部分
    #     data = data[rank::world_size]
    #
    #     if is_train_format:
    #         '''
    #           {
    #             "image": [],
    #             "conversations": [
    #             {
    #                 "from": "human",
    #                 "value": "<image>\nAre there any other construction vehicles of the same status as the construction vehicle?"
    #             },
    #             {
    #                 "from": "gpt",
    #                 "value": "no"
    #             }
    #             ],
    #             "history": {
    #             "split": "val",
    #             "sample_token": "c9b7e76f90594a84b5a4ea6750912d02",
    #             "question": "Are there any other construction vehicles of the same status as the construction vehicle?",
    #             "answer": "no",
    #             "num_hop": 0,
    #             "template_type": "exist",
    #             "token": "c9b7e76f90594a84b5a4ea6750912d02"
    #             }
    #         }
    #         '''
    #         # to
    #         '''
    #         {"id": "7ea15ebb-6ce9-4d4c-b13e-c20f295856ac",
    #         "image": "/data3/zliu/saved_bev_features_val/23cd0004606a4127acc8505a8de5c282.npy",
    #         "history": {"token": "23cd0004606a4127acc8505a8de5c282", "label": 16, "category_all": ["car", "truck", "trailer", "bus", "construction_vehicle", "bicycle", "motorcycle", "pedestrian", "traffic_cone", "barrier", "driveable_surface", "other_flat", "sidewalk", "terrain", "manmade", "vegetation", "free"]}, "question": "<image>\nYour task is to predict the 3D occupancy of the scene.\nThe scene area around you (in front, behind, left, and right) is divided into a 200x200 grid, with the bottom-left corner at (0, 0) and the top-right corner at (200, 200).\nThe height region is divided into 16 bins. We use <OCC>(x, y, z)</OCC> to represent the point at location (x, y) with a height of z.\nAssume you are located at the point <OCC>(100, 100, 0)</OCC>. Answer the below question.\nWhat object is occupying position <OCC>(20, 28, 10)</OCC>?\nIf there is an object, please provide its name; otherwise, answer 'free'.", "answer": "free"},
    #         '''
    #         new_data = []
    #         for item in data:
    #             new_item = {
    #                 "id": item['history']['token'],
    #                 "image": os.path.join("/data3/zliu/saved_bev_features_val/", item['history']['token'] + '.npy'),
    #                 "question": item['conversations'][0]['value'],
    #                 "answer": item['conversations'][1]['value'],
    #                 "history": item['history'],
    #             }
    #             new_data.append(new_item)
    #         data = new_data
    #
    #     # 进行完整的QA推理
    #     print("Performing QA inference...")
    #     responses = []
    #
    #     for batch_idx in tqdm(range(0, len(data), batch_size)):
    #         batch = data[batch_idx: batch_idx + batch_size]
    #         all_response = chatbot.chat(batch)
    #         responses.extend(all_response)
    #
    #     # 将 response 按顺序写入 data
    #     for i, ans in enumerate(responses):
    #         data[i]['pred'] = ans
    #
    #     # 收集所有进程的结果
    #     if world_size > 1:
    #         all_data = [None for _ in range(world_size)]
    #         dist.all_gather_object(all_data, data)
    #     else:
    #         all_data = [data]
    #
    #     if accelerator.is_local_main_process:
    #         # `all_data` is a list of lists, where each inner list is the data from one process.
    #         # We need to reconstruct the original order from the distributed slicing.
    #         reordered_data = [None] * sum(len(d) for d in all_data)
    #         for i, p_data in enumerate(all_data):
    #             for j, item in enumerate(p_data):
    #                 original_index = i + j * world_size
    #                 reordered_data[original_index] = item
    #
    #         correct_num = 0
    #         for item in reordered_data:
    #             # print(item['pred'],',', item['answer'])
    #             if item['pred'].lower() == item['answer'].lower():
    #                 correct_num += 1
    #         print(f'task {task_file} performance: {correct_num / len(reordered_data)}')
    #
    #         with open(os.path.join(chatbot.output_dir, task_file.replace('.json', '_results.json')), 'w', encoding='utf-8') as f:
    #             json.dump(reordered_data, f, ensure_ascii=False, indent=4)
    #         print(f"QA responses have been saved to {os.path.join(chatbot.output_dir, task_file)}")

    if accelerator.is_local_main_process:
        for task_file in ['nusceneqa_val_llava.json']:
            data = json.load(open(os.path.join(chatbot.output_dir, task_file.replace('.json', '_results.json'))))
            correct_num = 0
            # "history": {
            #     "split": "val",
            #     "sample_token": "d13f68b9973440a9b8c6c5b92f191513",
            #     "question": "Do the construction vehicle and the bus that is to the front left of the parked construction vehicle have the same status?",
            #     "answer": "no",
            #     "num_hop": 1,
            #     "template_type": "comparison",
            #     "token": "d13f68b9973440a9b8c6c5b92f191513"
            # }
            from collections import defaultdict
            val_results = defaultdict(list)

            for item in data:
                if item['pred'].lower() == item['answer'].lower():
                    correct_num += 1
                    correct = True
                else:
                    correct = False
                val_results[f"{item['history']['template_type']}_all"].append(correct)
                val_results[f"{item['history']['template_type']}_num_hop{item['history']['num_hop']}"].append(correct)

            print(f'task {task_file} performance: {correct_num / len(data)}')

            for sub_task in val_results:
                print(f'task {task_file} subtask {sub_task} {np.sum(val_results[sub_task]) / len(val_results[sub_task])}')