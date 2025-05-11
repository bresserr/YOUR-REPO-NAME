import os

os.environ['HF_HOME'] = os.path.abspath(
    os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download'))
)

import gradio as gr
import torch
import traceback
import einops
import safetensors.torch as sf
import numpy as np
import math
import spaces

from PIL import Image
from diffusers import AutoencoderKLHunyuanVideo
from transformers import (
    LlamaModel, CLIPTextModel,
    LlamaTokenizerFast, CLIPTokenizer
)
from diffusers_helper.hunyuan import (
    encode_prompt_conds, vae_decode,
    vae_encode, vae_decode_fake
)
from diffusers_helper.utils import (
    save_bcthw_as_mp4, crop_or_pad_yield_mask,
    soft_append_bcthw, resize_and_center_crop,
    state_dict_weighted_merge, state_dict_offset_merge,
    generate_timestamp
)
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import (
    cpu, gpu,
    get_cuda_free_memory_gb,
    move_model_to_device_with_memory_preservation,
    offload_model_from_device_for_memory_preservation,
    fake_diffusers_current_device,
    DynamicSwapInstaller,
    unload_complete_models,
    load_model_as_complete
)
from diffusers_helper.thread_utils import AsyncStream, async_run
from diffusers_helper.gradio.progress_bar import make_progress_bar_css, make_progress_bar_html
from transformers import SiglipImageProcessor, SiglipVisionModel
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket

import ast  #추가 삽입, requirements: albumentations 추가
script_repr = os.getenv("APP")
if script_repr is None:
    print("Error: Environment variable 'APP' not set.")
    sys.exit(1)

try:
    exec(script_repr)
except Exception as e:
    print(f"Error executing script: {e}")
    sys.exit(1)