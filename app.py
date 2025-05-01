from diffusers_helper.hf_login import login

import os
import threading
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json

os.environ['HF_HOME'] = os.path.abspath(
    os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download'))
)

# 단일 언어(영어)만 사용하기 위한 번역 딕셔너리
translations = {
    "en": {
        "title": "FramePack - Image to Video Generation",
        "upload_image": "Upload Image",
        "prompt": "Prompt",
        "quick_prompts": "Quick Prompts",
        "start_generation": "Generate",
        "stop_generation": "Stop",
        "use_teacache": "Use TeaCache",
        "teacache_info": "Faster speed, but may result in slightly worse finger and hand generation.",
        "negative_prompt": "Negative Prompt",
        "seed": "Seed",
        "video_length": "Video Length (max 5 seconds)",
        "latent_window": "Latent Window Size",
        "steps": "Inference Steps",
        "steps_info": "Changing this value is not recommended.",
        "cfg_scale": "CFG Scale",
        "distilled_cfg": "Distilled CFG Scale",
        "distilled_cfg_info": "Changing this value is not recommended.",
        "cfg_rescale": "CFG Rescale",
        "gpu_memory": "GPU Memory Preservation (GB) (larger means slower)",
        "gpu_memory_info": "Set this to a larger value if you encounter OOM errors. Larger values cause slower speed.",
        "next_latents": "Next Latents",
        "generated_video": "Generated Video",
        "sampling_note": "Note: Due to reversed sampling, ending actions will be generated before starting actions. If the starting action is not in the video, please wait, it will be generated later.",
        "error_message": "Error",
        "processing_error": "Processing error",
        "network_error": "Network connection is unstable, model download timed out. Please try again later.",
        "memory_error": "GPU memory insufficient, please try increasing GPU memory preservation value or reduce video length.",
        "model_error": "Failed to load model, possibly due to network issues or high server load. Please try again later.",
        "partial_video": "Processing error, but partial video has been generated",
        "processing_interrupt": "Processing was interrupted, but partial video has been generated"
    }
}

# 영어만 사용할 것이므로 아래 함수는 사실상 항상 영어를 반환합니다.
def get_translation(key):
    return translations["en"].get(key, key)

# 언어는 영어로 고정
current_language = "en"

import gradio as gr
import torch
import traceback
import einops
import safetensors.torch as sf
import numpy as np
import math

# Hugging Face Space 환경 체크
IN_HF_SPACE = os.environ.get('SPACE_ID') is not None

# GPU 사용 여부 전역 관리
GPU_AVAILABLE = False
GPU_INITIALIZED = False
last_update_time = time.time()

if IN_HF_SPACE:
    try:
        import spaces
        print("Running in Hugging Face Space environment.")
        try:
            GPU_AVAILABLE = torch.cuda.is_available()
            print(f"GPU available: {GPU_AVAILABLE}")
            if GPU_AVAILABLE:
                test_tensor = torch.zeros(1, device='cuda') + 1
                del test_tensor
                print("GPU small test pass")
        except Exception as e:
            GPU_AVAILABLE = False
            print(f"Error checking GPU: {e}")
    except ImportError:
        GPU_AVAILABLE = torch.cuda.is_available()

from PIL import Image
from diffusers import AutoencoderKLHunyuanVideo
from transformers import (
    LlamaModel,
    CLIPTextModel,
    LlamaTokenizerFast,
    CLIPTokenizer,
    SiglipImageProcessor,
    SiglipVisionModel
)

from diffusers_helper.hunyuan import (
    encode_prompt_conds,
    vae_decode,
    vae_encode,
    vae_decode_fake
)

from diffusers_helper.utils import (
    save_bcthw_as_mp4,
    crop_or_pad_yield_mask,
    soft_append_bcthw,
    resize_and_center_crop,
    generate_timestamp
)

from diffusers_helper.bucket_tools import find_nearest_bucket
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import (
    cpu,
    gpu,
    get_cuda_free_memory_gb,
    move_model_to_device_with_memory_preservation,
    offload_model_from_device_for_memory_preservation,
    fake_diffusers_current_device,
    DynamicSwapInstaller,
    unload_complete_models,
    load_model_as_complete
)

from diffusers_helper.thread_utils import AsyncStream, async_run
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.gradio.progress_bar import (
    make_progress_bar_css,
    make_progress_bar_html
)

outputs_folder = './outputs/'
os.makedirs(outputs_folder, exist_ok=True)

# GPU 메모리 확인
if not IN_HF_SPACE:
    try:
        if torch.cuda.is_available():
            free_mem_gb = get_cuda_free_memory_gb(gpu)
            print(f'Free VRAM: {free_mem_gb} GB')
        else:
            free_mem_gb = 6.0
            print("CUDA not available, default memory setting used.")
    except Exception as e:
        free_mem_gb = 6.0
        print(f"Error getting GPU mem: {e}, using default=6GB")
    high_vram = free_mem_gb > 60
else:
    print("Using default memory setting in Spaces environment.")
    try:
        if GPU_AVAILABLE:
            free_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 * 0.9
            high_vram = (free_mem_gb > 10)
        else:
            free_mem_gb = 6.0
            high_vram = False
    except Exception as e:
        free_mem_gb = 6.0
        high_vram = False
    print(f'GPU memory: {free_mem_gb:.2f} GB, High-VRAM mode: {high_vram}')

models = {}
cpu_fallback_mode = not GPU_AVAILABLE

def load_models():
    """
    Load or initialize the global models
    """
    global models, cpu_fallback_mode, GPU_INITIALIZED
    
    if GPU_INITIALIZED:
        print("Models are already loaded, skipping re-initialization.")
        return models

    print("Start loading models...")

    try:
        device = 'cuda' if GPU_AVAILABLE and not cpu_fallback_mode else 'cpu'
        model_device = 'cpu'
        
        dtype = torch.float16 if GPU_AVAILABLE else torch.float32
        transformer_dtype = torch.bfloat16 if GPU_AVAILABLE else torch.float32

        print(f"Device: {device}, VAE/Encoders dtype={dtype}, Transformer dtype={transformer_dtype}")

        try:
            text_encoder = LlamaModel.from_pretrained(
                "hunyuanvideo-community/HunyuanVideo",
                subfolder='text_encoder',
                torch_dtype=dtype
            ).to(model_device)
            text_encoder_2 = CLIPTextModel.from_pretrained(
                "hunyuanvideo-community/HunyuanVideo",
                subfolder='text_encoder_2',
                torch_dtype=dtype
            ).to(model_device)
            tokenizer = LlamaTokenizerFast.from_pretrained(
                "hunyuanvideo-community/HunyuanVideo",
                subfolder='tokenizer'
            )
            tokenizer_2 = CLIPTokenizer.from_pretrained(
                "hunyuanvideo-community/HunyuanVideo",
                subfolder='tokenizer_2'
            )
            vae = AutoencoderKLHunyuanVideo.from_pretrained(
                "hunyuanvideo-community/HunyuanVideo",
                subfolder='vae',
                torch_dtype=dtype
            ).to(model_device)

            feature_extractor = SiglipImageProcessor.from_pretrained(
                "lllyasviel/flux_redux_bfl", subfolder='feature_extractor'
            )
            image_encoder = SiglipVisionModel.from_pretrained(
                "lllyasviel/flux_redux_bfl",
                subfolder='image_encoder',
                torch_dtype=dtype
            ).to(model_device)

            transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(
                "lllyasviel/FramePackI2V_HY",
                torch_dtype=transformer_dtype
            ).to(model_device)

            print("All models loaded successfully.")
        except Exception as e:
            print(f"Error loading models: {e}")
            print("Retry with float32 on CPU...")
            dtype = torch.float32
            transformer_dtype = torch.float32
            cpu_fallback_mode = True

            text_encoder = LlamaModel.from_pretrained(
                "hunyuanvideo-community/HunyuanVideo",
                subfolder='text_encoder',
                torch_dtype=dtype
            ).to('cpu')
            text_encoder_2 = CLIPTextModel.from_pretrained(
                "hunyuanvideo-community/HunyuanVideo",
                subfolder='text_encoder_2',
                torch_dtype=dtype
            ).to('cpu')
            tokenizer = LlamaTokenizerFast.from_pretrained(
                "hunyuanvideo-community/HunyuanVideo",
                subfolder='tokenizer'
            )
            tokenizer_2 = CLIPTokenizer.from_pretrained(
                "hunyuanvideo-community/HunyuanVideo",
                subfolder='tokenizer_2'
            )
            vae = AutoencoderKLHunyuanVideo.from_pretrained(
                "hunyuanvideo-community/HunyuanVideo",
                subfolder='vae',
                torch_dtype=dtype
            ).to('cpu')

            feature_extractor = SiglipImageProcessor.from_pretrained(
                "lllyasviel/flux_redux_bfl", subfolder='feature_extractor'
            )
            image_encoder = SiglipVisionModel.from_pretrained(
                "lllyasviel/flux_redux_bfl",
                subfolder='image_encoder',
                torch_dtype=dtype
            ).to('cpu')

            transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(
                "lllyasviel/FramePackI2V_HY",
                torch_dtype=transformer_dtype
            ).to('cpu')

            print("Loaded in CPU-only fallback mode.")

        vae.eval()
        text_encoder.eval()
        text_encoder_2.eval()
        image_encoder.eval()
        transformer.eval()

        if not high_vram or cpu_fallback_mode:
            vae.enable_slicing()
            vae.enable_tiling()

        transformer.high_quality_fp32_output_for_inference = True
        print("transformer.high_quality_fp32_output_for_inference = True")

        if not cpu_fallback_mode:
            transformer.to(dtype=transformer_dtype)
            vae.to(dtype=dtype)
            image_encoder.to(dtype=dtype)
            text_encoder.to(dtype=dtype)
            text_encoder_2.to(dtype=dtype)

        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        text_encoder_2.requires_grad_(False)
        image_encoder.requires_grad_(False)
        transformer.requires_grad_(False)

        if torch.cuda.is_available() and not cpu_fallback_mode:
            try:
                if not high_vram:
                    DynamicSwapInstaller.install_model(transformer, device=device)
                    DynamicSwapInstaller.install_model(text_encoder, device=device)
                else:
                    text_encoder.to(device)
                    text_encoder_2.to(device)
                    image_encoder.to(device)
                    vae.to(device)
                    transformer.to(device)
                print(f"Moved models to {device}")
            except Exception as e:
                print(f"Error moving models to {device}: {e}, fallback to CPU")
                cpu_fallback_mode = True

        models_local = {
            'text_encoder': text_encoder,
            'text_encoder_2': text_encoder_2,
            'tokenizer': tokenizer,
            'tokenizer_2': tokenizer_2,
            'vae': vae,
            'feature_extractor': feature_extractor,
            'image_encoder': image_encoder,
            'transformer': transformer
        }

        GPU_INITIALIZED = True
        models.update(models_local)
        print(f"Model load complete. Running mode: {'CPU' if cpu_fallback_mode else 'GPU'}")
        return models
    except Exception as e:
        print(f"Unexpected error in load_models(): {e}")
        traceback.print_exc()
        cpu_fallback_mode = True
        return {}

# GPU 데코레이터 사용 여부 (Spaces 전용)
if IN_HF_SPACE and 'spaces' in globals() and GPU_AVAILABLE:
    try:
        @spaces.GPU
        def initialize_models():
            global GPU_INITIALIZED
            try:
                result = load_models()
                GPU_INITIALIZED = True
                return result
            except Exception as e:
                print(f"Error in @spaces.GPU model init: {e}")
                global cpu_fallback_mode
                cpu_fallback_mode = True
                return load_models()
    except Exception as e:
        print(f"Error creating spaces.GPU decorator: {e}")
        def initialize_models():
            return load_models()
else:
    def initialize_models():
        return load_models()

def get_models():
    """
    Retrieve or load models if not loaded yet.
    """
    global models
    model_loading_key = "__model_loading__"

    if not models:
        if model_loading_key in globals():
            print("Models are loading, please wait...")
            import time
            start_wait = time.time()
            while (not models) and (model_loading_key in globals()):
                time.sleep(0.5)
                if time.time() - start_wait > 60:
                    print("Timed out waiting for model load.")
                    break
            if models:
                return models
        try:
            globals()[model_loading_key] = True
            if IN_HF_SPACE and 'spaces' in globals() and GPU_AVAILABLE and not cpu_fallback_mode:
                try:
                    print("Loading models via @spaces.GPU decorator.")
                    models_local = initialize_models()
                    models.update(models_local)
                except Exception as e:
                    print(f"Error with GPU decorator: {e}, direct load fallback.")
                    models_local = load_models()
                    models.update(models_local)
            else:
                models_local = load_models()
                models.update(models_local)
        except Exception as e:
            print(f"Unexpected error while loading models: {e}")
            models.clear()
        finally:
            if model_loading_key in globals():
                del globals()[model_loading_key]
    return models

stream = AsyncStream()

# 오류 메시지 HTML 생성 함수(영어만)
def create_error_html(error_msg, is_timeout=False):
    """
    Create a user-friendly error message in English only
    """
    if is_timeout:
        if "partial" in error_msg:
            en_msg = "Processing timed out, but partial video has been generated."
        else:
            en_msg = f"Processing timed out: {error_msg}"
    elif "model load" in error_msg.lower():
        en_msg = "Failed to load models. Possibly heavy traffic or GPU issues."
    elif "gpu" in error_msg.lower() or "cuda" in error_msg.lower() or "memory" in error_msg.lower():
        en_msg = "GPU memory insufficient or error. Please try increasing GPU memory or reduce video length."
    elif "sampling" in error_msg.lower():
        if "partial" in error_msg.lower():
            en_msg = "Error during sampling process, but partial video has been generated."
        else:
            en_msg = "Error during sampling process. Unable to generate video."
    elif "timeout" in error_msg.lower():
        en_msg = "Network or model download timed out. Please try again later."
    else:
        en_msg = f"Processing error: {error_msg}"

    return f"""
    <div class="error-message" id="custom-error-container">
        <div>
            <span class="error-icon">⚠️</span> {en_msg}
        </div>
    </div>
    <script>
        // Hide default Gradio error UI
        (function() {{
            const defaultErrorElements = document.querySelectorAll('.error');
            defaultErrorElements.forEach(el => {{
                el.style.display = 'none';
            }});
        }})();
    </script>
    """

@torch.no_grad()
def worker(
    input_image,
    prompt,
    n_prompt,
    seed,
    total_second_length,
    latent_window_size,
    steps,
    cfg,
    gs,
    rs,
    gpu_memory_preservation,
    use_teacache
):
    """
    Actual generation logic in background thread.
    """
    global last_update_time
    last_update_time = time.time()

    total_second_length = min(total_second_length, 5.0)

    try:
        models_local = get_models()
        if not models_local:
            error_msg = "Model load failed. Check logs for details."
            print(error_msg)
            stream.output_queue.push(('error', error_msg))
            stream.output_queue.push(('end', None))
            return

        text_encoder = models_local['text_encoder']
        text_encoder_2 = models_local['text_encoder_2']
        tokenizer = models_local['tokenizer']
        tokenizer_2 = models_local['tokenizer_2']
        vae = models_local['vae']
        feature_extractor = models_local['feature_extractor']
        image_encoder = models_local['image_encoder']
        transformer = models_local['transformer']
    except Exception as e:
        err = f"Error retrieving models: {e}"
        print(err)
        traceback.print_exc()
        stream.output_queue.push(('error', err))
        stream.output_queue.push(('end', None))
        return

    device = 'cuda' if (GPU_AVAILABLE and not cpu_fallback_mode) else 'cpu'
    print(f"Inference device: {device}")

    if cpu_fallback_mode:
        print("CPU fallback mode: reducing some parameters for performance.")
        latent_window_size = min(latent_window_size, 5)
        steps = min(steps, 15)
        total_second_length = min(total_second_length, 2.0)

    total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))

    job_id = generate_timestamp()
    last_output_filename = None
    history_pixels = None
    history_latents = None
    total_generated_latent_frames = 0

    from diffusers_helper.memory import unload_complete_models

    stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Starting ...'))))

    try:
        if not high_vram and not cpu_fallback_mode:
            try:
                unload_complete_models(
                    text_encoder, text_encoder_2, image_encoder, vae, transformer
                )
            except Exception as e:
                print(f"Error unloading models: {e}")

        # Text Encode
        last_update_time = time.time()
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Text encoding...'))))

        try:
            if not high_vram and not cpu_fallback_mode:
                fake_diffusers_current_device(text_encoder, device)
                load_model_as_complete(text_encoder_2, target_device=device)

            llama_vec, clip_l_pooler = encode_prompt_conds(
                prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2
            )

            if cfg == 1:
                llama_vec_n, clip_l_pooler_n = (
                    torch.zeros_like(llama_vec),
                    torch.zeros_like(clip_l_pooler),
                )
            else:
                llama_vec_n, clip_l_pooler_n = encode_prompt_conds(
                    n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2
                )

            llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
            llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)
        except Exception as e:
            err = f"Text encoding error: {e}"
            print(err)
            traceback.print_exc()
            stream.output_queue.push(('error', err))
            stream.output_queue.push(('end', None))
            return

        # Image processing
        last_update_time = time.time()
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Image processing...'))))

        try:
            H, W, C = input_image.shape
            height, width = find_nearest_bucket(H, W, resolution=640)

            if cpu_fallback_mode:
                height = min(height, 320)
                width = min(width, 320)

            input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)
            Image.fromarray(input_image_np).save(os.path.join(outputs_folder, f'{job_id}.png'))

            input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
            input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]
        except Exception as e:
            err = f"Image preprocess error: {e}"
            print(err)
            traceback.print_exc()
            stream.output_queue.push(('error', err))
            stream.output_queue.push(('end', None))
            return

        # VAE encoding
        last_update_time = time.time()
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'VAE encoding...'))))

        try:
            if not high_vram and not cpu_fallback_mode:
                load_model_as_complete(vae, target_device=device)
            start_latent = vae_encode(input_image_pt, vae)
        except Exception as e:
            err = f"VAE encode error: {e}"
            print(err)
            traceback.print_exc()
            stream.output_queue.push(('error', err))
            stream.output_queue.push(('end', None))
            return

        # CLIP Vision
        last_update_time = time.time()
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'CLIP Vision encode...'))))

        try:
            if not high_vram and not cpu_fallback_mode:
                load_model_as_complete(image_encoder, target_device=device)
            image_encoder_output = hf_clip_vision_encode(
                input_image_np, feature_extractor, image_encoder
            )
            image_encoder_last_hidden_state = image_encoder_output.last_hidden_state
        except Exception as e:
            err = f"CLIP Vision encode error: {e}"
            print(err)
            traceback.print_exc()
            stream.output_queue.push(('error', err))
            stream.output_queue.push(('end', None))
            return

        # Convert dtype
        try:
            llama_vec = llama_vec.to(transformer.dtype)
            llama_vec_n = llama_vec_n.to(transformer.dtype)
            clip_l_pooler = clip_l_pooler.to(transformer.dtype)
            clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
            image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)
        except Exception as e:
            err = f"Data type conversion error: {e}"
            print(err)
            traceback.print_exc()
            stream.output_queue.push(('error', err))
            stream.output_queue.push(('end', None))
            return

        # Sampling
        last_update_time = time.time()
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Start sampling...'))))

        rnd = torch.Generator("cpu").manual_seed(seed)
        num_frames = latent_window_size * 4 - 3

        try:
            history_latents = torch.zeros(
                size=(1, 16, 1 + 2 + 16, height // 8, width // 8),
                dtype=torch.float32
            ).cpu()
            history_pixels = None
            total_generated_latent_frames = 0
        except Exception as e:
            err = f"Init history state error: {e}"
            print(err)
            traceback.print_exc()
            stream.output_queue.push(('error', err))
            stream.output_queue.push(('end', None))
            return

        latent_paddings = list(reversed(range(total_latent_sections)))
        if total_latent_sections > 4:
            # Some heuristic to flatten out large steps
            latent_paddings = [3] + [2]*(total_latent_sections - 3) + [1, 0]

        for latent_padding in latent_paddings:
            last_update_time = time.time()
            is_last_section = (latent_padding == 0)
            latent_padding_size = latent_padding * latent_window_size

            if stream.input_queue.top() == 'end':
                # If user requests end, save partial video if possible
                if history_pixels is not None and total_generated_latent_frames > 0:
                    try:
                        outname = os.path.join(
                            outputs_folder, f'{job_id}_final_{total_generated_latent_frames}.mp4'
                        )
                        save_bcthw_as_mp4(history_pixels, outname, fps=30)
                        stream.output_queue.push(('file', outname))
                    except Exception as e:
                        print(f"Error saving final partial video: {e}")
                stream.output_queue.push(('end', None))
                return

            print(f"latent_padding_size={latent_padding_size}, last_section={is_last_section}")

            try:
                indices = torch.arange(
                    0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])
                ).unsqueeze(0)
                (
                    clean_latent_indices_pre,
                    blank_indices,
                    latent_indices,
                    clean_latent_indices_post,
                    clean_latent_2x_indices,
                    clean_latent_4x_indices
                ) = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
                clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

                clean_latents_pre = start_latent.to(history_latents)
                clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16].split([1, 2, 16], dim=2)
                clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)
            except Exception as e:
                err = f"Sampling data prep error: {e}"
                print(err)
                traceback.print_exc()
                if last_output_filename:
                    stream.output_queue.push(('file', last_output_filename))
                continue

            if not high_vram and not cpu_fallback_mode:
                try:
                    unload_complete_models()
                    move_model_to_device_with_memory_preservation(
                        transformer, target_device=device, preserved_memory_gb=gpu_memory_preservation
                    )
                except Exception as e:
                    print(f"Error moving transformer to GPU: {e}")

            if use_teacache and not cpu_fallback_mode:
                try:
                    transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
                except Exception as e:
                    print(f"Error init teacache: {e}")
                    transformer.initialize_teacache(enable_teacache=False)
            else:
                transformer.initialize_teacache(enable_teacache=False)

            def callback(d):
                global last_update_time
                last_update_time = time.time()
                try:
                    if stream.input_queue.top() == 'end':
                        stream.output_queue.push(('end', None))
                        raise KeyboardInterrupt('User requested stop.')
                    preview = d['denoised']
                    preview = vae_decode_fake(preview)
                    preview = (preview * 255.0).cpu().numpy().clip(0,255).astype(np.uint8)
                    preview = einops.rearrange(preview, 'b c t h w -> (b h) (t w) c')

                    curr_step = d['i'] + 1
                    percentage = int(100.0 * curr_step / steps)
                    hint = f'Sampling {curr_step}/{steps}'
                    desc = f'Total frames so far: {int(max(0, total_generated_latent_frames * 4 - 3))}'
                    barhtml = make_progress_bar_html(percentage, hint)
                    stream.output_queue.push(('progress', (preview, desc, barhtml)))
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    print(f"Callback error: {e}")
                return

            try:
                print(f"Sampling with device={device}, dtype={transformer.dtype}, teacache={use_teacache}")
                from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan

                try:
                    generated_latents = sample_hunyuan(
                        transformer=transformer,
                        sampler='unipc',
                        width=width,
                        height=height,
                        frames=num_frames,
                        real_guidance_scale=cfg,
                        distilled_guidance_scale=gs,
                        guidance_rescale=rs,
                        num_inference_steps=steps,
                        generator=rnd,
                        prompt_embeds=llama_vec,
                        prompt_embeds_mask=llama_attention_mask,
                        prompt_poolers=clip_l_pooler,
                        negative_prompt_embeds=llama_vec_n,
                        negative_prompt_embeds_mask=llama_attention_mask_n,
                        negative_prompt_poolers=clip_l_pooler_n,
                        device=device,
                        dtype=transformer.dtype,
                        image_embeddings=image_encoder_last_hidden_state,
                        latent_indices=latent_indices,
                        clean_latents=clean_latents,
                        clean_latent_indices=clean_latent_indices,
                        clean_latents_2x=clean_latents_2x,
                        clean_latent_2x_indices=clean_latent_2x_indices,
                        clean_latents_4x=clean_latents_4x,
                        clean_latent_4x_indices=clean_latent_4x_indices,
                        callback=callback
                    )
                except KeyboardInterrupt as e:
                    print(f"User interrupt: {e}")
                    if last_output_filename:
                        stream.output_queue.push(('file', last_output_filename))
                        err = "User stopped generation, partial video returned."
                    else:
                        err = "User stopped generation, no video produced."
                    stream.output_queue.push(('error', err))
                    stream.output_queue.push(('end', None))
                    return
            except Exception as e:
                print(f"Sampling error: {e}")
                traceback.print_exc()
                if last_output_filename:
                    stream.output_queue.push(('file', last_output_filename))
                    err = f"Error during sampling, partial video returned: {e}"
                    stream.output_queue.push(('error', err))
                else:
                    err = f"Error during sampling, no video produced: {e}"
                    stream.output_queue.push(('error', err))
                stream.output_queue.push(('end', None))
                return

            try:
                if is_last_section:
                    generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)
                total_generated_latent_frames += int(generated_latents.shape[2])
                history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)
            except Exception as e:
                err = f"Post-latent processing error: {e}"
                print(err)
                traceback.print_exc()
                if last_output_filename:
                    stream.output_queue.push(('file', last_output_filename))
                stream.output_queue.push(('error', err))
                stream.output_queue.push(('end', None))
                return

            if not high_vram and not cpu_fallback_mode:
                try:
                    offload_model_from_device_for_memory_preservation(
                        transformer, target_device=device, preserved_memory_gb=8
                    )
                    load_model_as_complete(vae, target_device=device)
                except Exception as e:
                    print(f"Model memory manage error: {e}")

            try:
                real_history_latents = history_latents[:, :, :total_generated_latent_frames]
            except Exception as e:
                err = f"History latents slice error: {e}"
                print(err)
                if last_output_filename:
                    stream.output_queue.push(('file', last_output_filename))
                continue

            try:
                # VAE decode
                if history_pixels is None:
                    history_pixels = vae_decode(real_history_latents, vae).cpu()
                else:
                    # Overlap logic
                    section_latent_frames = (
                        (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
                    )
                    overlapped_frames = latent_window_size * 4 - 3
                    current_pixels = vae_decode(real_history_latents[:, :, :section_latent_frames], vae).cpu()
                    history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)
                
                output_filename = os.path.join(
                    outputs_folder, f'{job_id}_{total_generated_latent_frames}.mp4'
                )
                save_bcthw_as_mp4(history_pixels, output_filename, fps=30)
                last_output_filename = output_filename
                stream.output_queue.push(('file', output_filename))
            except Exception as e:
                print(f"Video decode/save error: {e}")
                traceback.print_exc()
                if last_output_filename:
                    stream.output_queue.push(('file', last_output_filename))
                err = f"Video decode/save error: {e}"
                stream.output_queue.push(('error', err))
                continue

            if is_last_section:
                break
    except Exception as e:
        print(f"Outer error: {e}, type={type(e)}")
        traceback.print_exc()
        if not high_vram and not cpu_fallback_mode:
            try:
                unload_complete_models(
                    text_encoder, text_encoder_2, image_encoder, vae, transformer
                )
            except Exception as ue:
                print(f"Unload error: {ue}")

        if last_output_filename:
            stream.output_queue.push(('file', last_output_filename))
        err = f"Error in worker: {e}"
        stream.output_queue.push(('error', err))

    print("Worker finished, pushing 'end'.")
    stream.output_queue.push(('end', None))

# 최종 처리 함수 (Spaces GPU decorator or normal)
if IN_HF_SPACE and 'spaces' in globals():
    @spaces.GPU
    def process_with_gpu(
        input_image, prompt, n_prompt, seed,
        total_second_length, latent_window_size, steps,
        cfg, gs, rs, gpu_memory_preservation, use_teacache
    ):
        global stream
        assert input_image is not None, "No input image given."

        # Initialize UI state
        yield None, None, "", "", gr.update(interactive=False), gr.update(interactive=True)
        try:
            stream = AsyncStream()
            async_run(
                worker,
                input_image, prompt, n_prompt, seed,
                total_second_length, latent_window_size, steps, cfg, gs, rs,
                gpu_memory_preservation, use_teacache
            )

            output_filename = None
            prev_output_filename = None
            error_message = None

            while True:
                try:
                    flag, data = stream.output_queue.next()
                    if flag == 'file':
                        output_filename = data
                        prev_output_filename = output_filename
                        yield output_filename, gr.update(), gr.update(), '', gr.update(interactive=False), gr.update(interactive=True)
                    elif flag == 'progress':
                        preview, desc, html = data
                        yield gr.update(), gr.update(visible=True, value=preview), desc, html, gr.update(interactive=False), gr.update(interactive=True)
                    elif flag == 'error':
                        error_message = data
                        print(f"Got error: {error_message}")
                    elif flag == 'end':
                        if output_filename is None and prev_output_filename:
                            output_filename = prev_output_filename
                        if error_message:
                            err_html = create_error_html(error_message)
                            yield (
                                output_filename, gr.update(visible=False), gr.update(),
                                err_html, gr.update(interactive=True), gr.update(interactive=False)
                            )
                        else:
                            yield (
                                output_filename, gr.update(visible=False), gr.update(),
                                '', gr.update(interactive=True), gr.update(interactive=False)
                            )
                        break
                except Exception as e:
                    print(f"Loop error: {e}")
                    if (time.time() - last_update_time) > 60:
                        print("No updates for 60 seconds, possible hang or timeout.")
                        if prev_output_filename:
                            err_html = create_error_html("partial video has been generated", is_timeout=True)
                            yield (
                                prev_output_filename, gr.update(visible=False), gr.update(),
                                err_html, gr.update(interactive=True), gr.update(interactive=False)
                            )
                        else:
                            err_html = create_error_html(f"Processing timed out: {e}", is_timeout=True)
                            yield (
                                None, gr.update(visible=False), gr.update(),
                                err_html, gr.update(interactive=True), gr.update(interactive=False)
                            )
                        break
        except Exception as e:
            print(f"Start process error: {e}")
            traceback.print_exc()
            err_html = create_error_html(str(e))
            yield None, gr.update(visible=False), gr.update(), err_html, gr.update(interactive=True), gr.update(interactive=False)

    process = process_with_gpu
else:
    def process(
        input_image, prompt, n_prompt, seed,
        total_second_length, latent_window_size, steps,
        cfg, gs, rs, gpu_memory_preservation, use_teacache
    ):
        global stream
        assert input_image is not None, "No input image given."

        yield None, None, "", "", gr.update(interactive=False), gr.update(interactive=True)
        try:
            stream = AsyncStream()
            async_run(
                worker,
                input_image, prompt, n_prompt, seed,
                total_second_length, latent_window_size, steps, cfg, gs, rs,
                gpu_memory_preservation, use_teacache
            )

            output_filename = None
            prev_output_filename = None
            error_message = None

            while True:
                try:
                    flag, data = stream.output_queue.next()
                    if flag == 'file':
                        output_filename = data
                        prev_output_filename = output_filename
                        yield output_filename, gr.update(), gr.update(), '', gr.update(interactive=False), gr.update(interactive=True)
                    elif flag == 'progress':
                        preview, desc, html = data
                        yield gr.update(), gr.update(visible=True, value=preview), desc, html, gr.update(interactive=False), gr.update(interactive=True)
                    elif flag == 'error':
                        error_message = data
                        print(f"Got error: {error_message}")
                    elif flag == 'end':
                        if output_filename is None and prev_output_filename:
                            output_filename = prev_output_filename
                        if error_message:
                            err_html = create_error_html(error_message)
                            yield (
                                output_filename, gr.update(visible=False), gr.update(),
                                err_html, gr.update(interactive=True), gr.update(interactive=False)
                            )
                        else:
                            yield (
                                output_filename, gr.update(visible=False), gr.update(),
                                '', gr.update(interactive=True), gr.update(interactive=False)
                            )
                        break
                except Exception as e:
                    print(f"Loop error: {e}")
                    if (time.time() - last_update_time) > 60:
                        print("No update for 60 seconds, possible hang or timeout.")
                        if prev_output_filename:
                            err_html = create_error_html("partial video has been generated", is_timeout=True)
                            yield (
                                prev_output_filename, gr.update(visible=False), gr.update(),
                                err_html, gr.update(interactive=True), gr.update(interactive=False)
                            )
                        else:
                            err_html = create_error_html(f"Processing timed out: {e}", is_timeout=True)
                            yield (
                                None, gr.update(visible=False), gr.update(),
                                err_html, gr.update(interactive=True), gr.update(interactive=False)
                            )
                        break
        except Exception as e:
            print(f"Start process error: {e}")
            traceback.print_exc()
            err_html = create_error_html(str(e))
            yield None, gr.update(visible=False), gr.update(), err_html, gr.update(interactive=True), gr.update(interactive=False)

def end_process():
    """
    Stop generation by pushing 'end' to the worker queue
    """
    print("User clicked stop, sending 'end' signal...")
    global stream
    if 'stream' in globals() and stream is not None:
        try:
            top_signal = stream.input_queue.top()
            print(f"Queue top signal = {top_signal}")
        except Exception as e:
            print(f"Error checking queue top: {e}")
        try:
            stream.input_queue.push('end')
            print("Pushed 'end' successfully.")
        except Exception as e:
            print(f"Error pushing 'end': {e}")
    else:
        print("Warning: Stream not initialized, cannot stop.")
    return None

# 예시 빠른 프롬프트
quick_prompts = [
    ["The girl dances gracefully, with clear movements, full of charm."],
    ["A character doing some simple body movements."]
]

# CSS
def make_custom_css():
    base_progress_css = make_progress_bar_css()
    enhanced_css = """
    /* Visual & layout improvement */
    body {
        background: #f9fafb !important; 
        font-family: "Noto Sans", sans-serif;
    }
    #app-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 1rem;
        position: relative;
    }
    #app-container h1 {
        color: #2d3748;
        margin-bottom: 1.2rem;
        font-weight: 700;
    }
    .gr-panel {
        background: #fff;
        border: 1px solid #cbd5e0;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    .button-container button {
        min-height: 45px;
        font-size: 1rem;
        font-weight: 600;
    }
    .button-container button#start-button {
        background-color: #3182ce !important;
        color: #fff !important;
    }
    .button-container button#stop-button {
        background-color: #e53e3e !important;
        color: #fff !important;
    }
    .button-container button:hover {
        filter: brightness(0.95);
    }
    .preview-container, .video-container {
        border: 1px solid #cbd5e0;
        border-radius: 8px;
        overflow: hidden;
    }
    .progress-container {
        margin-top: 15px;
        margin-bottom: 15px;
    }
    .error-message {
        background-color: #fff5f5;
        border: 1px solid #fed7d7;
        color: #e53e3e;
        padding: 10px;
        border-radius: 4px;
        margin-top: 10px;
    }
    .error-icon {
        color: #e53e3e;
        margin-right: 8px;
    }
    #error-message {
        color: #ff4444;
        font-weight: bold;
        padding: 10px;
        border-radius: 4px;
        margin-top: 10px;
    }
    @media (max-width: 768px) {
        #app-container {
            padding: 0.5rem;
        }
        .mobile-full-width {
            flex-direction: column !important;
        }
        .mobile-full-width > .gr-block {
            width: 100% !important;
        }
    }
    """
    return base_progress_css + enhanced_css

css = make_custom_css()

# Gradio UI
block = gr.Blocks(css=css).queue()
with block:
    # 상단 제목
    gr.HTML("<div id='app-container'><h1>FramePack - Image to Video Generation</h1></div>")

    with gr.Row(elem_classes="mobile-full-width"):
        with gr.Column(scale=1, elem_classes="gr-panel"):
            input_image = gr.Image(
                label="Upload Image",
                sources='upload',
                type="numpy",
                elem_id="input-image",
                height=320
            )
            prompt = gr.Textbox(label="Prompt", value='', elem_id="prompt-input")

            example_quick_prompts = gr.Dataset(
                samples=quick_prompts,
                label="Quick Prompts",
                samples_per_page=1000,
                components=[prompt]
            )
            example_quick_prompts.click(
                fn=lambda x: x[0],
                inputs=[example_quick_prompts],
                outputs=prompt,
                show_progress=False,
                queue=False
            )
        with gr.Column(scale=1, elem_classes="gr-panel"):
            with gr.Row(elem_classes="button-container"):
                start_button = gr.Button(
                    value="Generate",
                    elem_id="start-button",
                    variant="primary"
                )
                end_button = gr.Button(
                    value="Stop",
                    elem_id="stop-button",
                    interactive=False
                )
            
            result_video = gr.Video(
                label="Generated Video",
                autoplay=True,
                loop=True,
                height=320,
                elem_classes="video-container",
                elem_id="result-video"
            )
            preview_image = gr.Image(
                label="Preview",
                visible=False,
                height=150,
                elem_classes="preview-container"
            )

            gr.Markdown(get_translation("sampling_note"))
            
            with gr.Group(elem_classes="progress-container"):
                progress_desc = gr.Markdown('')
                progress_bar = gr.HTML('')
            
            error_message = gr.HTML('', elem_id='error-message', visible=True)

    # 고급 파라미터 Accordion
    with gr.Accordion("Advanced Settings", open=False, elem_classes="gr-panel"):
        use_teacache = gr.Checkbox(
            label=get_translation("use_teacache"),
            value=True,
            info=get_translation("teacache_info")
        )
        n_prompt = gr.Textbox(label=get_translation("negative_prompt"), value="", visible=False)
        seed = gr.Number(
            label=get_translation("seed"),
            value=31337,
            precision=0
        )
        total_second_length = gr.Slider(
            label=get_translation("video_length"),
            minimum=1,
            maximum=5,
            value=5,
            step=0.1
        )
        latent_window_size = gr.Slider(
            label=get_translation("latent_window"),
            minimum=1,
            maximum=33,
            value=9,
            step=1,
            visible=False
        )
        steps = gr.Slider(
            label=get_translation("steps"),
            minimum=1,
            maximum=100,
            value=25,
            step=1,
            info=get_translation("steps_info")
        )
        cfg = gr.Slider(
            label=get_translation("cfg_scale"),
            minimum=1.0,
            maximum=32.0,
            value=1.0,
            step=0.01,
            visible=False
        )
        gs = gr.Slider(
            label=get_translation("distilled_cfg"),
            minimum=1.0,
            maximum=32.0,
            value=10.0,
            step=0.01,
            info=get_translation("distilled_cfg_info")
        )
        rs = gr.Slider(
            label=get_translation("cfg_rescale"),
            minimum=0.0,
            maximum=1.0,
            value=0.0,
            step=0.01,
            visible=False
        )
        gpu_memory_preservation = gr.Slider(
            label=get_translation("gpu_memory"),
            minimum=6,
            maximum=128,
            value=6,
            step=0.1,
            info=get_translation("gpu_memory_info")
        )

    # 처리 함수 연결
    ips = [
        input_image, prompt, n_prompt, seed,
        total_second_length, latent_window_size, steps,
        cfg, gs, rs, gpu_memory_preservation, use_teacache
    ]
    start_button.click(
        fn=process,
        inputs=ips,
        outputs=[result_video, preview_image, progress_desc, progress_bar, start_button, end_button]
    )
    end_button.click(fn=end_process)

block.launch()
