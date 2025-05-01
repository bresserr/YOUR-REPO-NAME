from diffusers_helper.hf_login import login

import os
import threading
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json

os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download')))

# 영어/한국어 번역 딕셔너리
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
    },
    "ko": {
        "title": "FramePack - 이미지에서 동영상 생성",
        "upload_image": "이미지 업로드",
        "prompt": "프롬프트",
        "quick_prompts": "빠른 프롬프트 목록",
        "start_generation": "생성 시작",
        "stop_generation": "생성 중지",
        "use_teacache": "TeaCache 사용",
        "teacache_info": "더 빠른 속도를 제공하지만 손가락이나 손 생성 품질이 약간 떨어질 수 있습니다.",
        "negative_prompt": "부정 프롬프트",
        "seed": "랜덤 시드",
        "video_length": "동영상 길이 (최대 5초)",
        "latent_window": "잠재 윈도우 크기",
        "steps": "추론 스텝 수",
        "steps_info": "이 값을 변경하는 것은 권장되지 않습니다.",
        "cfg_scale": "CFG 스케일",
        "distilled_cfg": "증류된 CFG 스케일",
        "distilled_cfg_info": "이 값을 변경하는 것은 권장되지 않습니다.",
        "cfg_rescale": "CFG 재스케일",
        "gpu_memory": "GPU 메모리 보존 (GB) (값이 클수록 속도가 느려짐)",
        "gpu_memory_info": "OOM 오류가 발생하면 이 값을 더 크게 설정하십시오. 값이 클수록 속도가 느려집니다.",
        "next_latents": "다음 잠재 변수",
        "generated_video": "생성된 동영상",
        "sampling_note": "주의: 역순 샘플링 때문에, 종료 동작이 시작 동작보다 먼저 생성됩니다. 시작 동작이 동영상에 나타나지 않으면 기다려 주십시오. 나중에 생성됩니다.",
        "error_message": "오류 메시지",
        "processing_error": "처리 중 오류 발생",
        "network_error": "네트워크 연결이 불안정하여 모델 다운로드가 시간 초과되었습니다. 나중에 다시 시도해 주십시오.",
        "memory_error": "GPU 메모리가 부족합니다. GPU 메모리 보존 값을 늘리거나 동영상 길이를 줄여보세요.",
        "model_error": "모델 로드에 실패했습니다. 네트워크 문제 또는 서버 부하가 높을 수 있습니다. 나중에 다시 시도해 주십시오.",
        "partial_video": "처리 중 오류가 발생했지만 일부 동영상이 생성되었습니다.",
        "processing_interrupt": "처리 중 중단되었지만 일부 동영상이 생성되었습니다."
    }
}

# 다국어 텍스트를 반환하는 함수
def get_translation(key, lang="en"):
    if lang in translations and key in translations[lang]:
        return translations[lang][key]
    # 기본값(영어) 반환
    return translations["en"].get(key, key)

# 디폴트 언어를 영어로 설정
current_language = "en"

# 언어 전환 함수
def switch_language():
    global current_language
    current_language = "ko" if current_language == "en" else "en"
    return current_language

import gradio as gr
import torch
import traceback
import einops
import safetensors.torch as sf
import numpy as np
import math

# Spaces 환경 체크
IN_HF_SPACE = os.environ.get('SPACE_ID') is not None

# GPU 사용 여부 기록
GPU_AVAILABLE = False
GPU_INITIALIZED = False
last_update_time = time.time()

# Spaces 환경이라면, spaces 모듈 불러오기 시도
if IN_HF_SPACE:
    try:
        import spaces
        print("Hugging Face Space 환경에서 실행 중, spaces 모듈을 불러왔습니다.")
        
        # GPU 사용 가능 여부 확인
        try:
            GPU_AVAILABLE = torch.cuda.is_available()
            print(f"GPU available: {GPU_AVAILABLE}")
            if GPU_AVAILABLE:
                print(f"GPU device name: {torch.cuda.get_device_name(0)}")
                print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9} GB")
                
                # 작은 테스트 연산으로 실제 GPU 동작 확인
                test_tensor = torch.zeros(1, device='cuda')
                test_tensor = test_tensor + 1
                del test_tensor
                print("GPU 테스트 연산 성공")
            else:
                print("경고: CUDA는 가능하다고 하나 실제 GPU 디바이스를 찾을 수 없습니다.")
        except Exception as e:
            GPU_AVAILABLE = False
            print(f"GPU 확인 중 오류 발생: {e}")
            print("CPU 모드로 진행합니다.")
    except ImportError:
        print("spaces 모듈을 불러올 수 없습니다. Spaces 환경이 아닐 수 있습니다.")
        GPU_AVAILABLE = torch.cuda.is_available()

from PIL import Image
from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, state_dict_weighted_merge, state_dict_offset_merge, generate_timestamp
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, DynamicSwapInstaller, unload_complete_models, load_model_as_complete, IN_HF_SPACE as MEMORY_IN_HF_SPACE
from diffusers_helper.thread_utils import AsyncStream, async_run
from diffusers_helper.gradio.progress_bar import make_progress_bar_css, make_progress_bar_html
from transformers import SiglipImageProcessor, SiglipVisionModel
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket

outputs_folder = './outputs/'
os.makedirs(outputs_folder, exist_ok=True)

# Spaces 환경이 아닐 경우, VRAM 확인
if not IN_HF_SPACE:
    try:
        if torch.cuda.is_available():
            free_mem_gb = get_cuda_free_memory_gb(gpu)
            print(f'남은 VRAM: {free_mem_gb} GB')
        else:
            free_mem_gb = 6.0  # 기본값
            print("CUDA를 사용할 수 없으므로 기본 메모리 설정을 사용합니다.")
    except Exception as e:
        free_mem_gb = 6.0
        print(f"CUDA 메모리 확인 중 오류 발생: {e} / 기본값 사용")
        
    high_vram = free_mem_gb > 60
    print(f'high_vram 모드: {high_vram}')
else:
    # Spaces 환경에서 기본값 설정
    print("Spaces 환경에서 기본 메모리 설정 사용")
    try:
        if GPU_AVAILABLE:
            free_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 * 0.9
            high_vram = free_mem_gb > 10  # 조금 더 보수적으로 설정
        else:
            free_mem_gb = 6.0
            high_vram = False
    except Exception as e:
        print(f"GPU 메모리 확인 중 오류: {e}")
        free_mem_gb = 6.0
        high_vram = False
    
    print(f'GPU 메모리: {free_mem_gb:.2f} GB, High-VRAM 모드: {high_vram}')

# 전역 모델 참조
models = {}
cpu_fallback_mode = not GPU_AVAILABLE  # GPU가 불가능하면 CPU 모드로

def load_models():
    global models, cpu_fallback_mode, GPU_INITIALIZED
    
    if GPU_INITIALIZED:
        print("모델이 이미 로드되었습니다. 다시 로드하지 않습니다.")
        return models
    
    print("모델 로드를 시작합니다...")

    try:
        device = 'cuda' if GPU_AVAILABLE and not cpu_fallback_mode else 'cpu'
        model_device = 'cpu'  # 우선 CPU에 로드

        # 기본적으로 GPU면 float16, CPU면 float32
        dtype = torch.float16 if GPU_AVAILABLE else torch.float32
        transformer_dtype = torch.bfloat16 if GPU_AVAILABLE else torch.float32
        
        print(f"사용 디바이스: {device}, vae/text encoder dtype: {dtype}, transformer dtype: {transformer_dtype}")
        
        try:
            text_encoder = LlamaModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder', torch_dtype=dtype).to(model_device)
            text_encoder_2 = CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder_2', torch_dtype=dtype).to(model_device)
            tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
            tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')
            vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=dtype).to(model_device)

            feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
            image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=dtype).to(model_device)

            transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePackI2V_HY', torch_dtype=transformer_dtype).to(model_device)
            
            print("모든 모델을 성공적으로 로드했습니다.")
        except Exception as e:
            print(f"모델 로드 중 오류 발생: {e}")
            print("정밀도를 낮춰 다시 로드합니다...")

            dtype = torch.float32
            transformer_dtype = torch.float32
            cpu_fallback_mode = True
            
            text_encoder = LlamaModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder', torch_dtype=dtype).to('cpu')
            text_encoder_2 = CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder_2', torch_dtype=dtype).to('cpu')
            tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
            tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')
            vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=dtype).to('cpu')

            feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
            image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=dtype).to('cpu')

            transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePackI2V_HY', torch_dtype=transformer_dtype).to('cpu')
            
            print("CPU 모드로 모델 로드 성공")

        vae.eval()
        text_encoder.eval()
        text_encoder_2.eval()
        image_encoder.eval()
        transformer.eval()

        if not high_vram or cpu_fallback_mode:
            vae.enable_slicing()
            vae.enable_tiling()

        transformer.high_quality_fp32_output_for_inference = True
        print('transformer.high_quality_fp32_output_for_inference = True')

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
                    # 메모리 최적화
                    DynamicSwapInstaller.install_model(transformer, device=device)
                    DynamicSwapInstaller.install_model(text_encoder, device=device)
                else:
                    text_encoder.to(device)
                    text_encoder_2.to(device)
                    image_encoder.to(device)
                    vae.to(device)
                    transformer.to(device)
                print(f"모델을 {device}로 이동 완료")
            except Exception as e:
                print(f"{device}로 모델 이동 중 오류 발생: {e}")
                print("CPU 모드로 전환")
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
        print(f"모델 로드 완료. 현재 실행 모드: {'CPU' if cpu_fallback_mode else 'GPU'}")
        return models
    except Exception as e:
        print(f"모델 로드 중 예상치 못한 오류가 발생: {e}")
        traceback.print_exc()
        
        error_info = {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "cuda_available": torch.cuda.is_available(),
            "device": "cpu" if cpu_fallback_mode else "cuda",
        }
        
        try:
            with open(os.path.join(outputs_folder, "error_log.txt"), "w") as f:
                f.write(str(error_info))
        except:
            pass
            
        cpu_fallback_mode = True
        return {}

if IN_HF_SPACE and 'spaces' in globals() and GPU_AVAILABLE:
    try:
        @spaces.GPU
        def initialize_models():
            """@spaces.GPU 환경에서 모델을 초기화"""
            global GPU_INITIALIZED
            try:
                result = load_models()
                GPU_INITIALIZED = True
                return result
            except Exception as e:
                print(f"@spaces.GPU 모델 초기화 중 오류: {e}")
                traceback.print_exc()
                global cpu_fallback_mode
                cpu_fallback_mode = True
                return load_models()
    except Exception as e:
        print(f"spaces.GPU 데코레이터 생성 중 오류: {e}")
        def initialize_models():
            return load_models()

def get_models():
    """모델을 불러오거나 이미 불러왔다면 반환"""
    global models, GPU_INITIALIZED
    
    model_loading_key = "__model_loading__"
    
    if not models:
        if model_loading_key in globals():
            print("모델 로딩 중입니다. 대기 중...")
            import time
            start_wait = time.time()
            while not models and model_loading_key in globals():
                time.sleep(0.5)
                if time.time() - start_wait > 60:
                    print("모델 로딩 대기 시간 초과")
                    break
            
            if models:
                return models
            
        try:
            globals()[model_loading_key] = True
            
            if IN_HF_SPACE and 'spaces' in globals() and GPU_AVAILABLE and not cpu_fallback_mode:
                try:
                    print("GPU 데코레이터(@spaces.GPU)로 모델 로딩 시도")
                    models_local = initialize_models()
                    models.update(models_local)
                except Exception as e:
                    print(f"GPU 데코레이터 로딩 실패: {e} / 직접 로딩 시도")
                    models_local = load_models()
                    models.update(models_local)
            else:
                print("모델 직접 로딩 시도")
                models_local = load_models()
                models.update(models_local)
        except Exception as e:
            print(f"모델 로드 중 오류: {e}")
            traceback.print_exc()
            models.clear()
        finally:
            if model_loading_key in globals():
                del globals()[model_loading_key]
    
    return models

stream = AsyncStream()

@torch.no_grad()
def worker(input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache):
    global last_update_time
    last_update_time = time.time()
    
    total_second_length = min(total_second_length, 5.0)
    
    try:
        models_local = get_models()
        if not models_local:
            error_msg = "모델 로드에 실패했습니다. 로그를 확인하세요."
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
        error_msg = f"모델 가져오기 실패: {e}"
        print(error_msg)
        traceback.print_exc()
        stream.output_queue.push(('error', error_msg))
        stream.output_queue.push(('end', None))
        return
    
    device = 'cuda' if GPU_AVAILABLE and not cpu_fallback_mode else 'cpu'
    print(f"추론 디바이스: {device}")

    if cpu_fallback_mode:
        print("CPU 모드에서 일부 파라미터를 축소합니다.")
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

    stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Starting ...'))))

    try:
        if not high_vram and not cpu_fallback_mode:
            try:
                unload_complete_models(
                    text_encoder, text_encoder_2, image_encoder, vae, transformer
                )
            except Exception as e:
                print(f"모델 언로드 중 오류: {e}")
        
        # 텍스트 인코딩
        last_update_time = time.time()
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Text encoding ...'))))

        try:
            if not high_vram and not cpu_fallback_mode:
                fake_diffusers_current_device(text_encoder, device)
                load_model_as_complete(text_encoder_2, target_device=device)

            llama_vec, clip_l_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

            if cfg == 1:
                llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
            else:
                llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

            llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
            llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)
        except Exception as e:
            error_msg = f"텍스트 인코딩 오류: {e}"
            print(error_msg)
            traceback.print_exc()
            stream.output_queue.push(('error', error_msg))
            stream.output_queue.push(('end', None))
            return

        # 입력 이미지 처리
        last_update_time = time.time()
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Image processing ...'))))

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
            error_msg = f"이미지 전처리 오류: {e}"
            print(error_msg)
            traceback.print_exc()
            stream.output_queue.push(('error', error_msg))
            stream.output_queue.push(('end', None))
            return

        # VAE 인코딩
        last_update_time = time.time()
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'VAE encoding ...'))))

        try:
            if not high_vram and not cpu_fallback_mode:
                load_model_as_complete(vae, target_device=device)

            start_latent = vae_encode(input_image_pt, vae)
        except Exception as e:
            error_msg = f"VAE 인코딩 오류: {e}"
            print(error_msg)
            traceback.print_exc()
            stream.output_queue.push(('error', error_msg))
            stream.output_queue.push(('end', None))
            return

        # CLIP Vision 인코딩
        last_update_time = time.time()
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'CLIP Vision encoding ...'))))

        try:
            if not high_vram and not cpu_fallback_mode:
                load_model_as_complete(image_encoder, target_device=device)

            image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
            image_encoder_last_hidden_state = image_encoder_output.last_hidden_state
        except Exception as e:
            error_msg = f"CLIP Vision 인코딩 오류: {e}"
            print(error_msg)
            traceback.print_exc()
            stream.output_queue.push(('error', error_msg))
            stream.output_queue.push(('end', None))
            return

        # dtype 변환
        try:
            llama_vec = llama_vec.to(transformer.dtype)
            llama_vec_n = llama_vec_n.to(transformer.dtype)
            clip_l_pooler = clip_l_pooler.to(transformer.dtype)
            clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
            image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)
        except Exception as e:
            error_msg = f"dtype 변환 오류: {e}"
            print(error_msg)
            traceback.print_exc()
            stream.output_queue.push(('error', error_msg))
            stream.output_queue.push(('end', None))
            return

        # 샘플링 진행
        last_update_time = time.time()
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Start sampling ...'))))

        rnd = torch.Generator("cpu").manual_seed(seed)
        num_frames = latent_window_size * 4 - 3

        try:
            history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
            history_pixels = None
            total_generated_latent_frames = 0
        except Exception as e:
            error_msg = f"히스토리 상태 초기화 오류: {e}"
            print(error_msg)
            traceback.print_exc()
            stream.output_queue.push(('error', error_msg))
            stream.output_queue.push(('end', None))
            return

        latent_paddings = reversed(range(total_latent_sections))
        if total_latent_sections > 4:
            latent_paddings = [3] + [2]*(total_latent_sections - 3) + [1, 0]

        for latent_padding in latent_paddings:
            last_update_time = time.time()
            is_last_section = latent_padding == 0
            latent_padding_size = latent_padding * latent_window_size

            if stream.input_queue.top() == 'end':
                # 중단 신호 수신 시 부분 결과 반환
                if history_pixels is not None and total_generated_latent_frames > 0:
                    try:
                        output_filename = os.path.join(outputs_folder, f'{job_id}_final_{total_generated_latent_frames}.mp4')
                        save_bcthw_as_mp4(history_pixels, output_filename, fps=30)
                        stream.output_queue.push(('file', output_filename))
                    except Exception as e:
                        print(f"마지막 비디오 저장 오류: {e}")
                
                stream.output_queue.push(('end', None))
                return

            print(f'latent_padding_size = {latent_padding_size}, is_last_section = {is_last_section}')

            try:
                indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
                clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
                clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

                clean_latents_pre = start_latent.to(history_latents)
                clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
                clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)
            except Exception as e:
                error_msg = f"샘플링 데이터 준비 오류: {e}"
                print(error_msg)
                traceback.print_exc()
                if last_output_filename:
                    stream.output_queue.push(('file', last_output_filename))
                continue

            if not high_vram and not cpu_fallback_mode:
                try:
                    unload_complete_models()
                    move_model_to_device_with_memory_preservation(transformer, target_device=device, preserved_memory_gb=gpu_memory_preservation)
                except Exception as e:
                    print(f"transformer GPU 이동 오류: {e}")

            if use_teacache and not cpu_fallback_mode:
                try:
                    transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
                except Exception as e:
                    print(f"teacache 초기화 오류: {e}")
                    transformer.initialize_teacache(enable_teacache=False)
            else:
                transformer.initialize_teacache(enable_teacache=False)

            def callback(d):
                global last_update_time
                last_update_time = time.time()
                
                try:
                    if stream.input_queue.top() == 'end':
                        stream.output_queue.push(('end', None))
                        raise KeyboardInterrupt('사용자 중단 요청')

                    preview = d['denoised']
                    preview = vae_decode_fake(preview)

                    preview = (preview * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                    preview = einops.rearrange(preview, 'b c t h w -> (b h) (t w) c')

                    current_step = d['i'] + 1
                    percentage = int(100.0 * current_step / steps)
                    hint = f'Sampling {current_step}/{steps}'
                    desc = f'Total generated frames: {int(max(0, total_generated_latent_frames * 4 - 3))}, Video length: {max(0, (total_generated_latent_frames * 4 - 3) / 30) :.2f} seconds (FPS-30).'
                    stream.output_queue.push(('progress', (preview, desc, make_progress_bar_html(percentage, hint))))
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    print(f"콜백 중 오류: {e}")
                return

            try:
                sampling_start_time = time.time()
                print(f"샘플링 시작, device: {device}, dtype: {transformer.dtype}, TeaCache: {use_teacache and not cpu_fallback_mode}")
                
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
                        callback=callback,
                    )
                    
                    print(f"샘플링 완료. 소요 시간: {time.time() - sampling_start_time:.2f} 초")
                except KeyboardInterrupt as e:
                    print(f"사용자 중단: {e}")
                    if last_output_filename:
                        stream.output_queue.push(('file', last_output_filename))
                        error_msg = "사용자에 의해 중단되었지만, 일부 비디오가 생성되었습니다."
                    else:
                        error_msg = "사용자에 의해 중단되었습니다. 비디오가 생성되지 않았습니다."
                    
                    stream.output_queue.push(('error', error_msg))
                    stream.output_queue.push(('end', None))
                    return
            except Exception as e:
                print(f"샘플링 중 오류: {e}")
                traceback.print_exc()
                if last_output_filename:
                    stream.output_queue.push(('file', last_output_filename))
                    error_msg = f"샘플링 중 오류(일부 비디오 생성됨): {e}"
                    stream.output_queue.push(('error', error_msg))
                else:
                    error_msg = f"샘플링 중 오류: {e}"
                    stream.output_queue.push(('error', error_msg))
                stream.output_queue.push(('end', None))
                return

            try:
                if is_last_section:
                    generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)

                total_generated_latent_frames += int(generated_latents.shape[2])
                history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)
            except Exception as e:
                error_msg = f"생성된 잠재 변수 처리 오류: {e}"
                print(error_msg)
                traceback.print_exc()
                if last_output_filename:
                    stream.output_queue.push(('file', last_output_filename))
                stream.output_queue.push(('error', error_msg))
                stream.output_queue.push(('end', None))
                return

            if not high_vram and not cpu_fallback_mode:
                try:
                    offload_model_from_device_for_memory_preservation(transformer, target_device=device, preserved_memory_gb=8)
                    load_model_as_complete(vae, target_device=device)
                except Exception as e:
                    print(f"모델 메모리 관리 오류: {e}")

            try:
                real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]
            except Exception as e:
                error_msg = f"히스토리 잠재 변수 처리 오류: {e}"
                print(error_msg)
                if last_output_filename:
                    stream.output_queue.push(('file', last_output_filename))
                continue

            try:
                vae_start_time = time.time()
                print(f"VAE 디코딩 시작, 잠재 변수 크기: {real_history_latents.shape}")

                if history_pixels is None:
                    history_pixels = vae_decode(real_history_latents, vae).cpu()
                else:
                    section_latent_frames = (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
                    overlapped_frames = latent_window_size * 4 - 3

                    current_pixels = vae_decode(real_history_latents[:, :, :section_latent_frames], vae).cpu()
                    history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)

                print(f"VAE 디코딩 완료, 소요 시간: {time.time() - vae_start_time:.2f} 초")

                if not high_vram and not cpu_fallback_mode:
                    try:
                        unload_complete_models()
                    except Exception as e:
                        print(f"모델 언로드 중 오류: {e}")

                output_filename = os.path.join(outputs_folder, f'{job_id}_{total_generated_latent_frames}.mp4')

                save_start_time = time.time()
                save_bcthw_as_mp4(history_pixels, output_filename, fps=30)
                print(f"비디오 저장 완료, 소요 시간: {time.time() - save_start_time:.2f} 초")

                print(f'디코딩 완료. 현재 latent 크기: {real_history_latents.shape}, pixel 크기: {history_pixels.shape}')

                last_output_filename = output_filename
                stream.output_queue.push(('file', output_filename))
            except Exception as e:
                print(f"비디오 디코딩/저장 중 오류: {e}")
                traceback.print_exc()
                if last_output_filename:
                    stream.output_queue.push(('file', last_output_filename))
                error_msg = f"비디오 디코딩/저장 오류: {e}"
                stream.output_queue.push(('error', error_msg))
                continue

            if is_last_section:
                break
    except Exception as e:
        print(f"처리 중 오류 발생: {e} (type: {type(e)})")
        traceback.print_exc()
        
        if isinstance(e, KeyboardInterrupt):
            print("KeyboardInterrupt 발생")

        if not high_vram and not cpu_fallback_mode:
            try:
                unload_complete_models(
                    text_encoder, text_encoder_2, image_encoder, vae, transformer
                )
            except Exception as unload_error:
                print(f"언로드 오류: {unload_error}")
        
        if last_output_filename:
            stream.output_queue.push(('file', last_output_filename))
        
        error_msg = f"처리 중 오류: {e}"
        stream.output_queue.push(('error', error_msg))

    print("worker 함수 종료, 'end' 신호 전송")
    stream.output_queue.push(('end', None))
    return

if IN_HF_SPACE and 'spaces' in globals():
    @spaces.GPU
    def process_with_gpu(input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache):
        global stream
        assert input_image is not None, 'No input image!'

        yield None, None, '', '', gr.update(interactive=False), gr.update(interactive=True)

        try:
            stream = AsyncStream()
            async_run(worker, input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache)

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

                    if flag == 'progress':
                        preview, desc, html = data
                        yield gr.update(), gr.update(visible=True, value=preview), desc, html, gr.update(interactive=False), gr.update(interactive=True)
                    
                    if flag == 'error':
                        error_message = data
                        print(f"오류 메시지 수신: {error_message}")

                    if flag == 'end':
                        if output_filename is None and prev_output_filename is not None:
                            output_filename = prev_output_filename
                        
                        if error_message:
                            error_html = create_error_html(error_message)
                            yield output_filename, gr.update(visible=False), gr.update(), error_html, gr.update(interactive=True), gr.update(interactive=False)
                        else:
                            yield output_filename, gr.update(visible=False), gr.update(), '', gr.update(interactive=True), gr.update(interactive=False)
                        break
                except Exception as e:
                    print(f"출력 처리 중 오류: {e}")
                    current_time = time.time()
                    if current_time - last_update_time > 60:
                        print(f"처리가 {current_time - last_update_time:.1f}초 동안 정지됨. 타임아웃으로 간주.")
                        if prev_output_filename:
                            error_html = create_error_html("처리 시간이 초과되었지만 일부 동영상이 생성되었습니다.", is_timeout=True)
                            yield prev_output_filename, gr.update(visible=False), gr.update(), error_html, gr.update(interactive=True), gr.update(interactive=False)
                        else:
                            error_html = create_error_html(f"처리 시간 초과: {e}", is_timeout=True)
                            yield None, gr.update(visible=False), gr.update(), error_html, gr.update(interactive=True), gr.update(interactive=False)
                        break
        except Exception as e:
            print(f"프로세스 시작 오류: {e}")
            traceback.print_exc()
            error_msg = str(e)
            
            error_html = create_error_html(error_msg)
            yield None, gr.update(visible=False), gr.update(), error_html, gr.update(interactive=True), gr.update(interactive=False)
    
    process = process_with_gpu
else:
    def process(input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache):
        global stream
        assert input_image is not None, 'No input image!'

        yield None, None, '', '', gr.update(interactive=False), gr.update(interactive=True)

        try:
            stream = AsyncStream()
            async_run(worker, input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache)

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

                    if flag == 'progress':
                        preview, desc, html = data
                        yield gr.update(), gr.update(visible=True, value=preview), desc, html, gr.update(interactive=False), gr.update(interactive=True)
                    
                    if flag == 'error':
                        error_message = data
                        print(f"오류 메시지 수신: {error_message}")

                    if flag == 'end':
                        if output_filename is None and prev_output_filename is not None:
                            output_filename = prev_output_filename
                        
                        if error_message:
                            error_html = create_error_html(error_message)
                            yield output_filename, gr.update(visible=False), gr.update(), error_html, gr.update(interactive=True), gr.update(interactive=False)
                        else:
                            yield output_filename, gr.update(visible=False), gr.update(), '', gr.update(interactive=True), gr.update(interactive=False)
                        break
                except Exception as e:
                    print(f"출력 처리 중 오류: {e}")
                    current_time = time.time()
                    if current_time - last_update_time > 60:
                        print(f"{current_time - last_update_time:.1f}초 동안 진행이 없어 타임아웃으로 간주합니다.")
                        if prev_output_filename:
                            error_html = create_error_html("처리 시간이 초과되었지만 일부 동영상이 생성되었습니다.", is_timeout=True)
                            yield prev_output_filename, gr.update(visible=False), gr.update(), error_html, gr.update(interactive=True), gr.update(interactive=False)
                        else:
                            error_html = create_error_html(f"처리 시간 초과: {e}", is_timeout=True)
                            yield None, gr.update(visible=False), gr.update(), error_html, gr.update(interactive=True), gr.update(interactive=False)
                        break
        except Exception as e:
            print(f"프로세스 시작 오류: {e}")
            traceback.print_exc()
            error_msg = str(e)
            
            error_html = create_error_html(error_msg)
            yield None, gr.update(visible=False), gr.update(), error_html, gr.update(interactive=True), gr.update(interactive=False)

def end_process():
    print("사용자가 중지 버튼을 눌렀습니다. 종료 신호를 보냅니다...")
    if 'stream' in globals() and stream is not None:
        try:
            current_top = stream.input_queue.top()
            print(f"현재 입력 큐 top: {current_top}")
        except Exception as e:
            print(f"입력 큐 확인 오류: {e}")
        try:
            stream.input_queue.push('end')
            print("end 신호 전송 완료")
            try:
                current_top_after = stream.input_queue.top()
                print(f"신호 전송 후 입력 큐 top: {current_top_after}")
            except Exception as e:
                print(f"신호 전송 후 큐 상태 확인 오류: {e}")
        except Exception as e:
            print(f"end 신호 전송 오류: {e}")
    else:
        print("stream이 초기화되지 않아 종료 신호를 보낼 수 없습니다.")
    return None

quick_prompts = [
    'The girl dances gracefully, with clear movements, full of charm.',
    'A character doing some simple body movements.',
]
quick_prompts = [[x] for x in quick_prompts]

def make_custom_css():
    progress_bar_css = make_progress_bar_css()
    
    responsive_css = """
    /* progress_bar_css로부터 불러온 기본 설정 + 추가 */
    
    #app-container {
        max-width: 100%;
        margin: 0 auto;
    }
    
    #language-toggle {
        position: fixed;
        top: 10px;
        right: 10px;
        z-index: 1000;
        background-color: rgba(0, 0, 0, 0.7);
        color: white;
        border: none;
        border-radius: 4px;
        padding: 5px 10px;
        cursor: pointer;
        font-size: 14px;
    }
    
    h1 {
        font-size: 2rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .start-btn, .stop-btn {
        min-height: 45px;
        font-size: 1rem;
    }
    
    @media (max-width: 768px) {
        h1 {
            font-size: 1.5rem;
            margin-bottom: 0.5rem;
        }
        
        .mobile-full-width {
            flex-direction: column !important;
        }
        
        .mobile-full-width > .gr-block {
            min-width: 100% !important;
            flex-grow: 1;
        }
        
        .video-container {
            height: auto !important;
        }
        
        .button-container button {
            min-height: 50px;
            font-size: 1rem;
            touch-action: manipulation;
        }
        
        .slider-container input[type="range"] {
            height: 30px;
        }
    }
    
    @media (min-width: 769px) and (max-width: 1024px) {
        .tablet-adjust {
            width: 48% !important;
        }
    }
    
    @media (prefers-color-scheme: dark) {
        .dark-mode-text {
            color: #f0f0f0;
        }
        .dark-mode-bg {
            background-color: #2a2a2a;
        }
    }
    
    button, input, select, textarea {
        font-size: 16px;
    }
    
    button, .interactive-element {
        min-height: 44px;
        min-width: 44px;
    }
    
    .high-contrast {
        color: #fff;
        background-color: #000;
    }
    
    .progress-container {
        margin-top: 10px;
        margin-bottom: 10px;
    }
    
    #error-message {
        color: #ff4444;
        font-weight: bold;
        padding: 10px;
        border-radius: 4px;
        margin-top: 10px;
    }
    
    .error-message {
        background-color: rgba(255, 0, 0, 0.1);
        padding: 10px;
        border-radius: 4px;
        margin-top: 10px;
        border: 1px solid #ffcccc;
    }
    
    .error-msg-en, .error-msg-ko {
        font-weight: bold;
    }
    
    .error-icon {
        color: #ff4444;
        font-size: 18px;
        margin-right: 8px;
    }
    
    #error-message:empty {
        background-color: transparent;
        border: none;
        padding: 0;
        margin: 0;
    }
    
    .error {
        display: none !important;
    }
    """
    
    return progress_bar_css + responsive_css

css = make_custom_css()
block = gr.Blocks(css=css).queue()
with block:
    gr.HTML("""
        <div id="app-container">
            <button id="language-toggle" onclick="toggleLanguage()">한국어 / English</button>
        </div>
        <script>
            window.currentLang = "en";
            function toggleLanguage() {
                window.currentLang = (window.currentLang === "en") ? "ko" : "en";
                
                const elements = document.querySelectorAll('[data-i18n]');
                elements.forEach(el => {
                    const key = el.getAttribute('data-i18n');
                    const translations = {
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
                            "error_message": "Error"
                        },
                        "ko": {
                            "title": "FramePack - 이미지에서 동영상 생성",
                            "upload_image": "이미지 업로드",
                            "prompt": "프롬프트",
                            "quick_prompts": "빠른 프롬프트 목록",
                            "start_generation": "생성 시작",
                            "stop_generation": "생성 중지",
                            "use_teacache": "TeaCache 사용",
                            "teacache_info": "더 빠른 속도를 제공하지만 손가락이나 손 생성 품질이 약간 떨어질 수 있습니다.",
                            "negative_prompt": "부정 프롬프트",
                            "seed": "랜덤 시드",
                            "video_length": "동영상 길이 (최대 5초)",
                            "latent_window": "잠재 윈도우 크기",
                            "steps": "추론 스텝 수",
                            "steps_info": "이 값을 변경하는 것은 권장되지 않습니다.",
                            "cfg_scale": "CFG 스케일",
                            "distilled_cfg": "증류된 CFG 스케일",
                            "distilled_cfg_info": "이 값을 변경하는 것은 권장되지 않습니다.",
                            "cfg_rescale": "CFG 재스케일",
                            "gpu_memory": "GPU 메모리 보존 (GB) (값이 클수록 속도가 느려짐)",
                            "gpu_memory_info": "OOM 오류가 발생하면 이 값을 더 크게 설정하십시오. 값이 클수록 속도가 느려집니다.",
                            "next_latents": "다음 잠재 변수",
                            "generated_video": "생성된 동영상",
                            "sampling_note": "주의: 역순 샘플링 때문에, 종료 동작이 시작 동작보다 먼저 생성됩니다. 시작 동작이 나타나지 않으면 기다려 주십시오.",
                            "error_message": "오류 메시지"
                        }
                    };
                    
                    if (translations[window.currentLang] && translations[window.currentLang][key]) {
                        if (el.tagName === 'BUTTON') {
                            el.textContent = translations[window.currentLang][key];
                        } else if (el.tagName === 'LABEL') {
                            el.textContent = translations[window.currentLang][key];
                        } else {
                            el.innerHTML = translations[window.currentLang][key];
                        }
                    }
                });
                
                // bilingual-label 처리
                document.querySelectorAll('.bilingual-label').forEach(el => {
                    const enText = el.getAttribute('data-en');
                    const koText = el.getAttribute('data-ko');
                    el.textContent = (window.currentLang === 'en') ? enText : koText;
                });
                
                // data-lang 처리
                document.querySelectorAll('[data-lang]').forEach(el => {
                    el.style.display = (el.getAttribute('data-lang') === window.currentLang) ? 'block' : 'none';
                });
            }
            
            document.addEventListener('DOMContentLoaded', function() {
                setTimeout(() => {
                    // 매핑
                    const labelMap = {
                        "Upload Image": "upload_image",
                        "이미지 업로드": "upload_image",
                        "Prompt": "prompt",
                        "프롬프트": "prompt",
                        "Quick Prompts": "quick_prompts",
                        "빠른 프롬프트 목록": "quick_prompts",
                        "Generate": "start_generation",
                        "생성 시작": "start_generation",
                        "Stop": "stop_generation",
                        "생성 중지": "stop_generation"
                    };
                    
                    document.querySelectorAll('label, span, button').forEach(el => {
                        const text = el.textContent.trim();
                        if (labelMap[text]) {
                            el.setAttribute('data-i18n', labelMap[text]);
                        }
                    });
                    
                    const titleEl = document.querySelector('h1');
                    if (titleEl) titleEl.setAttribute('data-i18n', 'title');
                    
                    toggleLanguage();
                }, 1000);
            });
        </script>
    """)

    gr.HTML("<h1 data-i18n='title'>FramePack - Image to Video Generation</h1>")
    
    with gr.Row(elem_classes="mobile-full-width"):
        with gr.Column(scale=1, elem_classes="mobile-full-width"):
            input_image = gr.Image(
                sources='upload', 
                type="numpy", 
                label="Upload Image", 
                elem_id="input-image",
                height=320
            )
            
            prompt = gr.Textbox(
                label="Prompt", 
                value='',
                elem_id="prompt-input"
            )
            
            example_quick_prompts = gr.Dataset(
                samples=quick_prompts, 
                label='Quick Prompts', 
                samples_per_page=1000, 
                components=[prompt]
            )
            example_quick_prompts.click(
                lambda x: x[0], 
                inputs=[example_quick_prompts], 
                outputs=prompt, 
                show_progress=False, 
                queue=False
            )

            with gr.Row(elem_classes="button-container"):
                start_button = gr.Button(
                    value="Generate", 
                    elem_classes="start-btn", 
                    elem_id="start-button",
                    variant="primary"
                )
                
                end_button = gr.Button(
                    value="Stop", 
                    elem_classes="stop-btn", 
                    elem_id="stop-button",
                    interactive=False
                )

            with gr.Group():
                use_teacache = gr.Checkbox(
                    label='Use TeaCache', 
                    value=True, 
                    info='Faster speed, but may result in slightly worse finger and hand generation.'
                )

                n_prompt = gr.Textbox(label="Negative Prompt", value="", visible=False)
                
                seed = gr.Number(
                    label="Seed", 
                    value=31337, 
                    precision=0
                )
                
                with gr.Group(elem_classes="slider-container"):
                    total_second_length = gr.Slider(
                        label="Video Length (max 5 seconds)", 
                        minimum=1, 
                        maximum=5, 
                        value=5, 
                        step=0.1
                    )
                    
                    latent_window_size = gr.Slider(
                        label="Latent Window Size", 
                        minimum=1, 
                        maximum=33, 
                        value=9, 
                        step=1, 
                        visible=False
                    )
                    
                    steps = gr.Slider(
                        label="Inference Steps", 
                        minimum=1, 
                        maximum=100, 
                        value=25, 
                        step=1, 
                        info='Changing this value is not recommended.'
                    )

                    cfg = gr.Slider(
                        label="CFG Scale", 
                        minimum=1.0, 
                        maximum=32.0, 
                        value=1.0, 
                        step=0.01, 
                        visible=False
                    )
                    
                    gs = gr.Slider(
                        label="Distilled CFG Scale", 
                        minimum=1.0, 
                        maximum=32.0, 
                        value=10.0, 
                        step=0.01, 
                        info='Changing this value is not recommended.'
                    )
                    
                    rs = gr.Slider(
                        label="CFG Rescale", 
                        minimum=0.0, 
                        maximum=1.0, 
                        value=0.0, 
                        step=0.01, 
                        visible=False
                    )

                    gpu_memory_preservation = gr.Slider(
                        label="GPU Memory (GB)", 
                        minimum=6, 
                        maximum=128, 
                        value=6, 
                        step=0.1, 
                        info="Set this to a larger value if you encounter OOM errors. Larger values cause slower speed."
                    )

        with gr.Column(scale=1, elem_classes="mobile-full-width"):
            preview_image = gr.Image(
                label="Preview", 
                height=200, 
                visible=False,
                elem_classes="preview-container"
            )
            
            result_video = gr.Video(
                label="Generated Video", 
                autoplay=True, 
                show_share_button=True,
                height=512, 
                loop=True,
                elem_classes="video-container",
                elem_id="result-video"
            )
            
            gr.HTML("<div data-i18n='sampling_note'>Note: Due to reversed sampling, ending actions will be generated before starting actions.</div>")
            
            with gr.Group(elem_classes="progress-container"):
                progress_desc = gr.Markdown('', elem_classes='no-generating-animation')
                progress_bar = gr.HTML('', elem_classes='no-generating-animation')
            
            error_message = gr.HTML('', elem_id='error-message', visible=True)
    
    ips = [input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache]

    start_button.click(fn=process, inputs=ips, outputs=[
        result_video, preview_image, progress_desc, progress_bar, start_button, end_button
    ])
    end_button.click(fn=end_process)

block.launch()

def create_error_html(error_msg, is_timeout=False):
    en_msg = ""
    ko_msg = ""
    
    if is_timeout:
        if "부분" in error_msg or "partial" in error_msg:
            en_msg = "Processing timed out, but partial video has been generated."
            ko_msg = "처리 시간이 초과되었지만 일부 동영상이 생성되었습니다."
        else:
            en_msg = f"Processing timed out: {error_msg}"
            ko_msg = f"처리 시간 초과: {error_msg}"
    elif "모델 로드" in error_msg:
        en_msg = "Failed to load models. Possibly heavy traffic or GPU problem."
        ko_msg = "모델 로드에 실패했습니다. 과도한 트래픽 또는 GPU 문제일 수 있습니다."
    elif "GPU" in error_msg or "CUDA" in error_msg or "memory" in error_msg or "메모리" in error_msg:
        en_msg = "GPU memory insufficient or error. Increase GPU memory preservation or reduce video length."
        ko_msg = "GPU 메모리가 부족하거나 오류가 발생했습니다. GPU 메모리 보존 값을 늘리거나 동영상 길이를 줄여보세요."
    elif "샘플링 중 오류" in error_msg or "sampling process" in error_msg:
        if "부분" in error_msg or "partial" in error_msg:
            en_msg = "Error during sampling, but partial video has been generated."
            ko_msg = "샘플링 중 오류가 발생했지만 일부 동영상이 생성되었습니다."
        else:
            en_msg = "Error during sampling. Unable to generate video."
            ko_msg = "샘플링 중 오류가 발생했습니다. 비디오 생성에 실패했습니다."
    elif "네트워크" in error_msg or "Network" in error_msg or "ConnectionError" in error_msg or "ReadTimeoutError" in error_msg:
        en_msg = "Network is unstable, model download timed out. Please try again later."
        ko_msg = "네트워크가 불안정하여 모델 다운로드가 시간 초과되었습니다. 잠시 후 다시 시도해 주세요."
    elif "VAE" in error_msg or "디코딩" in error_msg or "decode" in error_msg:
        en_msg = "Error during video decoding or saving process. Try a different seed."
        ko_msg = "비디오 디코딩/저장 중 오류가 발생했습니다. 다른 시드를 시도해보세요."
    else:
        en_msg = f"Processing error: {error_msg}"
        ko_msg = f"처리 중 오류가 발생했습니다: {error_msg}"
    
    return f"""
    <div class="error-message" id="custom-error-container">
        <div class="error-msg-en" data-lang="en">
            <span class="error-icon">⚠️</span> {en_msg}
        </div>
        <div class="error-msg-ko" data-lang="ko">
            <span class="error-icon">⚠️</span> {ko_msg}
        </div>
    </div>
    <script>
        (function() {{
            const errorContainer = document.getElementById('custom-error-container');
            if (errorContainer) {{
                const currentLang = window.currentLang || 'en';
                const errMsgs = errorContainer.querySelectorAll('[data-lang]');
                errMsgs.forEach(msg => {{
                    msg.style.display = (msg.getAttribute('data-lang') === currentLang) ? 'block' : 'none';
                }});
                const defaultErrorElements = document.querySelectorAll('.error');
                defaultErrorElements.forEach(el => {{
                    el.style.display = 'none';
                }});
            }}
        }})();
    </script>
    """
