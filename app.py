from diffusers_helper.hf_login import login

import os
import threading
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json

os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download')))

# 添加中英双语翻译字典
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
    "zh": {
        "title": "FramePack - 图像到视频生成",
        "upload_image": "上传图像",
        "prompt": "提示词",
        "quick_prompts": "快速提示词列表",
        "start_generation": "开始生成",
        "stop_generation": "结束生成",
        "use_teacache": "使用TeaCache",
        "teacache_info": "速度更快，但可能会使手指和手的生成效果稍差。",
        "negative_prompt": "负面提示词",
        "seed": "随机种子",
        "video_length": "视频长度(最大5秒)",
        "latent_window": "潜在窗口大小",
        "steps": "推理步数",
        "steps_info": "不建议修改此值。",
        "cfg_scale": "CFG Scale",
        "distilled_cfg": "蒸馏CFG比例",
        "distilled_cfg_info": "不建议修改此值。",
        "cfg_rescale": "CFG重缩放",
        "gpu_memory": "GPU推理保留内存(GB)(值越大速度越慢)",
        "gpu_memory_info": "如果出现OOM错误，请将此值设置得更大。值越大，速度越慢。",
        "next_latents": "下一批潜变量",
        "generated_video": "生成的视频",
        "sampling_note": "注意：由于采样是倒序的，结束动作将在开始动作之前生成。如果视频中没有出现起始动作，请继续等待，它将在稍后生成。",
        "error_message": "错误信息",
        "processing_error": "处理过程出错",
        "network_error": "网络连接不稳定，模型下载超时。请稍后再试。",
        "memory_error": "GPU内存不足，请尝试增加GPU推理保留内存值或降低视频长度。",
        "model_error": "模型加载失败，可能是网络问题或服务器负载过高。请稍后再试。",
        "partial_video": "处理过程中出现错误，但已生成部分视频",
        "processing_interrupt": "处理过程中断，但已生成部分视频"
    }
}

# 语言切换功能
def get_translation(key, lang="en"):
    if lang in translations and key in translations[lang]:
        return translations[lang][key]
    # 默认返回英文
    return translations["en"].get(key, key)

# 默认语言设置
current_language = "en"

# 切换语言函数
def switch_language():
    global current_language
    current_language = "zh" if current_language == "en" else "en"
    return current_language

import gradio as gr
import torch
import traceback
import einops
import safetensors.torch as sf
import numpy as np
import math

# 检查是否在Hugging Face Space环境中
IN_HF_SPACE = os.environ.get('SPACE_ID') is not None

# 添加变量跟踪GPU可用性
GPU_AVAILABLE = False
GPU_INITIALIZED = False
last_update_time = time.time()

# 如果在Hugging Face Space中，导入spaces模块
if IN_HF_SPACE:
    try:
        import spaces
        print("在Hugging Face Space环境中运行，已导入spaces模块")
        
        # 检查GPU可用性
        try:
            GPU_AVAILABLE = torch.cuda.is_available()
            print(f"GPU available: {GPU_AVAILABLE}")
            if GPU_AVAILABLE:
                print(f"GPU device name: {torch.cuda.get_device_name(0)}")
                print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9} GB")
                
                # 尝试进行小型GPU操作，确认GPU实际可用
                test_tensor = torch.zeros(1, device='cuda')
                test_tensor = test_tensor + 1
                del test_tensor
                print("成功进行GPU测试操作")
            else:
                print("警告: CUDA报告可用，但未检测到GPU设备")
        except Exception as e:
            GPU_AVAILABLE = False
            print(f"检查GPU时出错: {e}")
            print("将使用CPU模式运行")
    except ImportError:
        print("未能导入spaces模块，可能不在Hugging Face Space环境中")
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

# 在Spaces环境中，我们延迟所有CUDA操作
if not IN_HF_SPACE:
    # 仅在非Spaces环境中获取CUDA内存
    try:
        if torch.cuda.is_available():
            free_mem_gb = get_cuda_free_memory_gb(gpu)
            print(f'Free VRAM {free_mem_gb} GB')
        else:
            free_mem_gb = 6.0  # 默认值
            print("CUDA不可用，使用默认的内存设置")
    except Exception as e:
        free_mem_gb = 6.0  # 默认值
        print(f"获取CUDA内存时出错: {e}，使用默认的内存设置")
        
    high_vram = free_mem_gb > 60
    print(f'High-VRAM Mode: {high_vram}')
else:
    # 在Spaces环境中使用默认值
    print("在Spaces环境中使用默认内存设置")
    try:
        if GPU_AVAILABLE:
            free_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 * 0.9  # 使用90%的GPU内存
            high_vram = free_mem_gb > 10  # 更保守的条件
        else:
            free_mem_gb = 6.0  # 默认值
            high_vram = False
    except Exception as e:
        print(f"获取GPU内存时出错: {e}")
        free_mem_gb = 6.0  # 默认值
        high_vram = False
    
    print(f'GPU内存: {free_mem_gb:.2f} GB, High-VRAM Mode: {high_vram}')

# 使用models变量存储全局模型引用
models = {}
cpu_fallback_mode = not GPU_AVAILABLE  # 如果GPU不可用，使用CPU回退模式

# 使用加载模型的函数
def load_models():
    global models, cpu_fallback_mode, GPU_INITIALIZED
    
    if GPU_INITIALIZED:
        print("模型已加载，跳过重复加载")
        return models
    
    print("开始加载模型...")
    
    try:
        # 设置设备，根据GPU可用性确定
        device = 'cuda' if GPU_AVAILABLE and not cpu_fallback_mode else 'cpu'
        model_device = 'cpu'  # 初始加载到CPU
        
        # 降低精度以节省内存
        dtype = torch.float16 if GPU_AVAILABLE else torch.float32
        transformer_dtype = torch.bfloat16 if GPU_AVAILABLE else torch.float32
        
        print(f"使用设备: {device}, 模型精度: {dtype}, Transformer精度: {transformer_dtype}")
        
        # 加载模型
        try:
            text_encoder = LlamaModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder', torch_dtype=dtype).to(model_device)
            text_encoder_2 = CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder_2', torch_dtype=dtype).to(model_device)
            tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
            tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')
            vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=dtype).to(model_device)

            feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
            image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=dtype).to(model_device)

            transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePackI2V_HY', torch_dtype=transformer_dtype).to(model_device)
            
            print("成功加载所有模型")
        except Exception as e:
            print(f"加载模型时出错: {e}")
            print("尝试降低精度重新加载...")
            
            # 降低精度重试
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
            
            print("使用CPU模式成功加载所有模型")

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

        # 设置模型精度
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
                    # DynamicSwapInstaller is same as huggingface's enable_sequential_offload but 3x faster
                    DynamicSwapInstaller.install_model(transformer, device=device)
                    DynamicSwapInstaller.install_model(text_encoder, device=device)
                else:
                    text_encoder.to(device)
                    text_encoder_2.to(device)
                    image_encoder.to(device)
                    vae.to(device)
                    transformer.to(device)
                print(f"成功将模型移动到{device}设备")
            except Exception as e:
                print(f"移动模型到{device}时出错: {e}")
                print("回退到CPU模式")
                cpu_fallback_mode = True
        
        # 保存到全局变量
        models = {
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
        print(f"模型加载完成，运行模式: {'CPU' if cpu_fallback_mode else 'GPU'}")
        return models
    except Exception as e:
        print(f"加载模型过程中发生错误: {e}")
        traceback.print_exc()
        
        # 记录更详细的错误信息
        error_info = {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "cuda_available": torch.cuda.is_available(),
            "device": "cpu" if cpu_fallback_mode else "cuda",
        }
        
        # 保存错误信息到文件，方便排查
        try:
            with open(os.path.join(outputs_folder, "error_log.txt"), "w") as f:
                f.write(str(error_info))
        except:
            pass
            
        # 返回空字典，允许应用继续尝试运行
        cpu_fallback_mode = True
        return {}


# 使用Hugging Face Spaces GPU装饰器
if IN_HF_SPACE and 'spaces' in globals() and GPU_AVAILABLE:
    try:
        @spaces.GPU
        def initialize_models():
            """在@spaces.GPU装饰器内初始化模型"""
            global GPU_INITIALIZED
            try:
                result = load_models()
                GPU_INITIALIZED = True
                return result
            except Exception as e:
                print(f"使用spaces.GPU初始化模型时出错: {e}")
                traceback.print_exc()
                global cpu_fallback_mode
                cpu_fallback_mode = True
                # 不使用装饰器再次尝试
                return load_models()
    except Exception as e:
        print(f"创建spaces.GPU装饰器时出错: {e}")
        # 如果装饰器出错，直接使用非装饰器版本
        def initialize_models():
            return load_models()


# 以下函数内部会延迟获取模型
def get_models():
    """获取模型，如果尚未加载则加载模型"""
    global models, GPU_INITIALIZED
    
    # 添加模型加载锁，防止并发加载
    model_loading_key = "__model_loading__"
    
    if not models:
        # 检查是否正在加载模型
        if model_loading_key in globals():
            print("模型正在加载中，等待...")
            # 等待模型加载完成
            import time
            start_wait = time.time()
            while not models and model_loading_key in globals():
                time.sleep(0.5)
                # 超过60秒认为加载失败
                if time.time() - start_wait > 60:
                    print("等待模型加载超时")
                    break
            
            if models:
                return models
            
        try:
            # 设置加载标记
            globals()[model_loading_key] = True
            
            if IN_HF_SPACE and 'spaces' in globals() and GPU_AVAILABLE and not cpu_fallback_mode:
                try:
                    print("使用@spaces.GPU装饰器加载模型")
                    models = initialize_models()
                except Exception as e:
                    print(f"使用GPU装饰器加载模型失败: {e}")
                    print("尝试直接加载模型")
                    models = load_models()
            else:
                print("直接加载模型")
                models = load_models()
        except Exception as e:
            print(f"加载模型时发生未预期的错误: {e}")
            traceback.print_exc()
            # 确保有一个空字典
            models = {}
        finally:
            # 无论成功与否，都移除加载标记
            if model_loading_key in globals():
                del globals()[model_loading_key]
    
    return models


stream = AsyncStream()


@torch.no_grad()
def worker(input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache):
    global last_update_time
    last_update_time = time.time()
    
    # 限制视频长度不超过5秒
    total_second_length = min(total_second_length, 5.0)
    
    # 获取模型
    try:
        models = get_models()
        if not models:
            error_msg = "模型加载失败，请检查日志获取详细信息"
            print(error_msg)
            stream.output_queue.push(('error', error_msg))
            stream.output_queue.push(('end', None))
            return
        
        text_encoder = models['text_encoder']
        text_encoder_2 = models['text_encoder_2']
        tokenizer = models['tokenizer']
        tokenizer_2 = models['tokenizer_2']
        vae = models['vae']
        feature_extractor = models['feature_extractor']
        image_encoder = models['image_encoder']
        transformer = models['transformer']
    except Exception as e:
        error_msg = f"获取模型时出错: {e}"
        print(error_msg)
        traceback.print_exc()
        stream.output_queue.push(('error', error_msg))
        stream.output_queue.push(('end', None))
        return
    
    # 确定设备
    device = 'cuda' if GPU_AVAILABLE and not cpu_fallback_mode else 'cpu'
    print(f"使用设备: {device} 进行推理")
    
    # 调整参数以适应CPU模式
    if cpu_fallback_mode:
        print("CPU模式下使用更精简的参数")
        # 减小处理大小以加快CPU处理
        latent_window_size = min(latent_window_size, 5)
        steps = min(steps, 15)  # 减少步数
        total_second_length = min(total_second_length, 2.0)  # CPU模式下进一步限制视频长度
    
    total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))

    job_id = generate_timestamp()
    last_output_filename = None
    history_pixels = None
    history_latents = None
    total_generated_latent_frames = 0

    stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Starting ...'))))

    try:
        # Clean GPU
        if not high_vram and not cpu_fallback_mode:
            try:
                unload_complete_models(
                    text_encoder, text_encoder_2, image_encoder, vae, transformer
                )
            except Exception as e:
                print(f"卸载模型时出错: {e}")
                # 继续执行，不中断流程

        # Text encoding
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
            error_msg = f"文本编码过程出错: {e}"
            print(error_msg)
            traceback.print_exc()
            stream.output_queue.push(('error', error_msg))
            stream.output_queue.push(('end', None))
            return

        # Processing input image
        last_update_time = time.time()
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Image processing ...'))))

        try:
            H, W, C = input_image.shape
            height, width = find_nearest_bucket(H, W, resolution=640)
            
            # 如果是CPU模式，缩小处理尺寸
            if cpu_fallback_mode:
                height = min(height, 320)
                width = min(width, 320)
                
            input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)

            Image.fromarray(input_image_np).save(os.path.join(outputs_folder, f'{job_id}.png'))

            input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
            input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]
        except Exception as e:
            error_msg = f"图像处理过程出错: {e}"
            print(error_msg)
            traceback.print_exc()
            stream.output_queue.push(('error', error_msg))
            stream.output_queue.push(('end', None))
            return

        # VAE encoding
        last_update_time = time.time()
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'VAE encoding ...'))))

        try:
            if not high_vram and not cpu_fallback_mode:
                load_model_as_complete(vae, target_device=device)

            start_latent = vae_encode(input_image_pt, vae)
        except Exception as e:
            error_msg = f"VAE编码过程出错: {e}"
            print(error_msg)
            traceback.print_exc()
            stream.output_queue.push(('error', error_msg))
            stream.output_queue.push(('end', None))
            return

        # CLIP Vision
        last_update_time = time.time()
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'CLIP Vision encoding ...'))))

        try:
            if not high_vram and not cpu_fallback_mode:
                load_model_as_complete(image_encoder, target_device=device)

            image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
            image_encoder_last_hidden_state = image_encoder_output.last_hidden_state
        except Exception as e:
            error_msg = f"CLIP Vision编码过程出错: {e}"
            print(error_msg)
            traceback.print_exc()
            stream.output_queue.push(('error', error_msg))
            stream.output_queue.push(('end', None))
            return

        # Dtype
        try:
            llama_vec = llama_vec.to(transformer.dtype)
            llama_vec_n = llama_vec_n.to(transformer.dtype)
            clip_l_pooler = clip_l_pooler.to(transformer.dtype)
            clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
            image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)
        except Exception as e:
            error_msg = f"数据类型转换出错: {e}"
            print(error_msg)
            traceback.print_exc()
            stream.output_queue.push(('error', error_msg))
            stream.output_queue.push(('end', None))
            return

        # Sampling
        last_update_time = time.time()
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Start sampling ...'))))

        rnd = torch.Generator("cpu").manual_seed(seed)
        num_frames = latent_window_size * 4 - 3

        try:
            history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
            history_pixels = None
            total_generated_latent_frames = 0
        except Exception as e:
            error_msg = f"初始化历史状态出错: {e}"
            print(error_msg)
            traceback.print_exc()
            stream.output_queue.push(('error', error_msg))
            stream.output_queue.push(('end', None))
            return

        latent_paddings = reversed(range(total_latent_sections))

        if total_latent_sections > 4:
            # In theory the latent_paddings should follow the above sequence, but it seems that duplicating some
            # items looks better than expanding it when total_latent_sections > 4
            # One can try to remove below trick and just
            # use `latent_paddings = list(reversed(range(total_latent_sections)))` to compare
            latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]

        for latent_padding in latent_paddings:
            last_update_time = time.time()
            is_last_section = latent_padding == 0
            latent_padding_size = latent_padding * latent_window_size

            if stream.input_queue.top() == 'end':
                # 确保在结束时保存当前的视频
                if history_pixels is not None and total_generated_latent_frames > 0:
                    try:
                        output_filename = os.path.join(outputs_folder, f'{job_id}_final_{total_generated_latent_frames}.mp4')
                        save_bcthw_as_mp4(history_pixels, output_filename, fps=30)
                        stream.output_queue.push(('file', output_filename))
                    except Exception as e:
                        print(f"保存最终视频时出错: {e}")
                
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
                error_msg = f"准备采样数据时出错: {e}"
                print(error_msg)
                traceback.print_exc()
                # 尝试继续下一轮迭代而不是完全终止
                if last_output_filename:
                    stream.output_queue.push(('file', last_output_filename))
                continue

            if not high_vram and not cpu_fallback_mode:
                try:
                    unload_complete_models()
                    move_model_to_device_with_memory_preservation(transformer, target_device=device, preserved_memory_gb=gpu_memory_preservation)
                except Exception as e:
                    print(f"移动transformer到GPU时出错: {e}")
                    # 继续执行，可能影响性能但不必终止

            if use_teacache and not cpu_fallback_mode:
                try:
                    transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
                except Exception as e:
                    print(f"初始化teacache时出错: {e}")
                    # 禁用teacache并继续
                    transformer.initialize_teacache(enable_teacache=False)
            else:
                transformer.initialize_teacache(enable_teacache=False)

            def callback(d):
                global last_update_time
                last_update_time = time.time()
                
                try:
                    # 首先检查是否有停止信号
                    print(f"【调试】回调函数: 步骤 {d['i']}, 检查是否有停止信号")
                    try:
                        queue_top = stream.input_queue.top()
                        print(f"【调试】回调函数: 队列顶部信号 = {queue_top}")
                        
                        if queue_top == 'end':
                            print("【调试】回调函数: 检测到停止信号，准备中断...")
                            try:
                                stream.output_queue.push(('end', None))
                                print("【调试】回调函数: 成功向输出队列推送end信号")
                            except Exception as e:
                                print(f"【调试】回调函数: 向输出队列推送end信号失败: {e}")
                                
                            print("【调试】回调函数: 即将抛出KeyboardInterrupt异常")
                            raise KeyboardInterrupt('用户主动结束任务')
                    except Exception as e:
                        print(f"【调试】回调函数: 检查队列顶部信号出错: {e}")
                        
                    preview = d['denoised']
                    preview = vae_decode_fake(preview)

                    preview = (preview * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                    preview = einops.rearrange(preview, 'b c t h w -> (b h) (t w) c')

                    current_step = d['i'] + 1
                    percentage = int(100.0 * current_step / steps)
                    hint = f'Sampling {current_step}/{steps}'
                    desc = f'Total generated frames: {int(max(0, total_generated_latent_frames * 4 - 3))}, Video length: {max(0, (total_generated_latent_frames * 4 - 3) / 30) :.2f} seconds (FPS-30). The video is being extended now ...'
                    stream.output_queue.push(('progress', (preview, desc, make_progress_bar_html(percentage, hint))))
                except KeyboardInterrupt as e:
                    # 捕获并重新抛出中断异常，确保它能传播到采样函数
                    print(f"【调试】回调函数: 捕获到KeyboardInterrupt: {e}")
                    print("【调试】回调函数: 重新抛出中断异常，确保传播到采样函数")
                    raise
                except Exception as e:
                    print(f"【调试】回调函数中出错: {e}")
                    # 不中断采样过程
                print(f"【调试】回调函数: 步骤 {d['i']} 完成")
                return

            try:
                sampling_start_time = time.time()
                print(f"开始采样，设备: {device}, 数据类型: {transformer.dtype}, 使用TeaCache: {use_teacache and not cpu_fallback_mode}")
                
                try:
                    print("【调试】开始sample_hunyuan采样流程")
                    generated_latents = sample_hunyuan(
                        transformer=transformer,
                        sampler='unipc',
                        width=width,
                        height=height,
                        frames=num_frames,
                        real_guidance_scale=cfg,
                        distilled_guidance_scale=gs,
                        guidance_rescale=rs,
                        # shift=3.0,
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
                    
                    print(f"【调试】采样完成，用时: {time.time() - sampling_start_time:.2f}秒")
                except KeyboardInterrupt as e:
                    # 用户主动中断
                    print(f"【调试】捕获到KeyboardInterrupt: {e}")
                    print("【调试】用户主动中断采样过程，处理中断逻辑")
                    
                    # 如果已经有生成的视频，返回最后生成的视频
                    if last_output_filename:
                        print(f"【调试】已有部分生成视频: {last_output_filename}，返回此视频")
                        stream.output_queue.push(('file', last_output_filename))
                        error_msg = "用户中断生成过程，但已生成部分视频"
                    else:
                        print("【调试】没有部分生成视频，返回中断消息")
                        error_msg = "用户中断生成过程，未生成视频"
                    
                    print(f"【调试】推送错误消息: {error_msg}")
                    stream.output_queue.push(('error', error_msg))
                    print("【调试】推送end信号")
                    stream.output_queue.push(('end', None))
                    print("【调试】中断处理完成，返回")
                    return
            except Exception as e:
                print(f"采样过程中出错: {e}")
                traceback.print_exc()
                
                # 如果已经有生成的视频，返回最后生成的视频
                if last_output_filename:
                    stream.output_queue.push(('file', last_output_filename))
                    
                    # 创建错误信息
                    error_msg = f"采样过程中出错，但已返回部分生成的视频: {e}"
                    stream.output_queue.push(('error', error_msg))
                else:
                    # 如果没有生成的视频，返回错误信息
                    error_msg = f"采样过程中出错，无法生成视频: {e}"
                    stream.output_queue.push(('error', error_msg))
                
                stream.output_queue.push(('end', None))
                return

            try:
                if is_last_section:
                    generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)

                total_generated_latent_frames += int(generated_latents.shape[2])
                history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)
            except Exception as e:
                error_msg = f"处理生成的潜变量时出错: {e}"
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
                    print(f"管理模型内存时出错: {e}")
                    # 继续执行

            try:
                real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]
            except Exception as e:
                error_msg = f"处理历史潜变量时出错: {e}"
                print(error_msg)
                
                if last_output_filename:
                    stream.output_queue.push(('file', last_output_filename))
                continue

            try:
                vae_start_time = time.time()
                print(f"开始VAE解码，潜变量形状: {real_history_latents.shape}")
                
                if history_pixels is None:
                    history_pixels = vae_decode(real_history_latents, vae).cpu()
                else:
                    section_latent_frames = (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
                    overlapped_frames = latent_window_size * 4 - 3

                    current_pixels = vae_decode(real_history_latents[:, :, :section_latent_frames], vae).cpu()
                    history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)

                print(f"VAE解码完成，用时: {time.time() - vae_start_time:.2f}秒")
                
                if not high_vram and not cpu_fallback_mode:
                    try:
                        unload_complete_models()
                    except Exception as e:
                        print(f"卸载模型时出错: {e}")

                output_filename = os.path.join(outputs_folder, f'{job_id}_{total_generated_latent_frames}.mp4')

                save_start_time = time.time()
                save_bcthw_as_mp4(history_pixels, output_filename, fps=30)
                print(f"保存视频完成，用时: {time.time() - save_start_time:.2f}秒")

                print(f'Decoded. Current latent shape {real_history_latents.shape}; pixel shape {history_pixels.shape}')

                last_output_filename = output_filename
                stream.output_queue.push(('file', output_filename))
            except Exception as e:
                print(f"视频解码或保存过程中出错: {e}")
                traceback.print_exc()
                
                # 如果已经有生成的视频，返回最后生成的视频
                if last_output_filename:
                    stream.output_queue.push(('file', last_output_filename))
                
                # 记录错误信息
                error_msg = f"视频解码或保存过程中出错: {e}"
                stream.output_queue.push(('error', error_msg))
                
                # 尝试继续下一次迭代
                continue

            if is_last_section:
                break
    except Exception as e:
        print(f"【调试】处理过程中出现错误: {e}, 类型: {type(e)}")
        print(f"【调试】错误详情:")
        traceback.print_exc()
        
        # 检查是否是中断类型异常
        if isinstance(e, KeyboardInterrupt):
            print("【调试】捕获到外层KeyboardInterrupt异常")

        if not high_vram and not cpu_fallback_mode:
            try:
                print("【调试】尝试卸载模型以释放资源")
                unload_complete_models(
                    text_encoder, text_encoder_2, image_encoder, vae, transformer
                )
                print("【调试】模型卸载成功")
            except Exception as unload_error:
                print(f"【调试】卸载模型时出错: {unload_error}")
                pass
        
        # 如果已经有生成的视频，返回最后生成的视频
        if last_output_filename:
            print(f"【调试】外层异常处理: 返回已生成的部分视频 {last_output_filename}")
            stream.output_queue.push(('file', last_output_filename))
        else:
            print("【调试】外层异常处理: 未找到已生成的视频")
        
        # 返回错误信息
        error_msg = f"处理过程中出现错误: {e}"
        print(f"【调试】外层异常处理: 推送错误信息: {error_msg}")
        stream.output_queue.push(('error', error_msg))

    # 确保总是返回end信号
    print("【调试】工作函数结束，推送end信号")
    stream.output_queue.push(('end', None))
    return


# 使用Hugging Face Spaces GPU装饰器处理进程函数
if IN_HF_SPACE and 'spaces' in globals():
    @spaces.GPU
    def process_with_gpu(input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache):
        global stream
        assert input_image is not None, 'No input image!'

        # 初始化UI状态
        yield None, None, '', '', gr.update(interactive=False), gr.update(interactive=True)

        try:
            stream = AsyncStream()

            # 异步启动worker
            async_run(worker, input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache)

            output_filename = None
            prev_output_filename = None
            error_message = None

            # 持续检查worker的输出
            while True:
                try:
                    flag, data = stream.output_queue.next()

                    if flag == 'file':
                        output_filename = data
                        prev_output_filename = output_filename
                        # 清除错误显示，确保文件成功时不显示错误
                        yield output_filename, gr.update(), gr.update(), '', gr.update(interactive=False), gr.update(interactive=True)

                    if flag == 'progress':
                        preview, desc, html = data
                        # 更新进度时不改变错误信息，并确保停止按钮可交互
                        yield gr.update(), gr.update(visible=True, value=preview), desc, html, gr.update(interactive=False), gr.update(interactive=True)
                    
                    if flag == 'error':
                        error_message = data
                        print(f"收到错误消息: {error_message}")
                        # 不立即显示，等待end信号

                    if flag == 'end':
                        # 如果有最后的视频文件，确保返回
                        if output_filename is None and prev_output_filename is not None:
                            output_filename = prev_output_filename
                        
                        # 如果有错误消息，创建友好的错误显示
                        if error_message:
                            error_html = create_error_html(error_message)
                            yield output_filename, gr.update(visible=False), gr.update(), error_html, gr.update(interactive=True), gr.update(interactive=False)
                        else:
                            # 确保成功完成时不显示任何错误
                            yield output_filename, gr.update(visible=False), gr.update(), '', gr.update(interactive=True), gr.update(interactive=False)
                        break
                except Exception as e:
                    print(f"处理输出时出错: {e}")
                    # 检查是否长时间没有更新
                    current_time = time.time()
                    if current_time - last_update_time > 60:  # 60秒没有更新，可能卡住了
                        print(f"处理似乎卡住了，已经 {current_time - last_update_time:.1f} 秒没有更新")
                        
                        # 如果有部分生成的视频，返回
                        if prev_output_filename:
                            error_html = create_error_html("处理超时，但已生成部分视频", is_timeout=True)
                            yield prev_output_filename, gr.update(visible=False), gr.update(), error_html, gr.update(interactive=True), gr.update(interactive=False)
                        else:
                            error_html = create_error_html(f"处理超时: {e}", is_timeout=True)
                            yield None, gr.update(visible=False), gr.update(), error_html, gr.update(interactive=True), gr.update(interactive=False)
                        break
                    
        except Exception as e:
            print(f"启动处理时出错: {e}")
            traceback.print_exc()
            error_msg = str(e)
            
            error_html = create_error_html(error_msg)
            yield None, gr.update(visible=False), gr.update(), error_html, gr.update(interactive=True), gr.update(interactive=False)
    
    process = process_with_gpu
else:
    def process(input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache):
        global stream
        assert input_image is not None, 'No input image!'

        # 初始化UI状态
        yield None, None, '', '', gr.update(interactive=False), gr.update(interactive=True)

        try:
            stream = AsyncStream()

            # 异步启动worker
            async_run(worker, input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache)

            output_filename = None
            prev_output_filename = None
            error_message = None

            # 持续检查worker的输出
            while True:
                try:
                    flag, data = stream.output_queue.next()

                    if flag == 'file':
                        output_filename = data
                        prev_output_filename = output_filename
                        # 清除错误显示，确保文件成功时不显示错误
                        yield output_filename, gr.update(), gr.update(), '', gr.update(interactive=False), gr.update(interactive=True)

                    if flag == 'progress':
                        preview, desc, html = data
                        # 更新进度时不改变错误信息，并确保停止按钮可交互
                        yield gr.update(), gr.update(visible=True, value=preview), desc, html, gr.update(interactive=False), gr.update(interactive=True)
                    
                    if flag == 'error':
                        error_message = data
                        print(f"收到错误消息: {error_message}")
                        # 不立即显示，等待end信号

                    if flag == 'end':
                        # 如果有最后的视频文件，确保返回
                        if output_filename is None and prev_output_filename is not None:
                            output_filename = prev_output_filename
                        
                        # 如果有错误消息，创建友好的错误显示
                        if error_message:
                            error_html = create_error_html(error_message)
                            yield output_filename, gr.update(visible=False), gr.update(), error_html, gr.update(interactive=True), gr.update(interactive=False)
                        else:
                            # 确保成功完成时不显示任何错误
                            yield output_filename, gr.update(visible=False), gr.update(), '', gr.update(interactive=True), gr.update(interactive=False)
                        break
                except Exception as e:
                    print(f"处理输出时出错: {e}")
                    # 检查是否长时间没有更新
                    current_time = time.time()
                    if current_time - last_update_time > 60:  # 60秒没有更新，可能卡住了
                        print(f"处理似乎卡住了，已经 {current_time - last_update_time:.1f} 秒没有更新")
                        
                        # 如果有部分生成的视频，返回
                        if prev_output_filename:
                            error_html = create_error_html("处理超时，但已生成部分视频", is_timeout=True)
                            yield prev_output_filename, gr.update(visible=False), gr.update(), error_html, gr.update(interactive=True), gr.update(interactive=False)
                        else:
                            error_html = create_error_html(f"处理超时: {e}", is_timeout=True)
                            yield None, gr.update(visible=False), gr.update(), error_html, gr.update(interactive=True), gr.update(interactive=False)
                        break
                    
        except Exception as e:
            print(f"启动处理时出错: {e}")
            traceback.print_exc()
            error_msg = str(e)
            
            error_html = create_error_html(error_msg)
            yield None, gr.update(visible=False), gr.update(), error_html, gr.update(interactive=True), gr.update(interactive=False)


def end_process():
    """停止生成过程函数 - 通过在队列中推送'end'信号来中断生成"""
    print("【调试】用户点击了停止按钮，发送停止信号...")
    # 确保stream已初始化
    if 'stream' in globals() and stream is not None:
        # 在推送前检查队列状态
        try:
            current_top = stream.input_queue.top()
            print(f"【调试】当前队列顶部信号: {current_top}")
        except Exception as e:
            print(f"【调试】检查队列状态出错: {e}")
            
        # 推送end信号
        try:
            stream.input_queue.push('end')
            print("【调试】成功推送end信号到队列")
            
            # 验证信号是否成功推送
            try:
                current_top_after = stream.input_queue.top()
                print(f"【调试】推送后队列顶部信号: {current_top_after}")
            except Exception as e:
                print(f"【调试】验证推送后队列状态出错: {e}")
                
        except Exception as e:
            print(f"【调试】推送end信号到队列失败: {e}")
    else:
        print("【调试】警告: stream未初始化，无法发送停止信号")
    return None


quick_prompts = [
    'The girl dances gracefully, with clear movements, full of charm.',
    'A character doing some simple body movements.',
]
quick_prompts = [[x] for x in quick_prompts]


# 创建一个自定义CSS，增加响应式布局支持
def make_custom_css():
    progress_bar_css = make_progress_bar_css()
    
    responsive_css = """
    /* 基础响应式设置 */
    #app-container {
        max-width: 100%;
        margin: 0 auto;
    }
    
    /* 语言切换按钮样式 */
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
    
    /* 页面标题样式 */
    h1 {
        font-size: 2rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    /* 按钮样式 */
    .start-btn, .stop-btn {
        min-height: 45px;
        font-size: 1rem;
    }
    
    /* 移动设备样式 - 小屏幕 */
    @media (max-width: 768px) {
        h1 {
            font-size: 1.5rem;
            margin-bottom: 0.5rem;
        }
        
        /* 单列布局 */
        .mobile-full-width {
            flex-direction: column !important;
        }
        
        .mobile-full-width > .gr-block {
            min-width: 100% !important;
            flex-grow: 1;
        }
        
        /* 调整视频大小 */
        .video-container {
            height: auto !important;
        }
        
        /* 调整按钮大小 */
        .button-container button {
            min-height: 50px;
            font-size: 1rem;
            touch-action: manipulation;
        }
        
        /* 调整滑块 */
        .slider-container input[type="range"] {
            height: 30px;
        }
    }
    
    /* 平板设备样式 */
    @media (min-width: 769px) and (max-width: 1024px) {
        .tablet-adjust {
            width: 48% !important;
        }
    }
    
    /* 黑暗模式支持 */
    @media (prefers-color-scheme: dark) {
        .dark-mode-text {
            color: #f0f0f0;
        }
        
        .dark-mode-bg {
            background-color: #2a2a2a;
        }
    }
    
    /* 增强可访问性 */
    button, input, select, textarea {
        font-size: 16px; /* 防止iOS缩放 */
    }
    
    /* 触摸优化 */
    button, .interactive-element {
        min-height: 44px;
        min-width: 44px;
    }
    
    /* 提高对比度 */
    .high-contrast {
        color: #fff;
        background-color: #000;
    }
    
    /* 进度条样式增强 */
    .progress-container {
        margin-top: 10px;
        margin-bottom: 10px;
    }
    
    /* 错误消息样式 */
    #error-message {
        color: #ff4444;
        font-weight: bold;
        padding: 10px;
        border-radius: 4px;
        margin-top: 10px;
    }
    
    /* 确保错误容器正确显示 */
    .error-message {
        background-color: rgba(255, 0, 0, 0.1);
        padding: 10px;
        border-radius: 4px;
        margin-top: 10px;
        border: 1px solid #ffcccc;
    }
    
    /* 处理多语言错误消息 */
    .error-msg-en, .error-msg-zh {
        font-weight: bold;
    }
    
    /* 错误图标 */
    .error-icon {
        color: #ff4444;
        font-size: 18px;
        margin-right: 8px;
    }
    
    /* 确保空错误消息不显示背景和边框 */
    #error-message:empty {
        background-color: transparent;
        border: none;
        padding: 0;
        margin: 0;
    }
    
    /* 修复Gradio默认错误显示 */
    .error {
        display: none !important;
    }
    """
    
    # 合并CSS
    combined_css = progress_bar_css + responsive_css
    return combined_css


css = make_custom_css()
block = gr.Blocks(css=css).queue()
with block:
    # 添加语言切换功能
    gr.HTML("""
        <div id="app-container">
            <button id="language-toggle" onclick="toggleLanguage()">中文/English</button>
        </div>
        <script>
            // 全局变量，存储当前语言
            window.currentLang = "en";
            
            // 语言切换函数
            function toggleLanguage() {
                window.currentLang = window.currentLang === "en" ? "zh" : "en";
                
                // 获取所有带有data-i18n属性的元素
                const elements = document.querySelectorAll('[data-i18n]');
                
                // 遍历并切换语言
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
                            "error_message": "Error",
                            "processing_error": "Processing error",
                            "network_error": "Network connection is unstable, model download timed out. Please try again later.",
                            "memory_error": "GPU memory insufficient, please try increasing GPU memory preservation value or reduce video length.",
                            "model_error": "Failed to load model, possibly due to network issues or high server load. Please try again later.",
                            "partial_video": "Processing error, but partial video has been generated",
                            "processing_interrupt": "Processing was interrupted, but partial video has been generated"
                        },
                        "zh": {
                            "title": "FramePack - 图像到视频生成",
                            "upload_image": "上传图像",
                            "prompt": "提示词",
                            "quick_prompts": "快速提示词列表",
                            "start_generation": "开始生成",
                            "stop_generation": "结束生成",
                            "use_teacache": "使用TeaCache",
                            "teacache_info": "速度更快，但可能会使手指和手的生成效果稍差。",
                            "negative_prompt": "负面提示词",
                            "seed": "随机种子",
                            "video_length": "视频长度(最大5秒)",
                            "latent_window": "潜在窗口大小",
                            "steps": "推理步数",
                            "steps_info": "不建议修改此值。",
                            "cfg_scale": "CFG Scale",
                            "distilled_cfg": "蒸馏CFG比例",
                            "distilled_cfg_info": "不建议修改此值。",
                            "cfg_rescale": "CFG重缩放",
                            "gpu_memory": "GPU推理保留内存(GB)(值越大速度越慢)",
                            "gpu_memory_info": "如果出现OOM错误，请将此值设置得更大。值越大，速度越慢。",
                            "next_latents": "下一批潜变量",
                            "generated_video": "生成的视频",
                            "sampling_note": "注意：由于采样是倒序的，结束动作将在开始动作之前生成。如果视频中没有出现起始动作，请继续等待，它将在稍后生成。",
                            "error_message": "错误信息",
                            "processing_error": "处理过程出错",
                            "network_error": "网络连接不稳定，模型下载超时。请稍后再试。",
                            "memory_error": "GPU内存不足，请尝试增加GPU推理保留内存值或降低视频长度。",
                            "model_error": "模型加载失败，可能是网络问题或服务器负载过高。请稍后再试。",
                            "partial_video": "处理过程中出现错误，但已生成部分视频",
                            "processing_interrupt": "处理过程中断，但已生成部分视频"
                        }
                    };
                    
                    if (translations[window.currentLang] && translations[window.currentLang][key]) {
                        // 根据元素类型设置文本
                        if (el.tagName === 'BUTTON') {
                            el.textContent = translations[window.currentLang][key];
                        } else if (el.tagName === 'LABEL') {
                            el.textContent = translations[window.currentLang][key];
                        } else {
                            el.innerHTML = translations[window.currentLang][key];
                        }
                    }
                });
                
                // 更新页面上其他元素
                document.querySelectorAll('.bilingual-label').forEach(el => {
                    const enText = el.getAttribute('data-en');
                    const zhText = el.getAttribute('data-zh');
                    el.textContent = window.currentLang === 'en' ? enText : zhText;
                });
                
                // 处理错误消息容器
                document.querySelectorAll('[data-lang]').forEach(el => {
                    el.style.display = el.getAttribute('data-lang') === window.currentLang ? 'block' : 'none';
                });
            }
            
            // 页面加载后初始化
            document.addEventListener('DOMContentLoaded', function() {
                // 添加data-i18n属性到需要国际化的元素
                setTimeout(() => {
                    // 给所有标签添加i18n属性
                    const labelMap = {
                        "Upload Image": "upload_image",
                        "上传图像": "upload_image",
                        "Prompt": "prompt",
                        "提示词": "prompt",
                        "Quick Prompts": "quick_prompts",
                        "快速提示词列表": "quick_prompts",
                        "Generate": "start_generation", 
                        "开始生成": "start_generation",
                        "Stop": "stop_generation",
                        "结束生成": "stop_generation",
                        // 添加其他标签映射...
                    };
                    
                    // 处理标签
                    document.querySelectorAll('label, span, button').forEach(el => {
                        const text = el.textContent.trim();
                        if (labelMap[text]) {
                            el.setAttribute('data-i18n', labelMap[text]);
                        }
                    });
                    
                    // 添加特定元素的i18n属性
                    const titleEl = document.querySelector('h1');
                    if (titleEl) titleEl.setAttribute('data-i18n', 'title');
                    
                    // 初始化标签语言
                    toggleLanguage();
                }, 1000);
            });
        </script>
    """)
    
    # 标题使用data-i18n属性以便JavaScript切换
    gr.HTML("<h1 data-i18n='title'>FramePack - Image to Video Generation / 图像到视频生成</h1>")
    
    # 使用带有mobile-full-width类的响应式行
    with gr.Row(elem_classes="mobile-full-width"):
        with gr.Column(scale=1, elem_classes="mobile-full-width"):
            # 添加双语标签 - 上传图像
            input_image = gr.Image(
                sources='upload', 
                type="numpy", 
                label="Upload Image / 上传图像", 
                elem_id="input-image",
                height=320
            )
            
            # 添加双语标签 - 提示词
            prompt = gr.Textbox(
                label="Prompt / 提示词", 
                value='',
                elem_id="prompt-input"
            )
            
            # 添加双语标签 - 快速提示词
            example_quick_prompts = gr.Dataset(
                samples=quick_prompts, 
                label='Quick Prompts / 快速提示词列表', 
                samples_per_page=1000, 
                components=[prompt]
            )
            example_quick_prompts.click(lambda x: x[0], inputs=[example_quick_prompts], outputs=prompt, show_progress=False, queue=False)

            # 按钮添加样式和双语标签
            with gr.Row(elem_classes="button-container"):
                start_button = gr.Button(
                    value="Generate / 开始生成", 
                    elem_classes="start-btn", 
                    elem_id="start-button",
                    variant="primary"
                )
                
                end_button = gr.Button(
                    value="Stop / 结束生成", 
                    elem_classes="stop-btn", 
                    elem_id="stop-button",
                    interactive=False
                )

            # 参数设置区域
            with gr.Group():
                use_teacache = gr.Checkbox(
                    label='Use TeaCache / 使用TeaCache', 
                    value=True, 
                    info='Faster speed, but may result in slightly worse finger and hand generation. / 速度更快，但可能会使手指和手的生成效果稍差。'
                )

                n_prompt = gr.Textbox(label="Negative Prompt / 负面提示词", value="", visible=False)  # Not used
                
                seed = gr.Number(
                    label="Seed / 随机种子", 
                    value=31337, 
                    precision=0
                )

                # 添加slider-container类以便CSS触摸优化
                with gr.Group(elem_classes="slider-container"):
                    total_second_length = gr.Slider(
                        label="Video Length (max 5 seconds) / 视频长度(最大5秒)", 
                        minimum=1, 
                        maximum=5, 
                        value=5, 
                        step=0.1
                    )
                    
                    latent_window_size = gr.Slider(
                        label="Latent Window Size / 潜在窗口大小", 
                        minimum=1, 
                        maximum=33, 
                        value=9, 
                        step=1, 
                        visible=False
                    )
                    
                    steps = gr.Slider(
                        label="Inference Steps / 推理步数", 
                        minimum=1, 
                        maximum=100, 
                        value=25, 
                        step=1, 
                        info='Changing this value is not recommended. / 不建议修改此值。'
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
                        label="Distilled CFG Scale / 蒸馏CFG比例", 
                        minimum=1.0, 
                        maximum=32.0, 
                        value=10.0, 
                        step=0.01, 
                        info='Changing this value is not recommended. / 不建议修改此值。'
                    )
                    
                    rs = gr.Slider(
                        label="CFG Rescale / CFG重缩放", 
                        minimum=0.0, 
                        maximum=1.0, 
                        value=0.0, 
                        step=0.01, 
                        visible=False
                    )

                    gpu_memory_preservation = gr.Slider(
                        label="GPU Memory (GB) / GPU推理保留内存(GB)", 
                        minimum=6, 
                        maximum=128, 
                        value=6, 
                        step=0.1, 
                        info="Set this to a larger value if you encounter OOM errors. Larger values cause slower speed. / 如果出现OOM错误，请将此值设置得更大。值越大，速度越慢。"
                    )

        # 右侧预览和结果列
        with gr.Column(scale=1, elem_classes="mobile-full-width"):
            # 预览图像
            preview_image = gr.Image(
                label="Preview / 预览", 
                height=200, 
                visible=False,
                elem_classes="preview-container"
            )
            
            # 视频结果容器
            result_video = gr.Video(
                label="Generated Video / 生成的视频", 
                autoplay=True, 
                show_share_button=True,  # 添加分享按钮
                height=512, 
                loop=True,
                elem_classes="video-container",
                elem_id="result-video"
            )
            
            # 双语说明
            gr.HTML("<div data-i18n='sampling_note' class='note'>Note: Due to reversed sampling, ending actions will be generated before starting actions. If the starting action is not in the video, please wait, it will be generated later.</div>")
            
            # 进度指示器
            with gr.Group(elem_classes="progress-container"):
                progress_desc = gr.Markdown('', elem_classes='no-generating-animation')
                progress_bar = gr.HTML('', elem_classes='no-generating-animation')
            
            # 错误信息区域 - 确保使用HTML组件以支持我们的自定义错误消息格式
            error_message = gr.HTML('', elem_id='error-message', visible=True)
    
    # 处理函数
    ips = [input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache]
    
    # 开始和结束按钮事件
    start_button.click(fn=process, inputs=ips, outputs=[result_video, preview_image, progress_desc, progress_bar, start_button, end_button])
    end_button.click(fn=end_process)


block.launch() 

# 创建友好的错误显示HTML
def create_error_html(error_msg, is_timeout=False):
    """创建双语错误消息HTML"""
    # 提供更友好的中英文双语错误信息
    en_msg = ""
    zh_msg = ""
    
    if is_timeout:
        en_msg = "Processing timed out, but partial video may have been generated" if "部分视频" in error_msg else f"Processing timed out: {error_msg}"
        zh_msg = "处理超时，但已生成部分视频" if "部分视频" in error_msg else f"处理超时: {error_msg}"
    elif "模型加载失败" in error_msg:
        en_msg = "Failed to load models. The Space may be experiencing high traffic or GPU issues."
        zh_msg = "模型加载失败，可能是Space流量过高或GPU资源不足。"
    elif "GPU" in error_msg or "CUDA" in error_msg or "内存" in error_msg or "memory" in error_msg:
        en_msg = "GPU memory insufficient or GPU error. Try increasing GPU memory preservation value or reduce video length."
        zh_msg = "GPU内存不足或GPU错误，请尝试增加GPU推理保留内存值或降低视频长度。"
    elif "采样过程中出错" in error_msg:
        if "部分" in error_msg:
            en_msg = "Error during sampling process, but partial video has been generated."
            zh_msg = "采样过程中出错，但已生成部分视频。"
        else:
            en_msg = "Error during sampling process. Unable to generate video."
            zh_msg = "采样过程中出错，无法生成视频。"
    elif "模型下载超时" in error_msg or "网络连接不稳定" in error_msg or "ReadTimeoutError" in error_msg or "ConnectionError" in error_msg:
        en_msg = "Network connection is unstable, model download timed out. Please try again later."
        zh_msg = "网络连接不稳定，模型下载超时。请稍后再试。"
    elif "VAE" in error_msg or "解码" in error_msg or "decode" in error_msg:
        en_msg = "Error during video decoding or saving process. Try again with a different seed."
        zh_msg = "视频解码或保存过程中出错，请尝试使用不同的随机种子。"
    else:
        en_msg = f"Processing error: {error_msg}"
        zh_msg = f"处理过程出错: {error_msg}"
    
    # 创建双语错误消息HTML - 添加有用的图标并确保CSS样式适用
    return f"""
    <div class="error-message" id="custom-error-container">
        <div class="error-msg-en" data-lang="en">
            <span class="error-icon">⚠️</span> {en_msg}
        </div>
        <div class="error-msg-zh" data-lang="zh">
            <span class="error-icon">⚠️</span> {zh_msg}
        </div>
    </div>
    <script>
        // 根据当前语言显示相应的错误消息
        (function() {{
            const errorContainer = document.getElementById('custom-error-container');
            if (errorContainer) {{
                const currentLang = window.currentLang || 'en'; // 默认英语
                const errMsgs = errorContainer.querySelectorAll('[data-lang]');
                errMsgs.forEach(msg => {{
                    msg.style.display = msg.getAttribute('data-lang') === currentLang ? 'block' : 'none';
                }});
                
                // 确保Gradio默认错误UI不显示
                const defaultErrorElements = document.querySelectorAll('.error');
                defaultErrorElements.forEach(el => {{
                    el.style.display = 'none';
                }});
            }}
        }})();
    </script>
    """ 