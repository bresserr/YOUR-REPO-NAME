#############################################
# from diffusers_helper.hf_login import login
# 필요시 HF 로그인 사용 (주석 해제 후)
#############################################

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
import time

# Hugging Face Spaces 환경 인지 확인
IN_HF_SPACE = os.environ.get('SPACE_ID') is not None

# --------- 번역 딕셔너리(영어 고정) ---------
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
        # 최대 4초로 UI 표기 수정
        "video_length": "Video Length (max 4 seconds)",
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
        "sampling_note": "Note: The model predicts future frames from past frames. If the start action isn't immediately visible, please wait for more frames.",
        "error_message": "Error",
        "processing_error": "Processing error",
        "network_error": "Network connection is unstable, model download timed out. Please try again later.",
        "memory_error": "GPU memory insufficient, please try increasing GPU memory preservation value or reduce video length.",
        "model_error": "Failed to load model, possibly due to network issues or high server load. Please try again later.",
        "partial_video": "Processing error, but partial video has been generated",
        "processing_interrupt": "Processing was interrupted, but partial video has been generated"
    }
}

def get_translation(key):
    return translations["en"].get(key, key)

#############################################
# diffusers_helper 관련 임포트
#############################################
from diffusers_helper.thread_utils import AsyncStream, async_run
from diffusers_helper.gradio.progress_bar import make_progress_bar_css, make_progress_bar_html
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
from diffusers_helper.utils import (
    generate_timestamp,
    save_bcthw_as_mp4,
    resize_and_center_crop,
    crop_or_pad_yield_mask,
    soft_append_bcthw
)
from diffusers_helper.bucket_tools import find_nearest_bucket
from diffusers_helper.hunyuan import (
    encode_prompt_conds, vae_encode, vae_decode, vae_decode_fake
)
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan

from diffusers import AutoencoderKLHunyuanVideo
from transformers import (
    LlamaModel, CLIPTextModel,
    LlamaTokenizerFast, CLIPTokenizer,
    SiglipVisionModel, SiglipImageProcessor
)

#############################################
# GPU 체크
#############################################
GPU_AVAILABLE = torch.cuda.is_available()
free_mem_gb = 0.0
high_vram = False
if GPU_AVAILABLE:
    try:
        free_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        high_vram = (free_mem_gb > 60)
    except:
        pass
print(f"GPU Available: {GPU_AVAILABLE}, free_mem_gb={free_mem_gb}, high_vram={high_vram}")

cpu_fallback_mode = not GPU_AVAILABLE
last_update_time = time.time()

#############################################
# 모델 로드 (전역)
#############################################
text_encoder = None
text_encoder_2 = None
tokenizer = None
tokenizer_2 = None
vae = None
feature_extractor = None
image_encoder = None
transformer = None

# 아래 로직은 질문에 제시된 '두 번째 코드'의 모델 로드 부분을 거의 그대로 사용
def load_global_models():
    global text_encoder, text_encoder_2, tokenizer, tokenizer_2
    global vae, feature_extractor, image_encoder, transformer
    global cpu_fallback_mode

    # 이미 로드되었으면 패스
    if transformer is not None:
        return

    # GPU 메모리 정보
    device = gpu if GPU_AVAILABLE else cpu

    # diffusers_helper.memory.get_cuda_free_memory_gb(gpu)로 더 정확히 구해도 됨
    print("Loading models...")

    # ======== 실 코드: 두 번째 예시 기준 =========
    # (1) 하이브리드 (if high_vram -> GPU로 로드, 아니면 CPU + DynamicSwap)

    # 반드시 float16, bfloat16로 로드
    text_encoder_local = LlamaModel.from_pretrained(
        "hunyuanvideo-community/HunyuanVideo",
        subfolder='text_encoder',
        torch_dtype=torch.float16
    ).cpu()

    text_encoder_2_local = CLIPTextModel.from_pretrained(
        "hunyuanvideo-community/HunyuanVideo",
        subfolder='text_encoder_2',
        torch_dtype=torch.float16
    ).cpu()

    tokenizer_local = LlamaTokenizerFast.from_pretrained(
        "hunyuanvideo-community/HunyuanVideo",
        subfolder='tokenizer'
    )
    tokenizer_2_local = CLIPTokenizer.from_pretrained(
        "hunyuanvideo-community/HunyuanVideo",
        subfolder='tokenizer_2'
    )

    vae_local = AutoencoderKLHunyuanVideo.from_pretrained(
        "hunyuanvideo-community/HunyuanVideo",
        subfolder='vae',
        torch_dtype=torch.float16
    ).cpu()

    feature_extractor_local = SiglipImageProcessor.from_pretrained(
        "lllyasviel/flux_redux_bfl", subfolder='feature_extractor'
    )
    image_encoder_local = SiglipVisionModel.from_pretrained(
        "lllyasviel/flux_redux_bfl",
        subfolder='image_encoder',
        torch_dtype=torch.float16
    ).cpu()

    # FramePack_F1_I2V_HY_20250503 (bfloat16)
    transformer_local = HunyuanVideoTransformer3DModelPacked.from_pretrained(
        'lllyasviel/FramePack_F1_I2V_HY_20250503',
        torch_dtype=torch.bfloat16
    ).cpu()

    # eval & dtype
    vae_local.eval()
    text_encoder_local.eval()
    text_encoder_2_local.eval()
    image_encoder_local.eval()
    transformer_local.eval()

    # VAE slicing for low VRAM
    if not high_vram:
        vae_local.enable_slicing()
        vae_local.enable_tiling()

    # 오프로드용
    transformer_local.high_quality_fp32_output_for_inference = True
    transformer_local.to(dtype=torch.bfloat16)
    vae_local.to(dtype=torch.float16)
    image_encoder_local.to(dtype=torch.float16)
    text_encoder_local.to(dtype=torch.float16)
    text_encoder_2_local.to(dtype=torch.float16)

    # requires_grad_(False)
    for m in [vae_local, text_encoder_local, text_encoder_2_local, image_encoder_local, transformer_local]:
        m.requires_grad_(False)

    # GPU 모드 & VRAM 많으면 전부 GPU
    # 그렇지 않으면 DynamicSwap
    if GPU_AVAILABLE:
        if not high_vram:
            DynamicSwapInstaller.install_model(transformer_local, device=gpu)
            DynamicSwapInstaller.install_model(text_encoder_local, device=gpu)
        else:
            text_encoder_local.to(gpu)
            text_encoder_2_local.to(gpu)
            image_encoder_local.to(gpu)
            vae_local.to(gpu)
            transformer_local.to(gpu)
    else:
        cpu_fallback_mode = True

    # 글로벌에 할당
    print("Model loaded.")
    text_encoder = text_encoder_local
    text_encoder_2 = text_encoder_2_local
    tokenizer = tokenizer_local
    tokenizer_2 = tokenizer_2_local
    vae = vae_local
    feature_extractor = feature_extractor_local
    image_encoder = image_encoder_local
    transformer = transformer_local

#############################################
# Worker 로직 (두 번째 코드) 그대로
#############################################
stream = AsyncStream()

outputs_folder = './outputs/'
os.makedirs(outputs_folder, exist_ok=True)

@torch.no_grad()
def worker(
    input_image, prompt, n_prompt, seed,
    total_second_length, latent_window_size, steps,
    cfg, gs, rs, gpu_memory_preservation, use_teacache
):
    """
    실제 샘플링 로직 (두 번째 코드 기반)
    """
    load_global_models()  # 모델 로딩
    global text_encoder, text_encoder_2, tokenizer, tokenizer_2
    global vae, feature_extractor, image_encoder, transformer
    global last_update_time

    # 최대 4초로 고정
    total_second_length = min(total_second_length, 4.0)

    total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))

    job_id = generate_timestamp()

    stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Starting ...'))))

    try:
        # GPU 적을 경우 Unload
        if not high_vram and GPU_AVAILABLE:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )

        # Text encoding
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Text encoding ...'))))

        if not high_vram and GPU_AVAILABLE:
            fake_diffusers_current_device(text_encoder, gpu)
            load_model_as_complete(text_encoder_2, target_device=gpu)

        llama_vec, clip_l_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
        if cfg == 1.0:
            llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
        else:
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        llama_vec, llama_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

        # Image processing
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Image processing ...'))))

        H, W, C = input_image.shape
        height, width = find_nearest_bucket(H, W, resolution=640)

        if cpu_fallback_mode:
            height = min(height, 320)
            width = min(width, 320)

        input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)

        Image.fromarray(input_image_np).save(os.path.join(outputs_folder, f'{job_id}.png'))

        input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]

        # VAE encode
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'VAE encoding ...'))))

        if not high_vram and GPU_AVAILABLE:
            load_model_as_complete(vae, target_device=gpu)
        start_latent = vae_encode(input_image_pt, vae)

        # CLIP Vision
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'CLIP Vision encoding ...'))))

        if not high_vram and GPU_AVAILABLE:
            load_model_as_complete(image_encoder, target_device=gpu)
        image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state

        # dtype
        llama_vec = llama_vec.to(transformer.dtype)
        llama_vec_n = llama_vec_n.to(transformer.dtype)
        clip_l_pooler = clip_l_pooler.to(transformer.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)

        # Start sampling
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Start sampling ...'))))

        rnd = torch.Generator("cpu").manual_seed(seed)

        # 초기 history latents
        history_latents = torch.zeros(size=(1, 16, 16 + 2 + 1, height // 8, width // 8), dtype=torch.float32).cpu()
        history_pixels = None

        # start_latent 붙이기
        history_latents = torch.cat([history_latents, start_latent.to(history_latents)], dim=2)
        total_generated_latent_frames = 1

        for section_index in range(total_latent_sections):
            if stream.input_queue.top() == 'end':
                stream.output_queue.push(('end', None))
                return

            print(f'Section {section_index+1}/{total_latent_sections}')

            if not high_vram and GPU_AVAILABLE:
                unload_complete_models()
                move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

            # teacache
            if use_teacache:
                transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
            else:
                transformer.initialize_teacache(enable_teacache=False)

            def callback(d):
                preview = d['denoised']
                preview = vae_decode_fake(preview)
                preview = (preview * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                preview = einops.rearrange(preview, 'b c t h w -> (b h) (t w) c')

                if stream.input_queue.top() == 'end':
                    stream.output_queue.push(('end', None))
                    raise KeyboardInterrupt('User stops generation.')

                current_step = d['i'] + 1
                percentage = int(100.0 * current_step / steps)
                hint = f'Sampling {current_step}/{steps}'
                desc = f'Section {section_index+1}/{total_latent_sections}'
                stream.output_queue.push(('progress', (preview, desc, make_progress_bar_html(percentage, hint))))
                return

            # indices
            frames_per_section = latent_window_size * 4 - 3
            indices = torch.arange(0, sum([1, 16, 2, 1, latent_window_size])).unsqueeze(0)
            (
                clean_latent_indices_start,
                clean_latent_4x_indices,
                clean_latent_2x_indices,
                clean_latent_1x_indices,
                latent_indices
            ) = indices.split([1, 16, 2, 1, latent_window_size], dim=1)

            clean_latent_indices = torch.cat([clean_latent_indices_start, clean_latent_1x_indices], dim=1)

            clean_latents_4x, clean_latents_2x, clean_latents_1x = history_latents[:, :, -19:, :, :].split([16, 2, 1], dim=2)
            clean_latents = torch.cat([start_latent.to(history_latents), clean_latents_1x], dim=2)

            try:
                generated_latents = sample_hunyuan(
                    transformer=transformer,
                    sampler='unipc',
                    width=width,
                    height=height,
                    frames=frames_per_section,
                    real_guidance_scale=cfg,
                    distilled_guidance_scale=gs,
                    guidance_rescale=rs,
                    num_inference_steps=steps,
                    generator=rnd,
                    prompt_embeds=llama_vec,
                    prompt_embeds_mask=llama_mask,
                    prompt_poolers=clip_l_pooler,
                    negative_prompt_embeds=llama_vec_n,
                    negative_prompt_embeds_mask=llama_mask_n,
                    negative_prompt_poolers=clip_l_pooler_n,
                    device=gpu if GPU_AVAILABLE else cpu,
                    dtype=torch.bfloat16,
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
            except KeyboardInterrupt:
                print("User cancelled.")
                stream.output_queue.push(('end', None))
                return
            except Exception as e:
                traceback.print_exc()
                stream.output_queue.push(('end', None))
                return

            total_generated_latent_frames += generated_latents.shape[2]
            history_latents = torch.cat([history_latents, generated_latents.to(history_latents)], dim=2)

            if not high_vram and GPU_AVAILABLE:
                offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=8)
                load_model_as_complete(vae, target_device=gpu)

            real_history_latents = history_latents[:, :, -total_generated_latent_frames:, :, :]

            if history_pixels is None:
                history_pixels = vae_decode(real_history_latents, vae).cpu()
            else:
                section_latent_frames = latent_window_size * 2
                overlapped_frames = frames_per_section
                current_pixels = vae_decode(real_history_latents[:, :, -section_latent_frames:], vae).cpu()
                history_pixels = soft_append_bcthw(history_pixels, current_pixels, overlapped_frames)

            if not high_vram and GPU_AVAILABLE:
                unload_complete_models()

            output_filename = os.path.join(outputs_folder, f'{job_id}_{total_generated_latent_frames}.mp4')
            save_bcthw_as_mp4(history_pixels, output_filename, fps=30, crf=16)  # CRF=16

            stream.output_queue.push(('file', output_filename))

    except:
        traceback.print_exc()
        if not high_vram and GPU_AVAILABLE:
            unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer)

    stream.output_queue.push(('end', None))
    return

def end_process():
    """
    중단 요청
    """
    global stream
    stream.input_queue.push('end')

# Gradio에서 이 worker 함수를 비동기로 호출
def process(
    input_image, prompt, n_prompt, seed,
    total_second_length, latent_window_size, steps,
    cfg, gs, rs, gpu_memory_preservation, use_teacache
):
    global stream
    if input_image is None:
        raise ValueError("No input image provided.")

    yield None, None, "", "", gr.update(interactive=False), gr.update(interactive=True)

    stream = AsyncStream()
    async_run(
        worker,
        input_image, prompt, n_prompt, seed,
        total_second_length, latent_window_size, steps,
        cfg, gs, rs, gpu_memory_preservation, use_teacache
    )

    output_filename = None
    prev_filename = None
    error_message = None

    while True:
        flag, data = stream.output_queue.next()
        if flag == 'file':
            output_filename = data
            prev_filename = output_filename
            yield output_filename, gr.update(), gr.update(), "", gr.update(interactive=False), gr.update(interactive=True)

        elif flag == 'progress':
            preview, desc, html = data
            yield gr.update(), gr.update(visible=True, value=preview), desc, html, gr.update(interactive=False), gr.update(interactive=True)

        elif flag == 'error':
            error_message = data
            print(f"Error: {error_message}")

        elif flag == 'end':
            if output_filename is None and prev_filename:
                output_filename = prev_filename
            # 에러가 있었으면 에러 표시
            if error_message:
                yield (
                    output_filename,  # 마지막 파일 (또는 None)
                    gr.update(visible=False),
                    gr.update(),
                    f"<div style='color:red;'>{error_message}</div>",
                    gr.update(interactive=True),
                    gr.update(interactive=False)
                )
            else:
                yield (
                    output_filename, gr.update(visible=False), gr.update(), "", gr.update(interactive=True), gr.update(interactive=False)
                )
            break

# UI CSS
def make_custom_css():
    base_progress_css = make_progress_bar_css()
    pastel_css = """
    body {
        background: #faf9ff !important; 
        font-family: "Noto Sans", sans-serif;
    }
    #app-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 1rem;
        position: relative;
    }
    #app-container h1 {
        color: #5F5AA2;
        margin-bottom: 1.2rem;
        font-weight: 700;
        text-shadow: 1px 1px 2px #bbb;
    }
    .gr-panel {
        background: #ffffffcc;
        border: 1px solid #e1dff0;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .button-container button {
        min-height: 45px;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 6px;
    }
    .button-container button#start-button {
        background-color: #A289E3 !important;
        color: #fff !important;
        border: 1px solid #a58de2;
    }
    .button-container button#stop-button {
        background-color: #F48A9B !important;
        color: #fff !important;
        border: 1px solid #f18fa0;
    }
    .button-container button:hover {
        filter: brightness(0.95);
    }
    .preview-container, .video-container {
        border: 1px solid #ded9f2;
        border-radius: 8px;
        overflow: hidden;
    }
    .progress-container {
        margin-top: 15px;
        margin-bottom: 15px;
    }
    .error-message {
        background-color: #FFF5F5;
        border: 1px solid #FED7D7;
        color: #E53E3E;
        padding: 10px;
        border-radius: 4px;
        margin-top: 10px;
        font-weight: 500;
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
    return base_progress_css + pastel_css

css = make_custom_css()

# 샘플 프롬프트
quick_prompts = [
    ["The girl dances gracefully, with clear movements, full of charm."],
    ["A character doing some simple body movements."]
]

# Gradio UI
block = gr.Blocks(css=css).queue()
with block:
    gr.HTML("<div id='app-container'><h1>FramePack - Image to Video Generation</h1></div>")

    with gr.Row(elem_classes="mobile-full-width"):
        # 왼쪽
        with gr.Column(scale=1, elem_classes="gr-panel"):
            input_image = gr.Image(
                label=get_translation("upload_image"),
                type="numpy",
                height=320
            )
            prompt = gr.Textbox(
                label=get_translation("prompt"),
                value=''
            )

            example_quick_prompts = gr.Dataset(
                samples=quick_prompts,
                label=get_translation("quick_prompts"),
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

        # 오른쪽
        with gr.Column(scale=1, elem_classes="gr-panel"):
            with gr.Row(elem_classes="button-container"):
                start_button = gr.Button(
                    value=get_translation("start_generation"),
                    elem_id="start-button",
                    variant="primary"
                )
                stop_button = gr.Button(
                    value=get_translation("stop_generation"),
                    elem_id="stop-button",
                    interactive=False
                )

            result_video = gr.Video(
                label=get_translation("generated_video"),
                autoplay=True,
                loop=True,
                height=320,
                elem_classes="video-container"
            )
            preview_image = gr.Image(
                label=get_translation("next_latents"),
                visible=False,
                height=150,
                elem_classes="preview-container"
            )
            gr.Markdown(get_translation("sampling_note"))

            with gr.Group(elem_classes="progress-container"):
                progress_desc = gr.Markdown('')
                progress_bar = gr.HTML('')

            error_message = gr.HTML('', visible=True)

    # Advanced
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
        # 기본 2초, 최대 4초
        total_second_length = gr.Slider(
            label=get_translation("video_length"),
            minimum=1,
            maximum=4,
            value=2,
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

    # 버튼 처리
    inputs_list = [
        input_image, prompt, n_prompt, seed,
        total_second_length, latent_window_size, steps,
        cfg, gs, rs, gpu_memory_preservation, use_teacache
    ]
    start_button.click(
        fn=process,
        inputs=inputs_list,
        outputs=[result_video, preview_image, progress_desc, progress_bar, start_button, stop_button]
    )
    stop_button.click(fn=end_process)

block.launch()
#############################################
# from diffusers_helper.hf_login import login
# 필요시 HF 로그인 사용 (주석 해제 후)
#############################################

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
import time

# Hugging Face Spaces 환경 인지 확인
IN_HF_SPACE = os.environ.get('SPACE_ID') is not None

# --------- 번역 딕셔너리(영어 고정) ---------
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
        # 최대 4초로 UI 표기 수정
        "video_length": "Video Length (max 4 seconds)",
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
        "sampling_note": "Note: The model predicts future frames from past frames. If the start action isn't immediately visible, please wait for more frames.",
        "error_message": "Error",
        "processing_error": "Processing error",
        "network_error": "Network connection is unstable, model download timed out. Please try again later.",
        "memory_error": "GPU memory insufficient, please try increasing GPU memory preservation value or reduce video length.",
        "model_error": "Failed to load model, possibly due to network issues or high server load. Please try again later.",
        "partial_video": "Processing error, but partial video has been generated",
        "processing_interrupt": "Processing was interrupted, but partial video has been generated"
    }
}

def get_translation(key):
    return translations["en"].get(key, key)

#############################################
# diffusers_helper 관련 임포트
#############################################
from diffusers_helper.thread_utils import AsyncStream, async_run
from diffusers_helper.gradio.progress_bar import make_progress_bar_css, make_progress_bar_html
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
from diffusers_helper.utils import (
    generate_timestamp,
    save_bcthw_as_mp4,
    resize_and_center_crop,
    crop_or_pad_yield_mask,
    soft_append_bcthw
)
from diffusers_helper.bucket_tools import find_nearest_bucket
from diffusers_helper.hunyuan import (
    encode_prompt_conds, vae_encode, vae_decode, vae_decode_fake
)
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan

from diffusers import AutoencoderKLHunyuanVideo
from transformers import (
    LlamaModel, CLIPTextModel,
    LlamaTokenizerFast, CLIPTokenizer,
    SiglipVisionModel, SiglipImageProcessor
)

#############################################
# GPU 체크
#############################################
GPU_AVAILABLE = torch.cuda.is_available()
free_mem_gb = 0.0
high_vram = False
if GPU_AVAILABLE:
    try:
        free_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        high_vram = (free_mem_gb > 60)
    except:
        pass
print(f"GPU Available: {GPU_AVAILABLE}, free_mem_gb={free_mem_gb}, high_vram={high_vram}")

cpu_fallback_mode = not GPU_AVAILABLE
last_update_time = time.time()

#############################################
# 모델 로드 (전역)
#############################################
text_encoder = None
text_encoder_2 = None
tokenizer = None
tokenizer_2 = None
vae = None
feature_extractor = None
image_encoder = None
transformer = None

# 아래 로직은 질문에 제시된 '두 번째 코드'의 모델 로드 부분을 거의 그대로 사용
def load_global_models():
    global text_encoder, text_encoder_2, tokenizer, tokenizer_2
    global vae, feature_extractor, image_encoder, transformer
    global cpu_fallback_mode

    # 이미 로드되었으면 패스
    if transformer is not None:
        return

    # GPU 메모리 정보
    device = gpu if GPU_AVAILABLE else cpu

    # diffusers_helper.memory.get_cuda_free_memory_gb(gpu)로 더 정확히 구해도 됨
    print("Loading models...")

    # ======== 실 코드: 두 번째 예시 기준 =========
    # (1) 하이브리드 (if high_vram -> GPU로 로드, 아니면 CPU + DynamicSwap)

    # 반드시 float16, bfloat16로 로드
    text_encoder_local = LlamaModel.from_pretrained(
        "hunyuanvideo-community/HunyuanVideo",
        subfolder='text_encoder',
        torch_dtype=torch.float16
    ).cpu()

    text_encoder_2_local = CLIPTextModel.from_pretrained(
        "hunyuanvideo-community/HunyuanVideo",
        subfolder='text_encoder_2',
        torch_dtype=torch.float16
    ).cpu()

    tokenizer_local = LlamaTokenizerFast.from_pretrained(
        "hunyuanvideo-community/HunyuanVideo",
        subfolder='tokenizer'
    )
    tokenizer_2_local = CLIPTokenizer.from_pretrained(
        "hunyuanvideo-community/HunyuanVideo",
        subfolder='tokenizer_2'
    )

    vae_local = AutoencoderKLHunyuanVideo.from_pretrained(
        "hunyuanvideo-community/HunyuanVideo",
        subfolder='vae',
        torch_dtype=torch.float16
    ).cpu()

    feature_extractor_local = SiglipImageProcessor.from_pretrained(
        "lllyasviel/flux_redux_bfl", subfolder='feature_extractor'
    )
    image_encoder_local = SiglipVisionModel.from_pretrained(
        "lllyasviel/flux_redux_bfl",
        subfolder='image_encoder',
        torch_dtype=torch.float16
    ).cpu()

    # FramePack_F1_I2V_HY_20250503 (bfloat16)
    transformer_local = HunyuanVideoTransformer3DModelPacked.from_pretrained(
        'lllyasviel/FramePack_F1_I2V_HY_20250503',
        torch_dtype=torch.bfloat16
    ).cpu()

    # eval & dtype
    vae_local.eval()
    text_encoder_local.eval()
    text_encoder_2_local.eval()
    image_encoder_local.eval()
    transformer_local.eval()

    # VAE slicing for low VRAM
    if not high_vram:
        vae_local.enable_slicing()
        vae_local.enable_tiling()

    # 오프로드용
    transformer_local.high_quality_fp32_output_for_inference = True
    transformer_local.to(dtype=torch.bfloat16)
    vae_local.to(dtype=torch.float16)
    image_encoder_local.to(dtype=torch.float16)
    text_encoder_local.to(dtype=torch.float16)
    text_encoder_2_local.to(dtype=torch.float16)

    # requires_grad_(False)
    for m in [vae_local, text_encoder_local, text_encoder_2_local, image_encoder_local, transformer_local]:
        m.requires_grad_(False)

    # GPU 모드 & VRAM 많으면 전부 GPU
    # 그렇지 않으면 DynamicSwap
    if GPU_AVAILABLE:
        if not high_vram:
            DynamicSwapInstaller.install_model(transformer_local, device=gpu)
            DynamicSwapInstaller.install_model(text_encoder_local, device=gpu)
        else:
            text_encoder_local.to(gpu)
            text_encoder_2_local.to(gpu)
            image_encoder_local.to(gpu)
            vae_local.to(gpu)
            transformer_local.to(gpu)
    else:
        cpu_fallback_mode = True

    # 글로벌에 할당
    print("Model loaded.")
    text_encoder = text_encoder_local
    text_encoder_2 = text_encoder_2_local
    tokenizer = tokenizer_local
    tokenizer_2 = tokenizer_2_local
    vae = vae_local
    feature_extractor = feature_extractor_local
    image_encoder = image_encoder_local
    transformer = transformer_local

#############################################
# Worker 로직 (두 번째 코드) 그대로
#############################################
stream = AsyncStream()

outputs_folder = './outputs/'
os.makedirs(outputs_folder, exist_ok=True)

@torch.no_grad()
def worker(
    input_image, prompt, n_prompt, seed,
    total_second_length, latent_window_size, steps,
    cfg, gs, rs, gpu_memory_preservation, use_teacache
):
    """
    실제 샘플링 로직 (두 번째 코드 기반)
    """
    load_global_models()  # 모델 로딩
    global text_encoder, text_encoder_2, tokenizer, tokenizer_2
    global vae, feature_extractor, image_encoder, transformer
    global last_update_time

    # 최대 4초로 고정
    total_second_length = min(total_second_length, 4.0)

    total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))

    job_id = generate_timestamp()

    stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Starting ...'))))

    try:
        # GPU 적을 경우 Unload
        if not high_vram and GPU_AVAILABLE:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )

        # Text encoding
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Text encoding ...'))))

        if not high_vram and GPU_AVAILABLE:
            fake_diffusers_current_device(text_encoder, gpu)
            load_model_as_complete(text_encoder_2, target_device=gpu)

        llama_vec, clip_l_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
        if cfg == 1.0:
            llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
        else:
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        llama_vec, llama_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

        # Image processing
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Image processing ...'))))

        H, W, C = input_image.shape
        height, width = find_nearest_bucket(H, W, resolution=640)

        if cpu_fallback_mode:
            height = min(height, 320)
            width = min(width, 320)

        input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)

        Image.fromarray(input_image_np).save(os.path.join(outputs_folder, f'{job_id}.png'))

        input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]

        # VAE encode
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'VAE encoding ...'))))

        if not high_vram and GPU_AVAILABLE:
            load_model_as_complete(vae, target_device=gpu)
        start_latent = vae_encode(input_image_pt, vae)

        # CLIP Vision
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'CLIP Vision encoding ...'))))

        if not high_vram and GPU_AVAILABLE:
            load_model_as_complete(image_encoder, target_device=gpu)
        image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state

        # dtype
        llama_vec = llama_vec.to(transformer.dtype)
        llama_vec_n = llama_vec_n.to(transformer.dtype)
        clip_l_pooler = clip_l_pooler.to(transformer.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)

        # Start sampling
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Start sampling ...'))))

        rnd = torch.Generator("cpu").manual_seed(seed)

        # 초기 history latents
        history_latents = torch.zeros(size=(1, 16, 16 + 2 + 1, height // 8, width // 8), dtype=torch.float32).cpu()
        history_pixels = None

        # start_latent 붙이기
        history_latents = torch.cat([history_latents, start_latent.to(history_latents)], dim=2)
        total_generated_latent_frames = 1

        for section_index in range(total_latent_sections):
            if stream.input_queue.top() == 'end':
                stream.output_queue.push(('end', None))
                return

            print(f'Section {section_index+1}/{total_latent_sections}')

            if not high_vram and GPU_AVAILABLE:
                unload_complete_models()
                move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

            # teacache
            if use_teacache:
                transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
            else:
                transformer.initialize_teacache(enable_teacache=False)

            def callback(d):
                preview = d['denoised']
                preview = vae_decode_fake(preview)
                preview = (preview * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                preview = einops.rearrange(preview, 'b c t h w -> (b h) (t w) c')

                if stream.input_queue.top() == 'end':
                    stream.output_queue.push(('end', None))
                    raise KeyboardInterrupt('User stops generation.')

                current_step = d['i'] + 1
                percentage = int(100.0 * current_step / steps)
                hint = f'Sampling {current_step}/{steps}'
                desc = f'Section {section_index+1}/{total_latent_sections}'
                stream.output_queue.push(('progress', (preview, desc, make_progress_bar_html(percentage, hint))))
                return

            # indices
            frames_per_section = latent_window_size * 4 - 3
            indices = torch.arange(0, sum([1, 16, 2, 1, latent_window_size])).unsqueeze(0)
            (
                clean_latent_indices_start,
                clean_latent_4x_indices,
                clean_latent_2x_indices,
                clean_latent_1x_indices,
                latent_indices
            ) = indices.split([1, 16, 2, 1, latent_window_size], dim=1)

            clean_latent_indices = torch.cat([clean_latent_indices_start, clean_latent_1x_indices], dim=1)

            clean_latents_4x, clean_latents_2x, clean_latents_1x = history_latents[:, :, -19:, :, :].split([16, 2, 1], dim=2)
            clean_latents = torch.cat([start_latent.to(history_latents), clean_latents_1x], dim=2)

            try:
                generated_latents = sample_hunyuan(
                    transformer=transformer,
                    sampler='unipc',
                    width=width,
                    height=height,
                    frames=frames_per_section,
                    real_guidance_scale=cfg,
                    distilled_guidance_scale=gs,
                    guidance_rescale=rs,
                    num_inference_steps=steps,
                    generator=rnd,
                    prompt_embeds=llama_vec,
                    prompt_embeds_mask=llama_mask,
                    prompt_poolers=clip_l_pooler,
                    negative_prompt_embeds=llama_vec_n,
                    negative_prompt_embeds_mask=llama_mask_n,
                    negative_prompt_poolers=clip_l_pooler_n,
                    device=gpu if GPU_AVAILABLE else cpu,
                    dtype=torch.bfloat16,
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
            except KeyboardInterrupt:
                print("User cancelled.")
                stream.output_queue.push(('end', None))
                return
            except Exception as e:
                traceback.print_exc()
                stream.output_queue.push(('end', None))
                return

            total_generated_latent_frames += generated_latents.shape[2]
            history_latents = torch.cat([history_latents, generated_latents.to(history_latents)], dim=2)

            if not high_vram and GPU_AVAILABLE:
                offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=8)
                load_model_as_complete(vae, target_device=gpu)

            real_history_latents = history_latents[:, :, -total_generated_latent_frames:, :, :]

            if history_pixels is None:
                history_pixels = vae_decode(real_history_latents, vae).cpu()
            else:
                section_latent_frames = latent_window_size * 2
                overlapped_frames = frames_per_section
                current_pixels = vae_decode(real_history_latents[:, :, -section_latent_frames:], vae).cpu()
                history_pixels = soft_append_bcthw(history_pixels, current_pixels, overlapped_frames)

            if not high_vram and GPU_AVAILABLE:
                unload_complete_models()

            output_filename = os.path.join(outputs_folder, f'{job_id}_{total_generated_latent_frames}.mp4')
            save_bcthw_as_mp4(history_pixels, output_filename, fps=30, crf=16)  # CRF=16

            stream.output_queue.push(('file', output_filename))

    except:
        traceback.print_exc()
        if not high_vram and GPU_AVAILABLE:
            unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer)

    stream.output_queue.push(('end', None))
    return

def end_process():
    """
    중단 요청
    """
    global stream
    stream.input_queue.push('end')

# Gradio에서 이 worker 함수를 비동기로 호출
def process(
    input_image, prompt, n_prompt, seed,
    total_second_length, latent_window_size, steps,
    cfg, gs, rs, gpu_memory_preservation, use_teacache
):
    global stream
    if input_image is None:
        raise ValueError("No input image provided.")

    yield None, None, "", "", gr.update(interactive=False), gr.update(interactive=True)

    stream = AsyncStream()
    async_run(
        worker,
        input_image, prompt, n_prompt, seed,
        total_second_length, latent_window_size, steps,
        cfg, gs, rs, gpu_memory_preservation, use_teacache
    )

    output_filename = None
    prev_filename = None
    error_message = None

    while True:
        flag, data = stream.output_queue.next()
        if flag == 'file':
            output_filename = data
            prev_filename = output_filename
            yield output_filename, gr.update(), gr.update(), "", gr.update(interactive=False), gr.update(interactive=True)

        elif flag == 'progress':
            preview, desc, html = data
            yield gr.update(), gr.update(visible=True, value=preview), desc, html, gr.update(interactive=False), gr.update(interactive=True)

        elif flag == 'error':
            error_message = data
            print(f"Error: {error_message}")

        elif flag == 'end':
            if output_filename is None and prev_filename:
                output_filename = prev_filename
            # 에러가 있었으면 에러 표시
            if error_message:
                yield (
                    output_filename,  # 마지막 파일 (또는 None)
                    gr.update(visible=False),
                    gr.update(),
                    f"<div style='color:red;'>{error_message}</div>",
                    gr.update(interactive=True),
                    gr.update(interactive=False)
                )
            else:
                yield (
                    output_filename, gr.update(visible=False), gr.update(), "", gr.update(interactive=True), gr.update(interactive=False)
                )
            break

# UI CSS
def make_custom_css():
    base_progress_css = make_progress_bar_css()
    pastel_css = """
    body {
        background: #faf9ff !important; 
        font-family: "Noto Sans", sans-serif;
    }
    #app-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 1rem;
        position: relative;
    }
    #app-container h1 {
        color: #5F5AA2;
        margin-bottom: 1.2rem;
        font-weight: 700;
        text-shadow: 1px 1px 2px #bbb;
    }
    .gr-panel {
        background: #ffffffcc;
        border: 1px solid #e1dff0;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .button-container button {
        min-height: 45px;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 6px;
    }
    .button-container button#start-button {
        background-color: #A289E3 !important;
        color: #fff !important;
        border: 1px solid #a58de2;
    }
    .button-container button#stop-button {
        background-color: #F48A9B !important;
        color: #fff !important;
        border: 1px solid #f18fa0;
    }
    .button-container button:hover {
        filter: brightness(0.95);
    }
    .preview-container, .video-container {
        border: 1px solid #ded9f2;
        border-radius: 8px;
        overflow: hidden;
    }
    .progress-container {
        margin-top: 15px;
        margin-bottom: 15px;
    }
    .error-message {
        background-color: #FFF5F5;
        border: 1px solid #FED7D7;
        color: #E53E3E;
        padding: 10px;
        border-radius: 4px;
        margin-top: 10px;
        font-weight: 500;
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
    return base_progress_css + pastel_css

css = make_custom_css()

# 샘플 프롬프트
quick_prompts = [
    ["The girl dances gracefully, with clear movements, full of charm."],
    ["A character doing some simple body movements."]
]

# Gradio UI
block = gr.Blocks(css=css).queue()
with block:
    gr.HTML("<div id='app-container'><h1>FramePack - Image to Video Generation</h1></div>")

    with gr.Row(elem_classes="mobile-full-width"):
        # 왼쪽
        with gr.Column(scale=1, elem_classes="gr-panel"):
            input_image = gr.Image(
                label=get_translation("upload_image"),
                type="numpy",
                height=320
            )
            prompt = gr.Textbox(
                label=get_translation("prompt"),
                value=''
            )

            example_quick_prompts = gr.Dataset(
                samples=quick_prompts,
                label=get_translation("quick_prompts"),
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

        # 오른쪽
        with gr.Column(scale=1, elem_classes="gr-panel"):
            with gr.Row(elem_classes="button-container"):
                start_button = gr.Button(
                    value=get_translation("start_generation"),
                    elem_id="start-button",
                    variant="primary"
                )
                stop_button = gr.Button(
                    value=get_translation("stop_generation"),
                    elem_id="stop-button",
                    interactive=False
                )

            result_video = gr.Video(
                label=get_translation("generated_video"),
                autoplay=True,
                loop=True,
                height=320,
                elem_classes="video-container"
            )
            preview_image = gr.Image(
                label=get_translation("next_latents"),
                visible=False,
                height=150,
                elem_classes="preview-container"
            )
            gr.Markdown(get_translation("sampling_note"))

            with gr.Group(elem_classes="progress-container"):
                progress_desc = gr.Markdown('')
                progress_bar = gr.HTML('')

            error_message = gr.HTML('', visible=True)

    # Advanced
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
        # 기본 2초, 최대 4초
        total_second_length = gr.Slider(
            label=get_translation("video_length"),
            minimum=1,
            maximum=4,
            value=2,
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

    # 버튼 처리
    inputs_list = [
        input_image, prompt, n_prompt, seed,
        total_second_length, latent_window_size, steps,
        cfg, gs, rs, gpu_memory_preservation, use_teacache
    ]
    start_button.click(
        fn=process,
        inputs=inputs_list,
        outputs=[result_video, preview_image, progress_desc, progress_bar, start_button, stop_button]
    )
    stop_button.click(fn=end_process)

block.launch()
