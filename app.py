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

# Check GPU memory
free_mem_gb = get_cuda_free_memory_gb(gpu)
high_vram = free_mem_gb > 60

print(f'Free VRAM {free_mem_gb} GB')
print(f'High-VRAM Mode: {high_vram}')

# Load models
text_encoder = LlamaModel.from_pretrained(
    "hunyuanvideo-community/HunyuanVideo",
    subfolder='text_encoder',
    torch_dtype=torch.float16
).cpu()
text_encoder_2 = CLIPTextModel.from_pretrained(
    "hunyuanvideo-community/HunyuanVideo",
    subfolder='text_encoder_2',
    torch_dtype=torch.float16
).cpu()
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
    torch_dtype=torch.float16
).cpu()

feature_extractor = SiglipImageProcessor.from_pretrained(
    "lllyasviel/flux_redux_bfl",
    subfolder='feature_extractor'
)
image_encoder = SiglipVisionModel.from_pretrained(
    "lllyasviel/flux_redux_bfl",
    subfolder='image_encoder',
    torch_dtype=torch.float16
).cpu()

transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(
    'lllyasviel/FramePack_F1_I2V_HY_20250503',
    torch_dtype=torch.bfloat16
).cpu()

# Evaluation mode
vae.eval()
text_encoder.eval()
text_encoder_2.eval()
image_encoder.eval()
transformer.eval()

# Slicing/Tiling for low VRAM
if not high_vram:
    vae.enable_slicing()
    vae.enable_tiling()

transformer.high_quality_fp32_output_for_inference = True
print('transformer.high_quality_fp32_output_for_inference = True')

# Move to correct dtype
transformer.to(dtype=torch.bfloat16)
vae.to(dtype=torch.float16)
image_encoder.to(dtype=torch.float16)
text_encoder.to(dtype=torch.float16)
text_encoder_2.to(dtype=torch.float16)

# No gradient
vae.requires_grad_(False)
text_encoder.requires_grad_(False)
text_encoder_2.requires_grad_(False)
image_encoder.requires_grad_(False)
transformer.requires_grad_(False)

# DynamicSwap if low VRAM
if not high_vram:
    DynamicSwapInstaller.install_model(transformer, device=gpu)
    DynamicSwapInstaller.install_model(text_encoder, device=gpu)
else:
    text_encoder.to(gpu)
    text_encoder_2.to(gpu)
    image_encoder.to(gpu)
    vae.to(gpu)
    transformer.to(gpu)

stream = AsyncStream()

outputs_folder = './outputs/'
os.makedirs(outputs_folder, exist_ok=True)

examples = [
    ["img_examples/1.png", "The girl dances gracefully, with clear movements, full of charm."],
    ["img_examples/2.jpg", "The man dances flamboyantly, swinging his hips and striking bold poses with dramatic flair."],
    ["img_examples/3.png", "The woman dances elegantly among the blossoms, spinning slowly with flowing sleeves and graceful hand movements."]
]

# Example generation (optional)
def generate_examples(input_image, prompt):
    t2v=False
    n_prompt=""
    seed=31337
    total_second_length=5
    latent_window_size=9
    steps=25
    cfg=1.0
    gs=10.0
    rs=0.0
    gpu_memory_preservation=6
    use_teacache=True
    mp4_crf=16

    global stream

    if t2v:
        default_height, default_width = 640, 640
        input_image = np.ones((default_height, default_width, 3), dtype=np.uint8) * 255
        print("No input image provided. Using a blank white image.")

    yield None, None, '', '', gr.update(interactive=False), gr.update(interactive=True)

    stream = AsyncStream()

    async_run(
        worker, input_image, prompt, n_prompt, seed,
        total_second_length, latent_window_size, steps,
        cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf
    )

    output_filename = None

    while True:
        flag, data = stream.output_queue.next()

        if flag == 'file':
            output_filename = data
            yield (
                output_filename,
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(interactive=False),
                gr.update(interactive=True)
            )

        if flag == 'progress':
            preview, desc, html = data
            yield (
                gr.update(),
                gr.update(visible=True, value=preview),
                desc,
                html,
                gr.update(interactive=False),
                gr.update(interactive=True)
            )

        if flag == 'end':
            yield (
                output_filename,
                gr.update(visible=False),
                gr.update(),
                '',
                gr.update(interactive=True),
                gr.update(interactive=False)
            )
            break

@torch.no_grad()
def worker(
    input_image, prompt, n_prompt, seed,
    total_second_length, latent_window_size, steps,
    cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf
):
    # Calculate total sections
    total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))

    job_id = generate_timestamp()

    stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Starting ...'))))

    try:
        # Unload if VRAM is low
        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )

        # Text encoding
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Text encoding ...'))))

        if not high_vram:
            fake_diffusers_current_device(text_encoder, gpu)
            load_model_as_complete(text_encoder_2, target_device=gpu)

        llama_vec, clip_l_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        if cfg == 1:
            llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
        else:
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

        # Process image
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Image processing ...'))))

        H, W, C = input_image.shape
        height, width = find_nearest_bucket(H, W, resolution=640)
        input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)

        Image.fromarray(input_image_np).save(os.path.join(outputs_folder, f'{job_id}.png'))

        input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]

        # VAE encoding
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'VAE encoding ...'))))

        if not high_vram:
            load_model_as_complete(vae, target_device=gpu)
        start_latent = vae_encode(input_image_pt, vae)

        # CLIP Vision
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'CLIP Vision encoding ...'))))

        if not high_vram:
            load_model_as_complete(image_encoder, target_device=gpu)
        image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state

        # Convert dtype
        llama_vec = llama_vec.to(transformer.dtype)
        llama_vec_n = llama_vec_n.to(transformer.dtype)
        clip_l_pooler = clip_l_pooler.to(transformer.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)

        # Start sampling
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Start sampling ...'))))

        rnd = torch.Generator("cpu").manual_seed(seed)

        history_latents = torch.zeros(
            size=(1, 16, 16 + 2 + 1, height // 8, width // 8),
            dtype=torch.float32
        ).cpu()
        history_pixels = None

        # Add start_latent
        history_latents = torch.cat([history_latents, start_latent.to(history_latents)], dim=2)
        total_generated_latent_frames = 1

        for section_index in range(total_latent_sections):
            if stream.input_queue.top() == 'end':
                stream.output_queue.push(('end', None))
                return

            print(f'section_index = {section_index}, total_latent_sections = {total_latent_sections}')

            if not high_vram:
                unload_complete_models()
                move_model_to_device_with_memory_preservation(
                    transformer, target_device=gpu,
                    preserved_memory_gb=gpu_memory_preservation
                )

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
                    raise KeyboardInterrupt('User ends the task.')

                current_step = d['i'] + 1
                percentage = int(100.0 * current_step / steps)
                hint = f'Sampling {current_step}/{steps}'
                desc = f'Total generated frames: {int(max(0, total_generated_latent_frames * 4 - 3))}'
                stream.output_queue.push(('progress', (preview, desc, make_progress_bar_html(percentage, hint))))
                return

            indices = torch.arange(
                0, sum([1, 16, 2, 1, latent_window_size])
            ).unsqueeze(0)
            (
                clean_latent_indices_start,
                clean_latent_4x_indices,
                clean_latent_2x_indices,
                clean_latent_1x_indices,
                latent_indices
            ) = indices.split([1, 16, 2, 1, latent_window_size], dim=1)

            clean_latent_indices = torch.cat([clean_latent_indices_start, clean_latent_1x_indices], dim=1)

            clean_latents_4x, clean_latents_2x, clean_latents_1x = history_latents[
                :, :, -sum([16, 2, 1]):, :, :
            ].split([16, 2, 1], dim=2)

            clean_latents = torch.cat(
                [start_latent.to(history_latents), clean_latents_1x],
                dim=2
            )

            generated_latents = sample_hunyuan(
                transformer=transformer,
                sampler='unipc',
                width=width,
                height=height,
                frames=latent_window_size * 4 - 3,
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
                device=gpu,
                dtype=torch.bfloat16,
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

            total_generated_latent_frames += int(generated_latents.shape[2])
            history_latents = torch.cat([history_latents, generated_latents.to(history_latents)], dim=2)

            if not high_vram:
                offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=8)
                load_model_as_complete(vae, target_device=gpu)

            real_history_latents = history_latents[:, :, -total_generated_latent_frames:, :, :]

            if history_pixels is None:
                history_pixels = vae_decode(real_history_latents, vae).cpu()
            else:
                section_latent_frames = latent_window_size * 2
                overlapped_frames = latent_window_size * 4 - 3

                current_pixels = vae_decode(
                    real_history_latents[:, :, -section_latent_frames:], vae
                ).cpu()
                history_pixels = soft_append_bcthw(
                    history_pixels, current_pixels, overlapped_frames
                )

            if not high_vram:
                unload_complete_models()

            output_filename = os.path.join(outputs_folder, f'{job_id}_{total_generated_latent_frames}.mp4')
            save_bcthw_as_mp4(history_pixels, output_filename, fps=30, crf=mp4_crf)

            print(f'Decoded. Latent shape {real_history_latents.shape}; pixel shape {history_pixels.shape}')

            stream.output_queue.push(('file', output_filename))

    except:
        traceback.print_exc()
        if not high_vram:
            unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer)

    stream.output_queue.push(('end', None))
    return

def get_duration(
    input_image, prompt, t2v, n_prompt,
    seed, total_second_length, latent_window_size,
    steps, cfg, gs, rs, gpu_memory_preservation,
    use_teacache, mp4_crf
):
    return total_second_length * 60

@spaces.GPU(duration=get_duration)
def process(
    input_image, prompt, t2v=False, n_prompt="", seed=31337,
    total_second_length=5, latent_window_size=9, steps=25,
    cfg=1.0, gs=10.0, rs=0.0, gpu_memory_preservation=6,
    use_teacache=True, mp4_crf=16
):
    global stream
    if t2v:
        default_height, default_width = 640, 640
        input_image = np.ones((default_height, default_width, 3), dtype=np.uint8) * 255
        print("No input image provided. Using a blank white image.")
    else:
        composite_rgba_uint8 = input_image["composite"]

        rgb_uint8 = composite_rgba_uint8[:, :, :3]
        mask_uint8 = composite_rgba_uint8[:, :, 3]

        h, w = rgb_uint8.shape[:2]
        background_uint8 = np.full((h, w, 3), 255, dtype=np.uint8)

        alpha_normalized_float32 = mask_uint8.astype(np.float32) / 255.0
        alpha_mask_float32 = np.stack([alpha_normalized_float32]*3, axis=2)

        blended_image_float32 = rgb_uint8.astype(np.float32) * alpha_mask_float32 + \
                                background_uint8.astype(np.float32) * (1.0 - alpha_mask_float32)

        input_image = np.clip(blended_image_float32, 0, 255).astype(np.uint8)

    yield None, None, '', '', gr.update(interactive=False), gr.update(interactive=True)

    stream = AsyncStream()

    async_run(
        worker, input_image, prompt, n_prompt, seed,
        total_second_length, latent_window_size, steps,
        cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf
    )

    output_filename = None

    while True:
        flag, data = stream.output_queue.next()

        if flag == 'file':
            output_filename = data
            yield (
                output_filename,
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(interactive=False),
                gr.update(interactive=True)
            )

        elif flag == 'progress':
            preview, desc, html = data
            yield (
                gr.update(),
                gr.update(visible=True, value=preview),
                desc,
                html,
                gr.update(interactive=False),
                gr.update(interactive=True)
            )

        elif flag == 'end':
            yield (
                output_filename,
                gr.update(visible=False),
                gr.update(),
                '',
                gr.update(interactive=True),
                gr.update(interactive=False)
            )
            break

def end_process():
    stream.input_queue.push('end')


quick_prompts = [
    'The girl dances gracefully, with clear movements, full of charm.',
    'A character doing some simple body movements.'
]
quick_prompts = [[x] for x in quick_prompts]


def make_custom_css():
    base_progress_css = make_progress_bar_css()
    extra_css = """
    body {
        background: #fafbfe !important;
        font-family: "Noto Sans", sans-serif;
    }
    #title-container {
        text-align: center;
        padding: 20px 0;
        background: linear-gradient(135deg, #a8c0ff 0%, #fbc2eb 100%);
        border-radius: 0 0 10px 10px;
        margin-bottom: 20px;
    }
    #title-container h1 {
        color: white;
        font-size: 2rem;
        margin: 0;
        font-weight: 800;
        text-shadow: 1px 2px 2px rgba(0,0,0,0.1);
    }
    .gr-panel {
        background: #ffffffcc;
        backdrop-filter: blur(4px);
        border: 1px solid #dcdcf7;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .gr-box > label {
        font-size: 0.9rem;
        font-weight: 600;
        color: #333;
    }
    .button-container button {
        min-height: 48px;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 8px;
        border: none !important;
    }
    .button-container button#start-button {
        background-color: #4b9ffa !important;
        color: #fff;
    }
    .button-container button#stop-button {
        background-color: #ef5d84 !important;
        color: #fff;
    }
    .button-container button:hover {
        filter: brightness(0.97);
    }
    .no-generating-animation {
        margin-top: 10px;
        margin-bottom: 10px;
    }
    """
    return base_progress_css + extra_css

css = make_custom_css()

block = gr.Blocks(css=css).queue()
with block:
    # Title (use gr.Group instead of gr.Box for older Gradio versions)
    with gr.Group(elem_id="title-container"):
        gr.Markdown("<h1>FramePack I2V</h1>")

    gr.Markdown("""
    ### Video diffusion, but feels like image diffusion
    FramePack I2V - a model that predicts future frames from past frames,
    letting you generate short animations from a single image plus text prompt.
    """)

    with gr.Row():
        with gr.Column():
            input_image = gr.ImageEditor(
                type="numpy",
                label="Image Editor (use Brush for mask)",
                height=320,
                brush=gr.Brush(colors=["#ffffff"])
            )
            prompt = gr.Textbox(label="Prompt", value='')
            t2v = gr.Checkbox(label="Only Text to Video (ignore image)?", value=False)

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

            with gr.Row(elem_classes="button-container"):
                start_button = gr.Button(value="Start Generation", elem_id="start-button")
                end_button = gr.Button(value="Stop Generation", elem_id="stop-button", interactive=False)

            total_second_length = gr.Slider(
                label="Total Video Length (Seconds)",
                minimum=1,
                maximum=5,
                value=2,
                step=0.1
            )

            with gr.Group():
                with gr.Accordion("Advanced Settings", open=False):
                    use_teacache = gr.Checkbox(
                        label='Use TeaCache',
                        value=True,
                        info='Faster speed, but may worsen hands/fingers.'
                    )
                    n_prompt = gr.Textbox(label="Negative Prompt", value="", visible=False)
                    seed = gr.Number(label="Seed", value=31337, precision=0)
                    latent_window_size = gr.Slider(
                        label="Latent Window Size",
                        minimum=1, maximum=33,
                        value=9, step=1,
                        visible=False
                    )
                    steps = gr.Slider(
                        label="Steps",
                        minimum=1, maximum=100,
                        value=25, step=1,
                        info='Not recommended to change drastically.'
                    )
                    cfg = gr.Slider(
                        label="CFG Scale",
                        minimum=1.0, maximum=32.0,
                        value=1.0, step=0.01,
                        visible=False
                    )
                    gs = gr.Slider(
                        label="Distilled CFG Scale",
                        minimum=1.0, maximum=32.0,
                        value=10.0, step=0.01,
                        info='Not recommended to change drastically.'
                    )
                    rs = gr.Slider(
                        label="CFG Re-Scale",
                        minimum=0.0, maximum=1.0,
                        value=0.0, step=0.01,
                        visible=False
                    )
                    gpu_memory_preservation = gr.Slider(
                        label="GPU Memory Preservation (GB)",
                        minimum=6, maximum=128,
                        value=6, step=0.1,
                        info="Increase if OOM occurs, but slower."
                    )
                    mp4_crf = gr.Slider(
                        label="MP4 Compression (CRF)",
                        minimum=0, maximum=100,
                        value=16, step=1,
                        info="Lower = better quality. 16 recommended."
                    )

        with gr.Column():
            preview_image = gr.Image(
                label="Preview Latents",
                height=200,
                visible=False
            )
            result_video = gr.Video(
                label="Finished Frames",
                autoplay=True,
                show_share_button=False,
                height=512,
                loop=True
            )
            progress_desc = gr.Markdown('', elem_classes='no-generating-animation')
            progress_bar = gr.HTML('', elem_classes='no-generating-animation')

    # Extra info
    gr.HTML("""
    <div style="text-align:center; margin-top:20px;">
      Share your outputs or get inspired by searching 
      <a href="https://x.com/search?q=framepack&f=live" target="_blank">#framepack</a> on Twitter!
    </div>
    """)

    ips = [
        input_image, prompt, t2v, n_prompt, seed,
        total_second_length, latent_window_size,
        steps, cfg, gs, rs, gpu_memory_preservation,
        use_teacache, mp4_crf
    ]
    start_button.click(
        fn=process,
        inputs=ips,
        outputs=[result_video, preview_image, progress_desc, progress_bar, start_button, end_button]
    )
    end_button.click(fn=end_process)

    # If you want examples, uncomment below:
    # gr.Examples(
    #     examples=examples,
    #     inputs=[input_image, prompt],
    #     outputs=[result_video, preview_image, progress_desc, progress_bar, start_button, end_button],
    #     fn=generate_examples,
    #     cache_examples=True
    # )

block.launch(share=True)
