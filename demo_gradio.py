
import os
import argparse



import torch
import gradio as gr


import numpy as np
import einops
import traceback

from PIL import Image
from diffusers import AutoencoderKLHunyuanVideo
from transformers import (
    LlamaModel, CLIPTextModel,
    LlamaTokenizerFast, CLIPTokenizer,
    SiglipImageProcessor, SiglipVisionModel
)

from diffusers_helper.hf_login import login
from diffusers_helper.hunyuan import (
    encode_prompt_conds, vae_decode, vae_encode,
    vae_decode_fake
)
from diffusers_helper.utils import (
    save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw,
    resize_and_center_crop, generate_timestamp
)
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import (
    gpu, get_cuda_free_memory_gb, unload_complete_models, load_model_as_complete,
    DynamicSwapInstaller, move_model_to_device_with_memory_preservation,
    offload_model_from_device_for_memory_preservation, fake_diffusers_current_device
)
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.thread_utils import AsyncStream, async_run


# --- Args and config ---
parser = argparse.ArgumentParser()
parser.add_argument('--share', action='store_true')
parser.add_argument('--server', type=str, default='0.0.0.0')
parser.add_argument('--port', type=int, required=False)
parser.add_argument('--inbrowser', action='store_true')
args = parser.parse_args()

os.environ['HF_HOME'] = os.path.abspath(
    os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download'))
)

print(args)

free_mem_gb = get_cuda_free_memory_gb(gpu)
high_vram = free_mem_gb > 60

print(f'Free VRAM {free_mem_gb} GB')
print(f'High-VRAM Mode: {high_vram}')

# --- Load models ---
text_encoder = LlamaModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder', torch_dtype=torch.float16).cpu()
text_encoder_2 = CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder_2', torch_dtype=torch.float16).cpu()
tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')
vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()

feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16).cpu()

transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePackI2V_HY', torch_dtype=torch.bfloat16).cpu()

vae.eval(), text_encoder.eval(), text_encoder_2.eval(), image_encoder.eval(), transformer.eval()





if not high_vram:
    vae.enable_slicing()
    vae.enable_tiling()

transformer.high_quality_fp32_output_for_inference = True


transformer.to(dtype=torch.bfloat16)
vae.to(dtype=torch.float16)
image_encoder.to(dtype=torch.float16)
text_encoder.to(dtype=torch.float16)
text_encoder_2.to(dtype=torch.float16)

for model in [vae, text_encoder, text_encoder_2, image_encoder, transformer]:
    model.requires_grad_(False)




if not high_vram:

    DynamicSwapInstaller.install_model(transformer, device=gpu)
    DynamicSwapInstaller.install_model(text_encoder, device=gpu)
else:
    transformer.to(gpu)

stream = AsyncStream()

outputs_folder = './outputs/'
os.makedirs(outputs_folder, exist_ok=True)

# --- UI + CSS ---
def make_progress_bar_css():
    return """
    body, .gradio-container {
        background-color: #000000 !important;
        color: #FFFFFF !important;
    }
    .gr-button, .gr-input, .gr-textbox, .gr-slider, .gr-checkbox {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
        border-color: #444 !important;
    }
    .gr-button:hover {
        background-color: #333 !important;
    }
    .gr-markdown {
        color: #ddd !important;
    }
    .gr-image-preview, .gr-video {
        background-color: #111 !important;
    }
    """

def end_process():
    stream.input_queue.push('end')






















































































































































































































def process(input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache):
    global stream
    assert input_image is not None, 'No input image!'

    yield None, None, '', '', gr.update(interactive=False), gr.update(interactive=True)

    stream = AsyncStream()

    async_run(worker, input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache)

    output_filename = None

    while True:
        flag, data = stream.output_queue.next()

        if flag == 'file':
            output_filename = data
            yield output_filename, gr.update(), gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=True)

        if flag == 'progress':
            preview, desc, html = data
            yield gr.update(), gr.update(visible=True, value=preview), desc, html, gr.update(interactive=False), gr.update(interactive=True)

        if flag == 'end':
            yield output_filename, gr.update(visible=False), gr.update(), '', gr.update(interactive=True), gr.update(interactive=False)
            break






quick_prompts = [
    'The girl dances gracefully, with clear movements, full of charm.',
    'A character doing some simple body movements.',
]
quick_prompts = [[x] for x in quick_prompts]


css = make_progress_bar_css()

block = gr.Blocks(css=css).queue()
with block:
    gr.Markdown('# FramePack')
                end_button = gr.Button(value="End Generation", interactive=False)

            with gr.Group():
                use_teacache = gr.Checkbox(label='Use TeaCache', value=True)
                n_prompt = gr.Textbox(label="Negative Prompt", value="", visible=False)

                seed = gr.Number(label="Seed", value=31337, precision=0)

                total_second_length = gr.Slider(label="Total Video Length (Seconds)", minimum=1, maximum=120, value=5, step=0.1)
                latent_window_size = gr.Slider(label="Latent Window Size", minimum=1, maximum=33, value=9, step=1, visible=False)
                steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=25, step=1)
                cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=1.0, step=0.01, visible=False)
                gs = gr.Slider(label="Distilled CFG Scale", minimum=1.0, maximum=32.0, value=10.0, step=0.01)
                rs = gr.Slider(label="CFG Re-Scale", minimum=0.0, maximum=1.0, value=0.0, step=0.01, visible=False)
                gpu_memory_preservation = gr.Slider(label="GPU Inference Preserved Memory (GB)", minimum=6, maximum=128, value=6, step=0.1)



        with gr.Column():
            preview_image = gr.Image(label="Next Latents", height=200, visible=False)
            result_video = gr.Video(label="Finished Frames", autoplay=True, show_share_button=False, height=512, loop=True)
            gr.Markdown('Note: The ending actions are generated before the start. Wait for full video.')
            progress_desc = gr.Markdown('', elem_classes='no-generating-animation')
            progress_bar = gr.HTML('', elem_classes='no-generating-animation')

    ips = [input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache]
    start_button.click(fn=process, inputs=ips, outputs=[result_video, preview_image, progress_desc, progress_bar, start_button, end_button])
    end_button.click(fn=end_process)


block.launch(
    server_name=args.server,
    server_port=args.port,