import streamlit as st
import io
import os
import math
import torch
import tempfile
from diffusion import GeneralizedDiffusion
from torchvision.utils import make_grid
from string import ascii_letters, digits
import time
from contextlib import contextmanager
import torch.nn.functional as F
from functools import partial
from PIL import Image
import cv2
import uuid

st.title("Diffusion Model Showcase")

dataset = st.sidebar.radio(
    "Dataset", options=["CIFAR-10", "CelebA", "AnimeFace"], index=1)
if dataset == "CIFAR-10":
    padding = 1
else:
    padding = 2

use_ddim = st.sidebar.checkbox("Use DDIM", value=False)
if use_ddim:
    schedule = st.sidebar.radio(
        "DDIM schedule", options=["linear", "quadratic"], index=0)
else:
    schedule = "linear"

timesteps = st.sidebar.number_input(
    "Number of denoising timesteps", 1, 1000, 100)

seed = st.sidebar.number_input("Random seed", 0, 65535, 1234)

n_images = st.sidebar.number_input("Number of images to generate", 1, 64, 9, step=1)
scale_factor = st.sidebar.number_input(
    "Scale factor of generated image(s)", 1.0, 4.0, 4.0, step=0.5)
mode = st.sidebar.radio(
    "Interpolation mode", options=["nearest", "bilinear", "bicubic"], index=2)

st.info(f"Dataset: {dataset}")
if use_ddim:
    st.info("DDIM is used for generation.")
st.info(f"Denoising with {timesteps} steps.")


def load_components(dataset, use_ddim, denoise_steps, denoise_schedule, n_images):
    eta = 0. if use_ddim else 1.
    model_var_type = "fixed-small" if use_ddim else \
        {"cifar10": "fixed-large"}.get(dataset, "fixed-small")
    diffusion = GeneralizedDiffusion(
        model_var_type=model_var_type,
        eta=eta, subseq_size=denoise_steps, schedule=denoise_schedule)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(f"./models/ddpm_{dataset}.pt", map_location=device)
    model.eval()
    input_shape = {"cifar10": (n_images, 3, 32, 32)}.get(dataset, (n_images, 3, 64, 64))
    return diffusion, model, input_shape


def _tensor2img(x, padding=2, scale_factor=1.0, mode="bicubic"):
    if x.ndim == 4:
        if x.shape[0] > 1:
            nrow = math.ceil(math.sqrt(x.shape[0]))
            x = make_grid(
                x, nrow=nrow, normalize=False, padding=padding, pad_value=-1)
        else:
            x = x.squeeze(0)
    if scale_factor != 1.0:
        x = F.interpolate(
            x.unsqueeze(0), scale_factor=scale_factor, mode=mode
        ).squeeze(0)
    x = x.permute(1, 2, 0)
    x = (x * 127.5 + 127.5).clamp(0, 255).round().to(torch.uint8)
    return x.numpy()


tensor2img = partial(_tensor2img, padding=padding, scale_factor=scale_factor, mode=mode)

progressive = st.sidebar.checkbox("Progressive sampling", value=False)
if progressive:
    animated = st.sidebar.checkbox("Animated", value=False)
else:
    animated = False


@st.cache(allow_output_mutation=True, show_spinner=False)
def generate(seed, progressive, **diffusion_kwargs):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    diffusion, model, input_shape = load_components(**diffusion_kwargs)
    device = next(model.parameters()).device
    noise = torch.randn(input_shape, device=device)
    with st.spinner("Waiting for the generation to be completed..."):
        if progressive:
            x_0, sample_path = diffusion.p_sample_progressive(model, noise)
            img = [x_0, ]
            for x_t in sample_path:
                img.append(x_t)
            img = img[-1::-1]
        else:
            img = [diffusion.p_sample(model, noise).cpu(), ]
    return img


def process_dataset_string(dataset_str):
    dataset_str = dataset_str.lower()
    dataset_str = "".join(ch for ch in dataset_str if ch in ascii_letters + digits)
    return dataset_str


@contextmanager
def timer(msg):
    start = time.perf_counter()
    yield None
    end = time.perf_counter()
    st.info(f"{msg} in {end - start:.2f}s.")


if st.button("Generate"):
    with timer("Image(s) generated"):
        diffusion_kwargs = dict(
            dataset=process_dataset_string(dataset), use_ddim=use_ddim,
            denoise_steps=timesteps, denoise_schedule=schedule, n_images=n_images)
        st.session_state["img"] = list(map(
            tensor2img, generate(seed, progressive, **diffusion_kwargs)))


def update_idx():
    st.session_state["image_index"] = st.session_state.get("image_slider", 0)


# noinspection PyUnresolvedReferences
def get_bytes(arr):
    buffer = io.BytesIO()
    if isinstance(arr, (list, tuple)):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmpfile = os.path.join(tmp_dir, f"{uuid.uuid4()}.mp4")
            try:
                writer = cv2.VideoWriter(
                    tmpfile, cv2.VideoWriter_fourcc(*"h264"), 24, arr[0].shape[:2])
                for a in arr:
                    writer.write(cv2.cvtColor(a, cv2.COLOR_RGB2BGR))  # RGB -> BGR
                writer.release()
                with open(tmpfile, "rb") as f:
                    buffer = f.read()
                os.remove(tmpfile)
                ext = "mp4"
            except:
                ims = [Image.fromarray(a) for a in arr]
                duration = max(42, 1000 // len(ims))
                duration = [duration for _ in range(len(ims))]
                duration[-1] = 1000
                ims[0].save(
                    buffer, format="WEBP", duration=duration,
                    append_images=ims[1:], save_all=True, loop=0)
                ext = "webp"
    else:
        Image.fromarray(arr).save(buffer, format="PNG")
        ext = "png"
    return buffer, ext


if "img" in st.session_state:
    if not animated:
        img_idx = st.session_state.get("image_index", 0) % len(st.session_state["img"])
        im_bytes, ext = get_bytes(st.session_state["img"][img_idx])
        if progressive:
            st.slider("Timestep", 0, timesteps, key="image_slider", on_change=update_idx)
    else:
        im_bytes, ext = get_bytes(st.session_state["img"])
    is_image = ext in {"png", "webp"}
    (st.image if is_image else st.video)(im_bytes)
    st.download_button(
        "Download image(s)" if is_image else "Download video",
        data=im_bytes,
        file_name=f"{uuid.uuid4()}.{ext}")
