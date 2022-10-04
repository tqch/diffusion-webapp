import streamlit as st
import io
import math
import torch
from diffusion import GeneralizedDiffusion
from torchvision.utils import make_grid
from string import ascii_letters
import time
from contextlib import contextmanager
import torch.nn.functional as F
from functools import partial
from PIL import Image
import uuid


st.title("Diffusion Model Showcase")

dataset = st.sidebar.radio(
    "Dataset", options=["CIFAR-10", "CelebA", "AnimeFace"], index=2)

use_ddim = st.sidebar.checkbox("Use DDIM", value=True)
if use_ddim:
    schedule = st.sidebar.radio(
        "DDIM schedule", options=["linear", "quadratic"], index=1)
else:
    schedule = "linear"

timesteps = st.sidebar.number_input(
    "Number of denoising timesteps", 1, 1000, 10)

seed = st.sidebar.number_input("Random seed", 0, 65535, 1234)

n_images = st.sidebar.number_input("Number of images to generate", 1, 64, 1, step=1)
scale_factor = st.sidebar.number_input(
    "Scale factor of generated image(s)", 1.0, 4.0, 1.0, step=0.5)
mode = st.sidebar.radio(
    "Interpolation mode", options=["nearest", "bilinear", "bicubic"], index=0)

st.info(f"Dataset: {dataset}")
if use_ddim:
    st.info("DDIM is used for generation.")
st.info(f"Denoising with {timesteps} steps.")


def load_components(dataset, use_ddim, denoise_steps, denoise_schedule, n_images):
    eta = 0. if use_ddim else 1.
    diffusion = GeneralizedDiffusion(
        model_var_type={"cifar10": "fixed-large"}.get(dataset, "fixed-small"),
        eta=eta, subseq_size=denoise_steps, schedule=denoise_schedule)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(f"./models/ddpm_{dataset}.pt", map_location=device)
    model.eval()
    input_shape = {"cifar10": (n_images, 3, 32, 32)}.get(dataset, (n_images, 3, 64, 64))
    return diffusion, model, input_shape


def _tensor2img(x, scale_factor=1.0, mode="bicubic"):
    if x.ndim == 4:
        if x.shape[0] > 1:
            nrow = math.ceil(math.sqrt(x.shape[0]))
            x = make_grid(x, nrow=nrow, normalize=False, pad_value=-1)
        else:
            x = x.squeeze(0)
    if scale_factor != 1.0:
        x = F.interpolate(
            x.unsqueeze(0), scale_factor=scale_factor, mode=mode
        ).squeeze(0)
    x = x.permute(1, 2, 0)
    x = (x * 127.5 + 127.5).clamp(0, 255).round().to(torch.uint8)
    return x.numpy()


tensor2img = partial(_tensor2img, scale_factor=scale_factor, mode=mode)

progressive = st.sidebar.checkbox("Progressive sampling", value=False)
if progressive:
    animated = st.sidebar.checkbox("Animated", value=False)
else:
    animated = False


@st.cache(show_spinner=False)
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
    dataset_str = "".join(ch for ch in dataset_str if ch in ascii_letters)
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


def get_bytes(arr):
    buffer = io.BytesIO()
    if isinstance(arr, (list, tuple)):
        ims = [Image.fromarray(a) for a in arr]
        duration = max(42, 1000 // len(ims))
        duration = [duration for _ in range(len(ims))]
        duration[-1] = 1000
        ims[0].save(
            buffer, format="WEBP", duration=duration,
            append_images=ims[1:], save_all=True, loop=0)
    else:
        Image.fromarray(arr).save(buffer, format="PNG")
    return buffer


if "img" in st.session_state:
    if not animated:
        img_idx = st.session_state.get("image_index", 0) % len(st.session_state["img"])
        im_bytes = get_bytes(st.session_state["img"][img_idx])
        ext = "png"
        if progressive:
            st.slider("Timestep", 0, timesteps, key="image_slider", on_change=update_idx)
    else:
        im_bytes = get_bytes(st.session_state["img"])
        ext = "webp"
    st.image(im_bytes)
    st.download_button(
        "Download image(s)", data=im_bytes,
        file_name=f"{uuid.uuid4()}.{ext}")
