# Streamlit Web APP for Diffusion-based Deep Generative Models

Check out associated model training & evaluation scripts at [[this repo]](https://github.com/tqch/ddpm-torch).

## Dependencies

see `requirements.txt` for details

- PyTorch
- NumPy
- Streamlit
- OpenCV-Python
- PIL

## Usage

```shell
streamlit run app.py
```

## Retrieve models

The models are stored in Git LFS. To retrieve the actual models, please run

```shell
git lfs pull
```

To first-time users of Git LFS: please refer to [[official website]](https://git-lfs.github.com/) for installation instructions

## Demo

![example](./demo/demo.gif)
