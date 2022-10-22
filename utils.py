import random
import torch
import numpy as np
from PIL import Image
from prompt_parser import get_learned_conditioning, get_multicond_learned_conditioning
from sampler import KDiffusionSampler
from load_model import load_novelAI
from face_restoration import FaceRestorerCodeFormer


def sample_to_image(samples, sd_model):
    x_sample = sd_model.decode_first_stage(samples[0:1].type(torch.float16))[0]
    x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
    x_sample = 255. * np.moveaxis(x_sample.cpu().numpy(), 0, 2)
    x_sample = x_sample.astype(np.uint8)
    return Image.fromarray(x_sample)


def create_random_tensors(
        shape,
        seeds,
        subseeds=None,
):
    xs = []

    for i, seed in enumerate(seeds):
        if subseeds is not None:
            torch.manual_seed(seed)

        torch.manual_seed(seed)
        noise = torch.randn(shape)

        xs.append(noise)

    x = torch.stack(xs).to("cuda")
    return x


LANCZOS = (Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)


def resize_image(resize_mode, im, width, height):
    if resize_mode == 0:
        res = im.resize((width, height), resample=LANCZOS)

    elif resize_mode == 1:
        ratio = width / height
        src_ratio = im.width / im.height

        src_w = width if ratio > src_ratio else im.width * height // im.height
        src_h = height if ratio <= src_ratio else im.height * width // im.width

        resized = im.resize((src_w, src_h), resample=LANCZOS)
        res = Image.new("RGB", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

    else:
        ratio = width / height
        src_ratio = im.width / im.height

        src_w = width if ratio < src_ratio else im.width * height // im.height
        src_h = height if ratio >= src_ratio else im.height * width // im.width

        resized = im.resize((src_w, src_h), resample=LANCZOS)
        res = Image.new("RGB", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

        if ratio < src_ratio:
            fill_height = height // 2 - src_h // 2
            res.paste(resized.resize((width, fill_height), box=(0, 0, width, 0)), box=(0, 0))
            res.paste(resized.resize((width, fill_height), box=(0, resized.height, width, resized.height)),
                      box=(0, fill_height + src_h))
        elif ratio > src_ratio:
            fill_width = width // 2 - src_w // 2
            res.paste(resized.resize((fill_width, height), box=(0, 0, 0, height)), box=(0, 0))
            res.paste(resized.resize((fill_width, height), box=(resized.width, 0, resized.width, height)),
                      box=(fill_width + src_w, 0))

    return res


def txt2img(
        sd_model,
        sampler,
        prompt,
        negative_prompt,
        *,
        height=512,
        width=512,
        steps=28,
        cfg=7,
        eta=0.21,
        seed=None,
        subseed=None,
        face_restoration: FaceRestorerCodeFormer = None,
        face_restoration_weight=1
):
    with torch.no_grad(), sd_model.ema_scope():
        with torch.autocast("cuda"):
            uc = get_learned_conditioning(
                sd_model,
                [negative_prompt],
                steps
            )

            c = get_multicond_learned_conditioning(
                sd_model,
                [prompt],
                steps
            )

            x = create_random_tensors(
                [4, height // 8, width // 8],
                seeds=[seed or int(random.randrange(4294967294))],
                subseeds=[subseed or int(random.randrange(4294967294))],
            )

            samples = sampler.txt2img(x, c, uc, steps, cfg, eta)

    if face_restoration:
        samples = sd_model.decode_first_stage(samples.type(torch.float16))
        samples = torch.clamp((samples + 1.0) / 2.0, min=0.0, max=1.0)[0]
        samples = 255. * np.moveaxis(samples.cpu().numpy(), 0, 2)
        samples = samples.astype(np.uint8)
        samples = face_restoration.restore(samples, w=face_restoration_weight)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        return Image.fromarray(samples)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    return sample_to_image(samples, sd_model)


def img2img(
        sd_model,
        sampler,
        img,
        prompt,
        negative_prompt,
        *,
        resize_mode=1,
        height=512,
        width=512,
        steps=28,
        cfg=7,
        denoising_strength=0.75,
        eta=0.21,
        seed=None,
        subseed=None,
        face_restoration: FaceRestorerCodeFormer = None,
        face_restoration_weight=1
):
    img = resize_image(resize_mode, img.convert("RGB"), width, height)
    img = np.array(img).astype(np.float16) / 255.0
    img = np.moveaxis(img, 2, 0)
    img = np.expand_dims(img, axis=0).repeat(1, axis=0)
    img = torch.from_numpy(img)
    img = 2. * img - 1.
    img = img.to("cuda")
    init_latent = sd_model.get_first_stage_encoding(sd_model.encode_first_stage(img))

    with torch.no_grad(), sd_model.ema_scope():
        with torch.autocast("cuda"):
            uc = get_learned_conditioning(
                sd_model,
                [negative_prompt],
                steps
            )

            c = get_multicond_learned_conditioning(
                sd_model,
                [prompt],
                steps
            )

            x = create_random_tensors(
                [4, height // 8, width // 8],
                seeds=[seed or int(random.randrange(4294967294))],
                subseeds=[subseed or int(random.randrange(4294967294))],
            )

            samples = sampler.img2img(init_latent, x, c, uc, steps, cfg, denoising_strength, eta)

    if face_restoration:
        samples = sd_model.decode_first_stage(samples.type(torch.float16))
        samples = torch.clamp((samples + 1.0) / 2.0, min=0.0, max=1.0)[0]
        samples = 255. * np.moveaxis(samples.cpu().numpy(), 0, 2)
        samples = samples.astype(np.uint8)
        samples = face_restoration.restore(samples, w=face_restoration_weight)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        return Image.fromarray(samples)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    return sample_to_image(samples, sd_model)


__all__ = [
    "load_novelAI",
    "KDiffusionSampler",
    "txt2img",
    "img2img",
    "FaceRestorerCodeFormer"
]
