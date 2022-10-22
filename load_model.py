import torch
import importlib
from omegaconf import OmegaConf
from hijack import hijack


def get_state_dict_from_checkpoint(pl_sd):
    if "state_dict" in pl_sd:
        return pl_sd["state_dict"]

    return pl_sd


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def load_model_weights(model, checkpoint_file, vae_file):
    print(f"Loading weights from {checkpoint_file}")

    pl_sd = torch.load(checkpoint_file, map_location="cpu")
    sd = get_state_dict_from_checkpoint(pl_sd)

    model.load_state_dict(sd, strict=False)

    model.half()

    print(f"Loading VAE weights from: {vae_file}")

    vae_ckpt = torch.load(vae_file, map_location="cpu")
    vae_dict = {k: v for k, v in vae_ckpt["state_dict"].items() if k[0:4] != "loss"}

    model.first_stage_model.load_state_dict(vae_dict)


def load_novelAI(checkpoint_file, vae_file, hypernetwork_file, config, CLIP_stop=2):
    sd_config = OmegaConf.load(config)
    sd_model = instantiate_from_config(sd_config.model)
    load_model_weights(sd_model, checkpoint_file, vae_file)
    sd_model.to("cuda")
    hijack(sd_model, hypernetwork_file, CLIP_stop)
    return sd_model
