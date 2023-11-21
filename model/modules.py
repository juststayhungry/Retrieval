import torch
from model.adapter import CustomCLIP
from model.clip1 import CLIP
from clip import clip
# from models.csp import get_csp, get_mix_csp

# DIR_PATH = os.path.dirname(os.path.realpath(__file__))

def load_clip_to_cpu(cfg):
  backbone_name = cfg.MODEL.BACKBONE.NAME
  openai_pretrained_model, _ = clip.load(backbone_name, device=cfg.device)

  return openai_pretrained_model

def get_model(config):
    if config.experiment_name == "adapter":
        clip_model = load_clip_to_cpu(config)
        clip_model.float()
        model = CustomCLIP(config,clip_model)
        print('Turning off gradients in both the image and the text encoder')
    
        for name, param in model.named_parameters():
            if 'adapter' not in name:
                param.requires_grad_(False)
        return model

    elif config.experiment_name == "clip":
        return  CLIP(config)
    else:
        raise NotImplementedError(
            "Error: Unrecognized Experiment Name {:s}.".format(
                config.experiment_name
            )
        )