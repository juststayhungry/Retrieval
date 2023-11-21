import torch.nn as nn
from torch.nn import functional as F

class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x

class CustomCLIP(nn.Module):

    def __init__(self, cfg, clip_model):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.text_encoder = clip_model.encode_text
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.adapter = Adapter(512, 4).to(clip_model.dtype)

            
    def forward(self, image,text):
        image_features = self.image_encoder(image.type(self.dtype))
        x = self.adapter(image_features)

        ratio = 0.2
        image_features = ratio * x + (1 - ratio) * image_features

        text_features = self.text_encoder(text)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return image_features, text_features 