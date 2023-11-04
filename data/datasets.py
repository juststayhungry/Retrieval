import torch
from torch.utils.data import Dataset
import os
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image


def _transform(n_px=224):
    return Compose([
        Resize(n_px, interpolation=Image.BICUBIC),
        CenterCrop(n_px),
        # lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.4225, 0.4012, 0.3659), (0.2681, 0.2635, 0.2763)), # COCO mean, std
    ])

class CLIP_COCO_dataset(Dataset):
    """CLIP_COCO_dataset. To train CLIP on COCO-Captions."""

    def __init__(self,phase,config, text_tokenizer, context_length=77, input_resolution=224):
        
        super(CLIP_COCO_dataset, self).__init__()
        if phase == 'train_on_seen':
            self.data = self.read_txt(config.seen_imgid_caption_dir)
        if phase == 'train_on_all':
            self.data = self.read_txt(config.seen_imgid_caption_dir)
            self.data.append(self.read_txt(config.unseen_imgid_caption_dir))
        elif phase == 'eval':
            self.data = self.read_txt(config.unseen_imgid_caption_dir)
        else:
            raise ValueError('Invalid transform')
        self.image_path = config.img_dir
        self.transform = _transform(input_resolution)
        self._tokenizer = text_tokenizer
        self.context_length = context_length

    def __len__(self):
        return len(self.data)

    def tokenize(self, text):
        sot_token = self._tokenizer.encoder["<|startoftext|>"]
        eot_token = self._tokenizer.encoder["<|endoftext|>"]
        tokens = [sot_token] + self._tokenizer.encode(text) + [eot_token]
        result = torch.zeros(self.context_length, dtype=torch.long)
        result[:len(tokens)] = torch.tensor(tokens)
        return result

    def read_txt(self, file_path):
        data = []
        with open(file_path, "r") as f:
            next(f)
            for line in f:
                # 去除行末尾的换行符
                line = line.strip()
                if line and line.count('\t') >= 1:#排除空行
                    image_id, caption = line.strip().split("\t")
                    image_id = int(image_id)
                    data.append({"id": image_id, "caption": caption})
        return data
    
    def load_image(self,image_id):
        # 构造图像文件名
        image_file = "000000" + str(image_id).zfill(6) + ".jpg"
        image_path = os.path.join(self.image_path, image_file)
        # 打开图像文件并返回
        image = Image.open(image_path).convert("RGB")
        return image

    def __getitem__(self, idx):
        item = self.data[idx]
        image_id, caption = item["id"], item["caption"]
        # 加载图像并进行预处理，这里以示例为准（可以根据实际情况进行修改）
        img = self.load_image(image_id)
        img_input = self.transform(img)
        text_input = self.tokenize(caption)
        #输出：预处理后的图像，tokenize后的caption
        return img_input, text_input
    
#text_tokenize的方式与CLIP用的不同？，为什么要自定义???

