import torch
from torch.utils.data import Dataset
import os
from torchvision.transforms import (CenterCrop, Compose, InterpolationMode,
                                    Normalize, RandomHorizontalFlip,
                                    RandomPerspective, RandomRotation, Resize,
                                    ToTensor)
from torchvision.transforms.transforms import RandomResizedCrop
from torch.utils.data.dataloader import DataLoader
from PIL import Image
n_px =224

def transform_image(split="train", imagenet=False,coco=False):
    if imagenet:
        # from czsl repo.
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        transform = Compose(
            [
                RandomResizedCrop(n_px),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize(
                    mean,
                    std,
                ),
            ]
        )
        return transform
    if coco:
        mean, std = (0.4225, 0.4012, 0.3659), (0.2681, 0.2635, 0.2763)
        transform = Compose(
            [
                RandomResizedCrop(n_px),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize(
                    mean,
                    std,
                ),
            ]
        )
        return transform

    if split == "test" or split == "val":
        transform = Compose(
            [
                Resize(n_px, interpolation=Image.BICUBIC),
                CenterCrop(n_px),
                lambda image: image.convert("RGB"),
                ToTensor(),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
    else:
        transform = Compose(
            [
                # RandomResizedCrop(n_px, interpolation=BICUBIC),
                Resize(n_px, interpolation=Image.BICUBIC),
                CenterCrop(n_px),
                RandomHorizontalFlip(),
                RandomPerspective(),
                RandomRotation(degrees=5),
                lambda image: image.convert("RGB"),
                ToTensor(),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    return transform

class CLIP_COCO_dataset(Dataset):
    """CLIP_COCO_dataset. To train CLIP on COCO-Captions."""

    def __init__(self,phase,config, text_tokenizer, context_length=77, input_resolution=224):
        
        super(CLIP_COCO_dataset, self).__init__()
        if phase == 'train_on_seen':
            self.data = self.read_txt(config.seen_imgid_caption_dir)
        elif phase == 'train_on_all':
            self.data = self.read_txt(config.seen_imgid_caption_dir)
            self.data.extend(self.read_txt(config.unseen_imgid_caption_dir))
        elif phase == 'eval':#eval on unseen image
            self.data = self.read_txt(config.unseen_imgid_caption_dir)
        else:
            raise ValueError('Invalid phase')
        self.image_path = config.img_dir
        self.transform = transform_image(coco=True)
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
        # return img_input,caption
#text_tokenize的方式与CLIP用的不同？，为什么要自定义???

################
class ImageLoader:
    def __init__(self, root):
        self.img_dir = root

    def __call__(self, img):
        file = '%s/%s' % (self.img_dir, img)
        img = Image.open(file).convert('RGB')
        return img


class CompositionDataset(Dataset):
    def __init__(
            self,
            root,
            phase,
            text_tokenizer,
            split='compositional-split-natural',
            imagenet=False
    ):
        self.root = root
        self.phase = phase
        self.split = split
        self._tokenizer = text_tokenizer
        self.context_length = 77


        self.transform = transform_image(phase, imagenet=imagenet)
        self.loader = ImageLoader(self.root + '/images/')

        self.attrs, self.objs, self.pairs, \
                self.train_pairs, self.val_pairs, \
                self.test_pairs = self.parse_split()

        self.seen_data,self.unseen_data  = self.get_split_info()

        print('# seen datas: %d | # unseen pairs datas: %d ' %
              (len(self.seen_data), len(self.unseen_data)))
            
        if self.phase == 'train_on_seen':
            self.data = self.seen_data
        elif self.phase == 'train_on_all':
            self.data = self.seen_data.extend(self.unseen_data)
        elif self.phase == 'eval':
            self.data = self.unseen_data_data

        self.train_pair_to_idx = dict(
            [(pair, idx) for idx, pair in enumerate(self.train_pairs)]
        )

        self.pair2idx = {pair: idx for idx, pair in enumerate(self.pairs)}

    def tokenize(self, text):
        sot_token = self._tokenizer.encoder["<|startoftext|>"]
        eot_token = self._tokenizer.encoder["<|endoftext|>"]
        tokens = [sot_token] + self._tokenizer.encode(text) + [eot_token]
        result = torch.zeros(self.context_length, dtype=torch.long)
        result[:len(tokens)] = torch.tensor(tokens)
        return result
    
    def get_split_info(self):
        data = torch.load(self.root + '/metadata_{}.t7'.format(self.split))
        seen_data,unseen_data = [], []
        for instance in data:
            image, attr, obj, settype = instance['image'], instance[
                'attr'], instance['obj'], instance['set']

            if attr == 'NA' or (attr,
                                obj) not in self.pairs or settype == 'NA':
                # ignore instances with unlabeled attributes
                # ignore instances that are not in current split
                continue

            data_i = [image, attr, obj]
            if settype == 'train':
                seen_data.append(data_i)
            else:
                pairs = tuple([attr,obj])
                if pairs not in self.train_pairs:
                    unseen_data.append(data_i)

        return seen_data, unseen_data

    def parse_split(self):
        def parse_pairs(pair_list):
            with open(pair_list, 'r') as f:
                pairs = f.read().strip().split('\n')
                # pairs = [t.split() if not '_' in t else t.split('_') for t in pairs]
                pairs = [t.split() for t in pairs]
                pairs = list(map(tuple, pairs))
                
            attrs, objs = zip(*pairs)
            return attrs, objs, pairs

        tr_attrs, tr_objs, tr_pairs = parse_pairs(
            '%s/%s/train_pairs.txt' % (self.root, self.split))
        vl_attrs, vl_objs, vl_pairs = parse_pairs(
            '%s/%s/val_pairs.txt' % (self.root, self.split))
        ts_attrs, ts_objs, ts_pairs = parse_pairs(
            '%s/%s/test_pairs.txt' % (self.root, self.split))

        all_attrs, all_objs = sorted(
            list(set(tr_attrs + vl_attrs + ts_attrs))), sorted(
                list(set(tr_objs + vl_objs + ts_objs)))
        all_pairs = sorted(list(set(tr_pairs + vl_pairs + ts_pairs)))

        return all_attrs, all_objs, all_pairs, tr_pairs, vl_pairs, ts_pairs

    def __getitem__(self, index):
        image, attr, obj = self.data[index]
        img = self.loader(image)
        img = self.transform(img)
        caption = 'a photo of a {} {}'.format(attr,obj)
        text = self.tokenize(caption)

        if self.phase == 'eval':
            data = [
                img, text, self.pair2idx[(attr, obj)]
            ]
        else:
            data = [
                img, text, self.train_pair_to_idx[(attr, obj)]
            ]
        return data

    def __len__(self):
        return len(self.data)

         