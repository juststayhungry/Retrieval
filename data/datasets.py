import torch
from torch.utils.data import Dataset
import os
from PIL import Image

class CustomDataset(Dataset):
    def __init__(
            self, 
            file_path,
            image_path,
            transform 
            ):
        super(CustomDataset,self).__init__()

        self.data = self.read_txt(file_path)
        self.transform = transform
        self.image_path = image_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_id, caption = item["id"], item["caption"]
        # 加载图像并进行预处理，这里以示例为准（可以根据实际情况进行修改）
        image = self.load_image(image_id)
        return image, caption

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
        return self.transform(image)#输出预处理后的image
