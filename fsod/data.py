"""VOC 检测 Dataset（torchvision Faster R-CNN 输入格式）。"""
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class VOCDetDataset(Dataset):
    def __init__(self, samples, train=False, hflip=0.5):
        self.samples = samples
        self.train = train
        self.hflip = hflip

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s["file"]).convert("RGB")
        boxes = torch.tensor(s["boxes"], dtype=torch.float32).reshape(-1, 4)
        labels = torch.tensor(s["labels"], dtype=torch.int64)

        if self.train and self.hflip > 0 and torch.rand(1).item() < self.hflip:
            w = img.width
            img = TF.hflip(img)
            if boxes.numel():
                x1 = boxes[:, 0].clone()
                x2 = boxes[:, 2].clone()
                boxes[:, 0] = w - 1 - x2
                boxes[:, 2] = w - 1 - x1

        t = TF.pil_to_tensor(img).float().div(255.0)
        t = TF.normalize(t, IMAGENET_MEAN, IMAGENET_STD)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx], dtype=torch.int64),
        }
        return t, target


def detection_collate(batch):
    imgs = [b[0] for b in batch]
    tgts = [b[1] for b in batch]
    return imgs, tgts
