# datasets/rtts_test.py

from pathlib import Path
import torch
import torch.utils.data
import torchvision
from PIL import Image
import xml.etree.ElementTree as ET
import datasets.transforms as T

class RTTSTestDataset(torch.utils.data.Dataset):
    def __init__(self, img_folder, ann_folder, transforms, image_set_file):
        self.img_folder = Path(img_folder)
        self.ann_folder = Path(ann_folder)
        self._transforms = transforms

        # !!!重要!!!: 请根据您的数据集修改类别
        self.classes = ('person', 'bicycle', 'car','motorbike','bus')

        with open(image_set_file, 'r') as f:
            self.ids = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_path = self.img_folder / f"{img_id}.png" # 假设图片是.jpg格式
        ann_path = self.ann_folder / f"{img_id}.xml"

        img = Image.open(img_path).convert("RGB")
        target = self.parse_voc_xml(ann_path)
        target['image_id'] = torch.tensor(idx)

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target

    def parse_voc_xml(self, ann_path):
        tree = ET.parse(ann_path)
        root = tree.getroot()
        size = root.find('size')
        w, h = int(size.find('width').text), int(size.find('height').text)

        boxes = []
        labels = []
        for obj in root.findall('object'):
            cls_name = obj.find('name').text
            if cls_name not in self.classes:
                continue
            labels.append(self.classes.index(cls_name))

            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text) - 1
            ymin = float(bndbox.find('ymin').text) - 1
            xmax = float(bndbox.find('xmax').text) - 1
            ymax = float(bndbox.find('ymax').text) - 1
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        target["area"] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        target["iscrowd"] = torch.zeros((len(boxes),), dtype=torch.int64)

        return target

def make_rtts_transforms():
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return T.Compose([
        T.RandomResize([800], max_size=1333),
        normalize,
    ])

def build(image_set, args):
    root = Path(args.voc_path)
    assert root.exists(), f'provided RTTS path {root} does not exist'

    img_folder = root / "Images"
    ann_folder = root / "Annotations"
    # 假设您会为RTTS数据集创建一个ImageSets/Main/val.txt文件
    image_set_file = root / 'ImageSets' / 'Main' / 'val.txt'


    dataset = RTTSTestDataset(
        img_folder=img_folder,
        ann_folder=ann_folder,
        transforms=make_rtts_transforms(),
        image_set_file=image_set_file
    )
    return dataset