# datasets/voc_fog.py

from pathlib import Path
import torch
import torch.utils.data
import torchvision
from PIL import Image
import xml.etree.ElementTree as ET
import datasets.transforms as T

class VocFogDataset(torch.utils.data.Dataset):
    def __init__(self, img_folder, ann_folder, transforms, image_set_file):
        self.img_folder = Path(img_folder)
        self.ann_folder = Path(ann_folder)
        self._transforms = transforms

        # !!!重要!!!: 请将这里的类别修改为您数据集的真实类别
        # 例如: self.classes = ('car', 'person', 'bicycle')
        self.classes = ('person', 'bicycle', 'car','motorbike','bus')

        # 从指定的 image_set 文件中读取文件名
        with open(image_set_file, 'r') as f:
            self.ids = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        # 假设图片都是 .jpg 格式
        img_path = self.img_folder / f"{img_id}.jpg"
        ann_path = self.ann_folder / f"{img_id}.xml"

        img = Image.open(img_path).convert("RGB")
        target = self.parse_voc_xml(ann_path)

        # image_id 在 voc_evaluator.py 中需要用到
        # 我们使用索引作为id
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

def make_voc_transforms(image_set):
    # 与COCO使用相同的变换
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])
    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])
    raise ValueError(f'unknown {image_set}')

def build(image_set, args):
    root = Path(args.voc_path)
    assert root.exists(), f'provided VOC path {root} does not exist'

    if image_set == 'train':
        # --- 训练阶段的路径 ---
        base_dir = root / 'train'
        img_folder = base_dir / "SnowyImages"  # 使用退化的训练图片  #todo
        ann_folder = base_dir / "Annotations"
        # 假设 ImageSets 文件在 train 文件夹内
        image_set_file = base_dir / 'ImageSets' / 'Main' / 'train.txt'
    elif image_set == 'val':
        # --- 验证/测试阶段的路径 ---
        base_dir = root / 'test'
        img_folder = base_dir / "SnowyImages" # 使用有雨的测试图片  #todo
        ann_folder = base_dir / "Annotations"
        # 假设 ImageSets 文件在 test 文件夹内, 并且文件名是 val.txt
        image_set_file = base_dir / 'ImageSets' / 'Main' / 'val.txt'
    else:
        raise ValueError(f'unknown image_set {image_set}')

    dataset = VocFogDataset(
        img_folder=img_folder,
        ann_folder=ann_folder,
        transforms=make_voc_transforms(image_set),
        image_set_file=image_set_file
    )
    return dataset