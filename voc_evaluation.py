# voc_evaluator.py (改造后)

import torch
import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
from pathlib import Path
import sys
from contextlib import redirect_stdout

from util.voc_eval_utils import get_map
from util import box_ops


@torch.no_grad()
def evaluate_voc(model, data_loader, device, output_dir, epoch, args):
    model.eval()

    # --- 1. 准备文件夹 ---
    # 每个 epoch 的结果都放在 output_dir 下的一个临时文件夹里
    map_out_path = Path(output_dir) / f'map_out_epoch_{epoch}'
    map_out_path.mkdir(exist_ok=True)
    (map_out_path / 'ground-truth').mkdir(exist_ok=True)
    (map_out_path / 'detection-results').mkdir(exist_ok=True)

    print(f"\n--- Starting VOC evaluation for Epoch {epoch} ---")

    # 获取类别名称和验证集文件名列表
    class_names = data_loader.dataset.classes
    image_ids = data_loader.dataset.ids

    # --- 2. 生成预测结果文件 ---
    print("Generating prediction results...")
    for samples, targets in tqdm(data_loader):
        samples = samples.to(device)
        outputs = model(samples)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        prob = out_logits.softmax(-1)
        scores, labels = prob[..., :-1].max(-1)

        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        img_h, img_w = orig_target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        image_idx = targets[0]['image_id'].item()
        image_name = image_ids[image_idx]

        with open(map_out_path / 'detection-results' / f"{image_name}.txt", "w") as f:
            for s, l, b in zip(scores[0], labels[0], boxes[0]):
                # 只保留置信度大于一个很小阈值的预测，避免文件过大
                if s.item() < 0.01:
                    continue
                class_name = class_names[l]
                confidence = s.item()
                xmin, ymin, xmax, ymax = b.tolist()
                f.write(f"{class_name} {confidence} {int(xmin)} {int(ymin)} {int(xmax)} {int(ymax)}\n")

    print("Prediction results generated.")

    # --- 3. 生成真实标签文件 ---
    print("Generating ground truth files...")
    voc_root = Path(args.voc_path)
    for image_name in tqdm(image_ids):
        # 验证集使用 test 文件夹下的 Annotations
        ann_file_path = voc_root / "test" / "Annotations" / f"{image_name}.xml"
        with open(map_out_path / "ground-truth" / f"{image_name}.txt", "w") as new_f:
            root = ET.parse(ann_file_path).getroot()
            for obj in root.findall('object'):
                obj_name = obj.find('name').text
                if obj_name not in class_names:
                    continue
                bndbox = obj.find('bndbox')
                left = bndbox.find('xmin').text
                top = bndbox.find('ymin').text
                right = bndbox.find('xmax').text
                bottom = bndbox.find('ymax').text
                new_f.write(f"{obj_name} {left} {top} {right} {bottom}\n")
    print("Ground truth files generated.")

    # --- 4. 计算 mAP 并保存到日志文件 ---
    print("Calculating mAP...")
    log_file_path = Path(output_dir) / f"log{epoch}.txt"
    with open(log_file_path, 'w') as f:
        with redirect_stdout(f):  # 将所有 print 输出重定向到日志文件
            print(f"--- Evaluation Results for Epoch {epoch} ---")
            get_map(min_overlap=0.5, draw_plot=False, path=str(map_out_path))

    print(f"mAP calculation done. Results saved to {log_file_path}")
    # 读取并打印 mAP 结果到控制台
    with open(log_file_path, 'r') as f:
        print("\n" + f.read())