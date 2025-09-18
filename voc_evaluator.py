# voc_evaluator.py (最终修正版 1)

import torch
import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
from pathlib import Path
from contextlib import redirect_stdout
from util.voc_eval_utils import get_map
from util import box_ops


@torch.no_grad()
def evaluate_voc(model, data_loader, device, output_dir, epoch, args):
    print(f"[DBG] evaluate_voc START epoch={epoch}") #todo
    model.eval()

    map_out_path = Path(output_dir) / f'map_out_epoch_{epoch}'
    gt_path = map_out_path / 'ground-truth'
    dr_path = map_out_path / 'detection-results'
    gt_path.mkdir(parents=True, exist_ok=True)
    dr_path.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Starting VOC evaluation for Epoch {epoch} ---")

    class_names = data_loader.dataset.classes
    image_ids = data_loader.dataset.ids

    print("Generating prediction results...")
    for samples, targets in tqdm(data_loader):
        samples = samples.to(device)
        outputs = model(samples)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0).to(device)

        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        prob = out_logits.softmax(-1)
        scores, labels = prob[..., :-1].max(-1)

        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        img_h, img_w = orig_target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        image_idx = targets[0]['image_id'].item()
        image_name = image_ids[image_idx]

        with open(dr_path / f"{image_name}.txt", "w") as f:
            for s, l, b in zip(scores[0], labels[0], boxes[0]):
                if s.item() < 0.01:
                    continue
                class_name = class_names[l]
                confidence = s.item()
                xmin, ymin, xmax, ymax = b.tolist()
                f.write(f"{class_name} {confidence} {int(xmin)} {int(ymin)} {int(xmax)} {int(ymax)}\n")

    print("Prediction results generated.")

    print("Generating ground truth files...")
    voc_root = Path(args.voc_path)
    # --- 关键修改：只为包含目标类别的图片生成 GT 文件 ---
    for image_name in tqdm(image_ids):
        ann_file_path = voc_root / "test" / "Annotations" / f"{image_name}.xml"

        objects_to_write = []
        root = ET.parse(ann_file_path).getroot()
        for obj in root.findall('object'):
            obj_name = obj.find('name').text
            # 只有当物体类别在我们定义的5个类别中时，才处理它
            if obj_name in class_names:
                bndbox = obj.find('bndbox')
                left = bndbox.find('xmin').text
                top = bndbox.find('ymin').text
                right = bndbox.find('xmax').text
                bottom = bndbox.find('ymax').text
                objects_to_write.append(f"{obj_name} {left} {top} {right} {bottom}\n")

        # 只有当这张图片里至少有一个我们感兴趣的物体时，才创建 GT 文件
        if objects_to_write:
            with open(gt_path / f"{image_name}.txt", "w") as new_f:
                new_f.writelines(objects_to_write)

    print("Ground truth files generated.")

    print("Calculating mAP...")
    log_file_path = Path(output_dir) / f"log{epoch}.txt"
    with open(log_file_path, 'w') as f:
        with redirect_stdout(f):
            print(f"--- Evaluation Results for Epoch {epoch} ---")
            get_map(0.5, draw_plot=False, path=str(map_out_path))

    print(f"mAP calculation done. Results saved to {log_file_path}")
    with open(log_file_path, 'r') as f:
        print("\n" + f.read())
    print(f"[DBG] evaluate_voc END epoch={epoch}")