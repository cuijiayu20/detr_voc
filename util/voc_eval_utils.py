# util/voc_eval_utils.py (最终完整版 2)

import glob
import json
import os
import shutil
import numpy as np


def file_lines_to_list(path):
    with open(path) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    return content


def voc_ap(rec, prec):
    rec.insert(0, 0.0)
    rec.append(1.0)
    mrec = rec[:]
    prec.insert(0, 0.0)
    prec.append(0.0)
    mpre = prec[:]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap, mrec, mpre


def get_map(MINOVERLAP, draw_plot, score_threhold=0.5, path='./map_out'):
    GT_PATH = os.path.join(path, 'ground-truth')
    DR_PATH = os.path.join(path, 'detection-results')
    TEMP_FILES_PATH = os.path.join(path, '.temp_files')
    RESULTS_FILES_PATH = os.path.join(path, "results")

    if os.path.exists(RESULTS_FILES_PATH):
        shutil.rmtree(RESULTS_FILES_PATH)
    os.makedirs(RESULTS_FILES_PATH)

    if os.path.exists(TEMP_FILES_PATH):
        shutil.rmtree(TEMP_FILES_PATH)
    os.makedirs(TEMP_FILES_PATH)

    ground_truth_files_list = glob.glob(GT_PATH + '/*.txt')
    if len(ground_truth_files_list) == 0:
        print("Error: No ground-truth files found for evaluation!")
        return

    gt_classes = []
    for txt_file in ground_truth_files_list:
        lines_list = file_lines_to_list(txt_file)
        for line in lines_list:
            if line:
                class_name = line.split()[0]
                if class_name not in gt_classes:
                    gt_classes.append(class_name)
    gt_classes = sorted(gt_classes)
    n_classes = len(gt_classes)

    for class_name in gt_classes:
        # Ground Truth
        bounding_boxes = []
        for txt_file in ground_truth_files_list:
            file_id = os.path.basename(txt_file).split(".txt", 1)[0]
            lines_list = file_lines_to_list(txt_file)
            for line in lines_list:
                if not line: continue
                parts = line.split()
                tmp_class_name = parts[0]
                if tmp_class_name == class_name:
                    bbox = " ".join(parts[1:5])
                    is_difficult = len(parts) == 6 and parts[5].lower() == "difficult"
                    bounding_boxes.append({"bbox": bbox, "difficult": is_difficult, "file_id": file_id})
        with open(os.path.join(TEMP_FILES_PATH, class_name + "_gt.json"), 'w') as f:
            json.dump(bounding_boxes, f)

        # Detection Results
        bounding_boxes = []
        dr_files_list = glob.glob(DR_PATH + '/*.txt')
        for txt_file in dr_files_list:
            file_id = os.path.basename(txt_file).split(".txt", 1)[0]
            lines_list = file_lines_to_list(txt_file)
            for line in lines_list:
                if not line: continue
                try:
                    tmp_class_name, confidence, left, top, right, bottom = line.split()
                    if tmp_class_name == class_name:
                        bbox = left + " " + top + " " + right + " " + bottom
                        bounding_boxes.append({"confidence": confidence, "file_id": file_id, "bbox": bbox})
                except ValueError:
                    continue
        with open(os.path.join(TEMP_FILES_PATH, class_name + "_dr.json"), 'w') as f:
            json.dump(bounding_boxes, f)

    sum_AP = 0.0
    ap_dictionary = {}

    with open(os.path.join(RESULTS_FILES_PATH, "results.txt"), 'w') as results_file:
        results_file.write("# AP and precision/recall per class\n")

        for class_index, class_name in enumerate(gt_classes):
            results_file.write("\n\n# Class: " + class_name + "\n")

            gt_file = os.path.join(TEMP_FILES_PATH, class_name + "_gt.json")
            dr_file = os.path.join(TEMP_FILES_PATH, class_name + "_dr.json")

            with open(gt_file) as f:
                gt_data = json.load(f)
            with open(dr_file) as f:
                dr_data = json.load(f)

            npos = 0
            for obj in gt_data:
                if not obj['difficult']:
                    npos += 1

            dr_data.sort(key=lambda x: float(x['confidence']), reverse=True)

            nd = len(dr_data)
            tp = np.zeros(nd)
            fp = np.zeros(nd)

            # --- 关键修改：更健壮地处理 GT 匹配 ---
            gt_file_map = {}
            for obj in gt_data:
                file_id = obj["file_id"]
                if file_id not in gt_file_map:
                    gt_file_map[file_id] = []
                gt_file_map[file_id].append(obj)

            for idx, detection in enumerate(dr_data):
                file_id = detection["file_id"]
                gt_boxes_for_image = gt_file_map.get(file_id, [])

                ovmax = -1
                gt_match = None

                bb = [float(x) for x in detection["bbox"].split()]

                for obj in gt_boxes_for_image:
                    if not obj["difficult"]:
                        bbgt = [float(x) for x in obj["bbox"].split()]
                        bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]), min(bb[2], bbgt[2]), min(bb[3], bbgt[3])]
                        iw = bi[2] - bi[0] + 1
                        ih = bi[3] - bi[1] + 1
                        if iw > 0 and ih > 0:
                            ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + \
                                 (bbgt[2] - bbgt[0] + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                            ov = iw * ih / ua
                            if ov > ovmax:
                                ovmax = ov
                                gt_match = obj

                if ovmax >= MINOVERLAP:
                    if not gt_match.get("used", False):
                        tp[idx] = 1
                        gt_match["used"] = True
                    else:
                        fp[idx] = 1
                else:
                    fp[idx] = 1

            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / float(npos) if npos > 0 else np.zeros_like(tp)
            prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

            ap, _, _ = voc_ap(rec.tolist(), prec.tolist())

            sum_AP += ap
            text = "{0:.2f}%".format(ap * 100) + " = " + class_name + " AP "
            results_file.write(text + "\n")
            ap_dictionary[class_name] = ap

        if n_classes > 0:
            mAP = sum_AP / n_classes
            print("mAP = {0:.2f}%".format(mAP * 100))
        else:
            print("mAP = 0.0% (No classes to evaluate)")
            mAP = 0.0

        results_file.write("\n\n" + "mAP = {0:.2f}%".format(mAP * 100))

    shutil.rmtree(TEMP_FILES_PATH)