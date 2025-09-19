import argparse
import time
import torch
from PIL import Image
from datasets import transforms as T # <

from models import build_model
from util.misc import collate_fn
import numpy as np


# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def get_args_parser():
    parser = argparse.ArgumentParser('DETR FPS Test script', add_help=False)

    # --- 必要参数 ---
    parser.add_argument('--resume', required=True, help='Path to the model checkpoint')
    parser.add_argument('--image_path', required=True, type=str, help='Path to the single image for testing')

    # --- 模型结构参数 (需要与训练时一致) ---
    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--position_embedding', default='sine', type=str)
    parser.add_argument('--enc_layers', default=6, type=int)
    parser.add_argument('--dec_layers', default=6, type=int)
    parser.add_argument('--dim_feedforward', default=2048, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--num_queries', default=100, type=int)
    parser.add_argument('--num_classes', default=5, type=int)
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--masks', action='store_true')
    parser.add_argument('--aux_loss', action='store_true')
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--lr_backbone', default=0, type=float)  # for build_model

    # --- 运行参数 ---
    parser.add_argument('--device', default='cuda', help='device to use for testing (cuda or cpu)')
    parser.add_argument('--warmup_iterations', default=10, type=int, help='Number of warmup iterations before timing')
    parser.add_argument('--test_iterations', default=100, type=int,
                        help='Number of iterations to average for FPS calculation')

    return parser


def main(args):
    device = torch.device(args.device)

    # --- 构建模型 ---
    # build_model 需要一些额外的参数，即使在推理时也是如此
    # 我们为它们提供默认值
    args.set_cost_class = 1
    args.set_cost_bbox = 5
    args.set_cost_giou = 2
    args.bbox_loss_coef = 5
    args.giou_loss_coef = 2
    args.eos_coef = 0.1

    model, _, _ = build_model(args)
    model.to(device)

    # --- 加载权重 ---
    print(f"Loading weights from {args.resume}")
    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # --- 准备图像 ---
    # 定义与验证集相同的图像变换
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print(f"Loading image from {args.image_path}")
    img = Image.open(args.image_path).convert('RGB')
    # 预处理图像
    img_tensor, _ = transform(img, None)

    # 将图像包装成模型期望的输入格式 (batch size = 1)
    inputs = [img_tensor.to(device)]

    # --- FPS 测试 ---
    print(f"Starting FPS test...")
    print(f"Warmup iterations: {args.warmup_iterations}")
    print(f"Test iterations:   {args.test_iterations}")

    # 预热
    with torch.no_grad():
        for _ in range(args.warmup_iterations):
            _ = model(inputs)

    # 计时
    t_start = time.perf_counter()
    with torch.no_grad():
        for _ in range(args.test_iterations):
            _ = model(inputs)

    # 如果使用CUDA，需要同步以获得准确的时间
    if args.device == 'cuda':
        torch.cuda.synchronize()

    t_end = time.perf_counter()

    # --- 计算并打印结果 ---
    total_time = t_end - t_start
    avg_time_per_image = total_time / args.test_iterations
    fps = 1 / avg_time_per_image

    print("\n--- FPS Test Results ---")
    print(f"Total time for {args.test_iterations} iterations: {total_time:.4f} seconds")
    print(f"Average time per image: {avg_time_per_image * 1000:.4f} ms")
    print(f"FPS: {fps:.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR FPS testing script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)