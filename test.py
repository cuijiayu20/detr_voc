# test.py
import argparse
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from models import build_model
from datasets import build_dataset
from voc_evaluator import evaluate_voc
import util.misc as utils

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    # --- 模型参数 ---
    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'))
    parser.add_argument('--enc_layers', default=6, type=int)
    parser.add_argument('--dec_layers', default=6, type=int)
    parser.add_argument('--dim_feedforward', default=2048, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--num_queries', default=100, type=int)
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--masks', action='store_true')
    parser.add_argument('--aux_loss', action='store_true') # 确保这个参数存在

    # --- 数据集参数 ---
    parser.add_argument('--dataset_file', required=True, type=str,
                        choices=['voc_fog', 'rtts_test', 'foggy_driving_voc'],
                        help='Dataset to use for testing.')
    parser.add_argument('--voc_path', required=True, type=str,
                        help='Path to the root of the dataset.')
    parser.add_argument('--num_classes', default=5, type=int,
                        help="Number of object classes")
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=2, type=int)

    # --- 评估参数 ---
    parser.add_argument('--resume', required=True, help='resume from checkpoint')
    parser.add_argument('--output_dir', default='test_results',
                        help='path where to save evaluation results')
    parser.add_argument('--device', default='cuda',
                        help='device to use for testing')
    parser.add_argument('--lr_backbone', default=0, type=float) # for build_model
    parser.add_argument('--dilation', action='store_true') # for build_model

    return parser

def main(args):
    device = torch.device(args.device)

    # --- 构建模型 ---
    model, _, _ = build_model(args)
    model.to(device)

    # --- 加载权重 ---
    print(f"Loading weights from {args.resume}")
    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    # --- 构建数据集和数据加载器 ---
    dataset_val = build_dataset(image_set='val', args=args)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn,
                                 num_workers=args.num_workers)

    # --- 创建输出目录 ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 开始评估 ---
    evaluate_voc(
        model=model,
        data_loader=data_loader_val,
        device=device,
        output_dir=args.output_dir,
        epoch=0, # or a custom identifier
        args=args
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR testing script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)