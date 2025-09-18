# datasets/__init__.py (修改后)
import torch.utils.data
import torchvision

# --- 我们将原来的直接导入修改为按需导入 ---

def get_coco_api_from_dataset(dataset):
    # 这个函数在我们的VOC流程中不会被有效调用，但保留它以防万一
    for _ in range(10):
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco
    # 添加一个返回 None 的路径，以避免在非COCO数据集上出错
    return None

def build_dataset(image_set, args):
    if args.dataset_file == 'coco':
        from .coco import build as build_coco
        return build_coco(image_set, args)
    if args.dataset_file == 'coco_panoptic':
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(image_set, args)
    if args.dataset_file == 'voc_fog':
        from .voc_fog import build as build_voc_fog
        return build_voc_fog(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')