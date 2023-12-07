import json
import torch
import os
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from util.misc import NestedTensor
from torchvision.transforms import ToTensor
from PIL import Image
import datasets.transforms as T


class BCCTVDataset(Dataset):
    def __init__(self, dataset_type: str, dataset_path: str, transformer):
        super().__init__()

        self.data_path = Path(dataset_path)
        self.to_tensor = ToTensor()
        self.transformer = transformer

        self._load_images(dataset_type)

    def _load_images(self, type: str):
        annotation = json.load(open(self.data_path / 'annotation.json', 'r'))
        image_ids = annotation[type]

        self.images = []

        for image_id in image_ids:
            self.images.append(annotation['images'][image_id])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        target = self.images[idx]

        image = Image.open(self.data_path / 'images' /
                           target['filename']).convert('RGB')

        w, h = image.size

        boxes = [[bbox['x'], bbox['y'], bbox['x'] + bbox['width'],
                  bbox['y'] + bbox['height']] for bbox in target['boxes']]
        classes = [bbox['category_id'] for bbox in target['boxes']]

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        classes = torch.as_tensor(classes, dtype=torch.int64)

        size = torch.tensor([int(h), int(w)])
        original_size = torch.as_tensor([int(h), int(w)])
        image_id = torch.tensor([target['id']])

        # clamp boxes to image
        boxes[:, 0].clamp_(min=0, max=w)
        boxes[:, 1].clamp_(min=0, max=h)
        boxes[:, 2].clamp_(min=0, max=w)
        boxes[:, 3].clamp_(min=0, max=h)

        target = {
            'image_id': image_id,
            'boxes': boxes,
            'labels': classes,
            'area': boxes[:, 3] * boxes[:, 2],
            'original_size': original_size,
            'size': size,
        }

        image, target = self.transformer(image, target)

        return image, target


def merge_images_to_batch(images):

    max_height = max([image.shape[1] for image, _ in images])
    max_width = max([image.shape[2] for image, _ in images])

    batch = torch.zeros((len(images), 3, max_height, max_width))
    mask = torch.ones((len(images), max_height, max_width), dtype=torch.bool)

    for i, (image, _) in enumerate(images):
        batch[i, :, :image.shape[1], :image.shape[2]] = image
        mask[i, :image.shape[1], :image.shape[2]] = False

    targets = [target for _, target in images]

    return NestedTensor(batch, mask), targets


def make_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [736, 768, 800]

    if image_set == 'train':
        if os.environ['DEVICE'] != 'cpu':
            return T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=1333, d=32),
                    T.Compose([
                        T.RandomResize([400, 500, 600]),
                        T.RandomSizeCrop(384, 600),
                        T.RandomResize(scales, max_size=1333, d=32),
                    ])
                ),
                normalize,
            ])
        else:
            return T.Compose([
                T.SquarePad(),
                T.RandomResize([256], max_size=256, d=16),
                normalize,
            ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build_dataset(dataset_type: str, args):
    transformer = make_transforms(dataset_type)
    dataset = BCCTVDataset(dataset_type, args.dataset_path, transformer)
    return dataset
