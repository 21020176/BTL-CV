import argparse
import datetime
import json
import random
import time
from pathlib import Path
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import torchvision.transforms as T

import numpy as np
import torch
from torch.utils.data import DataLoader
import datasets
import util.misc as utils
from util import box_ops
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model
import time
import os
import json


def get_args_parser():
    parser = argparse.ArgumentParser(
        'Deformable DETR Detector', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names',
                        default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names',
                        default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    parser.add_argument('--sgd', action='store_true')

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine',
                        default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4,
                        type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', default='./data/coco', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--num', default=10, type=int)
    parser.add_argument('--cache_mode', default=False,
                        action='store_true', help='whether to cache images on memory')
    parser.add_argument('--model_type', default='sfa-detr',
                        help='whether to use sfa-detr or detr-only')

    return parser


# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(640),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)
    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    NAMES = [
        "articulated_truck",
        "bicycle",
        "bus",
        "car",
        "motorcycle",
        "motorized_vehicle",
        "non-motorized_vehicle",
        "pedestrian",
        "pickup_truck",
        "single_unit_truck",
        "work_van"
    ]

    COLORS = [
        '#1abc9c',
        '#2ecc71',
        '#3498db',
        '#9b59b6',
        '#f1c40f',
        '#e67e22',
        '#e74c3c',
        '#ecf0f1',
        '#34495e',
        '#95a5a6',
        '#16a085'
    ]

    anno = json.load(open('./data/annotation.json'))

    font = ImageFont.truetype(
        '/workspace/ailab/dungpt/sfa-detr/data/Humor-Sans.ttf', 18)

    img_ids = random.sample(anno['val'], args.num)
    img_files = [anno['images'][idx] for idx in img_ids]
    print("img_files", img_files)
    from tqdm import tqdm
    #for img_obj in tqdm(img_files):
    t0 = time.time()
    # img_path = os.path.join('./data/images', img_obj['filename'])
    # out_imgName = './' + args.output_dir + \
    #     '/' + img_obj['filename'][:-4]+'.png'
    img_path = '/workspace/ailab/phucnd/TransWeather/dataweather/images/000000.jpg'
    out_imgName = './' + args.output_dir + \
         '/' + '1.png'
    im = Image.open(img_path)
    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)

    img = img.cuda()
    # propagate through the model
    outputs, _ = model(img)

    out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

    prob = out_logits.sigmoid()
    topk_values, topk_indexes = torch.topk(
        prob.view(out_logits.shape[0], -1), 100, dim=1)
    scores = topk_values
    topk_boxes = topk_indexes // out_logits.shape[2]
    labels = topk_indexes % out_logits.shape[2]
    boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
    boxes = torch.gather(
        boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

    print(boxes.shape)
    keep = scores[0] > 0.7
    boxes = boxes[0, keep]
    labels = labels[0, keep]

    # and from relative [0, 1] to absolute [0, height] coordinates
    im_h, im_w = im.size
    # print('im_h,im_w',im_h,im_w)
    target_sizes = torch.tensor([[im_w, im_h]])
    target_sizes = target_sizes.cuda()
    img_h, img_w = target_sizes.unbind(1)
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
    boxes = boxes * scale_fct[:, None, :]
    # plot_results
    source_img = Image.open(img_path).convert("RGBA")
    _source_img = Image.open(img_path).convert("RGBA")

    draw = ImageDraw.Draw(source_img)
    _draw = ImageDraw.Draw(_source_img)

    i = 0
    for xmin, ymin, xmax, ymax in boxes[0].tolist():
        label = labels[i] - 1
        draw.rectangle(((xmin, ymin), (xmax, ymax)),
                        outline=COLORS[label], width=3)
        draw.text((xmin + 5, ymin + 5), NAMES[label], font=font)

        i += 1

    # for box in img_obj['boxes']:
    #     xmin, ymin, w, h = box['x'], box['y'], box['width'], box['height']
    #     label = box['category_id'] - 1

    #     _draw.rectangle(((xmin, ymin), (xmin + w, ymin + h)),
    #                     outline=COLORS[label], width=3)
    #     _draw.text((xmin + 5, ymin + 5), NAMES[label], font=font)

    dst = Image.new('RGB', (source_img.width +
                    _source_img.width, source_img.height))
    dst.paste(_source_img, (0, 0))
    dst.paste(source_img, (_source_img.width, 0))

    dst.save(out_imgName, "png")
    # results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
    # print("Outputs",results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
