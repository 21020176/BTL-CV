import streamlit as st
from PIL import Image

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
import torch.nn as nn
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

from transweather_model import Transweather


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

    # dataset parameters
    # parser.add_argument('--output_dir', default='',
    #                     help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cpu',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='/content/drive/MyDrive/detr/checkpoints/clean_checkpoint0045.pth',
                        help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--num', default=1, type=int)
    parser.add_argument('--cache_mode', default=False,
                        action='store_true', help='whether to cache images on memory')
    parser.add_argument('--model_type', default='detr_only',
                        help='whether to use sfa-detr or detr-only')

    return parser

transform = T.Compose([
    T.Resize(640),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform2 = T.Compose([
    #T.Resize(480, 720),
    T.ToTensor(),
    T. Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

st.set_page_config(page_title="Computer Vision Test", page_icon=":tada:", layout="wide")

st.sidebar.title("Navigation")
    
navigation_menu = st.sidebar.selectbox("Pick one",
                                       ["About The Project", "Object Detection"])
def main(args):
    if navigation_menu == "About The Project":
        st.title("Object Detection in Adverse weather conditions")
        st.markdown("We use bla bla to . **StreamLit** to create the Web Graphical User Interface (GUI)")
        
    if navigation_menu == "Object Detection":
        
        
        uploaded_image = st.file_uploader("Choose an Image", type="jpg")
        col1, col2, col3 = st.columns(3)
        if uploaded_image is not None:
            with col1:
                st.title("Uploaded Image")
                st.image(uploaded_image, width = 500, use_column_width=False)
                         
            with col2:
                st.title("Restored Image")
                st.image(restore_image(uploaded_image), width = 500, use_column_width=False)
            
            with col3:
                st.title("Identify Objects")
                st.image(identify_object(restore_image(uploaded_image), args, uploaded_image), width = 500, use_column_width=False)
        
transform3 = T.ToPILImage()
#import torchvision.utils as utils
from torchvision.utils import save_image
from io import BytesIO 

def restore_image(image):
  device = torch.device(args.device)

  seed = args.seed + utils.get_rank()
  torch.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)

  net = Transweather()
  net = nn.DataParallel(net, device_ids=0)

  net.load_state_dict(torch.load('/content/drive/MyDrive/detr/checkpoints/best', map_location=torch.device('cpu')))
  net.eval()

  #print("image", type(image))
  img_first = Image.open(image).convert('RGB')
  b = BytesIO()
  img_first.save(b,format="jpeg")

  img = Image.open(b)
 
  img = img.resize((240,352), Image.ANTIALIAS)
  img = transform2(img).unsqueeze(0)  
  #print("img",img)
  pred = net(img)
  #print(pred)
  pred = torch.squeeze(pred)
  save_image(pred, '/content/drive/MyDrive/detr/tensor_image.png')
  pred = transform3(pred)
  pred = pred.resize((342, 228), Image.ANTIALIAS)

  #save_image(pred, '/content/drive/MyDrive/detr/tensor_image.png')
  #print("abc")
  #pred = pred.detach().numpy()

  #im = Image.fromarray(pred, mode="RGB")
  # source_img = Image.open(image).convert("RGBA")
  # dst = Image.new('RGB', (342, 228))
  # dst.paste(source_img, (0, 0))

  #restored_image = image
  return pred

def identify_object(image, args, source):

  utils.init_distributed_mode(args)
  print("git:\n  {}\n".format(utils.get_sha()))

  device = torch.device(args.device)

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
  font = ImageFont.truetype(
        '/workspace/ailab/dungpt/sfa-detr/Humor-Sans.ttf', 18)
  
  #img = Image.open(image)#.convert('RGB')
  #img = transform(img).unsqueeze(0)
  img = transform(image).unsqueeze(0)
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
  im_h, im_w = 342, 228#img.size
  target_sizes = torch.tensor([[im_w, im_h]])
  img_h, img_w = target_sizes.unbind(1)
  scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
  boxes = boxes * scale_fct[:, None, :]
  source_img = Image.open('/content/drive/MyDrive/detr/tensor_image.png').convert("RGBA")
  draw = ImageDraw.Draw(source_img) 
 
  i = 0
  for xmin, ymin, xmax, ymax in boxes[0].tolist():
      label = labels[i] - 1
      draw.rectangle(((xmin, ymin), (xmax, ymax)),
                      outline=COLORS[label], width=3)
      draw.text((xmin + 5, ymin + 5), NAMES[label], font=font)

      i += 1
  dst = Image.new('RGB', (342, 228))
  dst.paste(source_img, (0, 0))
  #identified_objects = image
  return dst#identified_objects


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        'Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    #a = restore_image("/content/drive/MyDrive/detr/000001.jpg")
    # if args.output_dir:
    #     Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

