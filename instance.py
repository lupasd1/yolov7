import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
# Apple MPS (Metal Performance Shaders) allows use of GPU for
# hardware acceleration (over using only CPU) for ML purposes, etc.
# top 2 lines set environmental variable to allow PyTorch to use CPU
# when MPS fails because the devs didn't implement some function for it
# ^^^ comment top 2 lines out if not using MacOS

import matplotlib.pyplot as plt
import torch
import cv2
import yaml
from torchvision import transforms
import numpy as np
# import models

from utils.datasets import letterbox
from utils.general import non_max_suppression_mask_conf

from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Boxes
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.layers import paste_masks_in_image


data_path = ('./data/hyp.scratch.mask.yaml')
torch_model = ('./tools/yolov7-mask.pt')
target_path = ('./tools/test6.png') # image to segment

# set device for tensor computations
device = "cuda:0" if torch.has_cuda else ("mps" if torch.has_mps else "cpu")
# first check if CUDA (nvidia gpu) available, then MPS (apple gpu), else cpu

with open(data_path) as f:
    hyp = yaml.load(f, Loader=yaml.FullLoader)
weights = torch.load(torch_model)
model = weights['model']
model = model.to(torch.float32).to(device) 
# TODO: investigate if model.to(data_type) and model.to(device) should be called in specific order,
# whether or not there's a difference
_ = model.eval()

image = cv2.imread(target_path)  # 504x378 image
image = letterbox(image, 640, stride=64, auto=True)[0]
image_ = image.copy()
image = transforms.ToTensor()(image)
image = torch.tensor(np.array([image.numpy()]))
image = image.to(device)

output = model(image)

inf_out, train_out, attn, mask_iou, bases, sem_output = output['test'], output['bbox_and_cls'], output['attn'], output['mask_iou'], output['bases'], output['sem']

bases = torch.cat([bases, sem_output], dim=1)
nb, _, height, width = image.shape
names = model.names
pooler_scale = model.pooler_scale
pooler = ROIPooler(output_size=hyp['mask_resolution'], scales=(pooler_scale,), sampling_ratio=1, pooler_type='ROIAlignV2', canonical_level=2)

output, output_mask, output_mask_score, output_ac, output_ab = non_max_suppression_mask_conf(inf_out, attn, bases, pooler, hyp, conf_thres=0.25, iou_thres=0.65, merge=False, mask_iou=None)

pred, pred_masks = output[0], output_mask[0]
base = bases[0]
bboxes = Boxes(pred[:, :4])
original_pred_masks = pred_masks.view(-1, hyp['mask_resolution'], hyp['mask_resolution'])
pred_masks = retry_if_cuda_oom(paste_masks_in_image)( original_pred_masks, bboxes, (height, width), threshold=0.5)
pred_masks_np = pred_masks.detach().cpu().numpy()
pred_cls = pred[:, 5].detach().cpu().numpy()
pred_conf = pred[:, 4].detach().cpu().numpy()
nimg = image[0].permute(1, 2, 0) * 255
nimg = nimg.cpu().numpy().astype(np.uint8)
nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
nbboxes = bboxes.tensor.detach().cpu().numpy().astype(int)
pnimg = nimg.copy()

for one_mask, bbox, cls, conf in zip(pred_masks_np, nbboxes, pred_cls, pred_conf):
    if conf < 0.25:
        continue
    color = [np.random.randint(255), np.random.randint(255), np.random.randint(255)]

    pnimg[one_mask] = pnimg[one_mask] * 0.5 + np.array(color, dtype=np.uint8) * 0.5
    pnimg = cv2.rectangle(pnimg, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

    # below code draws classifications (e.g. car, person]) and confidence scores (e.g. 0.98, 0.64)
    # label = '%s %.3f' % (names[int(cls)], conf)
    # t_size = cv2.getTextSize(label, 0, fontScale=0.5, thickness=1)[0]
    # c2 = bbox[0] + t_size[0], bbox[1] - t_size[1] - 3
    # pnimg = cv2.rectangle(pnimg, (bbox[0], bbox[1]), c2, color, -1, cv2.LINE_AA)  # filled
    # pnimg = cv2.putText(pnimg, label, (bbox[0], bbox[1] - 2), 0, 0.5, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

cv2.imshow('Output', pnimg)
cv2.waitKey(0)