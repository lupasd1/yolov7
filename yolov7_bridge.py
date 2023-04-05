# ---------------- TEMPLATE ---------------------------------------
# This is a template to help you start writing PythonBridge code  -
# -----------------------------------------------------------------

import rtmaps.core as rt
import rtmaps.types
from rtmaps.base_component import BaseComponent  # base class

import torch
import cv2
import yaml
from torchvision import transforms
# import torchvision
import numpy as np

from utils.datasets import letterbox
from utils.general import non_max_suppression_mask_conf

from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Boxes
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.layers import paste_masks_in_image

import os
data_path = os.path.abspath('data/hyp.scratch.mask.yaml')
data_path = data_path.replace("\\", "/")

torch_model = os.path.abspath('tools/yolov7-mask.pt')
torch_model = torch_model.replace("\\", "/")

# Python class that will be called from RTMaps.
class rtmaps_python(BaseComponent):

    # Constructor has to call the BaseComponent parent class
    def __init__(self):
        BaseComponent.__init__(self)  # call base class constructor

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        with open(data_path) as f:
            self.hyp = yaml.load(f, Loader=yaml.FullLoader)
        weights = torch.load(torch_model)
        model = weights['model']
        self.model = model.half().to(self.device)
        _ = self.model.eval()

    # Dynamic is called frequently:
    # - When loading the diagram
    # - When connecting or disconnecting a wire
    # Here you create your inputs, outputs and properties
    def Dynamic(self):
        # Adding an input called "in" of ANY type
        self.add_input("in", rtmaps.types.ANY)  # define an input

        # Define the output. The type is set to AUTO which means that the output will be typed automatically.
        # You don’t need to set the buffer_size, in that case it will be set automatically.
        self.add_output("out", rtmaps.types.AUTO)

    # Birth() will be called once at diagram execution startup
    def Birth(self):
        print("Passing through Birth()")

    # Core() is called every time you have a new inputs available, depending on your chosen reading policy
    def Core(self):
        # Just copy the input to the output here

        image = self.inputs["in"].ioelt.data.image_data

        image = letterbox(image, 640, stride=64, auto=True)[0]
        image_ = image.copy()
        image = transforms.ToTensor()(image)
        image = torch.tensor(np.array([image.numpy()]))
        image = image.to(self.device)
        image = image.half()

        output = self.model(image)

        inf_out, train_out, attn, mask_iou, bases, sem_output = output['test'], output['bbox_and_cls'], output['attn'], output['mask_iou'], output['bases'], output['sem']

        bases = torch.cat([bases, sem_output], dim=1)
        nb, _, height, width = image.shape
        names = self.model.names
        pooler_scale = self.model.pooler_scale
        pooler = ROIPooler(output_size=self.hyp['mask_resolution'], scales=(pooler_scale,), sampling_ratio=1,
                           pooler_type='ROIAlignV2', canonical_level=2)

        output, output_mask, output_mask_score, output_ac, output_ab = non_max_suppression_mask_conf(inf_out, attn,
                                                                                                     bases, pooler, self.hyp,
                                                                                                     conf_thres=0.25,
                                                                                                     iou_thres=0.65,
                                                                                                     merge=False,
                                                                                                     mask_iou=None)

        pred, pred_masks = output[0], output_mask[0]
        base = bases[0]
        bboxes = Boxes(pred[:, :4])
        original_pred_masks = pred_masks.view(-1, self.hyp['mask_resolution'], self.hyp['mask_resolution'])
        pred_masks = retry_if_cuda_oom(paste_masks_in_image)(original_pred_masks, bboxes, (height, width),
                                                             threshold=0.5)
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


        out = rtmaps.types.Ioelt()
        out.data = rtmaps.types.IplImage()
        out.data.image_data = pnimg
        out.data.color_model = "COLR"
        out.data.channel_seq = "RGB"

        self.write("out", out)

    # Death() will be called once at diagram execution shutdown
    def Death(self):
        print("Passing through Death()")



#
# cv2.imshow('Output', pnimg)
# cv2.waitKey(0)