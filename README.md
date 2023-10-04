# Official YOLOv7

Implementation of paper - [YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://arxiv.org/abs/2207.02696)

Instance segmentaion code is partially based on [BlendMask](https://arxiv.org/abs/2001.00309).

## Installation

1. MAKE SURE YOU ARE ON BRANCH MASK.
2. (Recommended) Create a virtual environment with `python -m venv venv` (Python 3.9 seems to work best).
3. (Recommended) Activate the virtual environment using `source ./venv/Scripts/activate` on Mac/Linux or `venv\Scripts\activate` on Windows.
4. Run `pip install -r requirements.txt` to install all dependancies.
5. If there is an error `No module named torch`, run `pip install torch`.
6. Run `pip install git+https://github.com/facebookresearch/detectron2` to install detectron2.
7. Run `pip3 install torch==1.13.1 torchvision==0.14.1 --index-url https://download.pytorch.org/whl/cu116` to install pyTorch with cuda support.
8. Cd into `yolov7\venv\lib\site-packages\torch\nn\modules` and open `upsampling.py`, find the forward function and remove `recompute_scale_factor=self.recompute_scale_factor` (don't ask why).
9. Run instance.py to test a singular video frame or open `yolov7_panoptic_final.rtd` in RTMaps for video.
10. In the Python Bridge component provide the path to the `yolov7_bridge.py` file and the path to your Python executable, which will probably be in `\venv\Scripts\python.exe`.
   
## Testing

[yolov7-mask.pt](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-mask.pt)

[[scripts]](./tools/instance.ipynb)

<div align="center">
    <a href="./">
        <img src="./figure/horses_instance.png" width="79%"/>
    </a>
</div>

## Citation

```
@article{wang2022yolov7,
  title={{YOLOv7}: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors},
  author={Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
  journal={arXiv preprint arXiv:2207.02696},
  year={2022}
}
```

## Acknowledgements

<details><summary> <b>Expand</b> </summary>

* [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)
* [https://github.com/WongKinYiu/yolor](https://github.com/WongKinYiu/yolor)
* [https://github.com/WongKinYiu/PyTorch_YOLOv4](https://github.com/WongKinYiu/PyTorch_YOLOv4)
* [https://github.com/WongKinYiu/ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)
* [https://github.com/Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
* [https://github.com/ultralytics/yolov3](https://github.com/ultralytics/yolov3)
* [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
* [https://github.com/DingXiaoH/RepVGG](https://github.com/DingXiaoH/RepVGG)
* [https://github.com/JUGGHM/OREPA_CVPR2022](https://github.com/JUGGHM/OREPA_CVPR2022)
* [https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose](https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose)

</details>
