# Official YOLOv7

Implementation of paper - [YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://arxiv.org/abs/2207.02696)

Instance segmentaion code is partially based on [BlendMask](https://arxiv.org/abs/2001.00309).

## Installation

1. MAKE SURE YOU ARE ON BRANCH MASK!
  > e.g. run `git clone -b mask [repolink]`
2. (Recommended) Create a virtual environment
  > Run `python -m venv venv` (Python 3.9 seems to work best).
3. (Recommended) Activate the virtual environment
  > Run  `source ./venv/Scripts/activate` on Mac/Linux or `venv\Scripts\activate` on Windows.
4. Install all dependencies
  > Run `pip install -r requirements.txt` (while in directory 'yolov7')
5. Download pre-trained YOLO model (if not already there)
> Download [yolov7-mask.pt](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-mask.pt) and place into the `/tools` folder
6. (Recommended) Enable CUDA GPU Acceleration. For NVIDIA GPUs only, skip if other GPU
  > Run `pip3 install torch==1.13.1 torchvision==0.14.1 --index-url https://download.pytorch.org/whl/cu116` to install pyTorch with cuda support
7. Run the instance segmenter!
  > Run `instance.py` to test a singular video frame or open `yolov7_panoptic_final.rtd` in RTMaps for video.
8. (RTMaps) In the Python Bridge component provide the path to the `yolov7_bridge.py` file and the path to your Python executable, which will probably be in `\venv\Scripts\python.exe` (if using virtual environment).

### Troubleshooting

9. `detectron2` can't be found AFTER installing dependencies
  > Run `pip install git+https://github.com/facebookresearch/detectron2` to install detectron2 as library
10. MPS & CPU can't upsample (operation not supported on cpu):
  > **i.** Find your torch module: run `pip show torch`. One of the lines should show Location: [path]<br>
  **2.** Copy this path and cd into it. Then, run `cd ./torch/nn/modules`. If we ls, we should see a bunch of python files.<br>
  **3.** Open `upsampling.py`, then Ctrl+F or Cmd+F and search the `forward` function. Then, remove/comment out the final parameter `recompute_scale_factor=self.recompute_scale_factor`<br>
  ... basically rescaling/upsampling is not supported on non-CUDA devices and this is a patch
   
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
