
# MTUNet

## Updates
- [2023/7/11]: Upload the dataset used in the experiment.

## Usage
### Requirements
- Python 3.10
- PyTorch 1.11.0(py3.10_cuda11.3_cudnn8.2.0)
- mmcv-full==1.4.8
- mmsegmentation==0.24.1
- mmdet==2.24.1

To use our code, please first install the mmcv-full and mmseg/mmdet following the official guidelines ([mmseg](https://mmsegmentation.readthedocs.io/zh_CN/latest/get_started.html#id2), [mmdet](https://mmdetection.readthedocs.io/zh_CN/latest/get_started.html#id2)) and prepare the datasets accordingly.
> Note: mmdetection and mmsegmentation have made huge compatibility change in their latest versions. Their latest version is not compatible with this repo. Make sure you install the correct version. We will update our code and make it compatible with their latest versions in the future, please stay tuned.

## Dataset
This repository contains the multi-task learning dataset(`sirst_mtl`) used in the experiment. The original images in this dataset come from [SIRST](https://github.com/YimianDai/sirst), and the object detection annotations and semantic segmentation annotations in the original dataset have been adapted to COCO format. Therefore, the data used for the object detection task includes three folders: `train2017`, `val2017`, and `annotations`, and the accuracy of the object detection annotations provided by the original dataset has been improved. The data used for the semantic segmentation task includes two folders: `img_dir` and `ann_dir`. In the semantic segmentation annotations, the pixel value of class `background` is 0, the pixel value of class `target` is 1, and the pixel value range is 0~255.

**Updating.   
Please stay tuned.**