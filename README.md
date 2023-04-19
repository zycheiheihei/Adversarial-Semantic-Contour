# Adversarial-Semantic-Contour

This is the official implementation code for the CVIU paper ["To Make Yourself Invisible with Adversarial Semantic Contours"](https://arxiv.org/abs/2303.00284).


## Installation

Please refer to [Installation](mmdetection/docs/en/get_started.md) for installation instructions.

Note that you should **install it from source**, since some of the source code in mmdetection have been modified to fit our need for ASC.

## Run the demo

To check the clean detection
```python
python test.py --imgdir data/demo/images --model frcn --task disappear
```

To generate adversarial example with Fixed Adeversarial Semantic Contour
```python
python fasc.py --root data/demo --model frcn --method fasc
```


To generate adversarial example with Optimized Adeversarial Semantic Contour
```python
python oasc.py --root data/demo --model frcn
```


## Reproduce the result

Here, we provide the dataset of 1000 Person from MSCOCO2017 and the corresponding result of F-ASC for Faster RCNN. [Download the zip file](https://drive.google.com/file/d/1qw5GxjWnUEGJD6rTzFHvqYpGtq9UckAW/view?usp=share_link) and place it under the directory `data`.

To check the clean detection rate (98.3%)
```
python test.py --imgdir data/coco_person/images --model frcn --task disappear
```


To reproduce the detection rate under attack by F-ASC (9.7%)
```
python test.py --imgdir data/coco_person/result/disappear/fasc_frcn --model frcn --task disappear
```

## Citation

```bibtex
@article{zhang2023make,
  title={To make yourself invisible with Adversarial Semantic Contours},
  author={Zhang, Yichi and Zhu, Zijian and Su, Hang and Zhu, Jun and Zheng, Shibao and He, Yuan and Xue, Hui},
  journal={Computer Vision and Image Understanding},
  volume={230},
  pages={103659},
  year={2023},
  publisher={Elsevier}
}
```