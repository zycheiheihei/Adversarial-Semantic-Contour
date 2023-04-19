# Adversarial-Semantic-Contour

## Installation

Please refer to [Installation](mmdetection/docs/en/get_started.md) for installation instructions.

Note that you should **install it from source in develop mode**, since some of the source code in mmdetection have been modified to fit our need for ASC.

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

Here, we provide the dataset of 1000 Person from MSCOCO2017 and the corresponding result of F-ASC for Faster RCNN.
