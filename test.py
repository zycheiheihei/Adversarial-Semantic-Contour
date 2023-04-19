import os
import numpy as np
import json
import cv2
from tqdm import tqdm

from utils import *
from mmdet import __version__
from mmdet.apis import init_detector, inference_detector, get_image_ready

import os

import argparse 
# 创建解析器
parser = argparse.ArgumentParser() 

#添加位置参数(positional arguments)
parser.add_argument('--model', type=str, default= 'frcn', help='using what model')
parser.add_argument('--imgdir', type=str, default='data/demo/images',help='img dir')
parser.add_argument('--task', type=str, default= 'disappear',help='task')
parser.add_argument('--annotations', type=str, default= 'data/coco_person/annotations.json',help='task')


def detected(bboxes, labels, gt_bbox, img_cat, task):
    if task == 'disappear':
        num = bboxes.shape[0]
        cnt = 0
        for i in range(num):
            iou = bbox_iou(bboxes[i][:4],gt_bbox[0])
            if iou > 0.5:
                if bboxes[i][-1]>0.5:
                    cnt+=1
                    return True
        return False
    
    elif task == 'misclass':
        num = bboxes.shape[0]
        cnt = 0
        for i in range(num):
            iou = bbox_iou(bboxes[i][:4],gt_bbox[0])
            if iou > 0.5 and labels[i]==img_cat:
                if bboxes[i][-1]>0.5:
                    cnt+=1
                    return True
        return False
    elif task =='shift':
        num = bboxes.shape[0]
        iou = 0
        maxIoU = 0
        cnt = 0
        for i in range(num):
            ciou = bbox_diou(gt_bbox[0],bboxes[i][:4],CIoU=True)
            # print(ciou)
            if bboxes[i][-1]>0.5:
                if ciou>maxIoU:
                    maxIoU = ciou
                    cnt+=1
        return maxIoU


def test(imgdir, model_type, task, annotations):

    model_config = {}
    model_config['frcn'] = ['mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py', 'mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth']
    model_config['mrcn'] = ['mmdetection/configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py', 'mmdetection/checkpoints/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth']

    model = init_detector(model_config[model_type][0], model_config[model_type][1], device='cpu').cuda()

    ImgAnns = json.load(open(annotations,'r'))

    CatIndex = json.load(open("data/Cat_cocoTommdet.json",'r'))

    counter = 0
    valid_pic = 0
    failure_cases = {}

    failure_cases[model_type] = []

    for imgID in tqdm(ImgAnns):
        img_info = ImgAnns[imgID]['info']
        img_ann = ImgAnns[imgID]['annotation']
        img_name = img_info['file_name']
        if img_name not in os.listdir(imgdir):
            continue
        
        img_path = os.path.join(imgdir, img_name)

        img_cat = CatIndex[str(img_ann['category_id'])]
        valid_pic += 1
        gt_bbox = xywh_to_xyxy(img_ann['bbox']).cuda()
        
        
        img_data = get_image_ready(model, img_path)
        img_data['img_metas'][0][0]['ASC'] = True
        # forward the model
        with torch.no_grad():
            results = model(return_loss=False, rescale=True, **img_data)

        results = results[0]
        bboxes = results[0]
        labels = results[1]

        print(imgID)

        if model_type == 'mrcn':
            bboxes = results[0][0]
            labels = results[0][1]
        success = detected(bboxes,labels,gt_bbox,img_cat,task)
        print(success)


        if task!='shift' and success==0:
            failure_cases[model_type].append(imgID)
        if task=='shift':
            if model_type == "ssd":
                if success>0.1:
                    failure_cases[model_type].append((imgID,float(success.cpu())))
            else:
                if success>0.05 and success<0.2:
                    failure_cases[model_type].append((imgID,float(success.cpu())))
            
        if success and task!='shift':
            counter+=1
        elif task == 'shift':
            counter+=success
        
        print(counter)
        
    print("Accuracy: {}/{} = {}".format(counter, valid_pic, counter/valid_pic))

    # with open(os.path.join(imgdir,"failure.json"),"w") as f:
    #     json.dump(failure_cases, f)

        
 
if __name__ == "__main__":
    args = parser.parse_args()
    img_dir = args.imgdir
    model = args.model
    task = args.task
    anno = args.annotations
    test(img_dir, model, task, anno)
