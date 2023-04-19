import cv2
import copy
from tqdm import tqdm
from mmdet import __version__
from mmdet.apis import init_detector, get_image_ready
import mmcv

from mmdet.core import *
import numpy as np
import torch.nn.functional as F

from utils import *
import json

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default= 'frcn', help='using what model')
parser.add_argument('--task', type=str, default= 'disappear', help='img dir')
parser.add_argument('--root', type=str, default='data/demo', help='using which method')



class OptimizedASC_Attacker(object):
    def __init__(self,task, model_type, root):
        self.learning_rate = 0.05
        self.task = task
        self.model_type = model_type
        self.root = root
        self.model_config = {}
        self.model_config['frcn'] = ['mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py', 'mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth']
        self.model_config['mrcn'] = ['mmdetection/configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py', 'mmdetection/checkpoints/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth']

        self.model = init_detector(self.model_config[model_type][0], self.model_config[model_type][1], device='cpu').cuda()

        self.mean = torch.Tensor([123.675, 116.28 , 103.53 ]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).cuda()
        self.std = torch.Tensor([58.395, 57.12 , 57.375]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).cuda()

        self.ImgAnns = json.load(open(os.path.join(self.root,'annotations.json'),'r'))

        self.CatIndex = json.load(open("data/Cat_cocoTommdet.json",'r'))

        self.maskTensor = None
        self.maskNumpy = None
        self.original_maskNumpy = None
        self.original_maskTensor = None

        self.imgTensor = None   # rgb
        self.imgNumpy = None    # bgr, for saving
        self.img_processed = None

        self.org_w = None
        self.org_h = None
        self.transformed_w = None
        self.transformed_h = None
        self.pad_w = None
        self.pad_h = None

        self.output = os.path.join(self.root,'result/'+task+'/oasc_'+model_type+'/')
        if not os.path.exists(self.output):
            os.makedirs(self.output)

    def Transform_Patch(self, patch):
        """
            patch: A patched image, RGB, h, w, 3
        """
        clamp_patch=torch.clamp(patch,0,255)
        unsqueezed_patch = clamp_patch.unsqueeze(0)
        resized_patch = F.interpolate(unsqueezed_patch,(self.transformed_h,self.transformed_w),mode='bilinear')
        normalized_patch = (resized_patch-self.mean)/self.std
        pad = (0,self.pad_w,0,self.pad_h)
        padded_patch = F.pad(normalized_patch, pad, "constant", value=0)
        return padded_patch

    def loadMask(self, img_name):
        filename = img_name.split('.')[0]+'.npy'
        maskpath = os.path.join(self.root,'fasc/'+filename)
        mask = np.load(maskpath)
        mask = mask[:,:,np.newaxis]
        mask = mask.repeat(3, axis=2)
        # h, w, 3
        self.original_maskNumpy = mask
        self.original_maskTensor = torch.Tensor(mask).permute([2,0,1]).cuda()

    def loadImage(self, img_name):
        path = os.path.join(self.root,'images/'+img_name)
        self.img_processed = get_image_ready(self.model, path)
        self.org_w = self.img_processed['img_metas'][0][0]['ori_shape'][1]
        self.org_h = self.img_processed['img_metas'][0][0]['ori_shape'][0]
        self.transformed_w = self.img_processed['img_metas'][0][0]['img_shape'][1]
        self.transformed_h = self.img_processed['img_metas'][0][0]['img_shape'][0]
        self.pad_w = self.img_processed['img_metas'][0][0]['pad_shape'][1]-self.transformed_w
        self.pad_h = self.img_processed['img_metas'][0][0]['pad_shape'][0]-self.transformed_h

        self.imgNumpy = mmcv.imread(path)   # h w 3
        # cv2.imwrite(self.output+img_name, self.imgNumpy)
        rgb_image = self.imgNumpy.copy()
        rgb_image = rgb_image[:,:,::-1].copy()
        img_tensor = torch.Tensor(rgb_image)
        self.imgTensor = img_tensor.permute([2,0,1]).cuda()    # 3 h w


    def generateNoise(self, type, h,w):
        """
        Generate a random patch as a starting point for optimization.

        :param type: Can be 'gray' or 'random'. Whether or not generate a gray or a random patch.
        :return:
        """
        if type == 'gray':
            adv_noise = torch.full((3, h, w), 0.5).cuda()
        elif type == 'random':
            adv_noise = torch.rand((3, h, w)).cuda()
        return adv_noise

    def sampleContour(self):
        mask = np.zeros_like(self.original_maskNumpy[:,:,0])
        
        for h in range(0,self.org_h,3):
            for w in range(0,self.org_w,3):
                numPixel = self.original_maskNumpy[h:min(h+3,self.org_h), w:min(w+3,self.org_w), 0].sum()
                kernel_h = min(3,self.org_h-h)
                kernel_w = min(3,self.org_w-w)
                Kernel = np.random.random(kernel_h*kernel_w)
                S = np.argsort(Kernel)
                for t in range(int(numPixel)):
                    coord = S[-1-t]
                    ch = int(coord/kernel_w)
                    cw = int(coord%kernel_w)
                    mask[h+ch][w+cw]=1
        mask = mask[:,:,np.newaxis]
        mask = mask.repeat(3, axis=2)
        self.maskNumpy = mask
        self.maskTensor = torch.Tensor(mask).permute([2,0,1]).cuda()
        

    def loss(self, bboxes, labels, gt_bbox, gt_cat):
        if self.task == 'disappear':
            num = bboxes.shape[0]
            loss = 0
            cnt = 0
            for i in range(num):
                iou = bbox_iou(bboxes[i][:4],gt_bbox[0])
                if iou > 0.45 and bboxes[i][-1]>0.45:
                    loss += bboxes[i][-1]
                    cnt+=1
            return loss
        elif self.task == 'misclass':
            num = bboxes.shape[0]
            loss = 0
            cnt = 0
            for i in range(num):
                iou = bbox_iou(bboxes[i][:4],gt_bbox[0])
                if iou > 0.45 and bboxes[i][-1]>0.45 and labels[i] == gt_cat:
                    loss += bboxes[i][-1]
                    cnt+=1
            return loss
        elif self.task == 'shift':
            num = bboxes.shape[0]
            iou = 0
            conf = 0
            maxIoU = 0
            cnt = 0
            loss = None
            Sum = False
            for i in range(num):
                iou = bbox_diou(bboxes[i][:4],gt_bbox[0])
                
                if bboxes[i][-1]>0.45:
                    if Sum:
                        if iou>0.45:
                            loss+=bboxes[i][-1]
                    else:
                        if iou>0.45:
                            loss = bboxes[i][-1]
                            Sum = True
                        else:
                            if iou>maxIoU:
                                loss = bboxes[i][-1]*iou
                    if iou>maxIoU:
                        maxIoU = iou
            return loss, maxIoU

    def train(self):
        for imgID in tqdm(self.ImgAnns):
            img_info = self.ImgAnns[imgID]['info']
            img_ann = self.ImgAnns[imgID]['annotation']
            img_name = img_info['file_name']

            # if img_name in os.listdir(self.output):
            #     continue

            gt_bbox = xywh_to_xyxy(img_ann['bbox']).cuda()
            gt_cat = self.CatIndex[str(img_ann['category_id'])]
            
            self.loadImage(img_name)
            self.loadMask(img_name)
            
            Successful = False
            if self.task=="shift":
                Successful = 10
            
            for r in range(3):
                if (self.task!='shift' and Successful) or (self.task=='shift' and Successful==0):
                    break
                
                self.sampleContour()
                adv_noise = self.generateNoise("gray",self.org_h,self.org_w)
                adv_noise.requires_grad_(True)

                optimizer = torch.optim.Adam([
                    {'params': adv_noise, 'lr': self.learning_rate}
                ], amsgrad=True)

                best_noise = None
                best_loss = 200
                update_round = 0

                for i in tqdm(range(1000)):
                    # TODO: update
                    update_round+=1
                    adv_noise_mmdet = adv_noise*255.0
                    IMG_TO_BE_SENT = copy.deepcopy(self.img_processed)
                    img_disturbed = self.imgTensor*(1-self.maskTensor)+adv_noise_mmdet*self.maskTensor

                    img_ready = self.Transform_Patch(img_disturbed)
                    IMG_TO_BE_SENT['img'][0] = img_ready
                    IMG_TO_BE_SENT['img_metas'][0][0]['ASC'] = True

                    results = self.model(return_loss=False, rescale=True, **IMG_TO_BE_SENT)
                    
                    results = results[0]
                    bboxes = results[0]
                    labels = results[1]

                    if self.model_type == 'mrcn':
                        bboxes = results[0][0]
                        labels = results[0][1]

                    loss_func = self.loss(bboxes, labels, gt_bbox, gt_cat)
                    
                    print(loss_func)
                    if self.task == 'shift':
                        maxiou = loss_func[1]
                        loss_func = loss_func[0]
                        if loss_func is None:
                            best_noise = adv_noise.clone()
                            best_loss = 0
                            Successful = 0
                            break
                        if maxiou < best_loss:
                            update_round = 0
                            best_loss = maxiou
                            best_noise = adv_noise.clone()
                    else:
                        if loss_func == 0:
                            best_noise = adv_noise.clone()
                            Successful=True
                            break
                        
                        if loss_func < best_loss:
                            best_loss = loss_func
                            best_noise = adv_noise.clone()
                            update_round = 0
                    loss_func.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    if update_round==200:
                        break
                
                
                print("{} minimum loss: {}".format(imgID, best_loss))
                if (self.task!='shift' and Successful) or (self.task=='shift' and best_loss<Successful):
                    if self.task=='shift':
                        Successful = best_loss
                    best_noise_np = best_noise.cpu().detach().numpy().transpose([1,2,0])*255
                    best_noise_np = np.round(best_noise_np)
                    best_noise_np = np.clip(best_noise_np,0,255)
                    patched_img = self.imgNumpy*(1-self.maskNumpy)+best_noise_np[:,:,::-1]*self.maskNumpy
                    cv2.imwrite(self.output+img_name,patched_img)

if __name__ == "__main__":
    args = parser.parse_args()
    attacker = OptimizedASC_Attacker(task=args.task, model_type=args.model, root=args.root)
    attacker.train()