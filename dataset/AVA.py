import os
import numpy as np
import torch.utils.data
from PIL import Image, ImageOps
from bbox import BBox
from typing import Tuple, List, Type, Iterator
import matplotlib.pyplot as plt
import PIL
import torch.utils.data.dataset
import torch.utils.data.sampler
from PIL import Image
from torch import Tensor
from torchvision.transforms import transforms
import cv2

class AVA():

    class info():
        def __init__(self, img_class, bbox,h,w,img_position):
            self.img_class = int(img_class)-1
            self.bbox = bbox
            self.height=h
            self.weight=w
            self.img_position=img_position
        def __repr__(self):
            return 'info[img_class={0}, bbox={1}]'.format(
                self.img_class, self.bbox)


    def __init__(self):
        self.bboxes=[]
        self.labels=[]
        self.image_ratios = []
        self.image_position=[]

        self.data_dic = {}
        self.data_size={}
        self.path_to_data_dir='/home/aiuser/'
        path_to_AVA_dir = os.path.join(self.path_to_data_dir, 'ava_v2.2', 'preproc','train_clips')
        path_to_videos = os.path.join(path_to_AVA_dir, 'clips')
        self.path_to_keyframe = os.path.join(path_to_AVA_dir, 'keyframes')
        path_to_video_ids_txt = os.path.join(path_to_AVA_dir, 'trainval.txt')

        for frame in sorted(os.listdir(self.path_to_keyframe)):
            img=os.listdir(os.path.join(self.path_to_keyframe, frame))[0]
            #print('img',img,os.listdir(os.path.join(self.path_to_keyframe, frame)))
            img=cv2.imread(os.path.join(self.path_to_keyframe, frame,img))
            #cv2.imshow('result.jpg',img)
            img_shape=img.shape
            self.data_size[frame]=(img_shape[0],img_shape[1])
        # print(self.data_size)
        with open(path_to_video_ids_txt, 'r') as f:
            data = f.readlines()
            for line in data:
                content = line.split(',')
                key=content[0]+"/"+str(int(content[1]))
                img_h=int(self.data_size[content[0]][0])
                img_w = int(self.data_size[content[0]][1])
                if key not in self.data_dic:
                    self.data_dic[key] = [AVA.info(content[6],BBox(  # convert to 0-based pixel index
                        left=float(content[2])*img_w - 1,
                        top=float(content[3])*img_h - 1,
                        right=float(content[4])*img_w - 1,
                        bottom=float(content[5])*img_h - 1),img_h,img_w,key)]
                else:
                    self.data_dic[key].append(AVA.info(content[6], BBox(  # convert to 0-based pixel index
                        left=float(content[2]) * img_w - 1,
                        top=float(content[3]) * img_h - 1,
                        right=float(content[4]) * img_w - 1,
                        bottom=float(content[5]) * img_h - 1), img_h, img_w, key))
            # print('data_dic:',self.data_dic)
        for key in self.data_dic:
            self.bboxes.append([item.bbox.tolist() for item in self.data_dic[key]])
            self.labels.append([item.img_class for item in self.data_dic[key]])
            width = int(self.data_dic[key][0].weight)
            height = int(self.data_dic[key][0].height)
            ratio = float(width / height)
            self.image_ratios.append(ratio)
            self.image_position.append(self.data_dic[key][0].img_position)

    def __len__(self) -> int:
        return len(self.bboxes)

    def num_classes(self):
        return 80

    def __getitem__(self, index: int) -> Tuple[str, Tensor, Tensor, Tensor, Tensor]:
        bboxes = torch.tensor(self.bboxes[index], dtype=torch.float)
        labels = torch.tensor(self.labels[index], dtype=torch.long)
        # print(int(self.image_position[index].split('/')[1]))
        #image = Image.open(self.path_to_keyframe+'/'+image_index[index].split('/')[0]+'/'+str(int(image_index[index].split('/')[1]))+".jpg")
        image = Image.open(self.path_to_keyframe+'/'+self.image_position[index]+".jpg")
        # random flip on only training mode
        # if self._mode == VOC2007.Mode.TRAIN and random.random() > 0.5:
        #     image = ImageOps.mirror(image)
        #     bboxes[:, [0, 2]] = image.width - bboxes[:, [2, 0]]  # index 0 and 2 represent `left` and `right` respectively
        self._image_min_side=600
        self._image_max_side=1000
        image, scale = self.preprocess(image, self._image_min_side, self._image_max_side)
        scale = torch.tensor(scale, dtype=torch.float)
        bboxes *= scale
        return self.image_position[index], image, scale, bboxes, labels


    def __getitem__(self, index: int) -> Tuple[str, Tensor, Tensor, Tensor, Tensor]:
        bboxes = torch.tensor(self.bboxes[index], dtype=torch.float)
        labels = torch.tensor(self.labels[index], dtype=torch.long)
        # print(int(self.image_position[index].split('/')[1]))
        #image = Image.open(self.path_to_keyframe+'/'+image_index[index].split('/')[0]+'/'+str(int(image_index[index].split('/')[1]))+".jpg")
        image = Image.open(self.path_to_keyframe+'/'+self.image_position[index]+".jpg")
        # random flip on only training mode
        # if self._mode == VOC2007.Mode.TRAIN and random.random() > 0.5:
        #     image = ImageOps.mirror(image)
        #     bboxes[:, [0, 2]] = image.width - bboxes[:, [2, 0]]  # index 0 and 2 represent `left` and `right` respectively
        self._image_min_side=600
        self._image_max_side=1000
        image, scale = self.preprocess(image, self._image_min_side, self._image_max_side)
        scale = torch.tensor(scale, dtype=torch.float)
        bboxes *= scale
        return self.image_position[index], image, scale, bboxes, labels



    def preprocess(self,image: PIL.Image.Image, image_min_side: float, image_max_side: float) -> Tuple[Tensor, float]:
        # resize according to the rules:
        #   1. scale shorter side to IMAGE_MIN_SIDE
        #   2. after scaling, if longer side > IMAGE_MAX_SIDE, scale longer side to IMAGE_MAX_SIDE
        scale_for_shorter_side = image_min_side / min(image.width, image.height)
        longer_side_after_scaling = max(image.width, image.height) * scale_for_shorter_side
        scale_for_longer_side = (image_max_side / longer_side_after_scaling) if longer_side_after_scaling > image_max_side else 1
        scale = scale_for_shorter_side * scale_for_longer_side

        transform = transforms.Compose([
            transforms.Resize((round(image.height * scale), round(image.width * scale))),  # interpolation `BILINEAR` is applied by default
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = transform(image)

        return image, scale

    def index2class(self):
        file_path = '/home/aiuser/ava_v2.2/ava_v2.2/ava_action_list_v2.0.csv'
        with open(file_path) as f:
            i2c_dic = {line.split(',')[0]: line.split(',')[1] for line in f}
        # print(i2c_dic)
        return i2c_dic

    def test(self,item_num):
        i2c_dic=self.index2class()
        for i in range(item_num):
            result=self.__getitem__(i)
            bboxes=result[3]
            labels=result[4]
            scale=result[2]
            print('scale:',scale)
            print ('bboxes:',bboxes)
            print ('labels:',labels)
            print('dir:',self.path_to_keyframe + '/' + result[0] + ".jpg")
            image = cv2.imread(self.path_to_keyframe + '/' + result[0] + ".jpg")
            count=0
            for bbox,lable in zip(bboxes,labels):
                count=count+1
                bbox=np.array(bbox)
                lable = int(lable)
                real_x_min = int(bbox[0]/scale)
                real_y_min = int(bbox[1]/scale)
                real_x_max = int(bbox[2]/scale)
                real_y_max = int(bbox[3]/scale)
                # 在每一帧上画矩形，frame帧,(四个坐标参数),（颜色）,宽度
                cv2.rectangle(image, (real_x_min, real_y_min), (real_x_max, real_y_max), (255, 255, 255), 4)
                cv2.putText(image, i2c_dic[str(lable+1)], (real_x_min + 30, real_y_min + 30 * count), cv2.FONT_HERSHEY_COMPLEX,\
                1,(255, 255, 0), 1, False)
            cv2.imshow('Frame', image)
            # 刷新视频
            cv2.waitKey()

if __name__ == '__main__':
    a=AVA()
    a.test(10)