import os
import logger
import numpy as np
import torch.utils.data
from PIL import Image, ImageOps
from bbox1 import BBox
from typing import Tuple, List, Type, Iterator
import operator
from torch import Tensor
import cv2
from torch.utils.data import DataLoader, Dataset
from config.eval_config import EvalConfig
class Imshow_result(Dataset):

    class info():
        def __init__(self, img_class,prob, bbox,h,w,img_position):
            self.img_class = [int(img_class)]
            self.prob = [prob]
            self.bbox = bbox
            self.height=h
            self.weight=w
            self.img_position=img_position
        def __repr__(self):
            return 'info[img_class={0}, bbox={1}]'.format(
                self.img_class, self.bbox)


    def __init__(self,imshow_result_dir,imshow_lable_dir):
        self.bboxes=[]
        self.labels=[]
        self.image_ratios = []
        self.image_position=[]
        self.widths=[]
        self.heights=[]
        self.probs=[]
        self.i2c_dic=self.index2class()
        self.data_dic = {}
        self.data_dic_real={}
        self.data_size={}
        self.data_format={}
        self.path_to_data_dir='/home/aiuser/'
        path_to_AVA_dir = os.path.join(self.path_to_data_dir, 'ava','ava', 'preproc_val')
        self.path_to_videos = os.path.join(path_to_AVA_dir, 'clips')
        self.path_to_keyframe = os.path.join(path_to_AVA_dir, 'keyframes')
        self.imshow_lable_dir=imshow_lable_dir
        path_to_result_ids_txt = os.path.join(path_to_AVA_dir, imshow_result_dir)
        #得到每个视频的大小，通过读取第一张keyframe
        self.get_video_size()
        # 得到每个视频的格式
        self.get_video_format()
        #读取文件，key是文件名(aa/0930)
        self.read_file_to_dic(path_to_result_ids_txt,self.data_dic)
        self.make_multi_lable(self.data_dic)
        #对字典中的数据进行整理，变成list的形式
        self.trans_dic_to_list()

        path_to_lable_ids_txt = os.path.join(path_to_AVA_dir, imshow_lable_dir)
        self.read_file_to_dic(path_to_lable_ids_txt, self.data_dic_real)
        self.make_multi_lable(self.data_dic_real)


    def get_video_size(self):
        for frame in sorted(os.listdir(self.path_to_keyframe)):
            img=os.listdir(os.path.join(self.path_to_keyframe, frame))[0]
            img=cv2.imread(os.path.join(self.path_to_keyframe, frame,img))
            img_shape=img.shape
            self.data_size[frame]=(img_shape[0],img_shape[1])

    def get_video_format(self):
        for video in sorted(os.listdir(self.path_to_videos)):
            video_0 = os.listdir(os.path.join(self.path_to_videos,\
                                              video))[0]
            self.data_format[video]='.'+video_0.split('.')[1]


    def read_file_to_dic(self,filename,dic):
        # with open("/home/aiuser/ava/ava/ava_val_v2.2.csv", 'r') as f:
        #     data = f.readlines()
        #     del_list=[]
        #     for line in data:
        #         content = line.split(',')
        #         del_list.append(content[0]+"/"+str(int(content[1])))
        #     print("del_list:",del_list)
        del_list=[]
        with open(filename, 'r') as f:
            data = f.readlines()
            for line in data:
                content = line.split(',')
                key=content[0]+"/"+str(int(content[1]))
                if key not in del_list:
                    img_h=int(self.data_size[content[0]][0])
                    img_w = int(self.data_size[content[0]][1])
                    if key not in dic:
                        dic[key] = [Imshow_result.info(content[6],float(content[7]),BBox(  # convert to 0-based pixel index
                            left=float(content[2])*img_w - 1,
                            top=float(content[3])*img_h - 1,
                            right=float(content[4])*img_w - 1,
                            bottom=float(content[5])*img_h - 1),img_h,img_w,key)]
                    else:
                        dic[key].append(Imshow_result.info(content[6],float(content[7]), BBox(  # convert to 0-based pixel index
                            left=float(content[2]) * img_w - 1,
                            top=float(content[3]) * img_h - 1,
                            right=float(content[4]) * img_w - 1,
                            bottom=float(content[5]) * img_h - 1), img_h, img_w, key))
                else:
                    print(key)

            # print('data_dic:',self.data_dic)

    def trans_dic_to_list(self):
        for key in self.data_dic:
            self.bboxes.append([item.bbox.tolist() for item in self.data_dic[key]])
            self.labels.append([item.img_class for item in self.data_dic[key]])
            self.probs.append([item.prob for item in self.data_dic[key]])
            width = int(self.data_dic[key][0].weight)
            self.widths.append(width)
            height = int(self.data_dic[key][0].height)
            self.heights.append(height)
            ratio = float(width / height)
            self.image_ratios.append(ratio)
            self.image_position.append(self.data_dic[key][0].img_position)

    def make_multi_lable(self,dic):
        for key in dic:
            pre=None
            #print("before:",dic[key])
            temp=[]
            for info in dic[key]:
                if pre==None:
                    pre=info
                    temp.append(info)
                elif operator.eq(info.bbox.tolist(),pre.bbox.tolist()):
                        temp[-1].img_class.append(info.img_class[0])
                        temp[-1].prob.append(info.prob[0])
                        #这是个陷坑
                        #dic[key].remove(info)
                else:
                    pre=info
                    temp.append(info)
            dic[key]=temp

    def __getitem__(self, index: int) -> Tuple[str, Tensor, Tensor, Tensor, Tensor]:
        bboxes = self.bboxes[index]
        labels = self.labels[index]
        # if self.imshow_lable_dir!=None:
        #     probs = [float(item) for item in self.probs[index]]
        return self.image_position[index], index, index, bboxes, labels,self.probs[index]

    def index2class(self):
        file_path = '/home/aiuser/ava_v2.2/ava_v2.2/ava_action_list_v2.0.csv'
        with open(file_path) as f:
            i2c_dic = {line.split(',')[0]: line.split(',')[1] for line in f}
        return i2c_dic

    def draw_bboxes_and_show(self, frame, frame_num, bboxes, labels, key_frame_start, key_frame_end, scale=1, probs=[]):

        if frame_num > key_frame_start and frame_num < key_frame_end:
            # Capture frame-by-frame
            if len(probs) == 0:  # 标签
                for bbox, lables in zip(bboxes, labels):
                    count=0
                    for lable in lables:
                        count = count + 1
                        bbox = np.array(bbox)
                        lable = int(lable)
                        real_x_min = int(bbox[0] / scale)
                        real_y_min = int(bbox[1] / scale)
                        real_x_max = int(bbox[2] / scale)
                        real_y_max = int(bbox[3] / scale)
                        # 在每一帧上画矩形，frame帧,(四个坐标参数),（颜色）,宽度
                        cv2.rectangle(frame,(real_x_min, real_y_min),(real_x_max, real_y_max),(17, 238, 105), 4)  # 绿色
                        cv2.putText(frame, self.i2c_dic[str(lable)].split("(")[0], (real_x_min + 15, real_y_min + 15 * count),
                                    cv2.FONT_HERSHEY_COMPLEX, \
                                    0.5, (17, 238, 105), 1, False)

            else:  # 预测
                for bbox, lables, prob in zip(bboxes, labels, probs):
                    count_2=0
                    for lable,p in zip(lables,prob):
                        count_2 = count_2 + 1
                        bbox = np.array(bbox)
                        lable = int(lable)
                        p = float(p)
                        real_x_min = int(bbox[0] / scale)
                        real_y_min = int(bbox[1] / scale)
                        real_x_max = int(bbox[2] / scale)
                        real_y_max = int(bbox[3] / scale)
                        # 在每一帧上画矩形，frame帧,(四个坐标参数),（颜色）,宽度
                        cv2.rectangle(frame, (real_x_min, real_y_min), (real_x_max, real_y_max), (0, 0, 255),
                                      4)  # 红色
                        cv2.putText(frame, self.i2c_dic[str(lable)].split("(")[0] + ':' + str(round(p, 2)),
                                    (real_x_min + 15, real_y_max - 15 * count_2),
                                    cv2.FONT_HERSHEY_COMPLEX, \
                                    0.5, (0, 0, 255), 1, False)

    def imshow(self,item_num,frame_start=0.55,frame_end=0.9):
        for i in range(item_num):
            result=self.__getitem__(i)
            print(result)
            name=result[0]
            real_bboxes=[item.bbox.tolist() for item in self.data_dic_real[name]]
            real_lables=[item.img_class for item in self.data_dic_real[name]]

            probs=result[5]
            #print(type(probs[0]))
            keep=0.2
            keep_labels=[]
            keep_probs=[]
            for n,p in enumerate(probs):
                kept_indices = list(np.where(np.array(p) > keep))
                keep_labels.append(np.array(result[4][n])[kept_indices])
                keep_probs.append(np.array(p)[kept_indices])
            #labels = np.array(result[4])[kept_indices]
            bboxes=np.array(result[3])
            # print ('bboxes:',real_bboxes)
            # print ('labels:',real_lables)
            # print('dir:',self.path_to_keyframe + '/' + result[0])
            print('image_position:',self.image_position)
            formate_key = self.image_position[i].split('/')[0]
            cap = cv2.VideoCapture(self.path_to_videos+'/'+self.image_position[i]+self.data_format[formate_key])
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            key_frame_start = int(frame_count * frame_start)
            key_frame_end = int(frame_count * frame_end)
            frame_num = 0
            count=0
            while (cap.isOpened()):
                ret, frame = cap.read()
                frame_num = frame_num + 1
                self.draw_bboxes_and_show(frame,frame_num, bboxes, keep_labels, key_frame_start, key_frame_end,probs=keep_probs)
                self.draw_bboxes_and_show(frame,frame_num, real_bboxes, real_lables, key_frame_start, key_frame_end)
                if ret == True:
                    count +=1
                    # 显示视频
                    cv2.imwrite('/home/aiuser/frames/%d.jpg' % count, frame)
                    cv2.imshow('Frame', frame)
                    # 刷新视频
                    cv2.waitKey(0)
                    # 按q退出
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break
                else:
                    break

if __name__ == '__main__':
    a = Imshow_result(imshow_result_dir="/home/aiuser/ava/ava/result.txt",imshow_lable_dir=EvalConfig.PATH_TO_LABLE)
    a.imshow(100)

