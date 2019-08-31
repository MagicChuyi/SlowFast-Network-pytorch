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
import operator
from torchvision.transforms import transforms
import cv2

from torchvision.transforms import transforms
from dataset.base import Base as DatasetBase
from get_ava_performance import ava_val
from config.config import Config
from config.eval_config import EvalConfig
from config.train_config import TrainConfig
import pandas as pa
from torch.utils.data import DataLoader, Dataset
class AVA_video(Dataset):

    class info():
        def __init__(self, img_class, bbox,h,w,img_position):
            self.img_class = [int(img_class)]
            self.bbox = bbox
            self.height=h
            self.weight=w
            self.img_position=img_position
        def __repr__(self):
            return 'info[img_class={0}, bbox={1}]'.format(
                self.img_class, self.bbox)

    def __init__(self,data_dir,discard=True):
        self.bboxes=[]
        self.labels=[]
        self.image_ratios = []
        self.image_position=[]
        self.widths=[]
        self.heights=[]
        #根据name获取detector_bbox
        self.detector_bboxes_list=[]
        #for debug
        self.name_list=[]
        self.i2c_dic=self.index2class()
        self.data_dic = {}
        self.data_dic_real={}
        self.data_size={}
        self.data_format={}
        self.detector_bbox_dic={}
        self.path_to_data_dir='/home/aiuser/'
        #path_to_AVA_dir = os.path.join(self.path_to_data_dir, 'ava', 'preproc_train')
        path_to_AVA_dir = os.path.join(self.path_to_data_dir, 'ava_v2.2', 'preproc', 'train_clips')
        self.path_to_videos = os.path.join(path_to_AVA_dir, 'clips')
        self.path_to_keyframe = os.path.join(path_to_AVA_dir, 'keyframes')
        self.discard=discard
        self.imshow_lable_dir=data_dir
        path_to_video_ids_txt = os.path.join(path_to_AVA_dir, data_dir)
        path_to_detector_result_txt=os.path.join(path_to_AVA_dir,Config.DETECTOR_RESULT_PATH)
        #得到每个视频的大小，通过读取第一张keyframe
        self.get_video_size()
        # 得到每个视频的格式
        self.get_video_format()
        #读取文件，key是文件名(aa/0930)
        self.read_file_to_dic(path_to_video_ids_txt,self.data_dic)
        self.make_multi_lable(self.data_dic)

        # 获取detector的predict_bbox
        self.read_file_to_dic(path_to_detector_result_txt, self.detector_bbox_dic)
        #print("detector_bbox_dic:",self.detector_bbox_dic)

        #对字典中的数据进行整理，变成list的形式
        self.trans_dic_to_list()


    def get_video_size(self):
        for frame in sorted(os.listdir(self.path_to_keyframe)):
            img=os.listdir(os.path.join(self.path_to_keyframe, frame))[0]
            img=cv2.imread(os.path.join(self.path_to_keyframe, frame,img))
            img_shape=img.shape
            self.data_size[frame]=(img_shape[0],img_shape[1])

    def get_video_format(self):
        for video in sorted(os.listdir(self.path_to_videos)):
            video_0 = os.listdir(os.path.join(self.path_to_videos, video))[0]
            self.data_format[video]='.'+video_0.split('.')[1]
        #print('data_format',self.data_format)

    def read_file_to_dic(self,filename,dic):
        with open(filename, 'r') as f:
            data = f.readlines()
            for line in data:
                content = line.split(',')
                key=content[0]+"/"+str(int(content[1]))
                img_h=int(self.data_size[content[0]][0])
                img_w = int(self.data_size[content[0]][1])
                if key not in dic:
                    dic[key] = [AVA_video.info(content[6],BBox(  # convert to 0-based pixel index
                        left=float(content[2])*img_w ,
                        top=float(content[3])*img_h ,
                        right=float(content[4])*img_w,
                        bottom=float(content[5])*img_h),img_h,img_w,key)]
                else:
                    dic[key].append(AVA_video.info(content[6], BBox(  # convert to 0-based pixel index
                        left=float(content[2]) * img_w,
                        top=float(content[3]) * img_h,
                        right=float(content[4]) * img_w,
                        bottom=float(content[5]) * img_h), img_h, img_w, key))
            # print('data_dic:',self.data_dic)
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
                        #这是个陷坑
                        #dic[key].remove(info)
                else:
                    pre=info
                    temp.append(info)
            dic[key]=temp
        #print("dic:",dic)




    def trans_dic_to_list(self):
        for key in self.data_dic:
            if(key in self.detector_bbox_dic):
                a=self.data_dic[key]
                self.bboxes.append([item.bbox.tolist() for item in self.data_dic[key]])
                self.labels.append([item.img_class for item in self.data_dic[key]])
                assert len(self.bboxes)==len(self.labels)
                self.detector_bboxes_list.append([item.bbox.tolist() for item in self.detector_bbox_dic[key]])
                width = int(self.data_dic[key][0].weight)
                self.widths.append(width)
                height = int(self.data_dic[key][0].height)
                self.heights.append(height)
                ratio = float(width / height)
                self.image_ratios.append(ratio)
                self.image_position.append(self.data_dic[key][0].img_position)
            else:
                continue

    def generate_one_hot(self,lable):
        one_hot_lable=np.zeros((len(lable),81))
        for i,box_lable in enumerate(lable):
            for one in box_lable:
                for j in range(81):
                    if j==int(one):
                        one_hot_lable[i][j]=1
        #print('one_hot_lable:',one_hot_lable)
        return one_hot_lable


    def __len__(self) -> int:
        return len(self.image_position)

    def num_classes(self):
        return 81

    def __getitem__(self, index: int) -> Tuple[str, Tensor, Tensor, Tensor, Tensor]:

        print("######## V1-VERSION ##########")


        buffer, scale,index = self.loadvideo(self.image_position, index, Config.IMAGE_MIN_SIDE, Config.IMAGE_MAX_SIDE, 1,self.discard)
        bboxes = torch.tensor(self.bboxes[index], dtype=torch.float)
        one_hot_lable=self.generate_one_hot(self.labels[index])
        labels = torch.tensor(one_hot_lable, dtype=torch.float)
        detector_bboxes=torch.tensor(self.detector_bboxes_list[index])
        #image = Image.open(self.path_to_keyframe+'/'+self.image_position[index]+".jpg")
        buffer = self.normalize(buffer)
        buffer = self.to_tensor(buffer,self.image_position,index)
        buffer=torch.tensor(buffer, dtype=torch.float)
        scale = torch.tensor(scale, dtype=torch.float)
        img=self.image_position[index]
        # print("####img:",img)
        # print("lable:",self.labels[index])
        # print("befor_bbox:", bboxes)
        # print("before_detector_bboxes:", detector_bboxes)
        bboxes *= scale
        detector_bboxes*= scale
        # print("bbox:", bboxes)
        # print("detector_bboxes:",detector_bboxes)
        # print("scale:",scale)
        return self.image_position[index], buffer, scale, bboxes, labels,detector_bboxes,(self.heights[index],self.widths[index])

    def normalize(self, buffer):
        # Normalize the buffer
        # buffer = (buffer - 128)/128.0
        norm = []
        for i, frame in enumerate(buffer):
            if np.shape(frame)[2]!=3:
                print(np.shape(frame))
            frame = (frame - np.array([[[128.0, 128.0, 128.0]]]))/128.0
            buffer[i] = frame
            norm.append(frame)
        # for n in np.array(norm,dtype="float32"):
        #     if len(pa.isna(n).nonzero()[1])!=0:
        #         cv2.imshow("demo", n)
        #         cv2.waitKey(0)
        return np.array(norm,dtype="float32")

    def to_tensor(self, buffer,image_position,index):
        # convert from [D, H, W, C] format to [C, D, H, W] (what PyTorch uses)
        # D = Depth (in this case, time), H = Height, W = Width, C = Channels
        if len(np.shape(buffer))!=4:
            print('WRONG:',image_position[index], np.shape(buffer))
        try:
            buffer.transpose([3, 0, 1, 2])
        except:
            print(image_position[index],np.shape(buffer))
        return buffer.transpose([3, 0, 1, 2])

    #/home/aiuser/ava_v2.2/preproc/train_clips/clips/cLiJgvrDlWw/1035.mp4
    def loadvideo(self,image_position,index,min_side,max_side,frame_sample_rate,discard):
        formate_key = image_position[index].split('/')[0]
        fname=self.path_to_videos + '/' + image_position[index] + self.data_format[formate_key]
        remainder = np.random.randint(frame_sample_rate)
        # initialize a VideoCapture object to read video data into a numpy array
        capture = cv2.VideoCapture(fname)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("frame_w,h:",fname,frame_width,frame_height)
        fps = int(capture.get(cv2.CAP_PROP_FPS))
        #print('fps:',fps,'frame_count:',frame_count)
        #训练时丢弃帧数过少的数据
        if True:
            while frame_count<80 or frame_height==0 or frame_width==0:
                capture.release()
                print('discard_video,frame_num:',frame_count,'dir:',fname,frame_height,frame_width)
                index = np.random.randint(self.__len__())
                formate_key = image_position[index].split('/')[0]
                fname = self.path_to_videos + '/' + image_position[index] + self.data_format[formate_key]
                capture = cv2.VideoCapture(fname)
                frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #将图片缩放，遵循frcnn方式
        if frame_count<80:
            print("fname no:",fname,frame_width,frame_height,min_side)
        try:
            scale_for_shorter_side = min_side / min(frame_width, frame_height)
        except:
            print("fname:",fname,frame_width,frame_height,min_side)
        if frame_height==0 or frame_width==0:
            print("WARNING:SHIT DATA")
        longer_side_after_scaling = max(frame_width, frame_height) * scale_for_shorter_side
        scale_for_longer_side = (
                    max_side / longer_side_after_scaling) if longer_side_after_scaling > max_side else 1
        scale = scale_for_shorter_side * scale_for_longer_side
        resize_height=round(frame_height * scale)
        resize_width=round(frame_width * scale)
        # create a buffer. Must have dtype float, so it gets converted to a FloatTensor by Pytorch later
        start_idx = 0
        end_idx = 0
        frame_keep_count=64
        if frame_count==120:
            start_idx=43
            end_idx=107
            #print("120", start_idx, end_idx)
        if frame_count==100:
            start_idx =30
            end_idx =94
            #print("120", start_idx, end_idx)
        if frame_count==93:
            start_idx =26
            end_idx =90
            #print("120", start_idx, end_idx)
        if frame_count!=120 and frame_count!=100 and frame_count!=93:
            #print("warning:without keep keyframe")
            end_idx=frame_count
            start_idx=end_idx-64-1

        buffer = np.zeros((frame_keep_count, resize_height, resize_width, 3), np.dtype('int8'))
        #buffer=[]
        #将数据填入空的buffer
        count = 0
        retaining = True
        sample_count = 0
        # read in each frame, one at a time into the numpy buffer array
        num=0
        while (count <= end_idx and retaining):
            num=num+1
            retaining, frame = capture.read()
            if count <= start_idx:
                count += 1
                continue
            if retaining is False or count>end_idx:
                break
            if count%frame_sample_rate == remainder and sample_count < frame_keep_count:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # will resize frames if not already final size
                if (frame_height != resize_height) or (frame_width != resize_width):
                    frame = cv2.resize(frame, (resize_width, resize_height))
                buffer[sample_count] = frame
                #buffer.append(frame)
                sample_count = sample_count + 1
            count += 1
        capture.release()
        # if len(buffer)<64:
        #     try:
        #         for i in range(64-len(buffer)):
        #             temp=buffer[-1]
        #             buffer.append(temp)
        #     except:
        #         print('fail padding',fname)
        #
        # if len(buffer)!=64:
        #     print('fail',fname)
        return buffer,scale,index

    def evaluate(self, path_to_results_dir: str,all_image_ids, bboxes: List[List[float]], classes: List[int], probs: List[float],img_size) -> Tuple[float, str]:
        self._write_results(path_to_results_dir,all_image_ids, bboxes, classes, probs,img_size)
        ava_val()

    def _write_results(self, path_to_results_dir: str, image_ids: List[str], bboxes: List[List[float]],
                       classes, probs: List[float],img_size):
        f = open(path_to_results_dir,mode='a+')
        for image_id, bbox, cls, prob in zip(image_ids, bboxes, classes, probs):
            #print(str(image_id.split('/')[0]),str(image_id.split('/')[1]), bbox[0]/int(img_size[1]), bbox[1], bbox[2], bbox[3],(int(cls)+1),prob,img_size[1],int(img_size[0]))
            x1=0 if bbox[0]/int(img_size[1])<0 else bbox[0]/int(img_size[1])
            y1=0 if bbox[1]/int(img_size[0])<0 else bbox[1]/int(img_size[0])
            x2=1 if bbox[2]/int(img_size[1])>1 else bbox[2]/int(img_size[1])
            y2=1 if bbox[3]/int(img_size[0])>1 else bbox[3]/int(img_size[0])
            print(str(image_id.split('/')[0]),str(image_id.split('/')[1]),x1,y1,x2,y2)
            for c,p in zip(cls,prob):
                f.write('{:s},{:s},{:f},{:f},{:f},{:f},{:s},{:s}\n'.format(str(image_id.split('/')[0]),str(image_id.split('/')[1]), x1, y1, x2, y2,str(c),str(p)))
        f.close()

    def index2class(self):
        file_path = '/media/aiuser/78C2F86DC2F830CC1/ava_v2.2/ava_v2.2/ava_action_list_v2.0.csv'
        with open(file_path) as f:
            i2c_dic = {line.split(',')[0]: line.split(',')[1] for line in f}
        return i2c_dic


    def draw_bboxes_and_show(self,frame,frame_num,bboxes,labels,key_frame_start,key_frame_end,scale=1,probs=[],color=(255, 255, 255)):

            if frame_num > key_frame_start and frame_num < key_frame_end:
                count = 0
                count_2=0
                # Capture frame-by-frame
                if len(probs)==0:
                    for bbox, lable in zip(bboxes, labels):
                        count = count + 1
                        bbox = np.array(bbox)
                        lable = int(lable)
                        real_x_min = int(bbox[0] / scale)
                        real_y_min = int(bbox[1] / scale)
                        real_x_max = int(bbox[2] / scale)
                        real_y_max = int(bbox[3] / scale)
                        # 在每一帧上画矩形，frame帧,(四个坐标参数),（颜色）,宽度
                        cv2.rectangle(frame, (real_x_min, real_y_min), (real_x_max, real_y_max), color, 4)
                        cv2.putText(frame, self.i2c_dic[str(lable)], (real_x_min + 30, real_y_min + 15 * count),
                                    cv2.FONT_HERSHEY_COMPLEX, \
                                    0.5, (255, 255, 0), 1, False)
                else:
                    for bbox,lable,prob in zip(bboxes,labels,probs):
                        count_2 = count_2 + 1
                        bbox=np.array(bbox)
                        lable = int(lable)
                        prob=float(prob)
                        print("probs",probs)
                        real_x_min = int(bbox[0]/scale)
                        real_y_min = int(bbox[1]/scale)
                        real_x_max = int(bbox[2]/scale)
                        real_y_max = int(bbox[3]/scale)
                        # 在每一帧上画矩形，frame帧,(四个坐标参数),（颜色）,宽度
                        cv2.rectangle(frame, (real_x_min, real_y_min), (real_x_max, real_y_max), (255, 255, 255), 4)
                        cv2.putText(frame, self.i2c_dic[str(lable)]+':'+str(prob), (real_x_min+30 , real_y_min+15*count_2 ), cv2.FONT_HERSHEY_COMPLEX,\
                        0.5,(255, 255, 0), 1, False)


    def test(self,item_num,frame_start=0.35,frame_end=0.95):
        for i in range(item_num):
            print(i)
            result=self.__getitem__(i)
            bboxes=result[3]
            labels=result[4]
            _scale=float(result[2])
            print('scale:',_scale)
            print ('bboxes:',bboxes)
            print ('labels:',labels)
            print('dir:',self.path_to_keyframe + '/' + result[0])
            # formate_key = self.image_position[i].split('/')[0]
            # cap = cv2.VideoCapture(self.path_to_videos + '/' + self.image_position[i] + self.data_format[formate_key])
            # frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # key_frame_start = int(frame_count * frame_start)
            # key_frame_end = int(frame_count * frame_end)
            # frame_num = 0
            # while (cap.isOpened()):
            #     ret, frame = cap.read()
            #     frame_num = frame_num + 1
            #     self.draw_bboxes_and_show(frame,frame_num,bboxes,labels,key_frame_start,key_frame_end,scale=_scale)
            #     if ret == True:
            #         # 显示视频
            #         cv2.imshow('Frame', frame)
            #         # 刷新视频
            #         cv2.waitKey(10)
            #         # 按q退出
            #         if cv2.waitKey(25) & 0xFF == ord('q'):
            #             break
            #     else:
            #         break
    def imshow(self,item_num,frame_start=0.35,frame_end=0.95):
        for i in range(item_num):
            result=self.__getitem__(i)
            name=result[0]
            real_bboxes=[item.bbox.tolist() for item in self.data_dic_real[name]]
            real_lables=[item.img_class for item in self.data_dic_real[name]]
            probs=result[5]
            print(type(probs[0]))
            kept_indices = list(np.where(np.array(probs) > 0.2))
            bboxes=np.array(result[3])[kept_indices]
            labels=np.array(result[4])[kept_indices]
            probs=np.array(probs)[kept_indices]
            scale=result[2]
            print('scale:',scale)
            print ('bboxes:',real_bboxes)
            print ('labels:',real_lables)
            print('dir:',self.path_to_keyframe + '/' + result[0])
            formate_key = self.image_position[i].split('/')[0]
            cap = cv2.VideoCapture(self.path_to_videos+'/'+self.image_position[i]+self.data_format[formate_key])
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            key_frame_start = int(frame_count * frame_start)
            key_frame_end = int(frame_count * frame_end)
            frame_num = 0
            while (cap.isOpened()):
                ret, frame = cap.read()
                frame_num = frame_num + 1
                self.draw_bboxes_and_show(frame,frame_num, bboxes, labels, key_frame_start, key_frame_end, scale=scale,probs=probs)
                #self.draw_bboxes_and_show(frame,frame_num, real_bboxes, real_lables, key_frame_start, key_frame_end,color=(255,0,255))
                if ret == True:
                    # 显示视频
                    cv2.imshow('Frame', frame)
                    # 刷新视频
                    cv2.waitKey(10)
                    # 按q退出
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break
                else:
                    break




# if __name__ == '__main__':
#     a=AVA_video(TrainConfig.TRAIN_DATA)
#     a.test(10)

if __name__ == '__main__':
    train_dataloader = \
        DataLoader(AVA_video(TrainConfig.TRAIN_DATA), batch_size=10, shuffle=True,collate_fn=DatasetBase.padding_collate_fn,num_workers=10)
    for n_iter, (_, image_batch, _, bboxes_batch, labels_batch,detector_bboxes_batch) in enumerate(train_dataloader):
        print("n_iter: ", n_iter)