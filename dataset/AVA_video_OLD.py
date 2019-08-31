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
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from dataset.base import Base as DatasetBase
from get_ava_performance import ava_val
from config.config import Config
from config.eval_config import EvalConfig
from config.train_config import TrainConfig
class AVA_video(Dataset):

    class info():
        def __init__(self, img_class, bbox,h,w,img_position):
            self.img_class = int(img_class)
            self.bbox = bbox
            self.height=h
            self.weight=w
            self.img_position=img_position
        def __repr__(self):
            return 'info[img_class={0}, bbox={1}]'.format(
                self.img_class, self.bbox)


    def __init__(self,mode):
        self.bboxes=[]
        self.labels=[]
        self.image_ratios = []
        self.image_position=[]
        self.widths=[]
        self.heights=[]



        self.data_dic = {}
        self.data_size={}
        self.data_format={}
        self.path_to_data_dir='/home/aiuser/'
        path_to_AVA_dir = os.path.join(self.path_to_data_dir, 'ava_v2.2', 'preproc','train_clips')
        self.path_to_videos = os.path.join(path_to_AVA_dir, 'clips')
        self.path_to_keyframe = os.path.join(path_to_AVA_dir, 'keyframes')
        path_to_video_ids_txt = os.path.join(path_to_AVA_dir, TrainConfig.TRAIN_DATA)
        #测试时时加载这个文件，用里面的数据送入forward
        if mode=='val':
            #path_to_video_ids_txt='/home/aiuser/ava_v2.2/val.txt'
            path_to_video_ids_txt = os.path.join(path_to_AVA_dir, EvalConfig.VAL_DATA)
        #得到每个视频的大小，通过读取第一张keyframe
        for frame in sorted(os.listdir(self.path_to_keyframe)):
            img=os.listdir(os.path.join(self.path_to_keyframe, frame))[0]
            img=cv2.imread(os.path.join(self.path_to_keyframe, frame,img))
            img_shape=img.shape
            self.data_size[frame]=(img_shape[0],img_shape[1])
        # 得到每个视频的格式
        for video in sorted(os.listdir(self.path_to_videos)):
            video_0 = os.listdir(os.path.join(self.path_to_videos, video))[0]
            self.data_format[video]='.'+video_0.split('.')[1]

        print('data_format',self.data_format)
        #读取文件，key是文件名(aa/0930)
        with open(path_to_video_ids_txt, 'r') as f:
            data = f.readlines()
            for line in data:
                content = line.split(',')
                key=content[0]+"/"+str(int(content[1]))
                img_h=int(self.data_size[content[0]][0])
                img_w = int(self.data_size[content[0]][1])
                if key not in self.data_dic:
                    self.data_dic[key] = [AVA_video.info(content[6],BBox(  # convert to 0-based pixel index
                        left=float(content[2])*img_w - 1,
                        top=float(content[3])*img_h - 1,
                        right=float(content[4])*img_w - 1,
                        bottom=float(content[5])*img_h - 1),img_h,img_w,key)]
                else:
                    self.data_dic[key].append(AVA_video.info(content[6], BBox(  # convert to 0-based pixel index
                        left=float(content[2]) * img_w - 1,
                        top=float(content[3]) * img_h - 1,
                        right=float(content[4]) * img_w - 1,
                        bottom=float(content[5]) * img_h - 1), img_h, img_w, key))
            # print('data_dic:',self.data_dic)
        #对字典中的数据进行整理，变成list的形式
        for key in self.data_dic:
            self.bboxes.append([item.bbox.tolist() for item in self.data_dic[key]])
            self.labels.append([item.img_class for item in self.data_dic[key]])
            width = int(self.data_dic[key][0].weight)
            self.widths.append(width)
            height = int(self.data_dic[key][0].height)
            self.heights.append(height)
            ratio = float(width / height)
            self.image_ratios.append(ratio)
            self.image_position.append(self.data_dic[key][0].img_position)
    #warning!!! return len(self.bboxes)
    def __len__(self) -> int:
        return len(self.image_position)

    def num_classes(self):
        return 81

    def __getitem__(self, index: int) -> Tuple[str, Tensor, Tensor, Tensor, Tensor]:
        buffer, scale,index = self.loadvideo(self.image_position, index, Config.IMAGE_MIN_SIDE, Config.IMAGE_MAX_SIDE, 1)
        bboxes = torch.tensor(self.bboxes[index], dtype=torch.float)
        labels = torch.tensor(self.labels[index], dtype=torch.long)
        #image = Image.open(self.path_to_keyframe+'/'+self.image_position[index]+".jpg")
        buffer = self.normalize(buffer)
        buffer = self.to_tensor(buffer)
        buffer=torch.tensor(buffer, dtype=torch.float)
        scale = torch.tensor(scale, dtype=torch.float)
        bboxes *= scale
        return self.image_position[index], buffer, scale, bboxes, labels,(self.heights[index],self.widths[index])

    def normalize(self, buffer):
        # Normalize the buffer
        # buffer = (buffer - 128)/128.0
        for i, frame in enumerate(buffer):
            frame = (frame - np.array([[[128.0, 128.0, 128.0]]]))/128.0
            buffer[i] = frame
        return buffer
    def to_tensor(self, buffer):
        # convert from [D, H, W, C] format to [C, D, H, W] (what PyTorch uses)
        # D = Depth (in this case, time), H = Height, W = Width, C = Channels
        return buffer.transpose((3, 0, 1, 2))

    def load_val_video(self,image_position,index,min_side,max_side,frame_sample_rate):
        formate_key = image_position[index].split('/')[0]
        fname = self.path_to_videos + '/' + image_position[index] + self.data_format[formate_key]
        remainder = np.random.randint(frame_sample_rate)
        # initialize a VideoCapture object to read video data into a numpy array
        capture = cv2.VideoCapture(fname)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        scale_for_shorter_side = min_side / min(frame_width, frame_height)
        longer_side_after_scaling = max(frame_width, frame_height) * scale_for_shorter_side
        scale_for_longer_side = (
                max_side / longer_side_after_scaling) if longer_side_after_scaling > max_side else 1
        scale = scale_for_shorter_side * scale_for_longer_side
        resize_height = round(frame_height * scale)
        resize_width = round(frame_width * scale)
        # create a buffer. Must have dtype float, so it gets converted to a FloatTensor by Pytorch later
        start_idx = 0
        end_idx = frame_count - 1
        frame_count_sample = frame_count // frame_sample_rate - 1
        if frame_count > 300:
            end_idx = np.random.randint(300, frame_count)
            start_idx = end_idx - 300
            frame_count_sample = 301 // frame_sample_rate - 1
        buffer = np.empty((frame_count_sample, resize_height, resize_width, 3), np.dtype('float32'))
        count = 0
        retaining = True
        sample_count = 0
        # read in each frame, one at a time into the numpy buffer array
        num = 0
        while (count <= end_idx and retaining):
            num = num + 1
            retaining, frame = capture.read()
            if count < start_idx:
                count += 1
                continue
            if retaining is False or count > end_idx:
                break
            if count % frame_sample_rate == remainder and sample_count < frame_count_sample:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # will resize frames if not already final size
                if (frame_height != resize_height) or (frame_width != resize_width):
                    frame = cv2.resize(frame, (resize_width, resize_height))
                buffer[sample_count] = frame
                sample_count = sample_count + 1
            count += 1
        capture.release()
        return buffer, scale

    #/home/aiuser/ava_v2.2/preproc/train_clips/clips/cLiJgvrDlWw/1035.mp4
    def loadvideo(self,image_position,index,min_side,max_side,frame_sample_rate):
        formate_key = image_position[index].split('/')[0]
        fname=self.path_to_videos + '/' + image_position[index] + self.data_format[formate_key]
        remainder = np.random.randint(frame_sample_rate)
        # initialize a VideoCapture object to read video data into a numpy array
        capture = cv2.VideoCapture(fname)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        while frame_count<80:
            print('discard_video,frame_num:',frame_count,'dir:',fname)
            index = np.random.randint(self.__len__())
            formate_key = image_position[index].split('/')[0]
            fname = self.path_to_videos + '/' + image_position[index] + self.data_format[formate_key]
            capture = cv2.VideoCapture(fname)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        scale_for_shorter_side = min_side / min(frame_width, frame_height)
        longer_side_after_scaling = max(frame_width, frame_height) * scale_for_shorter_side
        scale_for_longer_side = (
                    max_side / longer_side_after_scaling) if longer_side_after_scaling > max_side else 1
        scale = scale_for_shorter_side * scale_for_longer_side
        resize_height=round(frame_height * scale)
        resize_width=round(frame_width * scale)
        # create a buffer. Must have dtype float, so it gets converted to a FloatTensor by Pytorch later
        start_idx = 0
        end_idx = frame_count-1
        frame_count_sample = frame_count // frame_sample_rate - 1
        if frame_count>=80:
            start_idx = frame_count - 80
            frame_count_sample = 81 // frame_sample_rate - 1
        buffer = np.empty((frame_count_sample, resize_height, resize_width, 3), np.dtype('float32'))
        count = 0
        retaining = True
        sample_count = 0
        # read in each frame, one at a time into the numpy buffer array
        num=0
        while (count <= end_idx and retaining):
            num=num+1
            retaining, frame = capture.read()
            if count < start_idx:
                count += 1
                continue
            if retaining is False or count>end_idx:
                break
            if count%frame_sample_rate == remainder and sample_count < frame_count_sample:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # will resize frames if not already final size
                if (frame_height != resize_height) or (frame_width != resize_width):
                    frame = cv2.resize(frame, (resize_width, resize_height))
                buffer[sample_count] = frame
                sample_count = sample_count + 1
            count += 1
        capture.release()
        #print('num_pic',num)
        return buffer,scale,index

    def evaluate(self, path_to_results_dir: str, image_ids: List[str], bboxes: List[List[float]], classes: List[int], probs: List[float],img_size) -> Tuple[float, str]:
        self._write_results(path_to_results_dir, image_ids, bboxes, classes, probs,img_size)
        ava_val()

    def _write_results(self, path_to_results_dir: str, image_ids: List[str], bboxes: List[List[float]],
                       classes: List[int], probs: List[float],img_size):
        f = open(path_to_results_dir,mode='a+')
        for image_id, bbox, cls, prob in zip(image_ids, bboxes, classes, probs):
            #print(str(image_id.split('/')[0]),str(image_id.split('/')[1]), bbox[0]/int(img_size[1]), bbox[1], bbox[2], bbox[3],(int(cls)+1),prob,img_size[1],int(img_size[0]))
            x1=0 if bbox[0]/int(img_size[1])<0 else bbox[0]/int(img_size[1])
            y1=0 if bbox[1]/int(img_size[0])<0 else bbox[1]/int(img_size[0])
            x2=1 if bbox[2]/int(img_size[1])>1 else bbox[2]/int(img_size[1])
            y2=1 if bbox[3]/int(img_size[0])>1 else bbox[3]/int(img_size[0])
            print(str(image_id.split('/')[0]),str(image_id.split('/')[1]),x1,y1,x2,y2)
            f.write('{:s},{:s},{:f},{:f},{:f},{:f},{:s},{:f}\n'.format(str(image_id.split('/')[0]),str(image_id.split('/')[1]), x1, y1, x2, y2,str(cls),prob))
        f.close()



    def index2class(self):
        file_path = '/home/aiuser/ava_v2.2/ava_v2.2/ava_action_list_v2.0.csv'
        with open(file_path) as f:
            i2c_dic = {line.split(',')[0]: line.split(',')[1] for line in f}
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
            print('dir:',self.path_to_keyframe + '/' + result[0])
            formate_key = self.image_position[i].split('/')[0]
            cap = cv2.VideoCapture(self.path_to_videos+'/'+self.image_position[i]+self.data_format[formate_key])
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            key_frame_start=int(frame_count*0.35)
            key_frame_end =int(frame_count*0.95)
            frame_num=0
            while (cap.isOpened()):
                ret, frame = cap.read()
                frame_num=frame_num+1
                if frame_num>key_frame_start and frame_num<key_frame_end:
                    count = 0
                    # Capture frame-by-frame
                    for bbox,lable in zip(bboxes,labels):
                        count = count + 1
                        bbox=np.array(bbox)
                        lable = int(lable)
                        real_x_min = int(bbox[0]/scale)
                        real_y_min = int(bbox[1]/scale)
                        real_x_max = int(bbox[2]/scale)
                        real_y_max = int(bbox[3]/scale)
                        # 在每一帧上画矩形，frame帧,(四个坐标参数),（颜色）,宽度
                        cv2.rectangle(frame, (real_x_min, real_y_min), (real_x_max, real_y_max), (255, 255, 255), 4)
                        cv2.putText(frame, i2c_dic[str(lable)], (real_x_min+30 , real_y_min+15*count ), cv2.FONT_HERSHEY_COMPLEX,\
                        0.5,(255, 255, 0), 1, False)
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
#     a=AVA_video(mode='val')
#     a.test(10)

if __name__ == '__main__':
    train_dataloader = \
        DataLoader(AVA_video(mode="train"), batch_size=2, shuffle=True,collate_fn=DatasetBase.padding_collate_fn,num_workers=1,)
    for image_position, buffer, scale, bboxes, labels,(height,widths) in enumerate(train_dataloader):
        print(height,height)