import cv2
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
import torch
from config import params
import torch.backends.cudnn as cudnn
from lib import slowfastnet
from Config import Config

class Test_video(Dataset):
    def __init__(self,short_side):
        self.short_side=short_side
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

    def crop(self, buffer, crop_size):
        # randomly select time index for temporal jittering
        # time_index = np.random.randint(buffer.shape[0] - clip_len)
        # Randomly select start indices in order to crop the video
        height_index = np.random.randint(buffer.shape[1] - crop_size)
        width_index = np.random.randint(buffer.shape[2] - crop_size)

        # crop and jitter the video using indexing. The spatial crop is performed on
        # the entire array, so each frame is cropped in the same location. The temporal
        # jitter takes place via the selection of consecutive frames
        buffer = buffer[:,
                 height_index:height_index + crop_size,
                 width_index:width_index + crop_size, :]

        return buffer

    def generate_video_clip(self,split_span,keep_num,fname="/home/aiuser/Desktop/_7oWZq_s_Sk.mkv"):
        capture = cv2.VideoCapture(fname)
        #获取视频的基本信息
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(capture.get(cv2.CAP_PROP_FPS))
        #计算要切多少段，每段切多少帧
        print(frame_count,frame_width)
        split_len=fps*split_span
        split_time=frame_count/split_len
        if frame_height < frame_width:
            resize_height = np.random.randint(self.short_side[0], self.short_side[1] + 1)
            resize_width = int(float(resize_height) / frame_height * frame_width)
        else:
            resize_width = np.random.randint(self.short_side[0], self.short_side[1] + 1)
            resize_height = int(float(resize_width) / frame_width * frame_height)
        start_idx = 0
        end_idx = start_idx + split_len
        skip_span = split_len // keep_num if end_idx // keep_num > 0 else 1
        rem = split_len - skip_span * keep_num if split_len - skip_span * keep_num >= 0 else 0
        while split_time>0: #切多少段
           split_time=split_time-1
           start_idx = start_idx + rem // 2
           buffer = []
           sample_count=0
           #处理每一段视频
           while (start_idx<end_idx):
               start_idx=start_idx+1
               retaining, frame = capture.read()
               if(sample_count>=keep_num):
                   continue
               if start_idx % skip_span != 0 and start_idx!=0:
                   continue
               if retaining is False:
                   break
               frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
               if (frame_height != resize_height) or (frame_width != resize_width):
                   frame = cv2.resize(frame, (resize_width, resize_height))
               buffer.append(frame)
               # if len(pa.isna(frame).nonzero()[1]) != 0 or np.max(frame) > 255:
               #     print("discard:", buffer)
               sample_count=sample_count+1
           print(np.shape(buffer))
           if len(buffer)<keep_num:
               for i in range(keep_num-len(buffer)):
                   buffer.append(buffer[-1])
                   print("warning appen -1")
           start_idx=end_idx
           end_idx=end_idx+split_len
           #一段处理完返回
           for v in buffer:
               cv2.imshow("video",v)
               cv2.waitKey(0)
           list_buffer = buffer


           buffer=np.array(buffer)

           buffer = self.crop(buffer, 196)
           buffer = self.normalize(buffer)
           buffer = self.to_tensor(buffer)
           buffer=torch.tensor(buffer, dtype=torch.float).unsqueeze(0)
           yield buffer,list_buffer
        capture.release()

def validation(model, val_dataloader):
    model.eval()
    all_prob=[]
    all_pre=[]
    data=val_dataloader.generate_video_clip(20,64)
    with torch.no_grad():
        for step,(inputs,frame_list) in enumerate(data):
            inputs = inputs.cuda()
            outputs = model(inputs)
            max = np.max(np.array(outputs.cpu()), axis=1)
            all_prob.extend(max)
            # for frame in frame_list:
            #     cv2.imshow("frame",frame)
            #     cv2.waitKey(0)
            print("show over,pro=",torch.nn.functional.softmax(outputs))
            for item in np.array(outputs.cpu()):
                all_pre.extend(np.where(item == max)[0])
                print(np.where(item == max)[0])
    print(all_pre)

def main():
    cudnn.benchmark = False
    test_video=Test_video(short_side=[224,256])
    model = slowfastnet.resnet50(class_num=Config.CLASS_NUM)
    assert Config.LOAD_MODEL_PATH is not None
    print("load model from:", Config.LOAD_MODEL_PATH)
    pretrained_dict = torch.load(Config.LOAD_MODEL_PATH, map_location='cpu')
    try:
        model_dict = model.module.state_dict()
    except AttributeError:
        model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model = model.cuda(params['gpu'][0])
    validation(model, test_video)


if __name__ == '__main__':
    main()

