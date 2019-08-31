import cv2
import os
import numpy as np
from bbox import BBox
import torch
class imshow_result():
    class info():
        def __init__(self, img_class,prob, bbox,img_h, img_w,img_position):
            self.img_class = int(img_class)-1
            self.prob=prob
            self.bbox = bbox
            self.img_position=img_position
            self.height = img_h
            self.weight = img_w
        def __repr__(self):
            return 'info[img_class={0}, bbox={1}]'.format(
                self.img_class, self.bbox)
    def __init__(self):
        self.i2c_dic=self.index2class()

        self.bboxes = []
        self.labels = []
        self.probs=[]
        self.image_ratios = []
        self.image_position = []
        self.widths = []
        self.heights = []

        self.data_dic = {}
        self.data_dic_real = {}

        self.data_size = {}
        self.data_format = {}
        self.path_to_data_dir = '/home/aiuser/'
        path_to_AVA_dir = os.path.join(self.path_to_data_dir, 'ava_v2.2', 'preproc', 'train_clips')
        self.path_to_videos = os.path.join(path_to_AVA_dir, 'clips')
        self.path_to_keyframe = os.path.join(path_to_AVA_dir, 'keyframes')
        #path_to_video_ids_txt = os.path.join(path_to_AVA_dir, 'trainval.txt')
        path_to_video_ids_txt = '/home/aiuser/ava_v2.2/result.txt'
        path_to_real_ids_txt = '/home/aiuser/ava_v2.2/preproc/train_clips/trainval.txt'
        # 得到每个视频的大小，通过读取第一张keyframe
        for frame in sorted(os.listdir(self.path_to_keyframe)):
            img = os.listdir(os.path.join(self.path_to_keyframe, frame))[0]
            img = cv2.imread(os.path.join(self.path_to_keyframe, frame, img))
            img_shape = img.shape
            self.data_size[frame] = (img_shape[0], img_shape[1])
        # 得到每个视频的格式
        for video in sorted(os.listdir(self.path_to_videos)):
            video_0 = os.listdir(os.path.join(self.path_to_videos, video))[0]
            self.data_format[video] = '.' + video_0.split('.')[1]
        # 读取文件，key是文件名(aa/0930)
        with open(path_to_video_ids_txt, 'r') as f:
            data = f.readlines()
            for line in data:
                content = line.split(',')
                key = content[0] + "/" + str(int(content[1]))
                img_h = int(self.data_size[content[0]][0])
                img_w = int(self.data_size[content[0]][1])
                if key not in self.data_dic:
                    self.data_dic[key] = [imshow_result.info(content[6],content[7].replace("\n", ""), BBox(  # convert to 0-based pixel index
                        left=float(content[2]) * img_w - 1,
                        top=float(content[3]) * img_h - 1,
                        right=float(content[4]) * img_w - 1,
                        bottom=float(content[5]) * img_h - 1), img_h, img_w, key)]
                else:
                    self.data_dic[key].append(imshow_result.info(content[6],content[7].replace("\n", ""), BBox(  # convert to 0-based pixel index
                        left=float(content[2]) * img_w - 1,
                        top=float(content[3]) * img_h - 1,
                        right=float(content[4]) * img_w - 1,
                        bottom=float(content[5]) * img_h - 1), img_h, img_w, key))
        with open(path_to_real_ids_txt, 'r') as f:
            data = f.readlines()
            for line in data:
                content = line.split(',')
                key = content[0] + "/" + str(int(content[1]))
                img_h = int(self.data_size[content[0]][0])
                img_w = int(self.data_size[content[0]][1])
                if key not in self.data_dic_real:
                    self.data_dic_real[key] = [imshow_result.info(content[6], content[7].replace("\n", ""),
                                                                BBox(  # convert to 0-based pixel index
                                                                left=float(content[2]) * img_w - 1,
                                                                top=float(content[3]) * img_h - 1,
                                                                right=float(content[4]) * img_w - 1,
                                                                bottom=float(content[5]) * img_h - 1), img_h,img_w, key)]
                else:
                    self.data_dic_real[key].append(imshow_result.info(content[6], content[7].replace("\n", ""),
                                                                BBox(  # convert to 0-based pixel index
                                                                    left=float(content[2]) * img_w - 1,
                                                                    top=float(content[3]) * img_h - 1,
                                                                    right=float(content[4]) * img_w - 1,
                                                                    bottom=float(content[5]) * img_h - 1), img_h,img_w, key))
            # print('data_dic:',self.data_dic)
        # 对字典中的数据进行整理，变成list的形式
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
    def __getitem__(self, index: int):
        buffer, scale, index = self.loadvideo(self.image_position, index, 180, 280, 1)
        bboxes = torch.tensor(self.bboxes[index], dtype=torch.float)
        print(self.labels[index],self.probs[index])
        labels = torch.tensor(self.labels[index], dtype=torch.long)
        probs = [float(item) for item in self.probs[index]]
        #image = Image.open(self.path_to_keyframe+'/'+self.image_position[index]+".jpg")
        bboxes *= scale
        return self.image_position[index], buffer, scale, bboxes, labels,probs,(self.heights[index],self.widths[index])
    def __len__(self) -> int:
        return len(self.image_position)
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
    def index2class(self):
        file_path = '/home/aiuser/ava_v2.2/ava_v2.2/ava_action_list_v2.0.csv'
        with open(file_path) as f:
            i2c_dic = {line.split(',')[0]: line.split(',')[1] for line in f}
        print(i2c_dic)
        return i2c_dic


    def imshow_result(self,item_num):
        i2c_dic=self.i2c_dic
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
            key_frame_start=int(frame_count*0.35)
            key_frame_end =int(frame_count*0.95)
            frame_num=0
            while (cap.isOpened()):
                ret, frame = cap.read()
                frame_num=frame_num+1
                if frame_num>key_frame_start and frame_num<key_frame_end:
                    count = 0
                    count_2=0
                    for bbox, lable in zip(real_bboxes, real_lables):
                        count = count + 1
                        bbox = np.array(bbox)
                        lable = int(lable)
                        real_x_min = int(bbox[0])
                        real_y_min = int(bbox[1])
                        real_x_max = int(bbox[2])
                        real_y_max = int(bbox[3])
                        # 在每一帧上画矩形，frame帧,(四个坐标参数),（颜色）,宽度
                        cv2.rectangle(frame, (real_x_min, real_y_min), (real_x_max, real_y_max), (255, 0, 255), 4)
                        cv2.putText(frame, i2c_dic[str(lable + 1)], (real_x_min + 30, real_y_min + 15 * count),
                                    cv2.FONT_HERSHEY_COMPLEX, \
                                    0.5, (255, 0, 255), 1, False)

                    for bbox,lable,prob in zip(bboxes,labels,probs):
                        count_2 = count_2 + 1
                        bbox=np.array(bbox)
                        lable = int(lable)
                        prob=float(prob)
                        real_x_min = int(bbox[0]/scale)
                        real_y_min = int(bbox[1]/scale)
                        real_x_max = int(bbox[2]/scale)
                        real_y_max = int(bbox[3]/scale)
                        # 在每一帧上画矩形，frame帧,(四个坐标参数),（颜色）,宽度
                        cv2.rectangle(frame, (real_x_min, real_y_min), (real_x_max, real_y_max), (255, 255, 255), 4)
                        cv2.putText(frame, i2c_dic[str(lable+1)]+':'+str(prob), (real_x_min+30 , real_y_min+15*count_2 ), cv2.FONT_HERSHEY_COMPLEX,\
                        0.5,(255, 255, 0), 1, False)
                if ret == True:
                    # 显示视频
                    cv2.imshow('Frame', frame)
                    # 刷新视频
                    cv2.waitKey(25)
                    # 按q退出
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break
                else:
                    break


if __name__ == '__main__':
    ir=imshow_result()
    ir.imshow_result(20)