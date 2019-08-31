from typing import Tuple
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset.base import Base as DatasetBase
from model import Model
import cv2
import numpy as np
class Evaluator(object):
    def __init__(self, dataset: DatasetBase, path_to_results_dir: str):
        super().__init__()
        self._dataset = dataset
        self._dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=11, pin_memory=True)
        self._path_to_results_dir = path_to_results_dir

    def evaluate(self, model: Model) -> Tuple[float, str]:
        #all_image_ids, all_detection_bboxes, all_detection_classes, all_detection_probs = [], [], [], []

        with torch.no_grad():
            for _, (image_posision, image_batch, scale, _, _,detector_bboxes,img_size) in enumerate(tqdm(self._dataloader)):
                image_batch = image_batch.cuda()
                assert image_batch.shape[0] == 1, 'do not use batch size more than 1 on evaluation'

                detection_bboxes, detection_classes, detection_probs = \
                    model.eval().forward(image_batch,detector_bboxes_batch=detector_bboxes)

                scale = scale[0]

                detection_bboxes[:,[0,2]] /= scale[0]
                detection_bboxes[:,[1,3]] /= scale[1]

                #warning:may be wrong
                # if len(image_posision) < len(detection_bboxes):
                #     for i in range(len(detection_bboxes)-len(image_posision)):
                #         image_posision.append(image_posision[0])
                if not len(detection_bboxes.tolist())==len(detection_classes)==len(detection_probs):
                    print("%%%%",(np.round(detection_bboxes.tolist(),2)))
                    print(image_posision)
                    print(detection_classes)
                    print(detection_probs)
                    print(detector_bboxes)
                #千万小心确认这里
                assert len(detection_bboxes.tolist()) == len(detection_classes) == len(detection_probs)
                #all_detection_bboxes.append(detection_bboxes.tolist())
                #print("wait_write:",detection_bboxes.tolist())
                #all_detection_classes.append(detection_classes)
                #all_detection_probs.append(detection_probs)
                assert len(image_posision)==1
                #all_image_ids.append(image_posision[0])
                #for debug
                #self.imshow(detection_bboxes.tolist(),detection_classes,detection_probs)
                self._write_results(self._path_to_results_dir,[image_posision[0]], [detection_bboxes.tolist()], [detection_classes], [detection_probs],img_size)
        mean_ap, detail = self._dataset.evaluate(self._path_to_results_dir,all_image_ids, all_detection_bboxes, all_detection_classes, all_detection_probs,img_size)
        return mean_ap, detail


    def _write_results(self, path_to_results_dir, image_ids, bboxes,
                       classes, probs,img_size):
        f = open(path_to_results_dir,mode='a+')
        f1=open(path_to_results_dir+'1',mode='a+')
        f2 = open(path_to_results_dir + '2', mode='a+')
        # print(len(image_ids),len(bboxes),len(classes),len(probs))
        #assert len(image_ids)==len(bboxes)==len(classes)==len(probs)
        for image_id, _bbox, _cls, _prob in zip(image_ids, bboxes, classes, probs):
            # print("image_id:", image_id)
            # print("bbox:", _bbox)
            # print("cls:", _cls)
            # print("prob:", _prob)
            # print("info:", len(_bbox), len(_cls), len(_prob))
            assert len(_bbox) == len(_cls) == len(_prob)
            for bbox, cls, prob in zip(_bbox, _cls, _prob):
            #print(str(image_id.split('/')[0]),str(image_id.split('/')[1]), bbox[0]/int(img_size[1]), bbox[1], bbox[2], bbox[3],(int(cls)+1),prob,img_size[1],int(img_size[0]))
                x1=0 if bbox[0]/int(img_size[1])<0 else bbox[0]/int(img_size[1])
                y1=0 if bbox[1]/int(img_size[0])<0 else bbox[1]/int(img_size[0])
                x2=1 if bbox[2]/int(img_size[1])>1 else bbox[2]/int(img_size[1])
                y2=1 if bbox[3]/int(img_size[0])>1 else bbox[3]/int(img_size[0])
                # if x1>np.round(x1,3) or y1>np.round(y1,3) or x2>np.round(x2,3) or y2>np.round(y2,3):
                #     print(str(image_id.split('/')[0]), str(image_id.split('/')[1]), x1, y1, x2, y2)
                #     assert False
                for c,p in zip(cls,prob):

                    f.write('{:s},{:s},{:f},{:f},{:f},{:f},{:s},{:s}\n'.format(str(image_id.split('/')[0]),str(image_id.split('/')[1]), x1, y1, x2, y2,str(c),str(p)))
                    if p>0.1:
                         f1.write('{:s},{:s},{:f},{:f},{:f},{:f},{:s},{:s}\n'.format(str(image_id.split('/')[0]),str(image_id.split('/')[1]), x1, y1, x2, y2,str(c),str(p)))
                    if p>0.2:
                         f2.write('{:s},{:s},{:f},{:f},{:f},{:f},{:s},{:s}\n'.format(str(image_id.split('/')[0]),str(image_id.split('/')[1]), x1, y1, x2, y2,str(c),str(p)))
        f.close()

    def index2class(self):
        file_path = '/media/aiuser/78C2F86DC2F830CC1/ava_v2.2/ava_v2.2/ava_action_list_v2.0.csv'
        with open(file_path) as f:
            i2c_dic = {line.split(',')[0]: line.split(',')[1] for line in f}
        return i2c_dic

    def draw_bboxes_and_show(self,frame,bboxes,labels,probs=[],color=(255, 255, 255)):
            i2c_dic=self.index2class()
            if len(probs)==0:
                for bbox, lable in zip(bboxes, labels):
                    bbox = np.array(bbox)
                    real_x_min = int(bbox[0])
                    real_y_min = int(bbox[1])
                    real_x_max = int(bbox[2])
                    real_y_max = int(bbox[3])
                    # 在每一帧上画矩形，frame帧,(四个坐标参数),（颜色）,宽度
                    cv2.rectangle(frame, (real_x_min, real_y_min), (real_x_max, real_y_max), color, 4)
                    count=0
                    for l in lable:
                        cv2.putText(frame, i2c_dic[str(l)], (real_x_min + 30, real_y_min + 15 * count),
                                    cv2.FONT_HERSHEY_COMPLEX, \
                                    0.5, (255, 255, 0), 1, False)
            else:
                for bbox,lable,prob in zip(bboxes,labels,probs):
                    bbox=np.array(bbox)
                    #print("probs",prob)
                    real_x_min = int(bbox[0])
                    real_y_min = int(bbox[1])
                    real_x_max = int(bbox[2])
                    real_y_max = int(bbox[3])
                    # 在每一帧上画矩形，frame帧,(四个坐标参数),（颜色）,宽度
                    count_2=0
                    for l,p in zip(lable,prob):
                        cv2.rectangle(frame, (real_x_min, real_y_min), (real_x_max, real_y_max), (255, 255, 255), 4)
                        cv2.putText(frame, i2c_dic[str(l)]+':'+str(round(float(p),2)), (real_x_min+30 , real_y_min+15*count_2 ), cv2.FONT_HERSHEY_COMPLEX,\
                        0.5,(255, 255, 0), 1, False)
                        count_2+=1

    def imshow(self,bbox,cls,probs):
        #print("bbox ",bbox)
        cap = cv2.VideoCapture("/media/aiuser/78C2F86DC2F830CC1/ava_v2.2/preproc/train_clips/clips/b5pRYl_djbs/986.mp4")
        # frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # key_frame_start = int(frame_count * 0.3)
        # key_frame_end = int(frame_count * 0.9)
        while (cap.isOpened()):
            ret, frame = cap.read()
            self.draw_bboxes_and_show(frame, bbox, cls,probs=probs)
            # self.draw_bboxes_and_show(frame,frame_num, real_bboxes, real_lables, key_frame_start, key_frame_end,color=(255,0,255))
            if ret == True:
                # 显示视频
                cv2.imshow('Frame', frame)
                # 刷新视频
                cv2.waitKey(0)
                # 按q退出
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else:
                break