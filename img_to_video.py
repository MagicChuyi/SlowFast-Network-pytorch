import os
import cv2
img_root = '/home/aiuser/frames/'#这里写你的文件夹路径，比如：/home/youname/data/img/,注意最后一个文件夹要有斜杠
fps = 15   #保存视频的FPS，可以适当调整
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
videoWriter = cv2.VideoWriter('/home/aiuser/frames/saveVideo.avi',fourcc,fps,(656,480))
for i in range(121):
    if i>=10:
        frame = cv2.imread(img_root + str(i) + '.jpg')
        videoWriter.write(frame)
videoWriter.release()