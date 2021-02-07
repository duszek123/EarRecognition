import torch
import cv2

#data dir with train i validation picture
data_dir = '/home/pawel/Pulpit/picture_data'
#source video stream
camera_source = '/dev/video2'
#flag, false, not used 
save = False
#input picture size (px)
input_size = (224,224)
size_pict = input_size[0]
#part of the data from the database intended for training 
batch_size = 8
#numb of process core
num_workers = 4
#numb of train epoch
epoch_num = 2
#old variable not use
frame_iterator = 0
#flag, not use
flag_start = False

#use device in project - cpu or gpu(cuda)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#using video stream in project
video_stream = vid = cv2.VideoCapture(camera_source)
if not video_stream.isOpened():
    raise ValueError("Unable to open video source", camera_source)


