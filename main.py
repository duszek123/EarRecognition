import torch
import cv2

#import key program param
import program_param as pp
#module with class for transformation image
import data_transformation
#module with class for make or load cnn
import cnn_functions
#module depends on cerate GUI (tkinter)
import tkinter_window

#load avaible devices (cpu/gpu)
device = pp.device

print("--------------------------------")
print('Size pict = '+str(pp.size_pict))
print("--------------------------------")
print("liczba klas -> "+ str(data_transformation.num_of_class))
print("--------------------------------")
print("class -> "+str(data_transformation.ear_class))
print("--------------------------------")
print("urządzenie -> "+ str(device))
print("--------------------------------")

#create object with choose cnn
my_cnn = cnn_functions.CNN(cnn_type="load")

print('Liczba cech -> '+str(my_cnn.get_feature_num()))
#get model
model = my_cnn.get_model()
#put model to avaible device
model.to(device)

#window.attributes('-fullscreen', True) #uncomment to run program in full size window
app_window = tkinter_window.WindowApp()

#create necessary variables
result = 0
score = 0
probabilities = 0
pred_person = "0"
frame_iterator = 0


while True:
    #load one freame from chosen video source; ret is true when frame avaible
    ret, frame = pp.video_stream.read()
    if ret:
        #analyze every 10th frame
        if frame_iterator >= 10 and pp.flag_start:
            frame_iterator = 0
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #transformation frame to a form acceptable to cnn
            image_raw, image = data_transformation.preprocess(frame)
            #put image to cnn
            output = model(image)
            #get result from output network
            result, score = data_transformation.argmax(output)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            probabilities = probabilities.cpu().detach().data.numpy().squeeze()
            #make window with result
            app_window.draw_result(frame,image_raw,result,score,probabilities)

        else:
            frame_iterator += 1
            if frame_iterator > 10:
                frame_iterator = 0
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            app_window.draw_window(frame,result,score,probabilities)
            #w(self,frame,result,score,probabilities):


