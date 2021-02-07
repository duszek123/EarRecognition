import numpy as np

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import tkinter
import cv2
import PIL
from PIL import Image
from PIL import ImageTk
import os

import program_param as pp
import data_transformation as dt




class WindowApp():
    def __init__(self):
        self.window = tkinter.Tk()
        self.width = pp.video_stream.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = pp.video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.canvas = tkinter.Canvas(self.window, width=1920, height=1080)
        self.var = tkinter.StringVar()

        self.var.trace("w", self.__callback)
        self.__act_video_source()
        button = tkinter.Button(self.window, text="Rozpocznij/Zatrzymaj", command=self.__buttoncallback)
        button.place(x=1590, y=100)

        self.canvas.pack()

    def draw_result(self,frame,frame_raw,result,score,probabilities):
        # draw rectagle [SIZE_PICTxSIZE_PICT] int center frame
        a = (int((640 / 2) - (pp.size_pict + 20) / 2), int((480 / 2) - (pp.size_pict + 20) / 2))
        b = (int((640 / 2) + (pp.size_pict + 20) / 2), int((480 / 2) + (pp.size_pict + 20) / 2))
        cv2.rectangle(frame, a, b, (255, 0, 0), 2)

        cv2.putText(frame, '%s' % (result), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        cv2.putText(frame, '(score = %0.5f)' % (float(score)), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                    2)

        all_frame = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
        self.canvas.create_image(0, 0, image=all_frame, anchor=tkinter.NW)

        ear_frame = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame_raw))
        self.canvas.create_image(660, (640 - pp.input_size[0]) / 2, image=ear_frame, anchor=tkinter.W)

        self.canvas.create_text(760 + pp.input_size[0], 60, font=("Purisa", 22), text="Rozkład prawdopodobieństw: ",
                           anchor=tkinter.W)
        for i in range(0, len(dt.ear_class)):
            procet = int(float(probabilities[i]) * 100.0)
            text = "  " + str(dt.ear_class[i]) + ": " + str(probabilities[i]) + "  -> " + str(procet) + "%  "
            label = tkinter.Label(text=text, anchor=tkinter.W, font=("Purisa", 18))
            self.canvas.create_window(980 + pp.input_size[0], 110 + i * 30, window=label)

        if max(probabilities) > 0.50:
            pred_person = str(dt.ear_class[np.argmax(probabilities)])
            text = "  Rozpoznana osoba -> " + pred_person
            label = tkinter.Label(text=text, anchor=tkinter.CENTER, font=("Purisa", 22))
            win = self.canvas.create_window(1000, 350 + len(dt.ear_class) * 30, window=label)

            pilImage = Image.open(pp.data_dir + '/Train/' + pred_person + '/1_1.jpg')
            image = ImageTk.PhotoImage(pilImage)
            self.canvas.create_image(1050, 500 + len(dt.ear_class) * 30, image=image)
        else:
            text = "       Rozpoznana osoba ->             "
            label = tkinter.Label(text=text, anchor=tkinter.CENTER, font=("Purisa", 22))
            win = self.canvas.create_window(1000, 350 + len(dt.ear_class) * 30, window=label)
            pred_person = "0"

        self.window.update()

    def draw_window(self,frame,result,score,probabilities):
        # draw rectagle [SIZE_PICTxSIZE_PICT] int center frame
        a = (int((640 / 2) - (pp.size_pict + 20) / 2), int((480 / 2) - (pp.size_pict + 20) / 2))
        b = (int((640 / 2) + (pp.size_pict + 20) / 2), int((480 / 2) + (pp.size_pict + 20) / 2))
        cv2.rectangle(frame, a, b, (255, 0, 0), 2)

        cv2.putText(frame, '%s' % (result), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        cv2.putText(frame, '(score = %0.5f)' % (float(score)), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                    2)

        image_raw, image = dt.preprocess(frame)

        all_frame = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
        self.canvas.create_image(0, 0, image=all_frame, anchor=tkinter.NW)

        ear_frame = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(image_raw))
        self.canvas.create_image(660, (640 - pp.input_size[0]) / 2, image=ear_frame, anchor=tkinter.W)

        pred_person = str(dt.ear_class[np.argmax(probabilities)])

        if pred_person != "0":
            try:
                pilImage = Image.open(pp.data_dir + '/Train/' + pred_person + '/1_1.jpg')
                image = ImageTk.PhotoImage(pilImage)
                self.canvas.create_image(1050, 500 + len(dt.ear_class) * 30, image=image)
            except:
                pass

        self.window.update()

    def __callback(self,*args):
        pass
        # cv2.VideoCapture(pp.camera_source).release()
        # pp.camera_source = '/dev/' + str(self.var.get())
        # try:
        #     vid = cv2.VideoCapture(pp.camera_source)
        #     if not vid.isOpened():
        #         raise ValueError("Unable to open video source", pp.camera_source)
        # except:
        #     print("Unable to open video source " + str(pp.camera_source))

    def __buttoncallback(self,*args):
        pp.flag_start = not (pp.flag_start)
        print("Program on -> " + str(pp.flag_start))

    def __act_video_source(self):
        available_stream = []
        all_stream = os.listdir('/dev/')
        for item in all_stream:
            if item.find('video') != -1:
                available_stream.append(item)
        available_stream = sorted(available_stream)
        print('wszystkie streamy ' + str(available_stream))

        label = tkinter.Label(text="Źródło obrazu -> ")

        self.var.set(available_stream[0])
        stream_box = tkinter.OptionMenu(self.window, self.var, *available_stream)
        label.place(x=1587, y=65)
        stream_box.place(x=1700, y=60)


