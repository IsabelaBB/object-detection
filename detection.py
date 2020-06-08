#################################
# Created by IB. Barcelos
# June, 2020
##################################

import PIL
from PIL import Image, ImageTk
import tkinter as tk
import argparse
import datetime
import numpy as np
import cv2
import os

#########################################################################
#     Filtering
#########################################################################
def filter(frame, low_H, low_S, low_V, high_H, high_S, high_V, inverse):
  
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # converte para o canal HSV
  img = cv2.GaussianBlur(frame, (7, 7), 0.2) # aplica um filtro gaussiano para diminuir o ruído
  img = img & 0b11110000 # remove a informação de cor dos 3 bits menos significativos
  
  mask = cv2.inRange(img, (low_H, low_S, low_V), (high_H, high_S, high_V)) # filtra a imagem no intervalo especificado
  
  if(inverse == 1):
    mask = 255 - mask
  
  kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(3,3)) # define o elemento estruturante para a filtragem morfológica
  mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # filtra com fechamento, unindo os elementos muito próximos
  result = cv2.bitwise_and(frame, frame, mask=mask) # faz operação AND da imagem original com a filtrada para alterar a visualização

  img = cv2.cvtColor(result, cv2.COLOR_HSV2BGR) # volta para o canal BGR
  return img


#########################################################################
#     Detection
#########################################################################
def detect(original_img, image, max_objects=3, min_size=10):

  # se a imagem for colorida, converte para escala de cinza
  if(len(image.shape) > 2):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  
  contours, __ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # encontra os componentes conexos da imagem

  if len(contours) > 0:
    count = 0
    # para cada componente, em ordem decrescente de seu tamanho em pixels
    for cnt in sorted(contours, key = cv2.contourArea, reverse = True):
      
      # só pega os componentes com pelo menos o tamanho mínimo e numa quantidade máxima
      if cv2.contourArea(cnt) >= min_size and count < max_objects:
        rect = np.int32(cv2.boxPoints(cv2.minAreaRect(cnt)))
      
        rect = np.int0(cv2.boxPoints(cv2.minAreaRect(cnt))) # define o menor retângulo circunscrito no componente atual
        cv2.drawContours(original_img,[rect],0,(255,255,0),2) # desenha o retângulo, inficando que detectou um objeto
        count = count+1 # contagem de objetos para controlar quantos são detectados
      else:
        break;      

  return original_img


#########################################################################
#     Aplication class
#########################################################################
class Application:
  def __init__(self, camera=0):
    """ Initialize application which uses OpenCV + Tkinter. It displays
      a video stream in a Tkinter window and stores current snapshot on disk """
    self.vs = cv2.VideoCapture(camera) # capture video frames, 0 is your default video camera
    self.frame = None  # current image from the camera
 
    self.root = tk.Tk()  # initialize root window
    self.root.title("Color detect")  # set window title
    self.root.protocol('WM_DELETE_WINDOW', self.destructor)
    self.root.config(cursor="arrow")
    
    # read a single frame to get its size
    ok, self.frame = self.vs.read()


    ###########################################################################
    #### define 3 frames

    # frame with the captures
    self.frame_captures = tk.Frame(master=self.root, relief=tk.RAISED, borderwidth=1)
    self.frame_captures.grid(row=0, column=0, columnspan=20, sticky='n', padx=5, pady=5)

    # frame only with the colors scales
    self.frame_colorsScales = tk.Frame(master=self.root, relief=tk.RAISED, borderwidth=1)
    self.frame_colorsScales.grid(row=1, column=0, columnspan=10, sticky='n', padx=5, pady=5)

    # frame only with the conf.
    self.frame_conf = tk.Frame(master=self.root, relief=tk.RAISED, borderwidth=1)
    self.frame_conf.grid(row=1, column=10, columnspan=10, sticky='n', padx=5, pady=5)
    ###########################################################################


    ###########################################################################
    # define the panels (with the captures)
    label1 = tk.Label(master=self.frame_captures, text= 'Capture and detection', font=("Times Bold", 10))
    label1.grid(row=0, column=0, columnspan=10, padx=5)

    self.panel = tk.Label(master=self.frame_captures)  # initialize image panel
    self.panel.grid(row=1, column=0, columnspan=10, padx=5, pady=5)
    
    self.panel.bind("<Button-1>", self.leftclick)

    label2 = tk.Label(master=self.frame_captures, text= 'Filtering', font=("Times Bold", 10))
    label2.grid(row=0, column=10, columnspan=10, padx=5)

    self.panelB = tk.Label(master=self.frame_captures)  # initialize image panel
    self.panelB.grid(row=1, column=10, columnspan=10, padx=5, pady=5)
    ###########################################################################


    ###########################################################################
    # define the color scales (as barras de rolagem horizontais)
    label3 = tk.Label(master=self.frame_colorsScales, text= 'Color adjust', font=("Times Bold", 10))
    label3.grid(row=0, column=0, columnspan=10, pady=5)

    self.low_hue = tk.Scale(self.frame_colorsScales, label='Low Hue',from_=0, to=255, showvalue=1, orient=tk.HORIZONTAL, command=self.change_low_h, length=self.frame.shape[1]/2)
    self.low_hue.grid(row=1, column=0, columnspan=5)

    self.high_hue = tk.Scale(self.frame_colorsScales, label='High Hue', from_=0, to=255, showvalue=1, orient=tk.HORIZONTAL, command=self.change_high_h, length=self.frame.shape[1]/2)
    self.high_hue.grid(row=1, column=5, columnspan=5)

    self.low_sat = tk.Scale(self.frame_colorsScales, label='Low Saturation',from_=0, to=255, showvalue=1, orient=tk.HORIZONTAL, command=self.change_low_s, length=self.frame.shape[1]/2)
    self.low_sat.grid(row=2, column=0, columnspan=5)

    self.high_sat = tk.Scale(self.frame_colorsScales, label='High Saturation', from_=0, to=255, showvalue=1, orient=tk.HORIZONTAL, command=self.change_high_s, length=self.frame.shape[1]/2)
    self.high_sat.grid(row=2, column=5, columnspan=5)

    self.low_val = tk.Scale(self.frame_colorsScales, label='Low Value',from_=0, to=255, showvalue=1, orient=tk.HORIZONTAL, command=self.change_low_v, length=self.frame.shape[1]/2)
    self.low_val.grid(row=3, column=0, columnspan=5)

    self.high_val = tk.Scale(self.frame_colorsScales, label='High Value', from_=0, to=255, showvalue=1, orient=tk.HORIZONTAL, command=self.change_high_v, length=self.frame.shape[1]/2)
    self.high_val.grid(row=3, column=5, columnspan=5)
    ###########################################################################

    
    ###########################################################################
    ### define the conf. scales
    label4 = tk.Label(master=self.frame_conf, text= 'Configuration', font=("Times Bold", 10))
    label4.grid(row=0, column=0, columnspan=10, pady=5)

    self.maxObj = tk.Scale(self.frame_conf, label='Max Objects',from_=0, to=255, showvalue=1, orient=tk.HORIZONTAL, command=self.change_maxObj, length=self.frame.shape[1]/2)
    self.maxObj.grid(row=1, column=0, columnspan=10)

    self.minSize = tk.Scale(self.frame_conf, label='Min size', from_=1, to=10000, showvalue=1, orient=tk.HORIZONTAL, command=self.change_minSize, length=self.frame.shape[1]/2)
    self.minSize.grid(row=2, column=0, columnspan=10)

    self.inverse = tk.IntVar()
    self.inverseCheck = tk.Checkbutton(self.frame_conf, text='Inverse filter', onvalue = 1, offvalue = 0, variable=self.inverse)
    self.inverseCheck.grid(row=3, column=0, columnspan=10, pady=5)
    ###########################################################################
    
    
    ###########################################################################
    ### set default values
    self.low_hue.set(64)
    self.low_sat.set(64)
    self.low_val.set(64)

    self.high_hue.set(196)
    self.high_sat.set(196)
    self.high_val.set(196)

    self.minSize.set(50)
    ###########################################################################

    self.video_loop() # vai para o loop da aplicação
   

  #########################
  #     Events
  #########################
  # permite definir a cor central da faixa requerida apenas com um click na tela
  def leftclick(self, event):

    # get hsv color point
    point = self.frame[event.y,event.x]
    point = np.uint8([[[point[0],point[1],point[2] ]]])
    point = cv2.cvtColor(point, cv2.COLOR_BGR2HSV)[0][0]

    # get diff values (tentando manter a janela definida)
    diff_h = self.high_hue.get() - self.low_hue.get()
    diff_s = self.high_sat.get() - self.low_sat.get()
    diff_v = self.high_val.get() - self.low_val.get()
    
    # set new values
    self.low_hue.set(max(0,int(point[0]-(diff_h/2)-0.5)))
    self.high_hue.set(min(180,int(point[0]+(diff_h/2)+0.5)))
    self.low_sat.set(max(0,int(point[1]-(diff_s/2)-0.5)))
    self.high_sat.set(min(255,int(point[1]+(diff_s/2)+0.5)))
    self.low_val.set(max(0,int(point[2]-(diff_v/2)-0.5)))
    self.high_val.set(min(255,int(point[2]+(diff_v/2)+0.5)))


  def change_low_h(self, val):
    val = int(val)
    if(val <= self.high_hue.get()):
      self.low_hue.set(val)
    else:
      self.low_hue.set(self.high_hue.get())
    
  def change_low_s(self, val):
    val = int(val)
    if(val <= self.high_sat.get()):
      self.low_sat.set(val)
    else:
      self.low_sat.set(self.high_sat.get())

  def change_low_v(self, val):
    val = int(val)
    if(val <= self.high_val.get()):
      self.low_val.set(val)
    else:
      self.low_val.set(self.high_val.get())

  def change_high_h(self, val):
    val = int(val)
    if(val >= self.low_hue.get()):
      self.high_hue.set(val)
    else:
      self.high_hue.set(self.low_hue.get())

  def change_high_s(self, val):
    val = int(val)
    if(val >= self.low_sat.get()):
      self.high_sat.set(val)
    else:
      self.high_sat.set(self.low_sat.get())

  def change_high_v(self, val):
    val = int(val)
    if(val >= self.low_val.get()):
      self.high_val.set(val)
    else:
      self.high_val.set(self.low_val.get())

  def change_maxObj(self, val):
    val = int(val)
    self.maxObj.set(val)

  def change_minSize(self, val):
    val = int(val)
    self.minSize.set(val)


  #########################
  #     Aplication loop
  #########################
  def video_loop(self):
    """ Get frame from the video stream and show it in Tkinter """
    ok, self.frame = self.vs.read()  # read frame from video stream
    
    if ok:  # frame captured without any errors
      key = cv2.waitKey(100)
      
      frame_threshold = filter(self.frame, self.low_hue.get(), self.low_sat.get(), self.low_val.get(), self.high_hue.get(), self.high_sat.get(), self.high_val.get(), self.inverse.get())
      
      self.frame = detect(self.frame, frame_threshold, max_objects=self.maxObj.get(), min_size=self.minSize.get())

      # necessário para converter as imagens do opencv para o formato aceito pelo tkinter
      cv2image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGBA)  # convert colors from BGR to RGBA
      self.current_image = Image.fromarray(cv2image)  # convert image for PIL
      imgtk = ImageTk.PhotoImage(image=self.current_image)  # convert image for tkinter 
      self.panel.imgtk = imgtk  # anchor imgtk so it does not be deleted by garbage-collector  
      self.panel.config(image=imgtk)  # show the image

      cv2image_thresh = cv2.cvtColor(frame_threshold, cv2.COLOR_BGR2RGBA)  # convert colors from BGR to RGBA
      self.current_image_thresh = Image.fromarray(cv2image_thresh)  # convert image for PIL
      imgtk_thresh = ImageTk.PhotoImage(image=self.current_image_thresh)  # convert image for tkinter 
      self.panelB.imgtk = imgtk_thresh  # anchor imgtk so it does not be deleted by garbage-collector  
      self.panelB.config(image=imgtk_thresh)  # show the image

      #print(self.inverse.get())

    self.root.after(1, self.video_loop)  # call the same function after 30 milliseconds
 
 
  def destructor(self):
    """ Destroy the root object and release all resources """
    print("[INFO] closing...")
    self.root.destroy()
    self.vs.release()  # release web camera
    cv2.destroyAllWindows()  # it is not mandatory in this application
 



parser = argparse.ArgumentParser(description='Code for Thresholding Operations using inRange tutorial.')
parser.add_argument('--camera', help='Camera devide number.', default=0, type=int)
args = parser.parse_args()

# start the app
print("[INFO] starting...")
pba = Application(args.camera)
pba.root.mainloop()



