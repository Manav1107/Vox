#importing packages

import tkinter
from tkinter import *
from PIL import ImageTk, Image
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

#main window
root = Tk()
root.title('Languify32')
root.config(bg = '#fafab6')
root.attributes('-fullscreen',True)

#Title
label = Label(root,text = 'LANGUIFY',bg='#fafab6')
label.configure(fg='black', font=('Bodoni Bd BT',30))
label.place(relx = 0.5, rely= 0.04, anchor=CENTER)

#Subtitle
label1 = Label(root,text = '"Language simplified"',bg='#fafab6')
label1.configure(fg='black', font=('Caladea',25))
label1.place(relx = 0.5, rely= 0.11, anchor=CENTER)

#Video
vdo = Label(root)
vdo.grid(padx = (0,50))
vdo.configure(highlightthickness=3, highlightbackground='black', highlightcolor='black')
vdo.place(relx=0.05, rely=0.5, anchor=W)
cap = cv2.VideoCapture(0)

detector = HandDetector(maxHands=1)
classifier = Classifier("keras_model.h5", "labels.txt")
labels = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]

#Images
img_label = Label(root)
img_label.grid(padx = (0,50))
img_label.configure(highlightthickness=3, highlightbackground='black', highlightcolor='black')
img_label.place(relx=0.55, rely=0.475, relheight=0.48, relwidth=0.4, anchor=W)

#line
line = Label(root)
lines = ImageTk.PhotoImage(Image.open('line.png'))
line.configure(image=lines, bg='#fafab6')
line.place(relx = 0.51, rely=0.19)

#Starting letter
letter = ImageTk.PhotoImage(Image.open("Sign Images/A.jpg"))
img_label.configure(image=letter)
img_label.image = letter

#Translator heading
translatel = Label(root, text="Translator", bg='#fafab6')
translatel.configure(fg='black', font=('Exotc350 Bd BT',30))
translatel.place(relx = 0.25, rely= 0.88 , anchor=CENTER)

#Letter heading
translatel = Label(root, text="Hand Signs", bg='#fafab6')
translatel.configure(fg='black', font=('Exotc350 Bd BT',30))
translatel.place(relx = 0.75, rely= 0.88 , anchor=CENTER)

#main code
def video_frame():
    global letter
    global index
    ret, frame = cap.read()

    hands, img = detector.findHands(frame)
    #hand detection
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgcrop = frame[y:y + h, x:x + w]

        #Prediction
        image = imgcrop.copy()
        prediction, index = classifier.getPrediction(image)
        cv2.putText(frame, labels[index], (x, y - 40), cv2.FONT_HERSHEY_COMPLEX, 2, (225, 0, 255), 2)

        if prediction:
            path = 'Sign Images/' + labels[index] + '.jpg'
            x = Image.open(path)
            xr = x.resize((400,350))
            letters = ImageTk.PhotoImage(xr)
            img_label.configure(image=letters)
            img_label.image = letters
    #else:
        #img_label.configure(image=letter)
        #img_label.image = letter

    if frame is not None:
        converted_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(converted_image)
        img_tk = ImageTk.PhotoImage(image = img)
        vdo.configure(image = img_tk)
        vdo.image = img_tk
        vdo.after(1,video_frame)

count = -1

def next():
    global count

    if count == 25:
        count = -1
    count+=1
    n_img = 'Sign Images/' + labels[count] + '.jpg'
    n = Image.open(n_img)
    nr = n.resize((400, 350))
    n_tk = ImageTk.PhotoImage(nr)
    img_label.configure(image= n_tk)
    img_label.image = n_tk

def previous():
    global count
    if count == 0:
        count = 26
    count -= 1
    p_img = 'Sign Images/' + labels[count] + '.jpg'
    p = Image.open(p_img)
    pr = p.resize((400, 350))
    p_tk = ImageTk.PhotoImage(pr)
    img_label.configure(image=p_tk)
    img_label.image = p_tk

#next button
nxt = ImageTk.PhotoImage(Image.open('next_btn.png'))
m_next = Button(root, image=nxt, bg='#fafab6', borderwidth=0, command=next)
m_next.place(relx = 0.8, rely = 0.76, anchor=W)

#previous button
prev = ImageTk.PhotoImage(Image.open('prev_btn.png'))
m_prev = Button(root, image=prev, bg='#fafab6', borderwidth=0,command=previous)
m_prev.place(relx = 0.62, rely = 0.76, anchor=W)

video_frame()
input('Press enter to quit')
root.mainloop()


