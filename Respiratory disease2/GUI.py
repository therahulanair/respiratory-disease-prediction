from tkinter import *
import time
import re
#Import scikit-learn metrics module for accuracy calculation
import pickle
from PIL import Image, ImageTk  
import cv2
from tkinter.filedialog import askopenfile
from tensorflow.keras.models import model_from_json
import librosa
import numpy as np
json_file = open('BiGRUmodel.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("BiGRUmodel.h5")
print("Loaded model from disk")

def pp(a):
    global mylist
    mylist.insert(END, a)



def predict(val):
    global mylist
    print(val)
    data_x, sampling_rate = librosa.load(val,res_type='kaiser_fast')
    mfccs = np.mean(librosa.feature.mfcc(y=data_x, sr=sampling_rate, n_mfcc=40).T,axis=0)
    
    feat=np.array(mfccs)
    
    feat=np.reshape(feat,(1,40,1))

    dlist=['COPD','Healthy','Pneumonia','URTI']

    ypred=model.predict(feat)
    print("prediction-->",ypred)
    res=np.argmax(ypred,axis=1)[0]
    print("type-->",res)
    result=dlist[res]
    
    root.after(500, lambda : pp("Uploading  Audio"))
    root.after(1700, lambda : pp("Loading Bidirectional GRU Model"))
    root.after(2000, lambda : pp("MFCC Feature extraction"))
    root.after(2500, lambda : pp("Prediction using GRU model"))
    root.after(2800, lambda : pp("Result : "+result))
    root.after(3000, lambda : pp("============================"))
    root.after(3100, lambda :shrslt.config(text=result,fg="red"))
    
        
    
        
def browseim():
    global cimg,shrslt,E1
    path = askopenfile()
    n=path.name 
    print(n)
    E1.delete(0,"end")
    E1.insert(0, n)
    
def userHome():
    global root, mylist,shrslt,E1
    root = Tk()
    root.geometry("1200x700+0+0")
    root.title("Home Page")

    image = Image.open("lung.jpg")
    image = image.resize((1200, 700), Image.ANTIALIAS) 
    pic = ImageTk.PhotoImage(image)
    lbl_reg=Label(root,image=pic,anchor=CENTER)
    lbl_reg.place(x=0,y=0)
  
    #-----------------INFO TOP------------
    lblinfo = Label(root, font=( 'aria' ,20, 'bold' ),text="RESPIRATORY DISEASE DETECTION",fg="white",bg="#000955",bd=10,anchor='w')
    lblinfo.place(x=450,y=50)
 
    lblinfo3 = Label(root, font=( 'aria' ,20 ),text="input audio path",fg="#000955",anchor='w')
    lblinfo3.place(x=180,y=200)
    E1 = Entry(root,width=30,font="veranda 20")
    E1.place(x=50,y=260)
    mylist = Listbox(root,width=60, height=20,bg="white")

    mylist.place( x = 700, y = 200 )
    btntrn=Button(root,padx=10,pady=2, bd=4 ,fg="white",font=('ariel' ,16,'bold'),width=10, text="Browse", bg="red",command=lambda:browseim())
    btntrn.place(x=180, y=300)
    btnhlp=Button(root,padx=80,pady=8, bd=6 ,fg="white",font=('ariel' ,10,'bold'),width=7, text="Test", bg="blue",command=lambda:predict(E1.get()))
    btnhlp.place(x=150, y=400)
    lblinfo1 = Label(root, font=( 'aria' ,20, ),text="Result :",fg="red",bg="white",anchor=W)
    lblinfo1.place(x=200,y=480)
    shrslt = Label(root, font=( 'aria' ,20, ),text="",fg="blue",bg="white",anchor=W)
    shrslt.place(x=300,y=480)

   

    def qexit():
        root.destroy()
     

    root.mainloop()


userHome()