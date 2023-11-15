import random
import os
import numpy as np
import librosa.display
import soundfile
from sklearn.model_selection import train_test_split
import librosa
from Augmentation import add_noise, shift, stretch
import pickle


def features_extractor(file):
    #load the file (audio)
    audio, sample_rate = librosa.load(file, res_type='kaiser_fast') 
    #we extract mfcc
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    #in order to find out scaled feature we do mean of transpose of value
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    return mfccs_scaled_features


filepath=[]
root = "dataset2"
for path, subdirs, files in os.walk(root):
    for name in files:
        filepath.append(os.path.join(path, name))
        # print(os.path.join(path, name))

random.shuffle(filepath)
featurelist=[]
labellist=[]
for path in filepath:  
    try:
        # features = features_extractor(path)
        # print(features.shape)
        li1=path.split("\\")
        typ=li1[1]
        # print(typ)
        if(typ=="COPD"):
            data_x, sampling_rate = librosa.load(path,res_type='kaiser_fast')
            mfccs = np.mean(librosa.feature.mfcc(y=data_x, sr=sampling_rate, n_mfcc=40).T,axis=0)
            featurelist.append(mfccs)
            labellist.append(typ)
            print("copd")
        else:
            data_x, sampling_rate = librosa.load(path,res_type='kaiser_fast')
            mfccs = np.mean(librosa.feature.mfcc(y=data_x, sr=sampling_rate, n_mfcc=40).T,axis=0) 
  
            featurelist.append(mfccs)
            labellist.append(typ)
            data_noise = add_noise(data_x,0.005)
            mfccs_noise = np.mean(librosa.feature.mfcc(y=data_noise, sr=sampling_rate, n_mfcc=40).T,axis=0) 
            featurelist.append(mfccs_noise)
            labellist.append(typ) 
            data_shift = shift(data_x,1600)
            mfccs_shift = np.mean(librosa.feature.mfcc(y=data_shift, sr=sampling_rate, n_mfcc=40).T,axis=0) 
            featurelist.append(mfccs_shift)
            labellist.append(typ)  

            data_stretch = stretch(data_x,1.2)
            mfccs_stretch = np.mean(librosa.feature.mfcc(y=data_stretch, sr=sampling_rate, n_mfcc=40).T,axis=0) 
            featurelist.append(mfccs_stretch)
            labellist.append(typ) 
            data_stretch_2 = stretch(data_x,0.8)
            mfccs_stretch_2 = np.mean(librosa.feature.mfcc(y=data_stretch_2, sr=sampling_rate, n_mfcc=40).T,axis=0) 
            featurelist.append(mfccs_stretch_2)
            labellist.append(typ)  
            print("Type-->",typ)
        

    except:
        print(path)

dbfile=open("featlist2.pkl","wb")
pickle.dump(featurelist,dbfile)
dbfile.close()

dbfile1=open("labellist2.pkl","wb")
pickle.dump(labellist,dbfile1)
dbfile1.close()

print(featurelist)
print(labellist)