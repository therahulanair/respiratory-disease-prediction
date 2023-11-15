import numpy as np
import librosa
import os
import pandas as pd
data=pd.read_csv("patient_diagnosis.csv")
data.columns=['patient_id','disease']
print(data)
import shutil, os
def InstantiateAttributes(dir_):

    X_=[]
    y_=[]
    COPD=[]
    copd_count=0
    # print(os.listdir(dir_))
    # x=1/0
    for soundDir in (os.listdir(dir_)):
        # print(soundDir)
        # print(soundDir[-3:])
        # print(soundDir[:3])
        try:

            if(soundDir[-3:]=='wav'):
                audio_dir=dir_+soundDir
                print("audio directory--->",audio_dir)
                df_new = data[data['patient_id'] == int(soundDir[:3])]['disease']
                dflist=df_new.values.tolist()
                # p = list(data[data['patient_id']==int(soundDir[:3])]['disease'])[0]
                diss=dflist[0]
                dislist= ['Bronchiectasis' 'Bronchiolitis' 'COPD' 'Healthy']
                print("disease--->",dflist[0])
                if(diss not in dislist):
                    shutil.copy(audio_dir, 'dataset2/'+diss)
        except:
            print("error")
       
    
    # return X_,y_
InstantiateAttributes("audio_and_txt_files/")

