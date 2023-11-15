from keras import backend as K
from model import InstantiateModel
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adamax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
import numpy as np
import pickle
def trainModel(X, y):
    batch_size=X.shape[0]
    time_steps=X.shape[1]
    data_dim=X.shape[2]
    print("input shape==>",time_steps,data_dim)
    Input_Sample = Input(shape=(time_steps,data_dim))
    Output_ = InstantiateModel(Input_Sample)
    Model_Enhancer = Model(inputs=Input_Sample, outputs=Output_)
    print(Model_Enhancer.summary)
    Model_Enhancer.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')
    
    ES = EarlyStopping(monitor='val_loss', min_delta=0.5, patience=200, verbose=1, mode='auto', baseline=None,restore_best_weights=False)
    MC = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='auto', verbose=1, save_best_only=True)
    
    #class_weights = class_weight.compute_sample_weight('balanced',
	#                                                 np.unique(y[:,0],axis=0),
	#                                                 y[:,0])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    ModelHistory = Model_Enhancer.fit(X_train, y_train, batch_size=32, epochs=30,
                                  validation_data=(X_test, y_test),
                                  callbacks = [MC],
                                  #class_weight=class_weights,
                                  verbose=1)




featurelist=pickle.load( open( "featlist1.pkl", "rb" ) )
labellist=pickle.load( open( "labellist1.pkl", "rb" ) )
# label_binarizer = LabelBinarizer()
# labels = lb.fit_transform(labellist)
# n_classes = len(label_binarizer.classes_)
# print(n_classes)

# from tensorflow.keras.utils import to_categorical
# labels=to_categorical(labels, num_classes = 4)
# ydata=to_categorical(labels, 6)
le = LabelEncoder()
label = le.fit_transform(labellist)
print("label",label)
print("labtyp",type(label))
xdata=np.array(featurelist)
ydata=np.array(label)
xdata=np.reshape(xdata,(xdata.shape[0],xdata.shape[1],1))
print(xdata.shape)
# print(ydata.shape[1])
# # print(xdata[0])
# # print(xdata[0].shape)
# print(ydata.shape)
# print(ydata)

y_train = np.asarray(ydata).astype('float32').reshape((-1,1))
trainModel(xdata,y_train)