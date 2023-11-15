import random
import os
import numpy as np
from pyrsistent import plist
import librosa.display
import soundfile
from sklearn.model_selection import train_test_split
import librosa
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from keras.preprocessing import sequence
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
from keras.utils import np_utils
from keras.layers import GRU, LSTM, Embedding,Bidirectional 
featurelist=pickle.load( open( "featlist2.pkl", "rb" ) )
labellist=pickle.load( open( "labellist2.pkl", "rb" ) )
from keras.models import Sequential
from keras.layers import Dense, Dropout, LeakyReLU  
lb = LabelEncoder()
ydata = np_utils.to_categorical(lb.fit_transform(labellist))
print(ydata)
xdata=np.array(featurelist)
# print(xdata)
X_train, X_test, y_train, y_test = train_test_split(xdata, ydata, test_size=0.1, random_state=42)


X_train = np.reshape(X_train, (len(X_train), len(X_train[0]), 1))
print(X_train.shape)
X_val = np.reshape(X_test, (len(X_test), len(X_test[0]), 1))

print(X_val[0].shape)

print(X_train.shape)
num_labels = y_train.shape[1]
print(num_labels)
model = Sequential()

model.add(Bidirectional(GRU(256, activation='relu', recurrent_activation='hard_sigmoid')))

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(LeakyReLU())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))

#model.add(TimeDistributed(Dense(vocabulary)))
model.add(Dense(num_labels, activation='softmax'))
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
model.fit(X_train, y_train, batch_size=16, epochs=200, validation_data=(X_val, y_test))

# serialize model to JSON
model_json = model.to_json()
with open("BiGRUmodel.json", "w") as json_file:
       json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("BiGRUmodel.h5")
print("Saved model to disk")



# model = Sequential()

# model.add(Bidirectional(GRU(256, activation='relu', recurrent_activation='hard_sigmoid')))
# model.add(LeakyReLU())


# # model.add(Dense(128, activation='relu'))
# # model.add(LeakyReLU())
# # model.add(Dropout(0.5))
# # model.add(Dense(256, activation='relu'))
# # model.add(LeakyReLU())
# # model.add(Dropout(0.5))

# model.add(Dense(1024, activation='relu'))
# model.add(LeakyReLU())
# model.add(Dropout(0.5))

# # model.add(Dense(64, activation='relu'))
# # model.add(LeakyReLU())
# # model.add(Dropout(0.5))


# # model.add(Dense(32, activation='relu'))
# # model.add(LeakyReLU())
# # model.add(Dropout(0.5))

# # model.add(Dense(16, activation='relu'))
# # model.add(LeakyReLU())
# # model.add(Dropout(0.5))


# #model.add(TimeDistributed(Dense(vocabulary)))
# model.add(Dense(num_labels, activation='softmax'))
# model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
# model.fit(X_train, y_train, batch_size=16, epochs=200, validation_data=(X_val, y_test))