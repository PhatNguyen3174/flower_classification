import cv2
import os
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,InputLayer
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import to_categorical
import random as rn

image_directory='train/'

daisy=os.listdir(image_directory+ 'daisy/')
dandelion=os.listdir(image_directory+ 'dandelion/')
rose = os.listdir(image_directory+ 'rose/')
sunflower = os.listdir(image_directory+ 'sunflower/')
tulip = os.listdir(image_directory+ 'tulip/')
dataset=[]
label=[]

INPUT_SIZE=64


for i , image_name in enumerate(daisy):
    if(image_name.split('.')[-1]=='jpg'):
        image=cv2.imread(image_directory+ 'daisy/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize((INPUT_SIZE,INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(0)

for i , image_name in enumerate(rose):
    if(image_name.split('.')[-1]=='jpg'):
        image=cv2.imread(image_directory+ 'rose/'+image_name)
        image=Image.fromarray(image, 'RGB')
        image=image.resize((INPUT_SIZE,INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(1)

for i , image_name in enumerate(tulip):
    if(image_name.split('.')[-1]=='jpg'):
        image=cv2.imread(image_directory+ 'tulip/'+image_name)
        image=Image.fromarray(image, 'RGB')
        image=image.resize((INPUT_SIZE,INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(2)

for i , image_name in enumerate(dandelion):
    if(image_name.split('.')[-1]=='jpg'):
        image=cv2.imread(image_directory+ 'dandelion/'+image_name)
        image=Image.fromarray(image, 'RGB')
        image=image.resize((INPUT_SIZE,INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(3)

for i , image_name in enumerate(sunflower):
    if(image_name.split('.')[-1]=='jpg'):
        image=cv2.imread(image_directory+ 'sunflower/'+image_name)
        image=Image.fromarray(image, 'RGB')
        image=image.resize((INPUT_SIZE,INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(4)

        
dataset=np.array(dataset)
label=np.array(label)

x_train, x_test, y_train, y_test=train_test_split(dataset, label, test_size=0.002, random_state=0)


x_train=normalize(x_train, axis=1)
x_test=normalize(x_test, axis=1)



print(x_train.shape)
y_train=to_categorical(y_train , num_classes=5)
y_test=to_categorical(y_test , num_classes=5)



# Model Building
# 64,64,3

model=Sequential()
model.add(InputLayer(input_shape=(INPUT_SIZE,INPUT_SIZE,3)))
model.add(Conv2D(32, (3,3), ))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32, (3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Conv2D(64, (3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(5))
model.add(Activation('softmax'))





model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])


model.fit(x_train, y_train, 
batch_size=16, 
verbose=1, epochs=16, 
validation_data=(x_test, y_test),
shuffle=False)


model.save('FlowerCNN.h5')




