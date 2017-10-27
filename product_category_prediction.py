# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 10:37:22 2017

@author: SreeChaitanya.Y
"""


from keras.models import Model
#from scipy.misc import imread
import numpy as np
from keras.layers import Dense,GlobalAveragePooling2D,Flatten,Dropout
import pandas as pd
from keras.callbacks import ModelCheckpoint , EarlyStopping
from keras.preprocessing import image
from keras.applications import resnet50
from keras.optimizers import SGD
from keras.applications.resnet50 import preprocess_input
from sklearn.preprocessing import LabelEncoder
from keras.layers.normalization import BatchNormalization
#import h5py
from keras.utils.np_utils import to_categorical

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train_path = r'~\\hackerearth\\dl_1\\\\train_img\\'
test_path = r'~\\hackerearth\\dl_1\\test_img\\'

train_img=[]
for i in range(len(train)):
    temp_img = image.load_img(train_path+train['image_id'][i]+'.png',target_size=(256,256))
    temp_img=image.img_to_array(temp_img)
    train_img.append(temp_img)
train_img=np.array(train_img)

train_img=preprocess_input(train_img)

base_model = resnet50(weights='imagenet',include_top=False)

#add a global spatial average pooling layer
#train_img=preprocess_input(train_img)

x= base_model.output
#x = Flatten()(x)
x = GlobalAveragePooling2D()(x)

x = Dense(1024, activation="relu")(x)
#fully connected layer
#x = Dense(1024,activation='relu')(x)
#add a logistic layer -- for 25 classes
pred = Dense(25,activation='softmax')(x)


#freeze all lower level layers
for layer in base_model.layers:
    layer.trainable = False

#this is the model we will train
model = Model(inputs=base_model.inputs,outputs=pred)

checkpoint = ModelCheckpoint("resnet50_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')

#freeze all lower level layers
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer= 'sgd',loss='categorical_crossentropy',metrics=['accuracy'])

img_rows, img_cols = 256, 256 # Resolution of inputs
channel = 3
batch_size = 16
nb_epoch = 12


train_y=np.asarray(train['label'])
le = LabelEncoder()

train_y = le.fit_transform(train_y)

train_y=to_categorical(train_y)

train_y=np.array(train_y)
from sklearn.model_selection import train_test_split
#X_train, X_valid, Y_train, Y_valid=train_test_split(train_img,train_y,test_size=0.2, random_state=42)
X_train =  train_img
Y_train = train_y

#model.fit(X_train, Y_train,batch_size=batch_size,epochs=nb_epoch,shuffle=True,verbose=1,validation_data=(X_valid, Y_valid))
datagen = image.ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.25,
        shear_range=0.2,
        zoom_range=0.35,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='wrap')

model.fit_generator(datagen.flow(X_train, Y_train,
                    batch_size=batch_size),
                    nb_epoch=nb_epoch,
                    steps_per_epoch=X_train.shape[0] // batch_size,
                    verbose=1,
                    callbacks=[early,checkpoint])

#model.fit(train_img,train_y,batch_size=batch_size,epochs=nb_epoch,shuffle=True,verbose=1)
model.save('model.sav')
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

test_img=[]
for i in range(len(test)):
    temp_img = image.load_img(test_path+test['image_id'][i]+'.png',target_size=(256,256))
    temp_img=image.img_to_array(temp_img)
    test_img.append(temp_img)
test_img=np.array(test_img)

test_img=preprocess_input(test_img)

# we chose to train the top few layers of resnet50

for layer in model.layers[:10]:
    layer.trainable = False
for layer in model.layers[10:]:
    layer.trainable = True

sgd = SGD(lr=0.0001,momentum=0.9)
model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(X_train, Y_train,batch_size=batch_size,epochs=6,shuffle=True,verbose=1)
predictions_valid = model.predict(test_img, batch_size=batch_size, verbose=1)
y_pred = np.argmax(predictions_valid,axis=1)

from sklearn.metrics import f1_score

y_true = pd.read_csv(r'actual_categories.csv')['label']
y_true=le.fit_transform(y_true)
scr = f1_score(y_true, y_pred, average='micro')
labels = le.inverse_transform(y_pred)
print("final score : "+str(scr))
#sLength = len(test['image_id'])
#test['label'] = pd.Series(labels,np.random.randn(sLength), index=test.index)
test = pd.read_csv(r'test.csv')

test = pd.concat([test, pd.DataFrame(labels,columns=['label'])], axis=1)
test.to_csv(r'imag_prede.csv',index=False)
