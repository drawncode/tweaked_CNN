import keras
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Dropout, Conv2D, MaxPooling2D, Activation, BatchNormalization
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.optimizers import RMSprop
from keras.losses import binary_crossentropy, categorical_crossentropy
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import random
import os
import cv2
import pydot
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
from livelossplot import PlotLossesKeras


X = []
Y = {'Y_color':[], 'Y_width':[], 'Y_angle':[], 'Y_length':[]}


path = "line_dataset/"
images = os.listdir(path)
random.seed(42)
random.shuffle(images)

def Append(image):
    data=image[:-4].split('_')
    length = data[0]
    width = data[1] 
    angle = data[2]
    color = data[3]
    img=cv2.imread(path+image)
    X.append(img)
    Y['Y_color'].append(color)
    Y['Y_length'].append(length)
    Y['Y_width'].append(width)
    Y['Y_angle'].append(angle)

print("loading the data...........")
for image in images:
    Append(image)
print(len(X), "images loaded successfully.")

X = np.asarray(X)
X=X.astype('float32')/255.0
Y['Y_color']=np.array(Y['Y_color'])
Y['Y_length']=np.array(Y['Y_length'])
Y['Y_width']=np.array(Y['Y_width'])
Y['Y_angle']=np.array(Y['Y_angle'])
Y['Y_angle']=to_categorical(Y['Y_angle'],12)


split = train_test_split(X,Y['Y_color'],Y['Y_length'],Y['Y_width'],Y['Y_angle'],test_size=0.2, random_state=42)
(X_train,X_test,Y_color_train,Y_color_test,Y_length_train,Y_length_test,Y_width_train,Y_width_test,Y_angle_train,Y_angle_test) = split

print("Data split \n train set :",len(X_train),"\n test set :", len(X_test))
input = Input(shape=(28,28,3))
x = Conv2D(32,(3,3),use_bias=False, padding = 'same')(input)
x = BatchNormalization()(x)
x = Activation("relu")(x)
# x = Conv2D(32,(3,3),use_bias=False, padding = 'same')(x)
# x = BatchNormalization()(x)
# x = Activation("relu")(x)
x = MaxPooling2D((2,2))(x)
x = Conv2D(64,(3,3),use_bias= False, padding = 'same')(x)
x = BatchNormalization()(x)
# x = Activation("relu")(x)
# x = Conv2D(64,(3,3),use_bias= False, padding = 'same')(x)
# x = BatchNormalization()(x)
# x = Activation("relu")(x)
x = MaxPooling2D((2,2))(x)
x = Conv2D(128,(3,3),use_bias= False, padding = 'same')(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)
features = Flatten()(x)
x = Dense(64,use_bias=False)(features)
x = BatchNormalization()(x)
x = Activation("relu")(x)
output_1 = Dense(1,activation = 'sigmoid',name="color")(x)
x = Dense(64,use_bias=False)(features)
x = BatchNormalization()(x)
x = Activation("relu")(x)
output_2 = Dense(1,activation = 'sigmoid',name="length")(x)
x = Dense(64,use_bias=False)(features)
x = BatchNormalization()(x)
x = Activation("relu")(x)
output_3 = Dense(1,activation = 'sigmoid',name="width")(x)
x = Dense(128,use_bias=False)(features)
x = BatchNormalization()(x)
x = Activation("relu")(x)
# x = Dense(256,use_bias=False)(x)
# x = BatchNormalization()(x)
# x = Activation("relu")(x)
# x = Dense(128,use_bias=False)(x)
# x = BatchNormalization()(x)
# x = Activation("relu")(x)
output_4 = Dense(12,activation = 'softmax',name="angle")(x)
network = Model(input,[output_1,output_2,output_3,output_4])
# plot_model(network, to_file='model.png')

network.summary()

loss_1 = [binary_crossentropy, binary_crossentropy, binary_crossentropy, categorical_crossentropy]
loss_2 = {"dense_2":"binary_crossentropy","dense_4":"binary_crossentropy","dense_6":"binary_crossentropy","name1":"categorical_crossentropy"}
network.compile(optimizer = 'RMSprop', loss = loss_1, metrics = ['accuracy'])

print("\n\nStarting the training")
stats = network.fit(X_train, [Y_color_train,Y_length_train,Y_width_train,Y_angle_train], epochs = 2, validation_data = (X_test,[Y_color_test,Y_length_test,Y_width_test,Y_angle_test]), batch_size=16,verbose = 1,shuffle = True,callbacks=[PlotLossesKeras()])
print("\n\n Training completed successfully, saving the weights\n")

network.save_weights("weights_final.h5")

# print(stats.history)
history=stats
accuracy=[stats.history['angle_acc'],stats.history['color_acc'],stats.history['length_acc'],stats.history['width_acc']]
accuracy=[sum(x) for x in zip(*accuracy)]
accuracy =[x/4.0 for x in accuracy]
val_accuracy=[stats.history['val_angle_acc'],stats.history['val_color_acc'],stats.history['val_length_acc'],stats.history['val_width_acc']]
val_accuracy=[sum(x) for x in zip(*val_accuracy)]
val_accuracy =[x/4.0 for x in val_accuracy]

network.load_weights("weights_final.h5")

print("\n Getting the predictions on the test set\n")
predictions = network.predict(X_test, batch_size = 16, verbose =1)

predictions[3] = np.argmax(predictions[3], axis = 1)
Y_angle_test = np.argmax(Y_angle_test, axis =1)
pred_GT=[Y_color_test,Y_length_test,Y_width_test,Y_angle_test]
name=["color","length","width","angle"]


for i in range(4):
    predictions[i]=predictions[i].astype('int')
    pred_GT[i]=pred_GT[i].astype('int')
    predictions[i]=predictions[i].reshape(predictions[i].shape[0],)

print("\nF1-scores:")
for i in range(4):
    fs=f1_score(pred_GT[i],predictions[i],average = 'weighted')
    print(name[i]+" : ",fs)

print("\nConfusion matrices:")
for i in range(4):
    cm=confusion_matrix(pred_GT[i],predictions[i])
    print(name[i]+" : ")
    print(cm)



