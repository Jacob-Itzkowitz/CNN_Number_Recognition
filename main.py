import os
from PIL import Image
import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Input
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten
from keras import backend as k
import numpy

#for images in os.listdir(r"C:\Users\jacob.itzkowitz\PycharmProjects\ImageRecognitionNumbers\Test_Images"):
directory = r"C:\Users\jacob.itzkowitz\PycharmProjects\ImageRecognitionNumbers\Test_Images"
pathway = os.path.join(directory, "1.jpg")
image = Image.open(pathway)
image = image.resize((28, 28))
image = image.convert("L")
pixel_values = image.getdata()
Image_Test_data = []
for pixel_value in pixel_values:
    Image_Test_data.append(pixel_value)

rows, columns = 28, 28
Image_Test_data = numpy.array(Image_Test_data)
Image_Test_data = Image_Test_data.reshape(1, rows, columns)


(x_train, y_train), (x_test, y_test) = mnist.load_data()
if k.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, rows, columns)
    x_test = x_test.reshape(x_test.shape[0], 1, rows, columns)
    inpx = (1, rows, columns)

else:
    x_train = x_train.reshape(x_train.shape[0], rows, columns, 1)
    x_test = x_test.reshape(x_test.shape[0], rows, columns, 1)
    inpx = (rows, columns, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

inpx = Input(shape=inpx)
layer1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(inpx)
layer2 = Conv2D(64, (3, 3), activation='relu')(layer1)
layer3 = MaxPooling2D(pool_size=(3, 3))(layer2)
layer4 = Dropout(0.5)(layer3)
layer5 = Flatten()(layer4)
layer6 = Dense(250, activation='sigmoid')(layer5)
layer7 = Dense(10, activation='softmax')(layer6)

model = Model([inpx], layer7)
model.compile(optimizer=keras.optimizers.Adadelta(),
              loss=keras.losses.categorical_crossentropy,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=12, batch_size=500)

score = model.evaluate(x_test, y_test, verbose=0)
print('loss=', score[0])
print('accuracy=', score[1])
print(model.predict(Image_Test_data))

