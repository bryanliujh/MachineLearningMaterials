
#images must be structured based on folders name eg. Test_set, train_set folder -> dogs, cats folder
#image download from super data science website https://www.superdatascience.com/machine-learning/

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

classifier = Sequential()
#create 32 3x3 feature detector
# 3 channel (RGB) of 64x64 input image
classifier.add(Convolution2D(32, 3, 3, input_shape=(64,64,3), activation='relu'))
#2x2 pooled feature map
classifier.add(MaxPooling2D(pool_size=(2,2)))

#adding a second convolution layer will improve accuracy and prevent overfitting or can increase target size in train/test set
# 3 channel (RGB) of 64x64 input image
classifier.add(Convolution2D(32, 3, 3, activation='relu'))
#2x2 pooled feature map
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Flatten())

#hidden layer
classifier.add(Dense(units=128, activation='relu'))

#output layer (for this we want only one binary output (dog or cat))
classifier.add(Dense(units=1, activation='sigmoid'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)