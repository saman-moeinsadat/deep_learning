import os
os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'
from keras.applications import ResNet50
from keras.models import Sequential
from keras.layers import Dense, Flatten, GolabalAveragePooling2D
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator

num_classes = 2
resnet_path = '///'
my_new_model = Sequential()
my_new_model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_path))
my_new_model.add(Dense(num_classes, activation='softmax'))
my_new_model.layers[0].trainable = False
my_new_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
image_size = 224
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
train_data = data_generator.flow_from_directory(
        '///', targetsize=(image_size, image_size), batch_size=12,
        class_mode='categorical'
)
validation_data = data_generator.flow_from_directory(
        '///', target_size=(image_size, image_size), class_mode='categorical'
)
my_new_model.fit_generator(
    train_data, steps_per_epoch=3, validation_data=validation_data,
    validation_steps=1
)
