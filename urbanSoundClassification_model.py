import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Flatten, Activation, Dropout
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras import optimizers
import pandas as pd
import numpy as np


def append_ext(fn):
    return fn + '.jpg'


traindf = pd.read_csv(
    '~/python-projects/data_kaggle/urban_sound_classification/train/train.csv',
    dtype=str
)
testdf = pd.read_csv(
    '~/python-projects/data_kaggle/urban_sound_classification/test/test.csv',
    dtype=str
)
traindf['ID'] = traindf['ID'].apply(append_ext)
# print traindf.head()
testdf['ID'] = testdf['ID'].apply(append_ext)
# print testdf.head()
datagen = ImageDataGenerator(rescale=1./255., validation_split=0.25)
train_generator = datagen.flow_from_dataframe(
    dataframe=traindf,
    directory='/home/saman/python-projects/plaidml-venv/train_mfc',
    x_col='ID',
    y_col='Class',
    subset='training',
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode='categorical',
    target_size=(64, 64)
)
valid_generator = datagen.flow_from_dataframe(
    dataframe=traindf,
    directory='/home/saman/python-projects/plaidml-venv/train_mfc',
    x_col='ID',
    y_col='Class',
    subset='validation',
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode='categorical',
    target_size=(64, 64)
)
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=(64, 64, 3)))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.compile(
    optimizers.rmsprop(lr=0.0005, decay=1e-6),
    loss='categorical_crossentropy', metrics=['accuracy']
)
model.summary()
step_size_train = train_generator.n // train_generator.batch_size
step_size_valid = valid_generator.n // valid_generator.batch_size
model.fit_generator(
    generator=train_generator,
    steps_per_epoch=step_size_train,
    validation_data=valid_generator,
    validation_steps=step_size_valid,
    epochs=150
)
model.evaluate_generator(
    generator=valid_generator,
    steps=step_size_valid
)
# test_datagen = ImageDataGenerator(rescale=1./255.)
# test_generator = test_datagen.flow_from_dataframe(
#     dataframe=testdf,
#     directory='~/python-projects/data_kaggle/urban_sound_classification/test/test_mfc',
#     x_col='ID',
#     y_col=None,
#     batch_size=32,
#     seed=42,
#     shuffle=False,
#     class_mode=None,
#     target_size=(64, 64)
# )
# step_size_test = test_generator.n // test_generator.batch_size
# test_generator.reset()
# preds = model.predict_generator(
#     generator=test_generator,
#     steps=step_size_test,
#     verbose=1
# )
# predicted_class_indices = np.argmax(preds, axis=1)
# labels = train_generator.class_indices
# labels = dict((v, k) for k, v in labels.items())
# predictions = [labels[k] for k in predicted_class_indices]
