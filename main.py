import os
import numpy as np
from glob import glob

from keras import applications
from keras import optimizers
from keras .callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Flatten
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

train_data_dir = 'indian_currency_new/training/'
validation_data_dir = 'indian_currency_new/validation/'
epochs = 100
train_steps = 100
batch_size = 32
validation_steps = 20
saved_model_file_name = 'new_weights.h5'

img_width, img_height = 224, 224


model = applications.mobilenet.MobileNet(weights="imagenet", include_top=False,
                                         input_shape=(img_width, img_height, 3), pooling='avg')

for layer in model.layers[:-1]:
    layer.trainable = False

folders = glob('indian_currency_new/training/*')

x = Flatten()(model.output)
prediction = Dense(len(folders), activation='softmax')(x)
model_final = Model(inputs=model.input, outputs=prediction)

model_final.summary()

model_final.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.001), metrics=["accuracy"])

train_datagen = ImageDataGenerator(rescale=1./255,
                                   fill_mode='nearest',
                                   zoom_range=0.3,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2, rotation_range=360)

validation_datagen = ImageDataGenerator(rescale=1./255,
                                       fill_mode='nearest',
                                       zoom_range=0.3,
                                       rotation_range=30)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    class_mode='categorical')

print ("Training_labels are: ", validation_generator.class_indices)

# Save the model
checkpoint = ModelCheckpoint(saved_model_file_name, monitor='val_loss',
                             verbose=1, save_best_only=True, save_weights_only=False,
                             mode='auto', period=1)
earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')

# Train the model
history = model_final.fit_generator(train_generator,
                              steps_per_epoch=train_steps,
                              epochs=epochs,
                              validation_data=validation_generator,
                              validation_steps=validation_steps, workers=16,
                              callbacks=[checkpoint, earlystopping])

probabilities = model_final.predict_generator(validation_generator,
                                             workers=16, verbose=1)

# Confusion Matrix
y_true = ((validation_generator.classes))
y_pred = (np.argmax(probabilities, axis=1))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred)
print (cm)


                                            









