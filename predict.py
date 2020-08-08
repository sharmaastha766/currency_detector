from keras.models import load_model
validation_data_dir = 'indian_currency_new/validation/'
model_final = load_model()

validation_datagen = ImageDataGenerator(rescale=1./255,
                                       fill_mode='nearest',
                                       zoom_range=0.3,
                                       rotation_range=30)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    class_mode='categorical')

probabilities = model_final.predict_generator(validation_generator,
                                             workers=16, verbose=1)

# Confusion Matrix
y_true = ((validation_generator.classes))
y_pred = (np.argmax(probabilities, axis=1))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred)
print (cm)
