import cv2
import numpy as np
from keras.models import load_model

from matplotlib import pyplot as plt

# Training_labels are:  {'10': 0, '100': 1, '20': 2, '200': 3, '2000': 4, '50': 5, '500': 6, 'Background': 7}
img = cv2.imread('/content/indian_currency_new/validation/20/39.jpg')
img = cv2.resize(img, (224, 224))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
img = np.expand_dims(img, axis=0)

model_final = load_model('/content/new_weights.h5')

probabilities = model_final.predict(img)
y_pred = (np.argmax(probabilities, axis=1))
print (y_pred)
