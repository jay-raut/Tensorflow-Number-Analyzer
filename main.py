import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tkinter import *
from tkinter import filedialog

def display_image(path):
    model = tf.keras.models.load_model('/Users/jayraut/Documents/GitHub/Tensorflow-Number-Analyzer/model.h5')
    img = cv2.imread(path)[:, :, 0]
    img = cv2.resize(img, (28, 28))
    img = np.invert(np.array([img]))
    prediction = model.predict(img)
    print(f"Number is: {np.argmax(prediction)}")
    plt.title(f"Predicted Digit: {np.argmax(prediction)}")
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()

def recognize_image():
    root.destroy()
    display_image(root.filename)

root = Tk()
root.title("Get Image")
root.filename = filedialog.askopenfilename(title="Select Image File", filetypes=(("image files", "*.png"), ("image files", "*.jpg")))
button = Button(root, text="Recognize Image", command=recognize_image)
button.pack()
root.mainloop()
