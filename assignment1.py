import numpy as np
import cv2
import matplotlib.pyplot as plt

image = cv2.imread(r"C:\Users\slaba\OneDrive\Desktop\Y3\COMP-388-ComputerVision\Assignment1\victoria.jpg")

def cv2_imshow(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.axis(False)
    plt.show()

cv2_imshow(image)