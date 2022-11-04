import numpy as np
import cv2
import matplotlib.pyplot as plt

image = cv2.imread(r"C:\Users\slaba\OneDrive\Desktop\Y3\COMP-388-ComputerVision\Computer-Vision\victoria.jpg")

filter = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])

def cv2_imshow(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.axis(False)
    plt.show()

def get_image_dimensions(image):
    h = image.shape[0]
    w = image.shape[1]
    return(h,w)

def get_filter_dimensions(filter):
    h = filter.shape[0]
    w = filter.shape[1]
    return(h,w)

image_height, image_width = get_image_dimensions(image)

filter_height, filter_width = get_filter_dimensions(filter)

print("filter height: " + str(filter_height))
print("filter width: " + str(filter_width))

print("image height: " + str(image_height)) 
print("image width: " + str(image_width))

cv2_imshow(image)