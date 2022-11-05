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

#get height and width of an image
def get_image_dimensions(image):
    h = image.shape[0]
    w = image.shape[1]
    return(h,w)

#get height and width of the filter
def get_filter_dimensions(filter):
    h = filter.shape[0]
    w = filter.shape[1]
    return(h,w)

#adding padding to image rows based on the width of the current image
def rows_padding(image):
    _,image_width = get_image_dimensions(image)
    padding = np.zeros((image_width,), dtype=int)

    pad_image_rows = np.insert(image, 0, padding, axis=0)

    pad_image_rows = np.append(pad_image_rows, [padding], axis=0)

    return(pad_image_rows)

def col_padding(image):
    image_height,_ = get_image_dimensions(image)

    pad_left_side = np.zeros((image_height,), dtype=int)
    pad_right_side = np.zeros((image_height,1), dtype=int)
    
    pad_image_col = np.insert(image, 0, [pad_left_side], axis=1)

    pad_image_col = np.append(pad_image_col, pad_right_side, axis=1)

    return(pad_image_col)


#def filter2D(image,filter):
#gray = np.append(image, pad, axis=0)

image_height, image_width = get_image_dimensions(image)

filter_height, filter_width = get_filter_dimensions(filter)

print("filter height: " + str(filter_height))
print("filter width: " + str(filter_width))

print("image height: " + str(image_height)) 
print("image width: " + str(image_width))

cv2_imshow(image)