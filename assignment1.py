import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

#show the image, taken from the labs
def cv2_imshow(image):
    plt.imshow(image)
    plt.axis(False)
    plt.show()

#get height and width of an image
def get_image_dimensions(image):
    h = image.shape[0]
    w = image.shape[1]
    return(h,w)

#adding padding to image rows based on the width of the current image
def rows_padding(image):
    _,image_width = get_image_dimensions(image)
    padding = np.zeros((image_width,), dtype=int)

    pad_image_rows = np.insert(image, 0, padding, axis=0)

    pad_image_rows = np.append(pad_image_rows, [padding], axis=0)

    return(pad_image_rows)

#adding padding to image columns based on the height of the current image
def col_padding(image):
    image_height,_ = get_image_dimensions(image)

    pad_left_side = np.zeros((image_height,), dtype=int)
    pad_right_side = np.zeros((image_height,1), dtype=int)
    
    pad_image_col = np.insert(image, 0, [pad_left_side], axis=1)

    pad_image_col = np.append(pad_image_col, pad_right_side, axis=1)

    return(pad_image_col)

#apply padding based on the size of the filter
#i.e IF filter is 5 by 5, THEN pad 2 pixels 
def apply_padding(filter,image):
    size_of_filter = filter.shape[0]

    padded_image = image

    #the size of the filter indicates the number of pixel the code needs to add
    num = size_of_filter // 2

    for x in range(0,num):
        padded_image = rows_padding(padded_image)
        padded_image = col_padding(padded_image)

    return(padded_image)


def apply_convolution(filter,image):

    image_height, image_width = get_image_dimensions(image)

    size_of_filter = filter.shape[0]
    num = size_of_filter // 2
    
    #create a new array filled with zeros to pass the results of the convolution here
    #think of it as a blank canvas for the convolution results
    final_image = np.zeros((image_height,image_width), dtype=int)

    for x in range(num,image_height-num):
        for y in range(num,image_width-num):

            #get the area around the pixel of interest 
            #the area depends on the size of the filter hence the variable num
            pixel_area = image[x-num:x+(num+1), y-num:y+(num+1)]

            #multiply with filter and sum up to get the final value
            product = np.multiply(pixel_area,filter)
            sum = np.sum(product)
            
            #keep the values within the acceptable range for a grayscale image 0:255
            if sum > 255:
                final_image[x,y] = 255
            elif sum < 0:
                final_image[x,y] = 0
            else:
                final_image[x,y] = sum
            
    return final_image


image = cv2.imread(r"C:\Users\slaba\OneDrive\Desktop\Y3\COMP-388-ComputerVision\Computer-Vision\victoria.jpg")

#Step 1 turn input image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Filters
laplace1 = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
laplace2 = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
sobel1 = np.array([[-1,-2,0,2,1],[-2,-3,0,3,2],[-3,-5,0,5,3],[-2,-3,0,3,2],[-1,-2,0,2,1]])
sobel2 = np.array([[1,2,3,2,1],[2,3,5,3,2],[0,0,0,0,0],[-2,-3,-5,-3,-2],[-1,-2,-3,-2,-1]])
vertical_edge_detection = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
horizontal_edge_detection = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])

#First apply padding then convolution
#To experiment with other filters replace ONLY the first argument
#of the functions apply_padding & apply_convolution with the filter you desire
padded_image = apply_padding(laplace1,gray_image)
final_image = apply_convolution(laplace1,padded_image)
cv2_imshow(final_image)
print(final_image[0:10,0:10])

print("==============================================================================")

cv2_filtered_image = cv2.filter2D(gray_image,-1,laplace1)
cv2_imshow(cv2_filtered_image)
print(cv2_filtered_image[0:10,0:10])

