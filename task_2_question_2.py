import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread("./victoria.jpg")
img2 = cv2.imread("./victoria2.jpg")

#convert both images to grayscale
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

#create ORB object
orb = cv2.ORB_create()

# find the keypoints in image 1
img1_key_points = orb.detect(img1_gray,None)

# show key points on image
key_points_img1 = cv2.drawKeypoints(img1, img1_key_points, None, color=(255,0,0))
plt.imshow(key_points_img1)
plt.show()

# find the keypoints in image 2
img2_key_points = orb.detect(img2_gray,None)

# show keypoints on image
key_points_img2 = cv2.drawKeypoints(img2, img2_key_points, None, color=(255,0,0))
plt.imshow(key_points_img2)
plt.show()