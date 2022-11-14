import cv2 
import matplotlib.pyplot as plt

def cv2_imshow(image):
    cv2.imshow('My_Image',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# read images
img1 = cv2.imread('./victoria.jpg') 
img2 = cv2.imread('./victoria2.jpg') 

# convert to greyscale
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# create sift object
sift = cv2.SIFT_create()
orb = cv2.ORB_create()

sift_keypoints1, sift_descriptors1 = sift.detectAndCompute(img1,None)
sift_keypoints2, sift_descriptors2 = sift.detectAndCompute(img2,None)

orb_keypoints1, orb_descriptors1 = orb.detectAndCompute(img1,None)
orb_keypoints2, orb_descriptors2 = orb.detectAndCompute(img2,None)

# BFMatcher with default params
bf = cv2.BFMatcher()

sift_matches = bf.knnMatch(sift_descriptors1, sift_descriptors2, k=2)
orb_matches = bf.knnMatch(orb_descriptors1, orb_descriptors2, k=2)


# Apply ratio test
sift_good = []
for m,n in sift_matches:
    if m.distance < 0.75*n.distance:
        sift_good.append([m])

orb_good = []
for m,n in orb_matches:
    if m.distance < 0.75*n.distance:
        orb_good.append([m])


sift_img_result = cv2.drawMatchesKnn(img1, sift_keypoints1, img2, sift_keypoints2, sift_good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

orb_img_result =  cv2.drawMatchesKnn(img1, orb_keypoints1, img2, orb_keypoints2, orb_good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#plt.imshow(img_result),plt.show()
cv2_imshow(sift_img_result)
cv2_imshow(orb_img_result)
#print the number of matches for each method
print(len(orb_good))
print(len(sift_good))