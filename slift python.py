import cv2
import matplotlib.pyplot as plt
# %matplotlib inline

# read images
img1 = cv2.imread('/home/devdatta/Desktop/YOLO-object-detection-with-OpenCV-master/real-time-object-detection/WhatsApp Image 2024-03-19 at 1.42.53 AM.jpeg')
img2 = cv2.imread('/home/devdatta/Desktop/YOLO-object-detection-with-OpenCV-master/real-time-object-detection/WhatsApp Image 2024-03-19 at 1.42.54 AM (1).jpeg')
img3 = cv2.imread('/home/devdatta/Desktop/YOLO-object-detection-with-OpenCV-master/real-time-object-detection/WhatsApp Image 2024-03-19 at 1.42.54 AM (1).jpeg')


img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)


# create a sift object
sift = cv2.xfeatures2d.SIFT_create()

keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)
keypoints_3, descriptors_3 = sift.detectAndCompute(img3,None)


len(keypoints_1), len(keypoints_2), len(keypoints_3)
#feature matching
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True) # https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html

matches = bf.match(descriptors_1,descriptors_2)
matches = sorted(matches, key = lambda x:x.distance)

len(matches)
img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:150], img2, flags=2) # only plotting 150 keypoints
# Hide the numbers on the x and y axes
plt.xticks([])  # Hide x-axis numbers
plt.yticks([])  # Hide y-axis numbers
plt.imshow(img3), plt.show()