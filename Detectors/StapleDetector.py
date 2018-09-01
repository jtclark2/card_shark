import cv2
import numpy as np
from matplotlib import pyplot as plt
import time

#########################Read Raw#########################
img_name = "IMG_5881.JPG"

img_path = r".\..\RawImages\Staples" + "\\" + img_name
#img_path = r"C:\Users\Trevor\Documents\Coding Projects\StapleGooder\RawImages\BottleCaps" + "\\" + img_path
raw_img = cv2.imread(img_path,0)


#########################Correct Lighting#########################
local_lighting = cv2.medianBlur(raw_img,101)
img = cv2.addWeighted(raw_img,0.5,-local_lighting,0.5,0)

# kernel = np.ones((1,7), np.uint8)
# img = cv2.dilate(img, kernel, iterations=2)

#########################Edge filter#########################
# edges = cv2.Canny(img,10,80)


######################### Thresholding #########################

#Dynamic Threshold:
# thresh = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
#          cv2.THRESH_BINARY,21,5)

ret,thresh1 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY_INV)
ret,thresh2 = cv2.threshold(img, 135, 255, cv2.THRESH_BINARY)
thresh = cv2.addWeighted(thresh1,1.0,thresh2,1.0,0)
# thresh = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
#          cv2.THRESH_BINARY,21,5)


######################### Dilate and Erode #########################
kernel = np.ones((3,3), np.uint8)
img_erosion = cv2.erode(thresh, kernel, iterations=5)

kernel = np.ones((4,15), np.uint8)
img_dilation = cv2.dilate(img_erosion, kernel, iterations=9)


######################### Blob Detector #########################

scale_factor = .1
img_small = cv2.resize(img_dilation, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)


img_small = 255 - img_small
params = cv2.SimpleBlobDetector_Params()

# # Change thresholds
params.minThreshold = 10
params.maxThreshold = 300

# Filter by Area.
params.filterByArea = True
params.minArea = 200    #remember we're operating on shrunken image

# # Filter by Circularity
params.filterByCircularity = False
# params.minCircularity = 0.000001

# Filter by Convexity
params.filterByConvexity = False
# params.minConvexity = 0.000001

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0
params.maxInertiaRatio = .05

# Create a detector with the parameters
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3:
    detector = cv2.SimpleBlobDetector(params)
else:
    detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs.
keypoints = detector.detect(img_small)
index = 0
# print(dir(keypoints[index]))
# print(keypoints[index].angle)
# print(keypoints[index].class_id)
# print(keypoints[index].octave)
# print(keypoints[index].pt)
# print(keypoints[index].response)
# print(keypoints[index].size)

im_with_keypoints = cv2.drawKeypoints(img_small, keypoints, np.array([]), (0, 0, 255),
                                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

staples = []
for point in keypoints:
    (x,y) = point.pt
    (x,y) = (int(x / scale_factor), int(y / scale_factor))
    size = int(point.size / scale_factor)

    #May want to use fixed size (both for simplicity and standardizing for ML model
    left = x - ( size )
    right = x + ( size )
    top = y-(size//3)
    bottom = y+(size//3)
    # print((x,y,size))
    # print(left)
    # print(right)
    # print(top)
    # print(bottom)
    # print(im_with_keypoints.shape)
    staples.append( img[ top:bottom, left:right ] )

# rect = cv2.minAreaRect(cnt)
# bound_box = cv2.boxPoints(rect)
# bound_box = np.int0(bound_box)
# im = cv2.drawContours(im,[bound_box],0,(0,0,255),2)
# im_with_keypoints = cv2.drawContours(im_with_keypoints,roi,0,(0,0,255),2)


######################### Plot Images #########################

titles = ['Original Image', 'Lighting Correction', 'Thresholding',
           'Dilation', 'Erosion', 'Blob Detection']

images = [raw_img, img, thresh,
          img_erosion, img_dilation, im_with_keypoints]

for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()

staple_directory = r"C:\Users\Trevor\Documents\Coding Projects\StapleGooder\StapleImages" + "\\"
print(staple_directory)
count = len(staples)
for i in range(count):
    # plt.subplot((len(staples)+1)//2, 2, i+1),plt.imshow(staples[i],'gray')
    grid_size = int(count**.5)
    plt.subplot(grid_size, grid_size, i + 1), plt.imshow(staples[i], 'gray')
    plt.xticks([]),plt.yticks([])
    save_path = staple_directory + 'staple%f.png'%time.time()
    cv2.imwrite(save_path, staples[i])
plt.show()