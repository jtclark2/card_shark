import cv2
import numpy as np
from matplotlib import pyplot as plt
import time

# This detector is fairly specific, though I intend to eventually generalize,
# once I get a better feel for the error in the various detectors I'm creating

#########################Read Raw#########################
img_name = "IMG_5924.JPG"
img_path = r".\..\RawImages\BottleCaps" + "\\" + img_name
raw_img = cv2.imread(img_path,0)


#########################smooth and light adjust #########################
img = cv2.medianBlur(raw_img,49)
local_lighting = cv2.medianBlur(raw_img,251)

img = cv2.addWeighted(img,0.5,-local_lighting,0.5,0)
kernel = np.ones((1,7), np.uint8)
#########################Edge filter#########################
# edges = cv2.Canny(img,10,80)


######################### Thresholding #########################

#Dynamic Threshold:
# thresh = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
#          cv2.THRESH_BINARY,21,5)

ret,thresh = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY_INV)
# ret,thresh2 = cv2.threshold(img, 145, 255, cv2.THRESH_BINARY)
# thresh = cv2.addWeighted(thresh1,1.0,thresh2,1.0,0)
# thresh = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
#          cv2.THRESH_BINARY,21,5)


######################### Dilate and Erode #########################
img_dilation = thresh

for i in range(5):
    kernel = np.ones((5,5), np.uint8)
    img_erosion = cv2.erode(img_dilation, kernel, iterations=2)

    kernel = np.ones((5,5), np.uint8)
    img_dilation = cv2.dilate(img_erosion, kernel, iterations=3)

kernel = np.ones((5, 5), np.uint8)
img_dilation = cv2.dilate(img_erosion, kernel, iterations=6)


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
params.minArea = 100    #remember we're operating on shrunken image

# # Filter by Circularity
params.filterByCircularity = False
# params.minCircularity = 0.000001

# Filter by Convexity
params.filterByConvexity = False
# params.minConvexity = 0.000001

# Filter by Inertia
params.filterByInertia = False
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

objects = []
raw_width = height = raw_img.shape[0]
raw_height = raw_img.shape[1]

for point in keypoints:
    (x,y) = point.pt
    (x,y) = (int(x / scale_factor), int(y / scale_factor))
    size = int(point.size / scale_factor)

    #May want to use fixed size (both for simplicity and standardizing for ML model
    size = 100
    left = x - ( size )
    right = x + ( size )
    top = y-(size)
    bottom = y+(size)

    if(left < 0):
        print('*' * 10 + ' : left' + str( (x,y) ) )
        left = 0
        right = left + size*2
    if(right > raw_width):
        print('*' * 10 + ' : right' + str( (x,y) ) )
        right = raw_width
        left = right - size*2

    if(bottom < 0):
        print('*' * 10 + ' : bottom' + str( (x,y) ) )
        bottom = 0
        top = bottom+size*2
    if(top > raw_height):
        print('*' * 10 + ' : top' + str( (x,y) ) )
        top = raw_height
        bottom = top - size*2

    # print('*'*80)
    # print((x,y,size))
    # print(left)
    # print(right)
    # print(top)
    # print(bottom)
    # print(im_with_keypoints.shape)
    objects.append( raw_img[ top:bottom, left:right ] )

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
count = len(objects)
print(count)
for i in range(count):
    # plt.subplot((len(staples)+1)//2, 2, i+1),plt.imshow(staples[i],'gray')
    grid_size = int(count**.5)+1
    plt.subplot(grid_size, grid_size, i+1), plt.imshow(objects[i], 'gray')
    plt.xticks([]),plt.yticks([])
    save_path = staple_directory + 'staple%f.png'%time.time()
    cv2.imwrite(save_path, objects[i])
plt.show()