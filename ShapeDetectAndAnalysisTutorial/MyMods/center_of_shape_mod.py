###############################################################################
# Simple Image processing tutorial, based on:
# https://www.pyimagesearch.com/2016/02/01/opencv-center-of-contour/
#
#
###############################################################################


# import the necessary packages
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
#                 help="path to the input image")

#args = vars(ap.parse_args())

# Easier to set locally while testing
# args = {"image" : "shapes_and_colors.jpg"}

args = {"image" : "IMG_6394.JPG"}

# load the image, convert it to grayscale, blur it slightly,
# and threshold it
image = cv2.imread(args["image"])
#cv2.imshow("Image", image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#cv2.imshow("gray", gray)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#cv2.imshow("blurred", blurred)

# Having complex and possibly changing backgrounds, I'm applying an adaptive threshold
# Otsu's threshold basically using the image histogram, to attempt to split a bimodal distribution

# thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
ret3,thresh = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

cv2.imshow("thresh", thresh)

# find contours in the thresholded image
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1] #0 vs 1 based, depending on version of OpenCV

# loop over the contours
MIN_THRESH = 1000;
for c in cnts:
    # compute the center of the contour
    if cv2.contourArea(c) > MIN_THRESH:
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        # draw the contour and center of the shape on the image
        cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
        cv2.circle(image, (cX, cY), 7, (255, 255, 255), -1)
        cv2.putText(image, "center", (cX - 20, cY - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # show the image
        cv2.imshow("Image", image)
        cv2.waitKey(0)