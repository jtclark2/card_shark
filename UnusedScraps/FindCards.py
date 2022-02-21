#
# import imutils
# import cv2
#
# def find_cards(img_name)
#
#     #filter image
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     ret3,thresh = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#
#     # find contours in the thresholded image
#     cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
#         cv2.CHAIN_APPROX_SIMPLE)
#     cnts = cnts[0] if imutils.is_cv2() else cnts[1] #0 vs 1 based, depending on version of OpenCV
#
#     # loop over the contours
#     MIN_THRESH = 1000;
#     for c in cnts:
#         # compute the center of the contour
#         if cv2.contourArea(c) > MIN_THRESH:
#             #extract
