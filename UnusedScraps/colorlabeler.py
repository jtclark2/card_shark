# # import the necessary packages
# from scipy.spatial import distance as dist
# from collections import OrderedDict
# import numpy as np
# import cv2
#
#
# class ColorLabeler:
#     def __init__(self):
#         # initialize the colors dictionary, containing the color
#         # name as the key and the RGB tuple as the value
#         colors = OrderedDict({
#             "red": (128.08826013513513, 135.97761824324326, 225.00147804054055),
#             "green": (133.45031392076206, 184.2814462004763, 104.38081835895215),
#             "blue": (144.9800404975412, 105.16806479606595, 137.9408446630026)})
#
#         # allocate memory for the L*a*b* image, then initialize
#         # the color names list
#         self.lab = np.zeros((len(colors), 1, 3), dtype="uint8")
#         self.colorNames = []
#
#         # loop over the colors dictionary
#         for (i, (name, rgb)) in enumerate(colors.items()):
#             # update the L*a*b* array and the color names list
#             self.lab[i] = rgb
#             self.colorNames.append(name)
#
#         # convert the L*a*b* array from the RGB color space
#         # to L*a*b*
#         # self.lab = cv2.cvtColor(self.lab, cv2.COLOR_RGB2LAB)
#
#     def label(self, image, c):
#         # construct a mask for the contour, then compute the
#         # average L*a*b* value for the masked region
#         mask = np.zeros(image.shape[:2], dtype="uint8")
#         cv2.drawContours(image = mask,
#                          contours = c,
#                          contourIdx = -1,
#                          color = 255,
#                          thickness = 2)
#
#         # mask = cv2.erode(mask, None, iterations=2)
#         mean = cv2.mean(image, mask=mask)[:3]
#         # initialize the minimum distance found thus far
#         minDist = (np.inf, None)
#
#         # loop over the known L*a*b* color values
#         for (i, row) in enumerate(self.lab):
#             # compute the distance between the current L*a*b*
#             # color value and the mean of the image
#             d = dist.euclidean(row[0], mean)
#             # if the distance is smaller than the current distance,
#             # then update the bookkeeping variable
#             if d < minDist[0]:
#                 minDist = (d, i)
#         # return the name of the color with the smallest distance
#         return self.colorNames[minDist[1]]


# ### Display the outline in the matched hue (for development/testing only)
# for i in range(0,270,10):
#     swatch =  np.zeros((150, 250, 3), np.uint8)
#     swatch[:,:,V] = 255.
#     swatch[:,:,H] = i
#     swatch[:,:,S] = 255. # Saturating the color helps a ton, especially for the striped/empty shapes (use before converting back to BGR)
#     swatch = cv2.cvtColor(swatch, cv2.COLOR_HSV2BGR)
#     import random
#     cv2.imshow(f"Swatch-{i}", swatch)