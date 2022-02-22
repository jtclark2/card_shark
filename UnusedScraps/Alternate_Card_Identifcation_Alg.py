### The following methods were my first attempts...I got too sentimental to delete them
# def _identify_count(self, contours):
#     """
#     Simply count the contours found.
#     :param contours:
#     :return:
#     """
#     count = len(contours)
#     if count == 1:
#         return Count.one
#     elif count == 2:
#         return Count.two
#     elif count == 3:
#         return Count.three
#     else:
#         return
#
# def _identify_shape(self, contours):
#     MIN_SHAPE_CURVATURE = .02
#     MIN_SHAPE_SIZE = 10000
#     shape = None
#     total_area = 0
#     for c in contours:
#         if cv2.contourArea(c) >= MIN_SHAPE_SIZE:
#
#             #Shape Contour Metrics
#             area = cv2.contourArea(c)
#             total_area += area
#
#             perimeter = cv2.arcLength(c, True)
#             # vertices = cv2.approxPolyDP(c, MIN_SHAPE_CURVATURE * perimeter, True)
#             hull = cv2.convexHull(c)
#             convex_vertices = cv2.approxPolyDP(hull, MIN_SHAPE_CURVATURE * perimeter, True)
#             # area_convex = cv2.contourArea(hull)
#
#             ###Find Shape
#             if (len(convex_vertices) == 4):
#                 shape = Shape.diamond
#             elif ( (perimeter / area) > .027 ):  #TODO: currently dependant on image scale
#                 shape = Shape.wisp
#             else:
#                 shape = Shape.stadium
#
#     return shape
#
# def _identify_fill(self, gray, mask, contours):
#     # stripes have std dev of at least 9.9 in my small sample size, and all others are < 3.4
#     # Let's start with a thresh of 7
#
#     # Similar process with lighting, though variations in intensity may become a problem...
#     # Second pass, I could normalize off the outer edge of the card (which is always empty)
#
#     mask = cv2.erode(mask, None, iterations=10)
#     (mean, std_dev,) = cv2.meanStdDev(gray, mask=mask)
#
#     ###Find Fill
#     # print(mean, std_dev)
#     if (std_dev <10):
#         fill = Fill.solid
#     elif (mean < 20):
#         fill = Fill.striped
#     else:
#         fill = Fill.empty
#
#     return fill
#
#     # cv2.imshow("mask", inner_mask)
#     # cv2.imshow("inv_mask", outer_mask)
#
#
#
#     # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     # (total_mean, total_std_dev) = cv2.meanStdDev(gray)
#     # (inner_mean, inner_std_dev) = cv2.meanStdDev(gray, mask=outer_mask)
#     # (outer_mean, outer_std_dev) = cv2.meanStdDev(gray, mask=inner_mask)
#     # # print( inner_mean / outer_mean )
#     #
#     # relative_luminosity = inner_mean / outer_mean
#     # if (relative_luminosity > .97):
#     #     fill = Fill.empty
#     # elif (relative_luminosity > .75):
#     #     fill = Fill.striped
#     # else:
#     #     fill = Fill.solid
#     # # print(relative_luminosity)
#
# def _identify_color(self, image, mask, contours):
#
#     cl = ColorLabeler()
#     color = cl.label(image, contours)
#
#     if(color == "red"):
#         return Color.red
#     if(color == "green"):
#         return Color.green
#     if(color == "blue"):    #Todo: Yeah, I need to remap (2 wrongs make a right...kind of)
#         return Color.purple