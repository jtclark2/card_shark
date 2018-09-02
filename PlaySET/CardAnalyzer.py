from colorlabeler import ColorLabeler
from Card import *

import cv2
import numpy as np
import imutils

#min_curvature, aka "epsilon" is the allowable curvature as a fraction of the arclength
MIN_SHAPE_CURVATURE = .02
MIN_SHAPE_SIZE = 500    #TODO : add max shape size too (sometimes the shapes get combined in blurry images)

class CardAnalyzer:
    def identify_card(self, image):
        """
        Operation: Identify the properties of the incoming card image
        :param image: The input image to be processed.
        :return: Card, with appropriately defined color, shape, fill, and count
        """

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        shape = gray.shape
        mask = np.zeros(shape, np.uint8)

        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if imutils.is_cv2() else contours[1]

        SYMBOL_SIZE_THRESH = 1000

        contours = [c for c in contours if cv2.contourArea(c) >= SYMBOL_SIZE_THRESH]

        count = self.identify_count(contours)
        shape = self.identify_shape(contours)
        fill = self.identify_fill(gray, mask, contours)
        color = self.identify_color(image, mask, contours)

        if(count is None or
           shape is None or
           fill is None or
           color is None):
            # print("Found 4 sided object that didn't look like a card.")
            return None
        return Card(shape, color, count, fill)

    def identify_count(self, contours):
        count = len(contours)
        if count == 1:
            return Count.one
        elif count == 2:
            return Count.two
        elif count == 3:
            return Count.three
        else:
            return

    def identify_shape(self, contours):
        # TODO: Each card has 1, 2, or 3 symbols. We could repeat on each and copmare, but lets start simple
        shape = None
        total_area = 0
        for c in contours:
            if cv2.contourArea(c) >= MIN_SHAPE_SIZE:

                #Shape Contour Metrics
                area = cv2.contourArea(c)
                total_area += area

                perimeter = cv2.arcLength(c, True)
                # vertices = cv2.approxPolyDP(c, MIN_SHAPE_CURVATURE * perimeter, True)
                hull = cv2.convexHull(c)
                convex_vertices = cv2.approxPolyDP(hull, MIN_SHAPE_CURVATURE * perimeter, True)
                # area_convex = cv2.contourArea(hull)

                ###Find Shape
                if (len(convex_vertices) == 4):
                    shape = Shape.diamond
                elif ( (perimeter / area) > .027 ):  #TODO: currently dependant on image scale
                    shape = Shape.wisp
                else:
                    shape = Shape.stadium

        return shape

    def identify_fill(self, gray, mask, contours):
        # stripes have std dev of at least 9.9 in my small sample size, and all others are < 3.4
        # Let's start with a thresh of 7

        # Similar process with lighting, though variations in intensity may become a problem...
        # Second pass, I could normalize off the outer edge of the card (which is always empty)

        # cv2.fillPoly(mask, pts=contours, color=(1))
        cv2.drawContours(mask, contours, -1, 255, -1)
        mask = cv2.erode(mask, None, iterations=10)  # Avoid the edge...just want the texture

        (mean, std_dev,) = cv2.meanStdDev(gray, mask=mask)

        ###Find Fill
        if (std_dev > 7):
            fill = Fill.striped
        elif (mean > 160):
            fill = Fill.empty
        else:
            fill = Fill.solid

        return fill

    def identify_color(self, image, mask, contours):


        cl = ColorLabeler()
        color = cl.label(image, contours)

        if(color == "red"):
            return Color.red
        if(color == "green"):
            return Color.green
        if(color == "blue"):    #Todo: Yeah, I need to remap (2 wrongs make a right...kind of)
            return Color.purple
