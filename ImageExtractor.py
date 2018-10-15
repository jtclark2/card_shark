import cv2
import numpy as np
import imutils
from matplotlib import pyplot as plt

import Card

CARD_WIDTH = 400    #Width of extracted card images
CARD_HEIGHT = 600   #Height of extracted card images
MIN_CARD_SIZE = 10000   #Minimum pixel count for a blob to be considered a card
MAX_CARD_SIZE = 100000
MIN_CARD_CURVATURE = .10 #min_curvature, aka "epsilon" is the allowable curvature as a fraction of the arclength
BLUR = 11

class ROI:
    def __init__(self, index, contour=None, vertices=None):
        self.index = index
        self.contour = contour
        self.vertices = vertices

class ImageExtractor:

    ############Math Helper Methods#########
    def _get_angle(self, pt1, pt2):
        x = pt2[0] - pt1[0]
        y = pt2[1] - pt1[1]
        if (x == 0):
            if (y > 0):
                return 90
            else:
                return 270
        ang = np.rad2deg(np.arctan(y / x))
        if (x > 0):
            return ang
        else:
            if (y > 0):
                return 180 + ang
            else:
                return -180 + ang

    def _get_distance(self, pt1, pt2):
        x = pt2[0] - pt1[0]
        y = pt2[1] - pt1[1]
        return (x ** 2 + y ** 2) ** .5

    def get_histogram(self, image):
        # hist = cv2.calcHist([image], [0], None, [255], [0, 255])

        plt.hist(image.ravel(), 255, [0, 255]);
        plt.show(False)
        pass

    # Image search and extraction pipeline
    def pre_process_image(self, image, width=None):
        """
        Operation: pull image into memory and pre-process:
            -standardize input size
            -Future: color space?
        :param img_path: path to image
        :param size: desired output width of image
        :return: scaled image
        """

        # self.get_histogram(image)
        resized = imutils.resize(image, width=width)
        return resized

    def locate_ROIs(self, image):
        """
        Operation: Finds cards, and extracts the contours, which are then
            simplified into approximate polygons, which are returns
        :param image: The input image to be processed.
        :return: List of simplified polygons. These polygons are the corners of the cards.
        """

        # Gaussian blur is preferred for it's local smoothness, but it is VERY slow
        # Rescale image and operate in low res then scale back up to apply
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # light_correction = cv2.medianBlur(gray, 201, 0) #Correct aberation in image (brighter in center)
        height, width = gray.shape
        small_gray = cv2.resize(gray, (int(width/10), int(height/10)))
        light_correction = cv2.GaussianBlur(small_gray, (int(width/20)*2+1, int(width/20)*2+1), 0) #Reduce noise in the image
        light_correction = cv2.resize(light_correction, (width, height))
        light_correction = light_correction - np.amin(light_correction)

        #Todo: Division makes more sense, but gives us a type mistmatch - resolve later
        gray = cv2.subtract(gray, light_correction)

        #TODO: applying on a per card basis instead...seems effective so far. Delete this if performance is good
        # # im = [image[:,:,i] for i in range(3)] #Would allow iteration by channel
        # red = image[:,:,0]
        # green = image[:,:,1]
        # blue = image[:,:,2]
        # new_image = np.zeros(image.shape)
        # for (i, color) in enumerate([red,green,blue]):
        #     light_correction = cv2.medianBlur(gray, 801, 0)  # Reduce noise in the image
        #     light_correction = light_correction - np.amin(light_correction)
        #     ch_corr = cv2.subtract(gray, light_correction)
        #     new_image[:,:,i] = ch_corr


        blurred = cv2.medianBlur(gray, BLUR, 0) #Reduce noise in the image

        resized = imutils.resize(blurred, width=600)
        # cv2.imshow("Blurred Image", resized)

        min_thresh = np.amin(resized)#0
        max_thresh = np.amax(resized)#255
        ret,thresh_im = cv2.threshold(blurred,min_thresh,max_thresh,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # ret,thresh = cv2.threshold(blurred,min_thresh,max_thresh,cv2.THRESH_OTSU)

        # cv2.imshow("test_thresh", imutils.resize(thresh, width=600))
        # find contours in the thresholded image and initialize the
        # shape detector
        contours = cv2.findContours(thresh_im.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if imutils.is_cv2() else contours[1]

        ROIs = []
        self.filtered_contours = []
        for idx, contour in enumerate(contours):
            if MAX_CARD_SIZE <= cv2.contourArea(contour) or cv2.contourArea(contour) >= MIN_CARD_SIZE:
                ROIs.append( ROI(index=idx, contour=contour) )
        return ROIs

    def filter_for_rectangles(self, ROIs):
        """
        Keep it simple: Anything with 4 corners is a card
        """
        filtered_ROIs = []
        for ROI in ROIs:
            perimeter = cv2.arcLength(ROI.contour, True)
            vertices = cv2.approxPolyDP(ROI.contour, MIN_CARD_CURVATURE * perimeter, True)
            if (len(vertices) == 4):
                (x, y, w, h) = cv2.boundingRect(vertices)
                ar = w / float(h)
                if(.5 < ar and ar < (1/.5)): #intentionally wide, I've seen valid cards as low as 0.6
                    ROI.vertices = vertices
                    filtered_ROIs.append(ROI)

        return filtered_ROIs

    def extract_card_images(self, image, ROIs):
        """
        :param image:
        :param ROIs:
        :return:
        """
        # TODO: Add unit test to ensure that order is maintained, such that
        # ROI[idx] matches images[idx]

        images = []
        for ROI in ROIs:
            vertices = self._order_vertices(ROI.vertices)
            images.append( self._transform_and_extract_image(image, vertices) )

        return images

    def _order_vertices(self, vertices):
        """
        Opertaion: The vertices have a few challenges, that we want to tidy up before proceeding.
            1) There is an extra layer of indexing (and it's a middle layer, not outer). I need to
                got back and clean it up, but it was noted in this v2 vs v3 conditional, that
                is not worth digging into right now (maybe ever).
            2) Order is not gauranteed (and I need a very specific order for a transform I'll be using.
                Order, in this case, means counter-clockwise around the moment of inertia. Messing this up
                inverts the transforms, making them look like the world has been
                flipped inside out.
            3) Starting point: Related to order, I want to start with 2 points that
                make up a short side of the card. Orientation may matter

        More Conext:
            # Order of input points and output points needs to be consistent...
            # I'm making the hardcoded output points go counter-clockwise, so we need to figure out the clockwise input
            # Furthermore, we want the card to finish with a longer vertical aspect, so we need to start with one
            # of the shorter sides of the rectangle (don't worry about upside-down, because it's symmetric)

        :param vertices: 4 vertices of the card image
        :return: vertices (similar to the input, just better)
        """
        # TODO: Cleanup - avoid wrapping the extra layer in the first place (may be introduced by lib
        # compatibility issue

        # Unwrap inner index (undoing a previous mistake)
        vertices = np.float32(
            [vertices[0][0], vertices[1][0], vertices[2][0], vertices[3][0]])

        # Find the centroid
        center_x = (vertices[0][0] + vertices[1][0] + vertices[2][0] + vertices[3][0]) / 4.
        center_y = (vertices[0][1] + vertices[1][1] + vertices[2][1] + vertices[3][1]) / 4.
        center = np.float32([center_x, center_y])

        # Sorting angles: Order matters when running transforms
        # but I need to associate with vertices...so I use a built in sort and key lookup
        angle_list = []
        angle_dict = {}
        for vertex in vertices:
            angle = self._get_angle(center, vertex)
            angle_list.append(angle)
            angle_dict[angle] = vertex

        angle_list.sort()

        point = []
        for angle in angle_list:
            point.append(angle_dict[angle])

        side1 = self._get_distance(point[0], point[1])
        side2 = self._get_distance(point[1], point[2])

        # Start rotating so that the line drawn from pt1 to pt2 is a short side
        if (side1 < side2):
            vertices = np.float32(
                [point[0], point[1], point[2], point[3]])
        else:
            vertices = np.float32(
                [point[1], point[2], point[3], point[0]])


        return vertices

    def _transform_and_extract_image(self, image, input_vertices):

        output_vertices = np.float32([[0, 0], [CARD_WIDTH, 0], [CARD_WIDTH, CARD_HEIGHT], [0, CARD_HEIGHT]])

        M = cv2.getPerspectiveTransform(input_vertices, output_vertices)
        card_img = cv2.warpPerspective(image, M, (CARD_WIDTH, CARD_HEIGHT))
        return card_img

    def display_ROIs(self, image, ROIs, color=[255, 0, 0], line_thickness=1):
        contours = [ROI.contour for ROI in ROIs]
        cv2.drawContours(image=image,
                         contours=contours,
                         contourIdx= -1,
                         color=color,
                         thickness=line_thickness)
        # image = imutils.resize(image, width=600)
        # cv2.imshow("Rich Diagnostic View", image)

    #Visualization tools that apply to the original input image (not to individually extracted cards)
    def display_cards(self, card_collection, image, ROIs, color=[255, 0, 0], line_thickness=1, label = True, flip=False):
        contours = [ROI.contour for ROI in ROIs]
        for card in card_collection:
            cv2.drawContours(image=image,
                             contours=contours,
                             contourIdx=card.index,
                             color=color,
                             thickness=line_thickness)
            if label:
                image = self.annotate_card(image, card, contours[card.index])
        # image = imutils.resize(image, width=1000)
        if(flip):
            image = cv2.flip(image, -1)
        # cv2.imshow("Rich Diagnostic View", image)

    def display_key(self, image, key_dict, display_width = 800):
        image = imutils.resize(image, width=display_width)
        X = 20
        Y = 20
        for key in key_dict:
            Y += 20
            cv2.putText(image, key, (X, Y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, key_dict[key], 2)
        cv2.imshow("Rich Diagnostic View", image)

    def display_extracted_images(self, images):
        # Displays first 12 images extracted (or as many as are available)
        i=1
        for image in images:
            #plot
            (rows, columns) = (4,3)
            plt.subplot(rows, columns, i), plt.imshow(image)
            i+=1
            if(i > 12):
                break
        plt.show(False)

    def annotate_card(self, image, card, contour):
        cX_offset = -20
        cY_offset = -45

        M = cv2.moments(contour)
        cX = int(M['m10']/M['m00']) + cX_offset
        cY = int(M['m01']/M['m00']) + cY_offset

        cv2.putText(image, repr(card.count), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 2)
        cv2.putText(image, repr(card.shape), (cX, cY + 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 2)
        cv2.putText(image, repr(card.fill), (cX, cY + 20 * 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 2)
        cv2.putText(image, repr(card.color), (cX, cY + 20 * 3), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 2)

        return image