import cv2
import numpy as np
from matplotlib import pyplot as plt

import CardAnalyzer
import Card

CARD_WIDTH = 100    #Width of extracted card images
CARD_HEIGHT = 150   #Height of extracted card images
MIN_CARD_SIZE = 3000   #Minimum pixel count for a blob to be considered a card
MAX_CARD_SIZE = 100000
MIN_CARD_CURVATURE = .10 #min_curvature, aka "epsilon" is the allowable curvature as a fraction of the arclength
BLUR = 11

class ROI:
    def __init__(self, index, contour=None, vertices=None):
        self.index = index
        self.contour = contour
        self.vertices = vertices

class ImageExtractor:
    """
    This class is designed to find contours around cards, isolate them. Using those contours, the cards are isolated
    and projected into standardized rectangular images, which can be further analyzed.
    """

    def __init__(self):
        self.ROIs = []
        self.card_ROIs = []
        self.card_images = []

    ############Math Helper Methods#########
    def _get_direction_degrees(self, pt1, pt2):
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
        shape = image.shape
        resized = cv2.resize(image, (width * shape[1] // shape[0], width))
        return resized

    def detect_cards(self, image):
        """
        Each step in this is so important, I might just break it out to top-level processing
        :param image:
        :return:
        """
        # Filter #1 : create contours around Regions Of Interest(ROIs)
        self.ROIs = self.locate_ROIs(image)

        # Filter #2: Find 4 cornered, card-shaped ROIs (rectangular, good aspect ratio, etc.)
        self.card_ROIs = self.filter_for_rectangles(self.ROIs)

        # Filter #3: Keep only identifiable cards
        self.card_images = self.extract_card_images(image, self.card_ROIs)

        return self.card_images

    def identify_cards(self, images):
        card_analyzer = CardAnalyzer.HandTunedCardAnalyzer()

        cards = []
        for idx, card_image in enumerate(images):
            card = Card.Card(index=idx, image=card_image)
            card = card_analyzer.identify_card(card)
            if card is not None:
                cards.append(card)



        return cards

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
        height, width = gray.shape
        small_gray = cv2.resize(gray, (int(width/10), int(height/10)))
        light_correction = cv2.GaussianBlur(small_gray, (int(width/20)*2+1, int(width/20)*2+1), 0) #Reduce noise in the image
        light_correction = cv2.resize(light_correction, (width, height))
        light_correction = light_correction - np.amin(light_correction)
        # cv2.imwrite("ImageLibrary/PreCorrection.jpg", gray)
        # cv2.imwrite("ImageLibrary/light_correction.jpg", light_correction)

        gray = cv2.subtract(gray, light_correction)
        # cv2.imwrite("ImageLibrary/PostCorrection.jpg", gray)

        blurred = cv2.medianBlur(gray, BLUR, 0) #Reduce noise in the image

        min_thresh = np.amin(blurred)#0
        max_thresh = np.amax(blurred)#255
        ret,thresh_im = cv2.threshold(blurred,min_thresh,max_thresh,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # ret,thresh = cv2.threshold(blurred,min_thresh,max_thresh,cv2.THRESH_OTSU)

        # cv2.imshow("test_thresh", thresh_im)
        # cv2.imwrite("ImageLibrary/card_contour_thresh.jpg", thresh_im)
        # find contours in the thresholded image and initialize the
        # shape detector
        contours = cv2.findContours(thresh_im.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)

        # Quirky behavior in opencv (the magic numbers are major versions)
        major_version = int(cv2.__version__[0])  # I've primarily tested with v4.4.x
        contours = contours[0] if major_version in [2, 4] else contours[1]  # contour formatting is version dependant

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
                vertices = self._order_vertices(vertices)

                area_quadrilateral = \
                (1/2) * ( (vertices[0][0]*vertices[1][1] + vertices[1][0]*vertices[2][1] + vertices[2][0]*vertices[3][1] + vertices[3][0]*vertices[0][1]) \
                        - (vertices[1][0]*vertices[0][1] + vertices[2][0]*vertices[1][1] + vertices[3][0]*vertices[2][1] + vertices[0][0]*vertices[3][1]) )
                area_contour = cv2.contourArea(ROI.contour)
                # print(f"ratios: {area_quadrilateral/area_contour}") #, {area12/area_contour}, {area23/area_contour}, {area34/area_contour}, {area41/area_contour}")

                (x, y, w, h) = cv2.boundingRect(vertices)
                ar = w / float(h)
                # if (.5 < ar < (1 / .5) # This was a loose approx, and the quadrilateral addresses this and more
                if(area_quadrilateral/area_contour < 1.0): #intentionally wide, I've seen valid cards as low as 0.6
                    ROI.vertices = vertices
                    filtered_ROIs.append(ROI)

        return filtered_ROIs

    def extract_card_images(self, image, ROIs):
        """
        :param image:
        :param ROIs:
        :return:
        """

        images = []
        for ROI in ROIs:
            images.append( self._transform_and_extract_image(image, ROI.vertices) )

        return images

    def _order_vertices(self, vertices):
        """
        Purpose: Sort all points in order (based on angle from center of mass)

        Operation: The vertices have a few challenges, that we want to tidy up before proceeding.
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
        # Unwrap inner index (undoing a previous mistake)
        vertices = np.float32(
            [vertices[0][0], vertices[1][0], vertices[2][0], vertices[3][0]])

        # Find the centroid
        center_x = (vertices[0][0] + vertices[1][0] + vertices[2][0] + vertices[3][0]) / 4.
        center_y = (vertices[0][1] + vertices[1][1] + vertices[2][1] + vertices[3][1]) / 4.
        center = np.float32([center_x, center_y])

        # Sorting angles: Order matters when running transforms
        # but I need to associate with vertices...so I use a built in sort and key lookup
        dir_list = []
        angle_dict = {}
        for vertex in vertices:
            dir = self._get_direction_degrees(center, vertex)
            dir_list.append(dir)
            angle_dict[dir] = vertex

        dir_list.sort()

        point = []
        for dir in dir_list:
            point.append(angle_dict[dir])

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
