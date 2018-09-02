import cv2
import numpy as np
import imutils

CARD_WIDTH = 400    #Width of extracted card images
CARD_HEIGHT = 600   #Height of extracted card images
MIN_CARD_SIZE = 1000   #Minimum pixel count for a blob to be considered a card
MIN_CARD_CURVATURE = .10 #min_curvature, aka "epsilon" is the allowable curvature as a fraction of the arclength
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

    #Image search and extraction
    def get_image(self, img_path, width=None):
        """
        Operation: pull image into memory and pre-process, if requested
        :param img_path: path to image
        :param size: desired output width of image
        :return: scaled image
        """

        image = cv2.imread(img_path)
        resized = imutils.resize(image, width=width)
        return resized

    def find_cards(self, image):
        """
        Operation: Finds cards, and extracts the contours, which are then
            simplified into approximate polygons, which are returns
        :param image: The input image to be processed.
        :return: List of simplified polygons. These polygons are the corners of the cards.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        ret,thresh = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # find contours in the thresholded image and initialize the
        # shape detector
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if imutils.is_cv2() else contours[1]

        all_cards = []
        filtered_contours = []
        for c in contours:
            if cv2.contourArea(c) >= MIN_CARD_SIZE:
                filtered_contours.append(c)
                perimeter = cv2.arcLength(c, True)
                vertices = cv2.approxPolyDP(c, MIN_CARD_CURVATURE * perimeter, True)

                # Card better have 4 sides, or we messed up
                # In this case, don't return the bad value...
                # But we also shouldn't throw an exception for one missing card
                if (len(vertices) != 4):
                    # print("Invalid attempt to detect card with %d vertices." % len(vertices))
                    continue

                all_cards.append( self._cleanup_vertices(vertices) )

        return all_cards, filtered_contours

    def _cleanup_vertices(self, vertices):
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

    def transform_and_extract_image(self, image, input_vertices):

        output_vertices = np.float32([[0, 0], [CARD_WIDTH, 0], [CARD_WIDTH, CARD_HEIGHT], [0, CARD_HEIGHT]])

        M = cv2.getPerspectiveTransform(input_vertices, output_vertices)
        card_img = cv2.warpPerspective(image, M, (CARD_WIDTH, CARD_HEIGHT))

        return card_img
