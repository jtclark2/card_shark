# References that made this possible
#
# Image Processing:
# https://www.pyimagesearch.com/2016/02/08/opencv-shape-detection/
# https://www.pyimagesearch.com/2016/02/08/opencv-shape-detection/
# https://www.pyimagesearch.com/2016/02/15/determining-object-color-with-opencv/
#
# Thresholding:
# https://docs.opencv.org/3.4.0/d7/d4d/tutorial_py_thresholding.html
#
# Contours:
# https://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html
#
# Transform / Projection
# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html
#
# Histograms (didn't end up using this, but helped to investigate what was happening)
# https://docs.opencv.org/3.1.0/d1/db7/tutorial_py_histogram_begins.html
#
# And the opencv docs, which humble you, because...
# After spending hours masking and learning to run stats on images...turns out this one liner did it :)
# https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#void%20meanStdDev(InputArray%20src,%20OutputArray%20mean,%20OutputArray%20stddev,%20InputArray%20mask)
# https://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=drawcontours
# https://docs.opencv.org/3.0-beta/modules/refman.html

# import the necessary packages

# import sys
# sys.path.insert(0, r'C:\Users\Trevor\Documents\Coding Projects\ImageEvaluator\ShapeDetectAndAnalysisTutorial')
# import argparse
from colorlabeler import ColorLabeler
import imutils
import cv2
import numpy as np
from matplotlib import pyplot as plt
from enum import Enum


IMG_PATH = "IMG_6394.JPG"
# IMG_PATH = "IMG_6438.JPG"
# IMG_PATH = "IMG_6439.JPG"
# IMG_PATH = "IMG_6440.JPG"
# IMG_PATH = "IMG_6441.JPG"
IMG_PATH = "WebCam1.PNG"

MIN_THRESH = 1000   #Minimum pixel count for a blob to be considered a card
INPUT_WIDTH = None #width of image read in (use None to keep original size)
CARD_WIDTH = 400    #Width of extracted card images
CARD_HEIGHT = 600   #Height of extracted card images

#min_curvature, aka "epsilon" is the allowable curvature as a fraction of the arclength
# used when finding the vertices of the card
MIN_CARD_CURVATURE = .10
MIN_SHAPE_CURVATURE = .02

#############Classes#################
class Shape(Enum):
    diamond = 1
    stadium = 2 # I looked it up, and it's a a proper name (also discorectanble, and obround)
    wisp    = 3

    def __repr__(self):
        """Make Printing less noisy"""
        return str(self)[6:]+"-"

class Color(Enum):
    red     = 1
    green   = 2
    purple  = 3

    def __repr__(self):
        """Make Printing less noisy"""
        return str(self)[6:]+"-"

class Count(Enum):
    one     = 1
    two     = 2
    three   = 3

    def __repr__(self):
        """Make Printing less noisy"""
        return str(self)[6:]+"-"

class Fill(Enum):
    solid   = 1
    striped = 2
    empty   = 3

    def __repr__(self):
        """Make Printing less noisy"""
        return str(self)[5:]+"-"

class Card:
    def __init__(self, shape, color, count, fill, index = None):
        self.shape = shape
        self.color = color
        self.count = count
        self.fill = fill
        self.index = index

    def __repr__(self):
        return (repr(self.shape) + repr(self.color) + repr(self.count) + repr(self.fill))


############Helper Methods#########
def get_angle(pt1, pt2):
    x = pt2[0] - pt1[0]
    y = pt2[1] - pt1[1]
    if(x == 0):
        if(y > 0):
            return 90
        else:
            return 270
    ang = np.rad2deg( np.arctan(y/x) )
    if(x>0):
        return ang
    else:
        if(y>0):
            return 180 + ang
        else:
            return -180 + ang

def get_distance(pt1, pt2):
    x = pt2[0] - pt1[0]
    y = pt2[1] - pt1[1]
    return (x**2 + y**2)**.5

#########Core Methods##############
def get_image(img_path, width=None):
    """
    Operation: pull image into memory and pre-process, if requested
    :param img_path: path to image
    :param size: desired output width of image
    :return: scaled image
    """

    image = cv2.imread(img_path)
    resized = imutils.resize(image, width=width)
    return resized

def find_cards(image):
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
        if cv2.contourArea(c) >= MIN_THRESH:
            filtered_contours.append(c)
            perimeter = cv2.arcLength(c, True)
            vertices = cv2.approxPolyDP(c, MIN_CARD_CURVATURE * perimeter, True)

            # Card better have 4 sides, or we messed up
            # In this case, don't return the bad value...
            # But we also shouldn't throw an exception for one missing card
            if (len(vertices) != 4):
                # print("Invalid attempt to detect card with %d vertices." % len(vertices))
                continue

            all_cards.append( cleanup_vertices(vertices))

    return all_cards, filtered_contours

def cleanup_vertices(vertices):
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

    # Sorting angles, but I need to associate with vertices...so I use a built in sort and key lookup
    angle_list = []
    angle_dict = {}
    for vertex in vertices:
        angle = get_angle(center, vertex)
        angle_list.append(angle)
        angle_dict[angle] = vertex

    angle_list.sort()

    point = []
    for angle in angle_list:
        point.append(angle_dict[angle])

    side1 = get_distance(point[0], point[1])
    side2 = get_distance(point[1], point[2])

    # Start rotating so that the line drawn from pt1 to pt2 is a short side
    if (side1 < side2):
        vertices = np.float32(
            [point[0], point[1], point[2], point[3]])
    else:
        vertices = np.float32(
            [point[1], point[2], point[3], point[0]])


    return vertices

def transform_and_extract_image(image, input_vertices):

    output_vertices = np.float32([[0, 0], [CARD_WIDTH, 0], [CARD_WIDTH, CARD_HEIGHT], [0, CARD_HEIGHT]])

    M = cv2.getPerspectiveTransform(input_vertices, output_vertices)
    card_img = cv2.warpPerspective(image, M, (CARD_WIDTH, CARD_HEIGHT))

    return card_img

def identify_card(image):
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

    count = identify_count(contours)
    shape = identify_shape(contours)
    fill = identify_fill(gray, mask, contours)
    color = identify_color(image, mask, contours)

    if(count is None or
       shape is None or
       fill is None or
       color is None):
        # print("Found 4 sided object that didn't look like a card.")
        return None
    return Card(shape, color, count, fill)

def identify_count(contours):
    count = len(contours)
    if count == 1:
        return Count.one
    elif count == 2:
        return Count.two
    elif count == 3:
        return Count.three
    else:
        return

def identify_shape(contours):
    # TODO: Each card has 1, 2, or 3 symbols. We could repeat on each and copmare, but lets start simple
    shape = None
    total_area = 0
    for c in contours:
        if cv2.contourArea(c) >= MIN_THRESH:

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

def identify_fill(gray, mask, contours):
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

def identify_color(image, mask, contours):


    cl = ColorLabeler()
    color = cl.label(image, contours)

    if(color == "red"):
        return Color.red
    if(color == "green"):
        return Color.green
    if(color == "blue"):    #Todo: Yeah, I need to remap (2 wrongs make a right...kind of)
        return Color.purple

def find_sets(cards):
    # Start brute force (this game is intentionally computationally complex...but for humans, not computers)
    # Todo: Future optimization-we can get a bit clever with our patterns if this proves computationally straining...
    #   Certain cards can be eliminated from the pool, by virtue of having certain attributes, without
    #   having to check all possibilities
    i = 0
    Sets = []
    for idx1, card1 in enumerate(cards, start=0):
        for idx2, card2 in enumerate(cards[idx1 + 1:], start=idx1 + 1):
            for idx3, card3 in enumerate(cards[idx2 + 1:], start=idx2 + 1):
                i += 1
                if (check_set(card1, card2, card3)):
                    Sets.append(set([card1, card2, card3]))
                    valid = True
                else:
                    valid = False
    return Sets

def check_set(card1, card2, card3):

    # This conditional is intentionally awkward to intuit (that's why the game is fun)
    attributes = ['shape', 'color', 'count', 'fill']

    for attribute in attributes:
        all_same_condition =   (getattr(card1, attribute) == getattr(card2, attribute) and
                                getattr(card1, attribute) == getattr(card3, attribute) )
        all_different_condition =  (getattr(card1, attribute) != getattr(card2, attribute) and
                                    getattr(card1, attribute) != getattr(card3, attribute) and
                                    getattr(card2, attribute) != getattr(card3, attribute) )
        if(all_same_condition or all_different_condition):
            pass    #need all attributes to be True for a set
        else:
            return False

    #All attributes are True
    return True

def display_sets_text(sets):
    for s in sets:
        print(s)

def display_cards(single_set, image, contours, color = [255,0,0], line_thickness = 1):
        # x = [card.index for card in sets[0]]
        for card in single_set:
            cv2.drawContours(image=image,
                             contours=contours,
                             contourIdx=card.index,
                             color=color,
                             thickness=line_thickness)
        resized = imutils.resize(image, width=600)
        cv2.imshow("Image", resized)
        # cv2.waitKey(0)

def display_card(image, card):
    cX = 0
    cY = 15
    cv2.putText(image, repr(card.count), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 2)
    cv2.putText(image, repr(card.shape), (cX, cY + 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 2)
    cv2.putText(image, repr(card.fill), (cX, cY + 20*2), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 2)
    cv2.putText(image, repr(card.color), (cX, cY + 20*3), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 2)

    cv2.imshow("Image", image)
    cv2.waitKey(0)
    return

def process_image(image):
    (all_card_vertices, card_contours) = find_cards(image)
    i = 1
    cards = []
    card_images = [];
    for idx, card_vertices in enumerate(all_card_vertices):
        card_image = transform_and_extract_image(image, card_vertices)
        card_images.append(card_image)
        card = identify_card(card_image)
        if card is None:
            continue
        card.index = idx
        # display_card(card_image, card)
        cards.append(card)

    sets = find_sets(cards)

    display_sets_text(sets)
    if( len(sets) > 0):
        display_cards(sets[0], image, card_contours, color = [ 0, 255, 0], line_thickness = 5)
    else:
        display_cards(cards, image, card_contours, color=[255, 0, 0], line_thickness=5)
        cv2.imshow('frame',image)

    #Finish the first for loop before getting to plotting (easier for debug)
    # for card_image in card_images:
    #     #plot
    #     (rows, columns) = (4,3)
    #     plt.subplot(rows, columns, i), plt.imshow(card_image)
    #     i+=1
    # plt.show()


##################Main######################
mode = "run"
if mode == "debug":
    # process image from file (controlled, testing)
    import time
    t0 = time.time()

    #Get image from file
    image = get_image(IMG_PATH, width=INPUT_WIDTH)
    process_image(image)
    print("Image processed in: %f seconds" % (time.time() - t0))
else:
    # Process from webcam (doin' it live!)

    cap = cv2.VideoCapture(0)
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        process_image(frame)

        # Display the resulting frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()







def other_code_snippets():
    """
    Not wanting to throw out many of the useful methods snippets I wrote / found.
    I'll probably wipe these out when I'm more comfortable with OpenCV, but they've
    been great learning tools so far, and helpful reference
    """
    # cv2.imshow("Image", image)
    # cv2.waitKey(0)

    #############Color###############
    # color_mask = cv2.bitwise_and(image, image, mask=mask)
    #
    # red = image.copy()
    # # set blue and green channels to 0
    # red[:, :, 0] = 0
    # red[:, :, 1] = 0
    # red = (red.ravel())
    # red = red[red != 0]
    # r_mean = np.mean(red)
    # r_std_dev = np.std(red)
    #
    #
    # green = image.copy()
    # # set blue and red channels to 0
    # green[:, :, 0] = 0
    # green[:, :, 2] = 0
    # green = (green.ravel())
    # green = green[green != 0]
    # g_mean = np.mean(green)
    # g_std_dev = np.std(green)
    #
    # blue = image.copy()
    # # set green and red channels to 0
    # blue[:, :, 1] = 0
    # blue[:, :, 2] = 0
    # blue = (blue.ravel())
    # blue = blue[blue != 0]
    # b_mean = np.mean(blue)
    # b_std_dev = np.std(blue)
    #
    #
    # print( (int(r_mean), int(g_mean), int(b_mean)) )



    ###Realized this all existed in a single method already...now I know
    (mean, std_dev,) = cv2.meanStdDev(gray, mask=mask)
    # masked_img = cv2.bitwise_and(gray, gray, mask=mask)
    # data = masked_img.ravel()
    # before = len(data)
    # data = data[data != 0]  # remove zeros
    # after = len(data)
    # mean = np.mean(data)
    # std_dev = np.std(data)


    # gray.copyTo(masked_gray, mask)
    # masked_gray = gray * mask
    # total_intensity = cv2.sumElems(masked_gray)[0]

    channel = [0]  # only a single channel on grayscale
    bin_count = [256]  # this is full-scale (1 per possible uint8 intensity
    hist_range = [0, 256]  # ranging from 0 up to the max possible value of uint8

    # mask = np.zeros(image.shape[:2], np.uint8)
    # mask[100:300, 100:400] = 255
    # cv2.drawContours(mask, [c], -1, 255, -1)

    # Useful Snippet: https://docs.opencv.org/3.1.0/d1/db7/tutorial_py_histogram_begins.html
    hist_full = cv2.calcHist([data], channel, None, bin_count, hist_range)

    hist_mask = cv2.calcHist([gray], channel, mask, bin_count, hist_range)
    # hist_mask = [element[0] for element in hist_mask] #shell it (these methods keep adding dimensions)
    # Thresh  = outer_mean - outer_std_dev * k (where k is 1 to 3 ish)

    plt.subplot(221), plt.imshow(gray, 'gray')
    plt.subplot(222), plt.imshow(mask, 'gray')
    plt.subplot(223), plt.imshow(masked_img, 'gray')
    plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask)
    plt.xlim(hist_range)

    plt.show(False)  # false keep plt from blocking

    # #Realized there is a cleaner way to do this simply with drawContours
    # contourIdx = -1 # Draw all contours
    # color = 1
    # cv2.drawContours(mask, [c], -1, 255, -1)

    # inverse_mask = np.zeros(shape)
    # inverse_mask[:] = 255
    # cv2.fillPoly(inverse_mask, pts=contours, color=(0))
    # normalizer = (masked_gray + inverse_mask).min()

    # cv2.imshow("image", masked_gray)
    # cv2.waitKey()

    # create contour_maksed_image that's 0 where masked
    # sum contour_masked image area
    # Divide by brightest (ideal 90th percentile in case of noise)
    #   This makes the solid pixels count as one, and striped/blurred are some fraction


    # hist_mask = cv2.calcHist([gray], channel, mask, [bin_count], hist_range)



    # cv2.imshow("hist", hist)

    # plt.hist(gray.ravel(), 256, [0, 256]);
    # plt.show(False)


    # average_intensity = total_intensity/total_area
