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

from Card import *
from ImageExtractor import ImageExtractor
from CardAnalyzer import CardAnalyzer
from SetPlayer import SetPlayer

import imutils
import cv2
from matplotlib import pyplot as plt

IMG_PATH = "IMG_6394.JPG"
# IMG_PATH = "IMG_6438.JPG"
# IMG_PATH = "IMG_6439.JPG"
# IMG_PATH = "IMG_6440.JPG"
# IMG_PATH = "IMG_6441.JPG"
IMG_PATH = "WebCam1.PNG"

WIDTH = None #width of image read in (use None to keep original size)


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

    card_extractor = ImageExtractor()
    card_analyzer = CardAnalyzer()
    player = SetPlayer()

    (all_card_vertices, card_contours) = card_extractor.find_cards(image)
    i = 1
    cards = []
    card_images = [];
    for idx, card_vertices in enumerate(all_card_vertices):
        card_image = card_extractor.transform_and_extract_image(image, card_vertices)
        card_images.append(card_image)
        card = card_analyzer.identify_card(card_image)
        if card is None:
            continue
        card.index = idx
        # display_card(card_image, card)
        cards.append(card)

    sets = player.find_sets(cards)

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
    image = get_image(IMG_PATH, width = WIDTH)
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
