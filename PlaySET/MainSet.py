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

import Card
import ImageExtractor
import CardAnalyzer
import SetPlayer

import time
import imutils
import cv2
import numpy as np

def display_sets_text(sets):
    for s in sets:
        print(s)


##################Main######################
if __name__ == "__main__":
    ########Select tools (extractor, analyzer, player)#########
    image_extractor = ImageExtractor.ImageExtractor()

    card_analyzer = CardAnalyzer.HandTunedCardAnalyzer()  # Feature analyzer is better name
    # card_analyzer = CardAnalyzer.TemplateAnalyzer()

    player = SetPlayer.SetPlayer()

    #IMG_PATH = "IMG_6394.JPG"
    # IMG_PATH = "IMG_6438.JPG"
    # IMG_PATH = "IMG_6439.JPG"
    # IMG_PATH = "IMG_6440.JPG"
    # IMG_PATH = "IMG_6441.JPG"
    IMG_PATH = "WebCam1.PNG"

    SAVE_PATH ="ImageLibrary/%s.jpg"

    INPUT = "camera"

if INPUT == "saved_image":
    import time
    t0 = time.time()

    image = cv2.imread(IMG_PATH)
    image = image_extractor.pre_process_image(IMG_PATH)


    ROIs = image_extractor.locate_ROIs(image)
    four_corner_ROIs = image_extractor.filter_for_cards(ROIs)

    card_images = image_extractor.extract_images(image, four_corner_ROIs)

    cards =[]
    for idx, card_image in enumerate(card_images):
        card = Card.Card(index=idx, image=card_image)
        card = card_analyzer.identify_card(card)
        if card is not None:
            cards.append(card)


    sets = player.find_sets(cards)
    # display_sets_text(sets)

    image_extractor.display_cards(cards, image, ROIs, color=[255, 0, 0], line_thickness=2)

    if( len(sets) > 0):
        image_extractor.display_cards(sets[0], image, ROIs, color = [ 0, 255, 0], line_thickness = 5)

    print("Image processed in: %f seconds" % (time.time() - t0))
else:
    # Process from webcam (doin' it live!)

    cap = cv2.VideoCapture(0)
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if (frame is None):
            print("Yo! Close the previous connection fist!")
            break

        ROIs = image_extractor.locate_ROIs(frame)
        # image_extractor.display_ROIs(frame, ROIs, color=[255, 0, 0], line_thickness=2)
        four_corner_ROIs = image_extractor.filter_for_cards(ROIs)
        # image_extractor.display_ROIs(frame, four_corner_ROIs, color=[255, 0, 0], line_thickness=2)

        card_images = image_extractor.extract_images(frame, four_corner_ROIs)

        cards = []
        for idx, card_image in enumerate(card_images):
            card = Card.Card(index=idx, image=card_image)
            card = card_analyzer.identify_card(card)
            if card is not None:
                cards.append(card)

        sets = player.find_sets(cards)
        # display_sets_text(sets)

        image_extractor.display_cards(cards, frame, ROIs, color=[255, 0, 0], line_thickness=2)
        if (len(sets) > 0):
            image_extractor.display_cards(sets[0], frame, ROIs, color=[0, 255, 0], line_thickness=5)


        ####Results
        if(len(cards) > 0):
            # image_extractor.display_extracted_images([card.image for card in cards])

            #TODO: Heavy code duplication here
            gray = cv2.cvtColor(cards[0].image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
            contours = contours[0] if imutils.is_cv2() else contours[1]

            SYMBOL_SIZE_THRESH = 1000

            contours = [c for c in contours if cv2.contourArea(c) >= SYMBOL_SIZE_THRESH]

            shape = gray.shape
            mask = np.zeros(shape, np.uint8)
            mask = cv2.erode(mask, None, iterations=5)  # Avoid the edge...just want the texture

            cv2.drawContours(mask, contours, -1, 255, -1)

        key_input = cv2.waitKey(1)

        if (key_input & 0xFF == ord('s')): # Save
            name = "ImageSet/%s.jpg" % repr(time.time())
            cv2.imwrite(name, mask)

        if (key_input & 0xFF == ord('q')): # Quit
            break

        if (key_input & 0xFF == ord('c')): # Calibrate
        #     for()
        # else:
        #     #reset
            pass




    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


