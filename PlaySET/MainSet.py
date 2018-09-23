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
import cv2


class Game():

    def process_image(self, image):
        # Process image
        self.ROIs = image_extractor.locate_ROIs(image)
        # image_extractor.display_ROIs(image, ROIs, color=[255, 0, 0], line_thickness=2)
        self.four_corner_ROIs = image_extractor.filter_for_cards(self.ROIs)
        # image_extractor.display_ROIs(image, four_corner_ROIs, color=[255, 0, 0], line_thickness=2)

        self.card_images = image_extractor.extract_images(image, self.four_corner_ROIs)

        self.cards = []
        for idx, card_image in enumerate(self.card_images):
            card = Card.Card(index=idx, image=card_image)
            card = card_analyzer.identify_card(card)
            if card is not None:
                self.cards.append(card)

        self.sets = player.find_sets(self.cards)


##################Main######################
if __name__ == "__main__":
    ########Select tools (extractor, analyzer, player)#########
    image_extractor = ImageExtractor.ImageExtractor()
    card_analyzer = CardAnalyzer.HandTunedCardAnalyzer()  # Feature analyzer is better name
    player = SetPlayer.SetPlayer()
    game = Game()

    # IMG_PATH = "IMG_6441.JPG"
    IMG_PATH = "WebCam1.PNG"
    IMG_DIR ="ImageLibrary/%s.jpg"
    IMG_SOURCE = "camera"

if IMG_SOURCE == "saved_image":
    import time
    t0 = time.time()

    image = cv2.imread(IMG_PATH)
    game.process_image(image)


    print("Image processed in: %f seconds" % (time.time() - t0))

    image_extractor.display_cards(game.cards, image, game.ROIs, color=[255, 0, 0], line_thickness=2)
    if (len(game.sets) > 0):
        image_extractor.display_cards(game.sets[0], image, game.ROIs, color=[0, 255, 0], line_thickness=5)

    cv2.waitKey(0)
else:

    cap = cv2.VideoCapture(0)
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if (frame is None):
            print("Yo! Close the previous connection fist!")
            break

        game.process_image(frame)

        image_extractor.display_cards(game.cards, frame, game.ROIs, color=[255, 0, 0], line_thickness=2)
        if (len(game.sets) > 0):
            image_extractor.display_cards(game.sets[0], frame, game.ROIs, color=[0, 255, 0], line_thickness=5)



        key_input = cv2.waitKey(1)

        if (key_input & 0xFF == ord('s')): # Save
            name = "ImageSet/%s.jpg" % repr(time.time())
            cv2.imwrite(name, frame)

        if (key_input & 0xFF == ord('c')): # Calibrate
            card = Card.Card(index=0, image=game.card_images[0])

        if (key_input & 0xFF == ord('q')): # Quit
            cap.release()
            cv2.destroyAllWindows()
            break

    # When everything done, release the capture


