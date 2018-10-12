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

        # Filter #1 : create contours around Regions Of Interest(ROIs)
        self.ROIs = image_extractor.locate_ROIs(image)
        display_image = image.copy()
        image_extractor.display_ROIs(display_image, self.ROIs, color=[0, 0, 255], line_thickness=1)

        # Filter #2: Find 4 cornered, card-shaped rectangles images
        self.four_corner_ROIs = image_extractor.filter_for_rectangles(self.ROIs)
        image_extractor.display_ROIs(display_image, self.four_corner_ROIs, color=[255, 255, 0], line_thickness=2)


        # Filter #3: Keep only identifiable cards
        self.card_images = image_extractor.extract_card_images(image, self.four_corner_ROIs)
        if(len(self.card_images) > 0):
            cv2.imshow("First Card", self.card_images[0])

        self.cards = []
        for idx, card_image in enumerate(self.card_images):
            card = Card.Card(index=idx, image=card_image)
            card = card_analyzer.identify_card(card)
            if card is not None:
                self.cards.append(card)

        image_extractor.display_cards(self.cards, display_image, self.ROIs, color=[255, 0, 0], line_thickness=3)

        # Filter #4: Identify sets of cards
        self.sets = player.find_sets(self.cards)
        if(len(self.sets) > 1):
            image_extractor.display_cards(self.sets[0], display_image, self.ROIs, color=[0, 255, 0], line_thickness=3)


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

    if (len(game.sets) > 0):
        image_extractor.display_cards(game.sets[0], image, game.ROIs, color=[0, 255, 0], line_thickness=5)


    if (len(game.card_images) > 0):
        cv2.imshow("Card", game.card_images[0])

    cv2.waitKey(0)
else:
    # https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html
    CV_CAP_PROP_FPS = 5
    CV_CAP_PROP_CONTRAST = 11
    CV_CAP_PROP_SATURATION = 12
    CV_CAP_PROP_HUE = 13
    CV_CAP_PROP_GAIN = 14
    CV_CAP_PROP_EXPOSURE = 15
    CV_CAP_PROP_CONVERT_RGB=16

    CV_CAP_PROP_FRAME_WIDTH = 3
    CV_CAP_PROP_FRAME_HEIGHT = 4
    CV_CAP_PROP_BRIGHTNESS = 10

    # Intentionally set to specific values
    cap = cv2.VideoCapture(0)
    cap.set(CV_CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(CV_CAP_PROP_SATURATION, 100) # 0 - 100, values outside this range are ignored
    cap.set(CV_CAP_PROP_EXPOSURE, -4)
    cap.set(CV_CAP_PROP_HUE, 100)

    # set so that they wouldn't auto-adjust on me
    cap.set(CV_CAP_PROP_FPS, 1)
    cap.set(CV_CAP_PROP_BRIGHTNESS, 0)
    cap.set(CV_CAP_PROP_GAIN, 0)
    cap.set(CV_CAP_PROP_CONTRAST, 99)
    # cap.set(CV_CAP_PROP_CONVERT_RGB, True)


    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if (frame is None):
            print("Close the previous connection fist!")
            break

        frame = image_extractor.pre_process_image(frame)
        game.process_image(frame)
        #
        # image_extractor.display_cards(game.cards, frame, game.ROIs, color=[255, 0, 0], line_thickness=2)
        # if (len(game.sets) > 0):
        #     image_extractor.display_cards(game.sets[0], frame, game.ROIs, color=[0, 255, 0], line_thickness=5)
        #


        key_input = cv2.waitKey(1)

        if (key_input & 0xFF == ord('s')): # Save
            name = "ImageSet/%s.jpg" % repr(time.time())
            cv2.imwrite(name, frame)

        cal_lookup = {
             ord('0') : (None, Card.Fill.empty),
             ord('1') : (Card.Color.red, Card.Fill.empty),
             ord('2'): (Card.Color.green, Card.Fill.empty),
             ord('3'): (Card.Color.purple, Card.Fill.empty),
             ord('4'): (Card.Color.red, Card.Fill.striped),
             ord('5'): (Card.Color.green, Card.Fill.striped),
             ord('6'): (Card.Color.purple, Card.Fill.striped),
             ord('7'): (Card.Color.red, Card.Fill.solid),
             ord('8'): (Card.Color.green, Card.Fill.solid),
             ord('9'): (Card.Color.purple, Card.Fill.solid),
             ord('r'): 'reset',
        }

        edge_set = (ord('1'), ord('2'), ord('3'))

        key_stroke = (key_input & 0xFF)
        if ( key_stroke in cal_lookup): # Calibrate
            if (len(game.card_images) == 1):
                card = Card.Card(index=0, image=game.card_images[0])
                key = cal_lookup[key_input]
                if(key_stroke in edge_set):
                    try:
                        value = card_analyzer.corrected_edge_color
                        card_analyzer.calibrate_colors(key=key, value=value)
                    except:
                        print("This card does not appear to have fill='Empty'.")
                else: # any other
                    value = card_analyzer.corrected_color
                    card_analyzer.calibrate_colors(key=key, value=value)


        if (key_input & 0xFF == ord('q')): # Quit
            cap.release()
            cv2.destroyAllWindows()
            break

    # When everything done, release the capture


