import Card
import ImageExtractor
import CardAnalyzer
import SetPlayer

import time
import cv2

import Visualizer


class Game:

    def __init__(self, extractor, visualizer, player):
        self.extractor = extractor
        self.visualizer = visualizer
        self.player = player

    def find_cards(self, image):
        images = self.extractor.detect_cards(image)
        cards = []
        for idx, card_image in enumerate(images):
            card = Card.Card(index=idx, image=card_image)
            card = card_analyzer.identify_card(card)
            if card is not None:
                cards.append(card)

            max_cards_to_display = 3
            if idx < max_cards_to_display:
                cv2.imshow("Card # %d" % (idx), card_image)

        # Process image
        color_table = {
            "Card" : [255, 0, 0], # Red
            "Set" : [0, 255, 0], # Green
            "Raw_Contour" : [0, 0, 255], # Blue
        }

        display_image = image.copy() # Create a copy to add graphics on top of

        all_ROIs = self.extractor.ROIs
        card_ROIs = self.extractor.card_ROIs


        # Display
        self.visualizer.display_ROIs(display_image, all_ROIs, color=color_table["Raw_Contour"], line_thickness=3)
        self.visualizer.display_cards(cards, display_image, card_ROIs, color=color_table["Card"], line_thickness=3)


        # # Filter #4: Identify sets of cards
        sets = player.find_sets(cards)
        if(len(sets) > 0):
            self.visualizer.display_cards(sets[0], display_image, card_ROIs, color=color_table["Set"], line_thickness=3)

        self.visualizer.display_key(display_image, color_table, display_width = 1500)

        self.card_images = images

        return cards


##################Main######################
# Configure
# image_source = Camera
#
# get_image
# while image is not None:
    # get_image
    # pre_process_image
    # ROIs, images = detect_cards (image)
    # cards = identify_cards(images)
    # sets = determine_sets(cards)
    # display_contours() # probably comment out
    # display_cards()
    # display_sets()
    # detect_special_keys()

########Select tools (extractor, analyzer, player)#########
image_extractor = ImageExtractor.ImageExtractor()
card_analyzer = CardAnalyzer.HandTunedCardAnalyzer()  # Feature analyzer is better name
player = SetPlayer.SetPlayer()
visualizer = Visualizer.Visualizer()
game = Game(image_extractor, visualizer, player)

IMG_PATH = "ImageLibrary/IMG_6394.JPG"
IMG_DIR ="ImageLibrary/%s.jpg"
IMG_SOURCE = "saved_image"

if IMG_SOURCE == "saved_image":
    import time
    t0 = time.time()

    image = cv2.imread(IMG_PATH)
    image = image_extractor.pre_process_image(image)
    cards = game.find_cards(image)

    print("Image processed in: %f seconds" % (time.time() - t0))
    print("Press 'q' to close all windows.")
    cv2.waitKey(0)


if IMG_SOURCE == "camera":
    import Camera
    cam = Camera.Camera()
    cam.configure()
    while(True):
        frame = cam.read() # Capture frame-by-frame
        if frame is None:
            break
        frame = image_extractor.pre_process_image(frame)
        cards = game.find_cards(frame)

        key_input = cv2.waitKey(1)

        if (key_input & 0xFF == ord('s')): # Save
            name = "ImageLibrary/Capture%s.jpg" % repr(time.time())
            cv2.imwrite(name, frame)

        if (key_input & 0xFF == ord('q')): # Quit
            cam.release()
            cv2.destroyAllWindows()
            break
