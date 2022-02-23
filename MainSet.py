import time
import cv2
import imutils

import ImageExtractor
import CardAnalyzer
import SetPlayer
from Visualizer import Visualizer
import Camera


######## Configure setup #########
color_table = {
    "Raw_Contour": [0, 0, 255],  # Blue
    "Card": [255, 0, 0],  # Red
    "Set": [0, 255, 0],  # Green
}

image_extractor = ImageExtractor.ImageExtractor()
card_analyzer = CardAnalyzer.HandTunedCardAnalyzer()  # Feature analyzer is better name
player = SetPlayer.SetPlayer()

IMG_NAME = "IMG_6394.JPG"
# IMG_NAME = "WebCam1.PNG"
# IMG_NAME = "Capture_AllShades_MatchedToSilhouette.jpg"
# IMG_NAME = "RedReadsGreen.jpg" # looks like a rollover error in hue space
IMG_NAME = "CaptureBright.jpg"
IMG_NAME = "CaptureBackOfBox.jpg"

IMG_PATH = f"ImageLibrary/{IMG_NAME}"
IMG_DIR ="ImageLibrary/%s.jpg"
# IMG_SOURCE = "saved_image"
IMG_SOURCE = "saved_image" # "camera" #



# Processing Pipeline
def image_pipeline(image, image_extractor, color_table, player):

    images = image_extractor.detect_cards(image)
    tic = time.perf_counter()
    cards = image_extractor.identify_cards(images)
    print(f"Time to generate and display frame Time: {time.perf_counter() - tic}")
    sets = player.find_sets(cards)

    display_image = image.copy() # Create a copy to add graphics on top of
    Visualizer.overlay_ROIs(display_image, image_extractor.ROIs, color=color_table["Raw_Contour"], line_thickness=3)
    Visualizer.overlay_cards(cards, display_image, image_extractor.card_ROIs, color=color_table["Card"], line_thickness=3)

    if len(sets) > 0:
        Visualizer.overlay_cards(sets[0], display_image, image_extractor.card_ROIs, color=color_table["Set"], line_thickness=3)

    image = Visualizer.overlay_color_key(display_image, color_table)

    # TODO: cv2.resize causes some aliasing, but removes the imutils dependency (and that library has proven pretty unstable)
    # shape = image.shape
    # image = cv2.resize(image, (500, 500*shape[0]//shape[1]))
    image = imutils.resize(image, width=1000)
    cv2.imshow("Rich Diagnostic View", image)

# Read image, process, display image, and plot all cards found
if IMG_SOURCE == "saved_image":
    image = cv2.imread(IMG_PATH)

    image_pipeline(image, image_extractor, color_table, player)
    Visualizer.plot_extracted_cards(image_extractor.card_images)

    cv2.waitKey(0)

# Loop image capture and save/quit key interactions
if IMG_SOURCE == "camera":
    cam = Camera.Camera(1) # Starts at 0 (built-in laptop cam is usually 0, and USB cam is usually 1)
    cam.configure()
    while(True):
        image = cam.read() # Capture frame-by-frame
        if image is None:
            break

        image_pipeline(image, image_extractor, color_table, player)
        # Visualizer.plot_extracted_cards(image_extractor.card_images)

        key_input = cv2.waitKey(1)

        if (key_input & 0xFF == ord('s')): # Save
            name = "ImageLibrary/Capture%s.jpg" % repr(time.time())
            cv2.imwrite(name, image)

        if (key_input & 0xFF == ord('q')): # Quit
            cam.release()
            cv2.destroyAllWindows()
            break