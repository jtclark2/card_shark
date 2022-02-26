import time
import cv2
import imutils

import ImageExtractor
import CardAnalyzer
import SetPlayer
from Visualizer import Visualizer
import Camera


######## Configuration setup #########
SOURCE_TYPE = "video" # "image"

if SOURCE_TYPE == "video":
    # Either camera index or a video
    VIDEO_SOURCE = "Attempt2_WithoutGraphics.avi"
    VIDEO_SOURCE = 1

    if type(VIDEO_SOURCE) == str:
        VIDEO_SOURCE = f"VideoLibrary/{VIDEO_SOURCE}"

    record_video = False
    if record_video:
        FRAME_RATE = 8.0
        OUTPUT_FILE = "Output.avi"
        OUTPUT_VIDEO_PATH = f"VideoLibrary/Output_{FRAME_RATE}FPS_{OUTPUT_FILE}"
        raw = False


else:
    IMG_NAME = "IMG_6394.JPG"
    IMG_NAME = "WebCam1.PNG"
    # IMG_NAME = "Capture_AllShades_MatchedToSilhouette.jpg"
    # IMG_NAME = "RedReadsGreen.jpg" # looks like a rollover error in hue space
    # IMG_NAME = "CaptureBright.jpg"
    # IMG_NAME = "CaptureBackOfBox.jpg"
    # IMG_NAME = "ReadsAsThree.jpg"
    # IMG_NAME = "CaptureCardContourIssue.jpg"

    IMG_PATH = f"ImageLibrary/{IMG_NAME}"

OUTPUT_WIDTH = 1000

color_table = {
    "Raw_Contour": [0, 0, 255],  # Blue
    "Card": [255, 0, 0],  # Red
    "Set": [0, 255, 0],  # Green
}

image_extractor = ImageExtractor.ImageExtractor()
card_analyzer = CardAnalyzer.HandTunedCardAnalyzer()  # Feature analyzer is better name
player = SetPlayer.SetPlayer()

# Processing Pipeline
def image_pipeline(image, image_extractor, color_table, player):
    tic = time.perf_counter()
    start = time.perf_counter()
    images = image_extractor.detect_cards(image)
    extract_time = f"{time.perf_counter() - start: .5f}"
    start = time.perf_counter()
    cards = image_extractor.identify_cards(images)
    id_time = f"{time.perf_counter() - start: .5f}"
    start = time.perf_counter()
    sets = player.find_sets(cards)
    play_time = f"{time.perf_counter() - start: .5f}"

    start = time.perf_counter()
    display_image = image.copy() # Create a copy to add graphics on top of
    Visualizer.overlay_ROIs(display_image, image_extractor.ROIs, color=color_table["Raw_Contour"], line_thickness=3)
    Visualizer.overlay_cards(cards, display_image, image_extractor.card_ROIs, color=color_table["Card"], line_thickness=3)

    if len(sets) > 0:
        Visualizer.overlay_cards(sets[0], display_image, image_extractor.card_ROIs, color=color_table["Set"], line_thickness=3)

    image = Visualizer.overlay_color_key(display_image, color_table)
    fps = 1 / (time.perf_counter() - tic)
    image = Visualizer.display_fps(display_image, fps)

    # TODO: cv2.resize causes some aliasing, but removes the imutils dependency (and that library has proven pretty unstable)
    shape = image.shape
    image = cv2.resize(image, (OUTPUT_WIDTH*shape[1]//shape[0], OUTPUT_WIDTH))
    # image = imutils.resize(image, width=OUTPUT_WIDTH)
    cv2.imshow("Rich Diagnostic View", image)


    display_time = f"{time.perf_counter() - start: .5f}"
    # print(extract_time, id_time, play_time, display_time)
    return image

# Read image, process, display image, and plot all cards found
if SOURCE_TYPE == "image":
    image = cv2.imread(IMG_PATH)

    processed_image = image_pipeline(image, image_extractor, color_table, player)
    Visualizer.plot_extracted_cards(image_extractor.card_images)

    # Use to save results to share/show (saves processed form)
    while True:
        key_input = cv2.waitKey(1)
        if (key_input & 0xFF == ord('s')):  # Save
            name = "ImageLibrary/Capture%s.jpg" % repr(time.time())
            cv2.imwrite(name, processed_image)
        if (key_input & 0xFF == ord('q')):  # Quit
            cv2.destroyAllWindows()
            break

# Loop image capture and save/quit key interactions
if SOURCE_TYPE == "video":
    cam = Camera.Camera(VIDEO_SOURCE) # Starts at 0 (built-in laptop cam is usually 0, and USB cam is usually 1)
    cam.configure()
    image = cam.read()

    if record_video:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        if raw:
            shape = cam.shape
        else:
            shape = (OUTPUT_WIDTH, OUTPUT_WIDTH*cam.shape[1]//cam.shape[0])

        vid_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, FRAME_RATE, shape)


    while(True):
        image = cam.read() # Capture frame-by-frame
        if image is None:
            break

        if record_video and raw:
                vid_writer.write(image)

        processed_image = image_pipeline(image, image_extractor, color_table, player)

        if record_video and not raw:
            vid_writer.write(processed_image)

        key_input = cv2.waitKey(1)
        if (key_input & 0xFF == ord('s')): # Save
            name = "ImageLibrary/Capture%s.jpg" % repr(time.time())
            cv2.imwrite(name, image)
        if (key_input & 0xFF == ord('q')): # Quit
            cam.release()
            if record_video:
                vid_writer.release()
            cv2.destroyAllWindows()
            break