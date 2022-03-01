import time
import cv2

import ImageExtractor
import CardAnalyzer
import SetPlayer
from Visualizer import Visualizer
import Camera
import Card
import KeyboardInput


######## Configuration setup #########
SOURCE_TYPE = "video"
# SOURCE_TYPE = "image"

if SOURCE_TYPE == "video":
    # Either camera index or a video
    VIDEO_SOURCE = "Attempt2_WithoutGraphics.avi"
    # VIDEO_SOURCE = 1

    if type(VIDEO_SOURCE) == str:
        VIDEO_SOURCE = f"VideoLibrary/{VIDEO_SOURCE}"

    record_video = False
    if record_video:
        FRAME_RATE = 24.0
        OUTPUT_FILE = "Output.avi"
        OUTPUT_VIDEO_PATH = f"VideoLibrary/Output_{FRAME_RATE}FPS_{OUTPUT_FILE}"
        raw = False


else:
    IMG_NAME = "IMG_6394.JPG"
    # IMG_NAME = "IMG_6441.JPG"
    # IMG_NAME = "WebCam1.PNG"
    # IMG_NAME = "Capture_AllShades_MatchedToSilhouette.jpg"
    # IMG_NAME = "RedReadsGreen.jpg" # looks like a rollover error in hue space
    # IMG_NAME = "CaptureBright.jpg"
    IMG_NAME = "CaptureBackOfBox.jpg"
    # IMG_NAME = "ReadsAsThree.jpg"
    # IMG_NAME = "CaptureCardContourIssue.jpg"
    # IMG_NAME = "CaptureBright.jpg"
    # IMG_NAME = "CaptureBackOfBox.jpg"

    IMG_PATH = f"ImageLibrary/{IMG_NAME}"

OUTPUT_WIDTH = 480

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
    cards = []
    for idx, card_image in enumerate(images):
        card = Card.Card(index=idx, image=card_image)
        card = card_analyzer.identify_card(card)
        if card is not None:
            cards.append(card)
    id_time = f"{time.perf_counter() - start: .5f}"

    start = time.perf_counter()
    sets = player.find_sets(cards)
    play_time = f"{time.perf_counter() - start: .5f}"

    start = time.perf_counter()
    display_image = image.copy() # Create a copy to add graphics on top of
    Visualizer.overlay_ROIs(display_image, image_extractor.ROIs, color=color_table["Raw_Contour"], line_thickness=3)
    Visualizer.overlay_cards(cards, display_image, image_extractor.card_ROIs, color=color_table["Card"],
                             line_thickness=3, text_size=1.3)

    if len(sets) > 0:
        Visualizer.overlay_cards(sets[0], display_image, image_extractor.card_ROIs, color=color_table["Set"],
                                 line_thickness=3, show_labels=True, text_size=1.3)


    # TODO: cv2.resize causes minor aliasing, but removes the imutils dependency (which has been a bit unstable)
    shape = display_image.shape
    display_image = cv2.resize(display_image, (OUTPUT_WIDTH*shape[1]//shape[0], OUTPUT_WIDTH))

    display_image = Visualizer.overlay_color_key(display_image, color_table, text_size=1.5)
    fps = 1 / (time.perf_counter() - tic)
    display_image = Visualizer.display_fps(display_image, fps, text_size=1)

    # image = imutils.resize(image, width=OUTPUT_WIDTH)
    cv2.imshow("Game Window", display_image)
    overlay_time = f"{time.perf_counter() - start: .5f}"
    # print(extract_time, id_time, play_time, overlay_time)
    return image

# Read image, process, display image, and plot all cards found
if SOURCE_TYPE == "image":
    image = cv2.imread(IMG_PATH)
    print(f"Shape of incoming image: {image.shape}")

    while True:
        processed_image = image_pipeline(image, image_extractor, color_table, player)
        if KeyboardInput.listenToKeyBoard(image, image_extractor, card_analyzer):
            break
    Visualizer.plot_extracted_cards(image_extractor.card_images)

# Loop image capture and save/quit key interactions
if SOURCE_TYPE == "video":
    cam = Camera.Camera(VIDEO_SOURCE) # Starts at 0 (built-in laptop cam is usually 0, and USB cam is usually 1)
    cam.configure()
    image = cam.read()

    if image is None:
        raise ConnectionRefusedError("Did you remember to plug in the camera, and kill the other programs?")

    if record_video:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        if raw:
            shape = cam.shape
        else:
            shape = (OUTPUT_WIDTH, OUTPUT_WIDTH*cam.shape[1]//cam.shape[0])

        vid_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, FRAME_RATE, shape)

    pause = False
    while(True):
        if not pause:
            image = cam.read() # Capture frame-by-frame

        if image is None:
            break

        if record_video and raw:
                vid_writer.write(image)

        processed_image = image_pipeline(image, image_extractor, color_table, player)

        if record_video and not raw:
            vid_writer.write(processed_image)

        action = KeyboardInput.listenToKeyBoard(image, image_extractor, card_analyzer)
        if action == "quit": # TODO: Enum
            cam.release()
            if record_video:
                vid_writer.release()
            break
        if action == "pause":
            pause = not pause
