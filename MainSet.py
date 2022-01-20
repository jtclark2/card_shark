import time
import cv2

import Card
import ImageExtractor
import CardAnalyzer
import SetPlayer
import Visualizer


######## Configure setup #########
color_table = {
    "Card": [255, 0, 0],  # Red
    "Set": [0, 255, 0],  # Green
    "Raw_Contour": [0, 0, 255],  # Blue
}

image_extractor = ImageExtractor.ImageExtractor()
card_analyzer = CardAnalyzer.HandTunedCardAnalyzer()  # Feature analyzer is better name
player = SetPlayer.SetPlayer()
visualizer = Visualizer.Visualizer()

IMG_PATH = "ImageLibrary/IMG_6394.JPG"
IMG_DIR ="ImageLibrary/%s.jpg"
IMG_SOURCE = "saved_image"

if IMG_SOURCE == "saved_image":
    import time
    t0 = time.time()

    image = cv2.imread(IMG_PATH)

    image = image_extractor.pre_process_image(image)
    cards = image_extractor.find_cards(image)
    sets = player.find_sets(cards)
    print("Image processed in: %f seconds" % (time.time() - t0))

    display_image = image.copy() # Create a copy to add graphics on top of
    visualizer.overlay_ROIs(display_image, image_extractor.ROIs, color=color_table["Raw_Contour"], line_thickness=3)
    visualizer.overlay_cards(cards, display_image, image_extractor.card_ROIs, color=color_table["Card"], line_thickness=3)
    if len(sets) > 0:
        visualizer.overlay_cards(sets[0], display_image, image_extractor.card_ROIs, color=color_table["Set"], line_thickness=3)
    visualizer.overlay_color_key(display_image, color_table)
    visualizer.display_image(display_image, width=500)
    visualizer.plot_extracted_cards(image_extractor.card_images)

    cv2.waitKey(0)


if IMG_SOURCE == "camera":
    import Camera
    cam = Camera.Camera()
    cam.configure()
    while(True):
        image = cam.read() # Capture frame-by-frame
        if image is None:
            break

        image = image_extractor.pre_process_image(image)
        cards = image_extractor.find_cards(image)
        sets = player.find_sets(cards)

        display_image = image.copy()  # Create a copy to add graphics on top of
        visualizer.overlay_ROIs(display_image, image_extractor.ROIs, color=color_table["Raw_Contour"],
                                line_thickness=3)
        visualizer.overlay_cards(cards, display_image, image_extractor.card_ROIs, color=color_table["Card"],
                                 line_thickness=3)
        if len(sets) > 0:
            visualizer.overlay_cards(sets[0], display_image, image_extractor.card_ROIs, color=color_table["Set"],
                                     line_thickness=3)
        visualizer.overlay_color_key(display_image, color_table)
        visualizer.display_image(display_image, width=500)
        # visualizer.plot_extracted_cards(image_extractor.card_images)

        key_input = cv2.waitKey(1)

        if (key_input & 0xFF == ord('s')): # Save
            name = "ImageLibrary/Capture%s.jpg" % repr(time.time())
            cv2.imwrite(name, image)

        if (key_input & 0xFF == ord('q')): # Quit
            cam.release()
            cv2.destroyAllWindows()
            break


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