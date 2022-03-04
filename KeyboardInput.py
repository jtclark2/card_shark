import time
import Card
import cv2
from enum import Enum

class KeyBoardActions(Enum):
    quit = "quit"
    pause = "pause"

def listenToKeyBoard(image, image_extractor, card_analyzer):
    key_input = cv2.waitKey(1)
    if (key_input & 0xFF == ord('s')):  # Save Image
        name = "ImageLibrary/Capture%s.jpg" % repr(time.time())
        cv2.imwrite(name, cv2.resize(image, (640,480)))

    # Automatic calibration (refines output if cards are mostly correct already)
    if (key_input & 0xFF == ord('c')):  # Calibrate
        images = image_extractor.detect_cards(image)
        cards = image_extractor.identify_cards(images)
        card_analyzer.calibrate(cards)

    # Manual-ish calibrations
    if (key_input & 0xFF == ord('[')):
        card_analyzer.hollow_striped_thresh -= 0.05
        print(f"Adjusted hollow_striped_thresh: {card_analyzer.hollow_striped_thresh}")
    if (key_input & 0xFF == ord(']')):
        card_analyzer.hollow_striped_thresh += 0.05
        print(f"Adjusted hollow_striped_thresh: {card_analyzer.hollow_striped_thresh}")

    if (key_input & 0xFF == ord('(')):
        card_analyzer.striped_solid_thresh -= 0.1
        print(f"Adjusted empty_striped_thresh: {card_analyzer.striped_solid_thresh}")
    if (key_input & 0xFF == ord(')')):
        card_analyzer.striped_solid_thresh += 0.1
        print(f"Adjusted empty_striped_thresh: {card_analyzer.striped_solid_thresh}")

    # Color calibrations: Requires 1 card of that color
    if (key_input & 0xFF == ord('p')):
        images = image_extractor.detect_cards(image)
        cards = image_extractor.identify_cards(images)
        card_analyzer.calibrate_single_color(cards[0], Card.Color.purple)

    if (key_input & 0xFF == ord('r')):
        images = image_extractor.detect_cards(image)
        cards = image_extractor.identify_cards(images)
        card_analyzer.calibrate_single_color(cards[0], Card.Color.red)

    if (key_input & 0xFF == ord('g')):
        images = image_extractor.detect_cards(image)
        cards = image_extractor.identify_cards(images)
        card_analyzer.calibrate_single_color(cards[0], Card.Color.green)

    if (key_input & 0xFF == ord('d')):  # Diagnostic mode toggle
        card_analyzer.diagnostic_mode = not card_analyzer.diagnostic_mode

    if (key_input & 0xFF == ord('q')):  # Quit
        cv2.destroyAllWindows()
        return KeyBoardActions.quit

    if (key_input & 0xFF == ord(' ')):  # Quit
        cv2.destroyAllWindows()
        return KeyBoardActions.pause