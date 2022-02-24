import cv2
import imutils
from matplotlib import pyplot as plt
from enum import Enum


class Visualizer:

    @staticmethod
    def overlay_ROIs(image, ROIs, color=[255, 0, 0], line_thickness=1):
        """
        Draw simple contours on the image (modifies in place) using the built in
        drawContours method.
        """
        contours = [ROI.contour for ROI in ROIs]
        cv2.drawContours(image=image,
                         contours=contours,
                         contourIdx= -1,
                         color=color,
                         thickness=line_thickness)

    @staticmethod
    def overlay_cards(cards, image, ROIs, color=[255, 0, 0], line_thickness=1, show_labels = True):
        """
        Purpose: Apply outline and annotation to cards in image
        :param cards: List of cards with attributes to be overlaid. This excludes ROIs, but does include
            indexes which map to the ROIs, to determine which will be displayed.
        :param image: Main image that graphics will be
        :param ROIs: All regions Of Interest (which are really just wrappers for Contours)
        :param color: Color to display (BGR).
        :param line_thickness: Thickness of ROI border
        :param show_labels: (bool) Whether or not to overlay labels describing card status.
        :return: None
        """
        contours = [ROI.contour for ROI in ROIs]
        for card in cards:
            cv2.drawContours(image=image,
                             contours=contours,
                             contourIdx=card.index,
                             color=color,
                             thickness=line_thickness)
            if show_labels and contours is not None:
                image = Visualizer._annotate_card(image, card, contours[card.index])

    @staticmethod
    def overlay_color_key(image, key_dict, text_size=30):
        """

        :param image: The image to overlay text on top of
        :param key_dict: dict{text, color}
        :param text_size: Size that text will be display in (size in pixels)
        :return: The image with text overlaid
        """
        # image = imutils.resize(image, width=display_width)
        text_size = 30 # in pixels
        x,y = text_size//6, text_size//6
        for key in key_dict:
            y += text_size
            cv2.putText(image, key, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        text_size/30, key_dict[key], text_size//12)
            cv2.putText
        return image

    @staticmethod
    def plot_extracted_cards(images):
        """
        Purpose: Displays up to 12 images extracted.

        Side Effects:
            - Plots cards
            - Blocking: Not recommended for use with live video
        :param images: Images of cards to be plotted
        :return: None
        """
        i=1
        for image in images:
            # plot
            (rows, columns) = (3,4)
            rgb_image = im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            plt.subplot(rows, columns, i), plt.imshow(rgb_image)
            i+=1
            if(i > rows*columns):
                break
        plt.show()

    @staticmethod
    def _annotate_card(image, card, contour, size=1):
        """
        Annotates a card with it's attributes
        :param image: The image of the cards.
        :param card: The Card object, with underlying attributes
        :param contour: A contour outlining the card
        :return: The updated image, with contours added.
        """
        text_size = size
        spacing = int(size*20)
        text_thickness = int(size*2)

        cX_offset = -20
        cY_offset = -20

        M = cv2.moments(contour)
        cX = int(M['m10']/M['m00']) + cX_offset
        cY = int(M['m01']/M['m00']) + cY_offset

        cv2.putText(image, repr(card.count), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
                    text_size, (0, 0, 0), text_thickness)
        cv2.putText(image, repr(card.shape), (cX, cY + spacing), cv2.FONT_HERSHEY_SIMPLEX,
                    text_size, (0, 0, 0), text_thickness)
        cv2.putText(image, repr(card.fill), (cX, cY + spacing * 2), cv2.FONT_HERSHEY_SIMPLEX,
                    text_size, (0, 0, 0), text_thickness)
        cv2.putText(image, repr(card.color), (cX, cY + spacing * 3), cv2.FONT_HERSHEY_SIMPLEX,
                    text_size, (0, 0, 0), text_thickness)

        return image

