import cv2
import imutils
from matplotlib import pyplot as plt


class Visualizer:

    def display_ROIs(self, image, ROIs, color=[255, 0, 0], line_thickness=1):
        contours = [ROI.contour for ROI in ROIs]
        cv2.drawContours(image=image,
                         contours=contours,
                         contourIdx= -1,
                         color=color,
                         thickness=line_thickness)
        # image = imutils.resize(image, width=600)
        # cv2.imshow("Rich Diagnostic View", image)

    #Visualization tools that apply to the original input image (not to individually extracted cards)
    def display_cards(self, cards, image, ROIs, color=[255, 0, 0], line_thickness=1, label = True, flip=False):
        contours = [ROI.contour for ROI in ROIs]
        for card in cards:
            cv2.drawContours(image=image,
                             contours=contours,
                             contourIdx=card.index,
                             color=color,
                             thickness=line_thickness)
            if label:
                image = self.annotate_card(image, card, contours[card.index])
        # image = imutils.resize(image, width=1000)
        if(flip):
            image = cv2.flip(image, -1)
        # cv2.imshow("Rich Diagnostic View", image)

    def display_key(self, image, key_dict, display_width = 800):
        image = imutils.resize(image, width=display_width)
        X = 20
        Y = 20
        for key in key_dict:
            Y += 20
            cv2.putText(image, key, (X, Y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, key_dict[key], 2)
        cv2.imshow("Rich Diagnostic View", image)

    def display_extracted_images(self, images):
        # Displays first 12 images extracted (or as many as are available)
        i=1
        for image in images:
            #plot
            (rows, columns) = (4,3)
            plt.subplot(rows, columns, i), plt.imshow(image)
            i+=1
            if(i > 12):
                break
        plt.show(False)

    def annotate_card(self, image, card, contour):
        cX_offset = -20
        cY_offset = -45

        M = cv2.moments(contour)
        cX = int(M['m10']/M['m00']) + cX_offset
        cY = int(M['m01']/M['m00']) + cY_offset

        cv2.putText(image, repr(card.count), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 2)
        cv2.putText(image, repr(card.shape), (cX, cY + 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 2)
        cv2.putText(image, repr(card.fill), (cX, cY + 20 * 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 2)
        cv2.putText(image, repr(card.color), (cX, cY + 20 * 3), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 2)

        return image