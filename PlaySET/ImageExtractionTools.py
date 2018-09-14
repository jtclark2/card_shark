import ImageExtractor
import CardAnalyzer
import Card

import cv2
import imutils
import numpy as np

class Librarian:
    """
    The librarian gathers and manipulate stock libraries of data, such as sample images to train on.
    """
    def __init__(self, image_extractor):
        self.image_extractor = image_extractor
        self.card_analyzer = CardAnalyzer.HandTunedCardAnalyzer()

    def gather_images(self):
        cap = cv2.VideoCapture(0)
        analyzer = CardAnalyzer.AbstractCardAnalyzer()
        while (True):
            # Capture frame-by-frame
            ret, frame = cap.read()


            # Process batch
            images = self._extract_images(frame)
            cv2.drawContours(image=frame,
                             contours=self.contours,
                             contourIdx=-1,
                             color=[255,0,255],
                             thickness=2)
            cv2.imshow('frame', frame)

            if(len(images) >= 1):
                mask = analyzer.create_contrast_mask(images[0])
                cv2.imshow("Card Mask", images[0])

            key_input = cv2.waitKey(1)
            if(key_input & 0xFF == ord('s')):
                # self._save_images(images)
                self._save_masks(images)

            if(key_input & 0xFF == ord('q')):
                break

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

    def _save_masks(self, images, mode="Auto"):
        """

        :param images: Images to be saved.
        :param mode: "Manual" or "Auto", determines how naming is performed.
            "Auto" is easier, but only as good as the algorithm you are using, so
            some post processing verification is likely needed
        :return: None
        : Side-Effects: Saves images to the ./ImageSet/name
            All images are saved as .jpg
        """

        for image in images:
            if(mode == "Manual"):
                name = self._get_name_manual(image)
            elif(mode == "Auto"):
                card = self.card_analyzer.identify_card(image)
                if card is None:
                    #revert to manual naming if we fail to identify a card
                    name = self._get_name_manual(image)
                else:
                    name = repr(card.count) + repr(card.shape)
            else:
                raise Exception("I think you made a typo defining processing mode.")

            name = "ImageSet/%s.jpg" % name
            cv2.imwrite(name, mask)

    def _save_images(self, images, mode="Auto"):
        """

        :param images: Images to be saved.
        :param mode: "Manual" or "Auto", determines how naming is performed.
            "Auto" is easier, but only as good as the algorithm you are using, so
            some post processing verification is likely needed
        :return: None
        : Side-Effects: Saves images to the ./ImageSet/name
            All images are saved as .jpg
        """

        for image in images:
            if(mode == "Manual"):
                name = self._get_name_manual(image)
            elif(mode == "Auto"):
                card = self.card_analyzer.identify_card(image)
                # card = None
                if card is None:
                    #revert to manual naming if we fail to identify a card
                    name = self._get_name_manual(image)
                else:   #Semi-automatic
                    card.color = Card.Color.red
                    card.fill = Card.Fill.empty
                    name = repr(card)
            else:
                raise Exception("I think you made a typo defining processing mode.")

            name = "ImageSet/%s.jpg" % name
            cv2.imwrite(name, image)

    def _get_name_manual(self, image):
        name = self._guess_name(image)
        cv2.imshow("card", image)
        cv2.waitKey(1)
        name = input("Name the card: %s" % name)
        cv2.destroyWindow("card")
        return name

    def _extract_images(self, image):
        (all_card_vertices, self.contours) = self.image_extractor.find_cards(image)
        card_images = []
        for idx, card_vertices in enumerate(all_card_vertices):
            card_images.append( image_extractor.transform_and_extract_image(image, card_vertices) )
        return card_images

    def _guess_name(self, image):
        return "card_"

if __name__ == "__main__":
    image_extractor = ImageExtractor.ImageExtractor()
    lib = Librarian(image_extractor)
    lib.gather_images()