import cv2

class ImageExtractionTools:
    def __init__(self):
        pass

    def save_cards(self, image):
        images = extract_images()


        for image in images:
            name = self.guess_name(image)
            cv2.imshow(name, image)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            name = input("Name the card: _%s" % name)
            self.save_image(name, image)
        cv2.destroyAllWindows()

    def extract_images(self):
        (all_card_vertices, card_contours) = find_cards(image)
        cards = []
        card_images = [];
        for idx, card_vertices in enumerate(all_card_vertices):
            card_images.append( transform_and_extract_image(image, card_vertices) )

    def guess_name(self, image):
        return "card_"

    def verify_name(self, name):
        raise NotImplementedError

    def save_image(self, name, image):