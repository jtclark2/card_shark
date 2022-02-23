from colorlabeler import ColorLabeler
from Card import *

import cv2
import numpy as np
import imutils
import os


class HandTunedCardAnalyzer:
    """
    This analyzer is created with hand-tuned feature extraction operations. That includes contours and
    metrics derived from those contours, line fits, vertex counting, template matching, etc.
    Modern machine learning (notable CNNs) are not allowed in this filter. This is mostly for my learning,
    and for demonstrating the benefits of each approach.

    Pros / Cons:
        +1) Light weight, and little/no data collection required. Generally, I don't need any sample images ( though I
            may do some limited template matching which would require a simple image library).
        +2) Intuitive / Easy to understand algorithm.
            This model is built to think about card identification in a similar way to a player (me, in this case).
            If it's having trouble identifying a single feature, such as shape, I can go straight to the method
            that deals with that property, and follow the algorithm to understand exactly why it is identifying
            features in a particular way.
        +3) Very flexible and robust within the understood scope.
            This method is tuned to care about only aspects of the card I care about, and nothing else.
            I could easily add another color, or another shape, and since that concept is built into this analyzer,
            it could quickly be adapted.
        +4) Returns concrete and precise answer.
            If I want to measure something, or hone in on a very particular feature, I can be sure that is what
            the algorithm does. I can be fairly confident that unimportant aspects of the image do not impact the
            output in any way.
        +5) Processing speed is usually (not always) fairly fast compared to other methods.
        -6) Extensible is terrible!
            This method will only work for cards in the game of "Set". It doesn't extend to other card types at all!
        7) Only as good as the creator:
            There are many aspects of a hand-tuned approach that are specific to the implementation. If I work on
            them long enough, I can probably make the algorithm really good at handling them, but it just depends
            on how clever I am and how long I work at it.
                Examples:
                    -Robust to lighting?
                    -Overall Accuracy / F-score / Precision / Recall


    I'm allowing template matching to be used (though the TemplateAnalyA hand-tuned analyzer is light-weight (no images/models to store)
    """

    def __init__(self):
        self.mask_library = []
        SAVE_PATH = "CardTemplates/%s_%s.jpg"
        for shape in Shape: # ('stadium', 'wisp', 'diamond'):
            for count in Count: # ('one', 'two', 'three'):
                path = SAVE_PATH % (shape.value, count.value)
                im = cv2.imread(path)
                mask = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                card = Card(image=mask,shape=shape, count=count)
                self.mask_library.append(card)

        self.default_colors = {
                  (None, Fill.empty):(0.58464111,  0.57458059,  0.57275817),
                  (Color.red, Fill.solid):(0.04896699,  0.13915349,  0.98905942),
                  (Color.green, Fill.solid):(0.50235242,  0.85240304,  0.14509002),
                  (Color.purple, Fill.solid):(0.62991833,  0.46480677,  0.62221987),
                  (Color.red, Fill.striped):( 0.50945472,  0.56414269,  0.64976836),
                  (Color.green, Fill.striped):( 0.56960932,  0.62302764,  0.53608002),
                  (Color.purple, Fill.striped):(0.56364584,  0.56323305,  0.60421179),
                  }
        self.colors=self.default_colors

        # Edge pixels get blurred into the white, so the colors are similar, but brighter and whiter
        self.default_edge_colors = {(Color.red, Fill.empty):(0.52004241,  0.53068998,  0.66927127),
                            (Color.green, Fill.empty):(0.58003615,  0.61652044,  0.53241019),
                            (Color.purple, Fill.empty):(0.57135286,  0.5686256,   0.59179459)
                           }
        self.edge_colors = self.default_edge_colors

        self.cal_sum = np.array([0., 0., 0.])  # for the calibration routine
        self.count = 0

    def _distance(self, x, y=None):
        if(y is None):  #Find magnitude by comparing to 0
            y = [0]*len(x)

        assert(len(x) == len(y))

        sum = 0
        for (a,b) in zip(x,y):
            sum += (a-b)**2
        return sum**.5

    def copy_with_mask(self, image, mask):
        """
        Create a copy of an image that has the input image values in the feature region,
        and zeros in the background / masked region.

        :param image: Image to be copied.
        :param mask: Input mask (background / mask = 0, feature_pixels = 255)
        :return: masked image
        """
        masked_image = np.zeros(image.shape, np.uint8)
        idx = (mask != 0)
        masked_image[idx] = image[idx]
        return masked_image

    ### TODO: Strongly consider re-implementing, but it's just adding confusion for now
    # def calibrate_colors(self, key, value):
    #     if key is 'reset':
    #         self.cal_sum = np.array([0., 0., 0.])
    #         self.count = 0
    #         return
    #
    #     # Python is struggling with comparison of tuples, so we convert to strings first
    #     # ...not quite as good as comparing the objects, but it will work
    #
    #
    #     if (repr(key) in [repr(color) for color in self.colors]):
    #         self.cal_sum = np.array( [v+s for (v,s) in zip(value,self.cal_sum)] )
    #         self.count += 1
    #         self.colors[key] = self.cal_sum / self.count
    #         print("%s, %s, %s" % (self.colors[key][0], self.colors[key][1], self.colors[key][2]) )
    #
    #     if (repr(key) in [repr(color) for color in self.edge_colors]):
    #         self.cal_sum = np.array( [v+s for (v,s) in zip(value,self.cal_sum)] )
    #         self.count += 1
    #         self.edge_colors[key] = self.cal_sum / self.count
    #         print("edge: %s, %s, %s" % (self.edge_colors[key][0], self.edge_colors[key][1], self.edge_colors[key][2]) )

    def identify_card(self, card):
        """
        Purpose: Identify the properties of the incoming card image
        :param image: Rectangular image of a card.
        :return: Card, with appropriately defined color, shape, fill, and count
        """

        gray = cv2.cvtColor(card.image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh_val, thresh_img = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


        # TODO: Is this thresh_img.copy needed, or is it just slowing the program down?
        contours = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

        if imutils.is_cv2():
            contours = contours[0]
        if imutils.is_cv3():
            contours = contours[1]
        if imutils.is_cv4(): # This is the version I've tested with
            contours = contours[0]

        MIN_SHAPE_AREA = 1000
        contours = [c for c in contours if cv2.contourArea(c) >= MIN_SHAPE_AREA]

        shape = gray.shape
        mask = np.zeros(shape, np.uint8)
        cv2.drawContours(mask, contours, -1, 255, -1) # fill contours

        # TODO: if we crop this instead, we'll have less image to process, and a better frame rate
        # Mask off border (suppress any edge contours)
        border_width = 50
        mask[:border_width,:] = 0
        mask[-border_width:,:] = 0
        mask[:,:border_width] = 0
        mask[:,-border_width:] = 0

        (card.count, card.shape, _) = self._identify_count_and_shape(mask)
        (card.color, card.fill) = self._identify_color_and_fill(card.image, mask, contours)

        return card

    def _identify_count_and_shape(self, mask):
        """
        Purpose: Identify count and shape of the symbols on a card. There are only 9 possible silhouettes,
            describes by the combinations of 3 shapes and 3 counts. Since this is a reasonable number, it is reasonably
            easy to save idealized versions of these, and compare all future possibilities to those.

            This has proven to be very robust to lighting changes, and much more stable than the other hand-tuned attempts
            at interpreting contours as shapes.

        :param mask: Input mask (background = 0, feature_pixels = 255)
        :return: (count, shape, qualtiy_score)
        """

        #TODO: Recreate library raw: Eroded when gathering source images, so we apply same here
        mask = cv2.erode(mask, None, iterations=5)  # Avoid the edge...just want the texture

        best_match_score = -1000
        for template_card in self.mask_library:
            match_score = cv2.matchTemplate(mask, template_card.image, cv2.TM_CCOEFF_NORMED)

            if match_score > best_match_score:
                best_match_score = match_score
                best_match_card = template_card
        return (best_match_card.count, best_match_card.shape, best_match_score)

    def find_best_match(self, signature, lookup_table, worst_allowable_match = 1000000):
        """
        Find Closest Match, based on lookup table
        :param value: Value to be matched
        :param lookup_table: Lookup table to match against
        :return: Best Match key and value
        """
        smallest_match_error = worst_allowable_match #Init to large value
        best_match_signature = None
        best_match_key = None
        for key in lookup_table:
            possible_match = lookup_table[key]
            matchError = self._distance(signature, possible_match)
            if matchError < smallest_match_error:
                smallest_match_error = matchError
                best_match_key = key
                best_match_signature = possible_match
        return (best_match_key, best_match_signature)

    def _identify_color_and_fill(self, image, inner_mask, contours):

        # def equalize_histogram_layer(img):
        #     # This is a little slow...I think it might be worth it, but keep it in mind
        #     hist, bins = np.histogram(img.flatten(), 256, [0, 256])
        #     cdf = hist.cumsum()
        #     cdf_m = np.ma.masked_equal(cdf, 0)
        #     cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        #     cdf2 = np.ma.filled(cdf_m, 0).astype('uint8')
        #     equalized_img = cdf2[img]
        #     return equalized_img
        #
        # def equalize_histogram_image(img):
        #     equalized_img = np.zeros_like(img)
        #     equalized_img[:, :, 0] = equalize_histogram_layer(img[:, :, 0])
        #     equalized_img[:, :, 1] = equalize_histogram_layer(img[:, :, 1])
        #     equalized_img[:, :, 2] = equalize_histogram_layer(img[:, :, 2])
        #     return equalized_img
        #
        # image = equalize_histogram_image(image)

        # Create mask for white 'outer' area of card
        shape = inner_mask.shape
        outer_mask = np.zeros(shape, np.uint8) + 255
        idx = (inner_mask != 0)
        outer_mask[idx] = 0

        erosion_size = 20 # 3-5 seems like a good range to remove color blur from edges
        erosion_shape = cv2.MORPH_ELLIPSE
        element = cv2.getStructuringElement(erosion_shape, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                 (erosion_size, erosion_size))
        inner_mask = cv2.erode(inner_mask, element, iterations=1)

        erosion_size = 0
        element = cv2.getStructuringElement(erosion_shape, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                 (erosion_size, erosion_size))
        outer_mask = cv2.erode(outer_mask, element, iterations=1)

        contour_hidden_mask = cv2.bitwise_or(inner_mask,outer_mask)
        contour_exposed_mask = cv2.bitwise_not(contour_hidden_mask)

        # Normalize with the outer white portion of the card
        for idx in (0, 1, 2):
            (outer_mean, outer_std_dev) = cv2.meanStdDev(image[:, :, idx], mask=outer_mask)
            normalization_scalar = (128./outer_mean)
            image[:, :, idx] = cv2.multiply(image[:, :, idx],normalization_scalar)

        # color = np.zeros((3))
        # masked = cv2.bitwise_and(image, image, mask=contour_exposed_mask)
        # import random
        # cv2.imshow(f"Show silhouette color-{random.randint(1,1000)}", masked)

        # Average and mask
        # masked = np.zeros_like(image)
        color = np.zeros((1, 1, 3), np.uint8)
        for i in [0,1,2]:
            color[0,0,i] = np.mean(image[:, :, i], where=contour_exposed_mask.astype(bool))
            # masked[:,:,i] = np.mean(image[:, :, i], where=contour_exposed_mask.astype(bool))
        # masked = cv2.bitwise_and(masked, masked, mask=contour_exposed_mask)

        H, S, V = 0, 1, 2
        hsv_color = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
        hue = hsv_color[0][0][H]
        # hue = np.mean(masked[:, :, H], where=contour_exposed_mask.astype(bool))
        # (hue, _) = cv2.meanStdDev(image[:, :, H], mask=contour_exposed_mask)
        color_table = [Color.red, Color.green, Color.purple, Color.red]

        while hue > 180:
            hue -= 180
        red_hue, green_hue, purple_hue, red_hue_wrap = 0, 60, 140, 180
        color_match = [abs(hue-target) for target in [red_hue, green_hue, purple_hue, red_hue_wrap]]
        color_index = np.argmin(color_match)
        card_color = color_table[color_index]

        # Things to know about opencv's implementation of huespace
        # 1) It's from 0-255 (not 0-360). It will rollover after that
        # 2) The standard 0-360 scale corresponds to values of Hue values from 0-180...1 hue = 2 degrees in hue space
        # 3) That means that 180-255 and above is redundant

        # ### Display the outline in the matched hue (for development/testing only)
        # for i in range(0,270,10):
        #     swatch =  np.zeros((150, 250, 3), np.uint8)
        #     swatch[:,:,V] = 255.
        #     swatch[:,:,H] = i
        #     swatch[:,:,S] = 255. # Saturating the color helps a ton, especially for the striped/empty shapes (use before converting back to BGR)
        #     swatch = cv2.cvtColor(swatch, cv2.COLOR_HSV2BGR)
        #     import random
        #     cv2.imshow(f"Swatch-{i}", swatch)

        # print(f"Hue: {card_color} ({hue})")
        # disp_im = np.zeros_like(image)
        # disp_im[:,:,S] = 255. # Saturating the color helps a ton, especially for the striped/empty shapes (use before converting back to BGR)
        # disp_im[:,:,V] = 255.
        # disp_im[:,:,H] = hue
        # disp_im = cv2.bitwise_and(disp_im, disp_im, mask=contour_exposed_mask)
        # disp_im = cv2.cvtColor(disp_im, cv2.COLOR_HSV2BGR)
        # import random
        # cv2.imshow(f"Show silhouette color-{random.randint(0,10)}", disp_im)





        # ### Fill / Texture


        ### Alternate approach - slightly faster, but has some precision issues due to a type conversion
        #   TODO: No reason, you couldn't clean this one up and fix the precision problems though...It's the uint8,
        #   but it's pretty insistent these functions use them...it's just a matter of looking up the docs
        # inner_color = np.zeros((1, 1, 3), np.uint8)
        # background_color = np.zeros((1, 1, 3), np.uint8)
        # # We lose some precision here with the type conversion...not sure if I should worry about it or not...
        # for i in [0,1,2]:
        #     inner_color[0,0,i] = np.mean(image[:, :, i], where=inner_mask.astype(bool))
        #     background_color[0,0,i] = np.mean(image[:, :, i], where=outer_mask.astype(bool))
        #     # (inner_color[0,0,i], _) = cv2.meanStdDev(image[:, :, i], mask=inner_mask)
        #     # (background_color[0,0,i], _) = cv2.meanStdDev(image[:, :, i], mask=outer_mask)
        #
        # # print(inner_color)
        # # print(background_color)
        # hsv_color_inner = cv2.cvtColor(inner_color, cv2.COLOR_BGR2HSV)
        # hsv_color_background = cv2.cvtColor(background_color, cv2.COLOR_BGR2HSV)
        #
        # saturation = hsv_color_inner[0][0][S]
        # background_saturation = hsv_color_background[0][0][S]



        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = np.mean(hsv_image[:, :, S], where=inner_mask.astype(bool))
        background_saturation = np.mean(hsv_image[:, :, S], where=outer_mask.astype(bool))

        saturation_ratio = saturation/background_saturation


        if saturation_ratio > 10:
            fill = Fill.solid
        elif saturation_ratio > 1.3: # ????
            fill = Fill.striped
        else:
            fill = Fill.empty

        # if card_color == Color.purple:
        #     if saturation_ratio > 10:
        #         fill = Fill.solid
        #     elif saturation_ratio > 1.5:  # ????
        #         fill = Fill.striped
        #     else:
        #         fill = Fill.empty

        ### Display the fill region (for development only)
        # hsv_image[:,:,H] = 0.
        # hsv_image[:,:,V] = 255.
        # image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        # image = cv2.bitwise_and(image, image, mask=inner_mask)
        print(f"r{saturation/background_saturation}-s{saturation}-b{background_saturation}-")
        # cv2.imshow(f"r{saturation/background_saturation}-s{saturation}-b{background_saturation}-", image)

        return (card_color, fill)
