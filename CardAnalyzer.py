from Card import *

import cv2
import numpy as np
import imutils

class HandTunedCardAnalyzer:
    """
    This analyzer is created with hand-tuned feature extraction operations. I may create a CnnAnalyzer in the future,
    but I already have a bunch of projects with those, whereas this will teach me more about color-space transformations
    and traditional techniques

    Pros / Cons:
        +1) Light weight, and minimal data collection required (just the shape templates).
        +2) Intuitive / Easy to debug: If it's having trouble identifying a single attribute, I can go straight to
            the associated method and understand it is failing.
        +5) Processing speed is usually (not always) fairly fast compared to other methods...it's taking .2s per frame,
        which is plenty for my needs, but there is definitely still some low hanging fruit
        -6) Extensible: Not really. The underlying methods could be re-used, but the code is definitely specific to the
            SET deck and the current Card object.
            - If I moved the core of these methods (the image manipulation part) into an tools class, or an abstract
            base class, I don't think it would be too hard to then create different Card and Analyzer object for
            different decks (eg: SET, standard playing cards, etc.) I have tools to interpret shapes, colors, etc.
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

        self.cal_sum = np.array([0., 0., 0.])  # for the calibration routine
        self.count = 0

    def identify_card(self, card):
        """
        Purpose: Identify the properties of the incoming card image: fill, shape, color, count.
        :param image: Rectangular image of a SET card.
        :return: Card, with appropriately defined color, shape, fill, and count
        """

        gray = cv2.cvtColor(card.image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # This is a bit of a lazy hack to clean up the binarization (the opencv method doesn't quite have the flexibility
        # Adding a border basically tricks the threshold into shifting...this is NOT the right way to do this, but it's
        # a quick patch for now.
        # Eventually, I intend to use an adaptive threshold, fill the gaps in the shape, and findContours on that...
        # The tricky bit is connecting the shape in a smart way, (and not slowing the program down too much)
        # border_width = 5
        # border_val = 120
        # blurred[:border_width,:] = border_val
        # blurred[-border_width:,:] = border_val
        # blurred[:,:border_width] = border_val
        # blurred[:,-border_width:] = border_val

        height, width = blurred.shape
        small_gray = cv2.resize(blurred, (int(width/10), int(height/10)))
        light_correction = cv2.GaussianBlur(small_gray, (int(width/20)*2+1, int(width/20)*2+1), 0) #Reduce noise in the image
        light_correction = cv2.resize(light_correction, (width, height))
        light_correction = light_correction - np.amin(light_correction)
        new_blurred = light_correction
        #Todo: Division makes more sense, but gives us a type mistmatch - resolve later
        blurred = cv2.subtract(blurred, light_correction)

        thresh_val, thresh_img = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # thresh_img = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
        #                            cv2.THRESH_BINARY, 11, 2)
        # cv2.imshow("new_blurred", new_blurred)
        # cv2.imshow("light_correction", light_correction)
        # cv2.imshow("blurred", blurred)
        # cv2.imshow("thresh", thresh_img)

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

        # TODO: crop instead of masking for speed boost (400x600 vs. 300x500 --> 5/8 the pixels
        # I'll need to regenerate the templates though...so for now, I'll just crop as I pass into color_and_fill
        # Mask off border (suppress any edge contours - all card content is in the center)
        border_width = 50
        mask[:border_width,:] = 0
        mask[-border_width:,:] = 0
        mask[:,:border_width] = 0
        mask[:,-border_width:] = 0
        # cv2.imshow("mask", mask)

        (card.count, card.shape, _) = self._identify_count_and_shape(mask, card)
        (card.color, card.fill) = self._identify_color_and_fill(
            card.image[border_width:-border_width,border_width:-border_width],
            mask[border_width:-border_width,border_width:-border_width])

        return card

    def _intersection_over_union(self, im1, im2):
        intersection = np.sum(cv2.bitwise_and(im1, im2))
        union = np.sum(cv2.bitwise_or(im1, im2))
        return intersection/union

    def _identify_count_and_shape(self, mask, card):
        """
        Purpose: Identify count and shape of the symbols on a card. There are only 9 possible silhouettes,
            describes by the combinations of 3 shapes and 3 counts. Since this is a reasonable number, it is reasonably
            easy to save idealized versions of these, and compare all future possibilities to those.

            This has proven to be very robust to lighting changes, and much more stable than the other hand-tuned attempts
            at interpreting contours as shapes.

        :param mask: Input mask (background = 0, feature_pixels = 255)
        :return: (count, shape, qualtiy_score)
        """
        best_match_score = -1
        best_match_card = card # None
        for template_card in self.mask_library:

            # matchTemplate might be more robust??? However, IOU is definititely faster by 15-20x
            # match_score = cv2.matchTemplate(mask, template_card.image, cv2.TM_CCOEFF_NORMED)
            match_score = self._intersection_over_union(mask, template_card.image)

            if match_score > best_match_score:
                best_match_score = match_score
                best_match_card = template_card

        # Dev only: diagnostics
        #     print(f"\tspam...{template_card.shape.value}-{template_card.count.value} | {match_score}")
        #
        # for template_card in self.mask_library:
        #
        #     match_score = self._intersection_over_union(mask, template_card.image)
        #
        #     if match_score > .99*best_match_score:
        #         cv2.imshow(f"Intersection:{best_match_card.shape.value}-{best_match_card.count.value}:{best_match_score}",
        #                    cv2.bitwise_and(mask, template_card.image))
        #
        #         cv2.imshow(f"Union:{best_match_card.shape.value}-{best_match_card.count.value}:{best_match_score}",
        #                    cv2.bitwise_or(mask, template_card.image))
        #         print(f"\t{best_match_card.shape.value}-{best_match_card.count.value} | {match_score}")
        # print(f"Best_Match_Score for ({best_match_card.shape.value}-{best_match_card.count.value}: {best_match_score}")

        return (best_match_card.count, best_match_card.shape, best_match_score)

    def _identify_color_and_fill(self, image, inner_mask):

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

        # Average and mask
        color = np.zeros((1, 1, 3), np.uint8)
        for i in [0,1,2]:
            color[0,0,i] = np.mean(image[:, :, i], where=contour_exposed_mask.astype(bool))

        H, S, V = 0, 1, 2
        hsv_color = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
        hue = hsv_color[0][0][H]
        # hue = np.mean(masked[:, :, H], where=contour_exposed_mask.astype(bool))
        # (hue, _) = cv2.meanStdDev(image[:, :, H], mask=contour_exposed_mask)
        color_table = [Color.red, Color.green, Color.purple, Color.red]

        while hue > 180:
            hue -= 180
        red_hue, green_hue, purple_hue, red_hue_wrap = 5, 85, 150, 185
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
        # cv2.imshow(f"Show silhouette color", disp_im)
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


        if saturation_ratio > 9:
            fill = Fill.solid
        elif saturation_ratio > 1.3: # ????
            fill = Fill.striped
        else:
            fill = Fill.empty

        # Consider separate thresholds by color: empty should all be the same, but striped and solid vary
        # Purple tends to be a bit lower than green and red)...
        #    Alternately, we could add in color analysis on the center, so see if we find colored fill
        #    Note that detecting the "stripe" pattern is not helpful...It works in high quality images; however,
        #    everything works in high quality images. The stripes blur into non-existance before this method fails.

        ### Display the fill region (for development only)
        # hsv_image[:,:,H] = 0.
        # hsv_image[:,:,V] = 255.
        # image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        # image = cv2.bitwise_and(image, image, mask=inner_mask)
        # cv2.imshow(f"r{saturation/background_saturation}-s{saturation}-b{background_saturation}-", image)

        print(f"r{saturation/background_saturation}-s{saturation}-b{background_saturation}-")

        return (card_color, fill)