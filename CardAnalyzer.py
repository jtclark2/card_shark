from Card import *

import time
import cv2
import numpy as np

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

    def __init__(self, card_shape = (150,100), border_width=10):
        self.border_width = border_width
        self.card_shape = card_shape
        self.diagnostic_mode = True

        self.MIN_SHAPE_AREA = self.card_shape[0]*self.card_shape[1]//20

        # TODO: There are magical hand-tuned numbers throughout this class.
        # I marked these with "TODO: scale to Card Dimensions", and I'm working on cleaning it up
        self.mask_library = []
        SAVE_PATH = "CardTemplates/%s_%s.jpg"
        for shape in Shape: # ('stadium', 'wisp', 'diamond'):
            for count in Count: # ('one', 'two', 'three'):
                path = SAVE_PATH % (shape.value, count.value)
                im = cv2.imread(path)
                mask = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                if mask.shape != card_shape:
                    mask = cv2.resize(mask, (card_shape[1], card_shape[0])) # not a typo - order reversed for this op
                mask = self.crop_standard(mask)
                card = Card(image=mask,shape=shape, count=count)
                self.mask_library.append(card)

        self.cal_sum = np.array([0., 0., 0.])  # for the calibration routine
        self.count = 0

    def crop_standard(self, image):
        """
        Just make sure we're cropping templates and images in a standard way
        :param image: In image of a card/template
        :return: Cropped image
        """
        return image[self.border_width:-self.border_width,self.border_width:-self.border_width]

    def identify_card(self, card):
        """
        Purpose: Identify the properties of the incoming card image: fill, shape, color, count.
        :param image: Rectangular image of a SET card.
        :return: Card, with appropriately defined color, shape, fill, and count
        """
        card.image = self.crop_standard(card.image)

        start = time.perf_counter()
        mask = self.construct_feature_mask(card)
        feature_mask_time = f"{time.perf_counter() - start: .5f}"

        start = time.perf_counter()
        (card.count, card.shape, _) = self._identify_count_and_shape(mask, card)
        count_and_shape_time = f"{time.perf_counter() - start: .5f}"

        start = time.perf_counter()
        (card.color, card.fill) = self._identify_color_and_fill(card.image, mask, card.index)
        color_and_fill_time = f"{time.perf_counter() - start: .5f}"

        # print(feature_mask_time, count_and_shape_time, color_and_fill_time)
        return card

    def construct_feature_mask(self, card):
        start = time.perf_counter()
        # Find contours and build mask
        gray = cv2.cvtColor(card.image, cv2.COLOR_BGR2GRAY)
        gray_time = f"{time.perf_counter() - start: .5f}"
        start = time.perf_counter()

        # remove variation in background
        height, width = gray.shape
        small_size_approx = 50 # rough count in pixels, averaged across height and width
        light_correction_kernel_size_as_fraction = 1./2
        shrink_factor = int((height*width)**.5//small_size_approx) # the smaller the image, the faster the kernel sweep runs, but we want reasonable res
        small_gray = cv2.resize(gray, (width // shrink_factor, height // shrink_factor))
        kernel_size = int((small_gray.shape[0]*small_gray.shape[1])**.5 * light_correction_kernel_size_as_fraction)
        # light_correction = cv2.GaussianBlur(small_gray, (kernel_size * 2 + 1, kernel_size * 2 + 1), 0)
        light_correction = cv2.medianBlur(small_gray, kernel_size*2+1)
        light_correction = cv2.resize(light_correction, (width, height))
        light_correction = light_correction - np.amin(light_correction)
        balanced = cv2.subtract(gray, light_correction)
        light_corr_time = f"{time.perf_counter() - start: .5f}"
        start = time.perf_counter()
        # balanced = gray

        # Find threshold and extract contours
        thresh_val, thresh_img = cv2.threshold(balanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        contour_time = f"{time.perf_counter() - start: .5f}"
        start = time.perf_counter()

        # Quirky behavior in opencv (the magic numbers are major versions)
        major_version = int(cv2.__version__[0])  # I've primarily tested with v4.4.x
        contours = contours[0] if major_version in [2, 4] else contours[1]  # contour formatting is version dependant

        # Filter noisy junk contours
        contours = [c for c in contours if cv2.contourArea(c) >= self.MIN_SHAPE_AREA]
        filter_time = f"{time.perf_counter() - start: .5f}"
        start = time.perf_counter()

        # Draw contours onto a mask
        shape = gray.shape
        mask = np.zeros(shape, np.uint8)
        cv2.drawContours(mask, contours, -1, 255, -1)  # fill contours
        draw_time = f"{time.perf_counter() - start: .5f}"
        # print(gray_time, light_corr_time, contour_time, filter_time, draw_time)

        return mask

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
        :return: (count, shape, quality_score)
        """
        best_match_score = -1
        best_match_card = card # None
        for template_card in self.mask_library:

            # cv2.matchTemplate might be a bit more robust, but it's SLOW..IoU is 15-20x faster
            # match_score = cv2.matchTemplate(mask, template_card.image, cv2.TM_CCOEFF_NORMED)
            match_score = self._intersection_over_union(mask, template_card.image)

            if match_score > best_match_score:
                best_match_score = match_score
                best_match_card = template_card

        # Dev only: diagnostics
        if self.diagnostic_mode:
            for template_card in self.mask_library:

                match_score = self._intersection_over_union(mask, template_card.image)

                if match_score > .9999*best_match_score and card.index==0:
                    label = f"{best_match_card.shape.value}-{best_match_card.count.value}:{best_match_score}"
                    label = "" # simple print (less windows)
                    cv2.imshow(f"Intersection:{label}",
                               cv2.bitwise_and(mask, template_card.image))

                    cv2.imshow(f"Union:{label}",
                               cv2.bitwise_or(mask, template_card.image))
                    print(f"\t{best_match_card.shape.value}-{best_match_card.count.value} | {match_score}")
            print(f"Best_Match_Score for ({best_match_card.shape.value}-{best_match_card.count.value}: {best_match_score}")

        return (best_match_card.count, best_match_card.shape, best_match_score)

    def _identify_color_and_fill(self, image, inner_mask, id):
        # Create mask for white 'outer' area of card
        shape = inner_mask.shape
        outer_mask = np.zeros(shape, np.uint8) + 255
        idx = (inner_mask != 0)
        outer_mask[idx] = 0

        thickness_of_edge_mask_as_fraction = 1./20
        erosion_size = int(((shape[0]*shape[1])**.5)*thickness_of_edge_mask_as_fraction) # 3-5 seems like a good range to remove color blur from edges # TODO: scale to Card Dimensions
        erosion_shape = cv2.MORPH_ELLIPSE
        element = cv2.getStructuringElement(erosion_shape, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                 (erosion_size, erosion_size))
        inner_mask = cv2.erode(inner_mask, element, iterations=1)

        # erosion_size = 0 # TODO: scale to Card Dimensions
        # element = cv2.getStructuringElement(erosion_shape, (2 * erosion_size + 1, 2 * erosion_size + 1),
        #                          (erosion_size, erosion_size))
        # outer_mask = cv2.erode(outer_mask, element, iterations=1)

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


        if self.diagnostic_mode:
            print(f"Hue: {card_color} ({hue})")

            disp_im = np.zeros_like(image)
            disp_im[:,:,S] = 255. # Saturating the color helps a ton, especially for the striped/empty shapes (use before converting back to BGR)
            disp_im[:,:,V] = 255.
            disp_im[:,:,H] = hue
            disp_im = cv2.bitwise_and(disp_im, disp_im, mask=contour_exposed_mask)
            disp_im = cv2.cvtColor(disp_im, cv2.COLOR_HSV2BGR)
            masked_view = cv2.bitwise_and(image, image, mask=contour_exposed_mask)
            cv2.imshow(f"Color: Masked Image", cv2.resize(masked_view, (400,600)))
            cv2.imshow(f"Show silhouette color-{id}", disp_im)





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
        elif saturation_ratio > 2: # ????
            fill = Fill.striped
        else:
            fill = Fill.empty

        ### Display the fill region (for development only)
        if self.diagnostic_mode:
            hsv_image[:,:,V] = 255.
            image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
            image = cv2.bitwise_and(image, image, mask=inner_mask)
            cv2.imshow("Fill: Masked (Value@255)", image)

            print(f"r{saturation/background_saturation}-s{saturation}-b{background_saturation}-")

        return (card_color, fill)