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

    def __init__(self, card_shape = (150,100), border_width=0):
        self.border_width = border_width
        self.card_shape = card_shape
        self.diagnostic_mode = False

        self.MIN_SHAPE_AREA = self.card_shape[0]*self.card_shape[1]//20

        # TODO: There are magical hand-tuned numbers throughout this class.
        # I marked these with "TODO: scale to Card Dimensions", and I'm working on cleaning it up
        self.mask_library = []
        LOAD_PATH = "CardTemplates/%s_%s.jpg"
        for shape in Shape: # ('stadium', 'wisp', 'diamond'):
            for count in Count: # ('one', 'two', 'three'):
                path = LOAD_PATH % (shape.value, count.value)
                im = cv2.imread(path)
                mask = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                if mask.shape != card_shape:
                    mask = cv2.resize(mask, (card_shape[1], card_shape[0])) # not a typo - order reversed for this op
                mask = self.crop_standard(mask)
                card = Card(image=mask,shape=shape, count=count)
                self.mask_library.append(card)

        self.cal_sum = np.array([0., 0., 0.])  # for the calibration routine
        self.count = 0

        # calibration defaults
        self.empty_striped_thresh = 1
        self.striped_solid_thresh = 20

        self.hue_table = {"low_red": 5,
                          "green":85,
                          "purple":150,
                          "high_red":185}

    def calibrate_single_color(self, card, color):
        mask, _ = self.construct_feature_mask(card)
        offset_inner_mask, outer_mask = self.create_offset_masks(card.image, mask)
        _, hue = self.identify_color(card.image, offset_inner_mask, outer_mask)

        print(f"Updating Hue Table: ")
        print(f"Old:\tRed = {self.hue_table['low_red']} \tGreen = {self.hue_table['green']} \tPurple = {self.hue_table['purple']}")
        print(f"New:\tRed = {self.hue_table['low_red']} \tGreen = {self.hue_table['green']} \tPurple = {self.hue_table['purple']}")
        if color == Color.green:
            self.hue_table["green"] = hue
        if color == Color.purple:
            self.hue_table["purple"] = hue
        if color == Color.red:
            self.hue_table["low_red"] = hue
            self.hue_table["high_red"] = 180 + hue

    def calibrate(self, cards):
        """
        :param cards: A list of 3 cards. All of them need to be striped. One should be of each color
        :return: None
        Side Effect: Updates calibration values.
        Assumptions: Color are imperfect, but close enough to get it right most of the time. This
        calibration will shift the centers, but if the calibration does not know which color is which,
        it's not going to sort things out

        """
        # print(f"Old Thresholds: -> Empty - ({self.empty_striped_thresh}) - Striped - ({self.striped_solid_thresh}) - Solid <-")
        # saturation_ratios = []
        # for card in cards:
        #     mask = self.construct_feature_mask(card)
        #     offset_inner_mask, outer_mask = self.create_offset_masks(card.image, mask)
        #     fill, saturation_ratio = self.identify_fill(card.image, offset_inner_mask, outer_mask)
        #     saturation_ratios.append(saturation_ratio)
        #
        # self.empty_striped_thresh = (min(saturation_ratios))**0.5 # geometric mean between empty ~1, and the value we just found: a*sqrt(b/a)
        # self.striped_solid_thresh = max(saturation_ratios)*3 # If this is inconsistent, we'll need 3 solids for a geometric mean

        print(f"New Thresholds: Empty - ({self.empty_striped_thresh}) - Striped - ({self.striped_solid_thresh}) - Solid")

        all_colors = [Color.purple, Color.green, Color.red]
        for card in cards:
            if len(all_colors) == 0:
                return
            if card.color in all_colors:
                all_colors.remove(card.color)
                self.calibrate_single_color(card, card.color)

    def crop_standard(self, image):
        """
        Just make sure we're cropping templates and images in a standard way
        :param image: In image of a card/template
        :return: Cropped image
        """
        if self.border_width == 0:
            return image
        return image[self.border_width:-self.border_width,self.border_width:-self.border_width]

    def identify_card(self, card):
        """
        Purpose: Identify the properties of the incoming card image: fill, shape, color, count.
        :param image: Rectangular image of a SET card.
        :return: Card, with appropriately defined color, shape, fill, and count
        """
        if self.diagnostic_mode:
            print(f"############# Index: {card.index} #############")
        card.image = self.crop_standard(card.image)

        start = time.perf_counter()
        mask, contours = self.construct_feature_mask(card)
        feature_mask_time = f"{time.perf_counter() - start: .5f}"

        start = time.perf_counter()
        card.count = self._identify_count(contours)
        card.shape = self._identify_shape(contours,card)
        # (card.count, card.shape, _) = self._identify_count_and_shape(mask, card, contours)
        count_and_shape_time = f"{time.perf_counter() - start: .5f}"

        start = time.perf_counter()
        (card.color, card.fill) = self._identify_color_and_fill(card, mask, card.index)
        color_and_fill_time = f"{time.perf_counter() - start: .5f}"

        # print(feature_mask_time, count_and_shape_time, color_and_fill_time)
        return card

    def _identify_count(self, contours):
        if len(contours) == 1:
            count = Count.one
        elif len(contours) == 2:
            count = Count.two
        elif len(contours) == 3:
            count = Count.three
        else:
            count = None

        if self.diagnostic_mode:
            print(count)

        return count

    def _identify_shape(self, contours, card):
        if len(contours) == 0:
            return None

        hull = cv2.convexHull(contours[0])
        x, y, w, h = cv2.boundingRect(contours[0])

        area_box = w*h
        area_hull = cv2.contourArea(hull)
        area_contour = cv2.contourArea(contours[0])

        if (area_contour / area_hull < .9):
            shape = Shape.wisp
        else:
            if (area_contour / area_box < .7):
                shape = Shape.diamond
            else:
                shape = Shape.stadium

        if self.diagnostic_mode:
            print(shape)
            print(area_contour / area_box, area_contour / area_hull, area_box, area_hull, area_contour)
            display_image = card.image.copy()
            cv2.drawContours(display_image, contours, -1, color=[0, 255, 255], thickness=4)
            cv2.drawContours(display_image, hull, -1, color=[255, 0, 255], thickness=4)
            cv2.rectangle(display_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow("Card Contours", display_image)
        return shape

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
        contours = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

        return mask, contours

    def _intersection_over_union(self, im1, im2):
        intersection = np.sum(cv2.bitwise_and(im1, im2))
        union = np.sum(cv2.bitwise_or(im1, im2))
        return intersection/union

    def _identify_count_and_shape(self, mask, card, contours):
        """
        Purpose: Identify count and shape of the symbols on a card. There are only 9 possible silhouettes,
            describes by the combinations of 3 shapes and 3 counts. Since this is a reasonable number, it is reasonably
            easy to save idealized versions of these, and compare all future possibilities to those.

            This was my primary approach for a while, but I realized I could hand-tune them, and have now broken this
            into _identify_countm and _identify_shape. I'm leaving this in case I ever want to generalize to other
            card games. It still works very well, and with new silhouettes, it can work with any deck!

        Advantages: Generalizes very well to new games (just grab a new set of silhouettes). Reasonably robust...works
        99.9% of the time.
        Distadvantages:
            - Cards and camera need to be perfectly still...as they blur, everything starts to look more round,
             like a stadium.

        :param mask: Input mask (background = 0, feature_pixels = 255)
        :return: (count, shape, quality_score)
        """

        self.junk_shape_thresh = .6
        best_match_score = self.junk_shape_thresh
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
            print("#"*50)
            if best_match_card.shape is not None:
                cv2.imshow(f"Intersection:",
                           cv2.resize(cv2.bitwise_and(mask, best_match_card.image), (400, 600)))

                cv2.imshow(f"Union:",
                           cv2.resize(cv2.bitwise_or(mask, best_match_card.image), (400, 600)))
                print(f"Shape:({best_match_card.shape.value}, Count: {best_match_card.count.value}, score: {best_match_score}")
            else:
                print(f"Shape: None, Fill: None...All scores below: {self.junk_shape_thresh} ")

        return (best_match_card.count, best_match_card.shape, best_match_score)

    def _identify_color_and_fill(self, card, inner_mask, id):
        offset_inner_mask, outer_mask = self.create_offset_masks(card.image, inner_mask)

        card_color, _ = self.identify_color(card.image, offset_inner_mask, outer_mask)
        card.color = card_color

        fill, _ = self.identify_fill(card, offset_inner_mask, outer_mask)

        return (card_color, fill)

    def create_offset_masks(self, image, inner_mask):
        # Create mask for white 'outer' area of card
        shape = inner_mask.shape
        outer_mask = np.zeros(shape, np.uint8) + 255
        idx = (inner_mask != 0)
        outer_mask[idx] = 0
        thickness_of_edge_mask_as_fraction = 1. / 15
        erosion_size = int(((shape[0] * shape[
            1]) ** .5) * thickness_of_edge_mask_as_fraction)  # 3-5 seems like a good range to remove color blur from edges # TODO: scale to Card Dimensions
        erosion_shape = cv2.MORPH_ELLIPSE
        element = cv2.getStructuringElement(erosion_shape, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                            (erosion_size, erosion_size))
        offset_inner_mask = cv2.erode(inner_mask, element, iterations=1)
        # Normalize with the outer white portion of the card
        for idx in (0, 1, 2):
            (outer_mean, outer_std_dev) = cv2.meanStdDev(image[:, :, idx], mask=outer_mask)
            normalization_scalar = (128. / outer_mean)
            image[:, :, idx] = cv2.multiply(image[:, :, idx], normalization_scalar)
        return offset_inner_mask, outer_mask

    def identify_fill(self, card, offset_inner_mask, outer_mask):
        image = card.image
        # Fill / Texture
        H, S, V = 0, 1, 2
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = np.mean(hsv_image[:, :, S], where=offset_inner_mask.astype(bool))
        background_saturation = np.mean(hsv_image[:, :, S], where=outer_mask.astype(bool))
        # saturation_ratio = saturation / background_saturation # Tried this...the background was adding too much noise
        saturation_diff = saturation - background_saturation


        if card.color is not None:
            # R:5, P:1, G:3
            if card.color == Color.purple:
                color_correction = 1./1
            if card.color == Color.red:
                color_correction = 1./5
            if card.color == Color.green:
                color_correction = 1./2

            saturation_diff *= color_correction

        if saturation_diff > self.striped_solid_thresh:    # 9 is a reasonable value
            fill = Fill.solid
        elif saturation_diff > self.empty_striped_thresh:  # 1.5 is a reasonable value
            fill = Fill.striped
        else:
            fill = Fill.empty

        ### Display the fill region (for development only)
        if self.diagnostic_mode:
            hsv_image[:, :, V] = 255.
            image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
            fill_img = cv2.bitwise_and(image, image, mask=offset_inner_mask)
            background_img = cv2.bitwise_and(image, image, mask=outer_mask)
            cv2.imshow("Fill: Masked (Value@255)", cv2.resize(fill_img, (400, 600)))
            cv2.imshow("Background: Masked (Value@255)", cv2.resize(background_img, (400, 600)))

            print(f"Fill/Texture: {fill} Difference: {saturation_diff}, Inner:{saturation}, Background: b{background_saturation}-")
        return fill, saturation_diff

    def identify_color(self, image, offset_inner_mask, outer_mask):
        H, S, V = 0, 1, 2

        contour_hidden_mask = cv2.bitwise_or(offset_inner_mask, outer_mask)
        contour_exposed_mask = cv2.bitwise_not(contour_hidden_mask)
        # Average and mask
        color = np.zeros((1, 1, 3), np.uint8)
        for i in [0, 1, 2]:
            color[0, 0, i] = np.mean(image[:, :, i], where=contour_exposed_mask.astype(bool))
            # (color[0,0,i], _) = cv2.meanStdDev(image[:, :, i], mask=contour_exposed_mask) # handle precision/types differently than np

        # Things to know about opencv's implementation of HSV space (specifically hue)
        # 1) It's from 0-255 (not 0-360). It will rollover after that
        # 2) The standard 0-360 scale corresponds to values of Hue values from 0-180...1 hue = 2 degrees in hue space
        # 3) That means that 180-255 and above is redundant
        hsv_color = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
        hue = int(hsv_color[0][0][H]) # cast to ensure we can do some math without worrying about roll over
        color_table = [Color.red, Color.green, Color.purple, Color.red]

        while hue > 180:
            hue -= 180
        hue_error = [abs(hue - ideal_hue) for _, ideal_hue in self.hue_table.items()]
        color_index = np.argmin(hue_error)
        card_color = color_table[color_index]
        # print(hue, hue_error, color_index, card_color)
        if self.diagnostic_mode:
            print(f"Hue: {card_color} ({hue})")
            if card_color == Color.red and hue == 150:
                print("problems")

            # Maximize satuation and value, and use hue to show a BGR color image with intuitive hue information
            disp_im = np.zeros_like(image)
            disp_im[:, :, S] = 255.
            disp_im[:, :, V] = 255.
            disp_im[:, :, H] = hue
            disp_im = cv2.bitwise_and(disp_im, disp_im, mask=contour_exposed_mask)
            disp_im = cv2.cvtColor(disp_im, cv2.COLOR_HSV2BGR)
            cv2.imshow(f"Show silhouette color-{id}", cv2.resize(disp_im, (400, 600)))

            masked_view = cv2.bitwise_and(image, image, mask=contour_exposed_mask)
            cv2.imshow(f"Color: Masked Image", cv2.resize(masked_view, (400, 600)))
        return card_color, hue