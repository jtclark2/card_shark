from colorlabeler import ColorLabeler
from Card import *

import cv2
import numpy as np
import imutils
import os

#min_curvature, aka "epsilon" is the allowable curvature as a fraction of the arclength

MIN_SHAPE_CURVATURE = .02
MIN_SHAPE_SIZE = 10000    #TODO : add max shape size too (sometimes the shapes get combined in blurry images)

SAVE_PATH = "ImageLibrary/%s_%s.jpg"

#TODO: Lots of times, it finds the card and extracts the image, but doesn't show me the outline...
#TODO: Originally, I though it wasn't finding the card, but it must be...I need to investigate
#TODO: However, it may have been related to the Attribute finders returning None...which I've stabilized now?
class AbstractCardAnalyzer:

    def identify_card(self, image):
        """
        Returns a card (which could be specific "Card" class, or a generic Card.
        :param image:
        :return:
        """
        raise NotImplementedError

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

    def create_contrast_mask(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # find contours in the thresholded image and initialize the
        # shape detector
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if imutils.is_cv2() else contours[1]
        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask, contours, -1, 255, -1)

        return mask

    def _distance(self, x, y=None):
        if(y is None):  #Find magnitude by comparing to 0
            y = [0]*len(x)

        assert(len(x) == len(y))

        sum = 0
        for (a,b) in zip(x,y):
            sum += (a-b)**2
        return sum**.5


class HandTunedCardAnalyzer(AbstractCardAnalyzer):
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
        SAVE_PATH = "ImageLibrary/%s_%s.jpg"
        dir = "ImageSet/%s.jpg"
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

    def calibrate_colors(self, key, value):
        if key is 'reset':
            self.cal_sum = np.array([0., 0., 0.])
            self.count = 0
            return

        # Python is struggling with comparison of tuples, so we convert to strings first
        # ...not quite as good as comparing the objects, but it will work


        if (repr(key) in [repr(color) for color in self.colors]):
            self.cal_sum = np.array( [v+s for (v,s) in zip(value,self.cal_sum)] )
            self.count += 1
            self.colors[key] = self.cal_sum / self.count
            print("%s, %s, %s" % (self.colors[key][0], self.colors[key][1], self.colors[key][2]) )

        if (repr(key) in [repr(color) for color in self.edge_colors]):
            self.cal_sum = np.array( [v+s for (v,s) in zip(value,self.cal_sum)] )
            self.count += 1
            self.edge_colors[key] = self.cal_sum / self.count
            print("edge: %s, %s, %s" % (self.edge_colors[key][0], self.edge_colors[key][1], self.edge_colors[key][2]) )


    def identify_card(self, card):
        """
        Operation: Identify the properties of the incoming card image
        :param image: The input image to be processed.
        :return: Card, with appropriately defined color, shape, fill, and count
        """

        gray = cv2.cvtColor(card.image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if imutils.is_cv2() else contours[1]

        SYMBOL_SIZE_THRESH = 1000

        contours = [c for c in contours if cv2.contourArea(c) >= SYMBOL_SIZE_THRESH]


        shape = gray.shape
        mask = np.zeros(shape, np.uint8)
        # cv2.fillPoly(mask, pts=contours, color=(1))
        cv2.drawContours(mask, contours, -1, 255, -1)

        # card.count = self._identify_count(contours)
        # card.shape = self._identify_shape(contours)
        (card.count, card.shape, _) = self._identify_count_and_shape(mask)

        # card.fill = self._identify_fill(card.image, mask, contours)
        # card.color = self._identify_color(card.image, mask, contours)
        (card.color, card.fill) = self._identify_color_and_fill(card.image, mask, contours)

        return card

    def _identify_count_and_shape(self, source_image):
        """
        This is a simple template match on a mask. Rather than trying to extract and overlay
        a single contour, this seemed pretty straightforward.

        A bit slower, but robust to lighting, and much more stable than the other hand-tuned attempts.

        :param mask: Input mask (background = 0, feature_pixels = 255)
        :return: (count, shape)
        """

        #TODO: Recreate library raw
        #Eroded when gathering source images, so I'm doing it here too (should have should raw)
        source_image = cv2.erode(source_image, None, iterations=5)  # Avoid the edge...just want the texture

        best_match_score = -1000
        for template_card in self.mask_library:
            match_score = cv2.matchTemplate(source_image, template_card.image, cv2.TM_CCOEFF_NORMED)
            # cv2.imshow("source", source_image)
            # cv2.imshow("library", template_card.image)
            # cv2.waitKey(200)

            if match_score > best_match_score:
                best_match_score = match_score
                best_match_card = template_card
        # else:
        #     return (None, None, best_match_score)
        # print(template_card, "\t", match_score)
        return (best_match_card.count, best_match_card.shape, best_match_score)

    def _identify_count(self, contours):
        """
        Simply count the contours found.
        :param contours:
        :return:
        """
        count = len(contours)
        if count == 1:
            return Count.one
        elif count == 2:
            return Count.two
        elif count == 3:
            return Count.three
        else:
            return

    def _identify_shape(self, contours):
        # TODO: Each card has 1, 2, or 3 symbols. We could repeat on each and copmare, but lets start simple
        shape = None
        total_area = 0
        for c in contours:
            if cv2.contourArea(c) >= MIN_SHAPE_SIZE:

                #Shape Contour Metrics
                area = cv2.contourArea(c)
                total_area += area

                perimeter = cv2.arcLength(c, True)
                # vertices = cv2.approxPolyDP(c, MIN_SHAPE_CURVATURE * perimeter, True)
                hull = cv2.convexHull(c)
                convex_vertices = cv2.approxPolyDP(hull, MIN_SHAPE_CURVATURE * perimeter, True)
                # area_convex = cv2.contourArea(hull)

                ###Find Shape
                if (len(convex_vertices) == 4):
                    shape = Shape.diamond
                elif ( (perimeter / area) > .027 ):  #TODO: currently dependant on image scale
                    shape = Shape.wisp
                else:
                    shape = Shape.stadium

        return shape

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

        # Create mask for white 'outer' area of card
        shape = inner_mask.shape
        outer_mask = np.zeros(shape, np.uint8) + 255
        idx = (inner_mask != 0)
        outer_mask[idx] = 0

        # Erode, to avoid blurred edge colors
        inner_mask = cv2.erode(inner_mask, None, iterations=15)
        outer_mask = cv2.erode(outer_mask, None, iterations=15)

        # Gather Card Statistics by channel/color
        in_color = []
        out_color = []
        for idx in (0,1,2):
            (inner_mean, inner_std_dev) = cv2.meanStdDev(image[:,:,idx], mask=inner_mask)
            (outer_mean, outer_std_dev) = cv2.meanStdDev(image[:,:,idx], mask=outer_mask)

            in_color.append(inner_mean)
            out_color.append(outer_mean)

        self.corrected_color = [inner/(outer) for (inner,outer) in zip(in_color, out_color)]
        self.corrected_color = np.reshape(self.corrected_color, [3])
        normalization_factor = self._distance(self.corrected_color, np.array([0,0,0]))
        self.corrected_color = self.corrected_color / normalization_factor

        (best_match_key, best_match_signature) = self.find_best_match(self.corrected_color, self.colors)
        if(best_match_key is not None):
            (color, fill) = best_match_key
        else:
            (color, fill) = (None, None)

        #TODO: This part is still pretty buggy
        if(fill == Fill.empty ):
            edge_mask = np.zeros(shape, np.uint8)
            cv2.drawContours(edge_mask, contours, -1, [255,255,255],20)
            edge_mask_3D = np.zeros_like(image)

            mask_im = np.zeros_like(image)
            for idx in (0,1,2):
                channel = image[:,:,idx]
                channel = self.copy_with_mask(channel, edge_mask)
                mask_im[:,:,idx] = channel

            edge_color = []
            for idx in (0, 1, 2):
                (edge_mean, _) = cv2.meanStdDev(image[:, :, idx], mask=edge_mask)
                edge_color.append(edge_mean)

            self.corrected_edge_color = [edge / outer for (edge, outer) in zip(edge_color, out_color)]
            self.corrected_edge_color = np.reshape(self.corrected_edge_color, [3])
            normalization_factor = self._distance(self.corrected_edge_color, np.array([0,0,0]))
            self.corrected_edge_color = self.corrected_edge_color / normalization_factor
            print(self.corrected_edge_color)

            (best_match_key, best_match_signature) = self.find_best_match(self.corrected_edge_color, self.edge_colors)
            (color, _) = best_match_key

        return (color, fill)

    def _identify_fill(self, gray, mask, contours):
        # stripes have std dev of at least 9.9 in my small sample size, and all others are < 3.4
        # Let's start with a thresh of 7

        # Similar process with lighting, though variations in intensity may become a problem...
        # Second pass, I could normalize off the outer edge of the card (which is always empty)

        mask = cv2.erode(mask, None, iterations=10)
        (mean, std_dev,) = cv2.meanStdDev(gray, mask=mask)

        ###Find Fill
        # print(mean, std_dev)
        if (std_dev <10):
            fill = Fill.solid
        elif (mean < 20):
            fill = Fill.striped
        else:
            fill = Fill.empty

        return fill

        # cv2.imshow("mask", inner_mask)
        # cv2.imshow("inv_mask", outer_mask)



        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # (total_mean, total_std_dev) = cv2.meanStdDev(gray)
        # (inner_mean, inner_std_dev) = cv2.meanStdDev(gray, mask=outer_mask)
        # (outer_mean, outer_std_dev) = cv2.meanStdDev(gray, mask=inner_mask)
        # # print( inner_mean / outer_mean )
        #
        # relative_luminosity = inner_mean / outer_mean
        # if (relative_luminosity > .97):
        #     fill = Fill.empty
        # elif (relative_luminosity > .75):
        #     fill = Fill.striped
        # else:
        #     fill = Fill.solid
        # # print(relative_luminosity)

    def _identify_color(self, image, mask, contours):

        cl = ColorLabeler()
        color = cl.label(image, contours)

        if(color == "red"):
            return Color.red
        if(color == "green"):
            return Color.green
        if(color == "blue"):    #Todo: Yeah, I need to remap (2 wrongs make a right...kind of)
            return Color.purple

class TemplateAnalyzer(AbstractCardAnalyzer):
    """
    The TemplateAnalyzer is meant to be a highly generalizeable tool, that does a simple template match,
    comparing the input card against a library of 'known cards'. The analyzer is responsible for
    determining which card is the best match, and how good that match is.
    """

    #TODO: This analyzer kind of sucks at the moment, but could be powerful
    #   -1) This analyzer handles real-world lighting variations terribly!
    #       We could try to normalize against the white card background, or operate in a more
    #       robust color-space, but for now, it kind of stinks
    #   -2) It's a bit expensive to compare against all possible images
    #       I'm not sure if this is really an issue, but it's definitely on my mind, and worth further exploration
    #   +3) Still worth exploring though
    #       Under well-controlled lighting / background, if 1 second cycle time was reasonable, this could
    #       be a very quickly implemented solution, and adding new cards to the 'library' would be fast,
    #       require only a single image, and basically no manual tuning/customization

    SYMBOL_SIZE_THRESH = 1000

    def process_image(self, image):
        """
        Operation: Identify the properties of the incoming card image
        :param image: The input image to be processed.
        :return: A 3D mask (really 2.5D) of the image
        """

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        mask_3D = np.zeros(image.shape, np.uint8)

        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if imutils.is_cv2() else contours[1]
        contours = [c for c in contours if cv2.contourArea(c) >= self.SYMBOL_SIZE_THRESH]

        cv2.drawContours(mask_3D, contours, -1, [255,255,255], -1)

        #Crude smoothing method
        #TODO: would interleaving these in a loop make it more smooth and less "blocky"
        mask_3D = cv2.erode(mask_3D, None, iterations=10)
        mask_3D = cv2.dilate(mask_3D, None, iterations=10)

        return mask_3D

    def identify_card(self, image):

        mask = self.process_image(image)
        masked_image = self.copy_with_mask(image, mask)

        directory_path = r"./ImageSet/Verified/"
        paths = os.listdir(directory_path)
        for filename in os.listdir(r"./ImageSet/Verified/"):
            path = directory_path + filename
            template = cv2.imread(path)
            masked_template = self.copy_with_mask(template, mask)
            res = cv2.matchTemplate(masked_image, masked_template, cv2.TM_CCOEFF_NORMED)

            print(100*(1-res))
            cv2.imshow("template", masked_template)
            cv2.imshow("camera", masked_image)

            break   #Just 1 test image for now

        return Card(Shape.diamond, Color.purple, Count.one, Fill.solid)
        # return Card(shape, color, count, fill)