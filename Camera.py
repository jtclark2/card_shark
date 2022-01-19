import cv2


class Camera:
    def __init__(self, camera=0):
        self.cap = cv2.VideoCapture(camera)

    def configure(self):
        # https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html
        # CV_CAP_PROP_POS_MSEC = 0
        # CV_CAP_PROP_POS_FRAMES = 1
        # CV_CAP_PROP_POS_AVI_RATIO =  2
        CV_CAP_PROP_FRAME_WIDTH = 3
        CV_CAP_PROP_FRAME_HEIGHT = 4
        CV_CAP_PROP_FPS = 5
        CV_CAP_PROP_FOURCC = 6
        CV_CAP_PROP_FRAME_COUNT = 7
        CV_CAP_PROP_FORMAT = 8
        CV_CAP_PROP_MODE = 9
        CV_CAP_PROP_BRIGHTNESS = 10
        CV_CAP_PROP_CONTRAST = 11
        CV_CAP_PROP_SATURATION = 12
        CV_CAP_PROP_HUE = 13
        CV_CAP_PROP_GAIN = 14
        CV_CAP_PROP_EXPOSURE = 15
        CV_CAP_PROP_CONVERT_RGB=16
        CV_CAP_PROP_WHITE_BALANCE_U = 17
        CV_CAP_PROP_WHITE_BALANCE_V = 18
        CV_CAP_PROP_RECTIFICATION = 19
        CV_CAP_PROP_ISO_SPEED = 20
        CV_CAP_PROP_BUFFERSIZE = 21

        # Intentionally set to specific values
        self.cap.set(CV_CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(CV_CAP_PROP_FRAME_HEIGHT, 1080)
        # self.cap.set(CV_CAP_PROP_SATURATION, 100) # 0 - 100, values outside this range are ignored
        # self.cap.set(CV_CAP_PROP_HUE, 100)

        # self.cap.set(CV_CAP_PROP_MODE, 1)
        # self.cap.set(CV_CAP_PROP_FPS, 1)
        # self.cap.set(CV_CAP_PROP_BRIGHTNESS, 10)
        # self.cap.set(CV_CAP_PROP_CONTRAST, 255)
        # self.cap.set(CV_CAP_PROP_CONVERT_RGB, True)

        # set so that they wouldn't auto-adjust on me

        # Not supported with my hardware
        # cap.set(CV_CAP_PROP_WHITE_BALANCE_U, 99) # not supported with my current camera
        # cap.set(CV_CAP_PROP_WHITE_BALANCE_V, 99) # not supported with my current camera
        # cap.set(CV_CAP_PROP_EXPOSURE, -3) # not supported with my current camera
        # cap.set(CV_CAP_PROP_GAIN, 0) # not supported with my current camera

    def read(self):
        """
        Return:
            frame: I'm suppressing the bool. Just check if is None.
        """
        found_frame, frame = self.cap.read()
        if (frame is None):
            print("Available Camera Connection not found. Connect Camera, "
                  "and ensure camera is not owned by other instance.")
            self.release()
        return frame

    def release(self):
        self.cap.release()

    # def calibrate(self):
    #     """
    #     Probably not going to use this...I'm having some trouble with saturation on my detachable webcam, so
    #     lots of cards are getting washed out...I hoped to calibrate for lighting changes, but realistically
    #     the algorithm is quite robust if the data is there. My problems are with oversaturation, and there
    #     is no way around that in software
    #     """
        # cal_lookup = {
        #      ord('0') : (None, Card.Fill.empty),
        #      ord('1') : (Card.Color.red, Card.Fill.empty),
        #      ord('2'): (Card.Color.green, Card.Fill.empty),
        #      ord('3'): (Card.Color.purple, Card.Fill.empty),
        #      ord('4'): (Card.Color.red, Card.Fill.striped),
        #      ord('5'): (Card.Color.green, Card.Fill.striped),
        #      ord('6'): (Card.Color.purple, Card.Fill.striped),
        #      ord('7'): (Card.Color.red, Card.Fill.solid),
        #      ord('8'): (Card.Color.green, Card.Fill.solid),
        #      ord('9'): (Card.Color.purple, Card.Fill.solid),
        #      ord('r'): 'reset',
        # }
        #
        # edge_set = (ord('1'), ord('2'), ord('3'))
        #
        # key_stroke = (key_input & 0xFF)
        # if ( key_stroke in cal_lookup): # Calibrate
        #     if (len(game.card_images) == 1):
        #         card = Card.Card(index=0, image=game.card_images[0])
        #         key = cal_lookup[key_input]
        #         if(key_stroke in edge_set):
        #             try:
        #                 value = card_analyzer.corrected_edge_color
        #                 card_analyzer.calibrate_colors(key=key, value=value)
        #             except:
        #                 print("This card does not appear to have fill='Empty'.")
        #         else: # any other
        #             value = card_analyzer.corrected_color
        #             print(value)
        #             card_analyzer.calibrate_colors(key=key, value=value)
