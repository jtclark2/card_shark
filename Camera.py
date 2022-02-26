import cv2


class Camera:
    def __init__(self, camera=0):
        self.cap = cv2.VideoCapture(camera)
        self.shape = (None, None)

    def configure(self):
        """
        This class was intended to help stabilize the camera. Webcams often perform some light balance and color
        correction that I was trying to disable. I've found the available features are very specific to the camera,
        so I've stopped using them for now, but I'll leave this here for the moment.
        :return:
        """
        # https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html
        CV_CAP_PROP_POS_MSEC = 0
        CV_CAP_PROP_POS_FRAMES = 1
        CV_CAP_PROP_POS_AVI_RATIO =  2
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

        # Get basic info
        print(f"Camera Resolution: ({self.cap.get(CV_CAP_PROP_FRAME_WIDTH)}, {self.cap.get(CV_CAP_PROP_FRAME_HEIGHT)})")
        self.shape = (int(self.cap.get(CV_CAP_PROP_FRAME_WIDTH)), int(self.cap.get(CV_CAP_PROP_FRAME_HEIGHT)))

        # Intentionally set to specific values
        # self.cap.set(CV_CAP_PROP_FRAME_WIDTH, 1920)
        # self.cap.set(CV_CAP_PROP_FRAME_HEIGHT, 1080)
        # self.cap.set(CV_CAP_PROP_SATURATION, 100) # 0 - 100, values outside this range are ignored
        # self.cap.set(CV_CAP_PROP_HUE, 100)

        # self.cap.set(CV_CAP_PROP_MODE, 1)
        self.cap.set(CV_CAP_PROP_FPS, 4)
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
        # if (frame is None):
        #     print("Available Camera Connection not found. Connect Camera, "
        #           "and ensure camera is not owned by other instance.")
        #     self.release()
        return frame

    def release(self):
        self.cap.release()