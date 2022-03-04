# Card Shark (Set)
Using a web cam, this program identifies and plays your favorite card game, "Set"! Cards in a set will be highlighted
with a boarder in the video stream. 

# Set - How do you play?
[Official Rules on Youtube](https://youtu.be/NzXDfSFQ1c0)
Set is a card game in which 12 cards are laid out, and each player has to pick out sets of 3 cards that follow specific 
patterns. Each card is identified by combination of 4 features: shape, color, and fill, and count. A set is formed,
when each feature is either the same across all three cards, or unqiue across all three cards.
# Example:
Cards:
- (2 hollow green diamonds)
- (2 hollow green stadiums)
- (2 hollow green wisps   )

Explanation:
- All the cards have **2** copies of each shape.
- All the shapes are **green**.
- All the cards have **different** shapes.
- All the card are **hollow** inside the shapes
    
# Future Improvements
- Magic Numbers: I'm not proud of it, but a lot of them got in there while I was tuning. I need to consolidate 
configurable parameters and add a few comments explaining them
- Env setup instructions/script...dependencies are listed below, and it's not too hard, but PIL and opencv
can get finnicky if you don't grab the right versions, so I should add an explicit setup file
- Calibration: Could automate calibration further for tuning threshold...(fastish) unsupervised clustering algorithm would be great!
- ML classifier to identify cards...I probably won't at this point, since the existing tools work quite well. 
If I add it, it will just be for fun.
- Whip up a GUI with all the config options in MainSet.py. I should probably do this, but I won't, because I don't
really like building GUIs...If you're reading this, and you enjoy building UIs, I'd welcome the help!
- For the shape detection, I only use 1, even if there are 2 or 3 available...no issues now, but I am throwing out
potentially valuable info

# Notes on tools
- PIL and opencv work well together, but make sure you have the correct versions. They are a bit finnicky,
and I've found that the package managers don't always get it right (I think the build-dependencies were not updated
in a few versions of PIL...not sure though)
- HSV space is a little quirky in opencv
    - 0-360 degrees, but 1 byte is 0-255...Opencv chose 0-180 (+1 means 2 degrees Saturation)
    - at 180, a smooth transition wraps back to 0 (so 181 and 1 represent the same saturation value)
    - 255 --> 256 causes the rollover errors you would expect (just shift everything into the 0-180 range)

# Getting Started
1) The environment setup is described below. Start with that.
2) Once you're running, configure the inputs in `MainSet.py`. 
    - Select SOURCE_TYPE as "image" or "video"
    - set the input (either an image, or video source)
        - video source can be a file, or camera (0 based index)
3) While running, there are a handful of keyboard inputs available.
    - 's': **save** the current frame (useful for grabbing key frames from video
    - **calibration**: These usually are not needed, but they're here in case!
        - 'c': Automatic calibration (tunes all the colors/hues), provided they are on screen
        - 'r': Calibrate visible card to be the new ground truth of **Red**
        - 'g': Calibrate visible card to be the new ground truth of **Green**
        - 'p': Calibrate visible card to be the new ground truth of **Purple**
        - '[', ']', '(', ')': Tune Fill/texture thresholds (hollow, striped, solid)
    - 'q': **quit** (closes all windows and any active camera connection)
    - ' ' (space): **pause**

## Setting up the environment

I recommend using conda to create a new env for this install. I haven't tested recently...There were some
frustrating conflicts between version...hopefully aren't anymore...imutils has been removed, but major version
changes in opencv were not always tracked by PIL, so you may want to make sure both of those are up-to-date.

`conda create --name card_shark python=3 opencv numpy matplotlib pillow scipy`
`conda activate card_shark`

Just got it to work, but I can't isolate all the dependencies...I'll leave a hodgepodge of
notes here, and start trying to isolate the dependencies later:
- opencv...this installs fine, but for whatever reason, the Windows build of pycharm can't seem
to load cv2...I have no idea why (maybe it's import cv or cv4 instead, or something like that, 
    - 4.0.1... I also have opencv-python 4.4.0.44, but I'm not sure if that's relevant
but it doesn't work, so I ended up cloning my ml2 env instead)
- numpy (1.18.5 according to conda list, but the install record shows numpy-1.22.2)
- matplotlib (3.2.2)
- "Pillow" (not to be confused with "pillow")...again, I don't know what the deal is,
but there are two versions, and upper and lower case. I eventually got upper case to work 
`pip install --upgrade Pillow`...I ended up with Pillow-9.0.1
- Then I had a version mismatch with scipy, and ended up upgrading with 
`python -m pip install scipy --upgrade --force`, which upgraded to numpy-1.22.2, and scipy-1.8.0

- scipy (1.8.0)


On linux:
`sudo apt-get install libgtk2.0-dev`
`sudo apt-get install pkg-config`


# References that made this possible
Image Processing:
https://www.pyimagesearch.com/2016/02/08/opencv-shape-detection/
https://www.pyimagesearch.com/2016/02/15/determining-object-color-with-opencv/

Thresholding:
https://docs.opencv.org/3.4.0/d7/d4d/tutorial_py_thresholding.html

Contours:
https://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html

Transform / Projection
https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html

Histograms (didn't end up using this, but helped to investigate what was happening)
https://docs.opencv.org/3.1.0/d1/db7/tutorial_py_histogram_begins.html

And the opencv docs, which humble you, because...
After spending hours masking and learning to run stats on images...turns out this one liner did it :)
https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#void%20meanStdDev(InputArray%20src,%20OutputArray%20mean,%20OutputArray%20stddev,%20InputArray%20mask)
https://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=drawcontours
https://docs.opencv.org/3.0-beta/modules/refman.html