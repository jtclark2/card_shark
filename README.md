# Card Shark (Set)
Using a web cam, this program identifies and plays your favorite card game, "Set"! Cards in a set will be highlighted
with a boarder in the video stream. 

# Known Errors
### TODO:
This runs with opencv 3.X, but I put 4.1.0 in the notes below. This is in anticipation of updates I started on the linux
machine. Those will be out soon.

# Set - How do you play?
[Official Rules on Youtube](https://youtu.be/NzXDfSFQ1c0)
Set is a card game in which 12 cards are laid out, and each player has to pick out sets of 3 cards that follow specific 
patterns. Each card is identified by combination of 4 features: shape, color, and fill, and count. A set is formed,
when each feature is either the same across all three cards, or unqiue across all three cards.
# Example:
Cards:
- (2 green diamonds that are not filled in)
- (2 green stadiums that are not filled in)
- (2 green wisps    that are not filled in)
Explanation:
- All the cards have 2 copies of the symbol.
- All the cards are green.
- All the cards have different shapes.
- All the cards are not filled in  
    
# Future Improvements
- Speed up: When I extract card images, they're usually about ~100x150 pixels pulled from a 640x480 image; however,
I'm currently resizing them way up to 600x400...I could just shrink the card scale, but I have some dependencies
in the card analyzer, where I use some hard-coded values, and now it will be a bit tricky to re-tune everything
at a new scale...very doable, but I'll need to redo a bunch of hand-tuning, which I don't have time for right now
- organize and comment a little better...very cool program, but I wrote it a while ago, and while learning opencv.
    - Just improve the organization and readability
- I've considered adding an ML classifier for individual cards...I probably won't at this point, since the
existing tools work quite well. If I add it, it will just be for fun.
- Script env setup
- Whip up a GUI with all the config options in MainSet.py

# Notes on tools
- Camera Runs in BGR
- HSV space is a little quirky in opencv
    - 0-360 degrees, but 1 byte is 0-255...Opencv chose 0-180 (+1 means 2 degrees Saturation)
    - at 180, a smooth transition wraps back to 0 (so 181 and 1 represent the same saturation value)
    - 255 --> 256 causes the rollover errors you would expect (just shift everything into the 0-180 range)


# Getting Started
This is a lot of fun, and pretty easy to use. The only trick is setting up the environment dependencies (see below).

## Setting up the environment

I recommend using conda to create a new env for this install. I'll list You can try with pip, but some of the version
conflicts get a bit messy(PIL), and imutils isn't available.

`conda create --name card_shark python=3 opencv numpy matplotlib pillow scipy`
`conda activate card_shark`
`conda install -c conda-forge opencv=4.1.0`
`conda install -c conda-forge imutils`

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