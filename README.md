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
- organize and comment a little better...very cool program, but I wrote it a while ago, and while learning opencv.
    - Just improve the organization and readability
- Optimize the set finding algorithm. There is some low hanging fruit here, but even my brute force algorithm
barely takes any processing cycles compared to the image processing.
- Improve color correction. I have a cheap webcam, which struggles under some lighting conditions, particularly very 
yellow, indoor lighting. The color of the striped and empty purple and greens becomes very similar, even to my own eye.
    - A dynamic might help remove this yellowing effect, because this lighting is not even across the image
        - See **WebCam1.PNG** for a good example of this.
    - I'm running in RGB, and converting to something like HSV may simplify these challenges as well.
- I've considered adding an ML classifier...I probably won't at this point, just because the existing tools work quite
well. If I add it, it will just be for fun.
- Script env setup

## File Clean-up Status:
- Camera.py: Good enough for now
- Card.py: Good enough for now
- CardAnalyzer.py: **Needs a lot of work**
- Code Snippets.py: *Everything commented out (might delete)*
- colorlabeler.py: **Needs a lot of work**
- FindCards.py: *Just delete it, I think? No usages right now.*
- ImageExtractionTools.py: *Just delete, I think?  No usages right now.*
- ImageExtractor.py: **Needs review**
- MainSet.py: Good enough for now
- SetPlayer.py: Good enough for now...minor clean-up of comments
- Visualizer.py: Looks ok

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