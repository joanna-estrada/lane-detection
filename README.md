# TODO:
- Run the contents of this README.md file through chatgpt to make what I've written more succinct and clear (rather than the rambling nonsense that's in here).
- Create proper directories and sub-directories (i.e. /src, /src/test_videos)
    - NOTE: this will likely require changing some things with the video path's,
        which might require to import the 'os' python module to make it work more efficiently
- Remove the data class 'Line'. Its only used twice (Lines are represented by numpy arrays and at the time, I needed something easier to work with).
- Change variable names to be more appropriate, like : 'gang_shit_no_lame_shit'
    ps. look around line 650 to see what that variable is for.
- implement typing for the functions and methods
    ```python 
    import typing
    ```
    if you don't know a type of some function(as many passed are numpy or openCV objects) just run this code in the function:
    ```python 
    def foo(unknown_type):
        print(type(unknown_type))
        exit()
    ```
- implement pep-8 stylizing


# Lane Detection using Python and Open CV
## Project Goal
This project leverages the tools in Python and OpenCV to detect lanes in dash cam footage. My goal was to become more acquaintted with AI and computer vision algorithms to in my software to identify straight lanes, providing real-time visual feedback to users.

## What was implemented
In computer vision, a common tool for isolating objects in an image comes in the form of <masking>. To mask an image, we need to create a completely black (blank) image with the same height and width as the image we intend to mask. When we place a white polygon, such as a triangle or a square, within the blank image, we can merge this with the original, resulting in a new image containing contents of the original image only within the boundaries of the polygon.

```python
# NOTE: This exact code isn't in my program, but demonstrates how masks work
# copy and paste it into a new .py file to view this demo
import cv2
import numpy as np

# will save the image as a 2D array of pixel values
img = cv2.imread('lane.jpg')
img_height, img_width = img.shape[0],img.shape[1]

# create a blank with the same height and width (also a 2d numpy array, but with all 0s representing black pixels)
blank = np.zeros((img_height, img_width), dtype=np.uint8)

# creating a rectangular mask to cut out the top 1/3 of the image
rect_mask = cv2.rectangle(blank.copy(), (0, img_height//3+20), (img_width, img_height), 255, -1)
r_masked_img = cv2.bitwise_and(img, img, mask = rect_mask)

#displaying the images
cv2.imshow('Original Image', img)
cv2.imshow('Mask', rect_mask)
cv2.imshow('Masked Original Image', r_masked_img)

# press any key to exit
cv2.waitKey(0)
cv2.destroyAllWindows()
```
For dash cam frames, I went with a simple removal of the upper 1/3rd of the image since that is where most of the background is.


**Pre Processing**
To get started, I first needed to find a good mask to identify where the road was and cut out the background parts of the image. Most of the roads in the dashcam footage took up a trapezoidal area within the frame and I wanted to mask just this trapezoid. So I used the first 8 frames of a given dashcam vidoe as a way of priming my algorithm that would identify the images in real time.  


### *Color Thresholding*
A more advanced way of masking is through use of color thresholds. Given a range of colors, the 'cv2.inRange()' function can be used to essentially mask the image by color. With this in mind, I used a color threshold to capture a range of yellows and whites that were commonly found in dash cam footage.
This would cut out large portions of the picture, leaving behind the lanes along with a few remaining artifacts.


### *Sobel Edge Detection*
There are many ways of detecting edges, borders, or lines of an image in computer vision. One such was is sobel edge detection, which can be good for identifying linear patterns either in the vertical direction (cv2.Sobel(dx=1, dy=0)) or the horizontal directon (cv2.Sobel(dx=0, dy=1)).Since most lanes are usually oriented in steep curves in a dashcam, I used sobel in the x direction to identify the prominent vertically oriented edges within the image.

### *arbitrary masking for a region of interest*
This is one of the weaker portions of my project. I hard coded a triangular polygon to mask a portion of the image. It was intended for one specific video, but ended up working for a lot of the data I tried. So I just kept it
```python
# TODO: change this to be more of a trapezoid within the center of the image, 
```

*Hough Lines*
Basically, you run this and it tries to find lines in an image. I would try to identify some lines in the frame, then run it through a filter so that the lines nearest to the center of the image would remain.
I then calculated the average slope intercept, which also partitioned lines depending on whether their slope was positive (probably a lane on the left) OR negative (probably the lane on the right). Then averaged out all positive lines, and all the negative lines, to get a pair of averaged lines hopefully representing the lanes.
Since we used 8 framed, we then got the average of those 8 best fitting left and right line pairs. I used these averages as a basis for the lines could be, then formed a trapezoid around that region, large enough for where the road might be. then I made smaller masks from finding the center of the trapezoid and making a left portion and a right portion. this is easier to identify the left vs the right lanes rather than looking at both in one go. When you do that, it may confuse parts of one lane for the other.  

*Intended next step*
- test out whether there really are extra screen artifacts inside the pre-processing mask, cutting it down either with contour detection (changing the trapezoid mask) or changing the color thresholding ranges for yellow and white (changing the color thresholding mask)
- this is what I INTENDED to do, but I ended up doing something else in the process.

*Actual next step*
I basically used the mask found from the hough lines, most of the time, that was good enough to identify the lanes, there still are some artifacts that screw with it, but its okay. I ran another algorithm on this image, I tried to find the largest dotted line on the left and right side of the trapezoid, and the largest solid line on the right side of the trapezoid. I then recorded a skinnier line that would strike through that largest found line. 
Sometimes, it would confuse artifacts on one side of the screen as the largest solid line, or small pixel artifacts as part of the largest dotted line, but the method I found seemed to be a generally good enough.
What I found was at times, these largest dotted/solid lines would be exactly the same, and it would only be the same (that is to say, the averaged line that would strike through those largest found lines), when it was actually on a lane, regardless of whether the lane was solid or dotted. 
After finding that out, I would record whenever the largest solid and dotted lines were the same line. This ended up being a good enough indicator for whether the line was a lane or not.

So I used this information as a way of guiding the slope of what the lane should look like. This would help for real time processing, when filtering out lines from artifacts.

*Real time processing*
Some things are the same, but less complicated as I used the trapezoid mask along with the slopes I found, as 'heuristic' guides for identifying the lanes. The order of the real time algorithm was shorter:
- Color threshold the image
- apply a simple binary threshold on that image (not edge detection, but setting a color threshold, making any color above the threshold value white, any color below it black)
- apply the trapezoid mask found from pre-processing
- find the hough lines, then filter out the ones that were way off from the average slope
    - some times the hough lines came back with nothing, as my parameters were a little more strict here than before (i.e. I made the requirement to look for longer lines, before I was kind of loose on how long the line I was looking for needed to be)
    - I'd change the average slope along with the average slope of every 80th frame (just to shake it up a lil bit)
    - when the Hough Line identifier came back with zilch, I'd just display the average line (not the best, would've been better to do a variable line length, but ehh...)
- display those lines found onto the original image.

## Challenges faced along the way
*Line and HSV color handling in openCV*
One tricky thing was the way values such as HSV colors and lines are formatted in OpenCV. there is no dedicated color object or line object for these values, but instead, each value is represented in a numpy array. so a line would be an array of length four [x1, y1, x2,y2]. RGB values were simple enough [0-255,0-255,0-255], but it got trickier when handling HSV values. 
The HSV color wheel is typically represented with a cone. The Hue of a color represented in degrees(0-360), while the Saturation and Value of the color are represented as percentages (0%-100%). In openCV, the HSV values were represented by dividing the hue in half, and using 255 as 100% so HSV values would look like: [0-180, 0-255, 0-255].


*Knowing which Algorithm would work*
I mentioned before that I used a rectangle to cut out the upper 1/3rd rectangle as well as a triangle near the center the image as a simple way of ridding the background. The more advanced alternative would have been using a combination of thresholds along with a distance transform to segment the background from the foreground. 
Distance transforms are useful as you can use this to determine the distance between pixel clusters. One thing I experimented with was cutting out white pixel clusters depending on some distance they had to black pixels, with the intention of cutting out larger bulkier artifacts. but this had limited use, would at times cut out parts of the lanes, and generally leave behind the edges of the bulky object (meaning more ways to confuse something non linear for a line!)
I may try and implement this later such that the lowest point of the background could be used to find the horizon in the image, then cut out a trapezoid relative to that horizon.It would be cleaner 
Another thing I tried was the watershed algorithm. This was a pretty interesting way of segmenting the image. You can use it to identify contours of 
I was originally thinking of using this to detect the road vs non-road foreground and background. Didn't quite work, as it made a lot of partitions. 
Again, may work, but I didn't experiment enough.


## What I might try in the future:
- Succesfully implementing the watershed algorithm (to remove non-lane object image artifacts)
- Succesfully implementing the distance transform (in tandem with watershed)
- Application of perspective transform (this was used by a lot of other lane detection projects I saw)
- Maybe trying camera calibration and limiting distortion of the camera image. (this was also used by a lot of other lane detection projects I saw)
