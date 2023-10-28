# Lane-Detection
Alex Alvarez and Joanna Estrada
### Project Overview:
This project leverages the power of Python and OpenCV to detect lanes in dash cam footage. Aimed at enhancing driving safety and road awareness, the software identifies both straight and curved lanes, providing real-time visual feedback to users.


### Feature Specification:
- *Dash Cam Input*: Accepts dash cam footage as input to analyze road lanes.
 
- *Straight Lane Detection*:
  - Uses a region of interest, typically the lower triangular or trapezoidal portion of the screen, to focus on the road.
  - Implements edge detection techniques to identify straight lane markings.
  - Highlights the detected lanes for the user with overlays on the video feed.


- *Curved Lane Detection*:
  - Goal is to go beyond straight lane markings to detect and highlight curved lanes.
  - Will attempt at employing polynomial fitting to approximate the curve of the lanes in the view.


- *Real-time Feedback*:
  - Provides visual feedback to users by highlighting detected lanes on video input.


### Technical Specification:
Built using Python 3.11 and OpenCV 4.2, this application utilizes traditional computer vision techniques to assess lane detection in varying conditions.
