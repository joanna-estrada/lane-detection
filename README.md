# Lane Detection with Python and OpenCV
Alex Alvarez and Joanna Estrada

## Project Overview
This project uses Python and OpenCV for real-time lane detection in dash cam footage. The main objective is to explore AI and computer vision algorithms to identify straight lanes, providing immediate visual feedback.

# Project Details
## Lane Detection Algorithm
Preprocessing Steps
* Masking: Utilize masking to identify road areas in dash cam frames.
* Color Thresholding: Implement color thresholding to capture yellow and white lane markings.
* Sobel Edge Detection: Employ Sobel edge detection to identify vertical edges, common in lane markings.
* Arbitrary Masking for Region of Interest: Improve the hardcoded triangular polygon for more flexibility.

Hough Lines
* Use Hough lines to identify potential lane lines.
* Filter lines based on their proximity to the center of the image.
* Average positive and negative slope lines to obtain representative left and right lane lines.

Real-Time Processing
* Color Thresholding: Apply color thresholding to the image.
* Binary Thresholding: Use binary thresholding to simplify lane identification.
* Trapezoid Mask Application: Apply the trapezoid mask from preprocessing.
* Hough Lines Filtering: Identify and filter Hough lines based on average slope.
* Display Result: Display the identified lane lines on the original image.

# Challenges and Future Considerations
## Challenges Faced
* Handling HSV colors and lines in OpenCV.
* Choosing the right algorithm for background removal.

## Future Enhancements
* Watershed Algorithm: Implement the watershed algorithm for non-lane object artifact removal.
* Distance Transform: Experiment with distance transform for improved image segmentation.
* Perspective Transform: Apply perspective transform for better lane detection.
* Camera Calibration: Consider camera calibration to reduce distortion in the camera image.