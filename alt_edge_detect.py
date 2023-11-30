import cv2
import numpy as np

def color_thresholding(image):
    # initialize a blank image
    blank = np.zeros(image.shape[:2], dtype='uint8')
    # save the width and height of the image
    width = image.shape[0]
    height = image.shape[1]
    # create a rectangular mask that will display the approx. lower 2/3rd of the image 
    # this is done to remove the sky and buildings. 
    rect_mask = cv2.rectangle(blank.copy(), (0, image.shape[0]//3+20), (image.shape[1], image.shape[0]), 255, -1)
    r_masked_img = cv2.bitwise_and(image, image, mask = rect_mask)
    # Convert the image to the HSV color space
    # hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = cv2.cvtColor(r_masked_img, cv2.COLOR_BGR2HSV)
    # g_hsv = cv2.cvtColor(r_masked_img, cv2.COLOR_RGB2HSV)
    # hsv = cv2.GaussianBlur(g_hsv, (5,5),0)

    # Define the lower and upper thresholds for yellow color in HSV (adjusted for a broader range)
    lower_yellow = np.array([15, 50, 50])
    # upper_yellow = np.array([35, 255, 255])
    upper_yellow = np.array([40, 240, 240])

    # Define the lower and upper thresholds for white color in HSV (adjusted for a broader range)
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([255, 30, 255])

    # Create masks for yellow and white colors
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    # Combine the masks to get the regions of yellow and white lines
    combined_mask = cv2.bitwise_or(mask_yellow, mask_white)

    # Apply the mask to the original image to isolate yellow and white lines
    result = cv2.bitwise_and(image, image, mask=combined_mask)

    return result

def to_canny(img):
    # image = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    image = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    # Assuming 'image' is a color image in BGR format
    # sobelx_b = cv2.Sobel(image[:, :, 0], cv2.CV_64F, 1, 0, ksize=5)
    # sobelx_g = cv2.Sobel(image[:, :, 1], cv2.CV_64F, 1, 0, ksize=5)
    # sobelx_r = cv2.Sobel(image[:, :, 2], cv2.CV_64F, 1, 0, ksize=5)
    sobelx_b = cv2.Sobel(image[:, :, 0], cv2.CV_8UC1, 1, 0, ksize=5)
    sobelx_g = cv2.Sobel(image[:, :, 1], cv2.CV_8UC1, 1, 0, ksize=5)
    sobelx_r = cv2.Sobel(image[:, :, 2], cv2.CV_8UC1, 1, 0, ksize=5)
    rg = cv2.bitwise_or(sobelx_r,sobelx_g)
    b = cv2.bitwise_or(rg,sobelx_b)
    return b
    # gray = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
    # sobelx = sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    # return sobelx
    # return 
