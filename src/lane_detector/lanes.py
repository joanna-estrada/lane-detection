import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines:
        for x1,y1,x2,y2 in lines:
            cv.line(img=line_image, pt1=(x1,y1),pt2=(x2,y2), color=(255,0,0), thickness=10)
    return line_image 

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1,y1,x2,y2])

def canny(image):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)     # convert to gray scale
    blur = cv.GaussianBlur(gray, (5,5), 0)
    # application of 5x5 kernel window on our image. The size of the 
    # kernel depends on the situation but a 5x5 window is ideal for most cases
    canny = cv.Canny(blur, threshold1=50, threshold2=150 )     # to obtain edges of the image (1:3 ratio)
    return canny

def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([[(200, height), (1100, height), (550, 250)]])
    mask = np.zeros_like(image)
    cv.fillPoly(mask,polygons,255)
    masked_image = cv.bitwise_and(image, mask)
    return masked_image

def main():
    # image = cv.imread('lane.jpg')
    image = cv.imread('lane.jpg')
    # cv.imshow('Original', image)
    lane_image = np.copy(image)                                     # creating copy of the image
    cny = canny(lane_image)
    cv.imshow('Result of Canny', cny)                               # output to gray-scale img
    cropped_image = region_of_interest(cny)
    cv.imshow('Result of ROI', cropped_image)                       # changing it to show Region of Interest

    lines = cv.HoughLinesP(cropped_image, rho=2, theta=np.pi/180,
                           threshold=100, lines= np.array([]),
                           minLineLength=40, maxLineGap=5)

    cv.waitKey(0)                                          # display images for some time
    # plt.imshow(cny)
    # plt.show()


if __name__ == "__main__":
    main()