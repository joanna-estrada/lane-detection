import cv2
import numpy as np
from alt_edge_detect import *

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray,(5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

# def display_lines(image, lines):
#     line_image = np.zeros_like(image)
#     if lines is not None:
#         for line in lines:
#             print(line)
#             x1, y1, x2, y2 = line
#             x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#             cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
#     return line_image
 
def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            print(line)
            x1, y1, x2, y2 = line  # Extract the coordinates from the array
            # Filter out lines with extreme coordinates (you can adjust the threshold as needed)
            if abs(x1 - x2) > 10 and abs(y1 - y2) > 10:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to integers
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image


def region_of_interest(image):
    height = image.shape[0]
    # TODO: One thing that must be changed are the values in this numpy array, right now they are fixed, but they should be able to acclimate to different videos
    polygons = np.array([
    [(100, height), (1300, height), (550, 250)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def do_lines_nearest_to_center(image, lines):
    # Calculate the center point of the region of interest (ROI)
    roi_width = cropped_image.shape[1]
    roi_height = cropped_image.shape[0]
    center_x = roi_width // 2
    center_y = roi_height // 2

    # List to store line segments closest to the center
    closest_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]  # Extract line coordinates
        # Calculate the midpoint of the line segment
        midpoint_x = (x1 + x2) // 2
        midpoint_y = (y1 + y2) // 2
        # Calculate the Euclidean distance from the midpoint to the center
        distance_to_center = np.sqrt((midpoint_x - center_x)**2 + (midpoint_y - center_y)**2)

        # Set a threshold distance for considering lines
        threshold_distance = 50  # Adjust as needed

        # Check if the line is close enough to the center
        if distance_to_center < threshold_distance:
            closest_lines.append(line)
        
    return np.array(closest_lines)


#NOTE: This is for individual images, 
# image = cv2.imread('lane.jpg')
# lane_image = np.copy(image)

# canny_image = canny(lane_image)
# cropped_image = region_of_interest(canny_image)
# lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
# averaged_lines = average_slope_intercept(lane_image, lines)
# line_image = display_lines(lane_image, averaged_lines)

# combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)

# cv2.imshow('result',combo_image)
# cv2.waitKey(0)

#NOTE: This is for videos 
cap = cv2.VideoCapture('project_video.mp4')
while(cap.isOpened()):
    # get 1 frame from the video
    _, frame = cap.read()                                                       # frame is RGB
    # generate a canny image from the frame
    # canny_image = canny(frame)
    thresh_img = color_thresholding(frame)                                      # get RGB, returns HSV
    canny_image = to_canny(thresh_img)                                          #gets HSV, returns RGB
    # focus region to be the lower triangular portion of the image
    cropped_image = region_of_interest(canny_image)
    # cv2.imshow("haha",cropped_image)
    # cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2GRAY)
    lines = cv2.HoughLinesP(image=cropped_image,rho=21, theta=np.pi/180,
                            threshold=20, lines=np.array([]), 
                            minLineLength=10, maxLineGap=1)
    closest_to_middle = do_lines_nearest_to_center(cropped_image, lines)

    averaged_lines = average_slope_intercept(cropped_image, closest_to_middle)
    line_image = display_lines(frame, averaged_lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow('result',combo_image)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()