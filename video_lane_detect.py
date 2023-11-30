# TODO:  I need to see how it's drawing over the image that has been processed by the canny edge detection AND the bitwise and (region_of_interest)
import cv2
import numpy as np

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
    img_width = image.shape[1]
    boundary_x = img_width//2

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        # if slope < 0 and x1 < boundary_x and x2 < boundary_x:
        #     left_fit.append((slope, intercept))
        # elif slope > 0 and x1 > boundary_x and x2 > boundary_x:
        #     right_fit.append((slope, intercept))
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


def region_of_interest(image, frame_count):
    # image.shape[0] returns image height, image.shape[1] returns image width
    height = image.shape[0]

    # TODO: One thing that must be changed are the values in this numpy array, right now they are fixed, but they should be able to acclimate to different videos
    polygons = np.array([[(200, height), (1300, height), (550, 250)]])
    # polygons = np.array([[(200, height), (1300, height), (00, 300)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


# def main():
#     # #NOTE: This is for individual images, 
#     image = cv2.imread('lane.jpg')
#     lane_image = np.copy(image)

#     canny_image = canny(lane_image)
#     cropped_image = region_of_interest(canny_image, 0)
#     lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
#     averaged_lines = average_slope_intercept(lane_image, lines)
#     rgb_cropped = cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2BGR)

#     # line_image = display_lines(lane_image, averaged_lines)
#     # combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)

#     line_image = display_lines(rgb_cropped, averaged_lines)
#     combo_image = cv2.addWeighted(rgb_cropped, 0.8, line_image, 1, 1)

#     cv2.imshow('result',combo_image)
#     cv2.waitKey(0)


def main():
    # NOTE: This is for videos 
    cap = cv2.VideoCapture('project_video.mp4')
    i = 0
    while(cap.isOpened()):
        # get 1 frame from the video
        _, frame = cap.read()
        width = frame.shape[1]
        
        # generate a canny image from the frame
        canny_image = canny(frame)
        
        
        # cropped_image = region_of_interest(canny_image, i)                                          # focus region to be the lower triangular portion of the image
        lines = cv2.HoughLinesP(image=canny_image,rho=2, theta=np.pi/180,
                                threshold=80, lines=np.array([]), 
                                minLineLength=40, maxLineGap=5)
        averaged_lines = average_slope_intercept(frame, lines)
        rgb_cropped = cv2.cvtColor(canny_image, cv2.COLOR_GRAY2BGR)

        # # left_fit, right_fit = fit_lane_curves(frame, lines)
        # left_fit, right_fit = fit_lane_curves(rgb_cropped, lines)
        
        # # Use the polynomial functions to generate points along the left and right lane curves
        # left_lane_points = [(x, 
        # int(left_fit[0] * x**2 + left_fit[1] * x + left_fit[2])) for x in range(0, width)]
        # right_lane_points = [(x, 
        # int(right_fit[0] * x**2 + right_fit[1] * x + right_fit[2])) for x in range(0, width)]

        # # Draw the left and right lane curves on the image
        # cv2.polylines(rgb_cropped, [np.array(left_lane_points)], 
        # isClosed=False, color=(255, 0, 0), thickness=10)
        # cv2.polylines(rgb_cropped, [np.array(right_lane_points)], 
        # isClosed=False, color=(0, 0, 255), thickness=10)

        # line_image = display_lines(frame, averaged_lines)

        line_image = display_lines(rgb_cropped, averaged_lines)
        combo_image = cv2.addWeighted(rgb_cropped, 0.8, line_image, 1, 1)

        if i%20==0:
            cv2.imwrite(f'masked_frames/frame_{i}.jpg', combo_image)
        
        cv2.imshow('result',combo_image)
        i+=1
        if cv2.waitKey(1) == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()