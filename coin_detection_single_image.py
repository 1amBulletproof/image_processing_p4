#!/usr/bin/python

#Detect coins

#Example found here: http://blog.christianperone.com/2014/06/simple-and-effective-coin-segmentation-using-python-and-opencv/:

import numpy as np
import glob, cv2, sys

def main():

    image = sys.argv[1]

    input_img = cv2.imread(image)
    display_img("input img", input_img)

    #Ensure the image is grayscale
    gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    display_img("gray img", gray_img)

    #Blur the image to reduce the noise
    #blurred_gray_img = cv2.GaussianBlur(gray_img, (21, 21), 0)
    blurred_gray_img = cv2.blur(gray_img, (11, 11))
    display_img("blurred gray img", blurred_gray_img)

    #Adaptive threshold to control for different lighting but this will produce noise!
    threshold_mask = cv2.adaptiveThreshold(blurred_gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 61, 1)
    #threshold_mask = cv2.Canny(blurred_gray_img, 50, 100)
    display_img("Threshold mask", threshold_mask)

    kernel = np.ones((3, 3), np.uint8)

    #Remove the noise
    low_noise_mask = cv2.erode(threshold_mask, kernel, iterations=2)
    display_img("noise-removed mask", low_noise_mask)

    kernel2 = np.ones((3, 3), np.uint8)

    #Close (morphology): forms cohesive, closed shapes in the mask
    closed_mask = cv2.morphologyEx(low_noise_mask, cv2.MORPH_CLOSE,
                kernel2, iterations=11)
    display_img("Closed mask", closed_mask)

    #Disconnect anyone who got connected
    eroded_mask = cv2.erode(closed_mask, kernel2, iterations=7)
    display_img("eroded - mask", eroded_mask)

    #regain former shape  anyone who got connected
    enlarged_mask = cv2.dilate(eroded_mask, kernel2, iterations=7)
    display_img("enlarged - mask", enlarged_mask)

    #findContours - get the extreme boundaries
    closed_mask_clone = enlarged_mask.copy()
    contour_img, contours, hierarchy = cv2.findContours(closed_mask_clone, cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)

    print("There are " + str(len(contours)) + " contours in this image")

    '''
    print("contour_img = " + str(contour_img))
    print("contours = " + str(contours))
    print("hierarychy = " + str(hierarchy))
    '''
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 5000:
            print("Area = " + str(area))
            coin_type = classify_coin(area)
            print("This is a " + coin_type)
            ellipse = cv2.fitEllipse(contour)
            cv2.ellipse(input_img, ellipse, (0,255,0), 2)
            center_coord = get_center(contour)
            cv2.putText(input_img, coin_type, center_coord, cv2.FONT_HERSHEY_COMPLEX,2, 255)

    display_img("final img", input_img)

def display_img(description, img):
    cv2.imshow(description, img)
    cv2.waitKey(0)

def get_center(contour):
    pt1 = contour[0][0]
    print pt1
    number_of_pts = len(contour)
    pt2 = contour[number_of_pts/2][0]
    center = (pt1[0] - pt2[0], pt1[1] - pt2[1])
    return center

def classify_coin(area):
    if area < 5000:
        return "NOT A COIN"
    elif 5000 < area < 9500:
        return "Dime"
    elif 9500 < area < 11000:
        return "Penny"
    elif 11000 < area < 15000:
        return "Nickel"
    else:
        return "Quarter"

if __name__ == "__main__":
    main()
