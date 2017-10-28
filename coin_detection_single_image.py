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
    blurred_gray_img = cv2.GaussianBlur(gray_img, (15, 15), 0)
    display_img("blurred gray img", blurred_gray_img)

    #Adaptive threshold to control for different lighting but this will produce noise!
    #threshold_mask = cv2.adaptiveThreshold(blurred_gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #            cv2.THRESH_BINARY_INV, 11, 1)
    #use a small block to determine the mean to make noise smaller
    threshold_mask = cv2.adaptiveThreshold(blurred_gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 3, 1)
    display_img("Threshold mask", threshold_mask)

    #Close (morphology): reduces noise AND forms cohesive, closed shapes in the mask
    kernel = np.ones((3, 3), np.uint8)
    #closed_mask = cv2.morphologyEx(threshold_mask, cv2.MORPH_CLOSE,
                #kernel, iterations=4)
    closed_mask = cv2.morphologyEx(threshold_mask, cv2.MORPH_CLOSE,
                kernel, iterations=10)

    #Remove the noise...could try a filter of some kind instead of morphology here
    closed_mask = cv2.erode(closed_mask, kernel, iterations=10)
    display_img("Closed mask", closed_mask)

    #findContours - get the extreme boundaries (so you dont' worry about holes in the coins & get just the simple approximation of the contour
    closed_mask_clone = closed_mask.copy()
    contour_img, contours, hierarchy = cv2.findContours(closed_mask_clone, cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)

    print("There are " + str(len(contours)) + " coins in this image")

    '''
    print("contour_img = " + str(contour_img))
    print("contours = " + str(contours))
    print("hierarychy = " + str(hierarchy))
    '''
    for contour in contours:
        area = cv2.contourArea(contour)
        print("Area = " + str(area))
        ellipse = cv2.fitEllipse(contour)
        cv2.ellipse(input_img, ellipse, (0,255,0), 2)

    display_img("final img", input_img)

    '''
        for cnt in contours:
            area = cv2.contourArea(cnt)
        if area < 2000 or area > 4000:
            continue
    '''
def display_img(description, img):
    cv2.imshow(description, img)
    cv2.waitKey(2000)

if __name__ == "__main__":
    main()
