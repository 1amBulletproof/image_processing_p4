#!/usr/bin/python

#Detect coins

#Example found here: http://blog.christianperone.com/2014/06/simple-and-effective-coin-segmentation-using-python-and-opencv/:

import numpy as np
import cv2

def main():
    input_img = cv2.imread('../CoinImages/image_00.jpg')
    cv2.imshow("input img", input_img)
    cv2.waitKey(2000)

    #Ensure the image is grayscale
    gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray img", gray_img)
    cv2.waitKey(1000)

    #Blur the image to reduce the noise
    blurred_gray_img = cv2.GaussianBlur(gray_img, (15, 15), 0)
    cv2.imshow("blurred gray img", blurred_gray_img)
    cv2.waitKey(1000)

    #Adaptive threshold to control for different lighting but this will produce noise!
    #threshold_mask = cv2.adaptiveThreshold(blurred_gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #            cv2.THRESH_BINARY_INV, 11, 1)
    #use a small block to determine the mean to make noise smaller
    threshold_mask = cv2.adaptiveThreshold(blurred_gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 3, 1)
    cv2.imshow("Threshold mask", threshold_mask)
    cv2.waitKey(3000)

    #Close (morphology): reduces noise AND forms cohesive, closed shapes in the mask
    kernel = np.ones((3, 3), np.uint8)
    #closed_mask = cv2.morphologyEx(threshold_mask, cv2.MORPH_CLOSE,
                #kernel, iterations=4)
    closed_mask = cv2.morphologyEx(threshold_mask, cv2.MORPH_CLOSE,
                kernel, iterations=10)

    #Remove the noise...could try a filter of some kind instead of morphology here
    closed_mask = cv2.erode(closed_mask, kernel, iterations=10)
    cv2.imshow("Closed mask", closed_mask)
    cv2.waitKey(3000)

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
        ellipse = cv2.fitEllipse(contour)
        cv2.ellipse(input_img, ellipse, (0,255,0), 2)
        print(str(ellipse))


    cv2.imshow("final img", input_img)
    cv2.waitKey(2000)

    '''
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 2000 or area > 4000:
                continue
    '''

if __name__ == "__main__":
    main()
