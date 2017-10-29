#!/usr/bin/python

#Detect coins

#Example found here: http://blog.christianperone.com/2014/06/simple-and-effective-coin-segmentation-using-python-and-opencv/:

from matplotlib import pyplot as plt
import numpy as np
import glob, cv2

def main():
    image_paths = glob.glob("./CoinImages/*.jpg")
    #print images

    num_rows = 1
    num_cols = 3
    

    for image_path in image_paths:
        plot_pos = 1
        plt.figure(figsize=(13, 6))

        input_img = cv2.imread(image_path)
        #display_img("input img", input_img)
        plt.subplot(num_rows, num_cols, plot_pos)
        plt.imshow(input_img, aspect='auto')
        plt.title('Original'), plt.axis('off')
        plot_pos = plot_pos + 1
    
        #Ensure the image is grayscale
        gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        #display_img("gray img", gray_img)
    
        #Blur the image to reduce the noise
        #gausian, vs. mean, vs. blur (average)
        blurred_gray_img = cv2.blur(gray_img, (11, 11))
        #display_img("blurred gray img", blurred_gray_img)
    
        #Adaptive threshold to control for different lighting but this will produce noise!
        # gausian vs. mean!?
        threshold_mask = cv2.adaptiveThreshold(blurred_gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                    cv2.THRESH_BINARY_INV, 7, 1)
    
        #threshold_mask = cv2.Canny(blurred_gray_img, 50, 100)
        #display_img("Threshold mask", threshold_mask)
    
        kernel = np.ones((3, 3), np.uint8)
    
        #Remove the noise
        low_noise_mask = cv2.erode(threshold_mask, kernel, iterations=1)
        #display_img("noise-removed mask", low_noise_mask)
    
        kernel2 = np.ones((3, 3), np.uint8)
    
        #Close (morphology): forms cohesive, closed shapes in the mask
        closed_mask = cv2.morphologyEx(low_noise_mask, cv2.MORPH_CLOSE,
                    kernel2, iterations=17)
        #display_img("Closed mask", closed_mask)
    
        #Disconnect anyone who got connected
        eroded_mask = cv2.erode(closed_mask, kernel2, iterations=33)
        #display_img("eroded - mask", eroded_mask)
    
        #regain former shape  anyone who got connected
        enlarged_mask = cv2.dilate(eroded_mask, kernel2, iterations=33)
        #display_img("enlarged - mask", enlarged_mask)
        plt.subplot(num_rows, num_cols, plot_pos)
        plt.imshow(enlarged_mask, cmap='gray', aspect='auto')
        plt.title('Mask'), plt.axis('off')
        plot_pos = plot_pos + 1
    
        #findContours - get the extreme boundaries
        closed_mask_clone = enlarged_mask.copy()
        contour_img, contours, hierarchy = cv2.findContours(closed_mask_clone, cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE)
    
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 5000:
                coin_type = classify_coin(area)
                #coin_details = coin_type + " " + str(area)
                #print(coin_details)
                ellipse = cv2.fitEllipse(contour)
                cv2.ellipse(input_img, ellipse, (0,255,0), 2)
                cv2.putText(input_img, coin_type, (contour[0][0][0], contour[0][0][1]), cv2.FONT_HERSHEY_COMPLEX,2, (255,0, 0), 3)
    
        #display_img("final img", input_img)
        plt.subplot(num_rows, num_cols, plot_pos)
        plt.imshow(input_img, aspect='auto')
        plt.title('Final'), plt.axis('off')
        plot_pos = plot_pos + 1

        plt.tight_layout()
        plt.show()
    
    
def display_img(description, img):
    cv2.imshow(description, img)
    cv2.waitKey(0)
    
    
def classify_coin(area):
    if area < 5000:
        return "???"
    elif 5000 <= area < 8000:
        return "Dime"
    elif 8000 <= area < 10500:
        return "Penny"
    elif 10500 <= area < 15000:
        return "Nickel"
    elif 15000 <= area < 20000:
        return "Quarter"
    else:
        return "???"
    
    
if __name__ == "__main__":
    main()
