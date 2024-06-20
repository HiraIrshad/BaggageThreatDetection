from sklearn.svm import OneClassSVM
import pandas as pd
import os
from skimage.feature import hog
import cv2 as cv
from skimage import exposure
from sklearn.svm import LinearSVC
import numpy as np
import matplotlib.pyplot as plt


def remove_small_objects2(image, min_size):
    image[:,:] = 255 - image[:,:]
    # Find contours in the binary image
    contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Create an empty mask
    mask = np.zeros_like(image)

    # Loop through the contours and filter based on size
    for contour in contours:
        area = cv.contourArea(contour)
        if area >= min_size:
            cv.drawContours(mask, [contour], -1, 255, thickness=cv.FILLED)

    # Apply the mask to the original image
    result = cv.bitwise_and(image, image, mask=mask)
    result[:, :] = 255 - result[:, :]
    return result


def remove_small_objects(image, min_size):
    # Apply a binary threshold to get a binary image
    _, binary = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # Find contours in the binary image
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Create an empty mask
    mask = np.zeros_like(image)

    # Loop through the contours and filter based on size
    for contour in contours:
        area = cv.contourArea(contour)
        if area >= min_size:
            cv.drawContours(mask, [contour], -1, 255, thickness=cv.FILLED)

    # Apply the mask to the original image
    result = cv.bitwise_and(image, image, mask=mask)

    return result, mask


def fill_region(image):
    # Apply edge detection
    edges = cv.Canny(image, 30, 100)

    # Find contours
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Fill regions inside all contours
    filled_image = np.copy(image)
    for contour in contours:
        cv.fillPoly(filled_image, [contour], (255, 255, 255))  # Fill with white color

    return filled_image, contours



def remove_non_gun_objects(image):
    contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Calculate bounding box of the contour
        x, y, w, h = cv.boundingRect(contour)
        if w > 300 or h > 300:
            cv.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), thickness=cv.FILLED)
        if w < 35 or h < 35:
            cv.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), thickness=cv.FILLED)

    return image


def find_bounding_boxes(mask):
    """
    Find bounding boxes for objects in the mask.
    """
    min_width, min_height = 50, 50
    max_width, max_height = 200, 200
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        if min_width <= w <= max_width and min_height <= h <= max_height:
            bounding_boxes.append((x, y, w, h))
    return bounding_boxes


def extract_windows(image, bounding_boxes):
    windows = []
    for (x, y, w, h) in bounding_boxes:
        print(f"Bounding Box: (x={x}, y={y}, w={w}, h={h}), Size: {w}x{h}")

        window = image[y:y+h, x:x+w]

        # Resize the window to the specified size
        resized_window = cv.resize(window, (50,50))
        # window_mask = mask[y:y+h, x:x+w]
        windows.append(resized_window)
    return windows


def remove_background(image, mask, a):
    # cv.imshow("mask", mask*255)
    bounding_boxes = find_bounding_boxes(mask)
    thresh = np.zeros_like(image)
    b=0
    for (x, y, w, h) in bounding_boxes:
        thresh[y:y+h, x:x+w] = image[y:y+h, x:x+w]
        gun_contour = cv.resize(image[y:y+h, x:x+w], (100,100))
        # Set all non-zero pixels to 255
        gun_contour[gun_contour >= 140] = 255
        gun_contour[gun_contour < 140] = 0
        kernel = np.ones((3, 3), np.uint8)
        image = cv.erode(image, kernel, iterations=1)
        image = cv.dilate(image, kernel, iterations=1)
        # # image = cv.dilate(image, kernel, iterations=1)
        # # image = cv.erode(image, kernel, iterations=1)
        image = remove_small_objects2(image, 1000)
        # cv.imshow("gun", gun_contour)
        # cv.waitKey()
        cv.imwrite(f"test/{a}{b}.png", gun_contour)
        # X_train.append(gun_contour)
        # y_train.append(1)
        b += 1

    return thresh

def remove_background2(image, mask, a):
    image_r = remove_background_safe(image)
    contours, _ = cv.findContours(image_r, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    bounding_boxes_r = []
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        bounding_boxes_r.append((x, y, w, h))

    bounding_boxes = find_bounding_boxes(mask)
    b=0
    for (x, y, w, h) in bounding_boxes_r:
        for (x2, y2, w2, h2) in bounding_boxes:
            if (x2 <= x <= x2+w2) and (y2 <= y <= y2+h2):
                gun_contour = cv.resize(image_r[y:y+h, x:x+w], (100,100))
                # Set all non-zero pixels to 255
                gun_contour[gun_contour >= 40] = 255
                gun_contour[gun_contour < 40] = 0
                gun_contour[:,:] = 255 - gun_contour[:,:]
                # cv.imshow("gun", gun_contour)
                # cv.waitKey()
                cv.imwrite(f"test/{a}{b}.png", gun_contour)
        # X_train.append(gun_contour)
        # y_train.append(1)
                b += 1
    # Apply GaussianBlur to smooth the image
    # blurred = cv.GaussianBlur(thresh, (5, 5), 0)
    # cv.imshow("thresh", thresh)
    # cv.waitKey(0)

def remove_background_safe(image):
    # Apply GaussianBlur to smooth the image
    blurred = cv.GaussianBlur(image, (5, 5), 0)

    # Background subtraction using thresholding
    _, thresh = cv.threshold(blurred, 120, 255, cv.THRESH_BINARY_INV)

    kernel = np.ones((9, 9), np.uint8)
    thresh = cv.dilate(thresh, kernel, iterations=1)
    thresh = cv.erode(thresh, kernel, iterations=1)
    thresh = cv.dilate(thresh, kernel, iterations=1)
    thresh = cv.erode(thresh, kernel, iterations=1)
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv.erode(thresh, kernel, iterations=1)
    thresh = cv.dilate(thresh, kernel, iterations=1)

    min_size = 1500
    thresh, mask = remove_small_objects(thresh, min_size)
    thresh, boundary_points = fill_region(thresh)
    result = cv.bitwise_and(image, image, mask=thresh)
    result1 = remove_non_gun_objects(result)
    # cv.imshow("Safe", result1)
    # cv.waitKey(0)
    return result1


def find_non_gun_contours(image, a):
    contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        bounding_boxes.append((x, y, w, h))
    b = 0
    for (x, y, w, h) in bounding_boxes:
        gun_contour = cv.resize(image[y:y+h, x:x+w], (100,100))
        gun_contour[gun_contour >= 40] = 255
        gun_contour[gun_contour < 40] = 0
        gun_contour[:,:] = 255 - gun_contour[:,:]
        # cv.imshow("non_gun", gun_contour)
        # cv.waitKey()
        cv.imwrite(f"test2/{a}{b}.png", gun_contour)
        b += 1
        # X_train.append(gun_contour)
        # y_train.append(0)



safe_dir_path = "students_data/train/safe"
safe_files = [f for f in os.listdir(safe_dir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif'))]
# print(len(gun_files))
gun_dir_path = "students_data/train/gun"
gun_files = [f for f in os.listdir(gun_dir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif'))]
gun_ann_dir_path = "students_data/train/annotations/gun"
gun_ann_files = [f for f in os.listdir(gun_ann_dir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif'))]

X_train = []
y_train = []


for a in range(len(gun_files)):
    print(f"Iteration: {a}, {gun_files[a]}")
    bck_image_gun = cv.imread(f"{gun_dir_path}/{gun_files[a]}")
    ann_gun = cv.imread(f"{gun_ann_dir_path}/{gun_ann_files[a]}", 0)
    # print(bck_image_gun.shape)
    blue_channel, green_channel, red_channel = cv.split(bck_image_gun)
    bck_image_gun = green_channel
    # Remove background
    remove_background(bck_image_gun, ann_gun, gun_files[a])

for a in range(0, len(safe_files)):
    print(f"Iteration: {a}, {safe_files[a]}")
    # print(f"Iteration: {a}, {gun_ann_files[a]}")
    bck_image_gun = cv.imread(f"{safe_dir_path}/{safe_files[a]}")
    print(bck_image_gun.shape)
    blue_channel, green_channel, red_channel = cv.split(bck_image_gun)
    bck_image_gun = green_channel
    # Remove background
    image_gun = remove_background_safe(bck_image_gun)
    find_non_gun_contours(image_gun, safe_files[a])

