from sklearn.svm import OneClassSVM
import pandas as pd
import os
from skimage.feature import hog
import cv2 as cv
from skimage import exposure
from sklearn.svm import LinearSVC
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score

def IOU(predicted, mask):
    # Calculate the intersection
    intersection = cv.bitwise_and(predicted, mask)
    # Calculate the union
    union = cv.bitwise_or(predicted, mask)
    # Compute the IoU score
    iou_score = np.sum(intersection > 0) / np.sum(union > 0)
    return iou_score

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

def remove_non_gun_objects(image, ori_image):
    contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        # Calculate bounding box of the contour
        x, y, w, h = cv.boundingRect(contour)

        if w > 300 or h > 300:
            cv.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), thickness=cv.FILLED)
        if w < 15 or h < 15:
            cv.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), thickness=cv.FILLED)

    return image


def remove_background(image, ori_image):
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
    result1 = remove_non_gun_objects(result, ori_image)
    return result1


def find_contours(image):
    X_test_ = []
    contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        if (w < 200 or h < 200) or (w > 35 or h > 35):
            bounding_boxes.append((x, y, w, h))

    for (x, y, w, h) in bounding_boxes:
        gun_contour = cv.resize(image[y:y+h, x:x+w], (100,100))
        gun_contour[gun_contour >= 40] = 255
        gun_contour[gun_contour < 40] = 0
        gun_contour[:,:] = 255 - gun_contour[:,:]
        # cv.imshow("non_gun", gun_contour)
        # cv.waitKey(0)
        X_test_.append(gun_contour)
        # y_test.append(0)
    return X_test_

safe_dir_path = "students_data/test/safe"
safe_files = [f for f in os.listdir(safe_dir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif'))]
# print(len(gun_files))
gun_dir_path = "students_data/test/gun"
gun_anno_dir_path = "students_data/test/annotations/gun"
gun_ann_files = [f for f in os.listdir(gun_anno_dir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif'))]

gun_files = [f for f in os.listdir(gun_dir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif'))]
y_test = []
y_pred = []

import tensorflow
from keras.api.models import Sequential, load_model
from keras.api.layers import Conv2D, MaxPooling2D, Flatten, Dense

classification_model = load_model('classification_model.keras')


for a in range(len(gun_files)):
    print(f"Iteration: {a}, {gun_files[a]}")
    bck_image_gun = cv.imread(f"{gun_dir_path}/{gun_files[a]}")
    image_ann = cv.imread(f"{gun_anno_dir_path}/{gun_ann_files[a]}", 0)


    imager_ann = image_ann * 255



    blue_channel, green_channel, red_channel = cv.split(bck_image_gun)
    bck_image_gun = green_channel
    # Remove background
    image_gun = remove_background(bck_image_gun, bck_image_gun)
    cv.imwrite(f'segmented_output_gun/{gun_files[a]}', image_gun)
    new_width = 400
    new_height = 200
    new_size = (new_width, new_height)

    image1=cv.resize(image_gun,new_size)
    image2=cv.resize(image_ann,new_size)

    print(image1.shape)
    print(image2.shape)
    iou_score = IOU(image1, image2)

    X_test = find_contours(image_gun)

    if len(X_test) != 0:
        X_test = np.array(X_test)
        # Reshape X_data to match the input shape of the model
        X_test = X_test.reshape(-1, 100, 100, 1)
        # Predict labels
        y_pred_ = classification_model.predict(X_test)
        # Convert back to original labels
        y_pred_ = np.argmax(y_pred_, axis=1)
        print(y_pred_)
        if any(y_pred_) == 1:
            y_pred.append(1)
        else:
            y_pred.append(0)
    else:
        y_pred.append(0)
#     # image_gun = cv.resize(image_gun, (200,200))
#     # X_test.append(image_gun)
    y_test.append(1)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    image_ = cv.cvtColor(bck_image_gun, cv.COLOR_BGR2RGB)
    plt.imshow(image_)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    # _, thresh = cv.threshold(imager_ann, 0, 255, cv.THRESH_BINARY_INV)
    filtered_image = cv.cvtColor(imager_ann, cv.COLOR_BGR2RGB)
    plt.imshow(filtered_image)
    plt.title('Ground Truth Mask')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    segmented_image = cv.cvtColor(image_gun, cv.COLOR_BGR2RGB)
    plt.imshow(segmented_image)
    plt.title('Predicted Mask')
    plt.axis('off')
    plt.text(-1000, 300,f'Actual Label: {y_test[a]},Predicted Label: {y_pred[a]}, IOU score: {"%.4f" % iou_score}',
             fontsize=14, color='red')    # plt.show()
    plt.savefig(f'Results_Gun_Test/{gun_files[a]}.png')

for a in range(len(safe_files)):
    print(f"Iteration: {a}, {safe_files[a]}")
    # print(f"Iteration: {a}, {gun_ann_files[a]}")
    bck_image_gun = cv.imread(f"{safe_dir_path}/{safe_files[a]}")
    print(bck_image_gun.shape)
    blue_channel, green_channel, red_channel = cv.split(bck_image_gun)
    bck_image_gun = green_channel
    # Remove background
    image_gun = remove_background(bck_image_gun, bck_image_gun)
    cv.imwrite(f'segmented_output_safe/{safe_files[a]}', image_gun)

    # image_gun = cv.resize(image_gun, (200, 200))
    # X_test.append(image_gun)
    X_test = find_contours(image_gun)
    if len(X_test) != 0:
        X_test = np.array(X_test)
        # Reshape X_data to match the input shape of the model
        X_test = X_test.reshape(-1, 100, 100, 1)
        # Predict labels
        y_pred_ = classification_model.predict(X_test)
        # Convert back to original labels
        y_pred_ = np.argmax(y_pred_, axis=1)
        print(y_pred_)
        if any(y_pred_) == 1:
            y_pred.append(1)
        else:
            y_pred.append(0)
    else:
        y_pred.append(0)
    y_test.append(0)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    image_ = cv.cvtColor(bck_image_gun, cv.COLOR_BGR2RGB)
    plt.imshow(image_)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    segmented_image = cv.cvtColor(image_gun, cv.COLOR_BGR2RGB)
    plt.imshow(segmented_image)
    plt.title('Predicted Mask')
    plt.axis('off')
    # plt.show()
    plt.text(-200, 300,
             f'Actual Label: {y_test[a]}, Predicted Label: {y_pred[a]} ', fontsize=14, color='red')
    plt.savefig(f'Results_Gun_Test/{safe_files[a]}.png')



y_test = np.array(y_test)
y_pred = np.array(y_pred)

print(y_pred)
print(y_test)
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Plot the confusion matrix
cm = confusion_matrix(y_test, y_pred)
# Compute True Positives, False Positives, True Negatives, and False Negatives
TN, FP, FN, TP = cm.ravel()
# Compute the metrics
TPR = TP / (TP + FN)  # True Positive Rate (Recall)
FPR = FP / (FP + TN)  # False Positive Rate
TNR = TN / (TN + FP)  # True Negative Rate (Specificity)
FNR = FN / (FN + TP)  # False Negative Rate
accuracy = accuracy_score(y_test, y_pred)  # Accuracy
precision = precision_score(y_test, y_pred)  # Precision
dice_coefficient = 2 * (precision * TPR) / (precision + TPR)  # Dice Coefficient
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Safe', 'Gun'], yticklabels=['Safe', 'Gun'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig(f'Confusion Matrix.png')
print(f'True Positive Rate: {TPR:.2f}')
print(f'False Positive Rate: {FPR:.2f}')
print(f'True Negative Rate: {TNR:.2f}')
print(f'False Negative Rate: {FNR:.2f}')
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Dice Coefficient: {dice_coefficient:.2f}')



