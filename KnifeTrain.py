import numpy as np
import os
import cv2 as cv
import matplotlib.pyplot as plt


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
    # Apply a binary threshold to get a binary image
    _, binary = cv.threshold(result, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # print(result.shape)
    return binary


def clustering(image, a):
    # Apply GaussianBlur to smooth the image
    image = cv.GaussianBlur(image, (5, 5), 0)
    # image = histogram_equalization(image)
    # Reshape the image to a 2D array of pixels
    pixel_values = image.reshape((-1, 1))
    # Convert to float
    pixel_values = np.float32(pixel_values)
    # Define the criteria and apply kmeans()
    k = 5  # Number of clusters
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv.kmeans(pixel_values, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

    # Convert centers to 8-bit values
    centers = np.uint8(centers)

    # Identify the darkest cluster
    darkest_cluster = np.argmin(centers)

    # Create a mask where pixels belonging to the darkest cluster are set to 255 and others to 0
    mask = np.zeros(labels.shape, dtype=np.uint8)
    mask[labels == darkest_cluster] = 255

    # Reshape the mask back to the original image shape
    mask = mask.reshape(image.shape)

    # Apply the mask to the original image
    darkest_cluster_image = cv.bitwise_and(image, image, mask=mask)
    darkest_cluster_image  = remove_small_objects(darkest_cluster_image, 500)
    # Map the labels to the centers
    segmented_image = centers[labels.flatten()]

    # Reshape back to the original image shape
    segmented_image = segmented_image.reshape(image.shape)
    # Plot the original and segmented images
    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 3, 1)
    # plt.title('Original Image')
    # plt.imshow(image, cmap='gray')
    # plt.axis('off')
    #
    # plt.subplot(1, 3, 2)
    # plt.title('Segmented Image')
    # plt.imshow(segmented_image, cmap='gray')
    # plt.axis('off')
    #
    # plt.subplot(1, 3, 3)
    # plt.title('Darkest Cluster')
    # plt.imshow(darkest_cluster_image, cmap='gray')
    # plt.axis('off')
    # plt.savefig(f'Segmented_Knife/{a}.png')
    return darkest_cluster_image


def remove_background(image, a):
    darkest_cluster_image = clustering(image, a)
    # Apply GaussianBlur to smooth the image
    image = cv.GaussianBlur(image, (7, 7), 0)

    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            if 11 <= image[x,y] <= 69:
                image[x,y] = 255
            else:
                image[x,y] = 0

    min_size = 100
    thresh = remove_small_objects(image, min_size)

    contours, _ = cv.findContours(darkest_cluster_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        window = thresh[y:y+h, x:x+w]
        if window.max == 0:
            cv.rectangle(darkest_cluster_image, (x, y), (x + w, y + h), (0, 0, 0), thickness=cv.FILLED)
    kernel = np.ones((5, 5), np.uint8)
    darkest_cluster_image = cv.dilate(darkest_cluster_image, kernel, iterations=1)
    darkest_cluster_image = cv.erode(darkest_cluster_image, kernel, iterations=1)
    darkest_cluster_image = cv.dilate(darkest_cluster_image, kernel, iterations=1)
    darkest_cluster_image = cv.erode(darkest_cluster_image, kernel, iterations=1)
    return darkest_cluster_image


def find_contours(image):
    X_test_ = []
    contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        if w < 200 or h < 200:
            bounding_boxes.append((x, y, w, h))

    for (x, y, w, h) in bounding_boxes:
        gun_contour = cv.resize(image[y:y+h, x:x+w], (100,100))
        gun_contour[gun_contour >= 40] = 255
        gun_contour[gun_contour < 40] = 0
        gun_contour[:,:] = 255 - gun_contour[:,:]
        X_test_.append(gun_contour)
    return X_test_


def IOU(predicted, mask):
    # Calculate the intersection
    intersection = cv.bitwise_and(predicted, mask)
    # Calculate the union
    union = cv.bitwise_or(predicted, mask)
    # Compute the IoU score
    iou_score = np.sum(intersection > 0) / np.sum(union > 0)
    return iou_score


safe_dir_path = "dataCopy/test/safe"
safe_files = [f for f in os.listdir(safe_dir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif'))]
knife_dir_path = "dataCopy/test/knife"
# knife_dir_path = "test-20240518T172450Z-001/test/knife"
knife_files = [f for f in os.listdir(knife_dir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif'))]
knife_ann_dir_path = "dataCopy/test/annotations/knife"
# knife_ann_dir_path = "test-20240518T172450Z-001/test/annotations/knife"
knife_ann_files = [f for f in os.listdir(knife_ann_dir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif'))]

from keras.api.models import load_model


classification_model = load_model('classification_model_knife.keras')
y_pred, y_test = [], []

for a in range(len(knife_files)):
    print(f"Iteration: {a}, {knife_files[a]}")
    image_knife = cv.imread(f"{knife_dir_path}/{knife_files[a]}")
    image_ann = cv.imread(f"{knife_ann_dir_path}/{knife_ann_files[a]}", 0)
    blue_channel, green_channel, red_channel = cv.split(image_knife)
    image_knife = red_channel

    new_width = 400
    new_height = 200
    new_size = (new_width, new_height)

    # Resize the image
    imager_knife = cv.resize(image_knife, new_size)
    imager_ann = cv.resize(image_ann, new_size)

    imager_ann = imager_ann *255

    predicted_mask = remove_background(imager_knife, knife_files[a])
    iou_score = IOU(predicted_mask, imager_ann)
    X_test = find_contours(predicted_mask)
    if len(X_test) != 0:
        X_test = np.array(X_test)
        # Reshape X_data to match the input shape of the model
        X_test = X_test.reshape(-1, 100, 100, 1)
        # Predict labels
        y_pred_ = classification_model.predict(X_test)
        if any(y_pred_) == 1:
            y_pred.append(1)
        else:
            y_pred.append(0)
    else:
        y_pred.append(0)

    y_test.append(1)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    image_ = cv.cvtColor(imager_knife, cv.COLOR_BGR2RGB)
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
    segmented_image = cv.cvtColor(predicted_mask, cv.COLOR_BGR2RGB)
    plt.imshow(segmented_image)
    plt.title('Predicted Mask')
    plt.axis('off')

    plt.text(-1000, 300,
             f'Actual Label: {y_test[a]},       Predicted Label: {y_pred[a]},       IOU score: {"%.4f" % iou_score}',
             fontsize=14, color='red')
    plt.savefig(f'Results_Knife_Test/{knife_files[a]}.png')
    # plt.show()


for a in range(len(safe_files)):
    print(f"Iteration: {a}, {safe_files[a]}")
    image_knife = cv.imread(f"{safe_dir_path}/{safe_files[a]}")
    blue_channel, green_channel, red_channel = cv.split(image_knife)
    image_knife = red_channel

    new_width = 400
    new_height = 200
    new_size = (new_width, new_height)

    # Resize the image
    imager_knife = cv.resize(image_knife, new_size)
    predicted_mask = remove_background(imager_knife, safe_files[a])
    X_test = find_contours(predicted_mask)
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
    image_ = cv.cvtColor(imager_knife, cv.COLOR_BGR2RGB)
    plt.imshow(image_)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    segmented_image = cv.cvtColor(predicted_mask, cv.COLOR_BGR2RGB)
    plt.imshow(segmented_image)
    plt.title('Predicted Mask')
    plt.axis('off')
    plt.text(-200, 300,
             f'Actual Label: {y_test[a]}, Predicted Label: {y_pred[a]} ',
             fontsize=14, color='red')
    plt.savefig(f'Results_Knife_Test/{safe_files[a]}.png')
    # plt.show()


y_test = np.array(y_test)
y_pred = np.array(y_pred)

print(y_pred)
print(y_test)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score


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
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Safe', 'Knife'], yticklabels=['Safe', 'Knife'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
# Annotate with the metrics
print(f'True Positive Rate: {TPR:.2f}')
print(f'False Positive Rate: {FPR:.2f}')
print(f'True Negative Rate: {TNR:.2f}')
print(f'False Negative Rate: {FNR:.2f}')
print(f'Accuracy: {accuracy:.2f}')
print(f'Dice Coefficient: {dice_coefficient:.2f}')
plt.savefig(f'Confusion Matrix_knife.png')