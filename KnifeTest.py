import numpy as np
import os
import cv2 as cv
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
    # Apply a binary threshold to get a binary image
    _, binary = cv.threshold(result, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return binary


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


def clustering(image, a):
    # Apply GaussianBlur to smooth the image
    image = cv.GaussianBlur(image, (5, 5), 0)
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
    # # Plot the original and segmented images
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

    min_size = 500
    thresh = remove_small_objects(image, min_size)
    contours, _ = cv.findContours(darkest_cluster_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        window = thresh[y:y+h, x:x+w]
        if window.max == 0:
            cv.rectangle(darkest_cluster_image, (x, y), (x + w, y + h), (0, 0, 0), thickness=cv.FILLED)
    contours, _ = cv.findContours(darkest_cluster_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        bounding_boxes.append((x, y, w, h))
    b = 0
    for (x, y, w, h) in bounding_boxes:
        contour = cv.resize(darkest_cluster_image[y:y + h, x:x + w], (100, 100))
        contour[contour >= 40] = 255
        contour[contour < 40] = 0
        contour[:, :] = 255 - contour[:, :]
        cv.imwrite(f"test2_knife/{a}{b}.png", contour)
        b += 1

    return darkest_cluster_image


def remove_background_knife(image, mask, a):
    bounding_boxes = find_bounding_boxes(mask)
    thresh = np.zeros_like(image)
    b=0
    for (x, y, w, h) in bounding_boxes:
        thresh[y:y+h, x:x+w] = image[y:y+h, x:x+w]
        knife_contour = cv.resize(image[y:y+h, x:x+w], (100,100))
        # Set all non-zero pixels to 255
        knife_contour[knife_contour >= 140] = 255
        knife_contour[knife_contour < 140] = 0
        kernel = np.ones((3, 3), np.uint8)
        image = cv.erode(image, kernel, iterations=1)
        image = cv.dilate(image, kernel, iterations=1)
        image = remove_small_objects2(image, 500)
        cv.imwrite(f"test_knife/{a}{b}.png", knife_contour)
        b += 1

    return thresh


safe_dir_path = "dataCopy/train/safe"
safe_files = [f for f in os.listdir(safe_dir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif'))]
knife_dir_path = "train/train/knife"
knife_files = [f for f in os.listdir(knife_dir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif'))]
knife_ann_dir_path = "train/train/annotations/knife"
knife_ann_files = [f for f in os.listdir(knife_ann_dir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif'))]


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
    predicted_mask = remove_background_knife(imager_knife, imager_ann, knife_files[a])


    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    image_ = cv.cvtColor(imager_knife, cv.COLOR_BGR2RGB)
    plt.imshow(image_)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    _, thresh = cv.threshold(imager_ann, 0, 255, cv.THRESH_BINARY_INV)
    filtered_image = cv.cvtColor(thresh, cv.COLOR_BGR2RGB)
    plt.imshow(filtered_image)
    plt.title('Ground Truth Mask')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    segmented_image = cv.cvtColor(predicted_mask, cv.COLOR_BGR2RGB)
    plt.imshow(segmented_image)
    plt.title('Predicted Mask')
    plt.axis('off')
    plt.savefig(f'Results_Knife_Train/{knife_files[a]}.png')


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

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    image_ = cv.cvtColor(imager_knife, cv.COLOR_BGR2RGB)
    plt.imshow(image_)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    segmented_image = cv.cvtColor(predicted_mask, cv.COLOR_BGR2RGB)
    plt.imshow(segmented_image)
    plt.title('Predicted Mask')
    plt.axis('off')
    plt.savefig(f'Results_Knife_Train/{safe_files[a]}.png')
    # plt.show()