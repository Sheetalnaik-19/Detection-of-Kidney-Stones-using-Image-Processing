import cv2
import numpy as np

def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply GaussianBlur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred

def detect_edges(image):
    # Apply adaptive thresholding to get a binary image
    adaptive_thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    # Apply morphological operations to remove noise and close gaps
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
    # Apply Canny edge detection
    edges = cv2.Canny(morph, 50, 150)
    return edges

def find_contours(image):
    # Find contours in the edged image
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def draw_contours(image, contours):
    # Draw contours on the original image
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Filter out small contours
            cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
    return image

def check_for_kidney_stones(contours):
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Define a threshold for significant contours
            return True
    return False

def main():
    # Load the image
    image = cv2.imread('kidney_stone.jpg')
    if image is None:
        print("Could not read image")
        return

    # Preprocess the image
    preprocessed = preprocess_image(image)

    # Detect edges
    edges = detect_edges(preprocessed)

    # Find contours
    contours = find_contours(edges)

    # Check for kidney stones
    if check_for_kidney_stones(contours):
        print("Kidney stone detected")
        # Draw contours on the original image
        result = draw_contours(image.copy(), contours)
        cv2.imshow('Detected Kidney Stones', result)
    else:
        print("No kidney stone detected")
        cv2.imshow('No Kidney Stones Detected', image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
