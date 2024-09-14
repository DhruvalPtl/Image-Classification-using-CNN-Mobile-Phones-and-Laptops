import cv2

# Check OpenCV version
print("OpenCV version:", cv2.__version__)

# Simple test to see if the library loads correctly
img = cv2.imread("path_to_an_image.jpg")  # Replace with an actual image path
if img is None:
    print("Image not loaded correctly!")
else:
    print("OpenCV is installed and working correctly!")