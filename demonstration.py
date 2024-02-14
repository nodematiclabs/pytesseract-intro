import pytesseract
import cv2

import numpy as np

from PIL import Image, ImageEnhance

# Read the screenshot and convert to greyscale
cv2_image = cv2.imread('screenshot.png')
cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)

# Find contours of empty white space (isolating text/elements which have pure white backgrounds)
_, binary = cv2.threshold(cv2_image, 240, 255, cv2.THRESH_BINARY)
kernel = np.ones((25, 25), np.uint8)
closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
contours, hierarchy = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Extract the isolated text/elements, and put them on a "collage" image
collage = np.ones_like(cv2_image) * 255
for j, contour in enumerate(contours[:-1]):
    x, y, w, h = cv2.boundingRect(contour)
    collage[y:y+h, x:x+w] = cv2_image[y:y+h, x:x+w]
    cv2_image[y:y+h, x:x+w] = 255

# Write out the two splits of the image
cv2.imwrite('stripped.png', cv2_image)
cv2.imwrite('collage.png', collage)

# Image processing to improve text readability for the model
stripped_image = Image.fromarray(cv2_image)
collage_image = Image.fromarray(collage)
brightener = ImageEnhance.Brightness(stripped_image)
stripped_image = brightener.enhance(1.8)
contraster = ImageEnhance.Contrast(stripped_image)
stripped_image = contraster.enhance(1.8)
# sharpener = ImageEnhance.Sharpness(image)
# image = sharpener.enhance(1.5)
stripped_image.save('stripped_edited.png')

# Get and print extracted text
data = pytesseract.image_to_data(
    stripped_image,
    output_type=pytesseract.Output.DICT
)
collage_data = pytesseract.image_to_data(
    collage_image,
    output_type=pytesseract.Output.DICT
)
for key in data.keys():
    data[key] = data[key] + collage_data[key]
print(data)