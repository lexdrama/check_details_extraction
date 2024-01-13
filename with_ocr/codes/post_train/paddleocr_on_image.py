import cv2
from paddleocr import PaddleOCR, draw_ocr

ocr = PaddleOCR()

# Replace 'your_image_path' with the path to your image file
img_path = 'with_ocr/codes/post_train/1.JPG'

# Perform OCR on the image
result = ocr.ocr(img_path, cls=True)

# Draw the OCR result on the image
image = draw_ocr(img_path, result)

# Save the result image
output_path = 'output.jpg'
cv2.imwrite(output_path, image)

# Print the OCR result
for line in result:
    line_text = ' '.join([word_info[-1] for word_info in line])
    print(line_text)