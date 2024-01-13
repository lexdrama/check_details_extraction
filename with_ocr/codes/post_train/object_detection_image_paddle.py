
import os 
from paddleocr import PaddleOCR,draw_ocr
import cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# Initialize the PaddleOCR OCR reader
ocr = PaddleOCR(use_angle_cls=True,lang='en')

def save_ocr(img_path, out_path, result, font):
  save_path = os.path.join(out_path, img_path.split('/')[-1] + 'output')
 
  image = cv2.imread(img_path)
 
  boxes = [line[0] for line in result]
  txts = [line[1][0] for line in result]
  scores = [line[1][1] for line in result]
 
  im_show = draw_ocr(image, boxes, txts, scores,font_path=font)
  im_show = Image.fromarray(im_show)
  im_show.save('result.jpg')
  
  
  plt.imshow(im_show)

# Specify the path to your input image
image_path = 'with_ocr/codes/post_train/1.JPG'

out_path = './output_images'
font = './simfang.ttf'

result = ocr.ocr(image_path)

save_ocr(image_path, out_path, result, font)
