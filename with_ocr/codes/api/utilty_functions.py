
import cv2
import easyocr
import numpy as np
import tensorflow as tf

#Local imports
from Logging import Logging
from Config import ConfigReader


class check_process():
    def __init__(self, ):
        self.model_path = ""
        self.model = tf.saved_model.load(self.model_path)
        self.reader = easyocr.Reader(['en'])
          
    #function to read images 
    def read_image(self,image_path):
        try:
            image = cv2.imread(image_path)
            return image
        except Exception as image_read_error:
            print("Error while reading image is: %s"%image_read_error)
        
    #function for image processing
    def pre_process_image(image_as_np,expected_size):
        try:
            image_rgb = cv2.cvtColor(image_as_np, cv2.COLOR_BGR2RGB)
            dim = image_rgb.shape[0:2]
            image_np = cv2.resize(image_rgb, expected_size)
            return dim,image_np
        except Exception as image_processing_error:
            print("Error while processing image is: %s"%image_processing_error)
        
    # function to perform object detection
    def perform_object_detection(self,image):
        input_tensor = tf.convert_to_tensor(image)
        
        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis, ...]

        detections = self.model(input_tensor)

        return detections

    #Post process function
    def post_process_fun():
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        boxes=(detections['detection_boxes'])
        classes =(detections['detection_classes'])
        scores = (detections['detection_scores'])
                    
        cls = classes.tolist()

        dets = []
        for i, value in enumerate(cls):
            if scores[i] > threshold:
                (y1, x1) = (boxes[i][0], boxes[i][1])
                (y2, x2) = (boxes[i][2], boxes[i][3])                        
                dets.append([x1,y1,x2,y2,scores[i],cls[i]])

        for variables in dets :
            x1, y1, x2, y2, score, cls = variables

            #rescaling  to original dim
            pt1 = (int(x1 * dim[1]), int(y1 * dim[0])) 
            pt2 = (int(x2 * dim[1]), int(y2 * dim[0]))

            # Extract the ROI for OCR
            roi = image[pt1[1]:pt2[1], pt1[0]:pt2[0]]
            
            digit_list = [1,2,4]  # this list includes date, amount and micra

            # Perform OCR using EasyOCR
            # result = reader.readtext(roi)
            if cls in digit_list:
                result = reader.readtext(roi, allowlist ='0123456789', paragraph=True)
            else:
                result= reader.readtext(roi, paragraph=True)

            # Extract text from EasyOCR result
            text = result[0][1] if result else ""
            pass 

        #OCR function
        def ocr_function():
            pass

