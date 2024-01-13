import cv2
import easyocr
import numpy as np
import tensorflow as tf

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

#Local imports
from Logging import Logging
from Config import ConfigReader

# Initialize the logger
logger = Logging.get_logger('app_logs')


class check_process:
    def __init__(self, ):
        self.config = self.get_configuration()
        self.model_path = self.config['object_detection_config']['model_path']
        self.threshold  = float(self.config['object_detection_config']['threshold'])
        self.expected_size= int(self.config['object_detection_config']['expected_size'])


        self.model = tf.saved_model.load(self.model_path)
        self.reader = easyocr.Reader(['en'])

        self.app = FastAPI()

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],)
        
        self.app.post("/process_check")(self.run_simulation)

    # Read configuration using ConfigReader
    def get_configuration(self):
        try:
            conf = ConfigReader()
            config_path = 'conf.ini'
            config = conf.read_config_section(config_path)
            logger.info("Configuration has been read from file")
            return config
        except Exception as e:
            logger.error("Error reading configuration: %s", str(e))
          
    #function to read images 
    def read_image(self,image_path):
        try:
            self.image = cv2.imread(image_path)
            return self.image
        except Exception as image_read_error:
            print("Error while reading image is: %s"%image_read_error)
        
    #function for image processing
    def pre_process_image(self):
        try:
            self.image_original = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.dim = self.image_original.shape[0:2]
            self.image_np = cv2.resize(self.image_original, (self.expected_size,self.expected_size))
            return self.dim,self.image_np
        except Exception as image_processing_error:
            print("Error while processing image is: %s"%image_processing_error)
        
    # function to perform object detection
    def perform_object_detection(self):
        input_tensor = tf.convert_to_tensor(self.image_np)
        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis, ...]
        self.detections = self.model(input_tensor)
        return self.detections
    
    # # Dependency to check if the image file exists
    # def check_image_file_exists(image_path: FilePath = Query(..., description="Path to the image file")):
    #     if not image_path.is_file():
    #         raise HTTPException(status_code=404, detail="Image file not found")
    #     return image_path

    #Post process function
    def post_process_fun(self):
        num_detections = int(self.detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
        detections['num_detections'] = num_detections
        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
        boxes=(detections['detection_boxes'])
        classes =(detections['detection_classes'])
        scores = (detections['detection_scores'])
                    
        cls = classes.tolist()

        dets = []
        for i, value in enumerate(cls):
            if scores[i] > self.threshold:
                (y1, x1) = (boxes[i][0], boxes[i][1])
                (y2, x2) = (boxes[i][2], boxes[i][3])                        
                dets.append([x1,y1,x2,y2,scores[i],cls[i]])

        json_dict  = {}
        for variables in dets :
            x1, y1, x2, y2, score, cls = variables

            #rescaling  to original dim
            pt1 = (int(x1 * self.dim[1]), int(y1 * self.dim[0])) 
            pt2 = (int(x2 * self.dim[1]), int(y2 * self.dim[0]))
            # Extract the ROI for OCR
            roi = self.image_original[pt1[1]:pt2[1], pt1[0]:pt2[0]]
            digit_list = [1,2,4]  # this list includes date, amount and micra

            # Perform OCR using EasyOCR
            if cls in digit_list:
                result = self.reader.readtext(roi, allowlist ='0123456789', paragraph=True)
            else:
                result= self.reader.readtext(roi, paragraph=True)

            # Extract text from EasyOCR result
            text = result[0][1] if result else ""

            return json_dict
            
    #Route for process checks  
    async def process_check(self,image_path: str = Path(..., description="Path to the image file")):
        try:
            self.read_image(image_path)
            self.pre_process_image()
            self.perform_object_detection()
            response = self.post_process_fun()
            
            return JSONResponse(content=response)

        except Exception as e:
            return JSONResponse(content={"message": f"Error processing image: {str(e)}"}, status_code=500)
            pass

if __name__ == "__main__":
    app_instance = check_process()
    #Start the FastAPI server
    uvicorn.run(app_instance.app, host="localhost", port=8005)
    logger.info("[main]: app is running at 80 port")

