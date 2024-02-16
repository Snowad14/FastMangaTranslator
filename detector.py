from ultralytics import YOLO
from utils import SpeechBubble
import numpy as np
import os

class Detector:
    def __init__(self, model_path, use_tensorrt=True):
        self.model_path = model_path

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Error : Not Found the model  {model_path} in the model dir folder")
        
        if use_tensorrt:
            if not os.path.exists(model_path.replace(".pt", ".engine")):
                print(f"Warning : TensorRT engine not found for {model_path}. Creating one now...")
                model = YOLO(model_path)
                model.export(format="engine")
                print(f"TensorRT engine saved to {self.model_path}")
            self.model_path = model_path.replace(".pt", ".engine")

        self.model = YOLO(self.model_path, task="detect")
    
    def __adjust_bubble(self, bubbles):
        return bubbles
            
    def predict(self, img_path, imgsz=1024):
        result =  self.model.predict(img_path, imgsz=imgsz)[0]
        bboxes = result.boxes
        no_text_bubbles = []
        for box in bboxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            box_type = int(box.cls.item())
            no_text_bubbles.append(SpeechBubble(x1, y1, x2, y2, box_type))
        no_text_bubbles = self.__adjust_bubble(no_text_bubbles)
        return no_text_bubbles 


if __name__ == "__main__":
    detector = Detector("models/comic-speech-bubble-detector.pt", use_tensorrt=True)
    bubbles = detector.predict("images/ex2.png")
    


    # model = YOLO("models/bubble_detector.engine", task="detect")
    # result = model.predict("images/complex.jpg", imgsz=1024)
