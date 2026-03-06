import cv2
import numpy as np
import torch

class LegoDetector:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")
        self.target_features = None
        self.target_name = ""

    def set_target(self, name):
        with open(f"data/{name}.pkl", "rb") as f:
            data = pickle.load(f)
            self.target_features = data['features']
            self.target_name = name
    
    def detect(self, frame, treshold=0.5):
        results = self.model.predict(frame, conf=0.25, verbose=False)

        
    