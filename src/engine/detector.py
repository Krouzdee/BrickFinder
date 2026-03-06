import cv2
import torch
import numpy as np
from ultralytics import YOLO
from torchvision import models, transforms
from scipy.spatial.distance import cosine
from utils.storage import LegoStorage
from PIL import Image

class LegoDetector:
    def __init__(self, yolov8_model_path='тут будет путь к обученой модели'):
        self.storage = LegoStorage()
        self.detector = YOLO(yolov8_model_path)
        
        # Это экстрактор признаков
        resnet = models.resnet18(pretrained=True)
        self.encoder = torch.nn.Sequential(*(list(resnet.children())[:-1]))
        self.encoder.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.current_target_name = ""
        self.target_vector = None
        self.target_color_hist = None

    # Здесь получаем вектор
    def get_vector(self, cv2_img):
        img = Image.fromarray(cv2_img, cv2.COLOR_BGR2RGB)
        img_t = self.transform(img).unsqueeze(0)
        with torch.no_grad():
            vec = self.encoder(img_t)
        return vec.flatten().numpy()
    
    # Преобразуем фото в HSV 
    def get_color_hist(self, cv2_img):
        hsv = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist
    
    def add_new_break(self, frame, display_name):
        res = self.detector(frame, conf=0.4, verbose=False)
        if len(res) > 0 and len(res[0].boxes) > 0:
            b = res[0].boxes.xyxy[0].cpu().numpy().astype(int)
            crop = frame[b[1]:b[3], b[0]:b[2]]
        else:
            crop = frame

        vector = self.get_vector(crop)
        hist = self.get_color_hist(crop)

        self.storage.save_reference(display_name, crop, vector, hist)
        return True