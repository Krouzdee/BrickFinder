import cv2
import torch
import numpy as np
from ultralytics import YOLO
from torchvision import models, transforms
from scipy.spatial.distance import cosine
from ..utils import LegoStorage
from PIL import Image

class LegoDetector:
    def __init__(self, yolov8_model_path='yolov8n.pt'): # Потом тут будет другая модель
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

    def get_vector(self, cv2_img):
        img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img_rgb)
        img_t = self.transform(img).unsqueeze(0)
        with torch.no_grad():
            vec = self.encoder(img_t)
        return vec.flatten().numpy()
    
    # HSV фильтр
    def get_color_hist(self, cv2_img):
        hsv = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist
    
    # Добавление новых деталей в базу
    def add_new_target(self, frame, display_name):
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
    
    # Метод для смены цели 
    def switch_target(self, safe_name):
        data = self.storage.load_reference(safe_name)
        
        if data:
            self.target_vector = data['vector']
            self.target_color_hist = data['hist']
            self.current_target_name = data['name']
            return True
        return False

    # Обработка
    def process_frame(self, frame, threshold_percent=70):
        if self.target_vector is None:
            return frame
        threshold = threshold_percent / 100.0

        results = self.detector.predict(frame, conf=0.25, verbose=False)

        if len(results) > 0 and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    continue
        
            cur_vec = self.get_vector(roi)
            shape_sim = 1 - cosine(self.target_vector, cur_vec)
                
            cur_hist = self.get_color_hist(roi)
            color_sim = cv2.compareHist(self.target_color_hist, cur_hist, cv2.HISTCMP_CORREL)
            color_sim = max(0, color_sim)

            final_sim = (shape_sim * 0.6) + (color_sim * 0.4)

            if final_sim >= threshold:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                score_text = f"{self.current_target_name} {int(final_sim * 100)}%"
                    
                cv2.rectangle(frame, (x1, y1 - 25), (x1 + len(score_text) * 10, y1), (0, 255, 0), -1)
                           
                cv2.putText(frame, score_text, (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame