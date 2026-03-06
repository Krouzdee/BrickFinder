import os
import cv2
import pickle

class LegoStorage:
    def __init__(self, base_dir='data'):
        self.base_dir = base_dir
        # Добавил тут проверку на наличие папки пусть будет на всякий
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
            

    def save_reference(self, name, image, vector, hist):
        safe_name = name.replace(" ", "_")
        img_path = os.path.join(self.base_dir, f"{safe_name}.jpg")
        pkl_path = os.path.join(self.base_dir, f"{safe_name}.pkl")

        cv2.imwrite(img_path, image)
        data = {
            'name': name, # Имя для детальки
            'vector': vector, 
            'hist': hist
        }

        with open(pkl_path, 'wb') as f:
            pickle.dump(data, f)
        return img_path

    def load_reference(self, safe_name):
        pkl_path = os.path.join(self.base_dir, f"{safe_name}.pkl")
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                return pickle.load(f)
        return None

    def get_available_parts(self):
        parts = {}
        for f in os.listdir(self.base_dir):
            if f.endswith('.pkl'):
                safe_name = f.replace('.pkl', '')
                data = self.load_reference(safe_name)
                if data:
                    parts[safe_name] = data.get('name', safe_name)
        return parts