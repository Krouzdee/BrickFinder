import os
import cv2
import pickle
import re

class LegoStorage:
    """
    Класс для сохранения и загрузки данных о деталях Lego.
    """
    def __init__(self, base_dir='data'):
        self.base_dir = base_dir
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

    def make_safe_name(self, name):
        """
        Внутренний метод для создания безопасного имени файла
        """
        forbidden_chars = r'[\\/*?:"<>|]'
        safe = re.sub(forbidden_chars, '_', name)

        safe = re.sub(r'[\s-]+', '_', safe)

        safe = safe.strip('.')

        if not safe:
            safe = "unnamed"
        return safe
            

    def save_reference(self, name, image, vector, hist):
        """
        Сохраняет новую эталонную деталь

        Args:
            name (str): Имя детали, введенное пользователем.
            image (numpy.ndarray): Изображение детали.
            vector (numpy.ndarray): Вектор признаков формы.
            hist (numpy.ndarray): Гистограмма цвета.

        Returns:
            str: Путь к сохраненному изображению.
        """

        safe_name = self.make_safe_name(name)
        img_path = os.path.join(self.base_dir, f"{safe_name}.jpg")
        pkl_path = os.path.join(self.base_dir, f"{safe_name}.pkl")

        if os.path.exists(pkl_path):
            return False
    
        cv2.imwrite(img_path, image)
        data = {
            'name': name,
            'safe_name': safe_name, 
            'vector': vector, 
            'hist': hist
        }

        with open(pkl_path, 'wb') as f:
            pickle.dump(data, f)
        return img_path

    def load_reference(self, safe_name):
        """
        Загружает данные детали по ее безопасному имени

        Args:
            safe_name (str): Имя файла без расширения.

        Returns:
            dict: Словарь с данными детали или None, если файл не найден.
        """
        pkl_path = os.path.join(self.base_dir, f"{safe_name}.pkl")
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                return pickle.load(f)
        return None

    def delete_target(self, safe_name):
        """
        Удаляет деталь из базы данных.
    
        Args:
            safe_name (str): Безопасное имя детали (имя файла без расширения)
        
        Returns:
            bool: True если удаление успешно, False если файлы не найдены
        """
        img_path = os.path.join(self.base_dir, f"{safe_name}.jpg")
        pkl_path = os.path.join(self.base_dir, f"{safe_name}.pkl")
    
        deleted = False
    
        if os.path.exists(pkl_path):
            os.remove(pkl_path)
            deleted = True
    
        if os.path.exists(img_path):
            os.remove(img_path)
            deleted = True
    
        return deleted


    def get_available_parts(self):
        """
        Получает список всех сохраненных деталей

        Returns:
            dict: Словарь {безопасное_имя: оригинальное_имя}
        """
        parts = {}
        for f in os.listdir(self.base_dir):
            if f.endswith('.pkl'):
                safe_name = f.replace('.pkl', '')
                data = self.load_reference(safe_name)
                if data:
                    parts[safe_name] = data.get('name', safe_name)
        return parts