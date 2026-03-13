import cv2
import torch
import numpy as np
from ultralytics import YOLO
from torchvision import models, transforms
from torchvision.models import ConvNeXt_Tiny_Weights
from scipy.spatial.distance import cosine
from ..utils import LegoStorage
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict
from typing import Optional, Tuple, Dict, Any, List
from queue import Queue


class LegoDetector:
    """
    Основной класс компьютерного зрения
    """

    def __init__(self, yolov8_model_path: str = 'models/lego_detector200.pt') -> None:
        """
        Инициализация детектора.

        Args:
            yolov8_model_path: Путь к предобученной модели YOLOv8.
        """
        self.storage: LegoStorage = LegoStorage()

        self.device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.detector: YOLO = YOLO(yolov8_model_path)
        if self.device.type == 'cuda':
            self.detector.to('cuda')

        self.trackers: defaultdict = defaultdict(dict)
        self.track_id: int = 0
        convnext = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        self.encoder: torch.nn.Module = torch.nn.Sequential(
            convnext.features,
            convnext.avgpool,
            torch.nn.Flatten()
        )
        self.encoder.eval()
        self.encoder.to(self.device)

        if self.device.type == 'cuda':
            self.encoder = self.encoder.half()
        self.transform: transforms.Compose = transforms.Compose([
            transforms.Resize(232),  # Немного больше 224 для лучшего кропа
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.current_target_name: str = ""
        self.current_safe_name: str = ""
        self.target_vector: Optional[np.ndarray] = None
        self.target_color_hist: Optional[np.ndarray] = None
        self.vector_cache: Dict[Tuple[int, int, int, int], np.ndarray] = {}
        self.cache_size: int = 50
        self.processing_queue: Queue = Queue(maxsize=10)
        self.result_queue: Queue = Queue()

        self.feature_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self.frame_counter: int = 0
        self.recompute_interval: int = 5
        self.base_font_path: Optional[str] = None
        for candidate in ("arial.ttf", "DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
            try:
                ImageFont.truetype(candidate, 22)
                self.base_font_path = candidate
                break
            except Exception:
                continue

    def _clean_cache(self) -> None:
        """Очистка устаревших записей в кэше"""
        if len(self.vector_cache) > self.cache_size:
            items = list(self.vector_cache.items())
            self.vector_cache = dict(items[-self.cache_size // 2:])

    def get_vector(self, cv2_img: np.ndarray) -> np.ndarray:
        if cv2_img is None or cv2_img.size == 0:
            return np.zeros(768, dtype=np.float32)  # ConvNeXt Tiny выдает 768-dim вектор

        small = cv2.resize(cv2_img, (8, 8), interpolation=cv2.INTER_NEAREST)
        img_hash = hash(small.tobytes())

        if img_hash in self.vector_cache:
            return self.vector_cache[img_hash].copy()

        img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        img_t = self.transform(Image.fromarray(img_rgb)).unsqueeze(0).to(self.device)

        if self.device.type == 'cuda':
            img_t = img_t.half()

        with torch.inference_mode():
            vec = self.encoder(img_t)
            vec_np = vec.detach().cpu().float().numpy().flatten()

        norm = np.linalg.norm(vec_np)
        if norm > 1e-9:
            vec_np /= norm
        else:
            vec_np = np.zeros_like(vec_np)

        self.vector_cache[img_hash] = vec_np
        self._clean_cache()
        return vec_np

    def batch_get_vectors(self, rois: List[np.ndarray]) -> List[np.ndarray]:
        """
        Batch extraction of feature vectors for multiple ROIs.
        """
        if not rois:
            return []

        imgs_t = []
        for roi in rois:
            img_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img_rgb)
            img_t = self.transform(img)
            imgs_t.append(img_t)

        batch_t = torch.stack(imgs_t).to(self.device)
        if self.device.type == 'cuda':
            batch_t = batch_t.half()

        with torch.no_grad():
            vecs = self.encoder(batch_t)

        vecs_np = []
        for vec in vecs:
            v = vec.flatten().cpu().numpy()
            norm = np.linalg.norm(v)
            if norm > 0:
                v = v / norm
            vecs_np.append(v)
        return vecs_np

    def get_dominant_color(self, cv2_img: np.ndarray, k: int = 3) -> np.ndarray:
        """
        Находит доминантный цвет изображения с помощью K-means кластеризации.
        
        Args:
            cv2_img: Изображение в формате BGR
            k: Количество кластеров для K-means
        
        Returns:
            np.ndarray: Доминантный цвет в формате BGR
        """
        if cv2_img is None or cv2_img.size == 0:
            return np.array([0, 0, 0], dtype=np.uint8)
        hsv = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 50, 40]), np.array([180, 255, 255]))
        img_small = cv2.resize(cv2_img, (100, 100), interpolation=cv2.INTER_LINEAR)
        mask_small = cv2.resize(mask, (100, 100), interpolation=cv2.INTER_NEAREST)
        pixels = img_small[mask_small > 0].reshape(-1, 3)
        if len(pixels) < 10:
            pixels = img_small.reshape(-1, 3)
        pixels = np.float32(pixels)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        unique, counts = np.unique(labels, return_counts=True)
        dominant_idx = unique[np.argmax(counts)]
        dominant_color = centers[dominant_idx]

        return np.uint8(dominant_color)

    def get_color_hist(self, cv2_img: np.ndarray) -> np.ndarray:
        """
        Извлекает гистограмму цвета с использованием доминантного цвета.
        """
        dominant_color = self.get_dominant_color(cv2_img)
        hsv = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2HSV)
        dominant_hsv = cv2.cvtColor(np.uint8([[dominant_color]]), cv2.COLOR_BGR2HSV)[0][0]
        hue = dominant_hsv[0]
        tolerance = 20
        if hue - tolerance < 0:
            mask1 = cv2.inRange(hsv, np.array([0, 50, 40]), np.array([hue + tolerance, 255, 255]))
            mask2 = cv2.inRange(hsv, np.array([hue - tolerance + 180, 50, 40]), np.array([180, 255, 255]))
            mask = cv2.bitwise_or(mask1, mask2)
        elif hue + tolerance > 180:
            mask1 = cv2.inRange(hsv, np.array([hue - tolerance, 50, 40]), np.array([180, 255, 255]))
            mask2 = cv2.inRange(hsv, np.array([0, 50, 40]), np.array([hue + tolerance - 180, 255, 255]))
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            mask = cv2.inRange(hsv, np.array([hue - tolerance, 50, 40]),
                               np.array([hue + tolerance, 255, 255]))
        hist = cv2.calcHist([hsv], [0, 1], mask, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=1, beta=0, norm_type=cv2.NORM_L1)

        return hist


    def add_new_target(self, frame: np.ndarray, display_name: str) -> Optional[str]:
        """
        Добавляет новую деталь в базу на основе загруженного фото.
        Сначала пытается найти объект детектором и вырезать его.

        Args:
            frame (numpy.ndarray): Кадр с деталью.
            display_name (str): Имя детали, которое ввел пользователь.

        Returns:
            bool: True, если деталь успешно добавлена.

        """
        h, w = frame.shape[:2]
        if h * w > 640 * 480:
            scale = np.sqrt(640 * 480 / (h * w))
            new_size = (int(w * scale), int(h * scale))
            frame_small = cv2.resize(frame, new_size, interpolation=cv2.INTER_LINEAR)
        else:
            frame_small = frame

        res = self.detector(frame_small, conf=0.4, verbose=False)
        if len(res) > 0 and len(res[0].boxes) > 0:
            b = res[0].boxes.xyxy[0].cpu().numpy().astype(int)
            if h * w > 640 * 480:
                b = (b / scale).astype(int)
            crop = frame[b[1]:b[3], b[0]:b[2]]
        else:
            crop = frame

        vector = self.get_vector(crop)
        hist = self.get_color_hist(crop)

        img_path = self.storage.save_reference(display_name, crop, vector, hist)
        return img_path

    def switch_target(self, safe_name: str) -> bool:
        """
        Переключает детектор на поиск новой детали из базы.
        Загружает признаки детали из хранилища и сохраняет их в атрибуты класса.

        Args:
            safe_name (str): Безопасное имя детали.

        Returns:
            bool: True, если деталь найдена и загружена.
        """
        data = self.storage.load_reference(safe_name)

        if data:
            self.target_vector = data['vector']
            self.target_color_hist = data['hist']
            self.current_target_name = data['name']
            self.current_safe_name = safe_name
            self.vector_cache.clear()
            self.feature_cache.clear()
            return True
        return False

    def delete_target(self, safe_name: str) -> bool:
        """
        Удаляет деталь и сбрасывает текущую цель, если нужно

        Args:
            safe_name (str): Безопасное имя детали.

        Returns:
            bool: True если деталь удалилась
        """
        success = self.storage.delete_reference(safe_name)

        if success:
            if safe_name == self.current_safe_name:
                self.target_vector = None
                self.target_color_hist = None
                self.current_target_name = ""
                self.current_safe_name = ""
                self.vector_cache.clear()
                self.feature_cache.clear()
        return success

    def _score_to_bgr(self, score: float) -> Tuple[int, int, int]:
        """
        Преобразует значение уверенности [0..1] в цвет BGR.
        Интерполирует от красного (низкий) через желтый к зелёному (высокий).
        """
        s = float(np.clip(score, 0.0, 1.0))
        if s <= 0.5:
            t = s / 0.5
            r = int(255 * (1 - t) + 255 * t)
            g = int(0 * (1 - t) + 255 * t)
            b = 0
        else:
            t = (s - 0.5) / 0.5
            r = int(255 * (1 - t) + 0 * t)
            g = 255
            b = 0
        return (b, g, r)

    def _process_single_box(self, box: Any, frame: np.ndarray,
                            target_vector: np.ndarray,
                            target_color_hist: np.ndarray,
                            threshold: float, tid: int, cur_vec: np.ndarray, cur_hist: np.ndarray) -> Optional[Dict]:
        """
        Обработка одного бокса (возвращает отдельные компоненты уверенности).
        """
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        try:
            yolo_conf = float(box.conf)
        except Exception:
            yolo_conf = 1.0

        if (y2 - y1) < 10 or (x2 - x1) < 10:
            return None

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        shape_sim = 1.0 - cosine(target_vector, cur_vec)
        shape_sim = float(np.clip(shape_sim, 0.0, 1.0))

        color_sim = cv2.compareHist(target_color_hist, cur_hist, cv2.HISTCMP_CORREL)
        color_sim = (color_sim + 1.0) / 2.0
        color_sim = float(np.clip(color_sim, 0.0, 1.0))
        
        final_sim = (shape_sim + color_sim) / 2
        
        final_sim = float(np.clip(final_sim, 0.0, 1.0))

        print(f"[LOG] YOLO: {yolo_conf:.2f} | Shape (Vec+Geom): {shape_sim:.2f} | Color: {color_sim:.2f} => TOTAL: {final_sim:.3f}")

        if yolo_conf < 0.4:
            return None

        if final_sim >= threshold:
            return {
                'coords': (x1, y1, x2, y2),
                'score': final_sim,
                'name': self.current_target_name
            }
        return None

    def draw_label(self, frame: np.ndarray, text: str, x: int, y: int) -> np.ndarray:
        """
        Рисует читаемую подпись с поддержкой русского языка.
        Динамически подбирает размер шрифта и рисует полупрозрачный фон + обводку.
        """
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
        font_size = 100

        if self.base_font_path:
            try:
                font = ImageFont.truetype(self.base_font_path, font_size)
            except Exception:
                font = ImageFont.load_default()
        else:
            font = ImageFont.load_default()

        overlay = Image.new("RGBA", img_pil.size, (255, 255, 255, 0))
        draw_overlay = ImageDraw.Draw(overlay)

        bbox = draw_overlay.textbbox((0, 0), text, font=font, stroke_width=max(1, font_size // 18))
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        padding = max(6, int(font_size * 0.3))

        y_top = max(y - text_h - padding * 2, 0)

        rect_x0 = x
        rect_y0 = y_top
        rect_x1 = min(img_pil.size[0], x + text_w + padding * 2)
        rect_y1 = min(img_pil.size[1], y)

        bg_fill = (0, 160, 0, 200)
        draw_overlay.rectangle([rect_x0, rect_y0, rect_x1, rect_y1], fill=bg_fill, outline=None)

        composed = Image.alpha_composite(img_pil, overlay)

        draw_final = ImageDraw.Draw(composed)
        text_pos = (x + padding, rect_y0 + padding // 2)
        stroke_w = max(1, font_size // 18)

        draw_final.text(text_pos, text, font=font, fill=(255, 255, 255, 255),
                        stroke_width=stroke_w, stroke_fill=(0, 0, 0, 255))

        result = cv2.cvtColor(np.array(composed.convert("RGB")), cv2.COLOR_RGB2BGR)
        return result

    def _draw_label(self, frame: np.ndarray, text: str, x: int, y: int,
                    bg_color: Tuple[int, int, int] = (0, 200, 0), alpha: float = 0.6) -> np.ndarray:
        """
        Рисует полупрозрачный зелёный фон и белый текст (поддерживается кириллица).
        frame: входной кадр в формате BGR (OpenCV)
        text: строка для вывода
        x, y: левый нижний угол (x, y) рамки — текст будет размещён над этим y
        bg_color: фон (B, G, R)
        alpha: прозрачность фона [0..1]
        """
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb).convert("RGBA")

        overlay = Image.new("RGBA", img_pil.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)

        font_size = max(18, frame.shape[0] // 40)
        try:
            font = ImageFont.truetype(self.base_font_path, font_size)
        except Exception:
            font = ImageFont.load_default()

        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        pad_x = 6
        pad_y = 4

        rect_left = x
        rect_right = x + tw + 2 * pad_x
        rect_bottom = y
        rect_top = y - th - 2 * pad_y

        if rect_top < 0:
            rect_top = y
            rect_bottom = y + th + 2 * pad_y

        bg_r, bg_g, bg_b = int(bg_color[2]), int(bg_color[1]), int(bg_color[0])
        alpha_byte = int(255 * float(np.clip(alpha, 0.0, 1.0)))

        draw.rectangle([rect_left, rect_top, rect_right, rect_bottom],
                       fill=(bg_r, bg_g, bg_b, alpha_byte))

        text_color = (255, 255, 255, 255)
        text_x = rect_left + pad_x
        text_y = rect_top + pad_y
        draw.text((text_x, text_y), text, font=font, fill=text_color)

        out = Image.alpha_composite(img_pil, overlay).convert("RGB")
        out_np = np.array(out)
        out_bgr = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)
        return out_bgr

    def process_frame(self, frame: np.ndarray, threshold_percent: int = 70) -> np.ndarray:
        """
        Главный метод обработки кадра.

        Args:
            frame (numpy.ndarray): Исходный кадр с камеры.
            threshold_percent (int): Порог уверенности в процентах (0-100).

        Returns:
            numpy.ndarray: Кадр с нарисованными рамками.
        """
        self.frame_counter += 1

        if self.target_vector is None:
            return frame

        threshold = threshold_percent / 100.0

        h, w = frame.shape[:2]
        if h * w > 1280 * 720:
            scale = np.sqrt(1280 * 720 / (h * w))
            new_size = (int(w * scale), int(h * scale))
            frame_small = cv2.resize(frame, new_size, interpolation=cv2.INTER_LINEAR)
        else:
            frame_small = frame
            scale = 1.0

        results = list(self.detector.track(frame_small, conf=0.25, persist=True,
                                           verbose=False, stream=True))

        detected_boxes: List[Dict] = []

        if results:
            res = results[0]
            if res.boxes:
                boxes = res.boxes
                if boxes.id is not None:
                    tids = boxes.id.cpu().numpy().astype(int)
                else:
                    tids = np.arange(len(boxes))

                target_vector = self.target_vector
                target_color_hist = self.target_color_hist

                new_rois = []
                new_tids = []
                vecs = [None] * len(boxes)
                hists = [None] * len(boxes)

                for i, tid in enumerate(tids):
                    if tid in self.feature_cache and self.frame_counter % self.recompute_interval != 0:
                        vecs[i], hists[i] = self.feature_cache[tid]
                    else:
                        x1, y1, x2, y2 = map(int, boxes[i].xyxy[0])
                        pad = 8
                        x1p = max(0, x1 - pad); y1p = max(0, y1 - pad)
                        x2p = min(frame_small.shape[1], x2 + pad); y2p = min(frame_small.shape[0], y2 + pad)
                        roi = frame_small[y1p:y2p, x1p:x2p]
                        if roi.size == 0:
                            vecs[i] = None
                            hists[i] = None
                            continue
                        new_rois.append(roi)
                        new_tids.append(tid)

                if new_rois:
                    new_vecs = self.batch_get_vectors(new_rois)
                    for i, tid in enumerate(new_tids):
                        vec = new_vecs[i]
                        hist = self.get_color_hist(new_rois[i])
                        self.feature_cache[tid] = (vec, hist)
                        idx = list(tids).index(tid)
                        vecs[idx] = vec
                        hists[idx] = hist

                for i, box in enumerate(boxes):
                    tid = tids[i]
                    if vecs[i] is None or hists[i] is None:
                        continue
                    result = self._process_single_box(box, frame_small,
                                                      target_vector,
                                                      target_color_hist,
                                                      threshold, tid, vecs[i], hists[i])
                    if result:
                        x1, y1, x2, y2 = result['coords']
                        if scale != 1.0:
                            x1, y1, x2, y2 = [int(c / scale) for c in [x1, y1, x2, y2]]
                        detected_boxes.append({
                            'coords': (x1, y1, x2, y2),
                            'score': result['score'],
                            'name': result['name']
                        })

                current_tids = set(tids)
                self.feature_cache = {k: v for k, v in self.feature_cache.items() if k in current_tids}

        for det in detected_boxes:
            x1, y1, x2, y2 = det['coords']
            score = det['score']
            name = det['name']

            conf_percent = int(round(score * 100))

            label = f"{name} {conf_percent}%"

            box_color = self._score_to_bgr(score)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            frame = self._draw_label(frame, label, x1, y1, bg_color=box_color, alpha=0.7)

        return frame