import cv2
import numpy as np
from ultralytics import YOLO
from ..utils import LegoStorage
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict
from typing import Optional, Tuple, Dict, Any, List
from queue import Queue
import torch


class LegoDetector:
    """
    Основной класс компьютерного зрения
    """

    def __init__(self, model_path: str = 'models/lego_detector200.pt') -> None:
        """
        Инициализация детектора.

        Args:
            model_path: Путь к предобученной модели.
        """
        self.storage: LegoStorage = LegoStorage()

        self.device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.detector: YOLO = YOLO(model_path)
        if self.device.type == 'cuda':
            self.detector.to('cuda')

        self.trackers: defaultdict = defaultdict(dict)
        self.track_id: int = 0

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


    def remove_background(self, cv2_img: np.ndarray) -> np.ndarray:
        """
        Обрезает 10% с каждой стороны изображения.
        """
        h, w = cv2_img.shape[:2]
    
        m_h = int(h * 0.1)
        m_w = int(w * 0.1)
    
        cropped = cv2_img[m_h : h - m_h, m_w : w - m_w]
    
        return cropped


    @staticmethod
    def get_color_histogram(cv2_img: np.ndarray) -> np.ndarray:
        """
        Быстрое извлечение гистограммы цвета (оптимизированная версия)
        
        Args:
            cv2_img: Изображение в формате BGR
            
        Returns:
            np.ndarray: Нормализованная гистограмма цвета
        """
        if cv2_img is None or cv2_img.size == 0:
            return np.zeros(8*8*8, dtype=np.float32)
    
        h, w = cv2_img.shape[:2]
        if h * w > 20000:
            scale = np.sqrt(20000 / (h * w))
            new_size = (int(w * scale), int(h * scale))
            small_img = cv2.resize(cv2_img, new_size, interpolation=cv2.INTER_LINEAR)
        else:
            small_img = cv2_img

        rgb = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)

        hist = cv2.calcHist([rgb], [0, 1, 2], None, [6, 6, 6],
                            [0, 256, 0, 256, 0, 256])
    
        cv2.normalize(hist, hist, alpha=1, beta=0, norm_type=cv2.NORM_MINMAX)
    
        return hist.flatten()


    def _clean_cache(self) -> None:
        """Очистка устаревших записей в кэше"""
        if len(self.vector_cache) > self.cache_size:
            items = list(self.vector_cache.items())
            self.vector_cache = dict(items[-self.cache_size // 2:])


    @staticmethod
    def get_vector(cv2_img: np.ndarray) -> np.ndarray:
        if cv2_img is None or cv2_img.size == 0:
            return np.array([0.0], dtype=np.float32)

        h, w = cv2_img.shape[:2]
        if h == 0 or w == 0:
            return np.array([0.0], dtype=np.float32)

        aspect_ratio = max(w, h) / min(w, h)

        return np.array([aspect_ratio], dtype=np.float32)

    def batch_get_vectors(self, rois: List[np.ndarray]) -> List[np.ndarray]:
        """
        Групповая обработка соотношений сторон для нескольких обнаруженных объектов.
        """
        if not rois:
            return []

        vectors = []
        for roi in rois:
            vectors.append(self.get_vector(roi))

        return vectors

    def add_new_target(self, frame: np.ndarray, display_name: str) -> Optional[str]:
        """
        Добавляет новую деталь в базу, сохраняя оригинальный кроп и метаданные.
        """
        h, w = frame.shape[:2]
        if h * w > 640 * 480:
            scale = np.sqrt(640 * 480 / (h * w))
            new_size = (int(w * scale), int(h * scale))
            frame_small = cv2.resize(frame, new_size, interpolation=cv2.INTER_LINEAR)
        else:
            frame_small = frame
            scale = 1.0
    
        res = self.detector(frame_small, conf=0.5, verbose=False)
    
        if len(res) > 0 and len(res[0].boxes) > 0:
            b = res[0].boxes.xyxy[0].cpu().numpy().astype(int)
            if h * w > 640 * 480:
                b = (b / scale).astype(int)
            crop_original = frame[b[1]:b[3], b[0]:b[2]]
        else:
            crop_original = frame
    
        crop_no_bg = self.remove_background(crop_original)

        vector = self.get_vector(crop_original)
        color_hist = self.get_color_histogram(crop_no_bg)
    
        img_path = self.storage.save_reference(display_name, crop_no_bg, vector, color_hist)
        return img_path

    def switch_target(self, safe_name: str) -> bool:
        """
        Переключает детектор на поиск новой детали из базы.
        """
        data = self.storage.load_reference(safe_name)

        if data:
            self.target_vector = data['vector']
            self.target_color_hist = data['color_hist']
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

    def reset_target(self) -> bool:
        """
        Сбрасывает текущую цель поиска.

        Returns:
            bool: True если деталь удалилась
        """

        self.target_vector = None
        self.target_color_hist = None
        self.current_target_name = ""
        self.current_safe_name = ""
        self.vector_cache.clear()
        self.feature_cache.clear()
        return True

    @staticmethod
    def _score_to_bgr(score: float) -> Tuple[int, int, int]:
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
        return b, g, r

    def _process_single_box(self, box: Any, frame: np.ndarray,
                            target_vector: np.ndarray,
                            target_color_hist: np.ndarray,
                            threshold: float, cur_vec: np.ndarray) -> Optional[Dict]:
        """
        Обработка одного бокса с использованием сравнения соотношения сторон.
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

        roi_no_bg = self.remove_background(roi)

        cur_color_hist = self.get_color_histogram(roi_no_bg)

        target_aspect = target_vector[0]
        current_aspect = cur_vec[0]

        if target_aspect > 0:
            aspect_ratio = min(target_aspect, current_aspect) / max(target_aspect, current_aspect)
            shape_sim = float(aspect_ratio)
        else:
            shape_sim = 0.0

        dist = cv2.compareHist(target_color_hist, cur_color_hist, cv2.HISTCMP_BHATTACHARYYA)
        color_sim = (1 - dist) * 3
        color_sim = float(np.clip(color_sim, 0.0, 1.0))

        final_sim = shape_sim * color_sim
        final_sim = float(np.clip(final_sim, 0.0, 1.0))

        print(f"[LOG] YOLO: {yolo_conf:.3f} | Shape: {shape_sim:.3f} | Color (correlation): {color_sim:.3f} => TOTAL: {final_sim:.3f}")

        if final_sim >= threshold:
            return {
                'coords': (x1, y1, x2, y2),
                'score': final_sim,
                'name': self.current_target_name
            }
        return None

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

        results = list(self.detector.track(
            frame_small,
            conf=0.6,
            persist=True,
            verbose=False,
            stream=True
        ))

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

                        x1p = max(0, x1)
                        y1p = max(0, y1)
                        x2p = min(frame_small.shape[1], x2)
                        y2p = min(frame_small.shape[0], y2)

                        roi = frame_small[y1p:y2p, x1p:x2p]

                        if roi.size == 0:
                            vecs[i] = None
                            hists[i] = None
                            continue

                        new_rois.append(roi)
                        new_tids.append(tid)

                if new_rois:
                    new_vecs = self.batch_get_vectors(new_rois)

                    for j, tid in enumerate(new_tids):
                        roi_no_bg = self.remove_background(new_rois[j])
                        vec = new_vecs[j]
                        color_hist = self.get_color_histogram(roi_no_bg)


                        self.feature_cache[tid] = (vec, color_hist)

                        idx = list(tids).index(tid)

                        vecs[idx] = vec
                        hists[idx] = color_hist

                for i, box in enumerate(boxes):

                    if vecs[i] is None or hists[i] is None:
                        continue

                    result = self._process_single_box(
                        box,
                        frame_small,
                        target_vector,
                        target_color_hist,
                        threshold,
                        vecs[i]
                    )

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
                self.feature_cache = {
                    k: v for k, v in self.feature_cache.items()
                    if k in current_tids
                }

        for det in detected_boxes:
            x1, y1, x2, y2 = det['coords']
            score = det['score']
            name = det['name']

            conf_percent = int(round(score * 100))
            label = f"{name} {conf_percent}%"

            box_color = self._score_to_bgr(score)

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            frame = self._draw_label(
                frame,
                label,
                x1,
                y1,
                bg_color=box_color,
                alpha=0.7
            )

        return frame