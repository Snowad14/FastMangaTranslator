from enum import Enum
import numpy as np

lang_abrev = {
    "en": "en_US",
    "fr": "fr_FR",
    "es": "es_ES",
    "de": "de_DE",
    "it": "it_IT",
}

def check_collision(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2
    return (x1 < x4 and x2 > x3) and (y1 < y4 and y2 > y3)


def right_collision_overlap(bbox1, bbox2):
    x2, y2 = bbox1[2:]
    x3 = bbox2[0]
    return max(0, x2 - x3)

class BoxType(Enum):
    SpeechBox = 0
    TextBox = 1

class SpeechBubble:

    def __init__(self, x1, y1, x2, y2, type):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.text = None
        self.type = BoxType(type)
        self.inner_bbox = None
    
    def __repr__(self):
        return f"SpeechBubble(x1={self.x1}, y1={self.y1}, x2={self.x2}, y2={self.y2}, text={self.text})"

    def resize_bbox(self, offset):
        self.x1 -= offset
        self.y1 -= offset
        self.x2 += offset
        self.y2 += offset

    def set_bbox(self, bbox):
        self.x1, self.y1, self.x2, self.y2 = bbox

    def as_mask(self, img):
        height, width = img.shape[:2]
        mask = np.zeros((height, width), dtype=bool)
        if not self.inner_bbox:
            return mask
        inner_x1, inner_y1, inner_x2, inner_y2 = self.inner_bbox
        x1 = max(0, min(inner_x1, width))
        y1 = max(0, min(inner_y1, height))
        x2 = max(0, min(inner_x2, width))
        y2 = max(0, min(inner_y2, height))
        mask[y1:y2, x1:x2] = True
        return mask

    def get_bbox(self):
        return self.x1, self.y1, self.x2, self.y2
