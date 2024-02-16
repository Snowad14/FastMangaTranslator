import cv2
from models.patchmark import inpaint

class PatchMarkInpainter:
    def __init__(self, radius):
        self.radius = radius

    def inpaint(self, img, mask):
        return inpaint(img, mask, patch_size=self.radius)