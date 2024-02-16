from paddleocr import PaddleOCR
import numpy as np

class PaddleOCRModel:
    def __init__(self, lang='en', use_gpu=True):
        self.ocr = PaddleOCR(lang=lang, use_gpu=use_gpu)

    def process(self, image_content, topleft):
        results = self.ocr.ocr(image_content)

        if not results or not results[0]:
            return "", None

        lines = results[0]
        sentence = " ".join(line[1][0] for line in lines)

        coordinates = np.array([line[0] for line in lines])
        x1s, y1s = coordinates[:, 0, :].min(axis=0)
        x2s, y2s = coordinates[:, 2, :].max(axis=0)

        x1s += topleft[0]
        y1s += topleft[1]
        x2s += topleft[0]
        y2s += topleft[1]

        overall_bbox = int(x1s), int(y1s), int(x2s), int(y2s)
        
        return sentence, overall_bbox
        
if __name__ == "__main__":
    ocr = PaddleOCRModel(lang="fr")
    images = ["images/test_ocr.png", "images/test_ocr2.png", "images/test_ocr3.png"]
    for img in images:
        sentence , _ = ocr.process(img, (0, 0))
        print(sentence)


    