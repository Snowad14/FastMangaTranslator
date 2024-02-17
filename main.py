import os, gradio, json, cv2, sys, glob
import numpy as np
from PIL import Image, ImageDraw

from detector import Detector
from ocr import PaddleOCRModel
from inpainter import PatchMarkInpainter
from renderer import write_in_bbox
from utils import BoxType, check_collision, right_collision_overlap
from translator import TensorRT_LLM_Translator

# get command line argument
args = json.load(open("config.json"))

# Model configuring & loading
bubble_detector = Detector(
    model_path=f"{args.get('models_dir')}/comic-speech-bubble-detector.pt",
    use_tensorrt=args.get("use_tensorRT_detection")
)

# It could use tensorRT but but the gains would be too small to be worth it.
ocr_model = PaddleOCRModel(
    lang=args.get("source_lang"),
    use_gpu=args.get("use_ocr_gpu")
)

inpainter = PatchMarkInpainter(
    radius=args.get("inpaint_radius")
)

translator = TensorRT_LLM_Translator(
    trt_engine_path=args.get('trt_engine_path'),
    trt_engine_name=args.get('trt_engine_name'),
    tokenizer_dir_path=args.get('tokenizer_dir_path'),
    max_output_tokens=256,
    max_input_tokens=256,
    custom_prompt=args.get("prompt"),
    temperature=0.3
)


def process_image(input_file, use_inpainting=False):
    # PIL open png and convert to jpg
    if type(input_file) == str:
        image = np.array(Image.open(input_file).convert('RGB'))
    else:
        image = np.array(input_file.convert('RGB'))

    verbose_img = image.copy()

    # bubble detection
    bubbles = bubble_detector.predict(input_file)
    big_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    all_texts = []


    for i, bubble in enumerate(bubbles):
        if bubble.type == BoxType.SpeechBox: # often speech bubble are too big
            bubble.resize_bbox(-10)

        # Check collision with other bubbles and resolve them
        for other_bubble in bubbles:
            if other_bubble is not bubble and check_collision(bubble.get_bbox(), other_bubble.get_bbox()):
                overlap = right_collision_overlap(bubble.get_bbox(), other_bubble.get_bbox())
                print(f"Collide between {bubble.text} and {other_bubble.text}")
                print(f"Collide pixel count: {overlap}")

                # Move the bubble by subtracting the overlap amount.
                # check if the bubble is on the left side of the other bubble
                if bubble.get_bbox()[0] < other_bubble.get_bbox()[0]:
                    bubble.x2 -= overlap


        cropped_image = image[bubble.y1:bubble.y2, bubble.x1:bubble.x2]
        bubble.text, bubble.inner_bbox = ocr_model.process(cropped_image, (bubble.x1, bubble.y1))
        all_texts.append(bubble.text)

        if bubble.text and bubble.inner_bbox:
            cv2.rectangle(verbose_img, (bubble.x1, bubble.y1), (bubble.x2, bubble.y2), (0, 255, 0), 2)
            cv2.rectangle(verbose_img, (bubble.inner_bbox[0], bubble.inner_bbox[1]), (bubble.inner_bbox[2], bubble.inner_bbox[3]), (0, 0, 255), 2)
        
        arr = bubble.as_mask(image)
        big_mask = np.logical_or(big_mask, arr).astype(np.uint8)

    big_mask *= 255
    if use_inpainting:
        image = inpainter.inpaint(image, big_mask)
    else:
        image[big_mask == 255] = 255

    pillow_image = Image.fromarray(image)
    all_texts_translated = translator.translate_list(all_texts)

    # rendering
    for index, bubble in enumerate(bubbles):
        if bubble.text and bubble.inner_bbox:
            x1, y1, x2, y2 = bubble.inner_bbox
            if index < len(all_texts_translated):
                bubble.text = all_texts_translated[index]
            write_in_bbox(pillow_image , bubble.text, (x1, y1, x2, y2), args.get("font_path"), use_hyphenator=args.get("use_hyphenator"), hyphenator_lang=args.get("target_lang"))

    
    pillow_image.save("output.jpg")
    cv2.imwrite("mask.jpg", big_mask)
    cv2.imwrite("verbose.jpg", verbose_img)

    return pillow_image

if __name__ == "__main__":
    folder_images = args.get("folder_to_translate")

    if not os.path.exists("results"):
        os.makedirs("results")
    
    files = glob.glob(f"{folder_images}/*.jpg") + glob.glob(f"{folder_images}/*.jpeg") + glob.glob(f"{folder_images}/*.png")
    print("len files", len(files))

    for input_file in files:
        res = process_image(input_file)
        res.save(f"results/{os.path.basename(input_file)}")
        print(f"Image {input_file} processed and saved to results/{os.path.basename(input_file)}")