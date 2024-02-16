from PIL import Image, ImageDraw, ImageFont
from hyphen import Hyphenator
from functools import lru_cache
import hyphen.textwrap2 as textwrap
import numpy as np
import cv2, time
from hyphen.dictools import LANGUAGES as HYPHENATOR_LANGUAGES
from utils import lang_abrev

@lru_cache(maxsize=None)
def get_cached_font(font_path, font_size):
    return ImageFont.truetype(font_path, font_size)

def adjust_wrapping(text_list):
    adjusted_list = []
    carry_over = ''
    for text in text_list:
        if carry_over:
            text = carry_over + text
            carry_over = ''
        if text.endswith('-'):
            carry_over = text[-1]
            text = text[:-1]
        adjusted_list.append(text)
    if carry_over:
        adjusted_list.append(carry_over)
    return adjusted_list

def get_avg_char_width(font, text):
    widths = [font.getbbox(char)[2] for char in text if not char.isspace()]
    return sum(widths) / len(widths)

def draw_text_with_autofit(image, txt, xy, area_width, area_height, font_path, min_font_size=5, show_rect=False, use_hyphenator=False, hyphenator_lang="en_US"):
    hyphenator_lang 
    h_fr = Hyphenator(hyphenator_lang)
    x, y = xy
    draw = ImageDraw.Draw(image)
    bounding_box = [x, y, x + area_width, y + area_height]
    
    # Load font with an initial size
    font_size = 35
    font = get_cached_font(font_path, font_size)
    
    # Calculate wrapping width
    avg_char_width = get_avg_char_width(font, txt)
    wrap_at = max(int(area_width // avg_char_width), 1)
    if use_hyphenator:
        wrapped_text = textwrap.fill(txt, wrap_at, use_hyphenator=h_fr, break_long_words=False)
    else:
        wrapped_text = textwrap.fill(txt, wrap_at, break_long_words=False)
    bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    # Decrease the fontsize until the text fits
    while (w > area_width or h > area_height) and font_size > min_font_size:
        font_size -= 1
        font = ImageFont.truetype(font_path, font_size)
        avg_char_width = get_avg_char_width(font, txt)
        wrap_at = max(int(area_width // avg_char_width), 1)
        if use_hyphenator:
            wrapped_text = textwrap.fill(txt, wrap_at, use_hyphenator=h_fr, break_long_words=False)
        else:
            wrapped_text = textwrap.fill(txt, wrap_at, break_long_words=False)
        bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
    
    if use_hyphenator:
        wrapped_text = adjust_wrapping(wrapped_text.split('\n'))
        wrapped_text = '\n'.join(wrapped_text)

    # Calculate the position for centered text
    start_x = x + (area_width - w) / 2
    start_y = y + (area_height - h) / 2
    
    cropped_img = np.array(image.crop(bounding_box))
    gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)

    mean_color_value = np.mean(gray_img)
    pixel_diversity = np.std(cropped_img.flatten())

    font_color = "white" if mean_color_value < 128 else "black"

    if pixel_diversity < 25:
        draw.multiline_text((start_x, start_y), wrapped_text, font=font, fill=font_color, align='center')
    else:
        stroke_width = 2
        stroke_color = "black" if font_color == "white" else "white"
        # decrease the font size to fit the stroke
        font_size -= 4
        font = ImageFont.truetype(font_path, font_size)
        draw.multiline_text((start_x, start_y), wrapped_text, font=font, fill=stroke_color, align='center', stroke_width=stroke_width, stroke_fill=font_color)
    
    if show_rect:
        draw.rectangle((x, y, x + area_width, y + area_height), outline="black")

def write_in_bbox(image, text, bbox, font_path, **kwargs):
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    draw_text_with_autofit(image, text, (x1, y1), width, height, font_path, **kwargs)
