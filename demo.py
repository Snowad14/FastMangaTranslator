import gradio as gr
from PIL import Image
import time
from main import process_image

def process_and_show(input_image, use_inpainting):
    print(use_inpainting)
    start_time = time.time()
    original_image = Image.fromarray(input_image) if not isinstance(input_image, Image.Image) else input_image
    processed_image = process_image(original_image, use_inpainting)
    elapsed_time = time.time() - start_time
    return "{:.2f} seconds".format(elapsed_time), processed_image

def main():
    description = f"""
    <div style="text-align: center; font-size: 30px; font-weight: bold;">
        UltraFast Manga Translator
    </div>
    <div style="text-align: center; font-size: 24px;">
        Maybe not the best quality, the most complete, but the <strong>fastest</strong> manga translator.<br>
        Powered by TensorRT & TensorRT-LLM 🟢
    </div>
    """
    iface = gr.Interface(
        fn=process_and_show,
        inputs=[gr.Image(type="pil", label="Input Image"), gr.Checkbox(label="Use Inpainting")],
        outputs=[gr.components.Textbox(label="Time taken"), gr.Image(type="pil", label="Translated Image")],
        description=description,
        allow_flagging="never",
    )
    
    iface.launch()

if __name__ == "__main__":
    main()