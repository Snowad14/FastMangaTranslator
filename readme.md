# Ultra Fast Manga translator 📖
 
## Demo 📽️

![Demo of the project](assets/demo.gif)

️🚨 Japanese isn't supported, as it would require quite a lot of code modification, but above all because no LLM is good enough for it to be interesting, so it's a translator that aims to translate manga already translated mainly in English into other languages.  **Note :** The project should also works for webtoons

## Requirements 📋

- Nvidia GPU supporting *TensorRT-LLM*
- **8G** of VRAM is the absolute minimum (you'll have to use a very small model like phi-2, which is very bad at translation so it's more like **10G** to use bigger quantized models) 

## Installation ⚙️ 

- Install TensorRT-LLM for Windows from [tensorRT-LLM github repo](https://github.com/NVIDIA/TensorRT-LLM/tree/rel/windows)
-  Install requirements with :
```bash
# Install requirements
pip install -r requirements.txt
```
-  Download models from [this link](https://huggingface.co/Snowad/animeTagger/resolve/main/models.zip) and put them in the models/ folder
- The Bubble detection model uses TensorRT, but the conversion is done automatically.

## Setup 🔧

This is the longest part, you'll need to find an LLM to translate it. I'm using CroissantLLMChat-v0.1 for the demo, which translates from English to French, but it's not very good.

So I'd advise you to opt for a quantized 4bit AWQ model, even if it's a bit complicated to do. I recommend [OpenBuddy Mistral 7B](https://huggingface.co/OpenBuddy/openbuddy-mistral-7b-v17.1-32k) if you have a small GPU, and if you have a better one, [OpenBuddy Mistral 7Bx8](https://huggingface.co/OpenBuddy/openbuddy-mixtral-7bx8-v18.1-32k) or [OpenBuddy DeepSeek 67B](https://huggingface.co/OpenBuddy/openbuddy-deepseek-67b-v15.3-4k). They are good multilingual models in many languages

It's best to use an LLM, because OCR isn't perfect, so it needs to be able to understand the sentence and translate it even if some characters are missing, which is difficult for enc-dec models to do.

Once you have obtained your model, you need to add it to the `config.json` file :

- Set the source language of your manga in `source_lang`
- Set the desired target language of your manga in `target_lang`
- Set the path of your font (you need to download one) in `font_path`
- Edit the prompt in `prompt` to match with your target language, you can experiments different prompts by running `translator.py`
- Set `trt_engine_path` (folder path), `trt_engine_name` (.engine path) and `tokenizer_dir_path` (folder path)

## Usage 🏃

- To start the gradio demo, `python demo.py`
- To translate all images of a folder, edit `folder_to_translate` in the config and then run `python main.py`

## Todo 📝

- [ ]  Better speech bubble postprocessing (for larger text rendering)
- [ ]  Speeding up inpainting

## Acknowledgements

There are many other manga translation projects, but this one focuses on a quick translation, using the yolo model from [this page hugging face](https://huggingface.co/ogkalu/comic-speech-bubble-detector-yolov8m).

## Gen AI on RTX PCs Developer Contest Entry details

This project is a submission for the NVIDIA RTX PCs Developer Contest, under the General Generative AI Projects category.

Category: [General Generative AI Projects category](https://www.nvidia.com/en-us/ai-data-science/generative-ai/rtx-developer-contest)

**Tested on following system:**
- Operating System: Windows 11
  - Version: 23H2
  - OS Build: 10.0.22631
- TensorRT-LLM version: 0.7.1
  - CUDA version: 12.2
  - cuDNN version: 8.9.7.29 
  - GPU: NVIDIA RTX 3090
  - Driver version: 537.13
  - DataType: FP16
  - Python version: 3.10.13
  - PyTorch version: 2.1.0+cu121
