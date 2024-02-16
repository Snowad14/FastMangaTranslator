import torch, json, googletrans, re
from tensorrt_llm.runtime import ModelRunner
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional
from pathlib import Path
from trt_llama_api import TrtLlmAPI, ChatMessage, MessageRole, messages_to_prompt, completion_to_prompt

class TensorRT_LLM_Translator:
    def __init__(self, trt_engine_path, trt_engine_name, tokenizer_dir_path, max_output_tokens, max_input_tokens, custom_prompt, temperature=0.3):
        self.llm = TrtLlmAPI(
            model_path=trt_engine_path,
            engine_name=trt_engine_name,
            tokenizer_dir=tokenizer_dir_path,
            temperature=temperature,
            max_new_tokens=max_output_tokens,
            context_window=max_input_tokens,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            verbose=False
        )

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir_path)
        self.translate_prompt = custom_prompt

    def format_input(self, sentences: list) -> str:
        prompt = ["\n"]
        index = 1
        for line in sentences:
            if line:
                prompt.append(f"{index}. {line}\n\n")
                index += 1
        return "".join(prompt)
    
    def format_output(self, response: str) -> list:
        regex_pattern = r'^\d+\.\s(.+)$'
        matches = re.findall(regex_pattern, response, re.MULTILINE)
        return [i.strip() for i in matches if i]

    def translate_list(self, to_translate):
        chunk = self.format_input(to_translate)
        chat = [
            {"role": "user", "content": self.translate_prompt},
            {"role": "user", "content": chunk}
        ]
        prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        response = self.llm.complete_common(prompt, chat=False, formatted=True)
        text_only = json.loads(response)["choices"][0]["text"]
        return self.format_output(text_only)


if __name__ == "__main__":
    translator = TensorRT_LLM_Translator(
        trt_engine_path='models/croissant_engine',
        trt_engine_name='llama_bfloat16_tp1_rank0.engine',
        tokenizer_dir_path='models/croissant_engine/tokenizer',
        max_output_tokens=150,
        max_input_tokens=150,
        custom_prompt="Tu es traducteur professionnel, traduis ce texte en français.",
        temperature=0.3
    )

    test_trans = """
    1. WHAT DO YOU THINK YOU'RE DOING? \n
    2. YOU TWO\n
    3. WHAT A SPLENDID DISPLAY OF POWER WE JUST WITNESSED, STUDENTS.\n
    4. NOW IT'S TIME FOR INDIVIDUAL TRAINING\n
    5. HE'S OBVIOUSlY CHEATING\n
    6. IMPOSSIBLE!\n
    7. I HAVE ZERO MOTIVATION...\n
    8. HE'S JU5T A PLEB\n
    9. ALRIGHT EVERYONE\n
    10. WHATREMAINS OF THE TARGET\n
    """

    base3 = ['NO,NOT AT ALL.', 'AND ALWAYS MAKE SURETO DRAW THE EYE- BAGS.', 'EYEBAGS.. ARE THOSE BULGES BELOW YOUR EYES, RIGHT?', 'YEAH.', 'IDOMy MAKEUP IN A WAyTHAT MAKES MY EYESLOOK BIGGER.', 'IGUESS THE EYES.', 'IKNOW! THEY GOTTA BE SUPER HEAVY!']

    translation = translator.translate_list(base3)
    print(translation)
