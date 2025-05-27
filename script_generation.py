from tqdm import tqdm
from llm_utils import *
from utils import *


with open("INFO_EXTRACTED.txt", "r") as f:
    INFO_EXTRACTED = f.read()

with open("prompts/guide_gen_sys.txt", "r") as f:
    guide_sys = f.read()

guide_prompt = f"""{guide_sys}\n\nINFORMATION EXTRACTED:\n\n{INFO_EXTRACTED}"""
guide_str = get_gemini_response(guide_prompt)

with open("guide_str_gemini.txt", "w") as f:
    f.write(guide_str)
