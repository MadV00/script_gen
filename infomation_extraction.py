from tqdm import tqdm
from llm_utils import *
from utils import *
import whisper



with open("prompts/info_extraction_sys.txt", "r") as f:
    info_extraction_sys = f.read()

yt_urls = [
    "https://youtu.be/dTBduQ-gRL8",
    "https://youtu.be/K3N5o_qvlEk",
    "https://youtu.be/hMJ9f0IIHtk"
]
model_name = "large-v3"
vid_title = "Next Video : Why You Wouldn't Last a Day in Pre Historic Australia"


# download all vids to "yt_vids"
audio_files = batch_download(yt_urls, Path("yt_vids"))

# transcribe
model = whisper.load_model(model_name, device="cuda")
full_transcript = batch_transcribe(audio_files, model)
transcripts = full_transcript.strip()

out_path = Path("combined_transcripts.txt")
out_path.write_text(transcripts, encoding="utf-8")

info_prompt = info_extraction_sys.replace("YOUTUBE_TRANSCRIPTS_REPLACE_MEE", transcripts)
INFO_str = get_openai_response(info_prompt)

out_path = Path("INFO_EXTRACTED.txt")
out_path.write_text(INFO_str, encoding="utf-8")
