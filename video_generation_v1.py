import os
import re
import time
import torch
import whisper
import requests
import replicate
import subprocess
import numpy as np
from moviepy import *
from tqdm import tqdm
from PIL import Image
from openai import OpenAI
from pathlib import Path
from datetime import timedelta
import bisect, re
import bisect
from pathlib import Path
import numpy as np
from mutagen import File as MutagenFile
from moviepy.video.VideoClip import VideoClip



audio_file_path = r"data/Vid7_pt1.mp3"
transcript_txt_path = audio_file_path.replace(".mp3", "_transcription.txt")
output_dir = r"data/images_vid7_v3"
video_output_path = r"data/final_videos/output_video.mp4"
os.makedirs(output_dir, exist_ok=True)

image_size = "1792x1024"
delay_between_requests = 10
model_name = "base.en"
group_threshold = 20
batch_size = 10

client = OpenAI(
    api_key=API_KEY
)


base_style = """Digital oil painting, hyper-realistic oil-painting emulation, impasto brush strokes, 8 k render down-sampled, cinematic chiaroscuro, single ember/torch/fire key-light (≈2500 K) + warm rim, subsurface scatter on fire-facing skin, deep slate/charcoal shadows, 16 : 9 widescreen low-angle 35-50 mm POV, rugged prehistoric realism, impasto brush texture, fine film grain, atmospheric dust/smoke/haze, accents of ember orange #FF7F24, mild bloom & vignette, brutal survival tension, no modern elements."""


def _t(name):
    match = re.match(r"(\d{2})_(\d{2})_(\d{2})_\d{2}", name)
    if match:
        hh, mm, ss = map(int, match.groups())
        return timedelta(hours=hh, minutes=mm, seconds=ss).total_seconds()
    else:
        raise ValueError(f"Filename {name} doesn't match expected format HH_MM_SS_XX")


def images_to_timed_video_ffmpeg(
    img_folder,
    audio_path,
    output_path,
    fps=30,
    resize_to=(1792, 1024),
):
    img_folder_path = Path(img_folder) # Keep original as Path for consistency
    audio_path_obj = Path(audio_path) # Convert to Path object for consistency
    output_path_obj = Path(output_path) # Convert to Path object for consistency

    images = sorted(
        (
            p
            for p in img_folder_path.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        ),
        key=lambda p: _t(p.stem),
    )
    if not images:
        raise ValueError("No images found.")

    starts = [_t(p.stem) for p in images]
    if starts[0] != 0:
        # Allow for slght floating point inaccuracies if _t can return floats
        if not np.isclose(starts[0], 0):
            raise ValueError(f"First image must start at 00_00_00.* (got {images[0].stem} -> {starts[0]})")


    total_dur = MutagenFile(str(audio_path_obj)).info.length # MutagenFile might prefer string

    frames = [
        np.array(Image.open(p).resize(resize_to, Image.LANCZOS))[:, :, :3]  # ensure RGB
        for p in images
    ]

    def make_frame(t: float) -> np.ndarray:
        idx = bisect.bisect_right(starts, t) - 1
        idx = min(idx, len(frames) - 1)
        return frames[idx]


    temp_video_filename = output_path_obj.stem + "_video_tmp" + output_path_obj.suffix
    temp_video = output_path_obj.with_name(temp_video_filename)

    temp_video.parent.mkdir(parents=True, exist_ok=True)

    video_clip = VideoClip(make_frame, duration=total_dur)

    video_clip.write_videofile(
        str(temp_video), codec="libx264", audio=False, fps=fps,
    )

    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y", # Overwrite output files without asking
        "-i",
        str(temp_video),
        "-i",
        str(audio_path_obj),
        "-c:v",
        "copy", # Copy video stream as is (since it's already encoded)
        "-c:a",
        "aac",  # Re-encode audio to AAC (common for MP4)
        "-map",
        "0:v:0", # Map video from first input
        "-map",
        "1:a:0", # Map audio from second input
        "-shortest", # Finish encoding when the shortest input stream ends
        str(output_path_obj),
    ]
    try:
        print(f"Running FFmpeg command: {' '.join(cmd)}") # For debugging
        subprocess.run(cmd, check=True, capture_output=True, text=True) # capture_output for better error info
    except subprocess.CalledProcessError as e:
        print("FFmpeg merging failed.")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        raise
    
    if temp_video.exists():
        temp_video.unlink()
    print(f"Successfully created video: {output_path_obj}")


def timestamp_to_filename(start_time, index):
    h, m, s = (
        int(start_time // 3600),
        int((start_time % 3600) // 60),
        int(start_time % 60),
    )
    return f"{h:02d}_{m:02d}_{s:02d}_{index:02d}.png"


def write_image_from_url(image_url, output_path):
    response = requests.get(image_url)
    response.raise_for_status()
    with open(output_path, "wb") as f:
        f.write(response.content)


def generate_flux_image(prompt, seed=42):
    output = replicate.run(
        "black-forest-labs/flux-kontext-pro",
        input={"aspect_ratio": "16:9", "prompt": prompt, "seed": seed},
    )
    return output[0] if isinstance(output, list) else output


def get_segments(segments):
    grouped_segments = []
    temp_text = segments[0]["text"]
    group_start = segments[0]["start"]
    group_end = segments[0]["end"]

    for i in range(1, len(segments)):
        curr = segments[i]
        if group_end - group_start < group_threshold:
            temp_text += " " + curr["text"]
            group_end = curr["end"]
        else:
            grouped_segments.append(
                {"start": group_start, "end": group_end, "text": temp_text.strip()}
            )
            temp_text = curr["text"]
            group_start = curr["start"]
            group_end = curr["end"]

    grouped_segments.append(
        {"start": group_start, "end": group_end, "text": temp_text.strip()}
    )
    return grouped_segments


def create_llm_instruction_prompt(text_chunks, n=10):
    prompt_template = f"""You are an AI assistant specialized in creating visual prompts for text-to-image generation models. Your task is to analyze the provided chunk of YouTube script text and generate {n} sequential image prompts.
Instructions for generating the {n} prompts:
Read and Understand: Carefully read the entire provided text chunk to grasp the overall context, narrative flow, and key moments.
Sequential Prompts: The {n} prompts must follow the progression of events, descriptions, or ideas presented in the text chunk. Imagine these images will be displayed one after another as the script is read.
Prompt Structure (for each of the {n} prompts):
Each prompt must be a single sentence.
Each sentence must clearly describe:
1-2 Subjects: Identify the main characters, beings, or even significant objects central to that specific moment in the script. These are not limited to predefined lists; extract them from the text. (e.g., "A weary traveler," "Two children," "A towering ancient tree," "The glowing orb").
1 Emotion: Infer or identify a dominant emotion relevant to the subject(s) and the context of that moment in the script. (e.g., "curious," "frightened," "hopeful," "exhausted," "serene").
1 Action: Describe what the subject(s) are primarily doing or experiencing at that moment. (e.g., "discovering a hidden map," "fleeing from a shadow," "contemplating a vast landscape," "building a makeshift shelter," "sharing a quiet meal").
Conciseness and Clarity: Prompts should be direct and clear, suitable for a text-to-image model. Avoid overly complex sentence structures.
Focus on Visuals: Ensure each prompt describes a visually translatable image scene.
Derive from Text: All elements (subjects, emotions, actions) should be directly inspired by, or inferred from, the content of the provided YouTube script chunk. Do not invent elements not supported by the text.
Output Format:
Provide your response as a numbered list of {n} prompts, with each prompt on a new line. Do not include any other explanatory text or headers, just the {n} prompts.
Example of a single prompt's structure (do not use these specific examples unless the text supports them):
"A cave man hunting a mammoth with a spear." (Subjects: caveman,mammoth; Emotion: tension; Action: hunting)
"Cave man drinking water from a pond." (Subject: cave man; Emotion: thirsty; Action: drinking)

It doesn't matter if the subjects are not human, as long as they are clearly defined (Subjects can be pre hitoric human species or pre historic animal species (choose a specific species without being vague)), relevant to the text and are historically accurate. The prompts subjects and actions should grounded in the script's content and prehistoric realism .
ALL SUBJECTS MUST BE PRE-HISTORIC. NEVER MENTION JUST 'HUMAN' ALWAYS MENTION PRE-HISTORIC/PALEOTLITHIC HUMANS. SAME FOR ANIMALS, ALL ANIMALS MUST BE PRE-HISTORIC.

Make sure the sizes and proportions of the subjects are appropriate for the scene described, and that the actions are plausible within the context of pre historic realism.
Now, analyze the following YouTube script chunk and generate the {n} sequential image prompts in the form of a numbered list, following the above instructions:
[PASTE THE YOUTUBE SCRIPT CHUNK HERE]"""

    return prompt_template.replace("[PASTE THE YOUTUBE SCRIPT CHUNK HERE]", text_chunks)


def generate_prompts_from_text(text, n=10):
    prompt = create_llm_instruction_prompt(text, n=n)
    completion = client.chat.completions.create(
        model="o3",
        messages=[{"role": "user", "content": prompt}],
        # max_tokens=1000
    )
    return [
        line.split(". ", 1)[-1].strip()
        for line in completion.choices[0].message.content.strip().split("\n")
        if line.strip()
    ]


def get_dalle_image(prompt, seed=None):
    image_resp = client.images.generate(
        model="dall-e-3", prompt=prompt, size="1792x1024", quality="standard", n=1
    )
    image_url = image_resp.data[0].url
    return image_url


def main():
    print("Transcribing with Whisper...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(model_name, device=device)
    transcription = model.transcribe(audio_file_path, verbose=False)

    with open(transcript_txt_path, "w", encoding="utf-8") as f:
        f.write(transcription["text"])

    segments = transcription["segments"]
    grouped_segments = get_segments(segments)
    print(f"Grouped segments: {len(grouped_segments)}")

    with open("seg_with_prompt.txt", "w", encoding="utf-8") as log_f:
        for i in tqdm(range(0, len(grouped_segments), batch_size)):
            chunk_group = grouped_segments[i : i + batch_size]
            if len(chunk_group) < batch_size:
                break

            combined_text = ""
            for idx, seg in enumerate(chunk_group):
                combined_text += f"{idx+1}. {seg['text']}\n\n"

            visual_prompts = generate_prompts_from_text(
                combined_text, n=len(chunk_group)
            )
            assert len(visual_prompts) == batch_size

            for j, (seg, visual_prompt) in enumerate(zip(chunk_group, visual_prompts)):
                image_prompt = f"{visual_prompt}, {base_style}"
                try:
                    image_url = get_dalle_image(image_prompt, seed=42)

                    filename = timestamp_to_filename(seg["start"], i + j)
                    image_path = os.path.join(output_dir, filename)

                    write_image_from_url(image_url, image_path)
                    print(f"Saved: {image_path}")

                    log_f.write(
                        f"{seg['start']}-{seg['end']}: {seg['text']}\nPrompt: {visual_prompt}\n\n"
                    )
                    log_f.flush()

                except Exception as err:
                    print(f"Error in batch starting at segment {i}: {err}")
                    continue

                time.sleep(delay_between_requests)

    images_to_timed_video_ffmpeg(
        img_folder=output_dir, output_path=video_output_path, audio_path=audio_file_path
    )


if __name__ == "__main__":
    main()
