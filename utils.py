from pathlib import Path
from typing import Iterable, List
import yt_dlp as ydlp
import whisper
from tqdm import tqdm


# ────────────────────────────────────────────────
# 1.  Download helpers
# ────────────────────────────────────────────────
def download_as_mp3(url: str, out_dir: Path = Path("yt_vids")) -> Path:
    """
    Download *one* YouTube video and return the Path to the resulting .mp3.

    Raises FileNotFoundError or yt_dlp.utils.DownloadError on failure.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        # best available audio, fallback to best video+audio if necessary
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / "%(title)s.%(ext)s"),   # temp name before post-proc
        "noplaylist": True,
        "quiet": True,       # suppress console spam; remove if you want progress
        #
        # Post-processors convert to MP3 and embed metadata/thumbnail
        #
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "0",   # “0” ⇒ ~320 kbps
            },
            {"key": "EmbedThumbnail"},
            {"key": "FFmpegMetadata"},
        ],
    }

    with ydlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

    # yt-dlp gives us the *pre-post-processing* filename; change extension
    mp3_path = Path(ydl.prepare_filename(info)).with_suffix(".mp3")
    if not mp3_path.exists():
        raise FileNotFoundError(f"Expected {mp3_path} but it does not exist")
    return mp3_path


def batch_download(urls: Iterable[str], out_dir: Path = Path("yt_vids")) -> List[Path]:
    """Download many URLs, showing a nice progress bar."""
    return [download_as_mp3(u, out_dir) for u in tqdm(list(urls), desc="🎵  Downloading")]


# ────────────────────────────────────────────────
# 2.  Transcription helpers
# ────────────────────────────────────────────────
def transcribe_audio(mp3_path: Path, model) -> str:
    """Return plain-text transcription for a single MP3 file."""
    result = model.transcribe(str(mp3_path), verbose=False)
    return result["text"]


def batch_transcribe(mp3_paths: Iterable[Path], model) -> str:
    """Transcribe many MP3s and concatenate them with headers."""
    transcripts = ""
    for idx, p in enumerate(tqdm(list(mp3_paths), desc="✍️  Transcribing"), 1):
        text = transcribe_audio(p, model)
        transcripts += f"\n\nVideo Transcript #{idx}\n\n{text}"
    return transcripts.strip()


