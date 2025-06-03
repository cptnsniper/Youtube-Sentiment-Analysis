#!/usr/bin/env python3
"""
youtube_channel_scrape.py

Fetches metadata and transcripts for every video on a specified YouTube channel,
skipping any videos without a transcript. Outputs a CSV with:
  video_id, views, likes, comments, title, published_at, transcript
"""

import csv
import time
import datetime
import os
import re
import random

from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError
from youtube_transcript_api import YouTubeTranscriptApi, _errors

# ─── CONFIG ───────────────────────────────────────────────────────────────────
# You can supply any of these forms here:
#   - "https://www.youtube.com/@SomeHandle"
#   - "https://www.youtube.com/c/CustomName"
#   - "https://www.youtube.com/channel/UCxxxxxxxxxxxx"
#
# The script will normalize it to the channel’s “/videos” page automatically.
CHANNEL_URL = "https://www.youtube.com/@CosmicSkeptic"
OUTPUT_CSV   = "channel_videos.csv"

# Throttle between ytdlp calls to avoid rate-limits
MIN_PAUSE = 1.0  # seconds
MAX_PAUSE = 2.5  # seconds

# ─── YT-DLP INSTANCES ───────────────────────────────────────────────────────────
# 1) ydl_list: for getting the list of all videos on the channel (flat extract)
list_opts = {
    "quiet": True,
    "skip_download": True,
    "extract_flat": "in_playlist",   # ensures we get a “playlist” of uploads
    "nocheckcertificate": True,
    "ratelimit": 500_000,
}
ydl_list = YoutubeDL(list_opts)

# 2) ydl_meta: for fetching full metadata on each video
meta_opts = {
    "quiet": True,
    "skip_download": True,
    "nocheckcertificate": True,
    "ratelimit": 500_000,
}
ydl_meta = YoutubeDL(meta_opts)


# ─── UTILITY FUNCTIONS ─────────────────────────────────────────────────────────

def timestamp():
    return datetime.datetime.now().isoformat(sep=" ", timespec="seconds")


def normalize_to_videos_page(url: str) -> str:
    """
    Given a channel handle/URL, transform it into its “/videos” page.
    - If URL already ends with "/videos", return as-is.
    - Otherwise append "/videos".
    """
    if url.rstrip("/").endswith("/videos"):
        return url
    # If it ends with a known suffix like "/watch", "/playlist", etc., strip and append /videos
    # But generally, appending "/videos" is safe for any of the three forms:
    #   /@handle, /c/CustomName, /channel/UCID
    return url.rstrip("/") + "/videos"


def fetch_transcript(video_id: str) -> str:
    """
    Return the concatenated auto-generated transcript for `video_id`.
    Raises _errors.TranscriptsDisabled if no transcript is available,
    or ValueError if the transcript is empty.
    """
    segs = YouTubeTranscriptApi.get_transcript(video_id)
    text = " ".join(s["text"] for s in segs).strip()
    if not text:
        raise ValueError("empty transcript")
    return text


# ─── MAIN SCRAPING FUNCTION ────────────────────────────────────────────────────

def main():
    if not CHANNEL_URL:
        raise ValueError("Please set CHANNEL_URL to the target channel.")

    # 1) Normalize to the channel’s “/videos” page
    videos_page = normalize_to_videos_page(CHANNEL_URL)
    print(f"[{timestamp()}] Normalized channel URL → {videos_page}")

    # 2) Extract the “playlist” of all uploads on that page
    try:
        channel_info = ydl_list.extract_info(videos_page, download=False)
    except Exception as e:
        print(f"[{timestamp()}] ERROR: could not retrieve channel’s videos list:\n  {e}")
        return

    entries = channel_info.get("entries", [])
    if not entries:
        print(f"[{timestamp()}] No videos found on the channel (empty entries). Exiting.")
        return

    print(f"[{timestamp()}] Found {len(entries)} videos on the channel. Beginning per-video processing…\n")

    # 3) Prepare CSV
    fieldnames = [
        "video_id",
        "views",
        "likes",
        "comments",
        "title",
        "published_at",
        "transcript",
    ]
    os.makedirs(os.path.dirname(OUTPUT_CSV) or ".", exist_ok=True)
    csv_file = open(OUTPUT_CSV, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    processed = 0
    skipped_no_transcript = 0

    # 4) Loop through each entry (should be all video IDs)
    for idx, entry in enumerate(entries, start=1):
        video_id = entry.get("id")
        if not video_id:
            print(f"[{timestamp()}]   Entry #{idx} has no 'id', skipping.")
            continue

        print(f"[{timestamp()}] Processing video {idx}/{len(entries)}: ID={video_id}")

        # a) Fetch transcript (skip if not available)
        try:
            transcript = fetch_transcript(video_id)
        except _errors.TranscriptsDisabled:
            print(f"[{timestamp()}]   → No transcript available; skipping.")
            skipped_no_transcript += 1
            time.sleep(random.uniform(MIN_PAUSE, MAX_PAUSE))
            continue
        except Exception as e:
            print(f"[{timestamp()}]   → Transcript error ({e}); skipping.")
            skipped_no_transcript += 1
            time.sleep(random.uniform(MIN_PAUSE, MAX_PAUSE))
            continue

        # b) Fetch full metadata for the video
        try:
            vid_url = f"https://www.youtube.com/watch?v={video_id}"
            info = ydl_meta.extract_info(vid_url, download=False)
        except DownloadError as e:
            msg = str(e).lower()
            if "rate-limited" in msg:
                print(f"[{timestamp()}]   → Rate-limited fetching metadata; sleeping 60s.")
                time.sleep(60)
                # Retry once
                try:
                    info = ydl_meta.extract_info(vid_url, download=False)
                except Exception as e2:
                    print(f"[{timestamp()}]   → Retry failed ({e2}); skipping video.")
                    skipped_no_transcript += 1
                    time.sleep(random.uniform(MIN_PAUSE, MAX_PAUSE))
                    continue
            else:
                print(f"[{timestamp()}]   → DownloadError fetching metadata ({e}); skipping.")
                skipped_no_transcript += 1
                time.sleep(random.uniform(MIN_PAUSE, MAX_PAUSE))
                continue
        except Exception as e:
            print(f"[{timestamp()}]   → Error fetching metadata ({e}); skipping.")
            skipped_no_transcript += 1
            time.sleep(random.uniform(MIN_PAUSE, MAX_PAUSE))
            continue

        # c) Parse out fields (use 0/defaults if missing)
        views        = info.get("view_count", 0) or 0
        likes        = info.get("like_count", 0) or 0
        comments     = info.get("comment_count", 0) or 0
        title        = info.get("title", "") or ""
        published_at = info.get("upload_date", "") or ""  # typically "YYYYMMDD"

        # d) Write row to CSV
        writer.writerow({
            "video_id":     video_id,
            "views":        views,
            "likes":        likes,
            "comments":     comments,
            "title":        title,
            "published_at": published_at,
            "transcript":   transcript,
        })
        processed += 1

        # e) Throttle before next video
        pause = random.uniform(MIN_PAUSE, MAX_PAUSE)
        time.sleep(pause)

    csv_file.close()
    print(f"\n[{timestamp()}] Finished. Processed {processed} videos with transcripts.")
    print(f"[{timestamp()}] Skipped {skipped_no_transcript} videos due to missing/disabled transcripts.")
    print(f"[{timestamp()}] Output written to '{OUTPUT_CSV}'.\n")


if __name__ == "__main__":
    main()