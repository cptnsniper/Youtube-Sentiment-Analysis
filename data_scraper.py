#!/usr/bin/env python3
"""
youtube_stratified_no_api_rate_limit.py

1) Stratified sampling of YouTube videos by view-count tier  
2) Uses yt-dlp to search & scrape metadata (no API key)  
3) Uses youtube-transcript-api for transcripts  
4) Skips purely visual videos (no transcript)  
5) Handles YouTube rate-limiting by backing off and throttling
6) Dumps results to CSV for downstream NLP/stats
"""

import random
import csv
import time
import datetime

from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError
from youtube_transcript_api import YouTubeTranscriptApi, _errors

# ─── CONFIG ───────────────────────────────────────────────────────────────────
TARGET_PER_TIER = 50
COMMON_QUERIES  = [
    'the','and','vlog','news','how','review',
    'music','fun','daily','life'
]
OUTPUT_CSV = 'youtube_stratified_sample.csv'

# (tier_name, min_views, max_views)
TIERS = [
    ('viral',   1_000_000, float('inf')),
    ('popular',   100_000, 1_000_000),
    ('mid',        10_000,   100_000),
    ('niche',           0,    10_000),
]

# ─── YT-DLP SETUP ──────────────────────────────────────────────────────────────
ydl_opts = {
    'quiet': True,
    'skip_download': True,
    'extract_flat': False,
    'nocheckcertificate': True,
    # throttle bytes/sec to avoid being blocked
    'ratelimit': 500_000,
}
ydl = YoutubeDL(ydl_opts)


# ─── UTILITIES ─────────────────────────────────────────────────────────────────

def timestamp():
    """Return current time for debug prints."""
    return datetime.datetime.now().isoformat(sep=' ', timespec='seconds')


def fetch_transcript(video_id):
    """
    Fetch and concatenate auto-generated subtitles.
    Raises _errors.TranscriptsDisabled if no transcript,
    or ValueError if transcript is empty.
    """
    segs = YouTubeTranscriptApi.get_transcript(video_id)
    text = ' '.join(s['text'] for s in segs).strip()
    if not text:
        raise ValueError("empty transcript")
    return text


# ─── CORE SAMPLING FUNCTION ─────────────────────────────────────────────────────

def search_and_filter(min_views, max_views, needed):
    """
    Fill `needed` entries in the view-count range [min_views, max_views)
    by:
      - picking random seed queries
      - scraping ytsearch50:<query> via yt-dlp
      - filtering by view_count & transcript availability
      - backing off & throttling on rate limits
    """
    collected = {}
    tries = 0

    print(f"[{timestamp()}] START tier search: views ∈ [{min_views}, {max_views}), need {needed}")

    while len(collected) < needed:
        tries += 1
        q = random.choice(COMMON_QUERIES)
        print(f"[{timestamp()}] Iteration {tries}: searching 'ytsearch50:{q}'")

        # 1) attempt to extract search results, catch rate-limits
        try:
            info = ydl.extract_info(f"ytsearch50:{q}", download=False)
        except DownloadError as e:
            msg = str(e).lower()
            if 'rate-limited' in msg:
                print(f"[{timestamp()}] RATE-LIMITED by YouTube; sleeping 60s before retry")
                time.sleep(60)
            else:
                print(f"[{timestamp()}] DownloadError: {e}; sleeping 10s before retry")
                time.sleep(10)
            continue

        entries = info.get('entries', [])
        print(f"[{timestamp()}] Retrieved {len(entries)} entries, filtering…")

        # 2) filter entries
        for e in entries:
            vid   = e.get('id')
            views = e.get('view_count') or 0

            if vid in collected:
                continue
            if not (min_views <= views < max_views):
                continue

            # 3) check transcript
            try:
                transcript = fetch_transcript(vid)
            except _errors.TranscriptsDisabled:
                # no transcript available
                continue
            except Exception as ex:
                # empty transcript or other error
                print(f"[{timestamp()}] Skipping {vid}: transcript error ({ex})")
                continue

            # 4) accept video
            collected[vid] = {
                'video_id':     vid,
                'views':        views,
                'likes':        e.get('like_count', 0) or 0,
                'comments':     e.get('comment_count', 0) or 0,
                'title':        e.get('title', ''),
                'published_at': e.get('upload_date', ''),
                'transcript':   transcript,
            }
            print(f"[{timestamp()}] Accepted {vid} (views={views}) — {len(collected)}/{needed}")

            if len(collected) >= needed:
                break

        # 5) throttle between iterations to avoid hammering
        pause = random.uniform(1.0, 3.0)
        print(f"[{timestamp()}] Sleeping {pause:.1f}s before next iteration…")
        time.sleep(pause)

    print(f"[{timestamp()}] COMPLETED tier: collected {len(collected)} videos\n")
    return list(collected.values())


# ─── MAIN PIPELINE ────────────────────────────────────────────────────────────

def main():
    all_videos = []

    for name, lo, hi in TIERS:
        print(f"[{timestamp()}] ===== Sampling tier '{name}' =====")
        vids = search_and_filter(lo, hi, TARGET_PER_TIER)
        all_videos.extend(vids)

    # write results to CSV
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'video_id','views','likes','comments',
            'title','published_at','transcript'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for v in all_videos:
            writer.writerow(v)

    print(f"[{timestamp()}] DONE! Wrote {len(all_videos)} videos to '{OUTPUT_CSV}'")


if __name__ == '__main__':
    main()