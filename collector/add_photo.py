#!/usr/bin/env python3
"""Add one of Craig's own photos to the register.

Usage (on the MacBook, from ~/Arcade-Window):
  python3 collector/add_photo.py PHOTO.heic "caption" [BirdName ...]

Resizes the photo with sips, reads the time it was taken, copies it into data/photos/,
appends an entry to data/photos.json, commits and pushes. Bird names are optional and
attach the photo to those birds' cards. The iMac folds these commits into its own pushes.
"""
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHOTOS = os.path.join(ROOT, "data", "photos")
INDEX = os.path.join(ROOT, "data", "photos.json")
TZ = ZoneInfo("Australia/Sydney")


def taken_at(path):
    out = subprocess.run(["sips", "-g", "creation", path], capture_output=True, text=True).stdout
    m = re.search(r"creation:\s*(\d{4}):(\d{2}):(\d{2}) (\d{2}):(\d{2}):(\d{2})", out)
    if m:
        y, mo, d, h, mi, s = map(int, m.groups())
        return datetime(y, mo, d, h, mi, s, tzinfo=TZ)
    return datetime.fromtimestamp(os.path.getmtime(path), TZ)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    src, caption, birds = sys.argv[1], sys.argv[2], sys.argv[3:]
    os.makedirs(PHOTOS, exist_ok=True)
    when = taken_at(src)
    name = when.strftime("%Y%m%d-%H%M%S") + ".jpg"
    dst = os.path.join(PHOTOS, name)
    subprocess.run(["sips", "-s", "format", "jpeg", "-s", "formatOptions", "82", "-Z", "1400", src, "--out", dst],
                   check=True, capture_output=True)
    idx = {"photos": []}
    if os.path.exists(INDEX):
        with open(INDEX) as f:
            idx = json.load(f)
    idx["photos"] = [p for p in idx["photos"] if p["file"] != "photos/" + name]
    idx["photos"].insert(0, {"file": "photos/" + name, "time": when.isoformat(timespec="seconds"),
                             "caption": caption, "birds": birds, "by": "Craig"})
    idx["photos"].sort(key=lambda p: p["time"], reverse=True)
    with open(INDEX, "w") as f:
        json.dump(idx, f, ensure_ascii=False, indent=1)
    subprocess.run(["git", "pull", "-q", "--rebase"], cwd=ROOT, check=True)
    subprocess.run(["git", "add", "data/photos", "data/photos.json"], cwd=ROOT, check=True)
    subprocess.run(["git", "commit", "-q", "-m", f"Craig's photo {when.strftime('%-d %b %H:%M')}"], cwd=ROOT, check=True)
    subprocess.run(["git", "push", "-q"], cwd=ROOT, check=True)
    print("added", name, "taken", when.strftime("%a %-d %b %H:%M"), "birds:", ", ".join(birds) or "none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
