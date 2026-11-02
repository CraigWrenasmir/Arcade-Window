#!/usr/bin/env python3
"""Arcade Window collector.

Watches the view from the office window through the iMac camera and writes
small public derivatives into ../data:

  state.json        live numbers (brightness, sky colour, motion, counts)
  ledger.json       bird events, newest first
  birds.json        catalogue of named birds (naive re-identification)
  palettes.json     current palette and a history
  shadow/today.json shadow clock strips for the current day (ASCII grids)
  shadow/YYYY-MM-DD.json  archive of past days
  longexposure.jpg  running average of every frame since the start
  latest.jpg        the most recent frame, window head masked, reduced
  edges.png         edge drawing of the view for the Jangle panel
  birds/<id>.jpg    cropped box for each bird event

Raw frames never leave the machine. Nothing is kept except the derivatives.
"""
import json
import os
import random
import subprocess
import sys
import time
from collections import deque
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from names import POOL  # noqa: E402

TZ = ZoneInfo("Australia/Sydney")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
BIRDS_DIR = os.path.join(DATA, "birds")
SHADOW_DIR = os.path.join(DATA, "shadow")
for d in (DATA, BIRDS_DIR, SHADOW_DIR):
    os.makedirs(d, exist_ok=True)

W, H = 1280, 720
DEVICE = int(os.environ.get("ARCADE_CAM", "0"))
PROCESS_FPS = 3.0
PUSH_EVERY = int(os.environ.get("ARCADE_PUSH", "300"))
PALETTE_EVERY = 30
SHADOW_EVERY = 300
LONGEXP_EVERY = 300
LATEST_EVERY = 60
EDGES_EVERY = 3600
MAX_EVENTS_PER_HOUR = 40
LEDGER_KEEP = 400
MIN_BLOB = 8        # px at 640 wide
MAX_BLOB = 90
GLOBAL_CHANGE_SKIP = 0.06  # fraction of frame changed -> lighting shift, not a bird
WARMUP_FRAMES = 45          # ~15 s for auto-exposure to settle before anything is written

# Regions in 1280x720 coordinates, set for the frame-8 angle (2 Nov 2026).
WINDOW_HEAD = [(0, 0), (840, 0), (60, 330), (60, 720), (0, 720)]
SKY = [(640, 0), (1280, 0), (1280, 100), (735, 125), (640, 210)]
WALL = [(450, 300), (1280, 120), (1280, 480), (470, 530)]
DOME = [(70, 600), (1280, 530), (1280, 720), (70, 720)]
LEFTWALL = [(70, 330), (440, 180), (440, 520), (70, 600)]

ASCII_RAMP = " .:-=+*#%@"


def poly_mask(poly, shape=(H, W)):
    m = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(m, [np.array(poly, dtype=np.int32)], 255)
    return m


MASK_HEAD = poly_mask(WINDOW_HEAD)
MASK_VALID = cv2.bitwise_not(MASK_HEAD)
MASKS = {"sky": poly_mask(SKY), "wall": poly_mask(WALL), "dome": poly_mask(DOME), "leftwall": poly_mask(LEFTWALL)}


def now():
    return datetime.now(TZ)


def iso(dt=None):
    return (dt or now()).isoformat(timespec="seconds")


def jdump(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=0)
    os.replace(tmp, path)


def jload(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def region_stats(bgr, mask):
    mean = cv2.mean(bgr, mask=mask)[:3]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lum = cv2.mean(gray, mask=mask)[0]
    return {"rgb": [int(mean[2]), int(mean[1]), int(mean[0])], "lum": round(lum / 255.0, 3)}


def hexrgb(rgb):
    return "#%02x%02x%02x" % tuple(int(max(0, min(255, c))) for c in rgb)


def kmeans_palette(bgr, mask, k=5):
    pts = bgr[mask > 0]
    if len(pts) < k * 10:
        return []
    sample = pts[np.random.choice(len(pts), min(4000, len(pts)), replace=False)].astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(sample, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten(), minlength=k)
    order = np.argsort(-counts)
    out = []
    for i in order:
        b, g, r = centers[i]
        out.append({"hex": hexrgb((r, g, b)), "share": round(float(counts[i]) / len(sample), 3)})
    return out


def sky_word(lum, rgb):
    r, g, b = rgb
    if lum < 0.12:
        return "night"
    if lum < 0.3:
        return "dusk"
    if b > r + 25 and lum > 0.45:
        return "clear"
    if abs(r - b) < 18 and lum > 0.7:
        return "overcast bright"
    if abs(r - b) < 18:
        return "overcast"
    if r > b + 20:
        return "warm haze"
    return "pale"


def size_word(area):
    if area < 60:
        return "tiny"
    if area < 250:
        return "small"
    if area < 900:
        return "medium"
    return "large"


def species_guess(area, lum, moved):
    s = size_word(area)
    dark = lum < 0.35
    pale = lum > 0.7
    if s == "tiny" and moved:
        return "swallow or swift (guess)"
    if s == "tiny":
        return "sparrow or finch (guess)"
    if s == "small":
        return "noisy miner or starling (guess)" if dark else "sparrow or wren (guess)"
    if s == "medium":
        if pale:
            return "pigeon or dove (guess)"
        return "magpie or currawong (guess)" if dark else "pigeon (guess)"
    if pale:
        return "cockatoo or gull (guess)"
    return "crow or raven (guess)"


class Tracker:
    """Groups moving blobs into events. One event = one bird visit."""

    def __init__(self):
        self.active = None
        self.last_seen = 0.0

    def update(self, blobs, t, frame_small, frame_full):
        if blobs:
            b = max(blobs, key=lambda x: x[4])
            if self.active is None:
                self.active = {
                    "start": t, "boxes": [b], "best": None, "best_area": 0,
                    "path": [(b[0] + b[2] // 2, b[1] + b[3] // 2)], "frames": 0,
                }
            ev = self.active
            ev["boxes"].append(b)
            ev["path"].append((b[0] + b[2] // 2, b[1] + b[3] // 2))
            ev["frames"] += 1
            if b[4] > ev["best_area"]:
                ev["best_area"] = b[4]
                ev["best"] = (frame_full.copy(), b)
            self.last_seen = t
            return None
        if self.active and t - self.last_seen > 1.5:
            ev = self.active
            self.active = None
            ev["end"] = self.last_seen
            return ev
        return None


class Collector:
    def __init__(self):
        self.cap = cv2.VideoCapture(DEVICE, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
        self.cap.set(cv2.CAP_PROP_FPS, 10)
        self.bg = cv2.createBackgroundSubtractorMOG2(history=400, varThreshold=24, detectShadows=False)
        self.tracker = Tracker()
        self.longexp = None
        self.longexp_n = 0
        self.started = now()
        self.frames = 0
        self.last = {k: 0 for k in ("palette", "shadow", "longexp", "latest", "edges", "push", "state")}
        self.ledger = jload(os.path.join(DATA, "ledger.json"), {"events": []})
        self.birds = jload(os.path.join(DATA, "birds.json"), {"birds": [], "unused": list(POOL)})
        if not self.birds.get("unused"):
            self.birds["unused"] = list(POOL)
        random.shuffle(self.birds["unused"])
        self.palettes = jload(os.path.join(DATA, "palettes.json"), {"history": []})
        self.recent_events = deque(maxlen=MAX_EVENTS_PER_HOUR)
        self.motion_level = 0.0
        self.today = now().date()
        self.shadow = self.load_shadow_day()
        self.stats = {}

    # ---------- shadow clock ----------
    def load_shadow_day(self):
        p = os.path.join(SHADOW_DIR, "today.json")
        d = jload(p, None)
        if d and d.get("date") == str(self.today):
            return d
        return {"date": str(self.today), "cols": 48, "rows": 12, "strips": []}

    def roll_day_if_needed(self):
        d = now().date()
        if d != self.today:
            jdump(os.path.join(SHADOW_DIR, f"{self.today}.json"), self.shadow)
            self.today = d
            self.shadow = {"date": str(d), "cols": 48, "rows": 12, "strips": []}
            idx = sorted(f[:-5] for f in os.listdir(SHADOW_DIR) if f.endswith(".json") and f != "today.json")
            jdump(os.path.join(SHADOW_DIR, "index.json"), {"days": idx[-60:]})

    def shadow_strip(self, bgr):
        src = np.float32([WALL[0], WALL[1], WALL[2], WALL[3]])
        dst = np.float32([[0, 0], [480, 0], [480, 120], [0, 120]])
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(bgr, M, (480, 120))
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 0)
        small = cv2.resize(gray, (48, 12), interpolation=cv2.INTER_AREA).astype(np.float32)
        lit = max(1.0, float(np.percentile(small, 90)))   # the sunlit stone sets the reference
        ratio = small / lit                                 # 1.0 = full sun, lower = shade
        # 0.92 and above is lit (blank); 0.45 and below is deep shadow (densest character)
        shade = np.clip((0.92 - ratio) / (0.92 - 0.45), 0, 1)
        rows = []
        for r in range(12):
            rows.append("".join(ASCII_RAMP[int(v * (len(ASCII_RAMP) - 1))] for v in shade[r]))
        contrast = float(shade.mean())
        return rows, round(contrast, 3)

    # ---------- birds ----------
    def perch_cell(self, cx, cy):
        return f"{int(cx * 8 / 640)}:{int(cy * 5 / 360)}"

    def identify(self, sig):
        """Naive re-identification: same perch cell, size bucket and tone within 21 days."""
        cutoff = now() - timedelta(days=21)
        best = None
        for b in self.birds["birds"]:
            if b["size"] != sig["size"] or b["tone"] != sig["tone"]:
                continue
            if b["kind"] != sig["kind"]:
                continue
            if sig["kind"] == "percher" and b["cell"] != sig["cell"]:
                continue
            try:
                last = datetime.fromisoformat(b["last_seen"])
            except Exception:
                continue
            if last < cutoff:
                continue
            if best is None or last > datetime.fromisoformat(best["last_seen"]):
                best = b
        return best

    def new_bird(self, sig):
        if not self.birds["unused"]:
            self.birds["unused"] = list(POOL)
            random.shuffle(self.birds["unused"])
        name = self.birds["unused"].pop()
        b = {"id": f"b{len(self.birds['birds']) + 1:04d}", "name": name, "visits": 0,
             "first_seen": iso(), "last_seen": iso(), "species": sig["species"], **{k: sig[k] for k in ("size", "tone", "kind", "cell")}}
        self.birds["birds"].append(b)
        return b

    def finish_event(self, ev):
        t = now()
        self.recent_events.append(time.time())
        if len(self.recent_events) == MAX_EVENTS_PER_HOUR and time.time() - self.recent_events[0] < 3600:
            return
        frame, (x, y, w, h, area) = ev["best"]
        path = ev["path"]
        dx = path[-1][0] - path[0][0]
        dy = path[-1][1] - path[0][1]
        travel = float(np.hypot(dx, dy))
        dur = max(0.0, ev["end"] - ev["start"])
        moved = travel > 40 or dur < 2.0
        kind = "flyer" if moved else "percher"
        if kind == "flyer" and ev["frames"] < 2:
            return
        # scale blob box to full frame
        sx, sy = W / 640.0, H / 360.0
        X, Y, Wd, Hd = int(x * sx), int(y * sy), int(w * sx), int(h * sy)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blob = gray[max(0, Y):Y + Hd, max(0, X):X + Wd]
        lum = float(blob.mean()) / 255.0 if blob.size else 0.5
        tone = "dark" if lum < 0.35 else ("pale" if lum > 0.7 else "mid")
        sig = {"size": size_word(area), "tone": tone, "kind": kind,
               "cell": self.perch_cell(x + w / 2, y + h / 2), "species": species_guess(area, lum, moved)}
        bird = self.identify(sig)
        confidence = "likely"
        if bird is None:
            bird = self.new_bird(sig)
            confidence = "new"
        bird["visits"] += 1
        bird["last_seen"] = iso(t)
        eid = t.strftime("%Y%m%d-%H%M%S")
        # crop with margin, draw box, save small
        m = int(max(Wd, Hd) * 1.2) + 24
        cx, cy = X + Wd // 2, Y + Hd // 2
        x0, y0 = max(0, cx - m), max(0, cy - m)
        x1, y1 = min(W, cx + m), min(H, cy + m)
        crop = frame[y0:y1, x0:x1].copy()
        cv2.rectangle(crop, (X - x0, Y - y0), (X - x0 + Wd, Y - y0 + Hd), (0, 200, 255), 2)
        if crop.shape[1] > 320:
            crop = cv2.resize(crop, (320, int(crop.shape[0] * 320 / crop.shape[1])), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(BIRDS_DIR, f"{eid}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 78])
        direction = ""
        if kind == "flyer":
            direction = ("east" if dx > 0 else "west") if abs(dx) >= abs(dy) else ("down" if dy > 0 else "up")
        event = {
            "id": eid, "time": iso(t), "bird": bird["name"], "bird_id": bird["id"],
            "visit": bird["visits"], "confidence": confidence, "kind": kind,
            "species": sig["species"], "size": sig["size"], "tone": tone,
            "duration": round(dur, 1), "direction": direction,
            "box": [X, Y, Wd, Hd], "cell": sig["cell"], "image": f"birds/{eid}.jpg",
            "sky": self.stats.get("sky", {}).get("word", ""),
        }
        self.ledger["events"].insert(0, event)
        self.ledger["events"] = self.ledger["events"][:LEDGER_KEEP]
        keep = {e["image"].split("/")[-1] for e in self.ledger["events"]}
        for f in os.listdir(BIRDS_DIR):
            if f.endswith(".jpg") and f not in keep:
                os.remove(os.path.join(BIRDS_DIR, f))
        jdump(os.path.join(DATA, "ledger.json"), self.ledger)
        jdump(os.path.join(DATA, "birds.json"), self.birds)
        print(f"[bird] {event['time']} {bird['name']} ({confidence}, visit {bird['visits']}) {kind} {sig['species']}", flush=True)

    def detect(self, frame, t):
        small = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        fg = self.bg.apply(gray, learningRate=0.004)
        valid = cv2.resize(MASK_VALID, (640, 360), interpolation=cv2.INTER_NEAREST)
        fg = cv2.bitwise_and(fg, valid)
        changed = float(np.count_nonzero(fg)) / float(np.count_nonzero(valid))
        self.motion_level = round(changed, 4)
        if changed > GLOBAL_CHANGE_SKIP or self.frames < 60:
            self.tracker.update([], t, small, frame)
            return
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        fg = cv2.dilate(fg, np.ones((5, 5), np.uint8), iterations=1)
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blobs = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w < MIN_BLOB and h < MIN_BLOB:
                continue
            if w > MAX_BLOB or h > MAX_BLOB:
                continue
            ar = w / float(h)
            if ar > 5 or ar < 0.2:
                continue
            blobs.append((x, y, w, h, int(cv2.contourArea(c))))
        ev = self.tracker.update(blobs, t, small, frame)
        if ev and ev["best"] is not None:
            self.finish_event(ev)

    # ---------- periodic outputs ----------
    def do_palette(self, frame):
        pal = {"time": iso(), "regions": {}}
        for name, mask in MASKS.items():
            st = region_stats(frame, mask)
            pal["regions"][name] = {"hex": hexrgb(st["rgb"]), "lum": st["lum"]}
        pal["swatches"] = kmeans_palette(frame, MASK_VALID, 5)
        pal["sky_swatches"] = kmeans_palette(frame, MASKS["sky"], 3)
        self.palettes["current"] = pal
        hist = self.palettes.setdefault("history", [])
        if not hist or (datetime.fromisoformat(hist[-1]["time"]) + timedelta(minutes=10)) <= now():
            hist.append({"time": pal["time"], "sky": pal["regions"]["sky"]["hex"], "dome": pal["regions"]["dome"]["hex"],
                         "wall": pal["regions"]["wall"]["hex"], "brick": pal["regions"]["leftwall"]["hex"],
                         "swatches": [s["hex"] for s in pal["swatches"]]})
            self.palettes["history"] = hist[-1008:]
        jdump(os.path.join(DATA, "palettes.json"), self.palettes)

    def do_shadow(self, frame):
        rows, contrast = self.shadow_strip(frame)
        self.shadow["strips"].append({"time": now().strftime("%H:%M"), "contrast": contrast, "rows": rows})
        jdump(os.path.join(SHADOW_DIR, "today.json"), self.shadow)

    def do_longexp(self):
        if self.longexp is None:
            return
        img = np.clip(self.longexp, 0, 255).astype(np.uint8)
        img = self.masked(img)
        out = cv2.resize(img, (960, 540), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(DATA, "longexposure.jpg"), out, [cv2.IMWRITE_JPEG_QUALITY, 72])

    def masked(self, img):
        img = img.copy()
        img[MASK_HEAD > 0] = (18, 14, 12)
        return img

    def do_latest(self, frame):
        out = cv2.resize(self.masked(frame), (960, 540), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(DATA, "latest.jpg"), out, [cv2.IMWRITE_JPEG_QUALITY, 70])

    def do_edges(self, frame):
        gray = cv2.cvtColor(self.masked(frame), cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 7, 50, 50)
        e = cv2.Canny(gray, 60, 140)
        e = cv2.bitwise_and(e, MASK_VALID)
        out = cv2.resize(e, (640, 360), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(DATA, "edges.png"), out)

    def do_state(self, frame):
        sky = region_stats(frame, MASKS["sky"])
        wall = region_stats(frame, MASKS["wall"])
        dome = region_stats(frame, MASKS["dome"])
        brick = region_stats(frame, MASKS["leftwall"])
        allv = region_stats(frame, MASK_VALID)
        today = now().strftime("%Y-%m-%d")
        events_today = [e for e in self.ledger["events"] if e["time"].startswith(today)]
        self.stats = {
            "time": iso(), "started": iso(self.started), "frames": self.frames,
            "sky": {"hex": hexrgb(sky["rgb"]), "lum": sky["lum"], "word": sky_word(sky["lum"], sky["rgb"])},
            "wall": {"hex": hexrgb(wall["rgb"]), "lum": wall["lum"]},
            "dome": {"hex": hexrgb(dome["rgb"]), "lum": dome["lum"]},
            "brick": {"hex": hexrgb(brick["rgb"]), "lum": brick["lum"]},
            "overall_lum": allv["lum"], "motion": self.motion_level,
            "birds_today": len(events_today), "birds_total": len(self.ledger["events"]),
            "named_birds": len(self.birds["birds"]),
            "last_bird": self.ledger["events"][0] if self.ledger["events"] else None,
            "shadow_strips_today": len(self.shadow["strips"]),
        }
        jdump(os.path.join(DATA, "state.json"), self.stats)

    def do_push(self):
        try:
            r = subprocess.run(["git", "status", "--porcelain", "data"], cwd=ROOT, capture_output=True, text=True, timeout=30)
            if not r.stdout.strip():
                return
            # Fold onto whatever is on GitHub: the iMac's data is the truth, history is not.
            subprocess.run(["git", "fetch", "-q", "origin", "main"], cwd=ROOT, check=True, timeout=120)
            subprocess.run(["git", "reset", "-q", "--soft", "origin/main"], cwd=ROOT, check=True, timeout=60)
            subprocess.run(["git", "add", "data"], cwd=ROOT, check=True, timeout=60)
            subprocess.run(["git", "commit", "-q", "-m", f"window {now().strftime('%Y-%m-%d %H:%M')}"], cwd=ROOT, check=True, timeout=60)
            subprocess.run(["git", "push", "-q"], cwd=ROOT, check=True, timeout=120)
            print(f"[push] {iso()}", flush=True)
        except Exception as e:
            print(f"[push] failed: {e}", flush=True)

    # ---------- main loop ----------
    def run(self):
        print(f"[start] {iso()} device {DEVICE}", flush=True)
        interval = 1.0 / PROCESS_FPS
        next_t = time.time()
        while True:
            ok, frame = self.cap.read()
            if not ok:
                print("[cam] read failed, reopening", flush=True)
                time.sleep(2)
                self.cap.release()
                self.__init__()
                continue
            t = time.time()
            if t < next_t:
                continue
            next_t = t + interval
            if frame.shape[1] != W:
                frame = cv2.resize(frame, (W, H))
            self.frames += 1
            self.roll_day_if_needed()
            f32 = frame.astype(np.float32)
            if self.longexp is None:
                self.longexp = f32
                self.longexp_n = 1
            else:
                self.longexp_n += 1
                self.longexp += (f32 - self.longexp) / float(self.longexp_n)
            if self.frames <= WARMUP_FRAMES:
                self.longexp = None  # discard the settling frames from the average
                continue
            self.detect(frame, t)
            if t - self.last["state"] > 5:
                self.do_state(frame); self.last["state"] = t
            if t - self.last["palette"] > PALETTE_EVERY:
                self.do_palette(frame); self.last["palette"] = t
            if t - self.last["shadow"] > SHADOW_EVERY:
                self.do_shadow(frame); self.last["shadow"] = t
            if t - self.last["longexp"] > LONGEXP_EVERY:
                self.do_longexp(); self.last["longexp"] = t
            if t - self.last["latest"] > LATEST_EVERY:
                self.do_latest(frame); self.last["latest"] = t
            if t - self.last["edges"] > EDGES_EVERY:
                self.do_edges(frame); self.last["edges"] = t
            if PUSH_EVERY > 0 and t - self.last["push"] > PUSH_EVERY:
                self.do_push(); self.last["push"] = t


if __name__ == "__main__":
    Collector().run()
