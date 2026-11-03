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
import re
import subprocess
import urllib.request
import sys
import time
from collections import deque
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# RTSP over TCP from the GoPro: the UDP stream loses packets and every torn frame looks like motion
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
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
SOURCE = os.environ.get("ARCADE_SOURCE", "gopro")      # "gopro" over USB, or "cam" for the built-in FaceTime camera
GOPRO_FOV = os.environ.get("ARCADE_FOV", "4")           # 0 wide, 2 narrow, 3 superview, 4 linear
GOPRO_RES = os.environ.get("ARCADE_RES", "12")          # 4 480p, 7 720p, 12 1080p
GOPRO_PROTO = os.environ.get("ARCADE_PROTO", "RTSP")    # RTSP (TCP, lossless) or TS (UDP, drops packets)
FLIP = os.environ.get("ARCADE_FLIP", "1") == "1"        # the GoPro hangs upside down on the glass; its stream ignores orientation
PROCESS_FPS = 3.0
PUSH_EVERY = int(os.environ.get("ARCADE_PUSH", "300"))
PALETTE_EVERY = 30
SHADOW_EVERY = 300
LONGEXP_EVERY = 300
LATEST_EVERY = 60
EDGES_EVERY = 300
MAX_EVENTS_PER_HOUR = 40
LEDGER_KEEP = 400
DET_W, DET_H = 1280, 720   # detection runs on the full working frame: the GoPro's wide view makes birds small
MIN_BLOB = 9        # px at DET_W wide, measured after a 5 px dilation, so a 5 px bird passes and a 3 px speck does not
MAX_BLOB = 180
GLOBAL_CHANGE_SKIP = 0.12  # fraction of frame changed -> lighting shift, not a bird (the sky is near half this view)
WARMUP_FRAMES = 45          # ~15 s for auto-exposure to settle before anything is written

# Regions in 1280x720 coordinates, for the GoPro hanging upside down on the glass, fixed 3 Nov 2026
# (the stream is turned 180 degrees before these apply)
WINDOW_HEAD = [(0, 0), (165, 0), (0, 290)]
SKY = [(240, 0), (1280, 0), (1280, 245), (900, 318), (765, 287), (640, 362), (600, 335), (521, 288)]
WALL = [(500, 410), (1280, 290), (1280, 640), (500, 660)]
DOME = [(0, 665), (720, 720), (0, 720)]
LEFTWALL = [(0, 300), (487, 413), (487, 567), (0, 533)]
# the top edge of the pink wall against the sky flickers as thin cloud passes behind it: nothing counts there
PINK_EDGE = [(220, 0), (250, 0), (536, 288), (612, 352), (596, 372), (508, 290)]
# the open sky well above the roofline is cloud country: birds are only counted near the buildings
HIGH_SKY = [(234, 0), (1280, 0), (1280, 165), (900, 240), (765, 207), (640, 282), (600, 300), (521, 288)]
# thin cloud behind a hard line flickers: the parapet spikes and the gable slopes, leaving the apex itself open
PARAPET_EDGE = [(893, 304), (1280, 235), (1280, 257), (893, 326)]
GABLE_EDGE = [[(636, 354), (745, 292), (745, 310), (640, 370)], [(787, 292), (900, 312), (900, 328), (787, 310)]]
# the valley where the two gable slopes meet: cloud brightness changes there read as a blob (4 false entries 3 Nov pm)
VALLEY = [(594, 344), (652, 344), (652, 388), (594, 388)]

ASCII_RAMP = " .:-=+*#%@"

# Colour correction, applied to every frame before anything is measured or published.
# A 3x3 matrix on RGB plus an offset, fitted from reference patches (see calibrate.py).
# Identity until collector/colour.json exists.
COLOUR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "colour.json")


def load_colour():
    try:
        with open(COLOUR_PATH) as f:
            d = json.load(f)
        M = np.array(d["matrix"], dtype=np.float32).reshape(3, 3)
        off = np.array(d.get("offset", [0, 0, 0]), dtype=np.float32)
        return M, off
    except Exception:
        return None, None


COLOUR_M, COLOUR_OFF = load_colour()


def correct_colour(bgr):
    if COLOUR_M is None or SOURCE != "cam":
        return bgr
    rgb = bgr[:, :, ::-1].astype(np.float32)
    out = rgb @ COLOUR_M.T + COLOUR_OFF
    return np.clip(out, 0, 255).astype(np.uint8)[:, :, ::-1]


def poly_mask(poly, shape=(H, W)):
    m = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(m, [np.array(poly, dtype=np.int32)], 255)
    return m


MASK_HEAD = poly_mask(WINDOW_HEAD)
MASK_VALID = cv2.bitwise_and(cv2.bitwise_not(MASK_HEAD), cv2.bitwise_not(poly_mask(PINK_EDGE)))
MASK_VALID = cv2.bitwise_and(MASK_VALID, cv2.bitwise_not(poly_mask(HIGH_SKY)))
for _band in [PARAPET_EDGE, VALLEY] + GABLE_EDGE:
    MASK_VALID = cv2.bitwise_and(MASK_VALID, cv2.bitwise_not(poly_mask(_band)))
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
        return "swallow or swift"
    if s == "tiny":
        return "sparrow or finch"
    if s == "small":
        return "noisy miner or starling" if dark else "sparrow or wren"
    if s == "medium":
        if pale:
            return "pigeon or dove"
        return "magpie or currawong" if dark else "pigeon"
    if pale:
        return "cockatoo or gull"
    return "crow or raven"


class GoProSource:
    """A HERO9 or later over USB-C. The camera is a small network device on the cable; its web API starts
    a 1080p MPEG-TS stream over UDP to this machine, which OpenCV reads. Restarts the stream when it stalls."""
    STALL = 12.0

    def __init__(self):
        self.ip = None
        self.cap = None
        self.last_frame = 0.0
        self.last_health = 0.0
        self.health = {}
        self.open()

    @staticmethod
    def find_ip():
        try:
            out = subprocess.run(["ip", "-4", "-o", "addr"], capture_output=True, text=True, timeout=5).stdout
        except Exception:
            return None
        m = re.search(r"inet (172\.2\d\.1\d\d)\.\d+/24", out)
        return m.group(1) + ".51" if m else None

    def api(self, path, timeout=8):
        try:
            with urllib.request.urlopen(f"http://{self.ip}:8080{path}", timeout=timeout) as r:
                return r.read().decode()
        except Exception as e:
            return "ERR " + str(e)

    def open(self):
        if self.cap is not None:
            try: self.cap.release()
            except Exception: pass
            self.cap = None
        self.ip = self.find_ip()
        if not self.ip:
            print("[gopro] no USB network link; is the camera on?", flush=True)
            return False
        self.api("/gopro/camera/control/wired_usb?p=1")
        # a stream left running by a killed process makes the camera answer 500 to start; exit clears it
        self.api("/gopro/webcam/stop")
        self.api("/gopro/webcam/exit")
        time.sleep(2)
        for attempt in range(2):
            r = self.api(f"/gopro/webcam/start?res={GOPRO_RES}&fov={GOPRO_FOV}&port=8554&protocol={GOPRO_PROTO}")
            print(f"[gopro] {self.ip} start {GOPRO_PROTO} -> {r.strip().replace(chr(10), ' ').replace(chr(9), '')[:60]}", flush=True)
            if not r.startswith("ERR"):
                break
            time.sleep(5)
        time.sleep(2)
        if GOPRO_PROTO == "RTSP":
            self.cap = cv2.VideoCapture(f"rtsp://{self.ip}:554/live", cv2.CAP_FFMPEG)
        else:
            self.cap = cv2.VideoCapture("udp://0.0.0.0:8554?overrun_nonfatal=1&fifo_size=50000000&timeout=5000000", cv2.CAP_FFMPEG)
        self.last_frame = time.time()
        return self.cap.isOpened()

    def read(self):
        if self.cap is None or not self.cap.isOpened():
            time.sleep(5)
            self.open()
            return False, None
        ok, frame = self.cap.read()
        t = time.time()
        if ok and frame is not None:
            self.last_frame = t
            if t - self.last_health > 60:
                self.last_health = t
                self.poll_health()
            return True, frame
        if t - self.last_frame > self.STALL:
            print("[gopro] stream stalled, restarting", flush=True)
            self.open()
        return False, None

    def poll_health(self):
        try:
            st = json.loads(self.api("/gopro/camera/state", timeout=4)).get("status", {})
            self.health = {"model": "HERO13 Black", "battery": st.get("70"), "hot": bool(st.get("6")),
                           "busy": bool(st.get("8"))}
        except Exception:
            pass

    def release(self):
        self.api("/gopro/webcam/stop")
        self.api("/gopro/webcam/exit")
        if self.cap is not None:
            self.cap.release()


class CamSource:
    """The iMac's own FaceTime HD camera through V4L2."""
    def __init__(self):
        self.cap = cv2.VideoCapture(DEVICE, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
        self.cap.set(cv2.CAP_PROP_FPS, 10)
        self.health = {"model": "FaceTime HD"}

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()


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
        self.cap = GoProSource() if SOURCE == "gopro" else CamSource()
        self.bg = cv2.createBackgroundSubtractorMOG2(history=400, varThreshold=36, detectShadows=True)
        self.bg.setShadowThreshold(0.7)
        self.tracker = Tracker()
        self.longexp = None
        self.longexp_n = 0
        self.started = self.watching_since()
        self.frames = 0
        t0 = time.time()
        # periodic outputs wait their full interval after a start; only state, latest and push are prompt
        self.last = {"palette": t0, "shadow": t0, "longexp": t0, "latest": 0, "edges": 0, "push": 0, "state": 0}
        self.ledger = jload(os.path.join(DATA, "ledger.json"), {"events": []})
        self.birds = jload(os.path.join(DATA, "birds.json"), {"birds": [], "unused": list(POOL)})
        if not self.birds.get("unused"):
            self.birds["unused"] = list(POOL)
        random.shuffle(self.birds["unused"])
        self.palettes = jload(os.path.join(DATA, "palettes.json"), {"history": []})
        self.recent_events = deque(maxlen=MAX_EVENTS_PER_HOUR)
        self.motion_level = 0.0
        self.suppress_until = 0.0   # after the whole view shifts, relearn before trusting any blob
        self.relearn_frames = 0
        self.today = now().date()
        self.shadow = self.load_shadow_day()
        self.stats = {}

    def watching_since(self):
        """The date the watch began, kept in data/since.json so restarts do not reset it."""
        p = os.path.join(DATA, "since.json")
        d = jload(p, None)
        if d and d.get("since"):
            try:
                return datetime.fromisoformat(d["since"])
            except Exception:
                pass
        t = now()
        jdump(p, {"since": iso(t)})
        return t

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
        return f"{int(cx * 8 / DET_W)}:{int(cy * 5 / DET_H)}"

    def identify(self, sig):
        """Naive re-identification: same perch cell, size bucket and tone within 21 days."""
        cutoff = now() - timedelta(days=21)
        just_now = now() - timedelta(minutes=3)
        best = None
        for b in self.birds["birds"]:
            if b["size"] != sig["size"] or b["tone"] != sig["tone"]:
                continue
            try:
                last = datetime.fromisoformat(b["last_seen"])
            except Exception:
                continue
            if last < cutoff:
                continue
            # a bird leaving (or arriving at) a perch it was just seen on is the same bird
            same_spot_just_now = b["cell"] == sig["cell"] and last >= just_now
            if not same_spot_just_now:
                if b["kind"] != sig["kind"]:
                    continue
                if sig["kind"] == "percher" and b["cell"] != sig["cell"]:
                    continue
            if best is None or last > datetime.fromisoformat(best["last_seen"]):
                best = b
        return best

    def new_bird(self, sig):
        if not self.birds["unused"]:
            self.birds["unused"] = list(POOL)
            random.shuffle(self.birds["unused"])
        name = self.birds["unused"].pop()
        used = [int(x["id"][1:]) for x in self.birds["birds"] if x.get("id", "").startswith("b")]
        used += [int(e["bird_id"][1:]) for e in self.ledger["events"] if e.get("bird_id", "").startswith("b")]
        b = {"id": f"b{(max(used) if used else 0) + 1:04d}", "name": name, "visits": 0,
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
        moved = travel > 60 or dur < 2.0   # px at DET_W wide
        kind = "flyer" if moved else "percher"
        if kind == "flyer" and ev["frames"] < 2:
            return
        if ev["frames"] < 2:
            print(f"[skip] {iso(t)} seen in one frame only", flush=True)
            return
        if kind == "flyer" and dur > 15:
            print(f"[skip] {iso(t)} drifted for {dur:.0f} s, not a bird", flush=True)
            return
        # scale blob box to full frame
        sx, sy = W / float(DET_W), H / float(DET_H)
        # size words were set on the old camera at 640 wide; the GoPro sees wider, so a bird covers about half the area
        area = area * (640.0 / DET_W) ** 2 * 2.0
        X, Y, Wd, Hd = int(x * sx), int(y * sy), int(w * sx), int(h * sy)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blob = gray[max(0, Y):Y + Hd, max(0, X):X + Wd]
        lum = float(blob.mean()) / 255.0 if blob.size else 0.5
        # a bird stands out from what is around it; a patch of sky that brightened does not
        pad = 12
        ring = gray[max(0, Y - pad):Y + Hd + pad, max(0, X - pad):X + Wd + pad].astype(np.float32)
        if blob.size and ring.size:
            inner = float(blob.astype(np.float32).std())
            contrast = abs(float(blob.mean()) - float(ring.mean()))
            if contrast < 6 and inner < 8:
                print(f"[skip] {iso(t)} flat blob (contrast {contrast:.1f}, texture {inner:.1f})", flush=True)
                return
        # a shadow is the same surface, darker: same hue and saturation as the ring around it
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hin = hsv[max(0, Y):Y + Hd, max(0, X):X + Wd]
        hring = hsv[max(0, Y - pad):Y + Hd + pad, max(0, X - pad):X + Wd + pad]
        if hin.size and hring.size:
            dh = abs(float(hin[:, :, 0].mean()) - float(hring[:, :, 0].mean()))
            ds = abs(float(hin[:, :, 1].mean()) - float(hring[:, :, 1].mean()))
            darker = float(hin[:, :, 2].mean()) < float(hring[:, :, 2].mean()) - 8
            if darker and dh < 8 and ds < 18:
                print(f"[skip] {iso(t)} shadow (hue diff {dh:.1f}, sat diff {ds:.1f})", flush=True)
                return
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
        bird.setdefault("images", []).append(f"birds/{eid}.jpg")
        bird["images"] = bird["images"][-6:]
        # crop with margin, draw box, save small
        m = max(80, int(max(Wd, Hd) * 1.2) + 24)   # tiny birds still get a crop with some roof around them
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
        for b in self.birds["birds"]:
            keep.update(i.split("/")[-1] for i in b.get("images", []))
        for f in os.listdir(BIRDS_DIR):
            if f.endswith(".jpg") and f not in keep:
                os.remove(os.path.join(BIRDS_DIR, f))
        jdump(os.path.join(DATA, "ledger.json"), self.ledger)
        jdump(os.path.join(DATA, "birds.json"), self.birds)
        print(f"[bird] {event['time']} {bird['name']} ({confidence}, visit {bird['visits']}) {kind} {sig['species']}", flush=True)

    def in_sky(self, cx, cy):
        m = MASKS["sky"]
        return 0 <= cy < m.shape[0] and 0 <= cx < m.shape[1] and m[cy, cx] > 0

    def darker_than_ring(self, gray, x, y, w, h, pad=10, by=10.0):
        H, W = gray.shape[:2]
        inner = gray[max(0, y):min(H, y + h), max(0, x):min(W, x + w)]
        ring = gray[max(0, y - pad):min(H, y + h + pad), max(0, x - pad):min(W, x + w + pad)]
        return bool(inner.size and ring.size and float(inner.mean()) < float(ring.mean()) - by)

    def sharp_enough(self, gray, x, y, w, h, pad=3, need=90.0, texture=25.0):
        """Measured 3 Nov on real apex birds (max gradient 103-131, texture 39-55) against cloud gaps
        (gradient 53-95, texture 4-22): a bird has a hard edge and a varied interior."""
        H, W = gray.shape[:2]
        patch = gray[max(0, y - pad):min(H, y + h + pad), max(0, x - pad):min(W, x + w + pad)]
        inner = gray[max(0, y):min(H, y + h), max(0, x):min(W, x + w)]
        if patch.size < 16 or inner.size < 4:
            return False
        gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
        return float(np.max(np.hypot(gx, gy))) / 4.0 >= need and float(inner.std()) >= texture

    def stands_out(self, gray, x, y, w, h, pad=10, need=40.0):
        """A bird is darker or lighter than its surroundings by a clear margin. The edge of a cloud is not."""
        H, W = gray.shape[:2]
        inner = gray[max(0, y):min(H, y + h), max(0, x):min(W, x + w)]
        ring = gray[max(0, y - pad):min(H, y + h + pad), max(0, x - pad):min(W, x + w + pad)]
        if not inner.size or not ring.size:
            return False
        return abs(float(inner.mean()) - float(ring.mean())) >= need

    def debug_dump(self, small, fg, blobs, t):
        """What the detector saw, for tuning by eye. Kept locally, never pushed."""
        try:
            d = os.path.join(ROOT, "debug")
            os.makedirs(d, exist_ok=True)
            img = small.copy()
            for (x, y, w, h, a) in blobs:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 200, 255), 2)
            mask = cv2.cvtColor(fg, cv2.COLOR_GRAY2BGR)
            out = np.vstack([img, mask])
            out = cv2.resize(out, (960, 1080), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(d, now().strftime("%H%M%S") + ".jpg"), out, [cv2.IMWRITE_JPEG_QUALITY, 70])
            files = sorted(f for f in os.listdir(d) if f.endswith(".jpg"))
            for f in files[:-40]:
                os.remove(os.path.join(d, f))
        except Exception as e:
            print("[debug]", e, flush=True)

    def looks_like_shadow(self, small, x, y, w, h, pad=8):
        """Same hue and saturation as the ring around it, only darker: a shadow, not a bird."""
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV).astype(np.float32)
        H, W = hsv.shape[:2]
        inner = hsv[max(0, y):min(H, y + h), max(0, x):min(W, x + w)]
        ring = hsv[max(0, y - pad):min(H, y + h + pad), max(0, x - pad):min(W, x + w + pad)]
        if not inner.size or not ring.size:
            return False
        dh = abs(float(inner[:, :, 0].mean()) - float(ring[:, :, 0].mean()))
        ds = abs(float(inner[:, :, 1].mean()) - float(ring[:, :, 1].mean()))
        darker = float(inner[:, :, 2].mean()) < float(ring[:, :, 2].mean()) - 6
        return darker and dh < 12 and ds < 25

    def detect(self, frame, t):
        small = frame if frame.shape[1] == DET_W else cv2.resize(frame, (DET_W, DET_H), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        # a fresh model learns fast for a few seconds; until it has, nothing it says is judged
        rate = 0.05 if self.relearn_frames > 0 else 0.004
        fg = self.bg.apply(small, learningRate=rate)
        if self.relearn_frames > 0:
            self.relearn_frames -= 1
            self.motion_level = 0.0
            self.tracker.update([], t, small, frame)
            return
        fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)[1]   # shadows come back as 127: drop them
        valid = cv2.resize(MASK_VALID, (DET_W, DET_H), interpolation=cv2.INTER_NEAREST)
        fg = cv2.bitwise_and(fg, valid)
        changed = float(np.count_nonzero(fg)) / float(np.count_nonzero(valid))
        settled = self.frames >= WARMUP_FRAMES + 60   # background model has had ~20 s to learn
        self.motion_level = round(changed, 4) if settled else 0.0
        if changed > GLOBAL_CHANGE_SKIP and settled and t >= self.suppress_until:
            # the camera moved or the light jumped: start the background again and wait it out
            self.bg = cv2.createBackgroundSubtractorMOG2(history=400, varThreshold=36, detectShadows=True)
            self.bg.setShadowThreshold(0.7)
            self.relearn_frames = 45
            self.suppress_until = t + 60
            print(f"[view] {iso()} global change {changed:.2f}, relearning", flush=True)
            self.tracker.update([], t, small, frame)
            return
        if changed > GLOBAL_CHANGE_SKIP or not settled or t < self.suppress_until:
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
            if ar > 3 or ar < 0.45:      # birds are compact; vents, poles and tall shadows are not
                continue
            if self.looks_like_shadow(small, x, y, w, h, pad=16):
                continue
            if not self.stands_out(gray, x, y, w, h):
                continue
            if self.in_sky(x + w // 2, y + h // 2) and not self.darker_than_ring(gray, x, y, w, h):
                continue      # against the sky a bird is a silhouette; a bright blob up there is cloud
            if not self.sharp_enough(gray, x, y, w, h):
                continue      # a bird has a hard edge; a gap between clouds does not
            blobs.append((x, y, w, h, int(cv2.contourArea(c))))
        if blobs and t - self.last.get("debug", 0) > 20:
            self.last["debug"] = t
            self.debug_dump(small, fg, blobs, t)
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
        out = cv2.resize(self.masked(frame), (1280, 720), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(DATA, "latest.jpg"), out, [cv2.IMWRITE_JPEG_QUALITY, 74])

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
            "camera": getattr(self.cap, "health", {}),
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
            subprocess.run(["git", "reset", "-q", "--mixed", "origin/main"], cwd=ROOT, check=True, timeout=60)
            subprocess.run(["git", "add", "data"], cwd=ROOT, check=True, timeout=60)
            subprocess.run(["git", "commit", "-q", "-m", f"window {now().strftime('%Y-%m-%d %H:%M')}"], cwd=ROOT, check=True, timeout=60)
            subprocess.run(["git", "push", "-q"], cwd=ROOT, check=True, timeout=120)
            print(f"[push] {iso()}", flush=True)
        except Exception as e:
            print(f"[push] failed: {e}", flush=True)

    # ---------- main loop ----------
    def run(self):
        print(f"[start] {iso()} source {SOURCE}", flush=True)
        interval = 1.0 / PROCESS_FPS
        next_t = time.time()
        while True:
            ok, frame = self.cap.read()
            if not ok:
                if SOURCE == "gopro":
                    continue        # the source restarts its own stream
                print("[cam] read failed, reopening", flush=True)
                time.sleep(2)
                self.cap.release()
                self.__init__()
                continue
            t = time.time()
            if t < next_t:
                continue
            next_t = t + interval
            if FLIP and SOURCE == "gopro":
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            if frame.shape[1] != W:
                frame = cv2.resize(frame, (W, H))
            frame = correct_colour(frame)
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
