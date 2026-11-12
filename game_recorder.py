#!/usr/bin/env python3
"""
Game Recorder v2 - personal screen recorder for YouTube / Instagram / Shorts.

  * Output formats 16:9, 9:16, 4:5, 1:1 + layouts with webcam (overlay / stacked / only)
  * Microphone: push-to-talk (hold a key), toggle key, always on, or muted
  * Replay buffer: always keeps the last N seconds, press a key to save them
  * Text overlays (2): any font, colour, position, with {time} {date} {timer} tokens
  * Optional desktop audio (Stereo Mix / VB-Cable) mixed with the mic
  * Dark tabbed GUI with live preview, global hotkeys

Install:   pip install -r requirements.txt
Run:       python game_recorder.py
"""
import os
import sys
import time
import wave
import queue
import shutil
import threading
import subprocess
import datetime
import tempfile
import collections
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser

import cv2
import numpy as np
import mss
from PIL import Image, ImageTk, ImageDraw, ImageFont
import imageio_ffmpeg

try:
    import sounddevice as sd
except Exception:
    sd = None
try:
    from pynput import keyboard as pynput_keyboard
except Exception:
    pynput_keyboard = None

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
IS_WIN = sys.platform.startswith("win")
NOWIN = getattr(subprocess, "CREATE_NO_WINDOW", 0) if IS_WIN else 0

# --------------------------------------------------------------------------
# Options
# --------------------------------------------------------------------------
FORMATS = {
    "YouTube 16:9 - 1080p (1920x1080)": (1920, 1080),
    "YouTube 16:9 - 1440p (2560x1440)": (2560, 1440),
    "YouTube 16:9 - 720p (1280x720)": (1280, 720),
    "Shorts / Reels / TikTok 9:16 (1080x1920)": (1080, 1920),
    "Instagram Feed 4:5 (1080x1350)": (1080, 1350),
    "Instagram Square 1:1 (1080x1080)": (1080, 1080),
}
L_SCREEN = "Screen only"
L_OVERLAY = "Screen + webcam overlay"
L_STACK = "Stacked: screen top / webcam bottom"
L_CAM = "Webcam only"
LAYOUTS = [L_SCREEN, L_OVERLAY, L_STACK, L_CAM]
FIT_BLUR = "Fit (blurred background)"
FIT_CROP = "Crop to fill"
FIT_MODES = [FIT_BLUR, FIT_CROP]
CORNERS = ["Bottom right", "Bottom left", "Top right", "Top left"]
SHAPES = ["Rectangle", "Circle"]
TEXT_POS = ["Top left", "Top center", "Top right", "Middle left", "Center", "Middle right",
            "Bottom left", "Bottom center", "Bottom right"]
MIC_OFF = "Muted"
MIC_ALWAYS = "Always on"
MIC_HOLD = "Push-to-talk (hold key)"
MIC_TOGGLE = "Toggle (press key on/off)"
MIC_MODES = [MIC_HOLD, MIC_TOGGLE, MIC_ALWAYS, MIC_OFF]


# --------------------------------------------------------------------------
# Image helpers
# --------------------------------------------------------------------------
def cover(img, w, h, fx=0.5):
    """Scale to fill (w,h) and crop the overflow. fx = horizontal crop position 0..1."""
    ih, iw = img.shape[:2]
    s = max(w / iw, h / ih)
    nw, nh = max(w, int(round(iw * s))), max(h, int(round(ih * s)))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    r = cv2.resize(img, (nw, nh), interpolation=interp)
    x = int((nw - w) * fx)
    y = (nh - h) // 2
    return r[y:y + h, x:x + w]


def fit_blur(img, w, h):
    """Whole image visible, empty space filled with a dark blurred copy."""
    ih, iw = img.shape[:2]
    if abs(iw / ih - w / h) < 0.01:
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    small = cv2.resize(img, (max(2, iw // 10), max(2, ih // 10)), interpolation=cv2.INTER_AREA)
    bg = cover(small, max(2, w // 10), max(2, h // 10))
    bg = cv2.GaussianBlur(np.ascontiguousarray(bg), (0, 0), 4)
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)
    bg = cv2.convertScaleAbs(bg, alpha=0.5)
    s = min(w / iw, h / ih)
    nw, nh = int(round(iw * s)), int(round(ih * s))
    fg = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    x, y = (w - nw) // 2, (h - nh) // 2
    bg[y:y + nh, x:x + nw] = fg
    return bg


def place_screen(screen, w, h, cfg):
    if cfg["fit"] == FIT_BLUR:
        return fit_blur(screen, w, h)
    return np.ascontiguousarray(cover(screen, w, h, cfg["focus"]))


def placeholder(w, h, text):
    img = np.full((h, w, 3), 25, np.uint8)
    scale = max(0.6, w / 1200)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    cv2.putText(img, text, ((w - tw) // 2, (h + th) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (160, 160, 160), 2, cv2.LINE_AA)
    return img


_mask_cache = {}


def circle_mask(d):
    m = _mask_cache.get(d)
    if m is None:
        m = np.zeros((d, d), np.uint8)
        cv2.circle(m, (d // 2, d // 2), d // 2, 1, -1)
        m = m.astype(bool)
        _mask_cache[d] = m
    return m


def draw_cam_overlay(out, cam, cfg, W, H):
    size = max(80, int(W * cfg["size"] / 100))
    circle = cfg["shape"] == "Circle"
    cw = size
    ch = size if circle else int(size * 9 / 16)
    m = int(0.03 * min(W, H))
    corner = cfg["corner"]
    x = m if "left" in corner else W - cw - m
    y = m if "Top" in corner else H - ch - m
    if x < 0 or y < 0 or x + cw > W or y + ch > H:
        return
    tile = cover(cam, cw, ch)
    if circle:
        roi = out[y:y + ch, x:x + cw]
        mask = circle_mask(size)
        roi[mask] = tile[mask]
        cv2.circle(out, (x + size // 2, y + size // 2), size // 2, (255, 255, 255), 3, cv2.LINE_AA)
    else:
        out[y:y + ch, x:x + cw] = tile
        cv2.rectangle(out, (x, y), (x + cw - 1, y + ch - 1), (255, 255, 255), 3)


# ---- text overlay ----
def find_fonts():
    dirs = []
    if IS_WIN:
        dirs.append(os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts"))
    elif sys.platform == "darwin":
        dirs += ["/System/Library/Fonts/Supplemental", "/Library/Fonts", "/System/Library/Fonts"]
    else:
        dirs += ["/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/truetype/liberation",
                 "/usr/share/fonts/truetype"]
    wanted = [("Arial Bold", "arialbd.ttf"), ("Arial", "arial.ttf"), ("Impact", "impact.ttf"),
              ("Segoe UI Bold", "segoeuib.ttf"), ("Segoe UI", "segoeui.ttf"),
              ("Verdana Bold", "verdanab.ttf"), ("Consolas", "consola.ttf"),
              ("Comic Sans", "comic.ttf"), ("DejaVu Sans Bold", "DejaVuSans-Bold.ttf"),
              ("DejaVu Sans", "DejaVuSans.ttf"), ("Liberation Sans Bold", "LiberationSans-Bold.ttf")]
    found = {}
    for name, fn in wanted:
        for d in dirs:
            p = os.path.join(d, fn)
            if os.path.exists(p):
                found[name] = p
                break
    found["Built-in default"] = ""
    return found


_font_cache = {}


def load_font(path, px):
    k = (path, px)
    f = _font_cache.get(k)
    if f is None:
        try:
            f = ImageFont.truetype(path, px) if path else ImageFont.load_default(size=px)
        except Exception:
            try:
                f = ImageFont.load_default(size=px)
            except Exception:
                f = ImageFont.load_default()
        _font_cache[k] = f
    return f


def resolve_text(s, elapsed):
    now = datetime.datetime.now()
    e = int(elapsed)
    s = s.replace("\\n", "\n")
    return (s.replace("{time}", now.strftime("%H:%M:%S"))
             .replace("{date}", now.strftime("%d %b %Y"))
             .replace("{timer}", f"{e // 60:02d}:{e % 60:02d}"))


def render_text(s, font_path, px, color, bg, bg_alpha, outline):
    font = load_font(font_path, px)
    stroke = max(2, px // 14) if outline else 0
    spacing = max(2, px // 5)
    tmp = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    l, t, r, b = tmp.multiline_textbbox((0, 0), s, font=font, stroke_width=stroke, spacing=spacing)
    pad = int(px * 0.35) if bg else stroke + 2
    w, h = (r - l) + 2 * pad, (b - t) + 2 * pad
    img = Image.new("RGBA", (max(w, 2), max(h, 2)), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if bg:
        d.rounded_rectangle([0, 0, w - 1, h - 1], radius=pad, fill=(0, 0, 0, int(255 * bg_alpha)))
    d.multiline_text((pad - l, pad - t), s, font=font, fill=tuple(color) + (255,),
                     stroke_width=stroke, stroke_fill=(0, 0, 0, 255), spacing=spacing)
    return np.array(img)


_text_cache = {}


def apply_texts(out, texts, W, H, elapsed):
    for t in texts:
        if not t["enabled"] or not t["text"].strip():
            continue
        s = resolve_text(t["text"], elapsed)
        px = max(8, int(t["size"] * min(W, H) / 1080))
        key = (s, t["font"], px, t["color"], t["bg"], round(t["bg_alpha"], 2), t["outline"])
        rgba = _text_cache.get(key)
        if rgba is None:
            if len(_text_cache) > 64:
                _text_cache.clear()
            rgba = render_text(s, t["font"], px, t["color"], t["bg"], t["bg_alpha"], t["outline"])
            _text_cache[key] = rgba
        h, w = rgba.shape[:2]
        m = int(0.03 * min(W, H))
        pos = t["pos"]
        x = m if "left" in pos.lower() else (W - w - m if "right" in pos.lower() else (W - w) // 2)
        y = m if pos.startswith("Top") else (H - h - m if pos.startswith("Bottom") else (H - h) // 2)
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + w, W), min(y + h, H)
        if x1 <= x0 or y1 <= y0:
            continue
        sub = rgba[y0 - y:y1 - y, x0 - x:x1 - x]
        alpha = sub[..., 3:4].astype(np.float32) / 255.0
        fg = sub[..., [2, 1, 0]].astype(np.float32)
        roi = out[y0:y1, x0:x1]
        roi[:] = (fg * alpha + roi.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)


def compose(screen, cam, cfg, W, H, elapsed=0.0):
    layout = cfg["layout"]
    if cam is not None and cfg["mirror"]:
        cam = cv2.flip(cam, 1)

    if layout == L_CAM:
        out = cover(cam, W, H) if cam is not None else placeholder(W, H, "Webcam off / not found")
    elif layout == L_STACK:
        sh = int(H * cfg["split"] / 100) // 2 * 2
        top = place_screen(screen, W, sh, cfg)
        if cam is not None:
            bottom = cover(cam, W, H - sh)
        else:
            bottom = placeholder(W, H - sh, "Webcam off / not found")
        out = np.vstack([top, bottom])
    else:
        out = place_screen(screen, W, H, cfg)
        if layout == L_OVERLAY and cam is not None:
            draw_cam_overlay(out, cam, cfg, W, H)
    out = np.ascontiguousarray(out)
    apply_texts(out, cfg.get("texts", []), W, H, elapsed)
    return out


# --------------------------------------------------------------------------
# Webcam thread
# --------------------------------------------------------------------------
class Webcam(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.desired = None      # camera index or None
        self.frame = None
        self.error = None
        self.lock = threading.Lock()
        self.alive = True

    def get(self):
        with self.lock:
            return self.frame

    def run(self):
        cap, cur = None, "unset"
        while self.alive:
            want = self.desired
            if want != cur:
                if cap is not None:
                    cap.release()
                    cap = None
                with self.lock:
                    self.frame = None
                cur, self.error = want, None
                if want is not None:
                    backend = cv2.CAP_DSHOW if IS_WIN else cv2.CAP_ANY
                    cap = cv2.VideoCapture(want, backend)
                    if cap.isOpened():
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                        cap.set(cv2.CAP_PROP_FPS, 30)
                    else:
                        self.error = f"Could not open camera {want}"
                        cap.release()
                        cap = None
            if cap is not None:
                ok, f = cap.read()
                if ok:
                    with self.lock:
                        self.frame = f
                else:
                    time.sleep(0.02)
            else:
                time.sleep(0.05)
        if cap is not None:
            cap.release()


# --------------------------------------------------------------------------
# Audio
# --------------------------------------------------------------------------
def list_audio_inputs():
    """Returns (list of 'idx: name', default-input label or None)."""
    if sd is None:
        return [], None
    out, default = [], None
    try:
        try:
            api = sd.default.hostapi
        except Exception:
            api = None
        try:
            dflt = sd.default.device[0]
        except Exception:
            dflt = None
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0 and (api is None or d["hostapi"] == api):
                label = f"{i}: {d['name']}"
                out.append(label)
                if i == dflt:
                    default = label
    except Exception:
        pass
    return out, default


class AudioInput:
    """One sound-card input. Optionally gated (push-to-talk) -> silence when gate is closed.
    Can feed a wav writer (full recording) and/or a ring buffer (replay buffer)."""

    def __init__(self, dev, name, gate=None):
        info = sd.query_devices(dev)
        self.dev, self.name, self.gate = dev, name, gate
        self.ch = max(1, min(2, int(info["max_input_channels"])))
        self.sr = int(info["default_samplerate"])
        self.stream = None
        self.wf = None
        self.level = 0.0
        self.ring = collections.deque()
        self.ring_bytes = 0
        self.ring_cap = 0
        self.ring_on = False
        self.lock = threading.Lock()

    def _cb(self, indata, frames, t, status):
        raw = indata.tobytes()
        if self.gate is not None and not self.gate():
            raw = bytes(len(raw))
            self.level = 0.0
        else:
            self.level = float(np.abs(indata).max()) / 32768.0
        wf = self.wf
        if wf is not None:
            try:
                wf.writeframes(raw)
            except Exception:
                pass
        if self.ring_on:
            with self.lock:
                self.ring.append(raw)
                self.ring_bytes += len(raw)
                while self.ring_bytes > self.ring_cap and self.ring:
                    self.ring_bytes -= len(self.ring.popleft())

    def start(self):
        self.stream = sd.InputStream(device=self.dev, channels=self.ch, samplerate=self.sr,
                                     dtype="int16", callback=self._cb)
        self.stream.start()

    def stop(self):
        self.end_wav()
        self.ring_on = False
        try:
            if self.stream:
                self.stream.stop()
                self.stream.close()
        finally:
            self.stream = None

    def _open_wav(self, path):
        wf = wave.open(path, "wb")
        wf.setnchannels(self.ch)
        wf.setsampwidth(2)
        wf.setframerate(self.sr)
        return wf

    def begin_wav(self, path):
        self.wf = self._open_wav(path)

    def end_wav(self):
        wf, self.wf = self.wf, None
        if wf is not None:
            time.sleep(0.05)
            try:
                wf.close()
            except Exception:
                pass

    def enable_ring(self, secs):
        with self.lock:
            self.ring.clear()
            self.ring_bytes = 0
            self.ring_cap = int((secs + 10) * self.sr) * self.ch * 2
        self.ring_on = True

    def disable_ring(self):
        self.ring_on = False
        with self.lock:
            self.ring.clear()
            self.ring_bytes = 0

    def snap_chunks(self):
        with self.lock:
            return list(self.ring)

    def write_wav(self, chunks, secs, path):
        need = int(secs * self.sr) * self.ch * 2
        data = b"".join(chunks)
        if len(data) >= need:
            data = data[-need:]
        else:
            data = bytes(need - len(data)) + data
        wf = self._open_wav(path)
        try:
            wf.writeframes(data)
        finally:
            wf.close()


# --------------------------------------------------------------------------
# Video encoding (ffmpeg pipe)
# --------------------------------------------------------------------------
class VideoSink:
    def __init__(self, path, w, h, fps, gpu, log_path):
        self.log = open(log_path, "wb")
        self.log_path = log_path
        cmd = [FFMPEG, "-y", "-hide_banner", "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-an"]
        if gpu:
            cmd += ["-c:v", "h264_nvenc", "-rc", "vbr", "-cq", "21", "-b:v", "0"]
        else:
            cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-profile:v", "high"]
        cmd += ["-vf", "scale=out_color_matrix=bt709:out_range=tv", "-pix_fmt", "yuv420p",
                "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
                "-g", str(int(fps) * 2), "-movflags", "+faststart", path]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                     stderr=self.log, creationflags=NOWIN)

    def write(self, data):
        self.proc.stdin.write(data)

    def close(self):
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=120)
        except Exception:
            self.proc.kill()
        try:
            self.log.close()
        except Exception:
            pass
        return self.proc.returncode

    def log_tail(self, n=6):
        try:
            with open(self.log_path, "r", errors="ignore") as f:
                return "\n".join(f.read().strip().splitlines()[-n:])
        except Exception:
            return ""


def mux(video, audio_paths, out, concat=False):
    """Join video (or a concat list) with 0-2 wav files into out.mp4 (video is stream-copied)."""
    cmd = [FFMPEG, "-y", "-hide_banner"]
    if concat:
        cmd += ["-f", "concat", "-safe", "0"]
    cmd += ["-i", video]
    for a in audio_paths:
        cmd += ["-i", a]
    if not audio_paths:
        cmd += ["-c", "copy"]
    elif len(audio_paths) == 1:
        cmd += ["-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest"]
    else:
        cmd += ["-filter_complex",
                "[1:a]aresample=48000[a1];[2:a]aresample=48000[a2];"
                "[a1][a2]amix=inputs=2:duration=longest,volume=2[a]",
                "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest"]
    cmd += ["-movflags", "+faststart", out]
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, creationflags=NOWIN)
    return r.returncode == 0, r.stderr.decode(errors="ignore")[-400:]


class Timeline:
    """Constant-frame-rate clock: tells how many frames are due (duplicates a frame if we lag)."""

    def __init__(self, fps):
        self.fps, self.t0, self.written = fps, None, 0

    def due(self, now):
        if self.t0 is None:
            self.t0 = now
        target = int((now - self.t0) * self.fps) + 1
        return min(max(target - self.written, 0), self.fps)

    def next_time(self):
        return self.t0 + self.written / self.fps


class Rec:
    def __init__(self, sink, w, h, fps):
        self.sink, self.w, self.h, self.fps = sink, w, h, fps
        self.tl = Timeline(fps)
        self.stop_flag = False
        self.finished = threading.Event()
        self.error = None
        self.code = 0
        self.audio = []
        self.final = self.video_path = None


# --------------------------------------------------------------------------
# Replay buffer: rolling 3-second mp4 segments, newest N seconds are joined on demand
# --------------------------------------------------------------------------
class Seg:
    def __init__(self, path, sink):
        self.path, self.sink = path, sink
        self.frames = 0
        self.done = threading.Event()
        self.code = 0
        self.refs = 0


class Buffer:
    SEG_SECS = 3

    def __init__(self, w, h, fps, gpu, secs, workdir):
        self.w, self.h, self.fps, self.gpu = w, h, fps, gpu
        self.keep_secs = secs
        self.dir = workdir
        os.makedirs(workdir, exist_ok=True)
        self.tl = Timeline(fps)
        self.segs, self.cur, self.n = [], None, 0
        self.stop_flag = False
        self.save_req = None
        self.error = None
        self.lock = threading.Lock()
        self._new_seg()

    def _new_seg(self):
        self.n += 1
        path = os.path.join(self.dir, f"seg_{self.n:06d}.mp4")
        self.cur = Seg(path, VideoSink(path, self.w, self.h, self.fps, self.gpu, path + ".log"))

    @staticmethod
    def _close_async(seg):
        def run():
            try:
                seg.code = seg.sink.close()
            except Exception:
                seg.code = -1
            seg.done.set()
        threading.Thread(target=run, daemon=True).start()

    def rotate(self):
        old = self.cur
        if old.frames == 0:
            return
        with self.lock:
            self.segs.append(old)
        self._close_async(old)
        self._new_seg()
        self.prune()

    def write(self, data, n):
        for _ in range(n):
            if self.cur.frames >= self.SEG_SECS * self.fps:
                self.rotate()
            self.cur.sink.write(data)
            self.cur.frames += 1

    @staticmethod
    def _remove(seg):
        for p in (seg.path, seg.path + ".log"):
            try:
                os.remove(p)
            except Exception:
                pass

    def prune(self):
        with self.lock:
            while len(self.segs) > 1:
                total = sum(s.frames for s in self.segs)
                oldest = self.segs[0]
                if (total - oldest.frames >= self.keep_secs * self.fps
                        and oldest.refs == 0 and oldest.done.is_set()):
                    self.segs.pop(0)
                    self._remove(oldest)
                else:
                    break

    def snapshot(self, secs):
        """Finish the running segment and pick the newest segments covering >= secs."""
        self.rotate()
        need = int(secs * self.fps)
        chosen, total = [], 0
        with self.lock:
            for s in reversed(self.segs):
                chosen.append(s)
                s.refs += 1
                total += s.frames
                if total >= need:
                    break
        chosen.reverse()
        return chosen, total / self.fps

    def release(self, chosen):
        with self.lock:
            for s in chosen:
                s.refs -= 1

    def shutdown(self):
        cur = self.cur
        with self.lock:
            if cur.frames:
                self.segs.append(cur)
        self._close_async(cur)

        def cleanup():
            for s in list(self.segs) + [cur]:
                s.done.wait(30)
            t = time.time()
            while time.time() - t < 120 and any(s.refs for s in self.segs):
                time.sleep(0.5)
            shutil.rmtree(self.dir, ignore_errors=True)
        threading.Thread(target=cleanup, daemon=True).start()


def build_replay_clip(buf, chosen, dur, chunks, out_path):
    """Runs in a helper thread. Returns (ok, message)."""
    try:
        if not chosen:
            return False, "Replay buffer is still empty - wait a few seconds and try again."
        for s in chosen:
            s.done.wait(30)
        if any(s.code != 0 for s in chosen):
            return False, "A buffered segment is damaged (encoder problem). Try again."
        stamp = os.path.splitext(os.path.basename(out_path))[0]
        lst = os.path.join(buf.dir, stamp + "_list.txt")
        with open(lst, "w") as f:
            for s in chosen:
                p = s.path.replace("\\", "/").replace("'", "'\\''")
                f.write(f"file '{p}'\n")
        wavs = []
        for inp, ch in chunks:
            p = os.path.join(buf.dir, f"{stamp}_{inp.name}.wav")
            inp.write_wav(ch, dur, p)
            wavs.append(p)
        ok, err = mux(lst, wavs, out_path, concat=True)
        if ok:
            return True, f"Replay saved ({int(dur)} s): {out_path}"
        return False, f"Could not build replay: {err}"
    except Exception as e:
        return False, f"Could not build replay: {e}"
    finally:
        buf.release(chosen)


# --------------------------------------------------------------------------
# Worker thread: grabs the screen, composes frames, feeds the encoders
# --------------------------------------------------------------------------
class Worker(threading.Thread):
    IDLE_FPS = 15

    def __init__(self, app):
        super().__init__(daemon=True)
        self.app = app
        self.latest = None
        self.rec = None
        self.buf = None
        self.alive = True

    def _do_save(self, buf):
        secs, out = buf.save_req
        buf.save_req = None
        try:
            chosen, dur = buf.snapshot(secs)
            chunks = [(i, i.snap_chunks()) for i in list(self.app.inputs)]

            def job():
                ok, msg = build_replay_clip(buf, chosen, dur, chunks, out)
                self.app.q.put(("replay_done", ok, msg))
            threading.Thread(target=job, daemon=True).start()
        except Exception as e:
            self.app.q.put(("replay_done", False, f"Replay save failed: {e}"))

    def run(self):
        if IS_WIN:
            try:
                import ctypes
                ctypes.windll.winmm.timeBeginPeriod(1)   # accurate sleeps
            except Exception:
                pass
        q = self.app.q
        try:
            with mss.mss() as sct:
                while self.alive:
                    t_start = time.perf_counter()
                    rec, buf = self.rec, self.buf
                    if rec is not None and rec.stop_flag:
                        rec.code = rec.sink.close()
                        rec.finished.set()
                        self.rec = rec = None
                    if buf is not None and buf.stop_flag:
                        buf.shutdown()
                        self.buf = buf = None
                    if buf is not None and buf.save_req is not None:
                        self._do_save(buf)

                    cfg = self.app.cfg
                    mons = sct.monitors
                    mi = cfg["monitor"] if cfg["monitor"] < len(mons) else 1
                    try:
                        shot = sct.grab(mons[mi])
                    except Exception:
                        time.sleep(0.05)
                        continue
                    screen = cv2.cvtColor(np.asarray(shot), cv2.COLOR_BGRA2BGR)
                    src = rec or buf
                    W, H = (src.w, src.h) if src else FORMATS[cfg["format"]]
                    cam = self.app.cam.get() if cfg["uses_cam"] else None
                    elapsed = rec.tl.written / rec.fps if rec else 0.0
                    frame = compose(screen, cam, cfg, W, H, elapsed)
                    self.latest = frame

                    now = time.perf_counter()
                    nexts, data = [], None
                    if rec is not None:
                        n = rec.tl.due(now)
                        if n:
                            data = data or frame.tobytes()
                            try:
                                for _ in range(n):
                                    rec.sink.write(data)
                            except Exception as e:
                                rec.error = f"Encoder stopped: {e}\n{rec.sink.log_tail()}"
                                rec.stop_flag = True
                                q.put(("autostop",))
                            rec.tl.written += n
                        nexts.append(rec.tl.next_time())
                    if buf is not None:
                        n = buf.tl.due(now)
                        if n:
                            data = data or frame.tobytes()
                            try:
                                buf.write(data, n)
                            except Exception as e:
                                buf.error = f"Replay buffer encoder stopped: {e}\n{buf.cur.sink.log_tail()}"
                                buf.stop_flag = True
                                q.put(("buf_error", buf.error))
                            buf.tl.written += n
                        nexts.append(buf.tl.next_time())
                    if nexts:
                        delay = min(nexts) - time.perf_counter()
                    else:
                        delay = 1 / self.IDLE_FPS - (time.perf_counter() - t_start)
                    if delay > 0:
                        time.sleep(delay)
        except Exception as e:
            q.put(("crash", repr(e)))


# --------------------------------------------------------------------------
# Key names
# --------------------------------------------------------------------------
def keyname(key):
    ch = getattr(key, "char", None)
    if ch:
        return ch.lower()
    n = str(key)
    if n.startswith("Key."):
        return n[4:]
    vk = getattr(key, "vk", None)
    return f"vk{vk}" if vk is not None else n


# --------------------------------------------------------------------------
# GUI
# --------------------------------------------------------------------------
BG, PANEL, FG, MUTED, ACC = "#1b1c21", "#272930", "#ececf0", "#9aa0ac", "#e5484d"


class TextVars:
    def __init__(self, d):
        self.enabled = tk.BooleanVar(value=d["enabled"])
        self.text = tk.StringVar(value=d["text"])
        self.font = tk.StringVar(value=d["font"])
        self.size = tk.DoubleVar(value=d["size"])
        self.color = tk.StringVar(value=d["color"])
        self.pos = tk.StringVar(value=d["pos"])
        self.bg = tk.BooleanVar(value=d["bg"])
        self.bg_alpha = tk.DoubleVar(value=d["bg_alpha"])
        self.outline = tk.BooleanVar(value=d["outline"])

    def all(self):
        return [self.enabled, self.text, self.font, self.size, self.color, self.pos,
                self.bg, self.bg_alpha, self.outline]

    def as_dict(self, fonts):
        c = self.color.get().lstrip("#")
        rgb = tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))
        return dict(enabled=self.enabled.get(), text=self.text.get(),
                    font=fonts.get(self.font.get(), ""), size=self.size.get(), color=rgb,
                    pos=self.pos.get(), bg=self.bg.get(), bg_alpha=self.bg_alpha.get() / 100.0,
                    outline=self.outline.get())


class App:
    PREVIEW_W, PREVIEW_H = 440, 500

    def __init__(self, root):
        self.root = root
        root.title("Game Recorder")
        root.resizable(False, False)
        self.q = queue.Queue()
        self.cam = Webcam()
        self.cam.start()
        self.recording = False
        self.buffer_on = False
        self.rec = None
        self.inputs = []
        self.lockable = []
        self.cfg = {}
        self.keys = {}
        self.mic_held = False
        self.mic_toggled = False
        self._down = set()
        self.capture_target = None
        self.key_buttons = {}

        self.fonts = find_fonts()
        with mss.mss() as s:
            self.monitor_map = {f"Monitor {i} ({m['width']}x{m['height']})": i
                                for i, m in enumerate(s.monitors) if i > 0}
        self.audio_list, default_mic = list_audio_inputs()
        self.audio_list = ["None"] + self.audio_list
        first_font = next(iter(self.fonts))

        # variables
        self.v_format = tk.StringVar(value=list(FORMATS)[0])
        self.v_fps = tk.StringVar(value="30")
        self.v_monitor = tk.StringVar(value=list(self.monitor_map)[0])
        self.v_layout = tk.StringVar(value=L_OVERLAY)
        self.v_fit = tk.StringVar(value=FIT_BLUR)
        self.v_focus = tk.DoubleVar(value=50)
        self.v_split = tk.DoubleVar(value=60)
        self.v_gpu = tk.BooleanVar(value=False)
        self.v_min = tk.BooleanVar(value=True)
        self.v_folder = tk.StringVar(value=os.path.join(os.path.expanduser("~"), "Videos", "GameRecordings"))
        self.v_cam_on = tk.BooleanVar(value=True)
        self.v_cam_idx = tk.StringVar(value="Camera 0")
        self.v_shape = tk.StringVar(value="Rectangle")
        self.v_mirror = tk.BooleanVar(value=False)
        self.v_corner = tk.StringVar(value=CORNERS[0])
        self.v_size = tk.DoubleVar(value=25)
        self.v_mic_dev = tk.StringVar(value=default_mic or "None")
        self.v_mic_mode = tk.StringVar(value=MIC_HOLD)
        self.v_desk_dev = tk.StringVar(value="None")
        self.v_key_mic = tk.StringVar(value="f8")
        self.v_key_rec = tk.StringVar(value="f9")
        self.v_key_save = tk.StringVar(value="f10")
        self.v_replay = tk.BooleanVar(value=False)
        self.v_replay_secs = tk.IntVar(value=30)
        self.texts = [
            TextVars(dict(enabled=False, text="MY GAMEPLAY", font=first_font, size=90, color="#ffffff",
                          pos="Bottom center", bg=False, bg_alpha=55, outline=True)),
            TextVars(dict(enabled=False, text="REC {timer}", font=first_font, size=50, color="#ff4d4d",
                          pos="Top left", bg=True, bg_alpha=55, outline=False)),
        ]
        self.v_status = tk.StringVar(value="")
        self.v_msg = tk.StringVar(value="Ready.")
        self.v_mic_state = tk.StringVar(value="")

        self.refresh_keys()
        self.setup_style()
        self.build_ui()
        traced = [self.v_format, self.v_fps, self.v_monitor, self.v_layout, self.v_fit, self.v_focus,
                  self.v_split, self.v_cam_on, self.v_cam_idx, self.v_shape, self.v_mirror,
                  self.v_corner, self.v_size]
        for tv in self.texts:
            traced += tv.all()
        for v in traced:
            v.trace_add("write", self.refresh_cfg)
        for v in (self.v_key_mic, self.v_key_rec, self.v_key_save, self.v_mic_mode):
            v.trace_add("write", self.refresh_keys)
        self.refresh_cfg()
        self.refresh_keys()

        self.worker = Worker(self)
        self.worker.start()
        self.start_listener()
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.after(80, self.poll)

    # ---------------- styling ----------------
    def setup_style(self):
        r = self.root
        r.configure(bg=BG)
        st = ttk.Style()
        try:
            st.theme_use("clam")
        except tk.TclError:
            pass
        st.configure(".", background=BG, foreground=FG, fieldbackground=PANEL, bordercolor="#3a3d46",
                     lightcolor=BG, darkcolor=BG, troughcolor=PANEL, focuscolor=BG)
        st.configure("TFrame", background=BG)
        st.configure("TLabel", background=BG, foreground=FG)
        st.configure("Muted.TLabel", foreground=MUTED)
        st.configure("TLabelframe", background=BG, bordercolor="#3a3d46")
        st.configure("TLabelframe.Label", background=BG, foreground=MUTED)
        st.configure("TCheckbutton", background=BG, foreground=FG)
        st.map("TCheckbutton", background=[("active", BG)])
        st.configure("TNotebook", background=BG, borderwidth=0)
        st.configure("TNotebook.Tab", background=PANEL, foreground=MUTED, padding=(12, 6))
        st.map("TNotebook.Tab", background=[("selected", "#3a3d46")], foreground=[("selected", FG)])
        st.configure("TButton", background="#3a3d46", foreground=FG, padding=6)
        st.map("TButton", background=[("active", "#4a4e5a"), ("disabled", "#2a2c33")],
               foreground=[("disabled", "#6b7080")])
        st.configure("Rec.TButton", background=ACC, foreground="white", font=("Segoe UI", 11, "bold"), padding=8)
        st.map("Rec.TButton", background=[("active", "#f0595e"), ("disabled", "#6b3335")])
        st.configure("Save.TButton", background="#2f6fed", foreground="white", font=("Segoe UI", 10, "bold"), padding=6)
        st.map("Save.TButton", background=[("active", "#4a86ff"), ("disabled", "#2a3350")])
        st.map("TCombobox", fieldbackground=[("readonly", PANEL), ("disabled", "#202228")],
               foreground=[("readonly", FG), ("disabled", "#6b7080")], selectbackground=[("readonly", PANEL)],
               selectforeground=[("readonly", FG)])
        st.configure("TEntry", fieldbackground=PANEL, foreground=FG, insertcolor=FG)
        st.configure("TSpinbox", fieldbackground=PANEL, foreground=FG, arrowcolor=FG)
        st.configure("Horizontal.TProgressbar", background="#3ecf6e", troughcolor=PANEL)
        r.option_add("*TCombobox*Listbox.background", PANEL)
        r.option_add("*TCombobox*Listbox.foreground", FG)
        r.option_add("*TCombobox*Listbox.selectBackground", "#3a3d46")

    # ---------------- UI ----------------
    def combo(self, parent, var, values, width=34, lock=False):
        cb = ttk.Combobox(parent, textvariable=var, values=values, state="readonly", width=width)
        if lock:
            self.lockable.append(cb)
        return cb

    def row(self, parent, r, label, widget, **grid):
        ttk.Label(parent, text=label).grid(row=r, column=0, sticky="w", pady=3, padx=(0, 8))
        widget.grid(row=r, column=1, sticky="w", pady=3, **grid)

    def key_row(self, parent, r, label, target, var):
        ttk.Label(parent, text=label).grid(row=r, column=0, sticky="w", pady=3, padx=(0, 8))
        f = ttk.Frame(parent)
        f.grid(row=r, column=1, sticky="w")
        ttk.Label(f, textvariable=var, width=10, anchor="center", relief="groove").pack(side="left")
        b = ttk.Button(f, text="Change key", command=lambda: self.begin_capture(target))
        b.pack(side="left", padx=6)
        self.key_buttons[target] = b

    def build_ui(self):
        left = ttk.Frame(self.root, padding=(10, 10, 4, 10))
        left.grid(row=0, column=0, sticky="n")
        right = ttk.Frame(self.root, padding=(4, 10, 10, 10))
        right.grid(row=0, column=1, sticky="n")
        nb = ttk.Notebook(left)
        nb.pack()

        # ---- Video tab ----
        t = ttk.Frame(nb, padding=10)
        nb.add(t, text="Video")
        self.row(t, 0, "Format", self.combo(t, self.v_format, list(FORMATS), lock=True))
        self.row(t, 1, "FPS", self.combo(t, self.v_fps, ["24", "30", "60"], width=8, lock=True))
        self.row(t, 2, "Screen", self.combo(t, self.v_monitor, list(self.monitor_map)))
        self.row(t, 3, "Layout", self.combo(t, self.v_layout, LAYOUTS))
        self.row(t, 4, "Screen fit", self.combo(t, self.v_fit, FIT_MODES))
        self.row(t, 5, "Crop focus (L-R)", ttk.Scale(t, from_=0, to=100, variable=self.v_focus, length=240))
        self.row(t, 6, "Stacked: screen %", ttk.Scale(t, from_=35, to=80, variable=self.v_split, length=240))
        c = ttk.Checkbutton(t, text="Use NVIDIA GPU encoder (NVENC)", variable=self.v_gpu)
        c.grid(row=7, column=0, columnspan=2, sticky="w", pady=3)
        self.lockable.append(c)
        f = ttk.Frame(t)
        f.grid(row=8, column=0, columnspan=2, sticky="w", pady=(10, 0))
        ttk.Label(f, text="Save to").pack(anchor="w")
        e = ttk.Entry(f, textvariable=self.v_folder, width=38)
        e.pack(side="left")
        self.lockable.append(e)
        ttk.Button(f, text="Browse", command=self.browse).pack(side="left", padx=4)
        ttk.Button(f, text="Open", command=self.open_folder).pack(side="left")
        ttk.Checkbutton(t, text="Minimize this window when recording starts",
                        variable=self.v_min).grid(row=9, column=0, columnspan=2, sticky="w", pady=(8, 0))

        # ---- Webcam tab ----
        t = ttk.Frame(nb, padding=10)
        nb.add(t, text="Webcam")
        ttk.Checkbutton(t, text="Webcam on", variable=self.v_cam_on).grid(row=0, column=0, sticky="w", pady=3)
        self.combo(t, self.v_cam_idx, [f"Camera {i}" for i in range(5)], width=12).grid(row=0, column=1, sticky="w")
        ttk.Checkbutton(t, text="Mirror image", variable=self.v_mirror).grid(row=1, column=0, columnspan=2, sticky="w", pady=3)
        self.row(t, 2, "Shape", self.combo(t, self.v_shape, SHAPES, width=14))
        self.row(t, 3, "Corner", self.combo(t, self.v_corner, CORNERS, width=14))
        self.row(t, 4, "Overlay size %", ttk.Scale(t, from_=10, to=50, variable=self.v_size, length=220))
        ttk.Label(t, text="Used by the 'overlay' layout. Stacked / webcam-only\nlayouts fill their area automatically.",
                  style="Muted.TLabel").grid(row=5, column=0, columnspan=2, sticky="w", pady=(10, 0))

        # ---- Text tab ----
        t = ttk.Frame(nb, padding=6)
        nb.add(t, text="Text")
        for i, tv in enumerate(self.texts):
            self.build_text_frame(t, tv, f"Text {i + 1}").pack(fill="x", pady=3)
        ttk.Label(t, text="Tokens: {time}  {date}  {timer} (recording time).  \\n = new line.",
                  style="Muted.TLabel").pack(anchor="w")

        # ---- Audio tab ----
        t = ttk.Frame(nb, padding=10)
        nb.add(t, text="Audio / Mic")
        self.row(t, 0, "Microphone", self.combo(t, self.v_mic_dev, self.audio_list, lock=True))
        self.row(t, 1, "Mic mode", self.combo(t, self.v_mic_mode, MIC_MODES))
        self.key_row(t, 2, "Mic key", "mic", self.v_key_mic)
        self.mic_lbl = tk.Label(t, textvariable=self.v_mic_state, bg=BG, fg=MUTED, font=("Segoe UI", 10, "bold"))
        self.mic_lbl.grid(row=3, column=0, columnspan=2, sticky="w", pady=(8, 0))
        self.mic_bar = ttk.Progressbar(t, maximum=100, length=300)
        self.mic_bar.grid(row=4, column=0, columnspan=2, sticky="w", pady=3)
        self.row(t, 5, "Desktop audio", self.combo(t, self.v_desk_dev, self.audio_list, lock=True))
        ttk.Label(t, text=("Push-to-talk: the mic is recorded only while you hold the key;\n"
                           "otherwise silence. Toggle: press once = live, again = muted.\n"
                           "Desktop audio is always on (needs 'Stereo Mix' or VB-Cable).\n"
                           "The key still reaches your game, so pick one it doesn't use."),
                  style="Muted.TLabel").grid(row=6, column=0, columnspan=2, sticky="w", pady=(10, 0))
        if sd is None:
            ttk.Label(t, text="sounddevice not installed - no audio.", foreground=ACC).grid(
                row=7, column=0, columnspan=2, sticky="w")

        # ---- Replay & hotkeys tab ----
        t = ttk.Frame(nb, padding=10)
        nb.add(t, text="Replay / Keys")
        ttk.Checkbutton(t, text="Enable replay buffer (always keeps the last seconds)",
                        variable=self.v_replay, command=self.on_replay_toggle).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=3)
        sp = ttk.Spinbox(t, from_=10, to=120, increment=5, textvariable=self.v_replay_secs, width=6)
        self.lockable.append(sp)
        self.row(t, 1, "Seconds to keep", sp)
        self.key_row(t, 2, "Save replay key", "save", self.v_key_save)
        self.key_row(t, 3, "Start/Stop rec key", "rec", self.v_key_rec)
        ttk.Label(t, text=("The buffer encodes video continuously in the background\n"
                           "(uses CPU/GPU like a recording). Saved clips are a few\n"
                           "seconds longer than the chosen time and land in your\n"
                           "save folder as replay_<date>.mp4. Replay includes your\n"
                           "overlays, webcam and mic audio. Format/FPS are locked\n"
                           "while the buffer is on."),
                  style="Muted.TLabel").grid(row=4, column=0, columnspan=2, sticky="w", pady=(12, 0))

        # ---- Right side ----
        self.pf = tk.Frame(right, width=self.PREVIEW_W, height=self.PREVIEW_H, bg="#0d0d10")
        self.pf.pack()
        self.pf.pack_propagate(False)
        self.preview = tk.Label(self.pf, bg="#0d0d10")
        self.preview.pack(expand=True)

        self.btn = ttk.Button(right, style="Rec.TButton", command=self.toggle)
        self.btn.pack(fill="x", pady=(8, 3))
        self.btn_save = ttk.Button(right, style="Save.TButton", command=self.request_save)
        self.btn_save.pack(fill="x", pady=3)
        self.mic_lbl2 = tk.Label(right, textvariable=self.v_mic_state, bg=BG, fg=MUTED, anchor="w",
                                 font=("Segoe UI", 10, "bold"))
        self.mic_lbl2.pack(fill="x", pady=(6, 0))
        ttk.Label(right, textvariable=self.v_status, wraplength=self.PREVIEW_W).pack(anchor="w", pady=(4, 0))
        ttk.Label(right, textvariable=self.v_msg, wraplength=self.PREVIEW_W, style="Muted.TLabel").pack(anchor="w")
        self.update_button_labels()

    def build_text_frame(self, parent, tv, title):
        lf = ttk.LabelFrame(parent, text=title, padding=6)
        r1 = ttk.Frame(lf)
        r1.pack(fill="x")
        ttk.Checkbutton(r1, text="Show", variable=tv.enabled).pack(side="left")
        ttk.Entry(r1, textvariable=tv.text, width=36).pack(side="left", padx=6)
        r2 = ttk.Frame(lf)
        r2.pack(fill="x", pady=2)
        self.combo(r2, tv.font, list(self.fonts), width=18).pack(side="left")
        self.combo(r2, tv.pos, TEXT_POS, width=13).pack(side="left", padx=6)

        def pick():
            c = colorchooser.askcolor(color=tv.color.get(), title="Text colour")
            if c and c[1]:
                tv.color.set(c[1])
                sw.configure(bg=c[1])
        sw = tk.Button(r2, width=3, bg=tv.color.get(), command=pick, relief="flat")
        sw.pack(side="left")
        r3 = ttk.Frame(lf)
        r3.pack(fill="x", pady=2)
        ttk.Label(r3, text="Size").pack(side="left")
        ttk.Scale(r3, from_=20, to=220, variable=tv.size, length=110).pack(side="left", padx=4)
        ttk.Checkbutton(r3, text="Outline", variable=tv.outline).pack(side="left", padx=4)
        ttk.Checkbutton(r3, text="Box", variable=tv.bg).pack(side="left")
        ttk.Scale(r3, from_=10, to=100, variable=tv.bg_alpha, length=60).pack(side="left", padx=4)
        return lf

    # ---------------- config snapshots (read by worker thread) ----------------
    def refresh_cfg(self, *_):
        try:
            layout = self.v_layout.get()
            uses_cam = self.v_cam_on.get() and layout != L_SCREEN
            idx = int(self.v_cam_idx.get().split()[-1])
            self.cfg = dict(
                format=self.v_format.get(), fps=int(self.v_fps.get()),
                monitor=self.monitor_map.get(self.v_monitor.get(), 1),
                layout=layout, fit=self.v_fit.get(), focus=self.v_focus.get() / 100.0,
                mirror=self.v_mirror.get(), shape=self.v_shape.get(),
                corner=self.v_corner.get(), size=self.v_size.get(),
                split=self.v_split.get(), uses_cam=uses_cam,
                texts=[tv.as_dict(self.fonts) for tv in self.texts])
            self.cam.desired = idx if uses_cam else None
        except (tk.TclError, ValueError):
            pass

    def refresh_keys(self, *_):
        self.keys = dict(rec=self.v_key_rec.get(), save=self.v_key_save.get(),
                         mic=self.v_key_mic.get(), mic_mode=self.v_mic_mode.get())
        if self.keys["mic_mode"] != MIC_TOGGLE:
            self.mic_toggled = False
        if hasattr(self, "btn"):
            self.update_button_labels()

    def update_button_labels(self):
        k = self.keys
        if self.recording:
            self.btn.configure(text=f"■  Stop recording  ({k['rec'].upper()})")
        else:
            self.btn.configure(text=f"●  Start recording  ({k['rec'].upper()})")
        self.btn_save.configure(text=f"⟲  Save replay  ({k['save'].upper()})")

    def mic_open(self):
        m = self.keys.get("mic_mode")
        if m == MIC_ALWAYS:
            return True
        if m == MIC_HOLD:
            return self.mic_held
        if m == MIC_TOGGLE:
            return self.mic_toggled
        return False

    # ---------------- keys ----------------
    def start_listener(self):
        self.listener = None
        if not pynput_keyboard:
            self.v_msg.set("pynput not installed - global hotkeys / push-to-talk disabled.")
            return

        def on_press(key):
            name = keyname(key)
            if name in self._down:
                return
            self._down.add(name)
            tgt = self.capture_target
            if tgt:
                self.capture_target = None
                self.q.put(("setkey", tgt, name))
                return
            k = self.keys
            if name == k.get("rec"):
                self.q.put(("toggle",))
            elif name == k.get("save"):
                self.q.put(("save",))
            if name == k.get("mic"):
                if k.get("mic_mode") == MIC_HOLD:
                    self.mic_held = True
                elif k.get("mic_mode") == MIC_TOGGLE:
                    self.mic_toggled = not self.mic_toggled

        def on_release(key):
            name = keyname(key)
            self._down.discard(name)
            if name == self.keys.get("mic"):
                self.mic_held = False

        try:
            self.listener = pynput_keyboard.Listener(on_press=on_press, on_release=on_release)
            self.listener.daemon = True
            self.listener.start()
        except Exception as e:
            self.v_msg.set(f"Keyboard hooks unavailable: {e}")

    def begin_capture(self, target):
        if not self.listener:
            messagebox.showinfo("Keys", "pynput is not installed, so keys cannot be captured.")
            return
        self.capture_target = target
        self.key_buttons[target].configure(text="Press a key...")

    # ---------------- helpers ----------------
    def browse(self):
        d = filedialog.askdirectory(initialdir=self.v_folder.get())
        if d:
            self.v_folder.set(d)

    def open_folder(self):
        p = self.v_folder.get()
        os.makedirs(p, exist_ok=True)
        if IS_WIN:
            os.startfile(p)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", p])
        else:
            subprocess.Popen(["xdg-open", p])

    def refresh_locks(self):
        locked = self.recording or self.buffer_on
        for w in self.lockable:
            if locked:
                w.configure(state="disabled")
            else:
                w.configure(state="readonly" if isinstance(w, ttk.Combobox) else "normal")

    # ---------------- audio management ----------------
    def ensure_audio(self):
        if self.inputs or sd is None:
            return
        used = set()
        for name, var, gated in (("mic", self.v_mic_dev, True), ("desktop", self.v_desk_dev, False)):
            v = var.get()
            if v == "None" or v in used:
                continue
            used.add(v)
            try:
                inp = AudioInput(int(v.split(":")[0]), name, self.mic_open if gated else None)
                inp.start()
                self.inputs.append(inp)
            except Exception as e:
                messagebox.showwarning("Audio", f"Could not open '{v}':\n{e}\n\nContinuing without it.")

    def release_audio(self):
        if self.recording:
            return
        if self.buffer_on:
            return
        for inp in self.inputs:
            try:
                inp.stop()
            except Exception:
                pass
        self.inputs = []

    # ---------------- recording ----------------
    def toggle(self):
        if self.recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        cfg = self.cfg
        W, H = FORMATS[cfg["format"]]
        fps = cfg["fps"]
        folder = self.v_folder.get()
        try:
            os.makedirs(folder, exist_ok=True)
        except Exception as e:
            messagebox.showerror("Folder", f"Cannot use folder:\n{e}")
            return
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        final = os.path.join(folder, f"game_{stamp}.mp4")
        tmp = tempfile.gettempdir()

        self.ensure_audio()
        video_path = final if not self.inputs else os.path.join(tmp, f"gr_{stamp}_video.mp4")
        try:
            sink = VideoSink(video_path, W, H, fps, self.v_gpu.get(),
                             os.path.join(tmp, f"gr_{stamp}_ffmpeg.log"))
        except Exception as e:
            self.release_audio()
            messagebox.showerror("Encoder", f"Could not start ffmpeg:\n{e}")
            return

        rec = Rec(sink, W, H, fps)
        rec.final, rec.video_path = final, video_path
        for inp in self.inputs:
            p = os.path.join(tmp, f"gr_{stamp}_{inp.name}.wav")
            inp.begin_wav(p)
            rec.audio.append(p)
        self.rec = rec
        self.rec_start = time.time()
        self.worker.rec = rec
        self.recording = True
        self.refresh_locks()
        self.update_button_labels()
        if self.v_min.get():
            self.root.iconify()

    def stop_recording(self):
        if not self.recording:
            return
        rec = self.rec
        for inp in self.inputs:
            inp.end_wav()
        self.recording = False
        rec.stop_flag = True
        self.release_audio()
        self.update_button_labels()
        self.btn.configure(state="disabled")
        self.v_msg.set("Saving video...")
        self.root.deiconify()
        threading.Thread(target=self.finalize, args=(rec,), daemon=True).start()

    def finalize(self, rec):
        rec.finished.wait(30)
        secs = rec.tl.written / rec.fps if rec.fps else 0
        if rec.error or rec.code not in (0, None):
            msg = rec.error or f"ffmpeg exited with code {rec.code}\n{rec.sink.log_tail()}"
            self.q.put(("done", False, f"Recording failed:\n{msg}\n(If you use the GPU encoder, untick it and retry.)"))
            return
        if rec.audio:
            ok, err = mux(rec.video_path, rec.audio, rec.final)
            if ok:
                for p in [rec.video_path] + rec.audio:
                    try:
                        os.remove(p)
                    except Exception:
                        pass
            else:
                try:
                    os.replace(rec.video_path, rec.final)
                except Exception:
                    pass
                self.q.put(("done", True, f"Saved WITHOUT audio (mux failed): {rec.final}\n{err}"))
                return
        self.q.put(("done", True, f"Saved ({int(secs) // 60}:{int(secs) % 60:02d}): {rec.final}"))

    # ---------------- replay buffer ----------------
    def on_replay_toggle(self):
        if self.v_replay.get():
            self.start_buffer()
        else:
            self.stop_buffer()

    def start_buffer(self):
        if self.buffer_on:
            return
        cfg = self.cfg
        W, H = FORMATS[cfg["format"]]
        try:
            secs = max(5, min(300, int(float(self.v_replay_secs.get()))))
        except (tk.TclError, ValueError):
            secs = 30
        self.ensure_audio()
        for inp in self.inputs:
            inp.enable_ring(secs)
        workdir = os.path.join(tempfile.gettempdir(), f"gr_replay_{os.getpid()}_{int(time.time())}")
        try:
            buf = Buffer(W, H, cfg["fps"], self.v_gpu.get(), secs, workdir)
        except Exception as e:
            messagebox.showerror("Replay buffer", f"Could not start the encoder:\n{e}")
            self.v_replay.set(False)
            self.release_audio()
            return
        self.buf_secs = secs
        self.buffer_on = True
        self.worker.buf = buf
        self.refresh_locks()

    def stop_buffer(self):
        if not self.buffer_on:
            return
        self.buffer_on = False
        if self.worker.buf is not None:
            self.worker.buf.stop_flag = True
        for inp in self.inputs:
            inp.disable_ring()
        self.release_audio()
        self.refresh_locks()

    def request_save(self):
        buf = self.worker.buf
        if not self.buffer_on or buf is None:
            self.v_msg.set("Turn on the replay buffer first (Replay / Keys tab).")
            return
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = self.v_folder.get()
        try:
            os.makedirs(folder, exist_ok=True)
        except Exception as e:
            self.v_msg.set(f"Cannot use folder: {e}")
            return
        buf.save_req = (self.buf_secs, os.path.join(folder, f"replay_{stamp}.mp4"))
        self.v_msg.set("Saving replay...")

    # ---------------- main-thread polling ----------------
    def poll(self):
        try:
            while True:
                item = self.q.get_nowait()
                kind = item[0]
                if kind == "toggle":
                    if str(self.btn["state"]) != "disabled":
                        self.toggle()
                elif kind == "save":
                    self.request_save()
                elif kind == "setkey":
                    var = {"mic": self.v_key_mic, "rec": self.v_key_rec, "save": self.v_key_save}[item[1]]
                    var.set(item[2])
                    self.key_buttons[item[1]].configure(text="Change key")
                elif kind == "autostop":
                    if self.recording:
                        self.stop_recording()
                elif kind == "buf_error":
                    self.v_replay.set(False)
                    self.stop_buffer()
                    messagebox.showerror("Replay buffer", item[1] + "\n(If you use the GPU encoder, untick it and retry.)")
                elif kind == "done":
                    ok, msg = item[1], item[2]
                    self.v_msg.set(msg)
                    self.btn.configure(state="normal")
                    self.refresh_locks()
                    if not ok:
                        messagebox.showerror("Game Recorder", msg)
                elif kind == "replay_done":
                    self.v_msg.set(item[2])
                    if not item[1]:
                        messagebox.showwarning("Replay", item[2])
                elif kind == "crash":
                    messagebox.showerror("Game Recorder", f"Capture thread crashed:\n{item[1]}")
        except queue.Empty:
            pass

        parts = []
        if self.recording and self.rec:
            e = int(time.time() - self.rec_start)
            parts.append(f"● REC {e // 60:02d}:{e % 60:02d}  {self.rec.w}x{self.rec.h} @ {self.rec.fps} fps")
        if self.buffer_on:
            parts.append(f"⟲ Replay buffer ON ({self.buf_secs} s)")
        if not parts:
            parts.append("Idle")
        if self.cfg.get("uses_cam") and self.cam.error:
            parts.append(self.cam.error)
        self.v_status.set("   |   ".join(parts))

        live = self.mic_open() and self.v_mic_dev.get() != "None"
        self.v_mic_state.set("🎤 MIC LIVE" if live else "🎤 mic muted")
        col = "#3ecf6e" if live else MUTED
        self.mic_lbl.configure(fg=col)
        self.mic_lbl2.configure(fg=col)
        mic = next((i for i in self.inputs if i.name == "mic"), None)
        self.mic_bar["value"] = min(100, (mic.level * 140) if mic else 0)

        self.update_preview()
        self.root.after(80, self.poll)

    def update_preview(self):
        if self.root.state() == "iconic":
            return
        frame = self.worker.latest
        if frame is None:
            return
        h, w = frame.shape[:2]
        s = min(self.PREVIEW_W / w, self.PREVIEW_H / h)
        small = cv2.resize(frame, (max(2, int(w * s)), max(2, int(h * s))), interpolation=cv2.INTER_AREA)
        img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
        self.preview.configure(image=img)
        self.preview.image = img

    def on_close(self):
        if self.recording:
            if not messagebox.askyesno("Quit", "Recording is running. Stop and quit?"):
                return
            self.stop_recording()
            if self.rec:
                self.rec.finished.wait(10)
            time.sleep(1)
        if self.buffer_on:
            self.stop_buffer()
            time.sleep(0.5)
        for inp in self.inputs:
            try:
                inp.stop()
            except Exception:
                pass
        self.worker.alive = False
        self.cam.alive = False
        try:
            if self.listener:
                self.listener.stop()
        except Exception:
            pass
        self.root.destroy()


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()