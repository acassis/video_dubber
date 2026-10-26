#!/usr/bin/env python3
"""Track every visible face and measure its mouth, frame by frame.

usage: lips.py VIDEO OUT.npz

MediaPipe Face Landmarker (Tasks API, CPU) finds up to MAX_FACES faces per
frame and returns 478 landmarks and 52 blendshapes for each. For every face
this records:

  open    inner-lip gap / mouth width (0 = lips together)
  jaw     blendshape jawOpen
  round   max(mouthFunnel, mouthPucker): rounded lips (o, u, w)
  close   blendshape mouthClose (lips pressed: p, b, m)
  size    face width / frame width (how big the face is: close-up or not)
  yaw     |left-right asymmetry| of the face, 0 = frontal, 1 = profile
  x, y    face centre, as fractions of the frame

The face detector squeezes its input into 128x128, so a 16:9 frame makes a
standing presenter's face only ~11 px wide. Each frame is therefore searched
as CROPS: two overlapping, nearly square halves, and a face found in both
is kept once. Faces are tied into tracks within a shot by their position.
Shots are cut where the picture changes abruptly. The whole video is read
through one ffmpeg pipe at ANALYSIS_W pixels wide.
"""
import json
import os
import subprocess
import sys
import time

import numpy as np

MODEL = os.environ.get("DUBBER_FACE_MODEL", os.path.expanduser(
    "~/.local/share/dubber/vision/face_landmarker.task"))
ANALYSIS_W = 1280
CROPS = ((0.0, 0.6), (0.4, 1.0))  # x ranges, fractions of the width
SAME = 0.03       # two detections closer than this are one face
MAX_FACES = 4
CUT = 28.0        # mean |grey difference| (0-255) on a 64x36 thumbnail = cut
TRACK_DIST = 0.10  # a face may move this far (frame fraction) between frames
TRACK_GAP = 45     # frames a track may miss before it ends

# landmarks: inner lips 13 (top) 14 (bottom), corners 78/308;
# face sides 234/454, nose tip 1
UP, LO, ML, MR, FL, FR, NOSE = 13, 14, 78, 308, 234, 454, 1


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,avg_frame_rate,nb_frames:format=duration",
         "-of", "json", path], capture_output=True, text=True, check=True)
    j = json.loads(out.stdout)
    s = j["streams"][0]
    num, den = s["avg_frame_rate"].split("/")
    return int(s["width"]), int(s["height"]), float(num) / float(den), \
        float(j["format"]["duration"])


def frames(path, w, h):
    p = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", path, "-map", "0:v:0", "-vf",
         f"scale={w}:{h}", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
        stdout=subprocess.PIPE, bufsize=w * h * 3 * 4)
    n = w * h * 3
    while True:
        buf = p.stdout.read(n)
        if len(buf) < n:
            break
        yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    p.wait()


def measure(lm, bs, x0, cw, w, h):
    """Mouth and face measurements of one face found in the crop starting at
    pixel x0, cw pixels wide, of a w x h frame; None if degenerate."""
    pt = np.array([(x0 + lm[i].x * cw, lm[i].y * h) for i in
            (UP, LO, ML, MR, FL, FR, NOSE)])
    width = np.linalg.norm(pt[3] - pt[2])
    face = np.linalg.norm(pt[5] - pt[4])
    if width < 1 or face < 1:
        return None
    b = {c.category_name: c.score for c in bs}
    dl = np.linalg.norm(pt[6] - pt[4])  # nose to left side
    dr = np.linalg.norm(pt[6] - pt[5])
    return dict(
        open=float(np.linalg.norm(pt[1] - pt[0]) / width),
        jaw=float(b.get("jawOpen", 0)),
        round=float(max(b.get("mouthFunnel", 0), b.get("mouthPucker", 0))),
        close=float(b.get("mouthClose", 0)),
        size=float(face / w),
        yaw=float(abs(dl - dr) / (dl + dr)),
        x=float((pt[4, 0] + pt[5, 0]) / 2 / w),
        y=float(pt[6, 1] / h))


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src, out = sys.argv[1], sys.argv[2]
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    W, H, fps, dur = probe(src)
    w = ANALYSIS_W
    h = int(round(H * w / W / 2)) * 2
    opts = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL),
        running_mode=vision.RunningMode.VIDEO, num_faces=MAX_FACES,
        output_face_blendshapes=True,
        min_face_detection_confidence=0.5, min_tracking_confidence=0.5)
    crops = [(int(a * w), int(b * w)) for a, b in CROPS]
    dets = [vision.FaceLandmarker.create_from_options(opts) for _ in crops]

    keys = ("open", "jaw", "round", "close", "size", "yaw", "x", "y")
    rows = []       # (frame, track, *keys)
    shots = [0]
    tracks = {}     # live track id -> (x, y, last frame)
    next_id = 0
    prev = None
    t0 = time.time()
    n = 0
    for n, img in enumerate(frames(src, w, h)):
        thumb = img[::h // 36, ::w // 64].mean(axis=2)
        if prev is not None and np.abs(thumb - prev).mean() > CUT:
            shots.append(n)
            tracks.clear()
        prev = thumb
        faces = []
        for det, (x0, x1) in zip(dets, crops):
            res = det.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=np.ascontiguousarray(img[:, x0:x1])),
                int(round(n * 1000 / fps)))
            for lm, bs in zip(res.face_landmarks,
                              res.face_blendshapes or []):
                m = measure(lm, bs, x0, x1 - x0, w, h)
                if m and all(np.hypot(m["x"] - f["x"], m["y"] - f["y"])
                             > SAME for f in faces):
                    faces.append(m)
        # greedy nearest-centre matching, largest faces first
        used = set()
        for m in sorted(faces, key=lambda m: -m["size"]):
            best, bd = None, TRACK_DIST
            for tid, (x, y, last) in tracks.items():
                d = np.hypot(m["x"] - x, m["y"] - y)
                if tid not in used and n - last <= TRACK_GAP and d < bd:
                    best, bd = tid, d
            if best is None:
                best = next_id
                next_id += 1
            used.add(best)
            tracks[best] = (m["x"], m["y"], n)
            rows.append([n, best] + [m[k] for k in keys])
        if n % 1000 == 0 and n:
            el = time.time() - t0
            print(f"lips: {n / fps:6.0f}/{dur:.0f} s  "
                  f"({n / el:.0f} frames/s)", flush=True)
    for det in dets:
        det.close()
    r = np.array(rows, dtype=np.float32).reshape(-1, 2 + len(keys))
    np.savez_compressed(out, rows=r, keys=np.array(keys), fps=fps,
                frames=n + 1, shots=np.array(shots), width=W, height=H)
    print(f"lips: {n + 1} frames, {len(shots)} shots, {next_id} face tracks, "
          f"{len(r)} face measurements in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
