#!/usr/bin/env python3
"""Track every visible face and measure its mouth, frame by frame.

usage: lips.py VIDEO OUT.npz

MediaPipe Face Landmarker (Tasks API, CPU) returns 478 landmarks and 52
blendshapes per face. For every face this records:

  open    inner-lip gap / mouth width (0 = lips together)
  jaw     blendshape jawOpen
  round   max(mouthFunnel, mouthPucker): rounded lips (o, u, w)
  close   blendshape mouthClose (lips pressed: p, b, m)
  size    face width / frame width (how big the face is: close-up or not)
  yaw     |left-right asymmetry| of the face, 0 = frontal, 1 = profile
  x, y    face centre, as fractions of the frame

Two passes per frame:

* find: the frame at half size, searched as CROPS (two overlapping, nearly
  square halves; the detector squeezes its input into 128x128, so a 16:9
  frame makes a standing presenter's face only ~11 px wide). This only
  locates faces and follows them (VIDEO mode).
* measure: each face again, on a FINE x FINE crop cut from the
  full-resolution frame around it, so the mouth of a face 5% of the frame
  wide is seen with 1.5-4x more pixels. Measured on a TV panel, this helps
  less than expected: the lip gap's agreement with the voice rose from
  r=0.14 to 0.18, jawOpen's stayed at 0.19-0.20 (DESIGN.md section 11.1).

Faces are tied into tracks within a shot by their position; shots are cut
where the picture changes abruptly. The video is cut into JOBS time chunks
analysed in parallel processes; tracks are joined again at the seams.
"""
import json
import os
import subprocess
import sys
import time

import numpy as np

MODEL = os.environ.get("DUBBER_FACE_MODEL", os.path.expanduser(
    "~/.local/share/dubber/vision/face_landmarker.task"))
FIND_W = 960      # width of the frame used to find faces
CROPS = ((0.0, 0.6), (0.4, 1.0))  # x ranges, fractions of the width
FINE = 256        # side of the full-resolution crop a face is measured on
MARGIN = 1.7      # crop side / landmark extent of the face
SAME = 0.03       # two detections closer than this are one face
MAX_FACES = 4
CUT = 28.0        # mean |grey difference| (0-255) on a 64x36 thumbnail = cut
TRACK_DIST = 0.10  # a face may move this far (frame fraction) between frames
TRACK_GAP = 45     # frames a track may miss before it ends
KEYS = ("open", "jaw", "round", "close", "size", "yaw", "x", "y")

# landmarks: inner lips 13 (top) 14 (bottom), corners 78/308;
# face sides 234/454, nose tip 1
UP, LO, ML, MR, FL, FR, NOSE = 13, 14, 78, 308, 234, 454, 1


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,avg_frame_rate:format=duration",
         "-of", "json", path], capture_output=True, text=True, check=True)
    j = json.loads(out.stdout)
    s = j["streams"][0]
    num, den = s["avg_frame_rate"].split("/")
    return int(s["width"]), int(s["height"]), float(num) / float(den), \
        float(j["format"]["duration"])


def frames(path, W, H, start, count):
    """COUNT full-resolution RGB frames from frame START (accurate seek)."""
    p = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-ss", f"{start:.6f}", "-i", path,
         "-map", "0:v:0", "-frames:v", str(count), "-pix_fmt", "rgb24",
         "-f", "rawvideo", "-"], stdout=subprocess.PIPE, bufsize=W * H * 3 * 2)
    n = W * H * 3
    while True:
        buf = p.stdout.read(n)
        if len(buf) < n:
            break
        yield np.frombuffer(buf, np.uint8).reshape(H, W, 3)
    p.wait()


def points(lm, x0, y0, sx, sy):
    return np.array([(x0 + lm[i].x * sx, y0 + lm[i].y * sy) for i in
                     (UP, LO, ML, MR, FL, FR, NOSE)])


def mouth(lm, bs):
    """Scale-free mouth measurements of one landmark set."""
    pt = points(lm, 0, 0, 1, 1)
    width = np.linalg.norm(pt[3] - pt[2])
    if width <= 0:
        return None
    b = {c.category_name: c.score for c in bs}
    return dict(
        open=float(np.linalg.norm(pt[1] - pt[0]) / width),
        jaw=float(b.get("jawOpen", 0)),
        round=float(max(b.get("mouthFunnel", 0), b.get("mouthPucker", 0))),
        close=float(b.get("mouthClose", 0)))


def place(lm, x0, cw, w, h):
    """Where a face found in the crop at pixel x0 (cw wide) of the w x h
    finding frame is: size, turn, centre, and its landmark box."""
    pt = points(lm, x0, 0, cw, h)
    face = np.linalg.norm(pt[5] - pt[4])
    if face < 1:
        return None
    dl = np.linalg.norm(pt[6] - pt[4])  # nose to left side
    dr = np.linalg.norm(pt[6] - pt[5])
    xs = [x0 + p.x * cw for p in lm]
    ys = [p.y * h for p in lm]
    return dict(size=float(face / w), yaw=float(abs(dl - dr) / (dl + dr)),
                x=float((pt[4, 0] + pt[5, 0]) / 2 / w), y=float(pt[6, 1] / h),
                box=(min(xs) / w, min(ys) / h, max(xs) / w, max(ys) / h))


def fine_crop(img, box):
    """Square FINE x FINE crop of the full-resolution frame around BOX
    (fractions of the frame), padded with black at the borders."""
    import cv2
    H, W = img.shape[:2]
    cx, cy = (box[0] + box[2]) / 2 * W, (box[1] + box[3]) / 2 * H
    side = int(max((box[2] - box[0]) * W, (box[3] - box[1]) * H) * MARGIN)
    side = max(side, 16)
    x0, y0 = int(cx - side / 2), int(cy - side / 2)
    out = np.zeros((side, side, 3), np.uint8)
    a, b = max(x0, 0), max(y0, 0)
    c, d = min(x0 + side, W), min(y0 + side, H)
    if c <= a or d <= b:
        return None
    out[b - y0:d - y0, a - x0:c - x0] = img[b:d, a:c]
    return cv2.resize(out, (FINE, FINE), interpolation=cv2.INTER_AREA
                      if side > FINE else cv2.INTER_CUBIC)


def chunk(args):
    """Analyse frames [first, first + count): rows with chunk-local track
    ids, shot starts, and the thumbnails at both ends (to tell whether a
    seam falls on a cut)."""
    src, W, H, fps, first, count = args
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    w = FIND_W
    h = int(round(H * w / W / 2)) * 2
    find_opts = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL),
        running_mode=vision.RunningMode.VIDEO, num_faces=MAX_FACES,
        output_face_blendshapes=True,
        min_face_detection_confidence=0.5, min_tracking_confidence=0.5)
    fine_opts = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL),
        running_mode=vision.RunningMode.IMAGE, num_faces=1,
        output_face_blendshapes=True, min_face_detection_confidence=0.3)
    crops = [(int(a * w), int(b * w)) for a, b in CROPS]
    finders = [vision.FaceLandmarker.create_from_options(find_opts)
               for _ in crops]
    fine = vision.FaceLandmarker.create_from_options(fine_opts)
    import cv2
    rows, shots, tracks, next_id = [], [], {}, 0
    prev = thumb0 = None
    sharp = 0
    for k, full in enumerate(frames(src, W, H, first / fps, count)):
        n = first + k
        img = cv2.resize(full, (w, h), interpolation=cv2.INTER_AREA)
        thumb = img[::h // 36, ::w // 64].mean(axis=2)
        if thumb0 is None:
            thumb0 = thumb
        if prev is not None and np.abs(thumb - prev).mean() > CUT:
            shots.append(n)
            tracks.clear()
        prev = thumb
        faces = []
        for det, (x0, x1) in zip(finders, crops):
            res = det.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=np.ascontiguousarray(img[:, x0:x1])),
                int(round(k * 1000 / fps)))
            for lm, bs in zip(res.face_landmarks, res.face_blendshapes or []):
                p = place(lm, x0, x1 - x0, w, h)
                m = mouth(lm, bs)
                if p and m and all(np.hypot(p["x"] - f["x"], p["y"] - f["y"])
                                   > SAME for f in faces):
                    faces.append({**p, **m})
        for f in faces:     # measure the mouth again at full resolution
            c = fine_crop(full, f["box"])
            if c is None:
                continue
            r = fine.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                     data=np.ascontiguousarray(c)))
            if r.face_landmarks and r.face_blendshapes:
                m = mouth(r.face_landmarks[0], r.face_blendshapes[0])
                if m:
                    f.update(m)
                    sharp += 1
        used = set()        # greedy nearest-centre matching, largest first
        for f in sorted(faces, key=lambda f: -f["size"]):
            best, bd = None, TRACK_DIST
            for tid, (x, y, last) in tracks.items():
                d = np.hypot(f["x"] - x, f["y"] - y)
                if tid not in used and n - last <= TRACK_GAP and d < bd:
                    best, bd = tid, d
            if best is None:
                best = next_id
                next_id += 1
            used.add(best)
            tracks[best] = (f["x"], f["y"], n)
            rows.append([n, best] + [f[q] for q in KEYS])
    for d in finders + [fine]:
        d.close()
    return dict(first=first, count=count, rows=rows, shots=shots,
                ntracks=next_id, thumbs=(thumb0, prev), sharp=sharp)


def join(parts):
    """Number the tracks of all chunks globally and continue a track across
    a seam when the picture does not cut there and a face sits at the same
    place on both sides."""
    rows, shots, base = [], [0], 0
    prev_end = None     # {local id: (x, y)} at the last frame of the chunk
    prev_map = {}
    for p in parts:
        r = np.array(p["rows"], dtype=np.float32).reshape(-1, 2 + len(KEYS))
        ids = {}
        seam_cut = True
        if prev_end is not None and p["thumbs"][0] is not None and \
                prev_thumb is not None:
            seam_cut = np.abs(p["thumbs"][0] - prev_thumb).mean() > CUT
            if seam_cut:
                shots.append(p["first"])
        if not seam_cut and len(r):
            start = r[r[:, 0] == p["first"]]
            for row in start:
                for gid, (x, y) in prev_end.items():
                    if np.hypot(row[8] - x, row[9] - y) < TRACK_DIST and \
                            gid not in ids.values():
                        ids[int(row[1])] = gid
                        break
        for t in range(p["ntracks"]):
            if t not in ids:
                ids[t] = base
                base += 1
        if len(r):
            r[:, 1] = [ids[int(t)] for t in r[:, 1]]
        rows.append(r)
        shots += p["shots"]
        last = p["first"] + p["count"] - 1
        prev_end = {int(row[1]): (row[8], row[9]) for row in r if row[0] == last}
        prev_thumb = p["thumbs"][1]
    return np.concatenate(rows) if rows else np.zeros((0, 2 + len(KEYS))), \
        sorted(set(shots)), base


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src, out = sys.argv[1], sys.argv[2]
    W, H, fps, dur = probe(src)
    total = int(dur * fps)
    jobs = int(os.environ.get("DUBBER_LIPS_JOBS", 0)) or \
        max(1, min(8, (os.cpu_count() or 2) // 2))
    size = -(-total // jobs)
    work = [(src, W, H, fps, a, min(size, total - a))
            for a in range(0, total, size)]
    t0 = time.time()
    import multiprocessing as mpc
    with mpc.get_context("spawn").Pool(len(work)) as pool:
        parts = pool.map(chunk, work)
    rows, shots, ntracks = join(parts)
    frames_done = max(p["first"] + p["count"] for p in parts)
    np.savez_compressed(out, rows=rows.astype(np.float32), keys=np.array(KEYS),
                        fps=fps, frames=frames_done, shots=np.array(shots),
                        width=W, height=H)
    sharp = sum(p["sharp"] for p in parts)
    print(f"lips: {frames_done} frames, {len(shots)} shots, {ntracks} face "
          f"tracks, {len(rows)} face measurements ({sharp} at full "
          f"resolution) in {time.time() - t0:.0f} s ({jobs} processes)")


if __name__ == "__main__":
    main()
