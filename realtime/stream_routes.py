from flask import Blueprint, Response, request, render_template
import cv2, os, time
from .yolo_tracker import process_frame

realtime_bp = Blueprint("realtime", __name__)
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "uploads", "video")
os.makedirs(UPLOAD_DIR, exist_ok=True)


def _gen_frames(source, fruit_type):
    cap = cv2.VideoCapture(source)
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = process_frame(frame, fruit_type, frame_idx)
        frame_idx += 1
        _, buf = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
    cap.release()


@realtime_bp.route("/realtime")
def realtime_page():
    return render_template("realtime.html")


@realtime_bp.route("/realtime/webcam_feed")
def webcam_feed():
    fruit_type = request.args.get("fruit", "apple")
    return Response(_gen_frames(0, fruit_type), mimetype="multipart/x-mixed-replace; boundary=frame")


@realtime_bp.route("/realtime/upload_video", methods=["POST"])
def upload_video():
    f = request.files["video"]
    path = os.path.join(UPLOAD_DIR, f.filename)
    f.save(path)
    return {"video_path": f.filename}


@realtime_bp.route("/realtime/video_feed/<filename>")
def video_feed(filename):
    fruit_type = request.args.get("fruit", "apple")
    path = os.path.join(UPLOAD_DIR, filename)
    return Response(_gen_frames(path, fruit_type), mimetype="multipart/x-mixed-replace; boundary=frame")