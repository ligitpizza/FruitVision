from flask import Blueprint, Response, request, render_template, send_from_directory
import cv2, os
from .svm_yolo_tracker import process_frame, get_session_log, clear_session_log
from core_modules.pdf_report import generate_pdf_report_batch

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


@realtime_bp.route("/realtime/export_pdf", methods=["POST"])
def realtime_export_pdf():
    """Bundles every classification logged during the current real-time
    session into one combined PDF, the same way batch-analysis does."""
    session_results = get_session_log()
    if not session_results:
        return {"error": "No real-time classifications logged yet in this session."}, 400

    results_for_pdf = [
        {"filename": r["tag"], "label": r["label"], "confidence": r["confidence"], "image_path": r["image_path"]}
        for r in session_results
    ]
    out_path = generate_pdf_report_batch(results_for_pdf, model_tag="realtime_yolo")
    return send_from_directory(os.path.dirname(out_path), os.path.basename(out_path), as_attachment=True)


@realtime_bp.route("/realtime/clear_session", methods=["POST"])
def realtime_clear_session():
    clear_session_log()
    return {"status": "cleared"}