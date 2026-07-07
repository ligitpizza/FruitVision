from flask import Blueprint, Response, request, render_template, send_from_directory
import cv2, os
from .svm_yolo_tracker import (
    process_frame as svm_process_frame,
    get_session_log as svm_get_session_log,
    clear_session_log as svm_clear_session_log,
)
from .yolo_cls_tracker import (
    process_frame as yolo_cls_process_frame,
    get_session_log as yolo_cls_get_session_log,
    clear_session_log as yolo_cls_clear_session_log,
)
from core_modules.pdf_report import generate_pdf_report_batch

realtime_bp = Blueprint("realtime", __name__)
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "uploads", "video")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Engine registry: each entry is (process_frame, get_session_log,
# clear_session_log, pdf_model_tag). "svm" is the pre-existing default
# (COCO YOLO track + 4-member SVM ensemble); "yolo_cls" is the new
# fully-YOLO path (COCO YOLO track + pure-YOLO cls classification).
# Defaulting unknown/missing engine names to "svm" keeps every existing
# caller of these routes (that doesn't know about ?engine=) working exactly
# as before.
_ENGINES = {
    "svm": (svm_process_frame, svm_get_session_log, svm_clear_session_log, "realtime_yolo"),
    "yolo_cls": (yolo_cls_process_frame, yolo_cls_get_session_log, yolo_cls_clear_session_log, "yolo_pure_realtime"),
}


def _engine(name):
    return _ENGINES.get(name, _ENGINES["svm"])


def _gen_frames(source, fruit_type, engine_name):
    process_frame, _, _, _ = _engine(engine_name)
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
    engine_name = request.args.get("engine", "svm")
    return Response(
        _gen_frames(0, fruit_type, engine_name),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@realtime_bp.route("/realtime/upload_video", methods=["POST"])
def upload_video():
    f = request.files["video"]
    path = os.path.join(UPLOAD_DIR, f.filename)
    f.save(path)
    return {"video_path": f.filename}


@realtime_bp.route("/realtime/video_feed/<filename>")
def video_feed(filename):
    fruit_type = request.args.get("fruit", "apple")
    engine_name = request.args.get("engine", "svm")
    path = os.path.join(UPLOAD_DIR, filename)
    return Response(
        _gen_frames(path, fruit_type, engine_name),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@realtime_bp.route("/realtime/export_pdf", methods=["POST"])
def realtime_export_pdf():
    """Bundles every classification logged during the current real-time
    session into one combined PDF. Which session (SVM engine's or
    yolo_cls engine's) depends on ?engine= -- each engine keeps its own
    separate in-memory session log, so exporting one never pulls in the
    other's results."""
    engine_name = request.args.get("engine", "svm")
    _, get_session_log, _, pdf_model_tag = _engine(engine_name)

    session_results = get_session_log()
    if not session_results:
        return {"error": "No real-time classifications logged yet in this session."}, 400

    results_for_pdf = [
        {"filename": r["tag"], "label": r["label"], "confidence": r["confidence"], "image_path": r["image_path"]}
        for r in session_results
    ]
    out_path = generate_pdf_report_batch(results_for_pdf, model_tag=pdf_model_tag)
    return send_from_directory(os.path.dirname(out_path), os.path.basename(out_path), as_attachment=True)


@realtime_bp.route("/realtime/clear_session", methods=["POST"])
def realtime_clear_session():
    engine_name = request.args.get("engine", "svm")
    _, _, clear_session_log, _ = _engine(engine_name)
    clear_session_log()
    return {"status": "cleared"}