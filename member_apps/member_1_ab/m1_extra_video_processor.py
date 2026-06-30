import cv2

def process_video(video_path, predict_fn, sample_every_n=15):
    """
    Reads a video file, runs predict_fn on every Nth frame.
    predict_fn should be predict_ripeness from predict.py.
    Returns a list of (frame_index, label, confidence).
    """
    cap = cv2.VideoCapture(video_path)
    results = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_every_n == 0:
            try:
                label, confidence, bbox, _ = predict_fn(frame)
                results.append({"frame": frame_idx, "label": label, "confidence": confidence})
            except Exception as e:
                print(f"Frame {frame_idx} skipped: {e}")

        frame_idx += 1

    cap.release()
    return results