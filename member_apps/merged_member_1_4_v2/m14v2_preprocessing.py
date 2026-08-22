import cv2

def clean(image):
    """
    m14v2's preprocessing: reused as-is from member_1_ab / merged_member_1_4
    (Gaussian blur denoise + histogram equalization on the luminance
    channel). Same rationale as merged_member_1_4: this experiment is about
    combining FEATURE extraction (colour + shape from member 1, gabor from
    member 4, texture added on top for this v2), not the
    preprocessing/detection/calibration stages.
    """
    denoised = cv2.GaussianBlur(image, (5, 5), 0)

    ycrcb = cv2.cvtColor(denoised, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    enhanced = cv2.cvtColor(cv2.merge([y_eq, cr, cb]), cv2.COLOR_YCrCb2BGR)

    return enhanced
