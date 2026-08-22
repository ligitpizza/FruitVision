import cv2

def clean(image):
    """
    m14v3's preprocessing: reused as-is from member_1_ab / merged_member_1_4
    (Gaussian blur denoise + histogram equalization on the luminance
    channel). Kept unchanged on purpose -- v3's experiment is scoped to
    detection (watershed, from member 3) + calibration (deskew, from
    member 4) + mask-based feature extraction, not preprocessing.
    """
    denoised = cv2.GaussianBlur(image, (5, 5), 0)

    ycrcb = cv2.cvtColor(denoised, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    enhanced = cv2.cvtColor(cv2.merge([y_eq, cr, cb]), cv2.COLOR_YCrCb2BGR)

    return enhanced
