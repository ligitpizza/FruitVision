import cv2

def clean(image):
    """
    merged_member_1_4's preprocessing: reused as-is from member_1_ab
    (Gaussian blur denoise + histogram equalization on the luminance
    channel). This experiment is about combining FEATURE extraction
    (colour + shape from member 1, gabor from member 4), not the
    preprocessing/detection/calibration stages -- those are kept
    identical to member 1's for simplicity.
    """
    denoised = cv2.GaussianBlur(image, (5, 5), 0)

    ycrcb = cv2.cvtColor(denoised, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    enhanced = cv2.cvtColor(cv2.merge([y_eq, cr, cb]), cv2.COLOR_YCrCb2BGR)

    return enhanced
