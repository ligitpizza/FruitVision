import cv2

def clean(image):
    """
    Member 4's preprocessing: bilateral filtering (edge-preserving denoise,
    much cheaper than Non-Local Means) + CLAHE on the luminance channel for
    adaptive local contrast.
    """
    denoised = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

    ycrcb = cv2.cvtColor(denoised, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    y_eq = clahe.apply(y)
    enhanced = cv2.cvtColor(cv2.merge([y_eq, cr, cb]), cv2.COLOR_YCrCb2BGR)

    return enhanced
