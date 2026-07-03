import cv2

def clean(image):
    """
    Member 4's preprocessing: bilateral filtering (edge-preserving denoise,
    much cheaper than Non-Local Means) + CLAHE on the luminance channel for
    adaptive local contrast.

    Previously used cv2.fastNlMeansDenoisingColored, which compares patches
    across the whole image and was measured at ~1.1s/image -- responsible
    for 95%+ of total training time once YOLO was removed from detection.
    Bilateral filtering is edge-aware like NLM but weights only a local
    neighbourhood (spatial + intensity similarity), so it's a fraction of
    the cost while still preserving edges better than a plain Gaussian/
    median blur. CLAHE step is unchanged.
    """
    denoised = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

    ycrcb = cv2.cvtColor(denoised, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    y_eq = clahe.apply(y)
    enhanced = cv2.cvtColor(cv2.merge([y_eq, cr, cb]), cv2.COLOR_YCrCb2BGR)

    return enhanced