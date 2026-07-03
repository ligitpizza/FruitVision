import cv2

def clean(image):
    """
    Member 4's preprocessing: Non-Local Means denoising (compares patches
    across the whole image rather than just a local neighbourhood, so it
    removes noise while preserving fine texture better than Gaussian/
    median blur -- at noticeably higher compute cost) + CLAHE on the
    luminance channel for adaptive local contrast, same contrast approach
    as member 2 but paired with a heavier-weight denoiser.
    """
    denoised = cv2.fastNlMeansDenoisingColored(
        image, None, h=7, hColor=7, templateWindowSize=7, searchWindowSize=21
    )

    ycrcb = cv2.cvtColor(denoised, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    y_eq = clahe.apply(y)
    enhanced = cv2.cvtColor(cv2.merge([y_eq, cr, cb]), cv2.COLOR_YCrCb2BGR)

    return enhanced