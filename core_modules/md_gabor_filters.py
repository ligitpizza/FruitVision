import cv2
import numpy as np

def build_gabor_kernels():
    kernels = []
    for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
        kernel = cv2.getGaborKernel((21, 21), sigma=4.0, theta=theta,
                                     lambd=10.0, gamma=0.5, psi=0, ktype=cv2.CV_32F)
        kernels.append(kernel)
    return kernels

_KERNELS = build_gabor_kernels()

def extract_gabor(cleaned_img):
    """
    Extracts mean energy response across 4 Gabor orientations.
    Returns a 4-value feature vector.
    """
    gray = cv2.cvtColor(cleaned_img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    responses = []
    for kernel in _KERNELS:
        filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
        responses.append(np.mean(np.abs(filtered)))

    return np.array(responses, dtype=np.float32)

FEATURE_NAMES = ["gabor_0deg", "gabor_45deg", "gabor_90deg", "gabor_135deg"]


def visualize_gabor(cleaned_img):
    """Renders the normalized magnitude of the averaged Gabor response
    across all 4 orientations -- the same filtered signal extract_gabor()'s
    mean-energy values are computed from."""
    gray = cv2.cvtColor(cleaned_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    responses = [np.abs(cv2.filter2D(gray, cv2.CV_32F, kernel)) for kernel in _KERNELS]
    averaged = np.mean(responses, axis=0)
    normalized = cv2.normalize(averaged, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
