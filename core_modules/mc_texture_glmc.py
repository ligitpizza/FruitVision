import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

def extract_texture_glcm(cleaned_img):
    """
    Extracts Haralick texture features via Grey-Level Co-occurrence Matrix.
    Returns a 4-value feature vector.
    """
    gray = cv2.cvtColor(cleaned_img, cv2.COLOR_BGR2GRAY)
    gray = (gray / 4).astype(np.uint8)

    glcm = graycomatrix(gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                         levels=64, symmetric=True, normed=True)

    contrast = graycoprops(glcm, 'contrast').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    energy = graycoprops(glcm, 'energy').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()

    features = np.array([contrast, correlation, energy, homogeneity], dtype=np.float32)
    return features

FEATURE_NAMES = ["contrast", "correlation", "energy", "homogeneity"]


def visualize_texture(cleaned_img):
    """Renders the quantized grayscale image actually fed into the GLCM
    (same 64-level quantization used for feature extraction above), rescaled
    back to the 0-255 range for display."""
    gray = cv2.cvtColor(cleaned_img, cv2.COLOR_BGR2GRAY)
    quantized = (gray / 4).astype(np.uint8)
    rescaled = (quantized * 4).astype(np.uint8)
    return cv2.cvtColor(rescaled, cv2.COLOR_GRAY2BGR)
