import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

def extract_texture_glcm(cleaned_img):
    """
    Extracts Haralick texture features via Grey-Level Co-occurrence Matrix.
    Returns a 4-value feature vector.
    """
    gray = cv2.cvtColor(cleaned_img, cv2.COLOR_BGR2GRAY)
    gray = (gray / 4).astype(np.uint8)  # reduce to 64 grey levels for speed

    glcm = graycomatrix(gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                         levels=64, symmetric=True, normed=True)

    contrast = graycoprops(glcm, 'contrast').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    energy = graycoprops(glcm, 'energy').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()

    features = np.array([contrast, correlation, energy, homogeneity], dtype=np.float32)
    return features

FEATURE_NAMES = ["contrast", "correlation", "energy", "homogeneity"]