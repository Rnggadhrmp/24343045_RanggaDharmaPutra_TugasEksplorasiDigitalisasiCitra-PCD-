import cv2
import numpy as np
import matplotlib.pyplot as plt

def analyze_my_image(image_path):
    """Analyze your own image"""
    
    img = cv2.imread(image_path)
    
    if img is None:
        print("Gambar tidak ditemukan!")
        return
    
    print("---Analaisis Citra Pribadi---\n")
    
    height, width, channels = img.shape
    resolution = width * height
    print(f"Dimensi: {width} x {height}")
    print(f"Channels: {channels}")
    print(f"Resolusi: {resolution:,} pixels")
    
    aspect_ratio = width / height
    print(f"Aspect Ratio: {aspect_ratio:.2f}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"Ukuran Grayscale: {gray.shape}")
    
    print("\n=== Statistik Grayscale ===")
    print(f"Mean: {np.mean(gray):.2f}")
    print(f"Std Dev: {np.std(gray):.2f}")
    print(f"Min: {np.min(gray)}")
    print(f"Max: {np.max(gray)}")
    
    plt.figure(figsize=(12,4))
    
    plt.subplot(1,2,1)
    plt.hist(gray.ravel(), 256, [0,256])
    plt.title("Histogram Grayscale")
    
    plt.subplot(1,2,2)
    colors = ('b','g','r')
    for i, col in enumerate(colors):
        hist = cv2.calcHist([img],[i],None,[256],[0,256])
        plt.plot(hist, color=col)
    plt.title("Histogram RGB")
    
    plt.tight_layout()
    plt.show()
    
    return {
        "width": width,
        "height": height,
        "resolution": resolution,
        "aspect_ratio": aspect_ratio
    }

analyze_my_image("prend.jpeg")
