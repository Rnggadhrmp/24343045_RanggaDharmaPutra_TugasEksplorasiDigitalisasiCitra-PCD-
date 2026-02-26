import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.cluster import KMeans

print("===================================================")
print("PROYEK MINI - MODEL WARNA DAN DETEKSI OBJEK")
print("Nama : Rangga Dharma Putra")
print("NIM  : 24343045")
print("===================================================\n")

paths = ["terang.png", "normal.png", "redup.png"]
images = []

for path in paths:
    img = cv2.imread(path)
    if img is None:
        print(f"{path} tidak ditemukan, membuat citra sintetik...")
        img = np.zeros((300, 400, 3), dtype=np.uint8)
        cv2.circle(img, (200,150), 80, (0,0,255), -1)
    images.append(img)

def uniform_quantization(image, levels=16):
    step = 256 // levels
    return (image // step) * step

def non_uniform_quantization(image, levels=16):
    shape = image.shape
    if len(shape) == 2:
        flat = image.reshape(-1, 1)
    else:
        flat = image.reshape(-1, 3)
    flat = np.float32(flat)
    kmeans = KMeans(n_clusters=levels, random_state=0, n_init=10)
    kmeans.fit(flat)
    clustered = kmeans.cluster_centers_[kmeans.labels_]
    return clustered.reshape(shape).astype(np.uint8)

def calculate_metrics(original, processed):
    mse = np.mean((original.astype(float) - processed.astype(float))**2)
    psnr = 10*np.log10((255**2)/mse) if mse > 0 else float('inf')
    return mse, psnr

def calculate_memory(image, bit_depth=8):
    h, w = image.shape[:2]
    channels = 1 if len(image.shape)==2 else image.shape[2]
    return h * w * channels * bit_depth

for idx, img in enumerate(images):

    print("\n===================================")
    print(f"Analisis Image {idx+1}")
    print("===================================")

    img = cv2.resize(img, (800, 500))

    start = time.time()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_time = time.time() - start

    start = time.time()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_time = time.time() - start

    start = time.time()
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_time = time.time() - start

    print("Waktu konversi Gray :", gray_time)
    print("Waktu konversi HSV  :", hsv_time)
    print("Waktu konversi LAB  :", lab_time)

    start = time.time()
    gray_u = uniform_quantization(gray)
    hsv_u  = uniform_quantization(hsv)
    lab_u  = uniform_quantization(lab)
    uniform_time = time.time() - start

    start = time.time()
    gray_nu = non_uniform_quantization(gray)
    hsv_nu  = non_uniform_quantization(hsv)
    lab_nu  = non_uniform_quantization(lab)
    nonuniform_time = time.time() - start

    mse_gray_u, psnr_gray_u = calculate_metrics(gray, gray_u)
    mse_gray_nu, psnr_gray_nu = calculate_metrics(gray, gray_nu)

    mse_hsv_u, psnr_hsv_u = calculate_metrics(hsv, hsv_u)
    mse_hsv_nu, psnr_hsv_nu = calculate_metrics(hsv, hsv_nu)

    mse_lab_u, psnr_lab_u = calculate_metrics(lab, lab_u)
    mse_lab_nu, psnr_lab_nu = calculate_metrics(lab, lab_nu)

    print("\n--- PSNR RESULTS ---")
    print("Gray  Uniform     :", psnr_gray_u)
    print("Gray  NonUniform  :", psnr_gray_nu)
    print("HSV   Uniform     :", psnr_hsv_u)
    print("HSV   NonUniform  :", psnr_hsv_nu)
    print("LAB   Uniform     :", psnr_lab_u)
    print("LAB   NonUniform  :", psnr_lab_nu)
    print("\n--- MATRiks 5x5 Grayscale Original ---")
    print(gray[:5, :5])

    print("\n--- Matriks 5x5 Grayscale Uniform ---")
    print(gray_u[:5, :5])

    print("\n--- Matriks 5x5 Grayscale Non-Uniform ---")
    print(gray_nu[:5, :5])

    mem_original = calculate_memory(img, 8)
    mem_quantized = calculate_memory(img, 4)
    compression_ratio = mem_original / mem_quantized

    print("\nMemori RGB 8-bit :", mem_original)
    print("Memori RGB 4-bit :", mem_quantized)
    print("Rasio Kompresi   :", compression_ratio)

    _, gray_mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    hsv_mask = cv2.inRange(hsv, lower_red, upper_red)

    lab_a = lab[:,:,1]
    _, lab_mask = cv2.threshold(lab_a, 150, 255, cv2.THRESH_BINARY)

    plt.figure(figsize=(14,8))

    plt.subplot(2,3,1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"Original Image {idx+1}")
    plt.axis("off")

    plt.subplot(2,3,2)
    plt.imshow(gray, cmap='gray')
    plt.title("Gray Original")
    plt.axis("off")

    plt.subplot(2,3,3)
    plt.imshow(gray_u, cmap='gray')
    plt.title("Gray Uniform")
    plt.axis("off")

    plt.subplot(2,3,4)
    plt.imshow(gray_nu, cmap='gray')
    plt.title("Gray Non-Uniform")
    plt.axis("off")

    plt.subplot(2,3,5)
    plt.imshow(hsv_mask, cmap='gray')
    plt.title("HSV Segmentation")
    plt.axis("off")

    plt.subplot(2,3,6)
    plt.imshow(lab_mask, cmap='gray')
    plt.title("LAB Segmentation")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
    plt.figure(figsize=(14,4))

    plt.subplot(1,3,1)
    plt.hist(gray.ravel(), 256, [0,256])
    plt.title("Histogram Gray")

    plt.subplot(1,3,2)
    plt.hist(gray_u.ravel(), 256, [0,256])
    plt.title("Histogram Uniform")

    plt.subplot(1,3,3)
    plt.hist(gray_nu.ravel(), 256, [0,256])
    plt.title("Histogram Non-Uniform")

    plt.tight_layout()
    plt.show()