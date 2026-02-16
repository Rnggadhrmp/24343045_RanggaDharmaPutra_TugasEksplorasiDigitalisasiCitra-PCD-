import cv2
import numpy as np
import matplotlib.pyplot as plt

print("=== EKSPLORASI DIGITALISASI CITRA ===\n")

img = cv2.imread("prend.jpeg")

if img is None:
    print("Gambar tidak ditemukan.")
    exit()

height, width, channels = img.shape
resolution = width * height
aspect_ratio = width / height
bit_depth = img.dtype.itemsize * 8
memory_bytes = img.size * img.dtype.itemsize
memory_kb = memory_bytes / 1024
memory_mb = memory_kb / 1024

print("=== ANALISIS PARAMETER CITRA ===")
print(f"Dimensi           : {width} x {height}")
print(f"Channels          : {channels}")
print(f"Resolusi          : {resolution:,} piksel")
print(f"Aspect Ratio      : {aspect_ratio:.2f}")
print(f"Bit Depth         : {bit_depth}-bit per channel")
print(f"Ukuran Memori     : {memory_mb:.2f} MB")

print("\n=== REPRESENTASI MATRIKS (5x5 pertama) ===")
print(img[:5, :5])

print("\n=== REPRESENTASI VEKTOR (25 nilai pertama) ===")
vector = img.flatten()
print(vector[:25])


crop_img = img[200:600, 300:800]

resize_img = cv2.resize(img, (800, 600))

rotate_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

def show_images(images, titles):
    plt.figure(figsize=(15, 8))
    for i in range(len(images)):
        plt.subplot(2, 2, i+1)
        if len(images[i].shape) == 2:
            plt.imshow(images[i], cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        plt.title(titles[i])
        plt.axis("off")
    plt.tight_layout()
    plt.show()

show_images(
    [img, crop_img, resize_img, rotate_img],
    ["Original", "Cropped", "Resized", "Rotated 90°"]
)

print("\n=== SIMULASI PERUBAHAN RESOLUSI & BIT DEPTH ===")

new_width = width * 2
new_height = height * 2
new_bit_depth = bit_depth // 2

new_memory = new_width * new_height * channels * (new_bit_depth/8)
new_memory_mb = new_memory / (1024*1024)

print(f"Resolusi Baru     : {new_width} x {new_height}")
print(f"Bit Depth Baru    : {new_bit_depth}-bit")
print(f"Estimasi Memori   : {new_memory_mb:.2f} MB")

print("\n=== SELESAI ===")
