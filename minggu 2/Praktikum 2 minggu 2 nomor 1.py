import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('ruang tamu.png')

if image is None:
    print("File tidak ditemukan, menggunakan gambar sintetik.")
    image = np.zeros((300,400,3), dtype=np.uint8)
    cv2.rectangle(image,(50,50),(150,150),(0,0,255),-1)
    cv2.circle(image,(250,100),50,(0,255,0),-1)
    cv2.ellipse(image,(300,200),(80,40),30,0,360,(255,0,0),-1)

def analyze_color_model_suitability(image, application):
    
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if application == 'skin_detection':
        lower = np.array([0,20,70])
        upper = np.array([20,255,255])
        result = cv2.inRange(hsv, lower, upper)
        best_model = "HSV"
        
    elif application == 'shadow_removal':
        L,a,b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        L_enhanced = clahe.apply(L)
        merged = cv2.merge([L_enhanced,a,b])
        result = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
        best_model = "LAB"
        
    elif application == 'text_extraction':
        _, result = cv2.threshold(gray,0,255,
                                  cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        best_model = "Grayscale"
        
    elif application == 'object_detection':
        result = rgb
        best_model = "RGB"
        
    else:
        print("Aplikasi tidak dikenali.")
        return
    
    return result, best_model



applications = [
    'skin_detection',
    'shadow_removal',
    'text_extraction',
    'object_detection'
]

plt.figure(figsize=(12,8))

for i, app in enumerate(applications):
    
    result, model = analyze_color_model_suitability(image, app)
    
    plt.subplot(2,2,i+1)
    
    if len(result.shape)==2:
        plt.imshow(result, cmap='gray')
    else:
        plt.imshow(result)
    
    plt.title(f"{app}\nModel: {model}")
    plt.axis('off')
    
    print(f"{app} → Model terbaik: {model}")

plt.suptitle("Analisis Model Warna untuk Berbagai Aplikasi", fontsize=14)
plt.tight_layout()
plt.show()