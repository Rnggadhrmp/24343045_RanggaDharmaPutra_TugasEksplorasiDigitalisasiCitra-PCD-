import numpy as np
import matplotlib.pyplot as plt

def simulate_digitization(analog_function, sampling_rate, quantization_levels):
    
    x_cont = np.linspace(0, 1, 1000)
    y_cont = analog_function(x_cont)
    
    x_sample = np.linspace(0, 1, sampling_rate)
    y_sample = analog_function(x_sample)
    
    y_min = np.min(y_sample)
    y_max = np.max(y_sample)
    
    levels = np.linspace(y_min, y_max, quantization_levels)
    y_quantized = np.digitize(y_sample, levels)
    y_quantized = levels[y_quantized - 1]
    
    plt.figure(figsize=(10,6))
    
    plt.plot(x_cont, y_cont, label="Analog Signal")
    plt.stem(x_sample, y_sample, linefmt='g-', markerfmt='go', basefmt=" ", label="Sampled")
    plt.step(x_sample, y_quantized, where='mid', color='r', label="Quantized")
    
    plt.legend()
    plt.title("Simulasi Sampling dan Quantization")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()

simulate_digitization(lambda x: np.sin(2*np.pi*5*x), 20, 8)
