import numpy as np
import matplotlib.pyplot as plt

hW, hH = 600, 300
hFreq = 10

x = np.linspace(0, 2 * hW / (2 * hW + 1), 2 * hW - 1)
y = np.linspace(0, 2 * hW / (2 * hW + 1), 2 * hW - 1)

[X, Y] = np.meshgrid(x, y)
A = np.sin(hFreq * 2 * np.pi * X)

plt.imshow(A, "gray"), plt.show()
H, W = np.shape(A)

FT = np.fft.fft2(A)
FT = np.fft.fftshift(FT)
mag_F = abs(FT)
mag_F = np.divide (mag_F, np.max(mag_F))
phase_F = np.angle(FT)
plt.imshow(mag_F, "gray", vmin = 0, vmax=0.01), plt.title("mag"), plt.show()
plt.imshow(phase_F, "gray"), plt.title("phase"), plt.show()

# Phase shift
FT_new = np.zeros((H, W))*1j
x0 = 288
for u in range(0, W):
    FT_new[:, u] = FT[:, u] * np.exp(1j*2*np.pi*x0*u/1200)

plt.imshow(np.angle(FT_new), "gray"), plt.title("phase-shift"), plt.show()
# new imag
Im_new = np.fft.ifft2((FT_new))
plt.imshow(np.abs(Im_new), "gray"), plt.title("new image"), plt.show()