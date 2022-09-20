
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import color
from skimage import io

im = cv2.imread('zebra.bmp', cv2.IMREAD_GRAYSCALE)

FT = np.fft.fft2(im)
C_FT = np.fft.ifftshift(FT)
fig, axs = plt.subplots(2)
fig.suptitle('Vertically stacked subplots')
axs[0].imshow(im)
axs[1].imshow(np.log(1+abs(C_FT)))
plt.show()

# an attempt to generate the cylindrical HPF
# [a,b] = [len(FT), len(FT[0])]
a = FT.shape[0]
b = FT.shape[1]
LPF = np.zeros((a, b))
R = 50
for x in range(0, a):
    for y in range(0, b):
        if (x-a/2)**2 + (y-b/2)**2 <= R**2:
            LPF[x, y] = 1
HPF = 1 - LPF

plt.imshow(HPF, "gray"), plt.title("HPF")
plt.show()

# Decentralizing the mask
HPF_dec = np.fft.fftshift(HPF)
plt.imshow(HPF_dec,"gray"), plt.title("HPF_dec")
plt.show()

# Get high-pass image content

F_high = HPF * C_FT
Im1 = np.fft.ifft2(np.fft.fftshift(F_high))
plt.imshow(np.abs(Im1), "gray"), plt.title("Edges"), plt.show()

# Get low-pass image content
F_low = LPF * C_FT
Im2 = np.fft.ifft2(np.fft.fftshift(F_low))
plt.imshow(np.abs(Im2), "gray"), plt.title("Smoothness"), plt.show()


