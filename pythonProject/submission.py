import cv2
import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)
matplotlib.rcParams['image.cmap'] = 'gray'

import matplotlib
matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)
matplotlib.rcParams['image.cmap'] = 'gray'

im = np.zeros((10,10),dtype='uint8')
print(im);
plt.imshow(im)

im[0,1] = 1
im[-1,0]= 1
im[-2,-1]=1
im[2,2] = 1
im[5:8,5:8] = 1

print(im)
plt.imshow(im)

element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
print(element)
ksize = element.shape[0]

height,width = im.shape[:2]

ErodedEllipseKernel = cv2.erode(im, element)
print(ErodedEllipseKernel)
plt.imshow(ErodedEllipseKernel)

border = ksize // 2
paddedIm = np.zeros((height + border * 2, width + border * 2))
paddedIm = cv2.copyMakeBorder(im, border, border, border, border, cv2.BORDER_CONSTANT, value=1)
paddedErodedIm = paddedIm.copy()
# Create a VideoWriter object
# Use frame size as 50x50
###
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('erosionScratch.avi', fourcc, 10.0, (50, 50))
###
for h_i in range(border, height + border):
    for w_i in range(border, width + border):
        if paddedIm[h_i, w_i]:
            print("White Pixel Found @ {},{}".format(h_i, w_i))

            neighborhood = paddedIm[h_i - border: (h_i + border) + 1, w_i - border: (w_i + border) + 1]
            and_result = cv2.bitwise_and(neighborhood, element)

            min_value = np.min(and_result)
            paddedErodedIm[h_i, w_i] = min_value

            resized_im = cv2.resize(im * 255, (50, 50), interpolation=cv2.INTER_NEAREST)
            bgr_im = cv2.cvtColor(resized_im, cv2.COLOR_GRAY2BGR)
            out.write(bgr_im)

# Release the VideoWriter object
out.release()

# Display final image (cropped)
croppedErodedIm = paddedErodedIm[border:-border, border:-border]

plt.imshow(croppedErodedIm, cmap='gray')
plt.title('Cropped Final Eroded Image')
plt.show()

