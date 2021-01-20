import cv2

import os

img_array = []
for filename in sorted(os.listdir('/*Enter path*/')):
print(filename)
img = cv2.imread(filename)
height, width, layers = img.shape
size = (width, height)
img_array.append(img)

out = cv2.VideoWriter('/*Enter path*/project.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()

