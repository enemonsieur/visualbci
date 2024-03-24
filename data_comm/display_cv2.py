#%%
import numpy as np
import cv2
import time

win_identifier = "Test Image"
t_update = 1
n_width = 400
n_height = 200
n_frames = 5
n_rgb = 3

for i in range(n_frames):

    # imshow can display float [0, 1] or uint [0, 255]
    # imshow can display either RGB or grayscale
    r = np.random.rand(n_height, n_width, n_rgb)

    cv2.imshow(win_identifier, r)
    cv2.waitKey(1)
    time.sleep(t_update)

cv2.destroyWindow(win_identifier)


# %%
