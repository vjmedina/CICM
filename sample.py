import math
import numpy as np
from PIL import Image

import cicm


levels = 256
distances=[1]
angles=[0, math.radians(45), math.radians(90), math.radians(135)]
        
img = Image.open("peppers.png")

r_channel = np.array(img)[:,:,0]
g_channel = np.array(img)[:,:,1]

cicm_array = cicm.cicm(r_channel, g_channel, distances=distances, angles=angles, levels=levels)

cicm_image = Image.fromarray(cicm_array[:,:,0,0])

img.show("Sample image")
cicm_image.show("CICM")


