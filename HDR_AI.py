from __future__ import print_function
import numpy as np
from enlighten_inference import EnlightenOnnxModel
import cv2


#################### LOW LIGHT ENTHANCE  ####################
img = cv2.imread('./input_img.jpg')
model = EnlightenOnnxModel()

processed = model.predict(img)

cv2.imwrite('now_overexposed.jpg', processed)


####################  UNDEREXPOSED IMAGE  ####################
from PIL import Image
from skimage import exposure

# Öffne das Bild
image = Image.open("./input_img.jpg")

# Konvertiere das Bild in ein numpy-Array
image_array = np.array(image)

# Anwenden einer Gamma-Korrektur, um das Bild dunkler erscheinen zu lassen
image_array = exposure.adjust_gamma(image_array, gamma=1.8)

# Konvertiere das bearbeitete Bild zurück in ein PIL-Bild
image = Image.fromarray(np.uint8(image_array))

# Speichere das bearbeitete Bild
image.save("now_underexposed.jpg")


####################  MERGE TO HDR ####################
# Loading exposure images into a list
img_fn = [r"./now_underexposed.jpg", r"./input_img.jpg", r"./now_overexposed.jpg"]
img_list = [cv2.imread(fn) for fn in img_fn]
exposure_times = np.array([250, 450, 1300], dtype=np.float32)
# Merge exposures to HDR image
merge_debevec = cv2.createMergeDebevec()
hdr_debevec = merge_debevec.process(img_list, times=exposure_times.copy())
# Tonemap HDR image
tonemap1 = cv2.createTonemap(gamma=2)
res_debevec = tonemap1.process(hdr_debevec.copy())
# Exposure fusion using Mertens
merge_mertens = cv2.createMergeMertens()

# Convert datatype to 8-bit and save
res_debevec_8bit = np.clip(res_debevec*255, 0, 255).astype('uint8')

cv2.imwrite("HDR_Output.jpg", res_debevec_8bit)
