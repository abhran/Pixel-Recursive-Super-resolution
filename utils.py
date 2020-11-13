import numpy as np
from skimage.io import imsave
from skimage import img_as_ubyte



def softmax(x):
    exps = np.exp(x - np.max(x))
    return  exps/np.sum(exps)



def save_samples(np_imgs, img_path):
  imsave(img_path, img_as_ubyte(np_imgs))



def logits_2_pixel_value(logits, mu=1.1):
  rebalance_logits = logits * mu
  probs = softmax(rebalance_logits)
  pixels=(np.argmax(probs))
  return np.floor(pixels)

