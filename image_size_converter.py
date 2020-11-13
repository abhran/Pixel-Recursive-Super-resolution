import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm #for loading animation
import os
import numpy as np
from skimage.io import imsave

def get_data_list(datafolder):  
    print("Loading training data..........")
    z=os.listdir(datafolder)
    
    for i in tqdm(z):
        pass
    print()
    return  z



def get_training_data(datafolder,datalist,output_s,newfolder):

    for filename in tqdm(datalist):
        #Combines folder name and file name.
        path = os.path.join(datafolder,filename)
        #Opens an image as an Image object.
        image = Image.open(path)
        #Resizes to a desired size.
        imagehr = image.resize((output_s,output_s),Image.ANTIALIAS)
        # imagelr = image.resize((input_s,input_s),Image.ANTIALIAS)
        #Creates an array of pixel values from the image.
        output= np.asarray(imagehr)
        imsave(f'{newfolder}/{filename}',output)
        # pixel_array_lr = np.asarray(imagelr)

        # training_data_lr.append(pixel_array_lr)
        # training_data_hr.append(pixel_array_hr)

    #training_data is converted to a numpy array
    # hr = np.reshape(training_data_hr,(-1,output_s,output_s,channels))
    # lr = np.reshape(training_data_lr,(-1,input_s,input_s,channels))

    
class DataSet(object):
  def __init__(self,Image_data_folder):
    self.datalist=get_data_list(Image_data_folder)


data=DataSet('celeb')
get_training_data('celeb',data.datalist,32,'ramu')