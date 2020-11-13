import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm #for loading animation
import os

def get_data_list(datafolder):  
    print("Loading training data..........")
    z=os.listdir(datafolder)
    
    for i in tqdm(z):
        pass
    print()
    return  z




def get_training_data(datafolder,datalist,input_s,output_s,channels):

    training_data_lr = []
    training_data_hr = []
    #Finds all files in datafolder
    # print(filenames[(epoch-1)*batch_size:epoch*batch_size])
    for filename in datalist:
        #Combines folder name and file name.
        path = os.path.join(datafolder,filename)
        #Opens an image as an Image object.
        image = Image.open(path)
        #Resizes to a desired size.
        imagehr = image.resize((output_s,output_s),Image.ANTIALIAS)
        imagelr = image.resize((input_s,input_s),Image.ANTIALIAS)
        #Creates an array of pixel values from the image.
        pixel_array_hr= np.asarray(imagehr)
        pixel_array_lr = np.asarray(imagelr)

        training_data_lr.append(pixel_array_lr)
        training_data_hr.append(pixel_array_hr)

    #training_data is converted to a numpy array
    hr = np.reshape(training_data_hr,(-1,output_s,output_s,channels))
    lr = np.reshape(training_data_lr,(-1,input_s,input_s,channels))
    return hr,lr

# c=get_data_list("celebA")
# print(c)
# get_training_data('celebA',c,8,32,3,32,1)

class DataSet(object):
  def __init__(self,Image_data_folder,input_s,output_s,channels):
    self.datalist=get_data_list(Image_data_folder)
    self.hr,self.lr=get_training_data(Image_data_folder,self.datalist,input_s,output_s,channels)
