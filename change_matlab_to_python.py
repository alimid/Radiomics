# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 2018

@author: Alise Midtfjord
"""

#A script to convert the images to nrrd-files, and create masks for the images.
"-------------------------------------------"

import numpy as np
import pandas as pd
import nrrd
import scipy.io as sio
import os 
import os.path
 
"""-----------PET-images-----------"""

#Creating the table and setting the directory
n_images = 255
length = np.zeros(shape = (n_images,3)) 
tabell = pd.DataFrame(data = length, columns = ['pasientID', 'Image', 'Mask']) 
directory = os.getcwd() 
j = 0

#Loop over all images in the folder
for i in range(1,n_images): 
    
    #Need this because the names have two, one or zero zeroes before the number
    if i < 10: 
        
        #Insert the name of the pictures in the folder, from matlab
        filnavn = 'PETstackP00' + str(i) + '.mat' 
        kortnavn = 'PETstackP00' + str(i) 
        masknavn = '\PETstackP00' + str(i) + 'mask.nrrd' 
    elif i < 100:
        filnavn = 'PETstackP0' + str(i) + '.mat'
        kortnavn = 'PETstackP0' + str(i)
        masknavn = '\PETstackP0' + str(i) + 'mask.nrrd'
    else:
        filnavn = 'PETstackP' + str(i) + '.mat'
        kortnavn = 'PETstackP' + str(i)
        masknavn = '\PETstackP' + str(i) + 'mask.nrrd'
    if os.path.isfile(filnavn): 
        matlab = sio.loadmat(filnavn)
        bilde = matlab[kortnavn]
        
        #Create the image as a nrrd-file
        nrrd.write(kortnavn + '.nrrd', bilde) 
        maske = bilde
        
        #Create the mask
        maske[np.isnan(maske)] = 0 
        maske[maske != 0] = 1
        maske = maske.astype(int)
        nrrd.write(kortnavn + 'mask.nrrd', maske) 
        
        #Setting the values of the table
        tabell.iloc[j,0] = i 
        tabell.iloc[j,1] = directory +  kortnavn 
        j += 1  
        
    print(i)

#Make a excel-file with the paths. 
tabell.to_excel('paths.xlsx') 
    
"""-----------CT-images, with the same mask-----------"""

length = np.zeros(shape = (n_images,3)) 
tabell = pd.DataFrame(data = length, columns = ['pasientID', 'Image', 'Mask']) 
directory = os.getcwd() #
j = 0

for i in range(1,n_images): 
    if i < 10: 
        filnavn = 'CTstackP00' + str(i) + '.mat' 
        kortnavn = 'CTstackP00' + str(i) 
        masknavn = '\PETstackP00' + str(i) + 'mask.nrrd'
    elif i < 100:
        filnavn = 'CTstackP0' + str(i) + '.mat'
        kortnavn = 'CTstackP0' + str(i)
        masknavn = '\PETstackP0' + str(i) + 'mask.nrrd'
    else:
        filnavn = 'CTstackP' + str(i) + '.mat'
        kortnavn = 'CTstackP' + str(i)
        masknavn = '\PETstackP' + str(i) + 'mask.nrrd'
    if os.path.isfile(filnavn):
        matlab = sio.loadmat(filnavn)
        bilde = matlab[kortnavn]
        nrrd.write(kortnavn + '.nrrd', bilde) 
        
        tabell.iloc[j,0] = i 
        tabell.iloc[j,1] = directory +  kortnavn 
        tabell.iloc[j,2] = directory +  masknavn 
        j += 1  
        
    print(i) 

tabell.to_excel('paths_CT.xlsx')      
    