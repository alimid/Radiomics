# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 2018

@author: Alise Midtfjord
"""

#Extract features from a batch of images using PyRadiomics. 
"----------------------------------------------"

import os
from radiomics import featureextractor
import numpy as np
import pandas
import logging
import radiomics

def pyradiomics_batch(input_file_name, output_file_name):
    '''
    Extract features from a batch of images using radiomics.
    
    :param str input_file_name: Name of the excel-document that contains patient ID, path
                                the image and path to the mask of the image
    :param str output_file_name: Name of the excel-document that is going to 
                                 containing the extracted features
    '''
    
    #Finding the directory and setting up a logging file                            
    outPath = os.getcwd()
    inputXLSX = os.path.join(outPath, input_file_name)
    outputFilepath = os.path.join(outPath, output_file_name)
    progress_filename = os.path.join(outPath, 'pyrad_log.txt')
    rLogger = logging.getLogger('radiomics')
    handler = logging.FileHandler(filename=progress_filename, mode='w')
    handler.setFormatter(logging.Formatter('%(levelname)s:%(name)s: %(message)s'))
    rLogger.addHandler(handler)
    logger = rLogger.getChild('batch')
    radiomics.setVerbosity(logging.INFO)
    
    try:
    # Use pandas to read and transpose ('.T') the input data
    # The transposition is needed so that each column represents one test case. This is easier for iteration over the input cases
        flists = pandas.read_excel(inputXLSX).T
    except Exception:
        logger.error('Excel READ FAILED', exc_info=True)
        exit(-1)
    
    logger.info('Loading Done')
    logger.info('Patients: %d', len(flists.columns))
    
    #Change the settings
    settings = {}
    settings['binWidth'] = 2 #To get the right amount of bins
    settings['sigma'] = [1] 
    extractor = featureextractor.RadiomicsFeaturesExtractor(**settings)
    extractor.enableAllImageTypes()

    logger.info('Enabled input images types: %s', extractor._enabledImagetypes)
    logger.info('Enabled features: %s', extractor._enabledFeatures)
    logger.info('Current settings: %s', extractor.settings)
    
    results = pandas.DataFrame()
    
    for entry in flists:  # Loop over all columns (i.e. the test cases)
        logger.info("(%d/%d) Processing Patient: %s",
                    entry + 1,
                    len(flists.columns),
                    flists[entry]['pasientID'])
    
        imageFilepath = flists[entry]['Image']
        maskFilepath = flists[entry]['Mask']
        label = flists[entry].get('Label', None)
    
        if str(label).isdigit():
          label = int(label)
        else:
            label = None
            
        if (imageFilepath is not None) and (maskFilepath is not None):
          featureVector = flists[entry]  # This is a pandas Series
          featureVector['Image'] = os.path.basename(imageFilepath)
          featureVector['Mask'] = os.path.basename(maskFilepath)
    
          try:
            # PyRadiomics returns the result as an ordered dictionary, which can be easily converted to a pandas Series
            # The keys in the dictionary will be used as the index (labels for the rows), with the values of the features
            # as the values in the rows.
            result = pandas.Series(extractor.execute(imageFilepath, maskFilepath, label))
            featureVector = featureVector.append(result)
          except Exception:
              logger.error('FEATURE EXTRACTION FAILED:', exc_info=True)
              
        featureVector.name = entry
          # By specifying an 'outer' join, all calculated features are added to the data frame, including those not
          # calculated for previous cases. This also ensures we don't end up with an empty frame, as for the first patient
          # it is 'joined' with the empty data frame.
        results = results.join(featureVector, how='outer')
              
    logger.info('Extraction complete, writing Excel')
      # .T transposes the data frame, so that each line will represent one patient, with the extracted features as columns
    
    results.T.to_excel(outputFilepath, index=False, na_rep='NaN')
    logger.info('Excel writing complete')

if __name__ == "__main__":
    name_input = 'name_input_file.xlsx'
    name_output = 'name_output_file.xlsx'
    pyradiomics_batch(name_input,name_output)