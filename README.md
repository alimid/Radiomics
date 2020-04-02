# radiomics
These scripts were created to perform Radiomics on three-dimensional medical images (PET and CT). The scripts can be used to perform classification of all kinds of images (both two-dimensional and three-dimensional) by using classical texture analysis methods (e.g. GLCM). 

The script "batch.py" is used to extract the features from the images, using packages from PyRadiomics: https://pyradiomics.readthedocs.io/en/latest/. This is first order features as well as texture features, which could be used for further image classification. Here you can choose to use different kind of image filtration techniques (e.g. Wavelet, Laplacian) , as well as different kind of texture extraction methods (e.g. GLCM, shape, firstorder). You can change the settings for the feature extraction as described on the PyRadimics webside: https://pyradiomics.readthedocs.io/en/latest/customization.html. 

The script "change_matlab_to_python.py" is used to convert images created in matlab-format to nrrd-files, which is used in "batch.py".

The script "functions.py" is a package which containts all feature selection and classification methods imported into the script "classification.py". 

The script "classification.py" uses feature selection methods and classification methods defined in "functions.py" to classify images, and testes accuracy with a nested cross validation. Here you can also test the effect of using different amount of features in the classificaiton. 
