# CS221
Repository for CS221 project: "DeepLip: Using Recurrent and Convolutional NeuralNetworks to 'Lip-Read' from Image Sequences"

Authors: Pranav Upadhyayula, Michael Du, Sushil Upadhyayula

A Description of each of the files is as follows:
    1. 221_Main_Models.ipynb     This is a Jupyter notebook with all of our main code except what is defined below
    2. data_augmentation.py      This has the code we used to augment and tilt our images
    3. dataset_cropping.py       The given data was extremely noisy so we use OpenCV here to crop our images appropriately
    4. LSTM_data_collection.py   This is the code we used to get the vertical distance between lips that we input into our LSTM
