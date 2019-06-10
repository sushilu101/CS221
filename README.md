# CS221
Repository for CS221 project: "DeepLip: Using Recurrent and Convolutional NeuralNetworks to 'Lip-Read' from Image Sequences"

Authors: Pranav Upadhyayula, Michael Du, Sushil Upadhyayula

**A Description of each of the files is as follows:**
    <br/>
    221_Main_Models.ipynb     <br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;This is a Jupyter notebook with all of our main code except what is defined below<br/>
    data_augmentation.py      <br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;This has the code we used to augment and tilt our images<br/>
    dataset_cropping.py       <br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The given data was extremely noisy so we use OpenCV here to crop our images appropriately<br/>
    LSTM_data_collection.py   <br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;This is the code we used to get the vertical distance between lips that we input into our LSTM
