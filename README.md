STYLE TRANSFER -- Arman Mann

You will be needing Python 3.3+ to be installed on your system before starting.

To execute the code you will need to install the following python packages:

1. h5py
2. numpy
3. scipy
4. keras
5. tensorflow

Before executing the code, simply edit the file paths in the code to your content and style images on your device. Although any image may be used, a folder with some sample style images has been provided for simplicity. The specific places in the code where the image paths have to be edited are marked with relevant comments. 

To run, paste the following command in the command prompt and press Enter.

> python style.py

The program will automatically save the output images generated after each iteration in the same folder as the program file.
From the different images it can be gathered how the program iteratively reaches a state with minimum content and style loss in the output images through a process of gradient descent. Each image will take about 5-20 mins to be generated depending on the specified output image size. So please be patient.