Welcome to our model for breast cancer perdiction, to run this project you need to do the following:

1: First you need to have anconda3 environment to run the model
2: After that you need to activate it , example: C:/ProgramData/anaconda3/Scripts/activate
3: Then write "activate tf"
4: after that write "ipython -c "%run ProjectPart1.ipynb" " to train the model
5: You can also write "ipython -c "%run Demo.ipynb" " to try the demo on the save weights that we have in the project

Notes: 
-In each training you need to remove all the directories inside DatasetNew to train correctly , but keep DatasetNew folder dont delete it
-You can run the code without anaconda3 environment but it will run on CPU not GPU and it will be very slow
-For the Demo, when you run it, you will find a link in the command line to open gradio local host and try it


Link for our dataset : https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images
