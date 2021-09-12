1. Open DataCapture/WebcamCapture.py
2. Edit the home variable to your project directory
3. Edit the labels variable  to objects you want to detect, also edit objects variables to type of object to be detected
4. Run WebcamCapture.py; This will create necessary project directory, start your webcam for data capture and finally open the LebelImage script for creating xml files
5. Under Dataset Directory, find your images folder (directory will be named as the string assigned to the objects variables), group data to test and train. Ensure you also include xml files
6. Open Training/Train.py
7. Edit home and labels variable and Run the program to Train your model
6. Use StillImages.py and Webcam.py to test your model