import cv2, uuid, os, time
labels = ['PlateNumber']
Objects = 'licence'
number_imgs = 5
home = 'A:\projects\ANPR'  # Project Directory
os.chdir(home)
IMAGES_PATH = os.path.join(os.curdir, 'Datasets', Objects, 'CollectedImages')
LABELIMG_PATH = os.path.join(home, 'DataCapture','labelImages')
if not os.path.exists(IMAGES_PATH):
    os.system('mkdir {}'.format(IMAGES_PATH))

for label in labels:
    path = os.path.join(IMAGES_PATH, label)
    if not os.path.exists(path):
       print(os.system('mkdir {}'.format(path)))
