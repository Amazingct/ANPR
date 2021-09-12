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


# Capture Images
def capture():
    for label in labels:
        cap = cv2.VideoCapture(0)
        print('Collecting images for {}'.format(label))
        time.sleep(5)
        for imgnum in range(number_imgs):
            print('Collecting image {}'.format(imgnum))
            ret, frame = cap.read()
            imgname = os.path.join(IMAGES_PATH, label, label + '.' + '{}.jpg'.format(str(uuid.uuid1())))
            cv2.imwrite(imgname, frame)
            cv2.imshow('frame', frame)
            time.sleep(2)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()




# Clone Tzutalin Label Image Script
if not os.path.exists(LABELIMG_PATH):
    os.system('mkdir {}'.format(LABELIMG_PATH))
    os.system("git clone https://github.com/tzutalin/labelImg {}".format(LABELIMG_PATH))

'''
pyrcc5 takes a Qt Resource File (. qrc) and converts it into a 
Python module which can be imported into a PyQt5 application.
All files loaded by Qt that are prefixed with a colon will be 
loaded from the resources rather than the file system.
'''



#os.system('cd {} && pyrcc5 -o libs/resources.py resources.qrc'. format(LABELIMG_PATH))

# Capture
#capture()
# Now run the label Image Script
#os.system('cd {} && python labelImg.py'.format(LABELIMG_PATH))
