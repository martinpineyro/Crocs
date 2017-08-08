# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import io
import numpy 

#para que ande en laptop
#import sys
#video_capture = cv2.VideoCapture(0)


#-----------------------------------------------------------------------------
#       Load and configure Haar Cascade Classifiers
#-----------------------------------------------------------------------------
 
# location of OpenCV Haar Cascade Classifiers:
baseCascadePath = '/usr/local/share/OpenCV/haarcascades/'
 
# xml files describing our haar cascade classifiers
faceCascadeFilePath = baseCascadePath + 'haarcascade_frontalface_default.xml'
#noseCascadeFilePath = baseCascadePath + 'haarcascade_mcs_nose.xml'
 
# build our cv2 Cascade Classifiers
faceCascade = cv2.CascadeClassifier(faceCascadeFilePath)
#noseCascade = cv2.CascadeClassifier(noseCascadeFilePath)
 
#-----------------------------------------------------------------------------
#       Load and configure mustache (.png with alpha transparency)
#-----------------------------------------------------------------------------
 
# Load our overlay image: mustache.png
imgMustache = cv2.imread('crocs1.png',-1)
imgCrocs1= cv2.imread('crocs1.png',-1)
 
# Create the mask for the mustache
#orig_mask = imgMustache[:,:,3]
orig_mask = imgCrocs1[:,:,3]
 
# Create the inverted mask for the mustache
orig_mask_inv = cv2.bitwise_not(orig_mask)
 
# Convert mustache image to BGR
# and save the original image size (used later when re-sizing the image)
imgMustache = imgMustache[:,:,0:3]
imgCrocs1 = imgCrocs1[:,:,0:3]

origMustacheHeight, origMustacheWidth = imgMustache.shape[:2]
origCrocs1Height, origCrocs1Width = imgCrocs1.shape[:2]
 
#-----------------------------------------------------------------------------
#       Main program loop
#-----------------------------------------------------------------------------
 
# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
#camera.rotation = 90
#ret, frame = video_capture.read()

ancho = 1920
alto = 1080
#alto, ancho  = frame.shape[:2]

camera.resolution = (ancho, alto)
camera.framerate = 5
#camera.hflip = True
 
# allow the camera to warmup
time.sleep(0.1)

font = cv2.FONT_HERSHEY_SIMPLEX

while True:
	
    # Capture video feed
    stream = io.BytesIO()
    for foo in camera.capture_continuous(stream, format='jpeg'):	
 
	stream.truncate()
        stream.seek(0)
    	#Convert the picture into a numpy array
        buff = numpy.fromstring(stream.getvalue(), dtype=numpy.uint8)

    	#Now creates an OpenCV image
	frame = cv2.imdecode(buff, 1)

        # Create greyscale image from the video feed
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
        # Detect faces in input video stream
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.35,
            minNeighbors=5,
            minSize=(50, 50),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        
        # Iterate over each face found
        for (x, y, w, h) in faces:
            # Un-comment the next line for debug (draw box around all faces)
            #face = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

            # Dimensionar la careta
            crocs1Width =  int(round(1.5*w))
            crocs1Height = crocs1Width * origCrocs1Height / origCrocs1Width

            roi_gray = gray[y-(crocs1Height/2):y+(crocs1Height/2), x:x+w]
            roi_color = frame[y-(crocs1Height/2):y+(crocs1Height/2), x:x+w]

            # Centrar la careta 
            x1 = x - 50
            x2 = x + crocs1Width
            y1 = y - (crocs1Height/2)
            y2 = y + (crocs1Height/2)

            
            #cv2.putText(frame, '(x,y)', (x, y), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            #cv2.putText(frame, '(x1,y1)', (x1, y1), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            #cv2.putText(frame, '(x2,y2)', (x2, y2), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            
            
            # Check for clipping
            if x1 < 0:
                x1 = 0
            if y1 < 0:
                y1 = 0
            if x2 > ancho:
                x2 = ancho
            if y2 > alto:
                y2 = alto

            # Re-calculate the width and height of the mustache image
            crocs1Width = x2 - x1
            crocs1Height = y2 - y1
            

            # Re-size the original image and the masks to the mustache sizes
            # calcualted above
            
            crocs1 = cv2.resize(imgCrocs1, (crocs1Width,crocs1Height), interpolation = cv2.INTER_AREA)
            mask = cv2.resize(orig_mask, (crocs1Width,crocs1Height), interpolation = cv2.INTER_AREA)
            mask_inv = cv2.resize(orig_mask_inv, (crocs1Width,crocs1Height), interpolation = cv2.INTER_AREA)


            # take ROI for mustache from background equal to size of mustache image
            #roi = roi_color[:,:,0]
            #[row, col]
            #roi = frame[y1:(y1+crocs1Height), x1:(x1+crocs1Width)]
            roi = frame[y1:y2, x1:x2]
            #roi_color = frame[y1:(y1+crocs1Height), x1:(x1+crocs1Width)]
            roi_color = frame[y1:y2, x1:x2]

            # roi_bg contains the original image only where the mustache is not
            # in the region that is the size of the mustache.
            roi_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

            # roi_fg contains the image of the mustache only where the mustache is
            roi_fg = cv2.bitwise_and(crocs1,crocs1,mask = mask)
           
            # join the roi_bg and roi_fg
            dst = cv2.add(roi_bg,roi_fg)

            dst_w, dst_h = dst.shape[:2]

            # place the joined image, saved to dst back over the original image
            frame[y1:y2, x1:x2] = dst
            
            #break
    
            
        # Display the resulting frame
        cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
        cv2.imshow("window", frame)
        #cv2.imshow('Video', frame)
    
 
        # press any key to exit
        # NOTE;  x86 systems may need to remove: &amp;amp;amp;amp;amp;amp;amp;quot;&amp;amp;amp;amp;amp;amp;amp;amp; 0xFF == ord('q')&amp;amp;amp;amp;amp;amp;amp;quot;
        if cv2.waitKey(1) and 0xFF == ord('q'):
            break



