import numpy as np
import cv2
from time import sleep
import RPi.GPIO as GPIO


GPIO.setmode(GPIO.BOARD)   # Servo
GPIO.setwarnings(False)    # Servo
GPIO.setup(7, GPIO.OUT)    # Servo
p = GPIO.PWM(7, 25)        # Servo
p.start(2.5)               # Servo


DIR1 = 10
STEP1 = 8

DIR2 = 38
STEP2 = 36

CW = 1
CCW = 0

K = 0
counter = 0

GPIO.setmode(GPIO.BOARD)

GPIO.setup(DIR1, GPIO.OUT)
GPIO.setup(STEP1, GPIO.OUT)
GPIO.setup(DIR2, GPIO.OUT)
GPIO.setup(STEP2, GPIO.OUT)


up = 0
right = 0

vert = 0.025
hori = 0.001

quit_vert = 0.02
quit_hori = 0.01

hori_pulse = 2
vert_pulse = 1

hori_pulse_big = 20

loop_delay = 0.005


def rotate(DIR, STEP, C, INNER_DELAY, OUTER_DELAY, PULSE):
    GPIO.output(DIR,C)
    for x in range(PULSE):
        GPIO.output(STEP,GPIO.HIGH)
        sleep(INNER_DELAY) 
        GPIO.output(STEP,GPIO.LOW)
        sleep(INNER_DELAY)
    sleep(OUTER_DELAY)


def trigger():
    print("Gun Trigger")
    p.ChangeDutyCycle(2.5)  # turn towards 0 degree
    sleep(1)
    p.ChangeDutyCycle(5)  # turn towards 45 degrees
    sleep(1)
    p.ChangeDutyCycle(2.5)  # turn towards 0 degrees
    sleep(1)
    print("Sleeping for 10 sec")
    sleep(10)


def non_max_suppression_fast(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
	# initialize the list of picked indexes	
	pick = []
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int")






print("\nWelcome to Auto Aiming Turret\n")

while True:

    
    print("Please choose the mode of operation below\n1. Interactive Mode\n2. Automatic mode\n3. Test run")
    mode = input("Enter the no. of mode of operation here: ")


    if mode == "1":
        while True:

            control = input("\n").lower()
            
            
            if control == "w":
                print("UP\n")
                up = up + 2
                rotate(DIR2, STEP2, CW, vert, loop_delay, vert_pulse) 


            elif control == "s":
                print("DOWN\n")
                up = up - 2
                rotate(DIR2, STEP2, CCW, vert, loop_delay, vert_pulse) 


            elif control == "a":
                print("LEFT\n")
                right = right - 5
                rotate(DIR1, STEP1, CCW, hori, loop_delay, hori_pulse) 


            elif control == "d":
                print("RIGHT\n")
                right = right + 5
                rotate(DIR1, STEP1, CW, hori, loop_delay, hori_pulse) 

            
            elif control == " ":
                trigger()


            elif control == "quit":
                print("quitting\n")
                
                if up < 0:
                    down = up * (-1)   
                    rotate(DIR2, STEP2, CW, quit_vert, loop_delay, down)     
                
                if up > 0:
                    rotate(DIR2, STEP2, CCW, quit_vert, loop_delay, up)   
                    
                if right < 0:
                    left = right * (-1)  
                    rotate(DIR1, STEP1, CW, quit_hori, loop_delay, left)          
        
                if right > 0:
                    rotate(DIR1, STEP1, CCW, quit_hori, loop_delay, right)
        
                
                break


            else:
                print("INVALID INPUT\n")


    elif mode == "2":

        print("Please choose the mode of operation below\n1. Fast\n2. Accurate\n")
        mode2 = input("Enter the no. of mode of operation here: ")

        if mode2 == "1":
            # initialize the HOG descriptor/person detector
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

            cv2.startWindowThread()

            # open webcam video stream
            cap = cv2.VideoCapture(0)


            while(True):
                a = 0
                b = 0
                # Capture frame-by-frame
                ret, frame = cap.read()

                # resizing for faster detection
                frame = cv2.resize(frame, (426, 240))
                # using a greyscale picture, also for faster detection
                #gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

                # detect people in the image
                # returns the bounding boxes for the detected objects
                boxes, weights = hog.detectMultiScale(frame, winStride=(4,4), padding=(16,16) )
                

                boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
                box = non_max_suppression_fast(boxes, 0.4)

                cv2.rectangle(frame, ((frame.shape[1]//2)-10,(frame.shape[0]//2)-10), ((frame.shape[1]//2)+10,(frame.shape[0]//2)+10), (255,0,0), thickness = 2)
                


                for (xA, yA, xB, yB) in box:
                    # display the detected boxes in the colour picture
                    cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
                    a = int((xA+xB)/2)
                    b = int((yA+yB)/2)
                    
                    
                
                    if a == 0 and b == 0:
                        if K <= 0:
                            rotate(DIR1, STEP1, CW, hori, loop_delay, hori_pulse_big)
                            K = K-1
                            if K == -10:
                                K = 1
                        else:
                            rotate(DIR1, STEP1, CCW, hori, loop_delay, hori_pulse_big)
                            K = K+1
                            if K == 21:
                                K = 0




                    while b != 0 and b < frame.shape[0]//2 - 60:
                        print("UP\n")
                        up = up + 2
                        rotate(DIR2, STEP2, CCW, 0.1, loop_delay, vert_pulse) 


                    while b != 0 and b > frame.shape[0]//2 + 60:
                        print("DOWN\n")
                        up = up - 2
                        rotate(DIR2, STEP2, CW, 0.1, loop_delay, vert_pulse) 


                    while a != 0 and a < frame.shape[1]//2 - 40:
                        print("LEFT\n")
                        right = right - 5
                        rotate(DIR1, STEP1, CW, 0.1, loop_delay, hori_pulse) 


                    while a != 0 and a > frame.shape[1]//2 + 40:
                        print("RIGHT\n")
                        right = right + 5
                        rotate(DIR1, STEP1, CCW, 0.1, loop_delay, hori_pulse)



                cv2.imshow('frame',frame)





                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("quitting\n")
                    
                    if up < 0:
                        down = up * (-1)   
                        rotate(DIR2, STEP2, CW, quit_vert, loop_delay, down)     
                    
                    elif up > 0:
                        rotate(DIR2, STEP2, CCW, quit_vert, loop_delay, up)   
                        
                    if right < 0:
                        left = right * (-1)  
                        rotate(DIR1, STEP1, CW, quit_hori, loop_delay, left)          
            
                    elif right > 0:
                        rotate(DIR1, STEP1, CCW, quit_hori, loop_delay, right)
            
                    
                
                    break

            # When everything done, release the capture
            cap.release()

            # finally, close the window
            cv2.destroyAllWindows()
            cv2.waitKey(1)


        elif mode2 == "2":
            net = cv2.dnn.readNet('yolov3-tiny.weights','yolov3-tiny.cfg')
            classes = []
            with open('coco.names','r') as f:
                classes = f.read().splitlines()

            cap = cv2.VideoCapture(0)





            while True:
                _, img = cap.read()
                img = cv2.resize(img, (640, 480))
                height, width, _ = img.shape
                cv2.rectangle(img, (310, 230), (330, 250), (255, 0, 0), 2)

                blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)

                net.setInput(blob)
                output_layers_names = net.getUnconnectedOutLayersNames()
                layerOutputs = net.forward(output_layers_names)

                boxes = []
                confidences = []
                class_ids = []
                highest_area = 100

                for output in layerOutputs:
                    for detection in output:
                        scores = detection[5:]

                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        if confidence > 0.3:
                            center_x = int(detection[0] * width)
                            center_y = int(detection[1] * height)
                            w = int(detection[2] * width)
                            h = int(detection[3] * height)
                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)
                            
                            if class_id == 0:
                                print("Human Detected")
                                current_area = w*h
                                if current_area > highest_area:
                                    boxes = []
                                    confidences = []
                                    class_ids = []
                                    boxes.append([x, y, w, h])
                                    confidences.append((float(confidence)))
                                    class_ids.append(class_id)
                                    print(class_ids)
                                    highest_area = current_area
                                    print("Highest area is" +str(highest_area))

                                else:
                                    print("Lower area found")
                                
                            else:
                                print("No Human Found")
                                counter = counter + 1

                                if counter < 3:
                                    print("\n")
                                else:
                                    counter = 0
                                    if K <= 0:
                                        rotate(DIR1, STEP1, CW, hori, loop_delay, hori_pulse_big)
                                        K = K-1
                                        if K == -10:
                                            K = 1
                                        else: 
                                            print("\n")
                                    else:
                                        rotate(DIR1, STEP1, CCW, hori, loop_delay, hori_pulse_big)
                                        K = K+1
                                        if K == 21:
                                            K = 0
                                        else: 
                                            print("\n")
                            


                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)

                font = cv2.FONT_HERSHEY_PLAIN
                colors = np.random.uniform(0, 255, size=(len(boxes), 3))

                if len(indexes)>0:
                    for i in indexes.flatten():
                        x, y, w, h = boxes[i]
                        
                        label = str(classes[class_ids[i]])

                        confidence = str(round(confidences[i], 2))
                        color = colors[i]
                        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(img, label + "" + confidence, (x, y + 20), font, 2, (255, 255, 255), 2)
                        target_x = x + 0.5*w
                        target_y = y + 0.5*h
                        current_x = 320
                        current_y = 240
                        if target_x - current_x > 20 and target_x - current_x > 50:   #large right
                            rotate(DIR1, STEP1, CCW, hori, loop_delay, hori_pulse*3)
                            
                        elif target_x - current_x > 20:    #small right
                            rotate(DIR1, STEP1, CCW, hori, loop_delay, hori_pulse)
                            
                        elif current_x - target_x > 20 and current_x - target_x > 50:   #large left
                            rotate(DIR1, STEP1, CW, hori, loop_delay, hori_pulse*3)
                            
                        elif target_x - current_x > 20:    #small left
                            rotate(DIR1, STEP1, CW, hori, loop_delay, hori_pulse)
                            
                        else:
                            print("X Position Reached")

                        '''if target_y - current_y > 40 and target_y - current_y > 50:   #large up
                            rotate(DIR2, STEP2, CW, vert, loop_delay, vert_pulse)
                            continue'''
                        if target_y - current_y > 20:    #small up
                            rotate(DIR2, STEP2, CW, vert, loop_delay, vert_pulse)
                            
                        '''elif current_y - target_y > 40 and current_y - target_y > 50:   #large down
                            rotate(DIR2, STEP2, CCW, vert, loop_delay, vert_pulse)
                            continue'''
                        if target_y - current_y > 20:    #small down
                            rotate(DIR2, STEP2, CCW, vert, loop_delay, vert_pulse)
                            
                        else:
                            print("Y Position Reached")



                
                    #sleep(1)
                            
                cv2.imshow('Image', img)
                key = cv2.waitKey(1)
                if key == 27:
                    break

            cap.release()
            cv2.destroyAllWindows()

    elif mode =="0":
        while True:
            
            sleep(1.0)
            GPIO.output(DIR1, CW)
            for x in range(200):
                GPIO.output(STEP1, GPIO.HIGH)
                sleep(.025)
                GPIO.output(STEP1, GPIO.LOW)
                sleep(.025)
            sleep(1.0)
            GPIO.output(DIR1, CCW)
            for x in range(200):
                GPIO.output(STEP1, GPIO.HIGH)
                sleep(.025)
                GPIO.output(STEP1, GPIO.LOW)
                sleep(.025)
            sleep(1.0)
            GPIO.output(DIR2, CW)
            for x in range(10):
                GPIO.output(STEP2, GPIO.HIGH)
                sleep(.025)
                GPIO.output(STEP2, GPIO.LOW)
                sleep(.025)
            sleep(1.0)
            GPIO.output(DIR2, CCW)
            for x in range(10):
                GPIO.output(STEP2, GPIO.HIGH)
                sleep(.025)
                GPIO.output(STEP2, GPIO.LOW)
                sleep(.025)
            break
        


    else:
        GPIO.cleanup()
        break