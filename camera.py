import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

image_dir = "images/"

CAMERA_NUMBER = 0

def get_angle(back,front):
    #horizontal = (front[0],back[1])
    x_dist = abs(front[0]-back[0])
    horizontal = (front[0]+x_dist,back[1])
    
    a = math.atan2(front[1]-back[1],front[0]-back[0])
    b = math.atan2(horizontal[1]-back[1],horizontal[0]-back[0])

    if a < 0: 
        a += math.pi*2
    if b < 0: 
        b += math.pi*2

    angle = (math.pi*2 + b - a) if a > b else (b - a)
    #return (math.pi*2 + b - a) if a > b else (b - a)
    return angle

def get_colour_images():
    red = plt.imread(image_dir+"red.png")
    green = plt.imread(image_dir+"green.png")
    blue = plt.imread(image_dir+"blue.png")
    return [red,green,blue]

def get_colour_averages():
    colour_images = get_colour_images()
    avs = []
    for c in colour_images:
        av = np.mean(c,axis=(0,1))
        avs.append(av[0:3])
    return avs

def find_centres(image,tol=0.05,draw=False,frame=None):
    averages = get_colour_averages()
    red_upper = averages[0]+np.ones(3)*tol
    red_lower = averages[0]-np.ones(3)*tol
    green_upper = averages[1]+np.ones(3)*tol
    green_lower = averages[1]-np.ones(3)*tol
    blue_upper = averages[2]+np.ones(3)*tol
    blue_lower = averages[2]-np.ones(3)*tol

    upper_bounds = [red_upper,green_upper,blue_upper]
    lower_bounds = [red_lower,green_lower,blue_lower]

    centres = []

    for i in range(len(averages)):
        mask = cv2.inRange(image, lower_bounds[i], upper_bounds[i])
        res = cv2.bitwise_and(image,image, mask= mask)
        pix = np.where(mask==255)
        pixels = np.array([pix[0],pix[1]])
        centre = np.mean(pixels,axis=1)
        centres.append(centre)

    if draw and frame is not None:
        drawOnFeed(frame,centres,averages)
    return centres

def drawOnFeed(frame,centres,averages):
    for i in range(len(centres)):
        if not(np.isnan(centres[i][0]) or np.isnan(centres[i][1])):
            #newCol = (int(averages[i][0]*255),int(averages[i][1]*255),int(averages[i][2]*255))
            newCol = (0,0,0)
            newC = (int(round(centres[i][1])),int(round(centres[i][0]))) #Reversed because image
            cv2.circle(frame,newC,5,newCol,2)

def get_frame(cap):
    _, frame = cap.read() # Capture frame-by-frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert from BGR to RGB
    image = np.array(frame_rgb).astype(np.float)/255 # Convert from Int8 to float

    return image,frame


def camera_test(draw_colours = False):
    cap = cv2.VideoCapture(CAMERA_NUMBER)


    while True:
        image,frame = get_frame(cap)

        centres = find_centres(image,tol=0.1,draw=draw_colours,frame=frame)

        cv2.imshow('frame',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def get_coordinates(centres):
    coords = []
    for centre in centres:
        coords.append((int(round(centre[1])),int(round(centre[0]))))
    return coords


def display_text(frame,text,position,font=cv2.FONT_HERSHEY_SIMPLEX,fontScale = 1,fontColor = (255,255,255),lineType = 2):
    cv2.putText(frame,text, position, font, fontScale,fontColor,lineType)

def simple_localisation_test():

    cap =  cap = cv2.VideoCapture(CAMERA_NUMBER)
    coordinates = [(0,0),(0,0),(0,0)]

    while True:
        try:
            image,frame = get_frame(cap)
            centres = find_centres(image,tol=0.1,draw=False,frame=frame)
            coordinates = get_coordinates(centres)
        except:
            pass

        front = coordinates[0] # RED
        back = coordinates[1] # GREEN

        cv2.arrowedLine(frame,back,front,color=(0,0,0),thickness=2)

        angle = get_angle(back,front)
        display_text(frame,"Angle:{} degrees".format(int(math.degrees(angle))),(40,40))

        cv2.imshow('frame',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def find_qrs(frame):
    detect = cv2.QRCodeDetector()
    retval, decoded_info, points, _ = detect.detectAndDecodeMulti(frame)
    return retval, decoded_info, points


def qr_localiser():
    cap =  cap = cv2.VideoCapture(CAMERA_NUMBER)

    coords = {'front':(0,0),'back':(0,0)}

    while True:
        detected = {'front':False,'back':False}
        try:
            image,frame = get_frame(cap)
            retval, decoded_info, points = find_qrs(frame)
            if retval:
                for i in range(len(decoded_info)):
                    if decoded_info[i] in coords.keys():
                        detected[decoded_info[i]] = True
                        average = [0,0]
                        for corner in points[i]:
                            average[0] = average[0] + corner[0]
                            average[1] = average[1] + corner[1]
                        coords[decoded_info[i]] = (int(round(average[0]/len(points[i]))),int(round(average[1]/len(points[i]))))
        except:
            pass
        
        cv2.arrowedLine(frame,coords['back'],coords['front'],color=(0,0,0),thickness=2)
        angle = get_angle(coords['back'],coords['front'])
        display_text(frame,"Angle:{} degrees".format(int(math.degrees(angle))),(40,40))
        detected_str = ""
        if detected['front']:
            detected_str += "Front   "
        if detected['back']:
            detected_str += "Back"

        display_text(frame,detected_str,(40,80))

        cv2.imshow('frame',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



def take_picture(directory):
    cap = cv2.VideoCapture(1)
    ret, frame = cap.read() # Capture frame-by-frame
    print("Saving to {}".format(directory))
    cv2.imwrite(directory, frame)

    cap.release()

if __name__ == "__main__":
    #camera_test(draw_colours=True)
    #take_picture(image_dir+'calibration.png')
    #simple_localisation_test()
    qr_localiser()