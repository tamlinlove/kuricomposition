import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

image_dir = "calibration/"

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
    front = plt.imread(image_dir+"front.png")
    back = plt.imread(image_dir+"back.png")
    return [front,back]

def get_colour_averages():
    colour_images = get_colour_images()
    avs = []
    for c in colour_images:
        av = np.mean(c,axis=(0,1))
        avs.append(av[0:3])
    return avs

def detect_single_colour(image,upper,lower):
    mask = cv2.inRange(image, lower, upper)
    res = cv2.bitwise_and(image,image, mask= mask)
    pix = np.where(mask==255)
    pixels = np.array([pix[0],pix[1]])
    centre = np.mean(pixels,axis=1)
    return centre

def find_centres(image,tol=0.05,draw=False,frame=None,radius=30):
    averages = get_colour_averages()
    plt.imsave("front_average.png",np.ones((100,100,3))*averages[0].reshape(1,1,3))
    plt.imsave("back_average.png",np.ones((100,100,3))*averages[1].reshape(1,1,3))
    plt.close()
    front_upper = averages[0]+np.ones(3)*tol
    front_lower = averages[0]-np.ones(3)*tol
    back_upper = averages[1]+np.ones(3)*tol
    back_lower = averages[1]-np.ones(3)*tol

    upper_bounds = [front_upper,back_upper]
    lower_bounds = [front_lower,back_lower]

    centres = []

    '''
    for i in range(len(averages)):
        mask = cv2.inRange(image, lower_bounds[i], upper_bounds[i])
        res = cv2.bitwise_and(image,image, mask= mask)
        pix = np.where(mask==255)
        pixels = np.array([pix[0],pix[1]])
        centre = np.mean(pixels,axis=1)
        centres.append(centre)
    '''

    start_index = 0
    end_index = 1

    # First, detect front which is distinct
    centre = detect_single_colour(image,upper_bounds[start_index],lower_bounds[start_index])
    centres.append(centre)

    # Now, search for back within a radius of front
    centre_int = [int(centre[1]),int(centre[0])]
    start_point = (centre_int[0]-radius,centre_int[1]-radius)
    end_point = (centre_int[0]+radius,centre_int[1]+radius)
    cv2.rectangle(frame,start_point,end_point,(0,255,0),2)

    sub_image = image[centre_int[1]-radius:centre_int[1]+radius,centre_int[0]-radius:centre_int[0]+radius]
    centre = detect_single_colour(sub_image,upper_bounds[end_index],lower_bounds[end_index])
    centre = [centre[0]+centre_int[1]-radius,centre[1]+centre_int[0]-radius]
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


def display_text(frame,text,position=(40,40),font=cv2.FONT_HERSHEY_SIMPLEX,fontScale = 1,fontColor = (255,255,255),lineType = 2):
    cv2.putText(frame,text, position, font, fontScale,fontColor,lineType)

def get_position(image,front=(0,0),back=(0,0)):
    try:
        centres = find_centres(image,tol=0.1)
        coordinates = get_coordinates(centres)

        front = coordinates[0]
        back = coordinates[1]
        return front,back
    except:
        return front,back
    

def crop_image(image):
    image = image[:,80:-1]
    return image

def update_image(cap):
    image,frame = get_frame(cap)
    #image = crop_image(image)
    return image,frame

def draw_frame(frame,draw_cirlces=None):

    if draw_cirlces is not None:
        for point in draw_cirlces:
            cv2.circle(frame,draw_cirlces[point],5,(0,0,0),2)

    cv2.imshow('frame',frame)


def simple_localisation_test():

    cap =  cap = cv2.VideoCapture(CAMERA_NUMBER)
    coordinates = [(0,0),(0,0),(0,0)]

    while True:
        try:
            image,frame = get_frame(cap)
            image = image[:,80:-1]
            frame = frame[:,80:-1]
            #cv2.rectangle(frame,(80,0),(600,600),(0,0,255),2)
            centres = find_centres(image,tol=0.1,draw=True,frame=frame)
            coordinates = get_coordinates(centres)
        except Exception as e:
            #print(e)
            pass

        front = coordinates[0]
        back = coordinates[1]

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
    cap = cv2.VideoCapture(CAMERA_NUMBER)
    ret, frame = cap.read() # Capture frame-by-frame
    print("Saving to {}".format(directory))
    cv2.imwrite(directory, frame)

    cap.release()

def get_all_state_coords(top_left,bottom_right):
    X_OFFSET = top_left[0]
    Y_OFFSET = top_left[1]
    X_INTERVAL = (bottom_right[0]-top_left[0])//9
    Y_INTERVAL = (bottom_right[1]-top_left[1])//9



    points = []
    for i in range(10):
        for j in range(10):
            points.append((X_OFFSET+i*X_INTERVAL,Y_OFFSET+j*Y_INTERVAL))
    return points


def draw_states():
    cap =  cap = cv2.VideoCapture(CAMERA_NUMBER)

    while True:
        image,frame = get_frame(cap)

        top_left = (80,10)
        bottom_right = (620,460)

        cv2.rectangle(frame,top_left,bottom_right,(0,255,0),2)

        points = get_all_state_coords(top_left,bottom_right)
        for point in points:
            cv2.circle(frame,point,5,(0,0,255),1)

        #for i in range(10):
        #    cv2.line(frame,(0,i*(480//9)),(640,i*(480//9)),(255,0,0),2)

        cv2.imshow('frame',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()




if __name__ == "__main__":
    #camera_test(draw_colours=True)
    #take_picture(image_dir+'calibration.png')
    #simple_localisation_test()
    #qr_localiser()
    draw_states()
