from cmath import isnan
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import time

image_dir = "calibration/"

CAMERA_NUMBER = 0

directory = "footage/"
format = ".avi"

def get_output_filename(frame_name):
    #print(time.ctime(time.time()))
    timestamp = time.ctime(time.time()).split(" ")[3].split(":")
    #print(timestamp)
    filename = frame_name+"-"+timestamp[0]+"-"+timestamp[1]+"-"+timestamp[2]
    return filename

def display(frame_name,frame,out=None):
    cv2.imshow(frame_name,frame)
    if out:
        out.write(frame)


DIRECTIONS = ["UP","RIGHT","DOWN","LEFT"]

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
    front_dark = plt.imread(image_dir+"front_dark.png")
    back_dark = plt.imread(image_dir+"back_dark.png")
    return [front,back,front_dark,back_dark]

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

def find_centres(image,tol=0.1,draw=False,frame=None,radius=30):
    averages = get_colour_averages()
    #plt.imsave("front_average.png",np.ones((100,100,3))*averages[0].reshape(1,1,3))
    #plt.imsave("back_average.png",np.ones((100,100,3))*averages[1].reshape(1,1,3))
    #plt.close()
    front_upper = averages[0]+np.ones(3)*tol
    front_lower = averages[0]-np.ones(3)*tol
    back_upper = averages[1]+np.ones(3)*tol
    back_lower = averages[1]-np.ones(3)*tol
    front_upper_dark = averages[2]+np.ones(3)*tol
    front_lower_dark = averages[2]-np.ones(3)*tol
    back_upper_dark = averages[3]+np.ones(3)*tol
    back_lower_dark = averages[3]-np.ones(3)*tol

    upper_bounds = [front_upper,back_upper,front_upper_dark,back_upper_dark]
    lower_bounds = [front_lower,back_lower,front_lower_dark,back_lower_dark]

    centres = []

    start_index = 0
    end_index = 1
    start_index_dark = 2
    end_index_dark = 3

    # First, detect front which is distinct
    centre = detect_single_colour(image,upper_bounds[start_index],lower_bounds[start_index])
    if math.isnan(centre[0]):
        # Try detect dark instead
        centre = detect_single_colour(image,upper_bounds[start_index_dark],lower_bounds[start_index_dark])
    centres.append(centre)

    # Now, search for back within a radius of front
    centre_int = [int(centre[1]),int(centre[0])]
    start_point = (centre_int[0]-radius,centre_int[1]-radius)
    end_point = (centre_int[0]+radius,centre_int[1]+radius)
    cv2.rectangle(frame,start_point,end_point,(0,255,0),2)

    sub_image = image[centre_int[1]-radius:centre_int[1]+radius,centre_int[0]-radius:centre_int[0]+radius]
    centre = detect_single_colour(sub_image,upper_bounds[end_index],lower_bounds[end_index])
    if math.isnan(centre[0]+centre_int[1]-radius):
        centre = detect_single_colour(sub_image,upper_bounds[end_index_dark],lower_bounds[end_index_dark])
    centre = [centre[0]+centre_int[1]-radius,centre[1]+centre_int[0]-radius]
    centres.append(centre)



    if draw and frame is not None:
        drawOnFeed(frame,centres,averages)
    return centres

def find_centres_hue(image,draw=False,frame=None):
    image = np.float32(image)
    image_blur = cv2.GaussianBlur(image,(11,11),cv2.BORDER_DEFAULT)
    #image_blur = np.float32(image_blur)
    
    image_hsv = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    image_hue = (image_hsv[:,:,0]/255)*(image_hsv[:,:,1])

    image_thresh = image_hsv[:,:,1]
    image_thresh = np.where(image_thresh>0.3*np.max(image_thresh),1.0,0.0).reshape(image.shape[0],image.shape[1],1)
    #print(image_hue)
    image_mask = image*image_thresh
    image_mask = np.float32(image_mask)
    #cv2.imshow('frame_2',cv2.cvtColor(image_mask,cv2.COLOR_RGB2BGR))

    #plt.imsave("calibration_mask.png",image_mask)  

    centres = find_centres(image_mask)

    

    #plt.imsave("calibration_mask.png",image_mask)    

    

    if draw and frame is not None:
        drawOnFeed(frame,centres,None)
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

        display("frame",frame)

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
        #centres = find_centres(image,tol=0.2)
        centres = find_centres_hue(image)
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

def camera_sleep(cap,sleep_time=1,draw_circles=None):
    start_time = time.time()

    while True:
        _,frame = update_image(cap)
        draw_frame(frame,draw_cirlces=draw_circles)

        if time.time()-start_time > sleep_time:
            break

def draw_frame(frame,out=None,draw_cirlces=None):

    if draw_cirlces is not None:
        for point in draw_cirlces:
            cv2.circle(frame,draw_cirlces[point],5,(0,0,0),2)

    display("frame",frame,out)


def simple_localisation_test():

    cap =  cap = cv2.VideoCapture(CAMERA_NUMBER)
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    filename = get_output_filename("frame")
    out = cv2.VideoWriter(directory+filename+format, fourcc, 20.0, (int(cap.get(3)),int(cap.get(4))))
    coordinates = [(0,0),(0,0),(0,0)]

    while True:
        '''
        try:
            image,frame = get_frame(cap)
            #cv2.rectangle(frame,(80,0),(600,600),(0,0,255),2)
            #centres = find_centres(image,tol=0.1,draw=True,frame=frame)
            centres = find_centres_hue(image,draw=True,frame=frame)
            coordinates = get_coordinates(centres)
        except Exception as e:
            #print(e)
            pass
        '''
        try:
            image,frame = get_frame(cap)
            centres = find_centres_hue(image,draw=True,frame=frame)
            coordinates = get_coordinates(centres)
        except Exception as e:
            print(e)

        front = coordinates[0]
        back = coordinates[1]

        cv2.arrowedLine(frame,back,front,color=(0,0,0),thickness=2)

        angle = get_angle(back,front)
        display_text(frame,"Angle:{} degrees".format(int(math.degrees(angle))),(40,40))

        display("frame",frame,out)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            #plt.imsave("localise.png",cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            break

    cap.release()
    out.release()
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

    cols = []
    centre_coords = [(2,7),(7,2),(2,2),(7,7)]

    points = []
    for i in range(10):
        point_row = []
        for j in range(10):
            point_row.append((X_OFFSET+i*X_INTERVAL,Y_OFFSET+j*Y_INTERVAL))
            if (i,j) in centre_coords:
                cols.append((255,0,0))
            else:
                cols.append((0,0,255))
        points.append(point_row)
    return points,cols

def get_room_states(offset=40,x_offset=160,y_offset=0):

    x_offset = (x_offset+offset)//2
    y_offset = (y_offset+offset)//2

    top_left = (x_offset,y_offset)
    bottom_right = (640-x_offset,480-y_offset)

    points,_ = get_all_state_coords(top_left,bottom_right)

    return points

def get_goal_from_state(state,offset=40,x_offset=160,y_offset=0):
    points = get_room_states(offset=offset,x_offset=x_offset,y_offset=y_offset)
    


    position = state[0]
    goal = points[position[1]][position[0]]
    goal_direction = DIRECTIONS[state[1]]

    print(goal)

    return goal,goal_direction
    
    






def draw_states():
    cap =  cap = cv2.VideoCapture(CAMERA_NUMBER)

    while True:
        image,frame = get_frame(cap)

        # size = (640,480)
        '''
        EDGE_OFFSET = 100

        top_left = (EDGE_OFFSET,EDGE_OFFSET)
        bottom_right = (640-EDGE_OFFSET,480-EDGE_OFFSET)

        cv2.rectangle(frame,top_left,bottom_right,(0,255,0),2)

        points,cols = get_all_state_coords(top_left,bottom_right)
        for i in range(len(points)):
            cv2.circle(frame,points[i],5,cols[i],1)
        '''
        
        points = get_room_states()
        for i in range(len(points)):
            for j in range(len(points[i])):
                cv2.circle(frame,points[i][j],5,(255,0,0),1)


        #for i in range(10):
        #    cv2.line(frame,(0,i*(480//9)),(640,i*(480//9)),(255,0,0),2)

        cv2.imshow('frame',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            #plt.imsave("state_map.png",cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            break

    cap.release()
    cv2.destroyAllWindows()




if __name__ == "__main__":
    #camera_test(draw_colours=True)
    #take_picture(image_dir+'calibration.png')
    simple_localisation_test()
    #qr_localiser()
    #draw_states()
