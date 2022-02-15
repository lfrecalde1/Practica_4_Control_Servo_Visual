#!/home/fer/.virtualenvs/cv/bin/python
import numpy as np
import cv2 
def create_video():
    ## Read camera video
    capture = cv2.VideoCapture(0)

    ## place where the video will be save
    video = cv2.VideoWriter("Prueba_interior.avi", cv2.VideoWriter_fourcc(*'XVID'), 20.0, (640, 480))

    while (capture.isOpened()):
        ret, img = capture.read()
        if ret == True:
            cv2.imshow('Video', img)
            video.write(img)
            if cv2.waitKey(1) & 0xFF == ord('s'):
                break
        else: break
    capture.release()
    video.release()
    cv2.destroyAllWindows()
    return None

def read_video():
    cap = cv2.VideoCapture("Prueba_interior.avi")
    while (cap.isOpened()):
        ret, img = cap.read()
        if ret == True:
            cv2.imshow("Frame", img)
            if cv2.waitKey(30) == ord('q'):
                break
        else: break
    cap.release()
    cv2.destroyAllWindows()


def optical_flow():
    cap = cv2.VideoCapture("Prueba_interior.avi")
    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # Create some random colors
    color = np.random.randint(0,255,(100,3))
    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    while(1):
        ret,frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # Select good points
        if p1 is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]
        # draw the tracks
        for i,(new,old) in enumerate(zip(good_new, good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv2.line(mask, (int(a),int(b)),(int(c),int(d)), color[i].tolist(), 2)
            frame = cv2.circle(frame,(int(a),int(b)),5,color[i].tolist(),-1)
        img = cv2.add(frame,mask)
        cv2.imshow('frame',img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)

    cap.release()
    cv2.destroyAllWindows()
    return None

def main():
    print("hola luis fernando")
    #create_video()
    read_video()
    #optical_flow()

if __name__ == '__main__':
    main()
