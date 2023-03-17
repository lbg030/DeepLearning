import cv2
vidcap = cv2.VideoCapture('/Users/gwonsmpro/Documents/GitHub/DeepLearning/video_detection/helmet.mp4')
 
count = 0
 
while(vidcap.isOpened()):
    ret, image = vidcap.read()
 
    if(int(vidcap.get(1)) % 20 == 0):
        print('Saved frame number : ' + str(int(vidcap.get(1))))
        cv2.imwrite("/Users/gwonsmpro/Documents/GitHub/DeepLearning/video_detection/img/helmet_%d.jpg" % count, image)
        print('Saved frame%d.jpg' % count)
        count += 1

vidcap.release()