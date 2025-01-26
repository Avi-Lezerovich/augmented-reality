import cv2
import numpy as np

from perspective_warping.warper import ImageWarper

# ======= constants
template_path = r'C:\Users\Avi Lezerovich\Documents\GitHub\augmented-reality\data\tmp.png'
art_path = r'C:\Users\Avi Lezerovich\Documents\GitHub\augmented-reality\data\my_overlay.jpg'
video_path = r'C:\Users\Avi Lezerovich\Documents\GitHub\augmented-reality\data\video.mp4'
output_video_path = r'C:\Users\Avi Lezerovich\Documents\GitHub\augmented-reality\data\output_video.mp4'


cap = cv2.VideoCapture(video_path)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = int(cap.get(cv2.CAP_PROP_FPS))


# Use 'mp4v' codec for mp4 format
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (frame_width, frame_height))

warper = ImageWarper(template_path, art_path)

# ===== video input, output and metadata

# ========== run on all frames
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    print('frame:', frame_count)
    res = warper.apply_warp(frame)
    out.write(res)

# ======== end all
cap.release()
out.release()
cv2.destroyAllWindows()
