import cv2
import numpy as np

from perspective_warping.warper import ImageWarper
from core.camera import CameraCalibrator
from ar_rendering.cube_renderer import CubeRenderer
from utils.video_utils import VideoHandler

# ======= constants
template_path = r'C:\Users\Avi Lezerovich\Documents\GitHub\augmented-reality\data\tmp.png'
overlay_image_path = r'C:\Users\Avi Lezerovich\Documents\GitHub\augmented-reality\data\my_overlay.jpg'
video_path = r'C:\Users\Avi Lezerovich\Documents\GitHub\augmented-reality\data\v1.mp4'


# Calibrate camera
calibrator = CameraCalibrator()
calibrator.load_coefficients("data/calibration.xml")
K = calibrator.camera_matrix
dist_coeffs = calibrator.dist_coefs


# ====== create warper object
warper = ImageWarper(template_path, overlay_image_path)

# ====== create cube renderer object
renderer = CubeRenderer(K, dist_coeffs, warper.get_template_size()) 

# ===== video input, output and metadata
video = VideoHandler(video_path, save_video=True)


# ========== run on all frames
while True:
    ret, frame, frame_count= video.read_frame()
    if not ret or cv2.waitKey(1) == ord('q'):
        break
  
    print('frame:', frame_count)
    
    
    # Find homography
    H  = warper.find_homography(frame)
    frame = renderer.render_cube(frame, H)
        
        
    video.write_frame(frame)
    cv2.imshow('result',frame)
    
    

# ======== end all
video.release()
