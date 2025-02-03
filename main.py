import cv2
import numpy as np

from perspective_warping.warper import ImageWarper
from core.camera import CameraCalibrator
from ar_rendering.cube_renderer import CubeRenderer
from utils.video_utils import VideoHandler, video_to_frames


video_name = '1'
output_name = 'output_' + video_name

# ======= constants
template_path = r'C:\Users\Avi Lezerovich\Documents\GitHub\augmented-reality\data\tmp9.png'
overlay_image_path = r'C:\Users\Avi Lezerovich\Documents\GitHub\augmented-reality\data\my_overlay.jpg'
video_path = r'C:\Users\Avi Lezerovich\Documents\GitHub\augmented-reality\data\\'  + video_name + '.mp4'



# # Calibrate camera
calibrator = CameraCalibrator()
calibrator.load_coefficients("data/calibration.xml")
K = calibrator.camera_matrix
dist_coeffs = calibrator.dist_coefs


# ====== create warper object
warper = ImageWarper(template_path, overlay_image_path)

# ====== create cube renderer object
renderer = CubeRenderer(K, dist_coeffs, warper.get_template_size())

# ===== video input, output and metadata
video = VideoHandler(video_path, output_name)
# video = VideoHandler(video_path)

# ========== run on all frames
while True:
    ret, frame, frame_count = video.read_frame()
    if not ret or cv2.waitKey(1) == ord('q'):
        break
    
    
    print('frame:', frame_count)
    # Find homography
    H  = warper.find_homography(frame , display_matches=True)
    # match_frame = warper.get_match_frame()
    
    wrp = warper.apply_warp(frame, H)
    # cube =  renderer.render_cube(frame.copy(), H)

    # img = merge_images_side_by_side(match_frame, wrp, cube)
    # img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))
    video.write_frame(wrp)
    # cv2.imshow('result', wrp)


# ======== end all
video.release()
