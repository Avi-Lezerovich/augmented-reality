import cv2
import numpy as np

from perspective_warping.warper import ImageWarper
from core.camera import CameraCalibrator
from ar_rendering.cube_renderer import CubeRenderer
from utils.video_utils import VideoHandler, merge_images_side_by_side,  merge_images_top_to_bottom


video_name = '1'


# ======= constants
template_path = r'C:\Users\Avi Lezerovich\Documents\GitHub\augmented-reality\data\tmp9.png'
overlay_image_path = r'C:\Users\Avi Lezerovich\Documents\GitHub\augmented-reality\data\my_overlay.jpg'
video_path = r'C:\Users\Avi Lezerovich\Documents\GitHub\augmented-reality\data\\'  + video_name + '.mp4'

model_path = r'C:\Users\Avi Lezerovich\Documents\GitHub\augmented-reality\data\models\teapot.obj'


# # Calibrate camera
calibrator = CameraCalibrator()
calibrator.load_coefficients("data/calibration.xml")
K = calibrator.camera_matrix
dist_coeffs = calibrator.dist_coefs


# ====== create warper object
warper = ImageWarper(template_path, overlay_image_path)

# ====== create cube renderer object
renderer = CubeRenderer(K, dist_coeffs, warper.get_template_size(), model_path=model_path)

# ===== video input, output and metadata
video = VideoHandler(video_path)
wrp_out = VideoHandler(video_path, output_name='wrp_out')
cube_out = VideoHandler(video_path, output_name='cube_out')
model_out = VideoHandler(video_path, output_name='model_out')
all_out = VideoHandler(video_path, output_name='all_out')

prev_H = None 
# ========== run on all frames
while True:
    ret, frame, frame_count = video.read_frame()
    if not ret or cv2.waitKey(1) == ord('q'):
        break
    

    print('frame:', frame_count)
    # Find homography
    H  = warper.find_homography(frame , display_matches=True)
    
    match_frame = warper.get_match_frame()
    wrp = warper.apply_warp(frame, H)
    cube =  renderer.render_cube(frame.copy(), H)
    model = renderer.render_model(frame.copy(), H)

    img1 = merge_images_top_to_bottom(model, cube)
    img2 = merge_images_top_to_bottom(wrp, match_frame)
    
    img = merge_images_side_by_side(img1, img2)
    
    img_resized = cv2.resize(img, (video.frame_width, video.frame_height))

    wrp_out.write_frame(wrp)
    cube_out.write_frame(cube)
    model_out.write_frame(model)
    all_out.write_frame(img_resized)
    
    img = cv2.resize(img, (int(img.shape[0]/2), int(img.shape[1]/2)))

    cv2.imshow('result', img)
    prev_H = H

# ======== end all
video.release()
wrp_out.release()
cube_out.release()
model_out.release()
all_out.release()