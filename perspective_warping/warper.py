import cv2
import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.feature_utils import find_keypoints, find_matches, find_homography, draw_matches



class ImageWarper:
    def __init__(self, template_path, img_to_be_warp):
        # === template image keypoint and descriptors
        self.tmp_img =  cv2.imread(template_path)       
        self.tmp_kp, self.tmp_des = find_keypoints(self.tmp_img)
        
        

        overlay_image = self.new_method(img_to_be_warp)
        h, w = self.tmp_img.shape[:2]
            
        # Warp art image to fit book cover
        self.art_img = cv2.resize(overlay_image, (w, h))
        self.h, self.w = h, w
        self.prev_mask = None



    def new_method(self, img_to_be_warp):
        overlay_image = cv2.imread(img_to_be_warp)
        return overlay_image



    def apply_warp(self, img, H):   
        if H is None:
            return img
        
        warped_art = cv2.warpPerspective(self.art_img, H, (img.shape[1], img.shape[0]))
        
        # Create mask for blending (single channel)
        warped_mask = cv2.warpPerspective(np.ones((self.h, self.w), dtype=np.uint8) * 255, H, (img.shape[1], img.shape[0]))
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        mask[warped_mask > 0] = 255
        
        self.prev_mask = mask
        result = img.copy()
        result[mask > 0] = warped_art[mask > 0]
        
        return result
 
 
    def find_homography(self, img, display_matches=False):
        # ====== find keypoints matches of frame and template
        img_in_IRO = self.in_range_of_last_mask(img)
        img_kp, img_des = find_keypoints(img_in_IRO)
        
        matches = find_matches(self.tmp_des, img_des)
    
        # ======== find homography
        H  = find_homography(self.tmp_kp, img_kp, matches)
       

        if display_matches:
            self.matches_frame =  draw_matches(self.tmp_img, self.tmp_kp, img, img_kp , matches)
    
        return H 
    
    def get_match_frame(self):
        return self.matches_frame
    
    def get_template_size(self):
        return self.h, self.w
    
    
    def in_range_of_last_mask(self, img):
        if self.prev_mask is None:
            return img
    
        
        result = img.copy()
        result[self.prev_mask == 0] = 0
        return result