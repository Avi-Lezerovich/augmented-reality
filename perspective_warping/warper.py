import cv2
import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.feature_utils import find_keypoints, find_matches, find_homography



class ImageWarper:
    def __init__(self, template_path, img_to_be_warp):
        # === template image keypoint and descriptors
        self.tmp_img = cv2.imread(template_path)
        self.tmp_kp, self.tmp_des = find_keypoints(self.tmp_img)

        self.art_img = cv2.imread(img_to_be_warp)
        h, w = self.tmp_img.shape[:2]
            
        # Warp art image to fit book cover
        self.art_img = cv2.resize(self.art_img, (w, h))
        self.h, self.w = h, w

    def _warp_image(self, img ,H ,mask): 
        if H is None or mask is None:
            return img
        
        warped_art = cv2.warpPerspective(self.art_img, H, (img.shape[1], img.shape[0]))
        
        # Create mask for blending
        mask = np.zeros(img.shape, dtype=np.uint8)
        warped_mask = cv2.warpPerspective(np.ones((self.h, self.w), dtype=np.uint8) * 255, H, 
                                        (img.shape[1], img.shape[0]))
        mask[warped_mask > 0] = 255
        
        # Blend images
        result = img.copy()
        result[mask > 0] = warped_art[mask > 0]
        
        return result

    def apply_warp(self, img):
        img_kp, img_des = find_keypoints(img)
        matches = find_matches(self.tmp_des, img_des)
    
        # ======== find homography
        H , mask = find_homography(self.tmp_kp, img_kp, matches)

        # ++++++++ do warping of another image on template image
        return self._warp_image(img, H, mask)
