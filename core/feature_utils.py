import cv2
import numpy as np


class FeatureMatcher:
    def __init__(self):
        self.sift = cv2.SIFT_create()
        self.bf = cv2.BFMatcher()
        
    display = False
        

    def find_keypoints_and_matches(self, img1, img2):
        gray_1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray_2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        kp_l, desc_l = self.sift.detectAndCompute(gray_1, None)
        kp_r, desc_r = self.sift.detectAndCompute(gray_2, None)
        
       

        if self.display:
            test = cv2.drawKeypoints(img1, kp_l, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imshow('keypoint', test)
            cv2.waitKey(0)
            
        matches = self.bf.knnMatch(desc_l, desc_r, k=2)
        
        if self.display:
            img3 = cv2.drawMatchesKnn(img1, kp_l, img2, kp_r, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.imshow('matches', img3)
            cv2.waitKey(0)
            
        return kp_l, kp_r, matches
        
    
    def find_homography(self, kp1, kp2, matches):
        good_kp1 = np.array([kp1[m.queryIdx].pt for m in matches])
        good_kp2 = np.array([kp2[m.trainIdx].pt for m in matches])
        H, masked = cv2.findHomography(good_kp2, good_kp1, cv2.RANSAC, 5.0)
        
        return H, masked
    
