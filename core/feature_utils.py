import cv2
import numpy as np

sift = cv2.SIFT_create()
bf = cv2.BFMatcher()

def find_keypoints(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return sift.detectAndCompute(gray, None)
    
def find_matches(des1, des2):
    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        return []
    
    # Pass k as a positional argument
    matches = bf.knnMatch(des1, des2, 2)
    
    if matches is None or len(matches) <4:
        return []
    
    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:  # Apply ratio test
            good_matches.append(m)

    return good_matches

def find_homography(kp1, kp2, matches):
    if kp1 is None or kp2 is None or matches is None or len(matches) < 4: 
        return None
    
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H , _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    return H 

def draw_matches(img1, kp1, img2, kp2, matches):
    if matches is None:
        return 
    return cv2.drawMatches(img1, kp1, img2, kp2, matches, None)
