import cv2
import numpy as np

sift = cv2.SIFT_create()

def find_keypoints(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return sift.detectAndCompute(gray, None)
    

def find_matches(des1, des2):
    if des1 is None or des2 is None:
        return None
    
     # FLANN parameters and matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance: # Apply ratio test
            good_matches.append(m)

    return good_matches

def find_homography(kp1, kp2, matches):
    if kp1 is None or kp2 is None or len(matches) < 4: 
        return None, None
    
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    return cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)


