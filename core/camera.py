from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np


def calibrate_camera(image_dir ,image_format):
    img_mask =  images_path + "\*.jpeg"
    pattern_size = (9, 6)
    img_names = glob(img_mask)
    num_images = len(img_names)

    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
  

    obj_points = []
    img_points = []
    
    h, w = cv2.imread(img_names[0]).shape[:2]
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    images = glob(image_dir + "/*." + image_format)
    
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, pattern_size)
    
        # If found, add object points, image points (after refining them)
        if ret == True:
            obj_points.append(pattern_points)
    
            corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            img_points.append(corners2)
    
            # Draw and display the corners
            cv2.drawChessboardCorners(img, pattern_size, corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(0)
    
    cv2.destroyAllWindows()


    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)

    return [ret, mtx, dist, rvecs, tvecs]


   
def save_coefficients(ret, camera_matrix, dist_coefs, rvecs, tvecs, path):
    """
    Save the camera matrix and the distortion coefficients to given path/file.
    """
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    cv_file.write("rms", ret)
    cv_file.write("camera_matrix", camera_matrix)
    cv_file.write("dist_coefs", dist_coefs)
    cv_file.write("rvecs", np.array(rvecs))
    cv_file.write("tvecs", np.array(tvecs))
    cv_file.release()
    
    
def load_coefficients(path):
    """ Loads camera matrix and distortion coefficients. """
    
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    rms = cv_file.getNode("rms").real()
    camera_matrix = cv_file.getNode("camera_matrix").mat()
    dist_coefs = cv_file.getNode("dist_coefs").mat()
    rvecs = cv_file.getNode("rvecs").mat()
    tvecs = cv_file.getNode("tvecs").mat()
    
    cv_file.release()
    
    return [rms, camera_matrix, dist_coefs, rvecs, tvecs]



if __name__ == "__main__":
    images_path = r"C:\Users\Avi Lezerovich\Documents\GitHub\augmented-reality\data\images"
    
    # calibrate camera
    ret, mtx, dist, rvecs, tvecs = calibrate_camera(images_path,  "jpeg")
    save_coefficients(ret, mtx, dist, rvecs, tvecs, "calibration_data.yaml")
    print("Calibration complete")
    print("ret:", ret)
    print("mtx:", mtx)
    
    