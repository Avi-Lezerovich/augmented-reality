from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np

display = False



    def calibrate_camera(images_dir, image_format):
        """
        Calibrate the camera using chessboard images.

        Args:
            images_dir (str): Directory containing chessboard images.
            image_format (str): Format of the images (e.g., 'jpeg').

        Returns:
            ret (float): The overall RMS re-projection error.
            mtx (ndarray): Camera matrix.
            dist (ndarray): Distortion coefficients.
            rvecs (list): Rotation vectors.
            tvecs (list): Translation vectors.
        """
        img_mask = images_dir + '\\*.' + image_format
        pattern_size = (9, 6)
        square_size = 3.1
        img_names = glob(img_mask)
        
        
        pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
        pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
        pattern_points *= square_size
        
        obj_points = []
        img_points = []
        h, w = cv2.imread(img_names[0]).shape[:2]
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        for fname in img_names:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
            # Find the chess board corners
            found, corners = cv2.findChessboardCorners(gray, pattern_size)
        
            # If found, add object points, image points (after refining them)
            if not found:
                print("chessboard not found")
                continue
            obj_points.append(pattern_points)
            corners = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            img_points.append(corners)
        
            # Draw and display the corners
            if display:
                cv2.drawChessboardCorners(img, pattern_size, corners, found)
                cv2.imshow('img', img)
                cv2.waitKey(0)
        
        cv2.destroyAllWindows()

    
        rms, camera_matrix, dist_coefs, rvecs, tvecs  = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)
        
        mean_error = 0
        for i in range(len(obj_points)):
            imgpoints2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coefs)
            error = cv2.norm(img_points[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            mean_error += error
            
            
        print("#####################################################")
        print( "total error: {}".format(mean_error/len(obj_points)) )
        print("#####################################################")
        
        return [rms, camera_matrix, dist_coefs, rvecs, tvecs ]


    
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


    def print_calibrate_data(rms, camera_matrix, dist_coefs, rvecs, tvecs):
        print(f""" RMS: \n {rms} \n 
            Camera Matrix: \n {camera_matrix} \n 
            Distortion Coefficients: \n {dist_coefs.ravel()} \n """
            )

    if __name__ == "__main__":
        images_path = r"C:\Users\Avi Lezerovich\Documents\GitHub\augmented-reality\data\images"
        
        # calibrate camera
        rms, camera_matrix, dist_coefs, rvecs, tvecs  = calibrate_camera(images_path, "jpg")
        # save_coefficients(ret, mtx, dist, rvecs, tvecs, "data\calibration_data.yaml")
        print_calibrate_data(rms, camera_matrix, dist_coefs, rvecs, tvecs)
        
        