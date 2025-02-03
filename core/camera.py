from glob import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np


class CameraCalibrator:
    def __init__(self, pattern_size=(9, 6), square_size=2.4):
        self.pattern_size = pattern_size
        self.square_size = square_size
        self.camera_matrix = None
        self.dist_coefs = None
        self.rms = None
        self.rvecs = None
        self.tvecs = None
        self.mean_error = None

    def calibrate_camera(self, images_dir, image_format, display=False):
        """Enhanced calibration with validation checks"""
        img_mask = f"{images_dir}/*.{image_format}"
        img_names = glob(img_mask)

        pattern_points = self._create_pattern_points()
        obj_points, img_points = [], []

        # Process each image with progress tracking
        successful_images = 0
        for fname in img_names:
            success, corners = self._process_image(fname)
            if not success:
                continue

            obj_points.append(pattern_points)
            img_points.append(corners)
            successful_images += 1

            if display:
                self.visualize_calibration(fname, corners)
                cv2.waitKey(0)

        if display:
            cv2.destroyAllWindows()
            
        h, w = cv2.imread(img_names[0]).shape[:2]
        self.rms, self.camera_matrix, self.dist_coefs, self.rvecs, self.tvecs = \
            cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)

        # Calculate reprojection error
        self.mean_error = self._calculate_reprojection_error(
            obj_points, img_points)

        return {
            'rms': self.rms,
            'camera_matrix': self.camera_matrix,
            'dist_coefs': self.dist_coefs,
            'reprojection_error': self.mean_error,
            'successful_images': successful_images
        }

    def _calculate_reprojection_error(self, obj_points, img_points):
        """Calculate detailed reprojection error statistics"""
        errors = []
        for i in range(len(obj_points)):
            imgpoints2, _ = cv2.projectPoints(
                obj_points[i], self.rvecs[i], self.tvecs[i],
                self.camera_matrix, self.dist_coefs
            )
            error = cv2.norm(img_points[i], imgpoints2,
                             cv2.NORM_L2)/len(imgpoints2)
            errors.append(error)

        return {
            'mean': np.mean(errors),
            'std': np.std(errors),
            'max': np.max(errors)

        }

    def _create_pattern_points(self):
        pattern_points = np.zeros((np.prod(self.pattern_size), 3), np.float32)
        pattern_points[:, :2] = np.indices(self.pattern_size).T.reshape(-1, 2)
        pattern_points *= self.square_size
        return pattern_points

    def _process_image(self, fname):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, self.pattern_size)
        if not found:
            print(f"Chessboard not found in {fname}")
            return False, None
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                   (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        return True, corners

    def save_coefficients(self, path):
        """Save the camera matrix and the distortion coefficients to given path/file."""
        cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
        cv_file.write("rms", self.rms)
        cv_file.write("camera_matrix", self.camera_matrix)
        cv_file.write("dist_coefs", self.dist_coefs)
        cv_file.write("rvecs", np.array(self.rvecs))
        cv_file.write("tvecs", np.array(self.tvecs))
        
        cv_file.write("reprojection_error_mean", self.mean_error['mean'])
        cv_file.write("reprojection_error_std", self.mean_error['std'])
        cv_file.write("reprojection_error_max", self.mean_error['max'])
        cv_file.release()
        

    def load_coefficients(self, path):
        """Loads camera matrix and distortion coefficients."""
        cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
        self.rms = cv_file.getNode("rms").real()
        self.camera_matrix = cv_file.getNode("camera_matrix").mat()
        self.dist_coefs = cv_file.getNode("dist_coefs").mat()
        self.rvecs = cv_file.getNode("rvecs").mat()
        self.tvecs = cv_file.getNode("tvecs").mat()
        
        self.mean_error = {
            'mean': cv_file.getNode("reprojection_error_mean").real(),
            'std': cv_file.getNode("reprojection_error_std").real(),
            'max': cv_file.getNode("reprojection_error_max").real()
        }
        
        cv_file.release()

    def visualize_calibration(self, image_path, corners):
        """Visualize undistorted images"""
        img = cv2.imread(image_path)
        vis_img = img.copy()

        # Draw the corners
        cv2.drawChessboardCorners(vis_img, self.pattern_size, corners, True)
        cv2.imshow('Visualization', vis_img)

    def print_coefficients(self):
        print(f"RMS:\n {self.rms}")
        print(f"Camera matrix:\n {self.camera_matrix}\n")
        print(f"Distortion coefficients:\n {self.dist_coefs.ravel()}")

