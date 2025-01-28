import cv2
import numpy as np

class CubeRenderer:
    def __init__(self, camera_matrix, dist_coeffs, template_size, cube_scale=0.5):
        """
        Args:
            camera_matrix: Intrinsic camera matrix.
            dist_coeffs: Distortion coefficients.
            template_size: (w, h) in real-world units or a known reference.
            cube_scale: Multiplier to scale the cube relative to template_size.
        """
        self.K = camera_matrix
        self.dist = dist_coeffs

        # Store marker width, height (real-world units, e.g. cm)
        self.w, self.h = template_size

        # Create 3D points for the marker corners (Z=0 plane), scaled by template size
        self.template_corners_3d = np.float32([
            [0,     0,      0],
            [self.w,0,      0],
            [self.w,self.h, 0],
            [0,     self.h, 0]
        ])

        # Use the smaller dimension as the cube's base side length (or pick whichever you need)
        base_side = min(self.w, self.h) * cube_scale

        # Create 3D points for the cube
        self.cube_3d = np.float32([
            # Base (Z=0)
            [0,         0,         0],
            [base_side, 0,         0],
            [base_side, base_side, 0],
            [0,         base_side, 0],
            # Top (Z = -base_side)
            [0,         0,         -base_side],
            [base_side, 0,         -base_side],
            [base_side, base_side, -base_side],
            [0,         base_side, -base_side]
        ])

    def solve_pnp(self, warped_corners):
        """
        Solve Perspective-n-Point for cube rendering.
        """
        warped_corners = np.array(warped_corners, dtype=np.float32).reshape(4, 1, 2)
        success, rvec, tvec = cv2.solvePnP(
            self.template_corners_3d,
            warped_corners,
            self.K,
            self.dist,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        return (rvec, tvec) if success else (None, None)

    def _draw_cube(self, img, imgpts):
        imgpts = np.int32(imgpts).reshape(-1, 2)
        # Base
        img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -3)
        # Pillars
        for i, j in zip(range(4), range(4, 8)):
            img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255, 0, 0), 3)
        # Top
        img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)
        return img

    def render_cube(self, frame, H):
        """
        Render a 3D cube on the input image.
        """
        if H is None:
            return frame

        # Template corners in pixel space
        template_corners_2d = np.float32([
            [0, 0],
            [self.w, 0],
            [self.w, self.h],
            [0, self.h]
        ]).reshape(-1, 1, 2)

        # Perspective transform
        warped_corners = cv2.perspectiveTransform(template_corners_2d, H)

        # Solve PnP and project
        rvec, tvec = self.solve_pnp(warped_corners)
        if rvec is not None and tvec is not None:
            imgpts, _ = cv2.projectPoints(self.cube_3d, rvec, tvec, self.K, self.dist)
            frame = self._draw_cube(frame, imgpts)

        return frame