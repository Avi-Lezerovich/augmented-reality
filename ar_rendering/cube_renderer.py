import open3d as o3d
import cv2
import numpy as np

class CubeRenderer:
    def __init__(self, camera_matrix, dist_coeffs, template_size,
                 cube_scale=0.5, model_path=None, scale_factor=0.5):
        """
        Args:
            camera_matrix: Intrinsic camera matrix.
            dist_coeffs: Distortion coefficients.
            template_size: (w, h) in real-world units or a known reference.
            cube_scale: Multiplier to scale the cube relative to template_size.
            model_path: Optional path to a 3D model to render.
            scale_factor: Scale factor for the loaded 3D model.
        """
        self.K = camera_matrix
        self.dist = dist_coeffs
        self.w, self.h = map(int, template_size)
        self.prev_H = None

        # Create 3D points for the marker corners (Z=0 plane)
        self.template_corners_3d = np.float32([
            [0,     0,     0],
            [self.w, 0,     0],
            [self.w, self.h, 0],
            [0,     self.h, 0]
        ])

        # Prepare cube geometry if no model is supplied
        base_side = min(self.w, self.h) * cube_scale
        self.cube_3d = np.float32([
            # Base (Z=0)
            [0,         0,         0],
            [base_side, 0,         0],
            [base_side, base_side, 0],
            [0,         base_side, 0],
            # Top (Z=-base_side)
            [0,         0,         -base_side],
            [base_side, 0,         -base_side],
            [base_side, base_side, -base_side],
            [0,         base_side, -base_side]
        ])

        # Load and scale 3D model if provided
        self.mesh = None
        if model_path is not None:
            self.load_and_scale_model(model_path, scale_factor)

    def load_and_scale_model(self, model_path, scale_factor):
        """
        Loads, rotates, scales, and positions the 3D model, then colors it red.
        """
        self.mesh = o3d.io.read_triangle_mesh(model_path)
        self.mesh.compute_vertex_normals()

        # Rotate model 90 degrees around X-axis
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0]
        ], dtype=np.float64)
        bbox = self.mesh.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        self.mesh.rotate(rotation_matrix, center=center)

        # Scale to fit cube
        base_side = min(self.w, self.h) * scale_factor
        bbox = self.mesh.get_axis_aligned_bounding_box()
        max_dim = max(bbox.get_extent())
        if max_dim > 0:
            self.mesh.scale(base_side / max_dim, center=bbox.get_center())

        # Position at origin
        new_bbox = self.mesh.get_axis_aligned_bounding_box()
        new_min = new_bbox.get_min_bound()
        self.mesh.translate(-new_min)
        # Paint the model red
        self.mesh.paint_uniform_color([1, 0, 0])
        
    def solve_pnp(self, warped_corners):
        warped_corners = np.float32(warped_corners).reshape(-1, 1, 2)
        success, rvec, tvec = cv2.solvePnP(
            self.template_corners_3d, warped_corners, self.K, self.dist,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        return (rvec, tvec) if success else (None, None)

    def render_cube(self, frame, H):
        if H is None:
            return frame

        template_corners_2d = np.float32([
            [0, 0], [self.w, 0], [self.w, self.h], [0, self.h]
        ]).reshape(-1, 1, 2)
        warped_corners = cv2.perspectiveTransform(template_corners_2d, H)
        rvec, tvec = self.solve_pnp(warped_corners)
        if rvec is None or tvec is None:
            return frame

        cube_pts = self.cube_3d.reshape(-1, 1, 3)
        proj_pts, _ = cv2.projectPoints(cube_pts, rvec, tvec, self.K, self.dist)
        proj_pts = proj_pts.reshape(-1, 2).astype(int)
        for i, j in [(0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4), (0,4), (1,5), (2,6), (3,7)]:
            cv2.line(frame, tuple(proj_pts[i]), tuple(proj_pts[j]), (0,255,0), 2)
        return frame

    
    def render_model(self, frame, H):
        if H is None or self.mesh is None:
            return frame

        template_corners_2d = np.float32([
            [0, 0], [self.w, 0], [self.w, self.h], [0, self.h]
        ]).reshape(-1, 1, 2)
        warped_corners = cv2.perspectiveTransform(template_corners_2d, H)
        rvec, tvec = self.solve_pnp(warped_corners)
        if rvec is None or tvec is None:
            return frame

        vertices = np.asarray(self.mesh.vertices)
        verts_reshaped = vertices.reshape(-1, 1, 3).astype(np.float32)
        projected, _ = cv2.projectPoints(verts_reshaped, rvec, tvec, self.K, self.dist)
        
        # Draw projected vertices in red (BGR: (0, 0, 255))
        for pt in projected.reshape(-1, 2):
            cv2.circle(frame, tuple(pt.astype(int)), 2, (0, 0, 255), -1)
        return frame