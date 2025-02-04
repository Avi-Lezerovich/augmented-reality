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
        Loads, rotates, scales, and positions the 3D model, and applies the textures,
        ensuring it sits properly with z <= 0 (similar to your cube).
        """
        # 1) Load the model with materials and textures
        self.mesh = o3d.io.read_triangle_mesh(model_path, enable_post_processing=True)
        self.mesh.compute_vertex_normals()

        # 2) (Optional) Rotate model 90 degrees around the X-axis if needed
        #    Check sign of that last row to get the orientation you desire.
        rotation_matrix = np.array([
            [1,  0,  0],
            [0,  0,  1],
            [0, -1,  0]
        ], dtype=np.float64)

        bbox = self.mesh.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        self.mesh.rotate(rotation_matrix, center=center)

        # 3) Scale model so its largest dimension matches your chosen base_side
        base_side = min(self.w, self.h) * scale_factor
        bbox = self.mesh.get_axis_aligned_bounding_box()
        max_dim = max(bbox.get_extent())
        if max_dim > 0:
            self.mesh.scale(base_side / max_dim, center=bbox.get_center())

        # 4) First translate so that the bounding-box min corner is at (0,0,0)
        new_bbox = self.mesh.get_axis_aligned_bounding_box()
        new_min = new_bbox.get_min_bound()
        self.mesh.translate(-new_min)

        # 5) Then shift so that the bounding-box max z is at z=0;
        #    that means the entire model is behind the plane (z <= 0).
        shifted_bbox = self.mesh.get_axis_aligned_bounding_box()
        zmax = shifted_bbox.get_max_bound()[2]
        self.mesh.translate([0, 0, -zmax])

        # 6) Check textures
        if self.mesh.has_textures():
            print("Texture loaded successfully.")
        else:
            print("No texture found. Ensure the .mtl file and texture path are correct.")

        print("Mesh stats:")
        print("  Has normals?  ", self.mesh.has_vertex_normals())
        print("  Has colors?   ", self.mesh.has_vertex_colors())
        print("  Has textures? ", self.mesh.has_textures())

        
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

    
    def render_model_textured(self, frame, H):
        if H is None or self.mesh is None or not self.mesh.has_textures():
            return frame

        # Step 1: Solve PnP and get rvec, tvec
        template_corners_2d = np.float32([
            [0, 0], [self.w, 0], [self.w, self.h], [0, self.h]
        ]).reshape(-1, 1, 2)
        warped_corners = cv2.perspectiveTransform(template_corners_2d, H)
        rvec, tvec = self.solve_pnp(warped_corners)
        if rvec is None or tvec is None:
            return frame

        # Step 2: Suppose we have exactly 1 texture image in self.mesh.textures[0]
        # Convert that open3d Image to a NumPy array:
        if len(self.mesh.textures) == 0:
            return frame
        texture_o3d = self.mesh.textures[0]
        texture = np.asarray(texture_o3d)  # shape = (H, W, 3) or (H, W)
        
        # Step 3: For each triangle, get 3D vertices and 2D UVs
        triangles = np.asarray(self.mesh.triangles)
        triangle_uvs = np.asarray(self.mesh.triangle_uvs)  # 3 uvs per triangle vertex

        vertices_3d = np.asarray(self.mesh.vertices)

        for i, tri in enumerate(triangles):
            # tri is something like [v0, v1, v2]
            v0, v1, v2 = tri
            pts_3d = np.float32([
                vertices_3d[v0],
                vertices_3d[v1],
                vertices_3d[v2],
            ]).reshape(-1, 1, 3)

            # Project them
            pts_2d, _ = cv2.projectPoints(pts_3d, rvec, tvec, self.K, self.dist)
            pts_2d = pts_2d.reshape(-1, 2).astype(np.float32)

            # Get the corresponding UV coords for these vertices
            uv0, uv1, uv2 = triangle_uvs[3*i : 3*i + 3]  # each is (u, v) in [0..1]
            
            # Convert UV coords to pixel coords in the texture image
            Ht, Wt = texture.shape[:2]
            tex_tri = np.float32([
                [uv0[0] * Wt, uv0[1] * Ht],
                [uv1[0] * Wt, uv1[1] * Ht],
                [uv2[0] * Wt, uv2[1] * Ht],
            ])

            # Step 4: Compute a warp that maps the triangle in the texture to triangle in the frame
            M = cv2.getAffineTransform(tex_tri[:3], pts_2d[:3])

            # Step 5: Warp that triangular region from texture onto the frame
            # We can use warpAffine on a bounding rectangle or do piecewise warping
            # For a quick hack, do an affine warp of the bounding box, then mask with the triangle.
            bbox_x, bbox_y, bbox_w, bbox_h = cv2.boundingRect(pts_2d)
            if bbox_w > 0 and bbox_h > 0:
                # We'll warp the relevant region from the texture into a temporary patch
                patch = cv2.warpAffine(
                    texture, M, (frame.shape[1], frame.shape[0]),
                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
                )
                # Create a mask of the triangle in the same coordinate space
                mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                cv2.fillConvexPoly(mask, pts_2d.astype(int), 255)
                
                # Blend or copy the patch into the frame using the mask
                mask_bool = (mask == 255)
                frame[mask_bool] = patch[mask_bool]
        
        return frame