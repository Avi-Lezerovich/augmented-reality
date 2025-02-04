import cv2
import os


class VideoHandler:
    def __init__(self, input_path, output_name=None):
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Video file not found: {input_path}")

        self.cap = cv2.VideoCapture(input_path)
        if not self.cap.isOpened():
            raise IOError(f"Failed to open video: {input_path}")

        # Get input video properties
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_rate = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_count = 0

        self.writer = None
        if output_name:
            output_path = self._create_output_path(input_path, output_name)

            # Use XVID codec for Windows compatibility
            fourcc = cv2.VideoWriter_fourcc(*'XVID')

            self.writer = cv2.VideoWriter(
                output_path,
                fourcc,
                self.frame_rate,
                (self.frame_width, self.frame_height),
                isColor=True
            )

            if not self.writer.isOpened():
                raise IOError("Failed to initialize video writer")

    def _create_output_path(self, input_path, output_name):
        """Create output path in same directory as input with '_out' suffix"""
        directory = os.path.dirname(input_path)
        filename = os.path.basename(input_path)
        name, ext = os.path.splitext(filename)

        # Create output filename
        out_filename = f"{output_name}_result{ext}"

        return os.path.join(directory, out_filename)

    def read_frame(self):
        """Read next frame, returns (success, frame)"""
        ret, frame = self.cap.read()
        if ret:
            self.frame_count += 1
        return ret, frame, self.frame_count

    def write_frame(self, frame):
        """Write frame if writer exists"""
        if self.writer:
            self.writer.write(frame)

    def release(self):
        """Release video capture and writer"""
        self.cap.release()
        cv2.destroyAllWindows()
        if self.writer:
            self.writer.release()
        

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


def video_to_frames(input_path, output_dir):
    """Extract frames from video and save to output directory"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with VideoHandler(input_path) as video:
        while True:
            ret, frame, frame_count = video.read_frame()
            if not ret:
                break

            frame_filename = f"{frame_count:04d}.jpg"
            output_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(output_path, frame)
        video.release()
    print(f"Extracted {frame_count} frames to {output_dir}")

    return video.frame_count






def merge_images_side_by_side(img1, img2):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    common_height = min(h1, h2)
    new_w1 = int(w1 * (common_height / h1))
    new_w2 = int(w2 * (common_height / h2))
    
    
    # Resize images to common height and new widths

    # Resize images to common height and new widths
    img1_resized = cv2.resize(img1, (new_w1, common_height))
    img2_resized = cv2.resize(img2, (new_w2, common_height))

    return cv2.hconcat([img1_resized, img2_resized])

def merge_images_top_to_bottom(img1, img2):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    common_width = min(w1, w2)
    new_h1 = int(h1 * (common_width / w1))
    new_h2 = int(h2 * (common_width / w2))

    img1_resized = cv2.resize(img1, (common_width, new_h1))
    img2_resized = cv2.resize(img2, (common_width, new_h2))

    return cv2.vconcat([img1_resized, img2_resized])



