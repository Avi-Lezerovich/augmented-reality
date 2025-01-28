import cv2
import os

class VideoHandler:
    def __init__(self, input_path, save_video=False):
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
        if save_video:
            output_path = self._create_output_path(input_path)
            
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

            
    def _create_output_path(self, input_path):
        """Create output path in same directory as input with '_out' suffix"""
        directory = os.path.dirname(input_path)
        filename = os.path.basename(input_path)
        name, ext = os.path.splitext(filename)
        
        # Create output filename
        out_filename = f"{name}_result{ext}"
        
        return os.path.join(directory, out_filename)
    
    
    def read_frame(self):
        """Read next frame, returns (success, frame)"""
        ret, frame = self.cap.read()
        if ret:
            self.frame_count += 1
        return ret, frame , self.frame_count

    def write_frame(self, frame):
        """Write frame if writer exists"""
        if self.writer:
            self.writer.write(frame)

    def release(self):
        """Release video capture and writer"""
        self.cap.release()
        if self.writer:
            self.writer.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()