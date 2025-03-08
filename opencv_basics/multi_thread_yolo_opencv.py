import cv2 as cv
import time
import threading
from ultralytics import YOLO
import numpy as np
import math

class FrameReader:
    """Threaded frame reader for multiple sources."""
    def __init__(self, source, cfr, skip):
        self.cap = cv.VideoCapture(source)
        if not self.cap.isOpened():
            raise Exception(f"Error: Could not open source {source}")
        
        self.ret, self.frame = self.cap.read()
        self.cfr = cfr
        self.skip = skip
        self.cam_fps = self.cap.get(cv.CAP_PROP_FPS)
        self.lock = threading.Lock()
        self.running = True
        self.frameNo = 0

        # Start the thread
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        """Continuously reads frames in a separate thread."""
        while self.running:
            ret, frame = self.cap.read()
            self.frameNo += 1
            if not ret:
                self.running = False
                break

            if self.frameNo % int(self.cam_fps / self.cfr) != 0 and self.skip:
                continue

            with self.lock:
                self.ret, self.frame = ret, frame

    def get_frame(self):
        """Returns the latest frame."""
        with self.lock:
            return self.ret, self.frame

    def stop(self):
        """Stops the thread and releases the camera."""
        self.running = False
        self.thread.join()
        self.cap.release()

class YOLOv11MultiSource:
    """Handles multiple video sources with YOLOv11 inference."""
    def __init__(self, sources, model_name="yolo11n.pt", conf_threshold=0.5, cfr=5, skip=False, show=True, do_detection=True, model_plot=False):
        self.sources = sources  # List of camera indices or video files
        self.conf_threshold = conf_threshold
        self.show = show
        self.cfr = cfr
        self.skip = skip
        self.do_detection = do_detection
        self.model_plot = model_plot

        # Load YOLO model
        self.model = YOLO(model_name)

        # Create a separate reader for each source
        self.readers = {src: FrameReader(src, self.cfr, self.skip) for src in self.sources}

        # FPS tracking
        self.fps = {src: 0 for src in self.sources}
        self.frameNo = {src: 0 for src in self.sources}
        self.prev_time = {src: time.time() for src in self.sources}
        self.start_time = {src: self.prev_time[src] for src in self.sources}

        # Calculate dynamic grid size
        self.grid_size = self.calculate_grid_size(len(sources))
    
    def calculate_grid_size(self, num_sources):
        """Dynamically determines the best rows × cols grid for the number of sources."""
        rows = math.ceil(math.sqrt(num_sources))
        cols = math.ceil(num_sources / rows)
        return rows, cols

    def process_frame(self, frame):
        """Runs YOLO inference and draws bounding boxes."""
        results = self.model(frame, verbose=False, conf=self.conf_threshold)
        
        if self.model_plot:
            return results[0].plot()
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                conf = box.conf[0].item()  # Confidence score
                cls = int(box.cls[0].item())  # Class index
                label = f"{result.names[cls]}: {conf:.2f}"  # Class name and confidence

                # Draw bounding box
                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw label text
                cv.putText(frame, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame
    

    def create_grid(self, frames):
        """Arranges multiple frames into a grid."""
        rows, cols = self.grid_size
        blank_frame = np.zeros_like(next(iter(frames.values())))  # Black frame if missing
        
        # Ensure we have enough frames
        all_frames = list(frames.values())
        while len(all_frames) < rows * cols:
            all_frames.append(blank_frame)

        # Resize all frames to match the first frame's size
        h, w, _ = all_frames[0].shape
        all_frames = [cv.resize(f, (w, h)) for f in all_frames]

        # Create the grid
        grid_frames = [np.hstack(all_frames[i * cols:(i + 1) * cols]) for i in range(rows)]
        return np.vstack(grid_frames)

    def run(self):
        """Main loop to process frames from multiple sources."""
        while any(reader.running for reader in self.readers.values()):
            frames = {}
            for src, reader in self.readers.items():
                ret, frame = reader.get_frame()
                if not ret:
                    continue

                self.frameNo[src] += 1

                if self.do_detection:
                    frame = self.process_frame(frame)
                
                

                # Calculate FPS
                curr_time = time.time()
                elapsed_time = curr_time - self.start_time[src]
                avg_fps = self.frameNo[src] / elapsed_time if elapsed_time > 0 else 0

                self.fps[src] = 1 / (curr_time - self.prev_time[src])
                self.prev_time[src] = curr_time

                cv.putText(frame, f"FPS: {int(self.fps[src])}, Avg: {avg_fps:.2f}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                frames[src] = frame


            if self.show:
                grid_image = self.create_grid(frames)
                cv.imshow("Multi-Camera View", grid_image)


            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        # Stop all readers and close windows
        for reader in self.readers.values():
            reader.stop()
        cv.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    detector = YOLOv11MultiSource(model_name= "yolo11n.pt", sources = [0], 
                                  conf_threshold=0.5, cfr=2, skip=False, show=True, do_detection=False, model_plot=False)
    detector.run()