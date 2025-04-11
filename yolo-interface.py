import cv2
import numpy as np
import torch
import os
import time
from PIL import Image
import torch.nn as nn
from torchvision import transforms, models
import argparse
import threading
from collections import deque

# Constants
DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 600
PROCESSING_WIDTH = 640
PROCESSING_HEIGHT = 480
CONFIDENCE_THRESHOLD = 0.5
KEEP_FRAMES = 5  # For temporal smoothing

class SignClassifier(nn.Module):
    def __init__(self, num_classes):
        super(SignClassifier, self).__init__()
        # Use MobileNetV2 for better performance on the tablet
        self.model = models.mobilenet_v2(pretrained=True)
        self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)
        
    def forward(self, x):
        return self.model(x)

class TrafficSignSystem:
    def __init__(self, args):
        self.args = args
        self.source = args.source
        self.running = False
        self.frame = None
        self.processed_frame = None
        self.fps = 0
        self.detections = []
        self.frame_count = 0
        self.start_time = time.time()
        self.sign_history = deque(maxlen=KEEP_FRAMES)  # For temporal smoothing
        
        # Load detection model (YOLOv8 for sign detection)
        try:
            from ultralytics import YOLO
            self.detection_model = YOLO(args.detection_model)
            print(f"Loaded detection model: {args.detection_model}")
        except Exception as e:
            print(f"Error loading detection model: {e}")
            exit(1)
            
        # Load classification model
        self.load_classifier()
        
        # Setup video capture
        self.setup_video_capture()
        
    def load_classifier(self):
        try:
            # Get class names from folders or file
            if os.path.exists(self.args.classes_dir):
                self.class_names = sorted([d for d in os.listdir(self.args.classes_dir) 
                                      if os.path.isdir(os.path.join(self.args.classes_dir, d))])
            else:
                # Fallback to a predefined list of Polish sign classes
                self.class_names = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 
                                'A10', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'C1', 
                                'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'D1', 'D2', 'D3', 
                                'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'E1', 
                                'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10']
            
            num_classes = len(self.class_names)
            print(f"Loaded {num_classes} sign classes")
            
            # Initialize model
            self.classifier = SignClassifier(num_classes)
            self.classifier.load_state_dict(torch.load(self.args.classification_model, 
                                                       map_location=torch.device('cpu')))
            self.classifier.eval()
            
            # Preprocessing transforms
            self.preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
        except Exception as e:
            print(f"Error loading classification model: {e}")
            exit(1)
    
    def setup_video_capture(self):
        try:
            if self.source.startswith('rtsp://'):
                # For RTSP stream from Raspberry Pi
                self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # Reduce latency
            elif self.source.isdigit():
                # For webcam
                self.cap = cv2.VideoCapture(int(self.source))
            else:
                # For video file
                self.cap = cv2.VideoCapture(self.source)
                
            # Check if camera/video opened successfully
            if not self.cap.isOpened():
                print(f"Error: Unable to open video source {self.source}")
                exit(1)
                
            # Set resolution for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, PROCESSING_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PROCESSING_HEIGHT)
            
            print(f"Video source initialized: {self.source}")
            
        except Exception as e:
            print(f"Error setting up video capture: {e}")
            exit(1)
    
    def detect_signs(self, frame):
        try:
            # Run YOLOv8 detection
            results = self.detection_model(frame, verbose=False)
            
            detections = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    
                    if conf < CONFIDENCE_THRESHOLD:
                        continue
                    
                    # Right-side filtering (only consider signs on the right side of frame)
                    center_x = (x1 + x2) / 2
                    if center_x > frame.shape[1] / 2:
                        # Extract the sign region for classification
                        sign_img = frame[int(y1):int(y2), int(x1):int(x2)]
                        
                        if sign_img.size == 0:  # Skip empty detections
                            continue
                            
                        # Convert to PIL for classification preprocessing
                        sign_img_pil = Image.fromarray(cv2.cvtColor(sign_img, cv2.COLOR_BGR2RGB))
                        input_tensor = self.preprocess(sign_img_pil).unsqueeze(0)
                        
                        # Classify the sign
                        with torch.no_grad():
                            output = self.classifier(input_tensor)
                            _, predicted_idx = torch.max(output, 1)
                            sign_class = predicted_idx.item()
                            sign_name = self.class_names[sign_class]
                        
                        detections.append({
                            'class': sign_class,
                            'name': sign_name,
                            'confidence': conf,
                            'bbox': (x1, y1, x2, y2)
                        })
            
            # Update sign history for temporal smoothing
            self.sign_history.append(detections)
            
            # Perform temporal smoothing
            smoothed_detections = self.temporal_smoothing()
            return smoothed_detections
            
        except Exception as e:
            print(f"Error in sign detection: {e}")
            return []
    
    def temporal_smoothing(self):
        """Smooth detections over multiple frames to reduce flickering"""
        if not self.sign_history:
            return []
            
        # Count occurrences of each sign class
        sign_counts = {}
        for frame_detections in self.sign_history:
            for det in frame_detections:
                sign_name = det['name']
                if sign_name not in sign_counts:
                    sign_counts[sign_name] = {
                        'count': 1,
                        'confidence': det['confidence'],
                        'bbox': det['bbox'],
                        'class': det['class']
                    }
                else:
                    sign_counts[sign_name]['count'] += 1
                    # Update with highest confidence detection
                    if det['confidence'] > sign_counts[sign_name]['confidence']:
                        sign_counts[sign_name]['confidence'] = det['confidence']
                        sign_counts[sign_name]['bbox'] = det['bbox']
        
        # Only keep signs that appear in multiple frames
        min_count = max(1, len(self.sign_history) // 3)
        smoothed = []
        for sign_name, data in sign_counts.items():
            if data['count'] >= min_count:
                smoothed.append({
                    'name': sign_name,
                    'confidence': data['confidence'],
                    'bbox': data['bbox'],
                    'class': data['class']
                })
                
        return smoothed
    
    def detect_lanes(self, frame):
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply Canny edge detection
            edges = cv2.Canny(blur, 50, 150)
            
            # Define region of interest (bottom half of the image)
            height, width = edges.shape
            mask = np.zeros_like(edges)
            # Define a polygon for the ROI - focus on the road area
            polygon = np.array([
                [(0, height), (width, height), (width//2, height//2), (width//2, height//2), (0, height)]
            ], np.int32)
            cv2.fillPoly(mask, polygon, 255)
            masked_edges = cv2.bitwise_and(edges, mask)
            
            # Apply Hough transform to detect lines
            lines = cv2.HoughLinesP(masked_edges, 2, np.pi/180, 50, 
                                   minLineLength=40, maxLineGap=5)
            
            # Create blend image for lane lines
            line_image = np.zeros_like(frame)
            
            # Variables to store lane line parameters
            left_line_x = []
            left_line_y = []
            right_line_x = []
            right_line_y = []
            
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    
                    # Calculate slope to separate left and right lanes
                    if x2 - x1 == 0:  # Avoid division by zero
                        continue
                        
                    slope = (y2 - y1) / (x2 - x1)
                    
                    # Filter out horizontal lines
                    if abs(slope) < 0.5:
                        continue
                        
                    # Separate left and right lanes based on slope
                    if slope < 0:  # Left lane
                        left_line_x.extend([x1, x2])
                        left_line_y.extend([y1, y2])
                    else:  # Right lane
                        right_line_x.extend([x1, x2])
                        right_line_y.extend([y1, y2])
                
                # Draw the lanes if points exist
                if left_line_x and left_line_y:
                    # Fit line to points
                    left_poly = np.polyfit(left_line_y, left_line_x, deg=1)
                    
                    # Generate points along the fitted line
                    y_start = int(height * 0.6)  # Start from 60% down the image
                    y_end = height
                    x_start = int(left_poly[0] * y_start + left_poly[1])
                    x_end = int(left_poly[0] * y_end + left_poly[1])
                    
                    # Draw line
                    cv2.line(line_image, (x_start, y_start), (x_end, y_end), (0, 0, 255), 10)
                
                if right_line_x and right_line_y:
                    # Fit line to points
                    right_poly = np.polyfit(right_line_y, right_line_x, deg=1)
                    
                    # Generate points along the fitted line
                    y_start = int(height * 0.6)
                    y_end = height
                    x_start = int(right_poly[0] * y_start + right_poly[1])
                    x_end = int(right_poly[0] * y_end + right_poly[1])
                    
                    # Draw line
                    cv2.line(line_image, (x_start, y_start), (x_end, y_end), (0, 0, 255), 10)
            
            # Blend lane lines with original image
            result = cv2.addWeighted(frame, 0.8, line_image, 1, 0)
            return result
            
        except Exception as e:
            print(f"Error in lane detection: {e}")
            return frame
    
    def draw_ui(self, frame, detections):
        """Draw the user interface with detections and information"""
        try:
            # Resize frame for display
            display_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            
            # Draw detections
            for det in detections:
                x1, y1, x2, y2 = [int(coord * DISPLAY_WIDTH / PROCESSING_WIDTH) 
                                 if i % 2 == 0 else int(coord * DISPLAY_HEIGHT / PROCESSING_HEIGHT) 
                                 for i, coord in enumerate(det['bbox'])]
                
                # Draw bounding box
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add label with confidence
                label = f"{det['name']} ({det['confidence']:.2f})"
                cv2.rectangle(display_frame, (x1, y1-25), (x1+len(label)*8, y1), (0, 255, 0), -1)
                cv2.putText(display_frame, label, (x1, y1-7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # Draw FPS counter
            cv2.putText(display_frame, f"FPS: {self.fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Draw mode indicator
            mode = "Lane Detection: ON" if self.args.detect_lanes else "Lane Detection: OFF"
            cv2.putText(display_frame, mode, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw source indicator
            source_text = f"Source: {self.source}"
            cv2.putText(display_frame, source_text, (10, DISPLAY_HEIGHT - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            return display_frame
            
        except Exception as e:
            print(f"Error in UI drawing: {e}")
            return frame
    
    def process_frames(self):
        """Frame processing worker thread"""
        while self.running:
            if self.frame is None:
                time.sleep(0.01)
                continue
                
            frame_copy = self.frame.copy()
            
            # Detect traffic signs
            detections = self.detect_signs(frame_copy)
            
            # Detect lanes if enabled
            if self.args.detect_lanes:
                frame_copy = self.detect_lanes(frame_copy)
            
            # Draw UI
            display_frame = self.draw_ui(frame_copy, detections)
            
            # Calculate FPS
            self.frame_count += 1
            elapsed_time = time.time() - self.start_time
            if elapsed_time >= 1.0:
                self.fps = self.frame_count / elapsed_time
                self.frame_count = 0
                self.start_time = time.time()
            
            # Store the processed frame
            self.processed_frame = display_frame
            self.detections = detections
    
    def run(self):
        """Main method to run the system"""
        self.running = True
        
        # Start processing thread
        processing_thread = threading.Thread(target=self.process_frames)
        processing_thread.daemon = True
        processing_thread.start()
        
        try:
            while self.running:
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to receive frame. Exiting...")
                    break
                
                # Store the frame for processing
                self.frame = frame
                
                # Display the processed frame if available
                if self.processed_frame is not None:
                    cv2.imshow("Polish Traffic Sign Detection", self.processed_frame)
                
                # Check for key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('l'):
                    # Toggle lane detection
                    self.args.detect_lanes = not self.args.detect_lanes
                    print(f"Lane detection: {'ON' if self.args.detect_lanes else 'OFF'}")
                
        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            # Cleanup
            self.running = False
            if processing_thread.is_alive():
                processing_thread.join(timeout=1.0)
            self.cap.release()
            cv2.destroyAllWindows()
            print("System stopped")

def main():
    parser = argparse.ArgumentParser(description="Polish Traffic Sign Detection System")
    parser.add_argument("--source", type=str, default="0", 
                        help="Video source (0 for webcam, rtsp:// for stream, or video file path)")
    parser.add_argument("--detection_model", type=str, default="best.pt",
                        help="Path to YOLOv8 detection model")
    parser.add_argument("--classification_model", type=str, default="sign_classifier.pth",
                        help="Path to sign classification model")
    parser.add_argument("--classes_dir", type=str, default="data/classification",
                        help="Directory containing class folders for classification")
    parser.add_argument("--detect_lanes", action="store_true", 
                        help="Enable lane detection")
    
    args = parser.parse_args()
    
    # Create and run the system
    system = TrafficSignSystem(args)
    system.run()

if __name__ == "__main__":
    main()
