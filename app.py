
import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import time
from eye_detector import EyeStateDetector

class EyeStateApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.configure(bg='#f0f0f0')
        
        # Initialize video capture
        self.video_source = 0  # Default camera
        self.vid = cv2.VideoCapture(self.video_source)
        
        # Create detector
        self.detector = EyeStateDetector()
        
        # Create a canvas for the video
        self.canvas_width = 640
        self.canvas_height = 480
        self.canvas = tk.Canvas(window, width=self.canvas_width, height=self.canvas_height)
        self.canvas.pack(padx=10, pady=10)
        
        # Create status frame
        self.status_frame = ttk.Frame(window)
        self.status_frame.pack(pady=10, fill=tk.X)
        
        # Left Eye Status
        ttk.Label(self.status_frame, text="Left Eye:", font=('Arial', 14)).grid(row=0, column=0, padx=10)
        self.left_eye_status = ttk.Label(self.status_frame, text="Unknown", font=('Arial', 14, 'bold'))
        self.left_eye_status.grid(row=0, column=1, padx=10)
        
        ttk.Label(self.status_frame, text="Right Eye:", font=('Arial', 14)).grid(row=0, column=2, padx=10)
        self.right_eye_status = ttk.Label(self.status_frame, text="Unknown", font=('Arial', 14, 'bold'))
        self.right_eye_status.grid(row=0, column=3, padx=10)
        
        # Overall Status
        ttk.Label(self.status_frame, text="Status:", font=('Arial', 14)).grid(row=0, column=4, padx=10)
        self.overall_status = ttk.Label(self.status_frame, text="Unknown", font=('Arial', 14, 'bold'))
        self.overall_status.grid(row=0, column=5, padx=10)
        
        # Button frame
        self.btn_frame = ttk.Frame(window)
        self.btn_frame.pack(pady=10)
        
        # Start button
        self.start_btn = ttk.Button(self.btn_frame, text="Start", command=self.start_detection)
        self.start_btn.grid(row=0, column=0, padx=10)
        
        # Stop button
        self.stop_btn = ttk.Button(self.btn_frame, text="Stop", command=self.stop_detection)
        self.stop_btn.grid(row=0, column=1, padx=10)
        self.stop_btn["state"] = "disabled"
        
        # Exit button
        self.exit_btn = ttk.Button(self.btn_frame, text="Exit", command=self.on_close)
        self.exit_btn.grid(row=0, column=2, padx=10)
        
        # Variables for video processing
        self.processing = False
        self.thread = None
        
        # Status variables
        self.closed_count = 0
        self.consecutive_closed = 0
        self.max_consecutive_closed = 5  # Number of consecutive frames with closed eyes to change status
        
        # Set window close handler
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)
        
    def start_detection(self):
        """Start eye state detection"""
        if not self.processing:
            self.processing = True
            self.start_btn["state"] = "disabled"
            self.stop_btn["state"] = "normal"
            
            # Start processing thread
            self.thread = threading.Thread(target=self.update_frame)
            self.thread.daemon = True
            self.thread.start()
    
    def stop_detection(self):
        """Stop eye state detection"""
        if self.processing:
            self.processing = False
            self.start_btn["state"] = "normal"
            self.stop_btn["state"] = "disabled"
    
    def update_frame(self):
        """Update frame from video source"""
        while self.processing:
            ret, frame = self.vid.read()
            
            if ret:
                # Process the frame
                processed_frame, eye_states = self.detector.process_frame(frame)
                
                # Update eye states
                self.update_eye_states(eye_states)
                
                # Convert to display format
                self.photo = self.convert_frame_to_photo(processed_frame)
                
                # Update canvas
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            
            time.sleep(0.03)  # ~30 fps
    
    def update_eye_states(self, eye_states):
        """Update the eye state indicators"""
        if not eye_states:
            # No eyes detected
            self.left_eye_status.config(text="Not Detected", foreground="black")
            self.right_eye_status.config(text="Not Detected", foreground="black")
            return
        
        # Sort by position (usually left eye first, right eye second)
        # This is a simplification - in a real app, you'd track eyes more precisely
        if len(eye_states) >= 2:
            left_state = eye_states[0]
            right_state = eye_states[1]
            
            # Update left eye
            self.left_eye_status.config(
                text=left_state, 
                foreground="green" if left_state == "Open" else "red"
            )
            
            # Update right eye
            self.right_eye_status.config(
                text=right_state, 
                foreground="green" if right_state == "Open" else "red"
            )
            
            # Check overall state
            if left_state == "Closed" and right_state == "Closed":
                self.consecutive_closed += 1
            else:
                self.consecutive_closed = 0
            
            # Update overall status based on consecutive closed frames
            if self.consecutive_closed >= self.max_consecutive_closed:
                self.overall_status.config(text="Eyes Closed", foreground="red")
            else:
                self.overall_status.config(text="Eyes Open", foreground="green")
        
        elif len(eye_states) == 1:
            # Only one eye detected
            state = eye_states[0]
            self.left_eye_status.config(
                text=state, 
                foreground="green" if state == "Open" else "red"
            )
            self.right_eye_status.config(text="Not Detected", foreground="black")
            
            # Update overall based on the one eye
            if state == "Closed":
                self.consecutive_closed += 1
            else:
                self.consecutive_closed = 0
                
            if self.consecutive_closed >= self.max_consecutive_closed:
                self.overall_status.config(text="Eyes Closed", foreground="red")
            else:
                self.overall_status.config(text="Eyes Open", foreground="green")
    
    def convert_frame_to_photo(self, frame):
        """Convert OpenCV frame to PhotoImage for Tkinter"""
        # Convert from BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_frame)
        
        # Resize if needed
        if pil_image.width != self.canvas_width or pil_image.height != self.canvas_height:
            pil_image = pil_image.resize((self.canvas_width, self.canvas_height))
        
        # Convert to PhotoImage
        return ImageTk.PhotoImage(image=pil_image)
    
    def on_close(self):
        """Handle window close event"""
        # Stop processing
        self.processing = False
        
        # Release video resource
        if self.vid.isOpened():
            self.vid.release()
        
        # Close window
        self.window.destroy()

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = EyeStateApp(root, "Eye State Detection")
    root.mainloop()
        # Right Eye Status
