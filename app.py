import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
from utils.eye_detector import EyeStateDetector

class EyeDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Eye State Detection System")
        self.root.geometry("800x600")
        
        # Initialize detector
        self.detector = EyeStateDetector()
        
        # Video capture
        self.cap = None
        self.running = False
        
        # Current states
        self.left_eye_state = "Unknown"
        self.right_eye_state = "Unknown"
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Eye State Detection System", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Video frame
        self.video_label = ttk.Label(main_frame, text="Camera feed will appear here")
        self.video_label.grid(row=1, column=0, columnspan=3, pady=(0, 20))
        
        # Eye state display
        state_frame = ttk.LabelFrame(main_frame, text="Eye States", padding="10")
        state_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 20))
        
        # Left eye
        ttk.Label(state_frame, text="Left Eye:").grid(row=0, column=0, sticky=tk.W)
        self.left_eye_label = ttk.Label(state_frame, text="Unknown", 
                                       font=("Arial", 12, "bold"))
        self.left_eye_label.grid(row=0, column=1, padx=(10, 0), sticky=tk.W)
        
        # Right eye
        ttk.Label(state_frame, text="Right Eye:").grid(row=1, column=0, sticky=tk.W)
        self.right_eye_label = ttk.Label(state_frame, text="Unknown", 
                                        font=("Arial", 12, "bold"))
        self.right_eye_label.grid(row=1, column=1, padx=(10, 0), sticky=tk.W)
        
        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=3, pady=(0, 10))
        
        self.start_button = ttk.Button(button_frame, text="Start Detection", 
                                      command=self.start_detection)
        self.start_button.grid(row=0, column=0, padx=(0, 10))
        
        self.stop_button = ttk.Button(button_frame, text="Stop Detection", 
                                     command=self.stop_detection, state="disabled")
        self.stop_button.grid(row=0, column=1, padx=(0, 10))
        
        self.exit_button = ttk.Button(button_frame, text="Exit", 
                                     command=self.exit_app)
        self.exit_button.grid(row=0, column=2)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
    
    def start_detection(self):
        """Start eye detection"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.status_var.set("Error: Cannot open camera")
                return
            
            self.running = True
            self.start_button.config(state="disabled")
            self.stop_button.config(state="normal")
            self.status_var.set("Detection started")
            
            # Start detection thread
            self.detection_thread = threading.Thread(target=self.detection_loop)
            self.detection_thread.daemon = True
            self.detection_thread.start()
            
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
    
    def stop_detection(self):
        """Stop eye detection"""
        self.running = False
        if self.cap:
            self.cap.release()
        
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.status_var.set("Detection stopped")
        
        # Clear video display
        self.video_label.config(image="", text="Camera feed will appear here")
        
        # Reset eye states
        self.update_eye_states("Unknown", "Unknown")
    
    def detection_loop(self):
        """Main detection loop"""
        while self.running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Process frame
                result_frame, results = self.detector.process_frame(frame)
                
                # Draw results
                final_frame = self.detector.draw_results(result_frame, results)
                
                # Update eye states
                if results:
                    result = results[0]  # Use first face
                    self.update_eye_states(
                        result['left_ml_state'], 
                        result['right_ml_state']
                    )
                else:
                    self.update_eye_states("No Face", "No Face")
                
                # Convert frame for display
                self.update_video_display(final_frame)
                
            except Exception as e:
                print(f"Detection error: {e}")
                break
    
    def update_video_display(self, frame):
        """Update video display in GUI"""
        try:
            # Resize frame
            frame = cv2.resize(frame, (640, 480))
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(rgb_frame)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(pil_image)
            
            # Update label
            self.video_label.config(image=photo, text="")
            self.video_label.image = photo  # Keep a reference
            
        except Exception as e:
            print(f"Display error: {e}")
    
    def update_eye_states(self, left_state, right_state):
        """Update eye state labels"""
        self.left_eye_state = left_state
        self.right_eye_state = right_state
        
        # Update labels with colors
        left_color = "green" if left_state == "Open" else "red" if left_state == "Closed" else "black"
        right_color = "green" if right_state == "Open" else "red" if right_state == "Closed" else "black"
        
        self.left_eye_label.config(text=left_state, foreground=left_color)
        self.right_eye_label.config(text=right_state, foreground=right_color)
    
    def exit_app(self):
        """Exit the application"""
        self.stop_detection()
        self.root.quit()
        self.root.destroy()

def main():
    """Main function"""
    root = tk.Tk()
    app = EyeDetectionApp(root)
    
    # Handle window close
    root.protocol("WM_DELETE_WINDOW", app.exit_app)
    
    # Start GUI
    root.mainloop()

if __name__ == "__main__":
    main()