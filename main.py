import cv2
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk

class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Camera Capture App")
        
        # Initialize camera index and video capture object
        self.camera_index = 0
        self.cam = cv2.VideoCapture(self.camera_index)

        # Load Haar cascades for face and smile detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

        # Get supported resolutions
        self.supported_resolutions = self.get_supported_resolutions()

        # Set default resolution
        self.current_resolution = self.supported_resolutions[0]
        self.set_resolution(self.current_resolution)

        # Create GUI elements
        self.label = tk.Label(root, text="Press 'Next Camera' to switch, 'Capture Photo' to save")
        self.label.pack()

        self.toggle_button = tk.Button(root, text="Next Camera", command=self.toggle_camera)
        self.toggle_button.pack()

        self.capture_button = tk.Button(root, text="Capture Photo", command=self.capture_photo)
        self.capture_button.pack()

        self.quit_button = tk.Button(root, text="Quit", command=self.quit_app)
        self.quit_button.pack()

        # Resolution dropdown
        self.resolution_var = tk.StringVar(value=f"{self.current_resolution[0]}x{self.current_resolution[1]}")
        self.resolution_menu = ttk.Combobox(root, textvariable=self.resolution_var)
        self.resolution_menu['values'] = [f"{w}x{h}" for w, h in self.supported_resolutions]
        self.resolution_menu.pack()
        self.resolution_menu.bind("<<ComboboxSelected>>", self.change_resolution)

        # Create a label to display the camera feed
        self.image_label = tk.Label(root)
        self.image_label.pack()

        # Variable to handle countdown
        self.countdown = 0
        self.is_smiling = False
        self.frames_per_count = 10  # Number of frames to wait before decrementing countdown

        # Start the camera preview
        self.update_camera_display()

    def get_supported_resolutions(self):
        common_resolutions = [
            (640, 480), (800, 600), (1024, 768), 
            (1280, 720), (1920, 1080), (3840, 2160)
        ]
        supported_resolutions = []

        for resolution in common_resolutions:
            width, height = resolution
            self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            actual_width = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if (actual_width, actual_height) == resolution:
                supported_resolutions.append(resolution)

        # Reset to the initial resolution
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, common_resolutions[0][0])
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, common_resolutions[0][1])

        return supported_resolutions

    def toggle_camera(self):
        # Move to the next camera
        self.camera_index += 1
        self.cam.release()
        self.cam = cv2.VideoCapture(self.camera_index)
        try:
            self.cam = cv2.VideoCapture(self.camera_index)
        except IndexError:
            # If no cameras are found, reset the camera index and try again
            self.camera_index = 0
            self.cam.release()
            self.cam = cv2.VideoCapture(self.camera_index)

        # Get supported resolutions for the new camera
        self.supported_resolutions = self.get_supported_resolutions()

        if not self.supported_resolutions:
            # If no supported resolutions are found, reset to default
            self.camera_index = 0
            self.cam = cv2.VideoCapture(self.camera_index)
            self.supported_resolutions = self.get_supported_resolutions()

            # If the default camera also has no supported resolutions, quit the app
            if not self.supported_resolutions:
                messagebox.showerror("Error", "No supported resolutions found for the default camera. Exiting.")
                self.quit_app()
                return

        # Set the first supported resolution as the current resolution
        self.current_resolution = self.supported_resolutions[0]
        self.set_resolution(self.current_resolution)

        # Update the resolution dropdown
        self.resolution_menu['values'] = [f"{w}x{h}" for w, h in self.supported_resolutions]
        self.resolution_var.set(f"{self.current_resolution[0]}x{self.current_resolution[1]}")

        if not self.cam.isOpened():
            self.camera_index = 0
            self.cam = cv2.VideoCapture(self.camera_index)
            self.set_resolution(self.current_resolution)

    def set_resolution(self, resolution):
        width, height = resolution
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def change_resolution(self, event):
        resolution = self.resolution_var.get()
        width, height = map(int, resolution.split('x'))
        self.current_resolution = (width, height)
        self.set_resolution(self.current_resolution)

    def capture_photo(self):
        # Capture and save the current frame
        ret, frame = self.cam.read()
        if ret:
            filename = f"captured_image_{self.camera_index}.png"
            cv2.imwrite(filename, frame)
            messagebox.showinfo("Success", f"Photo saved as {filename}")
        else:
            messagebox.showerror("Error", "Failed to capture photo")

    def update_camera_display(self):
        # Capture a frame from the current camera
        ret, frame = self.cam.read()
        if ret:
            # Convert the frame to grayscale for face detection
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the frame
            faces = self.face_cascade.detectMultiScale(gray_frame, 1.3, 5)

            smiling_detected = False

            for (x, y, w, h) in faces:
                # Draw rectangle around the face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Get the region of interest (ROI) for smile detection within the face
                roi_gray = gray_frame[y:y + h, x:x + w]
                roi_color = frame[y:y + h, x:x + w]

                # Detect smiles within the face
                smiles = self.smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=35, minSize=(25, 25))

                if len(smiles) > 0:
                    smiling_detected = True
                    # Draw rectangle around the smile
                    for (sx, sy, sw, sh) in smiles:
                        cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)

            if smiling_detected:
                if not self.is_smiling:
                    self.countdown = 3
                    self.is_smiling = True
                    self.frame_count = 0  # Reset frame count when a smile is detected
                else:
                    self.frame_count += 1
                    if self.frame_count >= self.frames_per_count:
                        self.countdown -= 1
                        self.frame_count = 0  # Reset frame count after each decrement

                # Display countdown on the screen
                cv2.putText(frame, str(self.countdown), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

                if self.countdown == 0:
                    self.capture_photo()
                    self.is_smiling = False

            else:
                self.is_smiling = False

            # Convert the frame to RGB (OpenCV uses BGR by default)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert the frame to a format suitable for Tkinter
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)

            # Update the label with the new image
            self.image_label.imgtk = imgtk
            self.image_label.configure(image=imgtk)

        # Schedule the update_camera_display method to be called again after 10 milliseconds
        self.root.after(10, self.update_camera_display)

    def quit_app(self):
        # Release the camera
        if self.cam.isOpened():
            self.cam.release()
        self.root.quit()

if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root)
    root.protocol("WM_DELETE_WINDOW", app.quit_app)  # Handle window close button
    root.mainloop()
