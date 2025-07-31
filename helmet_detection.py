import cv2
import sys
import time
import pygame
from ultralytics import YOLO
from tkinter import Tk
from tkinter.filedialog import askopenfilename

class HelmetDetection:
    def __init__(self, model_path, alarm_sound_path):
        self.model = YOLO(model_path)
        pygame.mixer.init()
        self.alarm_sound = pygame.mixer.Sound(alarm_sound_path)
        self.alarm_channel = None
        self.alarm_on = False
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        self.no_helmet_frame_buffer = 0
        self.required_consecutive_no_helmet_frames = 15
        self.required_consecutive_helmet_frames = 10

    def start_alarm(self):
        if not self.alarm_on:
            print("[ALARM] No helmet detected! Alarm ON!")
            self.alarm_channel = self.alarm_sound.play(loops=-1)
            self.alarm_on = True

    def stop_alarm(self):
        if self.alarm_on:
            print("[ALARM] Helmet detected. Alarm OFF!")
            self.alarm_channel.stop()
            self.alarm_on = False

    def draw_overlay_text(self, frame, lines, start_y=30, color=(255, 255, 255)):
        y = start_y
        for line, col in lines:
            cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
            y += 30
        return y  # Return last y for chaining other overlays

    def detect_and_count(self, frame):
        results = self.model.predict(frame, imgsz=640, conf=0.7)
        helmet_count = 0
        no_helmet_count = 0

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()

            for box, cls_id, conf in zip(boxes, classes, confs):
                width = int(box[2] - box[0])
                height = int(box[3] - box[1])
                min_box_size = 50

                if width < min_box_size or height < min_box_size:
                    continue

                if cls_id == 0:  # Helmet
                    helmet_count += 1
                    color = (0, 255, 0)
                    label = f"Helmet {conf:.2f}"
                elif cls_id == 1:  # Head (No Helmet)
                    no_helmet_count += 1
                    color = (0, 0, 255)
                    label = f"No Helmet {conf:.2f}"
                else:
                    continue

                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
                cv2.putText(frame, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        total_persons = helmet_count + no_helmet_count
        print(f"Total Persons: {total_persons}, With Helmet: {helmet_count}, Without Helmet: {no_helmet_count}")

        # Buffering logic for Alarm
        if no_helmet_count > 0:
            self.no_helmet_frame_buffer += 1
        else:
            self.no_helmet_frame_buffer = 0  # Reset if no head detected without helmet

        if self.no_helmet_frame_buffer >= self.required_consecutive_no_helmet_frames:
            self.start_alarm()
        elif no_helmet_count == 0 and helmet_count > 0:
            self.required_consecutive_helmet_frames -= 1
            if self.required_consecutive_helmet_frames <= 0:
                self.stop_alarm()
                self.required_consecutive_helmet_frames = 10
        else:
            self.required_consecutive_helmet_frames = 10

        # Overlay counts with colors
        overlay_lines = [
            (f"Total Persons: {total_persons}", (0, 255, 255)),   # Yellow
            (f"With Helmet: {helmet_count}", (0, 255, 0)),        # Green
            (f"No Helmet: {no_helmet_count}", (0, 0, 255))        # Red
        ]
        last_y = self.draw_overlay_text(frame, overlay_lines, start_y=30)

        # FPS Calculation every 10 frames
        self.fps_frame_count += 1
        if self.fps_frame_count >= 10:
            fps = self.fps_frame_count / (time.time() - self.fps_start_time)
            self.fps_start_time = time.time()
            self.fps_frame_count = 0
            self.current_fps = fps  # Store FPS for display
        if hasattr(self, 'current_fps'):
            cv2.putText(frame, f'FPS: {self.current_fps:.2f}', (20, last_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Blinking ALARM Text
        if self.alarm_on and int(time.time() * 2) % 2 == 0:
            cv2.putText(frame, '*** ALARM ON ***', (150, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

        return frame

    def webcam_detection(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        print("Press ESC or Z to exit Webcam mode.")
        self.fps_start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            frame = cv2.resize(frame, (640, 480))
            result_frame = self.detect_and_count(frame)
            cv2.imshow('Helmet Detection - Webcam', result_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('z'):
                print("Exiting Webcam mode.")
                self.stop_alarm()
                break

        cap.release()
        cv2.destroyAllWindows()

    def video_detection(self):
        while True:
            Tk().withdraw()
            video_path = askopenfilename(title="Select Video File", filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
            if not video_path:
                print("No video selected. Returning to Main Menu.")
                return

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print("Error: Could not open video.")
                continue

            print("Press ESC or Z to exit Video mode.")
            self.fps_start_time = time.time()

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("End of video file.")
                    break

                frame = cv2.resize(frame, (640, 480))
                result_frame = self.detect_and_count(frame)
                cv2.imshow('Helmet Detection - Video', result_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('z'):
                    print("Exiting Video mode.")
                    self.stop_alarm()
                    break

            cap.release()
            cv2.destroyAllWindows()

            next_video = input("Load another video? (y/n): ").strip().lower()
            if next_video != 'y':
                break

    def image_detection(self):
        Tk().withdraw()
        image_path = askopenfilename(title="Select Image File", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if not image_path:
            print("No file selected.")
            return

        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image {image_path}")
            return

        result_image = self.detect_and_count(image)
        cv2.imshow('Helmet Detection - Image', result_image)
        cv2.imwrite('output_image_detected.jpg', result_image)
        print("Detection completed. Press any key to close image window.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        self.stop_alarm()

    def main_menu(self):
        while True:
            print("\nSelect Mode:")
            print("1. Webcam Detection")
            print("2. Image Detection")
            print("3. Video Detection")
            print("4. Exit")
            choice = input("Enter choice (1/2/3/4): ")

            if choice == '1':
                self.webcam_detection()
            elif choice == '2':
                self.image_detection()
            elif choice == '3':
                self.video_detection()
            elif choice == '4':
                print("Exiting program.")
                self.stop_alarm()
                pygame.mixer.quit()
                sys.exit()
            else:
                print("Invalid choice. Please select 1, 2, 3, or 4.")

if __name__ == "__main__":
    try:
        detector = HelmetDetection('runs/detect/train21/weights/best.pt', 'security-alarm-63578.mp3')
        detector.main_menu()
    except KeyboardInterrupt:
        print("\nProgram Interrupted. Exiting cleanly.")
        detector.stop_alarm()
        pygame.mixer.quit()
        cv2.destroyAllWindows()
        sys.exit()
