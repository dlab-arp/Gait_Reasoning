import cv2
import mediapipe as mp
import os
from os.path import join
import glob
from ultralytics import YOLO

# Set up folders (make sure your videos are in the INPUT_DIR)
INPUT_DIR = "/Videos"
OUTPUT_DIR = "/Videos_Blurred"

# Automatically create the output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

deb = 1
model = YOLO('yolov8n.pt')
# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
# model_selection=1 is optimized for faces further away from the camera (perfect for gait labs)
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)


def blur_heads_in_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Processing: {os.path.basename(video_path)} ...")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Run YOLO on the frame, strictly looking for class 0 (person)
        # verbose=False stops it from printing out every single frame detection
        results = model(frame, classes=0, verbose=False)

        # Draw the blur on the top 20% of the detected person
        for box in results[0].boxes:
            # Get the coordinates of the person's full body
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Calculate the head region (top 20% of the bounding box)
            head_height = int((y2 - y1) * 0.2)

            hx1, hy1 = x1, y1
            hx2, hy2 = x2, y1 + head_height

            # Add a slight width padding to ensure ears/hair are caught
            pad_x = int((hx2 - hx1) * 0.1)
            hx1 = max(0, hx1 - pad_x)
            hx2 = min(width, hx2 + pad_x)

            # Extract region and apply heavy blur
            head_region = frame[hy1:hy2, hx1:hx2]
            if head_region.size != 0:
                blurred = cv2.GaussianBlur(head_region, (99, 99), 30)
                frame[hy1:hy2, hx1:hx2] = blurred

        out.write(frame)

    cap.release()
    out.release()
    print(f"Done! Saved to: {output_path}")


#
subj_dirs = [x for x in os.listdir(INPUT_DIR) if os.path.isdir(join(INPUT_DIR, x))]

# Find all mp4 files in the input directory
for subj_dir in subj_dirs:
    fpath = os.path.join(INPUT_DIR, subj_dir)
    video_files = glob.glob(os.path.join(fpath, "*.MOV"))
    #remove lateral views
    video_files = [x for x in video_files if 'anterior' or 'posterior' in x.lower()]
    if not video_files:
        print(f"No video files found in '{INPUT_DIR}'. Did you put them in the right folder?")
    else:
        print(f"Found {len(video_files)} videos. Starting batch blur...")
        for video_path in video_files:
            filename = os.path.basename(video_path)
            output_path = os.path.join(OUTPUT_DIR, filename)
            blur_heads_in_video(video_path, output_path)
            deb = 1

print("All videos processed successfully!")