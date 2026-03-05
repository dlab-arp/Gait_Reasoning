import os
import cv2
import json
import torch
import argparse
from collections import Counter
from ultralytics import YOLO
#from transformers import AutoProcessor, AutoModelForCausalLM
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
import re
from collections import Counter


parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, required=True, help="Folder with blurred videos")
parser.add_argument("--output_dir", type=str, required=True, help="Folder to save cropped videos")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
JSON_LOG = os.path.join(args.output_dir, "bbox_log.json")

print("Loading YOLOv8...")
yolo_model = YOLO('yolov8n.pt')

print("Loading Cosmos Reason 2 (This takes a moment)...")
MODEL_ID = "nvidia/Cosmos-Reason2-8B"
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
vlm_model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

def query_cosmos(image, prompt):
    """Sends an image and text to Cosmos and returns the text response."""
    # 1. Official Qwen3-VL conversation format
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},  # Pass the raw image
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # 2. Use apply_chat_template to get the exact tags the model expects
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # 3. Process with 'pixel_values'
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt"
    ).to("cuda")

    with torch.no_grad():
        generated_ids = vlm_model.generate(**inputs, max_new_tokens=20)

    # Trim the prompt from the output
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return output_text

def extract_target_id(video_path, output_dir, video_filename):
    """Runs YOLO, picks 3 frames, asks Cosmos for the patient ID, takes a vote, and saves evidence."""
    # Create an evidence folder inside your output directory
    evidence_dir = os.path.join(output_dir, "evidence_frames")
    os.makedirs(evidence_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Sample frames at 25%, 50%, and 75% of the video
    sample_frames = [int(total_frames * 0.25), int(total_frames * 0.50), int(total_frames * 0.75)]
    results = yolo_model.track(video_path, persist=True, classes=[0], verbose=False)
    votes = []

    for idx in sample_frames:
        if idx >= len(results):
            continue
        res = results[idx]
        if res.boxes is None or res.boxes.id is None:
            continue
        valid_ids = res.boxes.id.int().tolist()
        annotated_frame = res.plot()  # Get the frame with YOLO boxes and IDs drawn on it

        # Save the first valid annotated frame we find as evidence for your presentation
        if idx==sample_frames[1]:
            evidence_path = os.path.join(evidence_dir, f"evidence_{video_filename}.jpg")
            cv2.imwrite(evidence_path, annotated_frame)

        # --- THE SPEED HACK ---
        if len(valid_ids) == 1:
            print(f"Only 1 person found at frame {idx} (ID: {valid_ids[0]}). Auto-voting.")
            votes.append(valid_ids[0])
            continue

        # --- THE VLM INFERENCE ---
        dynamic_prompt = (
            f"Act as a clinical biomechanics expert. Look at the numbered bounding boxes in this image. "
            f"The only valid ID numbers in this image are: {valid_ids}. "
            f"Which ID number corresponds to the pediatric patient exhibiting gait abnormalities? "
            f"Ignore adults, parents, or background individuals. "
            f"Output strictly a single integer ID from the provided list."
        )

        response = query_cosmos(annotated_frame, dynamic_prompt)
        print(f"Cosmos raw vote at frame {idx} (Valid IDs: {valid_ids}): {response}")

        match = re.search(r'\d+', response)
        if match:
            vote = int(match.group())
            if vote in valid_ids:
                votes.append(vote)
            else:
                print(f"Warning: Cosmos hallucinated ID {vote}. Ignoring vote.")

    if not votes:
        return None, results

    # Majority vote
    target_id = Counter(votes).most_common(1)[0][0]
    print(f"Consensus Target ID: {target_id}")
    return target_id, results


def crop_video(video_path, output_path, target_id, track_results):
    """Crops the video dynamically based on the target ID's bounding box."""
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # We'll set a standard crop size (e.g., 512x512) to make Phase 2 easier
    crop_size = 512
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (crop_size, crop_size))

    frame_idx = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success or frame_idx >= len(track_results):
            break

        res = track_results[frame_idx]
        boxes = res.boxes

        # Find the bounding box for our target ID in this specific frame
        target_box = None
        if boxes.id is not None:
            for i, obj_id in enumerate(boxes.id):
                if int(obj_id) == target_id:
                    target_box = boxes.xyxy[i]
                    break

        if target_box is not None:
            x1, y1, x2, y2 = map(int, target_box)
            # Find center of the bounding box
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Create standard crop around the center
            half_size = crop_size // 2
            h, w, _ = frame.shape

            # Keep crop within image boundaries
            y_start = max(0, cy - half_size)
            y_end = min(h, cy + half_size)
            x_start = max(0, cx - half_size)
            x_end = min(w, cx + half_size)

            crop = frame[y_start:y_end, x_start:x_end]

            # Resize to exactly 512x512 if it hit a boundary
            crop = cv2.resize(crop, (crop_size, crop_size))
            out.write(crop)

        frame_idx += 1

    cap.release()
    out.write(crop)  # fallback
    out.release()


def main():
    video_files = os.listdir(args.input_dir)
    print(f"Found {len(video_files)} videos in {args.input_dir}")

    log_data = {}
    if os.path.exists(JSON_LOG):
        with open(JSON_LOG, 'r') as f:
            log_data = json.load(f)

    for video in video_files:
        if video in log_data and log_data[video]["status"] == "success":
            continue  # Skip already processed videos

        print(f"\n--- Processing {video} ---")
        video_path = os.path.join(args.input_dir, video)
        output_path = os.path.join(args.output_dir, f"cropped_{video}")
        video_filename = os.path.splitext(video)[0]

        target_id, track_results = extract_target_id(video_path, args.output_dir, video_filename)
        if target_id is not None:
            crop_video(video_path, output_path, target_id, track_results)
            log_data[video] = {"target_id": target_id, "status": "success"}
        else:
            print(f"Failed to find consensus ID for {video}")
            log_data[video] = {"target_id": None, "status": "failed"}

        # Save progress after every video
        with open(JSON_LOG, 'w') as f:
            json.dump(log_data, f, indent=4)


if __name__ == "__main__":
    main()


