import gradio as gr
import cv2
import json
import torch
import numpy as np
from ultralytics import YOLO
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
import subprocess
import os
import re


def convert_to_mp4(input_path):
    """Standardizes any video format to a browser-playable H.264 MP4."""
    output_path = "standardized_input.mp4"
    if os.path.exists(output_path):
        os.remove(output_path)

    print(f"Standardizing video: {input_path}")
    # Force libx264 for the widest browser compatibility
    subprocess.call([
        'ffmpeg', '-y', '-i', input_path,
        '-vcodec', 'libx264', '-crf', '25',
        '-pix_fmt', 'yuv420p', '-acodec', 'aac', output_path
    ])
    return output_path


def interpret_results(scores):
    try:
        l_total = sum(scores['left'].values())
        r_total = sum(scores['right'].values())
        is_asym = scores.get('symmetry', 0) == 1

        sym_label = "ASYMMETRICAL" if is_asym else "SYMMETRICAL"
        status_color = "#d32f2f" if is_asym else "#2e7d32"  # Professional Red/Green

        impact_phrase = "No primary affected limb identified (Bilateral consistency)."
        if l_total != r_total:
            primary = "Left" if l_total < r_total else "Right"
            impact_phrase = f"Primary deviation observed in the <b>{primary}</b> limb."

        # Professional HTML Summary
        html_summary = f"""
        <div style="font-family: 'Segoe UI', Tahoma, Geneva, sans-serif; padding: 15px; border-left: 5px solid {status_color}; background-color: #f8f9fa;">
            <h3 style="margin: 0; color: #333;">Clinical Summary</h3>
            <hr style="border: 0; border-top: 1px solid #ddd; margin: 10px 0;">
            <p style="font-size: 1.1em; margin: 5px 0;">
                <b>Gait Pattern:</b> <span style="color: {status_color};"><b>{sym_label}</b></span>
            </p>
            <p style="margin: 5px 0;"><b>Left OGS:</b> {l_total}/18 | <b>Right OGS:</b> {r_total}/18</p>
            <p style="margin: 10px 0; font-style: italic; color: #555;">{impact_phrase}</p>
        </div>
        """
        return html_summary
    except Exception as e:
        return f"<p style='font-family: sans-serif; color: red;'>Awaiting Analysis... (Status: {e})</p>"


def process_gait_analysis(uploaded_file_path):
    # --- STEP 1: Standardize Input (MOV/AVI -> Web-Ready MP4) ---
    input_mp4 = "standardized_input.mp4"
    subprocess.call([
        'ffmpeg', '-y', '-i', uploaded_file_path,
        '-vcodec', 'libx264', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', input_mp4
    ])

    # --- STEP 2: Cosmos VLM Reasoning ---
    cap = cv2.VideoCapture(input_mp4)
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, 8, dtype=int)
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret: frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    prompt = (
              "You are a clinical biomechanics expert. Analyze this gait sequence. "
              "IMPORTANT CLINICAL SCALE (Except Symmetry): For all parameters, 3 = Normal/Healthy. 2 = Moderate Impairment, 1 = Severe Impairment and 0 = Extreme Impairment"
              "You must provide SEPARATE Observational Gait Scale (OGS) scores for the LEFT and RIGHT limbs. "
              "1. Knee position in midstance (0-3) - Left & Right\n"
              "2. Initial foot contact (0-3) - Left & Right\n"
              "3. Foot contact at midstance (-1 to 3) - Left & Right\n"
              "4. Timing of heel rise (0-3) - Left & Right\n"
              "5. Hindfoot at midstance (0-2) - Left & Right\n"
              "6. Base of support (0-3)\n"
              "7. Symmetry (0=Symmetrical, 1=Asymmetrical). To determine symmetry, you MUST explicitly compare the left and right sides."
              " If there are visible differences in step length, joint angles, or weight-bearing, OR if your assigned Left and Right OGS scores are not identical,"
              " you MUST score 1 (Asymmetrical). Otherwise, score 0.\n"
              "Explain your reasoning for each leg in <think> tags, then provide the scores in this JSON format: "
              "<answer> {\"left\": {\"knee\": X, \"init_contact\": X...}, \"right\": {\"knee\": X, ...}, \"symmetry\": X} </answer>"
              )

    messages = [{"role": "user",
                 "content": [*[{"type": "image", "image": f} for f in frames], {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=frames, return_tensors="pt").to("cuda")

    # 1. Generate response deterministically
    with torch.no_grad():
        generated_ids = vlm_model.generate(
            **inputs,
            max_new_tokens=1000,
            do_sample=True,  #
            temperature=0.2,  #
            top_p=0.9,
            repetition_penalty=1.1
        )

    # 2. Slice out the prompt
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

    # 3. Extract Reasoning
    if "<think>" in output_text and "</think>" in output_text:
        reasoning = output_text.split("<think>")[-1].split("</think>")[0].strip()
    else:
        reasoning = output_text.split("{")[0].strip()

    # 4. Extract JSON Scores (with Auto-Repair)
    try:
        json_match = re.search(r'(\{.*\})', output_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
            json_str = json_str.split("</answer>")[0].strip()

            # --- AUTO-CLOSE FIX ---
            if json_str.count('{') > json_str.count('}'):
                json_str += '}'

            scores = json.loads(json_str)

            # Scrub nested symmetry
            # for limb in ['left', 'right']:
            #     if limb in scores: scores[limb].pop('symmetry', None)
        else:
            raise ValueError("No JSON found in model output")

    except Exception as e:
        print(f"Extraction failed: {e}")
        scores = {"left": {"knee": 0}, "right": {"knee": 0}, "symmetry": 0}
        reasoning = f"EXTRACT_FAIL: Check terminal for raw output.\n\n{reasoning}"

    # --- STEP 3: Create Annotated Output ---
    temp_output = "raw_annotated.mp4"
    final_output = "final_web_report.mp4"

    cap = cv2.VideoCapture(input_mp4)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(temp_output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        pose_res = pose_model(frame, verbose=False)[0]
        frame = pose_res.plot(labels=False, boxes=False)
        cv2.putText(frame, "COSMOS-2 ANALYSIS ACTIVE", (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        writer.write(frame)
    cap.release()
    writer.release()

    subprocess.call([
        'ffmpeg', '-y', '-i', temp_output,
        '-vcodec', 'libx264', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', final_output
    ])

    summary_text = interpret_results(scores)

    return final_output, reasoning, scores, summary_text


# --- 1. INITIALIZATION ---
print("Initializing Clinical Gait Reasoner...")
pose_model = YOLO('yolov8n-pose.pt')
MODEL_ID = "nvidia/Cosmos-Reason2-8B"
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
vlm_model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
)

# --- 2. DASHBOARD UI ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo")) as demo:
    gr.HTML("<h1 style='text-align: center;'>Autonomous Clinical Gait Reasoner</h1>")

    with gr.Row():
        with gr.Column(scale=2):
            input_vid = gr.Video(label="Input Recording")
            analyze_btn = gr.Button("EXECUTE CLINICAL ANALYSIS", variant="primary")
            output_vid = gr.Video(label="Annotated Skeletal View")

        with gr.Column(scale=1):
            interpretation_box = gr.HTML("<p style='font-family: sans-serif;'>Awaiting Analysis...</p>")
            gr.Markdown("---")
            reasoning_out = gr.Textbox(label="Biomechanical Reasoning (CoT)", lines=15, interactive=False)
            json_out = gr.JSON(label="Structured Data")

    analyze_btn.click(
        fn=process_gait_analysis,
        inputs=input_vid,
        outputs=[output_vid, reasoning_out, json_out, interpretation_box]
    )

demo.launch(share=True)