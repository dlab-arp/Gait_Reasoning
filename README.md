# Gait-Reasoner: Autonomous Clinical Gait Workstation
Gait-Reasoner is an end-to-end, explainable AI pipeline that transforms standard 2D video into structured clinical biomechanical assessments. Built for pediatric Cerebral Palsy triage, it leverages the NVIDIA Cosmos-Reason2-8B Vision-Language Model to autonomously calculate the **Observational Gait Scale (OGS)** without the need for expensive, multi-camera infrared gait labs.

**The Clinical Problem**
Gait analysis is fundamental for assessing the severity of Cerebral Palsy in pediatric patients. However, the current gold standard requires highly specialized gait labs staffed by technical experts, utilizing expensive physical markers and infrared cameras. While accurate, it is incredibly time-consuming, expensive, and inaccessible to most patients.

**Gait-Reasoner democratizes clinical biomechanics by enabling at-home clinical analysis using standard cell phone videos, providing a powerful, scalable tool for rapid clinical triaging.**

**System Architecture**
Our pipeline bridges zero-shot computer vision with generative medical reasoning:
1. Dynamic Patient Isolation: YOLOv8-pose is utilized to track all persons in the video and generate person-ids. Cosmos-2 is used for idenitifying the person-id of the the patient, isolating the   relevant biomechanical regions and ignoring background clinic distractors. 
3. Temporal Frame Sampling: Keyframes representing a full gait cycle are extracted and pre-processed.
4. Chain-of-Thought (CoT) Biomechanics: nvidia/Cosmos-Reason2-8B evaluates the visual data, explicitly reasoning through complex bilateral kinematics (e.g., midstance knee extension, heel rise         timing, altered base of support).
5. Deterministic Clinical Extraction: The VLM's CoT is parsed using an engineered regex and auto-repair pipeline to ensure 100% compliant, structured JSON outputs.
6. Executive Summary UI: A Gradio dashboard provides an explainable "second opinion," presenting the bilateral OGS scores and immediately flagging bipedal asymmetry.

**Validation & Results**
A clinical tool must be rigorously validated. Gait-Reasoner was evaluated against ground-truth clinician scores across two datasets.
| Dataset | Patient Type | Samples | Symmetry Accuracy | Overall OGS MAE |
| :--- | :--- | :--- | :--- | :--- |
| **GPJATK** | Normal / Healthy Baseline | 58 | 82.76% | 2.38 |
| **Participant Data** | Severe Cerebral Palsy Pathologies | 285 | 62.11% | 5.73 |

**Quick Start & Installation**
Prerequisites
* Ubuntu/Linux environment (tested on WSL2/Ubuntu) or Windows.
* NVIDIA GPU (Tested on L40S and RTX 40-series) with CUDA installed.
* **FFmpeg** is *required* for web-video standardization.
# Install system dependencies (Ubuntu/Debian)
sudo apt update && sudo apt install ffmpeg -y

# Clone the repository
git clone https://github.com/dlab-arp/gait-reasoner.git
cd gait-reasoner

# Install Python dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers ultralytics gradio opencv-python pandas
