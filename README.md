# FoundationPose on MR6D Dataset

We used FoundationPose to load the models from the MR6D dataset and evaluated them on the validation subset.

## 🛠️ Environment Setup

- Official Project Repository: [NVlabs/FoundationPose](https://github.com/NVlabs/FoundationPose)
- This implementation was deployed on **AutoDL Cloud Server (NVIDIA RTX 4090)**. We only need to download the `val/` and `models/` folders from the MR6D dataset.

### Option 1: Use Pre-built AutoDL Image (Recommended)
No manual compilation or environment setup required.
- AutoDL Image Link: [NVlabs/FoundationPose](https://www.autodl.art/i/NVlabs/FoundationPose/FoundationPose)
- The code is pre-installed at: `~/chentianxing/FoundationPose`
- Activate the environment:
  ```bash
  conda activate fp
  cd ~/chentianxing/FoundationPose
  ```

### Option 2: Manual Conda Setup (Official)
Follow the [official conda installation guide](https://github.com/NVlabs/FoundationPose#env-setup-option-2-conda-experimental). You will need to:
1. Create a conda environment with Python 3.9
2. Install all dependencies (Eigen3, PyTorch, NVDiffRast, PyTorch3D, etc.)
3. Build the C++ extensions
4. Download the pre-trained weights and place them in the `weights/` folder

### Headless Mode (For Cloud Servers)
Since cloud servers do not have a display, run the demo in headless mode. Results will be saved to the specified debug directory:
```bash
python run_demo_headless.py
```

## 🚀 Usage

### Single Scene Detection
- **Every frame loads mask**:
  ```bash
  python run_mr6d_multiobj_scene.py \
    --scene_dir /root/mr6d_full/val/000000 \
    --models_dir /root/mr6d_full/models \
    --debug_dir /root/chentianxing/FoundationPose/debug
  ```

- **Only first frame loads mask** (more efficient, recommended):
  ```bash
  python run_mr6d_multiobj_scene_5.py \
    --scene_dir /root/mr6d_full/val/000000 \
    --models_dir /root/mr6d_full/models \
    --debug_dir /root/chentianxing/FoundationPose/debug
  ```

### Batch Multi-Scene Detection
Automatically process all scenes in the validation set:
```bash
bash /root/chentianxing/FoundationPose/run_remaining.sh
```

### Single Scene Evaluation
Calculate pose estimation metrics against ground truth:
```bash
python /root/chentianxing/FoundationPose/evaluate_metrics.py \
    --gt_json /root/mr6d_full/val/000002/scene_gt.json \
    --pred_json /root/chentianxing/FoundationPose/debug2/scene_gt_pred.json
```

### Batch Multi-Scene Evaluation
Automatically evaluate all processed scenes:
```bash
bash /root/chentianxing/FoundationPose/eval_all.sh
```

## 📊 MR6D Dataset Structure

- Dataset Link: [anas-gouda/mr6d](https://huggingface.co/datasets/anas-gouda/mr6d)
- Total Size: ~12.6 GB
- Validation Split: 8,478 samples

### Validation Set (`val/`)
```text
val/
├── 000000/
│   ├── scene_camera.json      # Camera intrinsic & extrinsic parameters
│   ├── scene_gt.json          # 3D/6D ground truth (poses, object IDs)
│   ├── scene_gt_info.json     # 2D bounding boxes, occlusion masks, visibility info
│   ├── rgb/                   # RGB images
│   ├── depth/                 # Depth maps
│   ├── mask/                  # Full object projection masks (including occluded parts)
│   └── mask_visib/            # Visible-only segmentation masks (excluding occluded parts)
├── 000001/
│   └── ...
└── ...
```

### 3D Object Models (`models/`)
```text
models/
├── models_info.json           # Object metadata (dimensions, IDs, etc.)
├── obj_000001.ply             # 3D mesh file for object 1
├── obj_000002.ply             # 3D mesh file for object 2
├── obj_000003.ply
├── obj_000004.ply
├── obj_000005.ply
├── obj_000005.png             # Texture/render image for object 5
├── obj_000006.ply
├── obj_000006.png
├── obj_000007.ply
├── obj_000007.png
├── obj_000008.ply
├── obj_000008.png
├── obj_000009.ply
├── obj_000009.png
├── obj_000010.ply
├── obj_000010.png
├── obj_000011.ply
├── obj_000011.png
├── obj_000012.ply
├── obj_000012.png
├── obj_000013.ply
├── obj_000013.png
├── obj_000014.ply
├── obj_000014.png
├── obj_000015.ply
├── obj_000015.png
├── obj_000016.ply
└── obj_000016.png

0 directories, 29 files
```
