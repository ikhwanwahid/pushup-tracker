# Running R3D-18 Study on Vast.ai (GPU)

## Prerequisites

- Vast.ai account with credits (~$1-5 is plenty)
- VS Code with **Remote - SSH** extension installed
- Your videos in a folder + filled-in annotations xlsx

---

## Step 1: Add your SSH key to Vast.ai

On your Mac terminal:

```bash
# Check if you already have a key
cat ~/.ssh/id_ed25519.pub
# If "No such file", generate one:
ssh-keygen -t ed25519
cat ~/.ssh/id_ed25519.pub
```

Copy the output → go to [vast.ai/account](https://cloud.vast.ai/account/) → paste under **SSH Keys**.

---

## Step 2: Rent a GPU instance

1. Go to [vast.ai/search](https://cloud.vast.ai/search/)
2. Filter by:
   - **GPU**: RTX 3060 or better
   - **Image**: `pytorch/pytorch`
   - **Disk**: 20GB+
   - **SSH**: enabled (check the "Direct SSH" option)
3. Pick cheapest one → click **Rent**
4. Wait for status to show **Running**
5. Copy the SSH command from the dashboard (looks like `ssh -p 12345 root@ssh5.vast.ai`)

---

## Step 3: Connect VS Code to the instance

1. Open VS Code
2. `Cmd+Shift+P` → type **"Remote-SSH: Connect to Host"**
3. Click **"Add New SSH Host"**
4. Paste the SSH command from Step 2
5. Select `~/.ssh/config` when prompted
6. `Cmd+Shift+P` → **"Remote-SSH: Connect to Host"** → select your Vast.ai host
7. A new VS Code window opens connected to the remote machine
8. When prompted, install **Python** and **Jupyter** extensions on the remote

---

## Step 4: Clone the repo on the instance

Open the VS Code terminal (`Ctrl+`` `) on the remote machine:

```bash
cd /workspace
git clone <your-repo-url> pushup-tracker
cd pushup-tracker/r3d_study
pip install -r requirements.txt
```

Verify GPU works:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Should print: True NVIDIA RTX ...
```

---

## Step 5: Upload your data

The videos and annotations are NOT in the repo, so you need to upload them.

**Option A — Drag and drop in VS Code:**
1. In VS Code file explorer (left panel), navigate to `/workspace/pushup-tracker/r3d_study/`
2. Create a `videos/` folder
3. Drag your `.mp4` files from Finder into the `videos/` folder
4. Drag your `annotations_template.xlsx` into `r3d_study/`

**Option B — scp from a separate Mac terminal:**

```bash
# Upload annotations
scp -P <PORT> "/Users/ikhyvicky/Library/CloudStorage/OneDrive-SingaporeManagementUniversity/Group 4 Deep Learning for Computer Vision Project - training_dataset/annotations_template.xlsx" root@<HOST>:/workspace/pushup-tracker/r3d_study/

# Upload videos folder
scp -P <PORT> -r "/Users/ikhyvicky/Library/CloudStorage/OneDrive-SingaporeManagementUniversity/Group 4 Deep Learning for Computer Vision Project - training_dataset/videos" root@<HOST>:/workspace/pushup-tracker/r3d_study/
```

Replace `<PORT>` and `<HOST>` with values from your SSH command.

---

## Step 6: Update notebook paths

Open `notebook.ipynb` in VS Code. Change cell 2 to:

```python
# ============================================================
# CONFIGURATION — Vast.ai paths
# ============================================================
ANNOTATIONS   = Path("annotations_template.xlsx")
VIDEO_DIR     = Path("videos")
KEYPOINT_DIR  = Path("keypoints")
OUTPUT_DIR    = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda"
```

Delete the `ONEDRIVE_DIR` and `DATASET_DIR` lines — those only work on your Mac.

---

## Step 7: Run the notebook

1. Open `notebook.ipynb` in VS Code
2. Select kernel: **Python 3** (the remote Python with torch installed)
3. Run All Cells (`Cmd+Shift+Enter` or use the Run All button)
4. Watch the results appear inline

Expected time:
| GPU | Total time |
|-----|-----------|
| RTX 3060 | ~30 min |
| RTX 3090 | ~15 min |
| RTX 4090 | ~10 min |

---

## Step 8: Download results

Results are saved in `r3d_study/outputs/`:
- `r3d_study_results.csv` — full comparison table
- `r3d_hp_grid.csv` — hyperparameter grid results
- `r3d_best.pt` — best model weights
- `*.png` — all figures

**To download:** right-click the `outputs/` folder in VS Code file explorer → **Download**.

Or from your Mac terminal:

```bash
scp -P <PORT> -r root@<HOST>:/workspace/pushup-tracker/r3d_study/outputs/ ~/Desktop/r3d_results/
```

---

## Step 9: Destroy the instance

**Important — you're billed while the instance is running.**

1. Go to Vast.ai dashboard
2. Click **Destroy** on your instance
3. Billing stops immediately

---

## Troubleshooting

**"CUDA not available"**
```bash
python -c "import torch; print(torch.cuda.is_available())"
# If False, the instance may not have GPU drivers. Destroy and rent a different one.
```

**"No module named ..."**
```bash
cd /workspace/pushup-tracker/r3d_study
pip install -r requirements.txt
```

**Notebook kernel not finding modules**
Make sure the notebook kernel matches the Python where you installed requirements. In VS Code, click the kernel selector (top right of notebook) and pick the correct Python.

**Upload is slow**
Zip locally first, upload the zip, unzip on the instance:
```bash
# On your Mac
cd /tmp
zip -r data.zip videos/ annotations_template.xlsx
scp -P <PORT> data.zip root@<HOST>:/workspace/pushup-tracker/r3d_study/

# On the instance
cd /workspace/pushup-tracker/r3d_study
unzip data.zip
```
