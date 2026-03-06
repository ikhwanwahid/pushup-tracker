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
   - **GPU**: RTX 3060 or better (RTX 3060 is plenty for this workload)
   - **Image**: `pytorch/pytorch`
   - **Disk**: 20GB+
   - **SSH**: select **Direct SSH** (not proxy)
3. Pick cheapest one → click **Rent**
4. Wait for status to show **Running**
5. Copy the SSH command from the dashboard (looks like `ssh -p 12345 root@199.68.217.31`)

---

## Step 3: Connect VS Code to the instance

1. Open VS Code
2. `Cmd+Shift+P` → **"Remote-SSH: Connect to Host"** → **"Add New SSH Host"**
3. Paste **only** the SSH command: `ssh -p 12345 root@199.68.217.31`
   - Do NOT include `-L 8080:localhost:8080` or other flags
4. Select `~/.ssh/config` when prompted
5. `Cmd+Shift+P` → **"Remote-SSH: Connect to Host"** → select the host (shown as the IP address)
6. A new VS Code window opens connected to the remote machine
7. Install extensions on remote: go to Extensions panel (`Cmd+Shift+X`), search and install **Python** and **Jupyter**

---

## Step 4: Fix DNS (if needed) and clone the repo

Open the VS Code terminal (`` Ctrl+` ``) on the remote machine.

Some Vast.ai instances can't resolve DNS. Test first:

```bash
ping -c 1 github.com
```

If it fails with "Temporary failure in name resolution", fix it:

```bash
echo "nameserver 8.8.8.8" > /etc/resolv.conf
```

Then clone and install:

```bash
cd /workspace
git clone -b phase3-feature-counting https://github.com/ikhwanwahid/pushup-tracker.git
cd pushup-tracker/r3d_study
pip install -r requirements.txt
```

Set up the Jupyter kernel:

```bash
pip install ipykernel
python -m ipykernel install --user --name r3d_study --display-name "R3D Study"
```

Set git identity (needed if you want to push from the instance):

```bash
git config user.email "your-email@example.com"
git config user.name "Your Name"
```

Verify GPU works:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Should print: True NVIDIA RTX ...
```

**Open the project:** File → Open Folder → `/workspace/pushup-tracker` → OK

---

## Step 5: Upload your data

The videos and annotations are NOT in the repo, so you need to upload them.

**Option A — Drag and drop in VS Code (easiest):**
1. In VS Code file explorer (left panel), navigate to `/workspace/pushup-tracker/r3d_study/`
2. Create a `videos/` folder
3. Drag your `.mp4` files from Finder into the `videos/` folder
4. Drag your `annotations_template.xlsx` into `r3d_study/`

**Option B — scp from a separate Mac terminal:**

```bash
# Upload annotations
scp -P <PORT> "annotations_template.xlsx" root@<HOST>:/workspace/pushup-tracker/r3d_study/

# Upload videos folder
scp -P <PORT> -r "videos/" root@<HOST>:/workspace/pushup-tracker/r3d_study/
```

**Option C — Zip first (faster for many files):**

```bash
# On your Mac
cd /path/to/your/data
zip -r data.zip videos/ annotations_template.xlsx
scp -P <PORT> data.zip root@<HOST>:/workspace/pushup-tracker/r3d_study/

# On the instance
cd /workspace/pushup-tracker/r3d_study
unzip data.zip
```

Replace `<PORT>` and `<HOST>` with values from your SSH command.

---

## Step 6: Update notebook paths

The notebook is already configured for Vast.ai. Verify cell 2 has:

```python
ANNOTATIONS   = Path("annotations_template.xlsx")
VIDEO_DIR     = Path("videos")
KEYPOINT_DIR  = Path("keypoints")
OUTPUT_DIR    = Path("outputs")
DEVICE = "cuda"
```

If you see `ONEDRIVE_DIR` lines, comment them out and use the above.

---

## Step 7: Run the notebook

1. Open `notebook.ipynb` in VS Code
2. Select kernel: click top-right kernel selector → pick **R3D Study**
3. Run All Cells (`Cmd+Shift+Enter` or use the Run All button)

### Expected timeline

| Step | Time |
|------|------|
| YOLO keypoint extraction (first run only) | ~15-30 min |
| Pre-loading video frames into RAM | ~15-25 min |
| Experiment A (full-frame, frozen) | ~10-15 min |
| Experiment B (YOLO-crop, frozen) | ~10-15 min |
| Unfreezing experiments | ~15-20 min |
| HP grid search (9 configs) | ~30-60 min |
| **Total (first run)** | **~1.5-2.5 hours** |

Keypoint extraction and preloading are one-time costs. If you rerun (e.g., with more data),
keypoints for existing videos are skipped. The preload happens each time the kernel restarts.

### Keeping the notebook running if you disconnect

SSH disconnects are common on Vast.ai. To avoid losing progress, convert to a script:

```bash
cd /workspace/pushup-tracker/r3d_study
jupyter nbconvert --to script notebook.ipynb
nohup python notebook.py > run_log.txt 2>&1 &
```

Then disconnect safely. Check progress anytime:

```bash
tail -f run_log.txt
```

Also run `caffeinate -i` in a separate Mac terminal to prevent your Mac from sleeping.

---

## Step 8: Download results

Before destroying the instance, download everything you need.

### What to download

| Folder/File | Why |
|---|---|
| `outputs/` | Results CSV, model weights, all figures |
| `keypoints/` | Pre-extracted YOLO keypoints (saves ~25 min on rerun) |
| `notebook.ipynb` | Notebook with outputs (if you edited on the instance) |

### How to download

**Option A — VS Code:** Right-click the folder → **Download**

**Option B — scp from Mac terminal:**

```bash
scp -P <PORT> -r root@<HOST>:/workspace/pushup-tracker/r3d_study/outputs/ ~/Desktop/r3d_results/
scp -P <PORT> -r root@<HOST>:/workspace/pushup-tracker/r3d_study/keypoints/ ~/Desktop/r3d_keypoints/
```

### Push to git (optional)

If you want to push notebook changes from the instance:

```bash
cd /workspace/pushup-tracker
git add r3d_study/notebook.ipynb
git commit -m "Updated notebook with training results"
git push
```

If `git push` fails with authentication errors, use a GitHub personal access token:
1. Go to github.com/settings/tokens → Generate new token (classic) → check `repo` scope
2. Push with: `git push https://USERNAME:<TOKEN>@github.com/USERNAME/pushup-tracker.git phase3-feature-counting`

---

## Step 9: Destroy the instance

**Important — you're billed while the instance exists (even when stopped).**

- **Stop** = pauses the instance, files preserved, but you still pay storage fees
- **Destroy** = deletes everything, billing stops immediately

1. Make sure you've downloaded `outputs/`, `keypoints/`, and the notebook
2. Go to Vast.ai dashboard
3. Click **Destroy** on your instance
4. Billing stops immediately

---

## Re-running with new data

When teammates add more videos:

1. Rent a new instance and follow Steps 3-6
2. Upload the new videos + updated `annotations_template.xlsx`
3. Upload your saved `keypoints/` folder (avoids re-extracting existing videos)
4. Run the notebook — only new videos get keypoint extraction
5. All experiments rerun on the full expanded dataset

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

**"Could not resolve host: github.com"**
```bash
echo "nameserver 8.8.8.8" > /etc/resolv.conf
```

**Notebook kernel not finding modules**
Click the kernel selector (top right of notebook) and pick **R3D Study** or the Python where you installed requirements.

**Kernel crashes during preload**
The frame preloading uses RAM. If the instance has limited RAM (<16GB), the kernel may crash. Rent an instance with more RAM, or reduce `N_FRAMES` in the notebook config.

**SSH disconnects mid-run**
Reconnect via VS Code (`Cmd+Shift+P` → Remote-SSH: Connect to Host). The instance is still running. However, the notebook kernel state is lost — you'll need to Run All again. To avoid this, use the `nohup` script approach (see Step 7).

**Git push authentication fails**
VS Code's credential helper doesn't work on remote instances. Use a personal access token (see Step 8).
