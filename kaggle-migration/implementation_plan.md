# Kaggle Migration and Retraining Plan

This plan details the exact steps needed to migrate the existing codebase to Kaggle, link your new full dataset, and run the training pipeline for both the predictor and the diffusion models. 

## User Review Required

> [!IMPORTANT]
> Kaggle notebooks have specific session time limits (usually 12 hours max) and reset their local disk state when the session ends or restarts. 
> To ensure we don't lose checkpoints, we must mount Google Drive, save to a Kaggle Output directory, or use Weights & Biases (wandb) for logging and checkpointing.

> [!NOTE]
> Please review the steps below and let me know if you want me to assist in creating a zip script to package your code, or if you prefer me to initialize the Kaggle notebook directly via Kaggle's API (if you have the kaggle.json credentials set up).

## Open Questions

> [!WARNING]
> 1. **Dataset format**: You mentioned you found the full dataset. Is it already uploaded to Kaggle as a Kaggle Dataset, or is it on a Google Drive / external link?
> 2. **Kaggle API setup**: Do you have Kaggle API credentials (`kaggle.json`) configured locally on this machine? If yes, we can push the code directly to Kaggle from here.

## Proposed Changes

To migrate the project and retrain, we need to take the following steps:

### Step 1: Package the Project Code
We need to bundle the `src`, `configs`, `scripts`, and `requirements.txt` into an archive so it can be uploaded as a Kaggle Dataset or directly pulled into a Kaggle Notebook.
- **Action**: Create a simple python/shell script to zip the repository (ignoring `.git`, `checkpoints`, `.venv`, and `data`).

### Step 2: Set Up Kaggle Notebook Environment
- **Create a New Notebook** on Kaggle.
- **Select Accelerator**: Turn on GPU (P100 or T4x2 depending on availability).
- **Add Data**: 
  - Add your full MRI dataset (via "Add Data" button).
  - Add the zipped codebase (as a dataset) OR clone it from a GitHub repository if you have pushed it there.

### Step 3: Notebook Initialization Setup
In the first cell of your Kaggle Notebook, run setup commands to extract the code and install dependencies:
```bash
!cp -r /kaggle/input/your-code-dataset/* /kaggle/working/
!pip install -r /kaggle/working/requirements.txt
```

### Step 4: Adjust Configuration Files for Kaggle
We need to point the config files (`configs/data.yaml`, `configs/diffusion.yaml`, `configs/predictor.yaml`) to the new dataset path on Kaggle, which will typically look like `/kaggle/input/your-dataset-name/`.
- **Action**: We can either create Kaggle-specific YAML files locally before zipping, or override the paths via command-line arguments during training.

### Step 5: Retrain the Models
You need to train the models in the correct order (as per previous phases).

1. **Train the Predictor First**:
   ```bash
   !python /kaggle/working/src/training/train_predictor.py --config /kaggle/working/configs/predictor.yaml
   ```
   > [!TIP]
   > Make sure the checkpoints are saved to `/kaggle/working/checkpoints/` so they persist in the output of the Kaggle notebook.

2. **Train the Diffusion Model**:
   Once the predictor finishes, use its features/conditions to train the diffusion model:
   ```bash
   !python /kaggle/working/src/training/train_diffusion.py --config /kaggle/working/configs/diffusion.yaml
   ```

### Step 6: Save and Export Checkpoints
At the end of your training cells, zip the `checkpoints` folder and create a download link, or ensure it is saved in `/kaggle/working/` so Kaggle automatically zips it as the output of the notebook version.
```bash
!zip -r /kaggle/working/checkpoints_final.zip /kaggle/working/checkpoints
```

## Verification Plan

### Manual Verification
- Check Kaggle Dataset path structures match what is in `data.yaml`.
- Run a dummy training epoch (1 epoch) on Kaggle to verify data loading and checkpoint saving works before committing to a 10-12 hour run.
- Monitor Kaggle memory usage to ensure the batch sizes defined in `configs` don't cause an OOM (Out of Memory) error on Kaggle GPUs.
