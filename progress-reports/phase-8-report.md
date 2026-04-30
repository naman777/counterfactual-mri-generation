# Phase 8 Completion Report

## Conditional Diffusion Model Training and Synthetic MRI Generation

**Project:** Explainable Counterfactual MRI Generation and Acquisition Parameter Control for Robust Medical Imaging AI
**Current Completed Phase:** Phase 8 / Step 8 — Conditional Diffusion Model Prototype
**Current Status:** Conditional diffusion generation pipeline successfully implemented and validated at prototype level.
**Decision:** Move ahead to Step 9 — counterfactual editing prototype, while continuing diffusion quality improvement later.

---

## 1. Project Context

The project goal is to build an explainable and controllable MRI transformation system. The system should allow a user to select an MRI slice and modify scanner/acquisition-related parameters such as:

```text
1. Scanner manufacturer
2. Magnetic field strength
3. EchoTime
```

The final expected output is not only a generated image, but an explainable counterfactual system that shows:

```text
Original MRI slice
Counterfactual MRI slice
Difference heatmap
Local SSIM map
Histogram comparison
Predictor output before and after generation
```

This matches the implementation plan, where the MVP output includes the original image, generated counterfactual image, difference heatmap, local SSIM map, histogram comparison, and acquisition-parameter predictor output before and after generation. 

The proposal also states that the system should train a conditional diffusion model, use an acquisition-parameter predictor, and show explainability outputs such as difference maps, local structural similarity maps, histogram shifts, and predictor-based target alignment. 

Before Phase 8, we had already completed:

```text
Step 5: Preprocessing and slice curation
Step 6: Acquisition-parameter predictor
Step 7: Baseline transformations and explainability functions
```

So Phase 8 started from a strong foundation: clean processed data, patient-level splits, trained predictor, classical baselines, and explanation metrics.

---

# 2. What Phase 8 Was Supposed to Do

According to the implementation plan, Step 8 required implementing a conditional diffusion model using:

```text
Image size: 128 × 128
Channels: 1
Diffusion training steps: 1000
Sampling steps: 50 or 100
Condition dropout: 10–20%
Conditioning:
  Manufacturer → embedding
  Field strength → embedding
  EchoTime → scalar MLP embedding
```

The definition of done for this phase was:

```text
The model should be able to sample synthetic MRI-like images conditioned on:
manufacturer
field strength
EchoTime
```

The implementation plan specifically says that at this stage, the outputs do not need to be perfect. They need to prove that the pipeline works. 

Therefore, our goal in Phase 8 was not yet full counterfactual editing. The goal was to prove that a conditional diffusion model can learn the distribution of processed MRI slices and generate MRI-like samples under target acquisition conditions.

---

# 3. Work Completed in Phase 8

## 3.1 Task 1: Created Diffusion Configuration

We first created:

```text
configs/diffusion.yaml
```

This configuration file became the central control file for diffusion training and sampling.

It included:

```text
Dataset paths
Image size
Image channel count
Conditioning columns
Condition embedding dimensions
U-Net model size
DDPM noise schedule
Training hyperparameters
Sampling hyperparameters
Checkpoint paths
Log paths
Predictor checkpoint path
```

Important configuration choices:

```text
image_size: 128
image_channels: 1
num_train_timesteps: 1000
sample_steps: 50
condition_dropout_prob: 0.10
batch_size: 4
mixed_precision: true
device: cuda
```

### Why we created a config file

We created a separate config file because diffusion training has many hyperparameters. Hardcoding these values inside Python files would make experiments difficult to reproduce or modify.

This also makes the project more evaluation-ready because we can clearly show:

```text
What dataset was used
What model settings were used
What training settings were used
What checkpoint paths were used
What sampling settings were used
```

---

## 3.2 Task 2: Created Diffusion Dataset Loader

We created:

```text
src/data/diffusion_dataset.py
```

The dataset loader reads:

```text
data/processed/ax_t1_128/splits/train.csv
data/processed/ax_t1_128/splits/val.csv
```

Each row contains:

```text
image_path
manufacturer_label
field_strength_label
echo_time_norm
```

The dataset returns:

```text
image tensor: [1, 128, 128]
manufacturer_label
field_strength_label
echo_time_norm
image_path
```

The original preprocessed `.npy` images were normalized to `[0, 1]`. For diffusion training, we converted them to `[-1, 1]` using:

```text
image = image * 2 - 1
```

### Dataset check result

The dataset check passed:

```text
Training samples: 4434
Image shape: torch.Size([1, 128, 128])
Image range: [-1.0, 1.0]
Batch shape: torch.Size([4, 1, 128, 128])
```

### Why this step was necessary

Diffusion models learn to denoise images. For stable training, the model expects consistent tensor shapes and intensity ranges. Since our processed MRI slices were already resized and normalized, the dataset loader only needed to:

```text
Load .npy file
Convert image to float tensor
Convert [0, 1] to [-1, 1]
Attach acquisition labels
Return a training-ready dictionary
```

This ensured that the diffusion model received clean and consistent inputs.

---

## 3.3 Task 3: Created Conditioning Module

We created:

```text
src/models/conditioning.py
```

This file implements:

```text
AcquisitionConditionEmbedding
```

The conditioning module converts three acquisition parameters into one condition embedding:

```text
manufacturer_label → embedding
field_strength_label → embedding
echo_time_norm → MLP embedding
combined vector → final condition embedding
```

The final condition embedding shape was:

```text
[B, 128]
```

It also supports classifier-free guidance through condition dropout.

### Conditioning check result

The conditioning module passed:

```text
Conditional embedding shape: torch.Size([8, 128])
Unconditional embedding shape: torch.Size([8, 128])
Eval determinism check passed
```

### Why this step was necessary

The diffusion U-Net cannot directly use labels like:

```text
GE MEDICAL SYSTEMS
SIEMENS
1.5T
3.0T
EchoTime = 9.216
```

These must be converted into numerical embeddings. The discrete variables are represented using embedding layers, while EchoTime is continuous, so it is passed through a small MLP.

This design directly follows the planned conditioning strategy:

```text
Manufacturer → embedding
Field strength → embedding
EchoTime → scalar MLP embedding
```

---

## 3.4 Task 4: Created Conditional U-Net

We created:

```text
src/models/diffusion_unet.py
```

The model implemented was:

```text
MRIConditionalUNet
```

Input:

```text
noisy image: [B, 1, 128, 128]
timestep: [B]
condition embedding: [B, 128]
```

Output:

```text
predicted noise: [B, 1, 128, 128]
```

The U-Net used:

```text
Sinusoidal timestep embedding
Residual blocks
GroupNorm
SiLU activation
Downsampling blocks
Upsampling blocks
Skip connections
Additive time-condition injection
```

Model size:

```text
U-Net parameters: 7,150,337
Conditioning parameters: 30,304
```

### U-Net check result

The U-Net check passed:

```text
Device: cuda
Input shape: torch.Size([2, 1, 128, 128])
Output shape: torch.Size([2, 1, 128, 128])
Backward pass successful
```

### Why we used a lightweight U-Net

We used a custom lightweight U-Net instead of a very large diffusion architecture because:

```text
The dataset has only 4434 training slices
The image size is 128×128
Training is being done on a laptop GPU
The goal is first prototype validation, not final image quality
```

A larger model could overfit or become too slow to train. The selected architecture is reasonable for proving the pipeline.

---

## 3.5 Task 5: Created DDPM Scheduler and Training Script

We created:

```text
src/models/ddpm_scheduler.py
src/training/train_diffusion.py
```

The scheduler implements the forward diffusion process:

```text
x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
```

The model learns to predict the added noise.

Training objective:

```text
Loss = MSE(predicted_noise, true_noise)
```

The training script handles:

```text
Dataset loading
Model creation
Scheduler creation
Random timestep sampling
Noise addition
Noise prediction
MSE loss
Backpropagation
Mixed precision training
Gradient clipping
Checkpoint saving
Validation loss
Training history logging
Training summary JSON
```

---

## 3.6 Task 5 Debug Training: 200-Step Test

Before real training, we ran:

```bash
python src/training/train_diffusion.py --config configs/diffusion.yaml --debug --max_steps 200
```

Debug training result:

```text
Train samples: 4434
Val samples: 987
Batch size: 4
U-Net parameters: 7150337
Conditioning parameters: 30304
Global step: 200
Train loss: 0.089236
Val loss: 0.045527
```

Saved files:

```text
checkpoints/diffusion/ax_t1_128/latest.pt
checkpoints/diffusion/ax_t1_128/best.pt
experiments/diffusion_logs/ax_t1_128/history.csv
experiments/diffusion_logs/ax_t1_128/training_summary.json
```

### Why debug training was necessary

Diffusion training is expensive. A short debug run allowed us to confirm:

```text
The training loop works
CUDA is being used
Loss is finite
Checkpoints are saving
Validation is running
Logs are created
Sampling can load checkpoints
```

Without this debug run, a full training failure would have wasted time.

---

## 3.7 Task 6: Created Sampling Script

We created:

```text
src/inference/sample_diffusion.py
```

The sampling script loads a trained checkpoint and generates synthetic MRI-like samples from random noise.

It supports:

```text
Loading best/latest checkpoint
Building condition grids
Classifier-free guidance
DDIM-style sampling with 50 steps
Saving PNG grids
Saving raw .npy sample tensors
```

Initial sampling from the 200-step checkpoint produced mostly noisy outputs. This was expected because 200 steps are not enough for real image generation.

### Why sampling was needed

Training loss alone is not enough to judge a diffusion model. A model may reduce MSE but still generate bad images. Therefore, visual sampling is essential.

Sampling answered the question:

```text
Can the model turn random noise into MRI-like images under target conditions?
```

After only 200 steps, the answer was: the pipeline works, but quality is not acceptable yet.

---

# 4. Full 10-Epoch Training

After debug testing, we trained the model for 10 full epochs.

Command:

```bash
python src/training/train_diffusion.py --config configs/diffusion.yaml
```

Training result:

```text
Train samples: 4434
Val samples: 987
Batch size: 4
Total epochs: 10
Final global step: 11080
```

Training history:

```text
Epoch 1:  train_loss = 0.0415958, val_loss = 0.0216885
Epoch 2:  train_loss = 0.0235593, val_loss = 0.0143286
Epoch 3:  train_loss = 0.0217040, val_loss = 0.0300803
Epoch 4:  train_loss = 0.0214241, val_loss = 0.0294518
Epoch 5:  train_loss = 0.0194531, val_loss = 0.0223620
Epoch 6:  train_loss = 0.0196377, val_loss = 0.0157839
Epoch 7:  train_loss = 0.0197111, val_loss = 0.0175686
Epoch 8:  train_loss = 0.0176032, val_loss = 0.0191558
Epoch 9:  train_loss = 0.0185260, val_loss = 0.0196420
Epoch 10: train_loss = 0.0174031, val_loss = 0.0150031
```

Final summary:

```json
{
  "train_samples": 4434,
  "val_samples": 987,
  "batch_size": 4,
  "global_step": 11080,
  "latest_epoch": 10,
  "latest_train_loss": 0.017403050489034018,
  "latest_val_loss": 0.015003071543760597,
  "best_val_loss": 0.014328610366210342,
  "debug_mode": false
}
```

These values came from the final training logs and summary file. 

---

# 5. Interpretation of Training Loss

## 5.1 Good signs

The model clearly learned:

```text
Train loss decreased from 0.0416 to 0.0174
Validation loss decreased compared with epoch 1
No NaN values occurred
No training crash occurred
Checkpoints saved correctly
```

This means the diffusion objective is working.

## 5.2 Why validation loss was not perfectly smooth

Validation loss fluctuated:

```text
Best val loss: epoch 2
Final val loss: epoch 10
```

This is normal in diffusion training because:

```text
Timesteps are randomly sampled
Noise is randomly sampled
Batch size is small
Validation loss is stochastic
Noise-prediction MSE does not always correlate perfectly with visual sample quality
```

The key learning from this phase is that the visually best checkpoint was not the lowest-validation-loss checkpoint.

---

# 6. Checkpoint Comparison

We compared four sampling results:

```text
Image 1: samples_best_gs1.png
Image 2: samples_latest_gs1.png
Image 3: samples_latest_gs3.png
Image 4: samples_best_gs3.png
```

Checkpoint information:

```text
best.pt:
  epoch = 2
  global_step = 2216
  best_val_loss = 0.0143286

latest.pt:
  epoch = 10
  global_step = 11080
  best_val_loss = 0.0143286
```

The sampling logs confirm that `best.pt` came from epoch 2 and `latest.pt` came from epoch 10. 

---

# 7. Visual Results and Evaluation

## 7.1 Image 1: `samples_best_gs1.png`

This image used:

```text
Checkpoint: best.pt
Epoch: 2
Guidance scale: 1.0
Sampling steps: 50
```

Result:

```text
Some structure is visible
Images are still highly noisy
MRI-like foreground/background is weak
Fine detail is not reliable
```

Verdict:

```text
Not good enough for moving forward
```

---

## 7.2 Image 2: `samples_latest_gs1.png`

This image used:

```text
Checkpoint: latest.pt
Epoch: 10
Guidance scale: 1.0
Sampling steps: 50
```

Result:

```text
Much cleaner than best.pt
Strong foreground/background separation
Large tissue-like regions appear
Images look significantly more MRI-like
Noise is reduced
Synthetic samples are visually coherent
```

Verdict:

```text
Good enough for first Step 9 prototype
```

---

## 7.3 Image 3: `samples_latest_gs3.png`

This image used:

```text
Checkpoint: latest.pt
Epoch: 10
Guidance scale: 3.0
Sampling steps: 50
```

Result:

```text
Also strong
Slightly more aggressive contrast
Tissue-like structures are visible
Some samples appear sharper
May introduce stronger intensity effects
```

Verdict:

```text
Usable for counterfactual editing experiments
```

---

## 7.4 Image 4: `samples_best_gs3.png`

This image used:

```text
Checkpoint: best.pt
Epoch: 2
Guidance scale: 3.0
Sampling steps: 50
```

Result:

```text
Still noisy
Guidance scale did not fix undertraining
Structure is weaker than latest.pt samples
```

Verdict:

```text
Not selected
```

---

# 8. Final Phase 8 Decision

The main decision from Phase 8 is:

```text
Use latest.pt, not best.pt.
```

Although `best.pt` had the lowest validation loss, `latest.pt` produced better visual samples.

This is important because diffusion model selection cannot rely only on MSE validation loss. Visual generation quality must also be inspected.

Final selected checkpoint:

```text
checkpoints/diffusion/ax_t1_128/latest.pt
```

Recommended backup copy:

```text
checkpoints/diffusion/ax_t1_128_selected/latest_epoch10_visual_best.pt
```

---

# 9. Phase 8 Completion Status

## Completed

```text
configs/diffusion.yaml
src/data/diffusion_dataset.py
src/models/conditioning.py
src/models/diffusion_unet.py
src/models/ddpm_scheduler.py
src/training/train_diffusion.py
src/inference/sample_diffusion.py
```

## Generated artifacts

```text
checkpoints/diffusion/ax_t1_128/latest.pt
checkpoints/diffusion/ax_t1_128/best.pt
experiments/diffusion_logs/ax_t1_128/history.csv
experiments/diffusion_logs/ax_t1_128/training_summary.json
reports/diffusion/ax_t1_128/samples/debug_samples_200_steps.png
reports/diffusion/ax_t1_128/samples/samples_best_gs1.png
reports/diffusion/ax_t1_128/samples/samples_latest_gs1.png
reports/diffusion/ax_t1_128/samples/samples_best_gs3.png
reports/diffusion/ax_t1_128/samples/samples_latest_gs3.png
```

## Status

```text
Diffusion code implementation: complete
Training loop: complete
Sampling pipeline: complete
Synthetic MRI-like generation: successful
Final image quality: still improvable
Ready for Step 9 prototype: yes
```

---

# 10. Why We Are Moving Ahead

We are moving ahead because Step 8’s prototype goal has been met.

The implementation plan says Step 8 is complete when the model can generate synthetic MRI-like images conditioned on manufacturer, field strength, and TE/TR. It also says the outputs at this stage do not need to be perfect; they need to prove the pipeline works. 

Our latest checkpoint does exactly that:

```text
It generates 128×128 MRI-like images
It accepts manufacturer condition
It accepts field strength condition
It accepts EchoTime condition
It supports classifier-free guidance
It saves reproducible output grids
```

Therefore, the correct next step is Step 9:

```text
Original image
→ partial noising
→ denoising under target condition
→ counterfactual image
```

The proposal describes this as the core counterfactual editing process: partially noise the original slice, then denoise it under a new target condition so that appearance changes while anatomy remains stable. 

---

# 11. Limitations of Phase 8

Even though Phase 8 is successful, it is not final.

Current limitations:

```text
Only 10 epochs of training completed
Synthetic images still contain some artifacts
Conditioning effect is not fully quantitatively validated
Generated samples are MRI-like but not yet high-fidelity
The model has not yet been tested for anatomy-preserving editing
The predictor has not yet been applied to generated diffusion outputs
No FID/MMD-style distribution metric has been computed yet
```

These limitations are acceptable at this stage because Phase 8 was a prototype milestone, not the final evaluation stage.

---

# 12. What We Should Not Claim Yet

We should not claim:

```text
The model generates clinically realistic MRI scans
The model is ready for medical use
The model perfectly controls manufacturer or field strength
The model preserves anatomy during editing
The model has completed final counterfactual validation
```

We can claim:

```text
The conditional diffusion pipeline has been implemented
The model trains successfully on 128×128 ax_t1 MRI slices
The model can generate MRI-like synthetic samples
The latest checkpoint produces much better visual samples than the early checkpoint
The project is ready to begin counterfactual editing experiments
```

This is a strong and defensible evaluation position.

---

# 13. Evaluation Questions and Answers

## Q1. What was the purpose of Phase 8?

The purpose of Phase 8 was to implement and train the conditional diffusion model. This model is the main generative component of the project. It learns to generate MRI-like images conditioned on scanner manufacturer, magnetic field strength, and EchoTime.

---

## Q2. Why did we implement diffusion only after preprocessing, predictor, and baselines?

Because diffusion training is expensive and sensitive to data quality. Before training diffusion, we needed:

```text
Clean processed images
Patient-level train/val/test splits
Reliable acquisition labels
A trained predictor for later validation
Baseline transformations for comparison
```

This follows the implementation plan, which places conditional diffusion after preprocessing, predictor training, and baseline creation. 

---

## Q3. Why did we use the `ax_t1_128` dataset?

Because this subset had already been selected as the first clean and balanced subset. It contains both manufacturers, both field strengths, and usable EchoTime variation. It also had already been preprocessed to 128×128, making it suitable for fast diffusion experiments.

---

## Q4. Why did we use 128×128 images instead of 256×256?

We used 128×128 images because this is the first diffusion prototype. It reduces GPU memory usage, speeds up training, and makes debugging easier. Once the pipeline is stable, the model can later be scaled to 256×256.

---

## Q5. Why did we convert images from `[0, 1]` to `[-1, 1]`?

Diffusion models commonly train more stably when image tensors are centered around zero. The processed MRI slices were stored in `[0, 1]`, but the diffusion model input was converted to `[-1, 1]` before training.

---

## Q6. Why did we create a separate diffusion dataset loader?

Because the predictor and diffusion model have different input requirements. The predictor uses images for classification/regression, while the diffusion model needs images scaled to `[-1, 1]` and paired with condition labels for denoising.

---

## Q7. Why did we create a conditioning module?

The model cannot directly use text labels like `GE MEDICAL SYSTEMS` or `SIEMENS`. The conditioning module converts manufacturer, field strength, and EchoTime into a numerical embedding that can be injected into the U-Net.

---

## Q8. Why are manufacturer and field strength embeddings used?

Manufacturer and field strength are discrete categorical variables. Embedding layers allow the model to learn meaningful vector representations for each category.

---

## Q9. Why is EchoTime passed through an MLP?

EchoTime is a continuous variable. A small MLP can map the normalized scalar value into a richer embedding space, making it easier for the U-Net to use EchoTime as a conditioning signal.

---

## Q10. Why did we use condition dropout?

Condition dropout is required for classifier-free guidance. During training, the model sometimes receives an unconditional embedding instead of the true condition. This teaches the model both conditional and unconditional denoising, which later allows guidance during sampling.

---

## Q11. What is classifier-free guidance?

Classifier-free guidance combines conditional and unconditional model predictions during sampling. It increases the influence of the target condition without needing a separate classifier.

In our sampling:

```text
guided_noise = uncond_noise + guidance_scale × (cond_noise - uncond_noise)
```

Higher guidance can make outputs more aligned with the condition, but it can also make images more aggressive or artifact-prone.

---

## Q12. Why did we test guidance scale 1 and 3?

The base methodology mentioned in the proposal uses classifier-free guidance and notes that guidance scale 3 with 50 reverse steps was a good balance in the base paper. 

We tested:

```text
guidance_scale = 1.0
guidance_scale = 3.0
```

because early models may behave better with lower guidance, while stronger guidance may improve target alignment later.

---

## Q13. Why did we use 50 sampling steps?

Fifty reverse steps are computationally practical and aligned with the implementation plan, which recommends 50 or 100 sampling steps for inference. 

Using 50 steps allowed us to quickly inspect generated images and compare checkpoints.

---

## Q14. Why did we run a 200-step debug training first?

The debug run verified that:

```text
CUDA works
Data loading works
Forward diffusion works
U-Net forward pass works
Loss computation works
Backpropagation works
Checkpoints save correctly
Sampling can load checkpoints
```

This reduced risk before running full training.

---

## Q15. Were the 200-step samples good?

No. The 200-step samples were mostly noisy. But that was expected. The purpose of the 200-step run was not image quality; it was pipeline validation.

---

## Q16. Were the 10-epoch samples better?

Yes. The 10-epoch `latest.pt` samples were significantly better. They showed:

```text
Clearer foreground/background separation
Large tissue-like regions
Less noise
More MRI-like appearance
Better visual coherence
```

This showed that the diffusion model was learning.

---

## Q17. Why was `best.pt` not selected?

`best.pt` came from epoch 2 because it had the lowest validation MSE. However, its generated images were visually noisier than `latest.pt`.

Therefore, `best.pt` was not selected for the next phase.

---

## Q18. Why was `latest.pt` selected?

`latest.pt` came from epoch 10 and produced much better visual samples. Even though its validation MSE was slightly worse than epoch 2, the generated images were more realistic and more useful for counterfactual editing.

---

## Q19. What does this teach us about diffusion evaluation?

It shows that noise-prediction validation loss is useful, but not sufficient. Diffusion models must also be evaluated visually and later through quantitative generation metrics.

---

## Q20. Is Phase 8 fully complete?

Phase 8 is complete as a prototype milestone. The model can generate synthetic MRI-like images conditioned on acquisition parameters. However, final diffusion quality can still be improved with longer training and more evaluation.

---

## Q21. Can we move to counterfactual editing now?

Yes. We can move to Step 9 prototype because the conditional generation pipeline is working. The next step is to test whether an original MRI slice can be partially noised and then denoised under a target condition.

---

## Q22. Does moving ahead mean we stop training diffusion?

No. It means we can begin Step 9 while also planning longer diffusion training later. The next practical goal is to create a first working counterfactual editing pipeline.

---

## Q23. What checkpoint should be used for Step 9?

Use:

```text
checkpoints/diffusion/ax_t1_128/latest.pt
```

or the protected copy:

```text
checkpoints/diffusion/ax_t1_128_selected/latest_epoch10_visual_best.pt
```

---

## Q24. What settings should be used first in Step 9?

Start with:

```text
edit_timestep: 100, 200, 300
guidance_scale: 1.0 and 3.0
reverse_steps: 50
checkpoint: latest.pt
image_size: 128
```

Avoid starting with edit timestep 400 because it may change anatomy too much.

---

## Q25. How will we evaluate the first counterfactual outputs?

We will evaluate using:

```text
Visual comparison
Absolute difference map
Local SSIM map
Histogram comparison
Predictor before/after output
Target alignment
Anatomy preservation
```

This matches the proposal’s explainability and validation strategy. 

---

# 14. Professor/Evaluation Explanation

A strong evaluation explanation would be:

```text
In this phase, we implemented the conditional diffusion generation module. We created a diffusion-specific dataset loader, conditioning module, lightweight conditional U-Net, DDPM scheduler, training script, and sampling script. The model was trained on 4434 preprocessed ax_t1 MRI slices at 128×128 resolution using manufacturer, field strength, and normalized EchoTime as conditioning inputs.

We first ran a 200-step debug training pass to verify the pipeline. After confirming that data loading, training, checkpointing, and sampling worked, we trained the model for 10 epochs. The final training reached 11080 optimization steps, with final train loss 0.0174 and validation loss 0.0150.

We compared the best validation checkpoint and the latest checkpoint using guidance scales 1 and 3. Although the best checkpoint had the lowest validation MSE, the latest checkpoint produced visually better MRI-like samples. This showed that diffusion model selection should consider visual quality, not only noise-prediction loss.

The latest checkpoint successfully generated synthetic MRI-like images conditioned on manufacturer, field strength, and EchoTime. Therefore, the conditional diffusion prototype milestone is complete, and the project is ready to move to counterfactual editing, where an original image will be partially noised and denoised toward a new target acquisition condition.
```

---

# 15. Current Project Status After Phase 8

Completed:

```text
Step 0: MVP scope frozen
Step 1: Repository/environment setup
Step 2: Dataset preparation
Step 3: Metadata extraction
Step 4: Metadata exploration and control selection
Step 5: Preprocessing and patient-level splits
Step 6: Acquisition-parameter predictor
Step 7: Baseline transformations and explainability functions
Step 8: Conditional diffusion model prototype
```

Remaining:

```text
Step 9: Counterfactual editing
Step 10: Predictor-shift validation integration
Step 11: Dashboard integration
Step 12: Quantitative evaluation experiments
Step 13: Final report and demo assets
```

---

# 16. Final Conclusion

Phase 8 was successful.

The project now has a working conditional diffusion generation pipeline. The model was trained on the processed ax_t1 dataset, conditioned on manufacturer, field strength, and EchoTime. The training pipeline, checkpointing, and sampling system all work correctly. The latest 10-epoch checkpoint produces visibly MRI-like synthetic outputs and is much better than the early best-validation checkpoint.

The most important learning from this phase is that diffusion model evaluation must include both loss curves and visual sample inspection. Based on the visual results, `latest.pt` is selected for the next phase.

The project should now move to:

```text
Step 9: Counterfactual Editing Prototype
```

where the model will generate a true counterfactual by partially noising an original MRI slice and denoising it under a new target acquisition condition.
