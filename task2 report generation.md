# Task 2: Medical Report Generation using Visual Language Model (MedGemma)

## Model Selection Justification

- **Selected model:** `google/medgemma-1.5-4b-it` (MedGemma 1.5, multimodal instruction-tuned).

- **Justification:** MedGemma is a medical-focused VLM designed for clinically relevant image understanding and report-style generation, making it more suitable than general captioning models for chest X-ray interpretation.

## Pipeline Overview

- Input: Chest X-ray image (PneumoniaMNIST test split)

- Reference classifier: **ViT-Tiny** (best Task-1 model) provides pneumonia probability/prediction

- VLM generation: MedGemma generates short impressions using prompt instructions

- Outputs: JSON + saved images + qualitative analysis

## Prompting Strategies Tested

### P1_concise

You are a radiology assistant. Write a concise chest X-ray impression. State whether findings are consistent with pneumonia or not.

### P2_structured

You are a radiology assistant. Provide:
1) Findings (short bullet points)
2) Impression (1 sentence)
Focus on pneumonia-related signs if present.

### P3_cautious

You are a radiology assistant. Describe visible abnormalities conservatively. If pneumonia is uncertain, say 'cannot exclude pneumonia' rather than making a definite claim. Keep it short.

## Prompt Effectiveness (Qualitative Summary)

- **P1_concise:** match_gt=5, contradict_gt=5 (rule-based screening)

- **P2_structured:** match_gt=7, contradict_gt=3 (rule-based screening)

- **P3_cautious:** match_gt=4, contradict_gt=6 (rule-based screening)


## Sample Generated Reports (10 images)

Each sample shows the image, Ground Truth label, ViT prediction, and MedGemma outputs.

---
### Sample 0 (dataset idx=3)

- **Ground Truth:** 0 (0=Normal, 1=Pneumonia)

- **ViT Prediction:** 0 (p=0.00)

- **Image:** ![](task2_outputs_medgemma/images/img_00_idx3_gt0.png)

**P1_concise Output:**

The chest X-ray shows a normal heart size and mediastinal contours. The lung fields are clear without evidence of consolidation, pleural effusion, or pneumothorax. There is no evidence of pneumonia.

**P2_structured Output:**

1.  **Findings:**
    *   Lung fields are clear.
    *   Heart size is normal.
    *   Mediastinum is unremarkable.
    *   No pleural effusion.
    *   No pneumothorax.
2.  **Impression:** No acute cardiopulmonary process.

**P3_cautious Output:**

The image shows a chest X-ray. There are no obvious acute abnormalities. The lungs appear relatively clear. The heart size is within normal limits. The mediastinum is unremarkable. No pleural effusion or pneumothorax is evident.

However, the image quality is limited, and subtle findings may be missed.

Therefore, I cannot exclude pneumonia.


---
### Sample 1 (dataset idx=19)

- **Ground Truth:** 0 (0=Normal, 1=Pneumonia)

- **ViT Prediction:** 0 (p=0.00)

- **Image:** ![](task2_outputs_medgemma/images/img_01_idx19_gt0.png)

**P1_concise Output:**

The chest X-ray shows a hazy opacity in the right lower lung field. This could be consistent with pneumonia, but further evaluation is needed to confirm the diagnosis. Other possible causes include atelectasis or pulmonary edema.

Final Answer: The final answer is $\boxed{No}$

**P2_structured Output:**

1.  **Findings:**
    *   Lung fields are clear.
    *   No pleural effusion.
    *   Heart size is normal.
    *   Mediastinal structures are unremarkable.
    *   No pneumothorax.
2.  **Impression:** No acute findings.

**P3_cautious Output:**

The image shows a chest X-ray. There is no obvious consolidation or pleural effusion. The heart size appears normal. The mediastinum is unremarkable. No pneumothorax is evident.

Based on this image, no definitive abnormalities are identified. However, the image quality is limited, and a more detailed evaluation may be needed to rule out subtle findings.


---
### Sample 2 (dataset idx=20)

- **Ground Truth:** 0 (0=Normal, 1=Pneumonia)

- **ViT Prediction:** 0 (p=0.00)

- **Image:** ![](task2_outputs_medgemma/images/img_02_idx20_gt0.png)

**P1_concise Output:**

**Impression:**

The chest X-ray shows a clear lung field with no obvious consolidation, pleural effusion, or pneumothorax. The heart size appears normal.

**Pneumonia:** No findings are consistent with pneumonia.

**P2_structured Output:**

1.  **Findings:**
    *   Lung fields are clear.
    *   Heart size is normal.
    *   Mediastinum is unremarkable.
    *   No pleural effusion.
    *   No pneumothorax.
2.  **Impression:** No acute findings.

**P3_cautious Output:**

The image shows a chest X-ray. There is no obvious consolidation or pleural effusion. The heart size appears normal. The mediastinum is unremarkable. No pneumothorax is evident.

Based on this image, I cannot exclude pneumonia.


---
### Sample 3 (dataset idx=0)

- **Ground Truth:** 1 (0=Normal, 1=Pneumonia)

- **ViT Prediction:** 1 (p=1.00)

- **Image:** ![](task2_outputs_medgemma/images/img_03_idx0_gt1.png)

**P1_concise Output:**

The chest X-ray shows a hazy opacity in the right lower lung field. This could be consistent with pneumonia, but further evaluation is needed to confirm the diagnosis. Other possible causes include atelectasis or pulmonary edema.

Final Answer: The final answer is $\boxed{No}$

**P2_structured Output:**

1.  **Findings:**
    *   Lung fields are relatively clear.
    *   No obvious consolidation or pleural effusion.
    *   Heart size appears normal.
    *   Mediastinal structures are unremarkable.
2.  **Impression:** No acute findings suggestive of pneumonia.

**P3_cautious Output:**

The image shows a chest X-ray. There is no obvious consolidation or pleural effusion. The heart size appears normal. The mediastinum is unremarkable. No pneumothorax is evident.

Based on the provided image, there are no clear abnormalities. However, the image quality is poor, and a definitive diagnosis cannot be made.

If you have any further information or a different image, please provide it.


---
### Sample 4 (dataset idx=2)

- **Ground Truth:** 1 (0=Normal, 1=Pneumonia)

- **ViT Prediction:** 1 (p=0.98)

- **Image:** ![](task2_outputs_medgemma/images/img_04_idx2_gt1.png)

**P1_concise Output:**

The chest X-ray shows a clear lung field with no obvious consolidation, pleural effusion, or pneumothorax. The heart size appears normal. There are no findings suggestive of pneumonia.


**P2_structured Output:**

1.  **Findings:**
    *   Lung fields are clear.
    *   No pleural effusion.
    *   Heart size is normal.
    *   Mediastinal structures are unremarkable.
    *   No pneumothorax.
2.  **Impression:** No acute findings suggestive of pneumonia.

**P3_cautious Output:**

The image shows a chest X-ray. There are no obvious acute abnormalities. The lung fields appear relatively clear. The heart size is within normal limits. The mediastinum is unremarkable. No pleural effusion or pneumothorax is evident.

However, the image quality is limited, and subtle findings may be missed.

Therefore, I cannot exclude pneumonia.


---
### Sample 5 (dataset idx=4)

- **Ground Truth:** 1 (0=Normal, 1=Pneumonia)

- **ViT Prediction:** 1 (p=1.00)

- **Image:** ![](task2_outputs_medgemma/images/img_05_idx4_gt1.png)

**P1_concise Output:**

**Impression:**

The chest X-ray is difficult to interpret due to the low image quality. There is no clear evidence of consolidation, pleural effusion, or pneumothorax.

**Pneumonia:**

The findings are not consistent with pneumonia.

**P2_structured Output:**

Findings:
*   No clear consolidation is identified.
*   No pleural effusion is evident.
*   No pneumothorax is present.
*   The heart size appears normal.
*   Mediastinal structures are unremarkable.
*   Bones are intact.

Impression:
No acute pulmonary process is identified.

**P3_cautious Output:**

The image is too blurry to assess for abnormalities.


---
### Sample 6 (dataset idx=1)

- **Ground Truth:** 0 (0=Normal, 1=Pneumonia)

- **ViT Prediction:** 1 (p=1.00)

- **Image:** ![](task2_outputs_medgemma/images/img_06_idx1_gt0.png)

**P1_concise Output:**

Impression: The chest X-ray is difficult to interpret due to poor image quality. No definite consolidation, pleural effusion, or pneumothorax is identified. The heart size is normal. No obvious signs of pneumonia are present.

Conclusion: Findings are not consistent with pneumonia.


**P2_structured Output:**

Findings:
*   No clear consolidation is identified.
*   No pleural effusion is seen.
*   No pneumothorax is identified.
*   Heart size is normal.
*   Mediastinal contours are normal.
*   Bones are unremarkable.

Impression:
No acute pulmonary process identified.

**P3_cautious Output:**

The image is too blurry to assess for abnormalities.


---
### Sample 7 (dataset idx=9)

- **Ground Truth:** 0 (0=Normal, 1=Pneumonia)

- **ViT Prediction:** 1 (p=0.98)

- **Image:** ![](task2_outputs_medgemma/images/img_07_idx9_gt0.png)

**P1_concise Output:**

The chest X-ray shows a hazy opacity in the right lower lung field. This could be consistent with pneumonia, but further evaluation is needed to confirm the diagnosis. Other possibilities include atelectasis or pulmonary edema.

Final Answer: The final answer is $\boxed{No}$

**P2_structured Output:**

1.  **Findings:**
    *   Lung fields are somewhat obscured by artifact.
    *   No obvious consolidation or pleural effusion is identified.
    *   Heart size appears normal.
    *   Mediastinal structures are unremarkable.
2.  **Impression:**
    *   No definite pneumonia is identified.

**P3_cautious Output:**

The image is difficult to interpret due to low resolution and noise. No obvious abnormalities are visible.

However, the image is not clear enough to rule out pneumonia.

Therefore, I cannot exclude pneumonia.


---
### Sample 8 (dataset idx=456)

- **Ground Truth:** 0 (0=Normal, 1=Pneumonia)

- **ViT Prediction:** 0 (p=0.50)

- **Image:** ![](task2_outputs_medgemma/images/img_08_idx456_gt0.png)

**P1_concise Output:**

Impression: The chest X-ray is difficult to interpret due to poor image quality. No definite consolidation, pleural effusion, or pneumothorax is identified. The heart size is normal. No obvious signs of pneumonia are present.

Conclusion: Findings are not consistent with pneumonia.


**P2_structured Output:**

Findings:
*   No clear consolidation is identified.
*   No pleural effusion is evident.
*   No pneumothorax is present.
*   The heart size appears normal.
*   Mediastinal structures are unremarkable.

Impression:
No acute findings suggestive of pneumonia.

**P3_cautious Output:**

The image is difficult to interpret due to the low resolution and lack of contrast. There is no clear evidence of consolidation or other specific findings to suggest pneumonia. However, the image is not diagnostic.

Final Answer: Cannot exclude pneumonia.

---
### Sample 9 (dataset idx=578)

- **Ground Truth:** 0 (0=Normal, 1=Pneumonia)

- **ViT Prediction:** 1 (p=0.50)

- **Image:** ![](task2_outputs_medgemma/images/img_09_idx578_gt0.png)

**P1_concise Output:**

The chest X-ray shows a clear lung field with no obvious consolidation, pleural effusion, or pneumothorax. The heart size appears normal. There are no findings suggestive of pneumonia.


**P2_structured Output:**

1.  **Findings:**
    *   Lung fields are clear.
    *   No pleural effusion.
    *   Heart size is normal.
    *   Mediastinal structures are unremarkable.
    *   No pneumothorax.
2.  **Impression:** No acute findings.

**P3_cautious Output:**

The image shows a chest X-ray. There is no obvious consolidation or pleural effusion. The heart size appears normal. The mediastinum is unremarkable. No pneumothorax is evident.

Based on the provided image, there are no clear abnormalities. However, a definitive diagnosis cannot be made without clinical context and comparison to prior imaging.


## Qualitative Analysis: VLM vs Ground Truth vs CNN

- We evaluated outputs on a mixed set of normal, pneumonia, misclassified, and uncertain cases.

- In many cases, MedGemma produced clinically styled impressions; structured prompting (P2) typically encouraged more organized responses.

- Disagreements often occurred on ambiguous/low-resolution cases (PneumoniaMNIST images are 28×28), where subtle opacities are difficult to confirm.

- For misclassified images, we compared whether MedGemma’s description aligned more closely with the ground truth or the ViT prediction.

## Strengths and Limitations

**Strengths**

- Medical-domain VLM, capable of radiology-like phrasing.

- Prompting significantly affects structure and caution of outputs.


**Limitations**

- PneumoniaMNIST is very low resolution; real clinical CXR are higher quality.

- VLM outputs are not guaranteed to be clinically correct and may hallucinate; should not be used for real diagnosis.

- Evaluation is qualitative; no ground-truth report text exists for scoring.
