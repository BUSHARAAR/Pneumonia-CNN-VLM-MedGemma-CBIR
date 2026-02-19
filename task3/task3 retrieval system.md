# Task 3: Semantic Image Retrieval System (CBIR)

## Objective
Build a content-based image retrieval (CBIR) system for chest X-ray images using medical image embeddings and a vector search index.

## Embedding Model Selection and Justification
- **Embedding backbone:** CLIPModel medical variant (MedCLIP/BiomedCLIP-style).
- **Justification:** CLIP-style medical encoders map images and clinical text into a shared embedding space, enabling both:
  - **Image→Image** similarity retrieval
  - **Text→Image** retrieval using natural language queries

All embeddings were L2-normalized to support cosine similarity search.

## Vector Database Implementation
- **Library:** FAISS
- **Index:** `IndexFlatIP` (Inner Product)
- **Similarity:** Cosine similarity (via normalized vectors)

Artifacts stored:
- `task3_outputs/embeddings.npy`
- `task3_outputs/labels.npy`
- `task3_outputs/faiss_index.bin`
- `task3_outputs/retrieval_figures/`

## Retrieval Interfaces
### 1) Image-to-Image Search
Given a query image index, retrieve top-k most similar images using FAISS nearest-neighbor search.

### 2) Text-to-Image Search
Given a clinical text prompt (e.g., “pneumonia opacity in lungs”), embed the text and retrieve the most relevant images from the index.

## Quantitative Evaluation (Precision@k)
Precision@k is computed as the fraction of retrieved top-k images sharing the same label as the query image (excluding the query image itself).

- Precision@1  = 0.8253
- Precision@3  = 0.8344
- Precision@5  = 0.8192
- Precision@10 = 0.8035

## Visualization of Retrieval Results
Saved examples are available under:
`task3_outputs/retrieval_figures/`

Figures include:
- Query image + top-k similar images (image→image)
- Text prompt + top-k retrieved images (text→image)

## Discussion and Failure Cases
- Retrieval quality is generally strong when images have clear patterns corresponding to normal vs pneumonia.
- Failure cases can occur on ambiguous low-resolution images (PneumoniaMNIST is 28×28), where pneumonia cues may be subtle.
- Text-to-image results are prompt-sensitive: more specific prompts (“opacity”, “infiltrate”) may retrieve pneumonia-like cases more consistently than broad prompts.

## Strengths and Limitations
**Strengths**
- Supports both image and text retrieval in a single shared embedding space.
- FAISS provides efficient and reproducible nearest-neighbor search.

**Limitations**
- Dataset resolution limits fine radiological detail; retrieval may group images based on coarse patterns.
- Precision@k uses binary labels only; it does not reflect deeper clinical similarity beyond the label.
