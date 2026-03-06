# Methodology: Sandwich Architecture for Lesion Segmentation

## 1. Proposed Architecture (The "Sandwich" Model)

To achieve precise and robust breast lesion segmentation, we propose a novel three-stage cascaded framework termed the "Sandwich Architecture". This design leverages the domain-specific localization capabilities of a standard convolutional network and the zero-shot boundary delineation power of a visual foundation model (SAM 3). 

The pipeline conceptually "sandwiches" a frozen foundation model between two U-Net architectures: a frozen Teacher U-Net for prompt generation (Stage 1), the frozen SAM 3 for boundary refinement (Stage 2), and a trainable Student U-Net for final feature fusion (Stage 3). 

### Stage 1: Coarse Localization (Teacher U-Net)
In the first stage, given an input ultrasound image $I \in \mathbb{R}^{3 \times H \times W}$, a pre-trained and frozen U-Net (Teacher) is utilized to generate a coarse segmentation probability map. We binarize this probability map and extract the minimal bounding box $B$ that encloses the predicted foreground region. This stage mimics human visual attention by providing a highly reliable, albeit coarse, Region of Interest (ROI) prior.

### Stage 2: Foundation Model Refinement (SAM 3)
In the second stage, we incorporate the Segment Anything Model 3 (SAM 3) as a frozen feature extractor and mask generator. SAM 3 is highly adept at capturing sharp object boundaries but often lacks domain-specific semantic understanding (e.g., distinguishing a true lesion from surrounding benign anatomical structures). We feed the original image $I$ and the bounding box prompt $B$ generated from Stage 1 into SAM 3. The foundation model outputs a refined, high-resolution mask $M_{SAM} \in \mathbb{R}^{1 \times H \times W}$ that tightly adheres to the lesion's physical boundaries.

### Stage 3: Feature Fusion and Final Prediction (Student U-Net)
While $M_{SAM}$ provides excellent boundary details, it may contain false positives (due to SAM's class-agnostic nature) or artifacts. To correct these and adapt the prediction strictly to the breast ultrasound domain, we introduce a final trainable U-Net (Student). We concatenate the original RGB image $I$ with the SAM-generated mask $M_{SAM}$ along the channel dimension to form a 4-channel input tensor $I_{fusion} \in \mathbb{R}^{4 \times H \times W}$. This U-Net acts as a final refiner, learning to fuse the original textural features with the structural prior provided by SAM 3, outputting the final precise lesion segmentation mask $M_{final}$.

## 2. Training Strategy and Optimization

A naive end-to-end training approach for the entire cascaded pipeline would be computationally prohibitive due to the heavy Transformer blocks within SAM 3. To address this, we employ a decoupled training strategy:

1. **Gradient Truncation and Freezing:** The parameters of the Stage 1 Teacher U-Net and the Stage 2 SAM 3 are strictly frozen. Furthermore, the non-differentiable bounding box extraction operation naturally truncates the gradient flow.
2. **Offline Mask Precomputation:** Because Stage 1 and Stage 2 are deterministic and frozen, the intermediate representation $M_{SAM}$ for any given training image is constant across all training epochs. We precompute and cache $M_{SAM}$ for the entire dataset offline. 
3. **Efficient U-Net Training:** During the actual training phase, the optimization objective is exclusively applied to the Stage 3 Student U-Net. The model fetches the cached $M_{SAM}$ and the original image $I$ to directly train the 4-channel network. 

This optimization strategy reduces the training complexity from a heavy multi-model pipeline to evaluating a single U-Net, accelerating the training process by an order of magnitude while maintaining the accuracy benefits derived from the foundation model. The network is optimized using a combination of Dice Loss and Binary Cross-Entropy (BCE) Loss calculated between $M_{final}$ and the ground truth annotations.
