# Approach 
## 1. Dataset Collection
- Went online to find a few wound datasets and came across the [Kaggle Wound Dataset](https://www.kaggle.com/datasets/yasinpratomo/wound-dataset?resource=download)
- The dataset included different types of wounds but I will only focus on the following wounds as these are the only types of wounds that require band aid in general
    - Abrasions
    - Cuts
- Created a quick python script and some manual eyeballing to acquire arm wound images and I have several cases including edge cases that I observe below and I aim to make my technique versatile:
    - Multiple wounds
    - Huge wounds
    - Small wounds
    - Abrasion
    - Cuts
    - Finger wound 
    - Images with background

## 2. Approach 
The goal is to create something fast and lightweight hence I begin by exploring traditional CV methods followed my Gaussian Mixture Model (gmm) and then [Wound-segmentation](https://github.com/uwm-bigdata/wound-segmentation) which I found online since the first 2 methods did not work that well. 

### 2.1 Simple CV
- skin mask in HSV+YCrCb to PCA on skin to get arm angle to wound = red (Lab a-channel outliers) or dark (HSV V outliers) inside skin mask to largest contour gives placement/size. Feel free to refer to [cv output](./cv_output.ipynb) for the images
- Analysis: unable to perform well on simple testcases 
![CV Approach](./outputs/cv/abrasions(10)_compare_cv.png)

### 2.2 Gaussian Mixture Model
- Lightweight way to separate “wound‑like” pixels from normal skin without training data
- Analysis: performance was good for trivial testcases but does not work well for multiple wounds or tiny wounds
![CV Approach](./outputs/gmm/abrasions(10)_compare_gmm.png)

### 2.3 Wound-segmentation (Deep learning github repo)
- Found this repository that is useful for our usecase while doing some literature review and research
- Analysis: performance was slightly better for wounds that are present in the dataset, so we will have to train the models on all the new data so that it will perform well, overall this method has the most potential
![CV Approach](./outputs/abrasions(10)_compare_woundseg.png)

## 3. References
- [Kaggle Wound Dataset](https://www.kaggle.com/datasets/yasinpratomo/wound-dataset?resource=download)
- [Wound-segmentation](https://github.com/uwm-bigdata/wound-segmentation)
- [Codex for some code edits](https://openai.com/codex/)

## 4. Appendix 
I understand that the word limit is 150-300 hence I added the details in the Appendix, feel free to read through it if you want more information. 

### 4.1 Simple CV
I used a lightweight ML-based segmentation step so it runs fast on a laptop without heavy models. First, I isolate the arm region with a skin mask that combines HSV and YCrCb thresholds, then keep the largest connected skin component. To estimate arm direction, I run PCA on the skin mask pixels and use the dominant eigenvector as the arm axis, which provides a stable rotation angle even with slight arm tilt.

For wound detection, I treat it as an unsupervised clustering problem within the skin area. I extract per-pixel features (Lab a/b channels for redness, HSV V for darkness, and a Laplacian texture magnitude for abrasions/cuts). I normalize these features and fit a small Gaussian Mixture Model (3–4 components). Components are scored by redness, darkness, and texture; one or two high-scoring components are selected as “wound-like.” After light morphology cleanup, I keep up to three connected components to support multiple wounds of different sizes.

The band-aid is generated procedurally (rounded rectangle with a lighter pad) to avoid external assets. Each wound component gets a scaled band-aid (with slight style variants), rotated to match the arm direction, and alpha-blended onto the original image. The script saves both the patched image and a side-by-side comparison, and can display results via matplotlib. A segmentation-based option using FastSAM is also available for more robust mask proposals; it ranks model masks by redness/texture before placing bandages.
