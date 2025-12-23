# Approach 
## 1. Dataset Collection
- Went online to find a few wound datasets and came across the [Kaggle Wound Dataset](https://www.kaggle.com/datasets/yasinpratomo/wound-dataset?resource=download)
- The dataset included different types of wounds but I will only focus on the following wounds as these are the only types of wounds that require band aid in general
    - Abrasions
    - Cuts
- Created a quick python script to acquire arm wound images and I have several cases that I observe below:
    - Multiple wounds
    - Huge wounds
    - Small wounds
    - Abrasion
    - Cuts
    - Finger wound 
    - Images with background

## 2. Approach 
### 2.1 Simple CV
- skin mask in HSV+YCrCb → PCA on skin to get arm angle → wound = red (Lab a-channel outliers) or dark (HSV V outliers) inside skin mask → largest contour gives placement/size.
- Analysis: unable to perform well on simple testcases
![CV Approach](./outputs/abrasions(10)_compare.png)

I approached the task with a lightweight, heuristic pipeline so it can run quickly without heavy ML models. First, I isolate the arm region using a combined skin mask in HSV and YCrCb color spaces. I then keep the largest connected skin region to approximate the arm. To estimate the arm’s direction, I run PCA on the skin mask pixels and take the dominant eigenvector as the arm axis, which gives a stable rotation angle even when the arm is slightly tilted.

For wound detection, I use two complementary cues within the skin mask: redness in the Lab a-channel (higher values indicate more red) and local darkness in HSV V-channel (wounds can be both red and dark). I normalize both channels by their mean and standard deviation inside the skin region, then threshold for red outliers or dark outliers. After light morphological cleanup, I select the largest wound-like contour. Its bounding box provides the center point and scale for the band-aid.

The band-aid itself is generated procedurally (rounded rectangle with a lighter pad) to avoid external assets. It is scaled based on wound size and rotated to match the arm direction, then alpha-blended onto the original image. The script saves both the patched image and a side-by-side comparison, and optionally displays the result using matplotlib.

# References
- [Kaggle Wound Dataset](https://www.kaggle.com/datasets/yasinpratomo/wound-dataset?resource=download)
