## 一、Project Introduction
### This project is a deep learning-based rice disease recognition system that can quickly and accurately identify common rice diseases (such as Rice Blast, BrownSpot, Bacterial Leaf Blight, etc.). It helps with the early diagnosis and prevention of diseases in agricultural production, reduces farmers' losses, and improves planting efficiency.
## 二、Technology Stack
1. Programming Language: Python 3.8+​
2. Deep Learning Framework: PyTorch​
3. Image Processing: OpenCV, PIL
## 三、Prepare Dataset
<p><strong>English:</strong> The recommended dataset structure is as follows:</p>
<pre><code id="xxx" lang="plain text">Final_Dataset/
├── train/          # Training set
│   ├── 0_Blast/     # Rice Leaf Blast
│   ├── 1_BrownSpot/ # Brown Spot 
│   ├── 2_Blight/    # Bacterial Leaf Blight 
│   └── 3_Tungro/    # Tungro 
├── val/            # Validation set (suggested structure)
│   ├── 0_Blast/
│   ├── 1_BrownSpot/
│   ├── 2_Blight/
│   └── 3_Tungro/
└── test/           # Test set (suggested structure)
    ├── 0_Blast/
    ├── 1_BrownSpot/
    ├── 2_Blight/
    └── 3_Tungro/</code></pre>

