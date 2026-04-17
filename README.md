<p><strong>一、Project Introduction</strong></p>
<p>This project is a deep learning-based rice disease recognition system that can quickly and accurately identify common rice diseases (such as Rice Blast, BrownSpot, Bacterial Leaf Blight, etc.). It helps with the early diagnosis and prevention of diseases in agricultural production, reduces farmers' losses, and improves planting efficiency.</p>
<p><strong>二、Technology Stack</strong></p>
<p>1. Programming Language: Python 3.8+</p>​
<p>2. Deep Learning Framework: PyTorch、pandas</p>​
<p>3. Image Processing: OpenCV</p>
<p><strong>三、Prepare Dataset</strong></p>
<p>The recommended dataset structure is as follows:</p>
<pre><code id="xxx" lang="plain text">1.Final_Dataset/
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
<p>2.test_noise</p>
<p><strong>四、Run the Project</strong></p>
<p>1.train_multitask.py</p>
<p>2.test_with_vcc.py</p>
