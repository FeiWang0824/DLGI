# DLGI
demo code for [Learning from simulation: An end-to-end deep-learning approach for computational ghost imaging](https://opg.optica.org/oe/fulltext.cfm?uri=oe-27-18-25560&id=417106).

## Citation
If you find this project useful, we would be grateful if you cite the **DLGI paper：**

Fei Wang, Hao Wang, Haichao Wang, Guowei Li, and Guohai Situ, Learning from simulation: An end-to-end deep-learning approach for computational ghost imaging. *Opt. Express* **27**, 25560-25572 (2019).

## Abstract
Artificial intelligence (AI) techniques such as deep learning (DL) for computational imaging usually require to experimentally collect a large set of labeled data to train a neural network. Here we demonstrate that a practically usable neural network for computational imaging can be trained by using simulation data. We take computational ghost imaging (CGI) as an example to demonstrate this method. We develop a one-step end-to-end neural network, trained with simulation data, to reconstruct two-dimensional images directly from experimentally acquired one-dimensional bucket signals, without the need of the sequence of illumination patterns. This is in particular useful for image transmission through quasi-static scattering media as little care is needed to take to simulate the scattering process when generating the training data. We believe that the concept of training using simulation data can be used in various DL-based solvers for general computational imaging.

## Overview
<img src='https://opg.optica.org/getimagev2.cfm?img=46zJdtApaV97%2B9ndrYMkWFLBoAafOUC8FhR2zAxpBpA%3D' width = '800'/>

## How to use
**Step 1: prepare required packages**

python 3.6

tensorflow 1.9

matplotlib 3.1.3

numpy 1.18.1

pillow 7.1.2

**Step 2: extract the data.zip file.**

**Step 3: run train.py to train the DNN using simulated data**

Our pretrained model is available at [DLGI_pretrained_model](https://drive.google.com/file/d/12DmpaUAVgL5srvdEyst4M0ZEYAwpF9Sw/view?usp=sharing).

**Step 4: run test.py to test on experimental data**

## Results
<img src='https://opg.optica.org/getimagev2.cfm?img=lpPlLTEXInXX6MLyDMvBFDVCRka7tl0PN9%2BqTuSQxwI%3D' width = '800'/>
## License
For academic and non-commercial use only.
