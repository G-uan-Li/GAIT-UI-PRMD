# gait analysis

#### 介绍
步态分析，Vicon全身骨骼数据，The codes in this repository are based on the eponymous research project <a href="https://arxiv.org/abs/1901.10435">A Deep Learning Framework for Assessing Physical Rehabilitation Exercises</a>. The proposed framework for automated quality assessment of physical rehabilitation exercises encompasses metrics for quantifying movement performance, scoring functions for mapping the performance metrics into numerical scores of movement quality, techniques for dimensionality reduction, and deep neural network models for regressing quality scores of input movements via supervised learning. 



#### 使用说明

#### 
# Neural Network Codes
The codes were developed using the Keras library.
* SpatioTemporalNN_Vicon - the proposed deep spatio-temporal model in the paper.
* CNN_Vicon - a basic convolutional neural network for predicting movement quality scores.
* RNN_Vicon - a basic recurrent neural network for predicting movement quality scores.
* Autoencoder_Dims_Reduction - a model for reducing the dimensionality of Vicon-captured movement data.
* SpatioTemporalNN_Kinect - implementation of the proposed deep learning model for predicting quality scores on Kinect-captured data.

# Distance Functions
The codes were developed using MATLAB.
* Maximum Variance - distance functions on reduced-dimensionality data using the maximum variance approach.
* PCA - distance functions on reduced-dimensionality data using PCA.
* Autoencoder - distance functions on reduced-dimensionality data using an autoencoder neural network.
* No Dimensionality Reduction - distance functions on full-body skeletal data (117 dimensions).


# Use
* Run "Prepare_Data_for_NN" to read the movements data, and perform pre-processing steps, such as length alignment and centering. Alternatively, skip this step, the outputs are saved in the Data folder (Data_Correct.csv and Data_Incorrect.csv).

* Run "Autoencoder_Dims_Reduction" to reduce the dimensionality of the movement data. Alternatively, skip this step, the outputs are saved in the Data folder (Autoencoder_Output_Correct.csv and Autoencoder_Output_Incorrect.csv).


* Run "Prepare_Labels_for_NN" to generate quality scores for the individual movement repetitions. Alternatively, skip this step, the outputs are saved in the Data folder (Labels_Correct.csv and Labels_Incorrect.csv)


* Run "SpatioTemporalNN_Vicon" to train the model and predict movement quality scores on the Vicon-captured movement data.


* Run "SpatioTemporalNN_Kinect" to train the model and predict movement quality scores on Kinect-captured movement data.
#### 参与贡献



#### 特技

1.  使用 Readme\_XXX.md 来支持不同的语言，例如 Readme\_en.md, Readme\_zh.md
2.  Gitee 官方博客 [blog.gitee.com](https://blog.gitee.com)
3.  你可以 [https://gitee.com/explore](https://gitee.com/explore) 这个地址来了解 Gitee 上的优秀开源项目
4.  [GVP](https://gitee.com/gvp) 全称是 Gitee 最有价值开源项目，是综合评定出的优秀开源项目
5.  Gitee 官方提供的使用手册 [https://gitee.com/help](https://gitee.com/help)
6.  Gitee 封面人物是一档用来展示 Gitee 会员风采的栏目 [https://gitee.com/gitee-stars/](https://gitee.com/gitee-stars/)
