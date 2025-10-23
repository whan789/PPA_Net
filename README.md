# PPA_Net
PPA-Net : Patch Pooling with Spectral Knowledge Distillation for Time Series Forecasting  
***[Whan Choi](https://github.com/whan789)**, ***[Kunwoo Kang](https://github.com/kunwookang)**, **[Shinwoong Kim](https://github.com/ALFEE19971029)**, **Sangyup Lee†**  
\* Equal Contribution, † Corresponding author  

This is the official repository for the paper "*PPA-Net : Patch Pooling with Spectral Knowledge Distillation for Time Series Forecasting*".

## Abstract
Recent advances in time-series forecasting have explored paradigms such as frequency- and time-domain represen- tations, yet they often consider these in isolation, limiting their ability to capture multi-scale dynamics. To address this, we propose PPA-Net, a novel hybrid forecasting model that leverages intra-model knowledge distillation to transfer stable global patterns from a frequency-domain branch to a patch-based branch. The core of our patch branch is the Patch Pooling Aggregator (PPA), which adaptively integrates general dynamics and critical local events into a single, ex- pressive summary token. This summary, processed alongside individual patch embeddings, enables robust modeling of both local and global features. Evaluations on seven bench- mark datasets demonstrate that PPA-Net achieves SOTA results on several datasets and consistently ranks within the top two across all, showing strong generalization ability and efficiency for long-term forecasting.

## Overall Architecture
PPA-Net leverages patch-wise summarization and spectral knowledge distillation to capture both local and global temporal dependencies.
It introduces a Patch Pooling Aggregator (PPA) that adaptively fuses contextual and extreme-value descriptors, and employs a frequency-domain teacher branch to guide the patch branch through intra-model distillation, achieving efficient and robust long-term forecasting.

<p align="center">
  <img src="img/Overall.png" width="700"/>
</p>

## Result of Experiment
We evaluate PPA-Net on several multivariate forecasting benchmarks.
PPA-Net achieves *consistently strong performance* across all datasets in terms of both MSE and MAE. (Best results are highlighted in **red**, and second-best in **blue**.)

<img src="img/experiment_result.png" width="800"/>

## Acknowledgement  

We appreciate the following Github repos a lot for their valuable code and efforts.
* TimeXer (https://github.com/thuml/TimeXer.git)
* iTransformer (https://github.com/thuml/iTransformer.git)
* RLinear (https://github.com/plumprc/RTSF.git)
* PatchTST (https://github.com/yuqinie98/PatchTST.git)
* DLinear (https://github.com/cure-lab/LTSF-Linear.git)
* FEDformer (https://github.com/MAZiqing/FEDformer.git)
* Autoformer (https://github.com/thuml/Autoformer.git)
* Time-Series-Library (https://github.com/thuml/Time-Series-Library.git)

## Contact
If you have any questions or want to use the code, feel free to contact:

* Whan Choi (whan789@yonsei.ac.kr)
* Kunwoo Kang (kunwoo2027@yonsei.ac.kr)
* Shinwoong Kim (kshw19971029@gmail.com)
