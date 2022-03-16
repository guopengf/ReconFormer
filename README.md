# ReconFormer
ReconFormer: Accelerated MRI Reconstruction Using Recurrent Transformer

Pytorch Code for the paper ["ReconFormer: Accelerated MRI Reconstruction Using Recurrent Transformer"](https://arxiv.org/abs/2201.09376)

**Updates**:

:rocket: : We release training and testing code \
:rocket: : We release pre-trained weights for fastMRI 

# Requirements

python=3.6  
pytorch=1.7.0

Please refer conda_environment.yml for more dependencies.

# Inroduction

Accelerating magnetic resonance imaging (MRI) reconstruction process is a challenging ill-posed inverse problem due to the excessive under-sampling operation in k-space.
In this paper, we propose a recurrent transformer model, namely ReconFormer, for MRI reconstruction which can iteratively reconstruct high fidelity magnetic resonance images from highly under-sampled k-space data. In particular, the proposed architecture is built upon Recurrent Pyramid Transformer Layers (RPTLs), which jointly exploits intrinsic multi-scale information at every architecture unit as well as the dependencies of the deep feature correlation through recurrent states. Moreover, the proposed ReconFormer is lightweight since it employs the recurrent structure for its parameter efficiency.

## Dataset Preparation

Prepare the dataset in the following structure for easy use of the code.The provided data loaders is ready for this this format and you may change it as your need.

```bash


                   |-- 
                   |                       
                   |                |--xxx.h5  
Dataset Folder-----|      |--train--|...
                   |      |         |...
                   |      |                  
                   |      |         
                   |--PD -|
                          |
                          |         |--xxx.h5 
                          |-- val --|...  
                                    |...
 ```

## Links for downloading the public datasets:

1) fastMRI Dataset - <a href="https://fastmri.med.nyu.edu/"> Link </a>
2) HPKS Dataset - We don't obtain the permission from Johns Hopkins Hospital to release this dataset. This dataset is available upon request which will be evaluated on a case-by-case basis.

Preprocessed fastMRI (OneDrive) - <a href="https://livejohnshopkins-my.sharepoint.com/:f:/g/personal/pguo4_jh_edu/EtXsMeyrJB1Pn-JOjM_UqhUBdY1KPrvs-PwF2fW7gERKIA?e=uuBINy"> Link </a>\
Password: pguo4@jhu.edu\
**Note:** In preprocessed fastMRI, We didn't modify the original fastMRI data and just make the format compatible with our DataLoader. 

# Run

## Clone this repo
```bash 
git clone git@github.com:guopengf/ReconFormer.git
```

## Set up conda environment
```bash
cd ReconFormer
conda env create -f conda_environment.yml
conda activate recon
```
## Train ReconFormer
```bash 
bash run_recon_exp.sh
```

## Monitor the traning process
```bash 
tensorboard --logdir 'Dir path for saving checkpoints'
```
## Test (Download [pre-trained weights](https://livejohnshopkins-my.sharepoint.com/:f:/g/personal/pguo4_jh_edu/Er37oIyNy3NBrXbeCQBp_fQBAxELR8UDaq6gHd-fjwRrSw) Password: pguo4@jhu.edu)
```bash 
bash run_recon_eval.sh
```
## Ackonwledgements

We give acknowledgements to [fastMRI](https://github.com/facebookresearch/fastMRI), [Swin-Transformer
](https://github.com/microsoft/Swin-Transformer), and [SwinIR](https://github.com/JingyunLiang/SwinIR).


# Citation
```bash
@article{guo2022reconformer,
  title={ReconFormer: Accelerated MRI Reconstruction Using Recurrent Transformer},
  author={Guo, Pengfei and Mei, Yiqun and Zhou, Jinyuan and Jiang, Shanshan and Patel, Vishal M},
  journal={arXiv preprint arXiv:2201.09376},
  year={2022}
}
```
