# Medical Image Segmentation

Contents
* Overview of project
* List of tutorials

### Project Overview

This project aims to explore the following capabilities in an MLOps platform:
* using terraform to create infrastructure
* exploring large models (using PyTorch, and some foundational models)
* creating a model selection pipeline for above models
* exploring experiment tracking services, both within AWS and third party options

### List of Tutorials
To aid in this project, several new softwares were explored. While learning about these, I discovered several tutorials that I found to be helpful. The tutorials I used are listed below:

* [terraform](https://www.youtube.com/playlist?list=PL5_Rrj9tYQAlgX9bTzlTN0WzU67ZeoSi_): basics of how to set up a terraform project
* [AWS Experiments](https://sagemaker-examples.readthedocs.io/en/latest/sagemaker-experiments/local_experiment_tracking/pytorch_experiment.html#Run-first-experiment): Reviews how to use AWS Sagemaker experiments using the MNIST dataset
* [Weights & Biases on AWS](https://wandb.ai/wandb/sm-pytorch-mnist-new/reports/Using-AWS-Sagemaker-and-Weights-Biases-Together-on-Digit-Recognition-with-MNIST---Vmlldzo4MTk3Nzg): Reviews how to use weights and biases on AWS using the MNIST dataset

### Fun Notes/Things that might be helpful to know
* Connecting AWS to VSCode: this can be found in the terraform tutorial linked above. It allows you to access AWS resources locally (so no more Sagemaker, unless you need the compute!)
* Terraform is great for easy setup and takedown...unless you need Sagemaker. Then it doesn't quite work (there are some dependencies it can't find and delete, and therefore does not delete all the resources)