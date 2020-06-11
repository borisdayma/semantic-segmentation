# Semantic Segmentation

*Research on optimizing neural networks for semantic segmentation on self-driving cars*

## Introduction

Self-driving cars require a deep understanding of their surroundings. Camera images are used to recognize road, pedestrians, cars, sidewalks, etc at a pixel level accuracy. In this repository, we aim at defining a neural network and optimizing it to perform semantic segmentation.

The AI framework used is fast.ai and the dataset is from [Berkeley Deep Drive](https://bdd-data.berkeley.edu/). It is highly diverse and present labeled segmentation data from a diverse range of cars, in multiple cities and weather conditions.

Every single experiment is automatically logged onto [Weights & Biases](https://www.wandb.com/) for easier analysis/interpretation of results and how to optimize the architecture.

## Usage

Dependencies can be installed through `requirements.txt` or `Pipfile`.

The dataset needs to be downloaded from [Berkeley Deep Drive](https://bdd-data.berkeley.edu/).

The following files are present in `src` folder:

- `pre_process.py` must be run once on the dataset to make it more user friendly (segmentation masks with consecutive values) ;
- `prototype.ipynb` is a Jupyter Notebook used to prototype our solution ;
- `train.py` is a script to run several experiments and log them on [Weights & Biases](https://www.wandb.com/).

## Results

See my results and conclusions:

- [Results & Conclusions](https://www.wandb.com/articles/semantic-segmentation-for-self-driving-cars)
- [W&B report](https://app.wandb.ai/borisd13/semantic-segmentation/reports?view=borisd13%2FSemantic%20Segmentation%20Report)
- [W&B runs](https://app.wandb.ai/borisd13/semantic-segmentation/?workspace=user-borisd13)
