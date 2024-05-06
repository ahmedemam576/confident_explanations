to cite this work://
@ARTICLE{10433174,
  author={Emam, Ahmed and Farag, Mohamed and Roscher, Ribana},
  journal={IEEE Geoscience and Remote Sensing Letters}, 
  title={Confident Naturalness Explanation (CNE): A Framework to Explain and Assess Patterns Forming Naturalness}, 
  year={2024},
  volume={21},
  number={},
  pages={1-5},
  keywords={Uncertainty;Measurement;Predictive models;Data models;Satellite images;Logistic regression;Land surface;Explainable machine learning (ML);naturalness index;pattern recognition;remote sensing;uncertainty quantification},
  doi={10.1109/LGRS.2024.3365196}}



# Confident Naturalness Explanation (CNE): A Framework to Explain and Assess Patterns Forming Naturalness in Fennoscandia with Confidence 
Protected natural areas represent regions nearly unaffected by human intervention. These regions, unspoiled by urbanization, agriculture, and other anthropogenic activities, host remarkable biodiversity and ecological processes. To fully comprehend these unaltered natural ecosystem processes, including water cycles and pollination dynamics, thorough research efforts become necessary. Machine learning models are utilized to analyze these protected natural areas from satellite imagery. To gain insights into the discriminative patterns underlying the notion of naturalness within these protected environments, the utilization of explainability techniques becomes essential. Moreover, addressing the inherent uncertainty embedded within machine learning models is imperative to achieve a comprehensive understanding of the concept of naturalness.
 Existing approaches aimed at explaining the concept of naturalness in these protected natural areas often encounter limitations. They either fall short in delivering explanations that are both complete and valid, or they struggle to offer a quantitative metric that accurately characterizes the confident significance of specific patterns that contribute to the concept of naturalness.
This paper proposes a novel approach that integrates explainable machine learning with uncertainty quantification to explain naturalness. A novel quantitative metric is designed to describe the pattern's confident contribution to the concept of naturalness. Furthermore, an uncertainty-aware segmentation mask is generated for each input sample, highlighting the regions where the model illustrates a lack of knowledge. The confident Naturalness Explanation (CNE) framework is a novel way to understand naturalness by providing complete, valid, objective, quantifiable explanations of the concept of naturalness in Fennoscandia.

## Features

- UNet architecture for image segmentation.
- Monte Carlo Dropout for uncertainty quantification.
- Data preprocessing and transformation pipelines.
- Training settings and hyperparameters.
- Evaluation of uncertainty at the class level.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- PIL (Pillow)
- pandas
- numpy
- OpenCV (cv2)
- tifffile
- tqdm

## Setup

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/yourusername/your-repo.git
## Usage

    Prepare your dataset and update the paths in the code to point to your data and labels.

    Set your desired hyperparameters and training settings in the code.

    Train the UNet model using the provided functions.

    Use Monte Carlo Dropout to assess uncertainty by running the mc_dropout_all_batches function.

    View uncertainty at the class level using the class_std function.

    Adjust and fine-tune the code for your specific project needs.
