# Incorporating Explainability to Image Captioning Models: An Improved Post-hoc Approach through Grad-CAM (IPAG)


IPAG an improved version of the XAI
method proposed by [Modafar Al-Shouha et. al.](https://ieeexplore.ieee.org/abstract/document/10158563), integrating a state-of-the-art captioning model and fine-tuning their pipeline
incorporating the Grad-CAM approach. Apart from providing accurate segmentation maps of the queries, our method provide additional saliency maps for better visual explanations.

## Acknowledgement

This project makes use of the following code and models:  
[IPICXAI](https://github.com/modafarshouha/PIC-XAI)  
[CLIP Explainability](https://github.com/sMamooler/CLIP_Explainability)  
[Detectron2](https://https//github.com/facebookresearch/detectron2) for image segmentation  
[BLIP](https://huggingface.co/Salesforce/blip-image-captioning-base) for image captioning  
[CLIP](https://github.com/openai/CLIP) for calculating similarity  

## Usage

To set up the environment and run the Jupyter Notebook, follow these steps:

### 1. Install Conda

Ensure that Conda is installed on your system. If not, download and install it from the [official Anaconda website](https://www.anaconda.com/products/distribution) or use [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for a minimal installation.

### 2. Create a Conda Environment

Create a Conda environment using the `requirements.yaml` file provided in the repository. Open a terminal or command prompt, navigate to the project directory, and run the following command:

```bash
conda env create -f requirements.yaml
```
### 3.Run the Jupyter Notebook
Launch Jupyter Notebook in the newly created environment and run ```IPAG.ipynb``` file

## Result
![Slide2](https://github.com/user-attachments/assets/ded07f8e-21fa-4567-a622-ac9cf5057562)
![Slide3](https://github.com/user-attachments/assets/e7d81a65-ac34-4346-943b-f2f84f645e43)


## Authors/Contributors

- Ashwant Ram A S - [GitHub Profile](https://github.com/Ashwanth-Ram)
- Harsh Kumar - [GitHub Profile](https://github.com/)
- Kamalnath M S - [GitHub Profile](https://github.com/KamalMS03)
- Dr. Jiji C V
