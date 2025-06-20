# Ma412

GitHub repository containing code and the system description report (in `.pdf` format) for the **Ma412** project: *Multi-Label Classification of Scientific Literature Using the NASA SciX Corpus*.

The project aims to evaluate and compare different multi-label classification models on scientific articles. The main task involves assigning multiple relevant tags to each scientific abstract using traditional machine learning models, neural networks, and transformer-based architectures like SciBERT.

---

## Project Dependencies

To run this project, ensure you have the following Python packages installed:

pandas: for data manipulation

matplotlib and seaborn: for data visualization

nltk: for natural language preprocessing

scikit-learn: for classical machine learning models and utilities

scikit-multilearn: for multi-label classification strategies like Binary Relevance, Classifier Chain, and Label Powerset

wordcloud: for visualizing word distributions

PyTorch: for deep learning models

transformers (Hugging Face): for using pre-trained models like SciBERT

tqdm: for progress bars

You can install all dependencies using:

bash
pip install -r requirements.txt

---

## Models and Methodology

This project explores various approaches to multi-label classification of scientific articles. Below is a brief overview of the implemented models and techniques:

### 1. Binary Relevance
Each label is treated as an independent binary classification problem. While simple and efficient, it does not consider label dependencies.

### 2. Classifier Chain
An improvement over Binary Relevance that models label dependencies by chaining classifiers. The prediction of one label is used as input for the next.

### 3. Label Powerset
This method transforms the problem into a multi-class classification by treating each unique label combination as a single class. It captures label correlations but may struggle with rare combinations.

### 4. Label Powerset + Random Forest
A traditional ensemble method (Random Forest) is used as the base classifier for the Label Powerset approach. It performs well with limited data but can struggle with high label cardinality.

### 5. Label Powerset + MLP
A multi-layer perceptron is used within the Label Powerset strategy to capture non-linear relationships. This model can be sensitive to hyperparameters and requires longer training times.

### 6. SciBERT (Transformers)
A pretrained BERT-based language model trained on scientific text (SciBERT) is fine-tuned for multi-label classification. It leverages contextual embeddings and outperforms traditional models in low-data or high-complexity settings.

---
## How to Run the Project

Clone the repository:

git clone https://github.com/AuraXenonMARL/Ma412.git

cd Ma412

Open the .ipynb (Jupyter Notebook) file included in the repository using Jupyter Notebook or JupyterLab.

Follow the instructions within the notebook. You simply need to:

Run each cell in sequential order.

Follow any prompts or explanations embedded in the markdown cells.

Choose the model configurations or adjust parameters as needed.

Review performance metrics printed or visualized at the end.

The notebook guides you step-by-step through:

---
## Notebook Structure

The Jupyter Notebook is organized as follows:

1. **Loading and Cleaning the Data**  
   - Handling missing text entries  
   - Removing articles with extremely few or many labels  

2. **Exploratory Data Analysis**  
   - Distribution of labels  
   - Label co-occurrence  
   - Text length statistics  

3. **Model Training and Evaluation**  
   - Each model is trained on a subset of the 50 most frequent labels  
   - Evaluation metrics include **Subset Accuracy**, **Hamming Loss**, and **F1 Score (micro average)**  
 

5. **Result Interpretation and Comparison**  
   - Final section compares the performance of all models side by side  
   - Conclusions and potential directions for improvement are discussed

---

##  Tips for Experimentation

- Due to computational constraints, running experiments on **50 or more labels** may take **over an hour**. It is recommended to **start with fewer labels** when testing models.
- SciBERT requires a GPU to run efficiently. For local execution, reduce dataset size or run on Google Colab with a GPU runtime.
- You can try other classifiers (e.g., XGBoost, SVC, etc.) by changing the `clf` in the pipeline for further experimentation.




