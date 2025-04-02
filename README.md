**ProtoNet Inference with Precision–Recall Evaluation**

This repository implements a [Prototypical Network (ProtoNet)](https://proceedings.neurips.cc/paper_files/paper/2017/file/cb8da6767461f2812ae4290eac7cbc42-Paper.pdf) for few-shot classification with a focus on robust inference on unbalanced datasets. Unlike many standard approaches, this project emphasizes inference by computing support set prototypes and evaluating predictions using precision–recall metrics. This detailed evaluation is key to our approach when dealing with imbalanced classes.

**Overview**

The project consists of three main components:

- **Training Pipeline:**

  The training script (`protonet_tf2.py`) builds and trains a ProtoNet model using a CNN backbone (e.g., WideResNet). It employs episodic training with support and query sets to learn discriminative features for few-shot learning tasks.

- **Evaluation Utilities:**

  The utility script (`protonet_precision_recall_heatmap.py`) provides functions to compute class-wise precision, recall, F1-scores, and generate a heatmap of the confusion matrix. This is essential for evaluating model performance on unbalanced datasets.

- **Inference Pipeline:**

  The inference script (`protonet_inference.py`) demonstrates how to perform inference with the trained ProtoNet. In this stage:

  - **Support Set Processing:**  
    The support set is loaded from CSV files and processed to compute class prototypes by averaging feature embeddings.

  - **Query Prediction:**  
    Test images are processed in batches, and predictions are obtained by computing Euclidean distances between query embeddings and the support prototypes. Softmax over these distances generates class probabilities.

  - **Precision–Recall Evaluation:**  
    Predictions and associated probabilities are saved to CSV, and a precision–recall heatmap is generated to highlight performance on an imbalanced dataset.

  This inference strategy, which emphasizes precision and recall, differentiates this approach from other attempts.

**Repository Structure**

- **protonet_precision_recall_heatmap.py**  
  Contains the `generate_precision_recall_heatmap` function. This utility reads a CSV file with ground truth and predicted labels, computes per-class metrics using scikit-learn's `classification_report`, and generates a combined visualization with a metrics table and a confusion matrix heatmap.

- **protonet_tf2.py**  
  Implements the training pipeline. It loads the dataset, samples episodes for few-shot learning, and trains a ProtoNet model using a CNN backbone (such as WideResNet). Training involves computing Euclidean distances between query samples and class prototypes to derive loss and accuracy.

- **protonet_inference.py**  
  Implements the inference pipeline. Key steps include:

  - **Loading Support Data:**  
    Reading support images and their labels from a CSV file.

  - **Computing Prototypes:**  
    Generating feature embeddings using the trained model and computing class prototypes.

  - **Predicting on Test Data:**  
    Processing test images in batches, computing Euclidean distances to the prototypes, and converting these distances into log-softmax predictions.

  - **Precision–Recall Visualization:**  
    Storing prediction results in CSV and generating a precision–recall heatmap to evaluate performance.

  This inference approach focuses on precision–recall metrics to better handle unbalanced data distributions.

**Running Inference**

The inference procedure is what differentiates this project. To perform inference and evaluate precision–recall metrics:

- **Prepare Support and Test Data:**  
  Ensure you have CSV files that list the support set (for prototype computation) and test set (for inference) with image paths and labels.

- **Run the Inference Script:**  
  `protonet_inference.py`  
  This script will:

  - Load support images and compute class prototypes.
  - Process test images batch by batch to compute predictions based on Euclidean distance.
  - Save prediction results and maximum probabilities in a CSV file.
  - Generate and display a precision–recall heatmap to assess performance on an unbalanced dataset.

**Evaluating Results**

The precision–recall heatmap generated during inference provides a visual summary of class-wise precision, recall, and overall model performance. This is crucial when dealing with imbalanced datasets, as it offers insights beyond overall accuracy.

**How Inference Differs**

Unlike many traditional few-shot learning approaches that focus primarily on accuracy, this project emphasizes:

- **Support-Based Prototype Computation:**  
  By averaging feature embeddings of support images, the method derives robust prototypes for each class.

- **Distance-Based Prediction:**  
  Euclidean distances between query embeddings and class prototypes are computed and transformed into prediction probabilities.

- **Precision–Recall Evaluation:**  
  The final evaluation uses precision–recall metrics, providing a more comprehensive understanding of performance on imbalanced datasets, where accuracy might be misleading.
