# **ML Models From Scratch**

In an era where trends like Neural Networks, NLP, and Retrieval-Augmented Generation (RAG) dominate, building a strong foundation in core machine learning concepts is more important than ever. This series is dedicated to creating some of the most common and widely used machine learning models **from scratch**. 

Each implementation is designed to be simple, clean, and easy to understand, making it ideal for learning and practical use. I hope you find it helpful! 😊

## **Models Implemented**
1. **Linear Regression**  
2. **Logistic Regression**  
3. **Decision Tree**
4. **Random Forest**
5. **K Nearest Neighbors**
6. **Support Vector Machine**
7. **KMeans Clustering**
8. **Transformer ( Encoder-Decoder )**

## **Installation and Setup**
### **Step 1: Install Required Libraries**
Before running the models, make sure to install the required dependencies.  
Run the following command:  
```bash
pip install -r requirements.txt
```

## **Usage**
### **Step 1: Navigate to the Model Directory**  
Each model is placed in its own directory. Navigate to the respective directory to execute the model.  
```bash
cd <model_directory_name>
```

**Example:**  
```bash
cd 01_Linear_Regression
```

### **Step 2: Run the Training Script**  
After navigating to the model directory, execute the `train.py` file to train the model.  
```bash
python train.py
```

### **Directory Structure**
```
ML_Models_From_Scratch/
│
├── 01_Linear_Regression/
│   ├── train.py
│   ├── results/
│   ├── src/
│   │   ├── linear_regression.py
│
├── 02_Logistic_Regression/
│   ├── train.py
│   ├── data/
│   ├── src/
│   │   ├── logistic_regression.py
│   │   ├── metrics.py
│
├── 03_Decision_Tree/
│   ├── train.py
│   ├── src/
│   │   ├── Node.py
│   │   ├── DecisionTree.py
│
├── 04_Random_Forest/
│   ├── train.py
│   ├── src/
│   │   ├── Node.py
│   │   ├── DecisionTree.py
│   │   ├── RandomForest.py
│
├── 05_KNN/
│   ├── train.py
│   ├── src/
│   │   ├── KNN.py
│
├── 06_SVM/
│   ├── train.py
│   ├── src/
│   │   ├── SVM.py
│
├── 07_KMeans_Clustering/
│   ├── train.py
│   ├── src/
│   │   ├── KMeans.py
│
├── 08_Transformer/
│   ├── src/
│   │   ├── decoder/
│   │   │   ├── decoder_block.py
│   │   │   ├── decoder.py
│   │   ├── encoder/
│   │   │   ├── encoder.py
│   │   │   ├── feed_forward_layer.py
│   │   │   ├── layer_normalization.py
│   │   │   ├── multi_headed_attention_block.py
│   │   ├── build_transformer_.py
│   │   ├── input_embedding.py
│   │   ├── positional_encoding.py
│   │   ├── projection_layer.py
│   │   ├── transformer.py
│ 
├── .gitignore
├── requirements.txt
└── README.md
```

## **Contact**
If you have any questions, suggestions, or feedback, feel free to reach out! 😊

---
