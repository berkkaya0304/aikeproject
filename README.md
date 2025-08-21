# Cardiotocography ML Analysis Project

## Overview

This project is a comprehensive machine learning analysis of cardiotocography (CTG) data, implemented as a Streamlit web application. The project demonstrates various machine learning techniques for fetal heart rate classification using real medical data.

## 🎯 Project Description

The project focuses on analyzing cardiotocography data to classify fetal heart rate patterns into different categories. It serves as a practical implementation of machine learning concepts including data preprocessing, model selection, hyperparameter tuning, and overfitting prevention.

## 📊 Dataset Information

- **Dataset**: `cardiotocography_v2.csv`
- **Source**: UCI Machine Learning Repository (Campos & Bernardes, 2000)
- **Shape**: 2,126 samples × 22 features
- **Target Variable**: `CLASS` (10 different classes)
- **Features**: 21 numerical features related to fetal heart rate monitoring

### Feature Categories
- **LB, AC, FM, UC, DL, DS, DP**: Various fetal heart rate measurements
- **ASTV, MSTV, ALTV, MLTV**: Short and long-term variability measures
- **Width, Min, Max, Nmax, Nzeros**: Statistical measures of heart rate patterns
- **Mode, Mean, Median, Variance, Tendency**: Central tendency and dispersion measures

## 🚀 Features

### 1. Interactive Web Interface
- **Streamlit-based UI** with sidebar controls
- **Real-time data upload** and processing
- **Dynamic parameter adjustment** for all models

### 2. Data Preprocessing Options
- **Missing Value Handling**: Median imputation strategy
- **Data Scaling**: Standardization and normalization
- **Dimensionality Reduction**: PCA with configurable components
- **Data Splitting**: Configurable train/validation split ratios

### 3. Machine Learning Models
- **Naive Bayes**: Gaussian implementation with var_smoothing tuning
- **Decision Tree**: Configurable depth and leaf constraints
- **Random Forest**: Ensemble method with hyperparameter controls
- **Support Vector Machine**: Multiple kernel options with C and gamma tuning

### 4. Model Evaluation
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score
- **Confusion Matrix Visualization**: Interactive heatmaps
- **Feature Importance Analysis**: For applicable models
- **Overfitting Detection**: Training vs validation performance comparison

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.7+
- pip package manager

### Installation Steps
```bash
# Clone the repository
git clone <repository-url>
cd aikeproject

# Install required packages
pip install -r requirements.txt

# Run the application
streamlit run lab4.py
```

### Required Dependencies
- streamlit
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## 📁 Project Structure
fetal-heart-rate-classifier/
├── lab4.py # Main Streamlit application
├── cardiotocography_v2.csv # Dataset file
├── ml_report_data.json # Detailed ML results and metrics
├── auto_generated_report.txt # Comprehensive analysis report
└── README.md # This file


## �� Usage Instructions

### 1. Launch the Application
```bash
streamlit run lab4.py
```

### 2. Upload Data
- Use the sidebar to upload `cardiotocography_v2.csv`
- The application will automatically load and display the dataset

### 3. Configure Parameters
- **Data Split**: Adjust validation set size (10%-40%)
- **Random State**: Set seed for reproducible results
- **Preprocessing**: Choose between Raw, Standardization, Normalization, or PCA
- **Model Selection**: Pick from available classifiers
- **Hyperparameters**: Tune model-specific parameters

### 4. Train and Evaluate
- Click "Train and Evaluate Model" button
- View performance metrics and visualizations
- Analyze confusion matrices and feature importances

## 📈 Key Results & Findings

### Best Performing Models
1. **Random Forest** (Standardized data): 83.1% accuracy
2. **Decision Tree** (Standardized data): 77.7% accuracy
3. **Naive Bayes** (Standardized data): 67.6% accuracy

### Data Preprocessing Impact
- **Standardization** consistently improves performance across all models
- **PCA** reduces dimensionality while maintaining reasonable accuracy
- **Raw data** performs poorly without proper scaling

### Overfitting Analysis
- Deep decision trees show significant overfitting (99.9% training vs 76.1% validation)
- Pruned trees demonstrate better generalization (77.2% training vs 74.4% validation)

## 🎓 Educational Value

This project demonstrates:
- **Data Mining**: Exploratory data analysis and feature understanding
- **Data Preparation**: Missing value handling and preprocessing techniques
- **Model Selection**: Comparison of different classification algorithms
- **Hyperparameter Tuning**: Impact of parameter selection on performance
- **Overfitting Prevention**: Techniques to improve model generalization
- **Evaluation Metrics**: Comprehensive model assessment

## 🔬 Technical Implementation

### Data Processing Pipeline
1. **Data Loading**: CSV file upload and validation
2. **Missing Value Handling**: Median imputation strategy
3. **Feature Engineering**: Statistical transformations and scaling
4. **Dimensionality Reduction**: PCA with variance retention analysis
5. **Model Training**: Configurable hyperparameter optimization
6. **Performance Evaluation**: Multi-metric assessment

### Model Architecture
- **Modular Design**: Easy addition of new algorithms
- **Parameter Validation**: Input sanitization and range checking
- **Memory Efficient**: Optimized data handling for large datasets
- **Reproducible Results**: Configurable random seeds

## 📚 References

- **Dataset**: Campos, D., & Bernardes, J. (2000). Cardiotocography. UCI Machine Learning Repository
- **Libraries**: scikit-learn, pandas, numpy, matplotlib, seaborn, streamlit
- **Course**: Artificial Intelligence and Knowledge Engineering - Lab 4

## 👥 Contributors

- Katarzyna Fojcik
- Joanna Szołomicka  
- Teddy Ferdinan

## 📝 License

This project is created for educational purposes as part of the Artificial Intelligence and Knowledge Engineering course.

## �� Contributing

This is an academic project, but suggestions and improvements are welcome. Please ensure any modifications maintain the educational objectives and code quality standards.

---

**Note**: This project is designed for educational purposes and should not be used for actual medical diagnosis without proper validation and medical expertise.

