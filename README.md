
# Loan Performance Prediction Analysis

![Project Overview](https://via.placeholder.com/800x300?text=Loan+Performance+Prediction)

## 📋 Project Overview

The **Loan Performance Prediction Analysis** project is a comprehensive data analysis and machine learning pipeline aimed at predicting loan defaults based on historical loan data. The analysis uncovers key insights about loan performance, borrower behavior, and default patterns.

### Key Features:
- Data Cleaning and Preprocessing
- Exploratory Data Analysis (EDA) with Visualizations
- Machine Learning Pipeline using Random Forest Classifier
- Model Performance Evaluation with Metrics
- Insights and Recommendations for Business Impact

---

## 🛠️ Tech Stack

### Languages & Tools:
- **Python**: Primary language for data processing and model training
- **Jupyter Notebook**: Interactive environment for code and analysis
- **Pandas**: Data manipulation and analysis
- **Seaborn & Matplotlib**: Data visualization
- **WordCloud**: Visualizing text data insights
- **Scikit-Learn**: Machine learning pipeline
- **Nbconvert**: Exporting notebooks to various formats (e.g., HTML, PDF)
- **Pandoc**: Document converter for exporting reports

---

## 📊 Dataset

The dataset contains loan information such as:
- Loan Amount
- Term
- Grade
- Purpose
- Interest Rate
- Employment Title
- Loan Status (`Fully Paid` or `Charged Off`)

The dataset is processed and visualized to uncover meaningful patterns and insights.

---

## 📈 Key Visualizations

1. **Distribution of Loan Amounts**  
   Visualizes the frequency distribution of loan amounts.

2. **Average Loan Amount by Grade**  
   Highlights the average loan amounts across loan grades.

3. **Most Common Loan Purposes**  
   A word cloud representing the most frequent purposes for loans.

4. **Confusion Matrix**  
   Depicts the performance of the machine learning model.

---

## 🧩 Challenges Faced

### 1. High-Cardinality Columns
- **Problem**: Columns like `emp_title` and `desc` had too many unique values, causing memory issues during one-hot encoding.
- **Solution**: Dropped high-cardinality columns.

### 2. Non-Numeric Columns in Model Input
- **Problem**: Columns like `term` were non-numeric, causing errors during model training.
- **Solution**: Converted such columns to numeric formats.

### 3. Class Imbalance
- **Problem**: The model struggled with predicting default loans due to class imbalance.
- **Solution**: Suggested oversampling or advanced class-balancing techniques.

---

## 🔍 Insights and Recommendations

1. Loans in higher grades (e.g., F, G) have significantly higher average loan amounts.
2. "Debt Consolidation" is the most common loan purpose, highlighting a primary use case for loans.
3. The model performs well in identifying non-default loans (class 0) but requires further improvements for predicting defaults (class 1).

---

## 🚀 How to Use

### Prerequisites
1. Install **Python 3.7+**.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Steps to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/loan-performance-prediction.git
   ```
2. Navigate to the project folder:
   ```bash
   cd loan-performance-prediction
   ```
3. Activate the virtual environment (if applicable):
   ```bash
   venv\Scriptsctivate  # Windows
   source venv/bin/activate  # macOS/Linux
   ```
4. Run the analysis:
   ```bash
   jupyter notebook loan_performance_analysis_with_images.ipynb
   ```
5. Export the notebook to PDF or HTML:
   ```bash
   jupyter nbconvert --to pdf loan_performance_analysis_with_images.ipynb
   ```

---

## 📁 Project Structure

```
loan-performance-prediction/
├── data/
│   └── lending_club_loan_two.csv   # Dataset file
├── images/
│   ├── Figure_1.png               # Loan Amount Distribution
│   ├── Figure_2.png               # Average Loan Amount by Grade
│   ├── Figure_3.png               # Most Common Loan Purposes
│   └── Figure_4.png               # Confusion Matrix
├── main.py                        # Main analysis and pipeline script
├── loan_performance_analysis_with_images.ipynb  # Jupyter Notebook
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

---

## ⚡ Dependencies

The project requires the following Python libraries:
- pandas
- numpy
- matplotlib
- seaborn
- wordcloud
- scikit-learn
- jupyter
- nbconvert
- pandoc (external dependency)

To install all dependencies:
```bash
pip install -r requirements.txt
```

---

## 📧 Contact

For any questions or feedback, feel free to reach out:
- **Name**: Henil
- **Email**: henilpatelmiteshbhai12@gmail.com
- **LinkedIn**: www.linkedin.com/in/henil-patel12

---

## 🌟 Acknowledgments

Special thanks to the open-source community and the datasets provided by LendingClub for enabling this project.

---
