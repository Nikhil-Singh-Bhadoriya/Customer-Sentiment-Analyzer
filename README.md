# Customer Sentiment Analyzer

A comprehensive sentiment analysis project that uses both rule-based (TextBlob) and machine learning approaches to classify text sentiment from Twitter data.

## 📋 Overview

This project analyzes sentiment from text data using the Sentiment140 dataset(Dataset is taken from kaggle). It performs text preprocessing, exploratory data analysis, sentiment scoring, and trains multiple machine learning models to classify text as positive or negative sentiment.
## Dataset
Dataset = https://www.kaggle.com/code/poojag718/sentiment-analysis-machine-learning-approach#Conclusion

## 🚀 Features

- **Text Preprocessing**: Advanced text cleaning with stopword removal, lemmatization, and regex-based filtering
- **TextBlob Sentiment Analysis**: Rule-based sentiment scoring with polarity and subjectivity metrics
- **Multiple ML Models**: Comparison of Logistic Regression, Naive Bayes, and Linear SVM classifiers
- **Hyperparameter Tuning**: Grid search optimization for best model performance
- **Comprehensive Evaluation**: Accuracy, precision, recall, F1-score, and confusion matrix analysis
- **Rich Visualizations**: Sentiment distribution plots, model comparison charts, and word clouds

## 📦 Dependencies

Install the required packages using pip:

```bash
pip install pandas numpy nltk textblob scikit-learn matplotlib seaborn wordcloud
```

### Required Libraries:
- **pandas** & **numpy**: Data manipulation and numerical operations
- **nltk**: Natural language processing (stopwords, lemmatization)
- **textblob**: Sentiment polarity and subjectivity analysis
- **scikit-learn**: Machine learning models, vectorization, and evaluation metrics
- **matplotlib** & **seaborn**: Data visualization
- **wordcloud**: Generate word cloud visualizations

## 📊 Dataset

The project uses the **Sentiment140 dataset** which contains Twitter messages labeled with sentiment:
- **0**: Negative sentiment
- **4**: Positive sentiment

### Dataset Files:
- `training.1600000.processed.noemoticon.csv` - Full training dataset (1.6M tweets)
- `testdata.manual.2009.06.14.csv` - Manually annotated test data *(used in this analysis)*
- `train.csv` / `test.csv` - Additional training/test splits

Dataset format: `target, ids, date, flag, user, text`

**Current Analysis Dataset:** testdata.manual.2009.06.14.csv

## 🔧 Project Structure

```
Customer Sentiment Analyzer/
├── sentiment_analysis.ipynb          # Main notebook with analysis
├── README.md                          # Project documentation
├── train.csv                          # Training data
├── test.csv                           # Test data
├── testdata.manual.2009.06.14.csv   # Manual test data
└── training.1600000.processed.noemoticon.csv  # Full training dataset
```

## 🛠️ Workflow

### 1. Data Loading & Exploration
- Load CSV files with proper encoding (latin-1/utf-8)
- Explore dataset structure and sentiment distribution
- Sample large datasets for efficient processing (up to 200k records)

### 2. Text Preprocessing
The cleaning pipeline includes:
- Convert to lowercase
- Remove URLs and special characters
- Remove stopwords (using NLTK)
- Lemmatization (reduce words to base form)
- Filter short tokens (length > 1)

### 3. TextBlob Sentiment Analysis
- Calculate **polarity** scores (-1 to 1: negative to positive)
- Calculate **subjectivity** scores (0 to 1: objective to subjective)
- Quick baseline sentiment understanding

### 4. Machine Learning Models

#### Models Trained:
1. **Logistic Regression** (baseline)
2. **Multinomial Naive Bayes**
3. **Linear SVM (LinearSVC)**
4. **Optimized Logistic Regression** (Grid Search with C=[0.5, 1.0, 2.0])
   - **Best parameter found:** C=2.0

#### Feature Extraction:
- **TF-IDF Vectorization** with max 5000 features
- N-grams: unigrams and bigrams (1, 2)

### 5. Model Evaluation
Metrics computed for each model:
- **Accuracy**: Overall correctness
- **Precision**: Positive prediction reliability
- **Recall**: Positive case coverage
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: True/false positive/negative breakdown

### 6. Visualizations
- **Sentiment Distribution**: Count plot of positive vs negative samples
- **Model Performance**: F1-score comparison bar chart
- **Confusion Matrix**: Heatmap showing prediction accuracy
- **Word Clouds**: Visual representation of most common words in positive and negative texts

## 📈 Results

The notebook trains and evaluates four models, ranking them by F1-score. Based on actual execution results:

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **LinearSVC** | **92.11%** | **90.48%** | **95.00%** | **92.68%** |
| LogisticRegression | 89.47% | 84.78% | 97.50% | 90.70% |
| LogReg_Grid (C=2.0) | 89.47% | 86.36% | 95.00% | 90.48% |
| MultinomialNB | 88.16% | 82.98% | 97.50% | 89.66% |

### Best Model: LinearSVC

The **Linear SVM (LinearSVC)** achieved the highest F1-score of **92.68%** and is selected as the best model.

**Detailed Classification Report:**
```
              precision    recall  f1-score   support

   Negative       0.94      0.89      0.91        36
   Positive       0.90      0.95      0.93        40

   accuracy                           0.92        76
```

**Confusion Matrix:**
```
                Predicted
              Neg    Pos
Actual Neg    32     4
       Pos     2    38
```

**Key Insights:**
- The LinearSVC model correctly classified **70 out of 76** test samples (92.11% accuracy)
- Only **4 false negatives** and **2 false positives** in the test set
- High recall (95%) for positive sentiment detection
- Balanced performance across both sentiment classes

## 🚦 Getting Started

1. **Clone or download** this repository
2. **Install dependencies**:
   ```bash
   pip install pandas numpy nltk textblob scikit-learn matplotlib seaborn wordcloud
   ```
3. **Download NLTK data**:
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('wordnet')
   nltk.download('omw-1.4')
   ```
4. **Ensure dataset files** are in the project directory
5. **Open and run** `sentiment_analysis.ipynb` in Jupyter Notebook or VS Code

## 💡 Usage

### Run the Entire Notebook:
Execute all cells sequentially to:
1. Load and explore data
2. Preprocess text
3. Perform TextBlob analysis
4. Train ML models
5. Evaluate and compare results
6. Generate visualizations

### Analyze Custom Text:
```python
# After running the notebook
def predict_sentiment(text):
    clean = clean_text(text)
    vec = vectorizer.transform([clean])
    prediction = best_model.predict(vec)[0]
    return "Positive" if prediction == 1 else "Negative"

# Example
predict_sentiment("I love this product! It's amazing!")
```

## 📝 Notes

- **Large Dataset Handling**: The notebook samples up to 200,000 records for faster processing
- **Encoding**: CSV files are loaded with latin-1 encoding (common for Sentiment140)
- **TextBlob Sample**: Sentiment scoring is performed on a 5,000-record sample to optimize speed
- **Train-Test Split**: 80-20 split with stratification to maintain class balance

## 🔍 Key Findings

- **LinearSVC outperforms** other models with 92.68% F1-score, achieving the best balance of precision and recall
- **High recall (95%)** for positive sentiment shows the model rarely misses positive cases
- **TF-IDF with bigrams** successfully captures phrase-level sentiment indicators
- **Minimal misclassifications:** Only 6 errors out of 76 test samples (4 false negatives, 2 false positives)
- **Grid search optimization** identified C=2.0 as the best regularization parameter for Logistic Regression
- **Balanced performance** across both sentiment classes (F1: 0.91 for negative, 0.93 for positive)
- Word clouds reveal distinct vocabulary patterns between positive and negative sentiments
- All models achieve **>88% accuracy**, indicating robust feature engineering and preprocessing

**Note**: This project is designed for learning and demonstration purposes. For production use, consider additional preprocessing, cross-validation, and model optimization strategies.
