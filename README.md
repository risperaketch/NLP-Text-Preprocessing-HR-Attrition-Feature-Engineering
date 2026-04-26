# NLP-Text-Preprocessing-HR-Attrition-Feature-Engineering

> **Two complete, end-to-end preprocessing pipelines in one notebook — an NLP pipeline for automated news classification across 127,375 articles, and a structured data pipeline preparing 1,470 HR records for attrition prediction. Score: 100 / 100.**

---

## Project Overview

This project demonstrates the full preprocessing workflow required before any machine learning model can be applied to real-world business data. It covers two distinct problem domains using two different datasets, each requiring different preprocessing strategies.

| Pipeline | Domain | Dataset Size | Core Techniques |
|---|---|---|---|
| **Part 1 — NLP** | News article classification | 127,375 articles | Stop word removal, regex cleaning, TF-IDF |
| **Part 2 — HR Analytics** | Employee attrition prediction | 1,470 employees | Imputation, discretization, OHE, target encoding |

---

## Business Context

### Part 1 — News Article Tagger
A news aggregator app needs an algorithm that automatically classifies incoming articles into categories (politics, sports, entertainment, technology, etc.) based on their content — without manual tagging. Before any classifier can be trained, raw article text must go through a rigorous NLP pipeline that transforms unstructured prose into a clean, numerical feature matrix.

### Part 2 — HR Attrition Predictor
Employee turnover costs organizations 50–200% of an employee's annual salary in recruitment, onboarding, and lost productivity. An HR analytics team wants to identify which employees are at risk of leaving so proactive retention strategies can be deployed. The raw HR dataset must be feature-engineered into a format suitable for binary classification before modeling begins.

---

## Datasets

### Dataset 1 — News Articles
| Property | Value |
|---|---|
| Total records | 127,393 |
| Records after cleaning | 127,375 |
| Columns | `date`, `text`, `label` |
| Categories | Politics, Sports, Entertainment, Technology, and more |
| Null rows removed | 18 (articles with no text content) |

### Dataset 2 — IBM HR Analytics
| Property | Value |
|---|---|
| Employees | 1,470 |
| Features | 32 original → 54 after encoding |
| Target variable | `Attrition` (Yes / No → 1 / 0) |
| Attrition rate | 16.1% (237 of 1,470 employees) |
| Majority missing columns | None found |

---
<img width="980" height="384" alt="image" src="https://github.com/user-attachments/assets/abcc18a6-1c12-4510-9581-b0248b6e877f" />

## Part 1 — NLP Pipeline: 5 Steps

### Step 1 — Remove Null Text Rows (`dropna`)
```python
df = df.dropna(subset=['text'])  # 127,393 → 127,375 rows
```
Articles with no text content have zero classification signal and must be removed before tokenization to avoid downstream errors.

---

### Step 2 — Lowercase Conversion (`.str.lower()`)
```python
df['text'] = df['text'].str.lower()
```
Without normalization, `Parliament`, `parliament`, and `PARLIAMENT` are three distinct TF-IDF features. Lowercasing collapses all morphological variants into a single token, reducing vocabulary size and improving feature consistency.

---

### Step 3 — Stop Word Removal (NLTK)
```python
stop_words = set(stopwords.words('english'))  # 179 words
df['text'] = df['text'].apply(lambda t: ' '.join([w for w in t.split() if w not in stop_words]))
```
High-frequency function words (the, will, is, and) appear in virtually every document and provide no discriminative information for category prediction. Removing them reduces noise and sharpens the signal from meaningful domain words.

**Before:** `"Farmers will get subsidies for turning fields back into wildflower meadows after brexit"`  
**After:** `"farmers get subsidies turning fields back wildflower meadows brexit"`

---

### Step 4 — Special Character and Number Removal (Regex)
```python
df['text'] = df['text'].str.replace(r'[^a-zA-Z]', ' ', regex=True)
df['text'] = df['text'].str.replace(r'\s+', ' ', regex=True).str.strip()
```
Numbers, percentages, quotation marks, and punctuation vary unpredictably across articles and contribute noise rather than signal. The regex pattern retains only alphabetic characters, and whitespace normalization ensures clean token boundaries.

---

### Step 5 — TF-IDF Vectorization
```python
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['text'])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
tfidf_df['label'] = df['label'].values
```
TF-IDF converts each article into a vector of word importance scores. Words distinctive to specific categories — `election`, `parliament` for politics; `goal`, `match` for sports — receive high scores and become primary classification features.

| TF-IDF Matrix Property | Value |
|---|---|
| Rows (articles) | 127,375 |
| Columns (vocabulary terms) | Tens of thousands |
| First feature | `aa` |
| Last feature | `zyxel` |
| Label column | Appended as final column |

<img width="984" height="384" alt="image" src="https://github.com/user-attachments/assets/26101ab1-aca7-44c1-b397-629908856276" />

---

## Part 2 — HR Attrition Pipeline: 5 Steps

### Step 6 — Variable Type Identification
```python
numerical_var   = [col for col in df.select_dtypes(include=['int64','float64']).columns if col != 'Attrition']
categorical_var = [col for col in df.select_dtypes(include=['object']).columns if col != 'Attrition']
```

| Type | Count | Examples |
|---|---|---|
| Numerical | **23** | Age, MonthlyIncome, DailyRate, YearsAtCompany, TotalWorkingYears |
| Categorical | **8** | BusinessTravel, Department, EducationField, Gender, JobRole |
| Target (excluded) | 1 | Attrition (processed separately in Q10) |

<img width="1583" height="308" alt="image" src="https://github.com/user-attachments/assets/e9956d6e-6e37-442e-aa11-f4502682ae2d" />

<img width="1384" height="409" alt="image" src="https://github.com/user-attachments/assets/909cb162-efc6-4ccc-85aa-1ed7f5240854" />


---

### Step 7 — Missing Value Imputation
```python
# Majority-missing columns (> 50%): drop
majority_missing_var = [col for col in df.columns if df[col].isnull().sum() > 0.5 * len(df)]

# Numerical: fill with median (robust to outliers)
for col in numerical_var:
    df[col] = df[col].fillna(df[col].median())

# Categorical: fill with mode (most frequent category)
for col in categorical_var:
    df[col] = df[col].fillna(df[col].mode()[0])
```

**Result:** No majority-missing columns detected. All columns retained. Zero missing values remain after imputation.

---

### Step 8 — Daily Rate Discretization (`pd.cut`)
```python
df['DailyRate_Category'] = pd.cut(df['DailyRate'], bins=3, labels=['low','medium','high'], include_lowest=True)
```
Converts continuous DailyRate (range: \$102–\$1,499) into equal-width tiers that can reveal non-linear relationships with attrition:

| Category | Count |
|---|---|
| Low | **484** |
| Medium | 502 |
| High | 484 |

<img width="1183" height="384" alt="image" src="https://github.com/user-attachments/assets/bdb0c170-3b72-413f-bbf7-9a4e723b882d" />

---

### Step 9 — One-Hot Encoding (`pd.get_dummies`)
```python
cat_vars_to_encode = [col for col in categorical_var if col != 'Attrition']
df = pd.get_dummies(df, columns=cat_vars_to_encode, drop_first=False)
```
Converts 8 categorical variables into binary dummy columns. `Attrition` is explicitly excluded — it is handled separately in Q10. The dataset expands from 32 columns to **54 columns** after encoding.

---

### Step 10 — Target Variable Encoding
```python
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
```

| Class | Count | Percentage |
|---|---|---|
| 0 (Stayed) | 1,233 | 83.9% |
| 1 (Left) | **237** | **16.1%** |


<img width="997" height="384" alt="image" src="https://github.com/user-attachments/assets/3f08af7a-a1e8-4041-b228-c3c126abfbfc" />


The 237 employees in the positive class (Attrition = 1) represent a class imbalance that will need to be addressed in downstream modeling (SMOTE, class weights, or threshold tuning).

---

## Final Score

| Question | Task | Technique | Result |
|---|---|---|---|
| Q1 | Remove null text rows | `dropna` | ✅ Pass |
| Q2 | Lowercase conversion | `.str.lower()` | ✅ Pass |
| Q3 | Stop word removal | NLTK stopwords | ✅ Pass |
| Q4 | Special character removal | Regex `[^a-zA-Z]` | ✅ Pass |
| Q5 | TF-IDF vectorization | `TfidfVectorizer` | ✅ Pass |
| Q6 | Variable type identification | `select_dtypes` | ✅ Pass |
| Q7 | Missing value imputation | Median / Mode | ✅ Pass |
| Q8 | DailyRate discretization | `pd.cut` | ✅ Pass |
| Q9 | One-hot encoding | `pd.get_dummies` | ✅ Pass |
| Q10 | Target encoding | `.map({'Yes':1,'No':0})` | ✅ Pass |

**Total: 100 / 100**

---

## Visualizations

| File | Contents |
|---|---|
| `news_category_distribution.png` | Horizontal bar chart — article count per news category |
| `tfidf_top_terms.png` | Highest TF-IDF terms in a sample article |
| `hr_attrition_eda.png` | Attrition count, income box plot, and age distribution |
| `numerical_distributions.png` | Histograms of 5 key numerical features |
| `dailyrate_discretization.png` | Before/after: continuous rate vs. discretized categories |
| `final_feature_matrix.png` | Final feature composition pie chart and column breakdown |

---

## Tech Stack

```
Python 3.10
├── pandas          — data loading, cleaning, groupby, get_dummies
├── NumPy           — numerical operations
├── scikit-learn    — TfidfVectorizer
├── nltk            — stopwords corpus
├── re              — regex special character removal
├── matplotlib      — all chart rendering and export
└── seaborn         — EDA visualizations
```

---

## How to Run

**Google Colab (recommended)**
```python
# Upload notebook → Runtime → Run all
# Datasets download automatically from Google Drive
```

**Local Jupyter**
```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn
jupyter notebook OkothAketch_A5.ipynb
```

> **Drive quota fallback:**
> ```python
> # News data:
> df = pd.read_csv('https://raw.githubusercontent.com/marineevy/datasets/main/news.csv')
> # HR data:
> df = pd.read_csv('https://raw.githubusercontent.com/marineevy/datasets/main/hr_analytics.csv')
> ```

---

## Skills Demonstrated

`NLP Text Preprocessing` `TF-IDF Vectorization` `Stop Word Removal` `Regex Cleaning` `Feature Engineering` `Missing Value Imputation` `Discretization` `One-Hot Encoding` `Target Encoding` `Binary Classification Preparation` `HR Analytics` `Python` `pandas` `scikit-learn` `NLTK` `seaborn`

---

## Author

**Aketch Adhiambo Okoth**  
MS Business Analytics — Montclair State University (GPA 3.8)  
[LinkedIn](https://linkedin.com/in/your-profile) · [Portfolio](https://your-portfolio-url.com)
