# Spam Classifier

A spam/ham email classifier using TF-IDF and Logistic Regression, built in August 2024.

## Dataset

- `mail_data.csv`, 5,572 messages, 2 columns: Category, Message
- 4,825 ham (86.6%), 747 spam (13.4%)
- No missing values

## Approach

1. Labels mapped to 0 (spam) and 1 (ham)
2. 70/30 train/test split (`random_state=3`)
3. `TfidfVectorizer` (English stopwords removed, `min_df=1`) fit on the training messages, vocabulary size 6,896
4. `LogisticRegression` trained on the TF-IDF features

## Results

| Metric | Value |
|---|---|
| Train accuracy | 96.6% |
| Test accuracy | 96.5% |
| Majority-class baseline (always predict ham) | 86.6% |

Test accuracy clears the baseline by about 10 points, so the model is adding real signal here, not just reflecting the class imbalance.

**Test set confusion matrix (232 actual spam, 1440 actual ham):**

| | Predicted spam | Predicted ham |
|---|---|---|
| Actual spam | 174 | 58 |
| Actual ham | 1 | 1439 |

- Spam precision: 99.4% (only 1 real message flagged as spam)
- Spam recall: 75.0% (58 of 232 actual spam messages got through as ham)
- Ham recall: 99.9%

The model is precision-heavy: it almost never blocks a real message by mistake, but it misses about 1 in 4 spam messages. For a spam filter that's arguably the safer failure mode (letting spam through beats blocking real mail), but it's worth stating plainly rather than only quoting the 96.5% headline accuracy.

## Corrections to the original README

The original README described this project as using "Naive Bayes and SVM" to achieve its results. Looking at the actual code, only `LogisticRegression` is trained and evaluated: there's no Naive Bayes or SVM model anywhere in the script. This rewrite reflects what the code actually does.

The script also builds a `clean_message` column using NeatText to strip stopwords, but that column is never used, the model is trained on the raw `Message` text. The cleaning step exists in the notebook but has no effect on the model's input.

## Tech Stack

Python, Pandas, NumPy, Seaborn, scikit-learn (`TfidfVectorizer`, `LogisticRegression`)

## How to Run

```bash
pip install pandas numpy scikit-learn seaborn
python spam_classifier.py
```

## What I Learned

This project was my first time looking past a single accuracy number into a confusion matrix. 96.5% accuracy sounds strong on its own, but the breakdown shows the model trades recall for precision on the minority class, missing a quarter of actual spam to almost never flag a real message. That's a real, measurable trade-off, not just a number to quote, and it's the kind of detail an accuracy score alone hides.
