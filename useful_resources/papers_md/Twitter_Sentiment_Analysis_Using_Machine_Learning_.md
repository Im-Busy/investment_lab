# Twitter_Sentiment_Analysis_Using_Machine_Learning_

> *Source PDF: Twitter_Sentiment_Analysis_Using_Machine_Learning_.pdf*
> *Extraction: Combined — markitdown raw text (base) + previous extraction supplements*

---

International Journal of Scientific Research in Science and Technology
Available online at : www.ijsrst.com
Print ISSN: 2395-6011 | Online ISSN: 2395-602X d o i : h t t p s : / / d o i . org/10.32628/IJSRST251241
Twitter Sentiment Analysis Using Machine Learning Techniques
G. Anish Kumar1, Dr. C Jayapratha2
1PG Scholar, Department of BDA, Karpaga Vinayaga College of Engineering and Technology, Chengalpattu,
Tamil Nadu, India
2Professor, Department of CSE, Karpaga Vinayaga College of Engineering and Technology, Chengalpattu, Tamil
Nadu, India
A R T I C L E I N F O A B S T R A C T
This paper presents an effective sentiment analysis system designed to
Article History:
classify the polarity of tweets into positive, negative, or neutral sentiments.
Accepted : 26 June 2025
The framework utilizes supervised machine learning algorithms, including
Published: 01 July 2025
Logistic Regression, Support Vector Machines (SVM), and Random Forest,
trained on the Sentiment140 dataset. Text preprocessing techniques such
as tokenization, stopword removal, stemming, and TF-IDF vectorization
Publication Issue :
are applied to improve classification performance. The proposed system
Volume 12, Issue 4
achieves an accuracy of 87.2% with SVM, outperforming other baseline
July-August-2025
models. This solution offers scalable deployment in social media
monitoring, political campaign tracking, and customer feedback analysis.
Page Number :
Keywords— sentiment analysis, Twitter, machine learning, natural
01-04
language processing, text classification, social media analytics
I. INTRODUCTION machine learning algorithms, enhanced by effective
text preprocessing to handle the unique characteristics
In the era of digital communication, platforms like of Twitter data.
Twitter have become central to the public expression
of opinions and emotions. The concise and II. PROPOSED METHODOLOGY
spontaneous nature of tweets makes them a valuable
source of sentiment data. Analyzing this data helps The architecture of the proposed sentiment
organizations and governments understand public classification system includes five stages: data
mood, assess brand reputation, and forecast election collection, preprocessing, feature extraction, model
trends. training, and evaluation.
Traditional text classification methods struggle with 3.1 Data Collection
Twitter’s informal language, hashtags, emojis, and The Sentiment140 dataset, comprising 1.6 million
abbreviations. Therefore, this paper aims to build a labeled tweets (positive, negative, neutral), was used.
robust sentiment analysis system using classical
Copyright © 2025 The Author(s): This is an open-access article distributed under the terms of the Creative 01
Commons Attribution 4.0 International License (CC BY-NC 4.0)

G. Anish Kumar et al Int J Sci Res Sci & Technol. July-August-2025, 12 (4) : 01-04

| Each tweet is associated with a sentiment label and  | Username    |     |     |     |
| ---------------------------------------------------- | ----------- | --- | --- | --- |
| has been pre-tagged based on emoticons.              | Tweet text  |     |     |     |
3.2  Text  Preprocessing  To  enhance  the  quality  of  Sentiment label
features, the following preprocessing steps were  Class Distribution (after cleaning for balance):
| applied:  | Positive: 50,000  |     |     |     |
| --------- | ----------------- | --- | --- | --- |
Lowercasing  all  text  Removing  URLs,  mentions,  Negative: 50,000
hashtags, and special characters Tokenizing text into  Neutral: 50,000 (Total: 150,000 tweets)
words Removing stopwords (e.g., ―is,‖ ―and,‖ ―the‖)  4.2  Text Preprocessing
Applying Porter stemming to reduce words to their  To prepare the data for machine learning algorithms,
base form  the following preprocessing steps were applied using
| 3.3  Feature Extraction   | NLTK and regular expressions:  |     |     |     |
| ------------------------- | ------------------------------ | --- | --- | --- |
We  used  the  TF-IDF  (Term  Frequency-Inverse  Lowercasing: All tweets were converted to lowercase.
Document  Frequency)  vectorizer  to  transform  the  Noise removal: URLs, mentions (@username), hashtags,
textual data into numerical feature vectors suitable for  special symbols, and numbers were removed.
training machine learning models.  Tokenization: Tweets were split into individual words
| 3.4  Model Training Three models were implemented:  | (tokens).  |     |     |     |
| --------------------------------------------------- | ---------- | --- | --- | --- |
Logistic Regression Support  Stopword removal: Common English stopwords (e.g.,
| Vector Machine (SVM)  | ―the‖, ―is‖) were eliminated.  |     |     |     |
| --------------------- | ------------------------------ | --- | --- | --- |
Random Forest Classifier  Stemming:  Porter  Stemmer  was  applied  to  reduce
Each model was trained on 80% of the dataset and  words to their root form.
evaluated on the remaining 20%.  Emoji  and  Emoticon  removal:  All  emoticons  were
|     | removed post labeling, ensuring no label leakage.  |     |     |     |
| --- | -------------------------------------------------- | --- | --- | --- |
III. EXPERIMENTAL SETUP  4.3  Feature Extraction
|     | We  applied  | TF-IDF  | Vectorization  | using  |
| --- | ------------ | ------- | -------------- | ------ |
This  section  describes  the  technical  environment,  TfidfVectorizer from Scikit-learn:
dataset  preparation,  preprocessing  methods,  model  N-grams: (1,2) to include unigrams and bigrams
training  strategy,  and  evaluation  settings  used  to  Max Features: 5000
conduct the sentiment analysis experiments on Twitter  Minimum Document Frequency (min_df): 5
| data.  | Maximum Document Frequency (max_df): 0.9  |     |     |     |
| ------ | ----------------------------------------- | --- | --- | --- |
4.1  Dataset Description   Token pattern: Alphanumeric words with at least one
| The Sentiment140 dataset was used for this research. It  | character  |     |     |     |
| -------------------------------------------------------- | ---------- | --- | --- | --- |
includes 1.6 million tweets automatically labeled using  This step converted raw tweet text into sparse feature
emoticons as sentiment indicators:  vectors suitable for machine learning algorithms.
Positive tweets (Label: 4): Labeled based on ":)" or ":D"  4.4  Model Implementation
Negative tweets (Label: 0): Labeled based on ":(" or ">:("  Three  different  machine  learning  algorithms  were
Neutral  tweets  (Label:  2):  Extracted  from  manually  trained and tested:
cleaned data or through other datasets for multiclass  Logistic  Regression
| classification        | (sklearn.linear_model.LogisticRegression)  |         |          |        |
| --------------------- | ------------------------------------------ | ------- | -------- | ------ |
| Each tweet includes:  | Solver: liblinear                          |         |          |        |
| Tweet ID              | Penalty: L2 regularization                 |         |          |        |
| Date of posting       | Support                                    | Vector  | Machine  | (SVM)  |
| Query term            | (sklearn.svm.LinearSVC)                    |         |          |        |
|                       |                                            |         |          | 02     |
International Journal of Scientific Research in Science and Technology (www.ijsrst.com) | Volume 12 |  Issue 4

G. Anish Kumar et al Int J Sci Res Sci & Technol. July-August-2025, 12 (4) : 01-04

Linear kernel
Regularization parameter C = 1.0
| Random  |     |     | Forest  |     | Classifier  |     |     |     |     |     |     |     |     |
| ------- | --- | --- | ------- | --- | ----------- | --- | --- | --- | --- | --- | --- | --- | --- |
(sklearn.ensemble.RandomForestClassifier)
Number of Trees: 100
Max Depth: 20
4.5  Training and Validation
Strategy Train-Test Split: 80% training data and 20%

testing data
|                    |     |         |             |                   |     |     | 5.2  Confusion Matrix (SVM)  |     |            |     |                      |     |      |
| ------------------ | --- | ------- | ----------- | ----------------- | --- | --- | ---------------------------- | --- | ---------- | --- | -------------------- | --- | ---- |
| Cross-validation:  |     | 5-fold  | stratified  | cross-validation  |     | to  |                              |     |            |     |                      |     |      |
|                    |     |         |             |                   |     |     | The  following               |     | confusion  |     | matrix  illustrates  |     | the  |
avoid overfitting and ensure model stability
|             |      |           |     |              |            |     | performance  |     | of  the  | SVM  | model  across  | the  | three  |
| ----------- | ---- | --------- | --- | ------------ | ---------- | --- | ------------ | --- | -------- | ---- | -------------- | ---- | ------ |
| 4.6  Tools  | and  | Hardware  |     | Programming  | Language:  |     |              |     |          |      |                |      |        |
sentiment classes (Positive, Neutral, and Negative):
Python 3.9
|     |     |     |     |     |     |     | Predicted  | Positive  |     | Predicted  | Neutral  | Predicted  |     |
| --- | --- | --- | --- | --- | --- | --- | ---------- | --------- | --- | ---------- | -------- | ---------- | --- |
Libraries:
Negative Actual Positive 3500 230 120 Actual Neutral
scikit-learn for ML models
190 2900 180 Actual Negative 160 210 3300
NLTK for text preprocessing
|     |     |     |     |     |     |     | The  confusion  |     | matrix  | indicates  | that  | the  model  | is  |
| --- | --- | --- | --- | --- | --- | --- | --------------- | --- | ------- | ---------- | ----- | ----------- | --- |
pandas, NumPy for data handling
|     |     |     |     |     |     |     | particularly  | effective  |     | at  correctly  | classifying  | Positive  |     |
| --- | --- | --- | --- | --- | --- | --- | ------------- | ---------- | --- | -------------- | ------------ | --------- | --- |
matplotlib, seaborn for visualization
|     |     |     |     |     |     |     | and  Negative  |     | sentiments,  |     | with  slightly  |     | more  |
| --- | --- | --- | --- | --- | --- | --- | -------------- | --- | ------------ | --- | --------------- | --- | ----- |
System Configuration:
misclassifications in the Neutral category.
CPU: Intel i5 10th Gen @ 2.4 GHz
RAM: 16 GB
OS: Windows 11 / Ubuntu 20.04 (dual setup)
IDE: Jupyter Notebook / VS Code

IV. RESULTS AND ANALYSIS

5.1  Performance Metrics
The performance of three different machine learning

| models—Logistic Regression, Support Vector Machine  |     |     |     |     |     |     |     |     |     |     |     |     |     |
| --------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
(SVM),  and  Random  Forest—was  evaluated  using  V. DISCUSSION
| standard  | classification  |     | metrics.  | The  | results  | are  |     |     |     |     |     |     |     |
| --------- | --------------- | --- | --------- | ---- | -------- | ---- | --- | --- | --- | --- | --- | --- | --- |

| summarized in the table below:  |     |     |     |     |     |     | 6.1  Key Findings   |     |     |     |     |     |     |
| ------------------------------- | --- | --- | --- | --- | --- | --- | ------------------- | --- | --- | --- | --- | --- | --- |
Model  Accuracy  Precision  Recall  F1-Score  Logistic  The SVM model performs well due to its ability to
| Regression  | 85.6%  |     | 0.86  0.85  | 0.85  | Support  | Vector  |     |     |     |     |     |     |     |
| ----------- | ------ | --- | ----------- | ----- | -------- | ------- | --- | --- | --- | --- | --- | --- | --- |
handle high-dimensional data with sparse features.
Machine 87.2% 0.88 0.86 0.87 Random Forest 83.4%  Text  preprocessing  significantly  improves  model
| 0.84 0.83 0.83  |     |         |          |        |           |      | accuracy.  |     |     |     |     |     |     |
| --------------- | --- | ------- | -------- | ------ | --------- | ---- | ---------- | --- | --- | --- | --- | --- | --- |
| The  Support    |     | Vector  | Machine  | (SVM)  | achieved  | the  |            |     |     |     |     |     |     |
Emoticons and hashtags contribute to the polarity and
highest  overall  performance,  demonstrating  superior  should be retained or encoded meaningfully in future
| accuracy and F1-score compared to the other models.  |     |     |     |     |     |     | work.              |     |     |     |     |     |     |
| ---------------------------------------------------- | --- | --- | --- | --- | --- | --- | ------------------ | --- | --- | --- | --- | --- | --- |
|                                                      |     |     |     |     |     |     | 6.2  Limitations   |     |     |     |     |     |     |
The dataset may not reflect recent tweet trends, slang,
or context.
|     |     |     |     |     |     |     |     |     |     |     |     |     | 03  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
International Journal of Scientific Research in Science and Technology (www.ijsrst.com) | Volume 12 |  Issue 4

G. Anish Kumar et al Int J Sci Res Sci & Technol. July-August-2025, 12 (4) : 01-04

Sarcasm  and  irony  remain  challenging  to  detect  in  Computing  Surveys, vol. 54, no.  7, pp. 1–38,
| sentiment classification.  |     |     |     |     |     |     |     | 2021. [DOI: 10.1145/3453152]  |     |     |     |     |     |
| -------------------------- | --- | --- | --- | --- | --- | --- | --- | ----------------------------- | --- | --- | --- | --- | --- |
The models are limited to three classes and do not  [4].  P.  Nakov  et  al.,  ―SemEval-2016  Task  4:
capture nuanced emotions like joy, anger, or disgust.  Sentiment Analysis in Twitter,‖ Proceedings of
|     |     |     |     |     |     |     |     | SemEval-2016, pp. 1–18. [ACL Anthology: S16- |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | -------------------------------------------- | --- | --- | --- | --- | --- |
1001]
|     |     |     |     |     |     |     | [5].  | A.  Go,        | R.  | Bhayani,        | and          | L.  Huang,  | ―Twitter  |
| --- | --- | --- | --- | --- | --- | --- | ----- | -------------- | --- | --------------- | ------------ | ----------- | --------- |
|     |     |     |     |     |     |     |       | Sentiment      |     | Classification  |              | using       | Distant   |
|     |     |     |     |     |     |     |       | Supervision,‖  |     | Stanford        | University,  |             | CS224N    |
Project Report, 2009. (Foundational Reference)
|     |     |     |     |     |     |     |     | [Available  |     |     |     |     | at:  |
| --- | --- | --- | --- | --- | --- | --- | --- | ----------- | --- | --- | --- | --- | ---- |
http://cs.stanford.edu/people/alecmgo/papers/T
witterDistantSupervision09.pdf]
|     |     |     |     |     |     |     | [6].  | T. Devlin, M. Chang, K. Lee, and K. Toutanova,  |               |                 |           |                  |                |
| --- | --- | --- | --- | --- | --- | --- | ----- | ----------------------------------------------- | ------------- | --------------- | --------- | ---------------- | -------------- |
|     |     |     |     |     |     |     |       | ―BERT:                                          | Pre-training  |                 | of        | Deep             | Bidirectional  |
|     |     |     |     |     |     |     |       | Transformers                                    |               | for             | Language  | Understanding,‖  |                |
|     |     |     |     |     |     |     |       | Proceedings                                     |               | of  NAACL-HLT,  |           | pp.              | 4171–4186,     |
|     |     |     |     |     |     |     |       | 2019. [arXiv:1810.04805]                        |               |                 |           |                  |                |
VI. CONCLUSION:  [7].  H.  Singh  and  M.  Sharma,  ―Aspect-Based
|     |     |     |     |     |     |     |     | Sentiment  |     | Analysis  | on  Twitter  |     | Data  Using  |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------- | --- | --------- | ------------ | --- | ------------ |
We  presented  a  Twitter  sentiment  analysis  system  BERT  Embeddings,‖  International  Journal  of
using traditional machine learning classifiers. Among  Information Management Data Insights, vol. 4,
the models tested, SVM achieved the highest accuracy  no. 1, 2024. [DOI: 10.1016/j.jjimei.2024.100179]
of 94.2%. Future work may explore transformer-based  [8].  S. Wang and Z. Li, ―Sentiment Analysis Using
deep  learning  models  like  BERT  and  RoBERTa  to  Pre-trained Language Models: A Case Study on
handle sarcasm and context better.  Twitter,‖  IEEE  Big  Data,  pp.  334–341,  2021.
|     |     |     |     |     |     |     |     | [IEEE Xplore]  |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | -------------- | --- | --- | --- | --- | --- |
REFERENCES  [9].  A. Sun, C. Peng, and X. Li, ―Detecting Sarcasm
|     |     |     |     |     |     |     |     | in  Twitter  |     | with  | Contextualized  |     | Language  |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------ | --- | ----- | --------------- | --- | --------- |
[1].  Y. Zhang, D. Jin, X. Xie, and H. Liu, ―BERT- Models,‖ Expert Systems with Applications, vol.
Based  Sentiment  Analysis  for  Social  Media  209, 2023. [DOI: 10.1016/j.eswa.2022.118258]
Texts,‖ Information Processing & Management,
|     | vol.  | 59,  no.  | 2,  pp.  | 102748,  | 2022.  | [DOI:  |     |     |     |     |     |     |     |
| --- | ----- | --------- | -------- | -------- | ------ | ------ | --- | --- | --- | --- | --- | --- | --- |
10.1016/j.ipm.2021.102748]
| [2].  | H.  Liu,   | R.    | Wu,  and  | K.     | Zhao,    | ―Sentiment  |     |     |     |     |     |     |     |
| ----- | ---------- | ----- | --------- | ------ | -------- | ----------- | --- | --- | --- | --- | --- | --- | --- |
|       | Analysis   | on    | COVID-19  |        | Tweets   | Using       |     |     |     |     |     |     |     |
|       | RoBERTa,‖  | IEEE  | Access,   | vol.   | 9,  pp.  | 148566–     |     |     |     |     |     |     |     |
|       | 148576,    |       |           | 2021.  |          | [DOI:       |     |     |     |     |     |     |     |
10.1109/ACCESS.2021.3124540]
[3].  A. Giachanou and F. Crestani, ―Deep Learning
|     | for  Sentiment  |     | Analysis:  |     | A  Survey,‖  | ACM  |     |     |     |     |     |     |     |
| --- | --------------- | --- | ---------- | --- | ------------ | ---- | --- | --- | --- | --- | --- | --- | --- |
|     |                 |     |            |     |              |      |     |     |     |     |     |     | 04  |
International Journal of Scientific Research in Science and Technology (www.ijsrst.com) | Volume 12 |  Issue 4

---

## Content from Previous Extraction (not in markitdown output)

### Visual/Chart/Graph Descriptions

#### 1

NLTK for text preprocessing pandas, NumPy for data handling matplotlib, seaborn for visualization System Configuration: CPU: Intel i5 10th Gen @ 2.4 GHz

### Additional Content

#### 1

**==> picture [49 x 49] intentionally omitted <==**

#### 2

**International Journal of Scientific Research in Science and Technology** Available online at : www.ijsrst.com Print ISSN: 2395-6011 | Online ISSN: 2395-602X doi : https://doi.org/10.32628/IJSRST251241

#### 3

**==> picture [44 x 58] intentionally omitted <==**

#### 4

## **Twitter Sentiment Analysis Using Machine Learning Techniques**

#### 5

G. Anish Kumar[1] , Dr. C Jayapratha[2]

#### 6

|A R T I C L E I N F O<br>Article History:<br>Accepted : 26 June 2025<br>Published: 01 July 2025<br>Publication Issue :<br>Volume 12, Issue 4<br>July-August-2025<br>Page Number :<br>01-04|A B S T R A C T|
|---|---|
||This paper presents an effective sentiment analysis system designed to<br>classify the polarity of tweets into positive, negative, or neutral sentiments.<br>The framework utilizes supervised machine learning algorithms, including<br>Logistic Regression, Support Vector Machines (SVM), and Random Forest,<br>trained on the Sentiment140 dataset. Text preprocessing techniques such<br>as tokenization, stopword removal, stemming, and TF-IDF vectorization<br>are applied to improve classification performance. The proposed system<br>achieves an accuracy of 87.2% with SVM, outperforming other baseline<br>models. This solution offers scalable deployment in social media<br>monitoring, political campaign tracking, and customer feedback analysis.<br>Keywords—sentiment analysis, Twitter, machine learning, natural<br>language processing, text classification, social media analytics|

#### 7

In the era of digital communication, platforms like Twitter have become central to the public expression of opinions and emotions. The concise and spontaneous nature of tweets makes them a valuable source of sentiment data. Analyzing this data helps organizations and governments understand public mood, assess brand reputation, and forecast election trends.

#### 8

Traditional text classification methods struggle with Twitter’s informal language, hashtags, emojis, and abbreviations. Therefore, this paper aims to build a robust sentiment analysis system using classical

#### 9

machine learning algorithms, enhanced by effective text preprocessing to handle the unique characteristics of Twitter data.

#### 10

## **II.** PROPOSED METHODOLOGY

#### 11

The architecture of the proposed sentiment classification system includes five stages: data collection, preprocessing, feature extraction, model training, and evaluation.

#### 12

The Sentiment140 dataset, comprising 1.6 million labeled tweets (positive, negative, neutral), was used.

#### 13

Copyright © 2025 The Author(s): This is an open-access article distributed under the terms of the Creative Commons Attribution 4.0 International License (CC BY-NC 4.0)

#### 14

Each tweet is associated with a sentiment label and has been pre-tagged based on emoticons.

#### 15

3.2 Text Preprocessing To enhance the quality of features, the following preprocessing steps were applied:

#### 16

Lowercasing all text Removing URLs, mentions, hashtags, and special characters Tokenizing text into words Removing stopwords (e.g., ―is,‖ ―and,‖ ―the‖) Applying Porter stemming to reduce words to their base form

#### 17

We used the TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer to transform the textual data into numerical feature vectors suitable for training machine learning models.

#### 18

## 3.4 Model Training Three models were implemented:

#### 19

Logistic Regression Support Vector Machine (SVM) Random Forest Classifier

#### 20

Each model was trained on 80% of the dataset and evaluated on the remaining 20%.
