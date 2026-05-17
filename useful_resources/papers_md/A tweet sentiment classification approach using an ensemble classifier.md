# A tweet sentiment classification approach using an ensemble classifier

> *Source PDF: A tweet sentiment classification approach using an ensemble classifier.pdf*
> *Extraction: Combined — markitdown raw text (base) + previous extraction supplements*

---

Contents lists available at ScienceDirect

International Journal of Cognitive Computing in Engineering

journal homepage: www.keaipublishing.com/en/journals/international-
journal-of-cognitive-computing-in-engineering/

A tweet sentiment classification approach using an ensemble classifier

Vidyashree KP a, Rajendra AB a, Gururaj HL b, Vinayakumar Ravi c,*, Moez Krichen d,e
a Department of Information Science and Engineering, Vidyavardhaka College of Engineering, Mysuru, India
b Department of Information Technology, Manipal Institute of Technology Bengaluru, Manipal Academy of Higher Education, Manipal, India
c Center for Artificial Intelligence, Prince Mohammad Bin Fahd University, Khobar, Saudi Arabia
d Department of Information Technology, Faculty of Computer Science and Information Technology (FCSIT), Al-Baha University, Alaqiq, 65779-7738, Saudi Arabia
e ReDCAD Laboratory, University of Sfax, Sfax, 3038, Tunisia

A R T I C L E  I N F O

A B S T R A C T

Keywords:
Adaptive boosting
Ensemble classifier
Sentiment analysis
Tweets
Twitter API

Social media users are more receptive to products or events and share their thoughts through raw textual data,
which is classified as semi-structured data. This data, which is presented using a variety of terminologies, is noisy
by  nature  but  yet  contains  important  information  and  superfluous  details,  giving  analysts  a  way  to  identify
patterns  and  knowledge.  This  hidden  information  must  be  extracted  from  language  data  in  order  to  make
informed  decisions  and  create  strategic  plans  for  entering  new  markets.  Among  the  most  prominent  fields  of
study are natural language processing (NLP) and data mining techniques, especially when it comes to sentiment
analysis—the process of  identifying the feelings and insights  concealed in  the data. Twitter is one of  the sig-
nificant microblogging platform with millions of users. These users use Twitter to share sentiments using hash
tags on different topics and to make status updates known as tweets. Twitter is therefore regarded as a significant
real-time source and as one of the most active opinion indicators. The volume of information is produced by
Twitter  is  enormous  and  manually  scanning  the  entire  data  set  is  difficult  process.  The  paper  proposed  an
ensemble classifier to categorize emotion of the tweets on the basis of polarities such as positive and negative.
In our study, we ensemble classifiers which is a combination of Random Forest (RF), Support Vector Machine
(SVM) and Decision Tree (DT). The data is collected from Twitter API and the Twitter data is analysed auton-
omously  to  define  public  view  on  particular  topic.  The  features  obtained  after  the  process  of  dimensionality
reduction  using  LDA  undergoes  the  stage  of  feature  selection  using  Wrapper  based  technique.  The  iterative
Wrapper based technique predict score for the features, the features with low score are ignored and high score is
proceeded  for  classification.  The  ensemble  classifier  used  Adaptive  Boosting  (AdaBoost)  technique  where  the
output from the Machine Learning (ML) classifiers are combined to produce a single output. Adaboost combines
the poor classifiers and extracts the prediction value to make a better classifier. The experimental results show
that  the  proposed  ensemble  classifier  provides  better  accuracy  of  93.42  %  that  is  comparatively  better  than
existing Convolutional Bidirectional - Long Short-Term Memory (ConvBiLSTM) classifier and Hybrid Lexicon-
Naïve Bayes Classifier (HL-NBC) which produce classification accuracy of 91.53 % and 89.61 % respectively.

1. Introduction

Twitter have about 319 million of active number of users in a month,
this is considered as a treasury for the organizations and individual with
strong political and economic background and keenly shows interest in
preserving their reputation (Minaee et al., 2019, Naseem et al., 2020).
Twitter acts as a platform to share the public opinion and their view on
the organizations or individuals. Sentiment analysis acts as a key tool for
the organizations to know about their capability and public views on
their  organization  by  analyzing  the  tweets  based  on  the  sentiment

(Gandhi et al., 2021). The analysis of sentiment in Twitter platform is
emerging popular among the public where they express their feedback
through texts (tweets) (Jianqiang et al., 2018). The Sentiment Analysis
(SA)  is  considered  as  one  of  the  predictable  research  field  of  Natural
Language Processing (NLP) and SA plays an important role in automated
detection of sentiment present in the text. Moreover, the SA is vastly
availed  based  on  reviews,  surveys  and  economic  report  etc.  (Akhtar
et al., 2019, Soumya and Pramod, 2022). In the process of SA tweets of
similar type of tweets are taken into consideration then it is categorized
into words and sentences, this is known as tokenization. The language

* Corresponding author.

E-mail address: vravi@pmu.edu.sa (V. Ravi).

https://doi.org/10.1016/j.ijcce.2024.04.001
Received 3 May 2023; Received in revised form 16 March 2024; Accepted 25 April 2024

InternationalJournalofCognitiveComputinginEngineering5(2024)170–177Availableonline1May20242666-3074/©2024TheAuthors.PublishingServicesbyElsevierB.V.onbehalfofKeAiCommunicationsCo.Ltd.ThisisanopenaccessarticleundertheCCBYlicense(http://creativecommons.org/licenses/by/4.0/).V. KP et al.

utilized  in  Twitter  is  universal,  not  in  proper  structure  and  varying
slangs. So, a pre-processing step must be involved in filtering the un-
wanted data to enhance the Sentiment Analysis results (Das et al., 2020,
Nurulhuda Zainuddin et al., 2018).

The classification models based on deep learning have emerged and
attained its peak in various tasks such as answering for the questions,
analysis of sentiments, classification and generation of images, analysis
of sentiments (AlBadani et al., 2022). Users of Twitter post many mes-
sages throughout the day that can include thoughts about brands, or-
ganizations,  things,  and  public  relations  (Khalid  et  al.,  2020).  These
opinions  can  be  divided  into  three  different  attitudes:  adverse,  favor-
able, and neutral. There are various levels that can be assigned to each
attitude. This technique is known as sentiment analysis, analyses and
extracts contextual knowledge from unprocessed data. (Kim and Jeong,
2019, Bibi et al., 2022, Sergiu Cosmin Nistor et al., 2021). Analysis of
sentiments takes place through two stages such as extraction of features
and  categorization  of  sentiments.  The  extraction  of  relevant  features
helps to choose the entities in a particular topic (Sergiu Cosmin Nistor
et al., 2021, Nurulhuda Zainuddin et al., 2018). Moreover, the words
that change from one dataset to other results in poor prediction capacity
according to the ML models, obtained using Contextual Analysis (CA). It
should be emphasised that without the aid of any linguistics resources,
the words are automatically clustered and significant nodes are deter-
mined. Improved prediction outcomes should be taken in to the account
while performing sentimental analysis from discrete real time dataset
the usage of individual classifiers does not result in better classification
accuracy. So this research considered this drawback as a motivation and
introduced a sentiment analysis model to analyze collected tweets from
Twitter  data  based  on  positive,  negative  and  neutral  tags.  Here  an
ensemble based classifier is proposed to classify emotion on the basis of
tweets  given  by  user.  Sentiment  analysis  mostly  looks  for  positive  or
negative expressions of opinion regarding a person, news story, or dig-
ital material. Understanding user sentiment allows businesses, service
providers, and individuals to reevaluate their approaches and enhance
their  offerings.  Additionally,  it  aids  businesses  in  obtaining  critical
feedback regarding their goods and areas in need of development.
This research is employed by the following contributions:

1.  To investigate how various pre-processing methods and subjectivity
analysis affect sentiment classification performance and accuracy.
2.  To  extract  the  features  that  significantly  influence  the  reviews’
sentiment by using an appropriate feature extraction mechanism.
3.  A feature selection approach is proposed to determine which features
have the most impact. The goal of this strategy is to increase cate-
gorization effectiveness in order to get pertinent opinions.

4.  A new ensemble classifier is proposed which is a combination of ML
classifiers (RF, DT, SVM) and ensemble classifier performs high level
classification of sentiment in tweets

Rest of the paper is organized in the following way; the literature
review  is  provided  in  Section  2.  The  methodology  of  the  proposed
technique is shown in Section 3. The results and analysis of the is dis-
cussed in Section 4 and finally, the conclusion of the paper is presented
in Section 5.

2. Related works

Sakirin Tam et.al (Tam et al., 2021) developed a combined frame-
work of Convolution neural network and Bidirectional Long Short-Term
Memory known as ConvBiLSTM which is a word embedding model that
converts  the  tweets  into  numerical  values.  The  CNN  present  in  the
framework  receives  embedded  features  as  an  input  and  provides  fea-
tures  with  small  dimensions.  The  Bi-LSTM  collects  the  data  from  the
CNN  layer  and  provides  classification  result.  The  framework  utilized
Word2Vec  and  GloVe  for  the  process  of  word  vectorization.  The
ConvBi-LSTM framework has the capability to capture both local and

global contextual sentence in Twitter. However, the word representation
present in the framework affects the classification accuracy of the whole
framework.

Anisha  P.  Rodrigues  and  Niranjan  N.  Chiplunkar  (Rodrigues  and
Chiplunkar,  2022)  developed  a  framework  based  on  Hybrid  Lex-
icon-Naïve  Bayes  Classifier  (HL-NBC)  for  classifying  emotion  of  the
tweets. Here, analysis of sentiment was based on classification of topics
where the tweets are classified and inappropriate tweets were filtered
out.  The  introduced  framework  utilized  Hadoop  framework  which
combines Apache Flume to gather the Twitter data and perform classi-
fication.  The  framework  based  on  HL-NBC  effectively  classifies  large
number  of  real  time  Twitter  data  and  analyze  the  concerned  tweets.
However, HL-NBC framework found difficulty in classifying the tweets
based on sarcastic opinions of the public.

Babacar  Gaye  et.al  (Gaye  et  al.,  2021)  have  presented  a  hybrid
stacked ensemble technique which was utilized in classifying the tweets
based on sentiments by using an ensemble based three Long Short-Term
Memory  (LSTM)  as  base  classifiers  and  Logistic  Regression  (LR)  as  a
meta classifier. In hybrid stacked ensemble technique, the tweets clas-
sification takes place with the sentiments obtained from Text Blob and
original sentiments. The hybrid stacked ensemble technique was effec-
tive with less consumption of time because it doesn’t need any feature
extraction  method.  However,  the  suggested  model  does  not  suit  for
classifying of sarcastic texts, fake reviews etc.

Muhammad Umer et.al (Umer et al., 2022) have introduced an Extra
Tree Convolutional Neural Network (ETCNN) based ensemble model for
classification of sentiment based on tweets related to COVID-19. The ET
and CNN models are combined using the methodology of soft voting. A
large  dataset  in  an  unstructured  form  was  obtained  from  publicly
available dataset, then the data was pre-processed and annotated using
TextBlob  and  VADER.  The  ETCNN  based  ensemble  model  utilized
TF-IDF  and  Word2Vec  model  to  perform  precise  classification  with
various ML models. However, the ETCNN based ensemble model was
limited for analysis of three feature approaches and it does not fit for
approaches based on annotations.

Mehmet Umut Salur and Ilhan Aydin (Salur and Aydin, 2020) have
introduced  a  deep  learning  model  which  combines  word  embedding
models  with  various  deep  learning  models  such  as  Long  Short-Term
Memory  (LSTM),  Gated  Recurrent  Unit  (GRU),  Bidirectional  Long
Short-Term  Memory  (Bi-LSTM)  or  Convolutional  Neural  Network
(CNN). The features were extracted from various deep learning models
and the text was classified accordingly. The character-level embedding
and FastText embedding was performed  using the algorithms of CNN
and LSTM. The features collected from CNN and LSTM was combined
and transferred to the softmax layer for classification. Generally, deep
learning model was much effective for huge datasets but in the proposed
model the dataset used was minimal in size.

Jawad Khan et al (Khan et al., 2023) have introduced a Sentiment
and  Context  Aware  Attention-based  Hybrid  Deep  Neural  Network
(SCA-HDNN) model along with attention mechanism which effectively
figure  out  the  salient  features  in  the  text.  Initially,  integrated  wide
coverage sentiment lexicons are used to detect the sentiment features
and BERT model is used to detect the sentiment of the extracted words.
Moreover, the suggested model utilized attention mechanism that allots
weight to the features and finally, CNN is used to classify the sentiment
of  the  tweets  by  reducing  the  feature  dimensionalities.  However,  the
suggested approach was not suitable to analyze the sentiment of the text
at the aspect level.

Fang, X., Zhan, J. (Fang and Zhan, 2015) The authors made the study
that addresses sentiment polarity categorization, a basic topic in senti-
ment analysis. The study uses a selection of online product reviews from
Amazon.com  as  its  data  source.  A  proposed  process  for  categorizing
sentiment polarity has been presented, complete with step-by-step in-
structions. Both review-level and sentence-level categorization experi-
ments have been conducted using machine learning classifiers.

• Rezaul Haque, Naimul Islam, Mayisha Tasneem, Amit Kumar Das

InternationalJournalofCognitiveComputinginEngineering5(2024)170–177171V. KP et al.

(Haque  et  al.,  2023)  For  multi-class  SA,  a  novel  CNN-based  LSTM
network called CLSTM was proposed. It can increase the classification
score  by  extracting  more  contextual  data  from  Bengali  social  media
texts. Additionally, it provided a performance comparison analysis be-
tween DL and ML classifiers using the CLSTM model were suggested. The
team  developed  a  web  application  based  on  the  best-performing  ma-
chine  learning  algorithm,  LR,  using  the  Flask  framework.  The  model
called CLSTM was developed to categorize real-world social media text
sentiments into four groups based on their proposed model. Unveiled a
multi-class  dataset  of  42,036  social  media  comments  in  Bengali  that
were divided into four groups political, sexual, acceptable, religious.

Zulfadzli  Drus,  Haliyana  Khalid  (Drus  and  Khalid,  2019),  Their
research examines relevant literature produced between 2014 and 2019
in order to better comprehend the application of sentiment analysis in
social  media  platforms.  Sentiment  analysis  is  a  method  that  divides
opinions into positive, negative, and natural sentiments by extracting,
converting, and interpreting opinions from texts using Natural Language
Processing  (NLP)  .  In  order  to  better  understand  their  customers  and
make the required decisions to improve their products or services, the
majority of the prior studies used sentiment analysis in product or movie
reviews .

3. An ensemble based classifier to classify the sentiment of
tweets

Sentiment analysis is considered as an important tool to collect the
information regarding the public view in applications related to daily
life. The proposed work extracts sentiment from the tweets posted by
user’s different situations. Moreover, ensemble based classifier has the
ability to classify the emotion from text. The classification of sentiment
analysis  using  ensemble  classifier  is  diagrammatically  represented  in
Fig. 1

The sentiment analysis model evaluated emotion of the tweet from
Twitter  API which  recognize the text based  on its nature. Analysis of
sentiment is a significant task which helps to deliver the exact sentiment
of the tweets. The proposed model has different phases such as (i) pre-
processing, (ii) extraction of feature, (iii) weighted feature selection and
(iv) classification.

3.1. Data acquisition and pre-processing

The raw data obtained from Twitter API dataset (Zhao et al., 2020)
and SST-2 (Tam et al., 2021) dataset is pre-processed to remove large
amount of unwanted data which complicates the process of classifica-
tion. Twitter data is frequently inappropriate for direct analysis. So, the
data should be normalized to offer an improved format before utilizing
any  methodologies.  In  the  suggested  methodology,  pre-processing

techniques are used to eliminate unnecessary contents. As a result, the
data can be used by all learning methods. One of the key steps in data
preparation, sometimes referred to as data mining, is the transformation
of  raw  data  into  an  understandable  format.  There  are  more  errors  in
real-world data due to its irregularity, fragmentation, and lack of exact
trends or behaviors. Pre-processing the data is therefore a key strategy
for resolving this issue that arises during sentimental analysis. The data
pre-processing  aids  in  improving  the  classification  accuracy  of  the
model. The steps involved in pre-processing is diagrammatically repre-
sented in Fig. 2 as follows

i)  Removal of stop words: The redundantly used words such as
adverbs,  prepositions,  articles  are  neglected  from  the  tweets.
Moreover, by removing these words, the dataset’s dimensionality
can be reduced. The removal of stop words aids in enhancement
of classification process.

ii)  Removal of blank space: The blank space present in between
the tweets must be removed, since the unnecessary blank space
consumes more time for classification.

iii)  Removal  of  punctuations:  Characters  such  as  “.”  and  "&"  is
detected  in  the  tweets  of  Twitter  which  must  be  removed  to
enhance the classification efficiency. The tags and punctuations
present in the data is removed and it is provided as input for the
feature selection process.

The words must be transformed into lower case since analysis of text
is case sensitive. When this process is not performed, the deep learning
model compute two words (for example, ‘Extraordinary’  and ‘extraor-
dinary’) individually which may affect the performance of the classifier.

3.2. Feature extraction

The pre-processed output undergoes the stage of feature extraction
where the features are separated using three types of feature extraction
process such as Bag of n-gram method, word 2 vector method and Term
Frequency –  Inverse Document Frequency (TF-IDF) method. The brief
description of these methods are provided in this section.

The bag of n-grams is used to check that continuous words exist in
textual  data  for  which  the  sentiment  should  be  analyzed.  This  model
consists of three main types: unigrams, bigrams, and trigrams. A single
word is represented by a unigram, a pair of words is represented by a
bigram,  and  three  or  more  words  are  represented  by  a  trigram.  The
words extracted using bag of n-gram is denoted as TSE
r  phrases
r
that  were  extracted  are  combined  with  the  word2vector  approach  to
produce a word vector representation. It is possible to identify and learn
the vector representation of each word. Word2vec offers the Continuous

. The TSE

Fig. 1. Process involved in classifying the sentiment from tweets.

InternationalJournalofCognitiveComputinginEngineering5(2024)170–177172V. KP et al.

Fig. 2. Steps in pre-processing.

Bag-Of-Words (CBOW) and skip-gram models as its two training tech-
niques.  This  technique  determines  the  degree  of  textual  similarity
among  the  words  present  in  the  certain  file.  Each  word  identifies  its
originality based on a bag of words, demonstrating the lack of contextual
association among two words. The extracted sentences are provided to
TF-IDF, which is one of the best techniques for removing the important
information from tweets. The value of TF is evaluated using the Eq. 1 as
follows:

T = NS
TN

(1)

Where the total number of terms is represented as .

The significant term is measured by utilizing IDF that specifies sig-
nificant terms present in the tweets. The IDF value is represented in Eq.
(2) as follows:

IDF = loge

(

)

ND
TD

(2)

Where  number  of  tweets  is  represented  by  TD  and  number  of  terms
present  in  tweet  is  represented  by  ND. At  last,  the  weighted  term  is
computed using the Eq. 3.

TF (cid:0)

IDF(s, DC) = TF(s, DC) × IDF(s)

(3)

Where s represents number of terms and DC represents text.

3.4. Feature selection

The features obtained after the process of dimensionality reduction
using  LDA  undergoes  the  stage  of  feature  selection.  The  process  of
reducing the amount of input variables while creating a model is known
as feature selection. By limiting the amount of input variables, feature
selection improves the model overall performance while also lowers the
computing costs. By neglecting the redundant information, the number
of input variables are lowered. The three types of feature selection al-
gorithms are filter technique, wrapper method, and embedding method.
Before applying a classification algorithm, the filter method performs
automatically to ignore features that look out of place. In the paper, the
characteristics with high calculated values are taken into account and
the features with less values are ignored.

The  Wrapper  technique  is  utilized  for  the  development  of  feature
selection  in  selecting  the  appropriate  features.  The  iterative  wrapper
approach  chooses  a  subset  of  characteristics  with  each  iteration.  The
number  of  iterations  rises  whenever  the  dataset  has  a  high  rate  of
dimensional  data.  Wrapper  approach  select  outstanding  features  and
provide excellent accuracy rates. Each feature in embedded type feature
selection model has an evaluation score. The high features are consid-
ered, and the features with low score is neglected. The wrapper method
requires less cost for computation and chooses exceptional features to
provide enhanced accuracy rate when related with existing techniques.

3.3. Dimensionality reduction using LDA

3.5. Classification using ensemble method

The selected features using Wrapper based technique is proceeded
for classification using ensemble classifier. Ensemble based method is a
collection of algorithms which is utilized for classification purpose. In
the  paper,  existing  ML  classifiers  are  combined  and  classification  is
performed using adaptive boosting method. An ensemble method was
utilized  for  classifying  the  tweets  based  on  their  sentiments.  Here,
Ensemble  method  is  a  combination  of  Random  Forest  (RF),  Decision
Tree  (DT),  and  Support  Vector  Machine  (SVM)  classifiers.  Generally,
ensemble based method exhibits better accuracy value when compared
with the existing algorithms. The process involved in Ensembling model
is diagrammatically represented in Fig. 3 as follows

3.5.1. Random Forest (RF)

Random Forests (RF) is a well-known method which is utilized for
the  classification  and  regression  problems.  It  has  developed  into  a
method to ensemble learning that is built on several decision trees. A
large number of decision trees are built during training period by RF

The  extracted  features  undergo  the  process  of  dimensionality
reduction where LDA focuses on creating a dataset with more charac-
teristics  and  strong  separability,  which  aids  in  cutting  down  on
computing  expenses.  LDA  supports  to  acquire  less  dimensional  space
while maximizing multi-class separations without compromising class
information. The objective of LDA is signified using Eq. 4 – Eq. 6 rep-
resented as follows:

aopt = argmax

aTSba
aTSwa

Where,

Sb =

∑c

i=1

ni(μi

(cid:0) μ)(μi

(cid:0) μ)T

Sw =

∑c

∑ni

(cid:0)

i=1

j=1

xj (cid:0) μi

)(cid:0)

)
T

xj (cid:0) μi

(4)

(5)

(6)

Where number of classes is denoted as c, μi is represented as mean and μ
is  denoted  as  total  mean  values.  The  between  class  scatter  matrix  is
represented as Sb  and Sw  is denoted as within class scatter matrix (Tam
et al., 2021). By enhancing the value for Eq. 2, the LDA’s efficiency will
be increased. Eigenvectors a for largest Eigen values is obtained from Eq.
7.

Sba = λSwa

(7)

In categorization of text, the issues arise in matrix Sw, since group of

terms in text is higher when related to total texts.

Fig. 3. Process involved in Ensemble model.

InternationalJournalofCognitiveComputinginEngineering5(2024)170–177173V. KP et al.

classifiers, which then provide a representation mode for classification.
By  decreasing  the  correlation  between  randomly  chosen  trees,  this
classifier can address the issue of overfitting and aids in boosting pre-
diction accuracy. In RF classifier with high dimensional feature space
provides  less  accuracy,  so  more  number  of  random  tress  must  be
generated.  The  general  procedure  to  construct  the  RF  model  with
training data D is mentioned in Eq. (8).
}

{(

D =

fi,Ci

)

N

i=1

⃒
⃒
⃒
⃒fi ∈ RF, C ∈ {1, 2, …, c}

(8)

Where the features are represented as fi, the set of classes and the count
of samples used in training is represented as Ci  and N respectively. Each
decision tree present in the RF classifier contains set of bagged samples
D1, D2,…., DP. The prediction of RF approach is defined as follows:

̂
C = majority

{

P}P
1

̂
C

(9)

Where the prediction class is represented as

P.

̂
C

In  this  paper,  RF  is  a  significant  classifier  which  provides  an
outstanding accuracy in the process of classifying the sentiments. The RF
classifier consist of hyper parameters such as number of trees, features,
tree depth acts as a significant parameter which helps in maintain better
accuracy.

3.5.2. Decision Tree (DT)

The  DT  classifier  performs  based  on  categorizing  the  region  into
multiple sub parts. DT classifier make use of every possible solution to
provide significant interaction among the variables. The classes present
in DT consist of two different categories of similar classes such as purity
class  and  impurity  classes.  The  impurity  class  present  in  DT  must  be
rectified  using  entropy  concept.  The  performance  of  DT  is  based  on
degree of the particular degree and this can be evaluated using Eq. (10)
as follows:

independent  inducer  is  constructed  independently  which  is  differed
from other inducers. The ensemble model exhibits better classification
accuracy  because  it  combines  the  classification  algorithms  which  is
utilized  to  perform  effectively.  This  method  evaluates  the  prediction
efficiency  of the  individual classifiers  and  combines  it using  adaptive
boosting technique.  The AdaBoosting  technique is less  prone  to over-
fitting, since the parameters are not jointly optimized. So, this research
utilized AdaBoost technique for the process of Ensemble which aids in
better classification accuracy.

3.6.1. Adaptive boosting (AdaBoost) algorithm

The adaptive boosting is a boosting technique which is created for
the  purpose  of  binary  classification.  AdaBoost  algorithm  uses  the
concept  of  boosting  which  is  ensemble  to  generate  a  robust  classifier
from the weaker classifiers. AdaBoost has capability to improve overall
efficiency of ML classifiers. Adaboost combines the poor classifiers and
extracts  the  prediction  value  to  make  a  better  classifier  known  as
ensemble classifier. The process of combining multiple classifiers along
with the training set produce weight in the final stage of voting. Ada-
boost  technique  considers  two  techniques  such  as  training  the  subset
with weak classifier and a random subset must be utilized for training
the  whole  group.  Secondly,  a  weight  factor  must  be  assigned  to  the
subset.  The  classifier  with  50  %  accuracy  provides  zero  weight  and
classifier with accuracy less than 50 % exhibits negative weight for the
classifier.

H(X) = SIGN

)

αt ht (x)

(

∑t

t=1

(12)

Where the output of the classifier for input x is represented as ht (x) and
the weight assigned to the classifier is denoted as αt.

The value of αt can be computed using the formula mentioned in Eq.

(13) as follows
(

)

(13)

1 (cid:0) E
E

∑

H = (cid:0)

p(x)logp(x)

αt = 0.5 × ln

(10)

Where the probability of the element id represented as p(x).

Where the error rate produced is denoted as E.

3.5.3. Support Vector Machine (SVM)

4. Results and analysis

SVM  accomplishes  binary  classification  for  linear  and  non-linear
versions. In general, the datasets are in linearly closed form, so main
objective of SVM is to capture the available surfaces and categorize the
samples into positive, neutral and negative on the basis of minimization
principal. SVM classifiers define the boundaries in a high dimensional
feature space. The hyper-plane classifies the vectorized text into three
categories such as positive, neutral and negative. The issues related to
optimization can be reduced using the Eq. (11) as follows:

{

α→∗

= argmin

(cid:0)

∑n

αi +

∑n

∑n

i=1

i=1

j=1

〉}

〈

→
x

→
i, x
j

αiαjyiyj

(11)

Where the feature vector is represented as x and the dimension of the
vector is denoted as d. The value of y lies among the range from {1, (cid:0) 1}.
The SVM classifier has the capability to categorize the dataset in an
individual  hyper  plane  and  in  non-linear  dataset  kernel  functions  are
utilized in a high dimensional spaces.

The overall efficiency of the developed ensemble method is evalu-
ated and the comparison is done with existing methodologies described
in related works.

4.1. Datasets

Twitter API dataset consist of tweets regarding users, spaces, direct
messages, lists, trends, media, places. Twitter API is one of the public
accessible dataset that contributes with the applications regarding real
world. Moreover, Twitter API dataset consist of tweets with long sen-
tences,  stop  words,  misspelled  words  etc.  To  remove  these  a  pre-
processing technique must be performed to ease the process of classifi-
cation.  Secondly,  the  data  is  collected  from  binary  labeled  Stanford
Sentiment  Treebank  (SST-2)  which  is  comprised  with  data  based  on
training, testing and validation with 67,349, 872 and 1821 sentences
respectively.

3.6. Ensemble model

4.2. Experimental setup

Ensemble methods are familiar in machine learning and recognition
of patterns and the ensemble methods combine the output of the weaker
algorithms to enhance the accuracy and classification efficiency of the
model. Ensemble methods are generally categorized into two types such
as dependent frame and independent frame. In dependent framework,
the  output  of  the  individual  inducer  affects  the  next  process  and  the

The system utilized in the classification process has i5 processor at
3.40 GHz on Ubuntu 16.4 OS with 8 GB RAM. The experiment is pro-
cessed for the tweets from Twitter API dataset. The performance of the
classifier is evaluated based on accuracy, precision, recall and f-1 score.

InternationalJournalofCognitiveComputinginEngineering5(2024)170–177174V. KP et al.

4.3. Performance metrics

The  performance  of  the  ensemble  classifier  in  classification  of
sentiment is evaluated based on accuracy, precision, recall and F 1 score.
Accuracy: It is stated as the total count of properly categorized data

to total count of data.

Accuracy =

TN + TP
TN + FP + TP + FN

(14)

Where, TN denotes true negative, TP denotes true positive, FP denotes
false positive and FN denotes false negative.

Precision: It is the total number of truly positive values to the total

number of positives.

Precision = TP

TP + FP

(15)

Fig. 4. Graphical representations for performance of classifiers.

Recall: It is defined as the ratio among the total count of Positives
that is properly categorized as Positive to total samples of positive and
negative values.

Table 2
Statistical analysis.

Recall = TP

TP + FN

(16)

Classifiers

RF
classifier

DT
classifier

SVM
classifier

Ensemble
classifier

F1-score: It is evaluated by using both recall and precision. when

values of precision and recall are 1, the F1 score will be one.
F1 score = 2 × Precision ∗ Recall
Precision + Recall

(17)

Best
Worst
Mean
Median
Standard

deviation

0.932
0.921
0.942
0.936
0.008

0.945
0.926
0.939
0.934
0.009

0.952
0.927
0.938
0.936
0.007

0.971
0.963
0. 965
0.947
0.005

4.4. Performance analysis

The proposed ensemble classifier is a combination of SVM, DT and
RF. The performance of the proposed ensemble classifier is compared
with the individual performance of SVM, DT and RF. Table 1 represented
below represents the performance of the proposed ensemble classifier in
classifying the sentiment of tweets.

Results from Table 1 and Fig. 4 shows that the ensemble classifier
attained  better  accuracy  of  93.42  %  during  classification  that  is
comparatively higher than existing ML classifiers. The better result is
due to the capability of ensemble classifier in combining the individual
predictions  of  RS,  SVM  and  DT.  An  ensemble  classifier  combines  the
predicted  values  and  provide  it  for  the  adaptive  boosting  technique
which  uses  the  concept  of  boosting  which  is  ensemble  to  generate  a
robust classifier from the weaker classifiers. AdaBoost has capability to
improve the overall efficiency of ML classifiers.

4.5. Statistic analysis

It is evaluated for ensemble classifier to evaluate the performance of
proposed  model  with  diverse  classification  models.  The  statistical
analysis is processed by considering the statistical metrics such as best,
worst, mean and standard deviation. The Table 2 and Fig. 5 presents the
statistical graph representation of the proposed classifier with existing
classification approaches.

The results from Table 2 and Fig. 5 shows that the proposed ensemble
classifier  effectively  classifies  the  statistical  value  of  tweets  by  evalu-
ating mean, median, worst and best values. By using ensemble classifier,
the classification process has been done effectively with less complex-
ities Fig. 6.

Table 1
Performance of different classifiers.

Classifiers

Accuracy (%)

Precision (%)

Recall (%)

F1 score (%)

RF classifier
DT classifier
SVM classifier
Ensemble classifier

91
89
86
93.42

87
83
89
91.29

80
85
78
89.65

81
84
83
87.32

Fig. 5. Graphical representation for statistical analysis.

Fig. 6. Graphical representations for comparison of classifiers for Twitter API.

InternationalJournalofCognitiveComputinginEngineering5(2024)170–177175V. KP et al.

4.6. Performance evaluation for SST-2 dataset

In this section, the performance of the ensemble classifier is evalu-
ated  with existing classifiers  such as RF,  DT and  SVM classifiers. The
data is collected from the SST-2 dataset is used to evaluate the perfor-
mance  of  the  proposed  classifier  with  existing  classification  methods.
The results obtained from the evaluation of classifier with the existing
approaches is presented in Table 3 as follows:

The  results  from  the  Table  3  shows  that  the  proposed  ensemble
classifier  have  achieved  better  classification  results.  The  accuracy  of
sentiment  classification  for  ensemble  classifier  is  96.53  %  which  is
comparatively higher  than the  existing classifiers  such as RF, DT and
SVM with 94.64 %, 91.56 % and 93.71 % respectively. The Fig. 7 shows
the  graphical  representation  of  evaluation  of  classification  for  SST-2
dataset Table 4.

4.7. Comparative analysis

This  section  provides  the  comparative  analysis  of  the  proposed
ensemble classifier with the existing Convolutional Bidirectional - Long
Short-Term  Memory  (ConvBiLSTM)  classifier  (Tam  et  al.,  2021),
HL-NBC (Rodrigues and Chiplunkar, 2022) and SCA-HDNN with BERT
(Khan et al., 2023). Table 4 represents the comparison of the proposed
ensemble  method  with  the  existing  classifiers  described  in  related
works.

The  results  from  Table  4  shows  that  ensemble  classifier  attained
better  performance in overall metrics. The results  are evaluated from
Twitter  API  and  SST-2  datasets,  the  proposed  ensemble  classifier
attained  better  accuracy  of  96.53  %  for  Twitter  API  dataset  that  is
comparatively  higher  than  ConvBiLSTM  (88.47  %)  %  and  HL-NBC
(89.61  %).  Similarly,  for  SST  2  dataset,  the  proposed  approach  ach-
ieves classification accuracy of 96.53 % which is comparatively higher
than the existing ConvBiLSTM with the accuracy percentage of 88.47
and SCA-HDNN with accuracy of 92 %. The better result of the proposed
approach is due to the process of Ensembling using adaptive boosting
technique where the prediction of best values from the individual fea-
tures of SVM, RF and DT is provided in adaptive boosting technique to
get better classification output.

5. Conclusion

Dependence on social media is unavoidable because it helps the users
to know about the day-to-day activities takes place in the world. Spe-
cifically, Twitter relies as a widely used platform which helps the people
to  share  their  opinion  on  different  topics.  The  sentiment  analysis  of
tweets is a challenging process because Twitter consist of vast number of
data.  So,  this  research  introduced  an  ensemble  classifier  with  BERT
mechanism  to  classify  the  sentiment  of  the  tweets  based  on  their  po-
larities. This research used Wrapper based technique for selecting the
relevant features and the Wrapper technique evaluates score for each
feature, the feature with low score is ignored and high score is proceeded
for  classification  using  ensemble  classifier.  The  proposed  ensemble
classifier which is a combination of ML classifiers such as SVM, RF and
DT. The individual prediction values are gathered from the ML classi-
fiers are fed to the AdaBoost technique that combines the poor classifiers
and  extracts  the  better  prediction  value  to  make  a  better  classifier.
AdaBoost has capability to improve the overall efficiency of ML classi-
fiers.  The  proposed  ensemble  classifier  effectively  classifies  the  senti-
ment  on  the  basis  of  paragraph-level  for  long  tweets.  The  developed
ensemble model achieves better accuracy of 93.42 % which is compar-
atively  better than existing ConvBiLSTM  classifier (91.53 %) and  HL-
NBC  classifier  (89.61  %).  However,  when  the  number  of  levels  get
increased in DT, the model become vulnerable and leads to high vari-
ance which effects overall efficiency of the ensemble classifier. In future,
the proposed ensemble classifier can be utilized to perform sentiment
analysis in different social platforms.

Table 3
Evaluating the classification performance for SST-2 dataset.

Classifiers

Accuracy (%)

Precision (%)

Recall (%)

F-1 score (%)

RF classifier
DT classifier
SVM classifier
Ensemble

classifier

94.64
91.56
93.71
96.53

89.93
92.82
91.39
95.87

90.06
88.13
85.32
91.24

90.21
89.99
88.62
93.91

Fig. 7. Graphical representation for evaluating the classification performance
for SST-2 dataset.

Table. 4
Comparative results for Twitter API and SST-2 dataset.

Classifiers

Datasets

Accuracy
(%)

Precision
(%)

Recall
(%)

F1
score
(%)

ConvBiLSTM (Tam

et al., 2021)

HL-NBC (Rodrigues
and Chiplunkar,
2022)
SCA-HDNN

With BERT (Khan
et al., 2023)

Ensemble classifier

Twitter
API
SST-2
Twitter
API

88.47

83.39

84.78

77.90

91.53
89.61

85.23
83.42

85.32
81.45

80.76
83.44

SST-2

92.0

90.8

93.0

91.9

Twitter
API
SST-2

93.42

91.29

89.65

87.32

96.53

95.87

91.24

93.91

Declaration of competing interest

The authors declare that they have no known competing financial
interests or personal relationships that could have appeared to influence
the work reported in this paper.

References

Akhtar, S., Ghosal, D., Ekbal, A., Bhattacharyya, P., & Kurohashi, S. (2019). All-in-one:
Emotion, sentiment and intensity prediction using a multi-task ensemble framework.
IEEE transactions on affective computing.

AlBadani, B., Shi, R., & Dong, J. (2022). A novel machine learning approach for

sentiment analysis on Twitter incorporating the universal language model fine-
tuning and SVM. Applied System Innovation, 5(1), 13.

Bibi, M., Arshad Abbasi, W., Aziz, W., Khalil, S., Uddin, M., Iwendi, C., & Gadekallu, T. R.

(2022). A novel unsupervised ensemble framework using concept-based linguistic
methods and machine learning for twitter sentiment analysis. Pattern Recognition
Letters, 158, 80–86.

Das, S., Das, D., & Kumar Kolya, A. (2020). Sentiment classification with GST tweet data

on LSTM based on polarity-popularity model. S¯adhan¯a, 45(1), 1–17.

Drus, Z., & Khalid, H. (2019). Sentiment Analysis in Social Media and Its Application:

Systematic Literature Review. Procedia Computer Science, 161, 707–714.

Fang, X., & Zhan, J. (2015). Sentiment analysis using product review data. Journal of Big

Data, 2, 5. https://doi.org/10.1186/s40537-015-0015-2

InternationalJournalofCognitiveComputinginEngineering5(2024)170–177176V. KP et al.

Gandhi, U. D., Kumar, P. M., Babu, G. C., & Karthick, G. (2021). Sentiment analysis on
twitter data by using convolutional neural network (CNN) and long short term
memory (LSTM). Wireless Personal Communications, 1–10.

Nistor, S. C., Moca, M., Moldovan, D., Oprean, D. B., & Nistor, R. L. (2021a). Building a

Twitter sentiment analysis system with recurrent neural networks. Sensors, 21(7),
2266.

Gaye, B., Zhang, D., & Wulamu, A. (2021). A Tweet sentiment classification approach

Nistor, S. C., Moca, M., Moldovan, D., Oprean, D. B., & Nistor, R. L. (2021b). Building a

using a hybrid stacked ensemble technique. Information, 12(9), 374.

Haque, R., Islam, N., & Tasneem, M. (2023). Amit Kumar Das, Multi-class sentiment

classification on Bengali social media comments using machine learning.
International Journal of Cognitive Computing in Engineering, (21-35).

Jianqiang, Z., Xiaolin, G., & Xuejun, Z. (2018). Deep convolution neural networks for

twitter sentiment analysis. IEEE access, 6, 23253–23260.

Khalid, M., Ashraf, I., Mehmood, A., Ullah, S., Ahmad, M., & Choi, G. S. (2020). GBSVM:
sentiment classification from unstructured reviews using ensemble classifier. Applied
Sciences, 10(8), 2788.

Khan, J., Ahmad, N., Khalid, S., Ali, F., & Lee, Y. (2023). Sentiment and Context-Aware

Hybrid DNN with attention for text sentiment classification. IEEE Access, 11,
28162–28179.

Kim, H., & Jeong, Y.-S. (2019). Sentiment classification using convolutional neural

networks. Applied Sciences, 9(11), 2347.

Minaee, Shervin, Elham Azimi, and AmirAli Abdolrashidi. "Deep-sentiment: Sentiment

analysis using ensemble of cnn and bi-lstm models." arXiv preprint arXiv:1
904.04206 (2019).

Naseem, U., Razzak, I., Musial, K., & Imran, M. (2020). Transformer based deep

intelligent contextual embedding for twitter sentiment analysis. Future Generation
Computer Systems, 113, 58–69.

Twitter sentiment analysis system with recurrent neural networks. Sensors, 21(7),
2266.

Rodrigues, A. P., & Chiplunkar, N. N. (2022). A new big data approach for topic

classification and sentiment analysis of Twitter data. Evolutionary Intelligence, 15(2),
877–887.

Salur, M. U., & Aydin, I. (2020). A novel hybrid deep learning model for sentiment

classification. IEEE Access, 8, 58080–58093.

Soumya, S., & Pramod, K. V. (2022). Hybrid deep learning approach for sentiment
classification of malayalam tweets. International Journal of Advanced Computer
Science and Applications, 13(4).
Tam, S., Ben Said, R., & Tanri¨over,

¨
O. (2021). A ConvBiLSTM deep learning model-
based approach for Twitter sentiment classification. IEEE Access, 9, 41283–41293.
Umer, M., Sadiq, S., Nappi, M., Sana, M. U., & Ashraf, I. (2022). ETCNN: extra tree and
convolutional neural network-based ensemble model for COVID-19 tweets sentiment
classification. Pattern Recognition Letters, 164, 224–231.

¨
O.

Zainuddin, N., Selamat, A., & Ibrahim, R. (2018a). Hybrid sentiment classification on
twitter aspect-based sentiment analysis. Applied Intelligence, 48(5), 1218–1232.
Zainuddin, N., Selamat, A., & Ibrahim, R. (2018b). Hybrid sentiment classification on
twitter aspect-based sentiment analysis. Applied Intelligence, 48(5), 1218–1232.
Zhao, C., Wang, S., & Li, D. (2020). Multi-source domain adaptation with joint learning
for cross-domain sentiment classification. Knowledge-Based Systems, 191, Article
105254.

InternationalJournalofCognitiveComputinginEngineering5(2024)170–177177

---

## Content from Previous Extraction (not in markitdown output)

### Visual/Chart/Graph Descriptions

#### 1

### Visual/Chart/Graph Descriptions

#### 2

A R T I C L E I N F O A B S T R A C T _Keywords:_ Social media users are more receptive to products or events and share their thoughts through raw textual data, Adaptive boosting which is classified as semi-structured data. This data, which is presented using a variety of terminologies, is noisy Ensemble classifier by nature but yet contains important information and superfluous details, giving analysts a way to identify Sentiment analysis Tweets Twitter API patterns and knowledge. This hidden information must be extracted from language data in order to make informed decisions and create strategic plans for entering new markets. Among the most prominent fields of study are natural language processing (NLP) and data mining techniques, especially when it comes to sentiment — analysis the process of identifying the feelings and insights concealed in the data. Twitter is one of the significant microblogging platform with millions of users. These users use Twitter to share sentiments using hash tags on different topics and to make status updates known as tweets. Twitter is therefore regarded as a significant real-time source and as one of the most active opinion indicators. The volume of information is produced by Twitter is enormous and manually scanning the entire data set is difficult process. The paper proposed an ensemble classifier to categorize emotion of the tweets on the basis of polarities such as positive and negative. In our study, we ensemble classifiers which is a combination of Random Forest (RF), Support Vector Machine (SVM) and Decision Tree (DT). The data is collected from Twitter API and the Twitter data is analysed autonomously to define public view on particular topic. The features obtained after the process of dimensionality reduction using LDA undergoes the stage of feature selection using Wrapper based technique. The iterative Wrapper based technique predict score for the features, the features with low score are ignored and high score is proceeded for classification. The ensemble classifier used Adaptive Boosting (Ada Boost) technique where the output from the Machine Learning (ML) classifiers are combined to produce a single output. Adaboost combines the poor classifiers and extracts the prediction value to make a better classifier. The experimental results show that the proposed ensemble classifier provides better accuracy of 93.42 % that is comparatively better than existing Convolutional Bidirectional - Long Short-Term Memory (Conv Bi LSTM) classifier and Hybrid Lexicon Naïve Bayes Classifier (HL-NBC) which produce classification accuracy of 91.53 % and 89.61 % respectively.

#### 3

**Fig. 4.** Graphical representations for performance of classifiers.

#### 4

**Fig. 5.** Graphical representation for statistical analysis.

#### 5

**Fig. 6.** Graphical representations for comparison of classifiers for Twitter API.

#### 6

**Fig. 7.** Graphical representation for evaluating the classification performance for SST-2 dataset.

### Additional Content

#### 1

# A tweet sentiment classification approach using an ensemble classifier

#### 2

> *Source PDF: A tweet sentiment classification approach using an ensemble classifier.pdf*
> *Extraction: Combined — markitdown raw text (base) + previous extraction supplements*

#### 3

## Content from Previous Extraction (not in markitdown output)

#### 4

International Journal of Cognitive Computing in Engineering 5 (2024) 170–177

#### 5

**==> picture [61 x 35] intentionally omitted <==**

#### 6

## Contents lists available at Science Direct

#### 7

## International Journal of Cognitive Computing in Engineering

#### 8

journal homepage: www.keaipublishing.com/en/journals/internationaljournal-of-cognitive-computing-in-engineering/

#### 9

**==> picture [58 x 72] intentionally omitted <==**

#### 10

## A tweet sentiment classification approach using an ensemble classifier

#### 11

**==> picture [29 x 30] intentionally omitted <==**

#### 12

## Vidyashree KP[a] , Rajendra AB[a] , Gururaj HL[b] , Vinayakumar Ravi[c][,][*] , Moez Krichen[d][,][e ]

#### 13

a _Department of Information Science and Engineering, Vidyavardhaka College of Engineering, Mysuru, India_

#### 14

b _Department of Information Technology, Manipal Institute of Technology Bengaluru, Manipal Academy of Higher Education, Manipal, India_

#### 15

c _Center for Artificial Intelligence, Prince Mohammad Bin Fahd University, Khobar, Saudi Arabia_

#### 16

d _Department of Information Technology, Faculty of Computer Science and Information Technology (FCSIT), Al-Baha University, Alaqiq, 65779-7738, Saudi Arabia_ e _Re DCAD Laboratory, University of Sfax, Sfax, 3038, Tunisia_

#### 17

_E-mail address:_ vravi@pmu.edu.sa (V. Ravi).

#### 18

Received 3 May 2023; Received in revised form 16 March 2024; Accepted 25 April 2024 Available online 1 May 2024

#### 19

2666-3074/© 2024 The Authors. Publishing Services by Elsevier B.V. on behalf of Ke Ai Communications Co. Ltd. This is an open access article under the CC BY license (http://creativecommons.org/licenses/by/4.0/).

#### 20

_International Journal of Cognitive Computing in Engineering 5 (2024) 170–177_
