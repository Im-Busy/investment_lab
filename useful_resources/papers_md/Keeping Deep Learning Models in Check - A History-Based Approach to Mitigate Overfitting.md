# Keeping Deep Learning Models in Check - A History-Based Approach to Mitigate Overfitting

> *Source PDF: Keeping Deep Learning Models in Check - A History-Based Approach to Mitigate Overfitting.pdf*
> *Extraction: Combined — markitdown raw text (base) + previous extraction supplements*

---

|     | Keeping       |     |     | Deep |     | Learning |     |     | Models |          | in  | Check:      |     | A   |     |
| --- | ------------- | --- | --- | ---- | --- | -------- | --- | --- | ------ | -------- | --- | ----------- | --- | --- | --- |
|     | History-Based |     |     |      |     | Approach |     |     | to     | Mitigate |     | Overfitting |     |     |     |
Hao Li∗, Gopi Krishnan Rajbahadur†, Dayi Lin†, Cor-Paul Bezemer∗, and Zhen Ming (Jack) Jiang‡
|             |     |          |     |              | ∗University |          | of Alberta. |         | {li.hao, bezemer}@ualberta.ca, |     |     |                       |     |     |     |
| ----------- | --- | -------- | --- | ------------ | ----------- | -------- | ----------- | ------- | ------------------------------ | --- | --- | --------------------- | --- | --- | --- |
|             |     | †Centre  |     |              |             |          |             |         | {gopi.krishnan.rajbahadur1,    |     |     | dayi.lin}@huawei.com, |     |     |     |
|             |     |          | for | Software     | Excellence, |          | Huawei      | Canada. |                                |     |     |                       |     |     |     |
|             |     |          |     |              |             | ‡York    | University. |         | zmjiang@cse.yorku.ca           |     |     |                       |     |     |     |
| Abstract—In |     | software |     | engineering, | deep        | learning | models      | are     |                                |     |     |                       |     |     |     |
increasinglydeployedforcriticaltaskssuchasbugdetectionand Training loss Training loss
codereview.However,overfittingremainsachallengethataffects Validation loss Validation loss
4202 naJ 81  ]ES.sc[  1v95301.1042:viXra
| the | quality, | reliability, | and | trustworthiness |     | of software |     | systems |     |     |     |     |     |     |     |
| --- | -------- | ------------ | --- | --------------- | --- | ----------- | --- | ------- | --- | --- | --- | --- | --- | --- | --- |
thatutilizedeeplearningmodels.Overfittingcanbe(1)prevented
(e.g.,usingdropoutorearlystopping)or(2)detectedinatrained
model(e.g.,usingcorrelation-basedapproaches).Bothoverfitting
detectionandpreventionapproachesthatarecurrentlyusedhave
| constraints |                | (e.g., | requiring   | modification |         | of the | model structure, |     |     |                 |     |     |                     |     |     |
| ----------- | -------------- | ------ | ----------- | ------------ | ------- | ------ | ---------------- | --- | --- | --------------- | --- | --- | ------------------- | --- | --- |
|             |                |        |             |              |         |        |                  |     |     | (a) Overfitting |     |     | (b) Non-overfitting |     |     |
| and         | high computing |        | resources). |              | In this | paper, | we propose       |     | a   |                 |     |     |                     |     |     |
simple, yet powerful approach that can both detect and prevent Fig. 1: Examples of overfit and non-overfit training histories.
| overfitting |          | based | on the | training | history | (i.e., validation |     | losses). |     |     |     |     |     |     |     |
| ----------- | -------- | ----- | ------ | -------- | ------- | ----------------- | --- | -------- | --- | --- | --- | --- | --- | --- | --- |
| Our         | approach | first | trains | a time   | series  | classifier        | on  | training |     |     |     |     |     |     |     |
historiesofoverfitmodels.Thisclassifieristhenusedtodetectif
| a trained | model       | is      | overfit.    | In addition, | our         | trained       | classifier | can      |                  |           |               |           |                   |           |          |
| --------- | ----------- | ------- | ----------- | ------------ | ----------- | ------------- | ---------- | -------- | ---------------- | --------- | ------------- | --------- | ----------------- | --------- | -------- |
|           |             |         |             |              |             |               |            |          | layers [69]      | or batch  | normalization |           | [31]. However,    |           | many of  |
| be used   | to          | prevent | overfitting | by           | identifying | the           | optimal    | point    |                  |           |               |           |                   |           |          |
|           |             |         |             |              |             |               |            |          | these approaches | are       | intrusive     | and       | require modifying |           | the data |
| to stop   | a           | model’s | training.   | We           | evaluate    | our           | approach   | on its   |                  |           |               |           |                   |           |          |
|           |             |         |             |              |             |               |            |          | or the model     | structure | and           | expertise | to execute        | correctly | and      |
| ability   | to identify |         | and prevent | overfitting  |             | in real-world |            | samples. |                  |           |               |           |                   |           |          |
We compare our approach against correlation-based detection even then, they may not work. For instance, adding dropout
approaches and the most commonly used prevention approach layers, a popularly used overfitting prevention scheme, when
(i.e., early stopping). Our approach achieves an F1 score of 0.91 set with a lower threshold or when added to the earlier
| which         | is at | least       | 5% higher | than      | the         | current      | best-performing |     |            |                     |            |     |             |                    |          |
| ------------- | ----- | ----------- | --------- | --------- | ----------- | ------------ | --------------- | --- | ---------- | ------------------- | ---------- | --- | ----------- | ------------------ | -------- |
|               |       |             |           |           |             |              |                 |     | layers may | cause unintentional |            |     | overfitting | [41]. Furthermore, |          |
| non-intrusive |       | overfitting |           | detection | approach.   | Furthermore, |                 | our |            |                     |            |     |             |                    |          |
|               |       |             |           |           |             |              |                 |     | even the   | non-intrusive       | prevention |     | approaches  | such               | as early |
| approach      | can   | stop        | training  | to avoid  | overfitting |              | at least        | 32% | of         |                     |            |     |             |                    |          |
thetimesearlierthanearlystoppingandhasthesameorabetter stoppingincurtrade-offsbetweenmodelaccuracyandtraining
rate of returning the best model. time [51]. For example, late stopping when using the early
Index Terms—Software engineering for AI, AI for Software stopping approach may improve model accuracy, but it will
| Engineering, |     | Overfitting, |     | Training | history, | Deep | learning |     |               |                  |       |             |              |     |           |
| ------------ | --- | ------------ | --- | -------- | -------- | ---- | -------- | --- | ------------- | ---------------- | ----- | ----------- | ------------ | --- | --------- |
|              |     |              |     |          |          |      |          |     | also increase | training         | time. | Conversely, | stopping     |     | too early |
|              |     |              |     |          |          |      |          |     | could result  | in a sub-optimal |       | model       | performance. |     |           |
I. INTRODUCTION
|     |     |     |     |     |     |     |     |     | Overfitting | detection |     | approaches | like | k-fold | cross- |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ----------- | --------- | --- | ---------- | ---- | ------ | ------ |
The use of Deep Learning (DL) models in software engi- validation, training the DL model with noisy data points
neering(SE)researchandsoftwareproductshasbeenskyrock- and observing if the added noise impacts the DL model’s
eting over the past decade. For instance, DL techniques have accuracy[87],checkingifthehypothesisofthetrainedmodel
beenusedforautomatedbugdetection[24],codereview[61], and the data are independent [81] can generally be resource
and software testing [73]. These applications underscore the intensive and time consuming. For instance, Xu et al. [84]
importance and ubiquity of DL in modern SE. report that training a DL model to find two semantically
OverfittingisoneofthefundamentalissuesthatplaguesDL linkable questions in StackOverflow takes about 14 hours.
models [33, 49, 67, 79, 85]. A DL model can be considered If one were to conduct a 5-fold cross-validation to detect if
overfitting if the model fits just the training data instead of the constructed DL model is overfitting, they would have to
learningthetargethypothesis[79].Anoverfitmodelincreases invest 70 hours, which might be prohibitive in practice.
the risk of inaccurate predictions, misleading feature impor- In this paper, we introduce OverfitGuard, an approach
tance, and wasted resources [27]. to both detect and prevent overfitting using training histories.
Currently, the problem of overfitting is addressed in SE Figure1illustratesexampletraininghistories(i.e.,thetraining
studies that use DL models by either (1) preventing it from and validation losses curves) of an overfit and a non-overfit
happening in the first place or (2) detecting it in a trained DL model. The training and validation losses of the overfit
DL model [79, 85]. Overfitting prevention approaches in- model both decrease at the beginning of the training process.
clude early stopping [48], data augmentation [64], regulariza- Followingthat,thevalidationlossincreaseswhilethetraining
tion [35], and modifying the DL model by adding dropout loss decreases, resulting in a large gap between the training

and validation losses. Such a trend indicates poor general- TABLE I: Studied time series classifiers.
| ization    | of the   | trained | model    | to new    | data. | Researchers     | have |            |     |             |     |     |     |     |     |
| ---------- | -------- | ------- | -------- | --------- | ----- | --------------- | ---- | ---------- | --- | ----------- | --- | --- | --- | --- | --- |
|            |          |         |          |           |       |                 |      | Classifier |     | Description |     |     |     |     |     |
| previously | employed |         | training | histories | for   | decision-making |      |            |     |             |     |     |     |     |     |
KNN-DTW∗
in areas such as quantitative data acquisition and model Uses K-Nearest Neighbors [25] with Dynamic Time
Warping[17]asthedistancemetrictoclassifytimeseries
selection [8, 9, 29, 45, 46, 70, 76]. Similarly, our approach data
trainsatimeseriesclassifieronasimulateddatasetoftraining HMM-GMM Uses Hidden Markov Model for modelling time series
dataandGaussianMixtureModelastheemissionsprob-
| histories | (i.e., | labelled | validation | loss | curves | over | epochs of |     |     |     |     |     |     |     |     |
| --------- | ------ | -------- | ---------- | ---- | ------ | ---- | --------- | --- | --- | --- | --- | --- | --- | --- | --- |
abilitydensity[23,32]
training) of DL models that overfit the training data. Our TSF Uses a random forest [6] for time series data using an
trainedtimeseriesclassifierdetectsoverfittinginatrainedDL ensembleoftimeseriestrees[16]
|          |           |     |                |     |              |           |     | TSBF |     | Time Series | Bag-of-Features |     | [5] extracts | features | based |
| -------- | --------- | --- | -------------- | --- | ------------ | --------- | --- | ---- | --- | ----------- | --------------- | --- | ------------ | -------- | ----- |
| model by | examining |     | the validation |     | loss history | (captured | as  |      |     |             |                 |     |              |          |       |
onthebag-of-featuresapproach[21]tocreatearandom
part of the training history). In contrast to existing overfitting forest
|           |             |           |              |     |          |       |            | SAX-VSM |     | Symbolic  | Aggregate | approXimation   |     | [38] converts | the   |
| --------- | ----------- | --------- | ------------ | --- | -------- | ----- | ---------- | ------- | --- | --------- | --------- | --------------- | --- | ------------- | ----- |
| detection | approaches, |           | our approach |     | does not | incur | additional |         |     |           |           |                 |     |               |       |
|           |             |           |              |     |          |       |            |         |     | data into | symbolic  | representations |     | and Vector    | Space |
| resources | or          | costs, as | the training |     | history  | (also | known as   |         |     |           |           |                 |     |               |       |
Model[50,57]transformsthemintovectorstocalculate
| the learning |     | curve) is | a natural | byproduct |     | of the | training |     |     |     |     |     |     |     |     |
| ------------ | --- | --------- | --------- | --------- | --- | ------ | -------- | --- | --- | --- | --- | --- | --- | --- | --- |
similarityforclassification
|     |     |     |     |     |     |     |     | BOSSVS |     | Bag-of-SFA | Symbols | in Vector | Space | [59] is similar | to  |
| --- | --- | --- | --- | --- | --- | --- | --- | ------ | --- | ---------- | ------- | --------- | ----- | --------------- | --- |
process.Furthermore,ourapproach(i.e.,thetrainedtimeseries
SAX-VSMbutuseSFA[60]totransformthedatainstead
| classifier) | can | be used | to prevent |     | overfitting | based | on the |     |     |     |     |     |     |     |     |
| ----------- | --- | ------- | ---------- | --- | ----------- | ----- | ------ | --- | --- | --- | --- | --- | --- | --- | --- |
ofSAX
| validation | losses | of recent | epochs | (e.g., | the | last 20 | epochs). |     |     |     |     |     |     |     |     |
| ---------- | ------ | --------- | ------ | ------ | --- | ------- | -------- | --- | --- | --- | --- | --- | --- | --- | --- |
∗ KNN-DTWhandlesvariable-lengthtimeseriesdata.
| Although    | our | approach        | is  | trained  | on a      | simulated | dataset, |     |     |     |     |     |     |     |     |
| ----------- | --- | --------------- | --- | -------- | --------- | --------- | -------- | --- | --- | --- | --- | --- | --- | --- | --- |
| we evaluate | it  | on a real-world |     | dataset, | collected | from      | papers   |     |     |     |     |     |     |     |     |
publishedintopAIvenueswithinthelast5years.Wegathered
Typically,adatasetisdividedintotraining,validation,andtest
| the training | histories |     | from these | papers |     | that are | explicitly |     |     |     |     |     |     |     |     |
| ------------ | --------- | --- | ---------- | ------ | --- | -------- | ---------- | --- | --- | --- | --- | --- | --- | --- | --- |
labelled as overfitting or non-overfitting by the authors as sets. While the training loss reflects how well the DL model
|            |        |     |                    |     |     |            |        | learns from | the | training | data during | the | training | process, | the |
| ---------- | ------ | --- | ------------------ | --- | --- | ---------- | ------ | ----------- | --- | -------- | ----------- | --- | -------- | -------- | --- |
| the ground | truth. | The | main contributions |     | of  | this paper | are as |             |     |          |             |     |          |          |     |
follows: validationlossisevaluatedbasedonthevalidationdata,which
|       |          |             |     |                      |     |     |             | serves as | a proxy | for   | evaluating | the           | model’s | performance |     |
| ----- | -------- | ----------- | --- | -------------------- | --- | --- | ----------- | --------- | ------- | ----- | ---------- | ------------- | ------- | ----------- | --- |
| • Our | approach | outperforms |     | the state-of-the-art |     |     | by at least |           |         |       |            |               |         |             |     |
|       |          |             |     |                      |     |     |             | on unseen | data.   | After | training   | is completed, |         | the trained | DL  |
5%intermsofF-scoreforoverfittingdetection,achieving
|     |          |          |                |     |            |             |     | model’s  | performance | is     | evaluated | using | the | test set that | has |
| --- | -------- | -------- | -------------- | --- | ---------- | ----------- | --- | -------- | ----------- | ------ | --------- | ----- | --- | ------------- | --- |
| an  | F-score  | of 0.91. |                |     |            |             |     |          |             |        |           |       |     |               |     |
|     |          |          |                |     |            |             |     | not been | exposed     | to the | model.    |       |     |               |     |
| Our | approach | has      | the capability |     | to prevent | overfitting |     | at       |             |        |           |       |     |               |     |
•
|       |     |         |            |          |     |                   |     | Researchers |     | and developers |     | can identify |     | potential | issues |
| ----- | --- | ------- | ---------- | -------- | --- | ----------------- | --- | ----------- | --- | -------------- | --- | ------------ | --- | --------- | ------ |
| least | 32% | earlier | than early | stopping |     | while maintaining |     |             |     |                |     |              |     |           |        |
(and often surpassing) the rate of reaching the optimal such as overfitting or underfitting by analyzing the training
epoch (i.e., the epoch that yields the best model). histories. For example, overfitting is often observed as an
increasingdivergencebetweentraininglossandvalidationloss
| • We | provide | a replication |     | package | [1] | containing | our |     |     |     |     |     |     |     |     |
| ---- | ------- | ------------- | --- | ------- | --- | ---------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
trained classifiers and labelled training histories which overtime(asillustratedinFigure1a).Inthispaper,wepropose
|               |               |             |           |              |              |               |            | an approach | that    | leverages      | training | history     |          | to automatically |       |
| ------------- | ------------- | ----------- | --------- | ------------ | ------------ | ------------- | ---------- | ----------- | ------- | -------------- | -------- | ----------- | -------- | ---------------- | ----- |
| can           | be directly   | used        | by other  | researchers. |              |               |            |             |         |                |          |             |          |                  |       |
|               |               |             |           |              |              |               |            | detect and  | prevent | overfitting    | in       | DL models.  |          |                  |       |
| Paper         | organization. |             | This      | paper        | is organized | as            | follows.   |             |         |                |          |             |          |                  |       |
| Section       | II provides   | background  |           | information  |              | about         | our study. |             |         |                |          |             |          |                  |       |
| Section       | III gives     | an          | overview  | of related   |              | work. Section | IV         |             |         |                |          |             |          |                  |       |
| introduces    | existing      | approaches  |           | for          | detecting    | and           | preventing |             |         |                |          |             |          |                  |       |
|               |               |             |           |              |              |               |            | B. Time     | series  | classification |          |             |          |                  |       |
| overfitting.  | Section       | V           | describes | the          | design       | of            | our study, |             |         |                |          |             |          |                  |       |
| while Section |               | VI provides |           | detailed     | information  |               | about our  |             |         |                |          |             |          |                  |       |
|               |               |             |           |              |              |               |            | Time        | series  | data consists  | of       | data points | recorded | over             | time, |
experimentalsetup.SectionsVIIandVIIIpresenttheresultsof
|     |     |     |     |     |     |     |     | with each | point | being | associated | with | a specific | timestamp |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --------- | ----- | ----- | ---------- | ---- | ---------- | --------- | --- |
ourstudy.SectionIXdiscussespotentialthreatstothevalidity
of our study. Finally, Section X concludes the paper. and its corresponding value. Time series classification is a
|     |     |     |     |     |     |     |     | machine | learning | task | that aims | to  | categorize | time | series |
| --- | --- | --- | --- | --- | --- | --- | --- | ------- | -------- | ---- | --------- | --- | ---------- | ---- | ------ |
II. BACKGROUND data into predefined classes. In the context of our study,
This section provides an introduction to the concepts of the we consider the training history of a DL model as time
|               |         |          |          |        |                 |     |     | series data,   | as             | the task | of identifying |         | whether | a DL      | model |
| ------------- | ------- | -------- | -------- | ------ | --------------- | --- | --- | -------------- | -------------- | -------- | -------------- | ------- | ------- | --------- | ----- |
| training      | history | in DL    | and time | series | classification. |     |     |                |                |          |                |         |         |           |       |
|               |         |          |          |        |                 |     |     | is overfitting | based          | on       | its training   | history | can     | be framed | as    |
| A. Leveraging |         | training | history  | in DL  |                 |     |     |                |                |          |                |         |         |           |       |
|               |         |          |          |        |                 |     |     | a time series  | classification |          | problem.       | Since   | there   | has been  | no    |
Training history, also known as the empirical learning prior systematic research on time series classifiers specifically
curve[46],providesvaluableinsightsintoaDLmodel’slearn- designedfortraininghistories,wehaveselectedsixclassifiers
ingprogressandperformancethroughoutthetrainingprocess. (showninTableI)thathavebeenreportedasbaselinesorstate-
The training history stores a record of metrics during the of-the-artinpriorstudies[2,77,78,82].Theseclassifierswere
training process, which are usually recorded in each training chosenduetotheirdemonstratedeffectivenessinvarioustime
iteration or epoch (as shown in Figure 1). Training loss and series classification tasks and their potential applicability to
validationlossarecommonlyusedmetricsintraininghistories. the overfitting detection problem in DL.

III. RELATEDWORK ferentiates overfitting patches based on semantic differences.
|               |     |             |       |     |     |     |     | Nilizadeh  | et al.         | [49] utilized | formal         | verification |                | to evaluate |
| ------------- | --- | ----------- | ----- | --- | --- | --- | --- | ---------- | -------------- | ------------- | -------------- | ------------ | -------------- | ----------- |
| A. Mitigating |     | overfitting | in SE |     |     |     |     |            |                |               |                |              |                |             |
|               |     |             |       |     |     |     |     | the degree | of overfitting |               | and identified |              | the challenges | posed       |
Overfitting poses a significant risk to the trustworthiness by program complexity and numeric issues.
| of software | systems |     | and the | research | studies | that | employ |     |     |     |     |     |     |     |
| ----------- | ------- | --- | ------- | -------- | ------- | ---- | ------ | --- | --- | --- | --- | --- | --- | --- |
Tothebestofourknowledge,boththeoverfittingprevention
DL models. SE researchers typically use either overfitting anddetectionmethodsusedinbothSEstudiesandinpractice
| detection | or prevention |     | methods | to  | mitigate | the problem | of  |           |            |     |           |     |             |            |
| --------- | ------------- | --- | ------- | --- | -------- | ----------- | --- | --------- | ---------- | --- | --------- | --- | ----------- | ---------- |
|           |               |     |         |     |          |             |     | fall prey | to several | key | concerns. | The | overfitting | prevention |
overfitting.Amongtheoverfittingpreventionmethods,dropout methods, typically require significant expertise to execute
isthemostcommonlyadoptedapproach[79,85].Researchers correctly and are intrusive (for instance, they may require
| have used | dropout | in  | various | domains | such | as  | code gen- |        |            |      |        |       |             |             |
| --------- | ------- | --- | ------- | ------- | ---- | --- | --------- | ------ | ---------- | ---- | ------ | ----- | ----------- | ----------- |
|           |         |     |         |         |      |     |           | one to | modify the | data | or the | model | structure). | Overfitting |
eration [39, 40], logging locations recommendation [37], detection approaches typically require retraining of the DL
| and comment |     | completion | [14, | 80]. | Regularization |     | is another |                |     |        |           |     |      |                 |
| ----------- | --- | ---------- | ---- | ---- | -------------- | --- | ---------- | -------------- | --- | ------ | --------- | --- | ---- | --------------- |
|             |     |            |      |      |                |     |            | model multiple |     | times, | which may | be  | very | costly in prac- |
prominent overfitting prevention strategy that has been used tice [20] and simple methods like early stop may stop the
in SE [4, 30, 84, 86], which requires adding another layer to trainingofDLmodelsuboptimally.Ourworkaddressesthese
| the model | structure | as  | well. For | example, | Zampetti |     | et al. [86] |         |             |                 |     |          |     |                 |
| --------- | --------- | --- | --------- | -------- | -------- | --- | ----------- | ------- | ----------- | --------------- | --- | -------- | --- | --------------- |
|           |           |     |           |          |          |     |             | gaps by | introducing | a history-based |     | approach |     | that serves the |
employed L2-norm regularization in training CNN and RNN dual purpose of detecting and preventing overfitting in a non-
modelstomanageself-admittedtechnicaldebtinsourcecode.
|          |             |        |          |            |             |           |         | intrusive                                              | manner | without | any need | for | DL model | retraining. |
| -------- | ----------- | ------ | -------- | ---------- | ----------- | --------- | ------- | ------------------------------------------------------ | ------ | ------- | -------- | --- | -------- | ----------- |
| Early    | stopping    | is     | another  | frequently | used        | technique |         | to                                                     |        |         |          |     |          |             |
|          |             |        |          |            |             |           |         | B. LeveragingtraininghistorytoimproveDLsoftwarequality |        |         |          |     |          |             |
| prevent  | overfitting | during | the      | training   | process     | of        | DL mod- |                                                        |        |         |          |     |          |             |
| els [13, | 28, 62].    | For    | example, | Shi        | et al. [62] | utilized  | early   |                                                        |        |         |          |     |          |             |
Whilethemachinelearningcommunityhasleveragedtrain-
stopping when training a deep Siamese network to identify ing histories (i.e., learning curves) for different tasks, the
hiddenfeaturerequestspostedinchatmessagesbydevelopers. SE community has seldom used training history to enhance
Other techniques like data augmentation [3, 19, 43] and data software quality. Mohr and van Rijn [46] conduct a survey
balancing [53, 75] are also employed to address overfitting. on approaches based on learning curves for decision-making
| For instance, | Bao | et al. | [3] | developed | a CNN-based |     | image |       |          |      |         |              |     |                 |
| ------------- | --- | ------ | --- | --------- | ----------- | --- | ----- | ----- | -------- | ---- | ------- | ------------ | --- | --------------- |
|               |     |        |     |           |             |     |       | in DL | domains, | such | as data | acquisition, |     | early stopping, |
classification model to filter out non-code and noisy-code and model selection. They also propose an approach called
frames from programming screencasts. To enhance training Learning Curve Cross-Validation (LCCV) [45] that iteratively
data diversity, they employed data augmentation techniques increasesthenumberoftrainingexamplesusedfortrainingto
such as rotation, scaling, translation, and shearing. select the best model from the candidates. Training histories
| In terms | of  | overfitting | detection |     | approaches, | Zhang | et al. |          |         |             |     |         |              |          |
| -------- | --- | ----------- | --------- | --- | ----------- | ----- | ------ | -------- | ------- | ----------- | --- | ------- | ------------ | -------- |
|          |     |             |           |     |             |       |        | can also | be used | to evaluate | the | trained | classifiers: | van Rijn |
[87] propose the perturbation validation (PV) assessment to et al. [76] propose an approach that recommends classifiers
determine whether a DL model fits the training data properly for a given dataset based on training histories using loss time
(i.e., ensure that it is neither overfitting nor underfit). Alterna- curves. Moreover, Hoiem et al. [29] investigate the use of
tively,somedetectionapproachescheckthehypothesisthatthe training histories to evaluate design choices for DL models,
trained DL model and the data are independent. For example, such as pretraining, architecture, and data augmentation.
Werpachowski et al. [81] check the hypothesis by comparing In this paper, we propose OverfitGuard, an approach
thetesterrorwiththeestimatedtesterrorbasedonadversarial that utilizes time-series classifiers to detect and prevent over-
examples of the test set. fitting by analyzing the training history of DL models. Our
Another popular approach towards detecting overfiting is approach aims to enhance the software quality of DL systems
to use model validation approaches. Tantithamthavorn et al. by improving the quality of the DL models themselves.
[72] evaluated 12 model validation techniques specifically for OverfitGuard is one of the first approaches that leverages
defect prediction models and concluded that out-of-sample traininghistorytoenhancethesoftwarequalityofDLsystems.
bootstrap emerges as the least biased and most stable tech- A related work by Tokui et al. [74] introduces an approach
nique that helps detect overfit. Damiani and Ardagna [15] calledNeuRecoverwhichalsoleveragestraininghistorytoim-
introduced a framework that validates DL models against provethequalityofDLmodels,particularlyforsafety-critical
desirednon-functionalpropertiesandstatisticallymonitorsthe applications. NeuRecover identifies the model parameters that
modeloutput. Straub[71]extendedthis byutilizingrandomly need to be modified (i.e., repaired) by analyzing the training
generated expert networks for model validation that focuses history to address specific failure types. Our approach shares
on performance characteristics. similaritieswithNeuRecover,butfocusesonthedetectionand
Otherthantheseapproaches,Smithetal.[67]discussedthe prevention of overfitting, which is important for maintaining
roleofhumanfactorsinoverfittingandprovidedacomparative the software quality of DL systems.
| analysis | between | automated | patches |     | and human-written |     | fixes. |     |     |     |     |     |     |     |
| -------- | ------- | --------- | ------- | --- | ----------------- | --- | ------ | --- | --- | --- | --- | --- | --- | --- |
IV. EXISTINGAPPROACH
| They reported |     | that overfitting |     | is not | solely a | machine-induced |     |     |     |     |     |     |     |     |
| ------------- | --- | ---------------- | --- | ------ | -------- | --------------- | --- | --- | --- | --- | --- | --- | --- | --- |
problem and suggested focusing on the contributing factors Inthissection,weintroducetheexistingapproachesthatwe
like test suite coverage and requirements-based testing. Xin useasbaselineapproachestocompareourproposedapproach
and Reiss [83] proposed a classification technique that dif- for overfitting detection and prevention in DL models.

5
4
3
ssoL
2
Training loss
1 Validation loss
Early stopping (p.=20)
Early stopping (p.=40)
|     |     | 0 25 | 50  | 75 100 | 125 150 | 175 | 200 |     |     |     |     |     |     |     |
| --- | --- | ---- | --- | ------ | ------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Epoch
(a)Examplesofearlystoppingwiththepatienceof20 (b) An demonstration of our overfitting prevention
and40epochs(stopatthe#71epochand#140epoch). approach with a rolling window.
|     |     |     | Fig. | 2: Early | stopping | and | our approach | for | overfitting | prevention. |     |     |     |     |
| --- | --- | --- | ---- | -------- | -------- | --- | ------------ | --- | ----------- | ----------- | --- | --- | --- | --- |
A. Overfitting detection is also used by PyTorch Ignite.2 As shown in Figure 2a, early
|     |     |     |     |     |     |     | stopping |       | with a      | patience | parameter | of 20 | epochs stops      | at the |
| --- | --- | --- | --- | --- | --- | --- | -------- | ----- | ----------- | -------- | --------- | ----- | ----------------- | ------ |
|     |     |     |     |     |     |     | #70      | epoch | and returns | the      | #50 with  | the   | lowest validation | loss   |
Correlation-basedapproaches.Oneapproachfordetecting
overfittinginDLmodelsistocomputethecorrelationbetween since no improvement occurs between epochs #50 and #70.
|     |     |     |     |     |     |     | Furthermore, |     | early | stopping | with | larger | patience values | (e.g., |
| --- | --- | --- | --- | --- | --- | --- | ------------ | --- | ----- | -------- | ---- | ------ | --------------- | ------ |
thetrainingandvalidationloss.Kronbergeretal.[34]propose
|             |      |                |          |            |            |             | 40   | epochs) | stops | later (at | the #140 | epoch) | but with | a lower |
| ----------- | ---- | -------------- | -------- | ---------- | ---------- | ----------- | ---- | ------- | ----- | --------- | -------- | ------ | -------- | ------- |
| computing   | the  | non-parametric |          | Spearman’s | rank       | correlation |      |         |       |           |          |        |          |         |
|             |      |                |          |            |            |             | loss | (at the | #100  | epoch).   |          |        |          |         |
| coefficient | [68] | between        | training | and        | validation | fitness     | in   |         |       |           |          |        |          |         |
Earlystoppingbasedonsmoothedvalidationlosscurves.
symbolicregressionmodelstodetectoverfitting.Similartoour
approach, correlation-based approaches also detect overfitting An alternate version of early stopping inspects the moving
|            |                 |          |     |             |           |           | average | of   | the | smoothed | validation | loss     | curves         | [47, 65] to |
| ---------- | --------------- | -------- | --- | ----------- | --------- | --------- | ------- | ---- | --- | -------- | ---------- | -------- | -------------- | ----------- |
| based on   | the training    | history. |     | Therefore,  | we select | them      | as      |      |     |          |            |          |                |             |
|            |                 |          |     |             |           |           | decide  | when | to  | stop the | training   | process. | After stopping | the         |
| a baseline | for comparison. |          | In  | this study, | we        | calculate | the     |      |     |          |            |          |                |             |
correlation metrics between the training and validation loss training process, this approach returns the best epoch which
|                     |             |               |                |                |            |           | has      | the lowest | validation |     | loss (not   | the smoothed | value). |     |
| ------------------- | ----------- | ------------- | -------------- | -------------- | ---------- | --------- | -------- | ---------- | ---------- | --- | ----------- | ------------ | ------- | --- |
| to detect           | overfitting | in DL         | models.        | The            | underlying | principle |          |            |            |     |             |              |         |     |
| is straightforward: |             | when          | no overfitting |                | occurs,    | the       | training |            |            |     |             |              |         |     |
|                     |             |               |                |                |            |           |          |            |            | V.  | OURAPPROACH |              |         |     |
| and validation      |             | losses should |                | be correlated, | whereas    |           | weak     |            |            |     |             |              |         |     |
correlation implies overfitting. Figure3showsanoverviewofourproposedapproach.Our
We determine the presence of overfitting by comparing approach uses a time series classifier to detect and prevent
the computed correlation-based metric between training and overfitting.TableIliststhestudiedtimeseriesclassifiers.First,
validation losses with a predetermined threshold value (deter- wecollectasimulateddataset(moredetailsonhowwecollect
mined in Section VI-D). We use three different correlation the data in Section VI) that contains training histories (i.e.,
metrics: Spearman, Pearson [26], and time-lagged Pearson training and validation loss curves, however, we only use the
correlation coefficients. Both Spearman and Pearson correla- validation loss curves in our approach) with labels indicating
tion coefficients are calculated since we do not know whether whether overfitting occurs in order to train our time series
the relationship between training and validation loss is lin- classifier. Second, we train and evaluate each studied time
ear or not. In addition, we compute the Pearson correlation seriesclassifieronallofthetraininghistoriesofthesimulated
coefficient between a 5-epoch lagged version of the training dataset. Finally, we use the trained time series classifier to
loss and the validation loss. This approach is inspired by perform both overfitting detection and prevention as follows.
autocorrelation [10], which measures the correlation between Overfitting detection. To detect overfitting in a trained DL
a time series data and a time-lagged version of itself. model, we first collect its validation losses over the training
|                |            |     |     |     |     |     | epochs. | We     | feed    | this loss | to our          | trained | time series | classifier |
| -------------- | ---------- | --- | --- | --- | --- | --- | ------- | ------ | ------- | --------- | --------------- | ------- | ----------- | ---------- |
| B. Overfitting | prevention |     |     |     |     |     |         |        |         |           |                 |         |             |            |
|                |            |     |     |     |     |     | to      | detect | whether | there     | is overfitting. |         | However,    | we cannot  |
Early stopping. One widely used approach for preventing directly feed these validation losses to our classifier, since the
overfittingisearlystopping,whichstopstrainingwhenthereis
lengthofthevalidationlossesmightnotbeofthesamelength
noimprovementinafixednumberofepochs(calledpatience) as that of the data used to train these time series classifiers.
andreturnstheepochwhichhasthelowestvalidationloss.We
|     |     |     |     |     |     |     | All | the studied |     | time series | classifiers, | with | the exception | of  |
| --- | --- | --- | --- | --- | --- | --- | --- | ----------- | --- | ----------- | ------------ | ---- | ------------- | --- |
implementation,1
| choose the                  | widely | used                                         | TensorFlow |     |     |     | which         |                                                             |     |     |     |     |           |     |
| --------------------------- | ------ | -------------------------------------------- | ---------- | --- | --- | --- | ------------- | ----------------------------------------------------------- | --- | --- | --- | --- | --------- | --- |
|                             |        |                                              |            |     |     |     |               | 2https://pytorch.org/ignite/generated/ignite.handlers.early |     |     |     |     | stopping. |     |
| 1https://tensorflow.org/api |        | docs/python/tf/keras/callbacks/EarlyStopping |            |     |     |     | EarlyStopping |                                                             |     |     |     |     |           |     |

Training the time series classifier
Continue training for
the model
Feed data (with
Trained time series
Tra w in i i t n h g l a h b is e t l o s ries l c a l b a e s l s s i ) fi e to r f t o im r e tr a s i e n r i i n e g s classifier or or
Extract the latest
Use the whole
history with a rolling
observed history
window
Overfitting detection using the training history
Perform inference to N Overfitting detection
Training identify overfitting Is overfitting? using the training
history history
Y
Interpolate the Stop training and
Validation losses v th a e li d s a a t m io e n l l e o n s g se th s a to s n O o v n e -o rf v it e ti r n fi g tt i o n r g retu h r a n s t h th e e e l p o o w c e h s t that O pr v e e v r e fi n tt t i i n o g n when
the data in training validation loss training a model
Fig. 3: Our approach for overfitting detection and prevention.
KNN-DTW, expect the length of the inputs used for training
Datasets
and inference to be the same. Therefore, we first linearly
Download the Simulate
interpolate the validation losses of the DL model for which datasets for overfitting by Label training Simulated training
overfitting training neural histories dataset (419
we need to detect overfit to the same length as the training simulation networks training histories)
histories used to train the studied time series classifiers. We
Search for Collect existing
Identify related
feedtheinterpolatedvalidationlossestoourtrainedtimeseries conferences papers that training history or Real-world test
classifier and perform inference to determine if the DL model and journals hav o e v s e a rf m itt p in le g s of t r r e a p in ro in d g u h ce is t t o h r e y trai d n a in ta g s h e i t s t ( o 4 r 0 ies)
is overfit.
Overfitting prevention. To prevent overfitting, we feed the Experiment Evaluation
training history (i.e., validation loss curve) of a DL model Choose Correlation- Compare
thresholds for based overfitting
thatisbeingtrainedtoourtrainedtimeseriesclassifierduring correlation-based approaches with detection
approaches thresholds approaches
the training process. The history is fed for inference in two
different ways: (1) as a rolling window: we extract the latest
Compare
Trained time
historyinafixedwindowsize(e.g.,thelatest20epochs),and Train our series Early stopping overfitting
approaches classifiers prevention
(2) as the whole observed history (from the first to the latest approaches
epochs). Our time series classifier detects if in the fed history
overfitting occurs. Similar to overfitting detection, we linearly
interpolate the data before feeding it into our model. If there Fig. 4: Overview of the experimental setup.
is no overfit occurring, we continue the training and repeat
the above procedure until the DL model has finished training.
For the rolling window, we move the window by a fixed step Python3.8andTensorFlow2.9.0.Thehardwareconfiguration
size (as shown in Figure 2b) and make another prediction. If for the experiments is detailed below:
ourmodeldetectsthepresenceofoverfittinginthefedhistory, • NVIDIA RTX 3090 GPU with 24 GB memory
we return the lowest validation loss in the observed epochs as • CUDA version 10.1.243
the best epoch. • cuDNN version 7.6.5
• Intel(R) Core(TM) i9-11900K CPU with a clock speed
VI. EXPERIMENTALSETUP of 3.50 GHz
• 64 GB of RAM
In this section, we introduce the datasets for training and
evaluating the studied overfitting detection and prevention B. Simulated training dataset
approaches, the experiments of our study, and the evalua-
We create a simulated dataset containing training histories
tion metrics for the studied approaches. Figure 4 shows an
with labels to determine a threshold for correlation-based
overview of the experimental setup.
approaches(seeSectionIV)andtotrainourproposedmethod
as described in Section V. We create this simulated dataset
A. Environment setting
by training neural networks of varying model complexities to
We conducted the experiments on an Ubuntu 20.04 oper- produce overfitting and non-overfitting samples. The process
ating system with a Linux kernel version of 5.15.0, utilizing is as follows:

TABLE II: Information about datasets used to simulate over-
of our manual labelling process, we follow the approach
fitting.
outlined by Ding et al. [18]. The first and second authors
of this paper independently labelled the 432 data points as
Dataset Type #In #Out #Examples
either“overfit”,“non-overfit”or“uncertain”anddiscussedthe
building regression 14 3 4,208
results.Inthefirstdiscussionround,theauthorsreacheda95%
cancer classification 9 2 699
card classification 51 2 690 agreement (410 data points), with both authors labelling 10
diabetes classification 8 2 768 data points as ”uncertain” and subsequently eliminating them.
flare regression 24 3 1,066
In the second round, the authors discussed the remaining 22
gene classification 120 3 3,175
glass classification 9 6 214 disagreements.Followingthediscussion,weeliminated3data
heart classification 35 2 920 points (labelled “uncertain” by both authors) and agreed on
hearta regression 35 1 920
the labels for the remaining 19 data points. The final dataset
horse classification 58 3 364
soybean classification 82 19 683 consistsof44overfitand375non-overfittraininghistories.We
thyroid classification 21 3 7,200 share the labelled training histories in our replication package
for other researchers to reuse.
Step 1 – Download the datasets for overfitting simulation. C. Real-world test dataset
We download 12 datasets representing real-world problems
from the Proben1 [52] benchmark set for training neural net-
To evaluate our approach using real-world data, we con-
works. These datasets were used by Prechelt [51] to simulate
ducted a survey of papers from conferences and journals to
traininghistoriesforstudyingearlystopping.Wechoosethese
gather examples of overfit and non-overfit DL models.
datasets over using SE datasets for two reasons: First, since
Step1–Identifyrelatedconferencesandjournals.Weiden-
we use the methodology used by Prechelt [51] to simulate
tify related conferences and journals based on the Computing
overfitting, we chose to stick with the datasets that they used. Research and Education Association of Australasia (CORE3)
Second, irrespective of the domain of the dataset, we assert and China Computer Federation (CCF4) ranking systems.
that how the phenomenon of overfit is represented by training
Under the CCF A rank, we have 7 conferences and 4 journals
andvalidationhistorieswillremainthesame.TableIIprovides
in the “Artificial Intelligence” field. Under the CORE A*
information about these datasets, which include 3 datasets for
rank, we have 16 conferences in the “machine learning” and
regression tasks and 9 datasets for classification tasks. All
“artificial intelligence” fields and 12 journals in the “artificial
of these datasets (except the “building” one) were originally
intelligence and image processing” field. After merging the
collectedfromtheUCImachinelearningrepository[7],which
results and accounting for overlaps between the two ranking
has been widely used in DL research [22, 36, 56, 63]. Each
systems, we obtained a final list of 17 conferences and 12
datasetispre-partitionedintotraining,validation,andtestsets
journals.
(50%,25%,and25%ofthedata,respectively)andpartitioned
Wecollectedourreal-worlddataintheartificialintelligence
three times to generate three distinct permutations, resulting
and machine learning domain as opposed to SE domain,
in a total of 36 datasets from Proben1.
becauseSEstudiestypicallydonotreportthetraininghistories
Step 2 – Simulate overfitting by training neural networks.
oftheoverfitDLmodelsandwerequiredcommunityaccepted
We train neural networks (NNs) with various architectures
examples of the training histories of both overfit and non-
on the collected 36 datasets. We do so to vary the model
overfit DL models.
complexitywhichinturnincreasesthechanceofproducingan
Step 2 – Search for papers that have samples of overfitting.
overfitted DL model, following the methodology of Prechelt
We found 33 full papers (see the replication package [1])
[51] used in their study. The input/output layer of each NN
containing the keyword ”overfit” (including variations such
contains the same number of nodes as the number of in-
as”overfitting”)inthetitlethatwerepublishedintheselected
put/output coefficients in the respective datasets (see Table II)
conferences and journals in the last 5 years. Five of these
and rectified linear units (ReLUs) are used for all hidden
papers provided samples of overfitting: P2 - Chatterjee and
layers. The structures of the NNs are as follows: (1) 6 one-
Mishchenko [11]; P4 - Chen et al. [12] P13 - Kim et al.
hidden-layerNNswith2,4,8,16,24,or32hiddennodes,and
[33]; P17 - Rice et al. [54] and P23 - Singla et al. [66].
(2) 6 two-hidden-layer NNs with hidden nodes (represented
Table III lists the papers and the number of collected samples
as first layer hidden nodes + second layer hidden nodes) of
of overfitting (some of them also provide samples of non-
2+2, 4+2, 4+4, 8+4, 8+8, 16+8. We use the mean square
overfitting).
error (MSE) as the loss function for regression problems, and
Step 3 – Collect existing training history or reproduce the
cross entropy as the loss function for classification problems.
traininghistory.PaperP17sharedthetraininghistory,making
All problems employ stochastic gradient descent (SGD) as
its replication straightforward. We replicated the other papers
the optimizer. To increase the likelihood of overfitting, we
thatprovideoverfittingsamplestocollectthetraininghistories
train these 12 neural network architectures on each of the 36
datasets for 1,000 epochs, producing 432 training histories.
3https://www.core.edu.au
Step 3 – Label training histories. To ensure the robustness 4https://ccf.atom.im/

TABLE III: Information about collected samples from sur-
approaches, we computed the precision, recall, and F-score
veyed papers.
for overfitting and non-overfitting samples in the real-world
test dataset. In addition, we calculated the average F-score to
Paper Labels for the training history in the #Overfit #Non-overfit
manuscript directly compare the classification performance of the studied
P2 “[...]thevalidationaccuracyofnn-randomis 2 0 approaches. To evaluate the time cost associated with training
9.73%(i.e.,closetochance)confirmingthat andusingthestudiedapproaches,wereportthetraining time
itishorriblyoverfit”
P4 “We first observe that the robust overfitting 3 3 (in seconds) for each approach on the simulated dataset and
prevailsinallBaselinecases” theinferencetime(inmilliseconds)forthereal-worlddataset.
“Ourmethodseffectivelymitigatestherobust
overfitting” Evaluation metrics for overfitting prevention. Ideally, an
P13 “Figure4[...]thatis,catastrophicoverfitting N/A∗ N/A∗ overfittingpreventionapproachreturnstheoptimalepoch(i.e.,
occurs.”
“Figure 6 shows that the proposed method the epoch that yields the best predictive performance for
alsosuccessfullypreventscatastrophicover- the DL model on the validation set) and stops the training
fitting[...]”
P17 “Figure24[...]Weseeclearrobustoverfitting 20 4 process as early as possible. We define the optimal rate
forthesmallertwooptionsinλ,andfindno of an overfitting prevention approach as the percentage of
overfittingbuthighlyregularizedmodelsfor
thelargertwooptions[...]” cases where the optimal epoch is successfully identified. To
P23 “These results therefore validate our claim 4 4 assess the speed of the approach, we introduce the delay
that low curvature activations reduce robust
overfitting” metric, which represents the epoch difference between the
∗ Cannotreproducethesameresultsasthepaper. stopped epoch and the best epoch. For example, a delay of 10
epochs occurs if the prevention approach stops at the 123th
epoch while the 113th epoch is the best one. In addition, we
of these samples. We executed the code from the papers with
report the DL model’s accuracy on the validation set when
available replication packages (P4, P13, and P23) to generate
the training process is stopped by the overfitting prevention
their training histories. However, we could not replicate the
approach.
results for paper P13. For paper P2, which did not provide a
replicationpackage,wefollowedthemethodologytoreplicate VII. RQ1:HOWWELLDOESOVERFITGUARD DETECT
the results and training history. In total, we collected 29 OVERFITTINGINTRAINEDDLMODELS?
training histories of overfit DL models and 11 of non-overfit
Motivation. Overfitting detection is an important task in
DL models (refer to Table III for details).
DL models since it helps in identifying whether a DL model
D. Experiments has learned to perform well on training data but fails to
generalize on unseen data. Accurate overfitting detection can
Overfitting detection.Wetrainedthetimeseriesclassifiers
assistresearchersanddevelopersinmakinginformeddecisions
based on the simulated dataset. For each classifier, a grid
regarding model selection, hyperparameter tuning, and other
search with 3-fold cross-validation was performed to tune
model performance improvements. This research question
the hyperparameters based on the simulated dataset. Once
investigates the performance of our proposed approach for
the optimal hyperparameters were identified, we proceeded
detecting overfitting in trained DL models and compares it
to train each time series classifier using all training histories
with existing correlation-based approaches.
and labels from the simulated dataset and saved the trained
Approach.WeusetheevaluationmetricsintroducedinSec-
classifierforfurtheruse.Forcorrelation-basedapproaches,we
tion VI-E to compare our approach with baseline approaches
also performed a grid search based on the simulated dataset
based on the real-world test dataset. Furthermore, we record
to select the optimal thresholds (ranging from -1 to 1) that
theF-scoreobtainedfromthe3-foldcross-validation(CV)for
yielded the best F-score.
ourapproachbasedonthesimulatedtrainingdatasettofurther
Overfitting prevention. We reused the trained time series
analyze the performance of our approach. Since we use the
classifiers from the previous step to perform inference during
entire simulated training dataset to determine the thresholds
thetrainingprocesstopreventoverfitting.Sincethevalidation
(without CV) for correlation-based approaches, we report the
loss curve is generally applicable to both classification and
F-score for correlation-based approaches based on the whole
regression tasks, we used it for overfitting prevention. We
simulated training dataset.
applied our approach to the trained DL models in every 10
Results. Overfit DL models can be detected by inspecting
epochs (i.e., the step size), with varying rolling window sizes
the training history, and our approach using time series
of 20, 40, 60, 80, and 100 epochs. We used early stopping
classifiersdemonstratesbetterclassificationperformancethan
based on the validation loss and set the patience values to
the correlation-based approaches for overfitting detection.
range from 5 to 115 epochs. We also applied early stopping
Table IV shows that our approach using KNN-DTW, TSBF,
based on smoothed validation loss curves generated by a 10-
and SAX-VSM generalizes well from the simulated dataset to
epoch moving average [47, 65].
the real-world dataset with the best F-score (0.91), followed
E. Evaluation
byTSFwhichoutperformsthebaselineapproachesaswell.In
Evaluation metrics for overfitting detection. To evaluate contrast, HMM-GMM performs poorly on both the simulated
the classification performance of the overfitting detection trainingandreal-worldtestdatasets.Onepossibleexplanation

TABLE IV: Results of the overfitting detection approaches on the simulated dataset (CV F-S: F-score of cross-validation)
and real-world dataset (Prec: precision; Rec: recall; F-s: F-score; Avg F-s: average F-score), and the time cost of training the
studied approaches on the simulated dataset and performing inference on the real-world dataset (per sample).
Detectionapproach Non-overfitting Overfitting Avg CV Training Inference
|     |     |     |          |     | Prec | Rec  | F-s Prec  | Rec  | F-s       | F-s F-s | time(s) | time(ms) |       |     |     |
| --- | --- | --- | -------- | --- | ---- | ---- | --------- | ---- | --------- | ------- | ------- | -------- | ----- | --- | --- |
|     |     |     | Spearman |     | 0.71 | 0.91 | 0.80 0.96 | 0.86 | 0.91 0.86 | 0.95    | 2.461   |          | 0.908 |     |     |
Corr.
|     |     |     | Pearson |     | 0.78 | 0.64 | 0.70 0.87 | 0.93 | 0.90 0.80 | 0.92 | 0.222 |     | 0.025 |     |     |
| --- | --- | --- | ------- | --- | ---- | ---- | --------- | ---- | --------- | ---- | ----- | --- | ----- | --- | --- |
based
|     |     |     | Autocorr |     | 0.80 | 0.73 | 0.76 0.90 | 0.93 | 0.92 0.84 | 0.80 | 0.233  |     | 0.026   |     |     |
| --- | --- | --- | -------- | --- | ---- | ---- | --------- | ---- | --------- | ---- | ------ | --- | ------- | --- | --- |
|     |     |     | KNN-DTW  |     | 0.79 | 1.00 | 0.88 1.00 | 0.90 | 0.95 0.91 | 0.97 | 0.001  |     | 180.512 |     |     |
|     |     |     | HMM-GMM  |     | 0.30 | 0.27 | 0.28 0.73 | 0.76 | 0.75 0.52 | 0.59 | 99.751 |     | 17.750  |     |     |
Time
|     |     |     | TSF |     | 0.77 | 0.91 | 0.83 0.96 | 0.90 | 0.93 0.88 | 0.99 | 0.311 |     | 17.209 |     |     |
| --- | --- | --- | --- | --- | ---- | ---- | --------- | ---- | --------- | ---- | ----- | --- | ------ | --- | --- |
series
|     |     |     | TSBF |     | 0.79 | 1.00 | 0.88 1.00 | 0.90 | 0.95 0.91 | 0.99 | 0.301 |     | 31.683 |     |     |
| --- | --- | --- | ---- | --- | ---- | ---- | --------- | ---- | --------- | ---- | ----- | --- | ------ | --- | --- |
(ours)
|             |           |        | BOSSVS     |      | 0.46        | 0.91 | 0.61 0.94     | 0.59     | 0.72 0.67 | 1.00        | 1.877      |        | 19.342     |     |           |
| ----------- | --------- | ------ | ---------- | ---- | ----------- | ---- | ------------- | -------- | --------- | ----------- | ---------- | ------ | ---------- | --- | --------- |
|             |           |        | SAX-VSM    |      | 0.83        | 0.91 | 0.87 0.96     | 0.93     | 0.95 0.91 | 0.96        | 0.912      |        | 17.474     |     |           |
|             |           |        |            |      |             |      |               | TABLE    | V: The    | optimal     | rate,      | median | delay,     | and | average   |
| is that the | extracted | state  | models     | (via | HMM)        | of   | the training  |          |           |             |            |        |            |     |           |
|             |           |        |            |      |             |      |               | accuracy | of our    | overfitting | prevention |        | approaches |     | using the |
| histories   | do not    | follow | a Gaussian |      | probability |      | distribution. |          |           |             |            |        |            |     |           |
Our approach with BOSSVS correctly identifies all the data whole observed history.
| in the simulated |     | dataset | but | performs | poorly | on  | the real- |     |     |     |     |     |     |     |     |
| ---------------- | --- | ------- | --- | -------- | ------ | --- | --------- | --- | --- | --- | --- | --- | --- | --- | --- |
world dataset. One reason could be that the extracted bag- Optimal Median Average
Classifier
|                |     |        |      |               |     |         |          |     |     |     | rate | delay |     | accuracy |     |
| -------------- | --- | ------ | ---- | ------------- | --- | ------- | -------- | --- | --- | --- | ---- | ----- | --- | -------- | --- |
| of-SFA symbols |     | (BOSS) | from | the simulated |     | dataset | does not |     |     |     |      |       |     |          |     |
generalize to the real-world dataset. In addition, we note KNN-DTW 0.95 43.5 0.42
|          |              |     |                   |     |            |     |         |     | HMM-GMM |     | 0.18 |     | 0.0  | 0.36 |     |
| -------- | ------------ | --- | ----------------- | --- | ---------- | --- | ------- | --- | ------- | --- | ---- | --- | ---- | ---- | --- |
| that the | investigated |     | correlation-based |     | approaches |     | perform |     |         |     |      |     |      |      |     |
|          |              |     |                   |     |            |     |         |     | TSF     |     | 0.90 |     | 35.0 | 0.42 |     |
reasonably well, with F-scores greater than 0.8. However, TSBF 0.83 31.0 0.41
our approach outperforms the correlation-based overfitting BOSSVS 0.65 21.0 0.41
detection approach by at least 5% on the studied real-world SAX-VSM 0.33 10.0 0.38
dataset.
Thestudiedtimeseriesclassifiersaremorecomputationally
intensive than correlation-based approaches for inference, yet VIII. RQ2:HOWWELLDOESOVERFITGUARD PREVENT
they are still useful in practice. As shown in Table IV, our OVERFITTINGDURINGTHETRAININGPROCESS?
| approach              | requires | more        | time     | for performing    |           | inference    | than         |              |             |                                            |               |            |              |          |              |
| --------------------- | -------- | ----------- | -------- | ----------------- | --------- | ------------ | ------------ | ------------ | ----------- | ------------------------------------------ | ------------- | ---------- | ------------ | -------- | ------------ |
|                       |          |             |          |                   |           |              |              | Motivation.  |             | Anothercriticalpartofdevelopingtrustworthy |               |            |              |          |              |
| the correlation-based |          | approaches. |          | For               | instance, | TSF          | has the      |              |             |                                            |               |            |              |          |              |
|                       |          |             |          |                   |           |              |              | and stable   | DL          | models                                     | is preventing |            | overfitting. |          | An effective |
| fastest inference     |          | time among  |          | the classifiers   |           | but is       | around 20    |              |             |                                            |               |            |              |          |              |
|                       |          |             |          |                   |           |              |              | overfitting  | prevention  |                                            | approach      | allows     | DL           | models   | to gener-    |
| times slower          | than     | the         | Spearman | correlation-based |           |              | approach     |              |             |                                            |               |            |              |          |              |
|                       |          |             |          |                   |           |              |              | alize better | on          | unseen                                     | data while    | minimizing |              | both     | training     |
| and around            | 700      | times       | slower   | than the          | other     | two          | correlation- |              |             |                                            |               |            |              |          |              |
|                       |          |             |          |                   |           |              |              | resources    | and         | computational                              |               | costs.     | This         | research | question     |
| based approaches.     |          | However,    |          | the speed         | of        | our approach |              | is           |             |                                            |               |            |              |          |              |
|                       |          |             |          |                   |           |              |              | evaluates    | the         | performance                                | of our        | proposed   |              | approach | for pre-     |
| not prohibitive       |          | in practice | since    | overfitting       |           | detection    | is only      |              |             |                                            |               |            |              |          |              |
|                       |          |             |          |                   |           |              |              | venting      | overfitting | during                                     | the training  |            | process      | compared | with         |
executedonceafterthetrainingiscomplete.Itisalsousefulto
|             |                        |            |     |               |                        |              |          | the frequently |           | used early | stopping        | approach.   |            |      |             |
| ----------- | ---------------------- | ---------- | --- | ------------- | ---------------------- | ------------ | -------- | -------------- | --------- | ---------- | --------------- | ----------- | ---------- | ---- | ----------- |
| notethatthe | trainingtimesofthetime |            |     |               | seriesclassifiersinour |              |          |                |           |            |                 |             |            |      |             |
|             |                        |            |     |               |                        |              |          | Approach.      |           | We assess  | our overfitting |             | prevention |      | approach    |
| approach    | are not                | excessive. | For | instance,     | the                    | training     | times of |                |           |            |                 |             |            |      |             |
|             |                        |            |     |               |                        |              |          | against        | the early | stopping   | method          |             | (both      | with | and without |
| TSF and     | TSBF                   | are around | 300 | milliseconds, |                        | and KNN-DTW, |          |                |           |            |                 |             |            |      |             |
|             |                        |            |     |               |                        |              |          | smoothing      | loss      | curves)    | using           | the metrics | introduced |      | in Sec-     |
ourbest-performingtimeseriesclassifier,canfinishtrainingin
1millisecond.However,KNN-DTWrequiresthelongesttime tion VI-E. To study the difference in delay across overfitting
|               |        |         |        |                  |        |      |            | prevention | approaches, |              | we performed |     | the | Mann-Whitney | U         |
| ------------- | ------ | ------- | ------ | ---------------- | ------ | ---- | ---------- | ---------- | ----------- | ------------ | ------------ | --- | --- | ------------ | --------- |
| for inference | which  | is      | around | 180 milliseconds |        | for  | a training |            |             |              |              |     |     |              |           |
|               |        |         |        |                  |        |      |            | test [44]  | at a        | significance | level        | of  | α = | 0.05 to      | determine |
| history.      | A fast | version | of DTW | [58]             | with a | time | complexity |            |             |              |              |     |     |              |           |
whetherthedistributionsofthedelayepochsofearlystopping
| of O(n) | is used | in experiments, |     | but | using KNN | with | DTW | is  |     |     |     |     |     |     |     |
| ------- | ------- | --------------- | --- | --- | --------- | ---- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
andourapproacharesignificantlydifferent.Wealsocomputed
| still computationally |           | intensive. |                |          |     |             |         |          |          |             |           |          |           |            |       |
| --------------------- | --------- | ---------- | -------------- | -------- | --- | ----------- | ------- | -------- | -------- | ----------- | --------- | -------- | --------- | ---------- | ----- |
|                       |           |            |                |          |     |             |         | Cliff’s  | delta d  | [42] effect | size to   | quantify | the       | difference | based |
| (cid:3)               |           |            |                |          |     |             | (cid:0) |          |          |             |           |          |           |            |       |
|                       |           |            |                |          |     |             |         | on the   | provided | thresholds  | [55].     |          |           |            |       |
| RQ1                   | Takeaway: |            | Our            | proposed |     | approach    |         |          |          |             |           |          |           |            |       |
|                       |           |            |                |          |     |             |         | Results. | Our      | proposed    | approach, |          | utilizing | KNN-DTW    | with  |
| demonstrates          |           | better     | classification |          |     | performance |         |          |          |             |           |          |           |            |       |
bothrollingwindowandwholeobservedhistory,hasasimilar
| than          | correlation-based |       |                | approaches  | for         | detecting |        |             |         |                 |             |             |          |           |             |
| ------------- | ----------------- | ----- | -------------- | ----------- | ----------- | --------- | ------ | ----------- | ------- | --------------- | ----------- | ----------- | -------- | --------- | ----------- |
|               |                   |       |                |             |             |           |        | or higher   | optimal | rate            | than        | early       | stopping | for       | overfitting |
| overfitting   |                   | in DL | models.        | Despite     |             | the       | higher |             |         |                 |             |             |          |           |             |
|               |                   |       |                |             |             |           |        | prevention. | Other   | studied         | classifiers |             | do not   | perform   | as well     |
| computational |                   | cost  | of the         | time series | classifiers |           | used   |             |         |                 |             |             |          |           |             |
|               |                   |       |                |             |             |           |        | as KNN-DTW  |         | for overfitting |             | prevention. |          | As Figure | 5 and       |
| in our        | approach,         |       | their training |             | time and    | inference |        |             |         |                 |             |             |          |           |             |
TableVshow,usingKNN-DTWwitheitherarollingwindow
| time | are still | practical. |     |     |     |     |     |     |     |     |     |     |     |     |     |
| ---- | --------- | ---------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
(cid:2) (cid:1) or the whole observed history outperforms early stopping at

TABLE VI: The median delay and average accuracy of early stopping (es) and our overfitting prevention approaches (using a
rolling window) with different window sizes (ws).
Prevention Mediandelay/ws Averageaccuracy/ws
approach
20 40 60 80 100 20 40 60 80 100
ES 20.0 40.0 60.0 80.0 100.0 0.40 0.42 0.42 0.43 0.43
ES(smoothed) 20.0 43.5 62.0 82.0 97.5 0.39 0.42 0.43 0.43 0.43
KNN-DTW 31.0 27.0 37.5 42.5 45.5 0.42 0.42 0.43 0.43 0.43
HMM-GMM 5.0 6.5 16.5 28.0 41.5 0.37 0.37 0.38 0.39 0.41
TSF 12.5 22.0 31.0 39.5 44.0 0.40 0.40 0.42 0.43 0.43
TSBF 8.5 15.0 25.0 37.0 47.0 0.38 0.39 0.40 0.41 0.42
BOSSVS 7.0 29.0 34.5 48.5 56.5 0.42 0.42 0.43 0.43 0.43
SAX-VSM 4.0 11.0 9.5 16.0 24.0 0.37 0.37 0.38 0.38 0.40
1.0
0.8
0.6
0.4
0.2
0.0
20 40 60 80 100
Window size (epoch)
etar
lamitpO
Patience (epoch)
5101520253035404550556065707580859095100 105 110 115
Early stopping KNN-DTW TSF SAX-VSM
ES (smoothed) HMM-GMM TSBF BOSSVS
Fig. 5: The optimal rate of our overfitting prevention ap-
proach (using a rolling window) and early stopping with
different patience values.
Fig. 6: Our approach using KNN-DTW (set the window size
as 40 epochs) stops earlier than the early stopping (set the
identifyingtheoptimalepoch.Inparticular,ourapproachwith patience parameter as 40 epochs) but both achieve the same
KNN-DTW based on the rolling window has a higher or the optimal epoch.
same optimal rate as both early stopping approaches when
using up to 80 epochs as the patience parameter and window
size.Forexample,ourapproachwithKNN-DTWobtains78% on both smoothed and non-smoothed validation loss) with the
optimalratewhensettingthewindowsizeto20epochs,while same or higher accuracy. As shown in Table VI, with the
bothearlystoppingapproachesachievelessthan50%optimal samenumberofepochsforthepatienceparameterandwindow
rate when the patience parameter is set to the same epochs. size, our approach can save training time (i.e., reducing delay
However, when the patience parameter is greater than 80 between the stopped epoch and the best epoch) compared to
epochs, both early stopping approaches can identify almost early stopping, except for a window size of 20 epochs. For
all of the optimal epochs. The reason is that 90% of the instance,whensettingboththepatienceparameterandwindow
training histories in the real-world dataset have around 200 size to 40 epochs, KNN-DTW and early stopping have the
epochs, hence, a large patience value makes it easy for early sameaverageaccuracy,butKNN-DTWhasamediandelayof
stopping to choose the optimal epoch. In addition, Table V 27epochswhileearlystoppinghasafixeddelayof40epochs.
shows that our approach with KNN-DTW based on the whole The significance test results indicate that the delay difference
observed history also obtains a higher optimal rate compared between KNN-DTW and early stopping is significant (except
to both early stopping approaches. For example, the KNN- when using a window size of 20), with a medium to large
DTW classifier achieves a 95% optimal rate with a median effect size. Furthermore, early stopping with smoothed loss
delay of 43.5 epochs, whereas early stopping approaches curveshasamediandelayof43.5,whichisslowerthanusing
achieve around 85% using the same number (i.e., between the original loss curves with the same patience parameter (40
40 to 45 epochs patience in Figure 5). epochs). Early stopping using the smoothed loss could hurt
Our approach using KNN-DTW and a rolling window can the performance of early stoppingand cannot compete with
stop training a DL model earlier than early stopping (based our approach. In comparison to the delay in early stopping,

the delay between the stopped epoch and the best epoch is overfittingprevention,andourapproachusingKNN-DTWstill
at least 32% shorter with our approach using KNN-DTW. outperformed early stopping.
| Figure 6 | provides | an example | where | both | early | stopping and |     |     |     |     |     |     |     |     |
| -------- | -------- | ---------- | ----- | ---- | ----- | ------------ | --- | --- | --- | --- | --- | --- | --- | --- |
our approach identify the optimal epoch, but our approach B. Internal Validity
stops 21 epochs earlier than early stopping (which stops with Ourproposedapproachreliesontheassumptionthatoverfit-
a 40 epochs delay). tingcanbedetectedandpreventedthroughtheanalysisofDL
Among our two approaches for overfitting prevention, we model training histories. However, certain cases of overfitting
recommendusingKNN-DTWwitharollingwindow.Although maynotbecapturedbyexaminingthetraininghistoriesalone.
|     |     |     |     |     |     |     | For instance, | data | leakage | caused | by data | augmentation |     | or  |
| --- | --- | --- | --- | --- | --- | --- | ------------- | ---- | ------- | ------ | ------- | ------------ | --- | --- |
usingthewholeobservedhistorymayachieveahigheroptimal
rate than using a rolling window for our approach, we note preprocessing in the entire dataset before data splitting (into
that we can predict the optimal epoch much earlier with the training, validation, and test sets) could lead to overfitting,
|                |     |          |            |       |           |            | but detecting | or  | preventing | it solely | by inspecting |     | the training |     |
| -------------- | --- | -------- | ---------- | ----- | --------- | ---------- | ------------- | --- | ---------- | --------- | ------------- | --- | ------------ | --- |
| rolling window |     | approach | for a very | small | trade-off | in optimal |               |     |            |           |               |     |              |     |
rate (with a similar average accuracy). As shown in Table VI history would be challenging.
andFigure5,ourapproachwithKNN-DTWachievesan83%
|         |           |          |       |       |        |           | C. External | Validity |     |     |     |     |     |     |
| ------- | --------- | -------- | ----- | ----- | ------ | --------- | ----------- | -------- | --- | --- | --- | --- | --- | --- |
| optimal | rate with | a median | delay | of 27 | epochs | and a 90% |             |          |     |     |     |     |     |     |
optimal rate with a median delay of 37.5 epochs using the We evaluated our proposed approach using a real-world
|        |         |        |           |               |          |     | dataset | that contains | training |     | histories | from top | AI venues. |     |
| ------ | ------- | ------ | --------- | ------------- | -------- | --- | ------- | ------------- | -------- | --- | --------- | -------- | ---------- | --- |
| window | size as | 40 and | 60 epochs | respectively. | However, | the |         |               |          |     |           |          |            |     |
median delay of KNN-DTW when using the whole observed However, it is still possible that the approach may not gener-
history is 43.5, while the rolling window approach with a alizewelltoalltypesofDLmodelsordatasets.Secondly,our
window size of 80 or more epochs can achieve a higher real-worldevaluationisbasedononly40datapoints(ofwhich
optimalrate(98%vs.95%accuracy)withashorterdelay(42.5 29traininghistoriesbelongtooverfitmodels),whichmightnot
beenoughdatapointstoclaimgeneralizabilityofourproposed
| vs. 43.5 | epochs). | In summary, |     | we suggest | using | the rolling |     |     |     |     |     |     |     |     |
| -------- | -------- | ----------- | --- | ---------- | ----- | ----------- | --- | --- | --- | --- | --- | --- | --- | --- |
window approach since it stops earlier with a relatively small approach. Please note that collecting authoritative examples
optimal rate drop using a small window (e.g., 40 epochs) and of overfit training history is very hard since researchers and
outperforms the whole observed history approach when using practitioners typically do not report the training history of
a large window size (e.g., 80 epochs). models that were overfit. In addition, collecting these data
(cid:3) (cid:0) points requires one to replicate the studies that report overfit
RQ2Takeaway:OurproposedapproachusingKNN- models, which is a very time and resource intensive task.
|     |      |           |        |     |                |     | Hence we | were | limited | to 40 | data points | for | the real-world |     |
| --- | ---- | --------- | ------ | --- | -------------- | --- | -------- | ---- | ------- | ----- | ----------- | --- | -------------- | --- |
| DTW | with | a rolling | window | or  | whole observed |     |          |      |         |       |             |     |                |     |
history outperforms early stopping for overfitting dataset in our study. However, we invite future research to
prevention and can stop training DL models earlier verify the validity of OverfitGuard using our replication
packageontheirownDLmodeltraininghistories.Inaddition,
| with | the same | or  | higher accuracy. |     | Among | our two |     |     |     |     |     |     |     |     |
| ---- | -------- | --- | ---------------- | --- | ----- | ------- | --- | --- | --- | --- | --- | --- | --- | --- |
approaches, we recommend using KNN-DTW with a the computational resources required to use the proposed
|         |         |           |                 |        |                |     | approach    | for inference |     | could limit   | its applicability |     | in specific |     |
| ------- | ------- | --------- | --------------- | ------ | -------------- | --- | ----------- | ------------- | --- | ------------- | ----------------- | --- | ----------- | --- |
| rolling | window  | for       | early stopping, |        | which achieves | a   |             |               |     |               |                   |     |             |     |
|         |         |           |                 |        |                |     | situations. | For instance, |     | the increased | computational     |     | cost        | may |
| high    | optimal | rate with | a shorter       | delay. |                |     |             |               |     |               |                   |     |             |     |
(cid:2) (cid:1) beprohibitiveinenvironmentswithconstrainedcomputational
|     |     |     |     |     |     |     | resources, | while | our approach |     | demonstrates | improved |     | perfor- |
| --- | --- | --- | --- | --- | --- | --- | ---------- | ----- | ------------ | --- | ------------ | -------- | --- | ------- |
IX. THREATSTOVALIDITY
|     |     |     |     |     |     |     | mance in | overfitting | detection |     | and prevention |     | than existing |     |
| --- | --- | --- | --- | --- | --- | --- | -------- | ----------- | --------- | --- | -------------- | --- | ------------- | --- |
approaches.
| A. Construct | Validity |     |     |     |     |     |     |     |     |     |     |     |     |     |
| ------------ | -------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
X. CONCLUSIONANDFUTUREWORK
| The construct |     | validity | of our | approach | may be | affected by |     |     |     |     |     |     |     |     |
| ------------- | --- | -------- | ------ | -------- | ------ | ----------- | --- | --- | --- | --- | --- | --- | --- | --- |
themanuallabellingprocessforthesimulatedtrainingdataset In this paper, we propose a non-intrusive overfitting de-
usedinoverfittingdetection.Thedefinitionofoverfittingisan
|     |     |     |     |     |     |     | tection | and prevention |     | approach | using | time | series | classi- |
| --- | --- | --- | --- | --- | --- | --- | ------- | -------------- | --- | -------- | ----- | ---- | ------ | ------- |
abstractconceptandmayresultinambiguityordisagreements fiers trained on the training history of DL models. Our
amongauthors.Tomitigatethisthreat,twoauthorslabelledthe approach (when using the KNN-DTW time series classifier)
training histories independently, achieving a 95% agreement has(1)betterclassificationperformancethancorrelation-based
rate. Following this, the authors engaged in multiple rounds approaches for overfitting detection, and (2) greater accuracy
| of discussions |     | to resolve | any disagreements |     | (as | detailed | in         |          |     |             |            |     |        |         |
| -------------- | --- | ---------- | ----------------- | --- | --- | -------- | ---------- | -------- | --- | ----------- | ---------- | --- | ------ | ------- |
|                |     |            |                   |     |     |          | than early | stopping | for | overfitting | prevention |     | with a | shorter |
Section VI-B). Despite these efforts, some subjectivity in the delay. We evaluate our approach on a real-world dataset of
labelling process might impact the validity of our results. labelled training histories collected from the papers published
Anotherpotentialthreattoconstructvalidityisthechoiceof at top AI venues in the last 5 years. Our approach can be a
themonitoringmetricusedinoverfittingprevention.Although usefultoolforresearchersanddevelopersofDLsoftware.We
validation loss is a widely used metric for monitoring the DL havesharedthetrainedtimeseriesclassifiersinthereplication
model performance during the training process, different DL package for reuse, along with all of the training histories
tasksmayrequirealternativemetrics.Weconductedadditional and labels. One limitation of our approach is that our best-
experiments using classification error (i.e., zero-one loss) for performing time series classifier takes longer to perform the

inferencerequiredtodetectandpreventoverfitthanthestudied
[20] W.FuandT.Menzies,“Easyoverhard:Acasestudyondeeplearning,”
inProceedingsofthe201711thjointmeetingonfoundationsofsoftware
| baselines. | We encourage |     | future | work | to optimize |     | time series |     |     |     |     |     |     |     |     |
| ---------- | ------------ | --- | ------ | ---- | ----------- | --- | ----------- | --- | --- | --- | --- | --- | --- | --- | --- |
engineering,2017,pp.49–60.
| classifiers | to enable | overfitting |     | detection |     | and prevention |     | in  |     |     |     |     |     |     |     |
| ----------- | --------- | ----------- | --- | --------- | --- | -------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
[21] Z.Fu,G.Lu,K.M.Ting,andD.Zhang,“Musicclassificationviathe
| real-time | with smaller |     | delays. | We also | encourage |     | the future |     |     |     |     |     |     |     |     |
| --------- | ------------ | --- | ------- | ------- | --------- | --- | ---------- | --- | --- | --- | --- | --- | --- | --- | --- |
bag-of-featuresapproach,”PatternRecognitionLetters,vol.32,no.14,
pp.1768–1777,2011.
| work to | investigate | if  | adding | more | real world | examples |     | to  |     |     |     |     |     |     |     |
| ------- | ----------- | --- | ------ | ---- | ---------- | -------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
[22] S.Gadde,A.Lakshmanarao,andS.Satyanarayana,“Smsspamdetection
| OverfitGuard’s |     | training |     | data or | using | online | training | to    |         |          |          |          |              |     |             |
| -------------- | --- | -------- | --- | ------- | ----- | ------ | -------- | ----- | ------- | -------- | -------- | -------- | ------------ | --- | ----------- |
|                |     |          |     |         |       |        |          | using | machine | learning | and deep | learning | techniques,” |     | in 2021 7th |
OverfitGuard
constantly update would improve its detec- InternationalConferenceonAdvancedComputingandCommunication
tion and prevention performance. Systems(ICACCS),vol.1,2021,pp.358–362.
|     |     |     |     |     |     |     |     | [23] J.-L.   | Gauvain  | and C.-H. | Lee, “Maximum |     | a posteriori | estimation | for  |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------ | -------- | --------- | ------------- | --- | ------------ | ---------- | ---- |
|     |     |     |     |     |     |     |     | multivariate | gaussian | mixture   | observations  |     | of markov    | chains,”   | IEEE |
REFERENCES Transactions on Speech and Audio Processing, vol. 2, no. 2, pp. 291–
298,1994.
[1] “Our replication package,” https://github.com/asgaardlab/OverfitGuard, [24] Q. Hanam, F. S. d. M. Brito, and A. Mesbah, “Discovering bug
patternsinJavaScript,”inProceedingsofthe201624thACMSIGSOFT
2023.
InternationalSymposiumonFoundationsofSoftwareEngineering(FSE
[2] B.S.AnamiandV.A.Bhandage,“Acomparativestudyofsuitabilityof
2016),2016,pp.144–156.
| certain | features | in classification |     | of bharatanatyam |     | mudra | images using |     |     |     |     |     |     |     |     |
| ------- | -------- | ----------------- | --- | ---------------- | --- | ----- | ------------ | --- | --- | --- | --- | --- | --- | --- | --- |
artificialneuralnetwork,”NeuralProcessingLetters,vol.50,no.1,pp. [25] D.J.Hand,“Principlesofdatamining,”Drugsafety,vol.30,no.7,pp.
| 741–769,2019. |          |     |         |        |         |          |            | 621–622,2007. |             |            |              |        |           |              |        |
| ------------- | -------- | --- | ------- | ------ | ------- | -------- | ---------- | ------------- | ----------- | ---------- | ------------ | ------ | --------- | ------------ | ------ |
|               |          |     |         |        |         |          |            | [26] J. Hauke | and T.      | Kossowski, | “Comparison  |        | of values | of Pearson’s | and    |
| [3] L. Bao,   | Z. Xing, | X.  | Xia, D. | Lo, M. | Wu, and | X. Yang, | “Psc2code: |               |             |            |              |        |           |              |        |
|               |          |     |         |        |         |          |            | Spearman’s    | correlation |            | coefficients | on the | same sets | of data,”    | Quaes- |
Denoisingcodeextractionfromprogrammingscreencasts,”ACMTrans.
tionesGeographicae,vol.30,no.2,pp.87–93,2011.
Softw.Eng.Methodol.,vol.29,no.3,Jun.2020.
[4] A. Barbez, F. Khomh, and Y.-G. Gue´he´neuc, “Deep learning anti- [27] D. M. Hawkins, “The problem of overfitting,” Journal of chemical
patterns from code metrics history,” in 2019 IEEE International Con- informationandcomputersciences,vol.44,no.1,pp.1–12,2004.
[28] T.Hoang,H.KhanhDam,Y.Kamei,D.Lo,andN.Ubayashi,“Deepjit:
| ference | on Software | Maintenance |     | and | Evolution | (ICSME), | 2019, pp. |     |            |               |           |     |              |        |         |
| ------- | ----------- | ----------- | --- | --- | --------- | -------- | --------- | --- | ---------- | ------------- | --------- | --- | ------------ | ------ | ------- |
|         |             |             |     |     |           |          |           | An  | end-to-end | deep learning | framework | for | just-in-time | defect | predic- |
114–124.
|     |     |     |     |     |     |     |     | tion,” | in 2019 | IEEE/ACM | 16th | International | Conference |     | on Mining |
| --- | --- | --- | --- | --- | --- | --- | --- | ------ | ------- | -------- | ---- | ------------- | ---------- | --- | --------- |
[5] M.G.Baydogan,G.Runger,andE.Tuv,“Abag-of-featuresframework
to classify time series,” IEEE Transactions on Pattern Analysis and SoftwareRepositories(MSR),2019,pp.34–45.
MachineIntelligence,vol.35,no.11,pp.2796–2802,2013. [29] D. Hoiem, T. Gupta, Z. Li, and M. Shlapentokh-Rothman, “Learning
|             |        |          |           |        |        |        |                | curves        | for analysis | of  | deep networks,” |           | in Proceedings |      | of the 38th |
| ----------- | ------ | -------- | --------- | ------ | ------ | ------ | -------------- | ------------- | ------------ | --- | --------------- | --------- | -------------- | ---- | ----------- |
| [6] G. Biau | and E. | Scornet, | “A random | forest | guided | tour,” | Test, vol. 25, |               |              |     |                 |           |                |      |             |
|             |        |          |           |        |        |        |                | International | Conference   |     | on Machine      | Learning, | vol.           | 139, | 18–24 Jul   |
no.2,pp.197–227,2016.
2021,pp.4287–4296.
| [7] C. Blake, |     | “UCI repository |     | of machine |     | learning | databases,” |     |     |     |     |     |     |     |     |
| ------------- | --- | --------------- | --- | ---------- | --- | -------- | ----------- | --- | --- | --- | --- | --- | --- | --- | --- |
https://archive.ics.uci.edu/ml/index.php,1998. [30] Q. Huang, A. Qiu, M. Zhong, and Y. Wang, “A code-description
|     |     |     |     |     |     |     |     |     |     |     |     |     |     | 2020 | IEEE 27th |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---- | --------- |
[8] J. Bornschein, F. Visin, and S. Osindero, “Small data, big decisions: representation learning model based on attention,” in
|       |           |        |            |          |                |     |             | International | Conference |     | on Software | Analysis, | Evolution |     | and Reengi- |
| ----- | --------- | ------ | ---------- | -------- | -------------- | --- | ----------- | ------------- | ---------- | --- | ----------- | --------- | --------- | --- | ----------- |
| Model | selection | in the | small-data | regime,” | in Proceedings |     | of the 37th |               |            |     |             |           |           |     |             |
neering(SANER),2020,pp.447–455.
InternationalConferenceonMachineLearning(ICML’20),2020.
[9] P.Brazdil,J.N.vanRijn,C.Soares,andJ.Vanschoren,Metalearning: [31] S. Ioffe and C. Szegedy, “Batch normalization: Accelerating deep
ApplicationstoAutomatedMachineLearningandDataMining,2022. network training by reducing internal covariate shift,” in International
conferenceonmachinelearning,2015,pp.448–456.
| [10] P. J. | Brockwell | and | R. A. | Davis, Introduction |     | to time | series and |     |     |     |     |     |     |     |     |
| ---------- | --------- | --- | ----- | ------------------- | --- | ------- | ---------- | --- | --- | --- | --- | --- | --- | --- | --- |
[32] S.Ji,B.Krishnapuram,andL.Carin,“Variationalbayesforcontinuous
forecasting,2002.
|                    |     |        |             |                |     |           |         | hidden | markov | models | and its application |     | to active | learning,” | IEEE |
| ------------------ | --- | ------ | ----------- | -------------- | --- | --------- | ------- | ------ | ------ | ------ | ------------------- | --- | --------- | ---------- | ---- |
| [11] S. Chatterjee |     | and A. | Mishchenko, | “Circuit-based |     | intrinsic | methods | to     |        |        |                     |     |           |            |      |
detectoverfitting,”inProceedingsofthe37thInternationalConference Transactions on Pattern Analysis and Machine Intelligence, vol. 28,
onMachineLearning,vol.119,13–18Jul2020,pp.1459–1468. no.4,pp.522–532,2006.
[33] H.Kim,W.Lee,andJ.Lee,“Understandingcatastrophicoverfittingin
[12] T.Chen,Z.Zhang,S.Liu,S.Chang,andZ.Wang,“Robustoverfitting
|        |           |     |          |         |               |     |               | single-step | adversarial | training,” | Proceedings |     | of the | AAAI | Conference |
| ------ | --------- | --- | -------- | ------- | ------------- | --- | ------------- | ----------- | ----------- | ---------- | ----------- | --- | ------ | ---- | ---------- |
| may be | mitigated | by  | properly | learned | smoothening,” | in  | International |             |             |            |             |     |        |      |            |
onArtificialIntelligence,vol.35,no.9,pp.8119–8127,May2021.
ConferenceonLearningRepresentations,2021.
[13] M. Choetkiertikul, H. K. Dam, T. Tran, T. Pham, A. Ghose, and [34] G.Kronberger,M.Kommenda,andM.Affenzeller,“Overfittingdetec-
T.Menzies,“Adeeplearningmodelforestimatingstorypoints,”IEEE tionandadaptivecovariantparsimonypressureforsymbolicregression,”
|     |     |     |     |     |     |     |     | in Proceedings |     | of the 13th | Annual | Conference | Companion |     | on Genetic |
| --- | --- | --- | --- | --- | --- | --- | --- | -------------- | --- | ----------- | ------ | ---------- | --------- | --- | ---------- |
TransactionsonSoftwareEngineering,vol.45,no.7,pp.637–656,2019.
andEvolutionaryComputation(GECCO’11),2011,pp.631–638.
| [14] A. Ciurumelea, |     | S. Proksch, |       | and H. C.       | Gall,    | “Suggesting | comment      |                   |     |         |        |          |                 |     |          |
| ------------------- | --- | ----------- | ----- | --------------- | -------- | ----------- | ------------ | ----------------- | --- | ------- | ------ | -------- | --------------- | --- | -------- |
|                     |     |             |       |                 |          |             |              | [35] J. Kukacˇka, | V.  | Golkov, | and D. | Cremers, | “Regularization |     | for deep |
| completions         | for | python      | using | neural language | models,” |             | in 2020 IEEE |                   |     |         |        |          |                 |     |          |
27th International Conference on Software Analysis, Evolution and learning:Ataxonomy,”arXivpreprintarXiv:1710.10686,2017.
Reengineering(SANER),2020,pp.456–467. [36] V. Kuleshov, N. Fenner, and S. Ermon, “Accurate uncertainties for
|                 |     |       |          |            |                  |     |          | deep | learning using | calibrated | regression,” |     | in Proceedings |     | of the 35th |
| --------------- | --- | ----- | -------- | ---------- | ---------------- | --- | -------- | ---- | -------------- | ---------- | ------------ | --- | -------------- | --- | ----------- |
| [15] E. Damiani | and | C. A. | Ardagna, | “Certified | machine-learning |     | models,” |      |                |            |              |     |                |     |             |
InternationalConferenceonMachineLearning,vol.80,10–15Jul2018,
| in SOFSEM | 2020: | Theory | and | Practice | of Computer |     | Science: 46th |     |     |     |     |     |     |     |     |
| --------- | ----- | ------ | --- | -------- | ----------- | --- | ------------- | --- | --- | --- | --- | --- | --- | --- | --- |
pp.2796–2804.
InternationalConferenceonCurrentTrendsinTheoryandPracticeof
Informatics, SOFSEM 2020, Limassol, Cyprus, January 20–24, 2020, [37] Z. Li, T.-H. Chen, and W. Shang, “Where shall we log? studying and
Proceedings,2020,p.3–15. suggestinglogginglocationsincodeblocks,”in202035thIEEE/ACM
|     |     |     |     |     |     |     |     | International | Conference |     | on Automated |     | Software | Engineering | (ASE), |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------- | ---------- | --- | ------------ | --- | -------- | ----------- | ------ |
[16] H.Deng,G.Runger,E.Tuv,andM.Vladimir,“Atimeseriesforestfor
2020,pp.361–372.
classificationandfeatureextraction,”InformationSciences,vol.239,pp.
|     |     |     |     |     |     |     |     | [38] J. Lin, | E. Keogh, | L. Wei, | and S. | Lonardi, | “Experiencing |     | sax: a novel |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------ | --------- | ------- | ------ | -------- | ------------- | --- | ------------ |
142–153,2013.
[17] H. Ding, G. Trajcevski, P. Scheuermann, X. Wang, and E. Keogh, symbolic representation of time series,” Data Mining and knowledge
discovery,vol.15,no.2,pp.107–144,2007.
| “Querying | and | mining | of time | series | data: Experimental |     | comparison |              |        |          |        |                  |     |          |            |
| --------- | --- | ------ | ------- | ------ | ------------------ | --- | ---------- | ------------ | ------ | -------- | ------ | ---------------- | --- | -------- | ---------- |
|           |     |        |         |        |                    |     |            | [39] F. Liu, | G. Li, | Y. Zhao, | and Z. | Jin, “Multi-task |     | learning | based pre- |
ofrepresentationsanddistancemeasures,”Proc.VLDBEndow.,vol.1,
trainedlanguagemodelforcodecompletion,”in202035thIEEE/ACM
no.2,pp.1542–1552,Aug.2008.
[18] Z.Ding,J.Chen,andW.Shang,“Towardstheuseofthereadilyavailable International Conference on Automated Software Engineering (ASE),
| testsfromthereleasepipelineasperformancetests:Arewethereyet?” |     |        |          |      |               |     |            | 2020,pp.473–485. |         |         |            |             |              |     |         |
| ------------------------------------------------------------- | --- | ------ | -------- | ---- | ------------- | --- | ---------- | ---------------- | ------- | ------- | ---------- | ----------- | ------------ | --- | ------- |
|                                                               |     |        |          |      |               |     |            | [40] Z. Liu,     | X. Xia, | M. Yan, | and S. Li, | “Automating | just-in-time |     | comment |
| in Proceedings                                                |     | of the | ACM/IEEE | 42nd | International |     | Conference | on               |         |         |            |             |              |     |         |
updating,”in202035thIEEE/ACMInternationalConferenceonAuto-
SoftwareEngineering(ICSE’20),2020,pp.1435–1446.
matedSoftwareEngineering(ASE),2020,pp.585–597.
[19] S.Fakhoury,V.Arnaoudova,C.Noiseux,F.Khomh,andG.Antoniol,
“Keepitsimple:Isdeeplearninggoodforlinguisticsmelldetection?” [41] Z. Liu, Z. Xu, J. Jin, Z. Shen, and T. Darrell, “Dropout reduces
in 2018 IEEE 25th International Conference on Software Analysis, underfitting,”arXivpreprintarXiv:2303.01500,2023.
|     |     |     |     |     |     |     |     | [42] J. D. | Long, D. | Feng, and | N. Cliff, | “Ordinal | Analysis | of  | Behavioral |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------- | -------- | --------- | --------- | -------- | -------- | --- | ---------- |
EvolutionandReengineering(SANER),2018,pp.602–611.

Data,”inHandbookofPsychology,Apr.2003,ch.25,pp.635–661. Applications:WithRExamples,ser.SpringerTextsinStatistics,2017.
[43] A.Mahadi,K.Tongay,andN.A.Ernst,“Cross-datasetdesigndiscussion [66] V. Singla, S. Singla, S. Feizi, and D. Jacobs, “Low curvature activa-
mining,” in 2020 IEEE 27th International Conference on Software tions reduce overfitting in adversarial training,” in Proceedings of the
Analysis,EvolutionandReengineering(SANER),2020,pp.149–160. IEEE/CVF International Conference on Computer Vision (ICCV), Oct.
[44] H.B.MannandD.R.Whitney,“Onatestofwhetheroneoftworandom 2021,pp.16423–16433.
variablesisstochasticallylargerthantheother,”AnnalsofMathematical [67] E. K. Smith, E. T. Barr, C. Le Goues, and Y. Brun, “Is the cure
Statistics,vol.18,pp.50–60,1947. worse than the disease? overfitting in automated program repair,” in
[45] F. Mohr and J. N. van Rijn, “Towards model selection using learning Proceedingsofthe201510thJointMeetingonFoundationsofSoftware
curve cross-validation,”in 8th ICMLWorkshop on automatedmachine Engineering(ESEC/FSE2015),2015,pp.532–543.
learning(AutoML),2021. [68] C.Spearman,“Theproofandmeasurementofassociationbetweentwo
[46] F. Mohr and J. N. van Rijn, “Learning curves for decision making in things,”TheAmericanjournalofpsychology,vol.100,no.3/4,pp.441–
supervised machine learning - A survey,” CoRR, vol. abs/2201.12150, 471,1987.
2022. [69] N.Srivastava,G.Hinton,A.Krizhevsky,I.Sutskever,andR.Salakhut-
[47] K.MolugaramandG.S.Rao,StatisticalTechniquesforTransportation dinov, “Dropout: a simple way to prevent neural networks from over-
Engineering,Jan.2017. fitting,” The journal of machine learning research, vol. 15, no. 1, pp.
[48] N.MorganandH.Bourlard,“Generalizationandparameterestimation 1929–1958,2014.
infeedforwardnets:Someexperiments,”Advancesinneuralinformation [70] B. Strang, P. v. d. Putten, J. N. v. Rijn, and F. Hutter, “Don’t rule out
processingsystems,vol.2,1989. simplemodelsprematurely:Alargescalebenchmarkcomparinglinear
[49] A. Nilizadeh, G. T. Leavens, X.-B. D. Le, C. S. Pa˘sa˘reanu, and D. R. and non-linear classifiers in openml,” in Advances in Intelligent Data
Cok, “Exploring true test overfitting in dynamic automated program AnalysisXVII,2018,pp.303–315.
repair using formal methods,” in 2021 14th IEEE Conference on [71] J.Straub,“Machinelearningperformancevalidationandtrainingusing
SoftwareTesting,VerificationandValidation(ICST),2021,pp.229–240. a‘perfect’expertsystem,”MethodsX,vol.8,p.101477,2021.
[50] T.Peng,L.Liu,andW.Zuo,“PUtextclassificationenhancedbyterm [72] C. Tantithamthavorn, S. McIntosh, A. E. Hassan, and K. Matsumoto,
frequency–inverse document frequency-improved weighting,” Concur- “An empirical comparison of model validation techniques for defect
rency and Computation: Practice and Experience, vol. 26, no. 3, pp. predictionmodels,”IEEETransactionsonSoftwareEngineering,vol.43,
728–741,2014. no.1,pp.1–18,2017.
[51] L.Prechelt,“EarlyStopping—ButWhen?”inNeuralNetworks:Tricks [73] Y. Tian, K. Pei, S. Jana, and B. Ray, “Deeptest: Automated testing
oftheTrade:SecondEdition,ser.LectureNotesinComputerScience, ofdeep-neural-network-drivenautonomouscars,”inProceedingsofthe
2012,pp.53–67. 40th International Conference on Software Engineering (ICSE ’18),
[52] L. Prechelt et al., “Proben1: A set of neural network benchmark 2018,pp.303–314.
problemsandbenchmarkingrules,”1994. [74] S. Tokui, S. Tokumoto, A. Yoshii, F. Ishikawa, T. Nakagawa, K. Mu-
[53] X. Ren, Z. Xing, X. Xia, D. Lo, X. Wang, and J. Grundy, “Neural nakata, and S. Kikuchi, “Neurecover: Regression-controlled repair of
network-based detection of self-admitted technical debt: From perfor- deep neural networks with training history,” in 2022 IEEE Interna-
mance to explainability,” ACM Trans. Softw. Eng. Methodol., vol. 28, tional Conference on Software Analysis, Evolution and Reengineering
no.3,Jul.2019. (SANER),2022,pp.1111–1121.
[54] L. Rice, E. Wong, and Z. Kolter, “Overfitting in adversarially robust [75] M. Tufano, C. Watson, G. Bavota, M. D. Penta, M. White, and
deeplearning,”inProceedingsofthe37thInternationalConferenceon D. Poshyvanyk, “An empirical study on learning bug-fixing patches
MachineLearning,vol.119,13–18Jul2020,pp.8093–8104. in the wild via neural machine translation,” ACM Trans. Softw. Eng.
[55] J.Romano,J.D.Kromrey,J.Coraggio,J.Skowronek,andL.Devine, Methodol.,vol.28,no.4,Sep.2019.
“ExploringmethodsforevaluatinggroupdifferencesontheNSSEand [76] J.N.vanRijn,S.M.Abdulrahman,P.Brazdil,andJ.Vanschoren,“Fast
othersurveys:Arethet-testandCohen’sdindicesthemostappropriate algorithm selection using learning curves,” in Advances in Intelligent
choices,”inannualmeetingoftheSouthernAssociationforInstitutional DataAnalysisXIV,2015,pp.298–309.
Research. Citeseer,2006,pp.1–51. [77] O.Varol,E.Ferrara,F.Menczer,andA.Flammini,“Earlydetectionof
[56] S.Sajeev,A.Maeder,S.Champion,A.Beleigoli,C.Ton,X.Kong,and promoted campaigns on social media,” EPJ data science, vol. 6, pp.
M. Shu, “Deep learning to improve heart disease risk prediction,” in 1–19,2017.
MachineLearningandMedicalEngineeringforCardiovascularHealth [78] Z. Wang, L. Wang, C. Huang, Z. Zhang, and X. Luo, “Soil-moisture-
and Intravascular Imaging and Computer Assisted Stenting, 2019, pp. sensor-based automated soil water content cycle classification with a
96–103. hybridsymbolicaggregateapproximationalgorithm,”IEEEInternetof
[57] G. Salton, A. Wong, and C.-S. Yang, “A vector space model for ThingsJournal,vol.8,no.18,pp.14003–14012,2021.
automaticindexing,”CommunicationsoftheACM,vol.18,no.11,pp. [79] C. Watson, N. Cooper, D. N. Palacio, K. Moran, and D. Poshyvanyk,
613–620,1975. “Asystematicliteraturereviewontheuseofdeeplearninginsoftware
[58] S. Salvador and P. Chan, “Toward accurate dynamic time warping in engineeringresearch,”ACMTrans.Softw.Eng.Methodol.,vol.31,no.2,
lineartimeandspace,”Intell.DataAnal.,vol.11,no.5,pp.561–580, Mar.2022.
Oct.2007. [80] B. Wei, Y. Li, G. Li, X. Xia, and Z. Jin, “Retrieve and refine:
[59] P. Scha¨fer, “Scalable time series classification,” Data Mining and Exemplar-basedneuralcommentgeneration,”inProceedingsofthe35th
KnowledgeDiscovery,vol.30,no.5,pp.1273–1298,2016. IEEE/ACMInternationalConferenceonAutomatedSoftwareEngineer-
[60] P. Scha¨fer and M. Ho¨gqvist, “SFA: A symbolic fourier approximation ing(ASE’20),2021,pp.349–360.
and index for similarity search in high dimensional datasets,” in Pro- [81] R.Werpachowski,A.Gyo¨rgy,andC.Szepesva´ri,“Detectingoverfitting
ceedings of the 15th International Conference on Extending Database via adversarial examples,” Advances in Neural Information Processing
Technology(EDBT’12),2012,pp.516–527. Systems,vol.32,2019.
[61] O. B. Sghaier and H. Sahraoui, “A multi-step learning approach to [82] X.Xi,E.Keogh,C.Shelton,L.Wei,andC.A.Ratanamahatana,“Fast
assistcodereview,”in2023IEEEInternationalConferenceonSoftware time series classification using numerosity reduction,” in Proceedings
Analysis,EvolutionandReengineering(SANER),2023,pp.450–460. of the 23rd international conference on Machine learning, 2006, pp.
[62] L. Shi, M. Xing, M. Li, Y. Wang, S. Li, and Q. Wang, “Detection of 1033–1040.
hidden feature requests from massive chat messages via deep siamese [83] Q.XinandS.P.Reiss,“Identifyingtest-suite-overfittedpatchesthrough
network,” in 2020 IEEE/ACM 42nd International Conference on Soft- test case generation,” in Proceedings of the 26th ACM SIGSOFT In-
wareEngineering(ICSE),2020,pp.641–653. ternationalSymposiumonSoftwareTestingandAnalysis(ISSTA2017),
[63] Q. Shi, R. Katuwal, P. Suganthan, and M. Tanveer, “Random vector 2017,pp.226–236.
functional link neural network based ensemble deep learning,” Pattern [84] B.Xu,D.Ye,Z.Xing,X.Xia,G.Chen,andS.Li,“Predictingseman-
Recognition,vol.117,p.107978,2021. ticallylinkableknowledgeindeveloperonlineforumsviaconvolutional
[64] C.ShortenandT.M.Khoshgoftaar,“Asurveyonimagedataaugmen- neuralnetwork,”in201631stIEEE/ACMInternationalConferenceon
tation for deep learning,” Journal of big data, vol. 6, no. 1, pp. 1–48, AutomatedSoftwareEngineering(ASE),2016,pp.51–62.
2019. [85] Y.Yang,X.Xia,D.Lo,andJ.Grundy,“Asurveyondeeplearningfor
[65] R. H. Shumway and D. S. Stoffer, Time Series Analysis and Its softwareengineering,”ACMComput.Surv.,vol.54,no.10s,Sep.2022.

| [86] F. Zampetti, | A. Serebrenik,    | and         | M. Di Penta, | “Automatically | learning          |
| ----------------- | ----------------- | ----------- | ------------ | -------------- | ----------------- |
| patterns          | for self-admitted | technical   | debt         | removal,”      | in 2020 IEEE 27th |
| International     | Conference        | on Software | Analysis,    | Evolution      | and Reengi-       |
neering(SANER),2020,pp.355–366.
| [87] J. M. Zhang, | E. T.             | Barr, B. Guedj, | M. Harman,      |     | and J. Shawe-Taylor, |
| ----------------- | ----------------- | --------------- | --------------- | --- | -------------------- |
| “Perturbed        | model validation: |                 | A new framework |     | to validate model    |
relevance,”CoRR,vol.abs/1905.10201,2019.

---

## Content from Previous Extraction (not in markitdown output)

### Visual/Chart/Graph Descriptions

#### 1

Abstract—In software engineering, deep learning models are
                                                                                                                            Training loss                              Training loss          increasingly deployed for critical tasks such as bug detection and                                                                                                                              Validation loss                            Validation loss
         code review. However, overfitting remains a challenge that affects
          the quality, reliability, and trustworthiness of software systems
          that utilize deep learning models. Overfitting can be (1) prevented
             (e.g., using dropout or early stopping) or (2) detected in a trained2024
        model (e.g., using correlation-based approaches). Both overfitting
          detection and prevention approaches that are currently used have
          constraints (e.g., requiring modification of the model structure,Jan                                                                                              (a) Overfitting                   (b) Non-overfitting        and high computing resources). In this paper, we propose a
18   simple,overfittingyet basedpowerfulon theapproachtrainingthathistorycan both(i.e.,detectvalidationand preventlosses).   Fig. 1: Examples of overfit and non-overfit training histories.
       Our approach  first trains a time series classifier on training
           histories of overfit models. This classifier is then used to detect if
        a trained model is overfit. In addition, our trained classifier can   layers [69] or batch normalization [31]. However, many of
        be used to prevent overfitting by identifying the optimal point
                                                                          these approaches are intrusive and require modifying the data          to stop a model’s training. We evaluate our approach on  its
           ability to identify and prevent overfitting in real-world samples.   or the model structure and expertise to execute correctly and[cs.SE]  We compare our approach against correlation-based detection   even then, they may not work. For instance, adding dropout
         approaches and the most commonly used prevention approach   layers, a popularly used overfitting prevention scheme, when
              (i.e., early stopping). Our approach achieves an F1 score of 0.91   set with a lower threshold or when added  to the  earlier
        which is at least 5% higher than the current best-performing
                                                                            layers may cause unintentional overfitting [41]. Furthermore,          non-intrusive overfitting detection approach. Furthermore, our
        approach can stop training to avoid overfitting at least 32% of   even the non-intrusive prevention approaches such as early
          the times earlier than early stopping and has the same or a better   stopping incur trade-offs between model accuracy and training
          rate of returning the best model.                              time [51]. For example, late stopping when using the early
           Index Terms—Software engineering for AI, AI for Software   stopping approach may improve model accuracy, but  it will
         Engineering, Overfitting, Training history, Deep learning
                                                                           also increase training time. Conversely, stopping too early
                                                                    could result in a sub-optimal model performance.
                                        I. INTRODUCTION
                                                                               Overfitting   detection  approaches   like   k-fold   cross-
         The use of Deep Learning (DL) models in software engi-   validation,  training the DL model with noisy data points
         neering (SE) research and software products has been skyrock-  and observing  if the added noise impacts the DL model’s
          eting over the past decade. For instance, DL techniques have   accuracy [87], checking if the hypothesis of the trained model
        been used for automated bug detection [24], code review [61],  and the data are independent [81] can generally be resourcearXiv:2401.10359v1  and software testing [73]. These applications underscore the   intensive and time consuming. For instance, Xu et al. [84]
        importance and ubiquity of DL in modern SE.                  report that training a DL model to find two semantically
            Overfitting is one of the fundamental issues that plagues DL   linkable questions in StackOverflow takes about 14 hours.
        models [33, 49, 67, 79, 85]. A DL model can be considered    If one were to conduct a 5-fold cross-validation to detect if
          overfitting if the model fits just the training data instead of   the constructed DL model is overfitting, they would have to
         learning the target hypothesis [79]. An overfit model increases   invest 70 hours, which might be prohibitive in practice.
         the risk of inaccurate predictions, misleading feature impor-     In this paper, we introduce OverfitGuard, an approach
          tance, and wasted resources [27].                                 to both detect and prevent overfitting using training histories.
            Currently, the problem of overfitting  is addressed in SE   Figure 1 illustrates example training histories (i.e., the training
          studies that use DL models by either (1) preventing  it from  and validation losses curves) of an overfit and a non-overfit
        happening in the  first place or (2) detecting  it in a trained  DL model. The training and validation losses of the overfit
      DL model [79, 85]. Overfitting prevention approaches  in-  model both decrease at the beginning of the training process.
         clude early stopping [48], data augmentation [64], regulariza-   Following that, the validation loss increases while the training
          tion [35], and modifying the DL model by adding dropout   loss decreases, resulting in a large gap between the training

#### 2

TABLE I: Studied time series classifiers.and validation losses. Such a trend indicates poor general-
ization of the trained model to new data. Researchers have                                                                                         Classifier       Description
previously employed training histories for decision-making
                                                   KNN-DTW∗   Uses K-Nearest Neighbors  [25] with Dynamic Time
in  areas such  as  quantitative  data  acquisition and model                   Warping [17] as the distance metric to classify time series
selection [8, 9, 29, 45, 46, 70, 76]. Similarly, our approach                      data
trains a time series classifier on a simulated dataset of training   HMM-GMM   Uses Hidden Markov Model for modelling time series
                                                                                                 data and Gaussian Mixture Model as the emissions prob-
histories (i.e., labelled validation loss curves over epochs of                         ability density [23, 32]
training) of DL models that overfit the training data. Our    TSF           Uses a random forest [6] for time series data using an
trained time series classifier detects overfitting in a trained DL                   ensemble of time series trees [16]
                                                        TSBF         Time Series Bag-of-Features [5] extracts features based
model by examining the validation loss history (captured as                  on the bag-of-features approach [21] to create a random
part of the training history). In contrast to existing overfitting                        forest
detection approaches, our approach does not incur additional    SAX-VSM     Symbolic Aggregate approXimation [38] converts the
                                                                                                 data  into symbolic  representations and Vector Space
resources or costs, as the training history (also known as                 Model [50, 57] transforms them into vectors to calculate
the learning curve)  is a natural byproduct of the training                        similarity for classification
process. Furthermore, our approach (i.e., the trained time series    BOSSVS      Bag-of-SFA Symbols in Vector Space [59] is similar to
                                                               SAX-VSM but use SFA [60] to transform the data instead
classifier) can be used to prevent overfitting based on the                      of SAX
validation losses of recent epochs (e.g., the last 20 epochs).    ∗KNN-DTW handles variable-length time series data.
  Although our approach is trained on a simulated dataset,
we evaluate it on a real-world dataset, collected from papers
published in top AI venues within the last 5 years. We gathered
                                                                   Typically, a dataset is divided into training, validation, and testthe training histories from these papers that are explicitly
                                                                           sets. While the training loss reflects how well the DL modellabelled as overfitting or non-overfitting by the authors as
                                                                  learns from the training data during the training process, thethe ground truth. The main contributions of this paper are as
                                                                  validation loss is evaluated based on the validation data, whichfollows:
                                                                serves as a proxy for evaluating the model’s performance
  • Our approach outperforms the state-of-the-art by at least
                                                     on unseen data. After training is completed, the trained DL
   5% in terms of F-score for overfitting detection, achieving
                                                          model’s performance is evaluated using the test set that has
    an F-score of 0.91.
                                                              not been exposed to the model.
  • Our approach has the capability to prevent overfitting at
     least 32% earlier than early stopping while maintaining     Researchers and developers can identify potential issues
     (and often surpassing) the rate of reaching the optimal   such as overfitting or underfitting by analyzing the training
    epoch (i.e., the epoch that yields the best model).           histories. For example, overfitting  is often observed as an
  • We provide a  replication package  [1] containing our   increasing divergence between training loss and validation loss
     trained classifiers and labelled training histories which   over time (as illustrated in Figure 1a). In this paper, we propose
    can be directly used by other researchers.               an approach that leverages training history to automatically
                                                                  detect and prevent overfitting in DL models.  Paper organization. This paper  is organized as follows.
Section II provides background information about our study.
Section  III gives an overview of related work. Section IV
introduces existing approaches for detecting and preventing
                                                             B. Time series classification
overfitting. Section V  describes  the design  of our  study,
while Section VI provides detailed information about our
                                                     Time series data consists of data points recorded over time,experimental setup. Sections VII and VIII present the results of
                                                           with each point being associated with a specific timestampour study. Section IX discusses potential threats to the validity
                                                       and  its corresponding value. Time series classification  is aof our study. Finally, Section X concludes the paper.
                                                       machine learning task  that aims to categorize time  series
                          II. BACKGROUND                        data into predefined  classes. In the context of our study,
  This section provides an introduction to the concepts of the  we consider the training history of a DL model as time
training history in DL and time series classification.             series data, as the task of identifying whether a DL model
                                                                               is overfitting based on its training history can be framed as
A. Leveraging training history in DL                                                           a time series classification problem. Since there has been no
  Training  history,  also known  as  the  empirical  learning   prior systematic research on time series classifiers specifically
curve [46], provides valuable insights into a DL model’s learn-   designed for training histories, we have selected six classifiers
ing progress and performance throughout the training process.  (shown in Table I) that have been reported as baselines or state-
The training history stores a record of metrics during the   of-the-art in prior studies [2, 77, 78, 82]. These classifiers were
training process, which are usually recorded in each training   chosen due to their demonstrated effectiveness in various time
iteration or epoch (as shown in Figure 1). Training loss and   series classification tasks and their potential applicability to
validation loss are commonly used metrics in training histories.   the overfitting detection problem in DL.

#### 3

III. RELATED WORK                         ferentiates overfitting patches based on semantic differences.
                                                             Nilizadeh et al. [49] utilized formal verification to evaluate
A. Mitigating overfitting in SE
                                                                 the degree of overfitting and identified the challenges posed
   Overfitting poses a significant risk to the trustworthiness  by program complexity and numeric issues.
of software systems and the research studies that employ    To the best of our knowledge, both the overfitting prevention
DL models. SE researchers typically use either overfitting  and detection methods used in both SE studies and in practice
detection or prevention methods to mitigate the problem of    fall prey to several key concerns. The overfitting prevention
overfitting. Among the overfitting prevention methods, dropout   methods,  typically  require  significant  expertise  to execute
is the most commonly adopted approach [79, 85]. Researchers   correctly and are intrusive (for instance, they may require
have used dropout in various domains such as code gen-  one to modify the data or the model structure). Overfitting
eration  [39,  40],  logging  locations recommendation  [37],   detection approaches typically require retraining of the DL
and comment completion [14, 80]. Regularization is another  model multiple times, which may be very costly in prac-
prominent overfitting prevention strategy that has been used   tice [20] and simple methods like early stop may stop the
in SE [4, 30, 84, 86], which requires adding another layer to   training of DL model sub optimally. Our work addresses these
the model structure as well. For example, Zampetti et al. [86]   gaps by introducing a history-based approach that serves the
employed L2-norm regularization in training CNN and RNN   dual purpose of detecting and preventing overfitting in a non-
models to manage self-admitted technical debt in source code.   intrusive manner without any need for DL model retraining.
  Early stopping  is another frequently used technique  to
prevent overfitting during the training process of DL mod-   B. Leveraging training history to improve DL software quality
els [13, 28, 62]. For example, Shi et al. [62] utilized early     While the machine learning community has leveraged train-
stopping when training a deep Siamese network to identify   ing histories  (i.e., learning curves) for different tasks, the
hidden feature requests posted in chat messages by developers.  SE community has seldom used training history to enhance
Other techniques like data augmentation [3, 19, 43] and data   software quality. Mohr and van Rijn [46] conduct a survey
balancing [53, 75] are also employed to address overfitting.  on approaches based on learning curves for decision-making
For instance, Bao et al. [3] developed a CNN-based image   in DL domains, such as data  acquisition, early stopping,
classification model to  filter out non-code and noisy-code  and model selection. They also propose an approach called
frames from programming screencasts. To enhance training   Learning Curve Cross-Validation (LCCV) [45] that iteratively
data diversity, they employed data augmentation techniques   increases the number of training examples used for training to
such as rotation, scaling, translation, and shearing.               select the best model from the candidates. Training histories
  In terms of overfitting detection approaches, Zhang et al.   can also be used to evaluate the trained classifiers: van Rijn
[87] propose the perturbation validation (PV) assessment to   et al. [76] propose an approach that recommends classifiers
determine whether a DL model fits the training data properly   for a given dataset based on training histories using loss time
(i.e., ensure that it is neither overfitting nor underfit). Alterna-   curves. Moreover, Hoiem et  al. [29] investigate the use of
tively, some detection approaches check the hypothesis that the   training histories to evaluate design choices for DL models,
trained DL model and the data are independent. For example,   such as pretraining, architecture, and data augmentation.
Werpachowski et al. [81] check the hypothesis by comparing     In this paper, we propose OverfitGuard, an approach
the test error with the estimated test error based on adversarial   that utilizes time-series classifiers to detect and prevent over-
examples of the test set.                                                fitting by analyzing the training history of DL models. Our
  Another popular approach towards detecting overfiting is   approach aims to enhance the software quality of DL systems
to use model validation approaches. Tantithamthavorn et al.  by improving  the  quality  of  the DL models themselves.
[72] evaluated 12 model validation techniques specifically for  OverfitGuard is one of the first approaches that leverages
defect prediction models and concluded that out-of-sample   training history to enhance the software quality of DL systems.
bootstrap emerges as the least biased and most stable tech-  A related work by Tokui et al. [74] introduces an approach
nique that helps detect overfit. Damiani and Ardagna [15]   called NeuRecover which also leverages training history to im-
introduced a framework  that validates DL models against   prove the quality of DL models, particularly for safety-critical
desired non-functional properties and statistically monitors the   applications. NeuRecover identifies the model parameters that
model output. Straub [71] extended this by utilizing randomly   need to be modified (i.e., repaired) by analyzing the training
generated expert networks for model validation that focuses   history to address specific failure types. Our approach shares
on performance characteristics.                                     similarities with NeuRecover, but focuses on the detection and
  Other than these approaches, Smith et al. [67] discussed the   prevention of overfitting, which is important for maintaining
role of human factors in overfitting and provided a comparative   the software quality of DL systems.
analysis between automated patches and human-written fixes.
                                                                               IV. EXISTING APPROACHThey reported that overfitting is not solely a machine-induced
problem and suggested focusing on the contributing factors     In this section, we introduce the existing approaches that we
like test suite coverage and requirements-based testing. Xin   use as baseline approaches to compare our proposed approach
and Reiss [83] proposed a classification technique that dif-   for overfitting detection and prevention in DL models.

#### 4

A. Overfitting detection                                                  is also used by PyTorch Ignite.2 As shown in Figure 2a, early
                                                             stopping with a patience parameter of 20 epochs stops at the
                                                     #70 epoch and returns the #50 with the lowest validation loss  Correlation-based approaches. One approach for detecting
                                                                since no improvement occurs between epochs #50 and #70.overfitting in DL models is to compute the correlation between
                                                            Furthermore, early stopping with larger patience values (e.g.,the training and validation loss. Kronberger et al. [34] propose
                                                     40 epochs) stops later (at the #140 epoch) but with a lowercomputing the non-parametric Spearman’s rank correlation
                                                                    loss (at the #100 epoch).coefficient [68] between  training and validation  fitness  in
                                                         Early stopping based on smoothed validation loss curves.symbolic regression models to detect overfitting. Similar to our
                                           An alternate version of early stopping inspects the movingapproach, correlation-based approaches also detect overfitting
                                                            average of the smoothed validation loss curves [47, 65] tobased on the training history. Therefore, we select them as
                                                             decide when to stop the training process. After stopping thea baseline for comparison. In this study, we calculate the
                                                                    training process, this approach returns the best epoch whichcorrelation metrics between the training and validation loss
                                                           has the lowest validation loss (not the smoothed value).to detect overfitting in DL models. The underlying principle
is straightforward: when no overfitting occurs, the training
                                                                              V. OUR APPROACH
and validation losses should be  correlated, whereas weak
correlation implies overfitting.                                   Figure 3 shows an overview of our proposed approach. Our
  We determine the presence of overfitting by comparing   approach uses a time series classifier to detect and prevent
the computed correlation-based metric between training and   overfitting. Table I lists the studied time series classifiers. First,
validation losses with a predetermined threshold value (deter-  we collect a simulated dataset (more details on how we collect
mined in Section VI-D). We use three different correlation   the data in Section VI) that contains training histories (i.e.,
metrics: Spearman, Pearson [26], and time-lagged Pearson   training and validation loss curves, however, we only use the
correlation coefficients. Both Spearman and Pearson correla-   validation loss curves in our approach) with labels indicating
tion coefficients are calculated since we do not know whether   whether overfitting occurs in order to train our time series
the relationship between training and validation loss is lin-    classifier. Second, we train and evaluate each studied time
ear or not. In addition, we compute the Pearson correlation   series classifier on all of the training histories of the simulated
coefficient between a 5-epoch lagged version of the training   dataset. Finally, we use the trained time series classifier to
loss and the validation loss. This approach  is inspired by   perform both overfitting detection and prevention as follows.
autocorrelation [10], which measures the correlation between     Overfitting detection. To detect overfitting in a trained DL
a time series data and a time-lagged version of itself.          model, we first collect its validation losses over the training
                                                             epochs. We feed this loss to our trained time series classifier
B. Overfitting prevention
                                                                    to detect whether there  is overfitting. However, we cannot
  Early stopping. One widely used approach for preventing   directly feed these validation losses to our classifier, since the
overfitting is early stopping, which stops training when there is   length of the validation losses might not be of the same length
no improvement in a fixed number of epochs (called patience)   as that of the data used to train these time series classifiers.
and returns the epoch which has the lowest validation loss. We   All the studied time series classifiers, with the exception of
choose the widely used TensorFlow implementation,1 which
                                                                                       2https://pytorch.org/ignite/generated/ignite.handlers.early stopping.
   1https://tensorflow.org/api docs/python/tf/keras/callbacks/EarlyStopping      EarlyStopping

#### 5

preventionhistory in a fixed window size (e.g., the latest 20 epochs), and             approachesTrain our               classifiersseries                Early stopping              overfitting
(2) as the whole observed history (from the first to the latest                                                                           approaches
epochs). Our time series classifier detects if in the fed history
overfitting occurs. Similar to overfitting detection, we linearly
interpolate the data before feeding it into our model. If there            Fig. 4: Overview of the experimental setup.
is no overfit occurring, we continue the training and repeat
the above procedure until the DL model has finished training.
For the rolling window, we move the window by a fixed step   Python 3.8 and TensorFlow 2.9.0. The hardware configuration
size (as shown in Figure 2b) and make another prediction. If   for the experiments is detailed below:
our model detects the presence of overfitting in the fed history,     • NVIDIA RTX 3090 GPU with 24 GB memory
we return the lowest validation loss in the observed epochs as     • CUDA version 10.1.243
the best epoch.                                                    • cuDNN version 7.6.5
                                                                   •  Intel(R) Core(TM) i9-11900K CPU with a clock speed
              VI. EXPERIMENTAL SETUP                        of 3.50 GHz
                                                                   • 64 GB of RAM
  In this section, we introduce the datasets for training and
evaluating the studied  overfitting detection and prevention   B. Simulated training dataset
approaches, the experiments of our study, and the evalua-
                                          We create a simulated dataset containing training histories
tion metrics for the studied approaches. Figure 4 shows an
                                                           with labels to determine a threshold for correlation-based
overview of the experimental setup.
                                                           approaches (see Section IV) and to train our proposed method
                                                               as described in Section V. We create this simulated dataset
A. Environment setting
                                                     by training neural networks of varying model complexities to
  We conducted the experiments on an Ubuntu 20.04 oper-   produce overfitting and non-overfitting samples. The process
ating system with a Linux kernel version of 5.15.0, utilizing    is as follows:

#### 6

TABLE V: The optimal  rate, median  delay, and averageis that the extracted state models (via HMM) of the training
                                                            accuracy of our overfitting prevention approaches using thehistories do not follow a Gaussian probability distribution.
                                                      whole observed history.Our approach with BOSSVS correctly identifies all the data
in the simulated dataset but performs poorly on the  real-
world dataset. One reason could be that the extracted bag-                    Classifier       Optimal   Median    Average
                                                                                                                               rate      delay   accuracy
of-SFA symbols (BOSS) from the simulated dataset does not
generalize to the real-world  dataset. In addition, we note           KNN-DTW        0.95      43.5        0.42
                                                   HMM-GMM       0.18        0.0        0.36
that the  investigated correlation-based approaches perform             TSF               0.90      35.0        0.42
reasonably well, with F-scores greater than 0.8. However,            TSBF              0.83      31.0        0.41
our approach outperforms the correlation-based  overfitting            BOSSVS          0.65      21.0        0.41
                                                             SAX-VSM         0.33      10.0        0.38
detection approach by at least 5% on the studied real-world
dataset.
  The studied time series classifiers are more computationally
intensive than correlation-based approaches for inference, yet    VIII. RQ2: HOW WELL DOES OV E R F I TGU A R D PREVENT
they are still useful in practice. As shown in Table IV, our       OVERFITTING DURING THE TRAINING PROCESS?
approach requires more time for performing inference than
                                                             Motivation. Another critical part of developing trustworthy
the correlation-based approaches. For instance, TSF has the
                                                       and stable DL models is preventing overfitting. An effective
fastest inference time among the classifiers but is around 20
                                                                      overfitting prevention approach allows DL models to gener-
times slower than the Spearman correlation-based approach
                                                                      alize better on unseen data while minimizing both training
and around 700 times slower than the other two correlation-
                                                               resources and computational  costs. This research question
based approaches. However, the speed of our approach  is
                                                                 evaluates the performance of our proposed approach for pre-
not prohibitive in practice since overfitting detection is only
                                                              venting overfitting during the training process compared with
executed once after the training is complete. It is also useful to
                                                                 the frequently used early stopping approach.
note that the training times of the time series classifiers in our
                                                       Approach. We assess our overfitting prevention approachapproach are not excessive. For instance, the training times of
                                                                 against the early stopping method (both with and withoutTSF and TSBF are around 300 milliseconds, and KNN-DTW,
                                                        smoothing loss curves) using the metrics introduced in Sec-our best-performing time series classifier, can finish training in
                                                                    tion VI-E. To study the difference in delay across overfitting1 millisecond. However, KNN-DTW requires the longest time
                                                              prevention approaches, we performed the Mann-Whitney Ufor inference which is around 180 milliseconds for a training
                                                                            test [44] at a significance level of α = 0.05 to determinehistory. A fast version of DTW [58] with a time complexity
                                                         whether the distributions of the delay epochs of early stoppingof O(n) is used in experiments, but using KNN with DTW is
                                                       and our approach are significantly different. We also computedstill computationally intensive.
                                                                       Cliff’s delta d [42] effect size to quantify the difference based
                                                     on the provided thresholds [55].
   RQ1   Takeaway:   Our   proposed   approach                                                                 Results. Our proposed approach, utilizing KNN-DTW with
    demonstrates   better   classification   performance                                                           both rolling window and whole observed history, has a similar
    than  correlation-based  approaches  for  detecting                                                           or higher optimal rate than early stopping for  overfitting
     overfitting  in DL  models.  Despite  the  higher                                                                 prevention. Other studied classifiers do not perform as well
    computational cost of the time series classifiers used                                                               as KNN-DTW for overfitting prevention. As Figure 5 and
     in our approach, their training time and inference                                                            Table V show, using KNN-DTW with either a rolling window
    time are still practical.                                                               or the whole observed history outperforms early stopping at

#### 7

the delay between the stopped epoch and the best epoch is   overfitting prevention, and our approach using KNN-DTW still
at least 32% shorter with our approach using KNN-DTW.   outperformed early stopping.
Figure 6 provides an example where both early stopping and
                                                             B. Internal Validityour approach identify the optimal epoch, but our approach
stops 21 epochs earlier than early stopping (which stops with    Our proposed approach relies on the assumption that overfit-
a 40 epochs delay).                                               ting can be detected and prevented through the analysis of DL
  Among our two approaches for overfitting prevention, we  model training histories. However, certain cases of overfitting
recommend using KNN-DTW with a rolling window. Although  may not be captured by examining the training histories alone.
using the whole observed history may achieve a higher optimal   For instance, data leakage caused by data augmentation or
rate than using a rolling window for our approach, we note   preprocessing in the entire dataset before data splitting (into
that we can predict the optimal epoch much earlier with the   training, validation, and test sets) could lead to overfitting,
rolling window approach for a very small trade-off in optimal   but detecting or preventing it solely by inspecting the training
rate (with a similar average accuracy). As shown in Table VI   history would be challenging.
and Figure 5, our approach with KNN-DTW achieves an 83%
                                                          C. External Validityoptimal rate with a median delay of 27 epochs and a 90%
optimal rate with a median delay of 37.5 epochs using the   We evaluated our proposed approach using a real-world
window size as 40 and 60 epochs respectively. However, the   dataset that contains training histories from top AI venues.
median delay of KNN-DTW when using the whole observed   However, it is still possible that the approach may not gener-
history  is 43.5, while the rolling window approach with a   alize well to all types of DL models or datasets. Secondly, our
window size of 80 or more epochs can achieve a higher   real-world evaluation is based on only 40 data points (of which
optimal rate (98% vs. 95% accuracy) with a shorter delay (42.5  29 training histories belong to overfit models), which might not
vs. 43.5 epochs). In summary, we suggest using the rolling   be enough data points to claim generalizability of our proposed
window approach since it stops earlier with a relatively small   approach. Please note that collecting authoritative examples
optimal rate drop using a small window (e.g., 40 epochs) and   of overfit training history is very hard since researchers and
outperforms the whole observed history approach when using   practitioners typically do not report the training history of
a large window size (e.g., 80 epochs).                      models that were overfit. In addition, collecting these data
                                                                 points requires one to replicate the studies that report overfit
   RQ2 Takeaway: Our proposed approach using KNN-       models, which  is a very time and resource intensive task.
  DTW with a  rolling window  or whole observed      Hence we were limited to 40 data points for the real-world
    history outperforms  early  stopping  for  overfitting       dataset in our study. However, we invite future research to
    prevention and can stop training DL models earlier       verify the validity of OverfitGuard using our replication
    with the same or higher accuracy. Among our two      package on their own DL model training histories. In addition,
    approaches, we recommend using KNN-DTW with a       the computational resources required  to use the proposed
     rolling window for early stopping, which achieves a      approach for inference could limit its applicability in specific
    high optimal rate with a shorter delay.                        situations. For instance, the increased computational cost may
                                                        be prohibitive in environments with constrained computational
                                                                  resources, while our approach demonstrates improved perfor-
              IX. THREATS TO VALIDITY               mance in overfitting detection and prevention than existing
                                                             approaches.
A. Construct Validity
                                                             X. CONCLUSION AND FUTURE WORK  The construct validity of our approach may be affected by
the manual labelling process for the simulated training dataset     In this paper, we propose a non-intrusive overfitting de-
used in overfitting detection. The definition of overfitting is an   tection and prevention approach using time  series  classi-
abstract concept and may result in ambiguity or disagreements   fiers  trained on  the  training  history  of DL models. Our
among authors. To mitigate this threat, two authors labelled the   approach (when using the KNN-DTW time series classifier)
training histories independently, achieving a 95% agreement   has (1) better classification performance than correlation-based
rate. Following this, the authors engaged in multiple rounds   approaches for overfitting detection, and (2) greater accuracy
of discussions to resolve any disagreements (as detailed in   than early stopping for overfitting prevention with a shorter
Section VI-B). Despite these efforts, some subjectivity in the   delay. We evaluate our approach on a real-world dataset of
labelling process might impact the validity of our results.       labelled training histories collected from the papers published
  Another potential threat to construct validity is the choice of   at top AI venues in the last 5 years. Our approach can be a
the monitoring metric used in overfitting prevention. Although   useful tool for researchers and developers of DL software. We
validation loss is a widely used metric for monitoring the DL   have shared the trained time series classifiers in the replication
model performance during the training process, different DL   package for reuse, along with  all of the training histories
tasks may require alternative metrics. We conducted additional  and labels. One limitation of our approach is that our best-
experiments using classification error (i.e., zero-one loss) for   performing time series classifier takes longer to perform the

#### 8

inference required to detect and prevent overfit than the studied    [20] W. Fu and T. Menzies, “Easy over hard: A case study on deep learning,”
baselines. We encourage future work to optimize time series          in Proceedings of the 2017 11th joint meeting on foundations of software
                                                                                    engineering, 2017, pp. 49–60.
classifiers to enable overfitting detection and prevention in    [21] Z. Fu, G. Lu, K. M. Ting, and D. Zhang, “Music classification via the
real-time with smaller delays. We also encourage the future         bag-of-features approach,” Pattern Recognition Letters, vol. 32, no. 14,
work to investigate  if adding more real world examples to         pp. 1768–1777, 2011.
                                                                               [22]  S. Gadde, A. Lakshmanarao, and S. Satyanarayana, “Sms spam detection
OverfitGuard’s training data or using online training to         using machine learning and deep learning techniques,” in 2021 7th
constantly update OverfitGuard would improve its detec-         International Conference on Advanced Computing and Communication
tion and prevention performance.                                        Systems (ICACCS), vol. 1, 2021, pp. 358–362.
                                                                               [23]  J.-L. Gauvain and C.-H. Lee, “Maximum a posteriori estimation for
                                                                                         multivariate gaussian mixture observations of markov chains,” IEEE
                 REFERENCES                                     Transactions on Speech and Audio Processing, vol. 2, no. 2, pp. 291–
                                                                                 298, 1994.
 [1] “Our replication package,” https://github.com/asgaardlab/OverfitGuard,    [24] Q. Hanam,  F.  S.  d. M.  Brito, and A. Mesbah, “Discovering bug
     2023.                                                                              patterns in JavaScript,” in Proceedings of the 2016 24th ACM SIGSOFT
                                                                                       International Symposium on Foundations of Software Engineering (FSE [2] B. S. Anami and V. A. Bhandage, “A comparative study of suitability of
      certain features in classification of bharatanatyam mudra images using         2016), 2016, pp. 144–156.
       artificial neural network,” Neural Processing Letters, vol. 50, no. 1, pp.    [25] D. J. Hand, “Principles of data mining,” Drug safety, vol. 30, no. 7, pp.
     741–769, 2019.                                                         621–622, 2007.
 [3] L. Bao, Z. Xing, X. Xia, D. Lo, M. Wu, and X. Yang, “Psc2code:    [26]  J. Hauke and T. Kossowski, “Comparison of values of Pearson’s and
     Denoising code extraction from programming screencasts,” ACM Trans.        Spearman’s correlation coefficients on the same sets of data,” Quaes-
      Softw. Eng. Methodol., vol. 29, no. 3, Jun. 2020.                                 tiones Geographicae, vol. 30, no. 2, pp. 87–93, 2011.
 [4] A. Barbez,  F. Khomh, and Y.-G. Gu´eh´eneuc, “Deep learning  anti-    [27] D. M. Hawkins, “The problem of overfitting,” Journal of chemical
      patterns from code metrics history,” in 2019 IEEE International Con-         information and computer sciences, vol. 44, no. 1, pp. 1–12, 2004.
      ference on Software Maintenance and Evolution (ICSME), 2019, pp.    [28]  T. Hoang, H. Khanh Dam, Y. Kamei, D. Lo, and N. Ubayashi, “Deepjit:
     114–124.                                                An end-to-end deep learning framework for just-in-time defect predic-
 [5] M. G. Baydogan, G. Runger, and E. Tuv, “A bag-of-features framework          tion,” in 2019 IEEE/ACM 16th International Conference on Mining
      to classify time series,” IEEE Transactions on Pattern Analysis and         Software Repositories (MSR), 2019, pp. 34–45.
     Machine Intelligence, vol. 35, no. 11, pp. 2796–2802, 2013.              [29] D. Hoiem, T. Gupta, Z. Li, and M. Shlapentokh-Rothman, “Learning
 [6] G. Biau and E. Scornet, “A random forest guided tour,” Test, vol. 25,         curves for analysis of deep networks,” in Proceedings of the 38th
                                                                                       International Conference on Machine Learning, vol. 139, 18–24 Jul     no. 2, pp. 197–227, 2016.
 [7] C.   Blake,  “UCI   repository   of  machine   learning   databases,”        2021, pp. 4287–4296.
      https://archive.ics.uci.edu/ml/index.php, 1998.                             [30] Q. Huang, A. Qiu, M. Zhong, and Y. Wang, “A code-description
 [8]  J. Bornschein, F. Visin, and S. Osindero, “Small data, big decisions:         representation learning model based on attention,” in 2020 IEEE 27th
    Model selection in the small-data regime,” in Proceedings of the 37th         International Conference on Software Analysis, Evolution and Reengi-
      International Conference on Machine Learning (ICML’20), 2020.              neering (SANER), 2020, pp. 447–455.
 [9]  P. Brazdil, J. N. van Rijn, C. Soares, and J. Vanschoren, Metalearning:    [31]  S.  Ioffe and C. Szegedy, “Batch normalization: Accelerating deep
     Applications to Automated Machine Learning and Data Mining, 2022.         network training by reducing internal covariate shift,” in International
[10]  P.  J. Brockwell and R. A. Davis, Introduction  to time  series and         conference on machine learning, 2015, pp. 448–456.
      forecasting, 2002.                                                        [32]  S. Ji, B. Krishnapuram, and L. Carin, “Variational bayes for continuous
[11]  S. Chatterjee and A. Mishchenko, “Circuit-based intrinsic methods to        hidden markov models and  its application to active learning,” IEEE
      detect overfitting,” in Proceedings of the 37th International Conference         Transactions on Pattern Analysis and Machine Intelligence, vol. 28,
     on Machine Learning, vol. 119, 13–18 Jul 2020, pp. 1459–1468.               no. 4, pp. 522–532, 2006.
[12]  T. Chen, Z. Zhang, S. Liu, S. Chang, and Z. Wang, “Robust overfitting    [33] H. Kim, W. Lee, and J. Lee, “Understanding catastrophic overfitting in
    may be mitigated by properly learned smoothening,” in International          single-step adversarial training,” Proceedings of the AAAI Conference
     Conference on Learning Representations, 2021.                           on Artificial Intelligence, vol. 35, no. 9, pp. 8119–8127, May 2021.
[13] M. Choetkiertikul, H. K. Dam, T. Tran, T. Pham, A. Ghose, and    [34] G. Kronberger, M. Kommenda, and M. Affenzeller, “Overfitting detec-
      T. Menzies, “A deep learning model for estimating story points,” IEEE          tion and adaptive covariant parsimony pressure for symbolic regression,”
     Transactions on Software Engineering, vol. 45, no. 7, pp. 637–656, 2019.          in Proceedings of the 13th Annual Conference Companion on Genetic
                                                                       and Evolutionary Computation (GECCO ’11), 2011, pp. 631–638.[14] A. Ciurumelea, S. Proksch, and H. C. Gall, “Suggesting comment
     completions for python using neural language models,” in 2020 IEEE    [35]  J. Kukaˇcka, V. Golkov, and D. Cremers, “Regularization  for deep
     27th International Conference on Software Analysis, Evolution and          learning: A taxonomy,” arXiv preprint arXiv:1710.10686, 2017.
     Reengineering (SANER), 2020, pp. 456–467.                              [36] V. Kuleshov, N. Fenner, and S. Ermon, “Accurate uncertainties for
[15] E. Damiani and C. A. Ardagna, “Certified machine-learning models,”        deep learning using calibrated regression,” in Proceedings of the 35th
      in SOFSEM 2020: Theory and Practice of Computer Science: 46th         International Conference on Machine Learning, vol. 80, 10–15 Jul 2018,
      International Conference on Current Trends in Theory and Practice of         pp. 2796–2804.
      Informatics, SOFSEM 2020, Limassol, Cyprus, January 20–24, 2020,    [37] Z. Li, T.-H. Chen, and W. Shang, “Where shall we log? studying and
     Proceedings, 2020, p. 3–15.                                                  suggesting logging locations in code blocks,” in 2020 35th IEEE/ACM
                                                                                       International Conference on Automated Software Engineering (ASE),[16] H. Deng, G. Runger, E. Tuv, and M. Vladimir, “A time series forest for
      classification and feature extraction,” Information Sciences, vol. 239, pp.        2020, pp. 361–372.
     142–153, 2013.                                                          [38]  J. Lin, E. Keogh, L. Wei, and S. Lonardi, “Experiencing sax: a novel
[17] H. Ding, G. Trajcevski,  P. Scheuermann, X. Wang, and E. Keogh,        symbolic representation of time series,” Data Mining and knowledge
     “Querying and mining of time series data: Experimental comparison         discovery, vol. 15, no. 2, pp. 107–144, 2007.
      of representations and distance measures,” Proc. VLDB Endow., vol. 1,    [39]  F. Liu, G. Li, Y. Zhao, and Z. Jin, “Multi-task learning based pre-
     no. 2, pp. 1542–1552, Aug. 2008.                                                trained language model for code completion,” in 2020 35th IEEE/ACM
                                                                                       International Conference on Automated Software Engineering (ASE),[18] Z. Ding, J. Chen, and W. Shang, “Towards the use of the readily available
      tests from the release pipeline as performance tests: Are we there yet?”        2020, pp. 473–485.
      in Proceedings of the ACM/IEEE 42nd International Conference on    [40] Z. Liu, X. Xia, M. Yan, and S. Li, “Automating just-in-time comment
     Software Engineering (ICSE ’20), 2020, pp. 1435–1446.                        updating,” in 2020 35th IEEE/ACM International Conference on Auto-
                                                                        mated Software Engineering (ASE), 2020, pp. 585–597.[19]  S. Fakhoury, V. Arnaoudova, C. Noiseux, F. Khomh, and G. Antoniol,
     “Keep it simple: Is deep learning good for linguistic smell detection?”    [41] Z. Liu, Z. Xu,  J.  Jin, Z. Shen, and T.  Darrell, “Dropout reduces
      in 2018 IEEE 25th International Conference on Software Analysis,          underfitting,” arXiv preprint arXiv:2303.01500, 2023.
     Evolution and Reengineering (SANER), 2018, pp. 602–611.               [42]  J. D. Long, D. Feng, and N. Cliff, “Ordinal Analysis of Behavioral

#### 9

Data,” in Handbook of Psychology, Apr. 2003, ch. 25, pp. 635–661.            Applications: With R Examples, ser. Springer Texts in Statistics, 2017.
[43] A. Mahadi, K. Tongay, and N. A. Ernst, “Cross-dataset design discussion    [66] V. Singla, S. Singla, S. Feizi, and D. Jacobs, “Low curvature activa-
     mining,” in 2020 IEEE 27th International Conference on Software          tions reduce overfitting in adversarial training,” in Proceedings of the
      Analysis, Evolution and Reengineering (SANER), 2020, pp. 149–160.        IEEE/CVF International Conference on Computer Vision (ICCV), Oct.
[44] H. B. Mann and D. R. Whitney, “On a test of whether one of two random        2021, pp. 16 423–16 433.
      variables is stochastically larger than the other,” Annals of Mathematical    [67] E. K. Smith, E. T. Barr, C. Le Goues, and Y. Brun, “Is the cure
       Statistics, vol. 18, pp. 50–60, 1947.                                       worse than the disease? overfitting in automated program repair,” in
[45]  F. Mohr and J. N. van Rijn, “Towards model selection using learning        Proceedings of the 2015 10th Joint Meeting on Foundations of Software
     curve cross-validation,” in 8th ICML Workshop on automated machine        Engineering (ESEC/FSE 2015), 2015, pp. 532–543.
     learning (AutoML), 2021.                                                [68] C. Spearman, “The proof and measurement of association between two
[46]  F. Mohr and J. N. van Rijn, “Learning curves for decision making in          things,” The American journal of psychology, vol. 100, no. 3/4, pp. 441–
     supervised machine learning - A survey,” CoRR, vol. abs/2201.12150,         471, 1987.
     2022.                                                                    [69] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhut-
[47] K. Molugaram and G. S. Rao, Statistical Techniques for Transportation         dinov, “Dropout: a simple way to prevent neural networks from over-
     Engineering, Jan. 2017.                                                                     fitting,” The journal of machine learning research, vol. 15, no. 1, pp.
[48] N. Morgan and H. Bourlard, “Generalization and parameter estimation        1929–1958, 2014.
      in feedforward nets: Some experiments,” Advances in neural information    [70] B. Strang, P. v. d. Putten, J. N. v. Rijn, and F. Hutter, “Don’t rule out
     processing systems, vol. 2, 1989.                                            simple models prematurely: A large scale benchmark comparing linear
[49] A. Nilizadeh, G. T. Leavens, X.-B. D. Le, C. S. P˘as˘areanu, and D. R.        and non-linear classifiers in openml,” in Advances in Intelligent Data
     Cok, “Exploring true test overfitting in dynamic automated program         Analysis XVII, 2018, pp. 303–315.
      repair using formal methods,”  in 2021 14th IEEE Conference on    [71]  J. Straub, “Machine learning performance validation and training using
     Software Testing, Verification and Validation (ICST), 2021, pp. 229–240.         a ‘perfect’ expert system,” MethodsX, vol. 8, p. 101477, 2021.
[50]  T. Peng, L. Liu, and W. Zuo, “PU text classification enhanced by term    [72] C. Tantithamthavorn, S. McIntosh, A. E. Hassan, and K. Matsumoto,
     frequency–inverse document frequency-improved weighting,” Concur-       “An empirical comparison of model validation techniques for defect
     rency and Computation: Practice and Experience, vol. 26, no. 3, pp.         prediction models,” IEEE Transactions on Software Engineering, vol. 43,
     728–741, 2014.                                                                no. 1, pp. 1–18, 2017.
[51] L. Prechelt, “Early Stopping — But When?” in Neural Networks: Tricks    [73] Y. Tian, K. Pei, S. Jana, and B. Ray, “Deeptest: Automated testing
      of the Trade: Second Edition, ser. Lecture Notes in Computer Science,         of deep-neural-network-driven autonomous cars,” in Proceedings of the
     2012, pp. 53–67.                                                           40th International Conference on Software Engineering (ICSE ’18),
[52] L. Prechelt  et  al., “Proben1: A  set of neural network benchmark        2018, pp. 303–314.
     problems and benchmarking rules,” 1994.                                 [74]  S. Tokui, S. Tokumoto, A. Yoshii, F. Ishikawa, T. Nakagawa, K. Mu-
[53] X. Ren, Z. Xing, X. Xia, D. Lo, X. Wang, and  J. Grundy, “Neural         nakata, and S. Kikuchi, “Neurecover: Regression-controlled repair of
     network-based detection of self-admitted technical debt: From perfor-        deep neural networks with training history,” in 2022 IEEE Interna-
    mance to explainability,” ACM Trans. Softw. Eng. Methodol., vol. 28,          tional Conference on Software Analysis, Evolution and Reengineering
     no. 3, Jul. 2019.                                                      (SANER), 2022, pp. 1111–1121.
[54] L. Rice, E. Wong, and Z. Kolter, “Overfitting in adversarially robust    [75] M. Tufano, C. Watson, G. Bavota, M. D.  Penta, M. White, and
     deep learning,” in Proceedings of the 37th International Conference on        D. Poshyvanyk, “An empirical study on learning bug-fixing patches
     Machine Learning, vol. 119, 13–18 Jul 2020, pp. 8093–8104.                    in the wild via neural machine translation,” ACM Trans. Softw. Eng.
[55]  J. Romano, J. D. Kromrey, J. Coraggio, J. Skowronek, and L. Devine,         Methodol., vol. 28, no. 4, Sep. 2019.
     “Exploring methods for evaluating group differences on the NSSE and    [76]  J. N. van Rijn, S. M. Abdulrahman, P. Brazdil, and J. Vanschoren, “Fast
      other surveys: Are the t-test and Cohen’sd indices the most appropriate         algorithm selection using learning curves,” in Advances in Intelligent
      choices,” in annual meeting of the Southern Association for Institutional        Data Analysis XIV, 2015, pp. 298–309.
     Research.   Citeseer, 2006, pp. 1–51.                                     [77] O. Varol, E. Ferrara, F. Menczer, and A. Flammini, “Early detection of
[56]  S. Sajeev, A. Maeder, S. Champion, A. Beleigoli, C. Ton, X. Kong, and        promoted campaigns on social media,” EPJ data science, vol. 6, pp.
    M. Shu, “Deep learning to improve heart disease risk prediction,” in        1–19, 2017.
     Machine Learning and Medical Engineering for Cardiovascular Health    [78] Z. Wang, L. Wang, C. Huang, Z. Zhang, and X. Luo, “Soil-moisture-
     and Intravascular Imaging and Computer Assisted Stenting, 2019, pp.         sensor-based automated soil water content cycle classification with a
     96–103.                                                                    hybrid symbolic aggregate approximation algorithm,” IEEE Internet of
[57] G. Salton, A. Wong, and C.-S. Yang, “A vector space model  for        Things Journal, vol. 8, no. 18, pp. 14 003–14 012, 2021.
     automatic indexing,” Communications of the ACM, vol. 18, no. 11, pp.    [79] C. Watson, N. Cooper, D. N. Palacio, K. Moran, and D. Poshyvanyk,
     613–620, 1975.                                                 “A systematic literature review on the use of deep learning in software
[58]  S. Salvador and P. Chan, “Toward accurate dynamic time warping in         engineering research,” ACM Trans. Softw. Eng. Methodol., vol. 31, no. 2,
      linear time and space,” Intell. Data Anal., vol. 11, no. 5, pp. 561–580,        Mar. 2022.
     Oct. 2007.                                                               [80] B. Wei, Y.  Li, G.  Li, X. Xia, and Z.  Jin, “Retrieve and  refine:
[59]  P.  Sch¨afer,  “Scalable time  series  classification,” Data Mining and        Exemplar-based neural comment generation,” in Proceedings of the 35th
     Knowledge Discovery, vol. 30, no. 5, pp. 1273–1298, 2016.               IEEE/ACM International Conference on Automated Software Engineer-
[60]  P. Sch¨afer and M. H¨ogqvist, “SFA: A symbolic fourier approximation         ing (ASE ’20), 2021, pp. 349–360.
     and index for similarity search in high dimensional datasets,” in Pro-    [81] R. Werpachowski, A. Gy¨orgy, and C. Szepesv´ari, “Detecting overfitting
     ceedings of the 15th International Conference on Extending Database         via adversarial examples,” Advances in Neural Information Processing
     Technology (EDBT ’12), 2012, pp. 516–527.                                  Systems, vol. 32, 2019.
[61] O. B. Sghaier and H. Sahraoui, “A multi-step learning approach to    [82] X. Xi, E. Keogh, C. Shelton, L. Wei, and C. A. Ratanamahatana, “Fast
      assist code review,” in 2023 IEEE International Conference on Software         time series classification using numerosity reduction,” in Proceedings
      Analysis, Evolution and Reengineering (SANER), 2023, pp. 450–460.            of the 23rd international conference on Machine learning, 2006, pp.
[62] L. Shi, M. Xing, M. Li, Y. Wang, S. Li, and Q. Wang, “Detection of        1033–1040.
     hidden feature requests from massive chat messages via deep siamese    [83] Q. Xin and S. P. Reiss, “Identifying test-suite-overfitted patches through
     network,” in 2020 IEEE/ACM 42nd International Conference on Soft-           test case generation,” in Proceedings of the 26th ACM SIGSOFT In-
     ware Engineering (ICSE), 2020, pp. 641–653.                                   ternational Symposium on Software Testing and Analysis (ISSTA 2017),
[63] Q. Shi, R. Katuwal, P. Suganthan, and M. Tanveer, “Random vector        2017, pp. 226–236.
      functional link neural network based ensemble deep learning,” Pattern    [84] B. Xu, D. Ye, Z. Xing, X. Xia, G. Chen, and S. Li, “Predicting seman-
     Recognition, vol. 117, p. 107978, 2021.                                               tically linkable knowledge in developer online forums via convolutional
[64] C. Shorten and T. M. Khoshgoftaar, “A survey on image data augmen-         neural network,” in 2016 31st IEEE/ACM International Conference on
      tation for deep learning,” Journal of big data, vol. 6, no. 1, pp. 1–48,        Automated Software Engineering (ASE), 2016, pp. 51–62.
     2019.                                                                    [85] Y. Yang, X. Xia, D. Lo, and J. Grundy, “A survey on deep learning for
[65] R. H. Shumway and D.  S.  Stoffer, Time  Series Analysis and  Its         software engineering,” ACM Comput. Surv., vol. 54, no. 10s, Sep. 2022.

### Additional Content

#### 1

# Keeping Deep Learning Models in Check - A History-Based Approach to Mitigate Overfitting

#### 2

Keeping Deep Learning Models in Check: A
      History-Based Approach to Mitigate Overfitting

#### 3

Hao Li∗, Gopi Krishnan Rajbahadur†, Dayi Lin†, Cor-Paul Bezemer∗, and Zhen Ming (Jack) Jiang‡
                                            ∗University of Alberta. {li.hao, bezemer}@ualberta.ca,
                   †Centre for Software Excellence, Huawei Canada. {gopi.krishnan.rajbahadur1, dayi.lin}@huawei.com,
                                             ‡York University. zmjiang@cse.yorku.ca

#### 4

Training loss
                1         Validation loss
                                Early stopping (p.=20)
                                Early stopping (p.=40)
                    0     25    50    75    100   125   150   175   200
                                Epoch

#### 5

(a) Examples of early stopping with the patience of 20        (b) An demonstration  of our  overfitting prevention
          and 40 epochs (stop at the #71 epoch and #140 epoch).       approach with a rolling window.

#### 6

Fig. 2: Early stopping and our approach for overfitting prevention.

#### 7

Training the time series classifier
                                                                                                       Continue training for
                                                                                                                    the model
                                          Feed data (with
                                                                            Trained time series
                                                               to                                                          time                                                                  series                       Training                                  histories            labels)                                                                                                    classifier                           or                            or                                                           classifier                                                                      for                                                                      training                           with labels

#### 8

Y
                                                         Interpolate the                                          Stop training and
                                                       validation losses to              Overfitting or                  return the epoch that    Overfitting                        Validation losses
                                                  the same length as             non-overfitting                 has the lowest      prevention when
                                                  the data in training                                                  validation loss       training a model

#### 9

Non-overfitting           Overfitting       Avg   CV    Training    Inference
                     Detection approach
                                             Prec   Rec     F-s   Prec   Rec     F-s     F-s     F-s   time (s)    time (ms)

#### 10

Spearman       0.71   0.91   0.80    0.96   0.86   0.91    0.86    0.95      2.461       0.908
                       Corr.
                             Pearson         0.78   0.64   0.70    0.87   0.93   0.90    0.80    0.92      0.222       0.025
                    based
                             Autocorr        0.80   0.73   0.76    0.90   0.93   0.92    0.84    0.80      0.233       0.026

#### 11

KNN-DTW     0.79   1.00   0.88    1.00   0.90   0.95    0.91    0.97      0.001     180.512
                 HMM-GMM    0.30   0.27   0.28    0.73   0.76   0.75    0.52    0.59     99.751      17.750
                 Time
                       TSF            0.77   0.91   0.83    0.96   0.90   0.93    0.88    0.99      0.311      17.209
                       series
                      TSBF          0.79   1.00   0.88    1.00   0.90   0.95    0.91    0.99      0.301      31.683
                      (ours)
                     BOSSVS       0.46   0.91   0.61    0.94   0.59   0.72    0.67    1.00      1.877      19.342
                     SAX-VSM      0.83   0.91   0.87    0.96   0.93   0.95    0.91    0.96      0.912      17.474

#### 12

Prevention              Median delay / ws                    Average accuracy / ws
                          approach
                                         20     40     60     80     100        20     40     60     80    100

#### 13

ES               20.0   40.0   60.0   80.0   100.0       0.40   0.42   0.42   0.43   0.43
                     ES (smoothed)   20.0   43.5   62.0   82.0     97.5       0.39   0.42   0.43   0.43   0.43

#### 14

Patience (epoch)
  5 101520253035404550556065707580859095100105110115
    1.0

#### 15

0.0              20         40         60         80         100
                    Window size (epoch)
                      Early stopping     KNN-DTW       TSF       SAX-VSM
                  ES (smoothed)     HMM-GMM      TSBF     BOSSVS

#### 16

[86]  F. Zampetti, A. Serebrenik, and M. Di Penta, “Automatically learning
      patterns for self-admitted technical debt removal,” in 2020 IEEE 27th
      International Conference on Software Analysis, Evolution and Reengi-
     neering (SANER), 2020, pp. 355–366.
[87]  J. M. Zhang, E. T. Barr, B. Guedj, M. Harman, and J. Shawe-Taylor,
     “Perturbed model  validation: A new framework  to  validate model
      relevance,” CoRR, vol. abs/1905.10201, 2019.
