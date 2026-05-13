<!-- Page 1 -->

Keeping Deep Learning Models in Check: A
History-Based Approach to Mitigate Overfitting
Hao Li∗, Gopi Krishnan Rajbahadur†, Dayi Lin†, Cor-Paul Bezemer∗, and Zhen Ming (Jack) Jiang‡
∗University of Alberta. {li.hao, bezemer}@ualberta.ca,
†Centre for Software Excellence, Huawei Canada. {gopi.krishnan.rajbahadur1, dayi.lin}@huawei.com,
‡York University. zmjiang@cse.yorku.ca
Abstract—In software engineering, deep learning models are
increasinglydeployedforcriticaltaskssuchasbugdetectionand Training loss Training loss
codereview. However, overfittingremainsachallengethataffects Validation loss Validation loss
the quality, reliability, and trustworthiness of software systems
thatutilizedeeplearningmodels. Overfittingcanbe (1) prevented
(e.g., usingdropoutorearlystopping) or (2) detectedinatrained
model (e.g., usingcorrelation-basedapproaches).Bothoverfitting
detectionandpreventionapproachesthatarecurrentlyusedhave
constraints (e.g., requiring modification of the model structure,
(a) Overfitting (b) Non-overfitting
and high computing resources). In this paper, we propose a
simple, yet powerful approach that can both detect and prevent Fig. 1: Examples of overfit and non-overfit training histories.
overfitting based on the training history (i.e., validation losses).
Our approach first trains a time series classifier on training
historiesofoverfitmodels. Thisclassifieristhenusedtodetectif
a trained model is overfit. In addition, our trained classifier can layers [69] or batch normalization [31]. However, many of
be used to prevent overfitting by identifying the optimal point
these approaches are intrusive and require modifying the data
to stop a model’s training. We evaluate our approach on its
or the model structure and expertise to execute correctly and
ability to identify and prevent overfitting in real-world samples.
We compare our approach against correlation-based detection even then, they may not work. For instance, adding dropout
approaches and the most commonly used prevention approach layers, a popularly used overfitting prevention scheme, when
(i.e., early stopping). Our approach achieves an F1 score of 0.91 set with a lower threshold or when added to the earlier
which is at least 5% higher than the current best-performing
layers may cause unintentional overfitting [41]. Furthermore,
non-intrusive overfitting detection approach. Furthermore, our
even the non-intrusive prevention approaches such as early
approach can stop training to avoid overfitting at least 32% of
thetimesearlierthanearlystoppingandhasthesameorabetter stoppingincurtrade-offsbetweenmodelaccuracyandtraining
rate of returning the best model. time [51]. For example, late stopping when using the early
Index Terms—Software engineering for AI, AI for Software stopping approach may improve model accuracy, but it will
Engineering, Overfitting, Training history, Deep learning
also increase training time. Conversely, stopping too early
could result in a sub-optimal model performance.
I. INTRODUCTION
Overfitting detection approaches like k-fold cross-
The use of Deep Learning (DL) models in software engi- validation, training the DL model with noisy data points
neering (SE) researchandsoftwareproductshasbeenskyrock- and observing if the added noise impacts the DL model’s
eting over the past decade. For instance, DL techniques have accuracy[87], checkingifthehypothesisofthetrainedmodel
beenusedforautomatedbugdetection[24], codereview[61], and the data are independent [81] can generally be resource
and software testing [73]. These applications underscore the intensive and time consuming. For instance, Xu et al. [84]
importance and ubiquity of DL in modern SE. report that training a DL model to find two semantically
Overfittingisoneofthefundamentalissuesthatplagues DL linkable questions in Stack Overflow takes about 14 hours.
models [33, 49, 67, 79, 85]. A DL model can be considered If one were to conduct a 5-fold cross-validation to detect if
overfitting if the model fits just the training data instead of the constructed DL model is overfitting, they would have to
learningthetargethypothesis[79].Anoverfitmodelincreases invest 70 hours, which might be prohibitive in practice.
the risk of inaccurate predictions, misleading feature impor- In this paper, we introduce Overfit Guard, an approach
tance, and wasted resources [27]. to both detect and prevent overfitting using training histories.
Currently, the problem of overfitting is addressed in SE Figure1 illustratesexampletraininghistories (i.e., thetraining
studies that use DL models by either (1) preventing it from and validation losses curves) of an overfit and a non-overfit
happening in the first place or (2) detecting it in a trained DL model. The training and validation losses of the overfit
DL model [79, 85]. Overfitting prevention approaches in- model both decrease at the beginning of the training process.
clude early stopping [48], data augmentation [64], regulariza- Followingthat, thevalidationlossincreaseswhilethetraining
tion [35], and modifying the DL model by adding dropout loss decreases, resulting in a large gap between the training
4202
na J
81
]ES.sc[
1 v95301.1042:vi Xra


<!-- Page 2 -->

TABLE I: Studied time series classifiers.
and validation losses. Such a trend indicates poor general-
ization of the trained model to new data. Researchers have
Classifier Description
previously employed training histories for decision-making
KNN-DTW∗ Uses K-Nearest Neighbors [25] with Dynamic Time
in areas such as quantitative data acquisition and model
Warping[17]asthedistancemetrictoclassifytimeseries
selection [8, 9, 29, 45, 46, 70, 76]. Similarly, our approach data
trainsatimeseriesclassifieronasimulateddatasetoftraining HMM-GMM Uses Hidden Markov Model for modelling time series
dataand Gaussian Mixture Modelastheemissionsprob-
histories (i.e., labelled validation loss curves over epochs of
abilitydensity[23,32]
training) of DL models that overfit the training data. Our TSF Uses a random forest [6] for time series data using an
trainedtimeseriesclassifierdetectsoverfittinginatrained DL ensembleoftimeseriestrees[16]
TSBF Time Series Bag-of-Features [5] extracts features based
model by examining the validation loss history (captured as
onthebag-of-featuresapproach[21]tocreatearandom
part of the training history). In contrast to existing overfitting forest
detection approaches, our approach does not incur additional SAX-VSM Symbolic Aggregate appro Ximation [38] converts the
data into symbolic representations and Vector Space
resources or costs, as the training history (also known as
Model[50,57]transformsthemintovectorstocalculate
the learning curve) is a natural byproduct of the training similarityforclassification
process. Furthermore, ourapproach (i.e., thetrainedtimeseries BOSSVS Bag-of-SFA Symbols in Vector Space [59] is similar to
SAX-VSMbutuse SFA[60]totransformthedatainstead
classifier) can be used to prevent overfitting based on the
of SAX
validation losses of recent epochs (e.g., the last 20 epochs).
∗ KNN-DTWhandlesvariable-lengthtimeseriesdata.
Although our approach is trained on a simulated dataset,
we evaluate it on a real-world dataset, collected from papers
publishedintop AIvenueswithinthelast5 years. Wegathered
Typically, adatasetisdividedintotraining, validation, andtest
the training histories from these papers that are explicitly
sets. While the training loss reflects how well the DL model
labelled as overfitting or non-overfitting by the authors as
learns from the training data during the training process, the
the ground truth. The main contributions of this paper are as
validationlossisevaluatedbasedonthevalidationdata, which
follows:
serves as a proxy for evaluating the model’s performance
• Our approach outperforms the state-of-the-art by at least
on unseen data. After training is completed, the trained DL
5%intermsof F-scoreforoverfittingdetection, achieving
model’s performance is evaluated using the test set that has
an F-score of 0.91.
not been exposed to the model.
• Our approach has the capability to prevent overfitting at
least 32% earlier than early stopping while maintaining Researchers and developers can identify potential issues
(and often surpassing) the rate of reaching the optimal such as overfitting or underfitting by analyzing the training
epoch (i.e., the epoch that yields the best model). histories. For example, overfitting is often observed as an
• We provide a replication package [1] containing our increasingdivergencebetweentraininglossandvalidationloss
trained classifiers and labelled training histories which overtime (asillustratedin Figure1 a).Inthispaper, wepropose
can be directly used by other researchers. an approach that leverages training history to automatically
detect and prevent overfitting in DL models.
Paper organization. This paper is organized as follows.
Section II provides background information about our study.
Section III gives an overview of related work. Section IV
introduces existing approaches for detecting and preventing
B. Time series classification
overfitting. Section V describes the design of our study,
while Section VI provides detailed information about our
Time series data consists of data points recorded over time,
experimentalsetup. Sections VIIand VIIIpresenttheresultsof
with each point being associated with a specific timestamp
ourstudy. Section IXdiscussespotentialthreatstothevalidity
and its corresponding value. Time series classification is a
of our study. Finally, Section X concludes the paper.
machine learning task that aims to categorize time series
II. BACKGROUND data into predefined classes. In the context of our study,
This section provides an introduction to the concepts of the we consider the training history of a DL model as time
training history in DL and time series classification. series data, as the task of identifying whether a DL model
is overfitting based on its training history can be framed as
A. Leveraging training history in DL
a time series classification problem. Since there has been no
Training history, also known as the empirical learning prior systematic research on time series classifiers specifically
curve[46], providesvaluableinsightsintoa DLmodel’slearn- designedfortraininghistories, wehaveselectedsixclassifiers
ingprogressandperformancethroughoutthetrainingprocess. (shownin Table I) thathavebeenreportedasbaselinesorstate-
The training history stores a record of metrics during the of-the-artinpriorstudies[2,77,78,82].Theseclassifierswere
training process, which are usually recorded in each training chosenduetotheirdemonstratedeffectivenessinvarioustime
iteration or epoch (as shown in Figure 1). Training loss and series classification tasks and their potential applicability to
validationlossarecommonlyusedmetricsintraininghistories. the overfitting detection problem in DL.


<!-- Page 3 -->

III. RELATEDWORK ferentiates overfitting patches based on semantic differences.
Nilizadeh et al. [49] utilized formal verification to evaluate
A. Mitigating overfitting in SE
the degree of overfitting and identified the challenges posed
Overfitting poses a significant risk to the trustworthiness by program complexity and numeric issues.
of software systems and the research studies that employ Tothebestofourknowledge, boththeoverfittingprevention
DL models. SE researchers typically use either overfitting anddetectionmethodsusedinboth SEstudiesandinpractice
detection or prevention methods to mitigate the problem of fall prey to several key concerns. The overfitting prevention
overfitting. Amongtheoverfittingpreventionmethods, dropout methods, typically require significant expertise to execute
isthemostcommonlyadoptedapproach[79,85].Researchers correctly and are intrusive (for instance, they may require
have used dropout in various domains such as code gen- one to modify the data or the model structure). Overfitting
eration [39, 40], logging locations recommendation [37], detection approaches typically require retraining of the DL
and comment completion [14, 80]. Regularization is another model multiple times, which may be very costly in prac-
prominent overfitting prevention strategy that has been used tice [20] and simple methods like early stop may stop the
in SE [4, 30, 84, 86], which requires adding another layer to trainingof DLmodelsuboptimally. Ourworkaddressesthese
the model structure as well. For example, Zampetti et al. [86] gaps by introducing a history-based approach that serves the
employed L2-norm regularization in training CNN and RNN dual purpose of detecting and preventing overfitting in a non-
modelstomanageself-admittedtechnicaldebtinsourcecode. intrusive manner without any need for DL model retraining.
Early stopping is another frequently used technique to
prevent overfitting during the training process of DL mod- B. Leveragingtraininghistorytoimprove DLsoftwarequality
els [13, 28, 62]. For example, Shi et al. [62] utilized early Whilethemachinelearningcommunityhasleveragedtrain-
stopping when training a deep Siamese network to identify ing histories (i.e., learning curves) for different tasks, the
hiddenfeaturerequestspostedinchatmessagesbydevelopers. SE community has seldom used training history to enhance
Other techniques like data augmentation [3, 19, 43] and data software quality. Mohr and van Rijn [46] conduct a survey
balancing [53, 75] are also employed to address overfitting. on approaches based on learning curves for decision-making
For instance, Bao et al. [3] developed a CNN-based image in DL domains, such as data acquisition, early stopping,
classification model to filter out non-code and noisy-code and model selection. They also propose an approach called
frames from programming screencasts. To enhance training Learning Curve Cross-Validation (LCCV) [45] that iteratively
data diversity, they employed data augmentation techniques increasesthenumberoftrainingexamplesusedfortrainingto
such as rotation, scaling, translation, and shearing. select the best model from the candidates. Training histories
In terms of overfitting detection approaches, Zhang et al. can also be used to evaluate the trained classifiers: van Rijn
[87] propose the perturbation validation (PV) assessment to et al. [76] propose an approach that recommends classifiers
determine whether a DL model fits the training data properly for a given dataset based on training histories using loss time
(i.e., ensure that it is neither overfitting nor underfit). Alterna- curves. Moreover, Hoiem et al. [29] investigate the use of
tively, somedetectionapproachescheckthehypothesisthatthe training histories to evaluate design choices for DL models,
trained DL model and the data are independent. For example, such as pretraining, architecture, and data augmentation.
Werpachowski et al. [81] check the hypothesis by comparing In this paper, we propose Overfit Guard, an approach
thetesterrorwiththeestimatedtesterrorbasedonadversarial that utilizes time-series classifiers to detect and prevent over-
examples of the test set. fitting by analyzing the training history of DL models. Our
Another popular approach towards detecting overfiting is approach aims to enhance the software quality of DL systems
to use model validation approaches. Tantithamthavorn et al. by improving the quality of the DL models themselves.
[72] evaluated 12 model validation techniques specifically for Overfit Guard is one of the first approaches that leverages
defect prediction models and concluded that out-of-sample traininghistorytoenhancethesoftwarequalityof DLsystems.
bootstrap emerges as the least biased and most stable tech- A related work by Tokui et al. [74] introduces an approach
nique that helps detect overfit. Damiani and Ardagna [15] called Neu Recoverwhichalsoleveragestraininghistorytoim-
introduced a framework that validates DL models against provethequalityof DLmodels, particularlyforsafety-critical
desirednon-functionalpropertiesandstatisticallymonitorsthe applications. Neu Recover identifies the model parameters that
modeloutput. Straub[71]extendedthis byutilizingrandomly need to be modified (i.e., repaired) by analyzing the training
generated expert networks for model validation that focuses history to address specific failure types. Our approach shares
on performance characteristics. similaritieswith Neu Recover, butfocusesonthedetectionand
Otherthantheseapproaches, Smithetal.[67]discussedthe prevention of overfitting, which is important for maintaining
roleofhumanfactorsinoverfittingandprovidedacomparative the software quality of DL systems.
analysis between automated patches and human-written fixes.
They reported that overfitting is not solely a machine-induced
IV. EXISTINGAPPROACH
problem and suggested focusing on the contributing factors Inthissection, weintroducetheexistingapproachesthatwe
like test suite coverage and requirements-based testing. Xin useasbaselineapproachestocompareourproposedapproach
and Reiss [83] proposed a classification technique that dif- for overfitting detection and prevention in DL models.


<!-- Page 4 -->

5
4
3
2
1
0 25 50 75 100 125 150 175 200
Epoch
sso L
Training loss
Validation loss
Early stopping (p.=20)
Early stopping (p.=40)
(a)Examplesofearlystoppingwiththepatienceof20 (b) An demonstration of our overfitting prevention
and40 epochs (stopatthe#71 epochand#140 epoch). approach with a rolling window.
Fig. 2: Early stopping and our approach for overfitting prevention.
A. Overfitting detection is also used by Py Torch Ignite.2 As shown in Figure 2 a, early
stopping with a patience parameter of 20 epochs stops at the
#70 epoch and returns the #50 with the lowest validation loss
Correlation-basedapproaches. Oneapproachfordetecting
since no improvement occurs between epochs #50 and #70.
overfittingin DLmodelsistocomputethecorrelationbetween
Furthermore, early stopping with larger patience values (e.g.,
thetrainingandvalidationloss. Kronbergeretal.[34]propose
40 epochs) stops later (at the #140 epoch) but with a lower
computing the non-parametric Spearman’s rank correlation
loss (at the #100 epoch).
coefficient [68] between training and validation fitness in
Earlystoppingbasedonsmoothedvalidationlosscurves.
symbolicregressionmodelstodetectoverfitting. Similartoour
An alternate version of early stopping inspects the moving
approach, correlation-based approaches also detect overfitting
average of the smoothed validation loss curves [47, 65] to
based on the training history. Therefore, we select them as
decide when to stop the training process. After stopping the
a baseline for comparison. In this study, we calculate the
training process, this approach returns the best epoch which
correlation metrics between the training and validation loss
has the lowest validation loss (not the smoothed value).
to detect overfitting in DL models. The underlying principle
is straightforward: when no overfitting occurs, the training
V. OURAPPROACH
and validation losses should be correlated, whereas weak
correlation implies overfitting. Figure3 showsanoverviewofourproposedapproach. Our
We determine the presence of overfitting by comparing approach uses a time series classifier to detect and prevent
the computed correlation-based metric between training and overfitting. Table Iliststhestudiedtimeseriesclassifiers. First,
validation losses with a predetermined threshold value (deter- wecollectasimulateddataset (moredetailsonhowwecollect
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
epochs. We feed this loss to our trained time series classifier
B. Overfitting prevention
to detect whether there is overfitting. However, we cannot
Early stopping. One widely used approach for preventing directly feed these validation losses to our classifier, since the
overfittingisearlystopping, whichstopstrainingwhenthereis lengthofthevalidationlossesmightnotbeofthesamelength
noimprovementinafixednumberofepochs (calledpatience) as that of the data used to train these time series classifiers.
andreturnstheepochwhichhasthelowestvalidationloss. We All the studied time series classifiers, with the exception of
choose the widely used Tensor Flow implementation,1 which
2 https://pytorch.org/ignite/generated/ignite.handlers.early stopping.
1 https://tensorflow.org/api docs/python/tf/keras/callbacks/Early Stopping Early Stopping


<!-- Page 5 -->

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
historyinafixedwindowsize (e.g., thelatest20 epochs), and Train our series Early stopping overfitting
approaches classifiers prevention
(2) as the whole observed history (from the first to the latest approaches
epochs). Our time series classifier detects if in the fed history
overfitting occurs. Similar to overfitting detection, we linearly
interpolate the data before feeding it into our model. If there Fig. 4: Overview of the experimental setup.
is no overfit occurring, we continue the training and repeat
the above procedure until the DL model has finished training.
For the rolling window, we move the window by a fixed step Python3.8 and Tensor Flow2.9.0.Thehardwareconfiguration
size (as shown in Figure 2 b) and make another prediction. If for the experiments is detailed below:
ourmodeldetectsthepresenceofoverfittinginthefedhistory, • NVIDIA RTX 3090 GPU with 24 GB memory
we return the lowest validation loss in the observed epochs as • CUDA version 10.1.243
the best epoch. • cu DNN version 7.6.5
• Intel (R) Core (TM) i9-11900 K CPU with a clock speed
VI. EXPERIMENTALSETUP of 3.50 GHz
• 64 GB of RAM
In this section, we introduce the datasets for training and
evaluating the studied overfitting detection and prevention B. Simulated training dataset
approaches, the experiments of our study, and the evalua-
We create a simulated dataset containing training histories
tion metrics for the studied approaches. Figure 4 shows an
with labels to determine a threshold for correlation-based
overview of the experimental setup.
approaches (see Section IV) andtotrainourproposedmethod
as described in Section V. We create this simulated dataset
A. Environment setting
by training neural networks of varying model complexities to
We conducted the experiments on an Ubuntu 20.04 oper- produce overfitting and non-overfitting samples. The process
ating system with a Linux kernel version of 5.15.0, utilizing is as follows:


<!-- Page 6 -->

TABLE II: Information about datasets used to simulate over-
of our manual labelling process, we follow the approach
fitting.
outlined by Ding et al. [18]. The first and second authors
of this paper independently labelled the 432 data points as
Dataset Type #In #Out #Examples
either“overfit”,“non-overfit”or“uncertain”anddiscussedthe
building regression 14 3 4,208
results. Inthefirstdiscussionround, theauthorsreacheda95%
cancer classification 9 2 699
card classification 51 2 690 agreement (410 data points), with both authors labelling 10
diabetes classification 8 2 768 data points as ”uncertain” and subsequently eliminating them.
flare regression 24 3 1,066
In the second round, the authors discussed the remaining 22
gene classification 120 3 3,175
glass classification 9 6 214 disagreements. Followingthediscussion, weeliminated3 data
heart classification 35 2 920 points (labelled “uncertain” by both authors) and agreed on
hearta regression 35 1 920
the labels for the remaining 19 data points. The final dataset
horse classification 58 3 364
soybean classification 82 19 683 consistsof44 overfitand375 non-overfittraininghistories. We
thyroid classification 21 3 7,200 share the labelled training histories in our replication package
for other researchers to reuse.
Step 1 – Download the datasets for overfitting simulation. C. Real-world test dataset
We download 12 datasets representing real-world problems
from the Proben1 [52] benchmark set for training neural net-
To evaluate our approach using real-world data, we con-
works. These datasets were used by Prechelt [51] to simulate
ducted a survey of papers from conferences and journals to
traininghistoriesforstudyingearlystopping. Wechoosethese
gather examples of overfit and non-overfit DL models.
datasets over using SE datasets for two reasons: First, since
Step1–Identifyrelatedconferencesandjournals. Weiden-
we use the methodology used by Prechelt [51] to simulate
tify related conferences and journals based on the Computing
overfitting, we chose to stick with the datasets that they used. Research and Education Association of Australasia (CORE3)
Second, irrespective of the domain of the dataset, we assert and China Computer Federation (CCF4) ranking systems.
that how the phenomenon of overfit is represented by training
Under the CCF A rank, we have 7 conferences and 4 journals
andvalidationhistorieswillremainthesame. Table IIprovides
in the “Artificial Intelligence” field. Under the CORE A*
information about these datasets, which include 3 datasets for
rank, we have 16 conferences in the “machine learning” and
regression tasks and 9 datasets for classification tasks. All
“artificial intelligence” fields and 12 journals in the “artificial
of these datasets (except the “building” one) were originally
intelligence and image processing” field. After merging the
collectedfromthe UCImachinelearningrepository[7], which
results and accounting for overlaps between the two ranking
has been widely used in DL research [22, 36, 56, 63]. Each
systems, we obtained a final list of 17 conferences and 12
datasetispre-partitionedintotraining, validation, andtestsets
journals.
(50%,25%, and25%ofthedata, respectively) andpartitioned
Wecollectedourreal-worlddataintheartificialintelligence
three times to generate three distinct permutations, resulting
and machine learning domain as opposed to SE domain,
in a total of 36 datasets from Proben1.
because SEstudiestypicallydonotreportthetraininghistories
Step 2 – Simulate overfitting by training neural networks.
oftheoverfit DLmodelsandwerequiredcommunityaccepted
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
as”overfitting”) inthetitlethatwerepublishedintheselected
put/output coefficients in the respective datasets (see Table II)
conferences and journals in the last 5 years. Five of these
and rectified linear units (Re LUs) are used for all hidden
papers provided samples of overfitting: P2 - Chatterjee and
layers. The structures of the NNs are as follows: (1) 6 one-
Mishchenko [11]; P4 - Chen et al. [12] P13 - Kim et al.
hidden-layer NNswith2,4,8,16,24, or32 hiddennodes, and
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
traininghistory. Paper P17 sharedthetraininghistory, making
All problems employ stochastic gradient descent (SGD) as
its replication straightforward. We replicated the other papers
the optimizer. To increase the likelihood of overfitting, we
thatprovideoverfittingsamplestocollectthetraininghistories
train these 12 neural network architectures on each of the 36
datasets for 1,000 epochs, producing 432 training histories.
3 https://www.core.edu.au
Step 3 – Label training histories. To ensure the robustness 4 https://ccf.atom.im/


<!-- Page 7 -->

TABLE III: Information about collected samples from sur-
approaches, we computed the precision, recall, and F-score
veyed papers.
for overfitting and non-overfitting samples in the real-world
test dataset. In addition, we calculated the average F-score to
Paper Labels for the training history in the #Overfit #Non-overfit
manuscript directly compare the classification performance of the studied
P2 “[...]thevalidationaccuracyofnn-randomis 2 0 approaches. To evaluate the time cost associated with training
9.73%(i.e., closetochance) confirmingthat andusingthestudiedapproaches, wereportthetraining time
itishorriblyoverfit”
P4 “We first observe that the robust overfitting 3 3 (in seconds) for each approach on the simulated dataset and
prevailsinall Baselinecases” theinferencetime (inmilliseconds) forthereal-worlddataset.
“Ourmethodseffectivelymitigatestherobust
overfitting” Evaluation metrics for overfitting prevention. Ideally, an
P13 “Figure4[...]thatis, catastrophicoverfitting N/A∗ N/A∗ overfittingpreventionapproachreturnstheoptimalepoch (i.e.,
occurs.”
“Figure 6 shows that the proposed method the epoch that yields the best predictive performance for
alsosuccessfullypreventscatastrophicover- the DL model on the validation set) and stops the training
fitting[...]”
P17 “Figure24[...]Weseeclearrobustoverfitting 20 4 process as early as possible. We define the optimal rate
forthesmallertwooptionsinλ, andfindno of an overfitting prevention approach as the percentage of
overfittingbuthighlyregularizedmodelsfor
thelargertwooptions[...]” cases where the optimal epoch is successfully identified. To
P23 “These results therefore validate our claim 4 4 assess the speed of the approach, we introduce the delay
that low curvature activations reduce robust
overfitting” metric, which represents the epoch difference between the
∗ Cannotreproducethesameresultsasthepaper. stopped epoch and the best epoch. For example, a delay of 10
epochs occurs if the prevention approach stops at the 123 th
epoch while the 113 th epoch is the best one. In addition, we
of these samples. We executed the code from the papers with
report the DL model’s accuracy on the validation set when
available replication packages (P4, P13, and P23) to generate
the training process is stopped by the overfitting prevention
their training histories. However, we could not replicate the
approach.
results for paper P13. For paper P2, which did not provide a
replicationpackage, wefollowedthemethodologytoreplicate VII. RQ1:HOWWELLDOESOVERFITGUARD DETECT
the results and training history. In total, we collected 29 OVERFITTINGINTRAINEDDLMODELS?
training histories of overfit DL models and 11 of non-overfit
Motivation. Overfitting detection is an important task in
DL models (refer to Table III for details).
DL models since it helps in identifying whether a DL model
D. Experiments has learned to perform well on training data but fails to
generalize on unseen data. Accurate overfitting detection can
Overfitting detection. Wetrainedthetimeseriesclassifiers
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
Approach. Weusetheevaluationmetricsintroducedin Sec-
classifierforfurtheruse. Forcorrelation-basedapproaches, we
tion VI-E to compare our approach with baseline approaches
also performed a grid search based on the simulated dataset
based on the real-world test dataset. Furthermore, we record
to select the optimal thresholds (ranging from -1 to 1) that
the F-scoreobtainedfromthe3-foldcross-validation (CV) for
yielded the best F-score.
ourapproachbasedonthesimulatedtrainingdatasettofurther
Overfitting prevention. We reused the trained time series
analyze the performance of our approach. Since we use the
classifiers from the previous step to perform inference during
entire simulated training dataset to determine the thresholds
thetrainingprocesstopreventoverfitting. Sincethevalidation
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
by TSFwhichoutperformsthebaselineapproachesaswell. In
Evaluation metrics for overfitting detection. To evaluate contrast, HMM-GMM performs poorly on both the simulated
the classification performance of the overfitting detection trainingandreal-worldtestdatasets. Onepossibleexplanation


<!-- Page 8 -->

TABLE IV: Results of the overfitting detection approaches on the simulated dataset (CV F-S: F-score of cross-validation)
and real-world dataset (Prec: precision; Rec: recall; F-s: F-score; Avg F-s: average F-score), and the time cost of training the
studied approaches on the simulated dataset and performing inference on the real-world dataset (per sample).
Non-overfitting Overfitting Avg CV Training Inference
Detectionapproach
Prec Rec F-s Prec Rec F-s F-s F-s time (s) time (ms)
Spearman 0.71 0.91 0.80 0.96 0.86 0.91 0.86 0.95 2.461 0.908
Corr.
Pearson 0.78 0.64 0.70 0.87 0.93 0.90 0.80 0.92 0.222 0.025
based
Autocorr 0.80 0.73 0.76 0.90 0.93 0.92 0.84 0.80 0.233 0.026
KNN-DTW 0.79 1.00 0.88 1.00 0.90 0.95 0.91 0.97 0.001 180.512
HMM-GMM 0.30 0.27 0.28 0.73 0.76 0.75 0.52 0.59 99.751 17.750
Time
TSF 0.77 0.91 0.83 0.96 0.90 0.93 0.88 0.99 0.311 17.209
series
TSBF 0.79 1.00 0.88 1.00 0.90 0.95 0.91 0.99 0.301 31.683
(ours)
BOSSVS 0.46 0.91 0.61 0.94 0.59 0.72 0.67 1.00 1.877 19.342
SAX-VSM 0.83 0.91 0.87 0.96 0.93 0.95 0.91 0.96 0.912 17.474
TABLE V: The optimal rate, median delay, and average
is that the extracted state models (via HMM) of the training
accuracy of our overfitting prevention approaches using the
histories do not follow a Gaussian probability distribution.
whole observed history.
Our approach with BOSSVS correctly identifies all the data
in the simulated dataset but performs poorly on the real-
world dataset. One reason could be that the extracted bag- Optimal Median Average
Classifier
rate delay accuracy
of-SFA symbols (BOSS) from the simulated dataset does not
generalize to the real-world dataset. In addition, we note KNN-DTW 0.95 43.5 0.42
HMM-GMM 0.18 0.0 0.36
that the investigated correlation-based approaches perform
TSF 0.90 35.0 0.42
reasonably well, with F-scores greater than 0.8. However, TSBF 0.83 31.0 0.41
our approach outperforms the correlation-based overfitting BOSSVS 0.65 21.0 0.41
SAX-VSM 0.33 10.0 0.38
detection approach by at least 5% on the studied real-world
dataset.
Thestudiedtimeseriesclassifiersaremorecomputationally
intensive than correlation-based approaches for inference, yet VIII. RQ2:HOWWELLDOESOVERFITGUARD PREVENT
they are still useful in practice. As shown in Table IV, our OVERFITTINGDURINGTHETRAININGPROCESS?
approach requires more time for performing inference than
Motivation. Anothercriticalpartofdevelopingtrustworthy
the correlation-based approaches. For instance, TSF has the
and stable DL models is preventing overfitting. An effective
fastest inference time among the classifiers but is around 20
overfitting prevention approach allows DL models to gener-
times slower than the Spearman correlation-based approach
alize better on unseen data while minimizing both training
and around 700 times slower than the other two correlation-
resources and computational costs. This research question
based approaches. However, the speed of our approach is
evaluates the performance of our proposed approach for pre-
not prohibitive in practice since overfitting detection is only
venting overfitting during the training process compared with
executedonceafterthetrainingiscomplete. Itisalsousefulto
the frequently used early stopping approach.
notethatthe trainingtimesofthetime seriesclassifiersinour
Approach. We assess our overfitting prevention approach
approach are not excessive. For instance, the training times of
against the early stopping method (both with and without
TSF and TSBF are around 300 milliseconds, and KNN-DTW,
smoothing loss curves) using the metrics introduced in Sec-
ourbest-performingtimeseriesclassifier, canfinishtrainingin
tion VI-E. To study the difference in delay across overfitting
1 millisecond. However, KNN-DTWrequiresthelongesttime
prevention approaches, we performed the Mann-Whitney U
for inference which is around 180 milliseconds for a training
test [44] at a significance level of α = 0.05 to determine
history. A fast version of DTW [58] with a time complexity
whetherthedistributionsofthedelayepochsofearlystopping
of O(n) is used in experiments, but using KNN with DTW is
andourapproacharesignificantlydifferent. Wealsocomputed
still computationally intensive.
(cid:3) (cid:0) Cliff’s delta d [42] effect size to quantify the difference based
on the provided thresholds [55].
RQ1 Takeaway: Our proposed approach
Results. Our proposed approach, utilizing KNN-DTW with
demonstrates better classification performance
bothrollingwindowandwholeobservedhistory, hasasimilar
than correlation-based approaches for detecting
or higher optimal rate than early stopping for overfitting
overfitting in DL models. Despite the higher
prevention. Other studied classifiers do not perform as well
computational cost of the time series classifiers used
as KNN-DTW for overfitting prevention. As Figure 5 and
in our approach, their training time and inference
Table Vshow, using KNN-DTWwitheitherarollingwindow
time are still practical.
or the whole observed history outperforms early stopping at
(cid:2) (cid:1)


<!-- Page 9 -->

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
lamitp O
Patience (epoch)
5101520253035404550556065707580859095100 105 110 115
Early stopping KNN-DTW TSF SAX-VSM
ES (smoothed) HMM-GMM TSBF BOSSVS
Fig. 5: The optimal rate of our overfitting prevention ap-
proach (using a rolling window) and early stopping with
different patience values.
Fig. 6: Our approach using KNN-DTW (set the window size
as 40 epochs) stops earlier than the early stopping (set the
identifyingtheoptimalepoch. Inparticular, ourapproachwith patience parameter as 40 epochs) but both achieve the same
KNN-DTW based on the rolling window has a higher or the optimal epoch.
same optimal rate as both early stopping approaches when
using up to 80 epochs as the patience parameter and window
size. Forexample, ourapproachwith KNN-DTWobtains78% on both smoothed and non-smoothed validation loss) with the
optimalratewhensettingthewindowsizeto20 epochs, while same or higher accuracy. As shown in Table VI, with the
bothearlystoppingapproachesachievelessthan50%optimal samenumberofepochsforthepatienceparameterandwindow
rate when the patience parameter is set to the same epochs. size, our approach can save training time (i.e., reducing delay
However, when the patience parameter is greater than 80 between the stopped epoch and the best epoch) compared to
epochs, both early stopping approaches can identify almost early stopping, except for a window size of 20 epochs. For
all of the optimal epochs. The reason is that 90% of the instance, whensettingboththepatienceparameterandwindow
training histories in the real-world dataset have around 200 size to 40 epochs, KNN-DTW and early stopping have the
epochs, hence, a large patience value makes it easy for early sameaverageaccuracy, but KNN-DTWhasamediandelayof
stopping to choose the optimal epoch. In addition, Table V 27 epochswhileearlystoppinghasafixeddelayof40 epochs.
shows that our approach with KNN-DTW based on the whole The significance test results indicate that the delay difference
observed history also obtains a higher optimal rate compared between KNN-DTW and early stopping is significant (except
to both early stopping approaches. For example, the KNN- when using a window size of 20), with a medium to large
DTW classifier achieves a 95% optimal rate with a median effect size. Furthermore, early stopping with smoothed loss
delay of 43.5 epochs, whereas early stopping approaches curveshasamediandelayof43.5, whichisslowerthanusing
achieve around 85% using the same number (i.e., between the original loss curves with the same patience parameter (40
40 to 45 epochs patience in Figure 5). epochs). Early stopping using the smoothed loss could hurt
Our approach using KNN-DTW and a rolling window can the performance of early stoppingand cannot compete with
stop training a DL model earlier than early stopping (based our approach. In comparison to the delay in early stopping,


<!-- Page 10 -->

the delay between the stopped epoch and the best epoch is overfittingprevention, andourapproachusing KNN-DTWstill
at least 32% shorter with our approach using KNN-DTW. outperformed early stopping.
Figure 6 provides an example where both early stopping and
B. Internal Validity
our approach identify the optimal epoch, but our approach
stops 21 epochs earlier than early stopping (which stops with Ourproposedapproachreliesontheassumptionthatoverfit-
a 40 epochs delay). tingcanbedetectedandpreventedthroughtheanalysisof DL
Among our two approaches for overfitting prevention, we model training histories. However, certain cases of overfitting
recommendusing KNN-DTWwitharollingwindow. Although maynotbecapturedbyexaminingthetraininghistoriesalone.
usingthewholeobservedhistorymayachieveahigheroptimal For instance, data leakage caused by data augmentation or
rate than using a rolling window for our approach, we note preprocessing in the entire dataset before data splitting (into
that we can predict the optimal epoch much earlier with the training, validation, and test sets) could lead to overfitting,
rolling window approach for a very small trade-off in optimal but detecting or preventing it solely by inspecting the training
rate (with a similar average accuracy). As shown in Table VI history would be challenging.
and Figure5, ourapproachwith KNN-DTWachievesan83%
C. External Validity
optimal rate with a median delay of 27 epochs and a 90%
optimal rate with a median delay of 37.5 epochs using the We evaluated our proposed approach using a real-world
window size as 40 and 60 epochs respectively. However, the dataset that contains training histories from top AI venues.
median delay of KNN-DTW when using the whole observed However, it is still possible that the approach may not gener-
history is 43.5, while the rolling window approach with a alizewelltoalltypesof DLmodelsordatasets. Secondly, our
window size of 80 or more epochs can achieve a higher real-worldevaluationisbasedononly40 datapoints (ofwhich
optimalrate (98%vs.95%accuracy) withashorterdelay (42.5 29 traininghistoriesbelongtooverfitmodels), whichmightnot
vs. 43.5 epochs). In summary, we suggest using the rolling beenoughdatapointstoclaimgeneralizabilityofourproposed
window approach since it stops earlier with a relatively small approach. Please note that collecting authoritative examples
optimal rate drop using a small window (e.g., 40 epochs) and of overfit training history is very hard since researchers and
outperforms the whole observed history approach when using practitioners typically do not report the training history of
a large window size (e.g., 80 epochs). models that were overfit. In addition, collecting these data
(cid:3) (cid:0) points requires one to replicate the studies that report overfit
RQ2 Takeaway:Ourproposedapproachusing KNN- models, which is a very time and resource intensive task.
DTW with a rolling window or whole observed Hence we were limited to 40 data points for the real-world
history outperforms early stopping for overfitting dataset in our study. However, we invite future research to
prevention and can stop training DL models earlier verify the validity of Overfit Guard using our replication
with the same or higher accuracy. Among our two packageontheirown DLmodeltraininghistories. Inaddition,
approaches, we recommend using KNN-DTW with a the computational resources required to use the proposed
rolling window for early stopping, which achieves a approach for inference could limit its applicability in specific
high optimal rate with a shorter delay. situations. For instance, the increased computational cost may
(cid:2) (cid:1) beprohibitiveinenvironmentswithconstrainedcomputational
resources, while our approach demonstrates improved perfor-
IX. THREATSTOVALIDITY mance in overfitting detection and prevention than existing
approaches.
A. Construct Validity
The construct validity of our approach may be affected by
X. CONCLUSIONANDFUTUREWORK
themanuallabellingprocessforthesimulatedtrainingdataset In this paper, we propose a non-intrusive overfitting de-
usedinoverfittingdetection. Thedefinitionofoverfittingisan tection and prevention approach using time series classi-
abstractconceptandmayresultinambiguityordisagreements fiers trained on the training history of DL models. Our
amongauthors. Tomitigatethisthreat, twoauthorslabelledthe approach (when using the KNN-DTW time series classifier)
training histories independently, achieving a 95% agreement has (1) betterclassificationperformancethancorrelation-based
rate. Following this, the authors engaged in multiple rounds approaches for overfitting detection, and (2) greater accuracy
of discussions to resolve any disagreements (as detailed in than early stopping for overfitting prevention with a shorter
Section VI-B). Despite these efforts, some subjectivity in the delay. We evaluate our approach on a real-world dataset of
labelling process might impact the validity of our results. labelled training histories collected from the papers published
Anotherpotentialthreattoconstructvalidityisthechoiceof at top AI venues in the last 5 years. Our approach can be a
themonitoringmetricusedinoverfittingprevention. Although usefultoolforresearchersanddevelopersof DLsoftware. We
validation loss is a widely used metric for monitoring the DL havesharedthetrainedtimeseriesclassifiersinthereplication
model performance during the training process, different DL package for reuse, along with all of the training histories
tasksmayrequirealternativemetrics. Weconductedadditional and labels. One limitation of our approach is that our best-
experiments using classification error (i.e., zero-one loss) for performing time series classifier takes longer to perform the


<!-- Page 11 -->

inferencerequiredtodetectandpreventoverfitthanthestudied [20] W.Fuand T.Menzies,“Easyoverhard:Acasestudyondeeplearning,”
baselines. We encourage future work to optimize time series in Proceedingsofthe201711 thjointmeetingonfoundationsofsoftware
engineering,2017, pp.49–60.
classifiers to enable overfitting detection and prevention in
[21] Z.Fu, G.Lu, K.M.Ting, and D.Zhang,“Musicclassificationviathe
real-time with smaller delays. We also encourage the future bag-of-featuresapproach,”Pattern Recognition Letters, vol.32, no.14,
work to investigate if adding more real world examples to pp.1768–1777,2011.
[22] S.Gadde, A.Lakshmanarao, and S.Satyanarayana,“Smsspamdetection
Overfit Guard’s training data or using online training to
using machine learning and deep learning techniques,” in 2021 7 th
constantly update Overfit Guard would improve its detec- International Conferenceon Advanced Computingand Communication
tion and prevention performance. Systems (ICACCS), vol.1,2021, pp.358–362.
[23] J.-L. Gauvain and C.-H. Lee, “Maximum a posteriori estimation for
multivariate gaussian mixture observations of markov chains,” IEEE
REFERENCES Transactions on Speech and Audio Processing, vol. 2, no. 2, pp. 291–
298,1994.
[1] “Our replication package,” https://github.com/asgaardlab/Overfit Guard, [24] Q. Hanam, F. S. d. M. Brito, and A. Mesbah, “Discovering bug
2023.
patternsin Java Script,”in Proceedingsofthe201624 th ACMSIGSOFT
International Symposiumon Foundationsof Software Engineering (FSE
[2] B.S.Anamiand V.A.Bhandage,“Acomparativestudyofsuitabilityof
certain features in classification of bharatanatyam mudra images using
2016),2016, pp.144–156.
artificialneuralnetwork,”Neural Processing Letters, vol.50, no.1, pp. [25] D.J.Hand,“Principlesofdatamining,”Drugsafety, vol.30, no.7, pp.
741–769,2019. 621–622,2007.
[3] L. Bao, Z. Xing, X. Xia, D. Lo, M. Wu, and X. Yang, “Psc2 code: [26] J. Hauke and T. Kossowski, “Comparison of values of Pearson’s and
Denoisingcodeextractionfromprogrammingscreencasts,”ACMTrans. Spearman’s correlation coefficients on the same sets of data,” Quaes-
Softw. Eng. Methodol., vol.29, no.3, Jun.2020. tiones Geographicae, vol.30, no.2, pp.87–93,2011.
[4] A. Barbez, F. Khomh, and Y.-G. Gue´he´neuc, “Deep learning anti- [27] D. M. Hawkins, “The problem of overfitting,” Journal of chemical
patterns from code metrics history,” in 2019 IEEE International Con- informationandcomputersciences, vol.44, no.1, pp.1–12,2004.
ference on Software Maintenance and Evolution (ICSME), 2019, pp. [28] T.Hoang, H.Khanh Dam, Y.Kamei, D.Lo, and N.Ubayashi,“Deepjit:
114–124. An end-to-end deep learning framework for just-in-time defect predic-
[5] M.G.Baydogan, G.Runger, and E.Tuv,“Abag-of-featuresframework tion,” in 2019 IEEE/ACM 16 th International Conference on Mining
to classify time series,” IEEE Transactions on Pattern Analysis and Software Repositories (MSR),2019, pp.34–45.
Machine Intelligence, vol.35, no.11, pp.2796–2802,2013. [29] D. Hoiem, T. Gupta, Z. Li, and M. Shlapentokh-Rothman, “Learning
[6] G. Biau and E. Scornet, “A random forest guided tour,” Test, vol. 25, curves for analysis of deep networks,” in Proceedings of the 38 th
no.2, pp.197–227,2016. International Conference on Machine Learning, vol. 139, 18–24 Jul
[7] C. Blake, “UCI repository of machine learning databases,” 2021, pp.4287–4296.
https://archive.ics.uci.edu/ml/index.php,1998. [30] Q. Huang, A. Qiu, M. Zhong, and Y. Wang, “A code-description
[8] J. Bornschein, F. Visin, and S. Osindero, “Small data, big decisions: representation learning model based on attention,” in 2020 IEEE 27 th
Model selection in the small-data regime,” in Proceedings of the 37 th International Conference on Software Analysis, Evolution and Reengi-
International Conferenceon Machine Learning (ICML’20),2020. neering (SANER),2020, pp.447–455.
[9] P.Brazdil, J.N.van Rijn, C.Soares, and J.Vanschoren, Metalearning: [31] S. Ioffe and C. Szegedy, “Batch normalization: Accelerating deep
Applicationsto Automated Machine Learningand Data Mining,2022.network training by reducing internal covariate shift,” in International
[10] P. J. Brockwell and R. A. Davis, Introduction to time series and conferenceonmachinelearning,2015, pp.448–456.
forecasting,2002. [32] S.Ji, B.Krishnapuram, and L.Carin,“Variationalbayesforcontinuous
[11] S. Chatterjee and A. Mishchenko, “Circuit-based intrinsic methods to hidden markov models and its application to active learning,” IEEE
detectoverfitting,”in Proceedingsofthe37 th International Conference Transactions on Pattern Analysis and Machine Intelligence, vol. 28,
on Machine Learning, vol.119,13–18 Jul2020, pp.1459–1468. no.4, pp.522–532,2006.
[12] T.Chen, Z.Zhang, S.Liu, S.Chang, and Z.Wang,“Robustoverfitting [33] H.Kim, W.Lee, and J.Lee,“Understandingcatastrophicoverfittingin
may be mitigated by properly learned smoothening,” in International single-step adversarial training,” Proceedings of the AAAI Conference
Conferenceon Learning Representations,2021. on Artificial Intelligence, vol.35, no.9, pp.8119–8127, May2021.
[13] M. Choetkiertikul, H. K. Dam, T. Tran, T. Pham, A. Ghose, and [34] G.Kronberger, M.Kommenda, and M.Affenzeller,“Overfittingdetec-
T.Menzies,“Adeeplearningmodelforestimatingstorypoints,”IEEE tionandadaptivecovariantparsimonypressureforsymbolicregression,”
Transactionson Software Engineering, vol.45, no.7, pp.637–656,2019. in Proceedings of the 13 th Annual Conference Companion on Genetic
[14] A. Ciurumelea, S. Proksch, and H. C. Gall, “Suggesting comment
and Evolutionary Computation (GECCO’11),2011, pp.631–638.
completions for python using neural language models,” in 2020 IEEE [35] J. Kukacˇka, V. Golkov, and D. Cremers, “Regularization for deep
27 th International Conference on Software Analysis, Evolution and learning:Ataxonomy,”ar Xivpreprintar Xiv:1710.10686,2017.
Reengineering (SANER),2020, pp.456–467. [36] V. Kuleshov, N. Fenner, and S. Ermon, “Accurate uncertainties for
[15] E. Damiani and C. A. Ardagna, “Certified machine-learning models,” deep learning using calibrated regression,” in Proceedings of the 35 th
in SOFSEM 2020: Theory and Practice of Computer Science: 46 th International Conferenceon Machine Learning, vol.80,10–15 Jul2018,
International Conferenceon Current Trendsin Theoryand Practiceof pp.2796–2804.
Informatics, SOFSEM 2020, Limassol, Cyprus, January 20–24, 2020, [37] Z. Li, T.-H. Chen, and W. Shang, “Where shall we log? studying and
Proceedings,2020, p.3–15. suggestinglogginglocationsincodeblocks,”in202035 th IEEE/ACM
[16] H.Deng, G.Runger, E.Tuv, and M.Vladimir,“Atimeseriesforestfor
International Conference on Automated Software Engineering (ASE),
classificationandfeatureextraction,”Information Sciences, vol.239, pp. 2020, pp.361–372.
142–153,2013. [38] J. Lin, E. Keogh, L. Wei, and S. Lonardi, “Experiencing sax: a novel
[17] H. Ding, G. Trajcevski, P. Scheuermann, X. Wang, and E. Keogh, symbolic representation of time series,” Data Mining and knowledge
“Querying and mining of time series data: Experimental comparison
discovery, vol.15, no.2, pp.107–144,2007.
ofrepresentationsanddistancemeasures,”Proc. VLDBEndow., vol.1, [39] F. Liu, G. Li, Y. Zhao, and Z. Jin, “Multi-task learning based pre-
no.2, pp.1542–1552, Aug.2008.
trainedlanguagemodelforcodecompletion,”in202035 th IEEE/ACM
[18] Z.Ding, J.Chen, and W.Shang,“Towardstheuseofthereadilyavailable
International Conference on Automated Software Engineering (ASE),
testsfromthereleasepipelineasperformancetests:Arewethereyet?” 2020, pp.473–485.
in Proceedings of the ACM/IEEE 42 nd International Conference on [40] Z. Liu, X. Xia, M. Yan, and S. Li, “Automating just-in-time comment
Software Engineering (ICSE’20),2020, pp.1435–1446. updating,”in202035 th IEEE/ACMInternational Conferenceon Auto-
[19] S.Fakhoury, V.Arnaoudova, C.Noiseux, F.Khomh, and G.Antoniol,
mated Software Engineering (ASE),2020, pp.585–597.
“Keepitsimple:Isdeeplearninggoodforlinguisticsmelldetection?” [41] Z. Liu, Z. Xu, J. Jin, Z. Shen, and T. Darrell, “Dropout reduces
in 2018 IEEE 25 th International Conference on Software Analysis, underfitting,”ar Xivpreprintar Xiv:2303.01500,2023.
Evolutionand Reengineering (SANER),2018, pp.602–611. [42] J. D. Long, D. Feng, and N. Cliff, “Ordinal Analysis of Behavioral


<!-- Page 12 -->

Data,”in Handbookof Psychology, Apr.2003, ch.25, pp.635–661. Applications:With RExamples, ser. Springer Textsin Statistics,2017.
[43] A.Mahadi, K.Tongay, and N.A.Ernst,“Cross-datasetdesigndiscussion [66] V. Singla, S. Singla, S. Feizi, and D. Jacobs, “Low curvature activa-
mining,” in 2020 IEEE 27 th International Conference on Software tions reduce overfitting in adversarial training,” in Proceedings of the
Analysis, Evolutionand Reengineering (SANER),2020, pp.149–160. IEEE/CVF International Conference on Computer Vision (ICCV), Oct.
[44] H.B.Mannand D.R.Whitney,“Onatestofwhetheroneoftworandom 2021, pp.16423–16433.
variablesisstochasticallylargerthantheother,”Annalsof Mathematical [67] E. K. Smith, E. T. Barr, C. Le Goues, and Y. Brun, “Is the cure
Statistics, vol.18, pp.50–60,1947. worse than the disease? overfitting in automated program repair,” in
[45] F. Mohr and J. N. van Rijn, “Towards model selection using learning Proceedingsofthe201510 th Joint Meetingon Foundationsof Software
curve cross-validation,”in 8 th ICMLWorkshop on automatedmachine Engineering (ESEC/FSE2015),2015, pp.532–543.
learning (Auto ML),2021. [68] C.Spearman,“Theproofandmeasurementofassociationbetweentwo
[46] F. Mohr and J. N. van Rijn, “Learning curves for decision making in things,”The Americanjournalofpsychology, vol.100, no.3/4, pp.441–
supervised machine learning - A survey,” Co RR, vol. abs/2201.12150, 471,1987.
2022. [69] N.Srivastava, G.Hinton, A.Krizhevsky, I.Sutskever, and R.Salakhut-
[47] K.Molugaramand G.S.Rao, Statistical Techniquesfor Transportation dinov, “Dropout: a simple way to prevent neural networks from over-
Engineering, Jan.2017. fitting,” The journal of machine learning research, vol. 15, no. 1, pp.
[48] N.Morganand H.Bourlard,“Generalizationandparameterestimation 1929–1958,2014.
infeedforwardnets:Someexperiments,”Advancesinneuralinformation [70] B. Strang, P. v. d. Putten, J. N. v. Rijn, and F. Hutter, “Don’t rule out
processingsystems, vol.2,1989. simplemodelsprematurely:Alargescalebenchmarkcomparinglinear
[49] A. Nilizadeh, G. T. Leavens, X.-B. D. Le, C. S. Pa˘sa˘reanu, and D. R. and non-linear classifiers in openml,” in Advances in Intelligent Data
Cok, “Exploring true test overfitting in dynamic automated program Analysis XVII,2018, pp.303–315.
repair using formal methods,” in 2021 14 th IEEE Conference on [71] J.Straub,“Machinelearningperformancevalidationandtrainingusing
Software Testing, Verificationand Validation (ICST),2021, pp.229–240. a‘perfect’expertsystem,”Methods X, vol.8, p.101477,2021.
[50] T.Peng, L.Liu, and W.Zuo,“PUtextclassificationenhancedbyterm [72] C. Tantithamthavorn, S. Mc Intosh, A. E. Hassan, and K. Matsumoto,
frequency–inverse document frequency-improved weighting,” Concur- “An empirical comparison of model validation techniques for defect
rency and Computation: Practice and Experience, vol. 26, no. 3, pp. predictionmodels,”IEEETransactionson Software Engineering, vol.43,
728–741,2014. no.1, pp.1–18,2017.
[51] L.Prechelt,“Early Stopping—But When?”in Neural Networks:Tricks [73] Y. Tian, K. Pei, S. Jana, and B. Ray, “Deeptest: Automated testing
ofthe Trade:Second Edition, ser. Lecture Notesin Computer Science, ofdeep-neural-network-drivenautonomouscars,”in Proceedingsofthe
2012, pp.53–67. 40 th International Conference on Software Engineering (ICSE ’18),
[52] L. Prechelt et al., “Proben1: A set of neural network benchmark 2018, pp.303–314.
problemsandbenchmarkingrules,”1994. [74] S. Tokui, S. Tokumoto, A. Yoshii, F. Ishikawa, T. Nakagawa, K. Mu-
[53] X. Ren, Z. Xing, X. Xia, D. Lo, X. Wang, and J. Grundy, “Neural nakata, and S. Kikuchi, “Neurecover: Regression-controlled repair of
network-based detection of self-admitted technical debt: From perfor- deep neural networks with training history,” in 2022 IEEE Interna-
mance to explainability,” ACM Trans. Softw. Eng. Methodol., vol. 28, tional Conference on Software Analysis, Evolution and Reengineering
no.3, Jul.2019. (SANER),2022, pp.1111–1121.
[54] L. Rice, E. Wong, and Z. Kolter, “Overfitting in adversarially robust [75] M. Tufano, C. Watson, G. Bavota, M. D. Penta, M. White, and
deeplearning,”in Proceedingsofthe37 th International Conferenceon D. Poshyvanyk, “An empirical study on learning bug-fixing patches
Machine Learning, vol.119,13–18 Jul2020, pp.8093–8104. in the wild via neural machine translation,” ACM Trans. Softw. Eng.
[55] J.Romano, J.D.Kromrey, J.Coraggio, J.Skowronek, and L.Devine, Methodol., vol.28, no.4, Sep.2019.
“Exploringmethodsforevaluatinggroupdifferencesonthe NSSEand [76] J.N.van Rijn, S.M.Abdulrahman, P.Brazdil, and J.Vanschoren,“Fast
othersurveys:Arethet-testand Cohen’sdindicesthemostappropriate algorithm selection using learning curves,” in Advances in Intelligent
choices,”inannualmeetingofthe Southern Associationfor Institutional Data Analysis XIV,2015, pp.298–309.
Research. Citeseer,2006, pp.1–51. [77] O.Varol, E.Ferrara, F.Menczer, and A.Flammini,“Earlydetectionof
[56] S.Sajeev, A.Maeder, S.Champion, A.Beleigoli, C.Ton, X.Kong, and promoted campaigns on social media,” EPJ data science, vol. 6, pp.
M. Shu, “Deep learning to improve heart disease risk prediction,” in 1–19,2017.
Machine Learningand Medical Engineeringfor Cardiovascular Health [78] Z. Wang, L. Wang, C. Huang, Z. Zhang, and X. Luo, “Soil-moisture-
and Intravascular Imaging and Computer Assisted Stenting, 2019, pp. sensor-based automated soil water content cycle classification with a
96–103. hybridsymbolicaggregateapproximationalgorithm,”IEEEInternetof
[57] G. Salton, A. Wong, and C.-S. Yang, “A vector space model for Things Journal, vol.8, no.18, pp.14003–14012,2021.
automaticindexing,”Communicationsofthe ACM, vol.18, no.11, pp. [79] C. Watson, N. Cooper, D. N. Palacio, K. Moran, and D. Poshyvanyk,
613–620,1975. “Asystematicliteraturereviewontheuseofdeeplearninginsoftware
[58] S. Salvador and P. Chan, “Toward accurate dynamic time warping in engineeringresearch,”ACMTrans. Softw. Eng. Methodol., vol.31, no.2,
lineartimeandspace,”Intell. Data Anal., vol.11, no.5, pp.561–580, Mar.2022.
Oct.2007. [80] B. Wei, Y. Li, G. Li, X. Xia, and Z. Jin, “Retrieve and refine:
[59] P. Scha¨fer, “Scalable time series classification,” Data Mining and Exemplar-basedneuralcommentgeneration,”in Proceedingsofthe35 th
Knowledge Discovery, vol.30, no.5, pp.1273–1298,2016. IEEE/ACMInternational Conferenceon Automated Software Engineer-
[60] P. Scha¨fer and M. Ho¨gqvist, “SFA: A symbolic fourier approximation ing (ASE’20),2021, pp.349–360.
and index for similarity search in high dimensional datasets,” in Pro- [81] R.Werpachowski, A.Gyo¨rgy, and C.Szepesva´ri,“Detectingoverfitting
ceedings of the 15 th International Conference on Extending Database via adversarial examples,” Advances in Neural Information Processing
Technology (EDBT’12),2012, pp.516–527. Systems, vol.32,2019.
[61] O. B. Sghaier and H. Sahraoui, “A multi-step learning approach to [82] X.Xi, E.Keogh, C.Shelton, L.Wei, and C.A.Ratanamahatana,“Fast
assistcodereview,”in2023 IEEEInternational Conferenceon Software time series classification using numerosity reduction,” in Proceedings
Analysis, Evolutionand Reengineering (SANER),2023, pp.450–460. of the 23 rd international conference on Machine learning, 2006, pp.
[62] L. Shi, M. Xing, M. Li, Y. Wang, S. Li, and Q. Wang, “Detection of 1033–1040.
hidden feature requests from massive chat messages via deep siamese [83] Q.Xinand S.P.Reiss,“Identifyingtest-suite-overfittedpatchesthrough
network,” in 2020 IEEE/ACM 42 nd International Conference on Soft- test case generation,” in Proceedings of the 26 th ACM SIGSOFT In-
ware Engineering (ICSE),2020, pp.641–653. ternational Symposiumon Software Testingand Analysis (ISSTA2017),
[63] Q. Shi, R. Katuwal, P. Suganthan, and M. Tanveer, “Random vector 2017, pp.226–236.
functional link neural network based ensemble deep learning,” Pattern [84] B.Xu, D.Ye, Z.Xing, X.Xia, G.Chen, and S.Li,“Predictingseman-
Recognition, vol.117, p.107978,2021. ticallylinkableknowledgeindeveloperonlineforumsviaconvolutional
[64] C.Shortenand T.M.Khoshgoftaar,“Asurveyonimagedataaugmen- neuralnetwork,”in201631 st IEEE/ACMInternational Conferenceon
tation for deep learning,” Journal of big data, vol. 6, no. 1, pp. 1–48, Automated Software Engineering (ASE),2016, pp.51–62.
2019. [85] Y.Yang, X.Xia, D.Lo, and J.Grundy,“Asurveyondeeplearningfor
[65] R. H. Shumway and D. S. Stoffer, Time Series Analysis and Its softwareengineering,”ACMComput. Surv., vol.54, no.10 s, Sep.2022.


<!-- Page 13 -->

[86] F. Zampetti, A. Serebrenik, and M. Di Penta, “Automatically learning
patterns for self-admitted technical debt removal,” in 2020 IEEE 27 th
International Conference on Software Analysis, Evolution and Reengi-
neering (SANER),2020, pp.355–366.
[87] J. M. Zhang, E. T. Barr, B. Guedj, M. Harman, and J. Shawe-Taylor,
“Perturbed model validation: A new framework to validate model
relevance,”Co RR, vol.abs/1905.10201,2019.
