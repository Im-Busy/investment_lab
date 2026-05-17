# Chapter4.MontesinosLpez2022_Chapter_OverfittingModelTuningAndEvalu

> *Source PDF: Chapter4.MontesinosLpez2022_Chapter_OverfittingModelTuningAndEvalu.pdf*
> *Extraction: Combined — previous structured extraction (base) + markitdown raw text*

---

# Chapter4.MontesinosLpez2022_Chapter_OverfittingModelTuningAndEvalu

> *Source PDF: Chapter4.MontesinosLpez2022_Chapter_OverfittingModelTuningAndEvalu.pdf*
> *Extraction: Combined — markitdown raw text (base) + previous extraction supplements*

---

Chapter 4
Overﬁtting, Model Tuning, and Evaluation
of Prediction Performance

4.1 The Problem of Overﬁtting and Underﬁtting

The overﬁtting phenomenon occurs when the statistical machine learning model
learns the training data set so well that it performs poorly on unseen data sets. In
other words, this means that the predicted values match the true observed values in
the training data set too well, causing what is known as overﬁtting. Overﬁtting
happens when a statistical machine learning model learns the systematic and noise
(random ﬂuctuations) parts in the training data to the extent that it negatively impacts
the performance of the statistical machine learning model on new data. This means
that the statistical machine learning model adapts very well to the noise as well as to
the signal that is present in the training data. The problem is that these concepts do
not apply to independent (new) data and negatively affect the model’s ability to
generalize. Overﬁtting is more probable when learning a loss function from a
complex statistical machine learning model (with more ﬂexibility). For this reason,
many nonparametric statistical machine learning models also include constraints in
the loss function to improve the learning process of the statistical machine learning
models. For example, artiﬁcial neural networks (ANN), mentioned later, are a
nonparametric statistical machine learning model that is very ﬂexible and is subject
to overﬁtting training data. This problem can be addressed by dropping out (setting
to zero) the weights of a certain percentage of hidden units in order to avoid
overﬁtting.

On the other hand, an underﬁtted phenomenon occurs when few predictors are
included in the statistical machine learning model, i.e., it is a very simple model that
poorly represents the complete picture of the predominant data pattern. This problem
also arises when the training data set is too small or not representative of the
population data. An underﬁtted model does a poor job of ﬁtting the training data
and for this reason it is not expected to satisfactorily predict new data points. This
implies that the predictions using unseen data are weak, since individuals are
perceived as strangers unfamiliar with the training data set.

© The Author(s) 2022
O. A. Montesinos López et al., Multivariate Statistical Machine Learning Methods
for Genomic Prediction, https://doi.org/10.1007/978-3-030-89010-0_4

109

110

4 Overﬁtting, Model Tuning, and Evaluation of Prediction Performance

Fig. 4.1 Schematic illustration of three models for classiﬁcation: (a) M1 with underﬁtting, (b) M2
with appropriate ﬁtting, and (c) M3 with overﬁtting

Consider a scattered series of points (y1, x1), . . ., (yn, xn), on a plane, to which we
want to adjust a statistical machine learning method. This means that we are looking
for the best f(xi) that explains the existing relationship between the response variable
(yi) and the predictors (x1, . . ., xn). We assume that we have three options for f(xi):
M1, the simple model plotted in Fig. 4.1a; M2, an intermediate model shown in
Fig. 4.1b; and M3, a complex model shown in Fig. 4.1c.

Under the classiﬁcation framework, the ﬁrst panel in Fig. 4.1 (left side, panel a)
shows an unsatisfactory ﬁt (underﬁtted) since the line does not cover most of the
points (has high bias) in the plot. As such, we expect that the prediction of unseen
data of this model, M1, will perform badly. In contrast, panel c of Fig. 4.1 shows an
almost perfect ﬁt, since the predicted line covers all the data points. While at ﬁrst
glance, you may think that model M3 will perform well when predicting unseen
data, this is actually untrue since the predicted line covers all points that are noise and
those that are signal (overﬁt); for this reason, this type of model also performs poorly
in the prediction of future data due to its complexity and high variance. Therefore,
the best option for predicting unseen data is model M2 (Fig. 4.1, panel b) since it
represents the predominant (smooth) pattern enough to represent the apparent data
pattern while maintaining a balance between bias and variance. For this reason, a
well-ﬁtted model is one that faithfully represents the sought-after predominant
pattern in the data, while ignoring the idiosyncrasies in the training data. As such,
a well-ﬁtted model in the testing set should be in the neighborhood of the model’s
accuracy based on the training data set, that is, the model’s accuracy in the testing set
should be approximately equal to that of the model’s accuracy in the training set. In
contrast, an overﬁtted model in the testing data set will be far from the neighborhood
of the model’s accuracy based on the training data set, and usually its prediction
performance is very high (good) in the training set and consequently low (bad) in the
testing set (Ratner 2017).

The paradox of overﬁtting is deﬁned as complex models that contain more
information about the training data, but less information about the testing data
(future data we want to predict). In statistical machine learning, overﬁtting is a
major issue and leads to some serious problems in research: (a) some relationships

4.2 The Trade-Off Between Prediction Accuracy and Model Interpretability

111

that seem statistically signiﬁcant are only noise, (b) the complexity of the statistical
machine learning model is very large for the amount of data provided, and (c) the
model in general is not replicable and predicts poorly.

Since the main goal of developing and implementing statistical machine learning
methods is to predict unseen data not used for training the statistical machine
learning algorithm, researchers are mainly interested in minimizing the testing
error (generalization error applicable to future samples) instead of minimizing the
training error that is applicable to the observed data used for training the statistical
machine learning algorithm.

According to Shalev-Shwartz and Ben-David (2014), if the learning fails, these

are some approaches to follow:

1. Increase the sample size of the training set.
2. Modify the hypothesis by (a) enlarging it, (b) reducing it, (c) completely changing
it, and (d) changing the parameters being used. We understand a hypothesis as the
models and their parameters under evaluation. This point is very important to
reach a reasonable model for your data.

3. Change the feature representation of the data.
4. Change the statistical machine learning algorithm used.

4.2 The Trade-Off Between Prediction Accuracy and Model

Interpretability

Accuracy is the ability of a statistical machine learning model to make correct
predictions and those models with more complexity (called ﬂexible models) are
better in terms of accuracy, while the simple, less complex models (called inﬂexible
models) are less accurate but more interpretable. Interpretability indicates to what
degree the model allows for human understanding of natural phenomena. For these
reasons, when the goal of the study is prediction, ﬂexible models should be used;
however, when the goal of the study is inference, inﬂexible models are more
appropriate because they more easily interpret the relationship between the response
variables and the predictor variables. As the complexity of the statistical machine
learning model increases, the bias is reduced and the variance increases. For this
reason, when more parameters are included in the statistical machine learning model,
the complexity of the model increases and the variance becomes the main concern
while the bias steadily falls. For example, James et al. (2013) state that the linear
regression model is a relatively inﬂexible method because it only generates linear
functions, while the support vector machine method is one of the most ﬂexible
statistical machine learning methods.

Before providing an analytical interpretation of the trade-off between the bias and
variance, we must understand the meaning of both concepts. Bias is the difference
between the expected prediction of our statistical machine learning model and the
true observed values. For example, assume that you poll a speciﬁc city where half of

112

4 Overﬁtting, Model Tuning, and Evaluation of Prediction Performance

Fig. 4.2 Graphical
representations of different
levels of bias and variance

the population is high-income and the other half is low-income. If you collected a
sample of high-income people, you would conclude that the entire city has high
income. This means that your conclusion is heavily biased since you only sampled
people with high income. On the other hand, error variance refers to the amount that
the estimate of the objective function will change using a different training data set.
In other words, the error variance accounts for the deviation of predictions from one
repetition to another using the same training set. Ideally, when a statistical machine
learning model with low error variance predicts a value, the predicted value should
remain almost the same, even when changing from one training data set to another;
however, if the model has high variance, then the predicted values of the statistical
machine learning method are affected by the values of the data set. We provide a
graphical visualization of bias and variance with a bull’s eye diagram (see Fig. 4.2).
We assume that the center of the target is a statistical machine learning model that
perfectly predicts the correct answers. As we move away from the bull’s eye, our
predictions get worse. Let us assume that we can repeat our entire statistical machine
learning model building process to get a number of separate hits on the target. Each
hit represents an individual realization of our statistical machine learning model,
given the chance variability in the training data we gathered. Sometimes we accu-
rately predict the observations of interest since we captured a representative sample
in our training data, while other times we obtain unreliable predictions since our
training data may be full of outliers or nonrepresentative values. The four combina-
tions of cases resulting from both high and low bias and variance are shown in
Fig. 4.2.

4.2 The Trade-Off Between Prediction Accuracy and Model Interpretability

113

Burger (2018) concludes that the best scenario is the one with low bias and low
variance, since samples are acceptable representatives of the population (Fig. 4.2),
while in the case of high bias and low variance, the samples are fairly consistent, but
not particularly representative of the population (Fig. 4.2). However, when there is
low bias and high variance, the samples vary widely in their consistency, and only
some may be representative of the population (Fig. 4.2). Finally, with high bias and
high variance, the samples are somewhat consistent, but unlikely to be representa-
tive of the population (Fig. 4.2).

If we denote the variable we are trying to predict as y and our covariates as xi, we
may assume that there is a relationship between y and xi, as that given in Eq. (1.1)
from Chap. 1, where the error term is normally distributed with a mean of zero and
variance σ2. The expected prediction error for a new observation with value x, using
a quadratic loss function, is given by Hastie et al. (2008, page 223):

(cid:2)
E y (cid:1)bf xif g

(cid:3)

2

(cid:3)
2

n

(cid:2)

(cid:3)

o

2

(cid:3)

ð

(cid:2)

(cid:2)
Þ2 þ E bf xif g
i
(cid:3)

(cid:2)

¼ E y (cid:1) f xif g
h
¼ Var yð Þ þ Bias bf xif g
i
(cid:3)
h
¼ Var Eð Þ þ Bias bf xif g

(cid:2)

2

2

þ E bf xið Þ (cid:1) E bf xif g
(cid:1) f xif g
(cid:3)
(cid:2)
þ Var bf xið Þ
(cid:3)
(cid:2)
þ Var bf xið Þ

,

where Bias is the result of misspecifying the statistical model f. Estimation variance
(the third term) is the result of using a sample to estimate f. The ﬁrst term is the error
(irreducible error) that results even if the model is correctly speciﬁed and accurately
estimated. This irreducible error is the noise term in the true relationship that cannot
fundamentally be reduced by any model. Given the true model and inﬁnite data to
train (calibrate) it, we should be able to reduce both the bias and variance terms to
0. However, in a world with imperfect models and ﬁnite data, there is a trade-off
between minimizing the bias and minimizing the variance. The above decomposition
reveals a source of the difference between explanatory and predictive modeling: In
explanatory modeling, the focus is on minimizing bias to obtain the most accurate
representation of the underlying theory. In contrast, predictive modeling seeks to
minimize the combination of bias and estimation variance, occasionally sacriﬁcing
theoretical accuracy for improved empirical precision (Shmueli 2010).

These four aspects impact every step of the modeling process, such that the

resulting f is markedly different in the explanatory and predictive contexts.

Let us assume that f is a reasonable operationalization of the true function (F)
relating constructs X and Y. Choosing a function f (cid:3) that is intentionally biased in
place of f is very undesirable from a theoretical–explanatory standpoint. However,
the election of f (cid:3) is desirable to f under the prediction approach. We show this using
the statistical model y ¼ β1x1 + β2x2 + β3x3 + E, which is assumed to be correctly
speciﬁed with respect to F. Using data, we obtain the estimated model bf , which has
the following properties:

114

4 Overﬁtting, Model Tuning, and Evaluation of Prediction Performance

(cid:3)
(cid:2)
Var bf xið Þ

(cid:2)
¼ Var bβ

Bias ¼ 0
2x2 þ bβ

1x1 þ bβ

3x3

(cid:3)

(cid:4)
¼ σ2xT XXT

(cid:5) 2 1

x,

where x is the vector x ¼ [x1, x2, x3]T and X is the design matrix based on all
predictors. Combining the squared bias with the variance gives, as expected, the
prediction error (EPE).

(cid:2)
E y (cid:1) bf xif g

(cid:3)
2

¼ σ2 þ 0 þ σ2xT XXT

(cid:4)

(cid:5) 2 1

h

(cid:4)

x = σ2 1 þ xT XXT

(cid:5)

(cid:1)1

i

:

x

In comparison, consider the estimated underspeciﬁed form bf

(cid:3)

xið Þ ¼ x1bγ. The bias

and variance here are

(cid:2)

Bias ¼ E bf xif g

(cid:3)

(cid:1)1xT
1

β

ð

1x1 þ β

2x2 þ β

3x3

Þ

(cid:4)

(cid:5)

(cid:1) f xif g ¼ x1 x1xT
1
2x2 þ β

(cid:1) β
ð

3x3

Þ

1x1 þ β
(cid:3)

(cid:2)
Var bf xið Þ

(cid:4)
¼ σ2x1 x1xT
1

(cid:5)

(cid:1)1xT
1

Combining the squared bias with the variance EPE is equal to

(cid:2)
E y (cid:1) bf xif g

(cid:3)
2

h

(cid:4)
¼ x1 x1xT
1

h

(cid:5)

(cid:1)1

1 x2β
2 þ x3β
xT
ð
i
(cid:5)
(cid:4)
:

(cid:1)1

xT
1

þ σ2 1 þ x1 x1xT
1

Þ (cid:1) x2β
ð

2 þ x3β

3

Þ

3

i
2

Although the bias of the underspeciﬁed model f (cid:3)(xi) is larger than that of f{xi}, its
variance can be smaller, and in some cases, so small that the overall EPE will be
lower for the underspeciﬁed model. Wu et al. (2007) showed the general result for an
underspeciﬁed linear regression model with multiple predictors. In particular, they
showed that the underspeciﬁed model that leaves out q predictors has a lower EPE
when the following inequality holds:

qσ2 > βT

2 XT

2 I (cid:1) H1
ð

ÞX2β

2

This means that the underspeciﬁed model produces more accurate predictions, in
terms of lower EPE, in the following situations: (a) when the data are very noisy (large
σ2); (b) when the true absolute values of the excluded parameters (in our example, β2
and β3) are small; (c) when the predictors are highly correlated; and (d) when the
sample size is small or the range of left-out variables is small (Shmueli 2010).

Hagerty and Srinivasan (1991) nicely summarize this situation: “We note that the
practice in applied research of concluding that a model with a higher predictive
validity is “truer,” is not a valid inference. This paper shows that a parsimonious but
less true model can have a higher predictive validity than a truer but less parsimo-
nious model.”

4.3 Cross-validation

115

4.3 Cross-validation

Cross-validation (CV) is a strategy for model selection or algorithm selection. CV
consists of splitting the data (at least once) for estimating the error of each algorithm.
Part of the data (the training set) is used for training each algorithm, and the
remaining part (the testing set) is used for estimating the error of the algorithm.
Then, CV selects the algorithm with the smallest estimated error. For this reason, CV
is used to evaluate the prediction performance of a statistical machine learning model
in out-of-sample data. This technique ensures that the data used for training the
statistical machine learning model are independent of the testing data set in which
the prediction performance is evaluated. It consists of repeating and recording the
arithmetic average obtained from the evaluation measures on different partitions.
Under k-fold CV, which is explained in greater detail later, this process is repeated a
total of k times, with each of the k groups getting the chance to play the role of the
test data, and the remaining k (cid:1) 1 groups used as training data. In this way, we obtain
k different estimates of the prediction error. As prediction performance is reported
the average of these estimates of prediction error. CV is used in data analysis to
validate the implemented models where the main objective is prediction and to
estimate the prediction performance of a statistical learning model that will be
carried out in practice. In other words, CV evaluates how well the statistical machine
learning model generalized new data not used for training the model. The results of
the CV largely depend on how the division between the training and testing sets is
carried out. For this reason, in the following sections, we provide the more popular
types of CV used in the implementation of statistical learning models.

4.3.1 The Single Hold-Out Set Approach

The single hold-out set or validation set approach consists of randomly dividing the
available data set into a training set and a validation or hold-out set (Fig. 4.3). The
statistical machine learning model is trained with the training set while the hold-out

I1 I2 I3

Complete data set

Training

Testing

I40 I5 I82...

I62 I45

In

I88

Fig. 4.3 Schematic representation of the hold-out set approach. A set of observations are randomly
split into a training set with individuals I40, I5, I82, among others, and into a testing set with
observations I45, I88, among others. The statistical machine learning model is ﬁtted on the training
set and its performance is evaluated on the validation set (James et al. 2013)

116

4 Overﬁtting, Model Tuning, and Evaluation of Prediction Performance

set (testing set) is used to study how well that statistical machine learning model
performs on unseen data. For example, 80% of the data can be used for training the
model and the remaining 20% of the data for testing it. One weakness of the hold-out
(validation) set approach is that it depends on just one training-testing split and its
performance depends on how the data are split into the training and testing sets.

4.3.2 The k-Fold Cross-validation

In k-fold CV, the data set is randomly divided into k complementary folds (groups)
of approximately equal size. One of the subsets is used as testing data and the rest
(k (cid:1) 1) as training data. Then k (cid:1) 1 folds are used for training the statistical machine
learning model and the remaining fold for evaluating the out-of-sample prediction
performance. For these reasons, the statistical machine learning model is ﬁtted
k times using a different partition (fold) as the testing set and the remaining k (cid:1) 1
as the training set. Finally, the arithmetic mean of the k folds is obtained and reported
as the prediction performance of the statistical machine learning model (see Fig. 4.4).
This method is very accurate because it combines k measures of ﬁtness resulting
from the k training and testing data sets into which the original data set was divided,
but at the cost of more computational resources. In practice, the choice of the number
of folds depends on the measurement of the data set, although 5 or 10 folds are the
most common choices.

Fig. 4.4 Schematic representation of the k-fold cross-validation with complementary subsets with
k ¼ 5

4.3 Cross-validation

117

It

is important

to point out

to reduce variability, we recommend
implementing the k-fold CV multiple times, each time using different complemen-
tary subsets to form the folds; the validation results are combined (e.g., averaged)
over the rounds (times) to give a better estimate of the statistical machine learning
model predictive performance.

that

4.3.3 The Leave-One-Out Cross-validation

The Leave-One-Out (or LOO) CV is very simple since each training data set is
created by including all the individuals except one, while the testing set only
includes the excluded individual. Thus, for n individuals in the full data set, we
have n different training and testing sets. This CV scheme wastes minimal data, as
only one individual is removed from the training set.

Regarding the k-fold cross-validation that was just explained, n models are built
from n individuals (samples) instead of k models, where n > k. Moreover, each
model is trained on n (cid:1) 1 samples rather than (k (cid:1) 1)n/k. In both cases, since k is
normally not too large and k < n, LOO is more computationally expensive than k-
fold cross-validation. In terms of prediction performance, LOO normally produces
high variance for the estimation of the test error. Because n (cid:1) 1 of the n samples is
used to build each statistical machine learning model, those constructed from folds
are virtually identical to each other and to the model built from the entire training set.
However, when the learning curve is steep for the evaluated training size, then ﬁve-
or ten-fold cross-validation usually overestimates the generalization error.

Learning curves (LC) are considered effective tools to monitor the performance
of the employee exposed to a new task. LCs provide a mathematical representation
of the learning process that takes place as the task is repeated. In statistical machine
learning the LC is a line plot of learning (y-axis) over experience (x-axis). Learning
curves are extensively used in statistical machine learning for algorithms that learn
(their parameters) incrementally over time, such as deep neural networks. In
general, there is considerable empirical evidence suggesting that ﬁve- or ten-fold
cross-validation should be preferred to LOO.

4.3.4 The Leave-m-Out Cross-validation

The Leave-m-Out (LmO) CV is very similar to LOO as it creates all the possible
training/test sets by removing m samples from the complete set. For n samples, this

produces

train-test pairs. Unlike LOO and k-fold, the test sets will overlap for

  !
n

m

m > 1. For example, in a Leave-2-Out CV with a data set with four samples (I1, I2,

118

4 Overﬁtting, Model Tuning, and Evaluation of Prediction Performance

I3, and I4), the total number of training-testing sets is equal to

  !
4

2

¼ 6; this means

that the six testing sets are: [I3, I4] [I1, I2], [I2, I4] [I1, I3], [I2, I3] [I1, I4], [I1, I4]
[I2, I3],[I1, I3] [I2, I4], and [I1,I2] [I3, I4], while the training sets are the comple-
mentary elements of each testing set.

4.3.5 Random Cross-validation

In this type of CV, the number of partitions (independent training-testing data set
splits) is deﬁned by the user, and more partitions are better. Each partition is
generated by randomly dividing the whole data set into two subsets: the training
(TRN) data set and the testing (TST) data set. The percentage of the whole data set
assigned to the TRN and TST data sets is also ﬁxed by the user. For example, for
each random partition, the user can decide that 80% of the whole data set can be
assigned to the TRN data set and the remaining 20% to the TST data set. Random
cross-validation is different from k-fold cross-validation because the partitions are
not mutually exclusive; this means that in the random cross-validation approach, one
observation can appear in more than one partition. Consequently, some samples
cannot be evaluated, whereas others can be evaluated more than once, meaning that
the testing and training subsets can be superimposed (Montesinos-López et al.
2018a, b). To control the randomness for reproducibility, we recommend using a
speciﬁc seed in the random number generator. We recommend using at least ten
random partitions to obtain enough accuracy in the estimate of prediction
performance.

4.3.6 The Leave-One-Group-Out Cross-validation

The Leave-One-Group-Out (LOGO) CV is useful when individuals are grouped
(in environments or years, or even another criterion), where the number of groups (g)
is at least two and the information of g (cid:1) 1 groups are used as the training set while
all individuals of the remaining group are used as the testing set. For example, in the
context of genomic selection, when the plant breeder is interested in predicting these
lines in another environment, the same (or different) lines were frequently evaluated
in g environments or years, that represent the groups. Jarquín et al. (2017) denotes
this type of CV strategy as CV1 in the context of plant breeding. Under this
approach, the predictions are reported for each of the g groups because the scientist
is interested in the prediction performance of each environment. Many times, the
groups are the years under study and the aim is to predict the information of a
complete year. However, when the groups are years and if we suspect that there is a
considerable correlation between observations that are near in time. Therefore, it is

4.3 Cross-validation

119

Fig. 4.5 Schematic representation of time series data with 5 years, using the previous year for
predicting the next year. However, in some practical applications, we are not interested in going too
far back in time since the training and testing sets will be less related the farther you go

imperative to evaluate our statistical machine learning model for time series data on
“future” observations. In this sense, the training sets are composed of the previous
years to predict the subsequent year. This method can also be seen as a variation of k-
fold CV, where the ﬁrst folds are used for training the statistical machine learning
model and the fold (k + 1) is the corresponding testing set. The main difference in
this CV method is that successive training sets are supersets of those that come
before them. Also, it adds all surplus data to the ﬁrst training partition, which is
always used to train the model (see Fig. 4.5).

Figure 4.5 shows that under this type of CV, for time series data, the predictions
are for g (cid:1) 1 years; individuals from the ﬁrst year are not predicted since a training
set is not available.

4.3.7 Bootstrap Cross-validation

First, we will deﬁne the bootstrapping method to understand how it is used in the CV
approach, which should then be straightforward. Bootstrapping is a type of
resampling method where, for example, B ¼ 10 samples of the same size are
repeatedly drawn, with replacement, from a single original sample. Afterward,
each of these B samples is used to estimate statistics (for example, the mean,
variance, median, minimum, etc.) of a population, and the average of all the
B sample estimates of the target statistic is reported as the ﬁnal estimate. In the
context of statistical machine learning, these samples are used to evaluate the
prediction performance of the algorithm under study for unseen data. One important
difference between this CV approach and all the procedures explained above is that
now the training set has the same size (number of observations) as the original
sample because the bootstrap method replaced some individuals more than once.
According to Kuhn and Johnson (2013), as a result, some observations will be
represented multiple times in the bootstrap sample, while others will not be selected
at all; those observations not selected are referred to as the testing set, however, this
CV strategy is quite different than the previously explained. Efron (1983) pointed
out that the prediction performance of the bootstrap samples tends to have less
uncertainty than the k-fold cross-validation since on average, 63.2% of the data
points are represented (for training) at least once in any sample size. For this reason,

120

4 Overﬁtting, Model Tuning, and Evaluation of Prediction Performance

Fig. 4.6 Schematic representation of bootstrap cross-validation

this CV approach has a bias similar to implementing a k ¼ two fold cross-validation,
and as the training set becomes smaller, the bias becomes more problematic. To
understand this CV method, we provide a simple example of how the training and
testing samples are constructed. If we have a sample with 12 individuals denoted as
I1, I2, . . ., I12, we will select B ¼ 5 bootstrap samples. Each bootstrap sample is
obtained with replacement and the individuals that appear in each one correspond to
the training sample; those that are not present will correspond to the testing set.
Figure 4.6 provides the ﬁve bootstrap samples; each training sample has the same
size as the original, however, only some individuals appear in each bootstrap sample,
while those individuals that do not appear are included in the testing set. For
example, in the ﬁrst fold, the training bootstrap sample contains seven different
individuals (I2, I3, I4, I6, I7, I8, and I9), while the testing set contains ﬁve
individuals (I1, I5, I10, I11, and I12). It is important to point out that since the
training sample has the same size as the original sample, some individuals in the
training sample are repeated at least twice; in the ﬁrst fold, the individual I4 are
repeated three times, whereas I6, I7, and I8 are repeated twice. Finally, similar to
other methods, the statistical machine learning model is trained with each training
set; likewise, the prediction performance of the model is evaluated in each testing
set. The average of these sample predictions is reported as the estimated testing error.

4.3.8

Incomplete Block Cross-validation

Incomplete block (IB) CV should be used when there are J treatments evaluated in
I blocks and the same treatments are evaluated in all the blocks. The idea behind this

4.3 Cross-validation

Table 4.1 TRN data set for
I ¼ 3 blocks and J ¼ 10
treatments with 70% of data
for training

Block
1
2
3

121

10
9
10

7
5
6

8
6
7

9
7
8

Treatments per block
1
2
1

5
4
4

3
3
2

CV method is that some treatments should be present in some blocks but absent in
others, whereas the same treatment should be present in at least one environment
(block). The theory of incomplete block designs developed in the experimental
designs statistical area can be used to construct the training set. For example,
under a balanced incomplete block (BIB) design, the term incomplete means that
all treatments in each block cannot be evaluated, whereas balanced means that each
pair of treatments occur together λ times. The training set is constructed by ﬁrst
deﬁning the % of individuals in the TRN set using the equation sI ¼ Jr ¼ NTRN,
where J represents the number of treatments under study, I represents the number of
blocks under study, r denotes the number of repetitions of each treatment, and s
denotes the treatments per block. For example, suppose that we had J ¼ 10 treat-
ments and I ¼ 3 blocks (that is, 30 individuals), and we decided to use NTRN ¼ 21
(70%) of the total individuals in the TRN set. Therefore, the number of treatments by
block can be obtained by solving (sI ¼ NTRN) for s, which results in s ¼ NTRN/I. This
means that s ¼ 21/3 ¼ 7 treatments per block. Then, the corresponding elements for
the training set can be obtained with the function ﬁnd.BIB(10, 3, 7) of the package
crossdes of the R statistical software. The numbers used in the function ﬁnd.BIB()
denote the treatments, the blocks, and the treatments per block, respectively. Finally,
the treatments that make up the TRN set are shown in Table 4.1.

According to Table 4.1, it is clear that each treatment is present in two blocks and
missing in one block. For example, in Block 1 the testing set includes treatments 2, 4,
and 6; in Block 2, the testing set is composed of treatments 1, 8, and 10; and in Block
3, the testing set is composed of treatments 3, 5, and 9.

4.3.9 Random Cross-validation with Blocks

Random cross-validation with blocks was proposed by Lopez-Cruz et al. (2015) and
belongs to the so-called replicated TRN-TST cross-validation that appears in the
publication of Daetwyler et al. (2012), since some individuals can never be part of
the training set. This algorithm, like the incomplete block cross-validation, is
appropriate when we are interested in evaluating J lines in I blocks or environments
and tries to mimic a prediction problem faced by breeders in incomplete ﬁeld trials
where lines are evaluated in some, but not all, target environments. The algorithm for
constructing the TRN-TST sets is described by the following steps: Step 1. Calculate
the total number of observations under study as N ¼ J (cid:4) I; Step 2. Deﬁne the
proportion of observations used for training and testing, that is, PTRN and PTST; Step
3. Calculate the size of the testing set NTST ¼ N (cid:4) PTST; Step 4. Choose NTST lines at

122

4 Overﬁtting, Model Tuning, and Evaluation of Prediction Performance

Table 4.2 TRN-TST data sets for J ¼ 10 lines and I ¼ 3 environments

Environment

1
2
3

L2

Lines
L1
TRN TST
TST
TRN TRN TST

L3
L4
TRN TRN TST

L5

L8

L6
L7
TRN TRN TST

L9
L10
TRN TRN
TRN
TRN TRN TRN TRN TRN TST

TRN TST

TST

TRN TRN TRN TRN TRN TST

random without replacement if J (cid:5) NTST, and with replacement otherwise; Step
5. Each chosen line will then be assigned to one of the I environments chosen at
random without replacement; Step 5. All the selected lines and environments will
form the training set, while the lines and environments that were not chosen will
form the corresponding testing set; and Step 6. Steps 1–5 are repeated depending on
the number of TRN-TST partitions required (Lopez-Cruz et al. 2015). This CV is
called CV2 in Jarquín et al. (2017). Next, we assume that we have ten lines or
treatments and three environments or blocks that will form the corresponding
training testing sets for only one partition: Step 1. The total number of observations
under study is N ¼ 10 (cid:4) 3 ¼ 30; Step 2. We deﬁne PTRN ¼ 0.7 and PTST ¼ 0.3; Step
3. The size of the testing set is NTST ¼ 30 (cid:4) 0.3 ¼ 9; Step 4. Since J ¼ 10 (cid:5) NTST ¼ 9,
we selected the following lines at random without replacement: L1, L2, L3, L4, L5,
L6, L7, L8, L9, and L10; and Step 5. Each chosen line was assigned to one of the
I ¼ 3 environments randomly chosen without replacement, as shown in Table 4.2. It
is important to point out that this CV strategy only differs from the incomplete block
cross-validation in the way the lines are allocated to blocks.

4.3.10 Other Options and General Comments

on Cross-validation

It is important to highlight that when the data set is considerably large, it is better to
randomly split it into three parts: a training set, a validation set (or tuning set), and a
testing set. The training set and testing set are used as explained before, while the
validation (tuning set) set is used to estimate the prediction error for model selection,
which is the process of estimating the performance of different models in order to
choose the best one, or to evaluate the chosen statistical machine learning model
with a range of values of tuning hyperparameters to select the combination of
hyperparameters with the best prediction performance and then use these
hyperparameters (or best model) to evaluate the prediction performance in the testing
set (Fig. 4.7). It is important to point out that Fig. 4.7 shows only one random split of
the data in terms of the training, testing, and validation sets.

For example, assume that our data set has 50,000 rows (observations) and that we
have decided to use 5000 of them for the testing set and another 5000 for the
validation set. This means that 40,000 rows are left for the training set. At this
point, we train our statistical machine learning model with each component of the

4.3 Cross-validation

123

Fig. 4.7 Schematic representation of the training, validation (tuning), and testing sets proposed by
Cook (2017)

grid of hyperparameters of the training set and evaluate the prediction performance
on the validation set, as shown in the middle of Fig. 4.7. Finally, we will pick the best
model (best subset of hyperparameters) in terms of prediction performance in the
validation set and we are ready to evaluate the prediction performance in the testing
set. Then, we will report the testing error on the testing set, as can be observed at the
bottom of Fig. 4.7 (Cook 2017). Under this approach, the testing and validation sets
have approximately the same size to guarantee a similar out-of-sample prediction
performance. Since it is difﬁcult to give general rules on how to choose the number
of observations in each of the three parts, a typical number might be 50% for
training, and 25% each for validation and testing (Hastie et al. 2008) or 70%,
15%, and 15% for training, validation, and testing, respectively. Another way to
ﬁnd the optimal setting of hyperparameters using the grid search, which is very
common in deep learning, consists of picking values for each parameter from a ﬁnite
set of options [e.g., number of epochs (100, 150, 200, 250, 300); batch sizes (25, 50,
75, 100, 125); number of layers (1, 2, 3, 4, 5); and types of activation functions
(RELU, Sigmoid), . . .] and training the statistical machine learning model with
every permutation of hyperparameter choices using the training set. Then, the
combination of hyperparameters with the best prediction performance on the vali-
dation set is chosen, and we report the prediction performance of the best selected
model (set of hyperparameters) in the testing set (Buduma 2017). The aforemen-
tioned examples use the validation data set as a proxy measure of the accuracy
during the hyperparameter optimization process. However, it can also be used as a
proxy measure of the accuracy for model selection, and instead of using a grid of
hyperparameters, we can use a set of different statistical machine learning models;
here, the best model is chosen instead of the best combination of hyperparameters. It
is important to understand that when you have more than one random partition
(training-testing-validation), as shown in Fig. 4.8, the same process provided in
Fig. 4.7 is followed, however, the average of all the partitions is reported as a
measure of prediction performance. Also, if more precision is required in the
estimated prediction performance, you can repeat the process given in Fig. 4.8

124

4 Overﬁtting, Model Tuning, and Evaluation of Prediction Performance

Fig. 4.8 Five partitions of training-validation-testing

multiple times and report the average of all repetitions as a measure of prediction
performance.

As mentioned above, this approach (Figs. 4.7 and 4.8) is used for a large data set
however, in the case of a smaller data set, we suggest modifying this approach to
avoid wasting too much training data in validation sets. This modiﬁcation consists of
performing an inner cross-validation approach since the training set is split into
complementary subsets, and each model is trained against a different combination of
these subsets, which is subsequently validated against the remaining parts. Once the
model hyperparameters have been selected, a ﬁnal model is trained with the whole
training set (reﬁtted) and the generalized prediction performance is measured on the
testing set. This approach was applied by Montesinos-López et al. (2018a, b), who
also split the original data into training and testing sets. Subsequently, each training
set was split again and 80% of the data was used for training a grid of
hyperparameters while the remaining 20% was used for validating (tuning) the
prediction performance and selecting the best combination of hyperparameters
with the best prediction performance. At this point, the deep learning algorithm
was reﬁtted with the whole training set, and with this they evaluated the out-of-
sample prediction. They called the conventional training-testing partition outer
cross-validation, while the split performed in each training set used for
hyperparameter tuning was called inner cross-validation. It should be highlighted
that there are no differences between outer and inner CV and training-validation-test.
Finally, it should also be mentioned that any type of the cross-validation strategies
mentioned in this section (random CV, k-fold CV, Bootstrap CV, IB CV, etc.,) can
be used in both outer and inner CV, such as is the case with the ﬁve-fold CV.

4.4 Model Tuning

A hyperparameter is a parameter whose value is set before the learning process
begins. Hyperparameters govern many aspects of the behavior of statistical machine
learning models, such as their ability to learn features from data, the models’
exhibited degree of generalizability in performance when presented with new data,
as well as the time and memory cost of training the model, since different
hyperparameters often result in models with signiﬁcantly different performance.
This means that tuning hyperparameter values is a critical aspect of the statistical
machine learning training process and a key element for the quality of the resulting

4.4 Model Tuning

125

Fig. 4.9 Schematic representation of the tuning process proposed by Kuhn and Johnson (2013)

prediction accuracies. However, choosing appropriate hyperparameters is challeng-
ing (Montesinos-López et al. 2018a). Hyperparameter tuning ﬁnds the best version
of a statistical machine learning model by running many training sets on the original
data set using the algorithm and ranges of values of hyperparameters as speciﬁed.
The hyperparameter values that provide the best performance in out-of-sample
prediction evaluated by the chosen metric are then selected.

There are many ways of searching for the best hyperparameters. However, a
general approach deﬁnes a set of candidate values for each hyperparameter. Each
value of this set of candidate values is then applied with a resample of the training set
of the chosen statistical machine learning method, where we aggregate all the hold-
out predictions from which the best hyperparameters are chosen and reﬁt the model
with the entire set (Kuhn and Johnson 2013). A schematic representation of the
tuning process proposed by Kuhn and Johnson (2013) is given in Fig. 4.9. It is
important to highlight that this process should be performed correctly because when
the same data are used for training and evaluating the prediction performance, the
prediction performance obtained is extremely optimistic.

For example, suppose a breeder is interested in developing an algorithm to
classify unseen plants as diseased or not diseased with an available training data
set. The goal is to minimize the rate of misclassiﬁcation or to maximize the
percentage of cases correctly classiﬁed (PCCC). Also, assume that you are new to
the world of statistical machine learning and that you only understand the k-nearest
neighbor method. Since this algorithm depends only on the hyperparameter called
the number of neighbors (k), the question is which value of k to choose in such a way
that the prediction performance of this algorithm will be the best in the sample
prediction of plants. To ﬁnd the best value of the k hyperparameter, you must specify
a range of values for k (for example, from 1 to 60 with increments of 1), then with a
part of your training data set, called the training-inner (or tuning that corresponds to
the training data in the inner loop) set, which is randomly selected. You proceed to
evaluate the 60 values of k with the k-nearest neighbor method and evaluate the

126

4 Overﬁtting, Model Tuning, and Evaluation of Prediction Performance

prediction performance in the remaining part of the training set (validation set).
Next, you select the value of k from this range of values that best predicts (according,
for example, to the PCCC) out-of-sample data (validation set) and use this value to
perform the prediction of the unseen plants not used for training the model (testing
set). This is a widely adopted practice that consists of searching for the parameter
(usually through brute force loops) that yields the best performance over a validation
set. However, the process illustrated here is very simple because the k-nearest
neighbor model only depends on a unique hyperparameter; however, there are
other statistical machine learning algorithms (for example, deep learning methods)
where the tuning process is required for a considerable amount of hyperparameters.
For this reason, we encourage caution when choosing the statistical machine learn-
ing algorithm, since the amount of work required for performing the tuning process
depends on the chosen method.

4.4.1 Why Is Model Tuning Important?

implementation,

Tuning the hyperparameters of the models is a key element to optimize your
statistical machine learning model to perform well in out-of-sample predictions.
The tuning process is more an art than a science because there is no unique formal
scientiﬁc procedure available in the literature. Nowadays, the tuning process is trial
and error that consists of implementing the statistical machine learning model many
times with different values of the hyperparameters and then comparing its perfor-
mance on the validation set in order to determine which set of hyperparameters
results in the most accurate model; for the ﬁnal
the set of
hyperparameters of the best model is used. As mentioned above, for the k-nearest
neighbor classiﬁer, we need to choose the number of neighbors (k) using the tuning
process to obtain the optimal prediction performance of this algorithm, while for
conventional Ridge regression, the parameter lambda (λ) is obtained by tuning to
improve the out-of-sample predictions. These two statistical machine learning algo-
rithms that we just mentioned need only one hyperparameter; however, other
statistical machine learning methods may require more hyperparameters, as exem-
pliﬁed by deep learning models that require at least three hyperparameters (number
of neurons, number of hidden layers, type of activation function, batch size, etc.).
After tuning the required hyperparameters, the statistical machine learning model
learns the parameters from the data to be used for the ﬁnal prediction of the testing
set. The choice of hyperparameters signiﬁcantly inﬂuences the time required to train
and test a statistical machine learning model. Hyperparameters can be continuous or
of the integer type; for this reason, there are mixed-type hyperparameter optimiza-
tion methods.

4.4 Model Tuning

127

4.4.2 Methods for Hyperparameter Tuning (Grid Search,

Random Search, etc.)

Manual tuning of statistical machine learning models is of course possible, but relies
heavily on the user’s expertise and understanding of the underlying problem.
Additionally, due to factors such as time-consuming model evaluations, nonlinear
hyperparameter interactions in the case of large models that consist of tens or even
hundreds of hyperparameters, manual tuning may not be feasible since it is equiv-
alent
the four most common approaches for
hyperparameter tuning reported in the literature are (a) grid search, (b) random
search, (c) Latin hypercube sampling, and (d) optimization (Koch et al. 2017).

to brute force. For this reason,

In the grid search method, each hyperparameter of interest is discretized into a
desired set of values to be studied where the models are trained and assessed for all
combinations of the values across all hyperparameters (that is, a “grid”). Although
fairly simple and straightforward to carry out, a grid search is appropriate when there
are only a few values for a limited number of hyperparameters. However, although
this is a comprehensive way of assessing different hyperparameter values, when
there are many values for some or many hyperparameters, it quickly becomes quite
costly due to the number of hyperparameters and the number of discrete levels of
each. For example, in Ridge regression, this approach is implemented as follows:
since λ is the hyperparameter to be tuned, we ﬁrst propose, for example, a grid of
100 values for this hyperparameter from λ ¼ 1010 to λ ¼ 10(cid:1)2; then we divide the
training set into ﬁve inner training sets and ﬁve inner testing (tuning) sets, where
each of the 100 values of the grid is ﬁtted using the inner training sets and the testing
error is evaluated with the inner testing sets. Then we get the average predicted test
error and pick one value out of the 100 values of the grid that produces the best
prediction performance. Next, we reﬁt the statistical machine learning method to the
whole training set using the picked value of λ, and ﬁnally perform the predictions for
the testing set using the learned parameters of the training set with the best picked
value of λ. In all the models with one hyperparameter, it is practical to implement the
grid search method, but for example, in deep learning models, which many times
require six hyperparameters to be tuned, if only three values are used for each
hyperparameter, there are 36 ¼ 729 combinations that need to be evaluated, quickly
becoming computationally impracticable.

A random search differs from a grid search in that rather than providing a
discrete set of values to explore each hyperparameter, we determine a statistical
distribution for each hyperparameter from which values may be randomly sampled.
This affords a much greater chance of ﬁnding effective values for each
hyperparameter. While Latin hypercube sampling is similar to the previous method,
it is a more structured approach (McKay 1992) since it is an experimental design in
which samples are exactly uniform across each hyperparameter but random in
combinations. These so-called low-discrepancy point sets attempt to ensure that
points are approximately equidistant from one another in order to ﬁll the space

128

4 Overﬁtting, Model Tuning, and Evaluation of Prediction Performance

efﬁciently. This sampling supports coverage across the entire range of each
hyperparameter and is more likely to ﬁnd good values of each hyperparameter.

The previous two methods for hyperparameter tuning are used to perform indi-
vidual experiments by building models with various hyperparameter values and
recording the model performance for each. Because each experiment is performed
in isolation, this process is parallelized, but is unable to use the information from one
experiment to improve the next experiment. Optimization methods, on the other
hand, consist of sequential model-based optimization where the results of previous
experiments are used to improve the sampling method of the next experiment. These
methods are designed to make intelligent use of fewer evaluations and thus save on
the overall computation time (Koch et al. 2017). Optimization algorithms that have
been used in statistical machine learning generally for hyperparameter tuning
include Broyden–Fletcher–Goldfarb–Shanno (BFGS) (Konen et al. 2011), covari-
ance matrix adaptation evolution strategy (CMA-ES) (Konen et al. 2011), particle
swarm (PS) (Renukadevi and Thangaraj 2014), tabu search (TS), genetic algorithms
(GA) (Lorena and de Carvalho 2008), and more recently, surrogate-based Bayesian
optimization (Dewancker et al. 2016). Also, recently the use of the response surface
methodology has been explored for tuning hyperparameters in random forest models
(Lujan-Moreno et al. 2018). However, the implementation of these optimization
methods is not straightforward because it requires expensive computation; also,
software development is required for implementing these algorithms automatically.
There have been advances in this direction for some machine learning algorithms in
the statistical analysis system (SAS), R and Python software (Koch et al. 2017). An
additional challenge is the potential unpredictable computation expense of training
and validating predictive models using different hyperparameter values. Finally,
although it is challenging, the tuning process often leads to hyperparameter settings
that are better than the default values, as it provides a heuristic validation of these
settings, giving greater assurance that a model conﬁguration with a higher accuracy
has not been overlooked.

4.5 Metrics for the Evaluation of Prediction Performance

The quality of prediction performance of any statistical machine learning method in
a given data set consists of evaluating how close the predicted values are to the true
observed ones. In other words, the prediction performance quantiﬁes the matching
degree between the predicted response value for a given observation and the true
response value for that observation (James et al. 2013). However, the metrics used
for quantifying the prediction performance depend on the type of response variable
under study; for this reason, we subsequently give the most popular metrics used for
this goal for four types of response variables.

4.5 Metrics for the Evaluation of Prediction Performance

129

4.5.1 Quantitative Measures of Prediction Performance

Before implementing a statistical machine learning model, we assume that we have
our training observation {(x1, y1), (x2, y2), . . ., (xn, yn)} and we estimate f, as bf , with
the chosen statistical machine learning model. Then we can make predictions for
each of the response values (yi) with bf xið Þ and compute the predicted values for each
of the n observations in the training set; with these values we can calculate the mean

square error (MSE) for the training data set as E ¼ 1
n

(cid:6)

(cid:2)

Pn

i¼1

(cid:3)
yi (cid:1) bf xið Þ

2

; however,

what we really want to predict are the values for unseen test observations that were
not used to train the statistical machine learning model. Assuming that the unseen
testing set is equal to {(xn + 1, yn + 1), (xn + 2, yn + 2), . . ., (xn + T, yn + T)}, the MSE for
the testing data set should be calculated as

MSETST ¼

1
T

XnþT

(cid:2)

yi (cid:1) bf xið Þ

(cid:3)
2

,

i¼nþ1

ð4:1Þ

where bf xið Þ is the prediction that bf gives to the ith observation. The MSETST with a
lower value will have better predictions, which means that the predicted values are
very close to the true observed values. Also, the square root of MSETST can be used
as a measure of prediction performance and is called root mean square error
(RMSE).

Pearson’s correlation coefﬁcient is a very popular measure of prediction perfor-

mance in plant breeding and can be calculated as

rTST ¼

r

P

(cid:2)
bf xið Þ (cid:1) bf xið Þ
q

nþT
i¼nþ1
(cid:2)
bf xið Þ (cid:1) bf xið Þ

ﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃ
(cid:3)2
P

nþT
i¼nþ1

(cid:3)
ð

yi (cid:1) yi

Þ
ﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃ
P
nþT
Þ2
i¼nþ1 yi (cid:1) yi
ð

,

ð4:2Þ

where bf xið Þ is the average of the T predictions that conform to the testing set, and yi is
the average of the T true observed values. In this case, the closer that the predictions
are to 1, the better the implemented statistical machine learning model will perform.
It is important to point out that Pearson’s correlation is deﬁned between (cid:1)1 and
1. However, to be convinced that the observed and predicted values match, it is a
common practice to perform a scatter plot of predicted versus observed (or vice
versa) values and when the observed and predicted values follow a straight line (45(cid:6)
diagonal line) from the bottom left corner to the top right corner, this indicates a
perfect match between the observed and predicted values. For this reason, Pearson’s
correlation as a metric should be complemented with the intercept and slope, since
the slope and intercept describe the consistency and the model bias, respectively
(Smith and Rose 1995; Mesple et al. 1996). To obtain the slope and intercept, the
observed values (as y) versus the predicted values (as x) are regressed and in addition

130

4 Overﬁtting, Model Tuning, and Evaluation of Prediction Performance

to Pearson’s correlation, these values should also be reported (slope and intercept).
The expected intercept should be zero and the slope 1, if the correlation obtained
between the observed and predicted values is high. It is important to avoid carrying
out the regression in the opposite way, i.e., using predicted values as y’s, and
observed values as x’s, since this leads to incorrect estimates of the slope and the
y-intercept. This denotes that a spurious effect is added to the regression parameters
when regressing predicted versus observed values and comparing them against the 1:
1 line. The user should also remember that underestimation of the slope and
overestimation of the y-intercept increase as Pearson’s correlation values decrease.
We strongly recommend that scientists evaluate their models by regressing observed
versus predicted values and test the signiﬁcance of slope ¼ 1 and intercept ¼ 0
(Piñeiro et al. 2008). Finally, it is important to recall that the square of Pearson’s
correlation can also be used as a metric for measuring prediction performance since it
represents the proportion of the total variance explained by the regression model and
is called the coefﬁcient of determination denoted as R2.

Next, we present the mean absolute error (MAE) metric that measures the
difference between two continuous variables (observed and predicted). The MAE
can be calculated with the following expression:

MAETST ¼

XnþT

(cid:8)
(cid:8)
(cid:8)

(cid:8)
(cid:8)
yi (cid:1) bf xið Þ
(cid:8)

i¼nþ1

1
T

ð4:3Þ

Below we present another metric used to evaluate the prediction performance of
any statistical machine learning model; it was proposed by Kim and Kim (2016) and
is called mean arctangent absolute percentage error (MAAPE) that is calculated as
follows:

P

nþT
i¼nþ1 arctan

(cid:8)
(cid:6)
(cid:8)
(cid:8)
(cid:8)

yi(cid:1)bf xið Þ
yi

(cid:9)

(cid:8)
(cid:8)
(cid:8)
(cid:8)

T

MAAPETST ¼

ð4:4Þ

Although MAAPE is ﬁnite when the response variable (i.e., yi ¼ 0) equals zero,
since it has a satisfactory trigonometric representation. However, because MAAPE’s
value is expressed in radians, it is less intuitive, in addition to being scale-free. This
metric is a modiﬁcation of the mean absolute percentage error (MAPE) which is
problematic because it is undeﬁned when the response variable is equal to zero
(yi ¼ 0). MAAPE is also asymmetric since division by zero is deﬁned and is not a
problem. It is important to point out that there are other metrics for measuring
prediction accuracy for continuous data, but we only presented the most
popular ones.

The distinction between the training and test MSE is important since we are not
interested in how well the statistical machine learning method performs in the
training data set, due to the fact that our main goal is to perform accurate predictions
in the unseen test data. For example, a plant breeder may be interested in developing
an algorithm to predict disease resistance of a plant in new environments based on

4.5 Metrics for the Evaluation of Prediction Performance

131

records that were collected in a set of environments. We can train the statistical
machine learning method with the information collected in the set of environments,
but the interest is not in how well the statistical machine learning method predicts in
those previously collected environments. An environmental scientist can also be
interested in predicting the average annual rainfall in a municipality in Mexico, using
data from the last 20 years to train the model. Such measures could include sea
surface temperature, time, the yearly rotation of the earth, among others. In this case,
the scientist is really interested in predicting the average rainfall of the next 1 or
2 years, not in accurately predicting the years measured in the training set.

4.5.2 Binary and Ordinal Measures of Prediction

Performance

The binary and ordinal response variables are very common in classiﬁcation prob-
lems, where the goal is to predict which category something falls into. An example
of a classiﬁcation problem is analyzing ﬁnancial data to determine if a client will be
granted credit or not. Another example is analyzing breeding data to predict if an
animal is at high risk for a certain disease or not. Below, we provide some popular
metrics to evaluate the prediction performance of this type of data.

The ﬁrst metric is called a confusion matrix, which is a tool to visualize the
performance of a statistical machine learning algorithm that is used in supervised
learning for classifying categorical and binary data. Each column of the matrix
represents the number of predictions in each class, while each row represents the
instances in the real class. One of the beneﬁts of confusion matrices is that they make
it easy to determine whether the system is confusing classes. Table 4.3 shows a
sample format of a confusion matrix of C classes.

With Eqs. (4.5–4.8) we can calculate the total number of false negatives (TFN),
false positives (TFP), true negatives (TTN) for each class i, and the total true
positives in the system, respectively:

XC

TFNi ¼

nij

j¼1

j6¼i

ð4:5Þ

Table 4.3 Confusion matrix with more than two classes

Observed values

Predicted values
Class 1
n11
n21
⋮

nC1

Class 2
n12
n22
⋮

nC2

Class 1
Class 2
⋮

Class C

. . .
. . .
. . .
⋮
. . .

Class C
n1C
n2C
⋮

nCC

132

4 Overﬁtting, Model Tuning, and Evaluation of Prediction Performance

XC

TFPi ¼

nji

j¼1

j6¼i

XC

XC

TTNi ¼

nji

j¼1

k¼1

j6¼i

k6¼i

XC

TTPall ¼

njj

j¼1

ð4:6Þ

ð4:7Þ

ð4:8Þ

Below, we deﬁne the sensitivity (Se), precision (P), and speciﬁcity (Sp). The
sensitivity indicates the ability of our statistical learning algorithm to determine the
proportion of true positives that are correctly identiﬁed by the test. The precision is
the proportion of correct classiﬁcation of our statistical machine learning model and
represents the proportion of cases correctly classiﬁed, while the speciﬁcity is the
ability of our statistical machine learning model to classify the true negative cases,
that is, the speciﬁcity is the proportion of true negatives that are correctly identiﬁed
by the test. Under the “one-versus-all basis,” where each category is compared with
the composed information of the remaining categories, we provide the expressions
for computing the generalized precision, sensitivity, and speciﬁcity for each class i:

Pi ¼

Sei ¼

Spi ¼

TTPall
TTPall þ TFPi
TTPall
TTPall þ TFNi
TTNall
TTNall þ TFPi

pCCC ¼

TTNall
PC

PC

nij

ð4:9Þ

ð4:10Þ

ð4:11Þ

ð4:12Þ

i¼1

j¼1

The term pCCC denotes the proportion of cases correctly classiﬁed, which is a
measure of the overall accuracy, and when multiplied by 100, denotes the percentage
of cases correctly classiﬁed. Many times, this is the only metric reported for
measuring prediction performance in multi-class problems. However, PCCC alone
is sometimes quite misleading as there may be a model with relatively “high”
accuracy, but it predicts the “unimportant” class labels fairly accurately (e.g.,
“unknown bucket”). However, the model may be making all sorts of mistakes on
the classes that are actually critical to the application. This problem is serious when
in the input data the number of samples of different classes is very unbalanced. For

4.5 Metrics for the Evaluation of Prediction Performance

133

Table 4.4 Confusion matrix
with two classes

Observed values

Predicted values
False
True
fn
tp
tn
fp
fn + tn
tp + fp

Sum
tp + fn
fp + tn
n

True
False
Sum

tp denotes true positives, fp denotes false positives, fn denotes
false negatives, tn denotes true negatives, and n denotes total
number of individuals

example, if there are 990 samples of class 1 and only 10 of class 2, the classiﬁer can
easily have a bias toward class 1. If the classiﬁer classiﬁes all samples as class 1, its
accuracy will be 99%. This does not mean that it is an appropriate classiﬁer, as it had
a 100% error when classifying the samples of class 2. For this reason, reporting this
metric with those reported in Eqs. (4.9–4.11) is recommended in order to have a
better picture of the prediction performance of any statistical machine learning
method (Ratner 2017). Also, it is important to highlight that when the problem
only has two classes, the confusion matrix is reduced to Table 4.4.

From Table 4.4 the PCCC is calculated as tpþtn

tnþfp ,
and P ¼ tp
tpþfp . Also, when there are only two classes, González-Camacho et al.
(2018) suggest calculating the Kappa coefﬁcient (κ) or Cohen’s Kappa, which is
deﬁned as

n , while the Se ¼ tp

tpþfn, Sp ¼ tn

κ ¼

P0 (cid:1) Pe
1 (cid:1) Pe

,

n (cid:4) tpþfp

n þ fpþtn

where P0 is the agreement between observed and predicted values and is computed
by the PCCC described above for two classes; Pe is the probability of agreement
calculated as Pe ¼ tpþfn
n (cid:4) fnþtn
, where fp is the number of false
n
positives, and fn is the number of false negatives (Table 4.4). This statistic can
take on values between (cid:1)1 and 1; a value of 0 means there is no agreement between
the observed and predicted classes, while a value of 1 indicates perfect agreement
between the model prediction and the observed classes. Negative values indicate that
a prediction may be incorrect; however large negative values seldom occur when
working with predictive models. Depending on the context, a Kappa value from 0.30
to 0.50 indicates reasonable agreement (Kuhn and Johnson 2013). The Kappa
coefﬁcient is appropriate when data are unbalanced, because it estimates the pro-
portion of cases that were correctly identiﬁed by taking into account coincidences
expected from chance alone (Fielding and Bell 1997). It is important to point out that
this statistic was originally designed to assess the agreement between two raters
(Cohen 1960).

Another popular metric for binary data is the Area Under the receiver operating
characteristic Curve (AUC–ROC) and it ranks the positive predictions higher than
the negative. The ROC curve is deﬁned as a plot of 1 (cid:1) speciﬁcity or false positive
rate (FPR) as the x-axis versus its model sensitivity as the y-axis. For a given set of

134

4 Overﬁtting, Model Tuning, and Evaluation of Prediction Performance

Fig. 4.10 ROC curve for
the synthetic data

thresholds τ, it is an effective method for evaluating the quality or performance of
diagnostic tests, and is widely used in statistical machine learning to evaluate the
prediction performance of learning algorithms. Since the mathematical construction
of this metric is not required in this book, we illustrate its calculation with one simple
example. Assume that the observed (y), predicted probabilities (pi) and predicted
values (ŷ) obtained after implementing a statistical machine learning model are
y ¼ {1, 0, 1, 1, 0, 0, 1, 1, 0, 1}, pi ¼ {0.6, 0.55, 0.8, 0.78, 0.3, 0.42, 0.9, 0.45,
0.3, 0.88}, and by ¼ f1, 1, 1, 1, 0, 0, 1, 0, 0, 1}; then, using the following R code, we
can obtain many metrics for binary data (Fig. 4.10):

########################Libraries required########################

library(caret)
library(pROC)
##Observed (y), predicted probability (pi) and predicted values of

synthetic data##

y=c(1, 0,1, 1, 0, 0, 1, 1, 0, 1)
pi=c(0.6, 0.55, 0.8, 0.78, 0.3, 0.42, 0.9, 0.45, 0.3, 0.88)
yhat=c(1, 1, 1, 1, 0, 0, 1, 0, 0, 1)
xtab <- table(y,yhat)
confusionMatrix(xtab)

plot.roc(y,pi) #####This make the ROC curve plot

Confusion Matrix and Statistics

yhat
y 0 1
0 3 1
1 1 5

4.5 Metrics for the Evaluation of Prediction Performance

135

Accuracy : 0.8

95% CI : (0.4439, 0.9748)

No Information Rate : 0.6
P-Value [Acc > NIR] : 0.1673

Kappa : 0.5833
Mcnemar's Test P-Value : 1.0000

Sensitivity : 0.7500
Speciﬁcity : 0.8333
Pos Pred Value : 0.7500
Neg Pred Value : 0.8333
Prevalence : 0.4000
Detection Rate : 0.3000
Detection Prevalence : 0.4000
Balanced Accuracy : 0.7917

'Positive' Class : 0

It is important to point out that when there are more than two classes, these
metrics (accuracy, sensitivity, speciﬁcity, etc.) can be calculated on a “one-versus-
all” basis that consists of using each class versus the pool of the remaining classes, as
was illustrated for the confusion matrix with more than two classes (James et al.
2013).

When the statistical machine learning model discriminates correctly between the
two groups, it produces a curve that coincides with the left and top sides of the plot.
Under this scenario, the perfect model would have 100% sensitivity and speciﬁcity,
and the ROC curve would be a single step between (0,0) and (0,1) and would remain
constant from (0,1) to (1,1); this implies that the area under the ROC curve of the
model would be 1. In general, the larger the area under the ROC curve, the better the
model in terms of prediction performance. On the other hand, a completely useless
statistical machine learning algorithm would give a straight line (45(cid:6) diagonal line)
from the bottom left corner to the top right corner of the plot. Different statistical
machine learning models or the same model with different
training sets or
hyperparameters can be compared by superimposing their ROC curves in the same
graph. In practice, most of the time the values in the two groups overlap, so the curve
often lies between these extremes.

From this ROC curve, we can obtain a global assessment of the prediction
performance of the statistical machine learning method by measuring the area
under the receiver operating characteristic curve. This area is equal to the probability
that a random individual (person) of the sample with the presence of the target has a
higher value of the measurement than a random individual without the target.

When a statistical machine learning model is unable to discriminate between the
positive and negative classes, this means that it has low discriminatory power.
Therefore, only in statistical machine learning models that had good discriminatory
power we can be conﬁdent of the predictions they provide and furthermore those
models that provide a curve that lies considerably above the curve will be better.

136

4 Overﬁtting, Model Tuning, and Evaluation of Prediction Performance

Matthews correlation coefﬁcient (MCC). Introduced in 1975 by Brian Matthews
(1975) and regarded by many scientists as the most informative score that connects
all four measures in a confusion matrix, the Matthews Correlation Coefﬁcient is
typically used in statistical machine learning to measure the quality of binary
classiﬁcations and it is particularly useful when there is a signiﬁcant imbalance in
class sizes (data). MCC is calculated according to the following expression:

MCC ¼

p

tp (cid:4) tn (cid:1) fp (cid:4) fn
ﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃ
Þ (cid:4) tn þ fn
tp þ fp
Þ
ð
ð

Þ (cid:4) tp þ fn

Þ (cid:4) tn þ fp

ð

ð

ð4:13Þ

If any of the denominator terms equals zero in (4.13) it will be set to 1 and MCC
becomes zero, which has been shown to be the correct limiting value. It returns a
value between (cid:1)1 and 1, where 1 means a perfect prediction, 0 means no better than
random and (cid:1)1 means a total disagreement between predicted and observed values.
Next, we present the Brier score (Brier 1950) for categorical or binary data that

can be computed as

BS ¼ T (cid:1)1

XnþT

XC

i¼nþ1

c¼1

bπic (cid:1) dic
ð

Þ2,

ð4:14Þ

where bπi denotes the estimated probabilities (predictive distribution) derived from
the estimated model for observation i and dic takes a value of 1 if the categorical
response observed for individual i falls into category c; otherwise, dic ¼ 0. The range
of BS in Eq. (4.14) for categorical data is between 0 and 2. For this reason, we
suggest dividing by 2, that is, BS/2, to obtain the Brier score bound between 0 and 1;
lower scores imply better predictions (Montesinos-López et al. 2015a, b).

Finally, we describe the use of negative log-likelihood (MLL) to evaluate the
prediction performance. This metric has the characteristic that better forecasts have
lower values and for this reason, it is analogous to the MSE. For categorical data,

MLL ¼ (cid:1)

"

1
T

XnþT

XC

i¼nþ1

c¼1

#

1 yi ¼ k
f

g log bπicð

Þ

,

(cid:10)

P

data

the MLL is

are
Þ þ 1 (cid:1) yi
ð

binary,
Þ log 1 (cid:1) bπi
ð

where 1{y(i) ¼ k} is an indicator variable taken the value of 1 when the ith
observation is assigned to category c, for c ¼ 1, 2, . . ., C takes place in the ith
the
observation. When
to
nþT
i¼nþ1 yi log bπið
MLL ¼ (cid:1) 1
. Following, we provide
½
T
some advantages of using the MML as a measure of prediction performance: (a) it
has a simple deﬁnition that, from a purely intuitive point of view, seems to be a
reasonable basis on which to compare forecasts; (b) it is mathematically optimal in
the sense that estimates of parameters of calibration models ﬁtted by maximizing the
likelihood are usually the most accurate possible estimates (see Cassella and Berger
2002); (c) it is a generalization to probabilistic forecasts of the most commonly used
skill score for single forecasts; (d) the properties of the likelihood have been studied

reduced

(cid:11)

Þ

(cid:7)

4.5 Metrics for the Evaluation of Prediction Performance

137

at great length over the last 90 years, and as such is well understood; (e) it is both a
measure of resolution and reliability; (f) likelihood can be used for both calibration
and assessment: this creates consistency between these two operations; (g) use of the
likelihood also creates consistency with other statistical modeling activities, since
most other statistical modeling uses the likelihood, which is important in cases where
the use of forecasts is simply a small part of a larger statistical modeling effort, as is
the case of our particular business; (h) likelihood can be used for all types of
response variables; and (i) likelihood can be used to compare multiple leads,
multiple variables, and multiple locations at the same time in a sensible way with
a single score even when these leads, variables, and locations are cross-correlated.

4.5.3 Count Measures of Prediction Performance

Spearman’s correlation and the MML are recommended to measure the prediction
performance for count data.

for

the

the

rank

ﬁnal

values

observed

rank for

show how to get

the observed values:

For the application of the Spearman’s correlation, the formula given in Eq. (4.2)
for Pearson’s correlation can be used; however, instead of using the observed and
predicted values directly, these are replaced by their corresponding ranks. For
example, assuming that the observed and predicted values are y ¼ {15, 9, 12,
27, 6, 3, 36, 15, 21, 30} and by ¼ f20, 17, 24, 25, 3, 3, 34, 22, 21, 33}, we can
thus
rangoy ¼
{5, 3, 4, 8, 2, 1, 10, 6, 7, 9}. However, in this vector, observations 1 and 8 are the
2 ¼ 5:5.
same, and as such, their positions are added and divided by two, that is, 5þ6
Therefore,
rangoy ¼
the
{5.5,3, 4, 8, 2, 1, 10,5.5,7, 9}. Now the range for the predicted values is rangoby ¼
g, but again, since values 5 and 6 are the same, we add their
4, 3, 7, 8, 1, 2, 10, 6, 5, 9
f
2 ¼ 1:5.
ranges, and as this is repeated twice, it is divided by two and we get 1þ2
the
Therefore,
rangoby ¼
of
values
the
4, 3, 7, 8, 1:5, 1:5, 10, 6, 5, 9
to obtain the Spearman correlation, we
f
used the expression given in Eq. (4.2) for Pearson’s correlation, and instead of
using the original observed and predicted values, we used rangoy and rangoby. The
interpretation of this metric is equal to that of the Pearson correlation, that is, when it
is closer to 1, the prediction performance of the implemented statistical learning
method is better. It should be noted that when the number of repeated values in the
observed and predicted values is greater than two, the adjusted range is the sum of
the repeated ranges divided by the number of repeated values; this new range is then
given to the repeated values. In this case, it is also important to regress the observed
versus the predicted values to obtain the intercept and slope using the ranges of the
observed and predicted values.

range
g . Finally,

predicted

ﬁnal

is

is

It is also possible to use the MLL criteria to assess the prediction performance for
count data; however, the new expression is now based on minus the log-likelihood of
a Poisson distribution, which is equal to

138

4 Overﬁtting, Model Tuning, and Evaluation of Prediction Performance

MLL ¼

1
T

XnþT

h
(cid:1)bf xið Þ þ yi log bf xið Þ

(cid:2)

i

i¼nþ1

Again, when the values of MLL are lower, the observed and predicted values are

closer to one another.

References

Brier GW (1950) Veriﬁcation of forecasts expressed in terms of probability. Mon Weather Rev 78:

1–3

Buduma M (2017) Fundamentals of deep learning, 1st edn. O’Reilly, Sabastopol, CA
Burger SV (2018) Introduction to machine learning with R. Rigorous mathematical analysis, 1st

edn. O’Reilly, Sabastopol, CA

Cassella G, Berger RL (2002) Statistical inference. Duxbury, Belmont, CA
Cohen J (1960) A coefﬁcient of agreement for national data. Educ Psychol Meas 20:37–46
Cook D (2017) Practical machine learning with H2O. O’Reilly Media, Inc, Sabastopol, CA
Daetwyler HD, Calus MPL, Pong-Wong R, de los Campos G, Hickey JM (2012) Genomic
prediction in animals and plants: simulation of data, validation, reporting and benchmarking.
Genetics 193:347–365

Dewancker I, McCourt M, Clark S, Hayes P, Johnson A, Ke G (2016) A stratiﬁed analysis of

Bayesian optimization methods. arXiv:1603.09441v1

Efron B (1983) Estimating the error rate of a prediction rule: improvement on cross-validation. J

Am Stat Assoc 78(382):316–331

Fielding AH, Bell JF (1997) A review of methods for the assessment of prediction errors in
conservation presence/absence models. Environ Conserv 24:38–49. https://doi.org/10.1017/
S0376892997000088

González-Camacho JM, Ornella L, Pérez-Rodríguez P, Gianola D, Dreisigacker S, Crossa J (2018)
Applications of machine learning methods to genomic selection in breeding wheat for rust
resistance. Plant Genome 11(2):1–15. https://doi.org/10.3835/plantgenome2017.11.0104
Hagerty MR, Srinivasan S (1991) Comparing the predictive powers of alternative multiple regres-

sion models. Psychometrika 56:77–85. MR1115296

Hastie T, Tibshirani R, Friedman J (2008) The elements of statistical learning: data mining,

inference, and prediction, Springer series in statistics, 2nd edn. Springer, New York

James G, Witten D, Hastie T, Tibshirani R (2013) An introduction to statistical learning: with

applications in R. Springer, New York

Jarquín D, Lemes da Silva C, Gaynor RC, Poland J, Fritz AR et al (2017) Increasing genomic-
enabled prediction accuracy by modeling genotype (cid:4) environment interactions in Kansas
wheat. Plant Genome 10(2):1–15. https://doi.org/10.3835/plantgenome2016.12.0130

Kim S, Kim H (2016) A new metric of absolute percentage error for intermittent demand forecasts.

Int J Forecast 32(3):669–679

Koch P, Wujek B, Golovidov O, Gardner S (2017) Automated hyperparameter tuning for effective
machine learning. In: Proceedings of the SAS global forum 2017 conference. SAS Institute Inc,
Cary, NC. http://support.sas.com/resources/papers/proceedings17/SAS514-2017.pdf

Konen W, Koch P, Flasch O, Bartz-Beielstein T, Friese M, Naujoks B (2011) Tuned data mining: a
benchmark study on different tuners. In: Proceedings of the 13th annual conference on genetic
and evolutionary computation (GECCO-2011). SIGEVO/ACM, New York
Kuhn M, Johnson K (2013) Applied predictive modeling. Springer, New York
Lopez-Cruz M, Crossa J, Bonnett D, Dreisigacker S, Poland J, Jannink J-L, Singh RP, Autrique E,
de los Campos, G. (2015) Increased prediction accuracy in wheat breeding trials using a marker
(cid:4) environment interaction genomic selection method. G3 5(4):569–582

References

139

Lorena AC, de Carvalho ACPLF (2008) Evolutionary tuning of SVM parameter values in

multiclass problems. Neurocomputing 71:3326–3334

Lujan-Moreno GA, Howard PR, Rojas OG, Montgomery DC (2018) Design of experiments and
response surface methodology to tune machine learning hyperparameters, with a random forest
case-study. Expert Syst Appl 109:195–205

Matthews BW (1975) Comparison of the predicted and observed secondary structure of T4 phage

lysozyme. Biochim Biophys Acta Protein Struct 405:442–451

McKay MD (1992) Latin hypercube sampling as a tool in uncertainty analysis of computer models.
In: Swain JJ, Goldsman D, Crain RC, Wilson JR (eds) Proceedings of the 24th conference on
winter simulation (WSC 1992). ACM, New York, pp 557–564

Mesple F, Troussellier M, Casellas C, Legendre P (1996) Evaluation of simple statistical criteria to

qualify a simulation. Ecol Model 88:9–18

Montesinos-López OA, Montesinos-López A, Pérez-Rodríguez P, de los Campos G, Eskridge KM,
Crossa J (2015a) Threshold models for genome-enabled prediction of ordinal categorical traits
in plant breeding. G3 5(1):291–300

Montesinos-López OA, Montesinos-López A, Crossa J, Burgueño J, Eskridge K (2015b) Genomic-
regression. G3

enabled prediction of ordinal data with Bayesian logistic ordinal
5(10):2113–2126. https://doi.org/10.1534/g3.115.021154

Montesinos-López A, Montesinos-López OA, Gianola D, Crossa J, Hernández-Suárez CM (2018a)
Multi-environment genomic prediction of plant traits using deep learners with a dense architec-
ture. G3 8(12):3813–3828. https://doi.org/10.1534/g3.118.200740

Montesinos-López OA, Montesinos-López A, Crossa J, Gianola D, Hernández-Suárez CM et al
(2018b) Multi-trait, multi-environment deep learning modeling for genomic-enabled prediction
of plant traits. G3 8(12):3829–3840. https://doi.org/10.1534/g3.118.200728

Piñeiro G, Perelman S, Guerschman JP, Paruelo JM (2008) How to evaluate models:

observed vs. predicted or predicted vs. observed? Ecol Model 216:316–322

Ratner B (2017) Statistical and machine-learning data mining. Techniques for better predictive
modelling and analysis of big data, 3rd edn. CRC Press Taylor & Francis Group, Boca
Raton, FL

Renukadevi NT, Thangaraj P (2014) Performance analysis of optimization techniques for medical

image retrieval. J Theor Appl Inf Technol 59:390–399

Shalev-Shwartz S, Ben-David S (2014) Understanding machine learning from theory to algorithms.

Cambridge University press, New York

Shmueli G (2010) To explain or to predict? Stat Sci 25(3):289–310
Smith EP, Rose KA (1995) Model goodness-of-ﬁt analysis using regression and related techniques.

Wu S, Harris T, Mcauley K (2007) The use of simpliﬁed or misspeciﬁed models: linear case. Canad

Ecol Model 77:49–64

J Chem Eng 85:386–398

Open Access This chapter is licensed under the terms of the Creative Commons Attribution 4.0
International License (http://creativecommons.org/licenses/by/4.0/), which permits use, sharing,
adaptation, distribution and reproduction in any medium or format, as long as you give appropriate
credit to the original author(s) and the source, provide a link to the Creative Commons license and
indicate if changes were made.

The images or other third party material in this chapter are included in the chapter's Creative
Commons license, unless indicated otherwise in a credit line to the material. If material is not
included in the chapter's Creative Commons license and your intended use is not permitted by
statutory regulation or exceeds the permitted use, you will need to obtain permission directly from
the copyright holder.



---

## Content from Previous Extraction (not in markitdown output)

### Visual/Chart/Graph Descriptions

#### 1

110 4 Overfitting, Model Tuning, and Evaluationof Prediction Performance
Fig.4.1 Schematicillustrationofthreemodelsforclassification:(a)M1 withunderfitting,(b)M2
withappropriatefitting, and (c)M3 withoverfitting
Considerascatteredseriesofpoints (y , x ),...,(y , x ), onaplane, towhichwe
1 1 n n
wanttoadjustastatisticalmachinelearningmethod. Thismeansthatwearelooking
forthebestf (x) thatexplainstheexistingrelationshipbetweentheresponsevariable
i
(y) andthepredictors (x ,..., x ).Weassumethat wehavethree options for f (x):
i 1 n i
M1, the simple model plotted in Fig. 4.1 a; M2, an intermediate model shown in
Fig.4.1 b;and M3, acomplexmodelshownin Fig.4.1 c.
Undertheclassificationframework, thefirstpanelin Fig.4.1(leftside, panela)
shows an unsatisfactory fit (underfitted) since the line does not cover most of the
points (has high bias) in theplot. As such, we expect that theprediction of unseen
dataofthismodel, M1, willperformbadly. Incontrast, panelcof Fig.4.1 showsan
almost perfect fit, since the predicted line covers all the data points. While at first
glance, you may think that model M3 will perform well when predicting unseen
data, thisisactuallyuntruesincethepredictedlinecoversallpointsthatarenoiseand
thosethataresignal (overfit);forthisreason, thistypeofmodelalsoperformspoorly
inthe predictionof future data due toits complexityandhigh variance. Therefore,
the best option for predicting unseen data is model M2 (Fig. 4.1, panel b) since it
represents the predominant (smooth) pattern enough to represent the apparent data
pattern while maintaining a balance between bias and variance. For this reason, a
well-fitted model is one that faithfully represents the sought-after predominant
pattern in the data, while ignoring the idiosyncrasies in the training data. As such,
a well-fitted model in the testing set should be in the neighborhood of the model’s
accuracybasedonthetrainingdataset, thatis, themodel’saccuracyinthetestingset
shouldbeapproximatelyequaltothatofthemodel’saccuracyinthetrainingset. In
contrast, anoverfittedmodelinthetestingdatasetwillbefarfromtheneighborhood
of the model’s accuracy based on the training data set, and usually its prediction
performanceisveryhigh (good) inthetrainingsetandconsequentlylow (bad) inthe
testingset (Ratner2017).
The paradox of overfitting is defined as complex models that contain more
information about the training data, but less information about the testing data
(future data we want to predict). In statistical machine learning, overfitting is a
major issue andleads tosome seriousproblemsin research: (a) some relationships

#### 2

112 4 Overfitting, Model Tuning, and Evaluationof Prediction Performance
Fig.4.2 Graphical
representationsofdifferent
levelsofbiasandvariance
the population is high-income and the other half is low-income. If you collected a
sample of high-income people, you would conclude that the entire city has high
income. Thismeansthatyourconclusionisheavilybiasedsinceyouonlysampled
peoplewithhighincome. Ontheotherhand, errorvariancereferstotheamountthat
theestimateoftheobjectivefunctionwillchangeusingadifferenttrainingdataset.
Inotherwords, theerrorvarianceaccountsforthedeviationofpredictionsfromone
repetitiontoanotherusingthesametrainingset. Ideally, whenastatisticalmachine
learningmodelwithlowerrorvariancepredictsavalue, thepredictedvalueshould
remainalmostthesame, evenwhenchangingfromonetrainingdatasettoanother;
however, ifthemodelhashighvariance, thenthepredictedvaluesofthestatistical
machine learning method are affected by the values of the data set. We provide a
graphicalvisualizationofbiasandvariancewithabull’seyediagram (see Fig.4.2).
We assume that the center ofthe target isa statistical machine learning model that
perfectly predicts the correct answers. As we move away from the bull’s eye, our
predictionsgetworse. Letusassumethatwecanrepeatourentirestatisticalmachine
learningmodelbuildingprocesstogetanumberofseparatehitsonthetarget. Each
hit represents an individual realization of our statistical machine learning model,
given the chance variability in the training data we gathered. Sometimes we accu-
ratelypredicttheobservationsofinterestsincewecapturedarepresentativesample
in our training data, while other times we obtain unreliable predictions since our
trainingdatamaybefullofoutliersornonrepresentativevalues. Thefourcombina-
tions of cases resulting from both high and low bias and variance are shown in
Fig.4.2.

#### 3

4.3 Cross-validation 117
It is important to point out that to reduce variability, we recommend
implementing the k-fold CV multiple times, each time using different complemen-
tary subsets to form the folds; the validation results are combined (e.g., averaged)
over the rounds (times) to give a better estimate of the statistical machine learning
modelpredictiveperformance.
4.3.3 The Leave-One-Out Cross-validation
The Leave-One-Out (or LOO) CV is very simple since each training data set is
created by including all the individuals except one, while the testing set only
includes the excluded individual. Thus, for n individuals in the full data set, we
havendifferent training andtesting sets. This CVschemewastesminimaldata, as
onlyoneindividualisremovedfromthetrainingset.
Regardingthek-foldcross-validationthatwasjustexplained, nmodelsarebuilt
from n individuals (samples) instead of k models, where n > k. Moreover, each
model is trained on n (cid:1) 1 samples rather than (k (cid:1) 1) n/k. In both cases, since k is
normally not too large and k < n, LOO is more computationally expensive than k-
fold cross-validation. In terms of prediction performance, LOO normally produces
highvariancefortheestimationofthetesterror. Becausen (cid:1)1 ofthensamplesis
usedtobuildeachstatistical machine learningmodel, thoseconstructedfromfolds
arevirtuallyidenticaltoeachotherandtothemodelbuiltfromtheentiretrainingset.
However, whenthelearningcurveissteepfortheevaluatedtrainingsize, thenfive-
orten-foldcross-validationusuallyoverestimatesthegeneralizationerror.
Learning curves (LC) areconsidered effective tools tomonitor theperformance
oftheemployeeexposedtoanewtask. LCsprovideamathematicalrepresentation
ofthelearningprocessthattakesplaceasthetaskisrepeated. Instatisticalmachine
learningthe LCisalineplotoflearning (y-axis) overexperience (x-axis).Learning
curvesareextensivelyusedinstatistical machinelearningforalgorithmsthatlearn
(their parameters) incrementally over time, such as deep neural networks. In
general, there is considerable empirical evidence suggesting that five- or ten-fold
cross-validationshouldbepreferredto LOO.
4.3.4 The Leave-m-Out Cross-validation
The Leave-m-Out (Lm O) CV is very similar to LOO as it creates all the possible
training/te stse!tsbyremovingmsamplesfromthecompleteset. Fornsamples, this
n
produces train-testpairs. Unlike LOOandk-fold, thetestsetswilloverlapfor
m
m>1.Forexample, ina Leave-2-Out CVwithadatasetwithfoursamples (I1, I2,

#### 4

120 4 Overfitting, Model Tuning, and Evaluationof Prediction Performance
Fig.4.6 Schematicrepresentationofbootstrapcross-validation
this CVapproachhasabiassimilartoimplementingak¼twofoldcross-validation,
and as the training set becomes smaller, the bias becomes more problematic. To
understand this CV method, we provide a simple example of how the training and
testingsamplesareconstructed. Ifwehaveasamplewith12 individualsdenotedas
I1, I2, ..., I12, we will select B ¼ 5 bootstrap samples. Each bootstrap sample is
obtainedwithreplacementandtheindividualsthatappearineachonecorrespondto
the training sample; those that are not present will correspond to the testing set.
Figure 4.6 provides the five bootstrap samples; each training sample has the same
sizeastheoriginal, however, onlysomeindividualsappearineachbootstrapsample,
while those individuals that do not appear are included in the testing set. For
example, in the first fold, the training bootstrap sample contains seven different
individuals (I2, I3, I4, I6, I7, I8, and I9), while the testing set contains five
individuals (I1, I5, I10, I11, and I12). It is important to point out that since the
training sample has the same size as the original sample, some individuals in the
training sample are repeated at least twice; in the first fold, the individual I4 are
repeated three times, whereas I6, I7, and I8 are repeated twice. Finally, similar to
other methods, the statistical machine learning model is trained with each training
set; likewise, the prediction performance of the model is evaluated in each testing
set. Theaverageofthesesamplepredictionsisreportedastheestimatedtestingerror.
4.3.8 Incomplete Block Cross-validation
Incompleteblock (IB)CVshouldbeusedwhenthereare Jtreatmentsevaluatedin
Iblocksandthesametreatmentsareevaluatedinalltheblocks. Theideabehindthis

#### 5

4.5 Metricsforthe Evaluationof Prediction Performance 129
4.5.1 Quantitative Measures of Prediction Performance
Beforeimplementingastatisticalmachinelearningmodel, weassumethatwehave
ourtrainingobservation{(x , y ),(x , y ),...,(x , y )} andweestimatef, as b f, with
1 1 2 2 n n
the chosen statistical machine learning model. Then we can make predictions for
b
eachoftheresponsevalues (y) withfðxÞandcomputethepredictedvaluesforeach
i i
ofthenobservationsinthetrainingset;withthesevalu (cid:6) eswecancalculatethemean
(cid:2) (cid:3)
Pn
b 2
square error (MSE) for thetraining dataset as E ¼1 y (cid:1) fðxÞ ;however,
n i i
i¼1
whatwereallywanttopredictarethevaluesforunseentestobservationsthatwere
not used to train the statistical machine learning model. Assuming that the unseen
testingsetisequalto{(x , y ),(x , y ),...,(x , y )}, the MSEfor
n+1 n+1 n+2 n+2 n+T n+T
thetestingdatasetshouldbecalculatedas
MSE ¼ 1
XnþT (cid:2)
y (cid:1) b fðxÞ
(cid:3)
2 , ð4:1Þ
TST T i i
i¼nþ1
b b
wherefðxÞisthepredictionthatf givestotheithobservation. The MSE witha
i TST
lowervaluewillhavebetterpredictions, whichmeansthatthepredictedvaluesare
veryclosetothetrueobservedvalues. Also, thesquarerootof MSE canbeused
TST
as a measure of prediction performance and is called root mean square error
(RMSE).
Pearson’scorrelationcoefficientisaverypopularmeasureofpredictionperfor-
manceinplantbreedingandcanbecalculatedas
(cid:2) (cid:3)
P
nþT b fðxÞ(cid:1) b fðxÞ ðy (cid:1) yÞ
r TST ¼rffi P ffiffiffiffiffi n ffi þ ffiffiffi T ffiffiffiffiffii (cid:2) ffi¼ffiffi b f nffiffiþ ð ffiffi x ffi1 ffiffi Þ ffiffiffi (cid:1) ffiffiffiffiffi b f i ffiffi ð ffiffi x ffiffiffi Þ ffiffi (cid:3) ffiffiffi 2 ffiffi q i ffi P ffiffiffiffiffi n ffi i þ ffiffiffi T ffiffiffiffiffi ð ffi i ffi y ffiffiffiffi (cid:1) ffiffiffiffiffi y ffiffiffi Þ ffiffi 2 ffiffi, ð4:2Þ
i¼nþ1 i i i¼nþ1 i i
b
wherefðxÞistheaverageofthe Tpredictionsthatconformtothetestingset, andy is
i i
theaverageofthe Ttrueobservedvalues. Inthiscase, thecloserthatthepredictions
areto1, thebettertheimplementedstatisticalmachinelearningmodelwillperform.
It is important to point out that Pearson’s correlation is defined between (cid:1)1 and
1. However, to be convinced that the observed and predicted values match, it is a
common practice to perform a scatter plot of predicted versus observed (or vice
versa) valuesandwhentheobservedandpredictedvaluesfollowastraightline (45(cid:6)
diagonal line) from the bottom left corner to the top right corner, this indicates a
perfectmatchbetweentheobservedandpredictedvalues. Forthisreason, Pearson’s
correlationasametricshouldbecomplemented with theinterceptandslope, since
the slope and intercept describe the consistency and the model bias, respectively
(Smith and Rose 1995; Mesple et al. 1996). To obtain the slope and intercept, the
observedvalues (asy) versusthepredictedvalues (asx) areregressedandinaddition

#### 6

4.5 Metricsforthe Evaluationof Prediction Performance 133
Table4.4 Confusionmatrix Predictedvalues
withtwoclasses
True False Sum
Observedvalues True tp fn tp+fn
False fp tn fp+tn
Sum tp+fp fn+tn n
tp denotes true positives, fp denotes false positives, fn denotes
false negatives, tn denotes true negatives, and n denotes total
numberofindividuals
example, ifthereare990 samplesofclass1 andonly10 ofclass2, theclassifiercan
easilyhaveabiastowardclass1.Iftheclassifierclassifiesallsamplesasclass1, its
accuracywillbe99%.Thisdoesnotmeanthatitisanappropriateclassifier, asithad
a100%errorwhenclassifyingthesamplesofclass2.Forthisreason, reportingthis
metric with those reported in Eqs. (4.9–4.11) is recommended in order to have a
better picture of the prediction performance of any statistical machine learning
method (Ratner 2017). Also, it is important to highlight that when the problem
onlyhastwoclasses, theconfusionmatrixisreducedto Table4.4.
From Table4.4 the PCCCiscalculatedastpþtn, whilethe S ¼ tp , S ¼ tn ,
n e tpþfn p tnþfp
and P¼ tp . Also, when there are only two classes, González-Camacho et al.
tpþfp
(2018) suggest calculating the Kappa coefficient (κ) or Cohen’s Kappa, which is
definedas
P (cid:1)P
κ ¼ 0 e,
1(cid:1)P
e
where P istheagreementbetweenobservedandpredictedvaluesandiscomputed
0
by the PCCC described above for two classes; P is the probability of agreement
e
calculated as P ¼tpþfn (cid:4) tpþfpþfpþtn (cid:4) fnþtn, where fp is the number of false
e n n n n
positives, and fn is the number of false negatives (Table 4.4). This statistic can
takeonvaluesbetween (cid:1)1 and1;avalueof0 meansthereisnoagreementbetween
the observed and predicted classes, while a value of 1 indicates perfect agreement
betweenthemodelpredictionandtheobservedclasses. Negativevaluesindicatethat
a prediction may be incorrect; however large negative values seldom occur when
workingwithpredictivemodels. Dependingonthecontext, a Kappavaluefrom0.30
to 0.50 indicates reasonable agreement (Kuhn and Johnson 2013). The Kappa
coefficient is appropriate when data are unbalanced, because it estimates the pro-
portion of cases that were correctly identified by taking into account coincidences
expectedfromchancealone (Fieldingand Bell1997).Itisimportanttopointoutthat
this statistic was originally designed to assess the agreement between two raters
(Cohen1960).
Anotherpopularmetricforbinarydataisthe Area Underthereceiveroperating
characteristic Curve (AUC–ROC) anditranksthepositivepredictionshigherthan
thenegative. The ROCcurveisdefinedasaplotof1(cid:1) specificityorfalsepositive
rate (FPR) asthex-axisversusitsmodelsensitivityasthey-axis. Foragivensetof

#### 7

134 4 Overfitting, Model Tuning, and Evaluationof Prediction Performance
Fig.4.10 ROCcurvefor
thesyntheticdata
thresholds τ, it is an effective method for evaluating the quality or performance of
diagnostic tests, and is widely used in statistical machine learning to evaluate the
predictionperformanceoflearningalgorithms. Sincethemathematicalconstruction
ofthismetricisnotrequiredinthisbook, weillustrateitscalculationwithonesimple
example. Assume that the observed (y), predicted probabilities (pi) and predicted
values (ŷ) obtained after implementing a statistical machine learning model are
y ¼ {1, 0, 1, 1, 0, 0, 1, 1, 0, 1}, pi ¼ {0.6, 0.55, 0.8, 0.78, 0.3, 0.42, 0.9, 0.45,
0.3,0.88}, andby¼f1,1,1,1,0,0,1,0,0,1};then, usingthefollowing Rcode, we
canobtainmanymetricsforbinarydata (Fig.4.10):
########################Librariesrequired########################
library (caret)
library (p ROC)
##Observed (y), predictedprobability (pi) and predictedvalues of
syntheticdata##
y=c (1,0,1,1,0,0,1,1,0,1)
pi=c (0.6,0.55,0.8,0.78,0.3,0.42,0.9,0.45,0.3,0.88)
yhat=c (1,1,1,1,0,0,1,0,0,1)
xtab<-table (y, yhat)
confusion Matrix (xtab)
plot.roc (y, pi)#####Thismakethe ROCcurveplot
Confusion Matrixand Statistics
yhat
y 01
031
115

#### 8

4.5 Metricsforthe Evaluationof Prediction Performance 135
Accuracy:0.8
95%CI:(0.4439,0.9748)
No Information Rate:0.6
P-Value[Acc>NIR]:0.1673
Kappa:0.5833
Mcnemar's Test P-Value:1.0000
Sensitivity:0.7500
Specificity:0.8333
Pos Pred Value:0.7500
Neg Pred Value:0.8333
Prevalence:0.4000
Detection Rate:0.3000
Detection Prevalence:0.4000
Balanced Accuracy:0.7917
'Positive'Class:0
It is important to point out that when there are more than two classes, these
metrics (accuracy, sensitivity, specificity, etc.) can be calculated on a “one-versus-
all”basisthatconsistsofusingeachclassversusthepooloftheremainingclasses, as
was illustrated for the confusion matrix with more than two classes (James et al.
2013).
Whenthestatisticalmachinelearningmodeldiscriminatescorrectlybetweenthe
twogroups, itproducesacurvethatcoincideswiththeleftandtopsidesoftheplot.
Underthisscenario, theperfectmodelwouldhave100%sensitivityandspecificity,
andthe ROCcurvewouldbeasinglestepbetween (0,0) and (0,1) andwouldremain
constant from (0,1) to (1,1); this implies that the area under the ROC curve of the
modelwouldbe1.Ingeneral, thelargertheareaunderthe ROCcurve, thebetterthe
modelintermsofpredictionperformance. Ontheotherhand, acompletelyuseless
statisticalmachinelearningalgorithmwouldgiveastraightline (45(cid:6) diagonalline)
from the bottom left corner to the top right corner of the plot. Different statistical
machine learning models or the same model with different training sets or
hyperparameterscanbecomparedbysuperimposingtheir ROCcurvesinthesame
graph. Inpractice, mostofthetimethevaluesinthetwogroupsoverlap, sothecurve
oftenliesbetweentheseextremes.
From this ROC curve, we can obtain a global assessment of the prediction
performance of the statistical machine learning method by measuring the area
underthereceiveroperatingcharacteristiccurve. Thisareaisequaltotheprobability
thatarandomindividual (person) ofthesamplewiththepresenceofthetargethasa
highervalueofthemeasurementthanarandomindividualwithoutthetarget.
Whenastatisticalmachinelearningmodelisunabletodiscriminatebetweenthe
positive and negative classes, this means that it has low discriminatory power.
Therefore, onlyinstatisticalmachinelearningmodelsthathadgooddiscriminatory
power we can be confident of the predictions they provide and furthermore those
modelsthatprovideacurvethatliesconsiderablyabovethecurvewillbebetter.

#### 9

References 139
Lorena AC, de Carvalho ACPLF (2008) Evolutionary tuning of SVM parameter values in
multiclassproblems. Neurocomputing71:3326–3334
Lujan-Moreno GA, Howard PR, Rojas OG, Montgomery DC(2018)Designofexperimentsand
responsesurfacemethodologytotunemachinelearninghyperparameters, witharandomforest
case-study. Expert Syst Appl109:195–205
Matthews BW(1975)Comparisonofthepredictedandobservedsecondarystructureof T4 phage
lysozyme. Biochim Biophys Acta Protein Struct405:442–451
Mc Kay MD(1992)Latinhypercubesamplingasatoolinuncertaintyanalysisofcomputermodels.
In:Swain JJ, Goldsman D, Crain RC, Wilson JR(eds)Proceedingsofthe24 thconferenceon
wintersimulation (WSC1992).ACM, New York, pp557–564
Mesple F, Troussellier M, Casellas C, Legendre P(1996)Evaluationofsimplestatisticalcriteriato
qualifyasimulation. Ecol Model88:9–18
Montesinos-López OA, Montesinos-López A, Pérez-Rodríguez P, delos Campos G, Eskridge KM,
Crossa J(2015 a)Thresholdmodelsforgenome-enabledpredictionofordinalcategoricaltraits
inplantbreeding. G35(1):291–300
Montesinos-López OA, Montesinos-López A, Crossa J, Burgueño J, Eskridge K(2015 b)Genomic-
enabled prediction of ordinal data with Bayesian logistic ordinal regression. G3
5(10):2113–2126.https://doi.org/10.1534/g3.115.021154
Montesinos-López A, Montesinos-López OA, Gianola D, Crossa J, Hernández-Suárez CM(2018 a)
Multi-environmentgenomicpredictionofplanttraitsusingdeeplearnerswithadensearchitec-
ture. G38(12):3813–3828.https://doi.org/10.1534/g3.118.200740
Montesinos-López OA, Montesinos-López A, Crossa J, Gianola D, Hernández-Suárez CM et al
(2018 b)Multi-trait, multi-environmentdeeplearningmodelingforgenomic-enabledprediction
ofplanttraits. G38(12):3829–3840.https://doi.org/10.1534/g3.118.200728
Piñeiro G, Perelman S, Guerschman JP, Paruelo JM (2008) How to evaluate models:
observedvs.predictedorpredictedvs.observed?Ecol Model216:316–322
Ratner B (2017) Statistical and machine-learning data mining. Techniques for better predictive
modelling and analysis of big data, 3 rd edn. CRC Press Taylor & Francis Group, Boca
Raton, FL
Renukadevi NT, Thangaraj P(2014)Performanceanalysisofoptimizationtechniquesformedical
imageretrieval. JTheor Appl Inf Technol59:390–399
Shalev-Shwartz S, Ben-David S(2014)Understandingmachinelearningfromtheorytoalgorithms.
Cambridge Universitypress, New York
Shmueli G(2010)Toexplainortopredict?Stat Sci25(3):289–310
Smith EP, Rose KA(1995)Modelgoodness-of-fitanalysisusingregressionandrelatedtechniques.
Ecol Model77:49–64
Wu S, Harris T, Mcauley K(2007)Theuseofsimplifiedormisspecifiedmodels:linearcase. Canad
JChem Eng85:386–398
Open Access Thischapterislicensedunderthetermsofthe Creative Commons Attribution4.0
International License (http://creativecommons.org/licenses/by/4.0/), which permits use, sharing,
adaptation, distributionandreproductioninanymediumorformat, aslongasyougiveappropriate
credittotheoriginalauthor (s) andthesource, providealinktothe Creative Commonslicenseand
indicateifchangesweremade.
Theimagesorotherthirdpartymaterialinthischapterareincludedinthechapter's Creative
Commons license, unless indicated otherwise in a credit line to the material. If material is not
included in the chapter's Creative Commons license and your intended use is not permitted by
statutoryregulationorexceedsthepermitteduse, youwillneedtoobtainpermissiondirectlyfrom
thecopyrightholder.

### Additional Content

#### 1

Chapter 4
fi
Over tting, Model Tuning, and Evaluation
of Prediction Performance
4.1 The Problem of Overfitting and Underfitting
The overfitting phenomenon occurs when the statistical machine learning model
learns the training data set so well that it performs poorly on unseen data sets. In
otherwords, thismeansthatthepredictedvaluesmatchthetrueobservedvaluesin
the training data set too well, causing what is known as overfitting. Overfitting
happens when a statistical machine learning model learns the systematic and noise
(randomfluctuations) partsinthetrainingdatatotheextentthatitnegativelyimpacts
theperformanceofthestatisticalmachinelearningmodelonnewdata. Thismeans
thatthestatisticalmachinelearningmodeladaptsverywelltothenoiseaswellasto
thesignalthatispresentinthetrainingdata. Theproblemisthattheseconceptsdo
not apply to independent (new) data and negatively affect the model’s ability to
generalize. Overfitting is more probable when learning a loss function from a
complexstatisticalmachinelearningmodel (withmoreflexibility).Forthisreason,
manynonparametricstatisticalmachinelearningmodelsalsoincludeconstraintsin
thelossfunctiontoimprovethelearningprocessofthestatisticalmachinelearning
models. For example, artificial neural networks (ANN), mentioned later, are a
nonparametricstatisticalmachinelearningmodelthatisveryflexibleandissubject
tooverfittingtrainingdata. Thisproblemcanbeaddressedbydroppingout (setting
to zero) the weights of a certain percentage of hidden units in order to avoid
overfitting.
On the other hand, an underfitted phenomenon occurs when few predictors are
includedinthestatisticalmachinelearningmodel, i.e., itisaverysimplemodelthat
poorlyrepresentsthecompletepictureofthepredominantdatapattern. Thisproblem
also arises when the training data set is too small or not representative of the
population data. An underfitted model does a poor job of fitting the training data
and for this reason it is not expected to satisfactorily predict new data points. This
implies that the predictions using unseen data are weak, since individuals are
perceivedasstrangersunfamiliarwiththetrainingdataset.
©The Author (s)2022 109
O.A.Montesinos Lópezetal., Multivariate Statistical Machine Learning Methods
for Genomic Prediction, https://doi.org/10.1007/978-3-030-89010-0_4

#### 2

4.2 The Trade-Off Between Prediction Accuracyand Model Interpretability 111
thatseemstatisticallysignificantareonlynoise,(b) thecomplexityofthestatistical
machine learning model is very large for the amount of data provided, and (c) the
modelingeneralisnotreplicableandpredictspoorly.
Sincethemaingoalofdevelopingandimplementingstatisticalmachinelearning
methods is to predict unseen data not used for training the statistical machine
learning algorithm, researchers are mainly interested in minimizing the testing
error (generalization error applicable to future samples) instead of minimizing the
training error that isapplicabletotheobserved data used for training thestatistical
machinelearningalgorithm.
According to Shalev-Shwartz and Ben-David (2014), if the learning fails, these
aresomeapproachestofollow:
1. Increasethesamplesizeofthetrainingset.
2. Modifythehypothesisby (a) enlargingit,(b) reducingit,(c) completelychanging
it, and (d) changingtheparametersbeingused. Weunderstandahypothesisasthe
models and their parameters under evaluation. This point is very important to
reachareasonablemodelforyourdata.
3. Changethefeaturerepresentationofthedata.
4. Changethestatisticalmachinelearningalgorithmused.
4.2 The Trade-Off Between Prediction Accuracyand Model
Interpretability
Accuracy is the ability of a statistical machine learning model to make correct
predictions and those models with more complexity (called flexible models) are
betterintermsofaccuracy, whilethesimple, lesscomplexmodels (calledinflexible
models) are less accurate but more interpretable. Interpretability indicates to what
degreethemodelallowsforhumanunderstandingofnaturalphenomena. Forthese
reasons, when the goal of the study is prediction, flexible models should be used;
however, when the goal of the study is inference, inflexible models are more
appropriatebecausetheymoreeasilyinterprettherelationshipbetweentheresponse
variables and the predictor variables. As the complexity of the statistical machine
learning model increases, the bias is reduced and the variance increases. For this
reason, whenmoreparametersareincludedinthestatisticalmachinelearningmodel,
the complexity of the model increases and the variance becomes the main concern
while the bias steadily falls. For example, James et al. (2013) state that the linear
regression model is a relatively inflexible method because it only generates linear
functions, while the support vector machine method is one of the most flexible
statisticalmachinelearningmethods.
Beforeprovidingananalyticalinterpretationofthetrade-offbetweenthebiasand
variance, we must understand the meaning of both concepts. Bias is the difference
between the expected prediction of our statistical machine learning model and the
trueobservedvalues. Forexample, assumethatyoupollaspecificcitywherehalfof

#### 3

4.2 The Trade-Off Between Prediction Accuracyand Model Interpretability 113
Burger (2018) concludesthatthebestscenarioistheonewithlowbiasandlow
variance, since samples are acceptable representatives ofthe population (Fig. 4.2),
whileinthecaseofhighbiasandlowvariance, thesamplesarefairlyconsistent, but
notparticularlyrepresentative ofthepopulation (Fig.4.2).However, whenthereis
lowbiasandhighvariance, thesamplesvarywidelyintheirconsistency, andonly
somemayberepresentativeofthepopulation (Fig.4.2).Finally, withhighbiasand
high variance, the samples are somewhat consistent, but unlikely to be representa-
tiveofthepopulation (Fig.4.2).
Ifwedenotethevariablewearetryingtopredictasyandourcovariatesasx, we
i
may assumethat there isarelationship between y andx, asthatgiven in Eq.(1.1)
i
from Chap.1, wheretheerrortermisnormallydistributedwithameanofzeroand
varianceσ2.Theexpectedpredictionerrorforanewobservationwithvaluex, using
aquadraticlossfunction, isgivenby Hastieetal.(2008, page223):
(cid:2) (cid:3) (cid:2) (cid:2) (cid:3) (cid:3) n (cid:2) (cid:3) o
E y (cid:1) b ffxg 2 ¼Eðy (cid:1) ffxgÞ2þ E b ffxg (cid:1) ffxg 2 þE b fðxÞ(cid:1)E b ffxg 2
i i i i i i
h (cid:2) (cid:3) i (cid:2) (cid:3)
b 2 b
¼VarðyÞþ Bias ffxg þVar fðxÞ
i i
h (cid:2) (cid:3) i (cid:2) (cid:3)
¼VarðEÞþ Bias b ffxg 2 þVar b fðxÞ ,
i i
where Biasistheresultofmisspecifyingthestatisticalmodelf. Estimationvariance
(thethirdterm) istheresultofusingasampletoestimatef. Thefirsttermistheerror
(irreducibleerror) thatresultsevenifthemodeliscorrectlyspecifiedandaccurately
estimated. Thisirreducibleerroristhenoiseterminthetruerelationshipthatcannot
fundamentally be reduced by any model. Given the true model and infinite data to
train (calibrate) it, we should be able to reduce both thebias and variancetermsto
0. However, in a world with imperfect models and finite data, there is a trade-off
betweenminimizingthebiasandminimizingthevariance. Theabovedecomposition
revealsasourceofthedifferencebetweenexplanatoryandpredictivemodeling:In
explanatory modeling, the focus is on minimizing bias to obtain the most accurate
representation of the underlying theory. In contrast, predictive modeling seeks to
minimize thecombination of bias and estimation variance, occasionally sacrificing
theoreticalaccuracyforimprovedempiricalprecision (Shmueli2010).
These four aspects impact every step of the modeling process, such that the
resultingfismarkedlydifferentintheexplanatoryandpredictivecontexts.
Let us assume that f is a reasonable operationalization of the true function (F)
relating constructs X and Y. Choosing a function f (cid:3) that is intentionally biased in
place of f is very undesirable from a theoretical–explanatory standpoint. However,
theelectionoff (cid:3) isdesirabletofunderthepredictionapproach. Weshowthisusing
the statistical model y ¼ β x + β x + β x + E, which is assumed to be correctly
1 1 2 2 3 3
specifiedwithrespectto F.Usingdata, weobtaintheestimatedmodel b f, whichhas
thefollowingproperties:

#### 4

114 4 Overfitting, Model Tuning, and Evaluationof Prediction Performance
Bias¼0
(cid:2) (cid:3) (cid:2) (cid:3) (cid:4) (cid:5)
Var b fðxÞ ¼Var bβ x þ bβ x þ bβ x ¼σ2 x T XXT 21 x,
i 1 1 2 2 3 3
where x is the vector x ¼ [x , x , x ]T and X is the design matrix based on all
1 2 3
predictors. Combining the squared bias with the variance gives, as expected, the
predictionerror (EPE).
(cid:2) (cid:3) (cid:4) (cid:5) h (cid:4) (cid:5) i
E y (cid:1) b ffxg 2 ¼σ2þ0þσ2 x T XXT 21 x=σ2 1þx T XXT (cid:1)1 x :
i
Incomparison, considertheestimatedunderspecifiedform b f (cid:3) ðxÞ¼x bγ.Thebias
i 1
andvariancehereare
(cid:2) (cid:3) (cid:4) (cid:5)
Bias¼E b ffxg (cid:1) ffxg¼x x x T (cid:1)1 x Tðβ x þβ x þβ x Þ
i i 1 1 1 1 1 1 2 2 3 3
(cid:1)ðβ x þβ x þβ x Þ
1 1 2 2 3 3
(cid:2) (cid:3) (cid:4) (cid:5)
Var b fðxÞ ¼σ2 x x x T (cid:1)1 x T
i 1 1 1 1
Combiningthesquaredbiaswiththevariance EPEisequalto
(cid:2) (cid:3) h (cid:4) (cid:5) i
E y (cid:1) b ffxg 2 ¼ x x x T (cid:1)1 x Tðx β þx β Þ(cid:1)ðx β þx β Þ 2
i 1 1 1 1 2 2 3 3 2 2 3 3
h (cid:4) (cid:5) i
þσ2 1þx x x T (cid:1)1 x T :
1 1 1 1
Althoughthebiasoftheunderspecifiedmodelf (cid:3)(x) islargerthanthatoff{x}, its
i i
variance can be smaller, and in some cases, so small that the overall EPE will be
lowerfortheunderspecifiedmodel. Wuetal.(2007) showedthegeneralresultforan
underspecified linear regression model with multiple predictors. In particular, they
showedthattheunderspecifiedmodelthatleavesoutqpredictorshasalower EPE
whenthefollowinginequalityholds:
qσ2 >βTXTðI(cid:1)H ÞX β
2 2 1 2 2
This means thatthe underspecifiedmodel producesmoreaccuratepredictions, in
termsoflower EPE, inthefollowingsituations:(a) whenthedataareverynoisy (large
σ2);(b) whenthetrueabsolutevaluesoftheexcludedparameters (inourexample,β
2
and β ) are small; (c) when the predictors are highly correlated; and (d) when the
3
samplesizeissmallortherangeofleft-outvariablesissmall (Shmueli2010).
Hagertyand Srinivasan (1991) nicelysummarizethissituation:“Wenotethatthe
practice in applied research of concluding that a model with a higher predictive
validityis“truer,”isnotavalidinference. Thispapershowsthataparsimoniousbut
less true model can have a higher predictive validity than a truer but less parsimo-
niousmodel.”

#### 5

116 4 Overfitting, Model Tuning, and Evaluationof Prediction Performance
set (testing set) is used to study how well that statistical machine learning model
performsonunseendata. Forexample,80%ofthedatacanbeusedfortrainingthe
modelandtheremaining20%ofthedatafortestingit. Oneweaknessofthehold-out
(validation) set approach is that it depends on just one training-testing split and its
performancedependsonhowthedataaresplitintothetrainingandtestingsets.
4.3.2 The k-Fold Cross-validation
Ink-fold CV, thedatasetisrandomlydividedintokcomplementaryfolds (groups)
of approximately equal size. One of the subsets is used as testing data and the rest
(k (cid:1)1) astrainingdata. Thenk (cid:1)1 foldsareusedfortrainingthestatisticalmachine
learning model and the remaining fold for evaluating the out-of-sample prediction
performance. For these reasons, the statistical machine learning model is fitted
k times using a different partition (fold) as the testing set and the remaining k (cid:1) 1
asthetrainingset. Finally, thearithmeticmeanofthekfoldsisobtainedandreported
asthepredictionperformanceofthestatisticalmachinelearningmodel (see Fig.4.4).
This method is very accurate because it combines k measures of fitness resulting
fromthektrainingandtestingdatasetsintowhichtheoriginaldatasetwasdivided,
butatthecostofmorecomputationalresources. Inpractice, thechoiceofthenumber
offoldsdependsonthemeasurementofthedataset, although5 or10 foldsarethe
mostcommonchoices.
Fig.4.4 Schematicrepresentationofthek-foldcross-validationwithcomplementarysubsetswith
k¼5

#### 6

118 4 Overfitting, Model Tuning, and Evaluationof Prediction Performance
!
4
I3, and I4), thetotalnumberoftraining-testingsetsisequalto ¼6;thismeans
2
thatthesixtestingsetsare:[I3, I4][I1, I2],[I2, I4][I1, I3],[I2, I3][I1, I4],[I1, I4]
[I2, I3],[I1, I3] [I2, I4], and [I1, I2] [I3, I4], while the training sets are the comple-
mentaryelementsofeachtestingset.
4.3.5 Random Cross-validation
In this type of CV, the number of partitions (independent training-testing data set
splits) is defined by the user, and more partitions are better. Each partition is
generated by randomly dividing the whole data set into two subsets: the training
(TRN) datasetandthetesting (TST) dataset. Thepercentageofthewholedataset
assigned to the TRN and TST data sets is also fixed by the user. For example, for
each random partition, the user can decide that 80% of the whole data set can be
assigned tothe TRN data set and the remaining 20% to the TST data set. Random
cross-validation is different from k-fold cross-validation because the partitions are
notmutuallyexclusive;thismeansthatintherandomcross-validationapproach, one
observation can appear in more than one partition. Consequently, some samples
cannotbeevaluated, whereasotherscanbeevaluatedmorethanonce, meaningthat
the testing and training subsets can be superimposed (Montesinos-López et al.
2018 a, b). To control the randomness for reproducibility, we recommend using a
specific seed in the random number generator. We recommend using at least ten
random partitions to obtain enough accuracy in the estimate of prediction
performance.
4.3.6 The Leave-One-Group-Out Cross-validation
The Leave-One-Group-Out (LOGO) CV is useful when individuals are grouped
(inenvironmentsoryears, orevenanothercriterion), wherethenumberofgroups (g)
isatleasttwoandtheinformationofg (cid:1)1 groupsareusedasthetrainingsetwhile
allindividualsoftheremaininggroupareusedasthetestingset. Forexample, inthe
contextofgenomicselection, whentheplantbreederisinterestedinpredictingthese
linesinanotherenvironment, thesame (ordifferent) lineswerefrequentlyevaluated
ing environments oryears, that representthe groups. Jarquínet al.(2017) denotes
this type of CV strategy as CV1 in the context of plant breeding. Under this
approach, thepredictionsarereportedforeachoftheggroupsbecausethescientist
is interested in the prediction performance of each environment. Many times, the
groups are the years under study and the aim is to predict the information of a
completeyear. However, whenthegroupsareyearsandifwesuspectthatthereisa
considerablecorrelationbetweenobservationsthatarenearintime. Therefore, itis

#### 7

4.3 Cross-validation 121
Table4.1 TRNdatasetfor Block Treatmentsperblock
I¼3 blocksand J¼10
1 1 3 5 7 8 9 10
treatmentswith70%ofdata
fortraining 2 2 3 4 5 6 7 9
3 1 2 4 6 7 8 10
CVmethodisthatsometreatmentsshouldbepresentinsomeblocksbutabsentin
others, whereas the same treatment should be present in at least one environment
(block). The theory of incomplete block designs developed in the experimental
designs statistical area can be used to construct the training set. For example,
under a balanced incomplete block (BIB) design, the term incomplete means that
alltreatmentsineachblockcannotbeevaluated, whereasbalancedmeansthateach
pair of treatments occur together λ times. The training set is constructed by first
defining the % of individuals in the TRN set using the equation s I ¼ Jr ¼ N ,
TRN
where Jrepresentsthenumberoftreatmentsunderstudy, Irepresentsthenumberof
blocks under study, r denotes the number of repetitions of each treatment, and s
denotes the treatments per block. For example, suppose that we had J ¼ 10 treat-
mentsand I¼3 blocks (thatis,30 individuals), andwedecidedtouse N ¼21
TRN
(70%) ofthetotalindividualsinthe TRNset. Therefore, thenumberoftreatmentsby
blockcanbeobtainedbysolving (s I¼N ) fors, whichresultsins¼N /I.This
TRN TRN
meansthats¼21/3¼7 treatmentsperblock. Then, thecorrespondingelementsfor
thetrainingsetcanbeobtainedwiththefunctionfind. BIB(10,3,7) ofthepackage
crossdes of the R statistical software. The numbers used in the function find. BIB()
denotethetreatments, theblocks, andthetreatmentsperblock, respectively. Finally,
thetreatmentsthatmakeupthe TRNsetareshownin Table4.1.
Accordingto Table4.1, itisclearthateachtreatmentispresentintwoblocksand
missinginoneblock. Forexample, in Block1 thetestingsetincludestreatments2,4,
and6;in Block2, thetestingsetiscomposedoftreatments1,8, and10;andin Block
3, thetestingsetiscomposedoftreatments3,5, and9.
4.3.9 Random Cross-validation with Blocks
Randomcross-validationwithblockswasproposedby Lopez-Cruzetal.(2015) and
belongs to the so-called replicated TRN-TST cross-validation that appears in the
publicationof Daetwyler etal.(2012), since some individuals cannever bepart of
the training set. This algorithm, like the incomplete block cross-validation, is
appropriatewhenweareinterestedinevaluating Jlinesin Iblocksorenvironments
andtriestomimicapredictionproblemfacedbybreedersinincompletefieldtrials
wherelinesareevaluatedinsome, butnotall, targetenvironments. Thealgorithmfor
constructingthe TRN-TSTsetsisdescribedbythefollowingsteps:Step1.Calculate
the total number of observations under study as N ¼ J (cid:4) I; Step 2. Define the
proportionofobservationsusedfortrainingandtesting, thatis, P and P ;Step
TRN TST
3.Calculatethesizeofthetestingset N ¼N(cid:4)P ;Step4.Choose N linesat
TST TST TST

#### 8

122 4 Overfitting, Model Tuning, and Evaluationof Prediction Performance
Table4.2 TRN-TSTdatasetsfor J¼10 linesand I¼3 environments
Environment Lines
L1 L2 L3 L4 L5 L6 L7 L8 L9 L10
1 TRN TST TRN TRN TST TRN TRN TST TRN TRN
2 TST TRN TRN TRN TRN TRN TST TRN TST TRN
3 TRN TRN TST TST TRN TRN TRN TRN TRN TST
random without replacement if J (cid:5) N , and with replacement otherwise; Step
TST
5. Each chosen line will then be assigned to one of the I environments chosen at
random without replacement; Step 5. All the selected lines and environments will
form the training set, while the lines and environments that were not chosen will
formthecorrespondingtestingset;and Step6.Steps1–5 arerepeateddependingon
the number of TRN-TST partitions required (Lopez-Cruz et al. 2015). This CV is
called CV2 in Jarquín et al. (2017). Next, we assume that we have ten lines or
treatments and three environments or blocks that will form the corresponding
trainingtestingsetsforonlyonepartition:Step1.Thetotalnumberofobservations
understudyis N¼10(cid:4)3¼30;Step2.Wedefine P ¼0.7 and P ¼0.3;Step
TRN TST
3.Thesizeofthetestingsetis N ¼30(cid:4)0.3¼9;Step4.Since J¼10(cid:5)N ¼9,
TST TST
weselectedthefollowinglinesatrandomwithoutreplacement:L1, L2, L3, L4, L5,
L6, L7, L8, L9, and L10; and Step 5. Each chosen line was assigned to one of the
I¼3 environmentsrandomlychosenwithoutreplacement, asshownin Table4.2.It
isimportanttopointoutthatthis CVstrategyonlydiffersfromtheincompleteblock
cross-validationinthewaythelinesareallocatedtoblocks.
4.3.10 Other Options and General Comments
on Cross-validation
Itisimportanttohighlightthatwhenthedatasetisconsiderablylarge, itisbetterto
randomlysplititintothreeparts:atrainingset, avalidationset (ortuningset), anda
testing set. The training set and testing set are used as explained before, while the
validation (tuningset) setisusedtoestimatethepredictionerrorformodelselection,
which is the process of estimating the performance of different models in order to
choose the best one, or to evaluate the chosen statistical machine learning model
with a range of values of tuning hyperparameters to select the combination of
hyperparameters with the best prediction performance and then use these
hyperparameters (orbestmodel) toevaluatethepredictionperformanceinthetesting
set (Fig.4.7).Itisimportanttopointoutthat Fig.4.7 showsonlyonerandomsplitof
thedataintermsofthetraining, testing, andvalidationsets.
Forexample, assumethatourdatasethas50,000 rows (observations) andthatwe
have decided to use 5000 of them for the testing set and another 5000 for the
validation set. This means that 40,000 rows are left for the training set. At this
point, we train our statistical machine learning model with each component of the

#### 9

4.3 Cross-validation 123
Fig.4.7 Schematicrepresentationofthetraining, validation (tuning), andtestingsetsproposedby
Cook (2017)
gridofhyperparametersofthetrainingsetandevaluatethepredictionperformance
onthevalidationset, asshowninthemiddleof Fig.4.7.Finally, wewillpickthebest
model (best subset of hyperparameters) in terms of prediction performance in the
validationsetandwearereadytoevaluatethepredictionperformanceinthetesting
set. Then, wewillreportthetestingerroronthetestingset, ascanbeobservedatthe
bottomof Fig.4.7(Cook2017).Underthisapproach, thetestingandvalidationsets
have approximately the same size to guarantee a similar out-of-sample prediction
performance. Sinceitisdifficulttogivegeneralrulesonhowtochoosethenumber
of observations in each of the three parts, a typical number might be 50% for
training, and 25% each for validation and testing (Hastie et al. 2008) or 70%,
15%, and 15% for training, validation, and testing, respectively. Another way to
find the optimal setting of hyperparameters using the grid search, which is very
commonindeeplearning, consistsofpickingvaluesforeachparameterfromafinite
setofoptions[e.g., numberofepochs (100,150,200,250,300);batchsizes (25,50,
75, 100, 125); number of layers (1, 2, 3, 4, 5); and types of activation functions
(RELU, Sigmoid), ...] and training the statistical machine learning model with
every permutation of hyperparameter choices using the training set. Then, the
combination of hyperparameters with the best prediction performance on the vali-
dation set is chosen, and we report the prediction performance of the best selected
model (set of hyperparameters) in the testing set (Buduma 2017). The aforemen-
tioned examples use the validation data set as a proxy measure of the accuracy
duringthe hyperparameter optimization process. However, it can also be usedas a
proxy measure of the accuracy for model selection, and instead of using a grid of
hyperparameters, we can use a set of different statistical machine learning models;
here, thebestmodelischoseninsteadofthebestcombinationofhyperparameters. It
is important to understand that when you have more than one random partition
(training-testing-validation), as shown in Fig. 4.8, the same process provided in
Fig. 4.7 is followed, however, the average of all the partitions is reported as a
measure of prediction performance. Also, if more precision is required in the
estimated prediction performance, you can repeat the process given in Fig. 4.8

#### 10

124 4 Overfitting, Model Tuning, and Evaluationof Prediction Performance
Fig.4.8 Fivepartitionsoftraining-validation-testing
multiple times and report the average of all repetitions as a measure of prediction
performance.
Asmentionedabove, thisapproach (Figs.4.7 and4.8) isusedforalargedataset
however, in the case of a smaller data set, we suggest modifying this approach to
avoidwastingtoomuchtrainingdatainvalidationsets. Thismodificationconsistsof
performing an inner cross-validation approach since the training set is split into
complementarysubsets, andeachmodelistrainedagainstadifferentcombinationof
thesesubsets, whichissubsequentlyvalidatedagainsttheremainingparts. Oncethe
modelhyperparametershavebeenselected, afinalmodelistrained withthewhole
trainingset (refitted) andthegeneralizedpredictionperformanceismeasuredonthe
testingset. Thisapproachwasappliedby Montesinos-Lópezetal.(2018 a, b), who
alsosplittheoriginaldataintotrainingandtestingsets. Subsequently, eachtraining
set was split again and 80% of the data was used for training a grid of
hyperparameters while the remaining 20% was used for validating (tuning) the
prediction performance and selecting the best combination of hyperparameters
with the best prediction performance. At this point, the deep learning algorithm
was refitted with the whole training set, and with this they evaluated the out-of-
sample prediction. They called the conventional training-testing partition outer
cross-validation, while the split performed in each training set used for
hyperparameter tuning was called inner cross-validation. It should be highlighted
thattherearenodifferencesbetweenouterandinner CVandtraining-validation-test.
Finally, itshould also bementioned thatany type ofthecross-validationstrategies
mentionedinthissection (random CV, k-fold CV, Bootstrap CV, IBCV, etc.,) can
beusedinbothouterandinner CV, suchasisthecasewiththefive-fold CV.
4.4 Model Tuning
A hyperparameter is a parameter whose value is set before the learning process
begins. Hyperparametersgovernmanyaspectsofthebehaviorofstatisticalmachine
learning models, such as their ability to learn features from data, the models’
exhibiteddegreeofgeneralizabilityinperformance when presented withnewdata,
as well as the time and memory cost of training the model, since different
hyperparameters often result in models with significantly different performance.
This means that tuning hyperparameter values is a critical aspect of the statistical
machinelearningtrainingprocessandakeyelementforthequalityoftheresulting

#### 11

4.4 Model Tuning 125
Fig.4.9 Schematicrepresentationofthetuningprocessproposedby Kuhnand Johnson (2013)
predictionaccuracies. However, choosingappropriatehyperparametersischalleng-
ing (Montesinos-Lópezetal.2018 a).Hyperparametertuningfindsthebestversion
ofastatisticalmachinelearningmodelbyrunningmanytrainingsetsontheoriginal
data set using the algorithm and ranges of values of hyperparameters as specified.
The hyperparameter values that provide the best performance in out-of-sample
predictionevaluatedbythechosenmetricarethenselected.
There are many ways of searching for the best hyperparameters. However, a
general approach defines a set of candidate values for each hyperparameter. Each
valueofthissetofcandidatevaluesisthenappliedwitharesampleofthetrainingset
ofthechosenstatisticalmachinelearningmethod, whereweaggregateallthehold-
outpredictionsfromwhichthebesthyperparametersarechosenandrefitthemodel
with the entire set (Kuhn and Johnson 2013). A schematic representation of the
tuning process proposed by Kuhn and Johnson (2013) is given in Fig. 4.9. It is
importanttohighlightthatthisprocessshouldbeperformedcorrectlybecausewhen
the same data are used for training and evaluating the prediction performance, the
predictionperformanceobtainedisextremelyoptimistic.
For example, suppose a breeder is interested in developing an algorithm to
classify unseen plants as diseased or not diseased with an available training data
set. The goal is to minimize the rate of misclassification or to maximize the
percentage of cases correctly classified (PCCC). Also, assume that you are new to
theworldofstatisticalmachinelearningandthatyouonlyunderstandthek-nearest
neighbor method. Since this algorithm depends only on the hyperparameter called
thenumberofneighbors (k), thequestioniswhichvalueofktochooseinsuchaway
that the prediction performance of this algorithm will be the best in the sample
predictionofplants. Tofindthebestvalueofthekhyperparameter, youmustspecify
arangeofvaluesfork (forexample, from1 to60 withincrementsof1), thenwitha
partofyourtrainingdataset, calledthetraining-inner (ortuningthatcorrespondsto
thetrainingdataintheinnerloop) set, whichisrandomlyselected. Youproceedto
evaluate the 60 values of k with the k-nearest neighbor method and evaluate the

#### 12

126 4 Overfitting, Model Tuning, and Evaluationof Prediction Performance
prediction performance in the remaining part of the training set (validation set).
Next, youselectthevalueofkfromthisrangeofvaluesthatbestpredicts (according,
forexample, tothe PCCC) out-of-sampledata (validationset) andusethisvalueto
performthepredictionoftheunseenplantsnotusedfortrainingthemodel (testing
set). This is a widely adopted practice that consists of searching for the parameter
(usuallythroughbruteforceloops) thatyieldsthebestperformanceoveravalidation
set. However, the process illustrated here is very simple because the k-nearest
neighbor model only depends on a unique hyperparameter; however, there are
other statistical machine learning algorithms (for example, deep learning methods)
wherethetuningprocessisrequiredforaconsiderableamountofhyperparameters.
Forthisreason, weencouragecautionwhenchoosingthestatisticalmachinelearn-
ingalgorithm, sincetheamountofworkrequiredforperformingthetuningprocess
dependsonthechosenmethod.
4.4.1 Why Is Model Tuning Important?
Tuning the hyperparameters of the models is a key element to optimize your
statistical machine learning model to perform well in out-of-sample predictions.
Thetuningprocessismoreanartthanasciencebecausethereisnouniqueformal
scientificprocedureavailableintheliterature. Nowadays, thetuningprocessistrial
anderrorthatconsistsofimplementingthestatisticalmachinelearningmodelmany
times with different values of the hyperparameters and then comparing its perfor-
mance on the validation set in order to determine which set of hyperparameters
results in the most accurate model; for the final implementation, the set of
hyperparameters of the best model is used. As mentioned above, for the k-nearest
neighborclassifier, weneedtochoosethenumberofneighbors (k) usingthetuning
process to obtain the optimal prediction performance of this algorithm, while for
conventional Ridge regression, the parameter lambda (λ) is obtained by tuning to
improvetheout-of-samplepredictions. Thesetwostatisticalmachinelearningalgo-
rithms that we just mentioned need only one hyperparameter; however, other
statistical machine learning methods may require more hyperparameters, as exem-
plifiedbydeeplearningmodelsthatrequireatleastthreehyperparameters (number
of neurons, number of hidden layers, type of activation function, batch size, etc.).
After tuning the required hyperparameters, the statistical machine learning model
learnstheparametersfromthedatatobeusedforthefinalpredictionofthetesting
set. Thechoiceofhyperparameterssignificantlyinfluencesthetimerequiredtotrain
andtestastatisticalmachinelearningmodel. Hyperparameterscanbecontinuousor
of the integer type; for this reason, there are mixed-type hyperparameter optimiza-
tionmethods.

#### 13

4.4 Model Tuning 127
4.4.2 Methods for Hyperparameter Tuning (Grid Search,
Random Search, etc.)
Manualtuningofstatisticalmachinelearningmodelsisofcoursepossible, butrelies
heavily on the user’s expertise and understanding of the underlying problem.
Additionally, due to factors such as time-consuming model evaluations, nonlinear
hyperparameterinteractionsinthecaseoflargemodelsthatconsistoftensoreven
hundreds of hyperparameters, manual tuning may not be feasible since it is equiv-
alent to brute force. For this reason, the four most common approaches for
hyperparameter tuning reported in the literature are (a) grid search, (b) random
search,(c)Latinhypercubesampling, and (d) optimization (Kochetal.2017).
In the grid search method, each hyperparameter of interest is discretized into a
desiredsetofvaluestobestudiedwherethemodelsaretrainedandassessedforall
combinations of the values across all hyperparameters (that is, a “grid”). Although
fairlysimpleandstraightforwardtocarryout, agridsearchisappropriatewhenthere
areonlyafewvaluesforalimitednumberofhyperparameters. However, although
this is a comprehensive way of assessing different hyperparameter values, when
therearemanyvaluesforsomeormanyhyperparameters, itquicklybecomesquite
costly due to the number of hyperparameters and the number of discrete levels of
each. For example, in Ridge regression, this approach is implemented as follows:
since λ is the hyperparameter to be tuned, we first propose, for example, a grid of
100 values for this hyperparameter from λ¼1010 toλ ¼10(cid:1)2;then we dividethe
training set into five inner training sets and five inner testing (tuning) sets, where
eachofthe100 valuesofthegridisfittedusingtheinnertrainingsetsandthetesting
errorisevaluatedwiththeinnertestingsets. Thenwegettheaveragepredictedtest
error and pick one value out of the 100 values of the grid that produces the best
predictionperformance. Next, werefitthestatisticalmachinelearningmethodtothe
wholetrainingsetusingthepickedvalueofλ, andfinallyperformthepredictionsfor
the testing set using the learned parameters of the training set with the best picked
valueofλ.Inallthemodelswithonehyperparameter, itispracticaltoimplementthe
grid search method, but for example, in deep learning models, which many times
require six hyperparameters to be tuned, if only three values are used for each
hyperparameter, thereare36¼729 combinationsthatneedtobeevaluated, quickly
becomingcomputationallyimpracticable.
A random search differs from a grid search in that rather than providing a
discrete set of values to explore each hyperparameter, we determine a statistical
distributionforeachhyperparameterfromwhichvaluesmayberandomlysampled.
This affords a much greater chance of finding effective values for each
hyperparameter. While Latinhypercubesamplingissimilartothepreviousmethod,
itisamorestructuredapproach (Mc Kay1992) sinceitisanexperimentaldesignin
which samples are exactly uniform across each hyperparameter but random in
combinations. These so-called low-discrepancy point sets attempt to ensure that
points are approximately equidistant from one another in order to fill the space

#### 14

128 4 Overfitting, Model Tuning, and Evaluationof Prediction Performance
efficiently. This sampling supports coverage across the entire range of each
hyperparameterandismorelikelytofindgoodvaluesofeachhyperparameter.
The previous two methods for hyperparameter tuning are used to perform indi-
vidual experiments by building models with various hyperparameter values and
recording the model performance for each. Because each experiment is performed
inisolation, thisprocessisparallelized, butisunabletousetheinformationfromone
experiment to improve the next experiment. Optimization methods, on the other
hand, consistofsequentialmodel-based optimizationwheretheresultsofprevious
experimentsareusedtoimprovethesamplingmethodofthenextexperiment. These
methodsaredesignedtomakeintelligentuseoffewerevaluationsandthussaveon
theoverallcomputationtime (Kochetal.2017).Optimizationalgorithmsthathave
been used in statistical machine learning generally for hyperparameter tuning
include Broyden–Fletcher–Goldfarb–Shanno (BFGS) (Konen et al. 2011), covari-
ance matrix adaptation evolution strategy (CMA-ES) (Konen et al. 2011), particle
swarm (PS)(Renukadeviand Thangaraj2014), tabusearch (TS), geneticalgorithms
(GA)(Lorenaandde Carvalho2008), andmorerecently, surrogate-based Bayesian
optimization (Dewanckeretal.2016).Also, recentlytheuseoftheresponsesurface
methodologyhasbeenexploredfortuninghyperparametersinrandomforestmodels
(Lujan-Moreno et al. 2018). However, the implementation of these optimization
methods is not straightforward because it requires expensive computation; also,
softwaredevelopmentisrequiredforimplementingthesealgorithmsautomatically.
Therehavebeenadvancesinthisdirectionforsomemachinelearningalgorithmsin
thestatisticalanalysissystem (SAS), Rand Pythonsoftware (Kochetal.2017).An
additional challenge isthe potential unpredictable computation expense of training
and validating predictive models using different hyperparameter values. Finally,
althoughitischallenging, thetuningprocessoftenleadstohyperparametersettings
that are better than the default values, as it provides a heuristic validation of these
settings, givinggreaterassurancethatamodelconfigurationwithahigheraccuracy
hasnotbeenoverlooked.
4.5 Metrics for the Evaluation of Prediction Performance
Thequalityofpredictionperformanceofanystatisticalmachinelearningmethodin
agivendatasetconsistsofevaluatinghowclosethepredictedvaluesaretothetrue
observed ones. In other words, the prediction performance quantifies the matching
degree between the predicted response value for a given observation and the true
responsevalue for that observation (James etal.2013).However, themetrics used
forquantifyingthepredictionperformancedependonthetypeofresponsevariable
understudy;forthisreason, wesubsequentlygivethemostpopularmetricsusedfor
thisgoalforfourtypesofresponsevariables.

#### 15

130 4 Overfitting, Model Tuning, and Evaluationof Prediction Performance
to Pearson’scorrelation, thesevaluesshouldalsobereported (slopeandintercept).
The expected intercept should be zero and the slope 1, if the correlation obtained
betweentheobservedandpredictedvaluesishigh. Itisimportanttoavoidcarrying
out the regression in the opposite way, i.e., using predicted values as y’s, and
observed values as x’s, since this leads to incorrect estimates of the slope and the
y-intercept. Thisdenotesthataspuriouseffectisaddedtotheregressionparameters
whenregressingpredictedversusobservedvaluesandcomparingthemagainstthe1:
1 line. The user should also remember that underestimation of the slope and
overestimation of the y-intercept increase as Pearson’s correlation values decrease.
Westronglyrecommendthatscientistsevaluatetheirmodelsbyregressingobserved
versus predicted values and test the significance of slope ¼ 1 and intercept ¼ 0
(Piñeiro et al. 2008). Finally, it is important to recall that the square of Pearson’s
correlationcanalsobeusedasametricformeasuringpredictionperformancesinceit
representstheproportionofthetotalvarianceexplainedbytheregressionmodeland
iscalledthecoefficientofdeterminationdenotedas R2.
Next, we present the mean absolute error (MAE) metric that measures the
difference between two continuous variables (observed and predicted). The MAE
canbecalculatedwiththefollowingexpression:
MAE ¼ 1
XnþT (cid:8)
(cid:8) (cid:8) y (cid:1) b fðxÞ
(cid:8)
(cid:8) (cid:8) ð4:3Þ
TST T i i
i¼nþ1
Belowwepresentanothermetricusedtoevaluatethepredictionperformanceof
anystatisticalmachinelearningmodel;itwasproposedby Kimand Kim (2016) and
iscalledmeanarctangentabsolutepercentageerror (MAAPE) thatiscalculatedas
follows:
(cid:6)(cid:8) (cid:8)(cid:9)
P (cid:8) b (cid:8)
nþT arctan (cid:8) (cid:8) y i (cid:1) fðxiÞ(cid:8) (cid:8)
i¼nþ1 y
MAAPE ¼ i ð4:4Þ
TST T
Although MAAPEisfinitewhentheresponsevariable (i.e., y ¼0) equalszero,
i
sinceithasasatisfactorytrigonometricrepresentation. However, because MAAPE’s
valueisexpressedinradians, itislessintuitive, inadditiontobeingscale-free. This
metric is a modification of the mean absolute percentage error (MAPE) which is
problematic because it is undefined when the response variable is equal to zero
(y ¼0). MAAPE isalso asymmetric since division by zero isdefined and isnota
i
problem. It is important to point out that there are other metrics for measuring
prediction accuracy for continuous data, but we only presented the most
popularones.
Thedistinctionbetweenthetrainingandtest MSEisimportantsincewearenot
interested in how well the statistical machine learning method performs in the
trainingdataset, duetothefactthatourmaingoalistoperformaccuratepredictions
intheunseentestdata. Forexample, aplantbreedermaybeinterestedindeveloping
analgorithm topredictdisease resistanceofa plant innew environments basedon

#### 16

4.5 Metricsforthe Evaluationof Prediction Performance 131
records that were collected in a set of environments. We can train the statistical
machinelearningmethodwiththeinformationcollectedinthesetofenvironments,
buttheinterestisnotinhowwellthestatisticalmachinelearningmethodpredictsin
those previously collected environments. An environmental scientist can also be
interestedinpredictingtheaverageannualrainfallinamunicipalityin Mexico, using
data from the last 20 years to train the model. Such measures could include sea
surfacetemperature, time, theyearlyrotationoftheearth, amongothers. Inthiscase,
the scientist is really interested in predicting the average rainfall of the next 1 or
2 years, notinaccuratelypredictingtheyearsmeasuredinthetrainingset.
4.5.2 Binary and Ordinal Measures of Prediction
Performance
The binary and ordinal response variables are very common in classification prob-
lems, wherethegoalistopredictwhichcategorysomethingfallsinto. Anexample
ofaclassificationproblemisanalyzingfinancialdatatodetermineifaclientwillbe
granted credit or not. Another example is analyzing breeding data to predict if an
animalisathighriskforacertaindiseaseornot. Below, weprovidesomepopular
metricstoevaluatethepredictionperformanceofthistypeofdata.
The first metric is called a confusion matrix, which is a tool to visualize the
performance of a statistical machine learning algorithm that is used in supervised
learning for classifying categorical and binary data. Each column of the matrix
represents the number of predictions in each class, while each row represents the
instancesintherealclass. Oneofthebenefitsofconfusionmatricesisthattheymake
it easy to determine whether the system is confusing classes. Table 4.3 shows a
sampleformatofaconfusionmatrixof Cclasses.
With Eqs.(4.5–4.8) wecancalculatethetotalnumberoffalsenegatives (TFN),
false positives (TFP), true negatives (TTN) for each class i, and the total true
positivesinthesystem, respectively:
XC
TFN ¼ n ð4:5Þ
i ij
j¼1
j6¼i
Table4.3 Confusionmatrixwithmorethantwoclasses
Predictedvalues
Class1 Class2 ... Class C
Observedvalues Class1 n n ... n
11 12 1 C
Class2 n n ... n
21 22 2 C
⋮ ⋮ ⋮ ⋮ ⋮
Class C n n ... n
C1 C2 CC

#### 17

132 4 Overfitting, Model Tuning, and Evaluationof Prediction Performance
XC
TFP ¼ n ð4:6Þ
i ji
j¼1
j6¼i
XC XC
TTN ¼ n ð4:7Þ
i ji
j¼1 k¼1
j6¼i k6¼i
XC
TTP ¼ n ð4:8Þ
all jj
j¼1
Below, we define the sensitivity (S ), precision (P), and specificity (S ). The
e p
sensitivityindicatestheabilityofourstatisticallearningalgorithmtodeterminethe
proportionoftruepositivesthatarecorrectlyidentifiedbythetest. Theprecisionis
theproportionofcorrectclassificationofourstatisticalmachinelearningmodeland
represents the proportion of cases correctly classified, while the specificity is the
ability ofourstatistical machinelearningmodel toclassify thetrue negativecases,
thatis, thespecificityistheproportionoftruenegativesthatarecorrectlyidentified
bythetest. Underthe“one-versus-allbasis,”whereeachcategoryiscomparedwith
the composedinformation of the remaining categories, we provide the expressions
forcomputingthegeneralizedprecision, sensitivity, andspecificityforeachclassi:
TTP
P ¼ all ð4:9Þ
i TTP þTFP
all i
TTP
S ¼ all ð4:10Þ
ei TTP þTFN
all i
TTN
S ¼ all ð4:11Þ
pi TTN þTFP
all i
TTN
p CCC¼ all ð4:12Þ
PC PC
n
ij
i¼1 j¼1
Thetermp CCC denotesthe proportionofcases correctlyclassified, whichisa
measureoftheoverallaccuracy, andwhenmultipliedby100, denotesthepercentage
of cases correctly classified. Many times, this is the only metric reported for
measuring prediction performance in multi-class problems. However, PCCC alone
is sometimes quite misleading as there may be a model with relatively “high”
accuracy, but it predicts the “unimportant” class labels fairly accurately (e.g.,
“unknown bucket”). However, the model may be making all sorts of mistakes on
theclassesthatareactuallycriticaltotheapplication. Thisproblemisseriouswhen
intheinputdatathenumberofsamplesofdifferentclassesisveryunbalanced. For

#### 18

136 4 Overfitting, Model Tuning, and Evaluationof Prediction Performance
Matthewscorrelationcoefficient (MCC).Introducedin1975 by Brian Matthews
(1975) andregardedbymanyscientistsasthemostinformativescorethatconnects
all four measures in a confusion matrix, the Matthews Correlation Coefficient is
typically used in statistical machine learning to measure the quality of binary
classifications and it is particularly useful when there is a significant imbalance in
classsizes (data).MCCiscalculatedaccordingtothefollowingexpression:
tp (cid:4) tn (cid:1) fp (cid:4) fn
MCC¼pffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffi ð4:13Þ
ðtpþfpÞ(cid:4)ðtpþfnÞ(cid:4)ðtnþfpÞ(cid:4)ðtnþfnÞ
Ifanyofthedenominatortermsequalszeroin (4.13) itwillbesetto1 and MCC
becomes zero, which has been shown to be the correct limiting value. It returns a
valuebetween (cid:1)1 and1, where1 meansaperfectprediction,0 meansnobetterthan
randomand (cid:1)1 meansatotaldisagreementbetweenpredictedandobservedvalues.
Next, wepresentthe Brierscore (Brier1950) forcategoricalorbinarydatathat
canbecomputedas
XnþT XC
BS¼T(cid:1)1 ðbπ (cid:1) d Þ2, ð4:14Þ
ic ic
i¼nþ1 c¼1
where bπ denotes the estimated probabilities (predictive distribution) derived from
i
the estimated model for observation i and d takes a value of 1 if the categorical
ic
responseobservedforindividualifallsintocategoryc;otherwise, d ¼0.Therange
ic
of BS in Eq. (4.14) for categorical data is between 0 and 2. For this reason, we
suggestdividingby2, thatis, BS/2, toobtainthe Brierscoreboundbetween0 and1;
lowerscoresimplybetterpredictions (Montesinos-Lópezetal.2015 a, b).
Finally, we describe the use of negative log-likelihood (MLL) to evaluate the
predictionperformance. Thismetrichasthecharacteristicthatbetterforecastshave
lowervaluesandforthisreason, itisanalogoustothe MSE.Forcategoricaldata,
" #
1
XnþT XC
MLL¼(cid:1) 1 fy ¼kglogðbπ Þ ,
T i ic
i¼nþ1 c¼1
where 1{y (i) ¼ k} is an indicator variable taken the value of 1 when the ith
observation is assigned to category c, for c ¼ 1, 2, ..., C takes place in the ith
observation. (cid:10)PWhen the data are binary, the (cid:11) MLL is reduced to
MLL ¼ (cid:1)1 nþT ½y logðbπÞþð1(cid:1) yÞlogð1(cid:1) bπÞ(cid:7) . Following, we provide
T i¼nþ1 i i i i
some advantages ofusing the MML as a measure ofpredictionperformance:(a) it
has a simple definition that, from a purely intuitive point of view, seems to be a
reasonablebasisonwhichtocompareforecasts;(b) itismathematicallyoptimalin
thesensethatestimatesofparametersofcalibrationmodelsfittedbymaximizingthe
likelihoodareusuallythemostaccuratepossibleestimates (see Cassellaand Berger
2002);(c) itisageneralizationtoprobabilisticforecastsofthemostcommonlyused
skillscoreforsingleforecasts;(d) thepropertiesofthelikelihoodhavebeenstudied

#### 19

4.5 Metricsforthe Evaluationof Prediction Performance 137
atgreatlengthoverthelast90 years, andassuchiswellunderstood;(e) itisbotha
measureofresolutionandreliability;(f) likelihoodcanbeusedforbothcalibration
andassessment:thiscreatesconsistencybetweenthesetwooperations;(g) useofthe
likelihood also creates consistency with other statistical modeling activities, since
mostotherstatisticalmodelingusesthelikelihood, whichisimportantincaseswhere
theuseofforecastsissimplyasmallpartofalargerstatisticalmodelingeffort, asis
the case of our particular business; (h) likelihood can be used for all types of
response variables; and (i) likelihood can be used to compare multiple leads,
multiple variables, and multiple locations at the same time in a sensible way with
asinglescoreevenwhentheseleads, variables, andlocationsarecross-correlated.
4.5.3 Count Measures of Prediction Performance
Spearman’s correlation and the MML are recommended to measure the prediction
performanceforcountdata.
Fortheapplicationofthe Spearman’scorrelation, theformulagivenin Eq.(4.2)
for Pearson’s correlation can be used; however, instead of using the observed and
predicted values directly, these are replaced by their corresponding ranks. For
example, assuming that the observed and predicted values are y ¼ {15, 9, 12,
27, 6, 3, 36, 15, 21, 30} and by¼f20, 17, 24, 25, 3, 3, 34, 22, 21, 33}, we can
thus show how to get the rank for the observed values: rango ¼
y
{5,3,4,8,2,1,10,6,7,9}. However, in this vector, observations 1 and 8 are the
same, andassuch, theirpositionsareaddedanddividedbytwo, thatis,5þ6¼5:5.
2
Therefore, the final rank for the observed values is rango ¼
y
{5.5,3,4,8,2,1,10,5.5,7,9}. Now the range for the predicted values is rangob¼
y
f4,3,7,8,1,2,10,6,5,9 g, butagain, sincevalues5 and6 arethesame, weaddtheir
ranges, and as this is repeated twice, it is divided by two and we get 1þ2¼1:5.
2
Therefore, the final range of the predicted values is rangob¼
y
f4,3,7,8,1:5,1:5,10,6,5,9 g. Finally, to obtain the Spearman correlation, we
used the expression given in Eq. (4.2) for Pearson’s correlation, and instead of
using the original observed and predicted values, we used rango
y
and rangob
y
. The
interpretationofthismetricisequaltothatofthe Pearsoncorrelation, thatis, whenit
is closer to 1, the prediction performance of the implemented statistical learning
methodisbetter. Itshouldbenotedthatwhenthenumberofrepeatedvaluesinthe
observedandpredicted valuesisgreaterthan two, theadjustedrange isthesumof
therepeatedrangesdividedbythenumberofrepeatedvalues;thisnewrangeisthen
giventotherepeatedvalues. Inthiscase, itisalsoimportanttoregresstheobserved
versusthepredictedvaluestoobtaintheinterceptandslopeusingtherangesofthe
observedandpredictedvalues.
Itisalsopossibletousethe MLLcriteriatoassessthepredictionperformancefor
countdata;however, thenewexpressionisnowbasedonminusthelog-likelihoodof
a Poissondistribution, whichisequalto

#### 20

138 4 Overfitting, Model Tuning, and Evaluationof Prediction Performance
1
XnþT h
b
(cid:2)
b
i
MLL¼ (cid:1) fðxÞþy log fðxÞ
T i i i
i¼nþ1
Again, whenthevaluesof MLLarelower, theobservedandpredictedvaluesare
closertooneanother.
References
Brier GW(1950)Verificationofforecastsexpressedintermsofprobability. Mon Weather Rev78:
1–3
Buduma M(2017)Fundamentalsofdeeplearning,1 stedn. O’Reilly, Sabastopol, CA
Burger SV(2018)Introductiontomachinelearningwith R.Rigorousmathematicalanalysis,1 st
edn. O’Reilly, Sabastopol, CA
Cassella G, Berger RL(2002)Statisticalinference. Duxbury, Belmont, CA
Cohen J(1960)Acoefficientofagreementfornationaldata. Educ Psychol Meas20:37–46
Cook D(2017)Practicalmachinelearningwith H O.O’Reilly Media, Inc, Sabastopol, CA
2
Daetwyler HD, Calus MPL, Pong-Wong R, de los Campos G, Hickey JM (2012) Genomic
predictioninanimalsandplants:simulationofdata, validation, reportingandbenchmarking.
Genetics193:347–365
Dewancker I, Mc Court M, Clark S, Hayes P, Johnson A, Ke G (2016) A stratified analysis of
Bayesianoptimizationmethods.ar Xiv:1603.09441 v1
Efron B(1983)Estimatingtheerrorrateofapredictionrule:improvementoncross-validation. J
Am Stat Assoc78(382):316–331
Fielding AH, Bell JF (1997) A review of methods for the assessment of prediction errors in
conservation presence/absence models. Environ Conserv 24:38–49. https://doi.org/10.1017/
S0376892997000088
González-Camacho JM, Ornella L, Pérez-Rodríguez P, Gianola D, Dreisigacker S, Crossa J(2018)
Applications of machine learning methods to genomic selection in breeding wheat for rust
resistance. Plant Genome11(2):1–15.https://doi.org/10.3835/plantgenome2017.11.0104
Hagerty MR, Srinivasan S(1991)Comparingthepredictivepowersofalternativemultipleregres-
sionmodels. Psychometrika56:77–85.MR1115296
Hastie T, Tibshirani R, Friedman J (2008) The elements of statistical learning: data mining,
inference, andprediction, Springerseriesinstatistics,2 ndedn. Springer, New York
James G, Witten D, Hastie T, Tibshirani R (2013) An introduction to statistical learning: with
applicationsin R.Springer, New York
Jarquín D, Lemesda Silva C, Gaynor RC, Poland J, Fritz ARet al (2017) Increasing genomic-
enabled prediction accuracy by modeling genotype (cid:4) environment interactions in Kansas
wheat. Plant Genome10(2):1–15.https://doi.org/10.3835/plantgenome2016.12.0130
Kim S, Kim H(2016)Anewmetricofabsolutepercentageerrorforintermittentdemandforecasts.
Int JForecast32(3):669–679
Koch P, Wujek B, Golovidov O, Gardner S(2017)Automatedhyperparametertuningforeffective
machinelearning. In:Proceedingsofthe SASglobalforum2017 conference. SASInstitute Inc,
Cary, NC.http://support.sas.com/resources/papers/proceedings17/SAS514-2017.pdf
Konen W, Koch P, Flasch O, Bartz-Beielstein T, Friese M, Naujoks B(2011)Tuneddatamining:a
benchmarkstudyondifferenttuners. In:Proceedingsofthe13 thannualconferenceongenetic
andevolutionarycomputation (GECCO-2011).SIGEVO/ACM, New York
Kuhn M, Johnson K(2013)Appliedpredictivemodeling. Springer, New York
Lopez-Cruz M, Crossa J, Bonnett D, Dreisigacker S, Poland J, Jannink J-L, Singh RP, Autrique E,
delos Campos, G.(2015)Increasedpredictionaccuracyinwheatbreedingtrialsusingamarker
(cid:4) environmentinteractiongenomicselectionmethod. G35(4):569–582


---

## Content Unique to Old Extraction (not found in markitdown output)

### 1. # Chapter4.MontesinosLpez2022_Chapter_OverfittingModelTuningAndEvalu...

# Chapter4.MontesinosLpez2022_Chapter_OverfittingModelTuningAndEvalu

### 2. > *Source PDF: Chapter4.MontesinosLpez2022_Chapter_OverfittingModelTuningAndEval...

> *Source PDF: Chapter4.MontesinosLpez2022_Chapter_OverfittingModelTuningAndEvalu.pdf*
> *Extraction: Combined — markitdown raw text (base) + previous extraction supplements*

### 3. ## Content from Previous Extraction (not in markitdown output)...

## Content from Previous Extraction (not in markitdown output)

### 4. ### Visual/Chart/Graph Descriptions...

### Visual/Chart/Graph Descriptions

### 5. 110 4 Overfitting, Model Tuning, and Evaluationof Prediction Performance
Fig.4.1...

110 4 Overfitting, Model Tuning, and Evaluationof Prediction Performance
Fig.4.1 Schematicillustrationofthreemodelsforclassification:(a)M1 withunderfitting,(b)M2
withappropriatefitting, and (c)M3 withoverfitting
Considerascatteredseriesofpoints (y , x ),...,(y , x ), onaplane, towhichwe
1 1 n n
wanttoadjustastatisticalmachinelearningmethod. Thismeansthatwearelooking
forthebestf (x) thatexplainstheexistingrelationshipbetweentheresponsevariable
i
(y) andthepredictors (x ,..., x ).Weassumethat wehavethree options for f (x):
i 1 n i
M1, the simple model plotted in Fig. 4.1 a; M2, an intermediate model shown in
Fig.4.1 b;and M3, acomplexmodelshownin Fig.4.1 c.
Undertheclassificationframework, thefirstpanelin Fig.4.1(leftside, panela)
shows an unsatisfactory fit (underfitted) since the line does not cover most of the
points (has high bias) in theplot. As such, we expect that theprediction of unseen
dataofthismodel, M1, willperformbadly. Incontrast, panelcof Fig.4.1 showsan
almost perfect fit, since the predicted line covers all the data points. While at first
glance, you may think that model M3 will perform well when predicting unseen
data, thisisactuallyuntruesincethepredictedlinecoversallpointsthatarenoiseand
thosethataresignal (overfit);forthisreason, thistypeofmodelalsoperformspoorly
inthe predictionof future data due toits complexityandhigh variance. Therefore,
the best option for predicting unseen data is model M2 (Fig. 4.1, panel b) since it
represents the predominant (smooth) pattern enough to represent the apparent data
pattern while maintaining a balance between bias and variance. For this reason, a
well-fitted model is one that faithfully represents the sought-after predominant
pattern in the data, while ignoring the idiosyncrasies in the training data. As such,
a well-fitted model in the testing set should be in the neighborhood of the model’s
accuracybasedonthetrainingdataset, thatis, themodel’saccuracyinthetestingset
shouldbeapproximatelyequaltothatofthemodel’saccuracyinthetrainingset. In
contrast, anoverfittedmodelinthetestingdatasetwillbefarfromtheneighborhood
of the model’s accuracy based on the training data set, and usually its prediction
performanceisveryhigh (good) inthetrainingsetandconsequentlylow (bad) inthe
testingset (Ratner2017).
The paradox of overfitting is defined as complex models that contain more
information about the training data, but less information about the testing data
(future data we want to predict). In statistical machine learning, overfitting is a
major issue andleads tosome seriousproblemsin research: (a) some relationships

### 6. 112 4 Overfitting, Model Tuning, and Evaluationof Prediction Performance
Fig.4.2...

112 4 Overfitting, Model Tuning, and Evaluationof Prediction Performance
Fig.4.2 Graphical
representationsofdifferent
levelsofbiasandvariance
the population is high-income and the other half is low-income. If you collected a
sample of high-income people, you would conclude that the entire city has high
income. Thismeansthatyourconclusionisheavilybiasedsinceyouonlysampled
peoplewithhighincome. Ontheotherhand, errorvariancereferstotheamountthat
theestimateoftheobjectivefunctionwillchangeusingadifferenttrainingdataset.
Inotherwords, theerrorvarianceaccountsforthedeviationofpredictionsfromone
repetitiontoanotherusingthesametrainingset. Ideally, whenastatisticalmachine
learningmodelwithlowerrorvariancepredictsavalue, thepredictedvalueshould
remainalmostthesame, evenwhenchangingfromonetrainingdatasettoanother;
however, ifthemodelhashighvariance, thenthepredictedvaluesofthestatistical
machine learning method are affected by the values of the data set. We provide a
graphicalvisualizationofbiasandvariancewithabull’seyediagram (see Fig.4.2).
We assume that the center ofthe target isa statistical machine learning model that
perfectly predicts the correct answers. As we move away from the bull’s eye, our
predictionsgetworse. Letusassumethatwecanrepeatourentirestatisticalmachine
learningmodelbuildingprocesstogetanumberofseparatehitsonthetarget. Each
hit represents an individual realization of our statistical machine learning model,
given the chance variability in the training data we gathered. Sometimes we accu-
ratelypredicttheobservationsofinterestsincewecapturedarepresentativesample
in our training data, while other times we obtain unreliable predictions since our
trainingdatamaybefullofoutliersornonrepresentativevalues. Thefourcombina-
tions of cases resulting from both high and low bias and variance are shown in
Fig.4.2.

### 7. 4.3 Cross-validation 117
It is important to point out that to reduce variability...

4.3 Cross-validation 117
It is important to point out that to reduce variability, we recommend
implementing the k-fold CV multiple times, each time using different complemen-
tary subsets to form the folds; the validation results are combined (e.g., averaged)
over the rounds (times) to give a better estimate of the statistical machine learning
modelpredictiveperformance.
4.3.3 The Leave-One-Out Cross-validation
The Leave-One-Out (or LOO) CV is very simple since each training data set is
created by including all the individuals except one, while the testing set only
includes the excluded individual. Thus, for n individuals in the full data set, we
havendifferent training andtesting sets. This CVschemewastesminimaldata, as
onlyoneindividualisremovedfromthetrainingset.
Regardingthek-foldcross-validationthatwasjustexplained, nmodelsarebuilt
from n individuals (samples) instead of k models, where n > k. Moreover, each
model is trained on n (cid:1) 1 samples rather than (k (cid:1) 1) n/k. In both cases, since k is
normally not too large and k < n, LOO is more computationally expensive than k-
fold cross-validation. In terms of prediction performance, LOO normally produces
highvariancefortheestimationofthetesterror. Becausen (cid:1)1 ofthensamplesis
usedtobuildeachstatistical machine learningmodel, thoseconstructedfromfolds
arevirtuallyidenticaltoeachotherandtothemodelbuiltfromtheentiretrainingset.
However, whenthelearningcurveissteepfortheevaluatedtrainingsize, thenfive-
orten-foldcross-validationusuallyoverestimatesthegeneralizationerror.
Learning curves (LC) areconsidered effective tools tomonitor theperformance
oftheemployeeexposedtoanewtask. LCsprovideamathematicalrepresentation
ofthelearningprocessthattakesplaceasthetaskisrepeated. Instatisticalmachine
learningthe LCisalineplotoflearning (y-axis) overexperience (x-axis).Learning
curvesareextensivelyusedinstatistical machinelearningforalgorithmsthatlearn
(their parameters) incrementally over time, such as deep neural networks. In
general, there is considerable empirical evidence suggesting that five- or ten-fold
cross-validationshouldbepreferredto LOO.
4.3.4 The Leave-m-Out Cross-validation
The Leave-m-Out (Lm O) CV is very similar to LOO as it creates all the possible
training/te stse!tsbyremovingmsamplesfromthecompleteset. Fornsamples, this
n
produces train-testpairs. Unlike LOOandk-fold, thetestsetswilloverlapfor
m
m>1.Forexample, ina Leave-2-Out CVwithadatasetwithfoursamples (I1, I2,

### 8. 120 4 Overfitting, Model Tuning, and Evaluationof Prediction Performance
Fig.4.6...

120 4 Overfitting, Model Tuning, and Evaluationof Prediction Performance
Fig.4.6 Schematicrepresentationofbootstrapcross-validation
this CVapproachhasabiassimilartoimplementingak¼twofoldcross-validation,
and as the training set becomes smaller, the bias becomes more problematic. To
understand this CV method, we provide a simple example of how the training and
testingsamplesareconstructed. Ifwehaveasamplewith12 individualsdenotedas
I1, I2, ..., I12, we will select B ¼ 5 bootstrap samples. Each bootstrap sample is
obtainedwithreplacementandtheindividualsthatappearineachonecorrespondto
the training sample; those that are not present will correspond to the testing set.
Figure 4.6 provides the five bootstrap samples; each training sample has the same
sizeastheoriginal, however, onlysomeindividualsappearineachbootstrapsample,
while those individuals that do not appear are included in the testing set. For
example, in the first fold, the training bootstrap sample contains seven different
individuals (I2, I3, I4, I6, I7, I8, and I9), while the testing set contains five
individuals (I1, I5, I10, I11, and I12). It is important to point out that since the
training sample has the same size as the original sample, some individuals in the
training sample are repeated at least twice; in the first fold, the individual I4 are
repeated three times, whereas I6, I7, and I8 are repeated twice. Finally, similar to
other methods, the statistical machine learning model is trained with each training
set; likewise, the prediction performance of the model is evaluated in each testing
set. Theaverageofthesesamplepredictionsisreportedastheestimatedtestingerror.
4.3.8 Incomplete Block Cross-validation
Incompleteblock (IB)CVshouldbeusedwhenthereare Jtreatmentsevaluatedin
Iblocksandthesametreatmentsareevaluatedinalltheblocks. Theideabehindthis

### 9. 4.5 Metricsforthe Evaluationof Prediction Performance 129
4.5.1 Quantitative Mea...

4.5 Metricsforthe Evaluationof Prediction Performance 129
4.5.1 Quantitative Measures of Prediction Performance
Beforeimplementingastatisticalmachinelearningmodel, weassumethatwehave
ourtrainingobservation{(x , y ),(x , y ),...,(x , y )} andweestimatef, as b f, with
1 1 2 2 n n
the chosen statistical machine learning model. Then we can make predictions for
b
eachoftheresponsevalues (y) withfðxÞandcomputethepredictedvaluesforeach
i i
ofthenobservationsinthetrainingset;withthesevalu (cid:6) eswecancalculatethemean
(cid:2) (cid:3)
Pn
b 2
square error (MSE) for thetraining dataset as E ¼1 y (cid:1) fðxÞ ;however,
n i i
i¼1
whatwereallywanttopredictarethevaluesforunseentestobservationsthatwere
not used to train the statistical machine learning model. Assuming that the unseen
testingsetisequalto{(x , y ),(x , y ),...,(x , y )}, the MSEfor
n+1 n+1 n+2 n+2 n+T n+T
thetestingdatasetshouldbecalculatedas
MSE ¼ 1
XnþT (cid:2)
y (cid:1) b fðxÞ
(cid:3)
2 , ð4:1Þ
TST T i i
i¼nþ1
b b
wherefðxÞisthepredictionthatf givestotheithobservation. The MSE witha
i TST
lowervaluewillhavebetterpredictions, whichmeansthatthepredictedvaluesare
veryclosetothetrueobservedvalues. Also, thesquarerootof MSE canbeused
TST
as a measure of prediction performance and is called root mean square error
(RMSE).
Pearson’scorrelationcoefficientisaverypopularmeasureofpredictionperfor-
manceinplantbreedingandcanbecalculatedas
(cid:2) (cid:3)
P
nþT b fðxÞ(cid:1) b fðxÞ ðy (cid:1) yÞ
r TST ¼rffi P ffiffiffiffiffi n ffi þ ffiffiffi T ffiffiffiffiffii (cid:2) ffi¼ffiffi b f nffiffiþ ð ffiffi x ffi1 ffiffi Þ ffiffiffi (cid:1) ffiffiffiffiffi b f i ffiffi ð ffiffi x ffiffiffi Þ ffiffi (cid:3) ffiffiffi 2 ffiffi q i ffi P ffiffiffiffiffi n ffi i þ ffiffiffi T ffiffiffiffiffi ð ffi i ffi y ffiffiffiffi (cid:1) ffiffiffiffiffi y ffiffiffi Þ ffiffi 2 ffiffi, ð4:2Þ
i¼nþ1 i i i¼nþ1 i i
b
wherefðxÞistheaverageofthe Tpredictionsthatconformtothetestingset, andy is
i i
theaverageofthe Ttrueobservedvalues. Inthiscase, thecloserthatthepredictions
areto1, thebettertheimplementedstatisticalmachinelearningmodelwillperform.
It is important to point out that Pearson’s correlation is defined between (cid:1)1 and
1. However, to be convinced that the observed and predicted values match, it is a
common practice to perform a scatter plot of predicted versus observed (or vice
versa) valuesandwhentheobservedandpredictedvaluesfollowastraightline (45(cid:6)
diagonal line) from the bottom left corner to the top right corner, this indicates a
perfectmatchbetweentheobservedandpredictedvalues. Forthisreason, Pearson’s
correlationasametricshouldbecomplemented with theinterceptandslope, since
the slope and intercept describe the consistency and the model bias, respectively
(Smith and Rose 1995; Mesple et al. 1996). To obtain the slope and intercept, the
observedvalues (asy) versusthepredictedvalues (asx) areregressedandinaddition

### 10. 4.5 Metricsforthe Evaluationof Prediction Performance 133
Table4.4 Confusionmatr...

4.5 Metricsforthe Evaluationof Prediction Performance 133
Table4.4 Confusionmatrix Predictedvalues
withtwoclasses
True False Sum
Observedvalues True tp fn tp+fn
False fp tn fp+tn
Sum tp+fp fn+tn n
tp denotes true positives, fp denotes false positives, fn denotes
false negatives, tn denotes true negatives, and n denotes total
numberofindividuals
example, ifthereare990 samplesofclass1 andonly10 ofclass2, theclassifiercan
easilyhaveabiastowardclass1.Iftheclassifierclassifiesallsamplesasclass1, its
accuracywillbe99%.Thisdoesnotmeanthatitisanappropriateclassifier, asithad
a100%errorwhenclassifyingthesamplesofclass2.Forthisreason, reportingthis
metric with those reported in Eqs. (4.9–4.11) is recommended in order to have a
better picture of the prediction performance of any statistical machine learning
method (Ratner 2017). Also, it is important to highlight that when the problem
onlyhastwoclasses, theconfusionmatrixisreducedto Table4.4.
From Table4.4 the PCCCiscalculatedastpþtn, whilethe S ¼ tp , S ¼ tn ,
n e tpþfn p tnþfp
and P¼ tp . Also, when there are only two classes, González-Camacho et al.
tpþfp
(2018) suggest calculating the Kappa coefficient (κ) or Cohen’s Kappa, which is
definedas
P (cid:1)P
κ ¼ 0 e,
1(cid:1)P
e
where P istheagreementbetweenobservedandpredictedvaluesandiscomputed
0
by the PCCC described above for two classes; P is the probability of agreement
e
calculated as P ¼tpþfn (cid:4) tpþfpþfpþtn (cid:4) fnþtn, where fp is the number of false
e n n n n
positives, and fn is the number of false negatives (Table 4.4). This statistic can
takeonvaluesbetween (cid:1)1 and1;avalueof0 meansthereisnoagreementbetween
the observed and predicted classes, while a value of 1 indicates perfect agreement
betweenthemodelpredictionandtheobservedclasses. Negativevaluesindicatethat
a prediction may be incorrect; however large negative values seldom occur when
workingwithpredictivemodels. Dependingonthecontext, a Kappavaluefrom0.30
to 0.50 indicates reasonable agreement (Kuhn and Johnson 2013). The Kappa
coefficient is appropriate when data are unbalanced, because it estimates the pro-
portion of cases that were correctly identified by taking into account coincidences
expectedfromchancealone (Fieldingand Bell1997).Itisimportanttopointoutthat
this statistic was originally designed to assess the agreement between two raters
(Cohen1960).
Anotherpopularmetricforbinarydataisthe Area Underthereceiveroperating
characteristic Curve (AUC–ROC) anditranksthepositivepredictionshigherthan
thenegative. The ROCcurveisdefinedasaplotof1(cid:1) specificityorfalsepositive
rate (FPR) asthex-axisversusitsmodelsensitivityasthey-axis. Foragivensetof

### 11. 134 4 Overfitting, Model Tuning, and Evaluationof Prediction Performance
Fig.4.1...

134 4 Overfitting, Model Tuning, and Evaluationof Prediction Performance
Fig.4.10 ROCcurvefor
thesyntheticdata
thresholds τ, it is an effective method for evaluating the quality or performance of
diagnostic tests, and is widely used in statistical machine learning to evaluate the
predictionperformanceoflearningalgorithms. Sincethemathematicalconstruction
ofthismetricisnotrequiredinthisbook, weillustrateitscalculationwithonesimple
example. Assume that the observed (y), predicted probabilities (pi) and predicted
values (ŷ) obtained after implementing a statistical machine learning model are
y ¼ {1, 0, 1, 1, 0, 0, 1, 1, 0, 1}, pi ¼ {0.6, 0.55, 0.8, 0.78, 0.3, 0.42, 0.9, 0.45,
0.3,0.88}, andby¼f1,1,1,1,0,0,1,0,0,1};then, usingthefollowing Rcode, we
canobtainmanymetricsforbinarydata (Fig.4.10):
########################Librariesrequired########################
library (caret)
library (p ROC)
##Observed (y), predictedprobability (pi) and predictedvalues of
syntheticdata##
y=c (1,0,1,1,0,0,1,1,0,1)
pi=c (0.6,0.55,0.8,0.78,0.3,0.42,0.9,0.45,0.3,0.88)
yhat=c (1,1,1,1,0,0,1,0,0,1)
xtab<-table (y, yhat)
confusion Matrix (xtab)
plot.roc (y, pi)#####Thismakethe ROCcurveplot
Confusion Matrixand Statistics
yhat
y 01
031
115

### 12. 4.5 Metricsforthe Evaluationof Prediction Performance 135
Accuracy:0.8
95%CI:(0....

4.5 Metricsforthe Evaluationof Prediction Performance 135
Accuracy:0.8
95%CI:(0.4439,0.9748)
No Information Rate:0.6
P-Value[Acc>NIR]:0.1673
Kappa:0.5833
Mcnemar's Test P-Value:1.0000
Sensitivity:0.7500
Specificity:0.8333
Pos Pred Value:0.7500
Neg Pred Value:0.8333
Prevalence:0.4000
Detection Rate:0.3000
Detection Prevalence:0.4000
Balanced Accuracy:0.7917
'Positive'Class:0
It is important to point out that when there are more than two classes, these
metrics (accuracy, sensitivity, specificity, etc.) can be calculated on a “one-versus-
all”basisthatconsistsofusingeachclassversusthepooloftheremainingclasses, as
was illustrated for the confusion matrix with more than two classes (James et al.
2013).
Whenthestatisticalmachinelearningmodeldiscriminatescorrectlybetweenthe
twogroups, itproducesacurvethatcoincideswiththeleftandtopsidesoftheplot.
Underthisscenario, theperfectmodelwouldhave100%sensitivityandspecificity,
andthe ROCcurvewouldbeasinglestepbetween (0,0) and (0,1) andwouldremain
constant from (0,1) to (1,1); this implies that the area under the ROC curve of the
modelwouldbe1.Ingeneral, thelargertheareaunderthe ROCcurve, thebetterthe
modelintermsofpredictionperformance. Ontheotherhand, acompletelyuseless
statisticalmachinelearningalgorithmwouldgiveastraightline (45(cid:6) diagonalline)
from the bottom left corner to the top right corner of the plot. Different statistical
machine learning models or the same model with different training sets or
hyperparameterscanbecomparedbysuperimposingtheir ROCcurvesinthesame
graph. Inpractice, mostofthetimethevaluesinthetwogroupsoverlap, sothecurve
oftenliesbetweentheseextremes.
From this ROC curve, we can obtain a global assessment of the prediction
performance of the statistical machine learning method by measuring the area
underthereceiveroperatingcharacteristiccurve. Thisareaisequaltotheprobability
thatarandomindividual (person) ofthesamplewiththepresenceofthetargethasa
highervalueofthemeasurementthanarandomindividualwithoutthetarget.
Whenastatisticalmachinelearningmodelisunabletodiscriminatebetweenthe
positive and negative classes, this means that it has low discriminatory power.
Therefore, onlyinstatisticalmachinelearningmodelsthathadgooddiscriminatory
power we can be confident of the predictions they provide and furthermore those
modelsthatprovideacurvethatliesconsiderablyabovethecurvewillbebetter.

### 13. References 139
Lorena AC, de Carvalho ACPLF (2008) Evolutionary tuning of SVM pa...

References 139
Lorena AC, de Carvalho ACPLF (2008) Evolutionary tuning of SVM parameter values in
multiclassproblems. Neurocomputing71:3326–3334
Lujan-Moreno GA, Howard PR, Rojas OG, Montgomery DC(2018)Designofexperimentsand
responsesurfacemethodologytotunemachinelearninghyperparameters, witharandomforest
case-study. Expert Syst Appl109:195–205
Matthews BW(1975)Comparisonofthepredictedandobservedsecondarystructureof T4 phage
lysozyme. Biochim Biophys Acta Protein Struct405:442–451
Mc Kay MD(1992)Latinhypercubesamplingasatoolinuncertaintyanalysisofcomputermodels.
In:Swain JJ, Goldsman D, Crain RC, Wilson JR(eds)Proceedingsofthe24 thconferenceon
wintersimulation (WSC1992).ACM, New York, pp557–564
Mesple F, Troussellier M, Casellas C, Legendre P(1996)Evaluationofsimplestatisticalcriteriato
qualifyasimulation. Ecol Model88:9–18
Montesinos-López OA, Montesinos-López A, Pérez-Rodríguez P, delos Campos G, Eskridge KM,
Crossa J(2015 a)Thresholdmodelsforgenome-enabledpredictionofordinalcategoricaltraits
inplantbreeding. G35(1):291–300
Montesinos-López OA, Montesinos-López A, Crossa J, Burgueño J, Eskridge K(2015 b)Genomic-
enabled prediction of ordinal data with Bayesian logistic ordinal regression. G3
5(10):2113–2126.https://doi.org/10.1534/g3.115.021154
Montesinos-López A, Montesinos-López OA, Gianola D, Crossa J, Hernández-Suárez CM(2018 a)
Multi-environmentgenomicpredictionofplanttraitsusingdeeplearnerswithadensearchitec-
ture. G38(12):3813–3828.https://doi.org/10.1534/g3.118.200740
Montesinos-López OA, Montesinos-López A, Crossa J, Gianola D, Hernández-Suárez CM et al
(2018 b)Multi-trait, multi-environmentdeeplearningmodelingforgenomic-enabledprediction
ofplanttraits. G38(12):3829–3840.https://doi.org/10.1534/g3.118.200728
Piñeiro G, Perelman S, Guerschman JP, Paruelo JM (2008) How to evaluate models:
observedvs.predictedorpredictedvs.observed?Ecol Model216:316–322
Ratner B (2017) Statistical and machine-learning data mining. Techniques for better predictive
modelling and analysis of big data, 3 rd edn. CRC Press Taylor & Francis Group, Boca
Raton, FL
Renukadevi NT, Thangaraj P(2014)Performanceanalysisofoptimizationtechniquesformedical
imageretrieval. JTheor Appl Inf Technol59:390–399
Shalev-Shwartz S, Ben-David S(2014)Understandingmachinelearningfromtheorytoalgorithms.
Cambridge Universitypress, New York
Shmueli G(2010)Toexplainortopredict?Stat Sci25(3):289–310
Smith EP, Rose KA(1995)Modelgoodness-of-fitanalysisusingregressionandrelatedtechniques.
Ecol Model77:49–64
Wu S, Harris T, Mcauley K(2007)Theuseofsimplifiedormisspecifiedmodels:linearcase. Canad
JChem Eng85:386–398
Open Access Thischapterislicensedunderthetermsofthe Creative Commons Attribution4.0
International License (http://creativecommons.org/licenses/by/4.0/), which permits use, sharing,
adaptation, distributionandreproductioninanymediumorformat, aslongasyougiveappropriate
credittotheoriginalauthor (s) andthesource, providealinktothe Creative Commonslicenseand
indicateifchangesweremade.
Theimagesorotherthirdpartymaterialinthischapterareincludedinthechapter's Creative
Commons license, unless indicated otherwise in a credit line to the material. If material is not
included in the chapter's Creative Commons license and your intended use is not permitted by
statutoryregulationorexceedsthepermitteduse, youwillneedtoobtainpermissiondirectlyfrom
thecopyrightholder.

### 14. Chapter 4
fi
Over tting, Model Tuning, and Evaluation
of Prediction Performance
...

Chapter 4
fi
Over tting, Model Tuning, and Evaluation
of Prediction Performance
4.1 The Problem of Overfitting and Underfitting
The overfitting phenomenon occurs when the statistical machine learning model
learns the training data set so well that it performs poorly on unseen data sets. In
otherwords, thismeansthatthepredictedvaluesmatchthetrueobservedvaluesin
the training data set too well, causing what is known as overfitting. Overfitting
happens when a statistical machine learning model learns the systematic and noise
(randomfluctuations) partsinthetrainingdatatotheextentthatitnegativelyimpacts
theperformanceofthestatisticalmachinelearningmodelonnewdata. Thismeans
thatthestatisticalmachinelearningmodeladaptsverywelltothenoiseaswellasto
thesignalthatispresentinthetrainingdata. Theproblemisthattheseconceptsdo
not apply to independent (new) data and negatively affect the model’s ability to
generalize. Overfitting is more probable when learning a loss function from a
complexstatisticalmachinelearningmodel (withmoreflexibility).Forthisreason,
manynonparametricstatisticalmachinelearningmodelsalsoincludeconstraintsin
thelossfunctiontoimprovethelearningprocessofthestatisticalmachinelearning
models. For example, artificial neural networks (ANN), mentioned later, are a
nonparametricstatisticalmachinelearningmodelthatisveryflexibleandissubject
tooverfittingtrainingdata. Thisproblemcanbeaddressedbydroppingout (setting
to zero) the weights of a certain percentage of hidden units in order to avoid
overfitting.
On the other hand, an underfitted phenomenon occurs when few predictors are
includedinthestatisticalmachinelearningmodel, i.e., itisaverysimplemodelthat
poorlyrepresentsthecompletepictureofthepredominantdatapattern. Thisproblem
also arises when the training data set is too small or not representative of the
population data. An underfitted model does a poor job of fitting the training data
and for this reason it is not expected to satisfactorily predict new data points. This
implies that the predictions using unseen data are weak, since individuals are
perceivedasstrangersunfamiliarwiththetrainingdataset.
©The Author (s)2022 109
O.A.Montesinos Lópezetal., Multivariate Statistical Machine Learning Methods
for Genomic Prediction, https://doi.org/10.1007/978-3-030-89010-0_4

### 15. 4.2 The Trade-Off Between Prediction Accuracyand Model Interpretability 111
that...

4.2 The Trade-Off Between Prediction Accuracyand Model Interpretability 111
thatseemstatisticallysignificantareonlynoise,(b) thecomplexityofthestatistical
machine learning model is very large for the amount of data provided, and (c) the
modelingeneralisnotreplicableandpredictspoorly.
Sincethemaingoalofdevelopingandimplementingstatisticalmachinelearning
methods is to predict unseen data not used for training the statistical machine
learning algorithm, researchers are mainly interested in minimizing the testing
error (generalization error applicable to future samples) instead of minimizing the
training error that isapplicabletotheobserved data used for training thestatistical
machinelearningalgorithm.
According to Shalev-Shwartz and Ben-David (2014), if the learning fails, these
aresomeapproachestofollow:
1. Increasethesamplesizeofthetrainingset.
2. Modifythehypothesisby (a) enlargingit,(b) reducingit,(c) completelychanging
it, and (d) changingtheparametersbeingused. Weunderstandahypothesisasthe
models and their parameters under evaluation. This point is very important to
reachareasonablemodelforyourdata.
3. Changethefeaturerepresentationofthedata.
4. Changethestatisticalmachinelearningalgorithmused.
4.2 The Trade-Off Between Prediction Accuracyand Model
Interpretability
Accuracy is the ability of a statistical machine learning model to make correct
predictions and those models with more complexity (called flexible models) are
betterintermsofaccuracy, whilethesimple, lesscomplexmodels (calledinflexible
models) are less accurate but more interpretable. Interpretability indicates to what
degreethemodelallowsforhumanunderstandingofnaturalphenomena. Forthese
reasons, when the goal of the study is prediction, flexible models should be used;
however, when the goal of the study is inference, inflexible models are more
appropriatebecausetheymoreeasilyinterprettherelationshipbetweentheresponse
variables and the predictor variables. As the complexity of the statistical machine
learning model increases, the bias is reduced and the variance increases. For this
reason, whenmoreparametersareincludedinthestatisticalmachinelearningmodel,
the complexity of the model increases and the variance becomes the main concern
while the bias steadily falls. For example, James et al. (2013) state that the linear
regression model is a relatively inflexible method because it only generates linear
functions, while the support vector machine method is one of the most flexible
statisticalmachinelearningmethods.
Beforeprovidingananalyticalinterpretationofthetrade-offbetweenthebiasand
variance, we must understand the meaning of both concepts. Bias is the difference
between the expected prediction of our statistical machine learning model and the
trueobservedvalues. Forexample, assumethatyoupollaspecificcitywherehalfof

### 16. 4.2 The Trade-Off Between Prediction Accuracyand Model Interpretability 113
Burg...

4.2 The Trade-Off Between Prediction Accuracyand Model Interpretability 113
Burger (2018) concludesthatthebestscenarioistheonewithlowbiasandlow
variance, since samples are acceptable representatives ofthe population (Fig. 4.2),
whileinthecaseofhighbiasandlowvariance, thesamplesarefairlyconsistent, but
notparticularlyrepresentative ofthepopulation (Fig.4.2).However, whenthereis
lowbiasandhighvariance, thesamplesvarywidelyintheirconsistency, andonly
somemayberepresentativeofthepopulation (Fig.4.2).Finally, withhighbiasand
high variance, the samples are somewhat consistent, but unlikely to be representa-
tiveofthepopulation (Fig.4.2).
Ifwedenotethevariablewearetryingtopredictasyandourcovariatesasx, we
i
may assumethat there isarelationship between y andx, asthatgiven in Eq.(1.1)
i
from Chap.1, wheretheerrortermisnormallydistributedwithameanofzeroand
varianceσ2.Theexpectedpredictionerrorforanewobservationwithvaluex, using
aquadraticlossfunction, isgivenby Hastieetal.(2008, page223):
(cid:2) (cid:3) (cid:2) (cid:2) (cid:3) (cid:3) n (cid:2) (cid:3) o
E y (cid:1) b ffxg 2 ¼Eðy (cid:1) ffxgÞ2þ E b ffxg (cid:1) ffxg 2 þE b fðxÞ(cid:1)E b ffxg 2
i i i i i i
h (cid:2) (cid:3) i (cid:2) (cid:3)
b 2 b
¼VarðyÞþ Bias ffxg þVar fðxÞ
i i
h (cid:2) (cid:3) i (cid:2) (cid:3)
¼VarðEÞþ Bias b ffxg 2 þVar b fðxÞ ,
i i
where Biasistheresultofmisspecifyingthestatisticalmodelf. Estimationvariance
(thethirdterm) istheresultofusingasampletoestimatef. Thefirsttermistheerror
(irreducibleerror) thatresultsevenifthemodeliscorrectlyspecifiedandaccurately
estimated. Thisirreducibleerroristhenoiseterminthetruerelationshipthatcannot
fundamentally be reduced by any model. Given the true model and infinite data to
train (calibrate) it, we should be able to reduce both thebias and variancetermsto
0. However, in a world with imperfect models and finite data, there is a trade-off
betweenminimizingthebiasandminimizingthevariance. Theabovedecomposition
revealsasourceofthedifferencebetweenexplanatoryandpredictivemodeling:In
explanatory modeling, the focus is on minimizing bias to obtain the most accurate
representation of the underlying theory. In contrast, predictive modeling seeks to
minimize thecombination of bias and estimation variance, occasionally sacrificing
theoreticalaccuracyforimprovedempiricalprecision (Shmueli2010).
These four aspects impact every step of the modeling process, such that the
resultingfismarkedlydifferentintheexplanatoryandpredictivecontexts.
Let us assume that f is a reasonable operationalization of the true function (F)
relating constructs X and Y. Choosing a function f (cid:3) that is intentionally biased in
place of f is very undesirable from a theoretical–explanatory standpoint. However,
theelectionoff (cid:3) isdesirabletofunderthepredictionapproach. Weshowthisusing
the statistical model y ¼ β x + β x + β x + E, which is assumed to be correctly
1 1 2 2 3 3
specifiedwithrespectto F.Usingdata, weobtaintheestimatedmodel b f, whichhas
thefollowingproperties:

### 17. 114 4 Overfitting, Model Tuning, and Evaluationof Prediction Performance
Bias¼0
...

114 4 Overfitting, Model Tuning, and Evaluationof Prediction Performance
Bias¼0
(cid:2) (cid:3) (cid:2) (cid:3) (cid:4) (cid:5)
Var b fðxÞ ¼Var bβ x þ bβ x þ bβ x ¼σ2 x T XXT 21 x,
i 1 1 2 2 3 3
where x is the vector x ¼ [x , x , x ]T and X is the design matrix based on all
1 2 3
predictors. Combining the squared bias with the variance gives, as expected, the
predictionerror (EPE).
(cid:2) (cid:3) (cid:4) (cid:5) h (cid:4) (cid:5) i
E y (cid:1) b ffxg 2 ¼σ2þ0þσ2 x T XXT 21 x=σ2 1þx T XXT (cid:1)1 x :
i
Incomparison, considertheestimatedunderspecifiedform b f (cid:3) ðxÞ¼x bγ.Thebias
i 1
andvariancehereare
(cid:2) (cid:3) (cid:4) (cid:5)
Bias¼E b ffxg (cid:1) ffxg¼x x x T (cid:1)1 x Tðβ x þβ x þβ x Þ
i i 1 1 1 1 1 1 2 2 3 3
(cid:1)ðβ x þβ x þβ x Þ
1 1 2 2 3 3
(cid:2) (cid:3) (cid:4) (cid:5)
Var b fðxÞ ¼σ2 x x x T (cid:1)1 x T
i 1 1 1 1
Combiningthesquaredbiaswiththevariance EPEisequalto
(cid:2) (cid:3) h (cid:4) (cid:5) i
E y (cid:1) b ffxg 2 ¼ x x x T (cid:1)1 x Tðx β þx β Þ(cid:1)ðx β þx β Þ 2
i 1 1 1 1 2 2 3 3 2 2 3 3
h (cid:4) (cid:5) i
þσ2 1þx x x T (cid:1)1 x T :
1 1 1 1
Althoughthebiasoftheunderspecifiedmodelf (cid:3)(x) islargerthanthatoff{x}, its
i i
variance can be smaller, and in some cases, so small that the overall EPE will be
lowerfortheunderspecifiedmodel. Wuetal.(2007) showedthegeneralresultforan
underspecified linear regression model with multiple predictors. In particular, they
showedthattheunderspecifiedmodelthatleavesoutqpredictorshasalower EPE
whenthefollowinginequalityholds:
qσ2 >βTXTðI(cid:1)H ÞX β
2 2 1 2 2
This means thatthe underspecifiedmodel producesmoreaccuratepredictions, in
termsoflower EPE, inthefollowingsituations:(a) whenthedataareverynoisy (large
σ2);(b) whenthetrueabsolutevaluesoftheexcludedparameters (inourexample,β
2
and β ) are small; (c) when the predictors are highly correlated; and (d) when the
3
samplesizeissmallortherangeofleft-outvariablesissmall (Shmueli2010).
Hagertyand Srinivasan (1991) nicelysummarizethissituation:“Wenotethatthe
practice in applied research of concluding that a model with a higher predictive
validityis“truer,”isnotavalidinference. Thispapershowsthataparsimoniousbut
less true model can have a higher predictive validity than a truer but less parsimo-
niousmodel.”

### 18. 116 4 Overfitting, Model Tuning, and Evaluationof Prediction Performance
set (te...

116 4 Overfitting, Model Tuning, and Evaluationof Prediction Performance
set (testing set) is used to study how well that statistical machine learning model
performsonunseendata. Forexample,80%ofthedatacanbeusedfortrainingthe
modelandtheremaining20%ofthedatafortestingit. Oneweaknessofthehold-out
(validation) set approach is that it depends on just one training-testing split and its
performancedependsonhowthedataaresplitintothetrainingandtestingsets.
4.3.2 The k-Fold Cross-validation
Ink-fold CV, thedatasetisrandomlydividedintokcomplementaryfolds (groups)
of approximately equal size. One of the subsets is used as testing data and the rest
(k (cid:1)1) astrainingdata. Thenk (cid:1)1 foldsareusedfortrainingthestatisticalmachine
learning model and the remaining fold for evaluating the out-of-sample prediction
performance. For these reasons, the statistical machine learning model is fitted
k times using a different partition (fold) as the testing set and the remaining k (cid:1) 1
asthetrainingset. Finally, thearithmeticmeanofthekfoldsisobtainedandreported
asthepredictionperformanceofthestatisticalmachinelearningmodel (see Fig.4.4).
This method is very accurate because it combines k measures of fitness resulting
fromthektrainingandtestingdatasetsintowhichtheoriginaldatasetwasdivided,
butatthecostofmorecomputationalresources. Inpractice, thechoiceofthenumber
offoldsdependsonthemeasurementofthedataset, although5 or10 foldsarethe
mostcommonchoices.
Fig.4.4 Schematicrepresentationofthek-foldcross-validationwithcomplementarysubsetswith
k¼5

### 19. 118 4 Overfitting, Model Tuning, and Evaluationof Prediction Performance
!
4
I3,...

118 4 Overfitting, Model Tuning, and Evaluationof Prediction Performance
!
4
I3, and I4), thetotalnumberoftraining-testingsetsisequalto ¼6;thismeans
2
thatthesixtestingsetsare:[I3, I4][I1, I2],[I2, I4][I1, I3],[I2, I3][I1, I4],[I1, I4]
[I2, I3],[I1, I3] [I2, I4], and [I1, I2] [I3, I4], while the training sets are the comple-
mentaryelementsofeachtestingset.
4.3.5 Random Cross-validation
In this type of CV, the number of partitions (independent training-testing data set
splits) is defined by the user, and more partitions are better. Each partition is
generated by randomly dividing the whole data set into two subsets: the training
(TRN) datasetandthetesting (TST) dataset. Thepercentageofthewholedataset
assigned to the TRN and TST data sets is also fixed by the user. For example, for
each random partition, the user can decide that 80% of the whole data set can be
assigned tothe TRN data set and the remaining 20% to the TST data set. Random
cross-validation is different from k-fold cross-validation because the partitions are
notmutuallyexclusive;thismeansthatintherandomcross-validationapproach, one
observation can appear in more than one partition. Consequently, some samples
cannotbeevaluated, whereasotherscanbeevaluatedmorethanonce, meaningthat
the testing and training subsets can be superimposed (Montesinos-López et al.
2018 a, b). To control the randomness for reproducibility, we recommend using a
specific seed in the random number generator. We recommend using at least ten
random partitions to obtain enough accuracy in the estimate of prediction
performance.
4.3.6 The Leave-One-Group-Out Cross-validation
The Leave-One-Group-Out (LOGO) CV is useful when individuals are grouped
(inenvironmentsoryears, orevenanothercriterion), wherethenumberofgroups (g)
isatleasttwoandtheinformationofg (cid:1)1 groupsareusedasthetrainingsetwhile
allindividualsoftheremaininggroupareusedasthetestingset. Forexample, inthe
contextofgenomicselection, whentheplantbreederisinterestedinpredictingthese
linesinanotherenvironment, thesame (ordifferent) lineswerefrequentlyevaluated
ing environments oryears, that representthe groups. Jarquínet al.(2017) denotes
this type of CV strategy as CV1 in the context of plant breeding. Under this
approach, thepredictionsarereportedforeachoftheggroupsbecausethescientist
is interested in the prediction performance of each environment. Many times, the
groups are the years under study and the aim is to predict the information of a
completeyear. However, whenthegroupsareyearsandifwesuspectthatthereisa
considerablecorrelationbetweenobservationsthatarenearintime. Therefore, itis

### 20. 4.3 Cross-validation 121
Table4.1 TRNdatasetfor Block Treatmentsperblock
I¼3 blo...

4.3 Cross-validation 121
Table4.1 TRNdatasetfor Block Treatmentsperblock
I¼3 blocksand J¼10
1 1 3 5 7 8 9 10
treatmentswith70%ofdata
fortraining 2 2 3 4 5 6 7 9
3 1 2 4 6 7 8 10
CVmethodisthatsometreatmentsshouldbepresentinsomeblocksbutabsentin
others, whereas the same treatment should be present in at least one environment
(block). The theory of incomplete block designs developed in the experimental
designs statistical area can be used to construct the training set. For example,
under a balanced incomplete block (BIB) design, the term incomplete means that
alltreatmentsineachblockcannotbeevaluated, whereasbalancedmeansthateach
pair of treatments occur together λ times. The training set is constructed by first
defining the % of individuals in the TRN set using the equation s I ¼ Jr ¼ N ,
TRN
where Jrepresentsthenumberoftreatmentsunderstudy, Irepresentsthenumberof
blocks under study, r denotes the number of repetitions of each treatment, and s
denotes the treatments per block. For example, suppose that we had J ¼ 10 treat-
mentsand I¼3 blocks (thatis,30 individuals), andwedecidedtouse N ¼21
TRN
(70%) ofthetotalindividualsinthe TRNset. Therefore, thenumberoftreatmentsby
blockcanbeobtainedbysolving (s I¼N ) fors, whichresultsins¼N /I.This
TRN TRN
meansthats¼21/3¼7 treatmentsperblock. Then, thecorrespondingelementsfor
thetrainingsetcanbeobtainedwiththefunctionfind. BIB(10,3,7) ofthepackage
crossdes of the R statistical software. The numbers used in the function find. BIB()
denotethetreatments, theblocks, andthetreatmentsperblock, respectively. Finally,
thetreatmentsthatmakeupthe TRNsetareshownin Table4.1.
Accordingto Table4.1, itisclearthateachtreatmentispresentintwoblocksand
missinginoneblock. Forexample, in Block1 thetestingsetincludestreatments2,4,
and6;in Block2, thetestingsetiscomposedoftreatments1,8, and10;andin Block
3, thetestingsetiscomposedoftreatments3,5, and9.
4.3.9 Random Cross-validation with Blocks
Randomcross-validationwithblockswasproposedby Lopez-Cruzetal.(2015) and
belongs to the so-called replicated TRN-TST cross-validation that appears in the
publicationof Daetwyler etal.(2012), since some individuals cannever bepart of
the training set. This algorithm, like the incomplete block cross-validation, is
appropriatewhenweareinterestedinevaluating Jlinesin Iblocksorenvironments
andtriestomimicapredictionproblemfacedbybreedersinincompletefieldtrials
wherelinesareevaluatedinsome, butnotall, targetenvironments. Thealgorithmfor
constructingthe TRN-TSTsetsisdescribedbythefollowingsteps:Step1.Calculate
the total number of observations under study as N ¼ J (cid:4) I; Step 2. Define the
proportionofobservationsusedfortrainingandtesting, thatis, P and P ;Step
TRN TST
3.Calculatethesizeofthetestingset N ¼N(cid:4)P ;Step4.Choose N linesat
TST TST TST

### 21. 122 4 Overfitting, Model Tuning, and Evaluationof Prediction Performance
Table4....

122 4 Overfitting, Model Tuning, and Evaluationof Prediction Performance
Table4.2 TRN-TSTdatasetsfor J¼10 linesand I¼3 environments
Environment Lines
L1 L2 L3 L4 L5 L6 L7 L8 L9 L10
1 TRN TST TRN TRN TST TRN TRN TST TRN TRN
2 TST TRN TRN TRN TRN TRN TST TRN TST TRN
3 TRN TRN TST TST TRN TRN TRN TRN TRN TST
random without replacement if J (cid:5) N , and with replacement otherwise; Step
TST
5. Each chosen line will then be assigned to one of the I environments chosen at
random without replacement; Step 5. All the selected lines and environments will
form the training set, while the lines and environments that were not chosen will
formthecorrespondingtestingset;and Step6.Steps1–5 arerepeateddependingon
the number of TRN-TST partitions required (Lopez-Cruz et al. 2015). This CV is
called CV2 in Jarquín et al. (2017). Next, we assume that we have ten lines or
treatments and three environments or blocks that will form the corresponding
trainingtestingsetsforonlyonepartition:Step1.Thetotalnumberofobservations
understudyis N¼10(cid:4)3¼30;Step2.Wedefine P ¼0.7 and P ¼0.3;Step
TRN TST
3.Thesizeofthetestingsetis N ¼30(cid:4)0.3¼9;Step4.Since J¼10(cid:5)N ¼9,
TST TST
weselectedthefollowinglinesatrandomwithoutreplacement:L1, L2, L3, L4, L5,
L6, L7, L8, L9, and L10; and Step 5. Each chosen line was assigned to one of the
I¼3 environmentsrandomlychosenwithoutreplacement, asshownin Table4.2.It
isimportanttopointoutthatthis CVstrategyonlydiffersfromtheincompleteblock
cross-validationinthewaythelinesareallocatedtoblocks.
4.3.10 Other Options and General Comments
on Cross-validation
Itisimportanttohighlightthatwhenthedatasetisconsiderablylarge, itisbetterto
randomlysplititintothreeparts:atrainingset, avalidationset (ortuningset), anda
testing set. The training set and testing set are used as explained before, while the
validation (tuningset) setisusedtoestimatethepredictionerrorformodelselection,
which is the process of estimating the performance of different models in order to
choose the best one, or to evaluate the chosen statistical machine learning model
with a range of values of tuning hyperparameters to select the combination of
hyperparameters with the best prediction performance and then use these
hyperparameters (orbestmodel) toevaluatethepredictionperformanceinthetesting
set (Fig.4.7).Itisimportanttopointoutthat Fig.4.7 showsonlyonerandomsplitof
thedataintermsofthetraining, testing, andvalidationsets.
Forexample, assumethatourdatasethas50,000 rows (observations) andthatwe
have decided to use 5000 of them for the testing set and another 5000 for the
validation set. This means that 40,000 rows are left for the training set. At this
point, we train our statistical machine learning model with each component of the

### 22. 4.3 Cross-validation 123
Fig.4.7 Schematicrepresentationofthetraining, validatio...

4.3 Cross-validation 123
Fig.4.7 Schematicrepresentationofthetraining, validation (tuning), andtestingsetsproposedby
Cook (2017)
gridofhyperparametersofthetrainingsetandevaluatethepredictionperformance
onthevalidationset, asshowninthemiddleof Fig.4.7.Finally, wewillpickthebest
model (best subset of hyperparameters) in terms of prediction performance in the
validationsetandwearereadytoevaluatethepredictionperformanceinthetesting
set. Then, wewillreportthetestingerroronthetestingset, ascanbeobservedatthe
bottomof Fig.4.7(Cook2017).Underthisapproach, thetestingandvalidationsets
have approximately the same size to guarantee a similar out-of-sample prediction
performance. Sinceitisdifficulttogivegeneralrulesonhowtochoosethenumber
of observations in each of the three parts, a typical number might be 50% for
training, and 25% each for validation and testing (Hastie et al. 2008) or 70%,
15%, and 15% for training, validation, and testing, respectively. Another way to
find the optimal setting of hyperparameters using the grid search, which is very
commonindeeplearning, consistsofpickingvaluesforeachparameterfromafinite
setofoptions[e.g., numberofepochs (100,150,200,250,300);batchsizes (25,50,
75, 100, 125); number of layers (1, 2, 3, 4, 5); and types of activation functions
(RELU, Sigmoid), ...] and training the statistical machine learning model with
every permutation of hyperparameter choices using the training set. Then, the
combination of hyperparameters with the best prediction performance on the vali-
dation set is chosen, and we report the prediction performance of the best selected
model (set of hyperparameters) in the testing set (Buduma 2017). The aforemen-
tioned examples use the validation data set as a proxy measure of the accuracy
duringthe hyperparameter optimization process. However, it can also be usedas a
proxy measure of the accuracy for model selection, and instead of using a grid of
hyperparameters, we can use a set of different statistical machine learning models;
here, thebestmodelischoseninsteadofthebestcombinationofhyperparameters. It
is important to understand that when you have more than one random partition
(training-testing-validation), as shown in Fig. 4.8, the same process provided in
Fig. 4.7 is followed, however, the average of all the partitions is reported as a
measure of prediction performance. Also, if more precision is required in the
estimated prediction performance, you can repeat the process given in Fig. 4.8

### 23. 124 4 Overfitting, Model Tuning, and Evaluationof Prediction Performance
Fig.4.8...

124 4 Overfitting, Model Tuning, and Evaluationof Prediction Performance
Fig.4.8 Fivepartitionsoftraining-validation-testing
multiple times and report the average of all repetitions as a measure of prediction
performance.
Asmentionedabove, thisapproach (Figs.4.7 and4.8) isusedforalargedataset
however, in the case of a smaller data set, we suggest modifying this approach to
avoidwastingtoomuchtrainingdatainvalidationsets. Thismodificationconsistsof
performing an inner cross-validation approach since the training set is split into
complementarysubsets, andeachmodelistrainedagainstadifferentcombinationof
thesesubsets, whichissubsequentlyvalidatedagainsttheremainingparts. Oncethe
modelhyperparametershavebeenselected, afinalmodelistrained withthewhole
trainingset (refitted) andthegeneralizedpredictionperformanceismeasuredonthe
testingset. Thisapproachwasappliedby Montesinos-Lópezetal.(2018 a, b), who
alsosplittheoriginaldataintotrainingandtestingsets. Subsequently, eachtraining
set was split again and 80% of the data was used for training a grid of
hyperparameters while the remaining 20% was used for validating (tuning) the
prediction performance and selecting the best combination of hyperparameters
with the best prediction performance. At this point, the deep learning algorithm
was refitted with the whole training set, and with this they evaluated the out-of-
sample prediction. They called the conventional training-testing partition outer
cross-validation, while the split performed in each training set used for
hyperparameter tuning was called inner cross-validation. It should be highlighted
thattherearenodifferencesbetweenouterandinner CVandtraining-validation-test.
Finally, itshould also bementioned thatany type ofthecross-validationstrategies
mentionedinthissection (random CV, k-fold CV, Bootstrap CV, IBCV, etc.,) can
beusedinbothouterandinner CV, suchasisthecasewiththefive-fold CV.
4.4 Model Tuning
A hyperparameter is a parameter whose value is set before the learning process
begins. Hyperparametersgovernmanyaspectsofthebehaviorofstatisticalmachine
learning models, such as their ability to learn features from data, the models’
exhibiteddegreeofgeneralizabilityinperformance when presented withnewdata,
as well as the time and memory cost of training the model, since different
hyperparameters often result in models with significantly different performance.
This means that tuning hyperparameter values is a critical aspect of the statistical
machinelearningtrainingprocessandakeyelementforthequalityoftheresulting

### 24. 4.4 Model Tuning 125
Fig.4.9 Schematicrepresentationofthetuningprocessproposedby...

4.4 Model Tuning 125
Fig.4.9 Schematicrepresentationofthetuningprocessproposedby Kuhnand Johnson (2013)
predictionaccuracies. However, choosingappropriatehyperparametersischalleng-
ing (Montesinos-Lópezetal.2018 a).Hyperparametertuningfindsthebestversion
ofastatisticalmachinelearningmodelbyrunningmanytrainingsetsontheoriginal
data set using the algorithm and ranges of values of hyperparameters as specified.
The hyperparameter values that provide the best performance in out-of-sample
predictionevaluatedbythechosenmetricarethenselected.
There are many ways of searching for the best hyperparameters. However, a
general approach defines a set of candidate values for each hyperparameter. Each
valueofthissetofcandidatevaluesisthenappliedwitharesampleofthetrainingset
ofthechosenstatisticalmachinelearningmethod, whereweaggregateallthehold-
outpredictionsfromwhichthebesthyperparametersarechosenandrefitthemodel
with the entire set (Kuhn and Johnson 2013). A schematic representation of the
tuning process proposed by Kuhn and Johnson (2013) is given in Fig. 4.9. It is
importanttohighlightthatthisprocessshouldbeperformedcorrectlybecausewhen
the same data are used for training and evaluating the prediction performance, the
predictionperformanceobtainedisextremelyoptimistic.
For example, suppose a breeder is interested in developing an algorithm to
classify unseen plants as diseased or not diseased with an available training data
set. The goal is to minimize the rate of misclassification or to maximize the
percentage of cases correctly classified (PCCC). Also, assume that you are new to
theworldofstatisticalmachinelearningandthatyouonlyunderstandthek-nearest
neighbor method. Since this algorithm depends only on the hyperparameter called
thenumberofneighbors (k), thequestioniswhichvalueofktochooseinsuchaway
that the prediction performance of this algorithm will be the best in the sample
predictionofplants. Tofindthebestvalueofthekhyperparameter, youmustspecify
arangeofvaluesfork (forexample, from1 to60 withincrementsof1), thenwitha
partofyourtrainingdataset, calledthetraining-inner (ortuningthatcorrespondsto
thetrainingdataintheinnerloop) set, whichisrandomlyselected. Youproceedto
evaluate the 60 values of k with the k-nearest neighbor method and evaluate the

### 25. 126 4 Overfitting, Model Tuning, and Evaluationof Prediction Performance
predict...

126 4 Overfitting, Model Tuning, and Evaluationof Prediction Performance
prediction performance in the remaining part of the training set (validation set).
Next, youselectthevalueofkfromthisrangeofvaluesthatbestpredicts (according,
forexample, tothe PCCC) out-of-sampledata (validationset) andusethisvalueto
performthepredictionoftheunseenplantsnotusedfortrainingthemodel (testing
set). This is a widely adopted practice that consists of searching for the parameter
(usuallythroughbruteforceloops) thatyieldsthebestperformanceoveravalidation
set. However, the process illustrated here is very simple because the k-nearest
neighbor model only depends on a unique hyperparameter; however, there are
other statistical machine learning algorithms (for example, deep learning methods)
wherethetuningprocessisrequiredforaconsiderableamountofhyperparameters.
Forthisreason, weencouragecautionwhenchoosingthestatisticalmachinelearn-
ingalgorithm, sincetheamountofworkrequiredforperformingthetuningprocess
dependsonthechosenmethod.
4.4.1 Why Is Model Tuning Important?
Tuning the hyperparameters of the models is a key element to optimize your
statistical machine learning model to perform well in out-of-sample predictions.
Thetuningprocessismoreanartthanasciencebecausethereisnouniqueformal
scientificprocedureavailableintheliterature. Nowadays, thetuningprocessistrial
anderrorthatconsistsofimplementingthestatisticalmachinelearningmodelmany
times with different values of the hyperparameters and then comparing its perfor-
mance on the validation set in order to determine which set of hyperparameters
results in the most accurate model; for the final implementation, the set of
hyperparameters of the best model is used. As mentioned above, for the k-nearest
neighborclassifier, weneedtochoosethenumberofneighbors (k) usingthetuning
process to obtain the optimal prediction performance of this algorithm, while for
conventional Ridge regression, the parameter lambda (λ) is obtained by tuning to
improvetheout-of-samplepredictions. Thesetwostatisticalmachinelearningalgo-
rithms that we just mentioned need only one hyperparameter; however, other
statistical machine learning methods may require more hyperparameters, as exem-
plifiedbydeeplearningmodelsthatrequireatleastthreehyperparameters (number
of neurons, number of hidden layers, type of activation function, batch size, etc.).
After tuning the required hyperparameters, the statistical machine learning model
learnstheparametersfromthedatatobeusedforthefinalpredictionofthetesting
set. Thechoiceofhyperparameterssignificantlyinfluencesthetimerequiredtotrain
andtestastatisticalmachinelearningmodel. Hyperparameterscanbecontinuousor
of the integer type; for this reason, there are mixed-type hyperparameter optimiza-
tionmethods.

### 26. 4.4 Model Tuning 127
4.4.2 Methods for Hyperparameter Tuning (Grid Search,
Rando...

4.4 Model Tuning 127
4.4.2 Methods for Hyperparameter Tuning (Grid Search,
Random Search, etc.)
Manualtuningofstatisticalmachinelearningmodelsisofcoursepossible, butrelies
heavily on the user’s expertise and understanding of the underlying problem.
Additionally, due to factors such as time-consuming model evaluations, nonlinear
hyperparameterinteractionsinthecaseoflargemodelsthatconsistoftensoreven
hundreds of hyperparameters, manual tuning may not be feasible since it is equiv-
alent to brute force. For this reason, the four most common approaches for
hyperparameter tuning reported in the literature are (a) grid search, (b) random
search,(c)Latinhypercubesampling, and (d) optimization (Kochetal.2017).
In the grid search method, each hyperparameter of interest is discretized into a
desiredsetofvaluestobestudiedwherethemodelsaretrainedandassessedforall
combinations of the values across all hyperparameters (that is, a “grid”). Although
fairlysimpleandstraightforwardtocarryout, agridsearchisappropriatewhenthere
areonlyafewvaluesforalimitednumberofhyperparameters. However, although
this is a comprehensive way of assessing different hyperparameter values, when
therearemanyvaluesforsomeormanyhyperparameters, itquicklybecomesquite
costly due to the number of hyperparameters and the number of discrete levels of
each. For example, in Ridge regression, this approach is implemented as follows:
since λ is the hyperparameter to be tuned, we first propose, for example, a grid of
100 values for this hyperparameter from λ¼1010 toλ ¼10(cid:1)2;then we dividethe
training set into five inner training sets and five inner testing (tuning) sets, where
eachofthe100 valuesofthegridisfittedusingtheinnertrainingsetsandthetesting
errorisevaluatedwiththeinnertestingsets. Thenwegettheaveragepredictedtest
error and pick one value out of the 100 values of the grid that produces the best
predictionperformance. Next, werefitthestatisticalmachinelearningmethodtothe
wholetrainingsetusingthepickedvalueofλ, andfinallyperformthepredictionsfor
the testing set using the learned parameters of the training set with the best picked
valueofλ.Inallthemodelswithonehyperparameter, itispracticaltoimplementthe
grid search method, but for example, in deep learning models, which many times
require six hyperparameters to be tuned, if only three values are used for each
hyperparameter, thereare36¼729 combinationsthatneedtobeevaluated, quickly
becomingcomputationallyimpracticable.
A random search differs from a grid search in that rather than providing a
discrete set of values to explore each hyperparameter, we determine a statistical
distributionforeachhyperparameterfromwhichvaluesmayberandomlysampled.
This affords a much greater chance of finding effective values for each
hyperparameter. While Latinhypercubesamplingissimilartothepreviousmethod,
itisamorestructuredapproach (Mc Kay1992) sinceitisanexperimentaldesignin
which samples are exactly uniform across each hyperparameter but random in
combinations. These so-called low-discrepancy point sets attempt to ensure that
points are approximately equidistant from one another in order to fill the space

### 27. 128 4 Overfitting, Model Tuning, and Evaluationof Prediction Performance
efficie...

128 4 Overfitting, Model Tuning, and Evaluationof Prediction Performance
efficiently. This sampling supports coverage across the entire range of each
hyperparameterandismorelikelytofindgoodvaluesofeachhyperparameter.
The previous two methods for hyperparameter tuning are used to perform indi-
vidual experiments by building models with various hyperparameter values and
recording the model performance for each. Because each experiment is performed
inisolation, thisprocessisparallelized, butisunabletousetheinformationfromone
experiment to improve the next experiment. Optimization methods, on the other
hand, consistofsequentialmodel-based optimizationwheretheresultsofprevious
experimentsareusedtoimprovethesamplingmethodofthenextexperiment. These
methodsaredesignedtomakeintelligentuseoffewerevaluationsandthussaveon
theoverallcomputationtime (Kochetal.2017).Optimizationalgorithmsthathave
been used in statistical machine learning generally for hyperparameter tuning
include Broyden–Fletcher–Goldfarb–Shanno (BFGS) (Konen et al. 2011), covari-
ance matrix adaptation evolution strategy (CMA-ES) (Konen et al. 2011), particle
swarm (PS)(Renukadeviand Thangaraj2014), tabusearch (TS), geneticalgorithms
(GA)(Lorenaandde Carvalho2008), andmorerecently, surrogate-based Bayesian
optimization (Dewanckeretal.2016).Also, recentlytheuseoftheresponsesurface
methodologyhasbeenexploredfortuninghyperparametersinrandomforestmodels
(Lujan-Moreno et al. 2018). However, the implementation of these optimization
methods is not straightforward because it requires expensive computation; also,
softwaredevelopmentisrequiredforimplementingthesealgorithmsautomatically.
Therehavebeenadvancesinthisdirectionforsomemachinelearningalgorithmsin
thestatisticalanalysissystem (SAS), Rand Pythonsoftware (Kochetal.2017).An
additional challenge isthe potential unpredictable computation expense of training
and validating predictive models using different hyperparameter values. Finally,
althoughitischallenging, thetuningprocessoftenleadstohyperparametersettings
that are better than the default values, as it provides a heuristic validation of these
settings, givinggreaterassurancethatamodelconfigurationwithahigheraccuracy
hasnotbeenoverlooked.
4.5 Metrics for the Evaluation of Prediction Performance
Thequalityofpredictionperformanceofanystatisticalmachinelearningmethodin
agivendatasetconsistsofevaluatinghowclosethepredictedvaluesaretothetrue
observed ones. In other words, the prediction performance quantifies the matching
degree between the predicted response value for a given observation and the true
responsevalue for that observation (James etal.2013).However, themetrics used
forquantifyingthepredictionperformancedependonthetypeofresponsevariable
understudy;forthisreason, wesubsequentlygivethemostpopularmetricsusedfor
thisgoalforfourtypesofresponsevariables.

### 28. 130 4 Overfitting, Model Tuning, and Evaluationof Prediction Performance
to Pear...

130 4 Overfitting, Model Tuning, and Evaluationof Prediction Performance
to Pearson’scorrelation, thesevaluesshouldalsobereported (slopeandintercept).
The expected intercept should be zero and the slope 1, if the correlation obtained
betweentheobservedandpredictedvaluesishigh. Itisimportanttoavoidcarrying
out the regression in the opposite way, i.e., using predicted values as y’s, and
observed values as x’s, since this leads to incorrect estimates of the slope and the
y-intercept. Thisdenotesthataspuriouseffectisaddedtotheregressionparameters
whenregressingpredictedversusobservedvaluesandcomparingthemagainstthe1:
1 line. The user should also remember that underestimation of the slope and
overestimation of the y-intercept increase as Pearson’s correlation values decrease.
Westronglyrecommendthatscientistsevaluatetheirmodelsbyregressingobserved
versus predicted values and test the significance of slope ¼ 1 and intercept ¼ 0
(Piñeiro et al. 2008). Finally, it is important to recall that the square of Pearson’s
correlationcanalsobeusedasametricformeasuringpredictionperformancesinceit
representstheproportionofthetotalvarianceexplainedbytheregressionmodeland
iscalledthecoefficientofdeterminationdenotedas R2.
Next, we present the mean absolute error (MAE) metric that measures the
difference between two continuous variables (observed and predicted). The MAE
canbecalculatedwiththefollowingexpression:
MAE ¼ 1
XnþT (cid:8)
(cid:8) (cid:8) y (cid:1) b fðxÞ
(cid:8)
(cid:8) (cid:8) ð4:3Þ
TST T i i
i¼nþ1
Belowwepresentanothermetricusedtoevaluatethepredictionperformanceof
anystatisticalmachinelearningmodel;itwasproposedby Kimand Kim (2016) and
iscalledmeanarctangentabsolutepercentageerror (MAAPE) thatiscalculatedas
follows:
(cid:6)(cid:8) (cid:8)(cid:9)
P (cid:8) b (cid:8)
nþT arctan (cid:8) (cid:8) y i (cid:1) fðxiÞ(cid:8) (cid:8)
i¼nþ1 y
MAAPE ¼ i ð4:4Þ
TST T
Although MAAPEisfinitewhentheresponsevariable (i.e., y ¼0) equalszero,
i
sinceithasasatisfactorytrigonometricrepresentation. However, because MAAPE’s
valueisexpressedinradians, itislessintuitive, inadditiontobeingscale-free. This
metric is a modification of the mean absolute percentage error (MAPE) which is
problematic because it is undefined when the response variable is equal to zero
(y ¼0). MAAPE isalso asymmetric since division by zero isdefined and isnota
i
problem. It is important to point out that there are other metrics for measuring
prediction accuracy for continuous data, but we only presented the most
popularones.
Thedistinctionbetweenthetrainingandtest MSEisimportantsincewearenot
interested in how well the statistical machine learning method performs in the
trainingdataset, duetothefactthatourmaingoalistoperformaccuratepredictions
intheunseentestdata. Forexample, aplantbreedermaybeinterestedindeveloping
analgorithm topredictdisease resistanceofa plant innew environments basedon

### 29. 4.5 Metricsforthe Evaluationof Prediction Performance 131
records that were coll...

4.5 Metricsforthe Evaluationof Prediction Performance 131
records that were collected in a set of environments. We can train the statistical
machinelearningmethodwiththeinformationcollectedinthesetofenvironments,
buttheinterestisnotinhowwellthestatisticalmachinelearningmethodpredictsin
those previously collected environments. An environmental scientist can also be
interestedinpredictingtheaverageannualrainfallinamunicipalityin Mexico, using
data from the last 20 years to train the model. Such measures could include sea
surfacetemperature, time, theyearlyrotationoftheearth, amongothers. Inthiscase,
the scientist is really interested in predicting the average rainfall of the next 1 or
2 years, notinaccuratelypredictingtheyearsmeasuredinthetrainingset.
4.5.2 Binary and Ordinal Measures of Prediction
Performance
The binary and ordinal response variables are very common in classification prob-
lems, wherethegoalistopredictwhichcategorysomethingfallsinto. Anexample
ofaclassificationproblemisanalyzingfinancialdatatodetermineifaclientwillbe
granted credit or not. Another example is analyzing breeding data to predict if an
animalisathighriskforacertaindiseaseornot. Below, weprovidesomepopular
metricstoevaluatethepredictionperformanceofthistypeofdata.
The first metric is called a confusion matrix, which is a tool to visualize the
performance of a statistical machine learning algorithm that is used in supervised
learning for classifying categorical and binary data. Each column of the matrix
represents the number of predictions in each class, while each row represents the
instancesintherealclass. Oneofthebenefitsofconfusionmatricesisthattheymake
it easy to determine whether the system is confusing classes. Table 4.3 shows a
sampleformatofaconfusionmatrixof Cclasses.
With Eqs.(4.5–4.8) wecancalculatethetotalnumberoffalsenegatives (TFN),
false positives (TFP), true negatives (TTN) for each class i, and the total true
positivesinthesystem, respectively:
XC
TFN ¼ n ð4:5Þ
i ij
j¼1
j6¼i
Table4.3 Confusionmatrixwithmorethantwoclasses
Predictedvalues
Class1 Class2 ... Class C
Observedvalues Class1 n n ... n
11 12 1 C
Class2 n n ... n
21 22 2 C
⋮ ⋮ ⋮ ⋮ ⋮
Class C n n ... n
C1 C2 CC

### 30. 132 4 Overfitting, Model Tuning, and Evaluationof Prediction Performance
XC
TFP ...

132 4 Overfitting, Model Tuning, and Evaluationof Prediction Performance
XC
TFP ¼ n ð4:6Þ
i ji
j¼1
j6¼i
XC XC
TTN ¼ n ð4:7Þ
i ji
j¼1 k¼1
j6¼i k6¼i
XC
TTP ¼ n ð4:8Þ
all jj
j¼1
Below, we define the sensitivity (S ), precision (P), and specificity (S ). The
e p
sensitivityindicatestheabilityofourstatisticallearningalgorithmtodeterminethe
proportionoftruepositivesthatarecorrectlyidentifiedbythetest. Theprecisionis
theproportionofcorrectclassificationofourstatisticalmachinelearningmodeland
represents the proportion of cases correctly classified, while the specificity is the
ability ofourstatistical machinelearningmodel toclassify thetrue negativecases,
thatis, thespecificityistheproportionoftruenegativesthatarecorrectlyidentified
bythetest. Underthe“one-versus-allbasis,”whereeachcategoryiscomparedwith
the composedinformation of the remaining categories, we provide the expressions
forcomputingthegeneralizedprecision, sensitivity, andspecificityforeachclassi:
TTP
P ¼ all ð4:9Þ
i TTP þTFP
all i
TTP
S ¼ all ð4:10Þ
ei TTP þTFN
all i
TTN
S ¼ all ð4:11Þ
pi TTN þTFP
all i
TTN
p CCC¼ all ð4:12Þ
PC PC
n
ij
i¼1 j¼1
Thetermp CCC denotesthe proportionofcases correctlyclassified, whichisa
measureoftheoverallaccuracy, andwhenmultipliedby100, denotesthepercentage
of cases correctly classified. Many times, this is the only metric reported for
measuring prediction performance in multi-class problems. However, PCCC alone
is sometimes quite misleading as there may be a model with relatively “high”
accuracy, but it predicts the “unimportant” class labels fairly accurately (e.g.,
“unknown bucket”). However, the model may be making all sorts of mistakes on
theclassesthatareactuallycriticaltotheapplication. Thisproblemisseriouswhen
intheinputdatathenumberofsamplesofdifferentclassesisveryunbalanced. For


---

## Raw Markitdown Extraction (full text)

Chapter 4
Overﬁtting, Model Tuning, and Evaluation
of Prediction Performance

4.1 The Problem of Overﬁtting and Underﬁtting

The overﬁtting phenomenon occurs when the statistical machine learning model
learns the training data set so well that it performs poorly on unseen data sets. In
other words, this means that the predicted values match the true observed values in
the training data set too well, causing what is known as overﬁtting. Overﬁtting
happens when a statistical machine learning model learns the systematic and noise
(random ﬂuctuations) parts in the training data to the extent that it negatively impacts
the performance of the statistical machine learning model on new data. This means
that the statistical machine learning model adapts very well to the noise as well as to
the signal that is present in the training data. The problem is that these concepts do
not apply to independent (new) data and negatively affect the model’s ability to
generalize. Overﬁtting is more probable when learning a loss function from a
complex statistical machine learning model (with more ﬂexibility). For this reason,
many nonparametric statistical machine learning models also include constraints in
the loss function to improve the learning process of the statistical machine learning
models. For example, artiﬁcial neural networks (ANN), mentioned later, are a
nonparametric statistical machine learning model that is very ﬂexible and is subject
to overﬁtting training data. This problem can be addressed by dropping out (setting
to zero) the weights of a certain percentage of hidden units in order to avoid
overﬁtting.

On the other hand, an underﬁtted phenomenon occurs when few predictors are
included in the statistical machine learning model, i.e., it is a very simple model that
poorly represents the complete picture of the predominant data pattern. This problem
also arises when the training data set is too small or not representative of the
population data. An underﬁtted model does a poor job of ﬁtting the training data
and for this reason it is not expected to satisfactorily predict new data points. This
implies that the predictions using unseen data are weak, since individuals are
perceived as strangers unfamiliar with the training data set.

© The Author(s) 2022
O. A. Montesinos López et al., Multivariate Statistical Machine Learning Methods
for Genomic Prediction, https://doi.org/10.1007/978-3-030-89010-0_4

109

110

4 Overﬁtting, Model Tuning, and Evaluation of Prediction Performance

Fig. 4.1 Schematic illustration of three models for classiﬁcation: (a) M1 with underﬁtting, (b) M2
with appropriate ﬁtting, and (c) M3 with overﬁtting

Consider a scattered series of points (y1, x1), . . ., (yn, xn), on a plane, to which we
want to adjust a statistical machine learning method. This means that we are looking
for the best f(xi) that explains the existing relationship between the response variable
(yi) and the predictors (x1, . . ., xn). We assume that we have three options for f(xi):
M1, the simple model plotted in Fig. 4.1a; M2, an intermediate model shown in
Fig. 4.1b; and M3, a complex model shown in Fig. 4.1c.

Under the classiﬁcation framework, the ﬁrst panel in Fig. 4.1 (left side, panel a)
shows an unsatisfactory ﬁt (underﬁtted) since the line does not cover most of the
points (has high bias) in the plot. As such, we expect that the prediction of unseen
data of this model, M1, will perform badly. In contrast, panel c of Fig. 4.1 shows an
almost perfect ﬁt, since the predicted line covers all the data points. While at ﬁrst
glance, you may think that model M3 will perform well when predicting unseen
data, this is actually untrue since the predicted line covers all points that are noise and
those that are signal (overﬁt); for this reason, this type of model also performs poorly
in the prediction of future data due to its complexity and high variance. Therefore,
the best option for predicting unseen data is model M2 (Fig. 4.1, panel b) since it
represents the predominant (smooth) pattern enough to represent the apparent data
pattern while maintaining a balance between bias and variance. For this reason, a
well-ﬁtted model is one that faithfully represents the sought-after predominant
pattern in the data, while ignoring the idiosyncrasies in the training data. As such,
a well-ﬁtted model in the testing set should be in the neighborhood of the model’s
accuracy based on the training data set, that is, the model’s accuracy in the testing set
should be approximately equal to that of the model’s accuracy in the training set. In
contrast, an overﬁtted model in the testing data set will be far from the neighborhood
of the model’s accuracy based on the training data set, and usually its prediction
performance is very high (good) in the training set and consequently low (bad) in the
testing set (Ratner 2017).

The paradox of overﬁtting is deﬁned as complex models that contain more
information about the training data, but less information about the testing data
(future data we want to predict). In statistical machine learning, overﬁtting is a
major issue and leads to some serious problems in research: (a) some relationships

4.2 The Trade-Off Between Prediction Accuracy and Model Interpretability

111

that seem statistically signiﬁcant are only noise, (b) the complexity of the statistical
machine learning model is very large for the amount of data provided, and (c) the
model in general is not replicable and predicts poorly.

Since the main goal of developing and implementing statistical machine learning
methods is to predict unseen data not used for training the statistical machine
learning algorithm, researchers are mainly interested in minimizing the testing
error (generalization error applicable to future samples) instead of minimizing the
training error that is applicable to the observed data used for training the statistical
machine learning algorithm.

According to Shalev-Shwartz and Ben-David (2014), if the learning fails, these

are some approaches to follow:

1. Increase the sample size of the training set.
2. Modify the hypothesis by (a) enlarging it, (b) reducing it, (c) completely changing
it, and (d) changing the parameters being used. We understand a hypothesis as the
models and their parameters under evaluation. This point is very important to
reach a reasonable model for your data.

3. Change the feature representation of the data.
4. Change the statistical machine learning algorithm used.

4.2 The Trade-Off Between Prediction Accuracy and Model

Interpretability

Accuracy is the ability of a statistical machine learning model to make correct
predictions and those models with more complexity (called ﬂexible models) are
better in terms of accuracy, while the simple, less complex models (called inﬂexible
models) are less accurate but more interpretable. Interpretability indicates to what
degree the model allows for human understanding of natural phenomena. For these
reasons, when the goal of the study is prediction, ﬂexible models should be used;
however, when the goal of the study is inference, inﬂexible models are more
appropriate because they more easily interpret the relationship between the response
variables and the predictor variables. As the complexity of the statistical machine
learning model increases, the bias is reduced and the variance increases. For this
reason, when more parameters are included in the statistical machine learning model,
the complexity of the model increases and the variance becomes the main concern
while the bias steadily falls. For example, James et al. (2013) state that the linear
regression model is a relatively inﬂexible method because it only generates linear
functions, while the support vector machine method is one of the most ﬂexible
statistical machine learning methods.

Before providing an analytical interpretation of the trade-off between the bias and
variance, we must understand the meaning of both concepts. Bias is the difference
between the expected prediction of our statistical machine learning model and the
true observed values. For example, assume that you poll a speciﬁc city where half of

112

4 Overﬁtting, Model Tuning, and Evaluation of Prediction Performance

Fig. 4.2 Graphical
representations of different
levels of bias and variance

the population is high-income and the other half is low-income. If you collected a
sample of high-income people, you would conclude that the entire city has high
income. This means that your conclusion is heavily biased since you only sampled
people with high income. On the other hand, error variance refers to the amount that
the estimate of the objective function will change using a different training data set.
In other words, the error variance accounts for the deviation of predictions from one
repetition to another using the same training set. Ideally, when a statistical machine
learning model with low error variance predicts a value, the predicted value should
remain almost the same, even when changing from one training data set to another;
however, if the model has high variance, then the predicted values of the statistical
machine learning method are affected by the values of the data set. We provide a
graphical visualization of bias and variance with a bull’s eye diagram (see Fig. 4.2).
We assume that the center of the target is a statistical machine learning model that
perfectly predicts the correct answers. As we move away from the bull’s eye, our
predictions get worse. Let us assume that we can repeat our entire statistical machine
learning model building process to get a number of separate hits on the target. Each
hit represents an individual realization of our statistical machine learning model,
given the chance variability in the training data we gathered. Sometimes we accu-
rately predict the observations of interest since we captured a representative sample
in our training data, while other times we obtain unreliable predictions since our
training data may be full of outliers or nonrepresentative values. The four combina-
tions of cases resulting from both high and low bias and variance are shown in
Fig. 4.2.

4.2 The Trade-Off Between Prediction Accuracy and Model Interpretability

113

Burger (2018) concludes that the best scenario is the one with low bias and low
variance, since samples are acceptable representatives of the population (Fig. 4.2),
while in the case of high bias and low variance, the samples are fairly consistent, but
not particularly representative of the population (Fig. 4.2). However, when there is
low bias and high variance, the samples vary widely in their consistency, and only
some may be representative of the population (Fig. 4.2). Finally, with high bias and
high variance, the samples are somewhat consistent, but unlikely to be representa-
tive of the population (Fig. 4.2).

If we denote the variable we are trying to predict as y and our covariates as xi, we
may assume that there is a relationship between y and xi, as that given in Eq. (1.1)
from Chap. 1, where the error term is normally distributed with a mean of zero and
variance σ2. The expected prediction error for a new observation with value x, using
a quadratic loss function, is given by Hastie et al. (2008, page 223):

(cid:2)
E y (cid:1)bf xif g

(cid:3)

2

(cid:3)
2

n

(cid:2)

(cid:3)

o

2

(cid:3)

ð

(cid:2)

(cid:2)
Þ2 þ E bf xif g
i
(cid:3)

(cid:2)

¼ E y (cid:1) f xif g
h
¼ Var yð Þ þ Bias bf xif g
i
(cid:3)
h
¼ Var Eð Þ þ Bias bf xif g

(cid:2)

2

2

þ E bf xið Þ (cid:1) E bf xif g
(cid:1) f xif g
(cid:3)
(cid:2)
þ Var bf xið Þ
(cid:3)
(cid:2)
þ Var bf xið Þ

,

where Bias is the result of misspecifying the statistical model f. Estimation variance
(the third term) is the result of using a sample to estimate f. The ﬁrst term is the error
(irreducible error) that results even if the model is correctly speciﬁed and accurately
estimated. This irreducible error is the noise term in the true relationship that cannot
fundamentally be reduced by any model. Given the true model and inﬁnite data to
train (calibrate) it, we should be able to reduce both the bias and variance terms to
0. However, in a world with imperfect models and ﬁnite data, there is a trade-off
between minimizing the bias and minimizing the variance. The above decomposition
reveals a source of the difference between explanatory and predictive modeling: In
explanatory modeling, the focus is on minimizing bias to obtain the most accurate
representation of the underlying theory. In contrast, predictive modeling seeks to
minimize the combination of bias and estimation variance, occasionally sacriﬁcing
theoretical accuracy for improved empirical precision (Shmueli 2010).

These four aspects impact every step of the modeling process, such that the

resulting f is markedly different in the explanatory and predictive contexts.

Let us assume that f is a reasonable operationalization of the true function (F)
relating constructs X and Y. Choosing a function f (cid:3) that is intentionally biased in
place of f is very undesirable from a theoretical–explanatory standpoint. However,
the election of f (cid:3) is desirable to f under the prediction approach. We show this using
the statistical model y ¼ β1x1 + β2x2 + β3x3 + E, which is assumed to be correctly
speciﬁed with respect to F. Using data, we obtain the estimated model bf , which has
the following properties:

114

4 Overﬁtting, Model Tuning, and Evaluation of Prediction Performance

(cid:3)
(cid:2)
Var bf xið Þ

(cid:2)
¼ Var bβ

Bias ¼ 0
2x2 þ bβ

1x1 þ bβ

3x3

(cid:3)

(cid:4)
¼ σ2xT XXT

(cid:5) 2 1

x,

where x is the vector x ¼ [x1, x2, x3]T and X is the design matrix based on all
predictors. Combining the squared bias with the variance gives, as expected, the
prediction error (EPE).

(cid:2)
E y (cid:1) bf xif g

(cid:3)
2

¼ σ2 þ 0 þ σ2xT XXT

(cid:4)

(cid:5) 2 1

h

(cid:4)

x = σ2 1 þ xT XXT

(cid:5)

(cid:1)1

i

:

x

In comparison, consider the estimated underspeciﬁed form bf

(cid:3)

xið Þ ¼ x1bγ. The bias

and variance here are

(cid:2)

Bias ¼ E bf xif g

(cid:3)

(cid:1)1xT
1

β

ð

1x1 þ β

2x2 þ β

3x3

Þ

(cid:4)

(cid:5)

(cid:1) f xif g ¼ x1 x1xT
1
2x2 þ β

(cid:1) β
ð

3x3

Þ

1x1 þ β
(cid:3)

(cid:2)
Var bf xið Þ

(cid:4)
¼ σ2x1 x1xT
1

(cid:5)

(cid:1)1xT
1

Combining the squared bias with the variance EPE is equal to

(cid:2)
E y (cid:1) bf xif g

(cid:3)
2

h

(cid:4)
¼ x1 x1xT
1

h

(cid:5)

(cid:1)1

1 x2β
2 þ x3β
xT
ð
i
(cid:5)
(cid:4)
:

(cid:1)1

xT
1

þ σ2 1 þ x1 x1xT
1

Þ (cid:1) x2β
ð

2 þ x3β

3

Þ

3

i
2

Although the bias of the underspeciﬁed model f (cid:3)(xi) is larger than that of f{xi}, its
variance can be smaller, and in some cases, so small that the overall EPE will be
lower for the underspeciﬁed model. Wu et al. (2007) showed the general result for an
underspeciﬁed linear regression model with multiple predictors. In particular, they
showed that the underspeciﬁed model that leaves out q predictors has a lower EPE
when the following inequality holds:

qσ2 > βT

2 XT

2 I (cid:1) H1
ð

ÞX2β

2

This means that the underspeciﬁed model produces more accurate predictions, in
terms of lower EPE, in the following situations: (a) when the data are very noisy (large
σ2); (b) when the true absolute values of the excluded parameters (in our example, β2
and β3) are small; (c) when the predictors are highly correlated; and (d) when the
sample size is small or the range of left-out variables is small (Shmueli 2010).

Hagerty and Srinivasan (1991) nicely summarize this situation: “We note that the
practice in applied research of concluding that a model with a higher predictive
validity is “truer,” is not a valid inference. This paper shows that a parsimonious but
less true model can have a higher predictive validity than a truer but less parsimo-
nious model.”

4.3 Cross-validation

115

4.3 Cross-validation

Cross-validation (CV) is a strategy for model selection or algorithm selection. CV
consists of splitting the data (at least once) for estimating the error of each algorithm.
Part of the data (the training set) is used for training each algorithm, and the
remaining part (the testing set) is used for estimating the error of the algorithm.
Then, CV selects the algorithm with the smallest estimated error. For this reason, CV
is used to evaluate the prediction performance of a statistical machine learning model
in out-of-sample data. This technique ensures that the data used for training the
statistical machine learning model are independent of the testing data set in which
the prediction performance is evaluated. It consists of repeating and recording the
arithmetic average obtained from the evaluation measures on different partitions.
Under k-fold CV, which is explained in greater detail later, this process is repeated a
total of k times, with each of the k groups getting the chance to play the role of the
test data, and the remaining k (cid:1) 1 groups used as training data. In this way, we obtain
k different estimates of the prediction error. As prediction performance is reported
the average of these estimates of prediction error. CV is used in data analysis to
validate the implemented models where the main objective is prediction and to
estimate the prediction performance of a statistical learning model that will be
carried out in practice. In other words, CV evaluates how well the statistical machine
learning model generalized new data not used for training the model. The results of
the CV largely depend on how the division between the training and testing sets is
carried out. For this reason, in the following sections, we provide the more popular
types of CV used in the implementation of statistical learning models.

4.3.1 The Single Hold-Out Set Approach

The single hold-out set or validation set approach consists of randomly dividing the
available data set into a training set and a validation or hold-out set (Fig. 4.3). The
statistical machine learning model is trained with the training set while the hold-out

I1 I2 I3

Complete data set

Training

Testing

I40 I5 I82...

I62 I45

In

I88

Fig. 4.3 Schematic representation of the hold-out set approach. A set of observations are randomly
split into a training set with individuals I40, I5, I82, among others, and into a testing set with
observations I45, I88, among others. The statistical machine learning model is ﬁtted on the training
set and its performance is evaluated on the validation set (James et al. 2013)

116

4 Overﬁtting, Model Tuning, and Evaluation of Prediction Performance

set (testing set) is used to study how well that statistical machine learning model
performs on unseen data. For example, 80% of the data can be used for training the
model and the remaining 20% of the data for testing it. One weakness of the hold-out
(validation) set approach is that it depends on just one training-testing split and its
performance depends on how the data are split into the training and testing sets.

4.3.2 The k-Fold Cross-validation

In k-fold CV, the data set is randomly divided into k complementary folds (groups)
of approximately equal size. One of the subsets is used as testing data and the rest
(k (cid:1) 1) as training data. Then k (cid:1) 1 folds are used for training the statistical machine
learning model and the remaining fold for evaluating the out-of-sample prediction
performance. For these reasons, the statistical machine learning model is ﬁtted
k times using a different partition (fold) as the testing set and the remaining k (cid:1) 1
as the training set. Finally, the arithmetic mean of the k folds is obtained and reported
as the prediction performance of the statistical machine learning model (see Fig. 4.4).
This method is very accurate because it combines k measures of ﬁtness resulting
from the k training and testing data sets into which the original data set was divided,
but at the cost of more computational resources. In practice, the choice of the number
of folds depends on the measurement of the data set, although 5 or 10 folds are the
most common choices.

Fig. 4.4 Schematic representation of the k-fold cross-validation with complementary subsets with
k ¼ 5

4.3 Cross-validation

117

It

is important

to point out

to reduce variability, we recommend
implementing the k-fold CV multiple times, each time using different complemen-
tary subsets to form the folds; the validation results are combined (e.g., averaged)
over the rounds (times) to give a better estimate of the statistical machine learning
model predictive performance.

that

4.3.3 The Leave-One-Out Cross-validation

The Leave-One-Out (or LOO) CV is very simple since each training data set is
created by including all the individuals except one, while the testing set only
includes the excluded individual. Thus, for n individuals in the full data set, we
have n different training and testing sets. This CV scheme wastes minimal data, as
only one individual is removed from the training set.

Regarding the k-fold cross-validation that was just explained, n models are built
from n individuals (samples) instead of k models, where n > k. Moreover, each
model is trained on n (cid:1) 1 samples rather than (k (cid:1) 1)n/k. In both cases, since k is
normally not too large and k < n, LOO is more computationally expensive than k-
fold cross-validation. In terms of prediction performance, LOO normally produces
high variance for the estimation of the test error. Because n (cid:1) 1 of the n samples is
used to build each statistical machine learning model, those constructed from folds
are virtually identical to each other and to the model built from the entire training set.
However, when the learning curve is steep for the evaluated training size, then ﬁve-
or ten-fold cross-validation usually overestimates the generalization error.

Learning curves (LC) are considered effective tools to monitor the performance
of the employee exposed to a new task. LCs provide a mathematical representation
of the learning process that takes place as the task is repeated. In statistical machine
learning the LC is a line plot of learning (y-axis) over experience (x-axis). Learning
curves are extensively used in statistical machine learning for algorithms that learn
(their parameters) incrementally over time, such as deep neural networks. In
general, there is considerable empirical evidence suggesting that ﬁve- or ten-fold
cross-validation should be preferred to LOO.

4.3.4 The Leave-m-Out Cross-validation

The Leave-m-Out (LmO) CV is very similar to LOO as it creates all the possible
training/test sets by removing m samples from the complete set. For n samples, this

produces

train-test pairs. Unlike LOO and k-fold, the test sets will overlap for

  !
n

m

m > 1. For example, in a Leave-2-Out CV with a data set with four samples (I1, I2,

118

4 Overﬁtting, Model Tuning, and Evaluation of Prediction Performance

I3, and I4), the total number of training-testing sets is equal to

  !
4

2

¼ 6; this means

that the six testing sets are: [I3, I4] [I1, I2], [I2, I4] [I1, I3], [I2, I3] [I1, I4], [I1, I4]
[I2, I3],[I1, I3] [I2, I4], and [I1,I2] [I3, I4], while the training sets are the comple-
mentary elements of each testing set.

4.3.5 Random Cross-validation

In this type of CV, the number of partitions (independent training-testing data set
splits) is deﬁned by the user, and more partitions are better. Each partition is
generated by randomly dividing the whole data set into two subsets: the training
(TRN) data set and the testing (TST) data set. The percentage of the whole data set
assigned to the TRN and TST data sets is also ﬁxed by the user. For example, for
each random partition, the user can decide that 80% of the whole data set can be
assigned to the TRN data set and the remaining 20% to the TST data set. Random
cross-validation is different from k-fold cross-validation because the partitions are
not mutually exclusive; this means that in the random cross-validation approach, one
observation can appear in more than one partition. Consequently, some samples
cannot be evaluated, whereas others can be evaluated more than once, meaning that
the testing and training subsets can be superimposed (Montesinos-López et al.
2018a, b). To control the randomness for reproducibility, we recommend using a
speciﬁc seed in the random number generator. We recommend using at least ten
random partitions to obtain enough accuracy in the estimate of prediction
performance.

4.3.6 The Leave-One-Group-Out Cross-validation

The Leave-One-Group-Out (LOGO) CV is useful when individuals are grouped
(in environments or years, or even another criterion), where the number of groups (g)
is at least two and the information of g (cid:1) 1 groups are used as the training set while
all individuals of the remaining group are used as the testing set. For example, in the
context of genomic selection, when the plant breeder is interested in predicting these
lines in another environment, the same (or different) lines were frequently evaluated
in g environments or years, that represent the groups. Jarquín et al. (2017) denotes
this type of CV strategy as CV1 in the context of plant breeding. Under this
approach, the predictions are reported for each of the g groups because the scientist
is interested in the prediction performance of each environment. Many times, the
groups are the years under study and the aim is to predict the information of a
complete year. However, when the groups are years and if we suspect that there is a
considerable correlation between observations that are near in time. Therefore, it is

4.3 Cross-validation

119

Fig. 4.5 Schematic representation of time series data with 5 years, using the previous year for
predicting the next year. However, in some practical applications, we are not interested in going too
far back in time since the training and testing sets will be less related the farther you go

imperative to evaluate our statistical machine learning model for time series data on
“future” observations. In this sense, the training sets are composed of the previous
years to predict the subsequent year. This method can also be seen as a variation of k-
fold CV, where the ﬁrst folds are used for training the statistical machine learning
model and the fold (k + 1) is the corresponding testing set. The main difference in
this CV method is that successive training sets are supersets of those that come
before them. Also, it adds all surplus data to the ﬁrst training partition, which is
always used to train the model (see Fig. 4.5).

Figure 4.5 shows that under this type of CV, for time series data, the predictions
are for g (cid:1) 1 years; individuals from the ﬁrst year are not predicted since a training
set is not available.

4.3.7 Bootstrap Cross-validation

First, we will deﬁne the bootstrapping method to understand how it is used in the CV
approach, which should then be straightforward. Bootstrapping is a type of
resampling method where, for example, B ¼ 10 samples of the same size are
repeatedly drawn, with replacement, from a single original sample. Afterward,
each of these B samples is used to estimate statistics (for example, the mean,
variance, median, minimum, etc.) of a population, and the average of all the
B sample estimates of the target statistic is reported as the ﬁnal estimate. In the
context of statistical machine learning, these samples are used to evaluate the
prediction performance of the algorithm under study for unseen data. One important
difference between this CV approach and all the procedures explained above is that
now the training set has the same size (number of observations) as the original
sample because the bootstrap method replaced some individuals more than once.
According to Kuhn and Johnson (2013), as a result, some observations will be
represented multiple times in the bootstrap sample, while others will not be selected
at all; those observations not selected are referred to as the testing set, however, this
CV strategy is quite different than the previously explained. Efron (1983) pointed
out that the prediction performance of the bootstrap samples tends to have less
uncertainty than the k-fold cross-validation since on average, 63.2% of the data
points are represented (for training) at least once in any sample size. For this reason,

120

4 Overﬁtting, Model Tuning, and Evaluation of Prediction Performance

Fig. 4.6 Schematic representation of bootstrap cross-validation

this CV approach has a bias similar to implementing a k ¼ two fold cross-validation,
and as the training set becomes smaller, the bias becomes more problematic. To
understand this CV method, we provide a simple example of how the training and
testing samples are constructed. If we have a sample with 12 individuals denoted as
I1, I2, . . ., I12, we will select B ¼ 5 bootstrap samples. Each bootstrap sample is
obtained with replacement and the individuals that appear in each one correspond to
the training sample; those that are not present will correspond to the testing set.
Figure 4.6 provides the ﬁve bootstrap samples; each training sample has the same
size as the original, however, only some individuals appear in each bootstrap sample,
while those individuals that do not appear are included in the testing set. For
example, in the ﬁrst fold, the training bootstrap sample contains seven different
individuals (I2, I3, I4, I6, I7, I8, and I9), while the testing set contains ﬁve
individuals (I1, I5, I10, I11, and I12). It is important to point out that since the
training sample has the same size as the original sample, some individuals in the
training sample are repeated at least twice; in the ﬁrst fold, the individual I4 are
repeated three times, whereas I6, I7, and I8 are repeated twice. Finally, similar to
other methods, the statistical machine learning model is trained with each training
set; likewise, the prediction performance of the model is evaluated in each testing
set. The average of these sample predictions is reported as the estimated testing error.

4.3.8

Incomplete Block Cross-validation

Incomplete block (IB) CV should be used when there are J treatments evaluated in
I blocks and the same treatments are evaluated in all the blocks. The idea behind this

4.3 Cross-validation

Table 4.1 TRN data set for
I ¼ 3 blocks and J ¼ 10
treatments with 70% of data
for training

Block
1
2
3

121

10
9
10

7
5
6

8
6
7

9
7
8

Treatments per block
1
2
1

5
4
4

3
3
2

CV method is that some treatments should be present in some blocks but absent in
others, whereas the same treatment should be present in at least one environment
(block). The theory of incomplete block designs developed in the experimental
designs statistical area can be used to construct the training set. For example,
under a balanced incomplete block (BIB) design, the term incomplete means that
all treatments in each block cannot be evaluated, whereas balanced means that each
pair of treatments occur together λ times. The training set is constructed by ﬁrst
deﬁning the % of individuals in the TRN set using the equation sI ¼ Jr ¼ NTRN,
where J represents the number of treatments under study, I represents the number of
blocks under study, r denotes the number of repetitions of each treatment, and s
denotes the treatments per block. For example, suppose that we had J ¼ 10 treat-
ments and I ¼ 3 blocks (that is, 30 individuals), and we decided to use NTRN ¼ 21
(70%) of the total individuals in the TRN set. Therefore, the number of treatments by
block can be obtained by solving (sI ¼ NTRN) for s, which results in s ¼ NTRN/I. This
means that s ¼ 21/3 ¼ 7 treatments per block. Then, the corresponding elements for
the training set can be obtained with the function ﬁnd.BIB(10, 3, 7) of the package
crossdes of the R statistical software. The numbers used in the function ﬁnd.BIB()
denote the treatments, the blocks, and the treatments per block, respectively. Finally,
the treatments that make up the TRN set are shown in Table 4.1.

According to Table 4.1, it is clear that each treatment is present in two blocks and
missing in one block. For example, in Block 1 the testing set includes treatments 2, 4,
and 6; in Block 2, the testing set is composed of treatments 1, 8, and 10; and in Block
3, the testing set is composed of treatments 3, 5, and 9.

4.3.9 Random Cross-validation with Blocks

Random cross-validation with blocks was proposed by Lopez-Cruz et al. (2015) and
belongs to the so-called replicated TRN-TST cross-validation that appears in the
publication of Daetwyler et al. (2012), since some individuals can never be part of
the training set. This algorithm, like the incomplete block cross-validation, is
appropriate when we are interested in evaluating J lines in I blocks or environments
and tries to mimic a prediction problem faced by breeders in incomplete ﬁeld trials
where lines are evaluated in some, but not all, target environments. The algorithm for
constructing the TRN-TST sets is described by the following steps: Step 1. Calculate
the total number of observations under study as N ¼ J (cid:4) I; Step 2. Deﬁne the
proportion of observations used for training and testing, that is, PTRN and PTST; Step
3. Calculate the size of the testing set NTST ¼ N (cid:4) PTST; Step 4. Choose NTST lines at

122

4 Overﬁtting, Model Tuning, and Evaluation of Prediction Performance

Table 4.2 TRN-TST data sets for J ¼ 10 lines and I ¼ 3 environments

Environment

1
2
3

L2

Lines
L1
TRN TST
TST
TRN TRN TST

L3
L4
TRN TRN TST

L5

L8

L6
L7
TRN TRN TST

L9
L10
TRN TRN
TRN
TRN TRN TRN TRN TRN TST

TRN TST

TST

TRN TRN TRN TRN TRN TST

random without replacement if J (cid:5) NTST, and with replacement otherwise; Step
5. Each chosen line will then be assigned to one of the I environments chosen at
random without replacement; Step 5. All the selected lines and environments will
form the training set, while the lines and environments that were not chosen will
form the corresponding testing set; and Step 6. Steps 1–5 are repeated depending on
the number of TRN-TST partitions required (Lopez-Cruz et al. 2015). This CV is
called CV2 in Jarquín et al. (2017). Next, we assume that we have ten lines or
treatments and three environments or blocks that will form the corresponding
training testing sets for only one partition: Step 1. The total number of observations
under study is N ¼ 10 (cid:4) 3 ¼ 30; Step 2. We deﬁne PTRN ¼ 0.7 and PTST ¼ 0.3; Step
3. The size of the testing set is NTST ¼ 30 (cid:4) 0.3 ¼ 9; Step 4. Since J ¼ 10 (cid:5) NTST ¼ 9,
we selected the following lines at random without replacement: L1, L2, L3, L4, L5,
L6, L7, L8, L9, and L10; and Step 5. Each chosen line was assigned to one of the
I ¼ 3 environments randomly chosen without replacement, as shown in Table 4.2. It
is important to point out that this CV strategy only differs from the incomplete block
cross-validation in the way the lines are allocated to blocks.

4.3.10 Other Options and General Comments

on Cross-validation

It is important to highlight that when the data set is considerably large, it is better to
randomly split it into three parts: a training set, a validation set (or tuning set), and a
testing set. The training set and testing set are used as explained before, while the
validation (tuning set) set is used to estimate the prediction error for model selection,
which is the process of estimating the performance of different models in order to
choose the best one, or to evaluate the chosen statistical machine learning model
with a range of values of tuning hyperparameters to select the combination of
hyperparameters with the best prediction performance and then use these
hyperparameters (or best model) to evaluate the prediction performance in the testing
set (Fig. 4.7). It is important to point out that Fig. 4.7 shows only one random split of
the data in terms of the training, testing, and validation sets.

For example, assume that our data set has 50,000 rows (observations) and that we
have decided to use 5000 of them for the testing set and another 5000 for the
validation set. This means that 40,000 rows are left for the training set. At this
point, we train our statistical machine learning model with each component of the

4.3 Cross-validation

123

Fig. 4.7 Schematic representation of the training, validation (tuning), and testing sets proposed by
Cook (2017)

grid of hyperparameters of the training set and evaluate the prediction performance
on the validation set, as shown in the middle of Fig. 4.7. Finally, we will pick the best
model (best subset of hyperparameters) in terms of prediction performance in the
validation set and we are ready to evaluate the prediction performance in the testing
set. Then, we will report the testing error on the testing set, as can be observed at the
bottom of Fig. 4.7 (Cook 2017). Under this approach, the testing and validation sets
have approximately the same size to guarantee a similar out-of-sample prediction
performance. Since it is difﬁcult to give general rules on how to choose the number
of observations in each of the three parts, a typical number might be 50% for
training, and 25% each for validation and testing (Hastie et al. 2008) or 70%,
15%, and 15% for training, validation, and testing, respectively. Another way to
ﬁnd the optimal setting of hyperparameters using the grid search, which is very
common in deep learning, consists of picking values for each parameter from a ﬁnite
set of options [e.g., number of epochs (100, 150, 200, 250, 300); batch sizes (25, 50,
75, 100, 125); number of layers (1, 2, 3, 4, 5); and types of activation functions
(RELU, Sigmoid), . . .] and training the statistical machine learning model with
every permutation of hyperparameter choices using the training set. Then, the
combination of hyperparameters with the best prediction performance on the vali-
dation set is chosen, and we report the prediction performance of the best selected
model (set of hyperparameters) in the testing set (Buduma 2017). The aforemen-
tioned examples use the validation data set as a proxy measure of the accuracy
during the hyperparameter optimization process. However, it can also be used as a
proxy measure of the accuracy for model selection, and instead of using a grid of
hyperparameters, we can use a set of different statistical machine learning models;
here, the best model is chosen instead of the best combination of hyperparameters. It
is important to understand that when you have more than one random partition
(training-testing-validation), as shown in Fig. 4.8, the same process provided in
Fig. 4.7 is followed, however, the average of all the partitions is reported as a
measure of prediction performance. Also, if more precision is required in the
estimated prediction performance, you can repeat the process given in Fig. 4.8

124

4 Overﬁtting, Model Tuning, and Evaluation of Prediction Performance

Fig. 4.8 Five partitions of training-validation-testing

multiple times and report the average of all repetitions as a measure of prediction
performance.

As mentioned above, this approach (Figs. 4.7 and 4.8) is used for a large data set
however, in the case of a smaller data set, we suggest modifying this approach to
avoid wasting too much training data in validation sets. This modiﬁcation consists of
performing an inner cross-validation approach since the training set is split into
complementary subsets, and each model is trained against a different combination of
these subsets, which is subsequently validated against the remaining parts. Once the
model hyperparameters have been selected, a ﬁnal model is trained with the whole
training set (reﬁtted) and the generalized prediction performance is measured on the
testing set. This approach was applied by Montesinos-López et al. (2018a, b), who
also split the original data into training and testing sets. Subsequently, each training
set was split again and 80% of the data was used for training a grid of
hyperparameters while the remaining 20% was used for validating (tuning) the
prediction performance and selecting the best combination of hyperparameters
with the best prediction performance. At this point, the deep learning algorithm
was reﬁtted with the whole training set, and with this they evaluated the out-of-
sample prediction. They called the conventional training-testing partition outer
cross-validation, while the split performed in each training set used for
hyperparameter tuning was called inner cross-validation. It should be highlighted
that there are no differences between outer and inner CV and training-validation-test.
Finally, it should also be mentioned that any type of the cross-validation strategies
mentioned in this section (random CV, k-fold CV, Bootstrap CV, IB CV, etc.,) can
be used in both outer and inner CV, such as is the case with the ﬁve-fold CV.

4.4 Model Tuning

A hyperparameter is a parameter whose value is set before the learning process
begins. Hyperparameters govern many aspects of the behavior of statistical machine
learning models, such as their ability to learn features from data, the models’
exhibited degree of generalizability in performance when presented with new data,
as well as the time and memory cost of training the model, since different
hyperparameters often result in models with signiﬁcantly different performance.
This means that tuning hyperparameter values is a critical aspect of the statistical
machine learning training process and a key element for the quality of the resulting

4.4 Model Tuning

125

Fig. 4.9 Schematic representation of the tuning process proposed by Kuhn and Johnson (2013)

prediction accuracies. However, choosing appropriate hyperparameters is challeng-
ing (Montesinos-López et al. 2018a). Hyperparameter tuning ﬁnds the best version
of a statistical machine learning model by running many training sets on the original
data set using the algorithm and ranges of values of hyperparameters as speciﬁed.
The hyperparameter values that provide the best performance in out-of-sample
prediction evaluated by the chosen metric are then selected.

There are many ways of searching for the best hyperparameters. However, a
general approach deﬁnes a set of candidate values for each hyperparameter. Each
value of this set of candidate values is then applied with a resample of the training set
of the chosen statistical machine learning method, where we aggregate all the hold-
out predictions from which the best hyperparameters are chosen and reﬁt the model
with the entire set (Kuhn and Johnson 2013). A schematic representation of the
tuning process proposed by Kuhn and Johnson (2013) is given in Fig. 4.9. It is
important to highlight that this process should be performed correctly because when
the same data are used for training and evaluating the prediction performance, the
prediction performance obtained is extremely optimistic.

For example, suppose a breeder is interested in developing an algorithm to
classify unseen plants as diseased or not diseased with an available training data
set. The goal is to minimize the rate of misclassiﬁcation or to maximize the
percentage of cases correctly classiﬁed (PCCC). Also, assume that you are new to
the world of statistical machine learning and that you only understand the k-nearest
neighbor method. Since this algorithm depends only on the hyperparameter called
the number of neighbors (k), the question is which value of k to choose in such a way
that the prediction performance of this algorithm will be the best in the sample
prediction of plants. To ﬁnd the best value of the k hyperparameter, you must specify
a range of values for k (for example, from 1 to 60 with increments of 1), then with a
part of your training data set, called the training-inner (or tuning that corresponds to
the training data in the inner loop) set, which is randomly selected. You proceed to
evaluate the 60 values of k with the k-nearest neighbor method and evaluate the

126

4 Overﬁtting, Model Tuning, and Evaluation of Prediction Performance

prediction performance in the remaining part of the training set (validation set).
Next, you select the value of k from this range of values that best predicts (according,
for example, to the PCCC) out-of-sample data (validation set) and use this value to
perform the prediction of the unseen plants not used for training the model (testing
set). This is a widely adopted practice that consists of searching for the parameter
(usually through brute force loops) that yields the best performance over a validation
set. However, the process illustrated here is very simple because the k-nearest
neighbor model only depends on a unique hyperparameter; however, there are
other statistical machine learning algorithms (for example, deep learning methods)
where the tuning process is required for a considerable amount of hyperparameters.
For this reason, we encourage caution when choosing the statistical machine learn-
ing algorithm, since the amount of work required for performing the tuning process
depends on the chosen method.

4.4.1 Why Is Model Tuning Important?

implementation,

Tuning the hyperparameters of the models is a key element to optimize your
statistical machine learning model to perform well in out-of-sample predictions.
The tuning process is more an art than a science because there is no unique formal
scientiﬁc procedure available in the literature. Nowadays, the tuning process is trial
and error that consists of implementing the statistical machine learning model many
times with different values of the hyperparameters and then comparing its perfor-
mance on the validation set in order to determine which set of hyperparameters
results in the most accurate model; for the ﬁnal
the set of
hyperparameters of the best model is used. As mentioned above, for the k-nearest
neighbor classiﬁer, we need to choose the number of neighbors (k) using the tuning
process to obtain the optimal prediction performance of this algorithm, while for
conventional Ridge regression, the parameter lambda (λ) is obtained by tuning to
improve the out-of-sample predictions. These two statistical machine learning algo-
rithms that we just mentioned need only one hyperparameter; however, other
statistical machine learning methods may require more hyperparameters, as exem-
pliﬁed by deep learning models that require at least three hyperparameters (number
of neurons, number of hidden layers, type of activation function, batch size, etc.).
After tuning the required hyperparameters, the statistical machine learning model
learns the parameters from the data to be used for the ﬁnal prediction of the testing
set. The choice of hyperparameters signiﬁcantly inﬂuences the time required to train
and test a statistical machine learning model. Hyperparameters can be continuous or
of the integer type; for this reason, there are mixed-type hyperparameter optimiza-
tion methods.

4.4 Model Tuning

127

4.4.2 Methods for Hyperparameter Tuning (Grid Search,

Random Search, etc.)

Manual tuning of statistical machine learning models is of course possible, but relies
heavily on the user’s expertise and understanding of the underlying problem.
Additionally, due to factors such as time-consuming model evaluations, nonlinear
hyperparameter interactions in the case of large models that consist of tens or even
hundreds of hyperparameters, manual tuning may not be feasible since it is equiv-
alent
the four most common approaches for
hyperparameter tuning reported in the literature are (a) grid search, (b) random
search, (c) Latin hypercube sampling, and (d) optimization (Koch et al. 2017).

to brute force. For this reason,

In the grid search method, each hyperparameter of interest is discretized into a
desired set of values to be studied where the models are trained and assessed for all
combinations of the values across all hyperparameters (that is, a “grid”). Although
fairly simple and straightforward to carry out, a grid search is appropriate when there
are only a few values for a limited number of hyperparameters. However, although
this is a comprehensive way of assessing different hyperparameter values, when
there are many values for some or many hyperparameters, it quickly becomes quite
costly due to the number of hyperparameters and the number of discrete levels of
each. For example, in Ridge regression, this approach is implemented as follows:
since λ is the hyperparameter to be tuned, we ﬁrst propose, for example, a grid of
100 values for this hyperparameter from λ ¼ 1010 to λ ¼ 10(cid:1)2; then we divide the
training set into ﬁve inner training sets and ﬁve inner testing (tuning) sets, where
each of the 100 values of the grid is ﬁtted using the inner training sets and the testing
error is evaluated with the inner testing sets. Then we get the average predicted test
error and pick one value out of the 100 values of the grid that produces the best
prediction performance. Next, we reﬁt the statistical machine learning method to the
whole training set using the picked value of λ, and ﬁnally perform the predictions for
the testing set using the learned parameters of the training set with the best picked
value of λ. In all the models with one hyperparameter, it is practical to implement the
grid search method, but for example, in deep learning models, which many times
require six hyperparameters to be tuned, if only three values are used for each
hyperparameter, there are 36 ¼ 729 combinations that need to be evaluated, quickly
becoming computationally impracticable.

A random search differs from a grid search in that rather than providing a
discrete set of values to explore each hyperparameter, we determine a statistical
distribution for each hyperparameter from which values may be randomly sampled.
This affords a much greater chance of ﬁnding effective values for each
hyperparameter. While Latin hypercube sampling is similar to the previous method,
it is a more structured approach (McKay 1992) since it is an experimental design in
which samples are exactly uniform across each hyperparameter but random in
combinations. These so-called low-discrepancy point sets attempt to ensure that
points are approximately equidistant from one another in order to ﬁll the space

128

4 Overﬁtting, Model Tuning, and Evaluation of Prediction Performance

efﬁciently. This sampling supports coverage across the entire range of each
hyperparameter and is more likely to ﬁnd good values of each hyperparameter.

The previous two methods for hyperparameter tuning are used to perform indi-
vidual experiments by building models with various hyperparameter values and
recording the model performance for each. Because each experiment is performed
in isolation, this process is parallelized, but is unable to use the information from one
experiment to improve the next experiment. Optimization methods, on the other
hand, consist of sequential model-based optimization where the results of previous
experiments are used to improve the sampling method of the next experiment. These
methods are designed to make intelligent use of fewer evaluations and thus save on
the overall computation time (Koch et al. 2017). Optimization algorithms that have
been used in statistical machine learning generally for hyperparameter tuning
include Broyden–Fletcher–Goldfarb–Shanno (BFGS) (Konen et al. 2011), covari-
ance matrix adaptation evolution strategy (CMA-ES) (Konen et al. 2011), particle
swarm (PS) (Renukadevi and Thangaraj 2014), tabu search (TS), genetic algorithms
(GA) (Lorena and de Carvalho 2008), and more recently, surrogate-based Bayesian
optimization (Dewancker et al. 2016). Also, recently the use of the response surface
methodology has been explored for tuning hyperparameters in random forest models
(Lujan-Moreno et al. 2018). However, the implementation of these optimization
methods is not straightforward because it requires expensive computation; also,
software development is required for implementing these algorithms automatically.
There have been advances in this direction for some machine learning algorithms in
the statistical analysis system (SAS), R and Python software (Koch et al. 2017). An
additional challenge is the potential unpredictable computation expense of training
and validating predictive models using different hyperparameter values. Finally,
although it is challenging, the tuning process often leads to hyperparameter settings
that are better than the default values, as it provides a heuristic validation of these
settings, giving greater assurance that a model conﬁguration with a higher accuracy
has not been overlooked.

4.5 Metrics for the Evaluation of Prediction Performance

The quality of prediction performance of any statistical machine learning method in
a given data set consists of evaluating how close the predicted values are to the true
observed ones. In other words, the prediction performance quantiﬁes the matching
degree between the predicted response value for a given observation and the true
response value for that observation (James et al. 2013). However, the metrics used
for quantifying the prediction performance depend on the type of response variable
under study; for this reason, we subsequently give the most popular metrics used for
this goal for four types of response variables.

4.5 Metrics for the Evaluation of Prediction Performance

129

4.5.1 Quantitative Measures of Prediction Performance

Before implementing a statistical machine learning model, we assume that we have
our training observation {(x1, y1), (x2, y2), . . ., (xn, yn)} and we estimate f, as bf , with
the chosen statistical machine learning model. Then we can make predictions for
each of the response values (yi) with bf xið Þ and compute the predicted values for each
of the n observations in the training set; with these values we can calculate the mean

square error (MSE) for the training data set as E ¼ 1
n

(cid:6)

(cid:2)

Pn

i¼1

(cid:3)
yi (cid:1) bf xið Þ

2

; however,

what we really want to predict are the values for unseen test observations that were
not used to train the statistical machine learning model. Assuming that the unseen
testing set is equal to {(xn + 1, yn + 1), (xn + 2, yn + 2), . . ., (xn + T, yn + T)}, the MSE for
the testing data set should be calculated as

MSETST ¼

1
T

XnþT

(cid:2)

yi (cid:1) bf xið Þ

(cid:3)
2

,

i¼nþ1

ð4:1Þ

where bf xið Þ is the prediction that bf gives to the ith observation. The MSETST with a
lower value will have better predictions, which means that the predicted values are
very close to the true observed values. Also, the square root of MSETST can be used
as a measure of prediction performance and is called root mean square error
(RMSE).

Pearson’s correlation coefﬁcient is a very popular measure of prediction perfor-

mance in plant breeding and can be calculated as

rTST ¼

r

P

(cid:2)
bf xið Þ (cid:1) bf xið Þ
q

nþT
i¼nþ1
(cid:2)
bf xið Þ (cid:1) bf xið Þ

ﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃ
(cid:3)2
P

nþT
i¼nþ1

(cid:3)
ð

yi (cid:1) yi

Þ
ﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃ
P
nþT
Þ2
i¼nþ1 yi (cid:1) yi
ð

,

ð4:2Þ

where bf xið Þ is the average of the T predictions that conform to the testing set, and yi is
the average of the T true observed values. In this case, the closer that the predictions
are to 1, the better the implemented statistical machine learning model will perform.
It is important to point out that Pearson’s correlation is deﬁned between (cid:1)1 and
1. However, to be convinced that the observed and predicted values match, it is a
common practice to perform a scatter plot of predicted versus observed (or vice
versa) values and when the observed and predicted values follow a straight line (45(cid:6)
diagonal line) from the bottom left corner to the top right corner, this indicates a
perfect match between the observed and predicted values. For this reason, Pearson’s
correlation as a metric should be complemented with the intercept and slope, since
the slope and intercept describe the consistency and the model bias, respectively
(Smith and Rose 1995; Mesple et al. 1996). To obtain the slope and intercept, the
observed values (as y) versus the predicted values (as x) are regressed and in addition

130

4 Overﬁtting, Model Tuning, and Evaluation of Prediction Performance

to Pearson’s correlation, these values should also be reported (slope and intercept).
The expected intercept should be zero and the slope 1, if the correlation obtained
between the observed and predicted values is high. It is important to avoid carrying
out the regression in the opposite way, i.e., using predicted values as y’s, and
observed values as x’s, since this leads to incorrect estimates of the slope and the
y-intercept. This denotes that a spurious effect is added to the regression parameters
when regressing predicted versus observed values and comparing them against the 1:
1 line. The user should also remember that underestimation of the slope and
overestimation of the y-intercept increase as Pearson’s correlation values decrease.
We strongly recommend that scientists evaluate their models by regressing observed
versus predicted values and test the signiﬁcance of slope ¼ 1 and intercept ¼ 0
(Piñeiro et al. 2008). Finally, it is important to recall that the square of Pearson’s
correlation can also be used as a metric for measuring prediction performance since it
represents the proportion of the total variance explained by the regression model and
is called the coefﬁcient of determination denoted as R2.

Next, we present the mean absolute error (MAE) metric that measures the
difference between two continuous variables (observed and predicted). The MAE
can be calculated with the following expression:

MAETST ¼

XnþT

(cid:8)
(cid:8)
(cid:8)

(cid:8)
(cid:8)
yi (cid:1) bf xið Þ
(cid:8)

i¼nþ1

1
T

ð4:3Þ

Below we present another metric used to evaluate the prediction performance of
any statistical machine learning model; it was proposed by Kim and Kim (2016) and
is called mean arctangent absolute percentage error (MAAPE) that is calculated as
follows:

P

nþT
i¼nþ1 arctan

(cid:8)
(cid:6)
(cid:8)
(cid:8)
(cid:8)

yi(cid:1)bf xið Þ
yi

(cid:9)

(cid:8)
(cid:8)
(cid:8)
(cid:8)

T

MAAPETST ¼

ð4:4Þ

Although MAAPE is ﬁnite when the response variable (i.e., yi ¼ 0) equals zero,
since it has a satisfactory trigonometric representation. However, because MAAPE’s
value is expressed in radians, it is less intuitive, in addition to being scale-free. This
metric is a modiﬁcation of the mean absolute percentage error (MAPE) which is
problematic because it is undeﬁned when the response variable is equal to zero
(yi ¼ 0). MAAPE is also asymmetric since division by zero is deﬁned and is not a
problem. It is important to point out that there are other metrics for measuring
prediction accuracy for continuous data, but we only presented the most
popular ones.

The distinction between the training and test MSE is important since we are not
interested in how well the statistical machine learning method performs in the
training data set, due to the fact that our main goal is to perform accurate predictions
in the unseen test data. For example, a plant breeder may be interested in developing
an algorithm to predict disease resistance of a plant in new environments based on

4.5 Metrics for the Evaluation of Prediction Performance

131

records that were collected in a set of environments. We can train the statistical
machine learning method with the information collected in the set of environments,
but the interest is not in how well the statistical machine learning method predicts in
those previously collected environments. An environmental scientist can also be
interested in predicting the average annual rainfall in a municipality in Mexico, using
data from the last 20 years to train the model. Such measures could include sea
surface temperature, time, the yearly rotation of the earth, among others. In this case,
the scientist is really interested in predicting the average rainfall of the next 1 or
2 years, not in accurately predicting the years measured in the training set.

4.5.2 Binary and Ordinal Measures of Prediction

Performance

The binary and ordinal response variables are very common in classiﬁcation prob-
lems, where the goal is to predict which category something falls into. An example
of a classiﬁcation problem is analyzing ﬁnancial data to determine if a client will be
granted credit or not. Another example is analyzing breeding data to predict if an
animal is at high risk for a certain disease or not. Below, we provide some popular
metrics to evaluate the prediction performance of this type of data.

The ﬁrst metric is called a confusion matrix, which is a tool to visualize the
performance of a statistical machine learning algorithm that is used in supervised
learning for classifying categorical and binary data. Each column of the matrix
represents the number of predictions in each class, while each row represents the
instances in the real class. One of the beneﬁts of confusion matrices is that they make
it easy to determine whether the system is confusing classes. Table 4.3 shows a
sample format of a confusion matrix of C classes.

With Eqs. (4.5–4.8) we can calculate the total number of false negatives (TFN),
false positives (TFP), true negatives (TTN) for each class i, and the total true
positives in the system, respectively:

XC

TFNi ¼

nij

j¼1

j6¼i

ð4:5Þ

Table 4.3 Confusion matrix with more than two classes

Observed values

Predicted values
Class 1
n11
n21
⋮

nC1

Class 2
n12
n22
⋮

nC2

Class 1
Class 2
⋮

Class C

. . .
. . .
. . .
⋮
. . .

Class C
n1C
n2C
⋮

nCC

132

4 Overﬁtting, Model Tuning, and Evaluation of Prediction Performance

XC

TFPi ¼

nji

j¼1

j6¼i

XC

XC

TTNi ¼

nji

j¼1

k¼1

j6¼i

k6¼i

XC

TTPall ¼

njj

j¼1

ð4:6Þ

ð4:7Þ

ð4:8Þ

Below, we deﬁne the sensitivity (Se), precision (P), and speciﬁcity (Sp). The
sensitivity indicates the ability of our statistical learning algorithm to determine the
proportion of true positives that are correctly identiﬁed by the test. The precision is
the proportion of correct classiﬁcation of our statistical machine learning model and
represents the proportion of cases correctly classiﬁed, while the speciﬁcity is the
ability of our statistical machine learning model to classify the true negative cases,
that is, the speciﬁcity is the proportion of true negatives that are correctly identiﬁed
by the test. Under the “one-versus-all basis,” where each category is compared with
the composed information of the remaining categories, we provide the expressions
for computing the generalized precision, sensitivity, and speciﬁcity for each class i:

Pi ¼

Sei ¼

Spi ¼

TTPall
TTPall þ TFPi
TTPall
TTPall þ TFNi
TTNall
TTNall þ TFPi

pCCC ¼

TTNall
PC

PC

nij

ð4:9Þ

ð4:10Þ

ð4:11Þ

ð4:12Þ

i¼1

j¼1

The term pCCC denotes the proportion of cases correctly classiﬁed, which is a
measure of the overall accuracy, and when multiplied by 100, denotes the percentage
of cases correctly classiﬁed. Many times, this is the only metric reported for
measuring prediction performance in multi-class problems. However, PCCC alone
is sometimes quite misleading as there may be a model with relatively “high”
accuracy, but it predicts the “unimportant” class labels fairly accurately (e.g.,
“unknown bucket”). However, the model may be making all sorts of mistakes on
the classes that are actually critical to the application. This problem is serious when
in the input data the number of samples of different classes is very unbalanced. For

4.5 Metrics for the Evaluation of Prediction Performance

133

Table 4.4 Confusion matrix
with two classes

Observed values

Predicted values
False
True
fn
tp
tn
fp
fn + tn
tp + fp

Sum
tp + fn
fp + tn
n

True
False
Sum

tp denotes true positives, fp denotes false positives, fn denotes
false negatives, tn denotes true negatives, and n denotes total
number of individuals

example, if there are 990 samples of class 1 and only 10 of class 2, the classiﬁer can
easily have a bias toward class 1. If the classiﬁer classiﬁes all samples as class 1, its
accuracy will be 99%. This does not mean that it is an appropriate classiﬁer, as it had
a 100% error when classifying the samples of class 2. For this reason, reporting this
metric with those reported in Eqs. (4.9–4.11) is recommended in order to have a
better picture of the prediction performance of any statistical machine learning
method (Ratner 2017). Also, it is important to highlight that when the problem
only has two classes, the confusion matrix is reduced to Table 4.4.

From Table 4.4 the PCCC is calculated as tpþtn

tnþfp ,
and P ¼ tp
tpþfp . Also, when there are only two classes, González-Camacho et al.
(2018) suggest calculating the Kappa coefﬁcient (κ) or Cohen’s Kappa, which is
deﬁned as

n , while the Se ¼ tp

tpþfn, Sp ¼ tn

κ ¼

P0 (cid:1) Pe
1 (cid:1) Pe

,

n (cid:4) tpþfp

n þ fpþtn

where P0 is the agreement between observed and predicted values and is computed
by the PCCC described above for two classes; Pe is the probability of agreement
calculated as Pe ¼ tpþfn
n (cid:4) fnþtn
, where fp is the number of false
n
positives, and fn is the number of false negatives (Table 4.4). This statistic can
take on values between (cid:1)1 and 1; a value of 0 means there is no agreement between
the observed and predicted classes, while a value of 1 indicates perfect agreement
between the model prediction and the observed classes. Negative values indicate that
a prediction may be incorrect; however large negative values seldom occur when
working with predictive models. Depending on the context, a Kappa value from 0.30
to 0.50 indicates reasonable agreement (Kuhn and Johnson 2013). The Kappa
coefﬁcient is appropriate when data are unbalanced, because it estimates the pro-
portion of cases that were correctly identiﬁed by taking into account coincidences
expected from chance alone (Fielding and Bell 1997). It is important to point out that
this statistic was originally designed to assess the agreement between two raters
(Cohen 1960).

Another popular metric for binary data is the Area Under the receiver operating
characteristic Curve (AUC–ROC) and it ranks the positive predictions higher than
the negative. The ROC curve is deﬁned as a plot of 1 (cid:1) speciﬁcity or false positive
rate (FPR) as the x-axis versus its model sensitivity as the y-axis. For a given set of

134

4 Overﬁtting, Model Tuning, and Evaluation of Prediction Performance

Fig. 4.10 ROC curve for
the synthetic data

thresholds τ, it is an effective method for evaluating the quality or performance of
diagnostic tests, and is widely used in statistical machine learning to evaluate the
prediction performance of learning algorithms. Since the mathematical construction
of this metric is not required in this book, we illustrate its calculation with one simple
example. Assume that the observed (y), predicted probabilities (pi) and predicted
values (ŷ) obtained after implementing a statistical machine learning model are
y ¼ {1, 0, 1, 1, 0, 0, 1, 1, 0, 1}, pi ¼ {0.6, 0.55, 0.8, 0.78, 0.3, 0.42, 0.9, 0.45,
0.3, 0.88}, and by ¼ f1, 1, 1, 1, 0, 0, 1, 0, 0, 1}; then, using the following R code, we
can obtain many metrics for binary data (Fig. 4.10):

########################Libraries required########################

library(caret)
library(pROC)
##Observed (y), predicted probability (pi) and predicted values of

synthetic data##

y=c(1, 0,1, 1, 0, 0, 1, 1, 0, 1)
pi=c(0.6, 0.55, 0.8, 0.78, 0.3, 0.42, 0.9, 0.45, 0.3, 0.88)
yhat=c(1, 1, 1, 1, 0, 0, 1, 0, 0, 1)
xtab <- table(y,yhat)
confusionMatrix(xtab)

plot.roc(y,pi) #####This make the ROC curve plot

Confusion Matrix and Statistics

yhat
y 0 1
0 3 1
1 1 5

4.5 Metrics for the Evaluation of Prediction Performance

135

Accuracy : 0.8

95% CI : (0.4439, 0.9748)

No Information Rate : 0.6
P-Value [Acc > NIR] : 0.1673

Kappa : 0.5833
Mcnemar's Test P-Value : 1.0000

Sensitivity : 0.7500
Speciﬁcity : 0.8333
Pos Pred Value : 0.7500
Neg Pred Value : 0.8333
Prevalence : 0.4000
Detection Rate : 0.3000
Detection Prevalence : 0.4000
Balanced Accuracy : 0.7917

'Positive' Class : 0

It is important to point out that when there are more than two classes, these
metrics (accuracy, sensitivity, speciﬁcity, etc.) can be calculated on a “one-versus-
all” basis that consists of using each class versus the pool of the remaining classes, as
was illustrated for the confusion matrix with more than two classes (James et al.
2013).

When the statistical machine learning model discriminates correctly between the
two groups, it produces a curve that coincides with the left and top sides of the plot.
Under this scenario, the perfect model would have 100% sensitivity and speciﬁcity,
and the ROC curve would be a single step between (0,0) and (0,1) and would remain
constant from (0,1) to (1,1); this implies that the area under the ROC curve of the
model would be 1. In general, the larger the area under the ROC curve, the better the
model in terms of prediction performance. On the other hand, a completely useless
statistical machine learning algorithm would give a straight line (45(cid:6) diagonal line)
from the bottom left corner to the top right corner of the plot. Different statistical
machine learning models or the same model with different
training sets or
hyperparameters can be compared by superimposing their ROC curves in the same
graph. In practice, most of the time the values in the two groups overlap, so the curve
often lies between these extremes.

From this ROC curve, we can obtain a global assessment of the prediction
performance of the statistical machine learning method by measuring the area
under the receiver operating characteristic curve. This area is equal to the probability
that a random individual (person) of the sample with the presence of the target has a
higher value of the measurement than a random individual without the target.

When a statistical machine learning model is unable to discriminate between the
positive and negative classes, this means that it has low discriminatory power.
Therefore, only in statistical machine learning models that had good discriminatory
power we can be conﬁdent of the predictions they provide and furthermore those
models that provide a curve that lies considerably above the curve will be better.

136

4 Overﬁtting, Model Tuning, and Evaluation of Prediction Performance

Matthews correlation coefﬁcient (MCC). Introduced in 1975 by Brian Matthews
(1975) and regarded by many scientists as the most informative score that connects
all four measures in a confusion matrix, the Matthews Correlation Coefﬁcient is
typically used in statistical machine learning to measure the quality of binary
classiﬁcations and it is particularly useful when there is a signiﬁcant imbalance in
class sizes (data). MCC is calculated according to the following expression:

MCC ¼

p

tp (cid:4) tn (cid:1) fp (cid:4) fn
ﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃ
Þ (cid:4) tn þ fn
tp þ fp
Þ
ð
ð

Þ (cid:4) tp þ fn

Þ (cid:4) tn þ fp

ð

ð

ð4:13Þ

If any of the denominator terms equals zero in (4.13) it will be set to 1 and MCC
becomes zero, which has been shown to be the correct limiting value. It returns a
value between (cid:1)1 and 1, where 1 means a perfect prediction, 0 means no better than
random and (cid:1)1 means a total disagreement between predicted and observed values.
Next, we present the Brier score (Brier 1950) for categorical or binary data that

can be computed as

BS ¼ T (cid:1)1

XnþT

XC

i¼nþ1

c¼1

bπic (cid:1) dic
ð

Þ2,

ð4:14Þ

where bπi denotes the estimated probabilities (predictive distribution) derived from
the estimated model for observation i and dic takes a value of 1 if the categorical
response observed for individual i falls into category c; otherwise, dic ¼ 0. The range
of BS in Eq. (4.14) for categorical data is between 0 and 2. For this reason, we
suggest dividing by 2, that is, BS/2, to obtain the Brier score bound between 0 and 1;
lower scores imply better predictions (Montesinos-López et al. 2015a, b).

Finally, we describe the use of negative log-likelihood (MLL) to evaluate the
prediction performance. This metric has the characteristic that better forecasts have
lower values and for this reason, it is analogous to the MSE. For categorical data,

MLL ¼ (cid:1)

"

1
T

XnþT

XC

i¼nþ1

c¼1

#

1 yi ¼ k
f

g log bπicð

Þ

,

(cid:10)

P

data

the MLL is

are
Þ þ 1 (cid:1) yi
ð

binary,
Þ log 1 (cid:1) bπi
ð

where 1{y(i) ¼ k} is an indicator variable taken the value of 1 when the ith
observation is assigned to category c, for c ¼ 1, 2, . . ., C takes place in the ith
the
observation. When
to
nþT
i¼nþ1 yi log bπið
MLL ¼ (cid:1) 1
. Following, we provide
½
T
some advantages of using the MML as a measure of prediction performance: (a) it
has a simple deﬁnition that, from a purely intuitive point of view, seems to be a
reasonable basis on which to compare forecasts; (b) it is mathematically optimal in
the sense that estimates of parameters of calibration models ﬁtted by maximizing the
likelihood are usually the most accurate possible estimates (see Cassella and Berger
2002); (c) it is a generalization to probabilistic forecasts of the most commonly used
skill score for single forecasts; (d) the properties of the likelihood have been studied

reduced

(cid:11)

Þ

(cid:7)

4.5 Metrics for the Evaluation of Prediction Performance

137

at great length over the last 90 years, and as such is well understood; (e) it is both a
measure of resolution and reliability; (f) likelihood can be used for both calibration
and assessment: this creates consistency between these two operations; (g) use of the
likelihood also creates consistency with other statistical modeling activities, since
most other statistical modeling uses the likelihood, which is important in cases where
the use of forecasts is simply a small part of a larger statistical modeling effort, as is
the case of our particular business; (h) likelihood can be used for all types of
response variables; and (i) likelihood can be used to compare multiple leads,
multiple variables, and multiple locations at the same time in a sensible way with
a single score even when these leads, variables, and locations are cross-correlated.

4.5.3 Count Measures of Prediction Performance

Spearman’s correlation and the MML are recommended to measure the prediction
performance for count data.

for

the

the

rank

ﬁnal

values

observed

rank for

show how to get

the observed values:

For the application of the Spearman’s correlation, the formula given in Eq. (4.2)
for Pearson’s correlation can be used; however, instead of using the observed and
predicted values directly, these are replaced by their corresponding ranks. For
example, assuming that the observed and predicted values are y ¼ {15, 9, 12,
27, 6, 3, 36, 15, 21, 30} and by ¼ f20, 17, 24, 25, 3, 3, 34, 22, 21, 33}, we can
thus
rangoy ¼
{5, 3, 4, 8, 2, 1, 10, 6, 7, 9}. However, in this vector, observations 1 and 8 are the
2 ¼ 5:5.
same, and as such, their positions are added and divided by two, that is, 5þ6
Therefore,
rangoy ¼
the
{5.5,3, 4, 8, 2, 1, 10,5.5,7, 9}. Now the range for the predicted values is rangoby ¼
g, but again, since values 5 and 6 are the same, we add their
4, 3, 7, 8, 1, 2, 10, 6, 5, 9
f
2 ¼ 1:5.
ranges, and as this is repeated twice, it is divided by two and we get 1þ2
the
Therefore,
rangoby ¼
of
values
the
4, 3, 7, 8, 1:5, 1:5, 10, 6, 5, 9
to obtain the Spearman correlation, we
f
used the expression given in Eq. (4.2) for Pearson’s correlation, and instead of
using the original observed and predicted values, we used rangoy and rangoby. The
interpretation of this metric is equal to that of the Pearson correlation, that is, when it
is closer to 1, the prediction performance of the implemented statistical learning
method is better. It should be noted that when the number of repeated values in the
observed and predicted values is greater than two, the adjusted range is the sum of
the repeated ranges divided by the number of repeated values; this new range is then
given to the repeated values. In this case, it is also important to regress the observed
versus the predicted values to obtain the intercept and slope using the ranges of the
observed and predicted values.

range
g . Finally,

predicted

ﬁnal

is

is

It is also possible to use the MLL criteria to assess the prediction performance for
count data; however, the new expression is now based on minus the log-likelihood of
a Poisson distribution, which is equal to

138

4 Overﬁtting, Model Tuning, and Evaluation of Prediction Performance

MLL ¼

1
T

XnþT

h
(cid:1)bf xið Þ þ yi log bf xið Þ

(cid:2)

i

i¼nþ1

Again, when the values of MLL are lower, the observed and predicted values are

closer to one another.

References

Brier GW (1950) Veriﬁcation of forecasts expressed in terms of probability. Mon Weather Rev 78:

1–3

Buduma M (2017) Fundamentals of deep learning, 1st edn. O’Reilly, Sabastopol, CA
Burger SV (2018) Introduction to machine learning with R. Rigorous mathematical analysis, 1st

edn. O’Reilly, Sabastopol, CA

Cassella G, Berger RL (2002) Statistical inference. Duxbury, Belmont, CA
Cohen J (1960) A coefﬁcient of agreement for national data. Educ Psychol Meas 20:37–46
Cook D (2017) Practical machine learning with H2O. O’Reilly Media, Inc, Sabastopol, CA
Daetwyler HD, Calus MPL, Pong-Wong R, de los Campos G, Hickey JM (2012) Genomic
prediction in animals and plants: simulation of data, validation, reporting and benchmarking.
Genetics 193:347–365

Dewancker I, McCourt M, Clark S, Hayes P, Johnson A, Ke G (2016) A stratiﬁed analysis of

Bayesian optimization methods. arXiv:1603.09441v1

Efron B (1983) Estimating the error rate of a prediction rule: improvement on cross-validation. J

Am Stat Assoc 78(382):316–331

Fielding AH, Bell JF (1997) A review of methods for the assessment of prediction errors in
conservation presence/absence models. Environ Conserv 24:38–49. https://doi.org/10.1017/
S0376892997000088

González-Camacho JM, Ornella L, Pérez-Rodríguez P, Gianola D, Dreisigacker S, Crossa J (2018)
Applications of machine learning methods to genomic selection in breeding wheat for rust
resistance. Plant Genome 11(2):1–15. https://doi.org/10.3835/plantgenome2017.11.0104
Hagerty MR, Srinivasan S (1991) Comparing the predictive powers of alternative multiple regres-

sion models. Psychometrika 56:77–85. MR1115296

Hastie T, Tibshirani R, Friedman J (2008) The elements of statistical learning: data mining,

inference, and prediction, Springer series in statistics, 2nd edn. Springer, New York

James G, Witten D, Hastie T, Tibshirani R (2013) An introduction to statistical learning: with

applications in R. Springer, New York

Jarquín D, Lemes da Silva C, Gaynor RC, Poland J, Fritz AR et al (2017) Increasing genomic-
enabled prediction accuracy by modeling genotype (cid:4) environment interactions in Kansas
wheat. Plant Genome 10(2):1–15. https://doi.org/10.3835/plantgenome2016.12.0130

Kim S, Kim H (2016) A new metric of absolute percentage error for intermittent demand forecasts.

Int J Forecast 32(3):669–679

Koch P, Wujek B, Golovidov O, Gardner S (2017) Automated hyperparameter tuning for effective
machine learning. In: Proceedings of the SAS global forum 2017 conference. SAS Institute Inc,
Cary, NC. http://support.sas.com/resources/papers/proceedings17/SAS514-2017.pdf

Konen W, Koch P, Flasch O, Bartz-Beielstein T, Friese M, Naujoks B (2011) Tuned data mining: a
benchmark study on different tuners. In: Proceedings of the 13th annual conference on genetic
and evolutionary computation (GECCO-2011). SIGEVO/ACM, New York
Kuhn M, Johnson K (2013) Applied predictive modeling. Springer, New York
Lopez-Cruz M, Crossa J, Bonnett D, Dreisigacker S, Poland J, Jannink J-L, Singh RP, Autrique E,
de los Campos, G. (2015) Increased prediction accuracy in wheat breeding trials using a marker
(cid:4) environment interaction genomic selection method. G3 5(4):569–582

References

139

Lorena AC, de Carvalho ACPLF (2008) Evolutionary tuning of SVM parameter values in

multiclass problems. Neurocomputing 71:3326–3334

Lujan-Moreno GA, Howard PR, Rojas OG, Montgomery DC (2018) Design of experiments and
response surface methodology to tune machine learning hyperparameters, with a random forest
case-study. Expert Syst Appl 109:195–205

Matthews BW (1975) Comparison of the predicted and observed secondary structure of T4 phage

lysozyme. Biochim Biophys Acta Protein Struct 405:442–451

McKay MD (1992) Latin hypercube sampling as a tool in uncertainty analysis of computer models.
In: Swain JJ, Goldsman D, Crain RC, Wilson JR (eds) Proceedings of the 24th conference on
winter simulation (WSC 1992). ACM, New York, pp 557–564

Mesple F, Troussellier M, Casellas C, Legendre P (1996) Evaluation of simple statistical criteria to

qualify a simulation. Ecol Model 88:9–18

Montesinos-López OA, Montesinos-López A, Pérez-Rodríguez P, de los Campos G, Eskridge KM,
Crossa J (2015a) Threshold models for genome-enabled prediction of ordinal categorical traits
in plant breeding. G3 5(1):291–300

Montesinos-López OA, Montesinos-López A, Crossa J, Burgueño J, Eskridge K (2015b) Genomic-
regression. G3

enabled prediction of ordinal data with Bayesian logistic ordinal
5(10):2113–2126. https://doi.org/10.1534/g3.115.021154

Montesinos-López A, Montesinos-López OA, Gianola D, Crossa J, Hernández-Suárez CM (2018a)
Multi-environment genomic prediction of plant traits using deep learners with a dense architec-
ture. G3 8(12):3813–3828. https://doi.org/10.1534/g3.118.200740

Montesinos-López OA, Montesinos-López A, Crossa J, Gianola D, Hernández-Suárez CM et al
(2018b) Multi-trait, multi-environment deep learning modeling for genomic-enabled prediction
of plant traits. G3 8(12):3829–3840. https://doi.org/10.1534/g3.118.200728

Piñeiro G, Perelman S, Guerschman JP, Paruelo JM (2008) How to evaluate models:

observed vs. predicted or predicted vs. observed? Ecol Model 216:316–322

Ratner B (2017) Statistical and machine-learning data mining. Techniques for better predictive
modelling and analysis of big data, 3rd edn. CRC Press Taylor & Francis Group, Boca
Raton, FL

Renukadevi NT, Thangaraj P (2014) Performance analysis of optimization techniques for medical

image retrieval. J Theor Appl Inf Technol 59:390–399

Shalev-Shwartz S, Ben-David S (2014) Understanding machine learning from theory to algorithms.

Cambridge University press, New York

Shmueli G (2010) To explain or to predict? Stat Sci 25(3):289–310
Smith EP, Rose KA (1995) Model goodness-of-ﬁt analysis using regression and related techniques.

Wu S, Harris T, Mcauley K (2007) The use of simpliﬁed or misspeciﬁed models: linear case. Canad

Ecol Model 77:49–64

J Chem Eng 85:386–398

Open Access This chapter is licensed under the terms of the Creative Commons Attribution 4.0
International License (http://creativecommons.org/licenses/by/4.0/), which permits use, sharing,
adaptation, distribution and reproduction in any medium or format, as long as you give appropriate
credit to the original author(s) and the source, provide a link to the Creative Commons license and
indicate if changes were made.

The images or other third party material in this chapter are included in the chapter's Creative
Commons license, unless indicated otherwise in a credit line to the material. If material is not
included in the chapter's Creative Commons license and your intended use is not permitted by
statutory regulation or exceeds the permitted use, you will need to obtain permission directly from
the copyright holder.
