# Overfitting, Underfitting and General Model Overconfidence and Under-Performance Pitfalls and Best Practices in Machine Learning and AI

> *Source PDF: Overfitting, Underfitting and General Model Overconfidence and Under-Performance Pitfalls and Best Practices in Machine Learning and AI.pdf*
> *Extraction: Combined — markitdown raw text (base) + previous extraction supplements*

---

Overfitting, Underfitting and General
Model Overconfidence
and Under- Performance Pitfalls and Best
Practices in Machine Learning and AI

Constantin Aliferis and Gyorgy Simon

Abstract

Avoiding  over  and  under  fitted  analyses  (OF,  UF)  and  models  is  critical  for
ensuring  as  high  generalization  performance  as  possible  and  is  of  profound
importance for the success of ML/AI modeling. In modern ML/AI practice OF/
UF are typically interacting with error estimator procedures and model selection,
as  well  as  with  sampling  and  reporting  biases  and  thus  need  be  considered
together in context. The more general situations of over confidence (OC) about
models and/or under-performing (UP) models can occur in many subtle and not
so subtle ways especially in the presence of high-dimensional data, modest or
small sample sizes, powerful learners and imperfect data designs. Because over/
under confidence about models are closely related to model complexity, model
selection, error estimation and sampling (as part of data design) we connect these
concepts with the material of chapters “An Appraisal and Operating Characteristics
of  Major  ML  Methods  Applicable  in  Healthcare  and  Health  Science,”  “Data
Design,” and “Evaluation”. These concepts are also closely related to statistical
significance and scientific reproducibility. We examine several common scenar-
ios  where  over  confidence  in  model  performance  and/or  model  under  perfor-
mance  occur  as  well  as  detailed  practices  for  preventing,  testing  and
correcting them.

C. Aliferis (*) · G. Simon
Institute for Health Informatics, University of Minnesota, Minneapolis, MN, USA
e-mail: constantinaibestpractices@gmail.com

© The Author(s) 2024
G. J. Simon, C. Aliferis (eds.), Artificial Intelligence and Machine Learning in
Health Care and Medical Sciences, Health Informatics,
https://doi.org/10.1007/978-3-031-39355-6_10

477

478

C. Aliferis and G. Simon

 The Critical Importance of Over Fitting and Under Fitting

As important over and underfitting are, they are commonly hard to grasp by non-
technical users of ML/AI models and systems, not because their basic definitions
are hard to comprehend and recall, but because they tend to be created or to manifest
in subtle ways and often can be hard to detect before they create significant errors at
the time of model application or testing on human subjects. There are also signifi-
cant and pervasive misconceptions about OF and UF, stemming from earlier stages
in the science of ML/AI which we will clarify.

We first present a few classical and pedagogical examples of OF and UF before

proceeding with more precise definitions and systematic ways to address them.

 Introductory (Pedagogical) Example 1: Who Is the Teacher?

Consider the following thought experiment involving a university course on ML/AI
taught by the two lead authors of the present book. The students are tasked with
creating a computable rule (i.e. a decision model that can run in a computer) that can
classify individuals in the classroom as belonging to either of 2 classes: (a) students,
(b)  teachers.  Numerous  variables  have  been  captured  including  the  structural,
behavioral and other characteristics of all individuals involved. Some of them are
shown in the table below:

Overfitting, Underfitting and General Model Overconfidence and…

479

)
e
l
b
a
i
r
a
v
e
s
n
o
p
s
e
r
(

r
o
t
t
c
u
r
t
s
n
i

S
U
T
A
T
S
E
U
R
T

e
s
r
u
o
c

s
a

e
u
g
o
l
a
t
a
c

U
n
i

d
e
t
s
i
L

r
e
h
c
a
e
T

r
e
h
c
a
e
T

t
n
e
d
u
t
S

t
n
e
d
u
t
S

t
n
e
d
u
t
S

t
n
e
d
u
t
S

t
n
e
d
u
t
S

t
n
e
d
u
t
S

t
n
e
d
u
t
s

g
n
i
t
i
d
u
A

A
T

r
e
r
u
t
c
e
l

t
s
e
u
G

s
e
Y

s
e
Y

o
N

o
N

o
N

o
N

o
N

o
N

s
e
Y

o
N

o
N

o
d
u
J

k
c
a
l
b

t
l
e
b

s
e
Y

o
N

o
N

o
N

o
N

o
N

o
N

o
N

o
N

o
N

o
N

t
u
o
s
d
n
a
H

s
t
n
e
m
n
g
i
s
s
a

r
e
d
n
e
G

s
r
a
e

W

s
e
s
s
a
l
g

s
a
H

s
a
H

e
h
t

o
t

l
a
i
r
e
t
a
m

t
n
e
c
c
a

d
r
a
e
b

s
s
a
l
c

s
e
e
r
g
e
D

s
t
n
e
s
e
r
P

s
e
Y

s
e
Y

o
N

o
N

o
N

o
N

o
N

o
N

o
N

s
e
Y

o
N

M

M

F

F

M

M

F

M

F

B
N

F

o
N

s
e
Y

o
N

s
e
Y

s
e
Y

s
e
Y

o
N

o
N

s
e
Y

o
N

s
e
Y

s
e
Y

s
e
Y

o
N

s
e
Y

o
N

o
N

o
N

s
e
Y

o
N

o
N

o
N

o
N

s
e
Y

o
N

o
N

s
e
Y

o
N

o
N

s
e
Y

o
N

o
N

o
N

s
e
Y

s
e
Y

s
e
Y

s
e
Y

D
h
P

,

D
M

D
h
P

c
S
B

N
R

s
e
Y

D
m
r
a
h
P

s
e
Y

s
e
Y

s
e
Y

s
e
Y

s
e
Y

o
N

c
S
B

D
M

c
S
B

,

D
M

S
M

D
h
P

c
S
B

d
n
a

t
i
u
s

s
r
a
e

W

-
e
m
o
S

s
e
m

i
t

e
i
t

o
N

o
N

o
N

s
e
Y

o
N

o
N

o
N

o
N

o
N

o
N

e
m
a
N

n
o
m
i
S

s
i
r
e
f
i
l

A

h
t
i

m
S

g
n
e
h
Z

h
g
n
i
S

r
u
e
l
F
a
L

n
a
m
k
c
i
B

-
o
d
a
p
a
P

s
o
l
u
o
p

g
n
a
h
C

z
t
r
a
w
h
c
S

s
e
n
o
J

480

C. Aliferis and G. Simon

In this thought experiment there is a number of models that would classify cor-
rectly the participants in the class with respect to whether they belong in the teacher
or  learner  class.  Some  examples  of  perfectly  accurate  models  to  identify  teach-
ers are:

•  Model 1: hands out assignments and has accent
•  Model 2: has PhD and is male
•  Model 3: wears suit & tie sometimes OR has judo black belt
•  Model 4: name ends in “on” or starts with “a”

The more variables we measure, the more such accurate models we can construct.
Such models can achieve 100% accuracy in this class, however they obviously do
not generalize well to other similar ML/AI classes across many other universities.
We say that these models are overfitted.
Consider now the following examples:

•  Model 6: Has beard and wears glasses
•  Model 7: Has accent
•  Model 8: Has beard or a judo black belt

These models can achieve modest/poor accuracy in this class, however they may
generalize to similar (low) accuracy to other similar ML/AI classes across the many
universities and alternative models exist with better generalizable accuracy. We say
that these models are underfitted.

Finally consider the following examples:
Model 9: person is listed in the university catalogue as instructor for the course

and hands out assignments.

This model can achieve 100% accuracy in this class and will generalize to simi-
lar extent to other ML/AI classes across the many universities. We say that such
models are neither under nor overfitted.

The intuitions gained are that with datasets with a large enough number of
variables, it is easy enough to come up with models that are very accurate in
the train (discovery) data but that will fail to generalize to the general popula-
tion. Also that performance in the training data is a poor indicator of perfor-
mance in the population.

 Introductory Example 2: Over and Undertraining ANNs

Figure  1  shows  a  classical  experiment  in  training  ANNs  [1].  As  the  number  of
weight update iterations increases, the model learns the training data really well and
its generalization performance also increases (i.e., error decreases). There is how-
ever  a  “breaking  point”  (or  inflection  point)  where  more  training  does  increase
accuracy in the training data but decreases generalization performance.

Overfitting, Underfitting and General Model Overconfidence and…

481

Fig. 1  Illustration of
overfitting and underfitting

UNDERFITTING

OVERRFITTING

MODEL ERROR

POPULATION
(GENERALIZATION)
ERROR

TRAINING
DATA ERROR

TRAINING
ITERATIONS

The models to the left of the optimal point are underfitted and to the right are

overfitted.

The intuition gained is that there is a level of model “fit” that is ideal for this
data and anything above or below that point will lead to worse generalization
performance.  Performance  in  the  training  data  remains  a  poor  indicator  of
performance in the population, however.

 Introductory Example 3: Rich Simon’s OF Demonstration for
Genomics-Driven Discovery

In bioinformatics where dimensionalities are typically very high and sample sizes
small, OF is a particularly important danger. Simon at al published the following
empirical  demonstration  showcasing  that  depending  on  how  gene  selection  and
model  error  estimation  are  conducted,  there  are  different  degrees  of  biasing  the
estimates of model generalization error [2].

Train
&
GS

GS

Test

“Biased resubstitution”

test

Train
&
GS

test

train

“Fully crossvalidated”

“Partially crossvalidated”

482

C. Aliferis and G. Simon

Simon et al. generated models using feature (gene) selection (GS) procedures in
data constructed so that there is no predictive signal. They examined 3 protocols for
combining the same feature selection and classification algorithms:

Protocol 1 “Biased resubstitution” is when the gene selection takes place on all

data and the error estimation also takes place on all data.

Protocol 2 “Full cross validation” is when feature selection is done on a train-
ing portion of the data, model is fitted in the training portion and error is estimated
in a separate testing portion.

Protocol  3  “Partial  cross-validation”  conducts  feature  selection  on  all  data,
then models are built in a training portion and model error is estimated in a separate
testing portion.

Unbiased  error  estimation  should  indicate  that  a  model  fit  in  such  data  should
have no signal, that is perform as well as random coin flipping. Indeed the “fully
cross validated” protocol 2 (which we referred to as nested cross validation chapter
“The Development Process and Lifecycle of Clinical Grade and Other Safety and
Performance-Sensitive  AI/ML  Models”)  is  indeed  unbiased.  No  cross  validation
(protocol 1) has large bias that can reach estimates of perfect classification if enough
variables are used. Partial cross validation has in this analysis setting intermediate
bias, smaller than nested cross validation and lower than no cross validation.

The intuitions gained are that we can produce highly biased error estimates if
our protocols are not set up properly to avoid bias, especially in high dimen-
sional data. Also the same algorithms for classification, feature selection
etc. can lead to dramatically different quality of results depending of how
they are arranged in modeling protocols.

 Introductory Example 4: Simple and Complex Surfaces

A classic demonstration of under and overfitting of a regression function of a con-
tinuous outcome Y given a continuous input X is given in Fig. 2. We see both train-
ing data (blue circles) and unsampled population data (white circles). As shown the
complex model (wiggly line) fits the training data perfectly but fails in the popula-
tion (future) data. A simpler model (straight line) does much worse in the training
data, but better in the future data [3].

In Fig. 2 above, the better model is one that is more complicated than the straight

line and less complicated than the complex wiggly one.

The intuition gained is that over/under-fitting are directly related to the com-
plexity of the decision surface and how well the training data is fit. Successful
data  analysis  methods  balance  training  data  fit  with  complexity  since:  too
complex model (to fit training data well) leads to overfitting (i.e., model does
not generalize) whereas: too simplistic models (to avoid overfitting) lead to
underfitting (will generalize but the fit to both the training and future data will
be low and predictive performance small).

Overfitting, Underfitting and General Model Overconfidence and…

483

Fig. 2  Further illustration
of overfitting (top, 2a) and
underfitting (bottom, 2b)

 Definitions of OF and UF; the Broader Pitfalls of Overconfidence
and of Under-Performing Models. Bias-Variance Error
Decomposition View

Training data error of a model M is the error of M on the training data used
to derive M.

True generalization error of a model M is the error of M on the population
or distribution, from which training data used to derive M, were sampled from.

Estimated generalization error of a model M is the estimated error (via an
error estimator procedure applied on data samples) of M on the population or
distribution from which training data used to derive M were sampled from.

In controlled conditions, for example in simulation experiments, the true gener-
alization error of any model can be known. Typically the true generalization error is
unknown when dealing in non-trivial real-world problems, not constructed in the
lab, however.

484

C. Aliferis and G. Simon

Typically  (i.e.,  unless  training  sample  is  enormous),  model  training  data  error
used to estimate true generalization error leads to downward-biased (unduly opti-
mistic) estimates of generalization error.

Overfitting a model to data is creating a model that (a) accurately represents
the training data, but (b) fails to generalize well to new data sampled from the same
distribution (because some of the learned patterns in the training data are not repre-
sentative of the population).

Alternatively, an overfitted model is often defined as a model that is more com-

plex than the ideal model for the data and problem at hand.

Finally  some  authors  define  overfitting  as  learning  “noise”  in  the  data,  that  is
learning idiosyncrasies of the training data that are not present in the population [1].
Similarly, the notion of OF applies at the method, modeling protocol and system
level whereby an overfitting ML/AI method, modeling system, or modeling proto-
col/data science stack have a propensity to overfit models to the data [4, 5].

An overfitting ML/AI method, system, stack, or protocol is one with the
propensity to generate models that overfit.

Conversely:

Underfitting a model to data is creating a model that represents the training
data sub- optimally and also fails to perform well in the general population.
More broadly, an under fitted model will have true generalization error that is
larger than the true generalization error of the best possible model that can be
fit with the data in hand.

An underfitting ML/AI method, system, stack, or protocol is one with the
propensity to generate models that underfit.

A model M cannot be both over and under fitted: the model, either describes
the  training  data  well  and  generalizes  worse,  or  it  is  describes  the  training  data
poorly and also generalizes poorly, or finally, it is ideally fitted and describes both
the training and population data as well as possible. From a complexity perspective,
a model is either more or less or equally complex than ideal for the data and prob-
lem at hand. We will now delve deeper into these concepts using BVDE (i.e., bias-
variance decomposition of a model’s error).

Bias-variance decomposition perspective on OF and UF. As detailed in chap-
ter  “Foundations  and  Properties  of AI/ML  Systems”,  the  BVDE  describes  that  a
model’s error is (excluding of course measurement noise and inherent stochasticity
in the data generating function) a combination of two error components: (a) a “bias”
component and (b) a “variance” component. Everything else being equal, a highly
biased model is less complex, while a lower bias one is more complex than ideal.
Small sample size corresponds to higher variance, while larger sample size, leads to
smaller variance error component. For a fixed sample size and data generating func-
tion, there is an optimal model complexity leading to smallest model error possible.

Overfitting, Underfitting and General Model Overconfidence and…

485

Model complexity above this ideal level corresponds to over-fitted models, while
smaller  complexity  corresponds  to  under-fitted  models.  Moreover,  as  a  model  is
increasingly overfitted because of increasing model complexity, the fit of the train-
ing data will improve and thus the error on the training data will decrease while the
true model generalization error in the population will increase.

We further point out a fundamental asymmetry between UF and OF: empiri-
cal proof of OF of a model M1 can be obtained by showing that the true generalization
error of M1 is higher than what is expected by the performance in training data.

However empirical proof of UF of a model M1 requires showing that there is at
least one other model M2 that has better true generalization error than M1. M2 does
not need be the optimal model attainable. M1 will also have larger generalization
error than the optimal model Mopt.

In other words UF is a relative property with respect to some optimal model (or
other higher-performing models) achievable under the circumstances and not an
intrinsic property of a model under examination. Therefore establishing or prevent-
ing UF is harder and more open-ended than establishing or preventing OF.

Whereas at face value establishing that a model is OF and/or UF requires calcu-
lating its true generalization error, in practice we can circumvent this (a priori for-
midable)  difficulty  by  application  of  unbiased  and  efficient  estimators  of  true
generalization  error  (see  chapters  “The  Development  Process  and  Lifecycle  of
Clinical Grade and Other Safety and Performance-Sensitive AI/ML Models” and
“Evaluation”). It is also possible to apply statistical theory to infer that a model’s
estimated generalization error is not accurate or to infer with high confidence from
small-sample estimates that true generalization errors E1 and E2 of models M1 and
M2 are not the same (see details in section “How ML/AI Model OC is Generated in
Common Practice” below).

We now address two concepts of broader significance:

Over confidence in a model (OC) occurs when the analyst’s estimated gen-
eralization (population) error of the model is smaller than the true generaliza-
tion error.

Under confidence in a model (UC) occurs when the analysts’ estimated gen-
eralization error of the model is higher than the true generalization error.

Over  performance  of  a  model  (OP)  (relative  to  a  lower  estimated  perfor-
mance expectation) occurs when the estimated population error of the model
is smaller than the true population error. It is thus obvious that Under confi-
dence in a model = Over performance of the same model.

Under performance of a model (UP) occurs when the true population error
of the model is lower than the best possible model, or other better performing
models (that can be estimated from same sample size, on average). We can
also talk about under performance relative to a high performance goal set dur-
ing  the  model  development  planning  stages  (chapter  “The  Development
Process and Lifecycle of Clinical Grade and Other Safety and Performance-
Sensitive AI/ML Models”).

486

C. Aliferis and G. Simon

Under confidence in a model (or equivalently over performance of a model)
is not considered in practice a serious concern, however it entails opportunity costs
that may be significant. For example, it may trigger expensive and time consuming
but unnecessary additional modeling efforts, and may deprive productive use of the
model (in healthcare or health science discovery) in the meanwhile.

Under fitting is a special case error of under-performance relative to the best/
better models that could be produced under the circumstances specifically due to
lower than ideal model complexity and so that the lower performance is reflected
both in the training data and in the population.

Over fitting may or may not be a case of over-confidence in the models pro-
duced (and over- interpretation of the modeling results). For example, overfitting is
a special case of overconfidence when the (low) error in the training data is misin-
terpreted as an indicator of the (higher) generalization error. If however, we overfit
a model to the training data but use an unbiased error estimator, then we will not
have either over or under confidence in this over fitted model.

The major (high-level) pitfalls that the present chapter addresses is:

Pitfall 10.1
Producing models in which we have over-confidence.

Pitfall 10.2
Producing models that under perform.

With corresponding high-level best practices:

Best Practice 10.1
Deploy  procedures  that  prevent,  diagnose  and  remedy  errors  of  overconfi-
dence in, or overfitting of models.

Best Practice 10.2
Deploy procedures that prevent, diagnose and remedy errors of model under-
performance or underfitting.

Before embarking into specific situations where OC occurs, and corresponding
remedies, we introduce fundamental general principles spanning statistics and ML
that underlie these phenomena. These have value for understanding and developing
general approaches to prevent OC and UP.

Overfitting, Underfitting and General Model Overconfidence and…

487

 Fundamental ML Insights About the Three Sources
of Overconfidence (OC), Under Performance (UP) and their
Relationship to OF and UF: Biased Data Design, Biased Error
Estimation, Poor Model Selection

 (a)  As discussed, OF in one of its classic definitions occurs when the (low) error in
the training data is misinterpreted as an indicator of the (higher) generalization
error.  This  creates  a  seriously  dangerous  OC  about  the  model  in  question.
Similarly, if UF means failing to model the training data correctly in both train-
ing data and the population and such error is readily available to calculate, why
do we ever under fit models?

The  answer  related  to  real-life  practice  is  that  circa  2023  no  professional
practitioner of ML uses raw training data error to estimate generalization error.
To the extent that even sophisticated but biased error estimators are used, the
problems of OF and UF can still occur. Fundamental insight: as we explained
previously, there is an ideal complexity for a model for which, per BVDE, gen-
eralization error is minimized. Unfortunately this complexity is rarely known a
priori,  hence  we  typically  use  empirical  data  analysis  procedures  to  find  the
right complexity. These procedures are in practice a combination of search over
a space of possible models (i.e., model selection) combined with an error esti-
mation procedure used to evaluate the merit candidate models examined. If the
error estimator is downward-biased (optimistic) and/or the model selection is
incomplete, then it is in practice possible to find models that appear as better
(=more  promising)  than  they  are  by  themselves  (hence  OF)  or  compared  to
alternative  models  (hence  UF).  In  either  case  we  encounter  OC  and/or  UP
problems.

 (b)  Insight: Another  way  to  view  OC  due  to  OF  is  that  of  learning  “statistical
noise”, or stated differently, over fitting by learning idiosyncratic characteristics
or complex patterns of the data sample(s) that are by definition not representa-
tive of the population. Even when the error estimation procedures are not naïve
(e.g. use of training data error to estimate population generalization error), the
total sample itself (comprising both training and validation datasets) is not rep-
resentative. This is an instance of the error due to high variance per the BVDE.
 (c)  Insight:  alternatively,  learning  such  idiosyncratic  characteristics  may  be  the
result of actively selecting (as opposed to randomly sampling) training data or
because of poor data design such that the training data is not representative of
the targeted population and application goals (non- random sampling or more
generally  mismatch  of  the  available  population  with  the  target  population  as
explained in the chapter on data design).

These  situations  (i.e.,  poor  sampling  methods,  poor  error  estimators)  can
lead to generalization error estimates that are too optimistic (biased downward)
and learning spurious patterns translating into error in future application of the
model that are not present in the training data.

488

C. Aliferis and G. Simon

 (d)  Insight: with regards to UP, poor choice of training data can lead to UP, and the
same  holds  true  for  error  estimator  biases.  In  addition,  poor  data  design  and
model selection deficiencies can create UP.

To summarize, from a ML lens, OC and UP originate in 3 different stages/
aspects of ML modeling, and their combination:

 1.  OC/UP created due to data design, primarily sampling, so that the mod-

els and generalization error estimates lead to OC or UP.

 2.  OC/UP created due to error estimation so that the models’ generaliza-

tion error estimates lead to OC or UP.

 3.  OC/UP created due to poor choice of model, that is the choice of model
family, model fitting algorithm, model selection procedure etc., are such
that models with OC/UP characteristics ensue.

 Additional  Insights  Spanning  ML  and  Statistics  as  they  Relate
to OC and UP: Reproducibility, Cross Validation, Nesting and Biased
Post-hoc Reporting

 (a)  Insight: reproducibility suggests generalizability; operationalizing reproduc-
ibility via independent data validation. A commonly-followed principle of bio-
medical science is that credible results (e.g., in our context, models that have
small generalizable error) have to successfully reproduce, that is, models must
have same performance in data independent of the data used to discover these
results.

We emphasize that as widely accepted this principle may be, its merits are
not immutable but hold under assumptions. For example, in a classical statisti-
cal context, if we wish to verify that a statistical association found in data D1
reproduces  in  data  D2,  everything  else  being  equal,  sufficiently  high  power
(type II error probability, of false negative results) and sufficiently low alpha
(type I error probability, of false positive results) must be in place if we wish to
conclude with high certainty that a reproducible result is a true one. Similarly,
if we wish to verify that a model built from data D1 and having estimated gen-
eralizable error in the population of E1, by applying on data D2 with estimated
error E2 = E1, then D1 and D2 must be sampled randomly from the same popu-
lation and have high enough sample size so that comparison of E1 and E2 is
sufficiently powered at low alpha levels.

Without these assumptions holding, an application of the principle of repro-
ducibility may generate both false positive and false negative validation of orig-
inal results.

The way we typically operationalize this principle of validity via repro-
ducibility,  is  by  well-designed  independent  data  set  validation,  and  very

Overfitting, Underfitting and General Model Overconfidence and…

489

commonly, holdout or cross validation varieties (although many other variants
exist, the general principles still apply).

 (b)  Insight: single dataset independent (holdout) validation is an unbiased estima-
tor and protects against p-hacking and HARKing. We can use data D1 for fitting
a model M1 and then apply it on independent data D2 and compare generaliz-
able error estimates. This is mathematically equivalent to randomly splitting the
original data into a training (TR) dataset and the rest used for validation or test-
ing (TE) dataset. This is the “holdout estimator” [3] and the error measurement
in TE is an unbiased generalizable error estimate.

Application  of  the  holdout  estimator  protects  against  so-called  HARKing
which  stands  for  “Hypothesis  testing After  Results  are  Known”  [6].  This  is
because the effective alpha of a null hypothesis rejected by both discovery (TR)
and  validation  (TR)  datasets  is:  (nominal  alpha  employed  in  TR  *  nominal
alpha employed in TE). The total expected false positives in the discovery data-
set TR will be (nominal alpha in TR * number of hypotheses tested). This is the
alpha  of  using  just  one  dataset. The  total  expected  false  positives  of  original
discovery followed by independent validation will be (nominal alpha in TR *
nominal alpha employed in TE * number of hypotheses tested). In other words
the independent data testing reduces false positives by alpha (which is a very
small number, typically never to exceed 5%).

For our purposes of modeling with ML/AI, the “hypotheses” in question are
typically one or more model(s) for which we test whether its estimated error
meets or exceeds a performance threshold.

If more than one validation datasets are used in sequential steps of indepen-
dent validation, the probability of false findings drops exponentially fast to the
number  of  validations. The  effective  alpha  of  being  selected  in  both TR  and
(k-1) TE datasets is alphak which is a very small probability. For example, if
alpha = 0.05 then the effective alpha of reproduced null hypothesis rejection in
two independent datasets sequentially is 0.000125. In other words, when more
than  one  validation  datasets/steps  are  used,  the  probability  of  false  findings
drops  exponentially  fast  to  the  number  of  validations. These  same  principles
apply to ML/AI models whose validity we want to test.

 (c)  Insight: multi-split data validation (NCV) is also unbiased and less susceptible
to sampling variation. Because the split of the original data into TR and TE is
subject to random sampling variation and some splits will not be such that both
TR and TE are representative of the population (even though TR + TE may be)
we often use n-fold cross validation (NFCV) which is also a (in practical terms)
unbiased estimator and less susceptible to bad random splits (see chapter “The
Development  Process  and  Lifecycle  of  Clinical  Grade  and  Other  Safety  and
Performance-Sensitive AI/ML  Models”).  NFCV  and  its  cousin,  the  repeated
NFCV  (RNFCV)  approximates  the  less-variant  but  computationally  very
expensive (and almost never used in practice) all splits cross validation where
all splits are employed and averaged over.

 (d)  Insight: NCV and NFCV can still be biased via information contamination unless
nesting is employed. Any procedure that transfers information about TE to TR can

490

C. Aliferis and G. Simon

“contaminate” the unbiasedness of the CV estimator and lead to bias. Recall the
seminal paper of Rich Simon et al. discussed earlier in this chapter, where they
showed  that  feature  selection  can  contaminate  (i.e.  bias)  CV- derived  error  esti-
mates if it is conducted in all (TR + TE) data and that when it is conducted sepa-
rately  in  TR  and  TE,  such  contamination  does  not  happen.  Here  is  how  this
contamination takes place: by conducting e.g., feature selection on all data (using
a univariate association filtering procedure in this case), we generate false positive
features that happen to appear significant in this (TR + TE) dataset. Then we fit
models with these false positives in TR. The model works well in TE because the
features are specifically selected to work well in all data (i.e., TR + TE). However
the model will not work in the population because the features used in it include
false positives (in Simon et al. experiment they were all false positives because the
data was constructed to be devoid of predictive signal). In other words, the chosen
features were within the false positive expectation of the feature selection proce-
dure and thus by definition will not generalize in the population).

Compare the above scenario with conducting feature selection separately in
TR and in TE. The false positives of TR will not generalize to TE (because the
effective  probability  of  such  random  success  is  alpha2  instead  of  alpha  (see
“Fundamental ML Insights about the Three Sources of over Confidence (OC),
under  Performance  (UP)  and  their  relationship  to  OF  and  UF:  Biased  Data
Design, Biased Error Estimation, Poor Model Selection”, be hence very small,
and  thus  no  strong  feature  selection  bias  will  manifest.  With  10,000  irrele-
vant features at nominal alpha = 0.05 we will obtain 50 false positives features
that will lead to biased error estimation. With separate discovery (TR) and vali-
dation (TE) stages we will obtain 50 false positive features from TR but only
2.5 (on average) of them will survive the statistical testing in TE. As it turns out
2.5 random features do not have enough capacity to overfit a random distribu-
tion  of  a  random  outcome  conditioned  on  10,000  variables  and  thus  the  CV
error estimate with independent feature selection is unbiased.

Important notes:

•  The exact same type of bias can happen for any data pre-processing or analysis
step including: data normalization, data imputation, any type of feature con-
struction or transformation, and in general any other data input to a model that
is created by processing the full data because these operations encode informa-
tion about the data distribution that modeling algorithms can detect (hence bias-
ing error estimates because of information transference about TE to TR).

•  CV contamination bias does not need to be linked to a supervised analysis
step, that is linking the data processing to the values of the response variable
we wish to model. Even access to the joint distribution of inputs in TE (i.e.,
without reference to the response variable’s values in TE) or to any subset of
variables in TE may lead to models that have biased CV error estimates. For
example, we may construct biased models and error estimates using Principal
Component Analysis of the data, a procedure that is usually -but falsely- con-
sidered “failsafe” from over fitting (see assignment 12).

Overfitting, Underfitting and General Model Overconfidence and…

491

•  Whereas CV contamination may create biased models/error estimates under
the  above  circumstances,  whether  it  will  happen  and  the  degree  of  bias
greatly depends on the safety measures employed by the unit-level (i.e.,
component) data procedures. For example, if in the classical Simon et al.
experiment the feature selection is tightly controlled for family-wise errors,
the  bias  will  be  eliminated  (see  assignment  1  and  later  chapter  “Lessons
Learned  from  Historical  Failures,  Limitations  and  Successes  of  Health AI/
ML. Enduring Problems and the Role of Best Practices””). This implies that
we can seek to avoid contamination altogether via two general approaches:
one is by application of component procedures that anticipate and avoid con-
tamination  bias,  or  by  employing  non-contaminating  protocols,  of  which
nested CV is the paradigmatic example. We can (and often do) also deploy
both measures in combination (see assignment 9).

 (e)  Insight: CV nesting decouples TE from TR data and eliminates possibility for
CV contamination bias. Recall form chapter “The Development Process and
Lifecycle of Clinical Grade and Other Safety and Performance-Sensitive AI/
ML Models” that nested CV is CV where we split TR and TE sets as in usual
CV  and  then  split  further  the  TR  sets  into  TRTR  (“traintrain”)  and  TRTE
(traintest”) subsets (i.e., applying an embedded “inner” loop of CV inside the
TR data of the “outer” CV loop). In addition, all data processing (prior to
modeling) happens separately in every TRTE dataset. Because this procedure
separates data and conducts data pre-processing operations locally inside TR
and subsets of it, there is no information transfer (aka contamination) about
the  modeling  from TE  to TR.  Consequently,  the  CV  contamination  bias  is
eliminated.  For  a  trace  of  how  nested  cross  validation  operates,  and  high
level pseudo code the reader is referred to chapter “The Development Process
and Lifecycle of Clinical Grade and Other Safety and Performance-Sensitive
AI/ML  Models”.  For  development  of  the  repeated  nested  balanced  n-fold
cross  validation  procedure  by  extending  the  hold  out  estimator  and  model
selection see [3].

 (f)  Insight: Holdout and NCV can also be biased via selective or post-hoc report-
ing. Both nesting and full protocol specification prevent this bias. In the context
of real-life model development it is still possible to introduce bias in HO (hold-
out) and CV if we employ selective and post-hoc reporting. This type of prob-
lem occurs when many models are generated and the better ones are reported
selectively.  When  this  is  a  conscious  decision  by  the  creators  of  models,  it
amounts to a fraudulent representation of their procedures and results. As we
will  see,  selective  reporting  may  occur  without  ill  intent,  however,  when  the
analysis plans and protocols are not well-controlled and not well-designed to
avoid OC.

 (g)  Insight: We can detect bias in biased modeling protocols if they are accurately
and  thoroughly  described  using  special  tests  (described  later  in  the  pres-
ent chapter). It is not possible to diagnose that a modeling protocol is biased if
it is not thoroughly or accurately described.

492

C. Aliferis and G. Simon

Equipped with the general principles leading to OC and UP we will next delve into
more details of producing OC/UP models. We will also enrich the discussion with
pitfalls and corresponding BPs to avoid OC/UP.

The reader is reminded that the specific pitfalls and best practices discussed here
are  strongly  connected  and  provide  additional  technical  depth  and  operational
(micro) guidance to the overall (macro and meso) strategies and best practices of
chapters  “Principles  of  Rigorous  Development  and  of  Appraisal  of  ML  and  AI
Methods  and  Systems”  (methods  development  and  validation)  and  “The
Development  Process  and  Lifecycle  of  Clinical  Grade  and  Other  Safety  and
Performance-Sensitive AI/ML Models” (model development and lifecycle).

 How ML/AI Model OC Is Generated in Common Practice

 Models Are Allowed to Have Inappropriately Large Complexity
with Respect to Data Generating Function Complexity
and to Available Sample Size

How do we measure model complexity? In older literature (and even today in some
cases) this situation, and overfitting more broadly, was misconstrued as having too
many parameters in the model (with respect to the available sample size, and the
fixed complexity of the data generating function). With newer methods, the number
of parameters does not matter as long as any number of available protective meth-
ods are used to reduce the effective model parameters and more generally the com-
plexity  of  the  fitted  models,  all  within  an  appropriate  family  of  learners  (i.e.,
functions that match the form of the data generating function) [3, 4].

The  distinction  between  original  parameters  and  effective  parameters  can  be
made  clear  using  the  example  of  SVMs  [3].  In  this  model  family  the  effective
parameters (the support vectors, in other words these data points defining the bound-
aries  of  each  label  class)  cannot  exceed  the  sample  size.  Thus  we  may  have  for
example 1,000,000 variables and sample size n = 1000 which automatically restricts
the number of effective parameters to at most 1000 (a 3-order of magnitude reduc-
tion over the initial data dimensionality in this example). It is also instructive that
SVM mathematical bounds on generalization error do not depend on original data
dimensionality but on the support vectors [4].

We have mentioned repeatedly so far that in general the complexity of the model
(the effective dimensionality of which is a major factor) must be balanced against
the complexity of the data generating function and the sample size. In some more
detail: (a) For a fixed complexity data generating function, and a fixed sample size
there is an optimal model complexity that leads to the model with the best true gen-
eralization error. Similarly, (b) for a fixed complexity data generating function, and
a fixed model complexity, there is a minimum sufficient sample size that leads to the
model with the true generalization error that is within an acceptable distance delta
from the optimal error achievable. This follows from the “Bias-variance decomposi-
tion of error” discussed earlier.

Overfitting, Underfitting and General Model Overconfidence and…

493

 Ignoring Statistical Uncertainty

Observing  that  a  model  has  an  acceptable  point  estimate  performance  does  not
guarantee  generalizable  performance  since  this  estimate  may  be  subject  to  large
variation due to small sample size. The 95% (or higher) confidence interval (CI) of
the estimate and/or the 95% credible interval (CrI) must also be taken into account
in order to establish that the performance estimate will generalize [7, 8].

X% Confidence Interval (CI) of a point estimate P of the performance of
model M developed from a fixed sample of size n: the range of values contain-
ing  X%  of  all  point  estimates  when  sampling  multiple  times,  developing  a
new model for each sample and estimating its generalization error (or other
performance  metric),  when  the  true  value  of  developing  a  model  from  this
population is P.

X%  Credible  Interval  (CrI)  of  a  point  estimate  P  of  the  performance  of
model M developed from sample of size n: the range containing with proba-
bility  X%  the  true  value  of  the  generalization  error  (or  other  performance
metric) of M applied to the population.

The CrI may be viewed—in Bayesian terms—as equivalent in meaning to the
credible region of the posterior probability density function of the model error
in the population containing X% of the total density function symmetrically
around  the  point  estimate,  given  the  data  and  prior  knowledge.  In  practice
however,  the  CrI  can  and  is  often  estimated  empirically  with  various
procedures.

A  subtler  way  to  fall  victim  to  small  sample-induced  statistical  variability  is
when the error estimator used has high variance that is not taken into account, or
when among various unbiased estimators a high variance one is used instead of the
lowest variance one. For example, both the holdout estimator and the n-fold cross
validation (NFCV) estimator are unbiased/near-unbiased (respectively) however the
holdout  has  higher  variance  than  the  NFCV  [3,  9]. This  is  why  over-reliance  on
“independent study verification: often (and falsely) treated by journals and others as
a “gold standard” is a pitfall that needs to be avoided.

 Using Biased Estimators or Introducing Bias in Unbiased Ones

When the procedure for estimating generalization error is not unbiased, or when the
unbiasedness is compromised by implementation decisions, the estimates will be
upward or downward biased. Omitting correction of this bias or applying an inap-
propriate correction leading to a downward error estimate leads to overconfidence.
At least two common situations lead to such OC in practice:

494

C. Aliferis and G. Simon

 (a)  The first case stems from using uncorrected or poorly corrected bootstrapping
which is a biased estimator because it employs resampling with replacement
[10].  The  classifier  model  is  produced  by  the  learning  procedures  who  see
unnatural replicates of the same true cases in the data and thus may appear to
perform better than in real life where such replicates do not exist (e.g., in omics
data where the combination of high dimensional data inputs are unique for each
subject, yet in bootstrapping based analyses they are viewed repeatedly as if
they were naturally occurring in the population).

 (b)  The second case occurs in temporal data analyses where there is a progressive
distribution shift overt time. If the application of an unbiased estimator such as
cross validation does not take into account time-dependent distribution changes,
then a temporal bias will be introduced into a nominally unbiased estimation
procedure (see chapter “Data Design”).

 Uncorrected Multiple Statistical Hypotheses Tests, “Data
Dredging”, “Fishing Expeditions”

The  problem  of  multiple  uncorrected  statistical  hypotheses  manifests  when  a
researcher (or a ML/AI discovery procedure) conducts not just one but many tests
of statistical hypotheses and does not address the combined effective false positive
rate error across the totality of all tests conducted. For example, consider an algo-
rithm (or researcher, the exact same principles apply) that conducts a test of associa-
tion between variables Vi and Vj in a data sample S. Such tests can be intermediate
steps of more complex algorithms, or be used to compare and evaluate models.

Assume that for the observed level of association, the data sample size and the
desired type II error (i.e., probability to reject the null hypothesis when it does not
hold, i.e., the “power” of the test), the type I error (i.e., the probability to generate a
false positive rejection under the null hypothesis) may be quite small, often set at the
level of at most 5%. Now if the researchers or ML/AI algorithm conducts for exam-
ple  1000  such  tests,  it  will  produce  1000*0.05  =  50  false  positive  results.
Epidemiologists describe the problematic practice of generating such false positives
as “data dredging” or “fishing expeditions” [11].

However  these  terms  falsely  imply  to  some  non-technical  audiences,  the
wrong idea that whenever a discovery algorithm conducts massive amounts of
statistical  tests  for  “unbiased”  or  “hypothesis  free”  discovery,  there  will  be
unavoidable massive numbers of false positives. This is not true, however, as
these  multiple  testing  can  be  corrected  with  many  powerful  and  practical
ways as we will show in section “Preventing, Detecting, and Managing UF/UP”.

Overfitting, Underfitting and General Model Overconfidence and…

495

The problem of uncorrected multiple hypothesis testing can also manifest when
we produce a number of models, the estimates of generalizable error of which have
considerable variance due to small sample size (as explained before), and we falsely
conclude that one or more have desirable performance because of uncorrected mul-
tiple testing of the significance or the performance estimates.

Note that so far it should be clear that the notion of OC is not confined to model
building and ML/AI, but is more general since it encapsulates broader notions of
producing generalizable knowledge from small sample data, and is also related to
classical statistical considerations of reliable estimation and inference.

 Selective Reporting of Results, “Filedrawer Bias”,
“Publication Bias”

Imagine a modeler that aims to build an outcome classifier model for outcome Ox
on the basis of data inputs V. He proceeds as follows: he develops 100 models by
using a variety of techniques and estimates the generalization error for each model.
Then he reports the model with the best error estimate but does not report that this
was selected out of 100 models. This setup will invariably lead to over fitted/OC
model reporting as evidenced by the following simple demonstration: imagine that
none of these models has error better than flipping a coin (i.e., by chance, e.g. 50%
accuracy for a binary outcome in a distribution with prior probability of positives
equal to 50%). However if the 99% CI of estimated error for (e.g., sample size = 50
the sample size used is [0.3, 0.68], then we expect that half of the models will show
accuracy >50% and some will be higher than 65%.

The data scientist of this hypothetical experiment may also apply a calcula-
tion of 95% CI of a model that performs at accuracy 68% in this sample size
level ([0.53, 0.80]) and may conduct a statistical test showing that the pro-
duced “best” model is statistically significantly better than random chance (at
5% alpha). All of this string of errors and over-interpretations is undetectable
by statistical tests or by reviewers unless they know the precise selection pro-
tocol employed.

 “Analysis Creep” and Uncontrolled Iterative Modeling

A milder, all-too-common and often well-intentioned version of the file drawer bias
pitfall  occurs  in  the  form  of  non-rigorous  iterative  analysis,  which  is  sometimes
refered to as bias due to “analysis creep”. We illustrate by extending the previous
hypothetical scenario:

Consider the data scientist of section “Selective Reporting of Results, ‘Filedrawer
Bias’, ‘Publication Bias’” who this time builds the outcome classifiers with careful

496

C. Aliferis and G. Simon

cross validation where the data modeling decisions are fixed in the training data and
error  estimation  takes  place  in  the  testing  data  portions  of  the  cross  validation.
Assume that estimated error is 20%. The PI of the scientific project (or project man-
ager of a commercial product based on the model) reviews the results and suggests
trying out an additional classifier algorithm. The data scientist does so (repeating
the same modeling protocol but with the new classifier) and produces a new model
with error 17%. An external consultant reviews the results and suggests specific data
transforms  and  feature  construction  which  lead  (by  application  of  same  analysis
protocol on the same data) to a model  with  error  14%. In yet  another “improve-
ment” step a new hire in the data science team decides to explore some recalibration
method that leads to a model with estimated 10% error. And so forth, until the final
model appears to be a near-perfect one.

The problem with this scenario is that no measures are taken to isolate the intro-
duction of new methods from the statistical variation of error estimation so that the
final model’s error estimates are not over fitted to the specific train-test configura-
tion used.

An especially challenging aspect of the “analysis creep” problem that makes
it invisible to even the most rigorous scientists who are not experts in data
science, is that each step may be well designed and perfectly appropriate as a
1-step analysis, but when a series of such steps is executed, then over fitting
and over confidence take place.

 Choice of a Few and Non-representative Datasets; Unusual
Populations; Broad Claims From Too Few or Too Easy Datasets

Just like non-rigorous (intentionally, or not) selection of models can create over fit-
ted models and over interpreted results, the same is true for datasets.

•  This  problem  is  particularly  common  in  ML/AI  method  development
where the developers of methods often choose themselves the datasets to
test  their  methods.  It  is  entirely  possible  that  developers  may  choose  to
report performance of their methods in data that is “friendly” to the meth-
ods, and to omit reporting (or event testing) performance on harder datasets.
•  This situation is also common in ML and data science competitions that pit
dozens or hundreds of methods against one another over one or a small
number of datasets such that the results are typically overly-specific to the
small  choice  of  datasets  (as  well  to  the  specific  data  design  and  perfor-
mance metrics used).

•  A related problem arises in applied discovery settings where multiple data

sets exist and only a subset is used for development and testing.

Overfitting, Underfitting and General Model Overconfidence and…

497

The  above  scenarios  create  a  lack  of  generalizability  problem  where  certain
methods or models perform well in some highly selected data or populations and
therefore  fail  to  generalize  beyond  those  (chapters  “Principles  of  Rigorous
Development and of Appraisal of ML and AI Methods and Systems” and “Lessons
Learned from Historical Failures, Limitations and Successes of AI/ML in Healthcare
and the Health Sciences. Enduring Problems, and the Role of BPs” give more details
in the context of limitations of competitions).

 Many Teams, Same Data, No Coordinated Unbiased Protocol

Yet  another  version  of  the  “analysis  creep”  and  uncontrolled  iterative  or  parallel
modeling  pitfalls  occurs  when  (typically)  in  cooperative  consortia  or  other  large
scale collaborative efforts, several teams are working on analyzing the same data
with different algorithms, analysis protocols, error estimators etc.

In the absence of a coordinated unbiased model selection and error estimation
protocol  that  applies  to  all  analyst  teams,  the  statistical  variation  of  even
good  modeling  methods  with  same  large-sample  performance  can  lead  to
large apparent differences in performance in the discovery data that do not
generalize to the population. The problem can be further compounded when
the various teams employ learning algorithms, model selection and error esti-
mation  methods  with  widely  different  characteristics  where  the  selected
model overall can simultaneously suffer from under-performance and over-
confidence  in  it  (see  chapter  “Lessons  Learned  from  Historical  Failures,
Limitations and Successes of AI/ML in Healthcare and the Health Sciences.
Enduring Problems, and the Role of Best Practices” for findings from a major
centralized benchmark study revealing related problems).

 Hard-to-Reproduce, Non-Standardized Data Input Steps

In some types of clinical as well in discovery data modeling the data inputs may be
subjectively assessed and these assessments may have a low degree of reliability
across different individuals or settings. If this aspect has not been incorporated in
the model performance estimation (e.g., a single observer is responsible for all sub-
jective assessments in both training and test data) then inflated performance esti-
mates are produced.

Examples of this pitfall exist in a variety of settings, for example in assessment
by surgeons of operation aspects, in pathologist evaluation of slides for less-than
ideally  standardized  features,  skin  lesion  assessments,  proteomic  spectra  peak
determinations, etc.

498

C. Aliferis and G. Simon

 Normalization/Data Transforms that Require Entirety of Sample

In some modeling settings data needs to be processed (e.g., via normalization, dis-
cretization or other operations) by looking across the totality of data, that is includ-
ing  train  and  test  data.  This  automatically  creates  the  previously-discussed
“contamination” of the error estimates of the test set by the train set weakening their
independence and creating a bias in the error estimates.

Moreover from a purely practical viewpoint, the application of these models on
new data not used in the model development/evaluation cycle is problematic because
it requires re-normalization (or re-processing) of all data starting from early model
development to the latest application.

 Learners Learn the Wrong Patterns via Spurious Co-Occurrence

A classical example from epidemiology, involving causality, is the “yellow finger”
which when is due to tar-staining from smoking, predicts several diseases resulting
from smoking (e.g., lung and cardiovascular diseases, cancer). However the classi-
fiers in these cases cannot discover that eliminating the yellow stains does not alter
the  disease  risk. That  requires  eliminating  the  confounding  cause  (smoking)  and
thus such purely predictive findings/models do not generalize when interventions
are considered.

 Selective Control of Factors that Can Lead to OC

Recall that R. Simon et al. showed the role of feature selection for error estimation
bias  in  high  dimensional  disease  classification  in  “complete”,  “partial”  and  “no”
cross validation. Every single analysis step and parameter (not just feature selec-
tion)  that  affects  the  performance  of  models  and  the  error  estimators  has  to  be
controlled accordingly. This includes hyperparameters, model families, normaliza-
tion discretization, imputation etc.

 OF and OC Problems in Patient-Specific Modeling

In recent years efforts have been made to develop models specifically tailored to
individual patients. These are commonly based on time series data obtained from
each individual and they may have predictive or causal foci [12, 13].

The  advantages  of  such  modeling  is  that  (1)  they--  may  avoid  masking  and
distribution- mix effects of population data and (2) may be able to focus more effec-
tively on mechanisms and characteristics of individuals for precision and personal-
ized  medicine.  Possible  disadvantages  are  that  they  (3)  may  require  dense  time
series data, (4) they may fail to leverage vast amounts of data from other individuals
that apply to all individuals in the population, (5) by their very nature they cannot
model severe and irrevocable, or rare or singular outcomes (e.g., death), (6) do not

Overfitting, Underfitting and General Model Overconfidence and…

499

deal well with abrupt distribution shifts of the individual (whereas same shifts may
be learnable at the population level and anticipated by population models) and (7)
do not provide guarantees for generalizing to other individuals.

Characteristics (6) and (7) are therefore related to possible over fitting and over

confidence.

 Bespoke and Hand-Created AI Models

In certain types of AI modeling, most commonly in mathematical and engineering-
based modeling that may incorporate limited data-driven aspects and is not fully
automated as in ML, the ability to generate large numbers of models by the same
individuals and over short periods of time is severely limited. This precludes the
collection of large numbers of models and datasets in which the success of such
models can be rigorously statistically evaluated. It is entirely possible under these
circumstances for a few models to be performing well in some task yet the overall
model building process is neither salable nor demonstrably generalizable. Creation
of a handful of successful models by hand and evaluation on a handful of cases, for
example a hand-crafted model to predict the safety of a drug in a specific RCT, may
say very little about whether this model would apply to other RCTs and even less
about whether the modeling methodology could be carried out by other modelers or
for other drugs.

 How UP ML/AI Models Are Commonly Created

 Not Considering the Right Method Family, Not Considering
Enough Method Families in Model Selection

A  common  reason  for  under  performant  models  is  not  exploring  the  right
model family in model selection. For example, when the data generating func-
tion  is  non-linear  and  discontinuous  but  we  explore  only  linear  regression
models. When the right family is not known a priori the corresponding prob-
lem is not exploring enough method families in model selection.
The above are common occurrences when data scientists have strong prefer-
ence  for  a  small  number  or  narrow  methods,  or  when  vendors  focus  on  a
specific technology that is used across diverse tasks even when it is not the
most appropriate for the task.

 Insufficient Data Preparation

This pitfall applies to all steps typically used for data preparation such as feature
construction  and  selection,  normalization,  discretization,  distribution  transforms,
etc. Such steps can greatly enhance model performance when employed correctly or
hurt performance when they are ignored or conducted sub optimally.

500

C. Aliferis and G. Simon

 Insufficient Model Selection Hyper Parameter Space

This pitfall occurs when the right model families are explored but without sufficient
exploration of their hyper parameter values.

 1-Step Modeling Attempts

In  chapter  “The  Development  Process  and  Lifecycle  of  Clinical  Grade  and  Other
Safety and Performance-Sensitive AI/ML Models” we elaborated on the importance
of initial modeling which typically must subsequently be refined and enhanced (since
the first attempt seldomly meets performance goals in complicated modeling prob-
lems). A 1-stage analysis procedure does not benefit from a graduated understanding
of the data and task at hand and their interaction with the modeling method deployed.

 Ignoring Best Known Methods

In some cases for specific domains and tasks prior theoretical and empirical work
has established the predominance or superiority of specific classes of methods. It is
therefore likely to under perform if these methods are not included in the model
selection, at a minimum as “starting points” with baseline performance that other
methods must match or exceed. See chapter “Principles of Rigorous Development
and of Appraisal of ML and AI Methods and Systems” for proper scope of empiri-
cally evaluating new methods and for appraising existing methods.

 Ignoring Official or Reference Specifications on Methods Use

A commonly-encountered pitfall is applying strong methods but in ways that are
inconsistent with suggested use. These suggested uses include: (a) the ways these
methods have been previously tested during reference method development and in
validation  phases  leading  to  established  strong  results;  (b)  the  specific  ways  the
inventors of these methods have used them in the primary (“official”) publications
associated  with  them.  Chapter  “Lessons  Learned  from  Historical  Failures,
Limitations  and  Successes  of  AI/ML  In  Healthcare  and  the  Health  Sciences.
Enduring Problems, and the Role of Best Practices” describes case studies where
such bias led to suboptimal performance.

 UP Problems in Models for Individual Patients

As explained previously in “OF and OC Problems in Patient-Specific Modeling”,
individual-specific modeling has a priori both advantages and disadvantages over

Overfitting, Underfitting and General Model Overconfidence and…

501

population-wide modeling. Characteristics (3), (4), (5), (6) and (7) are linked to pos-
sible under performance of such models relative to population based models.

 Failing to Obtain Power Sample Analysis and More Generally
to Control Effects of Sample Size on Modeling

In  the  absence  of  a  power-sample  analysis  or  knowledge  of  learning  curves,  the
modeler  does  not  know  whether  better  results  can  be  obtained  by  increasing  the
sample size, or what is the minimum sample size needed to reach the stated goals of
the modeling.

In ML the power-sample calculation problem is more complicated than
traditional statistical power-sample analysis. This is because in inferential
statistics  we  need  to  know  what  is  the  required  sample  size  to  reject  a  null
hypothesis  with  a  desired  alpha  and  power,  and  closed  formulas  exist  that
describe  this  relationship  for  applicable  statistical  tests.  In  ML  however,  in
addition to the need to test a model’s performance against a null hypothesis, we
first need to find a good model. Thus the power-sample calculation is further
complicated by the learning protocol’s learning curve that is the function that
describes the generalization error of the best model learned (on average) as a
function of sample size. Depending on the problem, learning curves may sug-
gest that we need more or less sample size than the one needed to reject a null
hypothesis centered on the model’s performance. To make things worse, learn-
ing curves are not known a priori for the vast majority of practical problems.

Based  on  the  above  we  can  summarize  on  the  pitfalls  related  to  OC  and  UP

including OF and UF.

Pitfall 10.1
Producing models in which we have over-confidence

10.1.1.  Models  are  allowed  to  have  inappropriately  large  complexity  with
respect to data generating function complexity and to available sample size.

10.1.2.  Ignoring  statistical  uncertainty  of  strong  point  estimates  of
perfromance.

10.1.3. Using biased estimators or introducing bias in unbiased ones.

10.1.4. Not correcting multiple statistical hypotheses tests.

10.1.5. Selectively reporting strongest models/results.

10.1.6. Conducting uncontrolled iterative modeling and succumbing to “anal-
ysis creep”.

502

C. Aliferis and G. Simon

10.1.7. Using non-representative datasets, unusual populations, and making
strong claims from too few or too easy datasets.

10.2.8. Not coordinating analysis over many teams and same data via appro-
priate unbiased protocols designed for collaborative work or competitions.

10.1.9. Using hard-to-reproduce, non-standardized data input steps.

10.1.10.  Employing  normalization/data  transforms  that  require  entirety
of sample.

10.1.11.  Allowing  learners  to  learn  the  wrong  patterns  via  spurious  co-
occurrence; uncontrolled structural relations and biased sampling; and ignor-
ing domain knowledge that reveals the above.

10.1.12. Controlling only some of the factors that can lead to OC.

10.1.13. Inappropriate modelling for individual patients and over interpreta-
tion of their generalizability.

10.1.14.  Insufficient  studies  of  scalability  and  generalizability  in  bespoke
hand-created AI models.

Pitfall 10.2
Producing models that are under performing

10.2.1. Not deploying the right model family in model selection; not explor-
ing enough method families in model selection.

10.2.2. Insufficient data preparation.

10.2.3.  Insufficient  exploration  of  the  hyper  parameter  space  during  model
selection.

10.2.4. 1-stage modeling attempts.

10.2.5. Ignoring best known methods for task and data at hand (either as base-
line comparators or starting point).

10.2.5.  Ignoring  official  specifications  and  prototypical  (reference)  use  of
employed methods.

10.2.6. Models for individual patients: lack of dense time series data, failure
to leverage population models that apply to the specific individual (including
ignoring or under modeling severe and irrevocable rare or singular outcomes),
not addressing well abrupt distribution shifts of the individual (whereas same
shifts may be learnable at the population level).

10.2.8. Failing to obtain power sample analysis and more generally control
effects of sample size on modeling.

Overfitting, Underfitting and General Model Overconfidence and…

503

 Preventing, Detecting, and Managing OC and OF

We now address specific best practices for preventing overfitted models and over
confidence in models.

 Manage Model Complexity with Respect to Data Generating
Function Complexity and to Available Sample Size

Whereas manually balancing complexity against sample size is a formidable hurdle,
well-designed modern ML methods, protocols, and systems encapsulate multiple
methods that achieve this balance automatically or semi-automatically:

 1.  Regularization: is a methodology whereas model parameters’ values are driven
to zero by model fitting algorithms, as much as data allows. Regularization is
broadly  used  by  “penalty+loss”  learners  (see  chapter  “An  Appraisal  and
Operating Characteristics of Major ML Methods Applicable in Healthcare and
Health Science”) for example SVMs, Lasso regression, regularized classical sta-
tistical variants such as regularized Cox regression, regularized Logistic regres-
sion,  regularized  Discriminant  Function  analysis,  etc.  The  “loss”  term  is  a
mathematical  expression  of  how  accurately  a  model  represents  training  data,
whereas  the  “penalty”  term  captures  the  combined  complexity  of  the  model
(e.g.,  sum  of  squared  weights  of  inputs).  Regularization  is  sometimes  closely
related to the notion of function smoothness, that is the preference for modeling
functions in which the impact of a small change in the data generating function
inputs leads to small changes to the classification output (this is mathematically
equivalent to the “maximum margin classifier” inductive bias of SVMs).

 2.  Dimensionality reduction: is a set of methods that map the original input vari-
ables to a much smaller number of mathematical combinations (see chapter “An
Appraisal  and  Operating  Characteristics  of  Major  ML  Methods Applicable  in
Healthcare and Health Science”). A prototypical example is Principal Component
Analysis (and variants) where the original input variables are replaced by inde-
pendent linear combination functions (the principal components, such that the
totality of data variance is captured by the totality of the principal components).
The  fitted  models  use  as  inputs  the  reduced  input  representation  or  a  subset
thereof.

 3.  Feature  selection:  (see  chapters  “Foundations  and  Properties  of  AI/ML
Systems”  and  “An  Appraisal  and  Operating  Characteristics  of  Major  ML
Methods Applicable in Healthcare and Health Science”) is a set of methods that
select a small number from the original variables such that ideally all informa-
tion about the response variable is retained and all redundant variables are dis-
carded. Strong feature selection helps reduce model complexity to the absolute
necessary (i.e., only those features that have indispensable information about the
response).

504

C. Aliferis and G. Simon

 4.  Bayesian Priors and Bayesian ensembles: in Bayesian maximum a posteriori
model selection-based modeling, prior probabilities over the model space con-
sidered can be used to ensure that models with appropriate (smaller) complexity
are given more attention in smaller sample sizes and as sample size grows more
complex  models  can  be  selected.  Moreover,  via  Bayesian  Model  Averaging,
many (all in theory, but just a few in practice in most practical settings) models
can be combined to provide an “ensemble” classification. Complex models are
expected to have smaller posteriors in small sample sizes than simpler ones and
thus to drive more the overall ensemble decisions whereas in larger sample sizes,
the opposite is true.

 5.  Algorithm-embedded complexity control: several ML algorithms have embed-
ded means to control complexity of produced models. For example, decision tree
learners  prune  the  trees  when  they  reach  branching  points  with  small  sample
sizes. Random forest learners apply feature selection at each branching point of
each  fitted  tree  and  forbid  trees  larger  than  a  set  size  (which  is  a  tunable
hyper - parameter). ANNs map large input spaces to potentially smaller hidden
layer spaces or incorporate pruning and other regularization steps. SVMs trans-
form non-linearly separable input spaces to linearly separable ones via kernel
functions.  Structured  Risk  Minimization  in  SVMs  progressively  considers
classes of models (corresponding to kernels) of strictly increasing complexity
with  guarantees  strictly  monotonic  improvements  in  generalization  error.
Boosting methods start from simpler models and extend them to address only the
cases not classified correctly. And so on (see chapter “An Appraisal and Operating
Characteristics  of  Major  ML  Methods  Applicable  in  Healthcare  and  Health
Science”).

 6.  Statistical model/data complexity measures: classical examples being the AIC
and BIC metrics [7] that are used in statistics to characterize models with respect
to their complexity and fit against the training data. Simpler models are preferred
by the human analysts, everything else being equal.

 7.  Model  selection  and  combination  approaches:  it  is  common  to  create  and
apply ML/AI model fitting and selection protocols that combine several of the
above approaches.

Using nested cross validation approaches to find the best models and estimate their
generalization error is a common approach that allows model complexity to grow
only as much it helps improving estimates of the true generalization error.

 Characterize and Manage Statistical Uncertainty

Typically this entails: (a) Calculating confidence and credible ntervals for models
(and of the models’ parameter values when appropriate). (b) Testing models against
the null hypothesis (commonly being that: there is no predictive signal in the data at
hand, or a network model has properties no different than a random one, etc.). This
is easily accomplished with a standard label reshuffling or other randomization tests

Overfitting, Underfitting and General Model Overconfidence and…

505

(chapter  “The  Development  Process  and  Lifecycle  of  Clinical  Grade  and  Other
Safety and Performance-Sensitive AI/ML Models”). (c) Conducting tests of model
stability (when appropriate) to sample size. (d) Obtaining measures of stability of
model output or decisions with respect to variation in the model inputs. (e) Reducing
sampling variance by choice of most powerful/low variance estimators and proto-
cols.  Especially  for  n-fold  cross  validation  schemes,  repeating  the  analysis  >50
times empirically has been shown to reduce train-test split variance [14].

 Use Unbiased Estimators or Correct Estimation Bias

Among  classical  estimators  and  model  selection  protocols  Repeated  n-fold  cross
validation (RNFCV) is a particularly robust error estimator which can also be used
for  powerful  model  selection  when  nested  (RNNFCV).  In  larger  sample  sizes  a
repeated  nested  holdout  may  be  a  more  computationally  efficient  alternative.
Finally, in very small sample situations a repeated nested leave on out can be used
as alternative to RNNFCV [3].

We also recommend “locking” models at some predefined stage in the modeling
process and not allowing further tampering with locked models. Publishing open-
box models can certainly provide a strong form of such locking, although it may
also be employed as an internal strategy during model development. A similar, very
stringent but much less practical best practice is preregistering ML studies and
models [15].

The exclusive use of independent data testing for establishing or testing for
generalizability is commonly and often required by journals, funding study
sections etc. It is NOT recommended however as a single, or “privileged” vali-
dation methodology in the present volume, since it is subject to between and
within-population sampling variation so that discrepancies between the mod-
el’s performance in the discovery + testing CV datasets and the independent
validation dataset may be due to: (1) sampling from a different population, or
(2) not having perfect power in the independent validation dataset (see section
“Additional  Notes  on  Strategies  and  Best  Practices  for  Detection, Analysis
and Managing Both OC and UP” for Details). The latter danger is especially
salient in domains where sample sizes are never very large, or are very costly,
ethically challenging, or slow to obtain. Nested CV by comparison eliminates
the  first  source  of  errors  since  we  ensure  that  the  discovery  and  validation
datasets come from the same population. See also chapter “Lessons Learned
from Historical Failures, Limitations and Successes of AI/ML in Healthcare
and the Health Sciences. Enduring Problems, and the Role of Best Practices”
for a striking demonstration of variability between discovery and validation
sets and results in a landmark benchmark study where same analyses were
done with original and swapped discovery-validation sequences [16].

506

C. Aliferis and G. Simon

 Correct or Control Multiple Statistical Hypotheses Tests

The venerable but outdated Bonferroni correction is not recommended since
it reduces the power of the discovery procedure dramatically relative to more
modern procedures. The Benjamini- Hochberg (or similar) methods for cor-
related and uncorrelated p-values can be used to more effectively control the
acceptable  ratio  of  false  positives  (e.g.,  the  analyst  can  set  thresholds  on
p- values that does not lead to more than 10% false positive rate on the reported
results) [17].

We also note that certain algorithms, for example constrained-based causal mod-
eling algorithms (e.g., GLL, LGL and others) have embedded control of false posi-
tives due to multiple statistical testing [18].

 Thoroughly Specify and Report the Procedure Used
to Obtain Results

The entirety of the analyses and modeling applied on data must be reported so
that the possibility for over fitting is properly assessed by third parties. We
emphasize  that  even  if  the  original  model  developers  share  their  modeling
algorithms and data in full, unless they specify the analyses employed in their
entirety (i.e., entirety of model selection and error estimation steps), it is not
possible to determine whether the models are over fitted without additional
verification  in  independent  data.  On  the  contrary,  when  the  entirety  of  the
analysis protocol is disclosed, both the final model’s over fitting and the whole
protocol’s propensity to over fit or OC can be assessed (often with a simple
label reshuffling test as demonstrated in chapter “The Development Process
and Lifecycle of Clinical Grade and Other Safety and Performance-Sensitive
AI/ML Models”).

 Conduct Iterative and Sequential Modeling via
Unbiased Protocols

A robust and proven such protocol is the previously mentioned RNNFCV protocol
in which every new method or parameter value added to the previous analysis steps
is incorporated in the nested model selection along with all other methods and the
whole modeling is repeated from scratch (ideally with previous model fitting
steps cached for improved tractability). Other such protocols may be constructed,
but it is essential for the modelers to establish first their robustness to bias due to
iterative modeling. Note that when we perform model selection over k algorithms
and their associated hyper parameter value sets in RNNFCV, the results are mathe-
matically equivalent if we first model-select over the first methods, then insert the

Overfitting, Underfitting and General Model Overconfidence and…

507

second, then the third and the winner of the first two, and so on until all methods
have been examined. This demonstrates that since doing all methods at once is not
overfitting/producing over-confident error estimates, the sequential procedure over
the same set of methods, will not either.

See assignment 8 for a practical demonstration of proper vs improper iterative

modeling.

 Use Representative Datasets, Appropriate Populations and Make
Generalizability Claims from Appropriate Datasets

In  all  modeling  settings,  clinical  or  biological  criteria  must  be  used  to  establish
appropriateness of data used. Rigorous phenotypic definition and extractions from
the EHR, for example, will ensure that only and all human subjects that apply to the
modeling goals are considered for discovery and validation (see chapter on Data
Design).

When new method development or validation and benchmarking are pursued,
we recommend using all publicly available datasets that apply to the task and
if they are too many, to use a randomly-stratified selection of representative
datasets (e.g., with certain distributions, dimensionalities, sample sizes, etc.).
See  also  chapter  “Principles  of  Rigorous  Development  and  of Appraisal  of
ML and AI Methods and Systems”.

In  situations  where  discovery  is  pursued  via  secondary  analysis  of  many
pre-existing datasets, it is very common for results to disagree among the
various datasets. One way to address the problem is to “Round-Robin” anal-
ysis and examine the nature and robustness of results when some of the data
are used for discovery and some for validation.

It is often useful in such situations of multi-dataset analyses, to adopt a differ-
ent perspective and focus not on whether model M or property P discov-
ered, e.g., in datasets 1–10 holds in datasets 11–20 but, rather, what are
the variant and invariant properties (both predictively and structurally)
across  this  collection  of  datasets  and  what  is  the  robustness  of  models
developed  in  a  subset  of  the  datasets  on  the  remainder  datasets  (with  the
round-robin analysis conducted so that it examines or approximates all pos-
sible discovery-validation splits).

Stated differently, the discrepancies between datasets (and models summariz-
ing properties of these data) should not be viewed automatically as “errors”
but  also  considered  as  potentially  valuable  indicators  of  systematic  differ-
ences between health care systems, research designs, model organisms etc.,
depending on the data measurement and sampling designs used.

508

C. Aliferis and G. Simon

 Coordinate Analysis over Many Teams and Same Data via
Appropriate Unbiased Protocols

The recommendations of “Conduct Iterative and Sequential Modeling Via Unbiased
Protocols” apply unchanged here as well.

 Use Easy-to-Reproduce, Standardized Data Input Steps

Techniques to facilitate this practice include automating all subjective input mea-
surements and establishing equivalence or sufficiency of their information content.
Alternatively,  establishing  that  subjective  measurements  can  be  standardized  via
protocols that ensure low interrater variability.

 Employ Normalization/Data Transforms that Do Not Require
Entirety of Sample

This  is  self-explanatory  and  follows  directly  from  the  nature  of  validation-to-
discovery  information  contamination  that  biases  CV  error  estimates  as  explained
previously.

 Prevent Learners from Learning the Wrong Patterns via Spurious
Co-Occurrence; Control Structural Relations and Biased
Sampling; and Incorporate Domain Knowledge and Face-Validity
Expert Testing that May Reveal Spurious Learning

Essential to the above are robust batch processing bias and error detection and cor-
rection protocols such as the ones routinely used in high-throughput omics assay-
based  studies.  Moreover,  causal  modeling  algorithms  can  reveal  spurious  and
confounded relations and patterns of bias. In addition, a diversity of datasets that
fully covers the space of application of the desired models must be used for training
and validation. Finally, model explanation techniques can be valuable by revealing
exactly  what  the  learning  algorithms  and  corresponding  models  created  by  those
have  learned  especially  when  combined  with  expert  review  of  such  models  (see
chapter  “The  Development  Process  and  Lifecycle  of  Clinical  Grade  and  Other
Safety and Performance-Sensitive AI/ML Models”).

These latter precautionary practices unfortunately rely on existence of sufficient
domain theory that can be used to detect anomalies in the models. It is entirely pos-
sible (and indeed common) in certain domains for robust such theory to be lacking,
however  (e.g.,  in  high-density  omics  studies,  or  complex  mental  health/human
behavior  and  other  domains  where  the  complex  mechanisms  governing  the  data
generating processes have not been conclusively  or completely established). It is
important in such cases to non over-interpret models and to test the propensity of

Overfitting, Underfitting and General Model Overconfidence and…

509

human  experts  to  construct  invalid  conceptual  explanations  in  support  of  models
(see chapters “Principles of Rigorous Development and of Appraisal of ML and AI
Methods and Systems” and “The Development Process and Lifecycle of Clinical
Grade  and  Other  Safety  and  Performance-Sensitive  AI/ML  Models”  for  expert
biases and modeling of expert judgment). It is equally important to always consider
the possibility that models encapsulate valid new knowledge previously unknown in
the field, thus an expert’s rejection of some model or result should be a piece of
evidence in evaluating model validity and not grounds for immediate dismissal.

 Control all Factors that Can Lead to OF/OC via TE→TR
Contamination, by Using Nested Model Selection

The most important way to operationalize this is to use a nested protocol for cross
validation and error estimation as previously explained, and at the same time ensure
that all possible data analysis steps that may transmit information from the test sets
to the train sets (and thus introduce bias in the error estimation) are isolated inside
the nested part of the protocol.

 Carefully Combine Modelling of Individual Patients
with Population Modeling

Most importantly, any individualized modeling must be compared with population
modeling and ensemble (combined) with population models whenever appropriate
and feasible.

 Do Not Over-Interpret the Generalizability of Bespoke
Hand- Created AI Models Unless Sufficient Validation Data Can
be Obtained to Support Such Claims

Unfortunately  bespoke,  hand-crafted  modeling  efforts  typically  cannot  -  by  their
very nature - be readily evaluated by automated procedures and data for such evalu-
ations  are  scarce.  It  is  prudent  in  such  cases  to  address  modeling  successes  and
failures as isolated incidences.

Whenever it is feasible to create automated computable procedures that repli-
cate the bespoke methodologies, this allows transitioning non-scalable human
modeling to scalable and testable AI/ML modeling with obvious advantages
for increasing the scale, scope, speed, cost-effectiveness and verifiability of
similar modeling in other problem domains/settings.

510

C. Aliferis and G. Simon

 Preventing, Detecting, and Managing UF/UP (Table 1)

Practices 10.5.1.–10.5.7. in Table 1 are self-explanatory and follow directly from
the principles presented earlier. The next best practice 10.5.8. require some explana-
tion, however:

 Conduct Power Sample Analysis and More Generally Characterize
the Effects of Sample Size on Modeling. Use Dynamic Sampling
Schemes Whenever Appropriate

With  regards  to  the  ML/AI  power-sample  planning  as  introduced  in  “Failing  to
Obtain Power Sample Analysis and more Generally Control Effects of Sample Size
on Modeling”, contrary to classical statistical hypothesis testing where closed for-
mulas exist to calculate the minimum required sample for achieving a desired alpha
(% of false positive rejections of the null, or type I error) and beta (probability of
false negative failure to reject the null under the alternative hypothesis, or type II
error, aka power) levels for some statistical test of choice, when designing ML/AI
modeling, two additional factors come into play the first relevant to predictive mod-
eling, and the second related to causal modeling:

 (a)  The learning curve of the used learning algorithm. The learning curve describes
the errors of the algorithm’s output as a function of sample size. Generally the
learning curves are not known and closed formulas do not exist.

Table 1  Lists specific practices for preventing under fitted models

10.5.1. Deploy and explore all appropriate learning method families in
model selection
10.5.2. Deploy and explore all relevant data preparation steps to the domain
and task at hand
10.5.3. Systematically and sufficiently explore the hyper parameter space
10.5.4. Anticipate several preliminary and refinement modeling stages and
incorporate in nested designs to avoid overfitting
10.5.5. Inform analyses by literature so that best known methods for task
and data at hand are explored
10.5.6. Follow theoretically and empirically proven specifications and
prototypical (reference) use of employed methods (including the official
specific ways the developers of methods have presented in the corresponding
primary publications)
10.5.7. Models for individual patients: Use dense time series data, leverage
population models that apply to the specific individual (including modeling
severe and irrevocable rare or singular outcomes), search for and model abrupt
distribution shifts of the individual (including learning and modeling shifts at the
population level)

Overfitting, Underfitting and General Model Overconfidence and…

511

 (b)  The causal sparsity (i.e., density or connectivity) of the causal process that
generates the data. A sparse causal data generating process requires less sample
size to be discovered because the sample size required for conditional indepen-
dence tests (CITs) that are at the core of causal structure discovery algorithms
increases exponentially (in unrestricted distributions) to the number of condi-
tioning variables in the CIT (see chapter “Foundations of Causal ML”). This
latter number is directly linked to the density of the generating causal graph.
After a causal graph has been discovered, causal effects of interventions need
be estimated and these estimations also require sample size that grows exponen-
tially to the controlled confounders which are similarly linked to the density of
the causal generating process.

Whereas  classical  statistical  power-sample  analysis  can  be  applied  once  a  good
model or causal structure has been identified (to reject the null predictive model or
the estimate causal effects, with high confidence) the sample size required for the
predictive model discovery or the causal structure discovery and effect estimation
are separate considerations. Indeed it is entirely possible for the sample size required
for the former to be larger, equal or smaller than the sample size required for the
latter.

10.5.8.  Best  practice  strategies  to  address  these  sample  size  and  power
design needs for ML/AI model building include:

 1.  Using sensitivity analysis for results over convenience samples by itera-
tively  reducing  available  sample  size  (sub-sampling  on  a  convenience
sample). If, for example, by reducing the sample size, models retain their
predictivity, this strengthens the empirical argument that the learning curve
has  reached  convergence  and  additional  sample  will  not  increase
performance.

 2.  Use of simulations, ideally with real life data where ground truth models
are known or re-simulation (as covered in chapter “Principles of Rigorous
Development and of Appraisal of ML and AI Methods and Systems”).
 3.  Use of domain knowledge (if it exists) about the nature of causal structure
underlying the data, or the nature of predictive or causal functions to be
learned.

 4.  Use of network-scientific knowledge about the nature of connectivity of

real life networks.

 5.  Reference to prior robust results in very similar domains to formulate and

justify assumptions about a successful analysis.

 6.  Use of dynamic sampling schemes such as adaptive trial designs, Bayesian

posterior updating or active learning-based sampling.

512

C. Aliferis and G. Simon

 Additional Notes on Strategies and Best Practices
for Detection, Analysis and Managing Both OC and UP

 (a)  Label  reshuffling  tests.  The  label  reshuffling  procedure  (as  for  example
employed  by  R.  Simon  et  al.  and  elaborated  in  chapter  “The  Development
Process  and  Lifecycle  of  Clinical  Grade  and  Other  Safety  and  Performance-
Sensitive AI/ML Models”) retains the joint distribution of input variables how-
ever it decouples (on average) the inputs to the response variable. Thus it creates
an “on average” null distribution from which we sample data and build models.
•  By comparing the performance of the best model we found to this null dis-
tribution we can test the statistical hypothesis that it is as good as random
choice (i.e., not reject the null).

•  Additionally, by looking at the mean of this null distribution we can establish
whether the overall analysis protocol biases the error estimates (under the
null) and by how much.

 (b)  Reanalysis  (with  both  original  and  unbiased  or  otherwise  improved  protocols,
including  single  coordinated  protocols  as  needed).  This  is  especially  important
when one wishes to verify the validity of models produced by third parties. For
example, when suspicion exists for selective reporting of analyses, or when suspi-
cion of under fitting exists. It is not possible to conduct definitive “forensic style”
re-analyses without having access to the full range of data and modeling protocols
used in the original analyses. We caution that the ability to run “black box code”
on the same data and reproduce the exactly same results, is inappropriately pre-
sented as a top-tier level of confidence by some guidelines (see discussion in chap-
ter “Lessons Learned from Historical Failures, Limitations and Successes of AI/
ML in Healthcare and the Health Sciences. Enduring Problems, and the Role of
Best Practices”), yet it does not suffice because depending on how the black box
operates, the models and performance estimates may be OC, UF, or both.

 (c)  “Safety  net”  model  application  measures  for  ensuring  that  a  model  is  not
applied  to  the  wrong  person  or  population  (see  chapters  “The  Development
Process  and  Lifecycle  of  Clinical  Grade  and  Other  Safety  and  Performance-
Sensitive AI/ML Models” and “Characterizing, Diagnosing and Managing the
Risk of Error of ML and AI Models in Clinical and Organizational Application”
for details on models' “knowledge cliff” and managing prediction risk).

 (d)  Model stability considerations. Variable coefficients or very large variation of
output given small change in input variable inputs must be dealt with caution as
they may imply OF/OC or UP/UF. However we note (without going into full
technical details that would require very substantial space to cover), that it is
entirely possible for unstable models, markers, causal edges, coefficients etc. to
be meaningful and reproducible because of underlying equivalence classes in
the data.
Therefore unstable findings should be examined more deeply, but not discarded

outright.

Overfitting, Underfitting and General Model Overconfidence and…

513

In summary the following best practices allow managing (i.e., preventing, diag-

nosis and/or correcting) OC and UP:

Best Practice 10.1
Deploy procedures that prevent, diagnose and remedy errors of over con-
fidence in, or over fitting of, models.

10.1.1.  Manage  model  complexity  with  respect  to  data  generating  function
complexity and to available sample size using:

 1.  Regularization
 2.  Dimensionality reduction
 3.  Feature selection
 4.  Bayesian Priors and Bayesian ensembles
 5.  Algorithm-embedded capacity control
 6.  Statistical model/data complexity measures
 7.  Model selection
 8.  Combination approaches

10.1.2. Characterize and manage statistical uncertainty.

10.1.3.  Use  unbiased  estimators  of  model  performance  or  correct  bias  of
biased estimates.

10.1.4.  Lock  models  at  predefined  stages  in  the  modeling  process  and  not
allow further tampering with locked models.

10.1.5. Correct multiple statistical hypotheses tests (explicitly or implicitly).

10.1.6.  Thoroughly  specify  and  report  the  entirety  of  procedures  used  to
obtain models so that independent verification of generalizability is possible.

10.1.7. Conduct iterative or sequential modeling via unbiased protocols.

10.1.8. Use representative datasets, appropriate populations and make gener-
alizability claims from appropriate datasets.

10.1.9. Coordinate analysis over many teams and same data via appropriate
unbiased protocols.

10.1.10. Use reproducible, standardized data input steps.

10.1.11. Employ normalization /data transforms that do not require entirety of
sample  (or  confine  such  within  discovery  and  validation  datasets
independently).

514

C. Aliferis and G. Simon

10.1.12. Prevent learners from learning the wrong patterns via spurious co-
occurrence; control structural relations and biased sampling; and incorporate
domain knowledge-based review that reveals spurious learning.

10.1.13. Control via nested model selection all (and not just a few) factors that
can lead to OC.

10.1.14. If possible, combine modelling of individual patients with population
modeling.

10.1.15. Do not over-interpret the generalizability of bespoke hand-created AI
models  unless  sufficient  number  of  validation  data  sets  can  be  obtained  to
support such claims. Consider creating computable versions of model hand-
crafting modeling when possible.

10.1.16. Use label reshuffling testing for evaluating the overfitting/overconfi-
dence bias of the whole analysis protocol.

10.1.17. Apply with appropriate caution Independent dataset validation and
be mindful of dangers of over-interpretation of positive and negative results.

10.1.18. Instead of pursuing strict and exact reproducibility across datasets,
study the variant and invariant findings from these datasets.

10.1.19. Whenever possible, use reanalysis (with both original and unbiased
or  otherwise  improved  protocols,  including  single  coordinated  protocols  as
needed) when verifying the validity of models produced by third parties.

10.1.20. Use domain knowledge and related face-validity tests by experts to
flag potential model errors. The experts themselves may be prone to biases or
domain theory may not cover models’ new findings so do not over-interpret
experts’ objections.

10.1.21. Apply “safety net” measures for ensuring that a model is not applied
to the wrong person or population.

10.1.22.  Examine  stability  of  models,  parameters  and  other  findings  and
examine more deeply unstable findings. Be aware that it is possible for unsta-
ble findings and models to be perfectly valid.

Best Practice 10.2
Deploy  procedures  that  prevent,  diagnose  and  remedy  errors  of  model
under-performance or underfitting.

10.5.1.  To  maximize  predictivity,  deploy  and  explore  all  relevant  learning
method families in model selection.

10.5.2. To maximize predictivity and generalizability, deploy and explore all
relevant data preparation steps to the domain and task at hand.

Overfitting, Underfitting and General Model Overconfidence and…

515

10.5.3. To maximize predictivity, systematically and sufficiently explore the
hyper parameter space.

10.5.4. Anticipate  several  preliminary  and  refinement  modeling  stages  and
incorporate them into sequential nested designs to avoid overfitting.

10.5.5.  Inform  analyses  by  methods  literature  so  that  best  known  methods
for task and data at hand are always explored along with novel methods.

10.5.6.  Follow  theoretically  and  empirically  proven  specifications  of  refer-
ence prototypical or official use of employed methods.

10.5.7.  In models for individual patients: use dense time series data, leverage
population models, search for and model abrupt distribution shifts of the indi-
vidual (including learning and modeling shifts at the population level).

10.5.8. Conduct power sample analysis and more generally characterize the
effects of sample size on modeling. In the absence of knowledge of learning
curves, use:

 1.  Dynamic sampling schemes whenever appropriate,
 2.  Sensitivity analysis for results over convenience samples by iteratively
reducing available sample size (sub-sampling on a convenience sample),

 3.  Simulations,
 4.  Domain knowledge,
 5.  Network-scientific knowledge,
 6.  Reference to prior robust results in very similar domains,
 7.  Dynamic sampling schemes.

Key Concepts Discussed in Chapter “Overfitting, Underfitting and
General Model Overconfidence and Under-Performance Pitfalls and
Best Practices in Machine Learning and AI”
Training data error of a model

True generalization error of a model

Estimated generalization error of a model

Overfitting a model to data

Overfitting ML/AI method, system, stack, or protocol

Underfitting a model to data

Underfitting ML/AI method, system, stack, or protocol

Over confidence in a model

Under confidence in a model

516

C. Aliferis and G. Simon

Over performance of a model

Under performance of a model

Confidence  Interval  (CI)  of  a  point  estimate  P  of  the  performance
of a model

Predictive  Interval  (PI)  of  a  point  estimate  P  of  the  performance  of
a model.

Analysis creep

Sequential, iterative and multi-team analyses

Pitfalls Discussed in Chapter “Overfitting, Underfitting and General
Model Overconfidence and Under-Performance Pitfalls and Best
Practices in Machine Learning and AI”

Pitfall 10.1.: Producing models in which we have over-confidence

10.1.1.  Models  are  allowed  to  have  inappropriately  large  complexity  with
respect to data generating function complexity and to available sample size.

10.1.2. Ignoring statistical uncertainty of point estimates of performance.

10.1.3. Using biased estimators or introducing bias in unbiased ones.

10.1.4. Not correcting multiple statistical hypotheses tests.

10.1.5. Selectively reporting strongest models/results.

10.1.6. Conducting uncontrolled iterative modeling and succumbing to “anal-
ysis creep”.

10.1.7. Using non-representative datasets, unusual populations, and making
strong claims from too few or too easy datasets.

10.1.8. Not coordinating analysis over many teams and same data via appro-
priate unbiased protocols designed for collaborative work or competitions.

10.1.9. Using hard-to-reproduce, non-standardized data input steps.

10.1.10.  Employing  normalization  /data  transforms  that  require  entirety
of sample.

10.1.11.  Allowing  learners  to  learn  the  wrong  patterns  via  spurious  co-
occurrence, uncontrolled structural relations and biased sampling; and ignor-
ing domain knowledge that reveals the above.

10.1.12. Controlling only some of the factors that can lead to OC.

10.1.13. Inappropriate modelling for individual patients and over interpreta-
tion of their generalizability.

Overfitting, Underfitting and General Model Overconfidence and…

517

10.1.14.  Insufficient  studies  of  scalability  and  generalizability  in  bespoke
hand-created AI models.

Pitfall 10.2.: Producing models that are under performing.

10.2.1. Not deploying the right model family in model selection; not explor-
ing enough method families in model selection.

10.2.2. Insufficient data preparation.

10.2.3.  Insufficient  exploration  of  the  hyper  parameter  space  during  model
selection.

10.2.4. 1-stage modeling.

10.2.5. Ignoring best known methods for task and data at hand (either as base-
line comparators or starting point).

10.2.5.  Ignoring  official  specifications  and  prototypical  (reference)  use  of
employed methods.

10.2.6. Models for individual patients: lack of dense time series data, failure
to leverage population models that apply to the specific individual (including
ignoring or under-modeling severe and irrevocable rare or singular outcomes),
not addressing well abrupt distribution shifts of the individual (whereas same
shifts may be learnable at the population level).

10.2.7. Failing to obtain power sample analysis and more generally control
effects of sample size on modeling.

Best Practices Discussed in Chapter “Overfitting, Underfitting and
General Model Overconfidence and Under-Performance Pitfalls and
Best Practices in Machine Learning and AI”

Best  Practice  10.1:  Deploy  procedures  that  prevent,  diagnose  and  remedy
errors of over confidence in, or over fitting of, models.

10.1.1.  Manage  model  complexity  with  respect  to  data  generating  function
complexity and to available sample size using:

 1.  Regularization,
 2.  Dimensionality reduction,
 3.  Feature selection,
 4.  Bayesian Priors and Bayesian ensembles,
 5.  Algorithm-embedded capacity control,
 6.  Statistical model/data complexity measures,
 7.  Model selection,
 8.  Combination approaches.

518

C. Aliferis and G. Simon

10.1.2 Characterize and manage statistical uncertainty.

10.1.3.  Use  unbiased  estimators  of  model  performance  or  correct  bias  of
biased estimates.

10.1.4.  Lock  models  at  predefined  stages  in  the  modeling  process  and  not
allow further tampering with locked models.

10.1.5. Correct multiple statistical hypotheses tests (explicitly or implicitly).

10.1.6.  Thoroughly  specify  and  report  the  entirety  of  procedures  used  to
obtain models so that independent verification of generalizability is possible.

10.1.7. Conduct iterative or sequential modeling via unbiased protocols.

10.1.8. Use representative datasets, appropriate populations, and make gener-
alizability claims from appropriate datasets.

10.1.9. Coordinate analysis over many teams and same data via appropriate
unbiased protocols.

10.1.10. Use reproducible, standardized data input steps.

10.1.11. Employ normalization /data transforms that do not require entirety of
sample  (or  confine  such  within  discovery  and  validation  datasets
independently).

10.1.12. Prevent learners from learning the wrong patterns via spurious co-
occurrence; control structural relations and biased sampling; and incorporate
domain knowledge-based review that reveals spurious learning.

10.1.13. Control via nested model selection all (and not just a few) factors that
can lead to OC.

10.1.14. Carefully combine modelling of individual patients with population
modeling.

10.1.15. Do not over-interpret the generalizability of bespoke hand-created AI
models  unless  sufficient  validation  data  can  be  obtained  to  support  such
claims. Consider creating computable versions of model hand-crafting mod-
eling when possible.

10.1.16. Use label reshuffling testing for evaluating the overfitting bias of the
whole analysis protocol.

10.1.17. Apply with appropriate caution independent dataset validation and
be mindful of dangers of over interpretation of positive and negative results.

10.1.18. Instead of pursuing strict and exact reproducibility across datasets,
study the variant and invariant findings across these datasets.

Overfitting, Underfitting and General Model Overconfidence and…

519

10.1.19. Whenever possible, use reanalysis (with both original and unbiased
or  otherwise  improved  protocols,  including  single  coordinated  protocols  as
needed) when verifying the validity of models produced by third parties.

10.1.20. Use domain knowledge and related face-validity tests by experts to
flag potential model errors. The expert themselves may be prone to biases or
domain theory may not cover models’ new findings so do not over-interpret
experts’ objections.

10.1.21. Apply “safety net” measures for ensuring that a model is not applied
to the wrong person or population.

10.1.22.  Examine  stability  of  models,  parameters  and  other  findings  and
examine more deeply unstable findings. Be aware that it is possible for unsta-
ble models to be perfectly valid.

Best  Practice  10.2:  Deploy  procedures  that  prevent,  diagnose  and  remedy
errors of model under performance or underfitting.

10.5.1.  Deploy  and  explore  all  relevant  learning  method  families  in  model
selection.

10.5.2. Deploy and explore all relevant data preparation steps to the domain
and task at hand.

10.5.3. Systematically and sufficiently explore the hyper parameter space.

10.5.4. Anticipate  several  preliminary  and  refinement  modeling  stages  and
incorporate in sequential nested designs to avoid overfitting.

10.5.5. Inform analyses by methods literature so that best known methods for
task and data at hand are always explored along with novel methods.

10.5.6.  Follow  theoretically  and  empirically  proven  specifications  of  refer-
ence prototypical or official use of employed methods.

10.5.7. In models for individual patients: use dense time series data, leverage
population models, search for and model abrupt distribution shifts of the indi-
vidual (including learning and modeling shifts at the population level).

10.5.8. Conduct power sample analysis and more generally characterize the
effects of sample size on modeling. Use:

 1.  Dynamic sampling schemes whenever appropriate,
 2.  Sensitivity analysis for results over convenience samples by iteratively
reducing available sample size (sub-sampling on a convenience sample),

 3.  Simulations,
 4.  Domain knowledge,
 5.  Network-scientific knowledge,
 6.  Reference to prior robust results in very similar domains,
 7.  Dynamic sampling schemes.

520

C. Aliferis and G. Simon

Classroom Assignments and Discussion Topics

Chapter  “Overfitting,  Underfitting  and  General  Model  Overconfidence  and
Under-Performance Pitfalls and Best Practices in Machine Learning and AI”

  1.  Consider Rich Simon’s experiment with the following modification: the feature
selection  method  used  is  based  on  univariate  association  using  a  Hockberg-
Benjamini control of false positive rate at 10%. This means that the features
will  be  ranked  by  p-value  of  the  association  with  the  response  variable  and
thresholded so that no more than 10% of selected features will be false positive
correlates of the response variable.
 (a)  How  many  features  will  be  selected  if  no  feature  has  true  signal  for  the

response?

 (b)  What will be the error estimation bias of complete, incomplete and no cv

schemes?

 (c)  Based on the above, does the Simon et al. conclusions hold regardless of

the feature selection method?

 (d)  How would you modify the Simon et al. guidance?

  2.  Is it possible that error of a classifier is 0 in the TR data, true optimal general-
ization error is >0 and this classifier is optimal? In other words, is it possible for
a model to be optimal yet overfitted?

  3.  Consider  the  following  scenario  describing  two  model  selection  procedures
MS1, MS2 and MS3 each considering and selecting different sets of models
with estimated and true generalization errors as described in the table. For sim-
plicity assume no other models can be fitted in this setting.

TR accuracy
Estimated generalization accuracy by CV
Estimated generalization accuracy by
Estimator X
True generalization accuracy
MS1 selects model
MS2 selects model
MS3 considers model

Model 1 Model 2 Model 3 Model 4
0.95
0.80
0.9

0.75
0.78
0.92

0.65
0.70
0.80

0.90
0.88
0.96

0.80
Yes
Yes
No

0.88
No
Yes
Yes

0.76
Yes
Yes
No

0.73
Yes
Yes
Yes

 (a)  What are you conclusions about OP, OC, OF, UF and UP of models 1 to 4?
 (b)  What is your assessment of the bias of the generalization accuracy estima-

tor X used here?

 (c)  Why CV does not exactly match the true generalization error, although it is

unbiased?

 (d)  Which models are selected by each of MS1 to MS3? How would you char-

acterize these model selectors?

  4.  [ADVANCED] The label reshuffling procedure tests whether a model is statis-
tically significantly different than a model without signal and simultaneously
whether  the  overall  modeling  protocol  has  a  propensity  for  producing  over

Overfitting, Underfitting and General Model Overconfidence and…

521

confident estimates. This holds under the null hypothesis (i.e. there is no signal
in the data).
 (a)  The  reshuffling  takes  place  only  in  the  response  variable  labels.  Explain
why we DO NOT randomize all variables (i.e., both inputs and outputs).
 (b)  Bonus/research topic open problem: can you think of possible procedures
that could test against alternative hypotheses (i.e., against a user-postulated
non-zero signal)?

  5.  Consider years 2020–2021 of the COVID epidemic whereas many factors were

constantly changing.
 (a)  What are some key factors that were changing?
 (b)  What challenges of the OF/UP/OC varieties does a situation like this cre-
ates for various types of AI/ML decision models? Consider ICU admission
decision models as an example.

  6.  Describe, by example or more general analysis, how a researcher can produce
clustering omics data so that a published cluster model exhibits good diagnostic
accuracy even though no such signal exists in the data.

  7.  Comment on the following position: “whenever the true signal in the data is
very high, it is more difficult to produce models with serious overconfidence
errors; conversely as true signal approaches zero, the magnitude of possible OC
error increases”. What are underlying assumptions in the above thesis?

  8.  Consider  the  following  (idealized  and  simplified)  modeling  situation. A  data
scientist is tasked by her manager to create a predictive model. She decides to
use nested hold out (equivalent to NNCV with one fold) as follows: the total
data is randomly split in mutually exclusive datasets TRTR TRTE and TE. She
ensures that the prior of the binary response variable is the same in all three
datasets. She considers 2 possible values for hyper-parameter H of ML algo-
rithm A and estimates accuracy (0/1 error) as follows:

Algorithm A
Accuracy in
TRTE for H = 1

Accuracy in
TRTE for H = 2

0.80

0.90

Best
value of
H
2

Accuracy in TE of
model with best value
of H
0.85

Finally reported
accuracy of best
model
0.85

She presents the results to the project manager who suggests that a second
algorithm B is used because it may increase accuracy. The results this time look
like this:

Algorithm B
Accuracy in
TRTE for H = 1

Accuracy in
TRTE for H = 2

0.85

0.80

Best
value of
H
1

Accuracy in TE of
model with best value
of H
0.80

Finally reported
accuracy of best
model
0.80

522

C. Aliferis and G. Simon

She presents the results to the project manager who still is not satisfied with
the  result  and  brings  in  a  consultant  who  suggests  that  a  third  algorithm  C  is
used. The results this time look like this:

Algorithm C
Accuracy in
TRTE for H = 1

Accuracy in
TRTE for H = 2

0.75

0.80

Best
value of
H
2

Accuracy in TE of
model with best value
of H
0.90

Finally reported
accuracy of best
model
0.90

Based on the above the manager and the consultant conclude that the model
produced  by  algorithm  C  is  the  best,  it  has  generalization  accuracy  0.90,  and
should be deployed.

The data scientist (who recently read chapter “Overfitting, Underfitting and
General  Model  Overconfidence  and  Under-Performance  Pitfalls  and  Best
Practices in Machine Learning and AI”) is not convinced however because she
now suspects that an “analysis creep” situation has occurred leading to overcon-
fidence in a model. She decides to conduct a nested holdout based analysis, this
time analyzing all 3 algorithms simultaneously in the nested part (inner loop) of
the cv design. The following table presents her results:

Best
algorithm/best
value of H:
Algorithm A,
H = 2

Accuracy in TE
of model with
best algorithm/
best value of H
0.85

Finally
reported
estimated
generalization
accuracy of
best model
0.85

Algorithm
A

Algorithm
B

Algorithm
C

Accuracy
in TRTE
for H = 1
0.80
Accuracy
in TRTR
for H = 1
0.85
Accuracy
in TRTR
for H = 1
0.75

Accuracy
in TRTE
for H = 2
0.90
Accuracy
in TRTE
for H = 2
0.80
Accuracy
in TRTE
for H = 2
0.80

The manager and the consultant are perplexed.
Assume the role of the data scientist and write a short report explaining how
an OC error occurred in the first round of analyses and why the second analysis
is unbiased. For simplicity, ignore the need to conduct tests of statistical differ-
ence between point estimates and interpret nominal accuracy point estimates as
true ones.

  9.  We saw that combining capacity control mechanisms provides augmented pro-

tections against overfitting/excessive capacity.
 (a)  Describe  an  existing  protocol  of  your  choice  (or  one  that  you  construct)

that combines 4 or more ways to control excessive model capacity.

(b)  Bonus  question  [ADVANCED]:  is  it  possible  for  such  combinations  to
have  negative  effects  on  ability  to  control  the  capacity  of  the  models
produced?

Overfitting, Underfitting and General Model Overconfidence and…

523

 10.  [ADVANCED]

 (a)  Show that in order to calculate the true positives rate (aka PPV) or a mod-
el’s decisions we need to know the prior of the distribution of the response
variable.

(b)  Show that an active learning sampling design alone does not allow the ana-

lyst to know this prior whereas a random sampling design does.

(c)  What are the implications therefore of an active learning design for control-

ling the PPV?

 11.  Is it ok to build under performing models in the context of exploratory research?
Present a few situations where it might be a good idea and some where it is a
bad idea.

 12.  “Ocam’s  Razor”  is  an  epistemological  principle  that  says  that  between  two
models that explain the data equally well, the simpler one is more likely to be
true. How does this principle relate to the BVDE? Can you think of a counter
example (HINT: consider causal modeling).

 13.  (a) Describe how PCA can lead to OC errors.

(b)  Consider R. Simon’s experiment where instead of feature selection the ana-
lyst  use  a  PC-mapping  of  the  input  data  such  that  correlation  with  the
response variable is maximized. Is this subject to the same bias that Simon
et al. described?

(c)  How would you conduct unbiased error estimation using NNFCV whereas

the data is PCA-transformed?

 14.  [ADVANCED] Wolpert uses NFLT and OTSE to argue in [19] that cross vali-
dation is not better as a model selection strategy than doing the exact opposite
(i.e., choose the model with highest error in the test data), which he coins “anti-
cross validation”. If he is right, then what that Chapter “Overfitting, Underfitting
and General Model Overconfidence and Under-Performance Pitfalls and Best
Practices  in  Machine  Learning  and AI”  teaches  as  best  practices  for  model
selection  and  error  estimation  is  only  a  heuristic  strategy  that  may  fail  in  as
many situations as the ones that it will succeed. Refute these claims.

HINT: focus your arguments either around the misalignment of OTSE with
real- life modeling objectives/performance metrics, or alternatively/additionally
with the misalignment of these objectives with averaging over all distributions
rather the distribution in hand.

References

1.  Mitchell TM. Machine learning, vol. 1. New York: McGraw-hill; 2007.
2.  Simon R, Radmacher MD, Dobbin K, McShane LM. Pitfalls in the use of DNA microarray

data for diagnostic and prognostic classification. J Natl Cancer Inst. 2003;95(1):14–8.

3.  Statnikov A.  A  gentle  introduction  to  support  vector  machines  in  biomedicine:  theory  and

methods, vol. 1. World Scientific; 2011.

524

C. Aliferis and G. Simon

4.  Aliferis CF, Statnikov A, Tsamardinos I. Challenges in the analysis of mass-throughput data:
a technical commentary from the statistical machine learning perspective. Cancer Informat.
2006;2:133–62.

5.  Aliferis CF, Statnikov A, Tsamardinos I, Schildcrout JS, Shepherd BE, Harrell FE Jr. Factors
influencing the statistical power of complex data analysis protocols for molecular signature
development from microarray data. PLoS One. 2009;4(3):e4922.

6.  Kerr  NL.  HARKing:  hypothesizing  after  the  results  are  known.  Personal  Soc  Psychol  Rev.

1998;2(3):196–217.

7.  Harrell FE. Regression modeling strategies: with applications to linear models, logistic regres-

sion, and survival analysis, vol. 608. New York: springer; 2001.

8.  Steyerberg EW. Applications of prediction models. New York: Springer; 2009. p. 11–31.
9.  Duda RO, Hart PE. Pattern classification. John Wiley & Sons; 2006.
10. Efron B, Tibshirani RJ. An introduction to the bootstrap. CRC press; 1994.
11. Andrade C. HARKing, cherry-picking, p-hacking, fishing expeditions, and data dredging and

mining as questionable research practices. J Clin Psychiatry. 2021;82(1):25941.

12. Neal  ML,  Kerckhoffs  R.  Current  progress  in  patient-specific  modeling.  Brief  Bioinform.

2010;11(1):111–26.

13. Visweswaran S, Cooper GF. Patient-specific models for predicting the outcomes of patients
with  community  acquired  pneumonia.  In:  AMIA  Annual  Symposium,  vol.  2005.  American
Medical Informatics Association; 2005. p. 759.

14. Braga-Neto UM, Dougherty ER. Is cross-validation valid for small-sample microarray clas-

sification? Bioinformatics. 2004;20(3):374–80.

15. McDermott  MB,  Wang  S,  Marinsek  N,  Ranganath  R,  Foschini  L,  Ghassemi
M. Reproducibility in machine learning for health research: still a ways to go. Sci Transl Med.
2021;13(586):eabb1655.

16. Shi  L,  Campbell  G,  Jones WD.  The  MicroArray  quality  control  (MAQC)-II  study  of  com-
mon practices for the development and validation of microarray-based predictive models. Nat
Biotechnol. 2010;28(8):827–38.

17. Benjamini  Y,  Hochberg  Y.  Controlling  the  false  discovery  rate:  a  practical  and  powerful
approach to multiple testing. J R Stat Soc Series B Stat Methodol. 1995;57(1):289–300.
18. Aliferis CF, Statnikov A, Tsamardinos I, Mani S, Koutsoukos XD. Local causal and Markov
blanket induction for causal discovery and feature selection for classification part II: analysis
and extensions. J Mach Learn Res. 2010;11(1):235–84.

19. Wolpert DH. What is important about the no free lunch theorems? In: Black box optimization,
machine learning, and no-free lunch theorems. Cham: Springer International Publishing; 2021.
p. 373–88.

Open Access    This chapter is licensed under the terms of the Creative Commons Attribution 4.0
International  License  (http://creativecommons.org/licenses/by/4.0/),  which  permits  use,  sharing,
adaptation, distribution and reproduction in any medium or format, as long as you give appropriate
credit to the original author(s) and the source, provide a link to the Creative Commons license and
indicate if changes were made.

The images or other third party material in this chapter are included in the chapter's Creative
Commons  license,  unless  indicated  otherwise  in  a  credit  line  to  the  material.  If  material  is  not
included  in  the  chapter's  Creative  Commons  license  and  your  intended  use  is  not  permitted  by
statutory regulation or exceeds the permitted use, you will need to obtain permission directly from
the copyright holder.



---

## Content from Previous Extraction (not in markitdown output)

### Visual/Chart/Graph Descriptions

#### 1

Overfitting, Underfitting and General Model Overconfidence and… 481
Fig. 1 Illustration of UNDERFITTING OVERRFITTING
overfitting and underfitting
POPULATION
MODEL ERROR
(GENERALIZATION)
ERROR
TRAINING
DATA ERROR
TRAINING
ITERATIONS
The models to the left of the optimal point are underfitted and to the right are
overfitted.
The intuition gained is that there is a level of model “fit” that is ideal for this
data and anything above or below that point will lead to worse generalization
performance. Performance in the training data remains a poor indicator of
performance in the population, however.
Introductory Example 3: Rich Simon’s OF Demonstration for
Genomics-Driven Discovery
In bioinformatics where dimensionalities are typically very high and sample sizes
small, OF is a particularly important danger. Simon at al published the following
empirical demonstration showcasing that depending on how gene selection and
model error estimation are conducted, there are different degrees of biasing the
estimates of model generalization error [2].
Train
& Test “Biased resubstitution”
GS
test
Train “Fully crossvalidated”
&
GS
test
GS train “Partially crossvalidated”

#### 2

Overfitting, Underfitting and General Model Overconfidence and… 483
Fig. 2 Further illustration
of overfitting (top, 2 a) and
underfitting (bottom, 2 b)
Definitions of OF and UF; the Broader Pitfalls of Overconfidence
and of Under-Performing Models. Bias-Variance Error
Decomposition View
Training data error of a model M is the error of M on the training data used
to derive M.
True generalization error of a model M is the error of M on the population
or distribution, from which training data used to derive M, were sampled from.
Estimated generalization error of a model M is the estimated error (via an
error estimator procedure applied on data samples) of M on the population or
distribution from which training data used to derive M were sampled from.
In controlled conditions, for example in simulation experiments, the true gener-
alization error of any model can be known. Typically the true generalization error is
unknown when dealing in non-trivial real-world problems, not constructed in the
lab, however.

### Additional Content

#### 1

479
U
ni
detsi L
odu J
stneser P
srae W
SUTATS
EURT
esruoc
sa
eugolatac
kcalb
tuo
sdna H
srae W
sa H
sa H
eht
ot
lairetam
dna
tius
) elbairav
esnopser (
rottcurtsni
tleb
stnemngissa
redne G
sessalg
tnecca
draeb
ssalc
seerge D
eit
ema N
rehcae T
se Y
se Y
se Y
M
o N
se Y
o N
se Y
Dh P
o N
nomi S
rehcae T
se Y
o N
se Y
M
se Y
se Y
se Y
se Y
, DM
-emo S
sirefil A
Dh P
semit
tnedut S
o N
o N
o N
F
o N
o N
o N
se Y
c SB
o N
htim S
tnedut S
o N
o N
o N
F
se Y
se Y
o N
se Y
NR
o N
gneh Z
tnedut S
o N
o N
o N
M
se Y
o N
se Y
se Y
Dmrah P
se Y
hgni S
tnedut S
o N
o N
o N
M
se Y
o N
o N
se Y
c SB
o N
ruel Fa L
tnedut S
o N
o N
o N
F
o N
o N
o N
se Y
DM
o N
namkci B
tnedut S
o N
o N
o N
M
o N
se Y
se Y
se Y
c SB
o N
-odapa P soluop
rerutcel
tseu G
se Y
o N
o N
F
se Y
o N
o N
se Y
, DM
o N
gnah C
SM
AT
o N
o N
se Y
BN
o N
o N
o N
se Y
Dh P
o N
seno J
tneduts
gnitidu A
o N
o N
o N
F
se Y
o N
o N
o N
c SB
o N
ztrawhc S
Overfitting, Underfitting and General Model Overconfidence and…
