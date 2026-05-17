# Detecting Overfitting via Adversarial Examples

> *Source PDF: Detecting Overfitting via Adversarial Examples.pdf*
> *Extraction: Combined — previous structured extraction (base) + markitdown raw text*

---

# Detecting Overfitting via Adversarial Examples

> *Source PDF: Detecting Overfitting via Adversarial Examples.pdf*
> *Extraction: Combined — markitdown raw text (base) + previous extraction supplements*

---

Detecting Overﬁtting via Adversarial Examples

Roman Werpachowski

András György

Csaba Szepesvári

DeepMind, London, UK
{romanw,agyorgy,szepi}@google.com

Abstract

The frequent reuse of test sets in popular benchmark problems raises doubts about
the credibility of reported test-error rates. Verifying whether a learned model is
overﬁtted to a test set is challenging as independent test sets drawn from the same
data distribution are usually unavailable, while other test sets may introduce a
distribution shift. We propose a new hypothesis test that uses only the original test
data to detect overﬁtting. It utilizes a new unbiased error estimate that is based
on adversarial examples generated from the test data and importance weighting.
Overﬁtting is detected if this error estimate is sufﬁciently different from the original
test error rate. We develop a specialized variant of our test for multiclass image
classiﬁcation, and apply it to testing overﬁtting of recent models to the popular
ImageNet benchmark. Our method correctly indicates overﬁtting of the trained
model to the training set, but is not able to detect any overﬁtting to the test set, in
line with other recent work on this topic.

1

Introduction

Deep neural networks achieve impressive performance on many important machine learning bench-
marks, such as image classiﬁcation [18, 19, 28, 27, 16], automated translation [2, 31] or speech
recognition [9, 15]. However, the benchmark datasets are used a multitude of times by researchers
worldwide. Since state-of-the-art methods are selected and published based on their performance
on the corresponding test set, it is typical to see results that continuously improve over time; see,
e.g., the discussion of Recht et al. [25] and Figure 1 for the performance improvement of classiﬁers
published for the popular CIFAR-10 image classiﬁcation benchmark [18].

This process may naturally lead to models over-
ﬁtted to the test set, rendering test error rate
(the average error measured on the test set) an
unreliable indicator of the actual performance.
Detecting whether a model is overﬁtted to the
test set is challenging, since independent test
sets drawn from the same data distribution are
generally not available, while alternative test
sets often introduce a distribution shift.

To estimate the performance of a model on un-
seen data, one may use generalization bounds
to get upper bounds on the expected error rate.
The generalization bounds are also applicable
when the model and the data are dependent (e.g.,
for cross validation or for error estimates based
on the training data or the reused test data), but they usually lead to loose error bounds. Therefore,
although much tighter bounds are available if the test data and the model are independent, comparing

Figure 1: Accuracy of image classiﬁers on the CIFAR-10
test set, by year of publication (data from [25]).

33rd Conference on Neural Information Processing Systems (NeurIPS 2019), Vancouver, Canada.

20102012201420162018year0.800.850.900.951.00accuracyconﬁdence intervals constructed around the training and test error rates leads to an underpowered test
for detecting the dependence of a model on the test set. Recently, several methods have been proposed
that allow the reuse of the test set while keeping the validity of test error rates [10]. However, these
are intrusive: they require the user to follow a strict protocol of interacting with the test set and are
thus not applicable in the more common situation when enforcing such a protocol is impossible.

In this paper we take a new approach to the challenge of detecting overﬁtting of a model to the test
set, and devise a non-intrusive statistical test that does not restrict the training procedure and is based
on the original test data. To this end, we introduce a new error estimator that is less sensitive to
overﬁtting to the data; our test rejects the independence of the model and the test data if the new
error estimate and the original test error rate are too different. The core novel idea is that the new
estimator is based on adversarial examples [14], that is, on data points1 that are not sampled from the
data distribution, but instead are cleverly crafted based on existing data points so that the model errs
on them. Several authors showed that the best models learned for the above-mentioned benchmark
problems are highly sensitive to adversarial attacks [14, 23, 30, 6, 7, 24]: for instance, one can often
create adversarial versions of images properly classiﬁed by a state-of-the-art model such that the
model will misclassify them, yet the adversarial perturbations are (almost) undetectable for a human
observer; see, e.g., Figure 2, where the adversarial image is obtained from the original one by a
carefully selected translation.

scale, weighing machine

The adversarial (error) estimator proposed in
this work uses adversarial examples (generated
from the test set) together with importance
weighting to take into account the change in
the data distribution (covariate shift) due to the
adversarial transformation. The estimator is un-
biased and has a smaller variance than the stan-
dard test error rate if the test set and the model
are independent.2 More importantly, since it is
based on adversarially generated data points, the
adversarial estimator is expected to differ sig-
niﬁcantly from the test error rate if the model
is overﬁtted to the test set, providing a way to
detect test set overﬁtting. Thus, the test error
rate and the adversarial error estimate (calcu-
lated based on the same test set) must be close if the test set and the model are independent, and are
expected to be different in the opposite case. In particular, if the gap between the two error estimates
is large, the independence hypothesis (i.e., that the model and the test set are independent) is dubious
and will be rejected. Combining results from multiple training runs, we develop another method to
test overﬁtting of a model architecture and training procedure (for simplicity, throughout the paper
we refer to both together as the model architecture). The most challenging aspect of our method is to
construct adversarial perturbations for which we can calculate importance weights, while keeping
enough degrees of freedom in the way the adversarial perturbations are generated to maximize power,
the ability of the test to detect dependence when it is present.

Figure 2: Adversarial example for the ImageNet dataset
generated by a (5, −5) translation: the original example
(left) is correctly classiﬁed by the VGG16 model [27] as
“scale, weighing machine,” the adversarially generated
example (right) is classiﬁed as “toaster,” while the image
class is the same for any human observer.

toaster

To understand the behavior of our tests better, we ﬁrst use them on a synthetic binary classiﬁcation
problem, where the tests are able to successfully identify the cases where overﬁtting is present. Then
we apply our independence tests to state-of-the-art classiﬁcation methods for the popular image
classiﬁcation benchmark, ImageNet [8]. As a sanity check, in all cases examined, our test rejects
(at conﬁdence levels close to 1) the independence of the individual models from their respective
training sets. Applying our method to VGG16 [27] and Resnet50 [16] models/architectures, their
independence to the ImageNet test set cannot be rejected at any reasonable conﬁdence. This is in
agreement with recent ﬁndings of [26], and provides additional evidence that despite of the existing
danger, it is likely that no overﬁtting has happened during the development of ImageNet classiﬁers.

The rest of the paper is organized as follows: In Section 2, we introduce a formal model for error
estimation using adversarial examples, including the deﬁnition of adversarial example generators.

1Throughout the paper, we use the words “example” and “point” interchangeably.
2Note that the adversarial error estimator’s goal is to estimate the error rate, not the adversarial error rate (i.e.,

the error rate on the adversarial examples).

2

The new overﬁtting-detection tests are derived in Section 3, and applied to a synthetic problem in
Section 4, and to the ImageNet image classiﬁcation benchmark in Section 5. Due to space limitations,
some auxiliary results, including the in-depth analysis of our method on the synthetic problem, are
relegated to the appendix.

2 Adversarial Risk Estimation

We consider a classiﬁcation problem with deterministic (noise-free) labels, which is a reasonable
assumption for many practical problems, such as image recognition (we leave the extension of our
method to noisy labels for future work). Let X ⊂ RD denote the input space and Y = {0, . . . , K −1}
the set of labels. Data is sampled from the distribution P over X , and the class label is determined
by the ground truth function f ∗ : X → Y. We denote a random vector drawn from P by X, and its
corresponding class label by Y = f ∗(X). We consider deterministic classiﬁers f : X → Y. The
performance of f is measured by the zero-one loss: L(f, x) = I(f (x) (cid:54)= f ∗(x)),3 and the expected
error (also known as the risk or expected risk in the learning theory literature) of the classiﬁer f is
deﬁned as R(f ) = E[I(f (X) (cid:54)= Y )] = (cid:82)
X L(f, x)dP(x).
Consider a test dataset S = {(X1, Y1) . . . , (Xm, Ym)} where the Xi are drawn from P independently
of each other and Yi = f ∗(Xi). In the learning setting, the classiﬁer f usually also depends on some
randomly drawn training data, hence is random itself. If f is (statistically) independent from S, then
L(f, X1), . . . , L(f, Xm) are i.i.d., thus the empirical error rate

(cid:98)RS(f ) =

1
m

m
(cid:88)

i=1

L(f, Xi) =

1
m

m
(cid:88)

i=1

I(f (Xi) (cid:54)= Yi)

is an unbiased estimate of R(f ) for all f ; that is, R(f ) = E[ (cid:98)RS(f )|f ]. If f and S are not indepen-
dent, the performance guarantees on the empirical estimates available in the independent case are
signiﬁcantly weakened; for example, in case of overﬁtting to S, the empirical error rate is likely to be
much smaller than the expected error.

Another well-known way to estimate R(f ) is to use importance sampling (IS) [17]: instead of
sampling from the distribution P, we sample from another distribution P (cid:48) and correct the estimate
by appropriate reweighting. Assuming P is absolutely continuous with respect to P (cid:48) on the set
X L(f, x)dP(x) = (cid:82)
E = {x ∈ X : L(f, x) (cid:54)= 0}, R(f ) = (cid:82)
E L(f, x)h(x)dP (cid:48)(x), where h = dP
dP (cid:48)
is the density (Radon-Nikodym derivative) of P with respect to P (cid:48) on E (h can be deﬁned to have
arbitrary ﬁnite values on X \ E). It is well known that the the corresponding empirical error estimator

(cid:98)R(cid:48)

S(cid:48)(f ) =

1
m

m
(cid:88)

L(f, X (cid:48)

i)h(X (cid:48)

i) =

m
(cid:88)

I(f (X (cid:48)

i) (cid:54)= Y (cid:48)

i )h(X (cid:48)
i)

(1)

1
m

i=1

1, Y (cid:48)

m, Y (cid:48)

1 ), . . . , (X (cid:48)

i=1
obtained from a sample S(cid:48) = {(X (cid:48)
(i.e., E[ (cid:98)RS(cid:48)(f )|f ] = R(f )) if f and S(cid:48) are independent.
S(cid:48) is minimized if P (cid:48) is the so-called zero-variance IS distribution, which is
The variance of (cid:98)R(cid:48)
supported on E with h(x) = R(f )
L(f,x) for all x ∈ E (see, e.g., [4, Section 4.2]). This suggest that an
effective sampling distribution P (cid:48) should concentrate on points where f makes mistakes, which also
facilitates that (cid:98)R(cid:48)
S(cid:48)(f ) become large if f is overﬁtted to S and hence (cid:98)RS(f ) is small. We achieve this
through the application of adversarial examples.

m)} drawn independently from P (cid:48) is unbiased

2.1 Generating adversarial examples

In this section we introduce a formal framework for generating adversarial examples. Given a
classiﬁcation problem with data distribution P and ground truth f ∗, an adversarial example generator
(AEG) for a classiﬁer f is a (measurable) mapping g : X → X such that

(G1) g preserves the class labels of the samples, that is, f ∗(x) = f ∗(g(x)) for P-almost all x;
(G2) g does not change points that are incorrectly classiﬁed by f , that is, g(x) = x if f (x) (cid:54)=

f ∗(x) for P-almost all x.

3For an event B, I(B) denotes its indicator function: I(B) = 1 if B happens and I(B) = 0 otherwise.

3

Figure 3: Generating adversarial examples. The top row depicts the original dataset S, with blue and orange
points representing the two classes. The classiﬁer’s prediction is represented by the color of the striped areas
(checkmarks and crosses denote if a point is correctly or incorrectly classiﬁed). The arrows show the adversarial
transformations via the AEG g, resulting in the new dataset S(cid:48); misclassiﬁed points are unchanged, while some
correctly classiﬁed points are moved, but their original class label is unchanged. If the original data distribution is
uniform over S, the transformation g is density preserving, but not measure preserving: after the transformation
the two rightmost correctly classiﬁed points in each class have probability 0, while the leftmost misclassiﬁed
point in each class has probability 3/16; hence, the density hg for the latter points is 1/3.

Figure 3 illustrates how an AEG works. In the literature, an adversarial example g(x) is usually
generated by staying in a small vicinity of the original data point x (with respect to, e.g., the 2- or the
max-norm) and assuming that the resulting label of g(x) is the same as that of x (see, e.g., [14, 6]).
This foundational assumption—which is in fact a margin condition on the distribution—is captured
in condition (G1). (G2) formalizes the fact that there is no need to change samples which are already
misclassiﬁed. Indeed, existing AEGs comply with this condition.

The performance of an AEG is usually measured by how successfully it generates misclassiﬁed
examples. Accordingly, we call a point g(x) a successful adversarial example if x is correctly
classiﬁed by f and f (g(x)) (cid:54)= f (x) (i.e., L(f, x) = 0 and L(f, g(x)) = 1).

In the development of our AEGs for image recognition tasks, we will make use of another condition.
For simplicity, we formulate this condition for distributions P that have a density ρ with respect
to the uniform measure on X , which is assumed to exist (notable cases are when X is ﬁnite, or
X = [0, 1]D or when X = RD; in the latter two cases the uniform measure is the Lebesgue measure).
The assumption states that the AEG needs to be density-preserving:

(G3) ρ(x) = ρ(g(x)) for P-almost all x.

Note that a density-preserving map may not be measure-preserving (the latter means that for all
measurable A ⊂ X , P(A) = P(g(A))).

We expect (G3) to hold when g perturbs its input by a small amount and if ρ is sufﬁciently smooth.
The assumption is reasonable for, e.g., image recognition problems (at least in a relaxed form,
ρ(x) ≈ ρ(g(x))) where we expect that very close images will have a similar likelihood as measured
by ρ. An AEG employing image translations, which satisﬁes (G3), will be introduced in Section 5.
Both (G1) and (G3) can be relaxed (to a soft margin condition or allowing a slight change in ρ, resp.)
at the price of an extra error term in the analysis that follows.

For a ﬁxed AEG g : X → X , let Pg be the distribution of g(X) where X ∼ P (Pg is known as the
pushforward measure of P under g). Further, let hg = dP
on E = {x : L(f, x) (cid:54)= 0} and arbitrary
dPg
otherwise. It is easy to see that, on E, hg(x) is well-deﬁned and hg ≤ 1. For any measurable A ⊂ E
Pg(A) = P(g(X) ∈ A) ≥ P(g(X) ∈ A, X ∈ E) = P(X ∈ A) = P(A)
where the second to last equality holds because g(X) = X for any X ∈ E under condition (G2).
Thus, P(A) ≤ Pg(A) for any measurable A ⊂ E, which implies that hg is well-deﬁned on E and
hg(x) ≤ 1 for all x ∈ E.

One may think that (G3) implies that hg(x) = 1 for all x ∈ E. However, this does not hold. For
example, if P is a uniform distribution, any g : X → supp P satisﬁes (G3), where supp P ⊂ X
denotes the support of the distribution P. This is also illustrated in Figure 3.

2.2 Risk estimation via adversarial examples

Combining the ideas of this section so far, we now introduce unbiased risk estimates based on
adversarial examples. Our goal is to estimate the error-rate of f through an adversarially generated

4

(cid:51)(cid:51)(cid:51)(cid:51)(cid:55)(cid:55)(cid:55)(cid:55)(cid:51)(cid:51)(cid:51)(cid:51)(cid:51)(cid:51)(cid:51)(cid:51)(cid:55)(cid:55)(cid:55)(cid:55)(cid:55)(cid:55)(cid:55)(cid:55)(cid:55)(cid:55)(cid:55)(cid:55)(cid:55)(cid:55)(cid:55)(cid:55)SS(cid:48)g1, Y1), . . . , (X (cid:48)

sample S(cid:48) = {(X (cid:48)
i = g(Xi) with
X1, . . . , Xm drawn independently from P and Yi = f ∗(Xi). Since g satisﬁes (G1) by deﬁnition, the
original example Xi and the corresponding adversarial example X (cid:48)
i have the same label Yi. Recalling
that hg = dP/dPg ≤ 1 on E = {x ∈ X : L(f, x) = 1}, one can easily show that the importance
weighted adversarial estimate

m, Ym)} obtained through an AEG g, where X (cid:48)

(cid:98)Rg(f ) =

1
m

m
(cid:88)

i=1

I(f (X (cid:48)

i) (cid:54)= Yi)hg(X (cid:48)
i)

(2)

obtained from (1) for the adversarial sample S(cid:48) has smaller variance than that of the empirical average
(cid:98)RS(f ), while both are unbiased estimates of R(f ). Recall that both (cid:98)Rg(f ) and (cid:98)RS(f ) are unbiased
estimates of R(f ) with expectation E[ (cid:98)Rg(f )] = E[ (cid:98)RS(f )] = R(f ), and so

V[ (cid:98)Rg(f )] =

≤

1
m
1
m

(cid:0)E[L(f, g(X))2hg(g(X))2] − R(f )2(cid:1)

(cid:0)E[L(f, g(X))hg(g(X))] − R2(f )(cid:1) =

1
m

(cid:0)R(f ) − R2(f )(cid:1) = V[ (cid:98)RS(f )] .

Intuitively, the more successful the AEG is (i.e., the more classiﬁcation error it induces), the smaller
the variance of the estimate (cid:98)Rg(f ) becomes.

3 Detecting overﬁtting

In this section we show how the risk estimates introduced in the previous section can be used to test
the independence hypothesis that

(H) the sample S and the model f are independent.

If (H) holds, E[ (cid:98)Rg(f )] = E[ (cid:98)RS(f )] = R(f ), and so the difference TS,g(f ) = (cid:98)Rg(f ) − (cid:98)RS(f )
is expected to be small. On the other hand, if f is overﬁtted to the dataset S (in which case
(cid:98)RS(f ) < R(f )), we expect (cid:98)RS(f ) and (cid:98)Rg(f ) to behave differently (the latter being less sensitive to
overﬁtting) since (i) (cid:98)Rg(f ) depends also on examples previously unseen by the training procedure;
(ii) the adversarial transformation g aims to increase the loss, countering the effect of overﬁtting;
(iii) especially in high dimensional settings, in case of overﬁtting one may expect that there are
misclassiﬁed points very close to the decision boundary of f which can be found by a carefully
designed AEG. Therefore, intuitively, (H) can be rejected if |TS,g(f )| exceeds some appropriate
threshold.

3.1 Test based on conﬁdence intervals

The simplest way to determine the threshold is based on constructing conﬁdence intervals for
these estimator based on concentration inequalities. Under (H), standard concentration inequal-
ities, such as the Chernoff or empirical Bernstein bounds [3], can be used to quantify how
fast (cid:98)RS and (cid:98)Rg(f ) concentrate around the expected error R(f ).
In particular, we use the
S = (1/m) (cid:80)m
i=1(L(f, Xi) − (cid:98)RS(f ))2 and
following empirical Bernstein bound [22]: Let ¯σ2
g = (1/m) (cid:80)m
i=1(L(f, g(Xi))hg(g(Xi)) − (cid:98)Rg(f ))2 denote the empirical variance of L(f, Xi)
¯σ2
and L(f, g(Xi))hg(g(Xi)), respectively. Then, for any 0 < δ ≤ 1, with probability at least 1 − δ,

| (cid:98)RS(f ) − R(f )| ≤ B(m, ¯σ2

S, δ, 1),

(3)

where B(m, σ2, δ, 1) =
and we used the fact that the range of L(f, x) is 1
(the last parameter of B is the range of the random variables considered). Similarly, with probability
at least 1 − δ,

+ 3 ln(3/δ)
m

(cid:113) 2σ2 ln(3/δ)
m

| (cid:98)Rg(f ) − R(f )| ≤ B(m, ¯σ2

g, δ, 1).

(4)

It follows trivially from the union bound that if the independence hypothesis (H) holds, the
above two conﬁdence intervals [ (cid:98)RS(f ) − B(m, ¯σ2
S, δ, 1)] and [ (cid:98)Rg(f ) −

S, δ, 1), (cid:98)RS(f ) + B(m, ¯σ2

5

g, δ, 1), (cid:98)RS(f ) + B(m, ¯σ2
B(m, ¯σ2
intersect with probability at least 1 − 2δ.

g, δ, 1)], which both contain R(f ) with probability at least 1 − δ,

On the other hand, if f and S are not independent, the performance guarantees (3) and (4) may be
violated and the conﬁdence intervals may become disjoint. If this is detected, we can reject the
independence hypothesis (H) at a conﬁdence level 1 − 2δ or, equivalently, with p-value 2δ. In other
words, we reject (H) if the absolute value of the difference of the estimates TS,g(f ) = (cid:98)Rg(f )− (cid:98)RS(f )
g, δ, 1) (note that E[TS,g(f ) = 0] if S and f are
exceeds the threshold B(m, ¯σ2
independent).

S, δ, 1) + B(m, ¯σ2

3.2 Pairwise test

A smaller threshold for |TS,g(f )|, and hence a more effective independence test, can be devised
if instead of independently estimating the behavior of (cid:98)RS and (cid:98)Rg(f ), one utilizes their apparent
correlation. Indeed, TS,g(f ) = (1/m) (cid:80)m
i=1 Ti,g(f ) where
Ti,g(f ) = L(f, g(Xi))hg(g(Xi)) − L(f, Xi)
and the two terms in Ti,g(f ) have the same mean and are typically highly correlated by the con-
struction of g. Thus, we can apply the empirical Bernstein bound [22] to the pairwise differences
Ti,g(f ) to set a tighter threshold in the test: if the independence hypothesis (H) holds (i.e., S and f
are independent), then for any 0 < δ < 1, with probability at least 1 − δ,

(5)

|TS,g(f )| ≤ B(m, ¯σ2

T , δ, U )

(6)

(cid:113) 2σ2 ln(3/δ)
m

with B(m, σ2, δ, U ) =
i=1(Ti(f ) − TS,g(f ))2 is
the empirical variance of the Ti,g(f ) terms and U = sup Ti,g(f ) − inf Ti,g(f ); we also used the fact
that the expectation of each Ti,g(f ), and hence that of TS,g(f ), is zero. Since hg ≤ 1 if L(f, x) = 1
(as discussed in Section 2.2), it follows that U ≤ 2, but further assumptions (such as g being density
preserving) can result in tighter bounds.

, where ¯σ2

+ 3U ln(3/δ)
m

T = (1/m) (cid:80)m

This leads to our pairwise dependence detection method:

if |TS,g(f )| > B(m, ¯σ2

T , δ, 2), reject (H) at a conﬁdence level 1 − δ (p-value δ).

For a given statistic (|TS,g(f )|, ¯σ2
T ), the largest conﬁdence level (smallest p-value) at which (H) can
be rejected can be calculated by setting the value of the statistic |TS,g(f )| − B(m, ¯σ2
T , δ, 2) to zero
and solving for δ. This leads to the following formula for the p-value (if the solution is larger than 1,
which happens when the bound (6) is loose, δ is capped at 1):
√

(cid:17)(cid:27)

T +3U |TS,g(f )|−¯σT

¯σ2

T +6U |TS,g(f )|

δ = min

(cid:26)

1, 3e− m

9U 2

(cid:16)

¯σ2

.

(7)

Note that in order for the test to work well, we not only need the test statistic TS,g(f ) to have a
small variance in case of independence (this could be achieved if g were the identity), but we also
need the estimators (cid:98)RS(f ) and (cid:98)Rg(f ) behave sufﬁciently differently if the independence assumption
is violated. The latter behavior is encouraged by stronger AEGs, as we will show empirically in
Section 5.2 (see Figure 5 in particular).

3.3 Dependence detector for randomized training

The dependence between the model and the test set can arise from (i) selecting the “best” random
seed in order to improve the test set performance and/or (ii) tweaking the model architecture (e.g.,
neural network structure) and hyperparameters (e.g., learning-rate schedule). If one has access to a
single instance of a trained model, these two sources cannot be disentangled. However, if the model
architecture and training procedure is fully speciﬁed and computational resources are adequate, it
is possible to isolate (i) and (ii) by retraining the model multiple times and calculating the p-value
for every training run separately. Assuming N models, let fj, j = 1, . . . , N denote the j-th trained
model and pj the p-value calculated using the pairwise independence test (6) (i.e., from Eq. 7 in
Section 3). We can investigate the degree to which (i) occurs by comparing the pj values with the
corresponding test set error rates RS(fj). To investigate whether (ii) occurs, we can average over the
randomness of the training runs.

6

For every example Xi ∈ S, consider the average test statistic ¯Ti = 1
j=1 Ti,gj (fj), where
N
Ti,gj (fj) is the statistic (5) calculated for example Xi and model fj with AEG gj selected for model
fj (note that AEGs are model-dependent by construction). If, for each i and j, the random variables
Ti(fj) are independent, then so are the ¯Ti (for all i). Hence, we can apply the pairwise dependence
detector (6) with ¯Ti instead of Ti, using the average ¯TS = (1/m) (cid:80)m
¯Ti with empirical variance
T,N = (1/m) (cid:80)m
i=1( ¯Ti − ¯TS)2, giving a single p-value pN . If the training runs vary enough in their
¯σ2
outcomes, different models fj err on different data points Xj, leading to ¯σ2
T , and therefore
strengthening the power of the dependence detector. For brevity, we call this independence test an
N -model test.

T,N < ¯σ2

i=1

(cid:80)N

4 Synthetic experiments

First we verify the effectiveness of our method
on a simple linear classiﬁcation problem. Due
to space limitations, we only convey high-level
results here, details are given in Appendix A.
We assume that the data is linearly separa-
ble with a margin and the density ρ is known.
We consider a linear classiﬁers of the form
f (x) = sgn(w(cid:62)x + b) trained with the cross-
entropy loss c, and we employ a one-step gra-
dient method (which is an L2 version of the
fast gradient-sign method of [14, 23]) to deﬁne
our AEG g, which tries to modify a correctly
classiﬁed point x with label y in the direction
of the gradient of the cost function, yielding
x(cid:48) = x−εyw/(cid:107)w(cid:107)2, where ε ≥ 0 is the strength
of the attack. To comply with the requirements
for an AEG, we deﬁne g as follows: g(x) = x(cid:48) if L(f, x) = 0 and f ∗(x) = f ∗(x(cid:48)) (corresponding
to (G2) and (G1), respectively), while g(x) = x otherwise. Therefore, if x(cid:48) is misclassiﬁed by f , x
and x(cid:48) are the only points mapped to x(cid:48) by g. This simple form of g and the knowledge of ρ allows to
compute the density hg, making it easy to compute the adversarial error estimate (2). Figure 4 shows
the average p-values produced by our N -model independence test for a dependent (solid lines) and
an independent (dashed lines) test set. It can be seen that in the dependent case the test can reject
independence with high conﬁdence for a large range of attack strength ε, while the independence
hypothesis is not rejected in the case of true independence. More details (including why only a range
of ε is suitable for detecting overﬁtting) are given in Appendix A.

Figure 4: Average p-values produced by the indepen-
dence test in a separable linear classiﬁcation problem
for the cases of both when the model is independent of
(dashed lines) and, resp., dependent on (solid lines) the
test set.

5 Testing overﬁtting on ImageNet

In the previous section we showed that the proposed adversarial-example-based dependence test
works for a synthetic problem where the densities can be computed exactly. In this section we apply
our estimates to a popular image classiﬁcation benchmark, ImageNet [8]; here the main issue is to
ﬁnd sufﬁciently strong AEGs that make computing the corresponding densities possible.

To facilitate the computation of the density hg, we only consider density-preserving AEGs as deﬁned
by (G3) (recall that (G3) is different from requiring hg = 1). Since in (2) and (5), hg(x) is multiplied
by L(f, x), we only need to determine the density hg for data points that are misclassiﬁed by f .

5.1 AEGs based on translations

To satisfy (G3), we implement the AEG using translations of images, which have recently been
proposed as means of generating adversarial examples [1]. Although relatively weak, such attacks ﬁt
our needs well: unless the images are procedurally centered, it is reasonable to assume that translating
them by a few pixels does not change their likelihood.4 We also make the natural assumption that
the small translations used do not change the true class of an image. Under these assumptions,

4Note that this assumption limits the applicability of our method, excluding such centered or essentially

centered image classiﬁcation benchmarks as MNIST [20] or CIFAR-10 [18].

7

10-210-1100101102†0.00.20.40.60.81.0p-valueN=1N=2N=10N=25N=100N=1N=2N=10N=25N=100translations by a few pixels satisfy conditions (G1) and (G3). An image-translating function g is a
valid AEG if it leaves all misclassiﬁed images in place (to comply with (G2)), and either leaves a
correctly classiﬁed image unchanged or applies a small translation.

The main beneﬁt of using a translational AEG g (with bounded translations) is that its density hg(x)
for an image x can be calculated exactly by considering the set of images x(cid:48) that can be mapped to x
by g (this is due to our assumption (G3)). We considered multiple ways for constructing translational
AEGs. The best version (selected based on initial evaluations on the ImageNet training set), which
we called the strongest perturbation, seeks a non-identical neighbor of a correctly classiﬁed image
x (neighboring images are the ones that are accessible through small translations) that causes the
classiﬁer to make an error with the largest conﬁdence.
Formally, we model images as 3D tensors in [0, 1]W ×H×C space, where C = 3 for RGB data, and
W and H are the width and height of the images, respectively. Let τv(x) denote the translation of
an image x by v ∈ Z2 pixels in the (X, Y) plane (here Z denotes the set of integers). To control
the amount of change, we limit the magnitude of translations and allow v ∈ Vε = {u ∈ Z2 :
u (cid:54)= (0, 0), (cid:107)u(cid:107)∞ ≤ ε} only, for some ﬁxed positive ε. Thus, we considers AEGs in the form
g(x) ∈ {τv(x) : v ∈ V} ∪ {x} if f (x) = f ∗(x) and g(x) = x otherwise (if x is correctly classiﬁed,
we attempt to translate it to ﬁnd an adversarial example in {τv(x) : v ∈ V} which is misclassiﬁed by
f , but x is left unchanged if no such point exists). Denoting the density of the pushforward measure
Pg by ρg, for any misclassiﬁed point x,

ρg(x) = ρ(x) +

(cid:88)

v∈V

ρ(τ−v(x))I(g(τ−v(x)) = x) = ρ(x)

1 +

(cid:32)

(cid:33)

I(g(τ−v(x)) = x)

(cid:88)

v∈V

where the second equality follows from (G3). Therefore, the corresponding density is

hg(x) = 1/(1 + n(x))

(8)
where n(x) = (cid:80)
I(g(τ−v(x)) = x) is the number of neighboring images which are mapped to
x by g. Note that given f and g, n(x) can be easily calculated by checking all possible translations
of x by −v for v ∈ V. It is easy to extend the above to non-deterministic perturbations, deﬁned as
distributions over AEGs, by replacing the indicator with its expectation P(g(τ−v(x)) = x|x, v) with
respect to the randomness of g, yielding

v∈V

hg(x) =

1

1 + (cid:80)

v∈V

P(g(τ−v(x)) = x|x, v)

.

(9)

If g is deterministic, we have hg(x) ≤ 1/2 for any successful adversarial example x. Hence, for such
g, the range U of the random variables Ti deﬁned in (5) has a tighter upper bound of 3/2 instead 2 (as
Ti ∈ [−1, 1/2]), leading to a tighter bound in (6) and a stronger pairwise independence test. In the
experiments, we use this stronger test. We provide additional details about the translational AEGs
used in Appendix B.

5.2 Tests of ImageNet models

We applied our test to check if state-of-the-art classiﬁers for the ImageNet dataset [8] have been
overﬁtted to the test set. In particular, we use the VGG16 classiﬁer of [27] and the Resnet50 classiﬁer
of [16]. Due to computational considerations, we only analyzed a single trained VGG16 model,
while the Resnet50 model was retrained 120 times. The models were trained using the parameters
recommended by their respective authors.

The preprocessing procedure of both architectures involves rescaling every image so that the smaller
of width and height is 256 and next cropping centrally to size 224 × 224. This means that translating
the image by v can be trivially implemented by shifting the cropping window by −v without any loss
of information for (cid:107)v(cid:107)∞ ≤ 16, because we have enough extra pixels outside the original, centrally
located cropping window. This implies that we can compute the densities of the translational AEGs
for any (cid:107)v(cid:107)∞ ≤ ε = (cid:98)16/3(cid:99) = 5 (see Appendix B.1 for detailed explanation). Because the ImageNet
data collection procedure did not impose any strict requirements on centering the images [8], it is
reasonable to assume (as we do) that small (lossless) translations respect the density-preserving
condition (G3).

In our ﬁrst experiment, we applied our pairwise independence test (6) with the AEGs described in
Appendix B (strongest, nearest, and the two random baselines) to all 1,271,167 training examples, as

8

Figure 5: p-values for the independence test on the ImageNet training set for different sample sizes and AEG
variants (left); original and adversarial risk estimates, (cid:98)RS(f ) and (cid:98)Rg(f ), on the ImageNet training set with
97.5% two-sided conﬁdence intervals for the ‘strongest attack’ AEG (right).

well as to a number of its randomly selected (uniformly without replacement) subsets of different
sizes. Besides this being a sanity check, we also used this experiment to select from different AEGs
and compare the performance of the pairwise independence test (6) to the basic version of the test
described in Section 3.1.

The left graph in Figure 5 shows that with the “strongest perturbation”, we were able to reject
independence of the trained model and the training samples at a conﬁdence level very close to 1 when
enough training samples are considered (to be precise, for the whole training set the conﬁdence level
is 99.9994%). Note, however, that the much weaker “smallest perturbation” AEG, as well as the
random transformations, are not able to detect the presence of overﬁtting. At the same time, the graph
on the right hand side shows the relative strength of the pairwise independence test compared to the
basic version based on independent conﬁdence interval estimates as described in detail in Section 3.1:
the 97.5%-conﬁdence intervals of the error estimates (cid:98)RS(f ) and (cid:98)Rg(f ) overlap, not allowing to
reject independence at a conﬁdence level of 95% (note that here S denotes the training set).

On the other hand, when applied to the test set, we obtained a p-value of 0.96, not allowing at all
to reject the independence of the trained model and the test set. This result could be explained by
the test being too weak, as no overﬁtting is detected to the training set at similar sample sizes (see
Figure 5), or simply the lack of overﬁtting. Similar results were obtained for Resnet50, where even
the N -model test with N = 120 independently trained models resulted a p value of 1, not allowing
to reject independence at any conﬁdence level. The view of no overﬁtting can be backed up in at
least two ways: ﬁrst, “manual” overﬁtting to the relatively large ImageNet test set is hard. Second,
since training an ImageNet model was just too computationally expensive until quite recently, only a
relatively small number of different architectures were developed for this problem, and the evolution
of their design was often driven by computational efﬁciency on the available hardware. On the other
hand, it is also possible that increasing N sufﬁciently might show evidence of overﬁtting (this is left
for future work).

6 Conclusions

We presented a method for detecting overﬁtting of models to datasets. It relies on an importance-
weighted risk estimate from a new dataset obtained by generating adversarial examples from the
original data points. We applied our method to the popular ImageNet image classiﬁcation task. For
this purpose, we developed a specialized variant of our method for image classiﬁcation that uses
adversarial translations, providing arguments for its correctness. Luckily, and in agreement with other
recent work on this topic [25, 26, 13, 21, 32], we found no evidence of overﬁtting of state-of-the-art
classiﬁers to the ImageNet test set.

The most challenging aspect of our methods is to construct adversarial perturbations for which we can
calculate the importance weights; ﬁnding stronger perturbations than the ones based on translations
for image classiﬁcation is an important question for the future. Another interesting research direction
is to consider extensions beyond image classiﬁcation, for example, by building on recent adversarial
attacks for speech-to-text methods [5], machine translation [11] or text classiﬁcation [12].

9

0.20.40.60.81.01.2sample size×1060.00.20.40.60.81.0P-valuevariantstrongestnearestrandomrandom20.20.40.60.81.01.2sample size×1060.1740.1760.1780.1800.1820.1840.1860.188cRS(f)cRg(f)Acknowledgements

We thank J. Uesato for useful discussions and advice about adversarial attack methods and sharing
their implementations [30] with us, as well as M. Rosca and S. Gowal for help with retraining image
classiﬁcation models. We also thank B. O’Donoghue for useful remarks about the manuscript, and L.
Schmidt for an in-depth discussion of their results on this topic. Finally, we thank D. Balduzzi, S.
Legg, K. Kavukcuoglu and J. Martens for encouragement, support, lively discussions and feedback.

References

[1] Aharon Azulay and Yair Weiss. Why do deep convolutional networks generalize so poorly to small image

transformations? 2018. arXiv:1805.12177.

[2] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning
to align and translate. In Proceedings of the International Conference on Learning Representations (ICLR),
2015.

[3] Stéphane Boucheron, Gábor Lugosi, and Pascal Massart. Concentration Inequalities: A Nonasymptotic

Theory of Independence. Oxford University Press, 2013.

[4] James Antonio Bucklew. Introduction to Rare Event Simulation. Springer New York, 2004.

[5] N. Carlini and D. Wagner. Audio adversarial examples: Targeted attacks on speech-to-text. In 2018 IEEE

Security and Privacy Workshops (SPW), May 2018.

[6] Nicholas Carlini and David A. Wagner. Adversarial examples are not easily detected: Bypassing ten
detection methods. In Proceedings of the 10th ACM Workshop on Artiﬁcial Intelligence and Security,
AISec@CCS 2017, Dallas, TX, USA, November 3, 2017, pages 3–14, 2017. URL http://doi.acm.org/
10.1145/3128572.3140444.

[7] Nicholas Carlini and David A. Wagner. Towards evaluating the robustness of neural networks. In 2017
IEEE Symposium on Security and Privacy, SP 2017, San Jose, CA, USA, May 22-26, 2017, pages 39–57,
2017. URL https://doi.org/10.1109/SP.2017.49.

[8] J. Deng, W. Dong, R. Socher, L. J. Li, Kai Li, and Li Fei-Fei. ImageNet: A large-scale hierarchical image
database. In 2009 IEEE Conference on Computer Vision and Pattern Recognition, pages 248–255, June
2009. doi: 10.1109/CVPR.2009.5206848.

[9] L. Deng, G. Hinton, and B. Kingsbury. New types of deep neural network learning for speech recognition
and related applications: an overview. In 2013 IEEE International Conference on Acoustics, Speech and
Signal Processing, pages 8599–8603. IEEE, May 2013.

[10] Cynthia Dwork, Vitaly Feldman, Moritz Hardt, Toniann Pitassi, Omer Reingold, and Aaron Roth. The
reusable holdout: Preserving validity in adaptive data analysis. Science, 349(6248):636–638, 2015.

[11] Javid Ebrahimi, Daniel Lowd, and Dejing Dou. On adversarial examples for character-level neural machine
translation. In Proceedings of the 27th International Conference on Computational Linguistics, COLING
2018, Santa Fe, New Mexico, USA, August 20-26, 2018, pages 653–663, 2018.

[12] Javid Ebrahimi, Anyi Rao, Daniel Lowd, and Dejing Dou. Hotﬂip: White-box adversarial examples for text
classiﬁcation. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics,
ACL 2018, Melbourne, Australia, July 15-20, 2018, Volume 2: Short Papers, pages 31–36, 2018.

[13] Vitaly Feldman, Roy Frostig, and Moritz Hardt. The advantages of multiple classes for reducing overﬁtting
from test set reuse. In Proceedings of the 36th International Conference on Machine Learning, pages
1892–1900, 2019.

[14] I. J. Goodfellow, J. Shlens, and C. Szegedy. Explaining and harnessing adversarial examples.

In

Proceedings of the International Conference on Learning Representations (ICLR), 2015.

[15] Alex Graves, Abdel-rahman Mohamed, and Geoffrey Hinton. Speech recognition with deep recurrent
neural networks. In 2013 IEEE International Conference on Acoustics, Speech and Signal Processing,
pages 6645–6649. IEEE, 2013.

[16] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition.
In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 770–778,
2016.

10

[17] H. Kahn and T. E. Harris. Estimation of particle transmission by random sampling. In Monte Carlo

Method, volume 12 of Applied Mathematics Series, pages 27–30. National Bureau of Standards, 1951.

[18] Alex Krizhevsky. Learning multiple layers of features from tiny images. Technical report, University of

Toronto, 2009.

[19] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. ImageNet classiﬁcation with deep convolutional
neural networks. In F. Pereira, C. J. C. Burges, L. Bottou, and K. Q. Weinberger, editors, Advances in
Neural Information Processing Systems 25, pages 1097–1105. Curran Associates, Inc., 2012.

[20] Yann LeCun and Corinna Cortes. MNIST handwritten digit database. http://yann.lecun.com/exdb/mnist/,

2010.

[21] Horia Mania, John Miller, Ludwig Schmidt, Moritz Hardt, and Benjamin Recht. Model similarity mitigates

test set overuse. 2019. arXiv:1905.12580.

[22] Volodymyr Mnih, Csaba Szepesvári, and Jean-Yves Audibert. Empirical Bernstein stopping. In Proceed-
ings of the 25th International Conference on Machine Learning, ICML ’08, pages 672–679, New York,
NY, USA, 2008. ACM.

[23] Nicolas Papernot, Patrick D. McDaniel, and Ian J. Goodfellow. Transferability in machine learning: from

phenomena to black-box attacks using adversarial samples. 2016. arXiv:1605.07277.

[24] Nicolas Papernot, Patrick McDaniel, Ian Goodfellow, Somesh Jha, Z Berkay Celik, and Ananthram Swami.
Practical black-box attacks against machine learning. In Proceedings of the 2017 ACM on Asia Conference
on Computer and Communications Security, pages 506–519. ACM, 2017.

[25] Benjamin Recht, Rebecca Roelofs, Ludwig Schmidt, and Vaishaal Shankar. Do CIFAR-10 classiﬁers

generalize to CIFAR-10? 2018. arXiv:1806.00451.

[26] Benjamin Recht, Rebecca Roelofs, Ludwig Schmidt, and Vaishaal Shankar. Do ImageNet classiﬁers

generalize to ImageNet? 2019. arXiv:1902.10811.

[27] K. Simonyan and A. Zisserman. Very deep convolutional networks for large-scale image recognition. In

Proceedings of the International Conference on Learning Representations (ICLR), 2015.

[28] Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru
Erhan, Vincent Vanhoucke, and Andrew Rabinovich. Going deeper with convolutions. In 2015 IEEE
Conference on Computer Vision and Pattern Recognition (CVPR), 2015. arXiv:1409.4842.

[29] T. Tieleman and G. Hinton. Lecture 6.5—RmsProp: Divide the gradient by a running average of its recent

magnitude. COURSERA: Neural Networks for Machine Learning, 2012.

[30] Jonathan Uesato, Brendan O’Donoghue, Aäron van den Oord, and Pushmeet Kohli. Adversarial risk and
the dangers of evaluating against weak attacks. In Proceedings of the 35th International Conference on
Machine Learning, volume 80 of Proceedings of Machine Learning Research, pages 5025–5034, 2018.
arXiv:1802.05666.

[31] Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V. Le, Mohammad Norouzi, Wolfgang Macherey, Maxim
Krikun, Yuan Cao, Qin Gao, Klaus Macherey, Jeff Klingner, Apurva Shah, Melvin Johnson, Xiaobing
Liu, Lukasz Kaiser, Stephan Gouws, Yoshikiyo Kato, Taku Kudo, Hideto Kazawa, Keith Stevens, George
Kurian, Nishant Patil, Wei Wang, Cliff Young, Jason Smith, Jason Riesa, Alex Rudnick, Oriol Vinyals,
Greg Corrado, Macduff Hughes, and Jeffrey Dean. Google’s neural machine translation system: Bridging
the gap between human and machine translation. 2016. arXiv:1609.08144.

[32] Chhavi Yadav and Léon Bottou. Cold case: The lost MNIST digits. May 2019. arXiv:1905.10498.

11

A Synthetic experiments

In this section, we present full details of the experiments on a simple synthetic classiﬁcation problem,
which we presented brieﬂy in Section 4. These experiments illustrate the power of the method of
Section 3. The advantage of the simple setup considered here is that we are able to compute the
density hg in an analytic form (see Figure 6 for an illustration).

A.1 Data distribution and model

Let X = R500 and consider an input distribution with a density ρ that is an equally weighted
± (µ±, σ2I) with
mixture of two 500-dimensional isotropic truncated Gaussian distributions N trunc
√
coordinate-wise standard deviation σ =
500 (I denotes the identity matrix of size 500 × 500),
means µ± = [±1, 0, 0, . . . , 0] and densities ρ± truncated in the ﬁrst dimension such that ρ+(x) = 0
if x1 ≤ 0.025 and ρ−(x) = 0 if x1 ≥ −0.025. The label of an input point x is f ∗(x) = sgn(x1),
which is the sign of its ﬁrst coordinate.
We consider linear classiﬁers of the form f (x) = sgn(w(cid:62)x + b) trained with the cross-entropy loss
c((w, b), x, y) = ln(1 + e−y(w(cid:62)x+b)) where y = f ∗(x). We employ a one-step gradient method
(which is an L2 version of the fast gradient-sign method of [14, 23]) to deﬁne our AEG g, which
tries to modify a correctly classiﬁed point x with label y in the direction of the gradient of the cost
function c: x(cid:48) = x + ε∇xc((w, b), x, y)/(cid:107)∇xc((w, b), x, y)(cid:107)2 for some ε > 0. For our speciﬁc
choice of c, the above simpliﬁes to x(cid:48) = x − εyw/(cid:107)w(cid:107)2. To comply with the requirements for an
AEG, we deﬁne g as follows: g(x) = x(cid:48) if L(f, x) = 0 and f ∗(x) = f ∗(x(cid:48)) (corresponding to
(G2) and (G1), respectively), while g(x) = x otherwise. Therefore, if x(cid:48) is misclassiﬁed by f , x

x2

y = −1

y = +1

xC

x(cid:48)
C

x(cid:48)
D

x(cid:48)
E

x(cid:48)
A

x(cid:48)
B

xA

0

xB

x1

w

xD

xE

Figure 6: Illustration of the data distribution and the linear model f (x) = sgn(w(cid:62)x + b) in two dimensions.
The blue and green gradients show the probability density ρ of the data with true labels y = −1 and y = 1,
respectively, while the white space between them is the margin with ρ = 0. The red line is the model’s
classiﬁcation boundary with its parameter vector w shown by the purple arrow. Depending on the label, w or
−w is the direction of translation used to perturbed the correctly classiﬁed data points, and the translations
used by the AEG g for speciﬁc points are depicted by grey arrows: solid arrows indicate the cases where
g(x) = x(cid:48)
(cid:54)= x, while dashed arrows are for candidate translations which are not performed by the AEG
because they would change the true label, f ∗(x(cid:48)) (cid:54)= f ∗(x), and hence g(x) = x (cid:54)= x(cid:48). Each original/perturbed
data point is represented by a color-coded circle: the inner color corresponds to the true label (dark blue for
y = −1 and dark green for y = 1) while the outer color to the model’s prediction (dark blue for f (x) = −1
and dark green for f (x) = 1). Points x(cid:48)
B can be obtained from xA and xB, respectively, by applying
the AEG, x(cid:48)
A = g(xA) and x(cid:48)
A, the density
(Radon-Nikodym derivative) can be obtained as hg(x(cid:48)
A)) ∈ (0, 1). In the case of
x(cid:48)
B) = ρ(x(cid:48)
B, hg(x(cid:48)
A) does not
depend on whether xA or x(cid:48)
A a “successful adversarial
example” while in the second case x(cid:48)
B)).
x(cid:48)
C is not a successful adversarial example since L(f, x(cid:48)
C according to our
deﬁnition). Points xD and xE are not perturbed by our AEG, since f ∗(xD) (cid:54)= f ∗(x(cid:48)
E).

A is called “originally misclassiﬁed” (a similar argument holds for hg(x(cid:48)

B)) = 0 due to the margin. Note that the formula for hg(x(cid:48)

A is in the original test set S; in the ﬁrst case we call x(cid:48)

B = g(xB). Since only xA is mapped to x(cid:48)

C ) = 0 (however, g(xC ) = x(cid:48)

D) and f ∗(xE) (cid:54)= f ∗(x(cid:48)

B)/(ρ(xB) + ρ(x(cid:48)

A)/(ρ(xA) + ρ(x(cid:48)

A by g, and g(x(cid:48)

A) = ρ(x(cid:48)

A and x(cid:48)

A) = x(cid:48)

12

and x(cid:48) are the only points mapped to x(cid:48) by g. Thus, the density at x(cid:48) after the transformation g is
ρ(cid:48)(x(cid:48)) = ρ(x) + ρ(x(cid:48))(1 − L(f, x))I(f ∗(x) = f ∗(x(cid:48))) and

hg(x(cid:48)) =

ρ(x(cid:48))
ρ(cid:48)(x(cid:48))

=

ρ(x(cid:48))
ρ(x(cid:48)) + ρ(x)(1 − L(f, x))I(f ∗(x) = f ∗(x(cid:48)))

(note that I(L(f, x) = 0) = 1 − L(f, x)).

A.2 Experiment setup

We present two experiments showing the behavior of our independence test: one where the training
and test sets are independent, and another where they are not.

In the ﬁrst experiment a linear classiﬁer was trained on a training set STr of size 500 for 50,000 steps
using the RMSProp optimizer [29] with batch size 100 and learning rate 0.01, obtaining zero (up to
numerical precision) ﬁnal training loss c and, consequently, 100% prediction accuracy on the training
data. Then the trained classiﬁer was tested on a large test set STe of size 10,000.5 Both sets were
drawn independently from ρ deﬁned above. We used a range of ε values matched to the scale of the
data distribution: from 10−2, which is the order of magnitude of the margin between two classes
(0.05), to 102, which is the order of magnitude of the width of the Gaussian distribution used for each
classes (σ =

500).

√

In the second experiment we consider the situation where the training and test sets are not independent.
To enhance the effects of this dependence, the setup was modiﬁed to make the training process more
amenable to overﬁtting by simulating a situation when the model has a wrong bias (this may happen in
practice if a wrong architecture or data preprocessing method is chosen, which, despite the modeler’s
best intentions, worsens the performance). Speciﬁcally, during training we added a penalty term
104w2
1 to the training loss c, decreased the size of the test set to 1000 and used 50% of the test data
for training (the ﬁnal penalized training loss was 0.25 with 100% prediction accuracy on the training
set). Note that the small training set and the large penalty on w1 yield classiﬁers that are essentially
independent of the only interesting feature x1 (recall that the true label of a point x is sgn(x1)) and
overﬁt to the noise in the data, resulting in a true model risk R(f ) ≈ 1/2.

A.3 Results

The results of the two experiments are shown in Figure 7, plotted against different perturbation
strengths: the left column corresponds to the ﬁrst experiment while the right column to the second.
The ﬁrst row presents the p-values for rejecting the independence hypothesis, calculated by repeating
the experiment (sampling data and training the classiﬁer) 100 times and applying the single-model
(Section 3, labelled as N = 1 in the plots) and N -model (Section 3.3, labelled as N = 2, 10, 25, 100
in the plots) independence test, and taking the average over models (or model sets of size N ) for each
ε. We also plot empirical 95% two-sided conﬁdence intervals (N ≤ 2) or, due to limited number of
p-values available after dividing 100 runs into disjoint bins of size N ≥ 10, ranges between minimum
and maximum value (N = 10, 25). For all methods of detecting dependence, it can be seen that
for the independent case the test is correctly not able to reject the independence hypothesis (the
average p-value is very close to 1, although in some runs it can drop to as low as 0.5). On the other
hand, for 10 ≤ ε ≤ 50, the non-independent model failed the independence test at conﬁdence level
1 − δ ≈ 100%, hence, in this range of ε our independence test reliably detects overﬁtting.

In fact, it is easy to argue that our test should only work for a limited range of ε, that is, it should
not reject independence for too small or too large values of ε. First we consider the case of small
ε values. Notice that except for points g(x) ε-close (in L2-norm) to the true decision boundary or
the decision boundary of f , g(x) is invertible: if g(x) is correctly classiﬁed and is ε-away from the
true decision boundary, there is exactly one point, x, which is translated to g(x), while if g(x) is
incorrectly classiﬁed and ε-away from the decision boundary of f , no translation leads to g(x) and
x = g(x); any other points are ε-close to the decision boundary of either f or f ∗. Thus, since ρ is
bounded, g(x) is invertible on a set of at least 1 − O(ε) probability (according to ρ). When ε → 0,
g(x) → x, and so ρ(g(x)) → ρ(x) for all points x with |x1| (cid:54)= 0.025 (since ρ is continuous in all
such x), implying hg(g(x)) ≈ 1 on these points. It also follows that L(f, x) (cid:54)= L(f, g(x)) can only

5The large number of test examples ensures that the random error in the empirical error estimate is negligible.

13

Figure 7: Risk and overﬁtting metrics for a synthetic problem with linear classiﬁers as a function of the
perturbation strengths ε (log scale). Left: unbiased model tested on a large, independent test set (in this case
(cid:98)RS(f ) ≈ (cid:98)Rg(f ) ≈ R(f )); right: trained model overﬁtted to the test set ( (cid:98)RS(f ) ≤ (cid:98)Rg(f ) while both are
smaller than R(f )). First row: Average p-value δ for the pairwise independence test with over 100 runs
(N = 1) or the N -model independence test (N > 1). The bounds plotted are either empirical 95% two-sided
conﬁdence intervals (N ≤ 2) or ranges between minimum and maximum value (N = 10, 25). Second row:
Empirical two-sided 97.5% conﬁdence intervals for the empirical test error rate (cid:98)RS(f ) and the adversarial risk
estimate (cid:98)Rg(f ). On the left, R(f ) ≈ (cid:98)RS(f ), while R(f ) is shown separately on the right. Third row: Average
densities (Radon-Nikodym derivatives) for originally misclassiﬁed points and for the new data points obtained
by successful adversarial transformations (with empirical 97.5% two-sided conﬁdence intervals). Fourth row:
The empirical test error rate (cid:98)RS(f ) and the adversarial risk estimate (cid:98)Rg(f ) for a single realization with 97.5%
two-sided conﬁdence intervals computed from Bernstein’s inequality, the adversarial error rate (cid:98)RS(cid:48) (f ), and the
expected error R(f ) (on the right, on the left R(f ) ≈ (cid:98)RS(f )). Fifth row: Histograms of p-values for selected ε
values over 100 runs.

14

10-210-1100101102†0.00.20.40.60.81.0p-valueN=1N=2N=10N=25N=10010-210-1100101102†0.00.20.40.60.81.0p-valueN=1N=2N=10N=25N=10010-210-1100101102†0.00.20.40.60.81.0errorcRg(f)cRS(f)10-210-1100101102†0.00.20.40.60.81.0errorcRg(f)cRS(f)R(f)10-210-1100101102†0.00.20.40.60.81.0average hgoriginal misclassifiedadversarial10-210-1100101102†0.00.20.40.60.81.0average hgoriginal misclassifiedadversarial10-210-1100101102†0.00.20.40.60.81.0errorcRg(f)cRS(f)cRS0(f)10-210-1100101102†0.00.20.40.60.81.0errorcRg(f)cRS(f)cRS0(f)R(f)0.00.20.40.60.81.0p-value020406080100countδ(†=0.1)δ(†=5)δ(†=20)0.00.20.40.60.81.0p-value020406080100countδ(†=0.1)δ(†=5)δ(†=6)δ(†=20)happen to a set of points with an O(ε) ρ-probability. This means that L(f, g(x))hg(g(x)) ≈ L(f, x)
on a set of 1 − O(ε) ρ-probability, and for these points Tg(x) = L(f, g(x))hg(g(x)) − L(f, x) ≈ 0.
Thus, Tg(X) ≈ 0 with ρ-probability 1 − O(ε). Unless the test set S is concentrated in large part
on the set of remaining points with O(ε) ρ-probability, the test statistic |TS,g(f )| = O(ε) with high
probability and our method will not reject the independence hypothesis for ε → 0.

When ε is large (ε → ∞), notice that for any point x with non-vanishing probability (i.e., with
ρ(x) > c for some c > 0), if g(x) (cid:54)= x than ρ(g(x)) ≈ 0. Therefore, for such an x, if L(f, x) = 0 and
L(f, g(x)) = 1, hg(g(x)) = ρ(g(x))/(ρ(x) + ρ(g(x))) ≈ 0, and so Tg(x) ≈ 0 (if L(f, g(x)) = 0,
we trivially have Tg(x) = 0). If L(f, x) = 1, we have g(x) = x. If g is invertible at x then
hg(x) = 1 and Tg(x) = 0. If g is not invertible, then there is another x(cid:48) such that g(x(cid:48)) = x;
however, if ρ(x) > c then ρ(x(cid:48)) ≈ 0 (since ε is large), and so hg(g(x)) = ρ(x)/(ρ(x) + ρ(x(cid:48))) ≈ 1,
giving Tg(x) ≈ 0. Therefore, for large ε, Tg(X) ≈ 0 with high probability (i.e., for points with
ρ(x) > c), so the independence hypothesis will not be rejected with high probability.

To better understand the behavior of the test, the second row of Figure 7 shows the empirical test error
rate (cid:98)RS(f ), the (unadjusted) adversarial error rate (cid:98)RS(cid:48)(f ), and the adversarial risk estimate (cid:98)Rg(f ),
together with their conﬁdence intervals. For the non-independent model, we also show the expected
error R(f ) (estimated over a large independent test set), while it is omitted for the independent model
where it approximately coincides with both (cid:98)RS(f ) and (cid:98)Rg(f ). While the reweighted adversarial error
estimate (cid:98)Rg(f ) remains the same for all perturbations in case of an independent test set (left column),
the adversarial error rate (cid:98)RS(cid:48)(f ) varies a lot for both the dependent and independent test sets. For
example, in the case when the test samples and the model f are not independent, it undershoots
the true error for ε < 10 and overshoots it for larger perturbations. For very large perturbations
(ε close to 100), the behavior of (cid:98)RS(cid:48)(f ) depends on the model f : in the independent case (cid:98)RS(cid:48)(f )
decreases back to (cid:98)RS(f ) because such large perturbations increasingly often change the true label
of the original example, so less and less adversarial points are generated. In the case when the data
and the model are not independent (right column), the adversarial perturbations are almost always
successful (i.e., lead to a valid adversarial example for most originally correctly classiﬁed points),
yielding an adversarial error rate close to one for large enough perturbations. This is because the
decision boundary of f is almost orthogonal to the true decision boundary, and so the adversarial
perturbations are parallel with the true boundary, almost never changing the true label of a point.

The plots of the densities (Radon-Nikodym derivatives), given in the third row of Figure 7, show
how the change in their values compensate the increase of the adversarial error rate (cid:98)RS(cid:48)(f ): in the
independent case, the effect is completely eliminated yielding an unbiased adversarial error estimate
(cid:98)Rg(f ), which is essentially constant over the whole range of ε (as shown in the ﬁrst row), while in
the non-independent case the similar densities do not bring back the adversarial error rate (cid:98)RS(cid:48)(f ) to
the test error rate (cid:98)RS(f ), allowing the test to detect overﬁtting. Note that the densities exhibit similar
trends (and values) in both cases, driven by the dependence of typical values of the ρ(x)/ρ(g(x))
ratio on the perturbation strength ε for originally misclassifed points (L(f, x) = 1) and for successful
adversarial examples (i.e., L(f, x) = 0 and L(f, g(x)) = 1).

To compare the behavior of our improved, pairwise test and the basic version, the fourth row of
Figure 7 depicts a single realization of the experiments where the 97.5% conﬁdence intervals (as
computed from Bernstein’s inequality) are shown for the estimates. For the independent case, the
conﬁdence intervals of (cid:98)RS(f ) and (cid:98)Rg(f ) overlap for all ε, and thus the basic test is not able to detect
overﬁtting. In the non-independent case, the conﬁdence intervals overlap for ε = 10 and ε = 75, thus
the basic test is not able to detect overﬁtting with at a 95% conﬁdence level, while the improved test
(second row) is able to reject the independence hypothesis for these ε values at the same conﬁdence
level.

Finally, in the ﬁfth row of Figure 7 we plotted the histograms of the empirical distribution of p-values
for both models, over 100 independent runs (between the runs, all the data was regenerated and the
models were retrained). For ε = 0.1, 5, 20, they concentrate heavily on either δ = 0 or δ = 1, and
have very thin tails extending far towards the opposite end of the [0, 1] interval. This explains the
surprisingly wide 95% conﬁdence intervals for p-values plotted in the ﬁrst row. In particular, the fact
that some p-values for the independent model are as low as 0.5 does not mean the independence test
is not reliable, because almost all calculated δ values are close or equal to 1, and the few outliers are

15

independent model, ε = 10

non-independent model, ε = 6

Figure 8: Histograms of p-values from N -model (N = 1, 2, 10) independence tests for both synthetic models
and selected ε values, over 100 runs.

a combined consequence of the ﬁnite sample size and the effectiveness of the AEG. The additional
ε = 6 histogram for the non-independent model illustrates a regime which is in between the single-
model pairwise test (Section 3) completely failing to reject the independence hypothesis and clearly
rejecting it.

To verify experimentally whether the N -model independence test can be a more powerful detector of
overﬁtting than the single-model version, in Figure 8 (right panel) we plotted p-value histograms for
N = 1, 2, 10 for the intermediate AEG strength ε = 6 applied to the non-independent model over 100
training runs. Indeed, as N increases, the concentration of p-values around in the low (δ ≤ 0.2) range
increases. For N > 10 we did not have enough values to plot a histogram: for N = 25 we obtained
δ = 0.1851, 0.1599, 0.0661 and 0.1941, while for N = 100 the p-value is 0.1153. The increase of
the test power becomes apparent when we compare the last value with the mean of p-values obtained
by testing every training run separately, equal 0.5984, and the median 0.6385.

For comparison, we also plotted in Figure 8 (left panel) the corresponding histograms for the
independent model and a slightly higher attack strength, ε = 10, at which the independence tests fails
for the overﬁtted model even without averaging (see Figure 7, ﬁrst row, right panel). The histograms
are all clustered in the δ region close to 1, indicating that the N -model test is not overly pessimistic.

B Translational AEGs for image classiﬁcation models

For image classiﬁcation we consider two translation variants that are used in constructing a transla-
tional AEG. For every correctly classiﬁed image x, we consider translations from Vε (for some ε),
choosing g(x) from the set G(x) = {τv(x) : v ∈ Vε} ∪ {x}. If all translations result in correctly
classiﬁed examples, we set g(x) = x. Otherwise, we use one of two possible ways to select g(x)
(and we call the resulting points successful adversarial examples):

• Strongest perturbation: Assuming the number of classes is K, let l(f, x) ∈ RK denote
the vector of the K class logits calculated by the model f for image x, and let lexc(f, x) =
max0≤i<K li(f, x) − ly(f, x). We deﬁne

gstrongest(x) = argmaxx(cid:48)∈G(x) lexc(f, x(cid:48)),

with ties broken deterministically by choosing the ﬁrst translation from the candidate set,
going top to bottom and left to right in row-major order. Thus, here we seek a non-identical
“neighbor” that causes the classiﬁer to err the most, reachable from x by translations within
a maximum range ε.

• Nearest misclassiﬁed neighbor: Here we aim to ﬁnd the nearest image in G(x) that is
misclassiﬁed. That is, letting d(x, x(cid:48)) = (cid:107)v(cid:107)2 if x(cid:48) = τv(x) and ∞ otherwise, we deﬁne
gnearest(x) := argminx(cid:48)∈G(x),L(f,x(cid:48))=1 d(x, x(cid:48))

with ties broken deterministically as above.

The two perturbation variants are successful on exactly the same set of images, hence they lead to
the same adversarial error rates (cid:98)RS(cid:48)(f ). However, they are characterized by different values of the
density hg and, consequently, yield different adversarial risk estimates (cid:98)Rg(f ) and associated p-values

16

0.00.20.40.60.81.0p-value024681012PDFδ(N=1)δ(N=2)δ(N=10)0.00.20.40.60.81.0p-value012345PDFδ(N=1)δ(N=2)δ(N=10)for the independence test. The main difference between them is that the “strongest” version is more
likely to map multiple images to the same adversarial example, thus decreasing the densities for
successful adversarial examples and, counterintuitively, increasing them for originally misclassiﬁed
points (as their neighbors are less likely to be mapped to these points).

To better see the effect of adversarial perturbations, we also consider two random baselines that do
not take into account the success of a translation in generating misclassiﬁed points: grandom(x) is
chosen uniformly at random from G(x) \ {x}, and grandom2(x) is chosen uniformly at random from
G(x).

B.1 Maximum translations

In practice, translating an image is not always simple, as the new image has to be padded with
new pixels. When (central) crops of a larger image are used (as is typical for ImageNet classiﬁers),
translations can easily be implemented as long as the resulting new cropping window stays within the
original image boundaries. Even if an image can be translated by a vector v, this limits our ability to
compute hg(x(cid:48)) for the adversarial image x(cid:48) by (8) or (9) for gstrongest or gnearest. Indeed, if an image
x is shifted by v ∈ Vε to generate adversarial example x(cid:48), we need to examine translations of x(cid:48) with
vectors in Vε to ﬁnd the neighbors x(cid:48)(cid:48) of x(cid:48) potentially contributing to n(x(cid:48)) when computing hg(x(cid:48)).
Finally we need to consider translations of x(cid:48)(cid:48) with vectors in Vε to determine the exact value they
contribute, that is, to compute the exact probabilities in (9) (see Figure 9 for an illustration). Thus,
to be able to compute the density hg for the adversarial points obtained by translations from Vε, we
might need to be able to perform translations within V3ε.

Figure 9: Image translations which need to be considered for a translational AEG with ε = 3. The red, blue and
green balls represent the center of the original image x, adversarial example x(cid:48) = g(x) and another image x(cid:48)(cid:48)
contributing to ρg(x(cid:48)), respectively, while the semi-translucent squares of corresponding colors represent the
possible translations which need to be considered for each of x, x(cid:48) and x(cid:48)(cid:48). Solid light grey arrows represent the
relationships x(cid:48) = g(x) and x(cid:48) = g(x(cid:48)(cid:48)). Finally, the dashed arrow and the semi-translucent grey ball represent
an alternative mapping, which has to be ruled out while calculating the value of g(x(cid:48)(cid:48)) and, consequently, of
hg(x(cid:48)). It is easy to see that the colored squares (which contain the translations needing to be evaluated) extend
as far as 3ε from the original image x.

17

xx'x''ε

---

## Content from Previous Extraction (not in markitdown output)

### Visual/Chart/Graph Descriptions

#### 1

Detecting Overfitting via Adversarial Examples
Roman Werpachowski András György Csaba Szepesvári
Deep Mind, London, UK
{romanw, agyorgy, szepi}@google.com
Abstract
Thefrequentreuseoftestsetsinpopularbenchmarkproblemsraisesdoubtsabout
thecredibilityofreportedtest-errorrates. Verifyingwhetheralearnedmodelis
overfittedtoatestsetischallengingasindependenttestsetsdrawnfromthesame
data distribution are usually unavailable, while other test sets may introduce a
distributionshift. Weproposeanewhypothesistestthatusesonlytheoriginaltest
datatodetectoverfitting. Itutilizesanewunbiasederrorestimatethatisbased
onadversarialexamplesgeneratedfromthetestdataandimportanceweighting.
Overfittingisdetectedifthiserrorestimateissufficientlydifferentfromtheoriginal
testerrorrate. Wedevelopaspecializedvariantofourtestformulticlassimage
classification, andapplyittotestingoverfittingofrecentmodelstothepopular
Image Netbenchmark. Ourmethodcorrectlyindicatesoverfittingofthetrained
modeltothetrainingset, butisnotabletodetectanyoverfittingtothetestset, in
linewithotherrecentworkonthistopic.
1 Introduction
Deepneuralnetworksachieveimpressiveperformanceonmanyimportantmachinelearningbench-
marks, such as image classification [18, 19, 28, 27, 16], automated translation [2, 31] or speech
recognition[9,15]. However, thebenchmarkdatasetsareusedamultitudeoftimesbyresearchers
worldwide. Sincestate-of-the-artmethodsareselectedandpublishedbasedontheirperformance
onthecorrespondingtestset, itistypicaltoseeresultsthatcontinuouslyimproveovertime; see,
e.g., thediscussionof Rechtetal.[25]and Figure1 fortheperformanceimprovementofclassifiers
publishedforthepopular CIFAR-10 imageclassificationbenchmark[18].
1.00
0.95
0.90
0.85
0.80
2010 2012 2014 2016 2018
year
ycarucca
Thisprocessmaynaturallyleadtomodelsover-
fitted to the test set, rendering test error rate
(theaverageerrormeasuredonthetestset) an
unreliableindicatoroftheactualperformance.
Detectingwhetheramodelisoverfittedtothe
test set is challenging, since independent test
setsdrawnfromthesamedatadistributionare
generally not available, while alternative test
setsoftenintroduceadistributionshift.
Toestimatetheperformanceofamodelonun-
seendata, onemayusegeneralizationbounds
togetupperboundsontheexpectederrorrate.
The generalization bounds are also applicable Figure1:Accuracyofimageclassifiersonthe CIFAR-10
whenthemodelandthedataaredependent (e.g., testset, byyearofpublication (datafrom[25]).
forcrossvalidationorforerrorestimatesbased
onthetrainingdataorthereusedtestdata), buttheyusuallyleadtolooseerrorbounds. Therefore,
althoughmuchtighterboundsareavailableifthetestdataandthemodelareindependent, comparing
33 rd Conferenceon Neural Information Processing Systems (Neur IPS2019), Vancouver, Canada.

#### 2

confidenceintervalsconstructedaroundthetrainingandtesterrorratesleadstoanunderpoweredtest
fordetectingthedependenceofamodelonthetestset. Recently, severalmethodshavebeenproposed
thatallowthereuseofthetestsetwhilekeepingthevalidityoftesterrorrates[10]. However, these
areintrusive: theyrequiretheusertofollowastrictprotocolofinteractingwiththetestsetandare
thusnotapplicableinthemorecommonsituationwhenenforcingsuchaprotocolisimpossible.
Inthispaperwetakeanewapproachtothechallengeofdetectingoverfittingofamodeltothetest
set, anddeviseanon-intrusivestatisticaltestthatdoesnotrestrictthetrainingprocedureandisbased
on the original test data. To this end, we introduce a new error estimator that is less sensitive to
overfittingtothedata; ourtestrejectstheindependenceofthemodelandthetestdataifthenew
errorestimateandtheoriginaltesterrorratearetoodifferent. Thecorenovelideaisthatthenew
estimatorisbasedonadversarialexamples[14], thatis, ondatapoints1 thatarenotsampledfromthe
datadistribution, butinsteadarecleverlycraftedbasedonexistingdatapointssothatthemodelerrs
onthem. Severalauthorsshowedthatthebestmodelslearnedfortheabove-mentionedbenchmark
problemsarehighlysensitivetoadversarialattacks[14,23,30,6,7,24]: forinstance, onecanoften
createadversarialversionsofimagesproperlyclassifiedbyastate-of-the-artmodelsuchthatthe
modelwillmisclassifythem, yettheadversarialperturbationsare (almost) undetectableforahuman
observer; see, e.g., Figure 2, where the adversarial image is obtained from the original one by a
carefullyselectedtranslation.
Theadversarial (error) estimator proposedin
thisworkusesadversarialexamples (generated
from the test set) together with importance
weighting to take into account the change in
thedatadistribution (covariateshift) duetothe
adversarialtransformation. Theestimatorisun-
biasedandhasasmallervariancethanthestan-
dardtesterrorrateifthetestsetandthemodel
scale, weighingmachine toaster
areindependent.2 Moreimportantly, sinceitis
Figure2:Adversarialexampleforthe Image Netdataset
basedonadversariallygenerateddatapoints, the
generatedbya (5,−5) translation:theoriginalexample
adversarial estimator is expected to differ sig-
(left) iscorrectlyclassifiedbythe VGG16 model[27]as
nificantly from the test error rate if the model
“scale, weighingmachine,”theadversariallygenerated
isoverfittedtothetestset, providingawayto
example (right) isclassifiedas“toaster,”whiletheimage
detect test set overfitting. Thus, the test error
classisthesameforanyhumanobserver.
rate and the adversarial error estimate (calcu-
latedbasedonthesametestset) mustbecloseifthetestsetandthemodelareindependent, andare
expectedtobedifferentintheoppositecase. Inparticular, ifthegapbetweenthetwoerrorestimates
islarge, theindependencehypothesis (i.e., thatthemodelandthetestsetareindependent) isdubious
andwillberejected. Combiningresultsfrommultipletrainingruns, wedevelopanothermethodto
testoverfittingofamodelarchitectureandtrainingprocedure (forsimplicity, throughoutthepaper
werefertobothtogetherasthemodelarchitecture). Themostchallengingaspectofourmethodisto
constructadversarialperturbationsforwhichwecancalculateimportanceweights, whilekeeping
enoughdegreesoffreedominthewaytheadversarialperturbationsaregeneratedtomaximizepower,
theabilityofthetesttodetectdependencewhenitispresent.
Tounderstandthebehaviorofourtestsbetter, wefirstusethemonasyntheticbinaryclassification
problem, wherethetestsareabletosuccessfullyidentifythecaseswhereoverfittingispresent. Then
we apply our independence tests to state-of-the-art classification methods for the popular image
classificationbenchmark, Image Net[8]. Asasanitycheck, inallcasesexamined, ourtestrejects
(at confidence levels close to 1) the independence of the individual models from their respective
trainingsets. Applyingourmethodto VGG16[27]and Resnet50[16]models/architectures, their
independencetothe Image Nettestsetcannotberejectedatanyreasonableconfidence. Thisisin
agreementwithrecentfindingsof[26], andprovidesadditionalevidencethatdespiteoftheexisting
danger, itislikelythatnooverfittinghashappenedduringthedevelopmentof Image Netclassifiers.
Therestofthepaperisorganizedasfollows: In Section2, weintroduceaformalmodelforerror
estimationusingadversarialexamples, includingthedefinitionofadversarialexamplegenerators.
1 Throughoutthepaper, weusethewords“example”and“point”interchangeably.
2 Notethattheadversarialerrorestimator’sgoalistoestimatetheerrorrate, nottheadversarialerrorrate (i.e.,
theerrorrateontheadversarialexamples).
2

#### 3

Thenewoverfitting-detectiontestsarederivedin Section3, andappliedtoasyntheticproblemin
Section4, andtothe Image Netimageclassificationbenchmarkin Section5. Duetospacelimitations,
someauxiliaryresults, includingthein-depthanalysisofourmethodonthesyntheticproblem, are
relegatedtotheappendix.
2 Adversarial Risk Estimation
Weconsideraclassificationproblemwithdeterministic (noise-free) labels, whichisareasonable
assumptionformanypracticalproblems, suchasimagerecognition (weleavetheextensionofour
methodtonoisylabelsforfuturework).Let X ⊂RDdenotetheinputspaceand Y ={0,..., K−1}
thesetoflabels. Dataissampledfromthedistribution P over X, andtheclasslabelisdetermined
bythegroundtruthfunctionf∗ :X →Y. Wedenotearandomvectordrawnfrom P by X, andits
correspondingclasslabelby Y = f∗(X). Weconsiderdeterministicclassifiersf : X → Y. The
performanceoff ismeasuredbythezero-oneloss: L(f, x)=I(f (x)(cid:54)=f∗(x)),3 andtheexpected
error (alsoknownastheriskorexpectedriskinthelearningtheoryliterature) oftheclassifierf is
definedas R(f)=E[I(f (X)(cid:54)=Y)]= (cid:82) L(f, x) d P(x).
X
Consideratestdataset S ={(X , Y )...,(X , Y )}wherethe X aredrawnfrom Pindependently
1 1 m m i
ofeachotherand Y =f∗(X ). Inthelearningsetting, theclassifierf usuallyalsodependsonsome
i i
randomlydrawntrainingdata, henceisrandomitself. Iff is (statistically) independentfrom S, then
L(f, X ),..., L(f, X ) arei.i.d., thustheempiricalerrorrate
1 m
m m
1 (cid:88) 1 (cid:88)
R(cid:98)S (f)=
m
L(f, X
i
)=
m
I(f (X
i
)(cid:54)=Y
i
)
i=1 i=1
isanunbiasedestimateof R(f) forallf;thatis, R(f)=E[R(cid:98)S (f)|f]. Iff and S arenotindepen-
dent, theperformanceguaranteesontheempiricalestimatesavailableintheindependentcaseare
significantlyweakened;forexample, incaseofoverfittingto S, theempiricalerrorrateislikelytobe
muchsmallerthantheexpectederror.
Another well-known way to estimate R(f) is to use importance sampling (IS) [17]: instead of
samplingfromthedistribution P, wesamplefromanotherdistribution P(cid:48) andcorrecttheestimate
by appropriate reweighting. Assuming P is absolutely continuous with respect to P(cid:48) on the set
E ={x∈X :L(f, x)(cid:54)=0}, R(f)= (cid:82) L(f, x) d P(x)= (cid:82) L(f, x) h (x) d P(cid:48)(x), whereh= d P
X E d P(cid:48)
isthedensity (Radon-Nikodymderivative) of P withrespectto P(cid:48) on E (hcanbedefinedtohave
arbitraryfinitevalueson X \E). Itiswellknownthatthethecorrespondingempiricalerrorestimator
m m
1 (cid:88) 1 (cid:88)
R(cid:98)
S
(cid:48)
(cid:48)
(f)=
m
L(f, X
i
(cid:48)) h (X
i
(cid:48))=
m
I(f (X
i
(cid:48))(cid:54)=Y
i
(cid:48)) h (X
i
(cid:48)) (1)
i=1 i=1
obtainedfromasample S(cid:48) ={(X(cid:48), Y(cid:48)),...,(X(cid:48) , Y(cid:48) )}drawnindependentlyfrom P(cid:48) isunbiased
1 1 m m
(i.e., E[R(cid:98)S(cid:48) (f)|f]=R(f)) iff and S(cid:48) areindependent.
The variance of R(cid:98)(cid:48) is minimized if P(cid:48) is the so-called zero-variance IS distribution, which is
S(cid:48)
supportedon E withh (x)= R(f) forallx∈E (see, e.g.,[4, Section4.2]). Thissuggestthatan
L(f, x)
effectivesamplingdistribution P(cid:48) shouldconcentrateonpointswheref makesmistakes, whichalso
facilitatesthat R(cid:98)
S
(cid:48)
(cid:48)
(f) becomelargeiff isoverfittedto Sandhence R(cid:98)S (f) issmall. Weachievethis
throughtheapplicationofadversarialexamples.
2.1 Generatingadversarialexamples
In this section we introduce a formal framework for generating adversarial examples. Given a
classificationproblemwithdatadistribution Pandgroundtruthf∗, anadversarialexamplegenerator
(AEG) foraclassifierf isa (measurable) mappingg :X →X suchthat
(G1) gpreservestheclasslabelsofthesamples, thatis, f∗(x)=f∗(g (x)) for P-almostallx;
(G2) g doesnotchangepointsthatareincorrectlyclassifiedbyf, thatis, g (x) = xiff (x) (cid:54)=
f∗(x) for P-almostallx.
3 Foranevent B, I(B) denotesitsindicatorfunction:I(B)=1 if Bhappensand I(B)=0 otherwise.
3

#### 4

(cid:51) (cid:51) (cid:51) (cid:51) (cid:55) (cid:55) (cid:55) (cid:55) (cid:51) (cid:51) (cid:51) (cid:51) (cid:55) (cid:55) (cid:55) (cid:55) S
g
(cid:55) (cid:55)
(cid:55) (cid:55)
(cid:51) (cid:51) (cid:55) (cid:55) (cid:55) (cid:55) (cid:51) (cid:51) (cid:55) (cid:55) (cid:55) (cid:55) S(cid:48)
Figure3:Generatingadversarialexamples. Thetoprowdepictstheoriginaldataset S, withblueandorange
pointsrepresentingthetwoclasses. Theclassifier’spredictionisrepresentedbythecolorofthestripedareas
(checkmarksandcrossesdenoteifapointiscorrectlyorincorrectlyclassified).Thearrowsshowtheadversarial
transformationsviathe AEGg, resultinginthenewdataset S(cid:48);misclassifiedpointsareunchanged, whilesome
correctlyclassifiedpointsaremoved, buttheiroriginalclasslabelisunchanged. Iftheoriginaldatadistributionis
uniformover S, thetransformationgisdensitypreserving, butnotmeasurepreserving:afterthetransformation
thetworightmostcorrectlyclassifiedpointsineachclasshaveprobability0, whiletheleftmostmisclassified
pointineachclasshasprobability3/16;hence, thedensityh forthelatterpointsis1/3.
g
Figure 3 illustrates how an AEG works. In the literature, an adversarial example g (x) is usually
generatedbystayinginasmallvicinityoftheoriginaldatapointx (withrespectto, e.g., the2-orthe
max-norm) andassumingthattheresultinglabelofg (x) isthesameasthatofx (see, e.g.,[14,6]).
Thisfoundationalassumption—whichisinfactamarginconditiononthedistribution—iscaptured
incondition (G1). (G2) formalizesthefactthatthereisnoneedtochangesampleswhicharealready
misclassified. Indeed, existing AEGscomplywiththiscondition.
The performance of an AEG is usually measured by how successfully it generates misclassified
examples. Accordingly, we call a point g (x) a successful adversarial example if x is correctly
classifiedbyf andf (g (x))(cid:54)=f (x)(i.e., L(f, x)=0 and L(f, g (x))=1).
Inthedevelopmentofour AEGsforimagerecognitiontasks, wewillmakeuseofanothercondition.
Forsimplicity, weformulatethisconditionfordistributions P thathaveadensityρwithrespect
to the uniform measure on X, which is assumed to exist (notable cases are when X is finite, or
X =[0,1]D orwhen X =RD;inthelattertwocasestheuniformmeasureisthe Lebesguemeasure).
Theassumptionstatesthatthe AEGneedstobedensity-preserving:
(G3) ρ(x)=ρ(g (x)) for P-almostallx.
Note that a density-preserving map may not be measure-preserving (the latter means that for all
measurable A⊂X, P(A)=P(g (A))).
Weexpect (G3) toholdwhengperturbsitsinputbyasmallamountandifρissufficientlysmooth.
The assumption is reasonable for, e.g., image recognition problems (at least in a relaxed form,
ρ(x)≈ρ(g (x))) whereweexpectthatverycloseimageswillhaveasimilarlikelihoodasmeasured
byρ. An AEGemployingimagetranslations, whichsatisfies (G3), willbeintroducedin Section5.
Both (G1) and (G3) canberelaxed (toasoftmarginconditionorallowingaslightchangeinρ, resp.)
atthepriceofanextraerrortermintheanalysisthatfollows.
Forafixed AEGg :X →X, let P bethedistributionofg (X) where X ∼P (P isknownasthe
g g
pushforwardmeasureof P underg). Further, leth = d P on E ={x : L(f, x)(cid:54)=0}andarbitrary
g d Pg
otherwise. Itiseasytoseethat, on E, h (x) iswell-definedandh ≤1. Foranymeasurable A⊂E
g g
P (A)=P(g (X)∈A)≥P(g (X)∈A, X ∈E)=P(X ∈A)=P(A)
g
wherethesecondtolastequalityholdsbecauseg (X) = X forany X ∈ E undercondition (G2).
Thus, P(A)≤P (A) foranymeasurable A⊂E, whichimpliesthath iswell-definedon E and
g g
h (x)≤1 forallx∈E.
g
Onemaythinkthat (G3) impliesthath (x) = 1 forallx ∈ E. However, thisdoesnothold. For
g
example, if P isauniformdistribution, anyg : X → supp P satisfies (G3), wheresupp P ⊂ X
denotesthesupportofthedistribution P. Thisisalsoillustratedin Figure3.
2.2 Riskestimationviaadversarialexamples
Combining the ideas of this section so far, we now introduce unbiased risk estimates based on
adversarialexamples. Ourgoalistoestimatetheerror-rateoff throughanadversariallygenerated
4

#### 5

B(m,σ¯
g
2,δ,1), R(cid:98)S (f)+B(m,σ¯
g
2,δ,1)], whichbothcontain R(f) withprobabilityatleast1−δ,
intersectwithprobabilityatleast1−2δ.
Ontheotherhand, iff and S arenotindependent, theperformanceguarantees (3) and (4) maybe
violated and the confidence intervals may become disjoint. If this is detected, we can reject the
independencehypothesis (H) ataconfidencelevel1−2δor, equivalently, withp-value2δ. Inother
words, wereject (H) iftheabsolutevalueofthedifferenceoftheestimates T
S, g
(f)=R(cid:98) g (f)−R(cid:98)S (f)
exceeds the threshold B(m,σ¯2,δ,1)+B(m,σ¯2,δ,1) (note that E[T (f) = 0] if S and f are
S g S, g
independent).
3.2 Pairwisetest
A smaller threshold for |T (f)|, and hence a more effective independence test, can be devised
S, g
if instead of independently estimating the behavior of R(cid:98)S and R(cid:98) g (f), one utilizes their apparent
correlation. Indeed, T (f)=(1/m)
(cid:80) m
T (f) where
S, g i=1 i, g
T (f)=L(f, g (X )) h (g (X ))−L(f, X ) (5)
i, g i g i i
and the two terms in T (f) have the same mean and are typically highly correlated by the con-
i, g
structionofg. Thus, wecanapplytheempirical Bernsteinbound[22]tothepairwisedifferences
T (f) tosetatighterthresholdinthetest: iftheindependencehypothesis (H) holds (i.e., S andf
i, g
areindependent), thenforany0<δ <1, withprobabilityatleast1−δ,
|T (f)|≤B(m,σ¯2,δ, U) (6)
S, g T
(cid:113)
with B(m,σ2,δ, U) = 2σ2 ln (3/δ) + 3 Uln (3/δ), whereσ¯2 = (1/m) (cid:80) m (T (f)−T (f))2 is
m m T i=1 i S, g
theempiricalvarianceofthe T (f) termsand U =sup T (f)−inf T (f);wealsousedthefact
i, g i, g i, g
thattheexpectationofeach T (f), andhencethatof T (f), iszero. Sinceh ≤1 if L(f, x)=1
i, g S, g g
(asdiscussedin Section2.2), itfollowsthat U ≤2, butfurtherassumptions (suchasgbeingdensity
preserving) canresultintighterbounds.
Thisleadstoourpairwisedependencedetectionmethod:
if|T (f)|>B(m,σ¯2,δ,2), reject (H) ataconfidencelevel1−δ(p-valueδ).
S, g T
Foragivenstatistic (|T (f)|,σ¯2), thelargestconfidencelevel (smallestp-value) atwhich (H) can
S, g T
berejectedcanbecalculatedbysettingthevalueofthestatistic|T (f)|−B(m,σ¯2,δ,2) tozero
S, g T
andsolvingforδ. Thisleadstothefollowingformulaforthep-value (ifthesolutionislargerthan1,
whichhappenswhenthebound (6) isloose,δiscappedat1):
(cid:26) (cid:16) √ (cid:17)(cid:27)
δ =min 1,3 e
−
9
m
U2
σ¯
T
2+3 U|TS, g (f)|−σ¯T σ¯
T
2+6 U|TS, g (f)|
. (7)
Note that in order for the test to work well, we not only need the test statistic T (f) to have a
S, g
smallvarianceincaseofindependence (thiscouldbeachievedifgweretheidentity), butwealso
needtheestimators R(cid:98)S (f) and R(cid:98) g (f) behavesufficientlydifferentlyiftheindependenceassumption
isviolated. Thelatterbehaviorisencouragedbystronger AEGs, aswewillshowempiricallyin
Section5.2(see Figure5 inparticular).
3.3 Dependencedetectorforrandomizedtraining
Thedependencebetweenthemodelandthetestsetcanarisefrom (i) selectingthe“best”random
seedinordertoimprovethetestsetperformanceand/or (ii) tweakingthemodelarchitecture (e.g.,
neuralnetworkstructure) andhyperparameters (e.g., learning-rateschedule). Ifonehasaccesstoa
singleinstanceofatrainedmodel, thesetwosourcescannotbedisentangled. However, ifthemodel
architectureandtrainingprocedureisfullyspecifiedandcomputationalresourcesareadequate, it
ispossibletoisolate (i) and (ii) byretrainingthemodelmultipletimesandcalculatingthep-value
foreverytrainingrunseparately. Assuming N models, letf , j =1,..., N denotethej-thtrained
j
modelandp thep-valuecalculatedusingthepairwiseindependencetest (6)(i.e., from Eq.7 in
j
Section3). Wecaninvestigatethedegreetowhich (i) occursbycomparingthep valueswiththe
j
correspondingtestseterrorrates R (f ). Toinvestigatewhether (ii) occurs, wecanaverageoverthe
S j
randomnessofthetrainingruns.
6

#### 6

For every example X ∈ S, consider the average test statistic T¯ = 1 (cid:80)N T (f ), where
i i N j=1 i, gj j
T (f ) isthestatistic (5) calculatedforexample X andmodelf with AEGg selectedformodel
i, gj j i j j
f (notethat AEGsaremodel-dependentbyconstruction). If, foreachiandj, therandomvariables
j
T (f ) areindependent, thensoarethe T¯ (foralli). Hence, wecanapplythepairwisedependence
i j i
detector (6) with T¯ insteadof T , usingtheaverage T¯ =(1/m) (cid:80) m T¯ withempiricalvariance
i i S i=1 i
σ¯2 =(1/m) (cid:80) m (T¯ −T¯ )2, givingasinglep-valuep . Ifthetrainingrunsvaryenoughintheir
T, N i=1 i S N
outcomes, differentmodelsf errondifferentdatapoints X , leadingtoσ¯2 <σ¯2, andtherefore
j j T, N T
strengtheningthepowerofthedependencedetector. Forbrevity, wecallthisindependencetestan
N-modeltest.
4 Syntheticexperiments
1.0
0.8
0.6
0.4
0.2
0.0
10-2 10-1 100 101 102
†
eulav-p
Firstweverifytheeffectivenessofourmethod
onasimplelinearclassificationproblem. Due
N=1 N=1
tospacelimitations, weonlyconveyhigh-level
N=2 N=2
results here, details are given in Appendix A. N=10 N=10
We assume that the data is linearly separa-
N=25 N=25
ble with a margin and the density ρ is known. N=100 N=100
We consider a linear classifiers of the form
f (x) = sgn (w (cid:62) x+b) trained with the cross-
entropylossc, andweemployaone-stepgra-
dient method (which is an L version of the
2
fastgradient-signmethodof[14,23]) todefine
Figure4: Averagep-valuesproducedbytheindepen-
our AEG g, which tries to modify a correctly
dencetestinaseparablelinearclassificationproblem
classified point x with label y in the direction forthecasesofbothwhenthemodelisindependentof
of the gradient of the cost function, yielding (dashedlines) and, resp., dependenton (solidlines) the
x (cid:48) =x−εyw/(cid:107) w (cid:107) , whereε≥0 isthestrength testset.
2
oftheattack. Tocomplywiththerequirements
foran AEG, wedefinegasfollows: g (x)=x (cid:48) if L(f, x)=0 andf∗(x)=f∗(x (cid:48))(corresponding
to (G2) and (G1), respectively), whileg (x)=xotherwise. Therefore, ifx (cid:48) ismisclassifiedbyf, x
andx (cid:48) aretheonlypointsmappedtox (cid:48) byg. Thissimpleformofgandtheknowledgeofρallowsto
computethedensityh , makingiteasytocomputetheadversarialerrorestimate (2). Figure4 shows
g
theaveragep-valuesproducedbyour N-modelindependencetestforadependent (solidlines) and
anindependent (dashedlines) testset. Itcanbeseenthatinthedependentcasethetestcanreject
independencewithhighconfidenceforalargerangeofattackstrengthε, whiletheindependence
hypothesisisnotrejectedinthecaseoftrueindependence. Moredetails (includingwhyonlyarange
ofεissuitablefordetectingoverfitting) aregivenin Appendix A.
5 Testingoverfittingon Image Net
In the previous section we showed that the proposed adversarial-example-based dependence test
worksforasyntheticproblemwherethedensitiescanbecomputedexactly. Inthissectionweapply
ourestimatestoapopularimageclassificationbenchmark, Image Net[8];herethemainissueisto
findsufficientlystrong AEGsthatmakecomputingthecorrespondingdensitiespossible.
Tofacilitatethecomputationofthedensityh , weonlyconsiderdensity-preserving AEGsasdefined
g
by (G3)(recallthat (G3) isdifferentfromrequiringh =1). Sincein (2) and (5), h (x) ismultiplied
g g
by L(f, x), weonlyneedtodeterminethedensityh fordatapointsthataremisclassifiedbyf.
g
5.1 AEGsbasedontranslations
To satisfy (G3), we implement the AEG using translations of images, which have recently been
proposedasmeansofgeneratingadversarialexamples[1]. Althoughrelativelyweak, suchattacksfit
ourneedswell:unlesstheimagesareprocedurallycentered, itisreasonabletoassumethattranslating
thembyafewpixelsdoesnotchangetheirlikelihood.4 Wealsomakethenaturalassumptionthat
the small translations used do not change the true class of an image. Under these assumptions,
4 Notethatthisassumptionlimitstheapplicabilityofourmethod, excludingsuchcenteredoressentially
centeredimageclassificationbenchmarksas MNIST[20]or CIFAR-10[18].
7

#### 7

translationsbyafewpixelssatisfyconditions (G1) and (G3). Animage-translatingfunctiongisa
valid AEGifitleavesallmisclassifiedimagesinplace (tocomplywith (G2)), andeitherleavesa
correctlyclassifiedimageunchangedorappliesasmalltranslation.
Themainbenefitofusingatranslational AEGg (withboundedtranslations) isthatitsdensityh (x)
g
foranimagexcanbecalculatedexactlybyconsideringthesetofimagesx (cid:48) thatcanbemappedtox
byg (thisisduetoourassumption (G3)). Weconsideredmultiplewaysforconstructingtranslational
AEGs. Thebestversion (selectedbasedoninitialevaluationsonthe Image Nettrainingset), which
wecalledthestrongestperturbation, seeksanon-identicalneighborofacorrectlyclassifiedimage
x (neighboringimagesaretheonesthatareaccessiblethroughsmalltranslations) thatcausesthe
classifiertomakeanerrorwiththelargestconfidence.
Formally, wemodelimagesas3 Dtensorsin[0,1]W×H×C space, where C =3 for RGBdata, and
W and H arethewidthandheightoftheimages, respectively. Letτ (x) denotethetranslationof
v
animagexbyv ∈ Z2 pixelsinthe (X, Y) plane (here Zdenotesthesetofintegers). Tocontrol
the amount of change, we limit the magnitude of translations and allow v ∈ V = {u ∈ Z2 :
ε
u (cid:54)= (0,0),(cid:107) u (cid:107) ≤ ε} only, for some fixed positive ε. Thus, we considers AEGs in the form
∞
g (x)∈{τ (x):v ∈V}∪{x}iff (x)=f∗(x) andg (x)=xotherwise (ifxiscorrectlyclassified,
v
weattempttotranslateittofindanadversarialexamplein{τ (x):v ∈V}whichismisclassifiedby
v
f, butxisleftunchangedifnosuchpointexists). Denotingthedensityofthepushforwardmeasure
P byρ , foranymisclassifiedpointx,
g g
(cid:32) (cid:33)
(cid:88) (cid:88)
ρ (x)=ρ(x)+ ρ(τ (x))I(g (τ (x))=x)=ρ(x) 1+ I(g (τ (x))=x)
g −v −v −v
v∈V v∈V
wherethesecondequalityfollowsfrom (G3). Therefore, thecorrespondingdensityis
h (x)=1/(1+n (x)) (8)
g
wheren (x)= (cid:80) I(g (τ (x))=x) isthenumberofneighboringimageswhicharemappedto
v∈V −v
xbyg. Notethatgivenf andg, n (x) canbeeasilycalculatedbycheckingallpossibletranslations
ofxby−vforv ∈V. Itiseasytoextendtheabovetonon-deterministicperturbations, definedas
distributionsover AEGs, byreplacingtheindicatorwithitsexpectation P(g (τ (x))=x|x, v) with
−v
respecttotherandomnessofg, yielding
1
h (x)= . (9)
g 1+ (cid:80) P(g (τ (x))=x|x, v)
v∈V −v
Ifgisdeterministic, wehaveh (x)≤1/2 foranysuccessfuladversarialexamplex. Hence, forsuch
g
g, therange U oftherandomvariables T definedin (5) hasatighterupperboundof3/2 instead2(as
i
T ∈[−1,1/2]), leadingtoatighterboundin (6) andastrongerpairwiseindependencetest. Inthe
i
experiments, weusethisstrongertest. Weprovideadditionaldetailsaboutthetranslational AEGs
usedin Appendix B.
5.2 Testsof Image Netmodels
Weappliedourtesttocheckifstate-of-the-art classifiers forthe Image Netdataset[8]havebeen
overfittedtothetestset. Inparticular, weusethe VGG16 classifierof[27]andthe Resnet50 classifier
of [16]. Due to computational considerations, we only analyzed a single trained VGG16 model,
whilethe Resnet50 modelwasretrained120 times. Themodelsweretrainedusingtheparameters
recommendedbytheirrespectiveauthors.
Thepreprocessingprocedureofbotharchitecturesinvolvesrescalingeveryimagesothatthesmaller
ofwidthandheightis256 andnextcroppingcentrallytosize224×224. Thismeansthattranslating
theimagebyvcanbetriviallyimplementedbyshiftingthecroppingwindowby−vwithoutanyloss
ofinformationfor (cid:107) v (cid:107) ≤16, becausewehaveenoughextrapixelsoutsidetheoriginal, centrally
∞
locatedcroppingwindow. Thisimpliesthatwecancomputethedensitiesofthetranslational AEGs
forany (cid:107) v (cid:107) ≤ε=(cid:98)16/3(cid:99)=5(see Appendix B.1 fordetailedexplanation).Becausethe Image Net
∞
datacollectionproceduredidnotimposeanystrictrequirementsoncenteringtheimages[8], itis
reasonable to assume (as we do) that small (lossless) translations respect the density-preserving
condition (G3).
Inourfirstexperiment, weappliedourpairwiseindependencetest (6) withthe AEGsdescribedin
Appendix B(strongest, nearest, andthetworandombaselines) toall1,271,167 trainingexamples, as
8

#### 8

1.0
0.8
0.6
0.4
0.2
0.0
0.2 0.4 0.6 0.8 1.0 1.2
sample size 106
×
eulav-P
0.188
0.186
RS(f)
Rcg (f)
variant 0.184
c
strongest 0.182
nearest
random 0.180
random2 0.178
0.176
0.174
0.2 0.4 0.6 0.8 1.0 1.2
sample size 106
×
Figure5:p-valuesfortheindependencetestonthe Image Nettrainingsetfordifferentsamplesizesand AEG
variants (left);originalandadversarialriskestimates, R(cid:98)S (f) and R(cid:98) g (f), onthe Image Nettrainingsetwith
97.5%two-sidedconfidenceintervalsforthe‘strongestattack’AEG(right).
wellastoanumberofitsrandomlyselected (uniformlywithoutreplacement) subsetsofdifferent
sizes. Besidesthisbeingasanitycheck, wealsousedthisexperimenttoselectfromdifferent AEGs
andcomparetheperformanceofthepairwiseindependencetest (6) tothebasicversionofthetest
describedin Section3.1.
The left graph in Figure 5 shows that with the “strongest perturbation”, we were able to reject
independenceofthetrainedmodelandthetrainingsamplesataconfidencelevelverycloseto1 when
enoughtrainingsamplesareconsidered (tobeprecise, forthewholetrainingsettheconfidencelevel
is99.9994%). Note, however, thatthemuchweaker“smallestperturbation”AEG, aswellasthe
randomtransformations, arenotabletodetectthepresenceofoverfitting. Atthesametime, thegraph
ontherighthandsideshowstherelativestrengthofthepairwiseindependencetestcomparedtothe
basicversionbasedonindependentconfidenceintervalestimatesasdescribedindetailin Section3.1:
the 97.5%-confidence intervals of the error estimates R(cid:98)S (f) and R(cid:98) g (f) overlap, not allowing to
rejectindependenceataconfidencelevelof95%(notethathere S denotesthetrainingset).
Ontheotherhand, whenappliedtothetestset, weobtainedap-valueof0.96, notallowingatall
torejecttheindependenceofthetrainedmodelandthetestset. Thisresultcouldbeexplainedby
thetestbeingtooweak, asnooverfittingisdetectedtothetrainingsetatsimilarsamplesizes (see
Figure5), orsimplythelackofoverfitting. Similarresultswereobtainedfor Resnet50, whereeven
the N-modeltestwith N =120 independentlytrainedmodelsresultedapvalueof1, notallowing
torejectindependenceatanyconfidencelevel. Theviewofnooverfittingcanbebackedupinat
leasttwoways: first,“manual”overfittingtotherelativelylarge Image Nettestsetishard. Second,
sincetrainingan Image Netmodelwasjusttoocomputationallyexpensiveuntilquiterecently, onlya
relativelysmallnumberofdifferentarchitecturesweredevelopedforthisproblem, andtheevolution
oftheirdesignwasoftendrivenbycomputationalefficiencyontheavailablehardware. Ontheother
hand, itisalsopossiblethatincreasing N sufficientlymightshowevidenceofoverfitting (thisisleft
forfuturework).
6 Conclusions
Wepresentedamethodfordetectingoverfittingofmodelstodatasets. Itreliesonanimportance-
weightedriskestimatefromanewdatasetobtainedbygeneratingadversarialexamplesfromthe
originaldatapoints. Weappliedourmethodtothepopular Image Netimageclassificationtask. For
thispurpose, wedevelopedaspecializedvariantofourmethodforimageclassificationthatuses
adversarialtranslations, providingargumentsforitscorrectness. Luckily, andinagreementwithother
recentworkonthistopic[25,26,13,21,32], wefoundnoevidenceofoverfittingofstate-of-the-art
classifierstothe Image Nettestset.
Themostchallengingaspectofourmethodsistoconstructadversarialperturbationsforwhichwecan
calculatetheimportanceweights;findingstrongerperturbationsthantheonesbasedontranslations
forimageclassificationisanimportantquestionforthefuture. Anotherinterestingresearchdirection
istoconsiderextensionsbeyondimageclassification, forexample, bybuildingonrecentadversarial
attacksforspeech-to-textmethods[5], machinetranslation[11]ortextclassification[12].
9

#### 9

Acknowledgements
Wethank J.Uesatoforusefuldiscussionsandadviceaboutadversarialattackmethodsandsharing
theirimplementations[30]withus, aswellas M.Roscaand S.Gowalforhelpwithretrainingimage
classificationmodels. Wealsothank B.O’Donoghueforusefulremarksaboutthemanuscript, and L.
Schmidtforanin-depthdiscussionoftheirresultsonthistopic. Finally, wethank D.Balduzzi, S.
Legg, K.Kavukcuogluand J.Martensforencouragement, support, livelydiscussionsandfeedback.
References
[1] Aharon Azulayand Yair Weiss. Whydodeepconvolutionalnetworksgeneralizesopoorlytosmallimage
transformations? 2018. ar Xiv:1805.12177.
[2] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neuralmachinetranslationbyjointlylearning
toalignandtranslate. In Proceedingsofthe International Conferenceon Learning Representations (ICLR),
2015.
[3] Stéphane Boucheron, Gábor Lugosi, and Pascal Massart. Concentration Inequalities:ANonasymptotic
Theoryof Independence. Oxford University Press,2013.
[4] James Antonio Bucklew. Introductionto Rare Event Simulation. Springer New York,2004.
[5] N.Carliniand D.Wagner. Audioadversarialexamples:Targetedattacksonspeech-to-text. In2018 IEEE
Securityand Privacy Workshops (SPW), May2018.
[6] Nicholas Carliniand David A.Wagner. Adversarialexamplesarenoteasilydetected: Bypassingten
detectionmethods. In Proceedingsofthe10 th ACMWorkshopon Artificial Intelligenceand Security,
AISec@CCS2017, Dallas, TX, USA, November3,2017, pages3–14,2017. URLhttp://doi.acm.org/
10.1145/3128572.3140444.
[7] Nicholas Carliniand David A.Wagner. Towardsevaluatingtherobustnessofneuralnetworks. In2017
IEEESymposiumon Securityand Privacy, SP2017, San Jose, CA, USA, May22-26,2017, pages39–57,
2017. URLhttps://doi.org/10.1109/SP.2017.49.
[8] J.Deng, W.Dong, R.Socher, L.J.Li, Kai Li, and Li Fei-Fei. Image Net:Alarge-scalehierarchicalimage
database. In2009 IEEEConferenceon Computer Visionand Pattern Recognition, pages248–255, June
2009. doi:10.1109/CVPR.2009.5206848.
[9] L.Deng, G.Hinton, and B.Kingsbury. Newtypesofdeepneuralnetworklearningforspeechrecognition
andrelatedapplications:anoverview. In2013 IEEEInternational Conferenceon Acoustics, Speechand
Signal Processing, pages8599–8603.IEEE, May2013.
[10] Cynthia Dwork, Vitaly Feldman, Moritz Hardt, Toniann Pitassi, Omer Reingold, and Aaron Roth. The
reusableholdout:Preservingvalidityinadaptivedataanalysis. Science,349(6248):636–638,2015.
[11] Javid Ebrahimi, Daniel Lowd, and Dejing Dou. Onadversarialexamplesforcharacter-levelneuralmachine
translation. In Proceedingsofthe27 th International Conferenceon Computational Linguistics, COLING
2018, Santa Fe, New Mexico, USA, August20-26,2018, pages653–663,2018.
[12] Javid Ebrahimi, Anyi Rao, Daniel Lowd, and Dejing Dou. Hotflip:White-boxadversarialexamplesfortext
classification. In Proceedingsofthe56 th Annual Meetingofthe Associationfor Computational Linguistics,
ACL2018, Melbourne, Australia, July15-20,2018, Volume2:Short Papers, pages31–36,2018.
[13] Vitaly Feldman, Roy Frostig, and Moritz Hardt. Theadvantagesofmultipleclassesforreducingoverfitting
fromtestsetreuse. In Proceedingsofthe36 th International Conferenceon Machine Learning, pages
1892–1900,2019.
[14] I. J. Goodfellow, J. Shlens, and C. Szegedy. Explaining and harnessing adversarial examples. In
Proceedingsofthe International Conferenceon Learning Representations (ICLR),2015.
[15] Alex Graves, Abdel-rahman Mohamed, and Geoffrey Hinton. Speechrecognitionwithdeeprecurrent
neuralnetworks. In2013 IEEEInternational Conferenceon Acoustics, Speechand Signal Processing,
pages6645–6649.IEEE,2013.
[16] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deepresiduallearningforimagerecognition.
In Proceedingsofthe IEEEConferenceon Computer Visionand Pattern Recognition, pages770–778,
2016.
10

#### 10

[17] H.Kahnand T.E.Harris. Estimationofparticletransmissionbyrandomsampling. In Monte Carlo
Method, volume12 of Applied Mathematics Series, pages27–30.National Bureauof Standards,1951.
[18] Alex Krizhevsky. Learningmultiplelayersoffeaturesfromtinyimages. Technicalreport, Universityof
Toronto,2009.
[19] Alex Krizhevsky, Ilya Sutskever, and Geoffrey EHinton. Image Netclassificationwithdeepconvolutional
neuralnetworks. In F.Pereira, C.J.C.Burges, L.Bottou, and K.Q.Weinberger, editors, Advancesin
Neural Information Processing Systems25, pages1097–1105.Curran Associates, Inc.,2012.
[20] Yann Le Cunand Corinna Cortes. MNISThandwrittendigitdatabase. http://yann.lecun.com/exdb/mnist/,
2010.
[21] Horia Mania, John Miller, Ludwig Schmidt, Moritz Hardt, and Benjamin Recht. Modelsimilaritymitigates
testsetoveruse. 2019. ar Xiv:1905.12580.
[22] Volodymyr Mnih, Csaba Szepesvári, and Jean-Yves Audibert. Empirical Bernsteinstopping. In Proceed-
ingsofthe25 th International Conferenceon Machine Learning, ICML’08, pages672–679, New York,
NY, USA,2008.ACM.
[23] Nicolas Papernot, Patrick D.Mc Daniel, and Ian J.Goodfellow. Transferabilityinmachinelearning:from
phenomenatoblack-boxattacksusingadversarialsamples. 2016. ar Xiv:1605.07277.
[24] Nicolas Papernot, Patrick Mc Daniel, Ian Goodfellow, Somesh Jha, ZBerkay Celik, and Ananthram Swami.
Practicalblack-boxattacksagainstmachinelearning. In Proceedingsofthe2017 ACMon Asia Conference
on Computerand Communications Security, pages506–519.ACM,2017.
[25] Benjamin Recht, Rebecca Roelofs, Ludwig Schmidt, and Vaishaal Shankar. Do CIFAR-10 classifiers
generalizeto CIFAR-10? 2018. ar Xiv:1806.00451.
[26] Benjamin Recht, Rebecca Roelofs, Ludwig Schmidt, and Vaishaal Shankar. Do Image Netclassifiers
generalizeto Image Net? 2019. ar Xiv:1902.10811.
[27] K.Simonyanand A.Zisserman. Verydeepconvolutionalnetworksforlarge-scaleimagerecognition. In
Proceedingsofthe International Conferenceon Learning Representations (ICLR),2015.
[28] Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru
Erhan, Vincent Vanhoucke, and Andrew Rabinovich. Goingdeeperwithconvolutions. In2015 IEEE
Conferenceon Computer Visionand Pattern Recognition (CVPR),2015. ar Xiv:1409.4842.
[29] T.Tielemanand G.Hinton. Lecture6.5—Rms Prop:Dividethegradientbyarunningaverageofitsrecent
magnitude. COURSERA:Neural Networksfor Machine Learning,2012.
[30] Jonathan Uesato, Brendan O’Donoghue, Aäronvanden Oord, and Pushmeet Kohli. Adversarialriskand
thedangersofevaluatingagainstweakattacks. In Proceedingsofthe35 th International Conferenceon
Machine Learning, volume80 of Proceedingsof Machine Learning Research, pages5025–5034,2018.
ar Xiv:1802.05666.
[31] Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V.Le, Mohammad Norouzi, Wolfgang Macherey, Maxim
Krikun, Yuan Cao, Qin Gao, Klaus Macherey, Jeff Klingner, Apurva Shah, Melvin Johnson, Xiaobing
Liu, Lukasz Kaiser, Stephan Gouws, Yoshikiyo Kato, Taku Kudo, Hideto Kazawa, Keith Stevens, George
Kurian, Nishant Patil, Wei Wang, Cliff Young, Jason Smith, Jason Riesa, Alex Rudnick, Oriol Vinyals,
Greg Corrado, Macduff Hughes, and Jeffrey Dean. Google’sneuralmachinetranslationsystem:Bridging
thegapbetweenhumanandmachinetranslation. 2016. ar Xiv:1609.08144.
[32] Chhavi Yadavand Léon Bottou. Coldcase:Thelost MNISTdigits. May2019. ar Xiv:1905.10498.
11

#### 11

A Syntheticexperiments
Inthissection, wepresentfulldetailsoftheexperimentsonasimplesyntheticclassificationproblem,
whichwepresentedbrieflyin Section4. Theseexperimentsillustratethepowerofthemethodof
Section3. Theadvantageofthesimplesetupconsideredhereisthatweareabletocomputethe
densityh inananalyticform (see Figure6 foranillustration).
g
A.1 Datadistributionandmodel
Let X = R500 and consider an input distribution with a density ρ that is an equally weighted
mixture of two 500-dimensional isotropic truncated Gaussian distributions Ntrunc (µ ,σ2 I) with
√ ± ±
coordinate-wise standard deviation σ = 500 (I denotes the identity matrix of size 500×500),
meansµ =[±1,0,0,...,0]anddensitiesρ truncatedinthefirstdimensionsuchthatρ (x)=0
± ± +
ifx ≤ 0.025 andρ (x) = 0 ifx ≥ −0.025. Thelabelofaninputpointxisf∗(x) = sgn (x ),
1 − 1 1
whichisthesignofitsfirstcoordinate.
Weconsiderlinearclassifiersoftheformf (x)=sgn (w (cid:62) x+b) trainedwiththecross-entropyloss
c ((w, b), x, y) = ln (1+e−y (w (cid:62) x+b)) wherey = f∗(x). Weemployaone-stepgradientmethod
(whichisan L versionofthefastgradient-signmethodof[14,23]) todefineour AEGg, which
2
triestomodifyacorrectlyclassifiedpointxwithlabelyinthedirectionofthegradientofthecost
function c: x (cid:48) = x+ε∇ c ((w, b), x, y)/(cid:107)∇ c ((w, b), x, y)(cid:107) for some ε > 0. For our specific
x x 2
choiceofc, theabovesimplifiestox (cid:48) =x−εyw/(cid:107) w (cid:107) . Tocomplywiththerequirementsforan
2
AEG, we define g as follows: g (x) = x (cid:48) if L(f, x) = 0 and f∗(x) = f∗(x (cid:48)) (corresponding to
(G2) and (G1), respectively), whileg (x) = xotherwise. Therefore, ifx (cid:48) ismisclassifiedbyf, x
x
2
y = −1 y = +1
x (cid:48)
x A
C x (cid:48)
B x A
x (cid:48)
C 0 x B x 1
x (cid:48) w
D
x
D
x (cid:48)
E
x
E
Figure6:Illustrationofthedatadistributionandthelinearmodelf (x)=sgn (w (cid:62) x+b) intwodimensions.
Theblueandgreengradientsshowtheprobabilitydensityρofthedatawithtruelabelsy = −1 andy = 1,
respectively, while the white space between them is the margin with ρ = 0. The red line is the model’s
classificationboundarywithitsparametervectorwshownbythepurplearrow. Dependingonthelabel, wor
−wisthedirectionoftranslationusedtoperturbedthecorrectlyclassifieddatapoints, andthetranslations
used by the AEG g for specific points are depicted by grey arrows: solid arrows indicate the cases where
g (x) = x (cid:48) (cid:54)= x, whiledashedarrowsareforcandidatetranslationswhicharenotperformedbythe AEG
becausetheywouldchangethetruelabel, f∗(x (cid:48))(cid:54)=f∗(x), andhenceg (x)=x (cid:54)=x (cid:48).Eachoriginal/perturbed
datapointisrepresentedbyacolor-codedcircle: theinnercolorcorrespondstothetruelabel (darkbluefor
y =−1 anddarkgreenfory =1) whiletheoutercolortothemodel’sprediction (darkblueforf (x)=−1
anddarkgreenforf (x)=1).Pointsx (cid:48) andx (cid:48) canbeobtainedfromx andx , respectively, byapplying
A B A B
the AEG, x (cid:48) =g (x ) andx (cid:48) =g (x ).Sinceonlyx ismappedtox (cid:48) byg, andg (x (cid:48) )=x (cid:48) , thedensity
A A B B A A A A
(Radon-Nikodymderivative) canbeobtainedash (x (cid:48) )=ρ(x (cid:48) )/(ρ(x )+ρ(x (cid:48) ))∈(0,1).Inthecaseof
g A A A A
x (cid:48) , h (x (cid:48) )=ρ(x (cid:48) )/(ρ(x )+ρ(x (cid:48) ))=0 duetothemargin. Notethattheformulaforh (x (cid:48) ) doesnot
B g B B B B g A
dependonwhetherx orx (cid:48) isintheoriginaltestset S;inthefirstcasewecallx (cid:48) a“successfuladversarial
A A A
example”whileinthesecondcasex (cid:48) iscalled“originallymisclassified”(asimilarargumentholdsforh (x (cid:48) )).
A g B
x (cid:48) is not a successful adversarial example since L(f, x (cid:48) ) = 0 (however, g (x ) = x (cid:48) according to our
C C C C
definition).Pointsx andx arenotperturbedbyour AEG, sincef∗(x )(cid:54)=f∗(x (cid:48) ) andf∗(x )(cid:54)=f∗(x (cid:48) ).
D E D D E E
12

#### 12

andx (cid:48) aretheonlypointsmappedtox (cid:48) byg. Thus, thedensityatx (cid:48) afterthetransformationg is
ρ(cid:48)(x (cid:48))=ρ(x)+ρ(x (cid:48))(1−L(f, x))I(f∗(x)=f∗(x (cid:48))) and
ρ(x (cid:48)) ρ(x (cid:48))
h (x (cid:48))= =
g ρ(cid:48)(x (cid:48)) ρ(x (cid:48))+ρ(x)(1−L(f, x))I(f∗(x)=f∗(x (cid:48)))
(notethat I(L(f, x)=0)=1−L(f, x)).
A.2 Experimentsetup
Wepresenttwoexperimentsshowingthebehaviorofourindependencetest: onewherethetraining
andtestsetsareindependent, andanotherwheretheyarenot.
Inthefirstexperimentalinearclassifierwastrainedonatrainingset S ofsize500 for50,000 steps
Tr
usingthe RMSPropoptimizer[29]withbatchsize100 andlearningrate0.01, obtainingzero (upto
numericalprecision) finaltraininglosscand, consequently,100%predictionaccuracyonthetraining
data. Thenthetrainedclassifierwastestedonalargetestset S ofsize10,000.5 Bothsetswere
Te
drawnindependentlyfromρdefinedabove. Weusedarangeofεvaluesmatchedtothescaleofthe
datadistribution: from10−2, whichistheorderofmagnitudeofthemarginbetweentwoclasses
(0.05), to102, whichistheorderofmagnitudeofthewidthofthe Gaussiandistributionusedforeach
√
classes (σ = 500).
Inthesecondexperimentweconsiderthesituationwherethetrainingandtestsetsarenotindependent.
Toenhancetheeffectsofthisdependence, thesetupwasmodifiedtomakethetrainingprocessmore
amenabletooverfittingbysimulatingasituationwhenthemodelhasawrongbias (thismayhappenin
practiceifawrongarchitectureordatapreprocessingmethodischosen, which, despitethemodeler’s
bestintentions, worsenstheperformance). Specifically, duringtrainingweaddedapenaltyterm
104 w2 tothetraininglossc, decreasedthesizeofthetestsetto1000 andused50%ofthetestdata
1
fortraining (thefinalpenalizedtraininglosswas0.25 with100%predictionaccuracyonthetraining
set). Notethatthesmalltrainingsetandthelargepenaltyonw yieldclassifiersthatareessentially
1
independentoftheonlyinterestingfeaturex (recallthatthetruelabelofapointxissgn (x )) and
1 1
overfittothenoiseinthedata, resultinginatruemodelrisk R(f)≈1/2.
A.3 Results
The results of the two experiments are shown in Figure 7, plotted against different perturbation
strengths: theleftcolumncorrespondstothefirstexperimentwhiletherightcolumntothesecond.
Thefirstrowpresentsthep-valuesforrejectingtheindependencehypothesis, calculatedbyrepeating
theexperiment (samplingdataandtrainingtheclassifier)100 timesandapplyingthesingle-model
(Section3, labelledas N =1 intheplots) and N-model (Section3.3, labelledas N =2,10,25,100
intheplots) independencetest, andtakingtheaverageovermodels (ormodelsetsofsize N) foreach
ε. Wealsoplotempirical95%two-sidedconfidenceintervals (N ≤2) or, duetolimitednumberof
p-valuesavailableafterdividing100 runsintodisjointbinsofsize N ≥10, rangesbetweenminimum
andmaximumvalue (N = 10,25). Forallmethodsofdetectingdependence, itcanbeseenthat
for the independent case the test is correctly not able to reject the independence hypothesis (the
averagep-valueisverycloseto1, althoughinsomerunsitcandroptoaslowas0.5). Ontheother
hand, for10≤ε≤50, thenon-independentmodelfailedtheindependencetestatconfidencelevel
1−δ ≈100%, hence, inthisrangeofεourindependencetestreliablydetectsoverfitting.
Infact, itiseasytoarguethatourtestshouldonlyworkforalimitedrangeofε, thatis, itshould
notrejectindependencefortoosmallortoolargevaluesofε. Firstweconsiderthecaseofsmall
εvalues. Noticethatexceptforpointsg (x)ε-close (in L -norm) tothetruedecisionboundaryor
2
thedecisionboundaryoff, g (x) isinvertible: ifg (x) iscorrectlyclassifiedandisε-awayfromthe
truedecisionboundary, thereisexactlyonepoint, x, whichistranslatedtog (x), whileifg (x) is
incorrectlyclassifiedandε-awayfromthedecisionboundaryoff, notranslationleadstog (x) and
x=g (x);anyotherpointsareε-closetothedecisionboundaryofeitherf orf∗. Thus, sinceρis
bounded, g (x) isinvertibleonasetofatleast1−O(ε) probability (accordingtoρ). Whenε→0,
g (x) → x, andsoρ(g (x)) → ρ(x) forallpointsxwith|x | =(cid:54) 0.025(sinceρiscontinuousinall
1
suchx), implyingh (g (x))≈1 onthesepoints. Italsofollowsthat L(f, x)(cid:54)=L(f, g (x)) canonly
g
5 Thelargenumberoftestexamplesensuresthattherandomerrorintheempiricalerrorestimateisnegligible.
13

#### 13

1.0
0.8
0.6
0.4
0.2
0.0
10-2 10-1 100 101 102
†
eulav-p
1.0
N=1 0.8
N=2
0.6
N=10
N=25 0.4
N=100 0.2
0.0
10-2 10-1 100 101 102
†
eulav-p
N=1
N=2
N=10
N=25
N=100
1.0
0.8
0.6
0.4
0.2
0.0
10-2 10-1 100 101 102
†
rorre
1.0
Rg (f) 0.8
Rc S(f) 0.6
c 0.4
0.2
0.0
10-2 10-1 100 101 102
†
rorre
Rg (f)
Rc S(f)
Rc (f)
1.0
0.8
0.6
0.4
0.2
0.0
10-2 10-1 100 101 102
†
gh
egareva
1.0
0.8
0.6
0.4
original misclassified
adversarial 0.2
0.0
10-2 10-1 100 101 102
†
gh
egareva
original misclassified
adversarial
1.0
0.8
0.6
0.4
0.2
0.0
10-2 10-1 100 101 102
†
rorre
1.0
Rg (f) 0.8
Rc S(f) 0.6
Rc S0 (f) 0.4
c 0.2
0.0
10-2 10-1 100 101 102
†
rorre
Rg (f)
Rc S(f)
Rc S0 (f)
Rc (f)
100
80
60
40
20
0
0.0 0.2 0.4 0.6 0.8 1.0
p-value
tnuoc
100
δ(†=0.1) 80
δ(†=5) 60
δ(†=20)
40
20
0
0.0 0.2 0.4 0.6 0.8 1.0
p-value
tnuoc
δ(†=0.1)
δ(†=5)
δ(†=6)
δ(†=20)
Figure 7: Risk and overfitting metrics for a synthetic problem with linear classifiers as a function of the
perturbationstrengthsε(logscale). Left:unbiasedmodeltestedonalarge, independenttestset (inthiscase
R(cid:98)S (f) ≈ R(cid:98) g (f) ≈ R(f)); right: trainedmodeloverfittedtothetestset (R(cid:98)S (f) ≤ R(cid:98) g (f) whilebothare
smaller than R(f)). First row: Average p-value δ for the pairwise independence test with over 100 runs
(N =1) orthe N-modelindependencetest (N >1).Theboundsplottedareeitherempirical95%two-sided
confidenceintervals (N ≤ 2) orrangesbetweenminimumandmaximumvalue (N = 10,25). Secondrow:
Empiricaltwo-sided97.5%confidenceintervalsfortheempiricaltesterrorrate R(cid:98)S (f) andtheadversarialrisk
estimate R(cid:98) g (f).Ontheleft, R(f)≈R(cid:98)S (f), while R(f) isshownseparatelyontheright. Thirdrow:Average
densities (Radon-Nikodymderivatives) fororiginallymisclassifiedpointsandforthenewdatapointsobtained
bysuccessfuladversarialtransformations (withempirical97.5%two-sidedconfidenceintervals).Fourthrow:
Theempiricaltesterrorrate R(cid:98)S (f) andtheadversarialriskestimate R(cid:98) g (f) forasinglerealizationwith97.5%
two-sidedconfidenceintervalscomputedfrom Bernstein’sinequality, theadversarialerrorrate R(cid:98)S(cid:48)(f), andthe
expectederror R(f)(ontheright, ontheleft R(f)≈R(cid:98)S (f)).Fifthrow:Histogramsofp-valuesforselectedε
valuesover100 runs.
14

#### 14

happentoasetofpointswithan O(ε)ρ-probability. Thismeansthat L(f, g (x)) h (g (x))≈L(f, x)
g
onasetof1−O(ε)ρ-probability, andforthesepoints T (x)=L(f, g (x)) h (g (x))−L(f, x)≈0.
g g
Thus, T (X) ≈ 0 withρ-probability1−O(ε). Unlessthetestset S isconcentratedinlargepart
g
onthesetofremainingpointswith O(ε)ρ-probability, theteststatistic|T (f)|=O(ε) withhigh
S, g
probabilityandourmethodwillnotrejecttheindependencehypothesisforε→0.
When ε is large (ε → ∞), notice that for any point x with non-vanishing probability (i.e., with
ρ(x)>cforsomec>0), ifg (x)(cid:54)=xthanρ(g (x))≈0.Therefore, forsuchanx, if L(f, x)=0 and
L(f, g (x))=1, h (g (x))=ρ(g (x))/(ρ(x)+ρ(g (x)))≈0, andso T (x)≈0(if L(f, g (x))=0,
g g
we trivially have T (x) = 0). If L(f, x) = 1, we have g (x) = x. If g is invertible at x then
g
h (x) = 1 and T (x) = 0. If g is not invertible, then there is another x (cid:48) such that g (x (cid:48)) = x;
g g
however, ifρ(x)>cthenρ(x (cid:48))≈0(sinceεislarge), andsoh (g (x))=ρ(x)/(ρ(x)+ρ(x (cid:48)))≈1,
g
giving T (x) ≈ 0. Therefore, forlargeε, T (X) ≈ 0 withhighprobability (i.e., forpointswith
g g
ρ(x)>c), sotheindependencehypothesiswillnotberejectedwithhighprobability.
Tobetterunderstandthebehaviorofthetest, thesecondrowof Figure7 showstheempiricaltesterror
rate R(cid:98)S (f), the (unadjusted) adversarialerrorrate R(cid:98)S(cid:48) (f), andtheadversarialriskestimate R(cid:98) g (f),
togetherwiththeirconfidenceintervals. Forthenon-independentmodel, wealsoshowtheexpected
error R(f)(estimatedoveralargeindependenttestset), whileitisomittedfortheindependentmodel
whereitapproximatelycoincideswithboth R(cid:98)S (f) and R(cid:98) g (f).Whilethereweightedadversarialerror
estimate R(cid:98) g (f) remainsthesameforallperturbationsincaseofanindependenttestset (leftcolumn),
theadversarialerrorrate R(cid:98)S(cid:48) (f) variesalotforboththedependentandindependenttestsets. For
example, in the case when the test samples and the model f are not independent, it undershoots
the true error for ε < 10 and overshoots it for larger perturbations. For very large perturbations
(εcloseto100), thebehaviorof R(cid:98)S(cid:48) (f) dependsonthemodelf: intheindependentcase R(cid:98)S(cid:48) (f)
decreasesbackto R(cid:98)S (f) becausesuchlargeperturbationsincreasinglyoftenchangethetruelabel
oftheoriginalexample, solessandlessadversarialpointsaregenerated. Inthecasewhenthedata
andthemodelarenotindependent (rightcolumn), theadversarialperturbationsarealmostalways
successful (i.e., leadtoavalidadversarialexampleformostoriginallycorrectlyclassifiedpoints),
yieldinganadversarialerrorrateclosetooneforlargeenoughperturbations. Thisisbecausethe
decisionboundaryoff isalmostorthogonaltothetruedecisionboundary, andsotheadversarial
perturbationsareparallelwiththetrueboundary, almostneverchangingthetruelabelofapoint.
Theplotsofthedensities (Radon-Nikodymderivatives), giveninthethirdrowof Figure7, show
howthechangeintheirvaluescompensatetheincreaseoftheadversarialerrorrate R(cid:98)S(cid:48) (f): inthe
independentcase, theeffectiscompletelyeliminatedyieldinganunbiasedadversarialerrorestimate
R(cid:98) g (f), whichisessentiallyconstantoverthewholerangeofε(asshowninthefirstrow), whilein
thenon-independentcasethesimilardensitiesdonotbringbacktheadversarialerrorrate R(cid:98)S(cid:48) (f) to
thetesterrorrate R(cid:98)S (f), allowingthetesttodetectoverfitting. Notethatthedensitiesexhibitsimilar
trends (andvalues) inbothcases, drivenbythedependenceoftypicalvaluesoftheρ(x)/ρ(g (x))
ratioontheperturbationstrengthεfororiginallymisclassifedpoints (L(f, x)=1) andforsuccessful
adversarialexamples (i.e., L(f, x)=0 and L(f, g (x))=1).
To compare the behavior of our improved, pairwise test and the basic version, the fourth row of
Figure7 depictsasinglerealizationoftheexperimentswherethe97.5%confidenceintervals (as
computedfrom Bernstein’sinequality) areshownfortheestimates. Fortheindependentcase, the
confidenceintervalsof R(cid:98)S (f) and R(cid:98) g (f) overlapforallε, andthusthebasictestisnotabletodetect
overfitting. Inthenon-independentcase, theconfidenceintervalsoverlapforε=10 andε=75, thus
thebasictestisnotabletodetectoverfittingwithata95%confidencelevel, whiletheimprovedtest
(secondrow) isabletorejecttheindependencehypothesisfortheseεvaluesatthesameconfidence
level.
Finally, inthefifthrowof Figure7 weplottedthehistogramsoftheempiricaldistributionofp-values
forbothmodels, over100 independentruns (betweentheruns, allthedatawasregeneratedandthe
modelswereretrained). Forε=0.1,5,20, theyconcentrateheavilyoneitherδ =0 orδ =1, and
haveverythintailsextendingfartowardstheoppositeendofthe[0,1]interval. Thisexplainsthe
surprisinglywide95%confidenceintervalsforp-valuesplottedinthefirstrow. Inparticular, thefact
thatsomep-valuesfortheindependentmodelareaslowas0.5 doesnotmeantheindependencetest
isnotreliable, becausealmostallcalculatedδvaluesarecloseorequalto1, andthefewoutliersare
15

#### 15

12
10
8
6
4
2
0
0.0 0.2 0.4 0.6 0.8 1.0
p-value
FDP
5
δ(N=1) 4
δ(N=2)
3
δ(N=10)
2
1
0
0.0 0.2 0.4 0.6 0.8 1.0
p-value
FDP
δ(N=1)
δ(N=2)
δ(N=10)
independentmodel,ε=10 non-independentmodel,ε=6
Figure8:Histogramsofp-valuesfrom N-model (N =1,2,10) independencetestsforbothsyntheticmodels
andselectedεvalues, over100 runs.
acombinedconsequenceofthefinitesamplesizeandtheeffectivenessofthe AEG.Theadditional
ε=6 histogramforthenon-independentmodelillustratesaregimewhichisinbetweenthesingle-
modelpairwisetest (Section3) completelyfailingtorejecttheindependencehypothesisandclearly
rejectingit.
Toverifyexperimentallywhetherthe N-modelindependencetestcanbeamorepowerfuldetectorof
overfittingthanthesingle-modelversion, in Figure8(rightpanel) weplottedp-valuehistogramsfor
N =1,2,10 fortheintermediate AEGstrengthε=6 appliedtothenon-independentmodelover100
trainingruns. Indeed, as N increases, theconcentrationofp-valuesaroundinthelow (δ ≤0.2) range
increases. For N >10 wedidnothaveenoughvaluestoplotahistogram: for N =25 weobtained
δ =0.1851,0.1599,0.0661 and0.1941, whilefor N =100 thep-valueis0.1153. Theincreaseof
thetestpowerbecomesapparentwhenwecomparethelastvaluewiththemeanofp-valuesobtained
bytestingeverytrainingrunseparately, equal0.5984, andthemedian0.6385.
For comparison, we also plotted in Figure 8 (left panel) the corresponding histograms for the
independentmodelandaslightlyhigherattackstrength,ε=10, atwhichtheindependencetestsfails
fortheoverfittedmodelevenwithoutaveraging (see Figure7, firstrow, rightpanel). Thehistograms
areallclusteredintheδregioncloseto1, indicatingthatthe N-modeltestisnotoverlypessimistic.
B Translational AEGsforimageclassificationmodels
Forimageclassificationweconsidertwotranslationvariantsthatareusedinconstructingatransla-
tional AEG.Foreverycorrectlyclassifiedimagex, weconsidertranslationsfrom V (forsomeε),
ε
choosingg (x) fromtheset G(x) = {τ (x) : v ∈ V }∪{x}. Ifalltranslationsresultincorrectly
v ε
classifiedexamples, wesetg (x) = x. Otherwise, weuseoneoftwopossiblewaystoselectg (x)
(andwecalltheresultingpointssuccessfuladversarialexamples):
• Strongest perturbation: Assuming the number of classes is K, let l (f, x) ∈ RK denote
thevectorofthe K classlogitscalculatedbythemodelf forimagex, andletl (f, x)=
exc
max l (f, x)−l (f, x). Wedefine
0≤i<K i y
g (x)=argmax l (f, x (cid:48)),
strongest x (cid:48)∈G(x) exc
withtiesbrokendeterministicallybychoosingthefirsttranslationfromthecandidateset,
goingtoptobottomandlefttorightinrow-majororder. Thus, hereweseekanon-identical
“neighbor”thatcausestheclassifiertoerrthemost, reachablefromxbytranslationswithin
amaximumrangeε.
• Nearest misclassified neighbor: Here we aim to find the nearest image in G(x) that is
misclassified. Thatis, lettingd (x, x (cid:48))=(cid:107) v (cid:107) ifx (cid:48) =τ (x) and∞otherwise, wedefine
2 v
g (x):=argmin d (x, x (cid:48))
nearest x (cid:48)∈G(x), L(f, x (cid:48))=1
withtiesbrokendeterministicallyasabove.
Thetwoperturbationvariantsaresuccessfulonexactlythesamesetofimages, hencetheyleadto
thesameadversarialerrorrates R(cid:98)S(cid:48) (f). However, theyarecharacterizedbydifferentvaluesofthe
densityh
g
and, consequently, yielddifferentadversarialriskestimates R(cid:98) g (f) andassociatedp-values
16

#### 16

fortheindependencetest. Themaindifferencebetweenthemisthatthe“strongest”versionismore
likely to map multiple images to the same adversarial example, thus decreasing the densities for
successfuladversarialexamplesand, counterintuitively, increasingthemfororiginallymisclassified
points (astheirneighborsarelesslikelytobemappedtothesepoints).
Tobetterseetheeffectofadversarialperturbations, wealsoconsidertworandombaselinesthatdo
nottakeintoaccountthesuccessofatranslationingeneratingmisclassifiedpoints: g (x) is
random
chosenuniformlyatrandomfrom G(x)\{x}, andg (x) ischosenuniformlyatrandomfrom
random2
G(x).
B.1 Maximumtranslations
In practice, translating an image is not always simple, as the new image has to be padded with
newpixels. When (central) cropsofalargerimageareused (asistypicalfor Image Netclassifiers),
translationscaneasilybeimplementedaslongastheresultingnewcroppingwindowstayswithinthe
originalimageboundaries. Evenifanimagecanbetranslatedbyavectorv, thislimitsourabilityto
computeh (x (cid:48)) fortheadversarialimagex (cid:48) by (8) or (9) forg org . Indeed, ifanimage
g strongest nearest
xisshiftedbyv ∈V togenerateadversarialexamplex (cid:48), weneedtoexaminetranslationsofx (cid:48) with
ε
vectorsin V tofindtheneighborsx (cid:48)(cid:48) ofx (cid:48) potentiallycontributington (x (cid:48)) whencomputingh (x (cid:48)).
ε g
Finallyweneedtoconsidertranslationsofx (cid:48)(cid:48) withvectorsin V todeterminetheexactvaluethey
ε
contribute, thatis, tocomputetheexactprobabilitiesin (9)(see Figure9 foranillustration). Thus,
tobeabletocomputethedensityh fortheadversarialpointsobtainedbytranslationsfrom V , we
g ε
mightneedtobeabletoperformtranslationswithin V .
3ε
ε
x x' x''
Figure9:Imagetranslationswhichneedtobeconsideredforatranslational AEGwithε=3.Thered, blueand
greenballsrepresentthecenteroftheoriginalimagex, adversarialexamplex (cid:48) =g (x) andanotherimagex (cid:48)(cid:48)
contributingtoρ (x (cid:48)), respectively, whilethesemi-translucentsquaresofcorrespondingcolorsrepresentthe
g
possibletranslationswhichneedtobeconsideredforeachofx, x (cid:48) andx (cid:48)(cid:48).Solidlightgreyarrowsrepresentthe
relationshipsx (cid:48) =g (x) andx (cid:48) =g (x (cid:48)(cid:48)).Finally, thedashedarrowandthesemi-translucentgreyballrepresent
analternativemapping, whichhastoberuledoutwhilecalculatingthevalueofg (x (cid:48)(cid:48)) and, consequently, of
h (x (cid:48)).Itiseasytoseethatthecoloredsquares (whichcontainthetranslationsneedingtobeevaluated) extend
g
asfaras3εfromtheoriginalimagex.
17

### Additional Content

#### 1

sample S(cid:48) = {(X(cid:48), Y ),...,(X(cid:48) , Y )} obtained through an AEG g, where X(cid:48) = g (X ) with
1 1 m m i i
X ,..., X drawnindependentlyfrom P and Y =f∗(X ). Sincegsatisfies (G1) bydefinition, the
1 m i i
originalexample X andthecorrespondingadversarialexample X(cid:48) havethesamelabel Y . Recalling
i i i
thath =d P/d P ≤1 on E ={x∈X : L(f, x)=1}, onecaneasilyshowthattheimportance
g g
weightedadversarialestimate
m
1 (cid:88)
R(cid:98) g (f)=
m
I(f (X
i
(cid:48))(cid:54)=Y
i
) h
g
(X
i
(cid:48)) (2)
i=1
obtainedfrom (1) fortheadversarialsample S(cid:48) hassmallervariancethanthatoftheempiricalaverage
R(cid:98)S (f), whilebothareunbiasedestimatesof R(f). Recallthatboth R(cid:98) g (f) and R(cid:98)S (f) areunbiased
estimatesof R(f) withexpectation E[R(cid:98) g (f)]=E[R(cid:98)S (f)]=R(f), andso
V[R(cid:98) g (f)]=
m
1 (cid:0)E[L(f, g (X))2 h
g
(g (X))2]−R(f)2(cid:1)
≤
m
1 (cid:0)E[L(f, g (X)) h
g
(g (X))]−R2(f) (cid:1) =
m
1 (cid:0) R(f)−R2(f) (cid:1) =V[R(cid:98)S (f)].
Intuitively, themoresuccessfulthe AEGis (i.e., themoreclassificationerroritinduces), thesmaller
thevarianceoftheestimate R(cid:98) g (f) becomes.
3 Detectingoverfitting
Inthissectionweshowhowtheriskestimatesintroducedintheprevioussectioncanbeusedtotest
theindependencehypothesisthat
(H) thesample S andthemodelf areindependent.
If (H) holds, E[R(cid:98) g (f)] = E[R(cid:98)S (f)] = R(f), and so the difference T
S, g
(f) = R(cid:98) g (f)−R(cid:98)S (f)
is expected to be small. On the other hand, if f is overfitted to the dataset S (in which case
R(cid:98)S (f)<R(f)), weexpect R(cid:98)S (f) and R(cid:98) g (f) tobehavedifferently (thelatterbeinglesssensitiveto
overfitting) since (i)R(cid:98) g (f) dependsalsoonexamplespreviouslyunseenbythetrainingprocedure;
(ii) theadversarialtransformationg aimstoincreasetheloss, counteringtheeffectofoverfitting;
(iii) especially in high dimensional settings, in case of overfitting one may expect that there are
misclassified points very close to the decision boundary of f which can be found by a carefully
designed AEG. Therefore, intuitively, (H) can be rejected if |T (f)| exceeds some appropriate
S, g
threshold.
3.1 Testbasedonconfidenceintervals
The simplest way to determine the threshold is based on constructing confidence intervals for
these estimator based on concentration inequalities. Under (H), standard concentration inequal-
ities, such as the Chernoff or empirical Bernstein bounds [3], can be used to quantify how
fast R(cid:98)S and R(cid:98) g (f) concentrate around the expected error R(f). In particular, we use the
following empirical Bernstein bound [22]: Let σ¯
S
2 = (1/m) (cid:80) m
i=1
(L(f, X
i
) − R(cid:98)S (f))2 and
σ¯
g
2 = (1/m) (cid:80) m
i=1
(L(f, g (X
i
)) h
g
(g (X
i
))−R(cid:98) g (f))2 denote the empirical variance of L(f, X
i
)
and L(f, g (X )) h (g (X )), respectively. Then, forany0<δ ≤1, withprobabilityatleast1−δ,
i g i
|R(cid:98)S (f)−R(f)|≤B(m,σ¯
S
2,δ,1), (3)
(cid:113)
where B(m,σ2,δ,1)= 2σ2 ln (3/δ) + 3 ln (3/δ) andweusedthefactthattherangeof L(f, x) is1
m m
(thelastparameterof Bistherangeoftherandomvariablesconsidered). Similarly, withprobability
atleast1−δ,
|R(cid:98) g (f)−R(f)|≤B(m,σ¯
g
2,δ,1). (4)
It follows trivially from the union bound that if the independence hypothesis (H) holds, the
abovetwoconfidenceintervals[R(cid:98)S (f)−B(m,σ¯
S
2,δ,1), R(cid:98)S (f)+B(m,σ¯
S
2,δ,1)]and[R(cid:98) g (f)−
5


---

## Content Unique to Old Extraction (not found in markitdown output)

### 1. # Detecting Overfitting via Adversarial Examples...

# Detecting Overfitting via Adversarial Examples

### 2. > *Source PDF: Detecting Overfitting via Adversarial Examples.pdf*
> *Extraction...

> *Source PDF: Detecting Overfitting via Adversarial Examples.pdf*
> *Extraction: Combined — markitdown raw text (base) + previous extraction supplements*

### 3. ## Content from Previous Extraction (not in markitdown output)...

## Content from Previous Extraction (not in markitdown output)

### 4. ### Visual/Chart/Graph Descriptions...

### Visual/Chart/Graph Descriptions

### 5. Detecting Overfitting via Adversarial Examples
Roman Werpachowski András György ...

Detecting Overfitting via Adversarial Examples
Roman Werpachowski András György Csaba Szepesvári
Deep Mind, London, UK
{romanw, agyorgy, szepi}@google.com
Abstract
Thefrequentreuseoftestsetsinpopularbenchmarkproblemsraisesdoubtsabout
thecredibilityofreportedtest-errorrates. Verifyingwhetheralearnedmodelis
overfittedtoatestsetischallengingasindependenttestsetsdrawnfromthesame
data distribution are usually unavailable, while other test sets may introduce a
distributionshift. Weproposeanewhypothesistestthatusesonlytheoriginaltest
datatodetectoverfitting. Itutilizesanewunbiasederrorestimatethatisbased
onadversarialexamplesgeneratedfromthetestdataandimportanceweighting.
Overfittingisdetectedifthiserrorestimateissufficientlydifferentfromtheoriginal
testerrorrate. Wedevelopaspecializedvariantofourtestformulticlassimage
classification, andapplyittotestingoverfittingofrecentmodelstothepopular
Image Netbenchmark. Ourmethodcorrectlyindicatesoverfittingofthetrained
modeltothetrainingset, butisnotabletodetectanyoverfittingtothetestset, in
linewithotherrecentworkonthistopic.
1 Introduction
Deepneuralnetworksachieveimpressiveperformanceonmanyimportantmachinelearningbench-
marks, such as image classification [18, 19, 28, 27, 16], automated translation [2, 31] or speech
recognition[9,15]. However, thebenchmarkdatasetsareusedamultitudeoftimesbyresearchers
worldwide. Sincestate-of-the-artmethodsareselectedandpublishedbasedontheirperformance
onthecorrespondingtestset, itistypicaltoseeresultsthatcontinuouslyimproveovertime; see,
e.g., thediscussionof Rechtetal.[25]and Figure1 fortheperformanceimprovementofclassifiers
publishedforthepopular CIFAR-10 imageclassificationbenchmark[18].
1.00
0.95
0.90
0.85
0.80
2010 2012 2014 2016 2018
year
ycarucca
Thisprocessmaynaturallyleadtomodelsover-
fitted to the test set, rendering test error rate
(theaverageerrormeasuredonthetestset) an
unreliableindicatoroftheactualperformance.
Detectingwhetheramodelisoverfittedtothe
test set is challenging, since independent test
setsdrawnfromthesamedatadistributionare
generally not available, while alternative test
setsoftenintroduceadistributionshift.
Toestimatetheperformanceofamodelonun-
seendata, onemayusegeneralizationbounds
togetupperboundsontheexpectederrorrate.
The generalization bounds are also applicable Figure1:Accuracyofimageclassifiersonthe CIFAR-10
whenthemodelandthedataaredependent (e.g., testset, byyearofpublication (datafrom[25]).
forcrossvalidationorforerrorestimatesbased
onthetrainingdataorthereusedtestdata), buttheyusuallyleadtolooseerrorbounds. Therefore,
althoughmuchtighterboundsareavailableifthetestdataandthemodelareindependent, comparing
33 rd Conferenceon Neural Information Processing Systems (Neur IPS2019), Vancouver, Canada.

### 6. confidenceintervalsconstructedaroundthetrainingandtesterrorratesleadstoanunderpo...

confidenceintervalsconstructedaroundthetrainingandtesterrorratesleadstoanunderpoweredtest
fordetectingthedependenceofamodelonthetestset. Recently, severalmethodshavebeenproposed
thatallowthereuseofthetestsetwhilekeepingthevalidityoftesterrorrates[10]. However, these
areintrusive: theyrequiretheusertofollowastrictprotocolofinteractingwiththetestsetandare
thusnotapplicableinthemorecommonsituationwhenenforcingsuchaprotocolisimpossible.
Inthispaperwetakeanewapproachtothechallengeofdetectingoverfittingofamodeltothetest
set, anddeviseanon-intrusivestatisticaltestthatdoesnotrestrictthetrainingprocedureandisbased
on the original test data. To this end, we introduce a new error estimator that is less sensitive to
overfittingtothedata; ourtestrejectstheindependenceofthemodelandthetestdataifthenew
errorestimateandtheoriginaltesterrorratearetoodifferent. Thecorenovelideaisthatthenew
estimatorisbasedonadversarialexamples[14], thatis, ondatapoints1 thatarenotsampledfromthe
datadistribution, butinsteadarecleverlycraftedbasedonexistingdatapointssothatthemodelerrs
onthem. Severalauthorsshowedthatthebestmodelslearnedfortheabove-mentionedbenchmark
problemsarehighlysensitivetoadversarialattacks[14,23,30,6,7,24]: forinstance, onecanoften
createadversarialversionsofimagesproperlyclassifiedbyastate-of-the-artmodelsuchthatthe
modelwillmisclassifythem, yettheadversarialperturbationsare (almost) undetectableforahuman
observer; see, e.g., Figure 2, where the adversarial image is obtained from the original one by a
carefullyselectedtranslation.
Theadversarial (error) estimator proposedin
thisworkusesadversarialexamples (generated
from the test set) together with importance
weighting to take into account the change in
thedatadistribution (covariateshift) duetothe
adversarialtransformation. Theestimatorisun-
biasedandhasasmallervariancethanthestan-
dardtesterrorrateifthetestsetandthemodel
scale, weighingmachine toaster
areindependent.2 Moreimportantly, sinceitis
Figure2:Adversarialexampleforthe Image Netdataset
basedonadversariallygenerateddatapoints, the
generatedbya (5,−5) translation:theoriginalexample
adversarial estimator is expected to differ sig-
(left) iscorrectlyclassifiedbythe VGG16 model[27]as
nificantly from the test error rate if the model
“scale, weighingmachine,”theadversariallygenerated
isoverfittedtothetestset, providingawayto
example (right) isclassifiedas“toaster,”whiletheimage
detect test set overfitting. Thus, the test error
classisthesameforanyhumanobserver.
rate and the adversarial error estimate (calcu-
latedbasedonthesametestset) mustbecloseifthetestsetandthemodelareindependent, andare
expectedtobedifferentintheoppositecase. Inparticular, ifthegapbetweenthetwoerrorestimates
islarge, theindependencehypothesis (i.e., thatthemodelandthetestsetareindependent) isdubious
andwillberejected. Combiningresultsfrommultipletrainingruns, wedevelopanothermethodto
testoverfittingofamodelarchitectureandtrainingprocedure (forsimplicity, throughoutthepaper
werefertobothtogetherasthemodelarchitecture). Themostchallengingaspectofourmethodisto
constructadversarialperturbationsforwhichwecancalculateimportanceweights, whilekeeping
enoughdegreesoffreedominthewaytheadversarialperturbationsaregeneratedtomaximizepower,
theabilityofthetesttodetectdependencewhenitispresent.
Tounderstandthebehaviorofourtestsbetter, wefirstusethemonasyntheticbinaryclassification
problem, wherethetestsareabletosuccessfullyidentifythecaseswhereoverfittingispresent. Then
we apply our independence tests to state-of-the-art classification methods for the popular image
classificationbenchmark, Image Net[8]. Asasanitycheck, inallcasesexamined, ourtestrejects
(at confidence levels close to 1) the independence of the individual models from their respective
trainingsets. Applyingourmethodto VGG16[27]and Resnet50[16]models/architectures, their
independencetothe Image Nettestsetcannotberejectedatanyreasonableconfidence. Thisisin
agreementwithrecentfindingsof[26], andprovidesadditionalevidencethatdespiteoftheexisting
danger, itislikelythatnooverfittinghashappenedduringthedevelopmentof Image Netclassifiers.
Therestofthepaperisorganizedasfollows: In Section2, weintroduceaformalmodelforerror
estimationusingadversarialexamples, includingthedefinitionofadversarialexamplegenerators.
1 Throughoutthepaper, weusethewords“example”and“point”interchangeably.
2 Notethattheadversarialerrorestimator’sgoalistoestimatetheerrorrate, nottheadversarialerrorrate (i.e.,
theerrorrateontheadversarialexamples).
2

### 7. Thenewoverfitting-detectiontestsarederivedin Section3, andappliedtoasyntheticpro...

Thenewoverfitting-detectiontestsarederivedin Section3, andappliedtoasyntheticproblemin
Section4, andtothe Image Netimageclassificationbenchmarkin Section5. Duetospacelimitations,
someauxiliaryresults, includingthein-depthanalysisofourmethodonthesyntheticproblem, are
relegatedtotheappendix.
2 Adversarial Risk Estimation
Weconsideraclassificationproblemwithdeterministic (noise-free) labels, whichisareasonable
assumptionformanypracticalproblems, suchasimagerecognition (weleavetheextensionofour
methodtonoisylabelsforfuturework).Let X ⊂RDdenotetheinputspaceand Y ={0,..., K−1}
thesetoflabels. Dataissampledfromthedistribution P over X, andtheclasslabelisdetermined
bythegroundtruthfunctionf∗ :X →Y. Wedenotearandomvectordrawnfrom P by X, andits
correspondingclasslabelby Y = f∗(X). Weconsiderdeterministicclassifiersf : X → Y. The
performanceoff ismeasuredbythezero-oneloss: L(f, x)=I(f (x)(cid:54)=f∗(x)),3 andtheexpected
error (alsoknownastheriskorexpectedriskinthelearningtheoryliterature) oftheclassifierf is
definedas R(f)=E[I(f (X)(cid:54)=Y)]= (cid:82) L(f, x) d P(x).
X
Consideratestdataset S ={(X , Y )...,(X , Y )}wherethe X aredrawnfrom Pindependently
1 1 m m i
ofeachotherand Y =f∗(X ). Inthelearningsetting, theclassifierf usuallyalsodependsonsome
i i
randomlydrawntrainingdata, henceisrandomitself. Iff is (statistically) independentfrom S, then
L(f, X ),..., L(f, X ) arei.i.d., thustheempiricalerrorrate
1 m
m m
1 (cid:88) 1 (cid:88)
R(cid:98)S (f)=
m
L(f, X
i
)=
m
I(f (X
i
)(cid:54)=Y
i
)
i=1 i=1
isanunbiasedestimateof R(f) forallf;thatis, R(f)=E[R(cid:98)S (f)|f]. Iff and S arenotindepen-
dent, theperformanceguaranteesontheempiricalestimatesavailableintheindependentcaseare
significantlyweakened;forexample, incaseofoverfittingto S, theempiricalerrorrateislikelytobe
muchsmallerthantheexpectederror.
Another well-known way to estimate R(f) is to use importance sampling (IS) [17]: instead of
samplingfromthedistribution P, wesamplefromanotherdistribution P(cid:48) andcorrecttheestimate
by appropriate reweighting. Assuming P is absolutely continuous with respect to P(cid:48) on the set
E ={x∈X :L(f, x)(cid:54)=0}, R(f)= (cid:82) L(f, x) d P(x)= (cid:82) L(f, x) h (x) d P(cid:48)(x), whereh= d P
X E d P(cid:48)
isthedensity (Radon-Nikodymderivative) of P withrespectto P(cid:48) on E (hcanbedefinedtohave
arbitraryfinitevalueson X \E). Itiswellknownthatthethecorrespondingempiricalerrorestimator
m m
1 (cid:88) 1 (cid:88)
R(cid:98)
S
(cid:48)
(cid:48)
(f)=
m
L(f, X
i
(cid:48)) h (X
i
(cid:48))=
m
I(f (X
i
(cid:48))(cid:54)=Y
i
(cid:48)) h (X
i
(cid:48)) (1)
i=1 i=1
obtainedfromasample S(cid:48) ={(X(cid:48), Y(cid:48)),...,(X(cid:48) , Y(cid:48) )}drawnindependentlyfrom P(cid:48) isunbiased
1 1 m m
(i.e., E[R(cid:98)S(cid:48) (f)|f]=R(f)) iff and S(cid:48) areindependent.
The variance of R(cid:98)(cid:48) is minimized if P(cid:48) is the so-called zero-variance IS distribution, which is
S(cid:48)
supportedon E withh (x)= R(f) forallx∈E (see, e.g.,[4, Section4.2]). Thissuggestthatan
L(f, x)
effectivesamplingdistribution P(cid:48) shouldconcentrateonpointswheref makesmistakes, whichalso
facilitatesthat R(cid:98)
S
(cid:48)
(cid:48)
(f) becomelargeiff isoverfittedto Sandhence R(cid:98)S (f) issmall. Weachievethis
throughtheapplicationofadversarialexamples.
2.1 Generatingadversarialexamples
In this section we introduce a formal framework for generating adversarial examples. Given a
classificationproblemwithdatadistribution Pandgroundtruthf∗, anadversarialexamplegenerator
(AEG) foraclassifierf isa (measurable) mappingg :X →X suchthat
(G1) gpreservestheclasslabelsofthesamples, thatis, f∗(x)=f∗(g (x)) for P-almostallx;
(G2) g doesnotchangepointsthatareincorrectlyclassifiedbyf, thatis, g (x) = xiff (x) (cid:54)=
f∗(x) for P-almostallx.
3 Foranevent B, I(B) denotesitsindicatorfunction:I(B)=1 if Bhappensand I(B)=0 otherwise.
3

### 8. (cid:51) (cid:51) (cid:51) (cid:51) (cid:55) (cid:55) (cid:55) (cid:55) (cid:51)...

(cid:51) (cid:51) (cid:51) (cid:51) (cid:55) (cid:55) (cid:55) (cid:55) (cid:51) (cid:51) (cid:51) (cid:51) (cid:55) (cid:55) (cid:55) (cid:55) S
g
(cid:55) (cid:55)
(cid:55) (cid:55)
(cid:51) (cid:51) (cid:55) (cid:55) (cid:55) (cid:55) (cid:51) (cid:51) (cid:55) (cid:55) (cid:55) (cid:55) S(cid:48)
Figure3:Generatingadversarialexamples. Thetoprowdepictstheoriginaldataset S, withblueandorange
pointsrepresentingthetwoclasses. Theclassifier’spredictionisrepresentedbythecolorofthestripedareas
(checkmarksandcrossesdenoteifapointiscorrectlyorincorrectlyclassified).Thearrowsshowtheadversarial
transformationsviathe AEGg, resultinginthenewdataset S(cid:48);misclassifiedpointsareunchanged, whilesome
correctlyclassifiedpointsaremoved, buttheiroriginalclasslabelisunchanged. Iftheoriginaldatadistributionis
uniformover S, thetransformationgisdensitypreserving, butnotmeasurepreserving:afterthetransformation
thetworightmostcorrectlyclassifiedpointsineachclasshaveprobability0, whiletheleftmostmisclassified
pointineachclasshasprobability3/16;hence, thedensityh forthelatterpointsis1/3.
g
Figure 3 illustrates how an AEG works. In the literature, an adversarial example g (x) is usually
generatedbystayinginasmallvicinityoftheoriginaldatapointx (withrespectto, e.g., the2-orthe
max-norm) andassumingthattheresultinglabelofg (x) isthesameasthatofx (see, e.g.,[14,6]).
Thisfoundationalassumption—whichisinfactamarginconditiononthedistribution—iscaptured
incondition (G1). (G2) formalizesthefactthatthereisnoneedtochangesampleswhicharealready
misclassified. Indeed, existing AEGscomplywiththiscondition.
The performance of an AEG is usually measured by how successfully it generates misclassified
examples. Accordingly, we call a point g (x) a successful adversarial example if x is correctly
classifiedbyf andf (g (x))(cid:54)=f (x)(i.e., L(f, x)=0 and L(f, g (x))=1).
Inthedevelopmentofour AEGsforimagerecognitiontasks, wewillmakeuseofanothercondition.
Forsimplicity, weformulatethisconditionfordistributions P thathaveadensityρwithrespect
to the uniform measure on X, which is assumed to exist (notable cases are when X is finite, or
X =[0,1]D orwhen X =RD;inthelattertwocasestheuniformmeasureisthe Lebesguemeasure).
Theassumptionstatesthatthe AEGneedstobedensity-preserving:
(G3) ρ(x)=ρ(g (x)) for P-almostallx.
Note that a density-preserving map may not be measure-preserving (the latter means that for all
measurable A⊂X, P(A)=P(g (A))).
Weexpect (G3) toholdwhengperturbsitsinputbyasmallamountandifρissufficientlysmooth.
The assumption is reasonable for, e.g., image recognition problems (at least in a relaxed form,
ρ(x)≈ρ(g (x))) whereweexpectthatverycloseimageswillhaveasimilarlikelihoodasmeasured
byρ. An AEGemployingimagetranslations, whichsatisfies (G3), willbeintroducedin Section5.
Both (G1) and (G3) canberelaxed (toasoftmarginconditionorallowingaslightchangeinρ, resp.)
atthepriceofanextraerrortermintheanalysisthatfollows.
Forafixed AEGg :X →X, let P bethedistributionofg (X) where X ∼P (P isknownasthe
g g
pushforwardmeasureof P underg). Further, leth = d P on E ={x : L(f, x)(cid:54)=0}andarbitrary
g d Pg
otherwise. Itiseasytoseethat, on E, h (x) iswell-definedandh ≤1. Foranymeasurable A⊂E
g g
P (A)=P(g (X)∈A)≥P(g (X)∈A, X ∈E)=P(X ∈A)=P(A)
g
wherethesecondtolastequalityholdsbecauseg (X) = X forany X ∈ E undercondition (G2).
Thus, P(A)≤P (A) foranymeasurable A⊂E, whichimpliesthath iswell-definedon E and
g g
h (x)≤1 forallx∈E.
g
Onemaythinkthat (G3) impliesthath (x) = 1 forallx ∈ E. However, thisdoesnothold. For
g
example, if P isauniformdistribution, anyg : X → supp P satisfies (G3), wheresupp P ⊂ X
denotesthesupportofthedistribution P. Thisisalsoillustratedin Figure3.
2.2 Riskestimationviaadversarialexamples
Combining the ideas of this section so far, we now introduce unbiased risk estimates based on
adversarialexamples. Ourgoalistoestimatetheerror-rateoff throughanadversariallygenerated
4

### 9. B(m,σ¯
g
2,δ,1), R(cid:98)S (f)+B(m,σ¯
g
2,δ,1)], whichbothcontain R(f) withprob...

B(m,σ¯
g
2,δ,1), R(cid:98)S (f)+B(m,σ¯
g
2,δ,1)], whichbothcontain R(f) withprobabilityatleast1−δ,
intersectwithprobabilityatleast1−2δ.
Ontheotherhand, iff and S arenotindependent, theperformanceguarantees (3) and (4) maybe
violated and the confidence intervals may become disjoint. If this is detected, we can reject the
independencehypothesis (H) ataconfidencelevel1−2δor, equivalently, withp-value2δ. Inother
words, wereject (H) iftheabsolutevalueofthedifferenceoftheestimates T
S, g
(f)=R(cid:98) g (f)−R(cid:98)S (f)
exceeds the threshold B(m,σ¯2,δ,1)+B(m,σ¯2,δ,1) (note that E[T (f) = 0] if S and f are
S g S, g
independent).
3.2 Pairwisetest
A smaller threshold for |T (f)|, and hence a more effective independence test, can be devised
S, g
if instead of independently estimating the behavior of R(cid:98)S and R(cid:98) g (f), one utilizes their apparent
correlation. Indeed, T (f)=(1/m)
(cid:80) m
T (f) where
S, g i=1 i, g
T (f)=L(f, g (X )) h (g (X ))−L(f, X ) (5)
i, g i g i i
and the two terms in T (f) have the same mean and are typically highly correlated by the con-
i, g
structionofg. Thus, wecanapplytheempirical Bernsteinbound[22]tothepairwisedifferences
T (f) tosetatighterthresholdinthetest: iftheindependencehypothesis (H) holds (i.e., S andf
i, g
areindependent), thenforany0<δ <1, withprobabilityatleast1−δ,
|T (f)|≤B(m,σ¯2,δ, U) (6)
S, g T
(cid:113)
with B(m,σ2,δ, U) = 2σ2 ln (3/δ) + 3 Uln (3/δ), whereσ¯2 = (1/m) (cid:80) m (T (f)−T (f))2 is
m m T i=1 i S, g
theempiricalvarianceofthe T (f) termsand U =sup T (f)−inf T (f);wealsousedthefact
i, g i, g i, g
thattheexpectationofeach T (f), andhencethatof T (f), iszero. Sinceh ≤1 if L(f, x)=1
i, g S, g g
(asdiscussedin Section2.2), itfollowsthat U ≤2, butfurtherassumptions (suchasgbeingdensity
preserving) canresultintighterbounds.
Thisleadstoourpairwisedependencedetectionmethod:
if|T (f)|>B(m,σ¯2,δ,2), reject (H) ataconfidencelevel1−δ(p-valueδ).
S, g T
Foragivenstatistic (|T (f)|,σ¯2), thelargestconfidencelevel (smallestp-value) atwhich (H) can
S, g T
berejectedcanbecalculatedbysettingthevalueofthestatistic|T (f)|−B(m,σ¯2,δ,2) tozero
S, g T
andsolvingforδ. Thisleadstothefollowingformulaforthep-value (ifthesolutionislargerthan1,
whichhappenswhenthebound (6) isloose,δiscappedat1):
(cid:26) (cid:16) √ (cid:17)(cid:27)
δ =min 1,3 e
−
9
m
U2
σ¯
T
2+3 U|TS, g (f)|−σ¯T σ¯
T
2+6 U|TS, g (f)|
. (7)
Note that in order for the test to work well, we not only need the test statistic T (f) to have a
S, g
smallvarianceincaseofindependence (thiscouldbeachievedifgweretheidentity), butwealso
needtheestimators R(cid:98)S (f) and R(cid:98) g (f) behavesufficientlydifferentlyiftheindependenceassumption
isviolated. Thelatterbehaviorisencouragedbystronger AEGs, aswewillshowempiricallyin
Section5.2(see Figure5 inparticular).
3.3 Dependencedetectorforrandomizedtraining
Thedependencebetweenthemodelandthetestsetcanarisefrom (i) selectingthe“best”random
seedinordertoimprovethetestsetperformanceand/or (ii) tweakingthemodelarchitecture (e.g.,
neuralnetworkstructure) andhyperparameters (e.g., learning-rateschedule). Ifonehasaccesstoa
singleinstanceofatrainedmodel, thesetwosourcescannotbedisentangled. However, ifthemodel
architectureandtrainingprocedureisfullyspecifiedandcomputationalresourcesareadequate, it
ispossibletoisolate (i) and (ii) byretrainingthemodelmultipletimesandcalculatingthep-value
foreverytrainingrunseparately. Assuming N models, letf , j =1,..., N denotethej-thtrained
j
modelandp thep-valuecalculatedusingthepairwiseindependencetest (6)(i.e., from Eq.7 in
j
Section3). Wecaninvestigatethedegreetowhich (i) occursbycomparingthep valueswiththe
j
correspondingtestseterrorrates R (f ). Toinvestigatewhether (ii) occurs, wecanaverageoverthe
S j
randomnessofthetrainingruns.
6

### 10. For every example X ∈ S, consider the average test statistic T¯ = 1 (cid:80)N T ...

For every example X ∈ S, consider the average test statistic T¯ = 1 (cid:80)N T (f ), where
i i N j=1 i, gj j
T (f ) isthestatistic (5) calculatedforexample X andmodelf with AEGg selectedformodel
i, gj j i j j
f (notethat AEGsaremodel-dependentbyconstruction). If, foreachiandj, therandomvariables
j
T (f ) areindependent, thensoarethe T¯ (foralli). Hence, wecanapplythepairwisedependence
i j i
detector (6) with T¯ insteadof T , usingtheaverage T¯ =(1/m) (cid:80) m T¯ withempiricalvariance
i i S i=1 i
σ¯2 =(1/m) (cid:80) m (T¯ −T¯ )2, givingasinglep-valuep . Ifthetrainingrunsvaryenoughintheir
T, N i=1 i S N
outcomes, differentmodelsf errondifferentdatapoints X , leadingtoσ¯2 <σ¯2, andtherefore
j j T, N T
strengtheningthepowerofthedependencedetector. Forbrevity, wecallthisindependencetestan
N-modeltest.
4 Syntheticexperiments
1.0
0.8
0.6
0.4
0.2
0.0
10-2 10-1 100 101 102
†
eulav-p
Firstweverifytheeffectivenessofourmethod
onasimplelinearclassificationproblem. Due
N=1 N=1
tospacelimitations, weonlyconveyhigh-level
N=2 N=2
results here, details are given in Appendix A. N=10 N=10
We assume that the data is linearly separa-
N=25 N=25
ble with a margin and the density ρ is known. N=100 N=100
We consider a linear classifiers of the form
f (x) = sgn (w (cid:62) x+b) trained with the cross-
entropylossc, andweemployaone-stepgra-
dient method (which is an L version of the
2
fastgradient-signmethodof[14,23]) todefine
Figure4: Averagep-valuesproducedbytheindepen-
our AEG g, which tries to modify a correctly
dencetestinaseparablelinearclassificationproblem
classified point x with label y in the direction forthecasesofbothwhenthemodelisindependentof
of the gradient of the cost function, yielding (dashedlines) and, resp., dependenton (solidlines) the
x (cid:48) =x−εyw/(cid:107) w (cid:107) , whereε≥0 isthestrength testset.
2
oftheattack. Tocomplywiththerequirements
foran AEG, wedefinegasfollows: g (x)=x (cid:48) if L(f, x)=0 andf∗(x)=f∗(x (cid:48))(corresponding
to (G2) and (G1), respectively), whileg (x)=xotherwise. Therefore, ifx (cid:48) ismisclassifiedbyf, x
andx (cid:48) aretheonlypointsmappedtox (cid:48) byg. Thissimpleformofgandtheknowledgeofρallowsto
computethedensityh , makingiteasytocomputetheadversarialerrorestimate (2). Figure4 shows
g
theaveragep-valuesproducedbyour N-modelindependencetestforadependent (solidlines) and
anindependent (dashedlines) testset. Itcanbeseenthatinthedependentcasethetestcanreject
independencewithhighconfidenceforalargerangeofattackstrengthε, whiletheindependence
hypothesisisnotrejectedinthecaseoftrueindependence. Moredetails (includingwhyonlyarange
ofεissuitablefordetectingoverfitting) aregivenin Appendix A.
5 Testingoverfittingon Image Net
In the previous section we showed that the proposed adversarial-example-based dependence test
worksforasyntheticproblemwherethedensitiescanbecomputedexactly. Inthissectionweapply
ourestimatestoapopularimageclassificationbenchmark, Image Net[8];herethemainissueisto
findsufficientlystrong AEGsthatmakecomputingthecorrespondingdensitiespossible.
Tofacilitatethecomputationofthedensityh , weonlyconsiderdensity-preserving AEGsasdefined
g
by (G3)(recallthat (G3) isdifferentfromrequiringh =1). Sincein (2) and (5), h (x) ismultiplied
g g
by L(f, x), weonlyneedtodeterminethedensityh fordatapointsthataremisclassifiedbyf.
g
5.1 AEGsbasedontranslations
To satisfy (G3), we implement the AEG using translations of images, which have recently been
proposedasmeansofgeneratingadversarialexamples[1]. Althoughrelativelyweak, suchattacksfit
ourneedswell:unlesstheimagesareprocedurallycentered, itisreasonabletoassumethattranslating
thembyafewpixelsdoesnotchangetheirlikelihood.4 Wealsomakethenaturalassumptionthat
the small translations used do not change the true class of an image. Under these assumptions,
4 Notethatthisassumptionlimitstheapplicabilityofourmethod, excludingsuchcenteredoressentially
centeredimageclassificationbenchmarksas MNIST[20]or CIFAR-10[18].
7

### 11. translationsbyafewpixelssatisfyconditions (G1) and (G3). Animage-translatingfunc...

translationsbyafewpixelssatisfyconditions (G1) and (G3). Animage-translatingfunctiongisa
valid AEGifitleavesallmisclassifiedimagesinplace (tocomplywith (G2)), andeitherleavesa
correctlyclassifiedimageunchangedorappliesasmalltranslation.
Themainbenefitofusingatranslational AEGg (withboundedtranslations) isthatitsdensityh (x)
g
foranimagexcanbecalculatedexactlybyconsideringthesetofimagesx (cid:48) thatcanbemappedtox
byg (thisisduetoourassumption (G3)). Weconsideredmultiplewaysforconstructingtranslational
AEGs. Thebestversion (selectedbasedoninitialevaluationsonthe Image Nettrainingset), which
wecalledthestrongestperturbation, seeksanon-identicalneighborofacorrectlyclassifiedimage
x (neighboringimagesaretheonesthatareaccessiblethroughsmalltranslations) thatcausesthe
classifiertomakeanerrorwiththelargestconfidence.
Formally, wemodelimagesas3 Dtensorsin[0,1]W×H×C space, where C =3 for RGBdata, and
W and H arethewidthandheightoftheimages, respectively. Letτ (x) denotethetranslationof
v
animagexbyv ∈ Z2 pixelsinthe (X, Y) plane (here Zdenotesthesetofintegers). Tocontrol
the amount of change, we limit the magnitude of translations and allow v ∈ V = {u ∈ Z2 :
ε
u (cid:54)= (0,0),(cid:107) u (cid:107) ≤ ε} only, for some fixed positive ε. Thus, we considers AEGs in the form
∞
g (x)∈{τ (x):v ∈V}∪{x}iff (x)=f∗(x) andg (x)=xotherwise (ifxiscorrectlyclassified,
v
weattempttotranslateittofindanadversarialexamplein{τ (x):v ∈V}whichismisclassifiedby
v
f, butxisleftunchangedifnosuchpointexists). Denotingthedensityofthepushforwardmeasure
P byρ , foranymisclassifiedpointx,
g g
(cid:32) (cid:33)
(cid:88) (cid:88)
ρ (x)=ρ(x)+ ρ(τ (x))I(g (τ (x))=x)=ρ(x) 1+ I(g (τ (x))=x)
g −v −v −v
v∈V v∈V
wherethesecondequalityfollowsfrom (G3). Therefore, thecorrespondingdensityis
h (x)=1/(1+n (x)) (8)
g
wheren (x)= (cid:80) I(g (τ (x))=x) isthenumberofneighboringimageswhicharemappedto
v∈V −v
xbyg. Notethatgivenf andg, n (x) canbeeasilycalculatedbycheckingallpossibletranslations
ofxby−vforv ∈V. Itiseasytoextendtheabovetonon-deterministicperturbations, definedas
distributionsover AEGs, byreplacingtheindicatorwithitsexpectation P(g (τ (x))=x|x, v) with
−v
respecttotherandomnessofg, yielding
1
h (x)= . (9)
g 1+ (cid:80) P(g (τ (x))=x|x, v)
v∈V −v
Ifgisdeterministic, wehaveh (x)≤1/2 foranysuccessfuladversarialexamplex. Hence, forsuch
g
g, therange U oftherandomvariables T definedin (5) hasatighterupperboundof3/2 instead2(as
i
T ∈[−1,1/2]), leadingtoatighterboundin (6) andastrongerpairwiseindependencetest. Inthe
i
experiments, weusethisstrongertest. Weprovideadditionaldetailsaboutthetranslational AEGs
usedin Appendix B.
5.2 Testsof Image Netmodels
Weappliedourtesttocheckifstate-of-the-art classifiers forthe Image Netdataset[8]havebeen
overfittedtothetestset. Inparticular, weusethe VGG16 classifierof[27]andthe Resnet50 classifier
of [16]. Due to computational considerations, we only analyzed a single trained VGG16 model,
whilethe Resnet50 modelwasretrained120 times. Themodelsweretrainedusingtheparameters
recommendedbytheirrespectiveauthors.
Thepreprocessingprocedureofbotharchitecturesinvolvesrescalingeveryimagesothatthesmaller
ofwidthandheightis256 andnextcroppingcentrallytosize224×224. Thismeansthattranslating
theimagebyvcanbetriviallyimplementedbyshiftingthecroppingwindowby−vwithoutanyloss
ofinformationfor (cid:107) v (cid:107) ≤16, becausewehaveenoughextrapixelsoutsidetheoriginal, centrally
∞
locatedcroppingwindow. Thisimpliesthatwecancomputethedensitiesofthetranslational AEGs
forany (cid:107) v (cid:107) ≤ε=(cid:98)16/3(cid:99)=5(see Appendix B.1 fordetailedexplanation).Becausethe Image Net
∞
datacollectionproceduredidnotimposeanystrictrequirementsoncenteringtheimages[8], itis
reasonable to assume (as we do) that small (lossless) translations respect the density-preserving
condition (G3).
Inourfirstexperiment, weappliedourpairwiseindependencetest (6) withthe AEGsdescribedin
Appendix B(strongest, nearest, andthetworandombaselines) toall1,271,167 trainingexamples, as
8

### 12. 1.0
0.8
0.6
0.4
0.2
0.0
0.2 0.4 0.6 0.8 1.0 1.2
sample size 106
×
eulav-P
0.188
...

1.0
0.8
0.6
0.4
0.2
0.0
0.2 0.4 0.6 0.8 1.0 1.2
sample size 106
×
eulav-P
0.188
0.186
RS(f)
Rcg (f)
variant 0.184
c
strongest 0.182
nearest
random 0.180
random2 0.178
0.176
0.174
0.2 0.4 0.6 0.8 1.0 1.2
sample size 106
×
Figure5:p-valuesfortheindependencetestonthe Image Nettrainingsetfordifferentsamplesizesand AEG
variants (left);originalandadversarialriskestimates, R(cid:98)S (f) and R(cid:98) g (f), onthe Image Nettrainingsetwith
97.5%two-sidedconfidenceintervalsforthe‘strongestattack’AEG(right).
wellastoanumberofitsrandomlyselected (uniformlywithoutreplacement) subsetsofdifferent
sizes. Besidesthisbeingasanitycheck, wealsousedthisexperimenttoselectfromdifferent AEGs
andcomparetheperformanceofthepairwiseindependencetest (6) tothebasicversionofthetest
describedin Section3.1.
The left graph in Figure 5 shows that with the “strongest perturbation”, we were able to reject
independenceofthetrainedmodelandthetrainingsamplesataconfidencelevelverycloseto1 when
enoughtrainingsamplesareconsidered (tobeprecise, forthewholetrainingsettheconfidencelevel
is99.9994%). Note, however, thatthemuchweaker“smallestperturbation”AEG, aswellasthe
randomtransformations, arenotabletodetectthepresenceofoverfitting. Atthesametime, thegraph
ontherighthandsideshowstherelativestrengthofthepairwiseindependencetestcomparedtothe
basicversionbasedonindependentconfidenceintervalestimatesasdescribedindetailin Section3.1:
the 97.5%-confidence intervals of the error estimates R(cid:98)S (f) and R(cid:98) g (f) overlap, not allowing to
rejectindependenceataconfidencelevelof95%(notethathere S denotesthetrainingset).
Ontheotherhand, whenappliedtothetestset, weobtainedap-valueof0.96, notallowingatall
torejecttheindependenceofthetrainedmodelandthetestset. Thisresultcouldbeexplainedby
thetestbeingtooweak, asnooverfittingisdetectedtothetrainingsetatsimilarsamplesizes (see
Figure5), orsimplythelackofoverfitting. Similarresultswereobtainedfor Resnet50, whereeven
the N-modeltestwith N =120 independentlytrainedmodelsresultedapvalueof1, notallowing
torejectindependenceatanyconfidencelevel. Theviewofnooverfittingcanbebackedupinat
leasttwoways: first,“manual”overfittingtotherelativelylarge Image Nettestsetishard. Second,
sincetrainingan Image Netmodelwasjusttoocomputationallyexpensiveuntilquiterecently, onlya
relativelysmallnumberofdifferentarchitecturesweredevelopedforthisproblem, andtheevolution
oftheirdesignwasoftendrivenbycomputationalefficiencyontheavailablehardware. Ontheother
hand, itisalsopossiblethatincreasing N sufficientlymightshowevidenceofoverfitting (thisisleft
forfuturework).
6 Conclusions
Wepresentedamethodfordetectingoverfittingofmodelstodatasets. Itreliesonanimportance-
weightedriskestimatefromanewdatasetobtainedbygeneratingadversarialexamplesfromthe
originaldatapoints. Weappliedourmethodtothepopular Image Netimageclassificationtask. For
thispurpose, wedevelopedaspecializedvariantofourmethodforimageclassificationthatuses
adversarialtranslations, providingargumentsforitscorrectness. Luckily, andinagreementwithother
recentworkonthistopic[25,26,13,21,32], wefoundnoevidenceofoverfittingofstate-of-the-art
classifierstothe Image Nettestset.
Themostchallengingaspectofourmethodsistoconstructadversarialperturbationsforwhichwecan
calculatetheimportanceweights;findingstrongerperturbationsthantheonesbasedontranslations
forimageclassificationisanimportantquestionforthefuture. Anotherinterestingresearchdirection
istoconsiderextensionsbeyondimageclassification, forexample, bybuildingonrecentadversarial
attacksforspeech-to-textmethods[5], machinetranslation[11]ortextclassification[12].
9

### 13. Acknowledgements
Wethank J.Uesatoforusefuldiscussionsandadviceaboutadversarialat...

Acknowledgements
Wethank J.Uesatoforusefuldiscussionsandadviceaboutadversarialattackmethodsandsharing
theirimplementations[30]withus, aswellas M.Roscaand S.Gowalforhelpwithretrainingimage
classificationmodels. Wealsothank B.O’Donoghueforusefulremarksaboutthemanuscript, and L.
Schmidtforanin-depthdiscussionoftheirresultsonthistopic. Finally, wethank D.Balduzzi, S.
Legg, K.Kavukcuogluand J.Martensforencouragement, support, livelydiscussionsandfeedback.
References
[1] Aharon Azulayand Yair Weiss. Whydodeepconvolutionalnetworksgeneralizesopoorlytosmallimage
transformations? 2018. ar Xiv:1805.12177.
[2] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neuralmachinetranslationbyjointlylearning
toalignandtranslate. In Proceedingsofthe International Conferenceon Learning Representations (ICLR),
2015.
[3] Stéphane Boucheron, Gábor Lugosi, and Pascal Massart. Concentration Inequalities:ANonasymptotic
Theoryof Independence. Oxford University Press,2013.
[4] James Antonio Bucklew. Introductionto Rare Event Simulation. Springer New York,2004.
[5] N.Carliniand D.Wagner. Audioadversarialexamples:Targetedattacksonspeech-to-text. In2018 IEEE
Securityand Privacy Workshops (SPW), May2018.
[6] Nicholas Carliniand David A.Wagner. Adversarialexamplesarenoteasilydetected: Bypassingten
detectionmethods. In Proceedingsofthe10 th ACMWorkshopon Artificial Intelligenceand Security,
AISec@CCS2017, Dallas, TX, USA, November3,2017, pages3–14,2017. URLhttp://doi.acm.org/
10.1145/3128572.3140444.
[7] Nicholas Carliniand David A.Wagner. Towardsevaluatingtherobustnessofneuralnetworks. In2017
IEEESymposiumon Securityand Privacy, SP2017, San Jose, CA, USA, May22-26,2017, pages39–57,
2017. URLhttps://doi.org/10.1109/SP.2017.49.
[8] J.Deng, W.Dong, R.Socher, L.J.Li, Kai Li, and Li Fei-Fei. Image Net:Alarge-scalehierarchicalimage
database. In2009 IEEEConferenceon Computer Visionand Pattern Recognition, pages248–255, June
2009. doi:10.1109/CVPR.2009.5206848.
[9] L.Deng, G.Hinton, and B.Kingsbury. Newtypesofdeepneuralnetworklearningforspeechrecognition
andrelatedapplications:anoverview. In2013 IEEEInternational Conferenceon Acoustics, Speechand
Signal Processing, pages8599–8603.IEEE, May2013.
[10] Cynthia Dwork, Vitaly Feldman, Moritz Hardt, Toniann Pitassi, Omer Reingold, and Aaron Roth. The
reusableholdout:Preservingvalidityinadaptivedataanalysis. Science,349(6248):636–638,2015.
[11] Javid Ebrahimi, Daniel Lowd, and Dejing Dou. Onadversarialexamplesforcharacter-levelneuralmachine
translation. In Proceedingsofthe27 th International Conferenceon Computational Linguistics, COLING
2018, Santa Fe, New Mexico, USA, August20-26,2018, pages653–663,2018.
[12] Javid Ebrahimi, Anyi Rao, Daniel Lowd, and Dejing Dou. Hotflip:White-boxadversarialexamplesfortext
classification. In Proceedingsofthe56 th Annual Meetingofthe Associationfor Computational Linguistics,
ACL2018, Melbourne, Australia, July15-20,2018, Volume2:Short Papers, pages31–36,2018.
[13] Vitaly Feldman, Roy Frostig, and Moritz Hardt. Theadvantagesofmultipleclassesforreducingoverfitting
fromtestsetreuse. In Proceedingsofthe36 th International Conferenceon Machine Learning, pages
1892–1900,2019.
[14] I. J. Goodfellow, J. Shlens, and C. Szegedy. Explaining and harnessing adversarial examples. In
Proceedingsofthe International Conferenceon Learning Representations (ICLR),2015.
[15] Alex Graves, Abdel-rahman Mohamed, and Geoffrey Hinton. Speechrecognitionwithdeeprecurrent
neuralnetworks. In2013 IEEEInternational Conferenceon Acoustics, Speechand Signal Processing,
pages6645–6649.IEEE,2013.
[16] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deepresiduallearningforimagerecognition.
In Proceedingsofthe IEEEConferenceon Computer Visionand Pattern Recognition, pages770–778,
2016.
10

### 14. [17] H.Kahnand T.E.Harris. Estimationofparticletransmissionbyrandomsampling. In ...

[17] H.Kahnand T.E.Harris. Estimationofparticletransmissionbyrandomsampling. In Monte Carlo
Method, volume12 of Applied Mathematics Series, pages27–30.National Bureauof Standards,1951.
[18] Alex Krizhevsky. Learningmultiplelayersoffeaturesfromtinyimages. Technicalreport, Universityof
Toronto,2009.
[19] Alex Krizhevsky, Ilya Sutskever, and Geoffrey EHinton. Image Netclassificationwithdeepconvolutional
neuralnetworks. In F.Pereira, C.J.C.Burges, L.Bottou, and K.Q.Weinberger, editors, Advancesin
Neural Information Processing Systems25, pages1097–1105.Curran Associates, Inc.,2012.
[20] Yann Le Cunand Corinna Cortes. MNISThandwrittendigitdatabase. http://yann.lecun.com/exdb/mnist/,
2010.
[21] Horia Mania, John Miller, Ludwig Schmidt, Moritz Hardt, and Benjamin Recht. Modelsimilaritymitigates
testsetoveruse. 2019. ar Xiv:1905.12580.
[22] Volodymyr Mnih, Csaba Szepesvári, and Jean-Yves Audibert. Empirical Bernsteinstopping. In Proceed-
ingsofthe25 th International Conferenceon Machine Learning, ICML’08, pages672–679, New York,
NY, USA,2008.ACM.
[23] Nicolas Papernot, Patrick D.Mc Daniel, and Ian J.Goodfellow. Transferabilityinmachinelearning:from
phenomenatoblack-boxattacksusingadversarialsamples. 2016. ar Xiv:1605.07277.
[24] Nicolas Papernot, Patrick Mc Daniel, Ian Goodfellow, Somesh Jha, ZBerkay Celik, and Ananthram Swami.
Practicalblack-boxattacksagainstmachinelearning. In Proceedingsofthe2017 ACMon Asia Conference
on Computerand Communications Security, pages506–519.ACM,2017.
[25] Benjamin Recht, Rebecca Roelofs, Ludwig Schmidt, and Vaishaal Shankar. Do CIFAR-10 classifiers
generalizeto CIFAR-10? 2018. ar Xiv:1806.00451.
[26] Benjamin Recht, Rebecca Roelofs, Ludwig Schmidt, and Vaishaal Shankar. Do Image Netclassifiers
generalizeto Image Net? 2019. ar Xiv:1902.10811.
[27] K.Simonyanand A.Zisserman. Verydeepconvolutionalnetworksforlarge-scaleimagerecognition. In
Proceedingsofthe International Conferenceon Learning Representations (ICLR),2015.
[28] Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru
Erhan, Vincent Vanhoucke, and Andrew Rabinovich. Goingdeeperwithconvolutions. In2015 IEEE
Conferenceon Computer Visionand Pattern Recognition (CVPR),2015. ar Xiv:1409.4842.
[29] T.Tielemanand G.Hinton. Lecture6.5—Rms Prop:Dividethegradientbyarunningaverageofitsrecent
magnitude. COURSERA:Neural Networksfor Machine Learning,2012.
[30] Jonathan Uesato, Brendan O’Donoghue, Aäronvanden Oord, and Pushmeet Kohli. Adversarialriskand
thedangersofevaluatingagainstweakattacks. In Proceedingsofthe35 th International Conferenceon
Machine Learning, volume80 of Proceedingsof Machine Learning Research, pages5025–5034,2018.
ar Xiv:1802.05666.
[31] Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V.Le, Mohammad Norouzi, Wolfgang Macherey, Maxim
Krikun, Yuan Cao, Qin Gao, Klaus Macherey, Jeff Klingner, Apurva Shah, Melvin Johnson, Xiaobing
Liu, Lukasz Kaiser, Stephan Gouws, Yoshikiyo Kato, Taku Kudo, Hideto Kazawa, Keith Stevens, George
Kurian, Nishant Patil, Wei Wang, Cliff Young, Jason Smith, Jason Riesa, Alex Rudnick, Oriol Vinyals,
Greg Corrado, Macduff Hughes, and Jeffrey Dean. Google’sneuralmachinetranslationsystem:Bridging
thegapbetweenhumanandmachinetranslation. 2016. ar Xiv:1609.08144.
[32] Chhavi Yadavand Léon Bottou. Coldcase:Thelost MNISTdigits. May2019. ar Xiv:1905.10498.
11

### 15. A Syntheticexperiments
Inthissection, wepresentfulldetailsoftheexperimentsonasim...

A Syntheticexperiments
Inthissection, wepresentfulldetailsoftheexperimentsonasimplesyntheticclassificationproblem,
whichwepresentedbrieflyin Section4. Theseexperimentsillustratethepowerofthemethodof
Section3. Theadvantageofthesimplesetupconsideredhereisthatweareabletocomputethe
densityh inananalyticform (see Figure6 foranillustration).
g
A.1 Datadistributionandmodel
Let X = R500 and consider an input distribution with a density ρ that is an equally weighted
mixture of two 500-dimensional isotropic truncated Gaussian distributions Ntrunc (µ ,σ2 I) with
√ ± ±
coordinate-wise standard deviation σ = 500 (I denotes the identity matrix of size 500×500),
meansµ =[±1,0,0,...,0]anddensitiesρ truncatedinthefirstdimensionsuchthatρ (x)=0
± ± +
ifx ≤ 0.025 andρ (x) = 0 ifx ≥ −0.025. Thelabelofaninputpointxisf∗(x) = sgn (x ),
1 − 1 1
whichisthesignofitsfirstcoordinate.
Weconsiderlinearclassifiersoftheformf (x)=sgn (w (cid:62) x+b) trainedwiththecross-entropyloss
c ((w, b), x, y) = ln (1+e−y (w (cid:62) x+b)) wherey = f∗(x). Weemployaone-stepgradientmethod
(whichisan L versionofthefastgradient-signmethodof[14,23]) todefineour AEGg, which
2
triestomodifyacorrectlyclassifiedpointxwithlabelyinthedirectionofthegradientofthecost
function c: x (cid:48) = x+ε∇ c ((w, b), x, y)/(cid:107)∇ c ((w, b), x, y)(cid:107) for some ε > 0. For our specific
x x 2
choiceofc, theabovesimplifiestox (cid:48) =x−εyw/(cid:107) w (cid:107) . Tocomplywiththerequirementsforan
2
AEG, we define g as follows: g (x) = x (cid:48) if L(f, x) = 0 and f∗(x) = f∗(x (cid:48)) (corresponding to
(G2) and (G1), respectively), whileg (x) = xotherwise. Therefore, ifx (cid:48) ismisclassifiedbyf, x
x
2
y = −1 y = +1
x (cid:48)
x A
C x (cid:48)
B x A
x (cid:48)
C 0 x B x 1
x (cid:48) w
D
x
D
x (cid:48)
E
x
E
Figure6:Illustrationofthedatadistributionandthelinearmodelf (x)=sgn (w (cid:62) x+b) intwodimensions.
Theblueandgreengradientsshowtheprobabilitydensityρofthedatawithtruelabelsy = −1 andy = 1,
respectively, while the white space between them is the margin with ρ = 0. The red line is the model’s
classificationboundarywithitsparametervectorwshownbythepurplearrow. Dependingonthelabel, wor
−wisthedirectionoftranslationusedtoperturbedthecorrectlyclassifieddatapoints, andthetranslations
used by the AEG g for specific points are depicted by grey arrows: solid arrows indicate the cases where
g (x) = x (cid:48) (cid:54)= x, whiledashedarrowsareforcandidatetranslationswhicharenotperformedbythe AEG
becausetheywouldchangethetruelabel, f∗(x (cid:48))(cid:54)=f∗(x), andhenceg (x)=x (cid:54)=x (cid:48).Eachoriginal/perturbed
datapointisrepresentedbyacolor-codedcircle: theinnercolorcorrespondstothetruelabel (darkbluefor
y =−1 anddarkgreenfory =1) whiletheoutercolortothemodel’sprediction (darkblueforf (x)=−1
anddarkgreenforf (x)=1).Pointsx (cid:48) andx (cid:48) canbeobtainedfromx andx , respectively, byapplying
A B A B
the AEG, x (cid:48) =g (x ) andx (cid:48) =g (x ).Sinceonlyx ismappedtox (cid:48) byg, andg (x (cid:48) )=x (cid:48) , thedensity
A A B B A A A A
(Radon-Nikodymderivative) canbeobtainedash (x (cid:48) )=ρ(x (cid:48) )/(ρ(x )+ρ(x (cid:48) ))∈(0,1).Inthecaseof
g A A A A
x (cid:48) , h (x (cid:48) )=ρ(x (cid:48) )/(ρ(x )+ρ(x (cid:48) ))=0 duetothemargin. Notethattheformulaforh (x (cid:48) ) doesnot
B g B B B B g A
dependonwhetherx orx (cid:48) isintheoriginaltestset S;inthefirstcasewecallx (cid:48) a“successfuladversarial
A A A
example”whileinthesecondcasex (cid:48) iscalled“originallymisclassified”(asimilarargumentholdsforh (x (cid:48) )).
A g B
x (cid:48) is not a successful adversarial example since L(f, x (cid:48) ) = 0 (however, g (x ) = x (cid:48) according to our
C C C C
definition).Pointsx andx arenotperturbedbyour AEG, sincef∗(x )(cid:54)=f∗(x (cid:48) ) andf∗(x )(cid:54)=f∗(x (cid:48) ).
D E D D E E
12

### 16. andx (cid:48) aretheonlypointsmappedtox (cid:48) byg. Thus, thedensityatx (cid:4...

andx (cid:48) aretheonlypointsmappedtox (cid:48) byg. Thus, thedensityatx (cid:48) afterthetransformationg is
ρ(cid:48)(x (cid:48))=ρ(x)+ρ(x (cid:48))(1−L(f, x))I(f∗(x)=f∗(x (cid:48))) and
ρ(x (cid:48)) ρ(x (cid:48))
h (x (cid:48))= =
g ρ(cid:48)(x (cid:48)) ρ(x (cid:48))+ρ(x)(1−L(f, x))I(f∗(x)=f∗(x (cid:48)))
(notethat I(L(f, x)=0)=1−L(f, x)).
A.2 Experimentsetup
Wepresenttwoexperimentsshowingthebehaviorofourindependencetest: onewherethetraining
andtestsetsareindependent, andanotherwheretheyarenot.
Inthefirstexperimentalinearclassifierwastrainedonatrainingset S ofsize500 for50,000 steps
Tr
usingthe RMSPropoptimizer[29]withbatchsize100 andlearningrate0.01, obtainingzero (upto
numericalprecision) finaltraininglosscand, consequently,100%predictionaccuracyonthetraining
data. Thenthetrainedclassifierwastestedonalargetestset S ofsize10,000.5 Bothsetswere
Te
drawnindependentlyfromρdefinedabove. Weusedarangeofεvaluesmatchedtothescaleofthe
datadistribution: from10−2, whichistheorderofmagnitudeofthemarginbetweentwoclasses
(0.05), to102, whichistheorderofmagnitudeofthewidthofthe Gaussiandistributionusedforeach
√
classes (σ = 500).
Inthesecondexperimentweconsiderthesituationwherethetrainingandtestsetsarenotindependent.
Toenhancetheeffectsofthisdependence, thesetupwasmodifiedtomakethetrainingprocessmore
amenabletooverfittingbysimulatingasituationwhenthemodelhasawrongbias (thismayhappenin
practiceifawrongarchitectureordatapreprocessingmethodischosen, which, despitethemodeler’s
bestintentions, worsenstheperformance). Specifically, duringtrainingweaddedapenaltyterm
104 w2 tothetraininglossc, decreasedthesizeofthetestsetto1000 andused50%ofthetestdata
1
fortraining (thefinalpenalizedtraininglosswas0.25 with100%predictionaccuracyonthetraining
set). Notethatthesmalltrainingsetandthelargepenaltyonw yieldclassifiersthatareessentially
1
independentoftheonlyinterestingfeaturex (recallthatthetruelabelofapointxissgn (x )) and
1 1
overfittothenoiseinthedata, resultinginatruemodelrisk R(f)≈1/2.
A.3 Results
The results of the two experiments are shown in Figure 7, plotted against different perturbation
strengths: theleftcolumncorrespondstothefirstexperimentwhiletherightcolumntothesecond.
Thefirstrowpresentsthep-valuesforrejectingtheindependencehypothesis, calculatedbyrepeating
theexperiment (samplingdataandtrainingtheclassifier)100 timesandapplyingthesingle-model
(Section3, labelledas N =1 intheplots) and N-model (Section3.3, labelledas N =2,10,25,100
intheplots) independencetest, andtakingtheaverageovermodels (ormodelsetsofsize N) foreach
ε. Wealsoplotempirical95%two-sidedconfidenceintervals (N ≤2) or, duetolimitednumberof
p-valuesavailableafterdividing100 runsintodisjointbinsofsize N ≥10, rangesbetweenminimum
andmaximumvalue (N = 10,25). Forallmethodsofdetectingdependence, itcanbeseenthat
for the independent case the test is correctly not able to reject the independence hypothesis (the
averagep-valueisverycloseto1, althoughinsomerunsitcandroptoaslowas0.5). Ontheother
hand, for10≤ε≤50, thenon-independentmodelfailedtheindependencetestatconfidencelevel
1−δ ≈100%, hence, inthisrangeofεourindependencetestreliablydetectsoverfitting.
Infact, itiseasytoarguethatourtestshouldonlyworkforalimitedrangeofε, thatis, itshould
notrejectindependencefortoosmallortoolargevaluesofε. Firstweconsiderthecaseofsmall
εvalues. Noticethatexceptforpointsg (x)ε-close (in L -norm) tothetruedecisionboundaryor
2
thedecisionboundaryoff, g (x) isinvertible: ifg (x) iscorrectlyclassifiedandisε-awayfromthe
truedecisionboundary, thereisexactlyonepoint, x, whichistranslatedtog (x), whileifg (x) is
incorrectlyclassifiedandε-awayfromthedecisionboundaryoff, notranslationleadstog (x) and
x=g (x);anyotherpointsareε-closetothedecisionboundaryofeitherf orf∗. Thus, sinceρis
bounded, g (x) isinvertibleonasetofatleast1−O(ε) probability (accordingtoρ). Whenε→0,
g (x) → x, andsoρ(g (x)) → ρ(x) forallpointsxwith|x | =(cid:54) 0.025(sinceρiscontinuousinall
1
suchx), implyingh (g (x))≈1 onthesepoints. Italsofollowsthat L(f, x)(cid:54)=L(f, g (x)) canonly
g
5 Thelargenumberoftestexamplesensuresthattherandomerrorintheempiricalerrorestimateisnegligible.
13

### 17. 1.0
0.8
0.6
0.4
0.2
0.0
10-2 10-1 100 101 102
†
eulav-p
1.0
N=1 0.8
N=2
0.6
N=10...

1.0
0.8
0.6
0.4
0.2
0.0
10-2 10-1 100 101 102
†
eulav-p
1.0
N=1 0.8
N=2
0.6
N=10
N=25 0.4
N=100 0.2
0.0
10-2 10-1 100 101 102
†
eulav-p
N=1
N=2
N=10
N=25
N=100
1.0
0.8
0.6
0.4
0.2
0.0
10-2 10-1 100 101 102
†
rorre
1.0
Rg (f) 0.8
Rc S(f) 0.6
c 0.4
0.2
0.0
10-2 10-1 100 101 102
†
rorre
Rg (f)
Rc S(f)
Rc (f)
1.0
0.8
0.6
0.4
0.2
0.0
10-2 10-1 100 101 102
†
gh
egareva
1.0
0.8
0.6
0.4
original misclassified
adversarial 0.2
0.0
10-2 10-1 100 101 102
†
gh
egareva
original misclassified
adversarial
1.0
0.8
0.6
0.4
0.2
0.0
10-2 10-1 100 101 102
†
rorre
1.0
Rg (f) 0.8
Rc S(f) 0.6
Rc S0 (f) 0.4
c 0.2
0.0
10-2 10-1 100 101 102
†
rorre
Rg (f)
Rc S(f)
Rc S0 (f)
Rc (f)
100
80
60
40
20
0
0.0 0.2 0.4 0.6 0.8 1.0
p-value
tnuoc
100
δ(†=0.1) 80
δ(†=5) 60
δ(†=20)
40
20
0
0.0 0.2 0.4 0.6 0.8 1.0
p-value
tnuoc
δ(†=0.1)
δ(†=5)
δ(†=6)
δ(†=20)
Figure 7: Risk and overfitting metrics for a synthetic problem with linear classifiers as a function of the
perturbationstrengthsε(logscale). Left:unbiasedmodeltestedonalarge, independenttestset (inthiscase
R(cid:98)S (f) ≈ R(cid:98) g (f) ≈ R(f)); right: trainedmodeloverfittedtothetestset (R(cid:98)S (f) ≤ R(cid:98) g (f) whilebothare
smaller than R(f)). First row: Average p-value δ for the pairwise independence test with over 100 runs
(N =1) orthe N-modelindependencetest (N >1).Theboundsplottedareeitherempirical95%two-sided
confidenceintervals (N ≤ 2) orrangesbetweenminimumandmaximumvalue (N = 10,25). Secondrow:
Empiricaltwo-sided97.5%confidenceintervalsfortheempiricaltesterrorrate R(cid:98)S (f) andtheadversarialrisk
estimate R(cid:98) g (f).Ontheleft, R(f)≈R(cid:98)S (f), while R(f) isshownseparatelyontheright. Thirdrow:Average
densities (Radon-Nikodymderivatives) fororiginallymisclassifiedpointsandforthenewdatapointsobtained
bysuccessfuladversarialtransformations (withempirical97.5%two-sidedconfidenceintervals).Fourthrow:
Theempiricaltesterrorrate R(cid:98)S (f) andtheadversarialriskestimate R(cid:98) g (f) forasinglerealizationwith97.5%
two-sidedconfidenceintervalscomputedfrom Bernstein’sinequality, theadversarialerrorrate R(cid:98)S(cid:48)(f), andthe
expectederror R(f)(ontheright, ontheleft R(f)≈R(cid:98)S (f)).Fifthrow:Histogramsofp-valuesforselectedε
valuesover100 runs.
14

### 18. happentoasetofpointswithan O(ε)ρ-probability. Thismeansthat L(f, g (x)) h (g (x)...

happentoasetofpointswithan O(ε)ρ-probability. Thismeansthat L(f, g (x)) h (g (x))≈L(f, x)
g
onasetof1−O(ε)ρ-probability, andforthesepoints T (x)=L(f, g (x)) h (g (x))−L(f, x)≈0.
g g
Thus, T (X) ≈ 0 withρ-probability1−O(ε). Unlessthetestset S isconcentratedinlargepart
g
onthesetofremainingpointswith O(ε)ρ-probability, theteststatistic|T (f)|=O(ε) withhigh
S, g
probabilityandourmethodwillnotrejecttheindependencehypothesisforε→0.
When ε is large (ε → ∞), notice that for any point x with non-vanishing probability (i.e., with
ρ(x)>cforsomec>0), ifg (x)(cid:54)=xthanρ(g (x))≈0.Therefore, forsuchanx, if L(f, x)=0 and
L(f, g (x))=1, h (g (x))=ρ(g (x))/(ρ(x)+ρ(g (x)))≈0, andso T (x)≈0(if L(f, g (x))=0,
g g
we trivially have T (x) = 0). If L(f, x) = 1, we have g (x) = x. If g is invertible at x then
g
h (x) = 1 and T (x) = 0. If g is not invertible, then there is another x (cid:48) such that g (x (cid:48)) = x;
g g
however, ifρ(x)>cthenρ(x (cid:48))≈0(sinceεislarge), andsoh (g (x))=ρ(x)/(ρ(x)+ρ(x (cid:48)))≈1,
g
giving T (x) ≈ 0. Therefore, forlargeε, T (X) ≈ 0 withhighprobability (i.e., forpointswith
g g
ρ(x)>c), sotheindependencehypothesiswillnotberejectedwithhighprobability.
Tobetterunderstandthebehaviorofthetest, thesecondrowof Figure7 showstheempiricaltesterror
rate R(cid:98)S (f), the (unadjusted) adversarialerrorrate R(cid:98)S(cid:48) (f), andtheadversarialriskestimate R(cid:98) g (f),
togetherwiththeirconfidenceintervals. Forthenon-independentmodel, wealsoshowtheexpected
error R(f)(estimatedoveralargeindependenttestset), whileitisomittedfortheindependentmodel
whereitapproximatelycoincideswithboth R(cid:98)S (f) and R(cid:98) g (f).Whilethereweightedadversarialerror
estimate R(cid:98) g (f) remainsthesameforallperturbationsincaseofanindependenttestset (leftcolumn),
theadversarialerrorrate R(cid:98)S(cid:48) (f) variesalotforboththedependentandindependenttestsets. For
example, in the case when the test samples and the model f are not independent, it undershoots
the true error for ε < 10 and overshoots it for larger perturbations. For very large perturbations
(εcloseto100), thebehaviorof R(cid:98)S(cid:48) (f) dependsonthemodelf: intheindependentcase R(cid:98)S(cid:48) (f)
decreasesbackto R(cid:98)S (f) becausesuchlargeperturbationsincreasinglyoftenchangethetruelabel
oftheoriginalexample, solessandlessadversarialpointsaregenerated. Inthecasewhenthedata
andthemodelarenotindependent (rightcolumn), theadversarialperturbationsarealmostalways
successful (i.e., leadtoavalidadversarialexampleformostoriginallycorrectlyclassifiedpoints),
yieldinganadversarialerrorrateclosetooneforlargeenoughperturbations. Thisisbecausethe
decisionboundaryoff isalmostorthogonaltothetruedecisionboundary, andsotheadversarial
perturbationsareparallelwiththetrueboundary, almostneverchangingthetruelabelofapoint.
Theplotsofthedensities (Radon-Nikodymderivatives), giveninthethirdrowof Figure7, show
howthechangeintheirvaluescompensatetheincreaseoftheadversarialerrorrate R(cid:98)S(cid:48) (f): inthe
independentcase, theeffectiscompletelyeliminatedyieldinganunbiasedadversarialerrorestimate
R(cid:98) g (f), whichisessentiallyconstantoverthewholerangeofε(asshowninthefirstrow), whilein
thenon-independentcasethesimilardensitiesdonotbringbacktheadversarialerrorrate R(cid:98)S(cid:48) (f) to
thetesterrorrate R(cid:98)S (f), allowingthetesttodetectoverfitting. Notethatthedensitiesexhibitsimilar
trends (andvalues) inbothcases, drivenbythedependenceoftypicalvaluesoftheρ(x)/ρ(g (x))
ratioontheperturbationstrengthεfororiginallymisclassifedpoints (L(f, x)=1) andforsuccessful
adversarialexamples (i.e., L(f, x)=0 and L(f, g (x))=1).
To compare the behavior of our improved, pairwise test and the basic version, the fourth row of
Figure7 depictsasinglerealizationoftheexperimentswherethe97.5%confidenceintervals (as
computedfrom Bernstein’sinequality) areshownfortheestimates. Fortheindependentcase, the
confidenceintervalsof R(cid:98)S (f) and R(cid:98) g (f) overlapforallε, andthusthebasictestisnotabletodetect
overfitting. Inthenon-independentcase, theconfidenceintervalsoverlapforε=10 andε=75, thus
thebasictestisnotabletodetectoverfittingwithata95%confidencelevel, whiletheimprovedtest
(secondrow) isabletorejecttheindependencehypothesisfortheseεvaluesatthesameconfidence
level.
Finally, inthefifthrowof Figure7 weplottedthehistogramsoftheempiricaldistributionofp-values
forbothmodels, over100 independentruns (betweentheruns, allthedatawasregeneratedandthe
modelswereretrained). Forε=0.1,5,20, theyconcentrateheavilyoneitherδ =0 orδ =1, and
haveverythintailsextendingfartowardstheoppositeendofthe[0,1]interval. Thisexplainsthe
surprisinglywide95%confidenceintervalsforp-valuesplottedinthefirstrow. Inparticular, thefact
thatsomep-valuesfortheindependentmodelareaslowas0.5 doesnotmeantheindependencetest
isnotreliable, becausealmostallcalculatedδvaluesarecloseorequalto1, andthefewoutliersare
15

### 19. 12
10
8
6
4
2
0
0.0 0.2 0.4 0.6 0.8 1.0
p-value
FDP
5
δ(N=1) 4
δ(N=2)
3
δ(N=10)
...

12
10
8
6
4
2
0
0.0 0.2 0.4 0.6 0.8 1.0
p-value
FDP
5
δ(N=1) 4
δ(N=2)
3
δ(N=10)
2
1
0
0.0 0.2 0.4 0.6 0.8 1.0
p-value
FDP
δ(N=1)
δ(N=2)
δ(N=10)
independentmodel,ε=10 non-independentmodel,ε=6
Figure8:Histogramsofp-valuesfrom N-model (N =1,2,10) independencetestsforbothsyntheticmodels
andselectedεvalues, over100 runs.
acombinedconsequenceofthefinitesamplesizeandtheeffectivenessofthe AEG.Theadditional
ε=6 histogramforthenon-independentmodelillustratesaregimewhichisinbetweenthesingle-
modelpairwisetest (Section3) completelyfailingtorejecttheindependencehypothesisandclearly
rejectingit.
Toverifyexperimentallywhetherthe N-modelindependencetestcanbeamorepowerfuldetectorof
overfittingthanthesingle-modelversion, in Figure8(rightpanel) weplottedp-valuehistogramsfor
N =1,2,10 fortheintermediate AEGstrengthε=6 appliedtothenon-independentmodelover100
trainingruns. Indeed, as N increases, theconcentrationofp-valuesaroundinthelow (δ ≤0.2) range
increases. For N >10 wedidnothaveenoughvaluestoplotahistogram: for N =25 weobtained
δ =0.1851,0.1599,0.0661 and0.1941, whilefor N =100 thep-valueis0.1153. Theincreaseof
thetestpowerbecomesapparentwhenwecomparethelastvaluewiththemeanofp-valuesobtained
bytestingeverytrainingrunseparately, equal0.5984, andthemedian0.6385.
For comparison, we also plotted in Figure 8 (left panel) the corresponding histograms for the
independentmodelandaslightlyhigherattackstrength,ε=10, atwhichtheindependencetestsfails
fortheoverfittedmodelevenwithoutaveraging (see Figure7, firstrow, rightpanel). Thehistograms
areallclusteredintheδregioncloseto1, indicatingthatthe N-modeltestisnotoverlypessimistic.
B Translational AEGsforimageclassificationmodels
Forimageclassificationweconsidertwotranslationvariantsthatareusedinconstructingatransla-
tional AEG.Foreverycorrectlyclassifiedimagex, weconsidertranslationsfrom V (forsomeε),
ε
choosingg (x) fromtheset G(x) = {τ (x) : v ∈ V }∪{x}. Ifalltranslationsresultincorrectly
v ε
classifiedexamples, wesetg (x) = x. Otherwise, weuseoneoftwopossiblewaystoselectg (x)
(andwecalltheresultingpointssuccessfuladversarialexamples):
• Strongest perturbation: Assuming the number of classes is K, let l (f, x) ∈ RK denote
thevectorofthe K classlogitscalculatedbythemodelf forimagex, andletl (f, x)=
exc
max l (f, x)−l (f, x). Wedefine
0≤i<K i y
g (x)=argmax l (f, x (cid:48)),
strongest x (cid:48)∈G(x) exc
withtiesbrokendeterministicallybychoosingthefirsttranslationfromthecandidateset,
goingtoptobottomandlefttorightinrow-majororder. Thus, hereweseekanon-identical
“neighbor”thatcausestheclassifiertoerrthemost, reachablefromxbytranslationswithin
amaximumrangeε.
• Nearest misclassified neighbor: Here we aim to find the nearest image in G(x) that is
misclassified. Thatis, lettingd (x, x (cid:48))=(cid:107) v (cid:107) ifx (cid:48) =τ (x) and∞otherwise, wedefine
2 v
g (x):=argmin d (x, x (cid:48))
nearest x (cid:48)∈G(x), L(f, x (cid:48))=1
withtiesbrokendeterministicallyasabove.
Thetwoperturbationvariantsaresuccessfulonexactlythesamesetofimages, hencetheyleadto
thesameadversarialerrorrates R(cid:98)S(cid:48) (f). However, theyarecharacterizedbydifferentvaluesofthe
densityh
g
and, consequently, yielddifferentadversarialriskestimates R(cid:98) g (f) andassociatedp-values
16

### 20. fortheindependencetest. Themaindifferencebetweenthemisthatthe“strongest”versioni...

fortheindependencetest. Themaindifferencebetweenthemisthatthe“strongest”versionismore
likely to map multiple images to the same adversarial example, thus decreasing the densities for
successfuladversarialexamplesand, counterintuitively, increasingthemfororiginallymisclassified
points (astheirneighborsarelesslikelytobemappedtothesepoints).
Tobetterseetheeffectofadversarialperturbations, wealsoconsidertworandombaselinesthatdo
nottakeintoaccountthesuccessofatranslationingeneratingmisclassifiedpoints: g (x) is
random
chosenuniformlyatrandomfrom G(x)\{x}, andg (x) ischosenuniformlyatrandomfrom
random2
G(x).
B.1 Maximumtranslations
In practice, translating an image is not always simple, as the new image has to be padded with
newpixels. When (central) cropsofalargerimageareused (asistypicalfor Image Netclassifiers),
translationscaneasilybeimplementedaslongastheresultingnewcroppingwindowstayswithinthe
originalimageboundaries. Evenifanimagecanbetranslatedbyavectorv, thislimitsourabilityto
computeh (x (cid:48)) fortheadversarialimagex (cid:48) by (8) or (9) forg org . Indeed, ifanimage
g strongest nearest
xisshiftedbyv ∈V togenerateadversarialexamplex (cid:48), weneedtoexaminetranslationsofx (cid:48) with
ε
vectorsin V tofindtheneighborsx (cid:48)(cid:48) ofx (cid:48) potentiallycontributington (x (cid:48)) whencomputingh (x (cid:48)).
ε g
Finallyweneedtoconsidertranslationsofx (cid:48)(cid:48) withvectorsin V todeterminetheexactvaluethey
ε
contribute, thatis, tocomputetheexactprobabilitiesin (9)(see Figure9 foranillustration). Thus,
tobeabletocomputethedensityh fortheadversarialpointsobtainedbytranslationsfrom V , we
g ε
mightneedtobeabletoperformtranslationswithin V .
3ε
ε
x x' x''
Figure9:Imagetranslationswhichneedtobeconsideredforatranslational AEGwithε=3.Thered, blueand
greenballsrepresentthecenteroftheoriginalimagex, adversarialexamplex (cid:48) =g (x) andanotherimagex (cid:48)(cid:48)
contributingtoρ (x (cid:48)), respectively, whilethesemi-translucentsquaresofcorrespondingcolorsrepresentthe
g
possibletranslationswhichneedtobeconsideredforeachofx, x (cid:48) andx (cid:48)(cid:48).Solidlightgreyarrowsrepresentthe
relationshipsx (cid:48) =g (x) andx (cid:48) =g (x (cid:48)(cid:48)).Finally, thedashedarrowandthesemi-translucentgreyballrepresent
analternativemapping, whichhastoberuledoutwhilecalculatingthevalueofg (x (cid:48)(cid:48)) and, consequently, of
h (x (cid:48)).Itiseasytoseethatthecoloredsquares (whichcontainthetranslationsneedingtobeevaluated) extend
g
asfaras3εfromtheoriginalimagex.
17

### 21. sample S(cid:48) = {(X(cid:48), Y ),...,(X(cid:48) , Y )} obtained through an AE...

sample S(cid:48) = {(X(cid:48), Y ),...,(X(cid:48) , Y )} obtained through an AEG g, where X(cid:48) = g (X ) with
1 1 m m i i
X ,..., X drawnindependentlyfrom P and Y =f∗(X ). Sincegsatisfies (G1) bydefinition, the
1 m i i
originalexample X andthecorrespondingadversarialexample X(cid:48) havethesamelabel Y . Recalling
i i i
thath =d P/d P ≤1 on E ={x∈X : L(f, x)=1}, onecaneasilyshowthattheimportance
g g
weightedadversarialestimate
m
1 (cid:88)
R(cid:98) g (f)=
m
I(f (X
i
(cid:48))(cid:54)=Y
i
) h
g
(X
i
(cid:48)) (2)
i=1
obtainedfrom (1) fortheadversarialsample S(cid:48) hassmallervariancethanthatoftheempiricalaverage
R(cid:98)S (f), whilebothareunbiasedestimatesof R(f). Recallthatboth R(cid:98) g (f) and R(cid:98)S (f) areunbiased
estimatesof R(f) withexpectation E[R(cid:98) g (f)]=E[R(cid:98)S (f)]=R(f), andso
V[R(cid:98) g (f)]=
m
1 (cid:0)E[L(f, g (X))2 h
g
(g (X))2]−R(f)2(cid:1)
≤
m
1 (cid:0)E[L(f, g (X)) h
g
(g (X))]−R2(f) (cid:1) =
m
1 (cid:0) R(f)−R2(f) (cid:1) =V[R(cid:98)S (f)].
Intuitively, themoresuccessfulthe AEGis (i.e., themoreclassificationerroritinduces), thesmaller
thevarianceoftheestimate R(cid:98) g (f) becomes.
3 Detectingoverfitting
Inthissectionweshowhowtheriskestimatesintroducedintheprevioussectioncanbeusedtotest
theindependencehypothesisthat
(H) thesample S andthemodelf areindependent.
If (H) holds, E[R(cid:98) g (f)] = E[R(cid:98)S (f)] = R(f), and so the difference T
S, g
(f) = R(cid:98) g (f)−R(cid:98)S (f)
is expected to be small. On the other hand, if f is overfitted to the dataset S (in which case
R(cid:98)S (f)<R(f)), weexpect R(cid:98)S (f) and R(cid:98) g (f) tobehavedifferently (thelatterbeinglesssensitiveto
overfitting) since (i)R(cid:98) g (f) dependsalsoonexamplespreviouslyunseenbythetrainingprocedure;
(ii) theadversarialtransformationg aimstoincreasetheloss, counteringtheeffectofoverfitting;
(iii) especially in high dimensional settings, in case of overfitting one may expect that there are
misclassified points very close to the decision boundary of f which can be found by a carefully
designed AEG. Therefore, intuitively, (H) can be rejected if |T (f)| exceeds some appropriate
S, g
threshold.
3.1 Testbasedonconfidenceintervals
The simplest way to determine the threshold is based on constructing confidence intervals for
these estimator based on concentration inequalities. Under (H), standard concentration inequal-
ities, such as the Chernoff or empirical Bernstein bounds [3], can be used to quantify how
fast R(cid:98)S and R(cid:98) g (f) concentrate around the expected error R(f). In particular, we use the
following empirical Bernstein bound [22]: Let σ¯
S
2 = (1/m) (cid:80) m
i=1
(L(f, X
i
) − R(cid:98)S (f))2 and
σ¯
g
2 = (1/m) (cid:80) m
i=1
(L(f, g (X
i
)) h
g
(g (X
i
))−R(cid:98) g (f))2 denote the empirical variance of L(f, X
i
)
and L(f, g (X )) h (g (X )), respectively. Then, forany0<δ ≤1, withprobabilityatleast1−δ,
i g i
|R(cid:98)S (f)−R(f)|≤B(m,σ¯
S
2,δ,1), (3)
(cid:113)
where B(m,σ2,δ,1)= 2σ2 ln (3/δ) + 3 ln (3/δ) andweusedthefactthattherangeof L(f, x) is1
m m
(thelastparameterof Bistherangeoftherandomvariablesconsidered). Similarly, withprobability
atleast1−δ,
|R(cid:98) g (f)−R(f)|≤B(m,σ¯
g
2,δ,1). (4)
It follows trivially from the union bound that if the independence hypothesis (H) holds, the
abovetwoconfidenceintervals[R(cid:98)S (f)−B(m,σ¯
S
2,δ,1), R(cid:98)S (f)+B(m,σ¯
S
2,δ,1)]and[R(cid:98) g (f)−
5


---

## Raw Markitdown Extraction (full text)

Detecting Overﬁtting via Adversarial Examples

Roman Werpachowski

András György

Csaba Szepesvári

DeepMind, London, UK
{romanw,agyorgy,szepi}@google.com

Abstract

The frequent reuse of test sets in popular benchmark problems raises doubts about
the credibility of reported test-error rates. Verifying whether a learned model is
overﬁtted to a test set is challenging as independent test sets drawn from the same
data distribution are usually unavailable, while other test sets may introduce a
distribution shift. We propose a new hypothesis test that uses only the original test
data to detect overﬁtting. It utilizes a new unbiased error estimate that is based
on adversarial examples generated from the test data and importance weighting.
Overﬁtting is detected if this error estimate is sufﬁciently different from the original
test error rate. We develop a specialized variant of our test for multiclass image
classiﬁcation, and apply it to testing overﬁtting of recent models to the popular
ImageNet benchmark. Our method correctly indicates overﬁtting of the trained
model to the training set, but is not able to detect any overﬁtting to the test set, in
line with other recent work on this topic.

1

Introduction

Deep neural networks achieve impressive performance on many important machine learning bench-
marks, such as image classiﬁcation [18, 19, 28, 27, 16], automated translation [2, 31] or speech
recognition [9, 15]. However, the benchmark datasets are used a multitude of times by researchers
worldwide. Since state-of-the-art methods are selected and published based on their performance
on the corresponding test set, it is typical to see results that continuously improve over time; see,
e.g., the discussion of Recht et al. [25] and Figure 1 for the performance improvement of classiﬁers
published for the popular CIFAR-10 image classiﬁcation benchmark [18].

This process may naturally lead to models over-
ﬁtted to the test set, rendering test error rate
(the average error measured on the test set) an
unreliable indicator of the actual performance.
Detecting whether a model is overﬁtted to the
test set is challenging, since independent test
sets drawn from the same data distribution are
generally not available, while alternative test
sets often introduce a distribution shift.

To estimate the performance of a model on un-
seen data, one may use generalization bounds
to get upper bounds on the expected error rate.
The generalization bounds are also applicable
when the model and the data are dependent (e.g.,
for cross validation or for error estimates based
on the training data or the reused test data), but they usually lead to loose error bounds. Therefore,
although much tighter bounds are available if the test data and the model are independent, comparing

Figure 1: Accuracy of image classiﬁers on the CIFAR-10
test set, by year of publication (data from [25]).

33rd Conference on Neural Information Processing Systems (NeurIPS 2019), Vancouver, Canada.

20102012201420162018year0.800.850.900.951.00accuracyconﬁdence intervals constructed around the training and test error rates leads to an underpowered test
for detecting the dependence of a model on the test set. Recently, several methods have been proposed
that allow the reuse of the test set while keeping the validity of test error rates [10]. However, these
are intrusive: they require the user to follow a strict protocol of interacting with the test set and are
thus not applicable in the more common situation when enforcing such a protocol is impossible.

In this paper we take a new approach to the challenge of detecting overﬁtting of a model to the test
set, and devise a non-intrusive statistical test that does not restrict the training procedure and is based
on the original test data. To this end, we introduce a new error estimator that is less sensitive to
overﬁtting to the data; our test rejects the independence of the model and the test data if the new
error estimate and the original test error rate are too different. The core novel idea is that the new
estimator is based on adversarial examples [14], that is, on data points1 that are not sampled from the
data distribution, but instead are cleverly crafted based on existing data points so that the model errs
on them. Several authors showed that the best models learned for the above-mentioned benchmark
problems are highly sensitive to adversarial attacks [14, 23, 30, 6, 7, 24]: for instance, one can often
create adversarial versions of images properly classiﬁed by a state-of-the-art model such that the
model will misclassify them, yet the adversarial perturbations are (almost) undetectable for a human
observer; see, e.g., Figure 2, where the adversarial image is obtained from the original one by a
carefully selected translation.

scale, weighing machine

The adversarial (error) estimator proposed in
this work uses adversarial examples (generated
from the test set) together with importance
weighting to take into account the change in
the data distribution (covariate shift) due to the
adversarial transformation. The estimator is un-
biased and has a smaller variance than the stan-
dard test error rate if the test set and the model
are independent.2 More importantly, since it is
based on adversarially generated data points, the
adversarial estimator is expected to differ sig-
niﬁcantly from the test error rate if the model
is overﬁtted to the test set, providing a way to
detect test set overﬁtting. Thus, the test error
rate and the adversarial error estimate (calcu-
lated based on the same test set) must be close if the test set and the model are independent, and are
expected to be different in the opposite case. In particular, if the gap between the two error estimates
is large, the independence hypothesis (i.e., that the model and the test set are independent) is dubious
and will be rejected. Combining results from multiple training runs, we develop another method to
test overﬁtting of a model architecture and training procedure (for simplicity, throughout the paper
we refer to both together as the model architecture). The most challenging aspect of our method is to
construct adversarial perturbations for which we can calculate importance weights, while keeping
enough degrees of freedom in the way the adversarial perturbations are generated to maximize power,
the ability of the test to detect dependence when it is present.

Figure 2: Adversarial example for the ImageNet dataset
generated by a (5, −5) translation: the original example
(left) is correctly classiﬁed by the VGG16 model [27] as
“scale, weighing machine,” the adversarially generated
example (right) is classiﬁed as “toaster,” while the image
class is the same for any human observer.

toaster

To understand the behavior of our tests better, we ﬁrst use them on a synthetic binary classiﬁcation
problem, where the tests are able to successfully identify the cases where overﬁtting is present. Then
we apply our independence tests to state-of-the-art classiﬁcation methods for the popular image
classiﬁcation benchmark, ImageNet [8]. As a sanity check, in all cases examined, our test rejects
(at conﬁdence levels close to 1) the independence of the individual models from their respective
training sets. Applying our method to VGG16 [27] and Resnet50 [16] models/architectures, their
independence to the ImageNet test set cannot be rejected at any reasonable conﬁdence. This is in
agreement with recent ﬁndings of [26], and provides additional evidence that despite of the existing
danger, it is likely that no overﬁtting has happened during the development of ImageNet classiﬁers.

The rest of the paper is organized as follows: In Section 2, we introduce a formal model for error
estimation using adversarial examples, including the deﬁnition of adversarial example generators.

1Throughout the paper, we use the words “example” and “point” interchangeably.
2Note that the adversarial error estimator’s goal is to estimate the error rate, not the adversarial error rate (i.e.,

the error rate on the adversarial examples).

2

The new overﬁtting-detection tests are derived in Section 3, and applied to a synthetic problem in
Section 4, and to the ImageNet image classiﬁcation benchmark in Section 5. Due to space limitations,
some auxiliary results, including the in-depth analysis of our method on the synthetic problem, are
relegated to the appendix.

2 Adversarial Risk Estimation

We consider a classiﬁcation problem with deterministic (noise-free) labels, which is a reasonable
assumption for many practical problems, such as image recognition (we leave the extension of our
method to noisy labels for future work). Let X ⊂ RD denote the input space and Y = {0, . . . , K −1}
the set of labels. Data is sampled from the distribution P over X , and the class label is determined
by the ground truth function f ∗ : X → Y. We denote a random vector drawn from P by X, and its
corresponding class label by Y = f ∗(X). We consider deterministic classiﬁers f : X → Y. The
performance of f is measured by the zero-one loss: L(f, x) = I(f (x) (cid:54)= f ∗(x)),3 and the expected
error (also known as the risk or expected risk in the learning theory literature) of the classiﬁer f is
deﬁned as R(f ) = E[I(f (X) (cid:54)= Y )] = (cid:82)
X L(f, x)dP(x).
Consider a test dataset S = {(X1, Y1) . . . , (Xm, Ym)} where the Xi are drawn from P independently
of each other and Yi = f ∗(Xi). In the learning setting, the classiﬁer f usually also depends on some
randomly drawn training data, hence is random itself. If f is (statistically) independent from S, then
L(f, X1), . . . , L(f, Xm) are i.i.d., thus the empirical error rate

(cid:98)RS(f ) =

1
m

m
(cid:88)

i=1

L(f, Xi) =

1
m

m
(cid:88)

i=1

I(f (Xi) (cid:54)= Yi)

is an unbiased estimate of R(f ) for all f ; that is, R(f ) = E[ (cid:98)RS(f )|f ]. If f and S are not indepen-
dent, the performance guarantees on the empirical estimates available in the independent case are
signiﬁcantly weakened; for example, in case of overﬁtting to S, the empirical error rate is likely to be
much smaller than the expected error.

Another well-known way to estimate R(f ) is to use importance sampling (IS) [17]: instead of
sampling from the distribution P, we sample from another distribution P (cid:48) and correct the estimate
by appropriate reweighting. Assuming P is absolutely continuous with respect to P (cid:48) on the set
X L(f, x)dP(x) = (cid:82)
E = {x ∈ X : L(f, x) (cid:54)= 0}, R(f ) = (cid:82)
E L(f, x)h(x)dP (cid:48)(x), where h = dP
dP (cid:48)
is the density (Radon-Nikodym derivative) of P with respect to P (cid:48) on E (h can be deﬁned to have
arbitrary ﬁnite values on X \ E). It is well known that the the corresponding empirical error estimator

(cid:98)R(cid:48)

S(cid:48)(f ) =

1
m

m
(cid:88)

L(f, X (cid:48)

i)h(X (cid:48)

i) =

m
(cid:88)

I(f (X (cid:48)

i) (cid:54)= Y (cid:48)

i )h(X (cid:48)
i)

(1)

1
m

i=1

1, Y (cid:48)

m, Y (cid:48)

1 ), . . . , (X (cid:48)

i=1
obtained from a sample S(cid:48) = {(X (cid:48)
(i.e., E[ (cid:98)RS(cid:48)(f )|f ] = R(f )) if f and S(cid:48) are independent.
S(cid:48) is minimized if P (cid:48) is the so-called zero-variance IS distribution, which is
The variance of (cid:98)R(cid:48)
supported on E with h(x) = R(f )
L(f,x) for all x ∈ E (see, e.g., [4, Section 4.2]). This suggest that an
effective sampling distribution P (cid:48) should concentrate on points where f makes mistakes, which also
facilitates that (cid:98)R(cid:48)
S(cid:48)(f ) become large if f is overﬁtted to S and hence (cid:98)RS(f ) is small. We achieve this
through the application of adversarial examples.

m)} drawn independently from P (cid:48) is unbiased

2.1 Generating adversarial examples

In this section we introduce a formal framework for generating adversarial examples. Given a
classiﬁcation problem with data distribution P and ground truth f ∗, an adversarial example generator
(AEG) for a classiﬁer f is a (measurable) mapping g : X → X such that

(G1) g preserves the class labels of the samples, that is, f ∗(x) = f ∗(g(x)) for P-almost all x;
(G2) g does not change points that are incorrectly classiﬁed by f , that is, g(x) = x if f (x) (cid:54)=

f ∗(x) for P-almost all x.

3For an event B, I(B) denotes its indicator function: I(B) = 1 if B happens and I(B) = 0 otherwise.

3

Figure 3: Generating adversarial examples. The top row depicts the original dataset S, with blue and orange
points representing the two classes. The classiﬁer’s prediction is represented by the color of the striped areas
(checkmarks and crosses denote if a point is correctly or incorrectly classiﬁed). The arrows show the adversarial
transformations via the AEG g, resulting in the new dataset S(cid:48); misclassiﬁed points are unchanged, while some
correctly classiﬁed points are moved, but their original class label is unchanged. If the original data distribution is
uniform over S, the transformation g is density preserving, but not measure preserving: after the transformation
the two rightmost correctly classiﬁed points in each class have probability 0, while the leftmost misclassiﬁed
point in each class has probability 3/16; hence, the density hg for the latter points is 1/3.

Figure 3 illustrates how an AEG works. In the literature, an adversarial example g(x) is usually
generated by staying in a small vicinity of the original data point x (with respect to, e.g., the 2- or the
max-norm) and assuming that the resulting label of g(x) is the same as that of x (see, e.g., [14, 6]).
This foundational assumption—which is in fact a margin condition on the distribution—is captured
in condition (G1). (G2) formalizes the fact that there is no need to change samples which are already
misclassiﬁed. Indeed, existing AEGs comply with this condition.

The performance of an AEG is usually measured by how successfully it generates misclassiﬁed
examples. Accordingly, we call a point g(x) a successful adversarial example if x is correctly
classiﬁed by f and f (g(x)) (cid:54)= f (x) (i.e., L(f, x) = 0 and L(f, g(x)) = 1).

In the development of our AEGs for image recognition tasks, we will make use of another condition.
For simplicity, we formulate this condition for distributions P that have a density ρ with respect
to the uniform measure on X , which is assumed to exist (notable cases are when X is ﬁnite, or
X = [0, 1]D or when X = RD; in the latter two cases the uniform measure is the Lebesgue measure).
The assumption states that the AEG needs to be density-preserving:

(G3) ρ(x) = ρ(g(x)) for P-almost all x.

Note that a density-preserving map may not be measure-preserving (the latter means that for all
measurable A ⊂ X , P(A) = P(g(A))).

We expect (G3) to hold when g perturbs its input by a small amount and if ρ is sufﬁciently smooth.
The assumption is reasonable for, e.g., image recognition problems (at least in a relaxed form,
ρ(x) ≈ ρ(g(x))) where we expect that very close images will have a similar likelihood as measured
by ρ. An AEG employing image translations, which satisﬁes (G3), will be introduced in Section 5.
Both (G1) and (G3) can be relaxed (to a soft margin condition or allowing a slight change in ρ, resp.)
at the price of an extra error term in the analysis that follows.

For a ﬁxed AEG g : X → X , let Pg be the distribution of g(X) where X ∼ P (Pg is known as the
pushforward measure of P under g). Further, let hg = dP
on E = {x : L(f, x) (cid:54)= 0} and arbitrary
dPg
otherwise. It is easy to see that, on E, hg(x) is well-deﬁned and hg ≤ 1. For any measurable A ⊂ E
Pg(A) = P(g(X) ∈ A) ≥ P(g(X) ∈ A, X ∈ E) = P(X ∈ A) = P(A)
where the second to last equality holds because g(X) = X for any X ∈ E under condition (G2).
Thus, P(A) ≤ Pg(A) for any measurable A ⊂ E, which implies that hg is well-deﬁned on E and
hg(x) ≤ 1 for all x ∈ E.

One may think that (G3) implies that hg(x) = 1 for all x ∈ E. However, this does not hold. For
example, if P is a uniform distribution, any g : X → supp P satisﬁes (G3), where supp P ⊂ X
denotes the support of the distribution P. This is also illustrated in Figure 3.

2.2 Risk estimation via adversarial examples

Combining the ideas of this section so far, we now introduce unbiased risk estimates based on
adversarial examples. Our goal is to estimate the error-rate of f through an adversarially generated

4

(cid:51)(cid:51)(cid:51)(cid:51)(cid:55)(cid:55)(cid:55)(cid:55)(cid:51)(cid:51)(cid:51)(cid:51)(cid:51)(cid:51)(cid:51)(cid:51)(cid:55)(cid:55)(cid:55)(cid:55)(cid:55)(cid:55)(cid:55)(cid:55)(cid:55)(cid:55)(cid:55)(cid:55)(cid:55)(cid:55)(cid:55)(cid:55)SS(cid:48)g1, Y1), . . . , (X (cid:48)

sample S(cid:48) = {(X (cid:48)
i = g(Xi) with
X1, . . . , Xm drawn independently from P and Yi = f ∗(Xi). Since g satisﬁes (G1) by deﬁnition, the
original example Xi and the corresponding adversarial example X (cid:48)
i have the same label Yi. Recalling
that hg = dP/dPg ≤ 1 on E = {x ∈ X : L(f, x) = 1}, one can easily show that the importance
weighted adversarial estimate

m, Ym)} obtained through an AEG g, where X (cid:48)

(cid:98)Rg(f ) =

1
m

m
(cid:88)

i=1

I(f (X (cid:48)

i) (cid:54)= Yi)hg(X (cid:48)
i)

(2)

obtained from (1) for the adversarial sample S(cid:48) has smaller variance than that of the empirical average
(cid:98)RS(f ), while both are unbiased estimates of R(f ). Recall that both (cid:98)Rg(f ) and (cid:98)RS(f ) are unbiased
estimates of R(f ) with expectation E[ (cid:98)Rg(f )] = E[ (cid:98)RS(f )] = R(f ), and so

V[ (cid:98)Rg(f )] =

≤

1
m
1
m

(cid:0)E[L(f, g(X))2hg(g(X))2] − R(f )2(cid:1)

(cid:0)E[L(f, g(X))hg(g(X))] − R2(f )(cid:1) =

1
m

(cid:0)R(f ) − R2(f )(cid:1) = V[ (cid:98)RS(f )] .

Intuitively, the more successful the AEG is (i.e., the more classiﬁcation error it induces), the smaller
the variance of the estimate (cid:98)Rg(f ) becomes.

3 Detecting overﬁtting

In this section we show how the risk estimates introduced in the previous section can be used to test
the independence hypothesis that

(H) the sample S and the model f are independent.

If (H) holds, E[ (cid:98)Rg(f )] = E[ (cid:98)RS(f )] = R(f ), and so the difference TS,g(f ) = (cid:98)Rg(f ) − (cid:98)RS(f )
is expected to be small. On the other hand, if f is overﬁtted to the dataset S (in which case
(cid:98)RS(f ) < R(f )), we expect (cid:98)RS(f ) and (cid:98)Rg(f ) to behave differently (the latter being less sensitive to
overﬁtting) since (i) (cid:98)Rg(f ) depends also on examples previously unseen by the training procedure;
(ii) the adversarial transformation g aims to increase the loss, countering the effect of overﬁtting;
(iii) especially in high dimensional settings, in case of overﬁtting one may expect that there are
misclassiﬁed points very close to the decision boundary of f which can be found by a carefully
designed AEG. Therefore, intuitively, (H) can be rejected if |TS,g(f )| exceeds some appropriate
threshold.

3.1 Test based on conﬁdence intervals

The simplest way to determine the threshold is based on constructing conﬁdence intervals for
these estimator based on concentration inequalities. Under (H), standard concentration inequal-
ities, such as the Chernoff or empirical Bernstein bounds [3], can be used to quantify how
fast (cid:98)RS and (cid:98)Rg(f ) concentrate around the expected error R(f ).
In particular, we use the
S = (1/m) (cid:80)m
i=1(L(f, Xi) − (cid:98)RS(f ))2 and
following empirical Bernstein bound [22]: Let ¯σ2
g = (1/m) (cid:80)m
i=1(L(f, g(Xi))hg(g(Xi)) − (cid:98)Rg(f ))2 denote the empirical variance of L(f, Xi)
¯σ2
and L(f, g(Xi))hg(g(Xi)), respectively. Then, for any 0 < δ ≤ 1, with probability at least 1 − δ,

| (cid:98)RS(f ) − R(f )| ≤ B(m, ¯σ2

S, δ, 1),

(3)

where B(m, σ2, δ, 1) =
and we used the fact that the range of L(f, x) is 1
(the last parameter of B is the range of the random variables considered). Similarly, with probability
at least 1 − δ,

+ 3 ln(3/δ)
m

(cid:113) 2σ2 ln(3/δ)
m

| (cid:98)Rg(f ) − R(f )| ≤ B(m, ¯σ2

g, δ, 1).

(4)

It follows trivially from the union bound that if the independence hypothesis (H) holds, the
above two conﬁdence intervals [ (cid:98)RS(f ) − B(m, ¯σ2
S, δ, 1)] and [ (cid:98)Rg(f ) −

S, δ, 1), (cid:98)RS(f ) + B(m, ¯σ2

5

g, δ, 1), (cid:98)RS(f ) + B(m, ¯σ2
B(m, ¯σ2
intersect with probability at least 1 − 2δ.

g, δ, 1)], which both contain R(f ) with probability at least 1 − δ,

On the other hand, if f and S are not independent, the performance guarantees (3) and (4) may be
violated and the conﬁdence intervals may become disjoint. If this is detected, we can reject the
independence hypothesis (H) at a conﬁdence level 1 − 2δ or, equivalently, with p-value 2δ. In other
words, we reject (H) if the absolute value of the difference of the estimates TS,g(f ) = (cid:98)Rg(f )− (cid:98)RS(f )
g, δ, 1) (note that E[TS,g(f ) = 0] if S and f are
exceeds the threshold B(m, ¯σ2
independent).

S, δ, 1) + B(m, ¯σ2

3.2 Pairwise test

A smaller threshold for |TS,g(f )|, and hence a more effective independence test, can be devised
if instead of independently estimating the behavior of (cid:98)RS and (cid:98)Rg(f ), one utilizes their apparent
correlation. Indeed, TS,g(f ) = (1/m) (cid:80)m
i=1 Ti,g(f ) where
Ti,g(f ) = L(f, g(Xi))hg(g(Xi)) − L(f, Xi)
and the two terms in Ti,g(f ) have the same mean and are typically highly correlated by the con-
struction of g. Thus, we can apply the empirical Bernstein bound [22] to the pairwise differences
Ti,g(f ) to set a tighter threshold in the test: if the independence hypothesis (H) holds (i.e., S and f
are independent), then for any 0 < δ < 1, with probability at least 1 − δ,

(5)

|TS,g(f )| ≤ B(m, ¯σ2

T , δ, U )

(6)

(cid:113) 2σ2 ln(3/δ)
m

with B(m, σ2, δ, U ) =
i=1(Ti(f ) − TS,g(f ))2 is
the empirical variance of the Ti,g(f ) terms and U = sup Ti,g(f ) − inf Ti,g(f ); we also used the fact
that the expectation of each Ti,g(f ), and hence that of TS,g(f ), is zero. Since hg ≤ 1 if L(f, x) = 1
(as discussed in Section 2.2), it follows that U ≤ 2, but further assumptions (such as g being density
preserving) can result in tighter bounds.

, where ¯σ2

+ 3U ln(3/δ)
m

T = (1/m) (cid:80)m

This leads to our pairwise dependence detection method:

if |TS,g(f )| > B(m, ¯σ2

T , δ, 2), reject (H) at a conﬁdence level 1 − δ (p-value δ).

For a given statistic (|TS,g(f )|, ¯σ2
T ), the largest conﬁdence level (smallest p-value) at which (H) can
be rejected can be calculated by setting the value of the statistic |TS,g(f )| − B(m, ¯σ2
T , δ, 2) to zero
and solving for δ. This leads to the following formula for the p-value (if the solution is larger than 1,
which happens when the bound (6) is loose, δ is capped at 1):
√

(cid:17)(cid:27)

T +3U |TS,g(f )|−¯σT

¯σ2

T +6U |TS,g(f )|

δ = min

(cid:26)

1, 3e− m

9U 2

(cid:16)

¯σ2

.

(7)

Note that in order for the test to work well, we not only need the test statistic TS,g(f ) to have a
small variance in case of independence (this could be achieved if g were the identity), but we also
need the estimators (cid:98)RS(f ) and (cid:98)Rg(f ) behave sufﬁciently differently if the independence assumption
is violated. The latter behavior is encouraged by stronger AEGs, as we will show empirically in
Section 5.2 (see Figure 5 in particular).

3.3 Dependence detector for randomized training

The dependence between the model and the test set can arise from (i) selecting the “best” random
seed in order to improve the test set performance and/or (ii) tweaking the model architecture (e.g.,
neural network structure) and hyperparameters (e.g., learning-rate schedule). If one has access to a
single instance of a trained model, these two sources cannot be disentangled. However, if the model
architecture and training procedure is fully speciﬁed and computational resources are adequate, it
is possible to isolate (i) and (ii) by retraining the model multiple times and calculating the p-value
for every training run separately. Assuming N models, let fj, j = 1, . . . , N denote the j-th trained
model and pj the p-value calculated using the pairwise independence test (6) (i.e., from Eq. 7 in
Section 3). We can investigate the degree to which (i) occurs by comparing the pj values with the
corresponding test set error rates RS(fj). To investigate whether (ii) occurs, we can average over the
randomness of the training runs.

6

For every example Xi ∈ S, consider the average test statistic ¯Ti = 1
j=1 Ti,gj (fj), where
N
Ti,gj (fj) is the statistic (5) calculated for example Xi and model fj with AEG gj selected for model
fj (note that AEGs are model-dependent by construction). If, for each i and j, the random variables
Ti(fj) are independent, then so are the ¯Ti (for all i). Hence, we can apply the pairwise dependence
detector (6) with ¯Ti instead of Ti, using the average ¯TS = (1/m) (cid:80)m
¯Ti with empirical variance
T,N = (1/m) (cid:80)m
i=1( ¯Ti − ¯TS)2, giving a single p-value pN . If the training runs vary enough in their
¯σ2
outcomes, different models fj err on different data points Xj, leading to ¯σ2
T , and therefore
strengthening the power of the dependence detector. For brevity, we call this independence test an
N -model test.

T,N < ¯σ2

i=1

(cid:80)N

4 Synthetic experiments

First we verify the effectiveness of our method
on a simple linear classiﬁcation problem. Due
to space limitations, we only convey high-level
results here, details are given in Appendix A.
We assume that the data is linearly separa-
ble with a margin and the density ρ is known.
We consider a linear classiﬁers of the form
f (x) = sgn(w(cid:62)x + b) trained with the cross-
entropy loss c, and we employ a one-step gra-
dient method (which is an L2 version of the
fast gradient-sign method of [14, 23]) to deﬁne
our AEG g, which tries to modify a correctly
classiﬁed point x with label y in the direction
of the gradient of the cost function, yielding
x(cid:48) = x−εyw/(cid:107)w(cid:107)2, where ε ≥ 0 is the strength
of the attack. To comply with the requirements
for an AEG, we deﬁne g as follows: g(x) = x(cid:48) if L(f, x) = 0 and f ∗(x) = f ∗(x(cid:48)) (corresponding
to (G2) and (G1), respectively), while g(x) = x otherwise. Therefore, if x(cid:48) is misclassiﬁed by f , x
and x(cid:48) are the only points mapped to x(cid:48) by g. This simple form of g and the knowledge of ρ allows to
compute the density hg, making it easy to compute the adversarial error estimate (2). Figure 4 shows
the average p-values produced by our N -model independence test for a dependent (solid lines) and
an independent (dashed lines) test set. It can be seen that in the dependent case the test can reject
independence with high conﬁdence for a large range of attack strength ε, while the independence
hypothesis is not rejected in the case of true independence. More details (including why only a range
of ε is suitable for detecting overﬁtting) are given in Appendix A.

Figure 4: Average p-values produced by the indepen-
dence test in a separable linear classiﬁcation problem
for the cases of both when the model is independent of
(dashed lines) and, resp., dependent on (solid lines) the
test set.

5 Testing overﬁtting on ImageNet

In the previous section we showed that the proposed adversarial-example-based dependence test
works for a synthetic problem where the densities can be computed exactly. In this section we apply
our estimates to a popular image classiﬁcation benchmark, ImageNet [8]; here the main issue is to
ﬁnd sufﬁciently strong AEGs that make computing the corresponding densities possible.

To facilitate the computation of the density hg, we only consider density-preserving AEGs as deﬁned
by (G3) (recall that (G3) is different from requiring hg = 1). Since in (2) and (5), hg(x) is multiplied
by L(f, x), we only need to determine the density hg for data points that are misclassiﬁed by f .

5.1 AEGs based on translations

To satisfy (G3), we implement the AEG using translations of images, which have recently been
proposed as means of generating adversarial examples [1]. Although relatively weak, such attacks ﬁt
our needs well: unless the images are procedurally centered, it is reasonable to assume that translating
them by a few pixels does not change their likelihood.4 We also make the natural assumption that
the small translations used do not change the true class of an image. Under these assumptions,

4Note that this assumption limits the applicability of our method, excluding such centered or essentially

centered image classiﬁcation benchmarks as MNIST [20] or CIFAR-10 [18].

7

10-210-1100101102†0.00.20.40.60.81.0p-valueN=1N=2N=10N=25N=100N=1N=2N=10N=25N=100translations by a few pixels satisfy conditions (G1) and (G3). An image-translating function g is a
valid AEG if it leaves all misclassiﬁed images in place (to comply with (G2)), and either leaves a
correctly classiﬁed image unchanged or applies a small translation.

The main beneﬁt of using a translational AEG g (with bounded translations) is that its density hg(x)
for an image x can be calculated exactly by considering the set of images x(cid:48) that can be mapped to x
by g (this is due to our assumption (G3)). We considered multiple ways for constructing translational
AEGs. The best version (selected based on initial evaluations on the ImageNet training set), which
we called the strongest perturbation, seeks a non-identical neighbor of a correctly classiﬁed image
x (neighboring images are the ones that are accessible through small translations) that causes the
classiﬁer to make an error with the largest conﬁdence.
Formally, we model images as 3D tensors in [0, 1]W ×H×C space, where C = 3 for RGB data, and
W and H are the width and height of the images, respectively. Let τv(x) denote the translation of
an image x by v ∈ Z2 pixels in the (X, Y) plane (here Z denotes the set of integers). To control
the amount of change, we limit the magnitude of translations and allow v ∈ Vε = {u ∈ Z2 :
u (cid:54)= (0, 0), (cid:107)u(cid:107)∞ ≤ ε} only, for some ﬁxed positive ε. Thus, we considers AEGs in the form
g(x) ∈ {τv(x) : v ∈ V} ∪ {x} if f (x) = f ∗(x) and g(x) = x otherwise (if x is correctly classiﬁed,
we attempt to translate it to ﬁnd an adversarial example in {τv(x) : v ∈ V} which is misclassiﬁed by
f , but x is left unchanged if no such point exists). Denoting the density of the pushforward measure
Pg by ρg, for any misclassiﬁed point x,

ρg(x) = ρ(x) +

(cid:88)

v∈V

ρ(τ−v(x))I(g(τ−v(x)) = x) = ρ(x)

1 +

(cid:32)

(cid:33)

I(g(τ−v(x)) = x)

(cid:88)

v∈V

where the second equality follows from (G3). Therefore, the corresponding density is

hg(x) = 1/(1 + n(x))

(8)
where n(x) = (cid:80)
I(g(τ−v(x)) = x) is the number of neighboring images which are mapped to
x by g. Note that given f and g, n(x) can be easily calculated by checking all possible translations
of x by −v for v ∈ V. It is easy to extend the above to non-deterministic perturbations, deﬁned as
distributions over AEGs, by replacing the indicator with its expectation P(g(τ−v(x)) = x|x, v) with
respect to the randomness of g, yielding

v∈V

hg(x) =

1

1 + (cid:80)

v∈V

P(g(τ−v(x)) = x|x, v)

.

(9)

If g is deterministic, we have hg(x) ≤ 1/2 for any successful adversarial example x. Hence, for such
g, the range U of the random variables Ti deﬁned in (5) has a tighter upper bound of 3/2 instead 2 (as
Ti ∈ [−1, 1/2]), leading to a tighter bound in (6) and a stronger pairwise independence test. In the
experiments, we use this stronger test. We provide additional details about the translational AEGs
used in Appendix B.

5.2 Tests of ImageNet models

We applied our test to check if state-of-the-art classiﬁers for the ImageNet dataset [8] have been
overﬁtted to the test set. In particular, we use the VGG16 classiﬁer of [27] and the Resnet50 classiﬁer
of [16]. Due to computational considerations, we only analyzed a single trained VGG16 model,
while the Resnet50 model was retrained 120 times. The models were trained using the parameters
recommended by their respective authors.

The preprocessing procedure of both architectures involves rescaling every image so that the smaller
of width and height is 256 and next cropping centrally to size 224 × 224. This means that translating
the image by v can be trivially implemented by shifting the cropping window by −v without any loss
of information for (cid:107)v(cid:107)∞ ≤ 16, because we have enough extra pixels outside the original, centrally
located cropping window. This implies that we can compute the densities of the translational AEGs
for any (cid:107)v(cid:107)∞ ≤ ε = (cid:98)16/3(cid:99) = 5 (see Appendix B.1 for detailed explanation). Because the ImageNet
data collection procedure did not impose any strict requirements on centering the images [8], it is
reasonable to assume (as we do) that small (lossless) translations respect the density-preserving
condition (G3).

In our ﬁrst experiment, we applied our pairwise independence test (6) with the AEGs described in
Appendix B (strongest, nearest, and the two random baselines) to all 1,271,167 training examples, as

8

Figure 5: p-values for the independence test on the ImageNet training set for different sample sizes and AEG
variants (left); original and adversarial risk estimates, (cid:98)RS(f ) and (cid:98)Rg(f ), on the ImageNet training set with
97.5% two-sided conﬁdence intervals for the ‘strongest attack’ AEG (right).

well as to a number of its randomly selected (uniformly without replacement) subsets of different
sizes. Besides this being a sanity check, we also used this experiment to select from different AEGs
and compare the performance of the pairwise independence test (6) to the basic version of the test
described in Section 3.1.

The left graph in Figure 5 shows that with the “strongest perturbation”, we were able to reject
independence of the trained model and the training samples at a conﬁdence level very close to 1 when
enough training samples are considered (to be precise, for the whole training set the conﬁdence level
is 99.9994%). Note, however, that the much weaker “smallest perturbation” AEG, as well as the
random transformations, are not able to detect the presence of overﬁtting. At the same time, the graph
on the right hand side shows the relative strength of the pairwise independence test compared to the
basic version based on independent conﬁdence interval estimates as described in detail in Section 3.1:
the 97.5%-conﬁdence intervals of the error estimates (cid:98)RS(f ) and (cid:98)Rg(f ) overlap, not allowing to
reject independence at a conﬁdence level of 95% (note that here S denotes the training set).

On the other hand, when applied to the test set, we obtained a p-value of 0.96, not allowing at all
to reject the independence of the trained model and the test set. This result could be explained by
the test being too weak, as no overﬁtting is detected to the training set at similar sample sizes (see
Figure 5), or simply the lack of overﬁtting. Similar results were obtained for Resnet50, where even
the N -model test with N = 120 independently trained models resulted a p value of 1, not allowing
to reject independence at any conﬁdence level. The view of no overﬁtting can be backed up in at
least two ways: ﬁrst, “manual” overﬁtting to the relatively large ImageNet test set is hard. Second,
since training an ImageNet model was just too computationally expensive until quite recently, only a
relatively small number of different architectures were developed for this problem, and the evolution
of their design was often driven by computational efﬁciency on the available hardware. On the other
hand, it is also possible that increasing N sufﬁciently might show evidence of overﬁtting (this is left
for future work).

6 Conclusions

We presented a method for detecting overﬁtting of models to datasets. It relies on an importance-
weighted risk estimate from a new dataset obtained by generating adversarial examples from the
original data points. We applied our method to the popular ImageNet image classiﬁcation task. For
this purpose, we developed a specialized variant of our method for image classiﬁcation that uses
adversarial translations, providing arguments for its correctness. Luckily, and in agreement with other
recent work on this topic [25, 26, 13, 21, 32], we found no evidence of overﬁtting of state-of-the-art
classiﬁers to the ImageNet test set.

The most challenging aspect of our methods is to construct adversarial perturbations for which we can
calculate the importance weights; ﬁnding stronger perturbations than the ones based on translations
for image classiﬁcation is an important question for the future. Another interesting research direction
is to consider extensions beyond image classiﬁcation, for example, by building on recent adversarial
attacks for speech-to-text methods [5], machine translation [11] or text classiﬁcation [12].

9

0.20.40.60.81.01.2sample size×1060.00.20.40.60.81.0P-valuevariantstrongestnearestrandomrandom20.20.40.60.81.01.2sample size×1060.1740.1760.1780.1800.1820.1840.1860.188cRS(f)cRg(f)Acknowledgements

We thank J. Uesato for useful discussions and advice about adversarial attack methods and sharing
their implementations [30] with us, as well as M. Rosca and S. Gowal for help with retraining image
classiﬁcation models. We also thank B. O’Donoghue for useful remarks about the manuscript, and L.
Schmidt for an in-depth discussion of their results on this topic. Finally, we thank D. Balduzzi, S.
Legg, K. Kavukcuoglu and J. Martens for encouragement, support, lively discussions and feedback.

References

[1] Aharon Azulay and Yair Weiss. Why do deep convolutional networks generalize so poorly to small image

transformations? 2018. arXiv:1805.12177.

[2] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning
to align and translate. In Proceedings of the International Conference on Learning Representations (ICLR),
2015.

[3] Stéphane Boucheron, Gábor Lugosi, and Pascal Massart. Concentration Inequalities: A Nonasymptotic

Theory of Independence. Oxford University Press, 2013.

[4] James Antonio Bucklew. Introduction to Rare Event Simulation. Springer New York, 2004.

[5] N. Carlini and D. Wagner. Audio adversarial examples: Targeted attacks on speech-to-text. In 2018 IEEE

Security and Privacy Workshops (SPW), May 2018.

[6] Nicholas Carlini and David A. Wagner. Adversarial examples are not easily detected: Bypassing ten
detection methods. In Proceedings of the 10th ACM Workshop on Artiﬁcial Intelligence and Security,
AISec@CCS 2017, Dallas, TX, USA, November 3, 2017, pages 3–14, 2017. URL http://doi.acm.org/
10.1145/3128572.3140444.

[7] Nicholas Carlini and David A. Wagner. Towards evaluating the robustness of neural networks. In 2017
IEEE Symposium on Security and Privacy, SP 2017, San Jose, CA, USA, May 22-26, 2017, pages 39–57,
2017. URL https://doi.org/10.1109/SP.2017.49.

[8] J. Deng, W. Dong, R. Socher, L. J. Li, Kai Li, and Li Fei-Fei. ImageNet: A large-scale hierarchical image
database. In 2009 IEEE Conference on Computer Vision and Pattern Recognition, pages 248–255, June
2009. doi: 10.1109/CVPR.2009.5206848.

[9] L. Deng, G. Hinton, and B. Kingsbury. New types of deep neural network learning for speech recognition
and related applications: an overview. In 2013 IEEE International Conference on Acoustics, Speech and
Signal Processing, pages 8599–8603. IEEE, May 2013.

[10] Cynthia Dwork, Vitaly Feldman, Moritz Hardt, Toniann Pitassi, Omer Reingold, and Aaron Roth. The
reusable holdout: Preserving validity in adaptive data analysis. Science, 349(6248):636–638, 2015.

[11] Javid Ebrahimi, Daniel Lowd, and Dejing Dou. On adversarial examples for character-level neural machine
translation. In Proceedings of the 27th International Conference on Computational Linguistics, COLING
2018, Santa Fe, New Mexico, USA, August 20-26, 2018, pages 653–663, 2018.

[12] Javid Ebrahimi, Anyi Rao, Daniel Lowd, and Dejing Dou. Hotﬂip: White-box adversarial examples for text
classiﬁcation. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics,
ACL 2018, Melbourne, Australia, July 15-20, 2018, Volume 2: Short Papers, pages 31–36, 2018.

[13] Vitaly Feldman, Roy Frostig, and Moritz Hardt. The advantages of multiple classes for reducing overﬁtting
from test set reuse. In Proceedings of the 36th International Conference on Machine Learning, pages
1892–1900, 2019.

[14] I. J. Goodfellow, J. Shlens, and C. Szegedy. Explaining and harnessing adversarial examples.

In

Proceedings of the International Conference on Learning Representations (ICLR), 2015.

[15] Alex Graves, Abdel-rahman Mohamed, and Geoffrey Hinton. Speech recognition with deep recurrent
neural networks. In 2013 IEEE International Conference on Acoustics, Speech and Signal Processing,
pages 6645–6649. IEEE, 2013.

[16] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition.
In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 770–778,
2016.

10

[17] H. Kahn and T. E. Harris. Estimation of particle transmission by random sampling. In Monte Carlo

Method, volume 12 of Applied Mathematics Series, pages 27–30. National Bureau of Standards, 1951.

[18] Alex Krizhevsky. Learning multiple layers of features from tiny images. Technical report, University of

Toronto, 2009.

[19] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. ImageNet classiﬁcation with deep convolutional
neural networks. In F. Pereira, C. J. C. Burges, L. Bottou, and K. Q. Weinberger, editors, Advances in
Neural Information Processing Systems 25, pages 1097–1105. Curran Associates, Inc., 2012.

[20] Yann LeCun and Corinna Cortes. MNIST handwritten digit database. http://yann.lecun.com/exdb/mnist/,

2010.

[21] Horia Mania, John Miller, Ludwig Schmidt, Moritz Hardt, and Benjamin Recht. Model similarity mitigates

test set overuse. 2019. arXiv:1905.12580.

[22] Volodymyr Mnih, Csaba Szepesvári, and Jean-Yves Audibert. Empirical Bernstein stopping. In Proceed-
ings of the 25th International Conference on Machine Learning, ICML ’08, pages 672–679, New York,
NY, USA, 2008. ACM.

[23] Nicolas Papernot, Patrick D. McDaniel, and Ian J. Goodfellow. Transferability in machine learning: from

phenomena to black-box attacks using adversarial samples. 2016. arXiv:1605.07277.

[24] Nicolas Papernot, Patrick McDaniel, Ian Goodfellow, Somesh Jha, Z Berkay Celik, and Ananthram Swami.
Practical black-box attacks against machine learning. In Proceedings of the 2017 ACM on Asia Conference
on Computer and Communications Security, pages 506–519. ACM, 2017.

[25] Benjamin Recht, Rebecca Roelofs, Ludwig Schmidt, and Vaishaal Shankar. Do CIFAR-10 classiﬁers

generalize to CIFAR-10? 2018. arXiv:1806.00451.

[26] Benjamin Recht, Rebecca Roelofs, Ludwig Schmidt, and Vaishaal Shankar. Do ImageNet classiﬁers

generalize to ImageNet? 2019. arXiv:1902.10811.

[27] K. Simonyan and A. Zisserman. Very deep convolutional networks for large-scale image recognition. In

Proceedings of the International Conference on Learning Representations (ICLR), 2015.

[28] Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru
Erhan, Vincent Vanhoucke, and Andrew Rabinovich. Going deeper with convolutions. In 2015 IEEE
Conference on Computer Vision and Pattern Recognition (CVPR), 2015. arXiv:1409.4842.

[29] T. Tieleman and G. Hinton. Lecture 6.5—RmsProp: Divide the gradient by a running average of its recent

magnitude. COURSERA: Neural Networks for Machine Learning, 2012.

[30] Jonathan Uesato, Brendan O’Donoghue, Aäron van den Oord, and Pushmeet Kohli. Adversarial risk and
the dangers of evaluating against weak attacks. In Proceedings of the 35th International Conference on
Machine Learning, volume 80 of Proceedings of Machine Learning Research, pages 5025–5034, 2018.
arXiv:1802.05666.

[31] Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V. Le, Mohammad Norouzi, Wolfgang Macherey, Maxim
Krikun, Yuan Cao, Qin Gao, Klaus Macherey, Jeff Klingner, Apurva Shah, Melvin Johnson, Xiaobing
Liu, Lukasz Kaiser, Stephan Gouws, Yoshikiyo Kato, Taku Kudo, Hideto Kazawa, Keith Stevens, George
Kurian, Nishant Patil, Wei Wang, Cliff Young, Jason Smith, Jason Riesa, Alex Rudnick, Oriol Vinyals,
Greg Corrado, Macduff Hughes, and Jeffrey Dean. Google’s neural machine translation system: Bridging
the gap between human and machine translation. 2016. arXiv:1609.08144.

[32] Chhavi Yadav and Léon Bottou. Cold case: The lost MNIST digits. May 2019. arXiv:1905.10498.

11

A Synthetic experiments

In this section, we present full details of the experiments on a simple synthetic classiﬁcation problem,
which we presented brieﬂy in Section 4. These experiments illustrate the power of the method of
Section 3. The advantage of the simple setup considered here is that we are able to compute the
density hg in an analytic form (see Figure 6 for an illustration).

A.1 Data distribution and model

Let X = R500 and consider an input distribution with a density ρ that is an equally weighted
± (µ±, σ2I) with
mixture of two 500-dimensional isotropic truncated Gaussian distributions N trunc
√
coordinate-wise standard deviation σ =
500 (I denotes the identity matrix of size 500 × 500),
means µ± = [±1, 0, 0, . . . , 0] and densities ρ± truncated in the ﬁrst dimension such that ρ+(x) = 0
if x1 ≤ 0.025 and ρ−(x) = 0 if x1 ≥ −0.025. The label of an input point x is f ∗(x) = sgn(x1),
which is the sign of its ﬁrst coordinate.
We consider linear classiﬁers of the form f (x) = sgn(w(cid:62)x + b) trained with the cross-entropy loss
c((w, b), x, y) = ln(1 + e−y(w(cid:62)x+b)) where y = f ∗(x). We employ a one-step gradient method
(which is an L2 version of the fast gradient-sign method of [14, 23]) to deﬁne our AEG g, which
tries to modify a correctly classiﬁed point x with label y in the direction of the gradient of the cost
function c: x(cid:48) = x + ε∇xc((w, b), x, y)/(cid:107)∇xc((w, b), x, y)(cid:107)2 for some ε > 0. For our speciﬁc
choice of c, the above simpliﬁes to x(cid:48) = x − εyw/(cid:107)w(cid:107)2. To comply with the requirements for an
AEG, we deﬁne g as follows: g(x) = x(cid:48) if L(f, x) = 0 and f ∗(x) = f ∗(x(cid:48)) (corresponding to
(G2) and (G1), respectively), while g(x) = x otherwise. Therefore, if x(cid:48) is misclassiﬁed by f , x

x2

y = −1

y = +1

xC

x(cid:48)
C

x(cid:48)
D

x(cid:48)
E

x(cid:48)
A

x(cid:48)
B

xA

0

xB

x1

w

xD

xE

Figure 6: Illustration of the data distribution and the linear model f (x) = sgn(w(cid:62)x + b) in two dimensions.
The blue and green gradients show the probability density ρ of the data with true labels y = −1 and y = 1,
respectively, while the white space between them is the margin with ρ = 0. The red line is the model’s
classiﬁcation boundary with its parameter vector w shown by the purple arrow. Depending on the label, w or
−w is the direction of translation used to perturbed the correctly classiﬁed data points, and the translations
used by the AEG g for speciﬁc points are depicted by grey arrows: solid arrows indicate the cases where
g(x) = x(cid:48)
(cid:54)= x, while dashed arrows are for candidate translations which are not performed by the AEG
because they would change the true label, f ∗(x(cid:48)) (cid:54)= f ∗(x), and hence g(x) = x (cid:54)= x(cid:48). Each original/perturbed
data point is represented by a color-coded circle: the inner color corresponds to the true label (dark blue for
y = −1 and dark green for y = 1) while the outer color to the model’s prediction (dark blue for f (x) = −1
and dark green for f (x) = 1). Points x(cid:48)
B can be obtained from xA and xB, respectively, by applying
the AEG, x(cid:48)
A = g(xA) and x(cid:48)
A, the density
(Radon-Nikodym derivative) can be obtained as hg(x(cid:48)
A)) ∈ (0, 1). In the case of
x(cid:48)
B) = ρ(x(cid:48)
B, hg(x(cid:48)
A) does not
depend on whether xA or x(cid:48)
A a “successful adversarial
example” while in the second case x(cid:48)
B)).
x(cid:48)
C is not a successful adversarial example since L(f, x(cid:48)
C according to our
deﬁnition). Points xD and xE are not perturbed by our AEG, since f ∗(xD) (cid:54)= f ∗(x(cid:48)
E).

A is called “originally misclassiﬁed” (a similar argument holds for hg(x(cid:48)

B)) = 0 due to the margin. Note that the formula for hg(x(cid:48)

A is in the original test set S; in the ﬁrst case we call x(cid:48)

B = g(xB). Since only xA is mapped to x(cid:48)

C ) = 0 (however, g(xC ) = x(cid:48)

D) and f ∗(xE) (cid:54)= f ∗(x(cid:48)

B)/(ρ(xB) + ρ(x(cid:48)

A)/(ρ(xA) + ρ(x(cid:48)

A by g, and g(x(cid:48)

A) = ρ(x(cid:48)

A and x(cid:48)

A) = x(cid:48)

12

and x(cid:48) are the only points mapped to x(cid:48) by g. Thus, the density at x(cid:48) after the transformation g is
ρ(cid:48)(x(cid:48)) = ρ(x) + ρ(x(cid:48))(1 − L(f, x))I(f ∗(x) = f ∗(x(cid:48))) and

hg(x(cid:48)) =

ρ(x(cid:48))
ρ(cid:48)(x(cid:48))

=

ρ(x(cid:48))
ρ(x(cid:48)) + ρ(x)(1 − L(f, x))I(f ∗(x) = f ∗(x(cid:48)))

(note that I(L(f, x) = 0) = 1 − L(f, x)).

A.2 Experiment setup

We present two experiments showing the behavior of our independence test: one where the training
and test sets are independent, and another where they are not.

In the ﬁrst experiment a linear classiﬁer was trained on a training set STr of size 500 for 50,000 steps
using the RMSProp optimizer [29] with batch size 100 and learning rate 0.01, obtaining zero (up to
numerical precision) ﬁnal training loss c and, consequently, 100% prediction accuracy on the training
data. Then the trained classiﬁer was tested on a large test set STe of size 10,000.5 Both sets were
drawn independently from ρ deﬁned above. We used a range of ε values matched to the scale of the
data distribution: from 10−2, which is the order of magnitude of the margin between two classes
(0.05), to 102, which is the order of magnitude of the width of the Gaussian distribution used for each
classes (σ =

500).

√

In the second experiment we consider the situation where the training and test sets are not independent.
To enhance the effects of this dependence, the setup was modiﬁed to make the training process more
amenable to overﬁtting by simulating a situation when the model has a wrong bias (this may happen in
practice if a wrong architecture or data preprocessing method is chosen, which, despite the modeler’s
best intentions, worsens the performance). Speciﬁcally, during training we added a penalty term
104w2
1 to the training loss c, decreased the size of the test set to 1000 and used 50% of the test data
for training (the ﬁnal penalized training loss was 0.25 with 100% prediction accuracy on the training
set). Note that the small training set and the large penalty on w1 yield classiﬁers that are essentially
independent of the only interesting feature x1 (recall that the true label of a point x is sgn(x1)) and
overﬁt to the noise in the data, resulting in a true model risk R(f ) ≈ 1/2.

A.3 Results

The results of the two experiments are shown in Figure 7, plotted against different perturbation
strengths: the left column corresponds to the ﬁrst experiment while the right column to the second.
The ﬁrst row presents the p-values for rejecting the independence hypothesis, calculated by repeating
the experiment (sampling data and training the classiﬁer) 100 times and applying the single-model
(Section 3, labelled as N = 1 in the plots) and N -model (Section 3.3, labelled as N = 2, 10, 25, 100
in the plots) independence test, and taking the average over models (or model sets of size N ) for each
ε. We also plot empirical 95% two-sided conﬁdence intervals (N ≤ 2) or, due to limited number of
p-values available after dividing 100 runs into disjoint bins of size N ≥ 10, ranges between minimum
and maximum value (N = 10, 25). For all methods of detecting dependence, it can be seen that
for the independent case the test is correctly not able to reject the independence hypothesis (the
average p-value is very close to 1, although in some runs it can drop to as low as 0.5). On the other
hand, for 10 ≤ ε ≤ 50, the non-independent model failed the independence test at conﬁdence level
1 − δ ≈ 100%, hence, in this range of ε our independence test reliably detects overﬁtting.

In fact, it is easy to argue that our test should only work for a limited range of ε, that is, it should
not reject independence for too small or too large values of ε. First we consider the case of small
ε values. Notice that except for points g(x) ε-close (in L2-norm) to the true decision boundary or
the decision boundary of f , g(x) is invertible: if g(x) is correctly classiﬁed and is ε-away from the
true decision boundary, there is exactly one point, x, which is translated to g(x), while if g(x) is
incorrectly classiﬁed and ε-away from the decision boundary of f , no translation leads to g(x) and
x = g(x); any other points are ε-close to the decision boundary of either f or f ∗. Thus, since ρ is
bounded, g(x) is invertible on a set of at least 1 − O(ε) probability (according to ρ). When ε → 0,
g(x) → x, and so ρ(g(x)) → ρ(x) for all points x with |x1| (cid:54)= 0.025 (since ρ is continuous in all
such x), implying hg(g(x)) ≈ 1 on these points. It also follows that L(f, x) (cid:54)= L(f, g(x)) can only

5The large number of test examples ensures that the random error in the empirical error estimate is negligible.

13

Figure 7: Risk and overﬁtting metrics for a synthetic problem with linear classiﬁers as a function of the
perturbation strengths ε (log scale). Left: unbiased model tested on a large, independent test set (in this case
(cid:98)RS(f ) ≈ (cid:98)Rg(f ) ≈ R(f )); right: trained model overﬁtted to the test set ( (cid:98)RS(f ) ≤ (cid:98)Rg(f ) while both are
smaller than R(f )). First row: Average p-value δ for the pairwise independence test with over 100 runs
(N = 1) or the N -model independence test (N > 1). The bounds plotted are either empirical 95% two-sided
conﬁdence intervals (N ≤ 2) or ranges between minimum and maximum value (N = 10, 25). Second row:
Empirical two-sided 97.5% conﬁdence intervals for the empirical test error rate (cid:98)RS(f ) and the adversarial risk
estimate (cid:98)Rg(f ). On the left, R(f ) ≈ (cid:98)RS(f ), while R(f ) is shown separately on the right. Third row: Average
densities (Radon-Nikodym derivatives) for originally misclassiﬁed points and for the new data points obtained
by successful adversarial transformations (with empirical 97.5% two-sided conﬁdence intervals). Fourth row:
The empirical test error rate (cid:98)RS(f ) and the adversarial risk estimate (cid:98)Rg(f ) for a single realization with 97.5%
two-sided conﬁdence intervals computed from Bernstein’s inequality, the adversarial error rate (cid:98)RS(cid:48) (f ), and the
expected error R(f ) (on the right, on the left R(f ) ≈ (cid:98)RS(f )). Fifth row: Histograms of p-values for selected ε
values over 100 runs.

14

10-210-1100101102†0.00.20.40.60.81.0p-valueN=1N=2N=10N=25N=10010-210-1100101102†0.00.20.40.60.81.0p-valueN=1N=2N=10N=25N=10010-210-1100101102†0.00.20.40.60.81.0errorcRg(f)cRS(f)10-210-1100101102†0.00.20.40.60.81.0errorcRg(f)cRS(f)R(f)10-210-1100101102†0.00.20.40.60.81.0average hgoriginal misclassifiedadversarial10-210-1100101102†0.00.20.40.60.81.0average hgoriginal misclassifiedadversarial10-210-1100101102†0.00.20.40.60.81.0errorcRg(f)cRS(f)cRS0(f)10-210-1100101102†0.00.20.40.60.81.0errorcRg(f)cRS(f)cRS0(f)R(f)0.00.20.40.60.81.0p-value020406080100countδ(†=0.1)δ(†=5)δ(†=20)0.00.20.40.60.81.0p-value020406080100countδ(†=0.1)δ(†=5)δ(†=6)δ(†=20)happen to a set of points with an O(ε) ρ-probability. This means that L(f, g(x))hg(g(x)) ≈ L(f, x)
on a set of 1 − O(ε) ρ-probability, and for these points Tg(x) = L(f, g(x))hg(g(x)) − L(f, x) ≈ 0.
Thus, Tg(X) ≈ 0 with ρ-probability 1 − O(ε). Unless the test set S is concentrated in large part
on the set of remaining points with O(ε) ρ-probability, the test statistic |TS,g(f )| = O(ε) with high
probability and our method will not reject the independence hypothesis for ε → 0.

When ε is large (ε → ∞), notice that for any point x with non-vanishing probability (i.e., with
ρ(x) > c for some c > 0), if g(x) (cid:54)= x than ρ(g(x)) ≈ 0. Therefore, for such an x, if L(f, x) = 0 and
L(f, g(x)) = 1, hg(g(x)) = ρ(g(x))/(ρ(x) + ρ(g(x))) ≈ 0, and so Tg(x) ≈ 0 (if L(f, g(x)) = 0,
we trivially have Tg(x) = 0). If L(f, x) = 1, we have g(x) = x. If g is invertible at x then
hg(x) = 1 and Tg(x) = 0. If g is not invertible, then there is another x(cid:48) such that g(x(cid:48)) = x;
however, if ρ(x) > c then ρ(x(cid:48)) ≈ 0 (since ε is large), and so hg(g(x)) = ρ(x)/(ρ(x) + ρ(x(cid:48))) ≈ 1,
giving Tg(x) ≈ 0. Therefore, for large ε, Tg(X) ≈ 0 with high probability (i.e., for points with
ρ(x) > c), so the independence hypothesis will not be rejected with high probability.

To better understand the behavior of the test, the second row of Figure 7 shows the empirical test error
rate (cid:98)RS(f ), the (unadjusted) adversarial error rate (cid:98)RS(cid:48)(f ), and the adversarial risk estimate (cid:98)Rg(f ),
together with their conﬁdence intervals. For the non-independent model, we also show the expected
error R(f ) (estimated over a large independent test set), while it is omitted for the independent model
where it approximately coincides with both (cid:98)RS(f ) and (cid:98)Rg(f ). While the reweighted adversarial error
estimate (cid:98)Rg(f ) remains the same for all perturbations in case of an independent test set (left column),
the adversarial error rate (cid:98)RS(cid:48)(f ) varies a lot for both the dependent and independent test sets. For
example, in the case when the test samples and the model f are not independent, it undershoots
the true error for ε < 10 and overshoots it for larger perturbations. For very large perturbations
(ε close to 100), the behavior of (cid:98)RS(cid:48)(f ) depends on the model f : in the independent case (cid:98)RS(cid:48)(f )
decreases back to (cid:98)RS(f ) because such large perturbations increasingly often change the true label
of the original example, so less and less adversarial points are generated. In the case when the data
and the model are not independent (right column), the adversarial perturbations are almost always
successful (i.e., lead to a valid adversarial example for most originally correctly classiﬁed points),
yielding an adversarial error rate close to one for large enough perturbations. This is because the
decision boundary of f is almost orthogonal to the true decision boundary, and so the adversarial
perturbations are parallel with the true boundary, almost never changing the true label of a point.

The plots of the densities (Radon-Nikodym derivatives), given in the third row of Figure 7, show
how the change in their values compensate the increase of the adversarial error rate (cid:98)RS(cid:48)(f ): in the
independent case, the effect is completely eliminated yielding an unbiased adversarial error estimate
(cid:98)Rg(f ), which is essentially constant over the whole range of ε (as shown in the ﬁrst row), while in
the non-independent case the similar densities do not bring back the adversarial error rate (cid:98)RS(cid:48)(f ) to
the test error rate (cid:98)RS(f ), allowing the test to detect overﬁtting. Note that the densities exhibit similar
trends (and values) in both cases, driven by the dependence of typical values of the ρ(x)/ρ(g(x))
ratio on the perturbation strength ε for originally misclassifed points (L(f, x) = 1) and for successful
adversarial examples (i.e., L(f, x) = 0 and L(f, g(x)) = 1).

To compare the behavior of our improved, pairwise test and the basic version, the fourth row of
Figure 7 depicts a single realization of the experiments where the 97.5% conﬁdence intervals (as
computed from Bernstein’s inequality) are shown for the estimates. For the independent case, the
conﬁdence intervals of (cid:98)RS(f ) and (cid:98)Rg(f ) overlap for all ε, and thus the basic test is not able to detect
overﬁtting. In the non-independent case, the conﬁdence intervals overlap for ε = 10 and ε = 75, thus
the basic test is not able to detect overﬁtting with at a 95% conﬁdence level, while the improved test
(second row) is able to reject the independence hypothesis for these ε values at the same conﬁdence
level.

Finally, in the ﬁfth row of Figure 7 we plotted the histograms of the empirical distribution of p-values
for both models, over 100 independent runs (between the runs, all the data was regenerated and the
models were retrained). For ε = 0.1, 5, 20, they concentrate heavily on either δ = 0 or δ = 1, and
have very thin tails extending far towards the opposite end of the [0, 1] interval. This explains the
surprisingly wide 95% conﬁdence intervals for p-values plotted in the ﬁrst row. In particular, the fact
that some p-values for the independent model are as low as 0.5 does not mean the independence test
is not reliable, because almost all calculated δ values are close or equal to 1, and the few outliers are

15

independent model, ε = 10

non-independent model, ε = 6

Figure 8: Histograms of p-values from N -model (N = 1, 2, 10) independence tests for both synthetic models
and selected ε values, over 100 runs.

a combined consequence of the ﬁnite sample size and the effectiveness of the AEG. The additional
ε = 6 histogram for the non-independent model illustrates a regime which is in between the single-
model pairwise test (Section 3) completely failing to reject the independence hypothesis and clearly
rejecting it.

To verify experimentally whether the N -model independence test can be a more powerful detector of
overﬁtting than the single-model version, in Figure 8 (right panel) we plotted p-value histograms for
N = 1, 2, 10 for the intermediate AEG strength ε = 6 applied to the non-independent model over 100
training runs. Indeed, as N increases, the concentration of p-values around in the low (δ ≤ 0.2) range
increases. For N > 10 we did not have enough values to plot a histogram: for N = 25 we obtained
δ = 0.1851, 0.1599, 0.0661 and 0.1941, while for N = 100 the p-value is 0.1153. The increase of
the test power becomes apparent when we compare the last value with the mean of p-values obtained
by testing every training run separately, equal 0.5984, and the median 0.6385.

For comparison, we also plotted in Figure 8 (left panel) the corresponding histograms for the
independent model and a slightly higher attack strength, ε = 10, at which the independence tests fails
for the overﬁtted model even without averaging (see Figure 7, ﬁrst row, right panel). The histograms
are all clustered in the δ region close to 1, indicating that the N -model test is not overly pessimistic.

B Translational AEGs for image classiﬁcation models

For image classiﬁcation we consider two translation variants that are used in constructing a transla-
tional AEG. For every correctly classiﬁed image x, we consider translations from Vε (for some ε),
choosing g(x) from the set G(x) = {τv(x) : v ∈ Vε} ∪ {x}. If all translations result in correctly
classiﬁed examples, we set g(x) = x. Otherwise, we use one of two possible ways to select g(x)
(and we call the resulting points successful adversarial examples):

• Strongest perturbation: Assuming the number of classes is K, let l(f, x) ∈ RK denote
the vector of the K class logits calculated by the model f for image x, and let lexc(f, x) =
max0≤i<K li(f, x) − ly(f, x). We deﬁne

gstrongest(x) = argmaxx(cid:48)∈G(x) lexc(f, x(cid:48)),

with ties broken deterministically by choosing the ﬁrst translation from the candidate set,
going top to bottom and left to right in row-major order. Thus, here we seek a non-identical
“neighbor” that causes the classiﬁer to err the most, reachable from x by translations within
a maximum range ε.

• Nearest misclassiﬁed neighbor: Here we aim to ﬁnd the nearest image in G(x) that is
misclassiﬁed. That is, letting d(x, x(cid:48)) = (cid:107)v(cid:107)2 if x(cid:48) = τv(x) and ∞ otherwise, we deﬁne
gnearest(x) := argminx(cid:48)∈G(x),L(f,x(cid:48))=1 d(x, x(cid:48))

with ties broken deterministically as above.

The two perturbation variants are successful on exactly the same set of images, hence they lead to
the same adversarial error rates (cid:98)RS(cid:48)(f ). However, they are characterized by different values of the
density hg and, consequently, yield different adversarial risk estimates (cid:98)Rg(f ) and associated p-values

16

0.00.20.40.60.81.0p-value024681012PDFδ(N=1)δ(N=2)δ(N=10)0.00.20.40.60.81.0p-value012345PDFδ(N=1)δ(N=2)δ(N=10)for the independence test. The main difference between them is that the “strongest” version is more
likely to map multiple images to the same adversarial example, thus decreasing the densities for
successful adversarial examples and, counterintuitively, increasing them for originally misclassiﬁed
points (as their neighbors are less likely to be mapped to these points).

To better see the effect of adversarial perturbations, we also consider two random baselines that do
not take into account the success of a translation in generating misclassiﬁed points: grandom(x) is
chosen uniformly at random from G(x) \ {x}, and grandom2(x) is chosen uniformly at random from
G(x).

B.1 Maximum translations

In practice, translating an image is not always simple, as the new image has to be padded with
new pixels. When (central) crops of a larger image are used (as is typical for ImageNet classiﬁers),
translations can easily be implemented as long as the resulting new cropping window stays within the
original image boundaries. Even if an image can be translated by a vector v, this limits our ability to
compute hg(x(cid:48)) for the adversarial image x(cid:48) by (8) or (9) for gstrongest or gnearest. Indeed, if an image
x is shifted by v ∈ Vε to generate adversarial example x(cid:48), we need to examine translations of x(cid:48) with
vectors in Vε to ﬁnd the neighbors x(cid:48)(cid:48) of x(cid:48) potentially contributing to n(x(cid:48)) when computing hg(x(cid:48)).
Finally we need to consider translations of x(cid:48)(cid:48) with vectors in Vε to determine the exact value they
contribute, that is, to compute the exact probabilities in (9) (see Figure 9 for an illustration). Thus,
to be able to compute the density hg for the adversarial points obtained by translations from Vε, we
might need to be able to perform translations within V3ε.

Figure 9: Image translations which need to be considered for a translational AEG with ε = 3. The red, blue and
green balls represent the center of the original image x, adversarial example x(cid:48) = g(x) and another image x(cid:48)(cid:48)
contributing to ρg(x(cid:48)), respectively, while the semi-translucent squares of corresponding colors represent the
possible translations which need to be considered for each of x, x(cid:48) and x(cid:48)(cid:48). Solid light grey arrows represent the
relationships x(cid:48) = g(x) and x(cid:48) = g(x(cid:48)(cid:48)). Finally, the dashed arrow and the semi-translucent grey ball represent
an alternative mapping, which has to be ruled out while calculating the value of g(x(cid:48)(cid:48)) and, consequently, of
hg(x(cid:48)). It is easy to see that the colored squares (which contain the translations needing to be evaluated) extend
as far as 3ε from the original image x.

17

xx'x''ε
