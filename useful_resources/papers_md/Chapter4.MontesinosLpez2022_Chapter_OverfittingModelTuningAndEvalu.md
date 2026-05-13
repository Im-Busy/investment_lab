<!-- Page 1 -->

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


<!-- Page 2 -->

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


<!-- Page 3 -->

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


<!-- Page 4 -->

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


<!-- Page 5 -->

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


<!-- Page 6 -->

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


<!-- Page 7 -->

4.3 Cross-validation 115
4.3 Cross-validation
Cross-validation (CV) is a strategy for model selection or algorithm selection. CV
consistsofsplittingthedata (atleastonce) forestimatingtheerrorofeachalgorithm.
Part of the data (the training set) is used for training each algorithm, and the
remaining part (the testing set) is used for estimating the error of the algorithm.
Then, CVselectsthealgorithmwiththesmallestestimatederror. Forthisreason, CV
isusedtoevaluatethepredictionperformanceofastatisticalmachinelearningmodel
in out-of-sample data. This technique ensures that the data used for training the
statistical machine learning model are independent of the testing data set in which
the prediction performance is evaluated. It consists of repeating and recording the
arithmetic average obtained from the evaluation measures on different partitions.
Underk-fold CV, whichisexplainedingreaterdetaillater, thisprocessisrepeateda
totalofktimes, witheachofthekgroupsgettingthechancetoplaytheroleofthe
testdata, andtheremainingk (cid:1)1 groupsusedastrainingdata. Inthisway, weobtain
k different estimates of the prediction error. As prediction performance is reported
the average of these estimates of prediction error. CV is used in data analysis to
validate the implemented models where the main objective is prediction and to
estimate the prediction performance of a statistical learning model that will be
carriedoutinpractice. Inotherwords, CVevaluateshowwellthestatisticalmachine
learningmodelgeneralizednewdatanotusedfortrainingthemodel. Theresultsof
the CVlargely depend onhowthedivisionbetweenthetrainingandtestingsetsis
carriedout. Forthisreason, inthefollowingsections, weprovidethemorepopular
typesof CVusedintheimplementationofstatisticallearningmodels.
4.3.1 The Single Hold-Out Set Approach
Thesinglehold-outsetorvalidationsetapproachconsistsofrandomlydividingthe
availabledataset intoatrainingsetandavalidationorhold-outset (Fig. 4.3).The
statisticalmachinelearningmodelistrainedwiththetrainingsetwhilethehold-out
Complete data set
I1 I2 I3 In
Training Testing
I40 I5 I82... I62 I45 I88
Fig.4.3 Schematicrepresentationofthehold-outsetapproach. Asetofobservationsarerandomly
split into a training set with individuals I40, I5, I82, among others, and into a testing set with
observations I45, I88, amongothers. Thestatisticalmachinelearningmodelisfittedonthetraining
setanditsperformanceisevaluatedonthevalidationset (Jamesetal.2013)


<!-- Page 8 -->

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


<!-- Page 9 -->

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


<!-- Page 10 -->

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


<!-- Page 11 -->

4.3 Cross-validation 119
Fig. 4.5 Schematic representation of time series data with 5 years, using the previous year for
predictingthenextyear. However, insomepracticalapplications, wearenotinterestedingoingtoo
farbackintimesincethetrainingandtestingsetswillbelessrelatedthefartheryougo
imperativetoevaluateourstatisticalmachinelearningmodelfortimeseriesdataon
“future” observations. In this sense, the training sets are composed of the previous
yearstopredictthesubsequentyear. Thismethodcanalsobeseenasavariationofk-
fold CV, where the first folds are used for training the statistical machine learning
model and the fold (k + 1) is the corresponding testing set. The main difference in
this CV method is that successive training sets are supersets of those that come
before them. Also, it adds all surplus data to the first training partition, which is
alwaysusedtotrainthemodel (see Fig.4.5).
Figure4.5 showsthatunderthistypeof CV, fortimeseriesdata, thepredictions
areforg (cid:1)1 years;individualsfromthefirstyeararenotpredictedsinceatraining
setisnotavailable.
4.3.7 Bootstrap Cross-validation
First, wewilldefinethebootstrappingmethodtounderstandhowitisusedinthe CV
approach, which should then be straightforward. Bootstrapping is a type of
resampling method where, for example, B ¼ 10 samples of the same size are
repeatedly drawn, with replacement, from a single original sample. Afterward,
each of these B samples is used to estimate statistics (for example, the mean,
variance, median, minimum, etc.) of a population, and the average of all the
B sample estimates of the target statistic is reported as the final estimate. In the
context of statistical machine learning, these samples are used to evaluate the
predictionperformanceofthealgorithmunderstudyforunseendata. Oneimportant
differencebetweenthis CVapproachandalltheproceduresexplainedaboveisthat
now the training set has the same size (number of observations) as the original
sample because the bootstrap method replaced some individuals more than once.
According to Kuhn and Johnson (2013), as a result, some observations will be
representedmultipletimesinthebootstrapsample, whileotherswillnotbeselected
atall;thoseobservationsnotselectedarereferredtoasthetestingset, however, this
CV strategy is quite different than the previously explained. Efron (1983) pointed
out that the prediction performance of the bootstrap samples tends to have less
uncertainty than the k-fold cross-validation since on average, 63.2% of the data
pointsarerepresented (fortraining) atleastonceinanysamplesize. Forthisreason,


<!-- Page 12 -->

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


<!-- Page 13 -->

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


<!-- Page 14 -->

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


<!-- Page 15 -->

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


<!-- Page 16 -->

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


<!-- Page 17 -->

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


<!-- Page 18 -->

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


<!-- Page 19 -->

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


<!-- Page 20 -->

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


<!-- Page 21 -->

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


<!-- Page 22 -->

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


<!-- Page 23 -->

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


<!-- Page 24 -->

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


<!-- Page 25 -->

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


<!-- Page 26 -->

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


<!-- Page 27 -->

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


<!-- Page 28 -->

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


<!-- Page 29 -->

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


<!-- Page 30 -->

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


<!-- Page 31 -->

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
