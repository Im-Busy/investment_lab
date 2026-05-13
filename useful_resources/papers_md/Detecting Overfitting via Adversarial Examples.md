<!-- Page 1 -->

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


<!-- Page 2 -->

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


<!-- Page 3 -->

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


<!-- Page 4 -->

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


<!-- Page 5 -->

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


<!-- Page 6 -->

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


<!-- Page 7 -->

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


<!-- Page 8 -->

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


<!-- Page 9 -->

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


<!-- Page 10 -->

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


<!-- Page 11 -->

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


<!-- Page 12 -->

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


<!-- Page 13 -->

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


<!-- Page 14 -->

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


<!-- Page 15 -->

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


<!-- Page 16 -->

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


<!-- Page 17 -->

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
