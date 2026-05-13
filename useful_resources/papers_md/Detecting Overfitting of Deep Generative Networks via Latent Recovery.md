<!-- Page 1 -->

Detecting Overfitting of Deep Generative Networks via Latent Recovery
Ryan Webster, Julien Rabin, Lo¨ıc Simonand Fre´de´ric Jurie
Normandie Univ., ENSICAEN, UNICAEN, CNRS, GREYC, France
ryan.webster@unicaen.fr
Abstract targety NN G(y) NN D(y)
Stateoftheartdeepgenerativenetworkshaveachieved
such realism that they can be suspected of memorizing
training images. It is why it is not uncommon to include
visualizations of training set nearest neighbors, to suggest
generatedimagesarenotsimplymemorized. Wearguethis
is not sufficient and motivates studying overfitting of deep
generatorswithmorescrutiny. Weaddressthisquestionby
i) showinghowsimplelossesarehighlyeffectiveatrecon-
structingimagesfordeepgeneratorsii) analyzingthestatis-
tics of reconstruction errors for training versus validation
images. Using this methodology, we show that pure GAN
modelsappeartogeneralizewell, incontrastwiththoseus-
inghybridadversariallosses, whichareamongstthemost
widelyappliedgenerativemethods. Wealsoshowthatstan-
dard GANevaluationmetricsfailtocapturememorization
forsomedeepgenerators. Finally, wenotetheramifications
of memorization on data privacy. Considering the already
widespreadapplicationofgenerativenetworks, weprovide
a step in the right direction towards the important yet in-
completepictureofgenerativeoverfitting.
1.Introductionand Related Work
In just a few short years, image generation with deep
networks has gone from niche to a center piece of ma-
chine learning. This was largely initiated by Generative
Adversarial Networks (GANs)[13]andsincethenincredi-
bleprogresshasbeenmade, fromdeepconvolutional GAN
(DCGAN) [32] producing artifacted faces, to progressive
GANS(PGGAN)[20]producingfaceswhicharevirtually
indistinguishable from real ones even to human observers
and at high resolution (see Fig. 1). While a large amount
of research has proposed new generative models, less re-
search has been devoted to the evaluation of such models.
Furthermore, evaluating overfitting of deep generators has
beenperformedviaintuitivevisualdemonstrations, suchas
training set nearest neighbor search and latent space inter-
polation [7,20]. Fig.1(lastcolumn) illustratesthenearest
neighbor (NN) test, where NN D(y) is the training dataset
D∈6
y
) loop () y (φ
) z (G
D∈
y
) roloc () y (φ
Figure 1: Rather than inspecting the most similar images
NND(y) inthetrainingdataset D forsampledgeneratedimages
G(z)(row3), weconsiderfindingthemostsimilarimageinthe
manifold NNG(y) ofgeneratedimages (column2). Asseeninthe
lasttworows, NNG(y) ismoremeaningfulundersometransfor-
mations. Analysisofthediscrepancybetweenreconstructionsof
thetrainset Dandreconstructionsoutside Dmakesitpossibleto
detectoverfittingforsomegenerators.
NNofafewimagesy. While NN D(y) withthe Euclidean
distanceisacommonheuristic (lastcolumnin Fig.1), itis
purelyvisualandsensitivetotransformationsofthetraining
data.
In contrast, we suggest to rely on the opposite method-
111273


<!-- Page 2 -->

ologybyoptimizingthelatentcodez ∼Z tofindthenear- sonswhydeepnetsgeneralizeevenwhenoverparametrized
est neighbors NN G(y) in the manifold of generated faces isanopenquestion, theyarecertainlynotimmunetoover-
G ={G(z)} z∼Z ofimagesfromthetraining (y ∈D) anda fitting. Intheextremecase, Zhangetal.[40]demonstrated
validationset (y 6∈ D). Notonlyisthisapproachmorero- randomlabelscanbeperfectlymemorizedevenonthelarge
bust, itprovidesuswithreconstructionserrorswhichcanbe scale Image Netdatabase.
analyzedfordifferentsetsofimages. Usingthisframework Despitethis, verylittleworkhasgoneintodefiningover-
thatwerefertoaslatentrecovery, weproposethefollowing fitting for generative models. In [1], the authors defined
contributions: generalizationfor GANsinalargelytheoreticalsetting. The
formulation was used to suggest a new GAN training pro-
• A demonstration of successful latent recovery across
tocol rather than provide an evaluation technique. In [2],
avarietyofgenerators. In Section2 weintroduceour
the support of a GAN generator, in terms of the number
optimizationprocedureandshowitismeaningfuleven
offaceidentitiesitcouldproduce, wasestimatedusingthe
ifthetargetimageiscorrupted.
birthday paradox heuristic. While crude, it suggested the
• Section3 introducesanovelmethodtonumericallyes- supportoffacescouldbequitelargewithrespecttothesize
timate overfitting in deep generators via statistics of ofthetrainingset. Theveryrecentworkof[14]attemptsto
recovery errors on test and train sets. Overfitting is numericallyestimatethenotionofoverfittingwitha Neural
undetectable for GANs, which is corroborated visu- Net Distance (NND).Thatis, theytrainaneuralnettodif-
ally in Fig. 3 and statistically in Table 1. Overfitting ferentiategeneratedsamplesfromrealsamplesandsimilar
ishoweverdetectableinhybridadversariallossessim- to [18], use the resulting divergence as a measure of qual-
ilar to Cycle GAN [42], and easily detectable in non- ity. Importantly, they are able to show a slight overfitting
adversarial generators such as GLO [4]. Finally, we for some GANs and show that this divergence penalizes a
show that standard evaluation metrics do not detect generatortriviallymemorizingthetrainset. Unfortunately,
overfittinginsomemodels. thisapproachrequiresamassivetestsetinordertotrainthe
NND, whichisunrealisticconsideringsuccessful GANsal-
1.1.Related Work
ready require massive train sets, not to mention the NND
Adversariallosseshaveseensuccessfulapplicationsina itselfneedstobetrained. Furthermore, the NNDmayfavor
variety of settings beyond just image generation: unpaired GANsifthedivergencetheyuseresemblesorisidenticalto
imagetoimagetranslationin Cycle GAN[42], faceattribute the GANunderevaluation.
modificationin Star GAN[8]andvariousimageinpainting Finally, a new class of generative models has recently
techniques[17,39]tonameafew. Thisprogresshascreated been proposed which involves invertible generators [10,
a huge need to evaluate generated image quality, which to 21]. Thesegeneratorsareattractiveastheyaremathemati-
somedegreehasnotbeenfullyanswered[6]. callywellmotivatedandadmitexactlog-likelihoodestima-
tions, viatakingthedeterminantofeachlayerjacobian.[28]
examined the log likelihoods for such models and showed
GAN Evaluation Metrics The Fre´chet Inception Dis-
that while some GAN models generalized nicely to vali-
tance (FID), recentlyintroducedin[16], hasbecomeastan-
dation samples, out of distribution samples, such as those
dard for evaluating the quality of generated GAN images.
takenfromcompletelydifferentdatasets, yieldhigherlike-
The FID is computed by computing the Fre´chet Distance
lihoods.[37]alsoexaminedlog-likelihoodinthegenerative
betweenfeaturesofthe Inceptionnetwork[34]modeledas
setting, andbothworksultimatelycautionedagainsttheuse
multivariate gaussians. Furthermore, it was demonstrated
oflog-likelihoodforgenerativeevaluation.
to be consistent with human evaluation. In the large scale
GANstudy[24], FIDwasusedtocompareahugevarietyof
GANs, wherein it was shown auxiliary factors such as hy- Memorization and Privacy Beyond these aspects of
perparametertuningcanobfuscatetruedifferencesbetween memorizationandpracticalevaluationofgeneratorsliesthe
GANs. In [33], notions of precision and recall are intro- importantanddebatedissueofprivacy: Howtoensurethat
ducedforgeneratedimages, tohelpcharacterizemodelfail- the data used for training cannot leak by some reverse en-
ure rather than providing a scalar in image quality. While gineering, such as reconstruction from features [25, 12] ?
these works have helped to compare GAN image quality, Because GANs have seen such widespread application, it
theydonotaddressoverfittingofthetrainingset. is imperative that we have better evaluation tools to assess
howmuchthesenetworkshaveoverfitthetrainingdata. For
Overfittingin Generative Networks Forimageclassifi- example, ifauserisusinganeuralnettoinpaintfacesasin
cation, a model is said to overfit when it performs signifi- [22]ortoperformsuper-resolutionenhancing[9], itseems
cantly better on training examples compared to test exam- necessary to ensure verbatim copies of training images do
ples, andsaidtogeneralizeotherwise. Whiletheexactrea- not appear, due to privacy or even copyright concerns. In-
11274


<!-- Page 3 -->

deed, severalattacksagainstmachinelearningsystemshave Experimental Validation In every experiment, we em-
beenexposedintheliterature[30]. Forinstance, authorsin ploy LBFGSandnoteditconvergesroughly10 xfasterthan
[12]designedaninversionattacktocoarselyrecoverfaces SGD(successfulrecoveryrequiringapproximately50 iter-
used during the training of a white box facial recognition ationsasopposedto500 in[5,23]). Although Eq.(NN ) is
G
neuralnetwork. Morerecently,[35]performedasuccessful highly non-convex, the proposed latent recovery optimiza-
membership attack, which is the ability to discern training tionworkswell, asshownin Fig.1 and Fig.3. Inparticular
examplesfromamodel, inapurelyblackboxsetting. Very for generated images y = G(z), where NN G(y) = z, a
recently[15]exploredthepotentialofmembershipattacks global minimum (verbatim copy) is consistently achieved
for GANs and exploited the tendency of the discriminator (see third row of Fig. 1). Every network analyzed in this
tooverfitthetrainingset. documentappearedtobeabletoverbatimrecovergenerated
images, anobservationalsonotedby[23]andexemplified
2.Reconstructionby Latent Code Recovery bythetightdistributionoferrorsnearzeroin Fig.4. Note
thatwealsoconsideredthewidelyusedperceptualloss[19]
Thissectionproposesamethodologyforreconstructing
bytakingφtobe VGG-19 features, witheithernoimprove-
the most similar images to target images with an existing
mentorevendegradationofvisualresults (seesupplemen-
generator. Inversion of deep representations has been al-
tary). Furthermore, we did not see any difference in the
ready addressed in the literature. [25] used a simple opti-
statisticalresultsof Section3 forperceptuallosses.
mizationproceduretomaximizeanoutputclassofa VGG-
like network. In the seminal works of [29, 36], a similar 2.2.Latent Code Recovery Under Distortion
inversion of deep nets unveiled adversarial examples. In
[24], generativenetworksareinvertedtostudyrecall, which It should be noted that Eq. (NN G ) by itself may not be
is the ability of the network to reproduce all images in the meaningful for some generators. For example, if the gen-
datasetandfinally[27]usedlatentrecoveryofa GANgen- erator is invertible, errorswill zero regardless of the target
eratortoevaluateitsquality. image. To verify that Eq. (NN G ) is meaningful, we want
Other works tackle recovering latent codes directly by to make sure the error is lower for images inside the con-
traininganencodernetworktosendimagesbackfromim- sideredmanifoldandlargeforthoseoutside. Todothis, we
age space to latent space, such as the BEGAN model [3] follow[16](usedtheretomotivatethe FID) whereinwetest
or Adversarially Learned Inference (AGI)[11]. In Genera- Eq.(NN G ) responsetovariousdistortions. Wechooseφto
tive Latent Optimization (GLO) [4], a generative model is beoneofthethreedistortionsthatareillustratedin Fig.2:
trained along with a fixed-size set of latent codes, so that • Smooth Vector Field Warp (Fig.2 a) Following [16,
theyareknownexplicitlywhentrainingfinishes. 41] we warp training images by bilinear interpolation
Inthispaper, wewillproceedbyrecoveringlatentcodes withasmooth2 Dvectorfield V = V ∗g, whichis
σd
viaoptimization, following[27,5,23,24]. Incontrastwith obtainedfromthe Gaussiansmoothinggofa Gaussian
[27], wewillultimatelybeconcernedcomparingimagere- randomvectorfield V(x, y)∼N(0, σ2);
d
coverybetweentrainandvalidationsets. • Corruption Noise Patches (Fig.2 b)As[22], wecor-
rupt training images by replacing patches of various
2.1.Latent Code Recoverywith Euclidean Loss
sizeswithfixed Gaussiannoisewithvarianceσ2;
d
Weexplorerecoverywithaeuclideanlossandfinditis • Additive Noise (Fig.2 c)Weaddnoisetoeachtraining
effective at recovering latent codes for a variety of GAN imagewith X =X+W , where W issampledfrom
n d d
methods. Here, we consider the following latent recovery a Gaussiandistribution W ∼N(0, σ2).
d d
optimizationproblem
z⋆(y)∈argminkφ(G(z))−φ(y) k2 2 (NN G ) Experimental Validation Fig.2 demonstratesafewfacts
z
aboutlatentrecovery. Byinspectionofrecoveredimages, it
where G is a deep generative network, z is the input la- appears robust enough to recover faces semantically simi-
tent vector and y is the target image. Using a solution z∗ lar to the ground truth even if the image has been heavily
of Problem (NN G ), we denote by NN G(y) = G(z∗) the distorted. Italsodemonstratestheprecisionofthenetwork,
Nearest Neighborrecoveryofagivenimagey inthesetof for example the three networks highlighted will reject im-
generated images, as opposed to the usual NN search in a agesonlyslightlyoutsidethemanifold. In Table1, wecan
dataset D: NN D(y) = argmin
x∈D
kx−yk. Inthiswork, see that not all networks share the same specificity. For
weconsidermostlyφastheidentity, butotheroperatorsare example, the GLO networks can recover distorted images
discussedinthenextparagraphandforapplicationssuchas with similar MRE’s to training images, which means the
super resolution in Section 4. Fig. 1 illustrates the differ- networksarelessprecise. Thisiscoupledwithalower FID
encebetweenthetwo NNsearchesonafewexamples. ofthenetwork, forexamplesee Fig.5.
11275


<!-- Page 4 -->

GANs for high resolution generation; progressive growing
of GANs [20], which we refer to as PGGAN and the zero
centered gradient penalty Resnet presented in [26], which
we refer to as MESCH. We train these three GANs on
Celeb A-HQwithatrainingsplitofthefirst26 kimagesand
thefirst70 kimagesof LSUNbedroomandtower. Wechose
thesesplitstopreservethequalityofeachmethod, as GAN
(a)Deformationbysmoothdiffeomorphism (warping) qualitysignificantlydegradeswithsmalldatasetsizes.
Generative Latent Optimization (GLO) Therecentlyin-
troduced Generative Latent Optimization (GLO) creates a
mapping from a fixed set of latent vectors to training im-
ages. The GLOobjectiveisasfollows
min L (G(z ), x ):=k G(z )−x k2 (2)
G X rec i i i i 2
(zi, xi)
(b)Unsupervisedinpainting (facecompletion)
wherex ∈Dreferstotrainingimages, z ∼N(0,1) sam-
i i
plesa Gaussiandistributionandthepairs (z , x ) aredrawn
i i
onceandforallbeforetrainingbegins1. Becauseweknow
thelatentdistributionis Gaussian, wecaneasilysamplethe
networkafteritistrained.
Auto Encoder Finally, we train a vanilla autoencoder on
Celeb A-HQwiththeobjective:
(c)Additivewhitenoise min L (G, E, x ) (3)
G, E X rec i
Figure 2: Median recovery error (MRE, see Eq. (4))
xi∈D
for 1800 test images on various GAN generators (PG- Hybrid Losses Weconsideragenerativemodelcombining
GAN[20], MESCH[26]and DCGAN[32]) undervarious boththeadversarialloss Eq.(1) witheuclideanautoencod-
distortionsφinlatentrecoveryoptimization (NN
G
)(seetext ingloss (3) whichwerefertoas AEGAN.
fordetails). Concerning models trained with a reconstruction loss
(GLO, AEGAN and AE), we selected these architectures
for a theoretical perspective, as they offer interesting win-
3.Using Latent Recoveryto Assess Overfitting dowsintohowgeneratorscanmemorize. Inparticular, we
willstudytheimpactofthetrainingsetsize N ontheover-
In this section, we train a variety of generative models
fitting inclination. For example, while we were unable to
with a training and validation split. Then, we analyze the
train a good quality GAN with a small set of images (say
difference between image recovery using Eq. (NN ), be-
G 256), GLOconvergesextremelyquicklyinsuchacase. See
tweentrainingandvalidationimages.
the GLO-256 networkin Fig.3(4 th row) wherememoriza-
3.1.Training Protocols tionisimmediatelyapparent. Asaresult, wewillreferre-
spectively to GLO-N, AEGAN-N and AE-N, to account
We summarize the details of each generative model be-
forthissize. Besides, forboththe AEand AEGANmodels,
low, intermsoftheirtrainingprocedureandpurposewithin
we forgo optimization in Eq. (NN ) and use the encoder
G
thiswork.
E (3) to recover the latent vector, as is natural for autoen-
GAN Generative Adversarial Networks (GAN) involve a
codermodels.
stochastictrainingprocedurewhichsimultaneouslytrainsa
Inthenextparagraphs, wewillproceedtoshowthatitis
discriminatorandagenerator. Theoriginal GAN[13]opti-
possible for generative networks to memorize in the sense
mizationproblemwrites
thatvalidationandtrainingsetshavesignificantlydifferent
recoveryerrordistributions.
maxmin E [L (D, G, z, x)] (1)
D G
z∼N(0,1), x∼pdata adv
3.2.Comparisonofrecoveryerrors
where L (D, G, z, x)=log (D(x))+log (1−D(G(z))).
adv
We examine three prominent GANs in the literature. First Figure 4 shows the histograms of recovery errors on
is DCGAN[32], asitisoneofthemostwidelyused GAN train (D in green) and validation (T in red) datasets from
architecturesandwithstilldecentperformanceacrossava- 1 Contraryto[4], wedonotoptimizethelatentspaceandfoundno
riety of datasets [24]. Then we study two state-of-the-art negativeimpactonthereconstructioncapacityandgenerationquality.
11276


<!-- Page 5 -->

ytegrat
NAGGP
HCSEM
OLG
NAGEA
y ∈D(train) y ∈T (validation)
Figure3:Latentrecoveryoftrainingimagesfrom D(left, greenframe) andtestimagesfrom T (right, redframe) for128×128
imagesof Celeba-HQ[20]. Fromtoprowtobottomarefirsttargetimages, andthenrecoveryfrom Progressive GANs[20]
(PGGAN), 0-GP resnet GAN [26] (MESCH), a GLO network [4], and finally a Cycle-GAN like network [42] (AEGAN).
While GLOobviouslyshowssomememorizationoftrainingexamples, itishardtovisuallyassesswhenoverfittinghappens
forothermethods, asdiscussedin Section3(withadditionaldetailsonarchitecturesandtraining).
Celeb A-HQ, for various generators. For the sake of read- 3.3.Statistical Analysis
ability, thedistributionoferrorsforgeneratedimages (yel-
In light of the previous results, we propose two simple
low) from G anddistortedimages (blue) areonlydisplayed
definitions to measure and detect overfitting without rely-
for PGGAN and MESCH. Confirming visual inspection
ingonhistogramsorimageinspection. First, tosummarize
from Fig. 1, observe that the recovery errors for generated
thedistributionoferrorstoasinglevalue, weconsiderthe
images (inyellow) arequitelow. Increasingthenumberof
Median Recovery Error (MRE), definedforagenerator G
iterationsandusingseveralrandominitializationsimprove
andadataset Y as
results, buthavenotbeenusedtoreducecomputationcosts.
MRE (Y)=median minky −G(z) k2 (4)
G n i o
Nowwearegoingtoconsiderthedistributionofrecov- z yi∈Y
ery errors for test and train. For GLO-N and AEGAN-N
Table 1 reports such values for other deep generators and
generators with N ∈ {128,1024,8192}, the difference is
otherdatasets.
clear, and is decreasing with the number of training im-
Thentomeasurethedistancebetweentwodistributions,
ages N. For very small datasets of N = 128, the train
thatistoestimatetowhichextentthegeneratoroverfitsthe
andvalidationerrordistributionsaredisjoint. Ontheother
trainingset, wesimplycomputethenormalized MRE-gap
hand, pure GAN models can not be successfully trained
betweenvalidation T andtrain Ddataset, whichwrites
withsmalldatasets. Wethereforeonlytrained PGGANand
MESH with full datasets and in both cases, the difference
MRE-gap =(MRE (T)−MRE (D))/MRE (T) (5)
ofrecoveryerrordistributionbetweenthetrain (green) and G G G G
validation (red) set is barely noticeable. Further statisti-
Thesevaluesarereported2 in Table1.
cal analysis in the next paragraph shows indeed that such
a small gap is very likely for two samples drawn from the 2 Noticethatothermetricscouldhavebeenused, suchasthe Wasser-
samelaw, demonstratinggeneralization. steindistance.
11277


<!-- Page 6 -->

(a)PGGAN (b)MESCH
(c)GLO-128 (d)AEGAN-128
Figure5: Comparisonof FIDversus Median Recovery Er-
ror (MRE) for various models computed over training im-
ages (in green) and validation images (in red). FID does
notdetectmemorizationin GLOmodels.
on Celeb A-HQand N = 32768 on LSUNbedroom, over-
(e)GLO-1024 (f)AEGAN-1024 fittingisnolongerdetectable.
However, usingtheproposedstatistics (p-value, normal-
ized MRE-gap) ismuchmorepracticaltodetectoverfitting
thanonlyinspectinghistogramsandeasiertothresholdthan
MREitself. Italsoillustratesthatsuchstatisticalprinciple
overrules empirical evaluation, as memorization is indeed
sometimesquitehardtotellfromsimplevisualinspection,
(g)GLO-8192 (h)AEGAN-8192 suchasforthe AEGANgeneratorin Fig.3.
Figure 4: Histograms of recovery errors on train D and 3.4.FIDDoes Not Detect Memorization
validation T datasetsfrom Celeb A-HQshowingthatover-
The FIDisthestandard GANevaluationmetricforim-
fittingisnothappeningfor PGGANand MESCHgenerators
ages[16], soitisnaturaltoaskwhetherthismetriccanbe
on the training dataset, but is for GLO-N and AEGAN-N
usedtodetectmemorization. Figure5 displays FIDscores
whentrainingforasmalldataset N ≤8192.
computedbetweengeneratedandtrainingimages (ingreen)
andgeneratedandtestimages (inred).Whilethemedianre-
Insteadofusinganempiricalthresholdtoautomatically coveryerror (MRE) isabletodetectmemorizationin GLO
assess if the amount of overfitting is significant regarding models, the FID is not sensitive to this (this fact was also
thesizeofthetrainingset, werelyonastatisticaltest. We noted by [14]). We do not suggest replacing the FID, but
computethep-valueofthe Kolmogorov-Smirnovtest (KS) rather using MRE to provide a more complete picture of
which measures the probability that two random samples generator performance. Besides, other metrics such as the
drawnfromthesamedistributionhavealargerdiscrepancy, precisionrecallintroducedin[33]canbeconsideredaswell
definedasthemaximumabsolutedifferencebetweencumu- to tackle more subtle statistical biases such as mode drop-
lativeempiricaldistributions, thantheoneobserved. pingversusmodeinvention.
Suchp-valuesaredisplayedin Table1, andathresholdof
1%isusedtodetectoverfitting (valuesarehighlighted). To 4.Discussionand Future Work
showtheconsistencybetweenthetwoproposedmetric, we
4.1.Noteson Applications
alsohighlightthevaluesof MRE-gapthatareabove10%.
Observe thatthe results aremostly confirming previous Recently, GANs have seen wide application to vari-
empiricalevidence: memorizationisstronglycorrelatedto ous face generation tasks, such as face attribute modifica-
thenumberofimagesseenduringtraining. Wealsoseethat tion [8], generative face completion [22] and face super-
the same overfitting occurs on different datasets (Celeb A- resolution[9]. Inasimilarvein, deepimageprior[38], re-
HQand LSUN), andforautoencoder (AE).At N =26000 covers images by first fixing a random latent vector, then
11278


<!-- Page 7 -->

Table1: Kolmogorov-Smirnov (KS) p-values, normalizedmedianerrordifference (MRE-gap)Eq.(5), and Medianrecovery
errors (MRE)Eq.(4) foravarietyofgenerators. Highlightedvaluesindicategeneratorsforwhichoverfittingofthetraining
sethasbeendetected:(inblue) withthe KStestusing1%thresholdonp-value,(ingreen) using10%thresholdon MRE-gap.
KSp-value MRE-gap MRE
trainvsval train val generated distortion
DCGAN 9.43 e-01 1.79 e-02 4.95 e-02 5.04 e-02 3.68 e-03 5.69 e-02
MESCH 4.55 e-01 6.96 e-03 3.40 e-02 3.43 e-02 1.77 e-02 4.63 e-02
PGGAN 2.22 e-01 2.22 e-02 3.31 e-02 3.39 e-02 1.78 e-02 4.65 e-02
GLO-128 0.00 e+00 9.70 e-01 9.94 e-04 3.30 e-02 5.10 e-05 9.32 e-03
GLO-1024 0.00 e+00 7.59 e-01 1.95 e-03 8.08 e-03 1.29 e-03 4.46 e-03
GLO-8192 2.25 e-18 1.75 e-01 3.00 e-03 3.64 e-03 1.04 e-03 3.20 e-03
GLO-26000 2.12 e-01 3.69 e-02 4.27 e-03 4.44 e-03 4.08 e-04 4.43 e-03
Celeb A-HQ
AE-128 0.00 e+00 9.68 e-01 3.36 e-03 1.06 e-01 N/A 1.80 e-02
AE-1024 0.00 e+00 9.35 e-01 4.19 e-03 6.45 e-02 N/A 1.80 e-02
AE-8192 0.00 e+00 7.60 e-01 8.04 e-03 3.34 e-02 N/A 1.67 e-02
AEGAN-128 0.00 e+00 9.02 e-01 1.54 e-02 1.57 e-01 N/A 2.82 e-02
AEGAN-1024 0.00 e+00 2.68 e-01 8.52 e-02 1.16 e-01 N/A 8.69 e-02
AEGAN-8192 3.17 e-27 1.61 e-01 7.42 e-02 8.84 e-02 N/A 7.55 e-02
AEGAN-26000 1.25 e-01 1.85 e-02 9.96 e-02 1.01 e-01 N/A 1.00 e-01
DCGAN(tower) 7.02 e-02 1.36 e-02 7.96 e-02 8.07 e-02 1.49 e-02 7.31 e-02
DCGAN(bedroom) 3.65 e-01 5.34 e-03 7.06 e-02 7.10 e-02 7.03 e-02 7.09 e-02
LSUN
GLO-8192(bedroom) 6.70 e-06 1.70 e-01 5.45 e-03 6.56 e-03 5.37 e-04 5.01 e-03
GLO-32768(bedroom) 2.62 e-01 5.40 e-02 6.58 e-03 6.25 e-03 8.40 e-04 5.44 e-03
DCGAN 2.41 e-01 8.85 e-02 3.00 e-02 2.75 e-02 6.89 e-03 -
GLO-1024 0.00 e+00 6.78 e-01 2.86 e-04 8.88 e-04 1.49 e-03 -
MNIST
GLO-16384 3.48 e-01 6.45 e-03 8.72 e-04 8.77 e-04 1.41 e-03 -
AEGAN-16384 7.43 e-02 2.29 e-02 4.56 e-02 4.67 e-02 N/A -
DCGAN 5.40 e-01 3.65 e-03 2.29 e-01 2.28 e-01 1.30 e-03 -
CIFAR10
GLO-1024 0.00 e+00 5.84 e-01 2.77 e-03 6.67 e-03 8.53 e-04 -
GLO-16384 3.48 e-01 6.45 e-03 8.72 e-04 8.77 e-04 1.41 e-03 -
optimizing over the parameters of a randomly initialized means that it will not be able to verbatim recover an iden-
generator. We apply (NN ) to face inpainting and super tityfoundinthedataset. Forsomeapplicationsthiscouldbe
G
resolutionfortworeasons;firstitshowsoff-the-shelf GAN seenasinadequate, suchasthedomaintranslationnetwork
generatorsarewellsuitedforavarietyofdownstreamtasks, of Star GAN[8], whereinauserwantstoretainidentitybut
whichisalsonotedin[39]andseconditprovidesadditional changefacialfeatures. Ontheotherhand, ifafacedataset
visualinsightintotheobservationsoftheprevioussection. isconsideredprivateorcopyrighted, notverbatimcopying
anytrainingimagecanbeseenasabenefitofthealgorithm.
Figure6 showstheprogressive GANgenerator[20]ap- Quantifying whether GAN generators really do generalize
plied to face inpainting (φ is a mask) and super-resolution withrespecttoidentity, usingafaceidentificationnetwork
(φisa64 xpooling). Whilethefaceinpaintingisartifacted, like VGG-Face[31], isaninterestingissuethatweleavefor
wenotethattheresultsaredecentwithoutanypostprocess- futurework.
ingandsimilartothosepresentedin[22](whilebeingnon-
feedforward). Asforsuper-resolution, weobtainresultsat
4.2.Future Work
leastonparwith[9]. Anintriguingpropertyoftheimages
isthattherecoveryissemanticallyaccurate, intermsofat- Our work is a part of a growing body of research con-
tributessuchasgender, facialfeaturesandpose, whilstre- cernedwithoverfittingofdeepgenerativemodels[15,14].
coveringafacethatappearstobeadifferentidentity. This For example, [14] takes a perspective of GAN evaluation,
happensdespitetheuseofimagesthatthe PGGANgener- arguing that evaluation with a neural net distance can pe-
ator [20] was trained on, which is in accordance with the nalizetrivialmemorizationofthedatasetwhereas FIDcan-
observations of Sec. 3. Put in another way, we believe the not. We have a similar perspective, albeit with the goal of
factthat PGGANhasgeneralizedwellto Celeb A-HQ, also merely detecting overfitting and with a simpler approach.
11279


<!-- Page 8 -->

φ(y) G(z∗(y)) y the visual interpretability of our results. For example, one
canseein Fig.3, thevisualreconstructionsdoinfactreveal
visual quality of train versus validation samples. Namely,
both training and validation images are well reconstructed
for the GAN methods. We believe further work must be
donetosynthesizetheresultsof[14,15]andourwork. One
promising area would be to explore other loss functions.
We experimented briefly with perceptual losses (see sup-
plementary) but leave this possibility open. We also think
recovery could be guided with a learned NND loss. An-
otherinterestingdirectionisanalysisoflocaloverfittingon
image patches. Preliminary experiments can be found the
insupplementarymaterial, whichalsoshowgeneralization
of GANgenerators. Finally, Eq.(NN ) hadmixedsuccess
G
formorecomplexdatasetssuchas LSUNintermsofvisual
quality. Wethinkthatsomedatasetsleadtomorecomplex
latent space with many local minima and direct the reader
tothesupplementarymaterialformoredetailsonoptimiza-
tion.
Conclusion In this work, we studied overfitting of deep
generators through latent recovery. We saw that a sim-
ple Euclidean loss was effective at recovering latent codes
and recovers plausible images even after image transfor-
mations. We used this fact to study whether a variety of
deep generators memorize training examples by asking if
the network can generate validation samples. Our statisti-
Figure 6: Off the shelf application of Eq. (NN G ) with a cal analysis revealed that overfitting was undetectable for
1024×1024 generator PGGAN. From left to right: trans- GANs, but detectable for hybrid adversarial methods like
formedimageφ(y), recoveredimage G(z∗(y)) andground AEGAN and non-adversarial methods like GLO, even for
truthimagey. Thefirsttworowsareimagesuper-resolution training sets of moderate sizes. Due to the ever-growing
(φisamask) andnexttwoareimageinpaintingofimages concerns on privacy or copyright of training data and the
downsampledbyafactorof64(φisanaveragepooling). already widespread application of generative methods, we
providemethodologythatisastepintherightdirectionto-
wardsanalysisofgenerativeoverfitting.
Unlike our approach, the NND [18, 14] was able to de-
tectslightoverfittingofsome GANs, however, themassive
Acknowledgments Thisworkwassupportedbyfundings
size of the validation set and the fact that the NND must
from Re´gion Normandie under grant RIN Norman D’eep.
beretrainedforeverygeneratorunderevaluationmakethe
Thisstudyhasalsobeencarriedoutwithfinancialsupport
analysiscomputationallyburdensome. Additionally, thear-
fromthe French State, managedbythe French National Re-
chitecture choice of the NND may be biased to reflect the
search Agency (ANRGOTMI)(ANR-16-CE33-0010-01).
GANlossfunctionandnotbeuniversalacrossmodels, such
as the reconstruction based GLO model considered in this
References
work. On the other hand, we present a simple, computa-
tionally tractable solution requiring modestly sized (here,
[1] S.Arora, R.Ge, Y.Liang, T.Ma, and Y.Zhang. Generaliza-
just a few thousand) validation images. We found the re-
tionandequilibriumingenerativeadversarialnets (GANs).
constructionbasedgenerator GLOtobeinterestingfroma
In Proceedingsofthe34 th International Conferenceon Ma-
theoreticalperspective. Forexample, optimizationunveiled chine Learning, volume70, pages224–232.PMLR,06–11
strong overfitting; training images are nearly verbatim re- Aug2017. 2
covered and validation images are blurry (e.g. Fig. 3, row [2] S. Arora, A. Risteski, and Y. Zhang. Do GANs learn the
4), whereas FID on GLO samples was insensitive to this distribution? some theory and empirics. In International
difference. Furthermore, amajoradvantageofourworkis Conferenceon Learning Representations,2018. 2
11280


<!-- Page 9 -->

[3] D. Berthelot, T. Schumm, and L. Metz. Began: boundary [19] J.Johnson, A.Alahi, and L.Fei-Fei. Perceptuallossesfor
equilibriumgenerativeadversarialnetworks. ar Xivpreprint real-time style transfer and super-resolution. In European
ar Xiv:1703.10717,2017. 3 Conference on Computer Vision, pages 694–711. Springer,
[4] P.Bojanowski, A.Joulin, D.Lopez-Pas, and A.Szlam. Opti- 2016. 3
mizingthelatentspaceofgenerativenetworks. In Proceed- [20] T. Karras, T. Aila, S. Laine, and J. Lehtinen. Progressive
ingsofthe35 th International Conferenceon Machine Learn- growingofgansforimprovedquality, stability, andvariation.
ing, volume80, pages600–609.PMLR,10–15 Jul2018. 2, Sixth International Conferenceon Learning Representations
3,4,5 (ICLR),2018. 1,4,5,7
[5] A.Bora, A.Jalal, E.Price, and A.G.Dimakis. Compressed [21] D.P.Kingmaand P.Dhariwal. Glow: Generativeflowwith
sensing using generative models. Thirty-fifth International invertible1 x1 convolutions. In Advancesin Neural Informa-
Conferenceon Machine Learning (ICML),2018. 3 tion Processing Systems, pages10236–10245,2018. 2
[6] A.Borji. Prosandconsofganevaluationmeasures. ar Xiv [22] Y. Li, S. Liu, J. Yang, and M.-H. Yang. Generative face
preprintar Xiv:1802.03446,2018. 2 completion. In The IEEE Conference on Computer Vision
[7] A. Brock, J. Donahue, and K. Simonyan. Large scale gan and Pattern Recognition (CVPR), volume1, page3,2017.2,
training for high fidelity natural image synthesis. In Pro- 3,6,7
ceedings of the 36 th International Conference on Machine [23] Z.C.Liptonand S.Tripathi. Preciserecoveryoflatentvec-
Learning,2019. 1 torsfromgenerativeadversarialnetworks. ICLR,2017. 3
[8] Y.Choi, M.Choi, M.Kim, J.-W.Ha, S.Kim, and J.Choo. [24] M.Lui, K.Kurach, M.Michalski, S.Gelly, and O.Bousquet.
Stargan: Unifiedgenerativeadversarialnetworksformulti- Areganscreatedequal? alarge-scalestudy. In Advancesin
domainimage-to-imagetranslation.ar Xivpreprint,2018.2, Neural Information Processing Systems (NIPS),2018. 2,3,
6,7 4
[9] R. Dahl, M. Norouzi, and J. Shlens. Pixel recursive super [25] A. Mahendran and A. Vedaldi. Understanding deep image
resolution. In2017 IEEEInternational Conferenceon Com- representations by inverting them. In Proceedings of the
puter Vision (ICCV), pages5449–5458, Oct2017. 2,6,7 IEEE conference on computer vision and pattern recogni-
[10] L.Dinh, J.Sohl-Dickstein, and S.Bengio. Densityestima- tion, pages5188–5196,2015. 2,3
tionusingrealnvp. In International Conferenceon Learning [26] L.Mescheder, A.Geiger, and S.Nowozin. Whichtraining
Representations,2017. 2 methods for gans do actually converge? In International
[11] V. Dumoulin, I. Belghazi, B. Poole, O. Mastropietro, Conferenceon Machine Learning, pages3478–3487,2018.
A. Lamb, M. Arjovsky, and A. Courville. Adversarially 4,5
learnedinference. ICLR,2017. 3 [27] L.Metz, B.Poole, D.Pfau, and J.Sohl-Dickstein. Unrolled
[12] M. Fredrikson, S. Jha, and T. Ristenpart. Model inversion generativeadversarialnetworks. ICLR,2017. 3
attacksthatexploitconfidenceinformationandbasiccoun- [28] E. Nalisnick, A. Matsukawa, Y. W. Teh, D. Gorur, and
termeasures. In Proceedingsofthe22 nd ACMSIGSACCon- B. Lakshminarayanan. Do deep generative models know
ference on Computer and Communications Security, pages what they don’t know? In International Conference on
1322–1333.ACM,2015. 2,3 Learning Representations,2019. 2
[13] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, [29] A.Nguyen, J.Yosinski, and J.Clune. Deepneuralnetworks
D.Warde-Farley, S.Ozair, A.Courville, and Y.Bengio. Gen- areeasilyfooled: Highconfidencepredictionsforunrecog-
erativeadversarialnets. In Advancesinneuralinformation nizableimages. In Proceedingsofthe IEEEConferenceon
processingsystems, pages2672–2680,2014. 1,4 Computer Vision and Pattern Recognition, pages 427–436,
[14] I.Gulrajani, C.Raffel, and L.Metz. Towards GANbench- 2015. 3
marks which require generalization. In International Con- [30] N.Papernot, P.Mc Daniel, A.Sinha, and M.Wellman. To-
ferenceon Learning Representations,2019. 2,6,7,8 wardsthescienceofsecurityandprivacyinmachinelearn-
[15] J. Hayes, L. Melis, G. Danezis, and E. De Cristofaro. ing. 3 rd IEEE European Symposium on Security and Pri-
Logan: Membership inference attacks against generative vacy,2018. 3
models. Proceedings on Privacy Enhancing Technologies, [31] O. M. Parkhi, A. Vedaldi, and A. Zisserman. Deep face
2019(1):133–152,2019. 3,7,8 recognition. In British Machine Vision Conference, 2015.
[16] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and 7
S.Hochreiter. Ganstrainedbyatwotime-scaleupdaterule [32] A.Radford, L.Metz, and S.Chintala. Unsupervisedrepre-
convergetoalocalnashequilibrium. In Advancesin Neural sentationlearningwithdeepconvolutionalgenerativeadver-
Information Processing Systems, pages6626–6637,2017.2, sarialnetworks. ar Xivpreprintar Xiv:1511.06434,2015. 1,
3,6 4
[17] S. Iizuka, E. Simo-Serra, and H. Ishikawa. Globally and [33] M. S. Sajjadi, O. Bachem, M. Lucic, O. Bousquet, and
locallyconsistentimagecompletion. ACMTransactionson S.Gelly. Assessinggenerativemodelsviaprecisionandre-
Graphics (TOG),36(4):107,2017. 2 call. ar Xivpreprintar Xiv:1806.00035,2018. 2,6
[18] D.J.Im, A.H.Ma, G.W.Taylor, and K.Branson. Quantita- [34] T.Salimans, I.Goodfellow, W.Zaremba, V.Cheung, A.Rad-
tivelyevaluating GANswithdivergencesproposedfortrain- ford, and X.Chen. Improvedtechniquesfortraininggans. In
ing. In International Conference on Learning Representa- Advancesin Neural Information Processing Systems, pages
tions,2018. 2,8 2234–2242,2016. 2
11281


<!-- Page 10 -->

[35] R.Shokri, M.Stronati, C.Song, and V.Shmatikov. Member-
shipinferenceattacksagainstmachinelearningmodels. In
Securityand Privacy (SP),2017 IEEESymposiumon, pages
3–18.IEEE,2017. 3
[36] C. Szegedy, W. Zaremba, I. Sutskever, J. Bruna, D. Erhan,
I.Goodfellow, and R.Fergus. Intriguingpropertiesofneural
networks. ar Xivpreprintar Xiv:1312.6199,2013. 3
[37] L.Theis, A.vanden Oord, and M.Bethge. Anoteonthe
evaluationofgenerativemodels. In International Conference
on Learning Representations, Apr2016. 2
[38] D. Ulyanov, A. Vedaldi, and V. Lempitsky. Deep image
prior. In Proceedingsofthe IEEEInternational Conference
on Computer Visionand Pattern Recognition,2018. 6
[39] R. A. Yeh, C. Chen, T.-Y. Lim, A. G. Schwing,
M.Hasegawa-Johnson, and M.N.Do. Semanticimagein-
paintingwithdeepgenerativemodels. In CVPR, volume2,
page4,2017. 2,7
[40] C. Zhang, S. Bengio, M. Hardt, B. Recht, and O. Vinyals.
Understandingdeeplearningrequiresrethinkinggeneraliza-
tion. ar Xivpreprintar Xiv:1611.03530,2016. 2
[41] R.Zhang, P.Isola, A.A.Efros, E.Shechtman, and O.Wang.
Theunreasonableeffectivenessofdeepfeaturesasapercep-
tualmetric. The IEEEConferenceon Computer Visionand
Pattern Recognition (CVPR),2018. 3
[42] J.-Y.Zhu, T.Park, P.Isola, and A.A.Efros. Unpairedimage-
to-image translation using cycle-consistent adversarial net-
works. ar Xivpreprint,2017. 2,5
11282
