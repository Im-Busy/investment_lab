<!-- Page 1 -->

Underreviewasaconferencepaperat ICLR2023
USING THE TRAINING HISTORY TO DETECT AND PRE-
VENT OVERFITTING IN DEEP LEARNING MODELS
Anonymousauthors
Paperunderdouble-blindreview
ABSTRACT
Overfittingoccursindeeplearningmodelswheninsteadoflearningfromthetrain-
ingdata, theytendtomemorizeit, resultinginpoorgeneralizability. Overfitting
can be (1) prevented (e.g., using dropout or early stopping) or (2) detected in a
trainedmodel (e.g., usingcorrelation-basedmethods). Weproposeamethodthat
canbothdetectandpreventoverfittingbasedonthetraininghistory (i.e., valida-
tionlosses). Ourmethodfirsttrainsatimeseriesclassifierontraininghistoriesof
overfitmodels. Thisclassifieristhenusedtodetectifatrainedmodelisoverfit.
In addition, our trained classifier can be used to prevent overfitting by identify-
ing the optimal point to stop a model’s training. We evaluate our method on its
ability to identify and prevent overfitting in real-world samples (collected from
papers published in the last 5 years at top AI venues). We compare our method
againstcorrelation-baseddetectionmethodsandthemostcommonlyusedpreven-
tionmethod (i.e., earlystopping). Ourmethodachievesan F1 scoreof0.91 which
isatleast5%higherthanthecurrentbest-performingnon-intrusiveoverfittingde-
tection method. In addition, our method can find the optimal epoch and avoid
overfitting at least 32% earlier than early stopping and achieve at least the same
rate (oftenbetter) ofachievingtheoptimalepochasearlystopping.
1 INTRODUCTION
Overfitting is one of the fundamental issues that plagues the field
ofmachinelearning (Nowlan&Hinton,1992;Ng,1997;Caruana
etal.,2000;Cawley&Talbot,2007;Erhanetal.,2010;Srivastava Training loss
etal.,2014;Zhaoetal.,2020), whichcanalsooccurwhentraining Validation loss
adeeplearning (DL) model. Anoverfitmodelincreasestheriskof
inaccurate predictions, misleading feature importance, and wasted
resources (Hawkins, 2004). Figure 1 shows example training his-
tories (i.e., the training and validation losses curves) of an overfit
and a non-overfit model. The training and validation losses of the
overfitmodelbothdecreaseatthebeginningofthetrainingprocess. (a)Overfitting
Followingthat, thevalidationlossincreaseswhilethetrainingloss
decreases, resulting in a large gap between the training and vali- Training loss
Validation loss
dation losses. Such a trend indicates that the trained model is not
generalizingwelltonewdata.
Currently, theproblemofoverfittingisaddressedbyeither (1) pre-
venting it from happening in the first place or (2) detecting it in
a trained model. Overfitting prevention methods stop overfitting
from happening through methods such as early stopping (Morgan (b)Non-overfitting
& Bourlard, 1989), data augmentation (Shorten & Khoshgoftaar,
2019), regularization (Kukacˇkaetal.,2017), modifyingthemodel Figure 1: Example training
byaddingdropoutlayers (Srivastavaetal.,2014) orbatchnormal- histories of overfit and non-
ization (Ioffe & Szegedy, 2015). Many of these methods are in- overfitmodels.
trusive and require modifying the data or the model structure and
expertisetoexecutecorrectly. Furthermore, eventhenon-intrusive
prevention methods such as early stopping incur a trade-off between model accuracy and training
time (Prechelt,2012). Forexample, whenusingtheearlystoppingmethod, stoppingtoolatemay
1


<!-- Page 2 -->

Underreviewasaconferencepaperat ICLR2023
improve model accuracy but also increase training time while stopping too early could result in a
modelthatperformssub-optimally.
Overfittingdetectionmethodstypicallyattempttoidentifyifatrainedmodelisoverfitbyretraining
themodelwithnoisydatapointsandobservingtheimpactofthesenoisydatapointsonthemodel’s
accuracy (as an overfit model can learn the noise to reduce the impact) (Zhang et al., 2019). Al-
ternatively, some detection methods check the hypothesis that the trained model and the data are
independent, e.g., Werpachowskietal.(2019) checkthehypothesisbycomparingthetesterrorwith
theestimatedtesterrorbasedonadversarialexamplesofthetestset. However, similartointrusive
overfitting prevention methods, significant expertise is typically required to use existing detection
methods. In addition, these methods require extra computational resources for activities such as
generatingadversarialexamples, retrainingthemodels, andconvertingthemodel.
Inthispaper, wearethefirsttoproposeamethodforbothoverfittingdetectionandpreventionbased
on training histories. Training histories have been used by researchers before to make decisions
suchasquantitativedataacquisitionandmodelselection (van Rijnetal.,2015;Strangetal.,2018;
Bornscheinetal.,2020;Mohr&van Rijn,2021;2022;Brazdiletal.,2022). Similarly, ourmethod
trains a time series classifier on a simulated dataset of training histories (i.e., labelled validation
losscurvesoverepochsoftraining) ofmodelsthatoverfitthetrainingdata. Ourtrainedtimeseries
classifierdetectsifatrained DLmodelisoverfittingthetrainingdatabyinspectingthevalidationloss
history (whichiscapturedaspartofthetraininghistory).Incontrasttoexistingoverfittingdetection
methods, our method does not incur additional resources or costs since the training history is a
byproductofthetrainingprocess. Additionally, ourmethod (i.e., thetrainedtimeseriesclassifier)
can be used to prevent overfitting based on the validation losses of recent epochs (e.g., the last 20
epochs).
Whilewetrainourmethodonasimulateddataset, weevaluateitonareal-worlddataset, collected
frompapersfromtop AIvenuesfromthelast5 years. Wecollectedthetraininghistoriesfromthese
papersthatareexplicitlylabelledasoverfittingornon-overfittingbytheauthorsasthegroundtruth.
Ourresultsshowthatourmethodoutperformsthestate-of-the-artbyatleast5%intermsof F-score
for overfitting detection, with an F-score of 0.91. In addition, our method can prevent overfitting
from happening at least 32% earlier than early stopping while having the same (and often better)
rateofachievingtheoptimalepoch.
2 BACKGROUND AND RELATED WORK
2.1 OVERFITTING
Overfittingisawell-knownandexploredproblemintheareaofmachinelearning (Nowlan&Hin-
ton, 1992; Ng, 1997; Caruana et al., 2000; Cawley & Talbot, 2007; Erhan et al., 2010; Srivastava
et al., 2014; Zhao et al., 2020). Recent research has further noted the widespread presence and
impactof overfittingin thesub-fieldsof machinelearning includingreinforcementlearning (Song
etal.,2020), adversariallearning (Riceetal.,2020), andrecommendersystems (Pengetal.,2021).
For recommender systems that deal with massive amounts of data every day, incremental model
updatesarerequiredtocatchthemostrecenttrend. However, theincrementallyupdatedmodelmay
overfittothemostcurrentdataandforgetpreviouslylearnedknowledge (Pengetal.,2021). Song
etal.(2020) studytheobservationaloverfittingregimeinreinforcementlearning, whichoverfitsto
onlyasmallproportionoftheobservationspace. Furthermore, Riceetal.(2020) reportthatover-
fittinghappensmorefrequentlyinadversarialtrainingthanintraditional DL.Overfittinghurtsthe
generalizabilityofatrainedmodel, butgenerallypredictingwhetheramodelwilloverfittoacertain
datasetbeforetrainingitisformallyundecidable (Bashiretal.,2020).Inthispaper, westudyhowto
detectifatrainedmodelisoverfitandhowoverfittingcanbepreventedfromhappeningduringthe
trainingprocess. Below, wegiveanoverviewofexistingmethodstodetectandpreventoverfitting,
andwedescribethemethodsthatweusedasbaselinestoevaluatetheaccuracyofourmethod.
2.2 OVERFITTINGDETECTION
In the field of symbolic regression, Kronberger et al. (2011) propose computing Spearman’s
non-parametric rank correlation coefficient (Spearman, 1987) between training and validation fit-
ness (i.e., anevaluationmetricforthesymbolicregressionmodel) todetectoverfitting. Researchers
2


<!-- Page 3 -->

Underreviewasaconferencepaperat ICLR2023
havealsostudiedhowtodetectoverfittingbyinjectingnoiseintothetrainingdataorgeneratingnew
data. Theytypicallyretrainthemodelthatisbeingtestedwiththisnoisyornewdataandobserve
theimpactonitsperformancetodetectifthemodelisoverfitting. Forinstance, Zhangetal.(2019)
propose a Perturbation Validation (PV) method, which retrains the model after injecting different
levelsofnoise (perturbation) intothelabels. Theyretrainthemodelforeachnoiselevelandcollect
thetraininghistorytocomputethe PVmeasurement. The PVmeasurementshowshowtheaccuracy
changes in response to the injected noise and indicates that overfitting is present if the accuracy
does not decrease significantly on the noisy data. Werpachowski et al. (2019) generate adversar-
ialexamplestodetectwhetheranimageclassificationmodelisoverfittothetestset. Chatterjee&
Mishchenko (2020) describeamethodthatconvertsmachinelearningmodelstologiccircuitsand
detects overfitting by inspecting rare patterns of handling training samples in the model. In this
paper, weproposeanoverfittingdetectionmethodbasedontimeseriesclassifiersthatonlyrelieson
thetraininghistoryofatrainedmodelanddoesnotinvolvemodelconversionorretraining. Wetrain
thetimeseriesclassifierswithtraininghistoriesandlabelsindicatingwhetherornotthereisoverfit-
ting, hence, theclassifierscanidentifyoverfittingfromthetraininghistoryofatrainedmodel. Since
similartoourmethod, correlation-basedmethodsdetectoverfittingbasedonthetraininghistoryas
well, weselectthemasthebaselinetocompareourmethodagainstandintroduceitbelow.
Correlation-based methods. Inspired by the overfitting detection method of Kronberger et al.
(2011), wecomputecorrelationmetricsbetweenthetrainingandvalidationlosstodetectoverfitting
in DL models. The idea behind this method is intuitive: the training and validation loss (similar
tothetrainingandvalidationfitnessinsymbolicregression) areexpectedtobestronglycorrelated
when there is no overfitting and the correlation should be weak when there is overfitting. The
calculated correlation metrics are compared with a threshold (more details on how we select the
threshold in Section 4) to determine if there is overfitting. We choose three correlation metrics:
Spearman, Pearson (Hauke&Kossowski,2011), andtime-lagged Pearsoncorrelationcoefficients.
We calculate both Spearman and Pearson correlation coefficients since we do not know whether
therelationshipbetweentrainingandvalidationlossislinear. Inaddition, wecomputethe Pearson
correlation coefficient between the time-lagged version (5-epoch lagged) of the training loss and
the validation loss. This method is inspired by autocorrelation (Brockwell & Davis, 2002), which
computesthecorrelationbetweenatimeseriesdataandatime-laggedversionofitself.
2.3 OVERFITTINGPREVENTION
Bejani & Ghatee (2021) identify three categories of overfitting prevention methods: passive, ac-
tiveandsemi-activemethods. Passivemethodsareemployedbeforetrainingamodel, andinclude
methodssuchashyper-parameteroptimizationandmodelselection. Forinstance, Sunetal.(2017)
improvethebackpropagationalgorithmtospeedupthetrainingprocessandavoidoverfitting. Xu
etal.(2021) introducealearningalgorithmbasedonaprobabilisticmodelforavoidingoverfitting
tothenoiseinthetrainingdata. Activemethodspreventoverfittingbyeitherimposingnoisetothe
data or model through methods like adding dropout layers or other regularization schemes so that
modelscannotmemorizethepatternsinthedata. Forinstance, Dropout (Srivastavaetal.,2014), a
simpleandpopularoverfittingpreventionmethod, randomlydisablesapartofthe DLmodelduring
trainingtopreventoverfitting. Finally, semi-activemethodschangethemodelarchitectureduringthe
trainingprocess. Theyeitherworkbyaddinghiddennodesorpruningexistingnodes. Allaforemen-
tionedoverfittingpreventionmethodsaretypicallyintrusive (i.e., theyrequireeithermodificationof
themodelinternalsordatathatisfedtothemodel) andrequireconsiderableexpertisetoaccurately
execute. Forinstance, usingpassivemethodssuchashyperparameteroptimizationtoavoidoverfit-
tingrequiresexpertiseontherightoptimizationmethodtochooseandtherightparameterstotune,
whichisavastareaofresearchinitself (Bergstra&Bengio,2012;Falkneretal.,2018;Bischletal.,
2021). Similarly, activeandsemi-activemethodssuchasdropoutorpruningrequireeitheradding
layersordynamicallyeditingthemodelstructure. Inaddition, eventhoughthesemethodshavebeen
knowntoavoidoverfittingtheycannotguaranteethatthemodeldoesnotoverfitandtheytypically
employmethodslike Earlystopping (Morgan&Bourlard,1989;Prechelt,2012) tofurtherpredict
if overfit might occur. Early stopping is a widely used overfitting prevention method (which we
explainbelow) thatisnon-intrusiveanddoesnotrequireconsiderableexpertisetoexecute.
Early stopping. Early stopping stops training when there is no improvement in a fixed number
of epochs (indicated by the patience parameter) and returns the best epoch which has the lowest
validation loss. The idea behind this method is that the training will converge or become overfit
3


<!-- Page 4 -->

Underreviewasaconferencepaperat ICLR2023
Training the time series classifier
Continue training for
the model
Feed data (with
Trained time series
Training histories labels) to time series classifier
with labels classifier for training
Extract the latest
Use the whole
history with a rolling
observed history
window
Overfitting detection using the training history
Training P i e d r e fo n r t m ify in o f v e e r r e fi n tt c in e g to N Is overfitting? O u v s e i r n fi g tt i t n h g e d tr e a t i e n c in ti g on
history history
Y
Interpolate the Stop training and
Validation losses v th a e li d s a a t m io e n l l e o n s g se th s a to s n O o v n e -o rf v it e tin rf g itt i o n r g retu h r a n s t h th e e e l p o o w c e h s t that O pr v e e v r e fi n tt t i i n o g n when
the data in training validation loss training a model
Figure2: Ourmethodforoverfittingdetectionandprevention.
whenthevalidationlossstopsimproving. However, Prechelt (2012) studiedthreestoppingcriteria
ofearlystoppingandfoundthatusingaslowstoppingcriterionwillincreasethetrainingtimewhile
producingonlyasmallimprovementingeneralization. Inthispaper, weproposeusingoverfitting
detection methods during the training process to prevent overfitting. Hence, our method prevents
overfitting based on the byproduct (i.e., the training history) of the training process. We compare
our method to the early stopping method, as both methods are non-intrusive and do not require
significantexpertise. Wealsoincludetheresultsofanalternativeversionofearlystoppingbasedon
smoothedvalidationlossin Appendix E.
3 OUR METHOD
Figure 2 shows an overview of our proposed method. Our method uses a time series classifier to
detect and prevent overfitting. To the best of our knowledge, we are the first to use a time series
classification-based method to detect and prevent overfitting. First, we collect a simulated dataset
(moredetailsonhowwecollectthedatain Section4) thatcontainstraininghistories (i.e., training
andvalidationlosscurves, howeverweonlyusethevalidationlosscurvesinourmethod) withlabels
indicatingwhetheroverfittingoccursinordertotrainourtimeseriesclassifier. Second, wetraina
timeseriesclassifieronallthetraininghistoriesofthesimulateddataset. Weevaluatesixstate-of-
the-arttimeseriesclassifiers (see Appendix A) toidentifythebest-performingone. Finally, weuse
thetrainedtimeseriesclassifiertoperformbothoverfittingdetectionandpreventionasfollows.
Overfittingdetection. Todetectoverfittinginatrainedmodel, wefirstcollectitsvalidationlosses
over the training epochs. We feed this loss to our trained time series classifier to detect if there is
overfitting. However, wecannotdirectlyfeedthesevalidationlossestoourclassifierasthelength
ofthevalidationlossesmightnotbeofthesamelengthasthatofthedatausedtotrainthesetime
seriesclassifiers. Except KNN-DTWallthestudiedtimeseriesclassifiersexpectthelengthofthe
inputsusedfortrainingforwhichtheinferenceismadetobethesame. Therefore, wefirstlinearly
interpolatethevalidationlossesofthemodelforwhichweneedtodetectoverfittothesamelength
as the training histories used to train the studied time series classifiers. We feed the interpolated
validationlossestoourtrainedtimeseriesclassifierandperforminferencetodetermineifthemodel
is overfit. Figure 4 shows how the linear interpolation process works; if we only have validation
lossesover8 epochsandourtimeseriesclassifierwastrainedover80 epochvalidationlossvalues,
weinterpolatethe8 epochlossesto80 sothatwecanfeedittothetrainedtimeseriesclassifier.
Overfitting prevention. To prevent overfitting, we feed the training history (i.e., validation loss
curve) of a DL model that is being trained to our trained time series classifier during the training
process. Thehistoryisfedforinferenceintwodifferentways: (1) asarollingwindow: weextract
thelatesthistoryinafixedwindowsize (e.g., thelatest20 epochs), and (2) asthewholeobserved
history (from the first to the latest epochs). Our time series classifier detects if in the fed history
4


<!-- Page 5 -->

Underreviewasaconferencepaperat ICLR2023
Datasets
Download the Simulate
datasets for overfitting by Label training Simulated training
overfitting training neural histories dataset (419
simulation networks training histories)
80
Identify related Search for Collect existing 60 conferences papers that training history or Real-world test
and journals have samples of reproduce the dataset (40 40
overfitting training history training histories)
20
0 2 4 6 8
Experiment Evaluation
Choose Compare
Correlation- thresholds for overfitting
based methods
correlation- detection
with thresholds based methods methods
Compare
Trained time Train our series Early stopping overfitting
methods classifiers prevention methods
Figure3: Overviewoftheexperimentalsetup.
sso L
Original data
80
60
40
20
0 20 40 60 80
Epoch
sso L
Linear interpolation
Figure 4: An example of lin-
earlyinterpolation.
overfitting occurs. Similar to overfitting detection, we linearly interpolate the data before feeding
it into our model. If there is no overfit occurring, we continue the training and repeat the above
procedureuntilthemodelhasfinishedtraining. Fortherollingwindow, wemovethewindowbya
fixedstepsizeandmakeanotherprediction. Ifourmodeldetectsthepresenceofoverfitinthefed
history, wereturnthelowestvalidationlossintheobservedepochsasthebestepoch.
4 EXPERIMENTAL SETUP
Inthissection, weintroducethedatasetsfortrainingandevaluatingthestudiedoverfittingdetection
and prevention methods, the experiments of our study, and the evaluation metrics for the studied
methods. Figure3 showsanoverviewoftheexperimentalsetup.
4.1 DATASETS
Simulatedtrainingdataset. Weusesimulatedtraininghistorieswithlabelstodeterminethethresh-
old for the correlation-based methods (see Section 2.2) and to train our method as we explain in
Section3. Wecreateoursimulateddatasetbytrainingneuralnetworkswithdifferentmodelcom-
plexitiestogeneratethetraininghistoryofoverfittingandnon-overfittingsamplesasfollows:
Step1–Downloadthedatasetsforoverfittingsimulation. Wedownload12 datasetsofreal-world
problemsfromthe Proben1(Precheltetal.,1994) benchmarksetfortrainingneuralnetworks (the
informationaboutthestudieddatasetscanbefoundin Appendix A).Thesedatasetswereusedby
Prechelt (2012) tosimulatetraininghistoriesforstudyingearlystopping. Furthermore, allofthese
datasets (exceptthe“building”one) wereoriginallycollectedfromthe UCImachinelearningrepos-
itory (Blake et al., 1998), which has been widely used in deep learning research (Kuleshov et al.,
2018; Sajeev et al., 2019; Shi et al., 2021; Gadde et al., 2021). These datasets are pre-partitioned
into training, validation, and test set (respectively 50%, 25%, and 25% of the data). In addition,
Proben1 partitionseachdatasetthreetimesinordertogeneratethreedistinctpermutations. Hence,
intotalwecollect36 datasetsfrom Proben1.
Step 2 – Simulate overfitting by training neural networks. We train Neural Networks (NNs) with
variousarchitecturesonthecollected36 datasets. Wedosotovarythemodelcomplexitywhichin
turnincreasesthechanceofproducinganoverfittedmodel, inthesamewayas Prechelt (2012) didin
theirstudy. Theinput/outputlayercontainsthesamenumberofnodesasthenumberofinput/output
coefficients of the datasets (see Appendix A) and rectified linear units (Re LUs) are used for all
hidden layers. The structures of the NNs are as follows: (1) 6 one-hidden-layer NNs with hidden
nodesof2,4,8,16,24,32, and (2)6 two-hidden-layer NNswithhiddennodes (representedasfirst
5


<!-- Page 6 -->

Underreviewasaconferencepaperat ICLR2023
layer hidden nodes + second layer hidden nodes) of 2+2, 4+2, 4+4, 8+4, 8+8, 16+8. We use the
meansquareerror (MSE) asthelossfunctionforregressionproblems, andcrossentropyastheloss
function for classification problems. Additionally, we used SGD as the optimizer for all of these
problems. Toincreasethelikelihoodofoverfitting, wetrainthese12 neuralnetworkarchitectures
oneachdataset (ofthecollected36 datasets) for1,000 epochs, producing432 traininghistories.
Step3–Labeltraininghistories. Totrainourproposedmethodandthecorrelation-basedmethods,
weneedtomanuallylabelthetraininghistoryaseither“overfit”, “non-overfit”or“uncertain”. To
ensure the robustness of our manual labelling process, we follow the approach outlined by Ding
etal.(2020). Thefirstandsecondauthorsofthispaperindependentlylabelledthe432 datapoints
anddiscussedtheresults. Inthefirstdiscussionround, theauthorsreacheda95%agreement (410
datapoints), and10 datapointsthatwerelabelledas“uncertain”bybothauthorswereeliminated.
Inthesecondround, wediscussedthe22 disagreements. Followingthediscussion, weeliminated
3 data points (labelled “uncertain” by both authors) and agreed on the labels for the remaining 19
data points. The final data set consists of 44 overfit and 375 non-overfit training histories. As an
alternative (automated) approachforcollectingthelabelsweexperimentedwithaheuristicmethod,
however, thismethoddidnotperformwell (see Appendix H).
Real-world test dataset. To evaluate our method on real-world data, we surveyed papers from
conferencesandjournalstocollectsamplesofoverfitandnon-overfitmodels:
Step 1 – Identify related conferences and journals. We identify related conferences and journals
based on the Computing Research and Education Association of Australasia (CORE1) and China
Computer Federation (CCF2) rankingsystems. Underthe CCFArank, wehave7 conferencesand
4 journalsinthe“Artificial Intelligence”field. Underthe COREA*rank, wehave16 conferences
inthe“machinelearning”and“artificialintelligence”fieldsand12 journalsinthe“artificialintelli-
genceandimageprocessing”field. Weget17 conferencesand12 journalsaftermergingtheresults
becauseoftheoverlapbetweenthesetworankingsystems.
Step 2 – Search for papers that have samples of overfitting. We found 33 full papers (see Ap-
pendix C) withthe“overfit”keyword (whichincludese.g.,“overfitting”) inthetitlethatwerepub-
lished at the selected conferences and journals in the last 5 years. Five papers contain samples of
overfitting: P2-Chatterjee&Mishchenko (2020);P4-Chenetal.(2021)P13-Kimetal.(2021);
P17-Riceetal.(2020) and P23-Singlaetal.(2021). Appendix Cliststhepapersandthenumber
ofcollectedsamplesofoverfitting (someofthemalsoprovidesamplesofnon-overfitting).
Step 3 – Collect existing training history or reproduce the training history. Paper P17 shared the
traininghistory, makingitsreplicationstraightforward. Wereplicatedtheotherpapersthatprovide
overfittingsamplestocollectthetraininghistoriesofthesesamples. Weranthecodefromthepapers
that provide replication packages (P4, P13, and P23) to generate the training history, but we were
unable to replicate paper P13’s results. We followed the methodology to replicate the results and
traininghistoryforpaper P2, whichdidnotprovideareplicationpackage. Intotal, wecollected29
traininghistoriesofoverfitmodelsand11 ofnon-overfitmodels (see Appendix Cfordetails).
4.2 EXPERIMENTS
Overfittingdetection. Wetrainthetimeseriesclassifiersbasedonthesimulateddataset. Weper-
formed a grid search with 3-fold cross validation to tune the hyperparameters for each classifier
based on the simulated dataset. After selecting the hyperparameters, we trained each time series
classifier using the training histories and labels from the simulated dataset and saved the trained
classifier. Furthermore, we search thresholds for the correlation-based methods on the simulated
dataset. Weperformagridsearchforthethresholdsbetween-1 and1 basedonthecollectedsimu-
lateddatasettoselectthethresholdwhichhasthebest F-score.
Overfitting prevention. We reused the trained time series classifiers from the previous step to
performinferenceduringthetrainingprocesstopreventoverfitting. Weappliedourmethodtothe
trainedmodelsinevery10 epochs (i.e., thestepsize) with20,40,60,80, and100 epochsasdifferent
rollingwindowsizes. Wesetthepatiencevaluesforearlystoppingfrom5 to115 epochs.
1 https://www.core.edu.au/conference-portal
2 https://ccf.atom.im/
6


<!-- Page 7 -->

Underreviewasaconferencepaperat ICLR2023
Table1:Resultsoftheoverfittingdetectionmethodsonthereal-worlddataset (Prec:precision;Rec:
recall;F-s: F-score;Avg F-s: average F-score), andthetimecostoftrainingthestudiedmethodson
thesimulateddatasetandperforminginferenceonthereal-worlddataset (persample).
Non-overfitting Overfitting Avg Training Inference
Detectionmethod
Prec Rec F-s Prec Rec F-s F-s time (s) time (ms)
Spearman 0.71 0.91 0.80 0.96 0.86 0.91 0.86 2.461 0.908
Correlation
Pearson 0.78 0.64 0.70 0.87 0.93 0.90 0.80 0.222 0.025
based
Autocorr 0.80 0.73 0.76 0.90 0.93 0.92 0.84 0.233 0.026
KNN-DTW 0.79 1.00 0.88 1.00 0.90 0.95 0.91 0.001 180.512
Time HMM-GMM 0.30 0.27 0.28 0.73 0.76 0.75 0.52 99.751 17.750
series TSF 0.77 0.91 0.83 0.96 0.90 0.93 0.88 0.311 17.209
classifier TSBF 0.79 1.00 0.88 1.00 0.90 0.95 0.91 0.301 31.683
(ours) BOSSVS 0.46 0.91 0.61 0.94 0.59 0.72 0.67 1.877 19.342
SAX-VSM 0.83 0.91 0.87 0.96 0.93 0.95 0.91 0.912 17.474
4.3 EVALUATION
Evaluationmetricsforoverfittingdetection. Toevaluatetheclassificationperformanceofover-
fittingdetectionmethods, wecalculatedtheprecision, recall, and F-scoreforoverfittingandnon-
overfittingsamplesinthereal-worldtestdataset. Inaddition, wecalculatedtheaverage F-scorefor
directlycomparingtheclassificationperformanceofthestudiedmethods. Toevaluatethetimecost
oftrainingandusingthestudiedmethods, wereportthetrainingtime (inseconds) ofeachmethod
onthesimulateddatasetandtheinferencetime (inmilliseconds) onthereal-worlddataset.
Evaluation metrics for overfitting prevention. Ideally, an overfitting prevention method returns
theoptimalepochthathastheoptimalpredictiveperformanceforthemodelonthevalidationsetand
stopsthetrainingprocessasfastaspossible. Wedefinetheoptimalrateofanoverfittingprevention
method as the percentage of times the optimal epoch is identified. To evaluate the speed of the
method, wedefinethedelayastheepochdifferencebetweenthestoppedepochandthebestepoch,
e.g., the delay will be 10 epochs if a prevention method stops at the 123 th epoch while the 113 th
epochisthebestone. Forearlystopping, thedelaywillbethesameasthepatienceparameter.
5 RESULTS
Overfitting detection. Our overfitting detection approach using time series classifiers (except
HMM-GMMand BOSSVS) hasabetterclassificationperformancethanthecorrelation-basedmeth-
odsforoverfittingdetection. Table1 showsthatourapproachusing KNN-DTW, TSBF, and SAX-
VSM have the best F-score (0.91) on the real-world dataset, followed by TSF which outperforms
thebaselinemethodsaswell. Eventhoughourapproachusing BOSSVSachievesan F-scoreof1
onthesimulateddataset (see Appendix B), itperformspoorlyonthereal-worlddataset. Fromthe
results on both simulated and real-world datasets, the HMM-GMM time series classifier performs
poorlyfortheproblemofoverfittingdetection. Wealsonotethattheinvestigatedcorrelation-based
methods have a reasonably good performance. All of the studied correlation-based overfitting de-
tectionmethodshave F-scoresabove0.8. However, ourapproachoutperformsthebestperforming
correlation-basedoverfittingdetectionapproachbyatleast5%onthestudiedreal-worlddataset. We
alsoreporttheresultsofusingperturbationvalidationforoverfittingdetectionasanexampleofan
intrusiveoverfittingdetectionmethodin Appendix D.
Thestudiedtimeseriesclassifiersaremorecomputationallyintensivethancorrelation-basedmeth-
odsforinference, yettheyarestillusefulinpractice. Asshownin Table1, ourmethodrequiresmore
timeforperforminginferencethanthecorrelation-basedmethods. Forinstance, TSFhasthefastest
inference time among the classifiers but is around 20 times slower than the Spearman correlation-
basedmethodandaround700 timesslowerthantheothertwocorrelation-basedmethods. However,
the speed of our method is not prohibitive in practice since overfitting detection is only executed
onceafterthetrainingiscomplete. Itisalsousefultonotethatthetrainingtimesofthetimeseries
classifiersinourapproacharenotexcessive. Forinstance, thetrainingtimesof TSFand TSBFare
around300 millisecondsand KNN-DTW, ourbestperformingtimeseriesclassifiercanfinishtrain-
7


<!-- Page 8 -->

Underreviewasaconferencepaperat ICLR2023
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
5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95100105110115
Early stopping
KNN-DTW
HMM-GMM
TSF
TSBF
SAX-VSM
BOSSVS
Figure5:Theoptimalrateofourmethods (usingarollingwindow) andearlystoppingwithdifferent
patiencevalues.
ingin1 millisecond. However, KNN-DTWrequiresthelongesttimeforinferencewhichisaround
180 millisecondsforatraininghistory.
Overfitting prevention. Our method with KNN-DTW (both using rolling window and whole ob-
servedhistory) isoftenmoreaccuratethanearlystoppingforoverfittingprevention. Otherstudied
classifiersdonotperformaswellas KNN-DTWforoverfittingprevention. As Figure5 and Table2
show, bothourapproachbasedonrollingwindowandwholeobservedhistoryhashigheraccuracy
than early stopping at identifying the optimal epoch. In particular, our approach with KNN-DTW
based on rolling window is more accurate (see Figure 5) than early stopping when using up to 80
epochs as the patience parameter and window size. For example, our method with KNN-DTW
obtains 78% accuracy when setting the window size to 20 epochs, while early stopping has 48%
accuracywhensettingthepatienceparametertothatnumberofepochs. However, wefindthatearly
stoppingachievesnearlyperfectaccuracywhenthepatienceislargerthan80 epochs. Thereasonis
that90%ofthetraininghistoriesinthereal-worlddatasethavearound200 epochs, hence, alarge
patience value makes it easy for early stopping to choose the optimal epoch. In addition, Table 2
shows that our method with KNN-DTW based on the whole observed history also obtains higher
accuracythanearlystopping. Forinstance, the KNN-DTWclassifierhas95%accuracywithame-
diandelayof43.5 epochswhileearlystoppinghasonly83%usingthesamenumber (i.e., between
40 to45 epochspatiencein Figure5).
Table 2: The optimal rate and median delay Table3: Themediandelayofouroverfittingpreven-
ofouroverfittingpreventionmethodsthatare tion methods that are based on the rolling window
basedonthewholeobservedhistory. withdifferentwindowsizes.
Optimum Median Windowsize (epoch)
Classifier Classifier
rate delay
20 40 60 80 100
KNN-DTW 0.95 43.5
KNN-DTW 31.0 27.0 37.5 42.5 45.5
HMM-GMM 0.18 0.0
HMM-GMM 5.0 6.5 16.5 28.0 41.5
TSF 0.90 35.0
TSF 12.5 22.0 31.0 39.5 44.0
TSBF 0.83 31.0
TSBF 8.5 15.0 25.0 37.0 47.0
BOSSVS 0.65 21.0
BOSSVS 7.0 29.0 34.5 48.5 56.5
SAX-VSM 0.33 10.0
SAX-VSM 4.0 11.0 9.5 16.0 24.0
Our method using KNN-DTW and a rolling window can stop training a DL model earlier than
early stopping. As shown in Table 3, with the same number of epochs for the patience parameter
and window size, our method with KNN-DTW based on the rolling window (except window size
20) can save training time (i.e., there is a smaller delay between the stopped epoch and the best
epoch) over early stopping. For instance, when setting both the patience parameter and window
size to 40 epochs, KNN-DTW and early stopping have the same accuracy and KNN-DTW has a
8


<!-- Page 9 -->

Underreviewasaconferencepaperat ICLR2023
(a) Our method stops earlier than the early stopping (b)Ourmethodstopslaterthantheearlystoppingbut
butbothachievethesameoptimalepoch. achievestheoptimalepoch.
Figure 6: Examples of overfitting prevention based on KNN-DTW (set the window size as 40
epochs) andearlystopping (setthepatienceparameteras40 epochs).
mediandelayof27 epochswhileearlystoppinghasafixeddelayof40 epochs (whichisthesame
as the patience parameter). In comparison to the delay in early stopping, the delay between the
stopped epoch and the best epoch is at least 32% shorter with our method using KNN-DTW (see
Appendix Fforthesignificancetestingresults).Figure6 ashowsanexampleinwhichearlystopping
andourmethodbothidentifytheoptimalepoch, butourmethodstops21 epochsearlierthanearly
stopping (whichstopswitha40 epochsdelay). Inaddition, ourmethoddoesnotsacrificeaccuracy
forashorterdelay (asshownin Figure5). Figure6 bshowsanexampleinwhichourmethodstops
later than early stopping but identifies the optimal epoch when using the same number of epochs
for the patience parameter and window size. Furthermore, our overfitting detection approach can
beusedwithzero-onevalidationloss;theresultsofourapproachandearlystoppingwithzero-one
validationlosscanbefoundin Appendix G.
Amongourtwoapproachesforoverfittingpreventionwerecommendtheusageof KNN-DTWwith
rolling window. Though using the whole observed history may be more accurate at predicting the
optimalepochthanusingarollingwindowforourapproach, wenotethatwecanpredicttheoptimal
epoch much earlier with the rolling window approach for a very small trade-off in accuracy. As
shown in the Table 3 and Figure 5, our method with KNN-DTW achieves 83% accuracy with a
mediandelayof27 epochsand90%accuracywithamediandelayof37.5 epochsusingthewindow
sizeas40 and60 epochsrespectively. However, themediandelayof KNN-DTWwhenusingwhole
observedhistoryis43.5 whileusingtherollingwindowwithawindowsizeof80 ormoreepochs
canachieveahigheraccuracy (98%vs. 95%accuracy) withashorterdelay (42.5 vs. 43.5 epochs).
Insummary, wesuggestusingtherollingwindowapproachsinceitisstopsearlierwitharelatively
smallaccuracydropusingasmallwindow (e.g.,40 epochs) andperformsbetterthantheobserved
wholehistoryapproachwhenusingalargewindowsize (e.g.,80 epochs).
6 CONCLUSION AND FUTURE WORK
Weproposeanon-intrusiveoverfittingdetectionandpreventionmethodthatisbasedontimeseries
classifiers trained on the training history of DL models. Our method (when using the KNN-DTW
time series classifier) has (1) better classification performance than correlation-based methods for
overfittingdetection, and (2) greateraccuracythanearlystoppingforoverfittingpreventionwitha
shorterdelay. Wedosousingareal-worlddatasetoflabelledtraininghistoriescollectedfromthe
paperspublishedattop AIvenuesinthelast5 years. Furthermore, thetrainedtimeseriesclassifiers
are included in our replication package for use by other researchers. One of the downsides of our
approachisthatourbestperformingtimeseriesclassifiertakeslongertoperformtheinferencere-
quiredtodetectandpreventoverfitthanthestudiedbaselines. Weencouragefutureworktooptimize
timeseriesclassifierstoenableoverfittingdetectionandpreventioninreal-timewithsmallerdelays.
9


<!-- Page 10 -->

Underreviewasaconferencepaperat ICLR2023
REPRODUCIBILITY STATEMENT
Thereplicationpackagecanbefoundat:https://github.com/anonymous-p/overfit_
detect. Thispackageincludesthecodefortrainingandusingthestudiedmethods, notebooksfor
analysingtheresults, thesimulatedandreal-worlddatasets, andthetrainedmodels. Furthermore, we
arecurrentlydevelopingatooltousethestudiedmethodsandwillmakeitpubliconcecompleted.
Experimental environment. We use Python 3.8 with Tensor Flow 2.9.0 and run experiments on
Ubuntu 20.04 with Linux kernel 5.15.0. The hardware specifications are as follows: (1) NVIDIA
RTX3090 GPUwith24 GBmemory (theversionsof CUDAandcu DNNare10.1.243 and7.6.5),
(2)3.50 GHz Intel (R)Core (TM) i9-11900 KCPU, and (3)64 GBRAM.
REFERENCES
Basavaraj SAnamiand Venkatesh ABhandage. Acomparativestudyofsuitabilityofcertainfea-
tures in classification of bharatanatyam mudra images using artificial neural network. Neural
Processing Letters,50(1):741–769,2019.
Daniel Bashir, George D. Montan˜ez, Sonia Sehra, Pedro Sandoval Segura, and Julius Lauw. An
Information-Theoretic Perspective on Overfitting and Underfitting. In Marcus Gallagher, Nour
Moustafa, and Erandi Lakshika (eds.), AI2020:Advancesin Artificial Intelligence, Lecture Notes
in Computer Science, pp.347–358, Cham,2020.Springer International Publishing. ISBN978-3-
030-64984-5. doi: 10.1007/978-3-030-64984-5 27.
Mustafa Gokce Baydogan, George Runger, and Eugene Tuv. Abag-of-featuresframeworktoclas-
sifytimeseries. IEEETransactionson Pattern Analysisand Machine Intelligence,35(11):2796–
2802,2013. doi: 10.1109/TPAMI.2013.72.
Mohammad Mahdi Bejaniand Mehdi Ghatee. Asystematicreviewonoverfittingcontrolinshallow
anddeepneuralnetworks. Artificial Intelligence Review,54(8):6391–6438,2021.
James Bergstraand Yoshua Bengio. Randomsearchforhyper-parameteroptimization. Journalof
machinelearningresearch,13(2),2012.
Ge´rard Biauand Erwan Scornet. Arandomforestguidedtour. Test,25(2):197–227,2016.
Bernd Bischl, Martin Binder, Michel Lang, Tobias Pielok, Jakob Richter, Stefan Coors, Janek
Thomas, Theresa Ullmann, Marc Becker, Anne-Laure Boulesteix, et al. Hyperparameter
optimization: Foundations, algorithms, best practices and open challenges. ar Xiv preprint
ar Xiv:2107.05847,2021.
C.Blake, E.Keogh, and Christopher Merz. UCIrepositoryofmachinelearningdatabases. 011998.
Jo¨rg Bornschein, Francesco Visin, and Simon Osindero. Small data, big decisions: Model selec-
tioninthesmall-dataregime. In Proceedingsofthe37 th International Conferenceon Machine
Learning, ICML’20.JMLR.org,2020.
Pavel Brazdil, Jan Nvan Rijn, Carlos Soares, and Joaquin Vanschoren. Metalearning: Applications
to Automated Machine Learningand Data Mining. Springer Nature,2022.
Peter JBrockwelland Richard ADavis. Introductiontotimeseriesandforecasting. Springer,2002.
Rich Caruana, Steve Lawrence, and CGiles. Overfittinginneuralnets:Backpropagation, conjugate
gradient, andearlystopping. Advancesinneuralinformationprocessingsystems,13,2000.
Gavin CCawleyand Nicola LCTalbot. Preventingover-fittingduringmodelselectionviabayesian
regularisationofthehyper-parameters. Journalof Machine Learning Research,8(4),2007.
Satrajit Chatterjee and Alan Mishchenko. Circuit-based intrinsic methods to detect overfitting.
In Hal Daume´ III and Aarti Singh (eds.), Proceedings of the 37 th International Conference on
Machine Learning, volume119 of Proceedingsof Machine Learning Research, pp.1459–1468.
PMLR,13–18 Jul2020.
10


<!-- Page 11 -->

Underreviewasaconferencepaperat ICLR2023
Tianlong Chen, Zhenyu Zhang, Sijia Liu, Shiyu Chang, and Zhangyang Wang. Robustoverfitting
may be mitigated by properly learned smoothening. In International Conference on Learning
Representations,2021.
Houtao Deng, George Runger, Eugene Tuv, and Martyanov Vladimir. A time series forest for
classificationandfeatureextraction. Information Sciences,239:142–153,2013.ISSN0020-0255.
doi: https://doi.org/10.1016/j.ins.2013.02.030.
Hui Ding, Goce Trajcevski, Peter Scheuermann, Xiaoyue Wang, and Eamonn Keogh. Queryingand
mining of time series data: Experimental comparison of representations and distance measures.
Proc. VLDB Endow., 1(2):1542–1552, aug 2008. ISSN 2150-8097. doi: 10.14778/1454159.
1454226.
Zishuo Ding, Jinfu Chen, and Weiyi Shang. Towards the use of the readily available tests from
the release pipeline as performance tests: Are we there yet? In Proceedings of the ACM/IEEE
42 nd International Conferenceon Software Engineering, ICSE’20, pp.1435—-1446, New York,
NY, USA, 2020. Association for Computing Machinery. ISBN 9781450371216. doi: 10.1145/
3377811.3380351.
Dumitru Erhan, Yoshua Bengio, Aaron Courville, Pierre-Antoine Manzagol, Pascal Vincent, and
Samy Bengio. Whydoesunsupervisedpre-traininghelpdeeplearning? J.Mach. Learn. Res.,11:
625—-660, mar2010. ISSN1532-4435.
Stefan Falkner, Aaron Klein, and Frank Hutter. Bohb: Robust and efficient hyperparameter opti-
mization at scale. In International Conference on Machine Learning, pp. 1437–1446. PMLR,
2018.
Zhouyu Fu, Guojun Lu, Kai Ming Ting, and Dengsheng Zhang. Musicclassificationviathebag-of-
featuresapproach. Pattern Recognition Letters,32(14):1768–1777,2011. ISSN0167-8655. doi:
https://doi.org/10.1016/j.patrec.2011.06.026.
Sridevi Gadde, A.Lakshmanarao, and S.Satyanarayana. Smsspamdetectionusingmachinelearn-
ing and deep learning techniques. In 2021 7 th International Conference on Advanced Com-
puting and Communication Systems (ICACCS), volume 1, pp. 358–362, 2021. doi: 10.1109/
ICACCS51430.2021.9441783.
J.-L.Gauvainand Chin-Hui Lee. Maximumaposterioriestimationformultivariategaussianmixture
observationsofmarkovchains. IEEETransactionson Speechand Audio Processing, 2(2):291–
298,1994. doi: 10.1109/89.279278.
David JHand. Principlesofdatamining. Drugsafety,30(7):621–622,2007.
Jan Haukeand Tomasz Kossowski. Comparisonofvaluesof Pearson’sand Spearman’scorrelation
coefficientsonthesamesetsofdata. Quaestiones Geographicae, 30(2):87–93, 2011. doi: doi:
10.2478/v10117-011-0021-1.
Douglas M Hawkins. The problem of overfitting. Journal of chemical information and computer
sciences,44(1):1–12,2004.
Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by
reducinginternalcovariateshift. In Internationalconferenceonmachinelearning, pp.448–456.
PMLR,2015.
Shihao Ji, B. Krishnapuram, and L. Carin. Variational bayes for continuous hidden markov mod-
els and its application to active learning. IEEE Transactions on Pattern Analysis and Machine
Intelligence,28(4):522–532,2006. doi: 10.1109/TPAMI.2006.85.
Hoki Kim, Woojin Lee, and Jaewook Lee. Understanding catastrophic overfitting in single-step
adversarialtraining. Proceedingsofthe AAAIConferenceon Artificial Intelligence,35(9):8119–
8127, May2021.
11


<!-- Page 12 -->

Underreviewasaconferencepaperat ICLR2023
Gabriel Kronberger, Michael Kommenda, and Michael Affenzeller. Overfittingdetectionandadap-
tive covariant parsimony pressure for symbolic regression. In Proceedings of the 13 th Annual
Conference Companion on Genetic and Evolutionary Computation, GECCO ’11, pp. 631–638,
New York, NY, USA,2011.Associationfor Computing Machinery. ISBN9781450306904. doi:
10.1145/2001858.2002060.
Jan Kukacˇka, Vladimir Golkov, and Daniel Cremers. Regularizationfordeeplearning:Ataxonomy.
ar Xivpreprintar Xiv:1710.10686,2017.
Volodymyr Kuleshov, Nathan Fenner, and Stefano Ermon. Accurateuncertaintiesfordeeplearning
using calibrated regression. In Jennifer Dy and Andreas Krause (eds.), Proceedings of the 35 th
International Conferenceon Machine Learning, volume80 of Proceedingsof Machine Learning
Research, pp.2796–2804.PMLR,10–15 Jul2018.
Jessica Lin, Eamonn Keogh, Li Wei, and Stefano Lonardi. Experiencing sax: a novel symbolic
representationoftimeseries. Data Miningandknowledgediscovery,15(2):107–144,2007.
Jeffrey D. Long, Du Feng, and Norman Cliff. Ordinal Analysis of Behavioral Data. In Irving B.
Weiner (ed.), Handbookof Psychology, chapter25, pp.635–661.John Wiley&Sons, Inc., Hobo-
ken, NJ, USA, April2003. ISBN978-0-471-26438-5. doi: 10.1002/0471264385.wei0225.
H.B.Mannand D.R.Whitney. Onatestofwhetheroneoftworandomvariablesisstochastically
largerthantheother. Annalsof Mathematical Statistics,18:50–60,1947.
Felix Mohrand Jan Nvan Rijn. Towardsmodelselectionusinglearningcurvecross-validation. In
8 th ICMLWorkshoponautomatedmachinelearning (Auto ML),2021.
Felix Mohrand Jan N.van Rijn. Learningcurvesfordecisionmakinginsupervisedmachinelearning
-Asurvey. Co RR, abs/2201.12150,2022. URLhttps://arxiv.org/abs/2201.12150.
Kumar Molugaram and G. Shanker Rao. Statistical Techniques for Transportation Engineer-
ing. Butterworth-Heinemann, January 2017. ISBN 978-0-12-811555-8. doi: 10.1016/
B978-0-12-811555-8.00012-X.
Nelson Morganand Herve´ Bourlard. Generalizationandparameterestimationinfeedforwardnets:
Someexperiments. Advancesinneuralinformationprocessingsystems,2,1989.
Andrew Y.Ng. Preventing”overfitting”ofcross-validationdata. In Proceedingsofthe Fourteenth
International Conferenceon Machine Learning, ICML’97, pp.245—-253, San Francisco, CA,
USA,1997.Morgan Kaufmann Publishers Inc. ISBN1558604863.
Steven J. Nowlan and Geoffrey E. Hinton. Simplifying neural networks by soft weight-sharing.
Neural Computation,4(4):473–493,1992. doi: 10.1162/neco.1992.4.4.473.
Danni Peng, Sinno Jialin Pan, Jie Zhang, and Anxiang Zeng. Learning an adaptive meta model-
generator for incrementally updating recommender systems. In Proceedings of the 15 th ACM
Conferenceon Recommender Systems, Rec Sys’21, pp.411—-421, New York, NY, USA,2021.
Associationfor Computing Machinery. ISBN9781450384582. doi: 10.1145/3460231.3474239.
Tao Peng, Lu Liu, and Wanli Zuo. PUtextclassificationenhancedbytermfrequency–inversedocu-
mentfrequency-improvedweighting. Concurrencyand Computation: Practiceand Experience,
26(3):728–741,2014. doi: https://doi.org/10.1002/cpe.3040.
Lutz Prechelt. Early Stopping—But When? In Gre´goire Montavon, Genevie`ve B.Orr, and Klaus-
Robert Mu¨ller (eds.), Neural Networks: Tricks of the Trade: Second Edition, Lecture Notes in
Computer Science, pp. 53–67. Springer, Berlin, Heidelberg, 2012. ISBN 978-3-642-35289-8.
doi: 10.1007/978-3-642-35289-8 5.
Lutz Precheltetal. Proben1:Asetofneuralnetworkbenchmarkproblemsandbenchmarkingrules.
1994.
Leslie Rice, Eric Wong, and Zico Kolter. Overfitting in adversarially robust deep learning. In
Hal Daume´ III and Aarti Singh (eds.), Proceedings of the 37 th International Conference on
Machine Learning, volume119 of Proceedingsof Machine Learning Research, pp.8093–8104.
PMLR,13–18 Jul2020.
12


<!-- Page 13 -->

Underreviewasaconferencepaperat ICLR2023
Jeanine Romano, Jeffrey DKromrey, Jesse Coraggio, Jeff Skowronek, and Linda Devine. Exploring
methods for evaluating group differences on the NSSE and other surveys: Are the t-test and
Cohen’sd indices the most appropriate choices. In annual meeting of the Southern Association
for Institutional Research, pp.1–51.Citeseer,2006.
Shelda Sajeev, Anthony Maeder, Stephanie Champion, Alline Beleigoli, Cheng Ton, Xianglong
Kong, and Minglei Shu. Deep learning to improve heart disease risk prediction. In Machine
Learning and Medical Engineering for Cardiovascular Health and Intravascular Imaging and
Computer Assisted Stenting, pp.96–103.Springer,2019.
Gerard Salton, Anita Wong, and Chung-Shu Yang. A vector space model for automatic indexing.
Communicationsofthe ACM,18(11):613–620,1975.
Patrick Scha¨fer. Scalabletimeseriesclassification. Data Miningand Knowledge Discovery,30(5):
1273–1298,2016.
Patrick Scha¨ferand Mikael Ho¨gqvist. SFA:Asymbolicfourierapproximationandindexforsimi-
laritysearchinhighdimensionaldatasets. In Proceedingsofthe15 th International Conference
on Extending Database Technology, EDBT’12, pp.516—-527, New York, NY, USA,2012.As-
sociationfor Computing Machinery. ISBN9781450307901. doi: 10.1145/2247596.2247656.
Qiushi Shi, Rakesh Katuwal, P.N.Suganthan, and M.Tanveer. Randomvectorfunctionallinkneural
network based ensemble deep learning. Pattern Recognition, 117:107978, 2021. ISSN 0031-
3203. doi: https://doi.org/10.1016/j.patcog.2021.107978.
Connor Shortenand Taghi MKhoshgoftaar. Asurveyonimagedataaugmentationfordeeplearning.
Journalofbigdata,6(1):1–48,2019.
Robert H.Shumwayand David S.Stoffer. Time Series Analysisand Its Applications: With RExam-
ples. Springer Textsin Statistics. Springer International Publishing, Cham, 2017. ISBN978-3-
319-52451-1978-3-319-52452-8. doi: 10.1007/978-3-319-52452-8.
Vasu Singla, Sahil Singla, Soheil Feizi, and David Jacobs. Lowcurvatureactivationsreduceoverfit-
tinginadversarialtraining. In Proceedingsofthe IEEE/CVFInternational Conferenceon Com-
puter Vision (ICCV), pp.16423–16433, October2021.
Xingyou Song, Yiding Jiang, Stephen Tu, Yilun Du, and Behnam Neyshabur. Observationalover-
fittinginreinforcementlearning. In8 th International Conferenceon Learning Representations,
ICLR2020, Addis Ababa, Ethiopia, April26-30,2020.Open Review.net,2020.
Charles Spearman. Theproofandmeasurementofassociationbetweentwothings. The American
journalofpsychology,100(3/4):441–471,1987.
Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov.
Dropout: a simple way to prevent neural networks from overfitting. The journal of machine
learningresearch,15(1):1929–1958,2014.
Benjamin Strang, Peter van der Putten, Jan N. van Rijn, and Frank Hutter. Don’t rule out sim-
plemodelsprematurely: Alargescalebenchmarkcomparinglinearandnon-linearclassifiersin
openml. In Wouter Duivesteijn, Arno Siebes, and Antti Ukkonen (eds.), Advancesin Intelligent
Data Analysis XVII, pp.303–315, Cham,2018.Springer International Publishing. ISBN978-3-
030-01768-2.
Xu Sun, Xuancheng Ren, Shuming Ma, and Houfeng Wang. me Prop: Sparsifiedbackpropagation
foraccelerateddeeplearningwithreducedoverfitting. In International Conferenceon Machine
Learning, pp.3299–3308.PMLR,2017.
Jan N.van Rijn, Salisu Mamman Abdulrahman, Pavel Brazdil, and Joaquin Vanschoren. Fastalgo-
rithmselectionusinglearningcurves. In Elisa Fromont, Tijl De Bie, and Matthijsvan Leeuwen
(eds.), Advances in Intelligent Data Analysis XIV, pp. 298–309, Cham, 2015. Springer Interna-
tional Publishing. ISBN978-3-319-24465-5.
Onur Varol, Emilio Ferrara, Filippo Menczer, and Alessandro Flammini. Early detection of pro-
motedcampaignsonsocialmedia. EPJdatascience,6:1–19,2017.
13


<!-- Page 14 -->

Underreviewasaconferencepaperat ICLR2023
Zhongju Wang, Long Wang, Chao Huang, Zijun Zhang, and Xiong Luo. Soil-moisture-sensor-based
automatedsoilwatercontentcycleclassificationwithahybridsymbolicaggregateapproximation
algorithm. IEEEInternetof Things Journal,8(18):14003–14012,2021.
Roman Werpachowski, Andra´s Gyo¨rgy, and Csaba Szepesva´ri. Detectingoverfittingviaadversarial
examples. Advancesin Neural Information Processing Systems,32,2019.
Xiaopeng Xi, Eamonn Keogh, Christian Shelton, Li Wei, and Chotirat Ann Ratanamahatana. Fast
time series classification using numerosity reduction. In Proceedings of the 23 rd international
conferenceon Machinelearning, pp.1033–1040,2006.
Le Xu, Lei Cheng, Ngai Wong, and Yik-Chung Wu. Overfitting avoidance in tensor train factor-
izationandcompletion: Prioranalysisandinference. In2021 IEEEInternational Conferenceon
Data Mining (ICDM), pp.1439–1444,2021. doi: 10.1109/ICDM51629.2021.00185.
Jie M. Zhang, Earl T. Barr, Benjamin Guedj, Mark Harman, and John Shawe-Taylor. Perturbed
modelvalidation: Anewframeworktovalidatemodelrelevance. Co RR, abs/1905.10201,2019.
URLhttp://arxiv.org/abs/1905.10201.
Minghang Zhao, Baoping Tang, Lei Deng, and Michael Pecht. Multiple wavelet regularized deep
residualnetworksforfaultdiagnosis. Measurement, 152:107331, 2020. ISSN0263-2241. doi:
https://doi.org/10.1016/j.measurement.2019.107331.
14


<!-- Page 15 -->

Underreviewasaconferencepaperat ICLR2023
A STUDIED DATASETS AND TIME SERIES CLASSIFIERS
Table4 showstheinformationaboutthestudieddatasets. Thereare3 datasetsforregressiontasks
and9 datasetsforclassificationtasks.
Table4: Informationaboutstudieddatasets.
Dataset Type #In #Out #Examples
building regression 14 3 4,208
cancer classification 9 2 699
card classification 51 2 690
diabetes classification 8 2 768
flare regression 24 3 1,066
gene classification 120 3 3,175
glass classification 9 6 214
heart classification 35 2 920
hearta regression 35 1 920
horse classification 58 3 364
soybean classification 82 19 683
thyroid classification 21 3 7,200
Table 5 introduces the studied time series classifiers. Since there has been no prior systematic
researchontimeseriesclassifiersfortraininghistoriesasareference, wechoosetheclassifiersthat
have been reported as baselines or state-of-the-art in studies (Xi et al., 2006; Varol et al., 2017;
Anami&Bhandage,2019;Wangetal.,2021)
Table5: Studiedtimeseriesclassifiers.
Classifier Description
KNN-DTW∗ Uses K-Nearest Neighbors (Hand,2007) and Dynamic Time Warping (Dingetal.,2008)
asthedistancemetric
HMM-GMM Uses Hidden Markov Modelformodelingtimeseriesdataand Gaussian Mixture Modelas
theemissionsprobabilitydensity (Gauvain&Lee,1994;Jietal.,2006)
TSF Usesarandomforest (Biau&Scornet,2016) fortimeseriesdatausinganensembleoftime
seriestrees (Dengetal.,2013)
TSBF Time Series Bag-of-Features (Baydoganetal.,2013) extractsfeaturesbasedonthebag-of-
featuresapproach (Fuetal.,2011) tocreatearandomforest
SAX-VSM Symbolic Aggregateappro Ximationtransformsthedataintosymbolicrepresentations (Lin
etal.,2007) and Vector Space Model (Saltonetal.,1975;Pengetal.,2014) transformsthem
intovectorstocalculatesimilarityforclassification
BOSSVS Bag-of-SFA Symbols in Vector Space (Scha¨fer, 2016) is similar to SAX-VSM but use
SFA(Scha¨fer&Ho¨gqvist,2012) totransformthedatainsteadof SAX
∗ Wedonotinterpolatethevalidationlossfor KNN-DTWsinceitdoesnotrequireaconstantlength
inputsignal.
B RESULTS OF TRAINING OVERFITTING DETECTION METHODS
Asmentionedin Section4.2, wetunethehyperparametersofthetimeseriesclassifiersbyperform-
ingagridsearchwith3-foldcrossvalidation. Table6 showsthatalltheclassifiersachieve F-scores
of more than 0.95 in the 3-fold cross validation except HMM-GMM which obtains only an aver-
age F-score of only 0.6. Table 7 shows the results of training correlation-based methods and our
method on the simulated dataset. Generally, the time series classifiers achieve better performance
onthesimulateddatasetthantheothermethods (exceptthe HMM-GMMclassifier) andthreeofthe
classifierscanevencorrectlyidentifyallthedatainthesimulateddataset.
Wenoticethat KNN-DTWperformswellonboththesimulatedandreal-worlddatasets. Onepos-
siblereasonfortheperformanceof KNN-DTWmightbethat DTWisgoodatmeasuringsimilarity
15


<!-- Page 16 -->

Underreviewasaconferencepaperat ICLR2023
Table6: The F-scoresofthetimeseriesclassifiersin3-foldcrossvalidation (CV) onthesimulated
dataset.
Classifier CV1 CV2 CV3 Avg Variance
KNN-DTW 0.98 0.98 0.96 0.97 0.00
HMM-GMM 0.41 0.38 1.00 0.59 0.08
TSF 0.98 0.98 1.00 0.99 0.00
TSBF 0.98 0.98 1.00 0.99 0.00
BOSSVS 1.00 1.00 1.00 1.00 0.00
SAX-VSM 0.96 0.96 0.96 0.96 0.00
Table 7: Results of overfitting detection methods on the simulated dataset. (Prec: precision; Rec:
recall;F-s: F-score;Avg F-s: macroaverage F-score)
Non-overfitting Overfitting Avg
Detectionmethod
Prec Rec F-s Prec Rec F-s F-s
Spearman 0.99 1.00 0.99 0.95 0.89 0.92 0.95
Correlation
Pearson 0.99 0.97 0.98 0.79 0.93 0.85 0.92
based
Autocorr 0.97 0.94 0.95 0.59 0.73 0.65 0.80
KNN-DTW 0.99 1.00 1.00 0.98 0.93 0.95 0.97
Time HMM-GMM 0.95 0.55 0.70 0.16 0.75 0.27 0.48
series TSF 1.00 1.00 1.00 1.00 1.00 1.00 1.00
classifier TSBF 1.00 1.00 1.00 1.00 1.00 1.00 1.00
(ours) BOSSVS 1.00 1.00 1.00 1.00 1.00 1.00 1.00
SAX-VSM 0.99 1.00 0.99 0.98 0.91 0.94 0.97
acrosscurves, whichaids KNNindistinguishingbetweenoverfitandnon-overfitsamples. Incon-
trast, HMM-GMMperformspoorlyonboththesimulatedtrainingandreal-worldtestdatasets. One
possible explanation is that the extracted state models (via HMM) of the curves do not follow a
Gaussianprobabilitydistribution. BOSSVScorrectlyidentifiesallthedatainthesimulateddataset
butperformspoorlyonthereal-worlddataset (asshownin Table1), whichmaybeduetooverfitting
thesimulateddataset.
C SURVEYED PAPERS
Table8 liststhe33 paperswesurveyedfromtop AIconferencesandjournals. Wefound5 papers
outofthe33 papersprovidesamplesofoverfitandfitmodelsandcollected40 samplesfromthese
papers (asshownin Table9).
Table8: Informationaboutthesurveyedpapers.
#P Authors Title Venue Year
P1 Belkinetal. Overfittingorperfectfitting? riskboundsforclassifica- Neur IPS 2018
tionandregressionrulesthatinterpolate.
P2 Chatterjee Circuit-basedintrinsicmethodstodetectoverfitting. ICML 2020
&
Mishchenko
P3 Chatterji & Foolishcrowdssupportbenignoverfitting. JMLR 2022
Long
P4 Chenetal. Robustoverfittingmaybemitigatedbyproperlylearned ICLR 2021
smoothening.
P5 d’Ascoli et Triple descent and the two kinds of overfitting: where Neur IPS 2020
al. &whydotheyappear?
Continuedonnextpage
16


<!-- Page 17 -->

Underreviewasaconferencepaperat ICLR2023
Table8–continuedfrompreviouspage
#P Authors Title Venue Year
P6 Feldman et Theadvantagesofmultipleclassesforreducingoverfit- ICML 2019
al. tingfromtestsetreuse.
P7 Feldman et Open problem: how fast can a multiclass test set be COLT 2019
al. overfit?
P8 Freietal. Benign overfitting without linearity: neural network COLT 2022
classifiers trained by gradient descent for noisy linear
data.
P9 Heetal. Sparse double descent: where network pruning aggra- ICML 2022
vatesoverfitting.
P10 Huangetal. Sparseprogressivedistillation:resolvingoverfittingun- ACL 2022
derpretrain-and-finetuneparadigm.
P11 Juetal. Overfitting can be harmless for basis pursuit, but only Neur IPS 2020
toadegree.
P12 Juetal. On the generalization power of overfitted two-layer ICML 2021
neuraltangentkernelmodels.
P13 Kimetal. Understanding catastrophic overfitting in single-step AAAI 2021
adversarialtraining.
P14 Koehler et Uniformconvergenceofinterpolators: Gaussianwidth, Neur IPS 2021
al. normboundsandbenignoverfitting.
P15 Liuetal. Overfittingthedata: compactneuralvideodeliveryvia ICCV 2021
content-awarefeaturemodulation.
P16 Mohammed Over-fitting in model selection with Gaussian process ICML 2017
&Cawley regression.
P17 Riceetal. Overfittinginadversariallyrobustdeeplearning. ICML 2020
P18 Roelofs et Ameta-analysisofoverfittinginmachinelearning. Neur IPS 2019
al.
P19 Rozendaal Overfitting for fun and profit: instance-adaptive data ICLR 2021
etal.compression.
P20 Russo & Howmuchdoesyourdataexplorationoverfit? control- IEEE 2020
Zou lingbiasviainformationusage. Trans.
Inf.
Theory
P21 Sanyaletal. Howbenignisbenignoverfitting? ICLR 2021
P22 Shamir Theimplicitbiasofbenignoverfitting. COLT 2022
P23 Singlaetal. Low curvature activations reduce overfitting in adver- ICCV 2021
sarialtraining.
P24 Songetal. Observationaloverfittinginreinforcementlearning. ICLR 2020
P25 Steck Autoencodersthatdon’toverfittowardstheidentity. Neur IPS 2020
P26 Sunetal. me Prop: sparsified back propagation for accelerated ICML 2017
deeplearningwithreducedoverfitting.
P27 Telgarsky Stochastic linear optimization never overfits with COLT 2022
quadratically-boundedlossesongeneraldata.
P28 Wangetal. Benignoverfittinginmulticlassclassification: allroads Neur IPS 2021
leadtointerpolation.
P29 Webster et Detecting overfitting of deep generative networks via CVPR 2019
al. latentrecovery.
P30 Werpachowski Detectingoverfittingviaadversarialexamples. Neur IPS 2019
etal.
P31 Xuetal. Overfitting avoidance in tensor train factorization and ICDM 2021
completion: prioranalysisandinference.
P32 Zhang & Labelconsistencyinoverfittedgeneralizedk-means. Neur IPS 2021
Amini
P33 Zhangetal. Why overfitting isn’t always bad: retrofitting cross- ACL 2020
lingualwordembeddingstodictionaries.
17


<!-- Page 18 -->

Underreviewasaconferencepaperat ICLR2023
Table9: Informationaboutcollectedsamplesfromsurveyedpapers.
Paper Labelsforthetraininghistoryinthemanuscript #Overfit #Non-overfit
P2 “[...] the validation accuracy of nn-random is 9.73% (i.e., 2 0
closetochance) confirmingthatitishorriblyoverfit”
P4 “We first observe that the robust overfitting prevails in all 3 3
Baselinecases”
“Ourmethodseffectivelymitigatestherobustoverfitting”
P13 “Figure4[...] thatis, catastrophicoverfittingoccurs.” N/A∗ N/A∗
“Figure6 showsthattheproposedmethodalsosuccessfully
preventscatastrophicoverfitting[...]”
P17 “Figure24[...]Weseeclearrobustoverfittingforthesmaller 20 4
two options in λ, and find no overfitting but highly regular-
izedmodelsforthelargertwooptions[...]”
P23 “Theseresultsthereforevalidateourclaimthatlowcurvature 4 4
activationsreducerobustoverfitting”
∗:Cannotreproducethesameresultsasthepaper.
Table 10: Results of the Perturbation Validation method for overfitting detection methods on the
simulateddataset.
Non-overfitting Overfitting Average
Precision Recall F-score Precision Recall F-score F-score
0.89 0.85 0.87 0.07 0.09 0.08 0.47
D PERTURBATION VALIDATION FOR OVERFITTING DETECTION
Inject noise into the Retrain the model
Training data of Model training data at levels with the training data Training histories
a dataset of 0.1, 0.2, and 0.3 at each noise level (at all noise
levels)
Calculate PV Overfitting or
PV > threshold?
measurement non-overfitting PV measurement
Figure7: Ourmethodforoverfittingdetectionandprevention.
Zhang et al. (2019) suggest the perturbation validation (PV) assessment to determine whether a
modelfitsthetrainingdataproperly (i.e., ensurethatitisneitheroverfittingnorunderfit). Asshown
in Figure7, Weinjectthreelevelsofnoise (0.1,0.2, and0.3) astheoriginalpaper (Zhangetal.,2019)
intothelabelsinthetrainingsetandretrainthemodel. Werepeatthetrainingforeachnoiseleveland
collect the training history to compute the PV measurement. The proposed PV measurement will
showhowmuchtheaccuracydecreasesinresponsetotheinjectednoiseandindicateifoverfitting
ispresent. Theideabehindthismethodisthatoverfitorunderfitmodelswouldloseaccuracymore
slowly when trained using the noise-injected training set than optimally-fitted models. Since the
calculated PV measurement is only one value and we compare it with a threshold to determine if
thereisoverfitting. Toselectthethreshold, weperformedagridsearchforthethresholdsbetween
-1 and1 basedonthe F-scoreonthesimulateddataset.
The PV measurement requires retraining the models, which takes a significant amount of compu-
tational time but results in poor performance in the simulated dataset (see Table 10). We do not
calculatethe PVmeasurementonthereal-worlddatasetsinceretrainingthemodelstakestoolong,
andtheperformanceonthesimulateddatasetdoesnotjustifythisextraeffort.
18


<!-- Page 19 -->

Underreviewasaconferencepaperat ICLR2023
Table11: Resultsofearlystoppingbasedonthesmoothedvalidationlosscurves.
Smooth Optimum Median Smooth Optimum Median
Patience Patience
epochs rate delay epochs rate delay
0 0.48 20.0 0 0.90 60.0
5 0.38 20.0 5 0.93 61.0
20 60
10 0.40 20.0 10 0.93 62.0
15 0.55 22.0 15 0.90 62.0
0 0.83 40.0 0 0.98 80.0
5 0.85 42.0 5 0.98 81.0
40 80
10 0.83 43.5 10 0.98 82.0
15 0.88 44.5 15 0.90 81.5
E EARLY STOPPING BASED ON SMOOTHED VALIDATION LOSS CURVES
Analternateversionofearlystoppinginspectsthemovingaverageofthesmoothedvalidationloss
curves (Shumway & Stoffer, 2017; Molugaram & Rao, 2017) to determine whether to stop the
trainingprocess. Afterterminatingthetrainingprocess, earlystoppingreturnsthebestepochwhich
has the lowest validation loss (not the smoothed value). Table 11 shows that using a smoothed
validation loss curve may increase the optimal rate and slightly increases the delay. However, it
doesnothavebetterperformancethanthebasicformofearlystoppingandcannotcompetewithour
approach. Figure8 showsanexampleinwhichearlystoppingachievestheoptimalepochafterusing
thesmoothedvalidationloss. However, thesmoothedcurvescanalsohurttheperformance, suchas
dropping10%oftheoptimalratewhenearlystoppingusingasmoothingwindowof5 epochsanda
patienceparameterof20 epochs. Figure9 illustratesanexampleofmissingtheoptimalepochafter
smoothingthevalidationlosscurve.
F SIGNIFICANCE TEST FOR THE RESULTS OF OVERFITTING PREVENTION
Tostudythedifferenceindelayacrossoverfittingpreventionapproaches, weperformedthe Mann-
Whitney Utest (Mann&Whitney,1947) atasignificancelevelofα = 0.05 todeterminewhether
thedistributionsofthedelayepochsofearlystoppingandourapproacharesignificantlydifferent.
Wealsocomputed Cliff’sdeltad (Longetal.,2003) effectsizetoquantifythedifferencebasedon
theprovidedthresholds (Romanoetal.,2006):
Before smoothing After smoothing
2.2 Training loss 2.2 Training loss
Validation loss Validation loss
2.0 Return epoch 2.0 Return epoch
Stop epoch Stop epoch
1.8 1.8
1.6 1.6
1.4 1.4
1.2 1.2
1.0 1.0
0.8 0.8
0.6 0.6
0 25 50 75 100 125 150 175 200 0 25 50 75 100 125 150 175 200
Epoch Epoch
Figure8: Anexampleinwhichearlystoppingachievestheoptimalepochbasedonsmoothedvali-
dationloss (thepatienceparameterissetto40 epochsandthesmoothingwindowsizeis5).
19


<!-- Page 20 -->

Underreviewasaconferencepaperat ICLR2023
Before smoothing After smoothing
2.2 Training loss 2.2 Training loss
Validation loss Validation loss
2.0 Return epoch 2.0 Return epoch
Stop epoch Stop epoch
1.8 1.8
1.6 1.6
1.4 1.4
1.2 1.2
1.0 1.0
0 25 50 75 100 125 150 175 200 0 25 50 75 100 125 150 175 200
Epoch Epoch
Figure9: Anexampleinwhichearlystoppingcannotachievetheoptimalepochbasedonsmoothed
validationloss (thepatienceparameterissetto20 epochsandthesmoothingwindowsizeis5).
Table12: Significancetestingresultsofthedelaysforoverfittingprevention. (Ws.: Windowsize)
Effect Cliff’s Effect Cliff’s
Classifier Ws. P Classifier Ws. P
Size dvalue Size dvalue
20 0.000 large -0.600 20 0.000 large -1.000
40 0.001 medium -0.405 40 0.000 large -0.956
BOSSVS 60 0.000 large -0.476 SAX-VSM 60 0.000 large -0.868
80 0.000 large -0.522 80 0.000 large -0.817
100 0.001 medium -0.449 100 0.000 large -0.681
20 0.000 large -1.000 20 0.000 large -0.875
40 0.000 large -0.966 40 0.000 large -0.893
HMM-GMM 60 0.000 large -0.859 TSBF 60 0.000 large -0.742
80 0.000 large -0.793 80 0.000 large -0.748
100 0.000 large -0.633 100 0.000 large -0.581
20 0.101 small 0.200 20 0.000 large -0.525
40 0.001 medium -0.392 40 0.000 large -0.804
KNN-DTW 60 0.000 large -0.485 TSF 60 0.000 large -0.700
80 0.000 large -0.541 80 0.000 large -0.709
100 0.000 large -0.517 100 0.000 large -0.619
 negligible, if |d|≤0.147

small, if 0.147<|d|≤0.33
Effectsize= (1)
medium, if 0.33<|d|≤0.474

large, if 0.474<|d|≤1
Table 12 shows that the distributions of delay of early stopping and our methods are significantly
different except for KNN-DTW with a window size of 20 epochs. In addition, the delays of our
methodsareshorterthanearlystopping (exceptfor KNN-DTWwithawindowsizeof20 epochs)
withatleastamediumeffectsize.
G USING ZERO-ONE LOSS FOR OVERFITTING PREVENTION
Overfittingpreventionmethodsmaystopthetrainingprocessbyinspectingthevalidationaccuracy
error (i.e., zero-one loss) rather than the loss utilized for optimizing the model. Figure 10 shows
the optimal rate of our approaches as well as early stopping based on zero-one validation loss.
When compared to the results in Figure 5 (which uses validation loss), the optimal rate of early
stopping based on zero-one loss is quite similar. The results show that the optimal rate improves
(withanaverageof2.5%) whenthepatiencevalueisbetween35 and90 epochsbutdeclines (withan
20


<!-- Page 21 -->

Underreviewasaconferencepaperat ICLR2023
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
5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95100105110115
Early stopping
KNN-DTW
HMM-GMM
TSF
TSBF
SAX-VSM
BOSSVS
Figure10: Theoptimalrateofourmethods (usingarollingwindow) andearlystoppingwithdiffer-
entpatiencevaluesbasedonzero-onevalidationlosscurves.
averageof2.9%) whenthepatiencevalueislessthan35 epochs. Forourmethodwith KNN-DTW,
theresultsshowthatusingzero-onelosscurvesincreasestheoptimalrate, whichstilloutperforms
earlystoppingandothertimeseriesclassifiers. However, thedelayofthe KNN-DTWalsoincreases
andstopslaterthanearlystoppingwhenthewindowsizeislessthanorequalto40 epochs (please
see Table 13 for details). Furthermore, we found that our approach with KNN-DTW has a higher
optimalratebutlongerdelaywhileusingthezero-onelosscurves (comparedtousingtheoriginal
losscurves).
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
5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95100105110115
Early stopping
KNN-DTW
HMM-GMM
TSF
TSBF
SAX-VSM
BOSSVS
Figure 11: The optimal rate of our methods (using a rolling window) and early stopping with dif-
ferentpatiencevalues. (Thisfigureisthesameas Figure5, weaddedithereforeasycomparisonto
Figure10.)
H A HEURISTIC METHOD FOR AUTOMATIC LABELLING
Wedevelopedaheuristicmethodforlabellingthetraininghistoryinthesimulateddatasetasover-
fittingbasedonthefollowingconditions:
• Thetraininglossandvalidationlossbothdecreaseinthefirstinc percentageofthetrain-
p
inghistory.
• Thetraininglossandvalidationlossbothdecreaseinthelastdec percentageofthetraining
p
history.
21


<!-- Page 22 -->

Underreviewasaconferencepaperat ICLR2023
Table13: Themediandelayandsignificanttestingofouroverfittingpreventionmethodsbasedon
zero-one validation loss curves with different window sizes. (Ws.: Window size; Md.: Median
delay)
Effect Cliff’s Effect Cliff’s
Ws. Classifier Md. P Classifier Md. P
size dvalue size dvalue
20 13.5 0.684 neg -0.050 5.0 0.000 large -0.950
40 46.0 0.109 small 0.198 5.5 0.000 large -0.976
60 BOSSVS 44.5 0.494 neg -0.087 SAX-VSM 11.5 0.000 large -0.884
80 46.5 0.115 small -0.202 14.5 0.000 large -0.806
100 49.5 0.029 small -0.284 26.5 0.000 large -0.716
20 2.0 0.000 large -0.850 7.5 0.000 large -0.750
40 5.0 0.000 large -0.739 14.5 0.000 large -0.867
60 HMM-GMM 8.0 0.000 large -0.656 TSBF 27.0 0.000 large -0.809
80 18.0 0.000 large -0.719 37.0 0.000 large -0.708
100 46.5 0.001 medium -0.431 45.0 0.000 large -0.561
20 48.5 0.000 large 0.650 12.5 0.001 medium -0.400
40 44.0 0.027 small 0.273 24.0 0.000 large -0.726
60 KNN-DTW 47.0 0.392 neg -0.109 TSF 31.0 0.000 large -0.630
80 48.5 0.194 small -0.166 42.5 0.000 large -0.596
100 56.5 0.024 small -0.292 47.0 0.000 large -0.479
• Thegapbetweenthetraininglossandvalidationlossexceedsgap percentageofthesum
p
ofthetrainingandvalidationloss.
Toselectthethresholds, weperformedagridsearchbetween10%to50%forinc anddec , anda
p p
gridsearchbetween1%to50%forgap . Thebestperformanceoftheheuristicmethodcanachieve
p
isa0.75 F-scorefortheoverfittingsampleswith0.96 precisionand0.61 recall. Theresultshowsthat
the heuristic method does not work well on the simulated dataset in comparison to other methods
(see Table7). Furthermore, theheuristicmethodperformspoorlyonthereal-worlddataset, witha
0.22 average F-score. Hence, humanlabelsarestillrequiredfortrainingthetimeseriesclassifiers.
22
