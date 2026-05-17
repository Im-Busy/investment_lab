# Circuit-Based Intrinsic Methods to Detect Overfitting

> *Source PDF: Circuit-Based Intrinsic Methods to Detect Overfitting.pdf*
> *Extraction: Combined — previous structured extraction (base) + markitdown raw text*

---

# Circuit-Based Intrinsic Methods to Detect Overfitting

> *Source PDF: Circuit-Based Intrinsic Methods to Detect Overfitting.pdf*
> *Extraction: Combined — markitdown raw text (base) + previous extraction supplements*

---

|     |     | Circuit-Based | Intrinsic           |     | Methods                                          | to Detect | Overfitting |     |     |
| --- | --- | ------------- | ------------------- | --- | ------------------------------------------------ | --------- | ----------- | --- | --- |
|     |     |               | SatrajitChatterjee1 |     | AlanMishchenko2                                  |           |             |     |     |
|     |     | Abstract      |                     |     | knowledge,suchas,theperformanceofthemodelonexam- |           |             |     |     |
plesheldoutfromthetrainingprocess,detailsoftheprocess
Thefocusofthispaperisonintrinsicmethodsto
usedtofindthemodel(e.g.,multiplehypothesistestingwith
| detectoverfitting. |     | Byintrinsicmethods,wemean |     |     |     |     |     |     |     |
| ------------------ | --- | ------------------------- | --- | --- | --- | --- | --- | --- | --- |
0202 guA 5  ]GL.sc[  2v19910.7091:viXra methodsthatrelyonlyonthemodelandthetrain- registration),orlimitationsofthefunctionfamilytowhich
themodelbelongs(e.g.,VCdimension,Rademachercom-
ingdata,asopposedtotraditionalmethods(we
plexity)orthoseofthesizeoftheparameterspaceofthe
callthemextrinsicmethods)thatrelyonperfor-
model(e.g.,AkaikeInformationCriterion).
manceonatestsetoronboundsfrommodelcom-
plexity. Weproposeafamilyofintrinsicmethods Weclassifymethodsrelyingontheknowledgeofthefunc-
| called | Counterfactual | Simulation | (CFS) | which |     |     |     |     |     |
| ------ | -------------- | ---------- | ----- | ----- | --- | --- | --- | --- | --- |
tionfamilyasextrinsicbecauseoftentheinformationrel-
analyze the flow of training examples through evant to overfitting is not directly represented in a given
themodelbyidentifyingandperturbingrarepat-
|     |     |     |     |     | model. | For example, | a model | could have been | found by |
| --- | --- | --- | --- | --- | ------ | ------------ | ------- | --------------- | -------- |
terns. ByapplyingCFStologiccircuitswegeta searchingamuchsmallerspacethanthatimpliednaïvelyby
methodthathasnohyper-parametersandworks thefunctionfamilytowhichitbelongs,duetoeitherexplicit
uniformlyacrossdifferenttypesofmodelssuch
regularizationortheregularizationimplicitintheoptimiza-
as neural networks, random forests and lookup tionorsearchprocedure. Conversely,thegivenmodelcould
| tables. | Experimentally,CFScanseparatemodels |     |     |     |     |     |     |     |     |
| ------- | ----------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- |
havebeenfoundbysearchingamuchlargerspaceofmod-
with different levels of overfit using only their elsthroughhyper-parametersearch,orbypickingthebest
logiccircuitrepresentationswithoutanyaccess modelfromanumberofdifferentmodelfamilies,butthe
| tothehighlevelstructure. |     |     | Bycomparinglookup |     |     |     |     |     |     |
| ------------------------ | --- | --- | ----------------- | --- | --- | --- | --- | --- | --- |
specificfinalmodelitselfdoesnotcarryanyvestigesofthe
tables, neural networks, and random forests us- largerspacethatwassearchedover. Bothofthesesituations
ingCFS,wegetinsightintowhyneuralnetworks arecommoninmodernmachinelearning.
| generalize. | Inparticular,wefindthatstochastic |     |     |     |     |     |     |     |     |
| ----------- | --------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- |
Intrinsicmethodsareofpracticalinterestsincewithmodern
| gradient | descent | in neural | nets does | not lead | to  |     |     |     |     |
| -------- | ------- | --------- | --------- | -------- | --- | --- | --- | --- | --- |
deeplearningmodels,wefindthatextrinsicestimatesbased
“bruteforce”memorization,butfindscommonpat-
onmodelcomplexityaretypicallyvacuoussincethesemod-
terns(whetherwetrainwithactualorrandomized
elsarepowerfulenoughtofitarbitrarydata(Zhangetal.,
labels),andneuralnetworksarenotunlikeforests
in this regard. Finally, we identify a limitation 2017). Consequently,practitionersresorttostudyingperfor-
manceonaholdoutdataset(orcrossvalidation),butthisis
withourproposalthatmakesitunsuitableinan
|                     |     |                         |     |     | unsatisfactoryforacoupleofreasons. |     |     | First,thismeansthat, |     |
| ------------------- | --- | ----------------------- | --- | --- | ---------------------------------- | --- | --- | -------------------- | --- |
| adversarialsetting, |     | butpointsthewaytofuture |     |     |                                    |     |     |                      |     |
inalowdatasetting,wecannotuseallthedatafortraining,
workonrobustintrinsicmethods.
|     |     |     |     |     | but                                     | have to keep | significant | portions aside | for validation |
| --- | --- | --- | --- | --- | --------------------------------------- | ------------ | ----------- | -------------- | -------------- |
|     |     |     |     |     | (e.g.,seediscussioninDietterich(1998)). |              |             | Second,itmay   |                |
1.Introduction bedifficulttoensureapristineholdoutthatisnottouched
duringtheresearchprocessparticularlyiftheprojectislong
Thispaperconsidersmethodstodetectoverfittingofamodel running. Even with a few queries to the hold out during
based only on the model and the training data. In termi- theresearchprocess,itispossibletostartfittingtothehold
nologythatweintroduce,wecallsuchmethodsintrinsic,
out(Dworketal.,2015).
incontrasttoextrinsicmethods,whichrelyonadditional
Intrinsicmethodsarealsointerestingfromatheoreticalper-
| 1Google, |     |     |     | 2Department |     |     |     |     |     |
| -------- | --- | --- | --- | ----------- | --- | --- | --- | --- | --- |
Mountain View, California, USA of spective. Imaginethatwehavesufficientcomputingpower
EECS,UniversityofCalifornia,Berkeley,California,USA.Corre- to,say,enumerateallneuralnetworks(andtheirweights)
spondenceto:SatrajitChatterjee<schatter@google.com>,Alan up to a certain size. Among all the networks that fit the
Mishchenko<alanmi@berkeley.edu>.
datawell,intrinsicmethodscoulddistinguishbetweenthose
|                               |        | 37th          |                       |            | networksthatgeneralizewellfromthosethatdonot,andwe |     |     |     |     |
| ----------------------------- | ------ | ------------- | --------------------- | ---------- | -------------------------------------------------- | --- | --- | --- | --- |
| Proceedings                   | of the | International | Conference            | on Machine |                                                    |     |     |     |     |
| Learning,Online,PMLR119,2020. |        |               | Copyright2020bytheau- |            |                                                    |     |     |     |     |
thor(s).

Circuit-BasedIntrinsicMethodstoDetectOverfitting
| couldviewthemodelasacertificateofgeneralization.1     |     |     |     | In  |     |     |     |     |     |     |     |
| ----------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| addition,iftheintrinsicmethodwasefficient,itwouldmean |     |     |     |     | x   |     |     |     |     |     |     |
thatsupervisedlearning(andnotjustfittingthetrainingdata)
isinNP.Intrinsicmethodscanalsohelpshedlightonwhy
|     |     |     |     |     |     | =   |     | =   |     | =   |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
neural networks trained with stochastic gradient descent x x x
|     |     |     |     |     |     | n -1 |     | 1   | 0   |     |     |
| --- | --- | --- | --- | --- | --- | ---- | --- | --- | --- | --- | --- |
...
| generalizeinspiteoftheirlargecapacity.              |     |     | Somerecentanal- |     |     |      |     |     |     |     |     |
| --------------------------------------------------- | --- | --- | --------------- | --- | --- | ---- | --- | --- | --- | --- | --- |
|                                                     |     |     |                 |     |     | s    |     | s   |     | s   |     |
| ysesbasedonnormalizedmargin,curvature,etc.(Bartlett |     |     |                 |     |     | n -1 |     | 1   |     | 0   |     |
et al., 2017; Rangamani et al., 2019; Arora et al., 2018; 1000000000 0 0 0
| Neyshaburetal.,2018)maybeseenasintrinsicestimates |     |     |     |     |     |     |     |     |     |     | y   |
| ------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
forgeneralizationalbeitspecializedtoneuralnetworks. y 1 y 1 y 1
|     |     |     |     |     |     | n -1 |     | 1   | 0   |     |     |
| --- | --- | --- | --- | --- | --- | ---- | --- | --- | --- | --- | --- |
Intrinsicmethodscanbeconsideredinthecontextofapro-
| tocolinvolvingtwoagents. |     | LetS | beapublicdatasetdrawn |     |     |     |     |     |     |     |     |
| ------------------------ | --- | ---- | --------------------- | --- | --- | --- | --- | --- | --- | --- | --- |
Figure1: Acircuitimplementingalookuptablethatmem-
| fromadistributionD |     | thatgeneratessamplesinfrequently |     |     |     |     |     |     |     |     |     |
| ------------------ | --- | -------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
(e.g.,quarterlyfinancialstatementsandstockmarketreturns orizes the training examples (x i ,y i ) for 0 ≤ i < n. We
|     |     |     |     |     | observe | that in | this extreme | case of | overfitting, | there | are |
| --- | --- | --- | --- | --- | ------- | ------- | ------------ | ------- | ------------ | ----- | --- |
ofpubliccompanies,orpublichealthdataontreatmentsand
|                   |                                 |     |     |     | signalsinthecircuit(thes |     |     | )thatidentifyspecifictraining |     |     |     |
| ----------------- | ------------------------------- | --- | --- | --- | ------------------------ | --- | --- | ----------------------------- | --- | --- | --- |
| patientoutcomes). | SupposeArthurwantstobuildamodel |     |     |     |                          |     |     | i                             |     |     |     |
fromS butinsteadofdoingsohimself,heoutsourcesitto examples. For e.g., s 0 is 1 only when x = x 0 and 0 for
|                              |     |     |                      |     | all | other x (assuming | the | x are distinct). |     | We say | that 1 |
| ---------------------------- | --- | --- | -------------------- | --- | --- | ----------------- | --- | ---------------- | --- | ------ | ------ |
| Merlin,anuntrustedadversary. |     |     | Merlincomesbackwitha |     |     | i                 |     | i                |     |        |        |
|                              |     |     |                      |     |     | rare pattern      |     | s                |     |        |        |
modelMbutdoesnotdiscloseanydetailsofhismodeling is a for the signal 0 . Based on this exam-
process. HowcanArthurconvincehimselfthatMisnot ple,weproposethattheoccurrenceofrarepatternsduring
simulationofatrainingsetthroughamodelindicatesover-
| horriblyoverfit? | Forexample,Mcouldsimplybealookup |     |     |     |     |     |     |     |     |     |     |
| ---------------- | -------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
table built from S. Normally Arthur would evaluate M fitting,andinthiswork,weexploretowhatextentsuchrare
patternscanbeusedtodetectoverfittinginmorecomplex
onnewsamplesfromD,butinoursetup,Arthurdoesnot
modelssuchasneuralnetworksandrandomforests.
| haveanysamplesotherthanthoseinS |                                  |     | sincealltheexisting |     |     |     |     |     |     |     |     |
| ------------------------------- | -------------------------------- | --- | ------------------- | --- | --- | --- | --- | --- | --- | --- | --- |
| dataispublic.                   | Now, ifArthuronlyhasaccesstoMasa |     |                     |     |     |     |     |     |     |     |     |
blackboxandhecanonlyevaluateMonelementsofS,it
appearsthereislittlehecandotodistinguishagoodmodel a directed acyclic graph (DAG) of fixed or floating point
adders,multipliers,andpointwisenon-linearities.Finally,at
| from a lookup | table. | But can | Arthur do better | if he has |     |     |     |     |     |     |     |
| ------------- | ------ | ------- | ---------------- | --------- | --- | --- | --- | --- | --- | --- | --- |
accesstotheinternalsignalsintheimplementationofM? thelowestlevelofabstraction,wecandescribethestructure
Thisisthecentralquestionofthispaper. ofMasaDAGofprimitivelogicgatessuchas2-inputAnd
|     |     |     |     |     | gates | and inverters, | i.e., | as a combinational |     | logic | circuit. |
| --- | --- | --- | --- | --- | ----- | -------------- | ----- | ------------------ | --- | ----- | -------- |
Wetakeafirststeptowardsansweringthisquestionbystudy-
Inoursetup,itisnaturaltoworkatthislowestlevel,i.e.,
inganaturally-motivatedfamilyofintrinsicmethods,called
logicgatessinceitallowsdifferentkindsofmodelssuch
CounterfactualSimulation(CFS),andevaluatingtheiref-
aslookuptables,randomforests,andneuralnetworksall
| ficacyexperimentallyonabenchmarkproblem. |     |     |     | Themain |                              |     |     |                    |     |     |     |
| ---------------------------------------- | --- | --- | --- | ------- | ---------------------------- | --- | --- | ------------------ | --- | --- | --- |
|                                          |     |     |     |         | tobemappedintothesameformat. |     |     | Thus,Merlinneednot |     |     |     |
ideabehindCFSistoanalyzetheflowofthetrainingexam- discloseevenwhattypeofmodelhehasbuilt,butsimply
plesinSthroughthestructureofM.Thisisonlyafirststep
providesArthurwithacombinationallogiccircuitforthe
sinceCFS,althoughpromisinginpractice,hassignificant
model.
| limitations. | Inparticular,ourexperimentsshowthateven |     |     |     |     |     |     |     |     |     |     |
| ------------ | --------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
ifwecouldproveboundsbasedonCFS,theywouldnotbe Tomakethisconcrete,considertheMNISTimageclassifi-
tightenoughinanadversarialsetting. However,wehope cationproblem(LeCun&Cortes,2010)whichwewilluse
thatthispaperencouragesresearchtoovercometheselimi- asarunningexample. Disthedistributionofhandwritten
tationsortoshowthatnosuchmethodcanexist,especially digits and their classes. S is a sample from D of 60,000
|                                      |     |     |     |     | imagesx                         | andtheircorrespondinglabelsy |     |     | (thus,0≤i< |               |     |
| ------------------------------------ | --- | --- | --- | --- | ------------------------------- | ---------------------------- | --- | --- | ---------- | ------------- | --- |
| forlearningtasksofpracticalinterest. |     |     |     |     |                                 | i                            |     |     | i          |               |     |
|                                      |     |     |     |     | 60000)i.e.,theMNISTtrainingset. |                              |     |     | Eachx      | i is6,272bits |     |
wide(correspondingto28×28pixels×8bitsperpixel),
2.CounterfactualSimulation(CFS)
y
|     |     |     |     |     | and | each i is 10 | bits wide | (for a 1-hot | representation |     | of  |
| --- | --- | --- | --- | --- | --- | ------------ | --------- | ------------ | -------------- | --- | --- |
ThestructureofMcanbedescribedatdifferentlevelsof the 10 possible classes). Therefore, a classifier to solve
thisproblemisacircuitwith6,272Booleaninputsand10
| abstraction.                                    | For instance, | if M         | is a fully connected | feed           |                 |     |     |     |     |     |     |
| ----------------------------------------------- | ------------- | ------------ | -------------------- | -------------- | --------------- | --- | --- | --- | --- | --- | --- |
| forwardneuralnetwork,wecandescribeitasasequence |               |              |                      |                | Booleanoutputs. |     |     |     |     |     |     |
| of layers.                                      | Going one     | level lower, | we can               | describe it as |                 |     |     |     |     |     |     |
SupposeMerlin’smodelfortheMNISTclassifierisasimple
1Keepingaholdoutsetwouldnothelpushere—whatwould lookuptable. Howwouldthecircuitforitlook? Figure1
thatevenmean? sketchesonepossibility. The6,272-bitinputxiscompared
|     |     |     |     |     | witheachoftheexamplesx |     |     | inturnandifthereisamatch, |     |     |     |
| --- | --- | --- | --- | --- | ---------------------- | --- | --- | ------------------------- | --- | --- | --- |
i

Circuit-BasedIntrinsicMethodstoDetectOverfitting
thecorresponding10-bitoutputy isselected.Ifnoexample OtherTypesofCFS.Thereareothervariantsofthepro-
i
matches,thenthemodel(arbitrarily)returnsthe1-hotvector ceduredescribedabove(whichwecallSimpleCFSorjust
representingclass‘0’. Now,ifwesimulatethiscircuiton CFS).OfparticularinterestisCompositeCFSwhichisuse-
examplesinS,wenoticethatthereareinternalsignalsin fulforcircuitswithgatesthathavemanyinputsorareat
thecircuitthatarecapableofidentifyingspecifictraining higherlevelsofabstraction. InCompositeCFS,welookat
examples. For example, the signal s (the output of the rarepatternsincombinationsofsignalsfeedingaparticular
0
x=x block)is1(true)forthetrainingexamplex and0 gateandperturbtheoutputofthatgate(byflippingit)when
| 0   |     |     |     |     |     | 0   |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
(false)forallothers. Inthiscase,wesay1isararepattern ararecombinationisseenattheinputs. Anotherpossibility
fors sinces rarelytakesonthevalue1onthetrainingset. istorandomizetheperturbationinsteadofalwaysflipping.
| 0   | 0   |     |     |     |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Formally,ifasignalsinMtakesonthevaluevatmostl Inourexperiments,wefoundthesevariantstoproducere-
timesonthetrainingsetS,wecallvanl-rarepatternfors. sults that are similar to those obtained from Simple CFS,
andsoweonlymentiontheminpassing.
| This observation | leads | to  | the first | of the | two main | ideas |     |     |     |     |     |     |     |     |
| ---------------- | ----- | --- | --------- | ------ | -------- | ----- | --- | --- | --- | --- | --- | --- | --- | --- |
behindCFS:Thepresenceofl-rarepatternssuggestsover-
fitting,and,therefore,poorgeneralizationsincetheyopen 3.ExperimentalResults
M
| up the possibility |     | that | has special | logic | to detect | and |     |     |     |     |     |     |     |     |
| ------------------ | --- | ---- | ----------- | ----- | --------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
CFSImplementation.Ourimplementationofl-CFSworks
| handlespecificexamples. |     |     | Acountofrarepatterns,however, |     |     |     |                          |     |     |     |                            |     |     |     |
| ----------------------- | --- | --- | ----------------------------- | --- | --- | --- | ------------------------ | --- | --- | --- | -------------------------- | --- | --- | --- |
|                         |     |     |                               |     |     |     | onadirectedacyclicgraphG |     |     |     | representingacombinational |     |     |     |
doesnotdirectlytranslateintoametricforgeneralization
|                                           |     |     |     |     |              |     | logic circuit | where  | each      | node | is either | the          | constant | 0, a  |
| ----------------------------------------- | --- | --- | --- | --- | ------------ | --- | ------------- | ------ | --------- | ---- | --------- | ------------ | -------- | ----- |
| (withoutbuildingapredictivemodelofthat!). |     |     |     |     | Furthermore, |     |               |        |           |      |           |              |          |       |
|                                           |     |     |     |     |              |     | primary       | input, | a 2-input | And  | gate,     | or a 2-input | Xor      | gate. |
althoughapatternmayberareitmayalsobeanobservabil-
|     |     |     |     |     |     |     | An edge | is either | a direct | connection |     | or an | inverter | and |
| --- | --- | --- | --- | --- | --- | --- | ------- | --------- | -------- | ---------- | --- | ----- | -------- | --- |
itydon’tcare(ODC),i.e.,itmayhavenoinfluenceonthe
representsaBooleanfunctionintermsoftheprimaryinputs.
| outputofthecircuit. |            | Forexample,ifthesignalwiththerare |          |       |             |     |         |           |       |              |     |       |         |       |
| ------------------- | ---------- | --------------------------------- | -------- | ----- | ----------- | --- | ------- | --------- | ----- | ------------ | --- | ----- | ------- | ----- |
|                     |            |                                   |          |       |             |     | This is | a variant | of an | And-Inverter |     | Graph | (Biere, | 2007; |
| pattern             | only feeds | into an                           | And gate | whose | other input | is  |         |           |       |              |     |       |         |       |
Chatterjee,2007),astandarddatastructureinmodernlogic
0whentherarepatternappears,thenthevalueoftherare
synthesisusedtohandlecircuitswithhundredsofmillions
patterndoesnotmatterindecidingtheoutputofthecircuit.
|     |     |     |     |     |     |     | ofnodes. | Whileconstructingthesegraphs, |     |     |     |     | wepropagate |     |
| --- | --- | --- | --- | --- | --- | --- | -------- | ----------------------------- | --- | --- | --- | --- | ----------- | --- |
Weaddressbothproblemswiththesecondmainideabehind constantsbutdonotextractcommonsub-expressions.
CFS:perturbedsimulationofatrainingexamplewherewe
|          |            |         |     |           |          |     | WemaketwopassesthroughthenodesofG |     |              |     |         |        | intopological |          |
| -------- | ---------- | ------- | --- | --------- | -------- | --- | --------------------------------- | --- | ------------ | --- | ------- | ------ | ------------- | -------- |
| simulate | an example | through | M   | as usual, | but when | we  |                                   |     |              |     |         |        |               |          |
|          |            |         |     |           |          |     | order starting                    |     | from primary |     | inputs. | In the | first         | pass, we |
encounteral-rarepattern,insteadofpropagatingittothe
|     |     |     |     |     |     |     | simulatethetrainingsetthroughG |     |     |     | toobtainthecountsof |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ------------------------------ | --- | --- | --- | ------------------- | --- | --- | --- |
fanouts(i.e.,gatesthatdependonthissignal),weperturbthe
|     |     |     |     |     |     |     | differentpatternsinthecircuit. |     |     |     | Inthesecondpass,weuse |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ------------------------------ | --- | --- | --- | --------------------- | --- | --- | --- |
patternandsimulatethefanoutswiththeperturbedpattern.
thecountsfromthefirstpasstoperturbthel-rarepatterns.
| A natural | perturbation | is       | to propagate | the     | opposite | value |           |      |      |            |     |           |         |      |
| --------- | ------------ | -------- | ------------ | ------- | -------- | ----- | --------- | ---- | ---- | ---------- | --- | --------- | ------- | ---- |
|           |              |          |              |         |          |       | In Simple | CFS, | this | boils down | to  | replacing | signals | that |
| instead   | of the rare  | pattern. | In our       | running | example, | this  |           |      |      |            |     |           |         |      |
takeonavalueof0onmostexampleswiththeconstant0
| corresponds                                     | to propagating |     | a 0 instead | of  | 1 for signal | s 0 |                                      |     |     |                           |     |                 |     |     |
| ----------------------------------------------- | -------------- | --- | ----------- | --- | ------------ | --- | ------------------------------------ | --- | --- | ------------------------- | --- | --------------- | --- | --- |
|                                                 |                |     |             |     |              |     | signalandlikewisefor1.               |     |     | CFSthusrunsinlineartimein |     |                 |     |     |
| tothemultiplexerwhensimulatingthetrainingimagex |                |     |             |     |              |     | .                                    |     |     |                           |     |                 |     |     |
|                                                 |                |     |             |     |              | 0   | thesizeofthegraphandthetrainingdata. |     |     |                           |     | Forperformance, |     |     |
Inthismanner,wepreventthemodelfromidentifyingx
0
|                           |     |     |                        |     |     |     | the simulations                |     | are done | in  | a bit parallel |                   | manner | for all |
| ------------------------- | --- | --- | ---------------------- | --- | --- | --- | ------------------------------ | --- | -------- | --- | -------------- | ----------------- | ------ | ------- |
| andweseethattheoutputforx |     |     | isnolongernecessarilyy |     |     |     | .                              |     |          |     |                |                   |        |         |
|                           |     |     | 0                      |     |     | 0   | trainingexamplesatthesametime. |     |          |     |                | Toavoidrunningout |        |         |
Wecallthismodifiedsimulationprocedurel-counterfactual
|     |     |     |     |     |     |     | of memory, | we  | use reference |     | counting | to  | recycle | storage |
| --- | --- | --- | --- | --- | --- | --- | ---------- | --- | ------------- | --- | -------- | --- | ------- | ------- |
simulationorl-CFSforshort.
forintermediatesimulatedvalueswhentheyarenolonger
Weperformperturbedsimulationforeachtrainingexample needed. A typical run of l-CFS in our experiments takes
inturnandmeasuretheresultingaverageaccuracyoverthe lessthan10minutesona3.7GHzXeonCPUandlessthan
| trainingset. | Wecallthisquantitythetrainingaccuracyob- |     |     |     |     |     | 2GBofRAM. |     |     |     |     |     |     |     |
| ------------ | ---------------------------------------- | --- | --- | --- | --- | --- | --------- | --- | --- | --- | --- | --- | --- | --- |
tainedthroughl-CFS.Inourrunningexampleofthelookup
|     |     |     |     |     |     |     | BenchmarkProblem. |     |     | Whilethediscussionfromthepre- |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ----------------- | --- | --- | ----------------------------- | --- | --- | --- | --- |
table, itiseasytoseethatthetrainingaccuracyobtained
|     |     |     |     |     |     |     | vious section |     | shows | how CFS | can | discover | overfit | when |
| --- | --- | --- | --- | --- | --- | --- | ------------- | --- | ----- | ------- | --- | -------- | ------- | ---- |
through1-CFSisnobetterthanrandomchance(sinceeach
|     |     |     |     |     |     |     | the model | is a | simple | lookup | table, | it is not | clear | if CFS |
| --- | --- | --- | --- | --- | --- | --- | --------- | ---- | ------ | ------ | ------ | --------- | ----- | ------ |
trainingexampleismappedtoclass‘0’under1-CFS).Now,
wouldbeeffectiveonneuralnetworkstrainedwithstochas-
| since random | chance | is what | one | would expect | to  | be the |              |         |     |        |           |      |           |     |
| ------------ | ------ | ------- | --- | ------------ | --- | ------ | ------------ | ------- | --- | ------ | --------- | ---- | --------- | --- |
|              |        |         |     |              |     |        | tic gradient | descent |     | (SGD). | To answer | this | question, | we  |
generalizationofthelookuptable(i.e.,itsaccuracyonD),
trained3neuralnetworksforMNISTinTensorFlow(Keras)
itistemptingtoconjecturethat1-CFStrainingaccuracyis
andcompiledthemdownintocombinationallogiccircuits.
| agoodestimateofaccuracyonD. |     |     |     | Althoughthatisnotthe |     |     |                                      |     |     |     |     |     |              |     |
| --------------------------- | --- | --- | --- | -------------------- | --- | --- | ------------------------------------ | --- | --- | --- | --- | --- | ------------ | --- |
|                             |     |     |     |                      |     |     | All3networkshavethesamearchitecture: |     |     |     |     |     | aninputlayer |     |
caseasweshallseeempiricallyinSection3,wefindthatthe
|     |     |     |     |     |     |     | of size | 784 (i.e., | 28  | × 28), | 3 fully | connected | ReLU | lay- |
| --- | --- | --- | --- | --- | --- | --- | ------- | ---------- | --- | ------ | ------- | --------- | ---- | ---- |
differenceintrainingaccuracybetweennormalsimulation
erswith256nodeseach,andafinalsoftmaxlayerwith10
andl-CFSisagoodmeasureofthedegreeofoverfitofM.
|     |     |     |     |     |     |     | outputs. | (Thus,thetotalnumberoftrainableparametersis |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | -------- | ------------------------------------------- | --- | --- | --- | --- | --- | --- |

Circuit-BasedIntrinsicMethodstoDetectOverfitting
335,114.) WealsoperformedsomeexperimentswithFash- Itisremarkablethatevenwhenaneuralnetworkisrepre-
ionMNIST(Xiaoetal.,2017)andtheresultsaresimilar. sentedataverylowlevelasalogiccircuit,relativeoverfit
canbedetectedusinganintuitivealgorithmwithnohyper-
Thefirsttwonetworks(nn-real-2andnn-real-100),
parameterstotune.
weretrainedontheMNISTtrainingsetfor2epochsand
100epochsrespectively. Theygettotraining(top-1)accu- Expt.2:ImpactofArchitecture.Therearemanydifferent
raciesof97%and99.90%respectively. Thethirdnetwork ways in which a neural network can be compiled down
(nn-random)wastrainedonavariantofMNISTwhere intologicgates. InExpt. 1,wemadecertainarchitectural
the output labels in the training set are permuted pseudo- choicesforthecircuit,butwhatifwehadchosendifferently?
randomly and trained for 300 epochs to get to a training To evaluate that, here we replace the multipliers used in
91.27%.2
accuracy of In all cases, we used the ADAM Expt. 1 with array multipliers (i.e., multipliers based on
optimizerwithdefaultparametersandbatchsizeof64. As the elementary algorithm for multiplication). Figure 2b
expected, nn-real-2 is the least overfit and gets to a showstheresultantCFScurves(withdashedlines)aswell
validationsetaccuracyof97%(i.e., hasanegligiblegen- astheoriginalcurves(solidlines)forreference. Thecurves
eralizationgap),nn-real-100ismoreoverfitgettingto donotcoincideindicatingthattheresultofCFSdepends
avalidationsetaccuracyof98.24%(agapof1.66%),and onthestructureofthecircuitandnotjustonthefunction
finally, the validation accuracy of nn-random is 9.73% implementedbythecircuit(sincethefunctionisthesame
(i.e.,closetochance)confirmingthatitishorriblyoverfit. inbothcases). However,forthesamechoiceofarchitecture,
wefindthatthefalloffinCFScurvesareagainindicativeof
| ConversiontoLogicCircuits. |     | Thisisdonebygenerating |     |     |     |     |     |
| -------------------------- | --- | ---------------------- | --- | --- | --- | --- | --- |
thedegreeofoverfit.
| logic subcircuits | composed | of 2-input | And or Xor gates |     |     |     |     |
| ----------------- | -------- | ---------- | ---------------- | --- | --- | --- | --- |
and inverters for each of the operations in the neural net- Expt. 3: ImpactofChoiceofPrimitives. Evenatthelow-
work. Weights and activations are represented by signed estlevelofabstraction,wecanchoosewhatprimitivesto
8-bitand16-bitfixedpointnumbersrespectivelywith6bits workwith. ToseehowthischoiceimpactsCFScurves,in
reservedforthefractionalpart. (Weightsfromtrainingare thisexperimentwedisallowXorgatesasprimitives(thus
clamped to [−2.0,2.0) before conversion to fixed point.) requiringthatonlyAndgatesandinvertersbeused). Fig-
Eachmultiply-accumulateunitmultipliesan8-bitconstant ure2cshowstheresultingCFScurves(dashed)aswellas
(theweight)witha16-bitinput(theactivation)andaccumu- theoriginals(solid). Onceagain,weseethatthecurvesdo
latesin24bitswithsaturation. Theconstantmultiplications notcoincideindicatingthatthechoiceofprimitivesmatters
are done by finding a minimal combination of bit-shifts but for the same choice of primitives, the falloff in CFS
(multiplicationsbypowersof2)andadditionsorsubtrac- curvesareagainindicativeofthedegreeofoverfit.
tions. Forexample,5×uisimplementedas4×u+uand
|                   |     |                         |     | Expt. 4: | Count of Rare Patterns. | Figure | 2d shows for |
| ----------------- | --- | ----------------------- | --- | -------- | ----------------------- | ------ | ------------ |
| 11×uas16×u−4×u−u. |     | ReLUsareimplementedwith |     |          |                         |        |              |
eachvalueofl,howmanyexampleshavenol-rarepatterns,
| acomparatorandamultiplexer. |     | Theoutputsofeachnet- |     |     |     |     |     |
| --------------------------- | --- | -------------------- | --- | --- | --- | --- | --- |
i.e.,cannotbepossiblyaffectedbyl-CFS.Weobservein
workarethe10signed16-bitactivationsbeforethesoftmax.
|     |     |     |     | particular, | that for nn-random, | there are | 54,094 exam- |
| --- | --- | --- | --- | ----------- | ------------------- | --------- | ------------ |
Whenevaluatingaccuracy(withCFSorwithout)wepick
|     |     |     |     | ples (about | 90% of the training | set) that have | no 1-rare |
| --- | --- | --- | --- | ----------- | ------------------- | -------------- | --------- |
theclasscorrespondingtothelargestofthe10activations
patterns. Thisisinsharpcontrasttoasimplelookuptable
| (top-1 accuracy). | The resulting | logic | circuits have 35 to |                                           |     |     |       |
| ----------------- | ------------- | ----- | ------------------- | ----------------------------------------- | --- | --- | ----- |
|                   |               |       |                     | whereeveryexamplewouldhavea1-rarepattern, |     |     | andin |
52millionAndorXorgatesand5500to6000logiclevels.
factslightlymorethannn-real-100.
Comparingthese
(Thesesizesalongwiththeneedtofitrandomdatadictated
curvesofcountstotheCFScurvesinFigure2aindicates
thechoiceofarchitectureandbenchmark.)
thatperturbationisanimportantpartofCFSandthatrarity
Expt. 1: EffectofSimpleCFS.Figure2ashowsthetrain- byitselfisacrudermeasureofoverfittingsinceitmaynot
| ingaccuraciesobtainedthroughl-CFSforeachofthethree |     |     |     | beobservable. |     |     |     |
| -------------------------------------------------- | --- | --- | --- | ------------- | --- | --- | --- |
networksaslvariesfrom1to1024(whichisabout1.7%of
|                               |                                         |                     |     | Expt. 5:    | Random Forests.     | Since CFS works | on the cir-  |
| ----------------------------- | --------------------------------------- | ------------------- | --- | ----------- | ------------------- | --------------- | ------------ |
| thenumberoftrainingexamples). |                                         | WecalltheseplotsCFS |     |             |                     |                 |              |
|                               |                                         |                     |     | cuit level, | it can check random | forests for     | overfit. Two |
| curves. Asl                   | increases,i.e.,asmorepatternsbecomerare |                     |     |             |                     |                 |              |
randomforestsweretrainedusingversion0.19.1ofScikit-
| and get perturbed, | the accuracy | falls eventually | reaching |                             |     |                         |     |
| ------------------ | ------------ | ---------------- | -------- | --------------------------- | --- | ----------------------- | --- |
|                    |              |                  |          | learn(Pedregosaetal.,2011). |     | Eachforesthas10treesand |     |
chance. However,itisinterestingthatthedropinaccuracy
|     |     |     |     | is trained | using the default settings, | except | for bootstrap- |
| --- | --- | --- | --- | ---------- | --------------------------- | ------ | -------------- |
ishighestfornn-random(e.g.,atl=64,thedropisabout
|                |                      |     |               | ping(toavoidnon-uniformweightsduringinference). |     |     | The |
| -------------- | -------------------- | --- | ------------- | ----------------------------------------------- | --- | --- | --- |
| 45%), somewhat | less for nn-real-100 |     | (20%) and the |                                                 |     |     |     |
firstforest(rf-real)wastrainedonMNISTwhereasthe
| leastfornn-real-2(1.4%). |     | Thusthefalloffinaccuracy |     |     |     |     |     |
| ------------------------ | --- | ------------------------ | --- | --- | --- | --- | --- |
secondforest(rf-random)wastrainedonMNISTwith
withlisanindicatorofthelevelofoverfitofanetwork.
theoutputlabelspseudo-randomlypermuted(asbeforewith
2Whileevaluatingrandomfortrainingaccuracy(withorwith- nn-random). Bothforestsreachperfecttrainingaccuracy.
outCFS),thepermutedlabelsareused. rf-realgets95.58%validationaccuracywhereasasex-

Circuit-BasedIntrinsicMethodstoDetectOverfitting
| 1.0                          |     |     |     | 1.0                          |     |     |     |     |
| ---------------------------- | --- | --- | --- | ---------------------------- | --- | --- | --- | --- |
| SFC-l htiw ycarucca gniniart |     |     |     | SFC-l htiw ycarucca gniniart |     |     |     |     |
| 0.8                          |     |     |     | 0.8                          |     |     |     |     |
| 0.6                          |     |     |     | 0.6                          |     |     |     |     |
| 0.4                          |     |     |     | 0.4 nn-real-2                |     |     |     |     |
nn-real-2 w/ array
nn-real-100
| 0.2 nn-real-2 |         |     |     | 0.2 nn-real-100 w/ array |     |         |     |     |
| ------------- | ------- | --- | --- | ------------------------ | --- | ------- | --- | --- |
| nn-real-100   |         |     |     | nn-random                |     |         |     |     |
| nn-random     |         |     |     | nn-random w/ array       |     |         |     |     |
| 0.0           |         |     |     | 0.0                      |     |         |     |     |
| 0 2           | 4       | 6 8 | 10  | 0                        | 2   | 4       | 6 8 | 10  |
|               | log2(l) |     |     |                          |     | log2(l) |     |     |
|               | (a)     |     |     |                          |     | (b)     |     |     |
60000
snrettap erar-l on /w selpmaxe .mun
| 1.0 |     |     |     |     |     |     | nn-real-2 |     |
| --- | --- | --- | --- | --- | --- | --- | --------- | --- |
nn-real-100
|     |     |     |     | 50000 |     |     | nn-random |     |
| --- | --- | --- | --- | ----- | --- | --- | --------- | --- |
SFC-l htiw ycarucca gniniart
0.8
40000
| 0.6            |     |     |     | 30000 |     |     |     |     |
| -------------- | --- | --- | --- | ----- | --- | --- | --- | --- |
| 0.4 nn-real-2  |     |     |     | 20000 |     |     |     |     |
nn-real-2 w/o xors
nn-real-100
10000
0.2 nn-real-100 w/o xors
nn-random
| nn-random w/o xors |     |     |     | 0   |     |         |     |     |
| ------------------ | --- | --- | --- | --- | --- | ------- | --- | --- |
|                    |     |     |     | 0   | 1   | 2 3     | 4 5 | 6   |
| 0.0 0 2            | 4   | 6 8 | 10  |     |     | log2(l) |     |     |
log2(l)
|     | (c) |     |     |                                          |     | (d) |     |     |
| --- | --- | --- | --- | ---------------------------------------- | --- | --- | --- | --- |
| 1.0 |     |     |     | esion teknalb htiw ycarucca gniniart 1.0 |     |     |     |     |
rf-random
rf-real
SFC-l htiw ycarucca gniniart
| 0.8 |     |     |     | 0.8 |     |     | nn-real-2 |     |
| --- | --- | --- | --- | --- | --- | --- | --------- | --- |
nn-real-100
nn-random
| 0.6 |     |     |     | 0.6 |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.4 |     |     |     | 0.4 |     |     |     |     |
rf-real
rf-random
| 0.2 |     |     |     | 0.2 |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
nn-real-2
nn-real-100
nn-random
| 0.0                          |         |              |     | 0.0                                      |     |         |              |     |
| ---------------------------- | ------- | ------------ | --- | ---------------------------------------- | --- | ------- | ------------ | --- |
| 0 2                          | 4 6     | 8 10         | 12  | 30 25                                    |     | 20      | 15 10        | 5   |
|                              | log2(l) |              |     |                                          |     | log2(p) |              |     |
|                              | (e)     |              |     |                                          |     | (f)     |              |     |
| 1.0                          |         |              |     | esion teknalb htiw ycarucca gniniart 1.0 |     |         |              |     |
|                              |         | epochs = 1   |     |                                          |     |         | epochs = 1   |     |
| SFC-l htiw ycarucca gniniart |         | epochs = 2   |     |                                          |     |         | epochs = 2   |     |
| 0.8                          |         | epochs = 5   |     | 0.8                                      |     |         | epochs = 5   |     |
|                              |         | epochs = 10  |     |                                          |     |         | epochs = 10  |     |
|                              |         | epochs = 20  |     |                                          |     |         | epochs = 20  |     |
| 0.6                          |         |              |     | 0.6                                      |     |         |              |     |
|                              |         | epochs = 50  |     |                                          |     |         | epochs = 50  |     |
|                              |         | epochs = 100 |     |                                          |     |         | epochs = 100 |     |
|                              |         | epochs = 200 |     |                                          |     |         | epochs = 200 |     |
| 0.4                          |         |              |     | 0.4                                      |     |         |              |     |
| 0.2                          |         |              |     | 0.2                                      |     |         |              |     |
| 0.0                          |         |              |     | 0.0                                      |     |         |              |     |
| 0 2                          | 4 6     | 8 10         | 12  | 30 28                                    | 26  | 24      | 22 20        | 18  |
|                              | log2(l) |              |     |                                          |     | log2(p) |              |     |
|                              | (g)     |              |     |                                          |     | (h)     |              |     |
Figure2: TheresultsofexperimentsinSection3. Plot(a)showstheCFScurvesfor3networkswithdifferentamountsof
overfitting;(b)and(c)showtheimpactofthechoiceofmultipliersandofprimitivelogicgatesrespectivelyonCFScurves;
(d)showshowmanyexamplesareunaffectedbyl-CFSsincetheydonothaveanyl-rarepatterns;(e)showsCFScurvesfor
randomforests;(f)showstrainingaccuracieswhenasignalisrandomlyflippedwithprobabilityp;and(g)and(h)showthe
differencesbetweenCFSandrandomflipsrespectivelyfor8networkstrainedonadatasetwithhighlabelnoise.

Circuit-BasedIntrinsicMethodstoDetectOverfitting
pectedrf-randomgetsnobetterthanchance.(rf-real ingthetrainingsetwhilerandomlyflippingthenodevalues
has about 14K nodes per tree whereas rf-random has withprobabilityp. Aspvariesfrom2−30to2−5,theresult-
about70Knodespertree.) ingnoisecurves(Figure2f)aresimilartotheCFScurves
|     |     |     |     |     |     | (Figure | 2e). However, | the | more overfit | nn-real-100 |     |
| --- | --- | --- | --- | --- | --- | ------- | ------------- | --- | ------------ | ----------- | --- |
Theforestsarecompileddowntocircuitsinastraightfor-
|     |     |     |     |     |     | does not | fall faster than | nn-real-2. |     | With | CFS, these |
| --- | --- | --- | --- | --- | --- | -------- | ---------------- | ---------- | --- | ---- | ---------- |
wardmanner.Eachtreeiscompiledseparatelyandproduces
curvesarewellseparated,andthegapbetweenneuralnets
| 1016-bitoutputs(oneperclass). |     |     | Thecorrespondingoutputs |     |     |     |     |     |     |     |     |
| ----------------------------- | --- | --- | ----------------------- | --- | --- | --- | --- | --- | --- | --- | --- |
andforestsismuchlarger.
areaddedacrossall10treesandtheclassoutputbythefor-
estcorrespondstotheclasswiththemaximumvalue. Each Expt. 7: Sensitivity. Tobetterunderstandthesensitivity
internalnodeinatreemapstoamultiplexercontrolledbya differencebetweenCFSandblanketnoise,wetrained38
8-bitcomparatortoimplementthethreshold,andeachleaf neuralnetworks(withthesamearchitectureasbeforebut
node corresponds to 10 16-bit constants representing the different number of epochs) on MNIST with exactly one
numberofexamplesthatoccupyeachclassinthatleaf(thus halfoflabelsrandomized(somaximumaccuracypossible
mostentriesarezero). Thecircuitforrf-realhasabout is about 55%). We show the CFS curves and the noise
700Knodeswhereasrf-randomhas3Mnodes. Bothare curvesfor8representativenetworksinFigures2gand2h
lessthan250logiclevelsdeep. (Thesearemuchsmaller respectively. Note the crossover of the CFS curves that
thanthecircuitsfortheneuralnetworks.) indicatesalargerfalloffforoverfitnetworkscomparedto
|     |     |     |     |     |     | the more | uniform degradation |     | of the | noise | curves. It is |
| --- | --- | --- | --- | --- | --- | -------- | ------------------- | --- | ------ | ----- | ------------- |
Figure2eshowstheCFScurvesforthetworandomforests.
|            |                                     |           |      |                       |     | fascinating                    | that all the | CFS | curves | cross over        | at a single |
| ---------- | ----------------------------------- | --------- | ---- | --------------------- | --- | ------------------------------ | ------------ | --- | ------ | ----------------- | ----------- |
| Onceagain, | weseethattheoverfitmodel(rf-random) |           |      |                       |     |                                |              |     |        |                   |             |
|            |                                     |           |      |                       |     | pointwithanaccuracyofabout50%. |              |     |        | Thisisdiscussedin |             |
| degrades   | faster than                         | the model | with | better generalization |     |                                |              |     |        |                   |             |
Section4.
(rf-real)confirmingthatCFSiseffectiveevenformod-
elsthatarefundamentallydifferentfromneuralnetworks. Expt. 8: Composite CFS. For completeness, Figure 3
However,itisinterestingtoseethatifwecompareacross shows the results of running Composite CFS instead of
modelfamiliesi.e. betweentheneuralnetworksfromExpt. Simple CFS in the setup of Expt. 7. We see that these
1(repeatedinFigure2eforconvenience)andtherandom curvesareverysimilartothoseofSimpleCFSinFigure2g.
| forests, | CFS is not | effective | at distinguishing |     | overfit. In |     |     |     |     |     |     |
| -------- | ---------- | --------- | ----------------- | --- | ----------- | --- | --- | --- | --- | --- | --- |
particular,nn-real-2whichisnotoverfitdegradesmore
4.Discussion
| rapidlythatrf-randomwhichishighlyoverfit. |     |     |     |     | Wedis- |     |     |     |     |     |     |
| ----------------------------------------- | --- | --- | --- | --- | ------ | --- | --- | --- | --- | --- | --- |
cussthisingreaterdetailinSection4. StructureDependence. Expt. 2and3showthattheresults
ofSimpleCFSdependnotjustonthefunctionbutonthe
| Expt.       | 6: BlanketNoise.                         | CFSmaybeseenasaddingatar- |     |     |     |                        |     |                                 |     |     |     |
| ----------- | ---------------------------------------- | ------------------------- | --- | --- | --- | ---------------------- | --- | ------------------------------- | --- | --- | --- |
|             |                                          |                           |     |     |     | structureofthecircuit. |     | (OtherCFSvariantsweinvestigated |     |     |     |
| getednoise. | Here,instead,weaddblanketnoisebysimulat- |                           |     |     |     |                        |     |                                 |     |     |     |
showthisbehavioraswell.)Asmallexampleprovidessome
insight. ConsidertheBooleanfunctionf(a,b,c)=aeval-
uatedonthetrainingsetcomprisingthefullBooleancube
|     |     |     |     |     |     | (i.e., all    | 8 combinations  | of  | a,b,c ∈ | {0,1}). | In addition |
| --- | --- | --- | --- | --- | --- | ------------- | --------------- | --- | ------- | ------- | ----------- |
|     |     |     |     |     |     | to the direct | implementation, |     | f can   | also be | implemented |
1.0
|     |                              |     |     | epochs = 1 |     | (redundantly)asa·b·c+a·¬b·c+a·b·¬c+a·¬b·¬c.         |     |     |     |     |     |
| --- | ---------------------------- | --- | --- | ---------- | --- | --------------------------------------------------- | --- | --- | --- | --- | --- |
|     | SFC-l htiw ycarucca gniniart |     |     | epochs = 2 |     |                                                     |     |     |     |     |     |
|     | 0.8                          |     |     |            |     | Itiseasytoseethatunder1-CFS,thedirectimplementation |     |     |     |     |     |
epochs = 5
epochs = 10
|     |     |     |     | epochs = 20 |     | is unchanged | (there | are no | 1-rarepatterns) |     | butthe redun- |
| --- | --- | --- | --- | ----------- | --- | ------------ | ------ | ------ | --------------- | --- | ------------- |
0.6
|     |     |     |     | epochs = 50 |     | dantimplementationmapstoconstant0(theoutputofeach |     |     |     |     |     |
| --- | --- | --- | --- | ----------- | --- | ------------------------------------------------- | --- | --- | --- | --- | --- |
epochs = 100
epochs = 200
|     | 0.4 |     |     |     |     | conjunctionis1onlyonce). |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | ------------------------ | --- | --- | --- | --- | --- |
Althoughthisisnotaproblemwhenthecompilationpro-
0.2
cesscanbecontrolled,thisisbadnewsintheadversarial
|     | 0.0 |     |     |      |     | setup. Agoodmodelwithapoorimplementationmayshow |     |     |     |     |     |
| --- | --- | --- | --- | ---- | --- | ----------------------------------------------- | --- | --- | --- | --- | --- |
|     | 0 2 | 4   | 6   | 8 10 | 12  |                                                 |     |     |     |     |     |
steeperdegradationunderCFSthanamoreoverfitmodel
log2(l)
|     |     |     |     |     |     | with better | implementation |     | (c.f. nn-real-2 |     | with array |
| --- | --- | --- | --- | --- | --- | ----------- | -------------- | --- | --------------- | --- | ---------- |
Figure 3: This plot shows the result of repeating Expt. 7 v/snn-real-100withoutXorsatl =256). Ideally,we
wouldliketofindavariantofCFSthatdoesnotdependon
butusingCompositeCFS(insteadofSimpleCFS)where
wecomputeandperturbrarepatternsacrossallthesignals structurebutonlyonthefunction.3 Intheabsenceofthat
thatfeedintoalogicgate,i.e.,acrossallthefaninsofagate 3Inprinciple,onecouldcanonizethecircuitstructurebefore
(insteadofcomputingandperturbingrarepatternsforeach
applyingCFS,saybybuildingReducedOrderBinaryDecisionDi-
| signalorfaninindependently). |     |     | TheresultswithComposite |     |     |     |     |     |     |     |     |
| ---------------------------- | --- | --- | ----------------------- | --- | --- | --- | --- | --- | --- | --- | --- |
agrams(ROBDDs),butthatwouldbecomputationallyprohibitive.
CFSareverysimilartothoseofSimpleCFS(Figure2g). Alternatively,onecouldlightlyoptimizethecircuitbeforeCFS

Circuit-BasedIntrinsicMethodstoDetectOverfitting
|     | 1.0 |     |     |     |     |     | 1.0 |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
epochs = 1
SFC-l htiw ycarucca gniniart SFC-l htiw ycarucca gniniart epochs = 2
|     | 0.8 |     |     |     |     |     | 0.8 |     |     |     | epochs = 5 |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---------- | --- |
epochs = 10
epochs = 20
|     | 0.6            |     |     |     |     |     | 0.6 |     |     |     | epochs = 50  |     |
| --- | -------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ------------ | --- |
|     | epochs = 1     |     |     |     |     |     |     |     |     |     | epochs = 100 |     |
|     | epochs = 2     |     |     |     |     |     |     |     |     |     | epochs = 200 |     |
|     | 0.4 epochs = 5 |     |     |     |     |     | 0.4 |     |     |     |              |     |
epochs = 10
epochs = 20
|     | 0.2 epochs = 50 |     |     |     |     |     | 0.2 |     |     |     |     |     |
| --- | --------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
epochs = 100
epochs = 200
|     | 0.0 |     |         |     |       |     | 0.0 |     |     |         |      |     |
| --- | --- | --- | ------- | --- | ----- | --- | --- | --- | --- | ------- | ---- | --- |
|     | 0 2 | 4   | 6       | 8   | 10 12 |     | 0   | 2   | 4   | 6       | 8 10 | 12  |
|     |     |     | log2(l) |     |       |     |     |     |     | log2(l) |      |     |
|     |     |     | (a)     |     |       |     |     |     |     | (b)     |      |     |
Figure4: TheseplotsshowtheresultofrepeatingExpt. 7whenwecorruptonlyathirdofthelabelsin(a)MNISTandin(b)
Fashion-MNIST.Inbothcases,justasinFigure2gweseethatallthecurvespassthroughapointclosetothemaximum
achievableaccuracy(around2/3). ThereasonforthisisdiscussedinSection4underGeneralizationinDeepLearning.
ideal,weviewthestructureofthecircuitasacertificateof thoughunbalancedhavenorarepatterns.
howwellthedatasetislearnt,andmakeitMerlin’srespon-
ThisexamplesuggestsawaytobreakCFS.IfMerlinwanted
| sibility | to find and present | the | most convincing |     | structure. |     |     |     |     |     |     |     |
| -------- | ------------------- | --- | --------------- | --- | ---------- | --- | --- | --- | --- | --- | --- | --- |
tobuildalookuptable,butpreventArthurfromidentifying
Fromthisperspective,intheaboveexample,thedirectim-
|     |     |     |     |     |     | it  | as such | (by applying |     | say 1-CFS), | instead | of building |
| --- | --- | --- | --- | --- | --- | --- | ------- | ------------ | --- | ----------- | ------- | ----------- |
plementationoff
(whichisnotimpactedby1-CFS)ismore
thecircuitinFigure1,Merlincouldbuildadecision(i.e.,
| convincingthantheredundantimplementationoff |     |     |     |     | (which |     |     |     |     |     |     |     |
| ------------------------------------------- | --- | --- | --- | --- | ------ | --- | --- | --- | --- | --- | --- | --- |
multiplexer)treebyrecursivelysplittingononevariable(i.e.,
isseverelyimpacted).
onecomponentofx)atatimebypickingacomponentthatis
Adversarial Attack on CFS. Based on the discussion balanced(i.e.,takeson0and1valuesroughlyequallyoften)
aboveitiseasytodesignawaytoarbitrarilydegradethe andleadstobothbranchesalsobeingbalanced(i.e.,have
performance of a circuit under CFS. But is the opposite noclasswitheithertoofewortoomanyexamples,although
possible? CanMerlinfitanarbitraryfunctionontheinputs havingnoexamplesofaclassorhavingonlyexamplesof
but compile it down to a circuit which does not degrade oneclassisfineandexpectedattheleafnodes). Inother
under CFS? Expt. 5. offers a clue. The overfit model words, Merlin could use a procedure similar to the usual
rf-randomfittedonrandomlabelsfallsoffmoreslowly decisiontreeconstructionprocedures,butwiththeimportant
than nn-real-2 which generalizes well. What is go- differenceoffavoringbalancedsplitsinsteadofunbalanced
ing on? The short answer is that although each tree in splits.(TheunbalancedsplitsarelikelywhyCFSiseffective
rf-randomisextremelyoverfitwithmostleavescontain- totheextentitisonrandomforests.)
ingonlyasingleexample,thecircuitnodeshavefewrare
Finally,assuggestedbyareviewerofthispaper,itwouldbe
patternsduetotheobservabilitydon’tcaresintroducedby
interestingtoextendthisidea(ultimatelybasedonShannon
themultiplexers.
decompositionandco-factoring)toanalgorithmthatcan
Againasimpleexampleisinstructive. Letf betheparity amplifythecountofrarepatternsinagivencircuitwithout
functiononnbits,i.e.,f(x 1 ,x 2 ,...,x n )=x 1 ⊕x 2 ...⊕ changingitsfunctionality.
| x . Consider | an tree       | implementation |     | of this | function ob- |                |     |     |               |     |         |            |
| ------------ | ------------- | -------------- | --- | ------- | ------------ | -------------- | --- | --- | ------------- | --- | ------- | ---------- |
| n            |               |                |     |         |              | Comparisonwith |     |     | BlanketNoise. |     | Basedon | Expt. 6and |
| tained       | using Shannon | decomposition  |     | which   | has a multi- |                |     |     |               |     |         |            |
7,webelievethatblanketnoiseislesssensitivethanCFS.
| plexeratthetopcontrolledbyx |          |                  | 1 andwith1⊕x |     | 2 ⊕x 3 ...⊕   |     |         |      |        |               |            |           |
| --------------------------- | -------- | ---------------- | ------------ | --- | ------------- | --- | ------- | ---- | ------ | ------------- | ---------- | --------- |
|                             |          |                  |              |     |               | Our | results | here | add to | the extensive | literature | on noise, |
| x andx                      | ⊕x ...⊕x | asitsdatainputs. |              |     | Ifthetraining |     |         |      |        |               |            |           |
n 2 3 n generalizationandfaulttoleranceinneuralnetworks(e.g.,
{0,1}n
| set is the | full Boolean | cube, | i.e., | it  | is easy to see |     |     |     |     |     |     |     |
| ---------- | ------------ | ----- | ----- | --- | -------------- | --- | --- | --- | --- | --- | --- | --- |
seeBernieretal.(2001)andthereferencestherein)byex-
thattherearenol-raresignalsforl<<nsinceeachinput
tendingthemtothecircuitlevel(wheredistinctionsbetween
tothemultiplexerisbalanced,i.e.,hasequalnumberof0s
activationorweightnoise,oradditiveormultiplicativenoise
and1s. SinceShannondecompositionisrecursive,asimilar
|     |     |     |     |     |     | disappear) |     | and to | other | model | families such | as random |
| --- | --- | --- | --- | --- | --- | ---------- | --- | ------ | ----- | ----- | ------------- | --------- |
argumentholdsforthelowerlevelsofthetreeuntilweget
|        |                  |          |       |          |         | forests. | Furthermore, |     | Expt. | 6   | presents a | direct compari- |
| ------ | ---------------- | -------- | ----- | -------- | ------- | -------- | ------------ | --- | ----- | --- | ---------- | --------------- |
| to the | leaves which are | constant | 0 and | constant | 1 which |          |              |     |       |     |            |                 |
sonofthefault-toleranceofneuralnetsandforestswhere
butthatmaynotbeenough. forestsareseentobeabout1000xmorefault-toleranttobit

Circuit-BasedIntrinsicMethodstoDetectOverfitting
flips. Thisislikelyduetotheredundancyfromensembling. obviouslyapply. Wehavenotstudiedifnormalizedmargin
Italsosuggeststhatnoise-basedintrinsicmethodscouldbe can be exploited by an adversary. Similarly, most mea-
easilyfooledbyanadversarybyaddingredundancy. suresbasedontheshallownessofminimaarenotadequate
sincetheyarenotscale-invariant(Dinhetal.,2017)andwe
| Generalization | in Deep | Learning. | Why | do  | neural nets |     |     |     |     |     |     |     |
| -------------- | ------- | --------- | --- | --- | ----------- | --- | --- | --- | --- | --- | --- | --- |
havenotinvestigatedifmorerecentworkonscale-invariant
trainedwithSGDgeneralizewhentheyhavesufficient(ef-
|                   |             |     |                |      |         | measures   | (Rangamani |           | et al., 2019) | can        | be exploited. | In   |
| ----------------- | ----------- | --- | -------------- | ---- | ------- | ---------- | ---------- | --------- | ------------- | ---------- | ------------- | ---- |
| fective) capacity | to memorize |     | their training | set? | This is |            |            |           |               |            |               |      |
|                   |             |     |                |      |         | comparison | to         | these and | other         | approaches | for neural    | net- |
anopenresearchquestion(Zhangetal.,2017;Arpitetal.,
works(Aroraetal.,2018;Neyshaburetal.,2018),CFSis
| 2017; Bartlett | et al., 2017; | Arora | et al., | 2018; | Neyshabur |     |     |     |     |     |     |     |
| -------------- | ------------- | ----- | ------- | ----- | --------- | --- | --- | --- | --- | --- | --- | --- |
fundamentallymorediscrete,whichmakesitapplicableto
| et al., 2018). | Expt 5. shows | that | this question |     | is not lim- |                       |     |     |                              |     |     |     |
| -------------- | ------------- | ---- | ------------- | --- | ----------- | --------------------- | --- | --- | ---------------------------- | --- | --- | --- |
|                |               |      |               |     |             | alargerclassofmodels. |     |     | However,incontrasttotheother |     |     |     |
itedtonets—thesamecouldbeaskedforrandomforests
approaches,wedonothaveanytheoreticalboundsyetwhile
aswell. One(informal)answerforforestsisthatdecision
ourresultsindicatethatwithoutfurtherrefinementstoCFS
treeconstructionprocedureslookforcommonpatternsbe-
itself,anygeneralizationboundsfroml-CFSwouldlikely
| tweenexamples. | Whenexamplessharecommonality,they |     |     |     |     |     |     |     |     |     |     |     |
| -------------- | --------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
bevacuousinpractice.
arecombinedintocommonleafnodesandthemodelgen-
| eralizes, | whereas when | there | is little | commonality, | each |     |     |     |     |     |     |     |
| --------- | ------------ | ----- | --------- | ------------ | ---- | --- | --- | --- | --- | --- | --- | --- |
5.ConclusionandFutureWork
exampleisitsownleaf(sothetrainingsetisfitwell)butthe
| modelfailstogeneralize. |     | Couldthesamethingbegoingon |     |     |     |     |     |     |     |     |     |     |
| ----------------------- | --- | -------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
OurmainresultisthatCFSbasedonaddingsmallamounts
| withSGDandnetworks? |     | CFSonnn-randomandExpt4. |     |     |     |     |     |     |     |     |     |     |
| ------------------- | --- | ----------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
oftargetednoiseatthelogiccircuitlevelcandetectoverfit.
providedirectevidencethatevenonrandomdata,netsdo
|     |     |     |     |     |     | This is | remarkable | because | at  | this level | of representation |     |
| --- | --- | --- | --- | --- | --- | ------- | ---------- | ------- | --- | ---------- | ----------------- | --- |
not“brute-forcememorize”butidentifycommonpatterns
|             |                    |        |          |        |            | we have | lost most       | aspects | of      | the structure | of the           | model, |
| ----------- | ------------------ | ------ | -------- | ------ | ---------- | ------- | --------------- | ------- | ------- | ------------- | ---------------- | ------ |
| in the data | (a question        | raised | in Zhang | et al. | (2017) and |         |                 |         |         |               |                  |        |
|             |                    |        |          |        |            | such as | the distinction |         | between | weights       | and activations, |        |
| discussedin | Arpitetal.(2017)). |        |          |        |            |         |                 |         |         |               |                  |        |
orevenwhetherthemodelisaneuralnetwork,arandom
Inthiscontext,itisinterestingtostudywhytheCFScurves forest,oralookuptable. Furthermore,variationssuchas
forthedifferentnetworksinExpt7. intersectatacommon perturbingonlyrarepatternsinsinglesignalsoracrossthe
pointcorrespondingapproximatelytotheachievableaccu- faninsofagateleadtoqualitativelysimilarresultsandthere
racy. Thisholdsforotheramountsoflabelrandomization arevariants(suchasSimpleCFS)thatarenaturallyfreeof
| e.g.,seeFigure4fortheCFScurveswhenonethirdofthe |                                 |     |     |     |     | hyper-parameters. |     |     |     |     |     |     |
| ----------------------------------------------- | ------------------------------- | --- | --- | --- | --- | ----------------- | --- | --- | --- | --- | --- | --- |
| labelsarecorrupted.                             | Roughlyhalfoftheexamplesareeasy |     |     |     |     |                   |     |     |     |     |     |     |
Bystudyingrarepatterns,wefindthatSGDdoesnotlead
sincetheyhavecorrectlabelsandarelearntinthefirstfew
to“bruteforce”memorization,butfindscommonpatterns
| epochs. | The remaining | examples | with | corrupt | labels are |     |     |     |     |     |     |     |
| ------- | ------------- | -------- | ---- | ------- | ---------- | --- | --- | --- | --- | --- | --- | --- |
(whethertrainingisdonewithrandomizedlabelsoractual
| harder and | learnt only | in later | epochs | by the models | that |     |     |     |     |     |     |     |
| ---------- | ----------- | -------- | ------ | ------------- | ---- | --- | --- | --- | --- | --- | --- | --- |
labels), andneuralnetworksarenotunlikeforestsinthis
| aretrainedlonger. | WithCFS,theaccuracyofthosemodels |     |     |     |     |     |     |     |     |     |     |     |
| ----------------- | -------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
regard.Byaddingblanketnoise,randomforestsarefoundto
breaksdownearliersincethehardexampleshavemorerare
beabout1000xmoreresilienttonoisethanneuralnetworks
| patterns | than the easy | examples, | and the | accuracy | on the |     |     |     |     |     |     |     |
| -------- | ------------- | --------- | ------- | -------- | ------ | --- | --- | --- | --- | --- | --- | --- |
whichcouldbeusefulwhenimplementingmachinelearning
easyexamplesthusformsalimitingcurveforallmodels.
systemswithunreliablelowlevelcomponents.
Thisprovidesmore(anddirect)evidencefortheclaimin
Arpit et al. (2017, §1) that “SGD learns simpler patterns Thereareseveraldirectionsforfuturework. Weanalyzeflat
firstbeforememorizing.” Furthermore,wecouldidentify circuits, but with a clever implementation that constructs
“simplerpatterns”asexamplesthathavefewerrarepatterns thecircuiton-the-flyfromahigherlevelspecification,the
| and “memorizing” | as what | is  | required | for examples | that |              |     |       |           |         |          |      |
| ---------------- | ------- | --- | -------- | ------------ | ---- | ------------ | --- | ----- | --------- | ------- | -------- | ---- |
|                  |         |     |          |              |      | computations | can | scale | to larger | models. | We could | also |
havemorerarepatterns. Thus,learningsimplerpatternsand applyCFSathigherlevelsofabstraction(perhapsaspartof
memorizationarenotfundamentallydifferentbutlieattwo themodelevaluationprocessinframeworks,suchasScikit
endsofaspectrum. LearnandTensorflowEstimators)thoughatthatlevelthere
|              |                                 |     |     |     |     | are moredegrees |     | of freedomin |     | theimplementation |     | (e.g., |
| ------------ | ------------------------------- | --- | --- | --- | --- | --------------- | --- | ------------ | --- | ----------------- | --- | ------ |
| RelatedWork. | Onemaybetemptedtoviewmarginasan |     |     |     |     |                 |     |              |     |                   |     |        |
whatkindofnoisetoadd).
intrinsicmeasuretoestimatethegeneralizationofamodel.
However, when we have models with intermediate repre- Thenotionofrarityconsideredinthisworkmayberegarded
sentations, the notion of margin by itself is not adequate as a local notion, since we count the patterns at a signal
sinceanadversarycanoverfittoafavorableintermediate (or a group of signals in the case of Composite CFS) in
representationthatiseasilylinearlyseparated(butotherwise isolation. It is possible to extend this notion to a global
arbitrary). However,recentworkinthisarea(e.g.,Bartlett notion of rarity by propagating occurrence counts along
etal.(2017))hasfocusedonmarginsnormalizedbyspectral withsignalvaluesduringsimulation. Inthecorresponding
complexity(i.e.,ameasurerelatedtotheLipschitzconstant CFS, a pattern is perturbed when its global rarity drops
ofthenetwork)andinthatcasetheaboveargumentdoesnot

Circuit-BasedIntrinsicMethodstoDetectOverfitting
| belowthegiventhreshold. |                | Preliminaryexperimentsshow |              |                | References                              |     |     |     |     |     |          |     |
| ----------------------- | -------------- | -------------------------- | ------------ | -------------- | --------------------------------------- | --- | --- | --- | --- | --- | -------- | --- |
| that this               | more stringent | notion                     | of rarity is | more effective |                                         |     |     |     |     |     |          |     |
|                         |                |                            |              |                | Arora,S.,Ge,R.,Neyshabur,B.,andZhang,Y. |     |     |     |     |     | Stronger |     |
indetectingoverfitinthecaseofrandomforests,andmay
|     |     |     |     |     | generalization |     | bounds | for deep | nets | via | a compression |     |
| --- | --- | --- | --- | --- | -------------- | --- | ------ | -------- | ---- | --- | ------------- | --- |
offerawaytopartiallymitigatetheadversarialattackon
|     |     |     |     |     | approach. | CoRR, | abs/1802.05296, |     |     | 2018. | URL | http: |
| --- | --- | --- | --- | --- | --------- | ----- | --------------- | --- | --- | ----- | --- | ----- |
CFSoutlinedintheprevioussection.
//arxiv.org/abs/1802.05296.
Finally,basedoninsightsfromouranalysisofSimpleCFS,
wewouldliketocontinuethesearchforanintrinsicmethod Arpit,D.,Jastrzebski,S.K.,Ballas,N.,Krueger,D.,Bengio,
|     |     |     |     |     | E., Kanwal, |     | M. S., | Maharaj, | T., Fischer, |     | A., Courville, |     |
| --- | --- | --- | --- | --- | ----------- | --- | ------ | -------- | ------------ | --- | -------------- | --- |
thatdoesnotdependonthemodelstructureandisadversar-
|     |     |     |     |     | A.C.,Bengio,Y.,andLacoste-Julien,S. |     |     |     |     |     | Acloserlook |     |
| --- | --- | --- | --- | --- | ----------------------------------- | --- | --- | --- | --- | --- | ----------- | --- |
iallyrobust,ortoshowthatsuchamethoddoesnotexist,
|                                          |     |     |     |             | at memorization |     | in  | deep networks. |     | In Proceedings |     | of  |
| ---------------------------------------- | --- | --- | --- | ----------- | --------------- | --- | --- | -------------- | --- | -------------- | --- | --- |
| evenforlearningtasksofpracticalinterest. |     |     |     | Asageneral- |                 |     |     |                |     |                |     |     |
the34thInternationalConferenceonMachineLearning,
izationofthisidea,itisinterestingtocontemplatelearning
ICML2017,Sydney,NSW,Australia,6-11August2017,
algorithmsthatproducecertificatesofgeneralization,much
|     |     |     |     |     | pp. 233–242, |     | 2017. | URL | http://proceedings. |     |     |     |
| --- | --- | --- | --- | --- | ------------ | --- | ----- | --- | ------------------- | --- | --- | --- |
likeaBooleansatisfiabilitysolvercanproduceacertificate
mlr.press/v70/arpit17a.html.
ofsatisfiabilityorofunsatisfiability.
|             |                               |     |     |        | Bartlett,P.L.,Foster,D.J.,andTelgarsky,M.J. |     |     |     |     |     | Spectrally- |     |
| ----------- | ----------------------------- | --- | --- | ------ | ------------------------------------------- | --- | --- | --- | --- | --- | ----------- | --- |
| Postscript: | TheTheoryofCoherentGradients. |     |     | Theob- |                                             |     |     |     |     |     |             |     |
servations in thispaper from applyingCFS toneural net- normalizedmarginboundsforneuralnetworks.InGuyon,
works—particularly,theabsenceof“bruteforce”memo- I.,Luxburg,U.V.,Bengio,S.,Wallach,H.,Fergus,R.,
|     |     |     |     |     | Vishwanathan, |     | S., and | Garnett, | R.  | (eds.), | Advances | in  |
| --- | --- | --- | --- | --- | ------------- | --- | ------- | -------- | --- | ------- | -------- | --- |
rizationwithrandomlabels,andtheexistenceofeasyand
hardexamplesasdiscussedinSection4—inspiredthede- Neural Information Processing Systems 30, pp. 6240–
6249.CurranAssociates,Inc.,2017.
velopmentofatheorycalledCoherentGradients(CG)that
providesasimpleandintuitiveexplanationofgeneralization
|     |     |     |     |     | Bernier, | J., Ortega, | J., | Ros Vidal, | E., | Rojas, | I., and | Pri- |
| --- | --- | --- | --- | --- | -------- | ----------- | --- | ---------- | --- | ------ | ------- | ---- |
innetworkstrainedwith(stochastic)gradientdescent.
|     |     |     |     |     | eto, A. | A   | quantitative | study | of fault | tolerance, |     | noise |
| --- | --- | --- | --- | --- | ------- | --- | ------------ | ----- | -------- | ---------- | --- | ----- |
ThekeyideainCGisasfollows. Sincetheoverallgradient immunity, and generalization ability of mlps. Neural
g istheaverageofthegradientsoftheindividualtraining Computation, 12:2941–2964, 01 2001. doi: 10.1162/
examples,theremaybecertaincomponents(directions)ofg 089976600300014782.
whicharesignificantlystrongerthanothercomponentsdue
|     |     |     |     |     | Biere, A. | Aigerlibraryandtools, |     |     | 2007. | URLhttp:// |     |     |
| --- | --- | --- | --- | --- | --------- | --------------------- | --- | --- | ----- | ---------- | --- | --- |
tothegradientsofmultipleexamplesbeinginagreement
|                    |            |                                 |            |             | fmv.jku.at/aiger. |     |     | [Online;accessed1-July-2020]. |     |            |     |      |
| ------------------ | ---------- | ------------------------------- | ---------- | ----------- | ----------------- | --- | --- | ----------------------------- | --- | ---------- | --- | ---- |
| onthosecomponents, |            | andtherebyreinforcingeachother. |            |             |                   |     |     |                               |     |            |     |      |
| Since the          | changes to | the trainable                   | parameters | of the net- |                   |     |     |                               |     |            |     |      |
|                    |            |                                 |            |             | Chatterjee,       | S.  | On  | Algorithms                    | for | Technology |     | Map- |
workareproportionaltothegradient,thebiggestchangesto
|     |     |     |     |     | ping. | PhD | thesis, | EECS | Department, |     | Univer- |     |
| --- | --- | --- | --- | --- | ----- | --- | ------- | ---- | ----------- | --- | ------- | --- |
theparametersarebiasedtobenefitmultipleexamples,and
|     |     |     |     |     | sity | of California, |     | Berkeley, | Aug | 2007. |     | URL |
| --- | --- | --- | --- | --- | ---- | -------------- | --- | --------- | --- | ----- | --- | --- |
thereforelikelytogeneralizewell(basedonastabilityargu- http://www2.eecs.berkeley.edu/Pubs/
ment).However,iftherearenosuchstrongcomponents,i.e.,
TechRpts/2007/EECS-2007-100.html.
alltheper-examplegradientsareroughlyorthogonal,then
eachexampleisfittedindependently,andthiscorresponds Chatterjee,S. Coherentgradients: Anapproachtounder-
tomemorization(andpoorgeneralization,againfromasta- standing generalization in gradient descent-based opti-
InProceedingsoftheInternationalConference
| bilityargument). | ThusCGprovidesanuniformexplanation |     |     |     | mization. |     |     |     |     |     |     |     |
| ---------------- | ---------------------------------- | --- | --- | --- | --------- | --- | --- | --- | --- | --- | --- | --- |
ofbothgeneralizationandmemorizationinneuralnetworks. onLearningRepresentationsICLR,2020. URLhttps:
//openreview.net/forum?id=ryeFY0EFwS.
PleaseseeChatterjee(2020)andZielinskietal.(2020)for
amoredetaileddevelopmentofthisidea,includingananal- Dietterich, T. G. Approximate statistical tests for
ysisofeasyandhardexamplesfromthisperspective,anda
|     |     |     |     |     | comparing |     | supervised | classification |     | learning |     | algo- |
| --- | --- | --- | --- | --- | --------- | --- | ---------- | -------------- | --- | -------- | --- | ----- |
naturalmodificationtogradientdescentthatsignificantlyre- rithms. Neural Comput., 10(7):1895–1923, Oc-
ducesoverfittingbysuppressingthosegradientcomponents tober 1998. ISSN 0899-7667. doi: 10.1162/
thatarenotcommontomanyexamples.
|     |     |     |     |     | 089976698300017197. |     |     | URLhttp://dx.doi.org/ |     |     |     |     |
| --- | --- | --- | --- | --- | ------------------- | --- | --- | --------------------- | --- | --- | --- | --- |
10.1162/089976698300017197.
Acknowledgments
|     |     |     |     |     | Dinh, L., | Pascanu, |     | R., Bengio, |     | S., and | Bengio, | Y.  |
| --- | --- | --- | --- | --- | --------- | -------- | --- | ----------- | --- | ------- | ------- | --- |
ThefirstauthorthanksMicheleCovell,AliRahimi,Alex Sharp minima can generalize for deep nets. CoRR,
Alemi,ShumeetBaluja,SergeyIoffe,TomasIzo,Shankar abs/1703.04933, 2017. URL http://arxiv.org/
abs/1703.04933.
| Krishnan,    | Rahul Sukthankar, | and    | Jay Yagnik    | for helpful |     |     |     |     |     |     |     |     |
| ------------ | ----------------- | ------ | ------------- | ----------- | --- | --- | --- | --- | --- | --- | --- | --- |
| discussions. | The second        | author | was supported | in part by  |     |     |     |     |     |     |     |     |
SRCContract2867.001.

Circuit-BasedIntrinsicMethodstoDetectOverfitting
Dwork, C., Feldman, V., Hardt, M., Pitassi, T., Reingold, Rangamani,A.,Nguyen,N.H.,Kumar,A.,Phan,D.,Chin,
O.,andRoth,A. Thereusableholdout: Preservingvalid- S.H.,andTran,T.D. AScaleInvariantFlatnessMea-
ity in adaptive data analysis. Science, 349(6248):636– sure for Deep Network Minima. arXiv e-prints, art.
638, 2015. ISSN 0036-8075. doi: 10.1126/science. arXiv:1902.02434,Feb2019.
| aaa9375. | URL https://science.sciencemag. |     |     |                                 |     |                |        |
| -------- | ------------------------------- | --- | --- | ------------------------------- | --- | -------------- | ------ |
|          |                                 |     |     | Xiao,H.,Rasul,K.,andVollgraf,R. |     | Fashion-mnist: | anovel |
org/content/349/6248/636.
imagedatasetforbenchmarkingmachinelearningalgo-
LeCun,Y.andCortes,C.MNISThandwrittendigitdatabase.
|     |     |     |     | rithms. arXiv, | 2017. URL | https://arxiv.org/ |     |
| --- | --- | --- | --- | -------------- | --------- | ------------------ | --- |
http://yann.lecun.com/exdb/mnist/, 2010. URL http: abs/1708.07747.
//yann.lecun.com/exdb/mnist/.
Zhang,C.,Bengio,S.,Hardt,M.,Recht,B.,andVinyals,O.
| Neyshabur, | B., Li, Z., | Bhojanapalli, | S., LeCun, | Y., and |     |     |     |
| ---------- | ----------- | ------------- | ---------- | ------- | --- | --- | --- |
Understandingdeeplearningrequiresrethinkinggeneral-
| Srebro, | N. Towards | understanding | the role | of over- |     |     |     |
| ------- | ---------- | ------------- | -------- | -------- | --- | --- | --- |
ization. InProceedingsoftheInternationalConference
parametrization in generalization of neural networks. onLearningRepresentationsICLR,2017.
| CoRR,abs/1805.12076,2018. |     | URLhttp://arxiv. |     |     |     |     |     |
| ------------------------- | --- | ---------------- | --- | --- | --- | --- | --- |
org/abs/1805.12076. Zielinski, P., Krishnan, S., and Chatterjee, S. Weak and
stronggradientdirections:Explainingmemorization,gen-
| Pedregosa, | F., Varoquaux, | G., Gramfort, | A., Michel,       | V.,          |              |             |                  |
| ---------- | -------------- | ------------- | ----------------- | ------------ | ------------ | ----------- | ---------------- |
|            |                |               |                   | eralization, | and hardness | of examples | at scale. ArXiv, |
| Thirion,   | B., Grisel,    | O., Blondel,  | M., Prettenhofer, | P.,          |              |             |                  |
abs/2003.07422,2020. URLhttps://arxiv.org/
Weiss,R.,Dubourg,V.,Vanderplas,J.,Passos,A.,Cour-
abs/2003.07422.
napeau,D.,Brucher,M.,Perrot,M.,andDuchesnay,E.
| Scikit-learn: | Machine | learning in | Python. Journal | of  |     |     |     |
| ------------- | ------- | ----------- | --------------- | --- | --- | --- | --- |
MachineLearningResearch,12:2825–2830,2011.

---

## Content from Previous Extraction (not in markitdown output)

### Visual/Chart/Graph Descriptions

#### 1

Circuit-Based Intrinsic Methodsto Detect Overfitting
couldviewthemodelasacertificateofgeneralization.1 In
addition, iftheintrinsicmethodwasefficient, itwouldmean
thatsupervisedlearning (andnotjustfittingthetrainingdata)
isin NP.Intrinsicmethodscanalsohelpshedlightonwhy
neural networks trained with stochastic gradient descent x x x
n -1 1 0
generalizeinspiteoftheirlargecapacity. Somerecentanal-
s s
ysesbasedonnormalizedmargin, curvature, etc.(Bartlett n -1 0
et al., 2017; Rangamani et al., 2019; Arora et al., 2018;
Neyshaburetal.,2018) maybeseenasintrinsicestimates y
y y y forgeneralizationalbeitspecializedtoneuralnetworks.
n -1 1 0
Intrinsicmethodscanbeconsideredinthecontextofapro-
tocolinvolvingtwoagents. Let S beapublicdatasetdrawn
fromadistribution D thatgeneratessamplesinfrequently
(e.g., quarterlyfinancialstatementsandstockmarketreturns
ofpubliccompanies, orpublichealthdataontreatmentsand
patientoutcomes). Suppose Arthurwantstobuildamodel
from S butinsteadofdoingsohimself, heoutsourcesitto
Merlin, anuntrustedadversary. Merlincomesbackwitha
model Mbutdoesnotdiscloseanydetailsofhismodeling
process. Howcan Arthurconvincehimselfthat Misnot
horriblyoverfit? Forexample, Mcouldsimplybealookup
table built from S. Normally Arthur would evaluate M
onnewsamplesfrom D, butinoursetup, Arthurdoesnot
haveanysamplesotherthanthosein S sincealltheexisting
dataispublic. Now, if Arthuronlyhasaccessto Masa
blackboxandhecanonlyevaluate Monelementsof S, it
appearsthereislittlehecandotodistinguishagoodmodel
from a lookup table. But can Arthur do better if he has
accesstotheinternalsignalsintheimplementationof M?
Thisisthecentralquestionofthispaper.
Wetakeafirststeptowardsansweringthisquestionbystudy-
inganaturally-motivatedfamilyofintrinsicmethods, called
Counterfactual Simulation (CFS), andevaluatingtheiref-
ficacyexperimentallyonabenchmarkproblem. Themain
ideabehind CFSistoanalyzetheflowofthetrainingexam-
plesin Sthroughthestructureof M.Thisisonlyafirststep
since CFS, althoughpromisinginpractice, hassignificant
limitations. Inparticular, ourexperimentsshowthateven
ifwecouldproveboundsbasedon CFS, theywouldnotbe
tightenoughinanadversarialsetting. However, wehope
thatthispaperencouragesresearchtoovercometheselimi-
tationsortoshowthatnosuchmethodcanexist, especially
forlearningtasksofpracticalinterest.
2.Counterfactual Simulation (CFS)
Thestructureof Mcanbedescribedatdifferentlevelsof
abstraction. For instance, if M is a fully connected feed
forwardneuralnetwork, wecandescribeitasasequence
of layers. Going one level lower, we can describe it as
1 Keepingaholdoutsetwouldnothelpushere—whatwould
thatevenmean?
...
x
= = =
s
1
1000000000 0 0 0
1 1 1
Figure1: Acircuitimplementingalookuptablethatmem-
orizes the training examples (x , y ) for 0 ≤ i < n. We i i
observe that in this extreme case of overfitting, there are
signalsinthecircuit (thes ) thatidentifyspecifictraining i
examples. For e.g., s is 1 only when x = x and 0 for 0 0
all other x (assuming the x are distinct). We say that 1 i i
is a rare pattern for the signal s . Based on this exam- 0
ple, weproposethattheoccurrenceofrarepatternsduring
simulationofatrainingsetthroughamodelindicatesover-
fitting, andinthiswork, weexploretowhatextentsuchrare
patternscanbeusedtodetectoverfittinginmorecomplex
modelssuchasneuralnetworksandrandomforests.
a directed acyclic graph (DAG) of fixed or floating point
adders, multipliers, andpointwisenon-linearities. Finally, at
thelowestlevelofabstraction, wecandescribethestructure
of Masa DAGofprimitivelogicgatessuchas2-input And
gates and inverters, i.e., as a combinational logic circuit.
Inoursetup, itisnaturaltoworkatthislowestlevel, i.e.,
logicgatessinceitallowsdifferentkindsofmodelssuch
aslookuptables, randomforests, andneuralnetworksall
tobemappedintothesameformat. Thus, Merlinneednot
discloseevenwhattypeofmodelhehasbuilt, butsimply
provides Arthurwithacombinationallogiccircuitforthe
model.
Tomakethisconcrete, considerthe MNISTimageclassifi-
cationproblem (Le Cun&Cortes,2010) whichwewilluse
asarunningexample. Disthedistributionofhandwritten
digits and their classes. S is a sample from D of 60,000
imagesx andtheircorrespondinglabelsy (thus,0≤i< i i
60000) i.e., the MNISTtrainingset. Eachx is6,272 bits
i
wide (correspondingto28×28 pixels×8 bitsperpixel),
and each y is 10 bits wide (for a 1-hot representation of
i
the 10 possible classes). Therefore, a classifier to solve
thisproblemisacircuitwith6,272 Booleaninputsand10
Booleanoutputs.
Suppose Merlin’smodelforthe MNISTclassifierisasimple
lookuptable. Howwouldthecircuitforitlook? Figure1
sketchesonepossibility. The6,272-bitinputxiscompared
witheachoftheexamplesx inturnandifthereisamatch,
i

#### 2

Circuit-Based Intrinsic Methodsto Detect Overfitting
thecorresponding10-bitoutputy isselected. Ifnoexample Other Typesof CFS.Thereareothervariantsofthepro-
i
matches, thenthemodel (arbitrarily) returnsthe1-hotvector ceduredescribedabove (whichwecall Simple CFSorjust
representingclass‘0’. Now, ifwesimulatethiscircuiton CFS).Ofparticularinterestis Composite CFSwhichisuse-
examplesin S, wenoticethatthereareinternalsignalsin fulforcircuitswithgatesthathavemanyinputsorareat
thecircuitthatarecapableofidentifyingspecifictraining higherlevelsofabstraction. In Composite CFS, welookat
examples. For example, the signal s (the output of the rarepatternsincombinationsofsignalsfeedingaparticular
0
x=x block) is1(true) forthetrainingexamplex and0 gateandperturbtheoutputofthatgate (byflippingit) when
0 0
(false) forallothers. Inthiscase, wesay1 isararepattern ararecombinationisseenattheinputs. Anotherpossibility
fors sinces rarelytakesonthevalue1 onthetrainingset. istorandomizetheperturbationinsteadofalwaysflipping.
0 0
Formally, ifasignalsin Mtakesonthevaluevatmostl Inourexperiments, wefoundthesevariantstoproducere-
timesonthetrainingset S, wecallvanl-rarepatternfors. sults that are similar to those obtained from Simple CFS,
andsoweonlymentiontheminpassing.
This observation leads to the first of the two main ideas
behind CFS:Thepresenceofl-rarepatternssuggestsover-
fitting, and, therefore, poorgeneralizationsincetheyopen 3.Experimental Results
up the possibility that M has special logic to detect and
CFSImplementation. Ourimplementationofl-CFSworks
handlespecificexamples. Acountofrarepatterns, however,
onadirectedacyclicgraph G representingacombinational
doesnotdirectlytranslateintoametricforgeneralization
logic circuit where each node is either the constant 0, a
(withoutbuildingapredictivemodelofthat!). Furthermore,
primary input, a 2-input And gate, or a 2-input Xor gate.
althoughapatternmayberareitmayalsobeanobservabil-
An edge is either a direct connection or an inverter and
itydon’tcare (ODC), i.e., itmayhavenoinfluenceonthe
representsa Booleanfunctionintermsoftheprimaryinputs.
outputofthecircuit. Forexample, ifthesignalwiththerare
This is a variant of an And-Inverter Graph (Biere, 2007;
pattern only feeds into an And gate whose other input is
Chatterjee,2007), astandarddatastructureinmodernlogic
0 whentherarepatternappears, thenthevalueoftherare
synthesisusedtohandlecircuitswithhundredsofmillions
patterndoesnotmatterindecidingtheoutputofthecircuit.
ofnodes. Whileconstructingthesegraphs, wepropagate
Weaddressbothproblemswiththesecondmainideabehind constantsbutdonotextractcommonsub-expressions.
CFS:perturbedsimulationofatrainingexamplewherewe
Wemaketwopassesthroughthenodesof G intopological
simulate an example through M as usual, but when we
order starting from primary inputs. In the first pass, we
encounteral-rarepattern, insteadofpropagatingittothe
simulatethetrainingsetthrough G toobtainthecountsof
fanouts (i.e., gatesthatdependonthissignal), weperturbthe
differentpatternsinthecircuit. Inthesecondpass, weuse
patternandsimulatethefanoutswiththeperturbedpattern.
thecountsfromthefirstpasstoperturbthel-rarepatterns.
A natural perturbation is to propagate the opposite value
In Simple CFS, this boils down to replacing signals that
instead of the rare pattern. In our running example, this
takeonavalueof0 onmostexampleswiththeconstant0
corresponds to propagating a 0 instead of 1 for signal s
0 signalandlikewisefor1. CFSthusrunsinlineartimein
tothemultiplexerwhensimulatingthetrainingimagex .
0 thesizeofthegraphandthetrainingdata. Forperformance,
Inthismanner, wepreventthemodelfromidentifyingx
0 the simulations are done in a bit parallel manner for all
andweseethattheoutputforx isnolongernecessarilyy .
0 0 trainingexamplesatthesametime. Toavoidrunningout
Wecallthismodifiedsimulationprocedurel-counterfactual
of memory, we use reference counting to recycle storage
simulationorl-CFSforshort.
forintermediatesimulatedvalueswhentheyarenolonger
Weperformperturbedsimulationforeachtrainingexample needed. A typical run of l-CFS in our experiments takes
inturnandmeasuretheresultingaverageaccuracyoverthe lessthan10 minutesona3.7 GHz Xeon CPUandlessthan
trainingset. Wecallthisquantitythetrainingaccuracyob- 2 GBof RAM.
tainedthroughl-CFS.Inourrunningexampleofthelookup
Benchmark Problem. Whilethediscussionfromthepre-
table, itiseasytoseethatthetrainingaccuracyobtained
vious section shows how CFS can discover overfit when
through1-CFSisnobetterthanrandomchance (sinceeach
the model is a simple lookup table, it is not clear if CFS
trainingexampleismappedtoclass‘0’under1-CFS).Now,
wouldbeeffectiveonneuralnetworkstrainedwithstochas-
since random chance is what one would expect to be the
tic gradient descent (SGD). To answer this question, we
generalizationofthelookuptable (i.e., itsaccuracyon D),
trained3 neuralnetworksfor MNISTin Tensor Flow (Keras)
itistemptingtoconjecturethat1-CFStrainingaccuracyis
andcompiledthemdownintocombinationallogiccircuits.
agoodestimateofaccuracyon D. Althoughthatisnotthe
All3 networkshavethesamearchitecture: aninputlayer
caseasweshallseeempiricallyin Section3, wefindthatthe
of size 784 (i.e., 28 × 28), 3 fully connected Re LU lay-
differenceintrainingaccuracybetweennormalsimulation
erswith256 nodeseach, andafinalsoftmaxlayerwith10
andl-CFSisagoodmeasureofthedegreeofoverfitof M.
outputs. (Thus, thetotalnumberoftrainableparametersis

#### 3

Circuit-Based Intrinsic Methodsto Detect Overfitting
335,114.) Wealsoperformedsomeexperimentswith Fash- Itisremarkablethatevenwhenaneuralnetworkisrepre-
ion MNIST(Xiaoetal.,2017) andtheresultsaresimilar. sentedataverylowlevelasalogiccircuit, relativeoverfit
canbedetectedusinganintuitivealgorithmwithnohyper-
Thefirsttwonetworks (nn-real-2 andnn-real-100),
parameterstotune.
weretrainedonthe MNISTtrainingsetfor2 epochsand
100 epochsrespectively. Theygettotraining (top-1) accu- Expt.2:Impactof Architecture. Therearemanydifferent
raciesof97%and99.90%respectively. Thethirdnetwork ways in which a neural network can be compiled down
(nn-random) wastrainedonavariantof MNISTwhere intologicgates. In Expt. 1, wemadecertainarchitectural
the output labels in the training set are permuted pseudo- choicesforthecircuit, butwhatifwehadchosendifferently?
randomly and trained for 300 epochs to get to a training To evaluate that, here we replace the multipliers used in
accuracy of 91.27%.2 In all cases, we used the ADAM Expt. 1 with array multipliers (i.e., multipliers based on
optimizerwithdefaultparametersandbatchsizeof64. As the elementary algorithm for multiplication). Figure 2 b
expected, nn-real-2 is the least overfit and gets to a showstheresultant CFScurves (withdashedlines) aswell
validationsetaccuracyof97%(i.e., hasanegligiblegen- astheoriginalcurves (solidlines) forreference. Thecurves
eralizationgap), nn-real-100 ismoreoverfitgettingto donotcoincideindicatingthattheresultof CFSdepends
avalidationsetaccuracyof98.24%(agapof1.66%), and onthestructureofthecircuitandnotjustonthefunction
finally, the validation accuracy of nn-random is 9.73% implementedbythecircuit (sincethefunctionisthesame
(i.e., closetochance) confirmingthatitishorriblyoverfit. inbothcases). However, forthesamechoiceofarchitecture,
wefindthatthefalloffin CFScurvesareagainindicativeof
Conversionto Logic Circuits. Thisisdonebygenerating
thedegreeofoverfit.
logic subcircuits composed of 2-input And or Xor gates
and inverters for each of the operations in the neural net- Expt. 3: Impactof Choiceof Primitives. Evenatthelow-
work. Weights and activations are represented by signed estlevelofabstraction, wecanchoosewhatprimitivesto
8-bitand16-bitfixedpointnumbersrespectivelywith6 bits workwith. Toseehowthischoiceimpacts CFScurves, in
reservedforthefractionalpart. (Weightsfromtrainingare thisexperimentwedisallow Xorgatesasprimitives (thus
clamped to [−2.0,2.0) before conversion to fixed point.) requiringthatonly Andgatesandinvertersbeused). Fig-
Eachmultiply-accumulateunitmultipliesan8-bitconstant ure2 cshowstheresulting CFScurves (dashed) aswellas
(theweight) witha16-bitinput (theactivation) andaccumu- theoriginals (solid). Onceagain, weseethatthecurvesdo
latesin24 bitswithsaturation. Theconstantmultiplications notcoincideindicatingthatthechoiceofprimitivesmatters
are done by finding a minimal combination of bit-shifts but for the same choice of primitives, the falloff in CFS
(multiplicationsbypowersof2) andadditionsorsubtrac- curvesareagainindicativeofthedegreeofoverfit.
tions. Forexample,5×uisimplementedas4×u+uand
Expt. 4: Count of Rare Patterns. Figure 2 d shows for
11×uas16×u−4×u−u. Re LUsareimplementedwith
eachvalueofl, howmanyexampleshavenol-rarepatterns,
acomparatorandamultiplexer. Theoutputsofeachnet-
i.e., cannotbepossiblyaffectedbyl-CFS.Weobservein
workarethe10 signed16-bitactivationsbeforethesoftmax.
particular, that for nn-random, there are 54,094 exam-
Whenevaluatingaccuracy (with CFSorwithout) wepick
ples (about 90% of the training set) that have no 1-rare
theclasscorrespondingtothelargestofthe10 activations
patterns. Thisisinsharpcontrasttoasimplelookuptable
(top-1 accuracy). The resulting logic circuits have 35 to
whereeveryexamplewouldhavea1-rarepattern, andin
52 million Andor Xorgatesand5500 to6000 logiclevels.
factslightlymorethannn-real-100. Comparingthese
(Thesesizesalongwiththeneedtofitrandomdatadictated
curvesofcountstothe CFScurvesin Figure2 aindicates
thechoiceofarchitectureandbenchmark.)
thatperturbationisanimportantpartof CFSandthatrarity
Expt. 1: Effectof Simple CFS.Figure2 ashowsthetrain- byitselfisacrudermeasureofoverfittingsinceitmaynot
ingaccuraciesobtainedthroughl-CFSforeachofthethree beobservable.
networksaslvariesfrom1 to1024(whichisabout1.7%of
Expt. 5: Random Forests. Since CFS works on the cir-
thenumberoftrainingexamples). Wecalltheseplots CFS
cuit level, it can check random forests for overfit. Two
curves. Asl increases, i.e., asmorepatternsbecomerare
randomforestsweretrainedusingversion0.19.1 of Scikit-
and get perturbed, the accuracy falls eventually reaching
learn (Pedregosaetal.,2011). Eachforesthas10 treesand
chance. However, itisinterestingthatthedropinaccuracy
is trained using the default settings, except for bootstrap-
ishighestfornn-random (e.g., atl=64, thedropisabout
ping (toavoidnon-uniformweightsduringinference). The
45%), somewhat less for nn-real-100 (20%) and the
firstforest (rf-real) wastrainedon MNISTwhereasthe
leastfornn-real-2(1.4%). Thusthefalloffinaccuracy
secondforest (rf-random) wastrainedon MNISTwith
withlisanindicatorofthelevelofoverfitofanetwork.
theoutputlabelspseudo-randomlypermuted (asbeforewith
2 Whileevaluatingrandomfortrainingaccuracy (withorwith- nn-random). Bothforestsreachperfecttrainingaccuracy.
out CFS), thepermutedlabelsareused. rf-realgets95.58%validationaccuracywhereasasex-

#### 4

Circuit-Based Intrinsic Methodsto Detect Overfitting
1.0
0.8
0.6
0.4
0.2
0.0
0 2 4 6 8 10
log2(l)
SFC-l
htiw
ycarucca
gniniart
1.0
0.8
0.6
0.4
nn-real-2 0.2
nn-real-100
nn-random
0.0
0 2 4 6 8 10
log2(l)
(a)
SFC-l
htiw
ycarucca
gniniart
nn-real-2
nn-real-2 w/ array
nn-real-100
nn-real-100 w/ array
nn-random
nn-random w/ array
(b)
1.0
0.8
0.6
0.4
0.2
0.0
0 2 4 6 8 10
log2(l)
SFC-l
htiw
ycarucca
gniniart
60000
50000
40000
30000
nn-real-2 20000
nn-real-2 w/o xors
nn-real-100 10000
nn-real-100 w/o xors
nn-random
nn-random w/o xors 0
0 1 2 3 4 5 6
log2(l)
(c)
snrettap
erar-l
on
/w
selpmaxe
.mun
nn-real-2
nn-real-100
nn-random
(d)
1.0
0.8
0.6
0.4
0.2
0.0
0 2 4 6 8 10 12
log2(l)
SFC-l
htiw
ycarucca
gniniart
1.0
0.8
0.6
0.4
rf-real
rf-random
nn-real-2 0.2
nn-real-100
nn-random
0.0
30 25 20 15 10 5
log2(p)
(e)
esion
teknalb
htiw
ycarucca
gniniart
rf-random
rf-real
nn-real-2
nn-real-100
nn-random
(f)
1.0
0.8
0.6
0.4
0.2
0.0
0 2 4 6 8 10 12
log2(l)
SFC-l
htiw
ycarucca
gniniart
1.0
epochs = 1
epochs = 2
epochs = 5 0.8
epochs = 10
epochs = 20
epochs = 50 0.6
epochs = 100
epochs = 200
0.4
0.2
0.0
30 28 26 24 22 20 18
log2(p)
(g)
esion
teknalb
htiw
ycarucca
gniniart
epochs = 1
epochs = 2
epochs = 5
epochs = 10
epochs = 20
epochs = 50
epochs = 100
epochs = 200
(h)
Figure2: Theresultsofexperimentsin Section3. Plot (a) showsthe CFScurvesfor3 networkswithdifferentamountsof
overfitting;(b) and (c) showtheimpactofthechoiceofmultipliersandofprimitivelogicgatesrespectivelyon CFScurves;
(d) showshowmanyexamplesareunaffectedbyl-CFSsincetheydonothaveanyl-rarepatterns;(e) shows CFScurvesfor
randomforests;(f) showstrainingaccuracieswhenasignalisrandomlyflippedwithprobabilityp;and (g) and (h) showthe
differencesbetween CFSandrandomflipsrespectivelyfor8 networkstrainedonadatasetwithhighlabelnoise.

#### 5

Circuit-Based Intrinsic Methodsto Detect Overfitting
pectedrf-randomgetsnobetterthanchance.(rf-real
has about 14 K nodes per tree whereas rf-random has
about70 Knodespertree.)
Theforestsarecompileddowntocircuitsinastraightfor-
wardmanner. Eachtreeiscompiledseparatelyandproduces
1016-bitoutputs (oneperclass). Thecorrespondingoutputs
areaddedacrossall10 treesandtheclassoutputbythefor-
estcorrespondstotheclasswiththemaximumvalue. Each
internalnodeinatreemapstoamultiplexercontrolledbya
8-bitcomparatortoimplementthethreshold, andeachleaf
node corresponds to 10 16-bit constants representing the
numberofexamplesthatoccupyeachclassinthatleaf (thus
mostentriesarezero). Thecircuitforrf-realhasabout
700 Knodeswhereasrf-randomhas3 Mnodes. Bothare
lessthan250 logiclevelsdeep. (Thesearemuchsmaller
thanthecircuitsfortheneuralnetworks.)
Figure2 eshowsthe CFScurvesforthetworandomforests.
Onceagain, weseethattheoverfitmodel (rf-random)
degrades faster than the model with better generalization
(rf-real) confirmingthat CFSiseffectiveevenformod-
elsthatarefundamentallydifferentfromneuralnetworks.
However, itisinterestingtoseethatifwecompareacross
modelfamiliesi.e. betweentheneuralnetworksfrom Expt.
1(repeatedin Figure2 eforconvenience) andtherandom
forests, CFS is not effective at distinguishing overfit. In
particular, nn-real-2 whichisnotoverfitdegradesmore
rapidlythatrf-randomwhichishighlyoverfit. Wedis-
cussthisingreaterdetailin Section4.
Expt. 6: Blanket Noise. CFSmaybeseenasaddingatar-
getednoise. Here, instead, weaddblanketnoisebysimulat-
1.0
0.8
0.6
0.4
0.2
0.0
0 2 4 6 8 10 12
log2(l)
SFC-l
htiw
ycarucca
gniniart
ingthetrainingsetwhilerandomlyflippingthenodevalues
withprobabilityp. Aspvariesfrom2−30 to2−5, theresult-
ingnoisecurves (Figure2 f) aresimilartothe CFScurves
(Figure 2 e). However, the more overfit nn-real-100
does not fall faster than nn-real-2. With CFS, these
curvesarewellseparated, andthegapbetweenneuralnets
andforestsismuchlarger.
Expt. 7: Sensitivity. Tobetterunderstandthesensitivity
differencebetween CFSandblanketnoise, wetrained38
neuralnetworks (withthesamearchitectureasbeforebut
different number of epochs) on MNIST with exactly one
halfoflabelsrandomized (somaximumaccuracypossible
is about 55%). We show the CFS curves and the noise
curvesfor8 representativenetworksin Figures2 gand2 h
respectively. Note the crossover of the CFS curves that
indicatesalargerfalloffforoverfitnetworkscomparedto
the more uniform degradation of the noise curves. It is
fascinating that all the CFS curves cross over at a single
pointwithanaccuracyofabout50%. Thisisdiscussedin
Section4.
Expt. 8: Composite CFS. For completeness, Figure 3
shows the results of running Composite CFS instead of
Simple CFS in the setup of Expt. 7. We see that these
curvesareverysimilartothoseof Simple CFSin Figure2 g.
4.Discussion
Structure Dependence. Expt. 2 and3 showthattheresults
of Simple CFSdependnotjustonthefunctionbutonthe
structureofthecircuit. (Other CFSvariantsweinvestigated
showthisbehavioraswell.)Asmallexampleprovidessome
insight. Considerthe Booleanfunctionf (a, b, c)=aeval-
uatedonthetrainingsetcomprisingthefull Booleancube
(i.e., all 8 combinations of a, b, c ∈ {0,1}). In addition
to the direct implementation, f can also be implemented
epochs = 1 (redundantly) asa·b·c+a·¬b·c+a·b·¬c+a·¬b·¬c.
epochs = 2
epochs = 5 Itiseasytoseethatunder1-CFS, thedirectimplementation
epochs = 10
epochs = 20 is unchanged (there are no 1-rarepatterns) butthe redun-
epochs = 50
dantimplementationmapstoconstant0(theoutputofeach
epochs = 100
epochs = 200 conjunctionis1 onlyonce).
Althoughthisisnotaproblemwhenthecompilationpro-
cesscanbecontrolled, thisisbadnewsintheadversarial
setup. Agoodmodelwithapoorimplementationmayshow
steeperdegradationunder CFSthanamoreoverfitmodel
with better implementation (c.f. nn-real-2 with array
Figure 3: This plot shows the result of repeating Expt. 7 v/snn-real-100 without Xorsatl =256). Ideally, we
butusing Composite CFS(insteadof Simple CFS) where wouldliketofindavariantof CFSthatdoesnotdependon
wecomputeandperturbrarepatternsacrossallthesignals structurebutonlyonthefunction.3 Intheabsenceofthat
thatfeedintoalogicgate, i.e., acrossallthefaninsofagate
3 Inprinciple, onecouldcanonizethecircuitstructurebefore
(insteadofcomputingandperturbingrarepatternsforeach
applying CFS, saybybuilding Reduced Order Binary Decision Di-
signalorfaninindependently). Theresultswith Composite agrams (ROBDDs), butthatwouldbecomputationallyprohibitive.
CFSareverysimilartothoseof Simple CFS(Figure2 g). Alternatively, onecouldlightlyoptimizethecircuitbefore CFS

#### 6

Circuit-Based Intrinsic Methodsto Detect Overfitting
1.0
0.8
0.6
0.4
0.2
0.0
0 2 4 6 8 10 12
log2(l)
SFC-l
htiw
ycarucca
gniniart
1.0
0.8
0.6
epochs = 1
epochs = 2
epochs = 5 0.4
epochs = 10
epochs = 20
epochs = 50 0.2
epochs = 100
epochs = 200
0.0
0 2 4 6 8 10 12
log2(l)
(a)
SFC-l
htiw
ycarucca
gniniart
epochs = 1
epochs = 2
epochs = 5
epochs = 10
epochs = 20
epochs = 50
epochs = 100
epochs = 200
(b)
Figure4: Theseplotsshowtheresultofrepeating Expt. 7 whenwecorruptonlyathirdofthelabelsin (a)MNISTandin (b)
Fashion-MNIST.Inbothcases, justasin Figure2 gweseethatallthecurvespassthroughapointclosetothemaximum
achievableaccuracy (around2/3). Thereasonforthisisdiscussedin Section4 under Generalizationin Deep Learning.
ideal, weviewthestructureofthecircuitasacertificateof thoughunbalancedhavenorarepatterns.
howwellthedatasetislearnt, andmakeit Merlin’srespon-
Thisexamplesuggestsawaytobreak CFS.If Merlinwanted
sibility to find and present the most convincing structure.
tobuildalookuptable, butprevent Arthurfromidentifying
Fromthisperspective, intheaboveexample, thedirectim-
it as such (by applying say 1-CFS), instead of building
plementationoff (whichisnotimpactedby1-CFS) ismore
thecircuitin Figure1, Merlincouldbuildadecision (i.e.,
convincingthantheredundantimplementationoff (which
multiplexer) treebyrecursivelysplittingononevariable (i.e.,
isseverelyimpacted).
onecomponentofx) atatimebypickingacomponentthatis
Adversarial Attack on CFS. Based on the discussion balanced (i.e., takeson0 and1 valuesroughlyequallyoften)
aboveitiseasytodesignawaytoarbitrarilydegradethe andleadstobothbranchesalsobeingbalanced (i.e., have
performance of a circuit under CFS. But is the opposite noclasswitheithertoofewortoomanyexamples, although
possible? Can Merlinfitanarbitraryfunctionontheinputs havingnoexamplesofaclassorhavingonlyexamplesof
but compile it down to a circuit which does not degrade oneclassisfineandexpectedattheleafnodes). Inother
under CFS? Expt. 5. offers a clue. The overfit model words, Merlin could use a procedure similar to the usual
rf-randomfittedonrandomlabelsfallsoffmoreslowly decisiontreeconstructionprocedures, butwiththeimportant
than nn-real-2 which generalizes well. What is go- differenceoffavoringbalancedsplitsinsteadofunbalanced
ing on? The short answer is that although each tree in splits.(Theunbalancedsplitsarelikelywhy CFSiseffective
rf-randomisextremelyoverfitwithmostleavescontain- totheextentitisonrandomforests.)
ingonlyasingleexample, thecircuitnodeshavefewrare
Finally, assuggestedbyareviewerofthispaper, itwouldbe
patternsduetotheobservabilitydon’tcaresintroducedby
interestingtoextendthisidea (ultimatelybasedon Shannon
themultiplexers.
decompositionandco-factoring) toanalgorithmthatcan
Againasimpleexampleisinstructive. Letf betheparity amplifythecountofrarepatternsinagivencircuitwithout
functiononnbits, i.e., f (x , x ,..., x )=x ⊕x ...⊕ changingitsfunctionality.
1 2 n 1 2
x . Consider an tree implementation of this function ob-
n Comparisonwith Blanket Noise. Basedon Expt. 6 and
tained using Shannon decomposition which has a multi-
7, webelievethatblanketnoiseislesssensitivethan CFS.
plexeratthetopcontrolledbyx andwith1⊕x ⊕x ...⊕
1 2 3 Our results here add to the extensive literature on noise,
x andx ⊕x ...⊕x asitsdatainputs. Ifthetraining
n 2 3 n generalizationandfaulttoleranceinneuralnetworks (e.g.,
set is the full Boolean cube, i.e., {0,1}n it is easy to see
see Bernieretal.(2001) andthereferencestherein) byex-
thattherearenol-raresignalsforl<<nsinceeachinput
tendingthemtothecircuitlevel (wheredistinctionsbetween
tothemultiplexerisbalanced, i.e., hasequalnumberof0 s
activationorweightnoise, oradditiveormultiplicativenoise
and1 s. Since Shannondecompositionisrecursive, asimilar
disappear) and to other model families such as random
argumentholdsforthelowerlevelsofthetreeuntilweget
forests. Furthermore, Expt. 6 presents a direct compari-
to the leaves which are constant 0 and constant 1 which
sonofthefault-toleranceofneuralnetsandforestswhere
butthatmaynotbeenough. forestsareseentobeabout1000 xmorefault-toleranttobit

#### 7

Circuit-Based Intrinsic Methodsto Detect Overfitting
flips. Thisislikelyduetotheredundancyfromensembling. obviouslyapply. Wehavenotstudiedifnormalizedmargin
Italsosuggeststhatnoise-basedintrinsicmethodscouldbe can be exploited by an adversary. Similarly, most mea-
easilyfooledbyanadversarybyaddingredundancy. suresbasedontheshallownessofminimaarenotadequate
sincetheyarenotscale-invariant (Dinhetal.,2017) andwe
Generalization in Deep Learning. Why do neural nets
havenotinvestigatedifmorerecentworkonscale-invariant
trainedwith SGDgeneralizewhentheyhavesufficient (ef-
measures (Rangamani et al., 2019) can be exploited. In
fective) capacity to memorize their training set? This is
comparison to these and other approaches for neural net-
anopenresearchquestion (Zhangetal.,2017;Arpitetal.,
works (Aroraetal.,2018;Neyshaburetal.,2018), CFSis
2017; Bartlett et al., 2017; Arora et al., 2018; Neyshabur
fundamentallymorediscrete, whichmakesitapplicableto
et al., 2018). Expt 5. shows that this question is not lim-
alargerclassofmodels. However, incontrasttotheother
itedtonets—thesamecouldbeaskedforrandomforests
approaches, wedonothaveanytheoreticalboundsyetwhile
aswell. One (informal) answerforforestsisthatdecision
ourresultsindicatethatwithoutfurtherrefinementsto CFS
treeconstructionprocedureslookforcommonpatternsbe-
itself, anygeneralizationboundsfroml-CFSwouldlikely
tweenexamples. Whenexamplessharecommonality, they
bevacuousinpractice.
arecombinedintocommonleafnodesandthemodelgen-
eralizes, whereas when there is little commonality, each
exampleisitsownleaf (sothetrainingsetisfitwell) butthe 5.Conclusionand Future Work
modelfailstogeneralize. Couldthesamethingbegoingon
Ourmainresultisthat CFSbasedonaddingsmallamounts
with SGDandnetworks? CFSonnn-randomand Expt4.
oftargetednoiseatthelogiccircuitlevelcandetectoverfit.
providedirectevidencethatevenonrandomdata, netsdo
This is remarkable because at this level of representation
not“brute-forcememorize”butidentifycommonpatterns
we have lost most aspects of the structure of the model,
in the data (a question raised in Zhang et al. (2017) and
such as the distinction between weights and activations,
discussedin Arpitetal.(2017)).
orevenwhetherthemodelisaneuralnetwork, arandom
Inthiscontext, itisinterestingtostudywhythe CFScurves forest, oralookuptable. Furthermore, variationssuchas
forthedifferentnetworksin Expt7. intersectatacommon perturbingonlyrarepatternsinsinglesignalsoracrossthe
pointcorrespondingapproximatelytotheachievableaccu- faninsofagateleadtoqualitativelysimilarresultsandthere
racy. Thisholdsforotheramountsoflabelrandomization arevariants (suchas Simple CFS) thatarenaturallyfreeof
e.g., see Figure4 forthe CFScurveswhenonethirdofthe hyper-parameters.
labelsarecorrupted. Roughlyhalfoftheexamplesareeasy
Bystudyingrarepatterns, wefindthat SGDdoesnotlead
sincetheyhavecorrectlabelsandarelearntinthefirstfew
to“bruteforce”memorization, butfindscommonpatterns
epochs. The remaining examples with corrupt labels are
(whethertrainingisdonewithrandomizedlabelsoractual
harder and learnt only in later epochs by the models that
labels), andneuralnetworksarenotunlikeforestsinthis
aretrainedlonger. With CFS, theaccuracyofthosemodels
regard. Byaddingblanketnoise, randomforestsarefoundto
breaksdownearliersincethehardexampleshavemorerare
beabout1000 xmoreresilienttonoisethanneuralnetworks
patterns than the easy examples, and the accuracy on the
whichcouldbeusefulwhenimplementingmachinelearning
easyexamplesthusformsalimitingcurveforallmodels.
systemswithunreliablelowlevelcomponents.
Thisprovidesmore (anddirect) evidencefortheclaimin
Arpit et al. (2017, §1) that “SGD learns simpler patterns Thereareseveraldirectionsforfuturework. Weanalyzeflat
firstbeforememorizing.” Furthermore, wecouldidentify circuits, but with a clever implementation that constructs
“simplerpatterns”asexamplesthathavefewerrarepatterns thecircuiton-the-flyfromahigherlevelspecification, the
and “memorizing” as what is required for examples that computations can scale to larger models. We could also
havemorerarepatterns. Thus, learningsimplerpatternsand apply CFSathigherlevelsofabstraction (perhapsaspartof
memorizationarenotfundamentallydifferentbutlieattwo themodelevaluationprocessinframeworks, suchas Scikit
endsofaspectrum. Learnand Tensorflow Estimators) thoughatthatlevelthere
are moredegrees of freedomin theimplementation (e.g.,
Related Work. Onemaybetemptedtoviewmarginasan
whatkindofnoisetoadd).
intrinsicmeasuretoestimatethegeneralizationofamodel.
However, when we have models with intermediate repre- Thenotionofrarityconsideredinthisworkmayberegarded
sentations, the notion of margin by itself is not adequate as a local notion, since we count the patterns at a signal
sinceanadversarycanoverfittoafavorableintermediate (or a group of signals in the case of Composite CFS) in
representationthatiseasilylinearlyseparated (butotherwise isolation. It is possible to extend this notion to a global
arbitrary). However, recentworkinthisarea (e.g., Bartlett notion of rarity by propagating occurrence counts along
etal.(2017)) hasfocusedonmarginsnormalizedbyspectral withsignalvaluesduringsimulation. Inthecorresponding
complexity (i.e., ameasurerelatedtothe Lipschitzconstant CFS, a pattern is perturbed when its global rarity drops
ofthenetwork) andinthatcasetheaboveargumentdoesnot

#### 8

Circuit-Based Intrinsic Methodsto Detect Overfitting
Dwork, C., Feldman, V., Hardt, M., Pitassi, T., Reingold, Rangamani, A., Nguyen, N.H., Kumar, A., Phan, D., Chin,
O., and Roth, A. Thereusableholdout: Preservingvalid- S.H., and Tran, T.D. AScale Invariant Flatness Mea-
ity in adaptive data analysis. Science, 349(6248):636– sure for Deep Network Minima. ar Xiv e-prints, art.
638, 2015. ISSN 0036-8075. doi: 10.1126/science. ar Xiv:1902.02434, Feb2019.
aaa9375. URL https://science.sciencemag.
Xiao, H., Rasul, K., and Vollgraf, R. Fashion-mnist: anovel
org/content/349/6248/636.
imagedatasetforbenchmarkingmachinelearningalgo-
Le Cun, Y.and Cortes, C.MNISThandwrittendigitdatabase. rithms. ar Xiv, 2017. URL https://arxiv.org/
http://yann.lecun.com/exdb/mnist/, 2010. URL http: abs/1708.07747.
//yann.lecun.com/exdb/mnist/.
Zhang, C., Bengio, S., Hardt, M., Recht, B., and Vinyals, O.
Neyshabur, B., Li, Z., Bhojanapalli, S., Le Cun, Y., and
Understandingdeeplearningrequiresrethinkinggeneral-
Srebro, N. Towards understanding the role of over-
ization. In Proceedingsofthe International Conference
parametrization in generalization of neural networks.
on Learning Representations ICLR,2017.
Co RR, abs/1805.12076,2018. URLhttp://arxiv.
org/abs/1805.12076. Zielinski, P., Krishnan, S., and Chatterjee, S. Weak and
stronggradientdirections:Explainingmemorization, gen-
Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V.,
eralization, and hardness of examples at scale. Ar Xiv,
Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P.,
abs/2003.07422,2020. URLhttps://arxiv.org/
Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cour-
abs/2003.07422.
napeau, D., Brucher, M., Perrot, M., and Duchesnay, E.
Scikit-learn: Machine learning in Python. Journal of
Machine Learning Research,12:2825–2830,2011.

### Additional Content

#### 1

Circuit-Based Intrinsic Methods to Detect Overfitting
Satrajit Chatterjee1 Alan Mishchenko2
Abstract knowledge, suchas, theperformanceofthemodelonexam-
plesheldoutfromthetrainingprocess, detailsoftheprocess
Thefocusofthispaperisonintrinsicmethodsto
usedtofindthemodel (e.g., multiplehypothesistestingwith
detectoverfitting. Byintrinsicmethods, wemean
registration), orlimitationsofthefunctionfamilytowhich
methodsthatrelyonlyonthemodelandthetrain-
themodelbelongs (e.g., VCdimension, Rademachercom-
ingdata, asopposedtotraditionalmethods (we
plexity) orthoseofthesizeoftheparameterspaceofthe
callthemextrinsicmethods) thatrelyonperfor-
model (e.g., Akaike Information Criterion).
manceonatestsetoronboundsfrommodelcom-
plexity. Weproposeafamilyofintrinsicmethods Weclassifymethodsrelyingontheknowledgeofthefunc-
called Counterfactual Simulation (CFS) which tionfamilyasextrinsicbecauseoftentheinformationrel-
analyze the flow of training examples through evant to overfitting is not directly represented in a given
themodelbyidentifyingandperturbingrarepat- model. For example, a model could have been found by
terns. Byapplying CFStologiccircuitswegeta searchingamuchsmallerspacethanthatimpliednaïvelyby
methodthathasnohyper-parametersandworks thefunctionfamilytowhichitbelongs, duetoeitherexplicit
uniformlyacrossdifferenttypesofmodelssuch regularizationortheregularizationimplicitintheoptimiza-
as neural networks, random forests and lookup tionorsearchprocedure. Conversely, thegivenmodelcould
tables. Experimentally, CFScanseparatemodels havebeenfoundbysearchingamuchlargerspaceofmod-
with different levels of overfit using only their elsthroughhyper-parametersearch, orbypickingthebest
logiccircuitrepresentationswithoutanyaccess modelfromanumberofdifferentmodelfamilies, butthe
tothehighlevelstructure. Bycomparinglookup specificfinalmodelitselfdoesnotcarryanyvestigesofthe
tables, neural networks, and random forests us- largerspacethatwassearchedover. Bothofthesesituations
ing CFS, wegetinsightintowhyneuralnetworks arecommoninmodernmachinelearning.
generalize. Inparticular, wefindthatstochastic
Intrinsicmethodsareofpracticalinterestsincewithmodern
gradient descent in neural nets does not lead to
deeplearningmodels, wefindthatextrinsicestimatesbased
“bruteforce”memorization, butfindscommonpat-
onmodelcomplexityaretypicallyvacuoussincethesemod-
terns (whetherwetrainwithactualorrandomized
elsarepowerfulenoughtofitarbitrarydata (Zhangetal.,
labels), andneuralnetworksarenotunlikeforests
2017). Consequently, practitionersresorttostudyingperfor-
in this regard. Finally, we identify a limitation
manceonaholdoutdataset (orcrossvalidation), butthisis
withourproposalthatmakesitunsuitableinan
unsatisfactoryforacoupleofreasons. First, thismeansthat,
adversarialsetting, butpointsthewaytofuture
inalowdatasetting, wecannotuseallthedatafortraining,
workonrobustintrinsicmethods.
but have to keep significant portions aside for validation
(e.g., seediscussionin Dietterich (1998)). Second, itmay
1.Introduction bedifficulttoensureapristineholdoutthatisnottouched
duringtheresearchprocessparticularlyiftheprojectislong
Thispaperconsidersmethodstodetectoverfittingofamodel running. Even with a few queries to the hold out during
based only on the model and the training data. In termi- theresearchprocess, itispossibletostartfittingtothehold
nologythatweintroduce, wecallsuchmethodsintrinsic, out (Dworketal.,2015).
incontrasttoextrinsicmethods, whichrelyonadditional
Intrinsicmethodsarealsointerestingfromatheoreticalper-
1 Google, Mountain View, California, USA 2 Department of spective. Imaginethatwehavesufficientcomputingpower
EECS, Universityof California, Berkeley, California, USA.Corre- to, say, enumerateallneuralnetworks (andtheirweights)
spondenceto:Satrajit Chatterjee<schatter@google.com>, Alan
up to a certain size. Among all the networks that fit the
Mishchenko<alanmi@berkeley.edu>.
datawell, intrinsicmethodscoulddistinguishbetweenthose
Proceedings of the 37 th International Conference on Machine networksthatgeneralizewellfromthosethatdonot, andwe
Learning, Online, PMLR119,2020. Copyright2020 bytheau-
thor (s).
0202
gu A
5
]GL.sc[
2 v19910.7091:vi Xra

#### 2

Circuit-Based Intrinsic Methodsto Detect Overfitting
belowthegiventhreshold. Preliminaryexperimentsshow References
that this more stringent notion of rarity is more effective
Arora, S., Ge, R., Neyshabur, B., and Zhang, Y. Stronger
indetectingoverfitinthecaseofrandomforests, andmay
generalization bounds for deep nets via a compression
offerawaytopartiallymitigatetheadversarialattackon
approach. Co RR, abs/1802.05296, 2018. URL http:
CFSoutlinedintheprevioussection.
//arxiv.org/abs/1802.05296.
Finally, basedoninsightsfromouranalysisof Simple CFS,
Arpit, D., Jastrzebski, S.K., Ballas, N., Krueger, D., Bengio,
wewouldliketocontinuethesearchforanintrinsicmethod
E., Kanwal, M. S., Maharaj, T., Fischer, A., Courville,
thatdoesnotdependonthemodelstructureandisadversar-
A.C., Bengio, Y., and Lacoste-Julien, S. Acloserlook
iallyrobust, ortoshowthatsuchamethoddoesnotexist,
at memorization in deep networks. In Proceedings of
evenforlearningtasksofpracticalinterest. Asageneral-
the34 th International Conferenceon Machine Learning,
izationofthisidea, itisinterestingtocontemplatelearning
ICML2017, Sydney, NSW, Australia,6-11 August2017,
algorithmsthatproducecertificatesofgeneralization, much
pp. 233–242, 2017. URL http://proceedings.
likea Booleansatisfiabilitysolvercanproduceacertificate
mlr.press/v70/arpit17 a.html.
ofsatisfiabilityorofunsatisfiability.
Postscript: The Theoryof Coherent Gradients. Theob- Bartlett, P.L., Foster, D.J., and Telgarsky, M.J. Spectrally-
servations in thispaper from applying CFS toneural net- normalizedmarginboundsforneuralnetworks. In Guyon,
works—particularly, theabsenceof“bruteforce”memo- I., Luxburg, U.V., Bengio, S., Wallach, H., Fergus, R.,
rizationwithrandomlabels, andtheexistenceofeasyand Vishwanathan, S., and Garnett, R. (eds.), Advances in
hardexamplesasdiscussedin Section4—inspiredthede- Neural Information Processing Systems 30, pp. 6240–
velopmentofatheorycalled Coherent Gradients (CG) that 6249.Curran Associates, Inc.,2017.
providesasimpleandintuitiveexplanationofgeneralization
Bernier, J., Ortega, J., Ros Vidal, E., Rojas, I., and Pri-
innetworkstrainedwith (stochastic) gradientdescent.
eto, A. A quantitative study of fault tolerance, noise
Thekeyideain CGisasfollows. Sincetheoverallgradient immunity, and generalization ability of mlps. Neural
g istheaverageofthegradientsoftheindividualtraining Computation, 12:2941–2964, 01 2001. doi: 10.1162/
examples, theremaybecertaincomponents (directions) ofg 089976600300014782.
whicharesignificantlystrongerthanothercomponentsdue
Biere, A. Aigerlibraryandtools, 2007. URLhttp://
tothegradientsofmultipleexamplesbeinginagreement
fmv.jku.at/aiger. [Online;accessed1-July-2020].
onthosecomponents, andtherebyreinforcingeachother.
Since the changes to the trainable parameters of the net-
Chatterjee, S. On Algorithms for Technology Map-
workareproportionaltothegradient, thebiggestchangesto
ping. Ph D thesis, EECS Department, Univer-
theparametersarebiasedtobenefitmultipleexamples, and
sity of California, Berkeley, Aug 2007. URL
thereforelikelytogeneralizewell (basedonastabilityargu-
http://www2.eecs.berkeley.edu/Pubs/
ment).However, iftherearenosuchstrongcomponents, i.e.,
Tech Rpts/2007/EECS-2007-100.html.
alltheper-examplegradientsareroughlyorthogonal, then
eachexampleisfittedindependently, andthiscorresponds Chatterjee, S. Coherentgradients: Anapproachtounder-
tomemorization (andpoorgeneralization, againfromasta- standing generalization in gradient descent-based opti-
bilityargument). Thus CGprovidesanuniformexplanation mization. In Proceedingsofthe International Conference
ofbothgeneralizationandmemorizationinneuralnetworks. on Learning Representations ICLR,2020. URLhttps:
//openreview.net/forum?id=rye FY0 EFw S.
Pleasesee Chatterjee (2020) and Zielinskietal.(2020) for
amoredetaileddevelopmentofthisidea, includingananal- Dietterich, T. G. Approximate statistical tests for
ysisofeasyandhardexamplesfromthisperspective, anda comparing supervised classification learning algo-
naturalmodificationtogradientdescentthatsignificantlyre- rithms. Neural Comput., 10(7):1895–1923, Oc-
ducesoverfittingbysuppressingthosegradientcomponents tober 1998. ISSN 0899-7667. doi: 10.1162/
thatarenotcommontomanyexamples. 089976698300017197. URLhttp://dx.doi.org/
10.1162/089976698300017197.
Acknowledgments
Dinh, L., Pascanu, R., Bengio, S., and Bengio, Y.
Thefirstauthorthanks Michele Covell, Ali Rahimi, Alex Sharp minima can generalize for deep nets. Co RR,
Alemi, Shumeet Baluja, Sergey Ioffe, Tomas Izo, Shankar abs/1703.04933, 2017. URL http://arxiv.org/
Krishnan, Rahul Sukthankar, and Jay Yagnik for helpful abs/1703.04933.
discussions. The second author was supported in part by
SRCContract2867.001.


---

## Content Unique to Old Extraction (not found in markitdown output)

### 1. # Circuit-Based Intrinsic Methods to Detect Overfitting...

# Circuit-Based Intrinsic Methods to Detect Overfitting

### 2. > *Source PDF: Circuit-Based Intrinsic Methods to Detect Overfitting.pdf*
> *Ext...

> *Source PDF: Circuit-Based Intrinsic Methods to Detect Overfitting.pdf*
> *Extraction: Combined — markitdown raw text (base) + previous extraction supplements*

### 3. ## Content from Previous Extraction (not in markitdown output)...

## Content from Previous Extraction (not in markitdown output)

### 4. ### Visual/Chart/Graph Descriptions...

### Visual/Chart/Graph Descriptions

### 5. Circuit-Based Intrinsic Methodsto Detect Overfitting
couldviewthemodelasacertifi...

Circuit-Based Intrinsic Methodsto Detect Overfitting
couldviewthemodelasacertificateofgeneralization.1 In
addition, iftheintrinsicmethodwasefficient, itwouldmean
thatsupervisedlearning (andnotjustfittingthetrainingdata)
isin NP.Intrinsicmethodscanalsohelpshedlightonwhy
neural networks trained with stochastic gradient descent x x x
n -1 1 0
generalizeinspiteoftheirlargecapacity. Somerecentanal-
s s
ysesbasedonnormalizedmargin, curvature, etc.(Bartlett n -1 0
et al., 2017; Rangamani et al., 2019; Arora et al., 2018;
Neyshaburetal.,2018) maybeseenasintrinsicestimates y
y y y forgeneralizationalbeitspecializedtoneuralnetworks.
n -1 1 0
Intrinsicmethodscanbeconsideredinthecontextofapro-
tocolinvolvingtwoagents. Let S beapublicdatasetdrawn
fromadistribution D thatgeneratessamplesinfrequently
(e.g., quarterlyfinancialstatementsandstockmarketreturns
ofpubliccompanies, orpublichealthdataontreatmentsand
patientoutcomes). Suppose Arthurwantstobuildamodel
from S butinsteadofdoingsohimself, heoutsourcesitto
Merlin, anuntrustedadversary. Merlincomesbackwitha
model Mbutdoesnotdiscloseanydetailsofhismodeling
process. Howcan Arthurconvincehimselfthat Misnot
horriblyoverfit? Forexample, Mcouldsimplybealookup
table built from S. Normally Arthur would evaluate M
onnewsamplesfrom D, butinoursetup, Arthurdoesnot
haveanysamplesotherthanthosein S sincealltheexisting
dataispublic. Now, if Arthuronlyhasaccessto Masa
blackboxandhecanonlyevaluate Monelementsof S, it
appearsthereislittlehecandotodistinguishagoodmodel
from a lookup table. But can Arthur do better if he has
accesstotheinternalsignalsintheimplementationof M?
Thisisthecentralquestionofthispaper.
Wetakeafirststeptowardsansweringthisquestionbystudy-
inganaturally-motivatedfamilyofintrinsicmethods, called
Counterfactual Simulation (CFS), andevaluatingtheiref-
ficacyexperimentallyonabenchmarkproblem. Themain
ideabehind CFSistoanalyzetheflowofthetrainingexam-
plesin Sthroughthestructureof M.Thisisonlyafirststep
since CFS, althoughpromisinginpractice, hassignificant
limitations. Inparticular, ourexperimentsshowthateven
ifwecouldproveboundsbasedon CFS, theywouldnotbe
tightenoughinanadversarialsetting. However, wehope
thatthispaperencouragesresearchtoovercometheselimi-
tationsortoshowthatnosuchmethodcanexist, especially
forlearningtasksofpracticalinterest.
2.Counterfactual Simulation (CFS)
Thestructureof Mcanbedescribedatdifferentlevelsof
abstraction. For instance, if M is a fully connected feed
forwardneuralnetwork, wecandescribeitasasequence
of layers. Going one level lower, we can describe it as
1 Keepingaholdoutsetwouldnothelpushere—whatwould
thatevenmean?
...
x
= = =
s
1
1000000000 0 0 0
1 1 1
Figure1: Acircuitimplementingalookuptablethatmem-
orizes the training examples (x , y ) for 0 ≤ i < n. We i i
observe that in this extreme case of overfitting, there are
signalsinthecircuit (thes ) thatidentifyspecifictraining i
examples. For e.g., s is 1 only when x = x and 0 for 0 0
all other x (assuming the x are distinct). We say that 1 i i
is a rare pattern for the signal s . Based on this exam- 0
ple, weproposethattheoccurrenceofrarepatternsduring
simulationofatrainingsetthroughamodelindicatesover-
fitting, andinthiswork, weexploretowhatextentsuchrare
patternscanbeusedtodetectoverfittinginmorecomplex
modelssuchasneuralnetworksandrandomforests.
a directed acyclic graph (DAG) of fixed or floating point
adders, multipliers, andpointwisenon-linearities. Finally, at
thelowestlevelofabstraction, wecandescribethestructure
of Masa DAGofprimitivelogicgatessuchas2-input And
gates and inverters, i.e., as a combinational logic circuit.
Inoursetup, itisnaturaltoworkatthislowestlevel, i.e.,
logicgatessinceitallowsdifferentkindsofmodelssuch
aslookuptables, randomforests, andneuralnetworksall
tobemappedintothesameformat. Thus, Merlinneednot
discloseevenwhattypeofmodelhehasbuilt, butsimply
provides Arthurwithacombinationallogiccircuitforthe
model.
Tomakethisconcrete, considerthe MNISTimageclassifi-
cationproblem (Le Cun&Cortes,2010) whichwewilluse
asarunningexample. Disthedistributionofhandwritten
digits and their classes. S is a sample from D of 60,000
imagesx andtheircorrespondinglabelsy (thus,0≤i< i i
60000) i.e., the MNISTtrainingset. Eachx is6,272 bits
i
wide (correspondingto28×28 pixels×8 bitsperpixel),
and each y is 10 bits wide (for a 1-hot representation of
i
the 10 possible classes). Therefore, a classifier to solve
thisproblemisacircuitwith6,272 Booleaninputsand10
Booleanoutputs.
Suppose Merlin’smodelforthe MNISTclassifierisasimple
lookuptable. Howwouldthecircuitforitlook? Figure1
sketchesonepossibility. The6,272-bitinputxiscompared
witheachoftheexamplesx inturnandifthereisamatch,
i

### 6. Circuit-Based Intrinsic Methodsto Detect Overfitting
thecorresponding10-bitoutpu...

Circuit-Based Intrinsic Methodsto Detect Overfitting
thecorresponding10-bitoutputy isselected. Ifnoexample Other Typesof CFS.Thereareothervariantsofthepro-
i
matches, thenthemodel (arbitrarily) returnsthe1-hotvector ceduredescribedabove (whichwecall Simple CFSorjust
representingclass‘0’. Now, ifwesimulatethiscircuiton CFS).Ofparticularinterestis Composite CFSwhichisuse-
examplesin S, wenoticethatthereareinternalsignalsin fulforcircuitswithgatesthathavemanyinputsorareat
thecircuitthatarecapableofidentifyingspecifictraining higherlevelsofabstraction. In Composite CFS, welookat
examples. For example, the signal s (the output of the rarepatternsincombinationsofsignalsfeedingaparticular
0
x=x block) is1(true) forthetrainingexamplex and0 gateandperturbtheoutputofthatgate (byflippingit) when
0 0
(false) forallothers. Inthiscase, wesay1 isararepattern ararecombinationisseenattheinputs. Anotherpossibility
fors sinces rarelytakesonthevalue1 onthetrainingset. istorandomizetheperturbationinsteadofalwaysflipping.
0 0
Formally, ifasignalsin Mtakesonthevaluevatmostl Inourexperiments, wefoundthesevariantstoproducere-
timesonthetrainingset S, wecallvanl-rarepatternfors. sults that are similar to those obtained from Simple CFS,
andsoweonlymentiontheminpassing.
This observation leads to the first of the two main ideas
behind CFS:Thepresenceofl-rarepatternssuggestsover-
fitting, and, therefore, poorgeneralizationsincetheyopen 3.Experimental Results
up the possibility that M has special logic to detect and
CFSImplementation. Ourimplementationofl-CFSworks
handlespecificexamples. Acountofrarepatterns, however,
onadirectedacyclicgraph G representingacombinational
doesnotdirectlytranslateintoametricforgeneralization
logic circuit where each node is either the constant 0, a
(withoutbuildingapredictivemodelofthat!). Furthermore,
primary input, a 2-input And gate, or a 2-input Xor gate.
althoughapatternmayberareitmayalsobeanobservabil-
An edge is either a direct connection or an inverter and
itydon’tcare (ODC), i.e., itmayhavenoinfluenceonthe
representsa Booleanfunctionintermsoftheprimaryinputs.
outputofthecircuit. Forexample, ifthesignalwiththerare
This is a variant of an And-Inverter Graph (Biere, 2007;
pattern only feeds into an And gate whose other input is
Chatterjee,2007), astandarddatastructureinmodernlogic
0 whentherarepatternappears, thenthevalueoftherare
synthesisusedtohandlecircuitswithhundredsofmillions
patterndoesnotmatterindecidingtheoutputofthecircuit.
ofnodes. Whileconstructingthesegraphs, wepropagate
Weaddressbothproblemswiththesecondmainideabehind constantsbutdonotextractcommonsub-expressions.
CFS:perturbedsimulationofatrainingexamplewherewe
Wemaketwopassesthroughthenodesof G intopological
simulate an example through M as usual, but when we
order starting from primary inputs. In the first pass, we
encounteral-rarepattern, insteadofpropagatingittothe
simulatethetrainingsetthrough G toobtainthecountsof
fanouts (i.e., gatesthatdependonthissignal), weperturbthe
differentpatternsinthecircuit. Inthesecondpass, weuse
patternandsimulatethefanoutswiththeperturbedpattern.
thecountsfromthefirstpasstoperturbthel-rarepatterns.
A natural perturbation is to propagate the opposite value
In Simple CFS, this boils down to replacing signals that
instead of the rare pattern. In our running example, this
takeonavalueof0 onmostexampleswiththeconstant0
corresponds to propagating a 0 instead of 1 for signal s
0 signalandlikewisefor1. CFSthusrunsinlineartimein
tothemultiplexerwhensimulatingthetrainingimagex .
0 thesizeofthegraphandthetrainingdata. Forperformance,
Inthismanner, wepreventthemodelfromidentifyingx
0 the simulations are done in a bit parallel manner for all
andweseethattheoutputforx isnolongernecessarilyy .
0 0 trainingexamplesatthesametime. Toavoidrunningout
Wecallthismodifiedsimulationprocedurel-counterfactual
of memory, we use reference counting to recycle storage
simulationorl-CFSforshort.
forintermediatesimulatedvalueswhentheyarenolonger
Weperformperturbedsimulationforeachtrainingexample needed. A typical run of l-CFS in our experiments takes
inturnandmeasuretheresultingaverageaccuracyoverthe lessthan10 minutesona3.7 GHz Xeon CPUandlessthan
trainingset. Wecallthisquantitythetrainingaccuracyob- 2 GBof RAM.
tainedthroughl-CFS.Inourrunningexampleofthelookup
Benchmark Problem. Whilethediscussionfromthepre-
table, itiseasytoseethatthetrainingaccuracyobtained
vious section shows how CFS can discover overfit when
through1-CFSisnobetterthanrandomchance (sinceeach
the model is a simple lookup table, it is not clear if CFS
trainingexampleismappedtoclass‘0’under1-CFS).Now,
wouldbeeffectiveonneuralnetworkstrainedwithstochas-
since random chance is what one would expect to be the
tic gradient descent (SGD). To answer this question, we
generalizationofthelookuptable (i.e., itsaccuracyon D),
trained3 neuralnetworksfor MNISTin Tensor Flow (Keras)
itistemptingtoconjecturethat1-CFStrainingaccuracyis
andcompiledthemdownintocombinationallogiccircuits.
agoodestimateofaccuracyon D. Althoughthatisnotthe
All3 networkshavethesamearchitecture: aninputlayer
caseasweshallseeempiricallyin Section3, wefindthatthe
of size 784 (i.e., 28 × 28), 3 fully connected Re LU lay-
differenceintrainingaccuracybetweennormalsimulation
erswith256 nodeseach, andafinalsoftmaxlayerwith10
andl-CFSisagoodmeasureofthedegreeofoverfitof M.
outputs. (Thus, thetotalnumberoftrainableparametersis

### 7. Circuit-Based Intrinsic Methodsto Detect Overfitting
335,114.) Wealsoperformedso...

Circuit-Based Intrinsic Methodsto Detect Overfitting
335,114.) Wealsoperformedsomeexperimentswith Fash- Itisremarkablethatevenwhenaneuralnetworkisrepre-
ion MNIST(Xiaoetal.,2017) andtheresultsaresimilar. sentedataverylowlevelasalogiccircuit, relativeoverfit
canbedetectedusinganintuitivealgorithmwithnohyper-
Thefirsttwonetworks (nn-real-2 andnn-real-100),
parameterstotune.
weretrainedonthe MNISTtrainingsetfor2 epochsand
100 epochsrespectively. Theygettotraining (top-1) accu- Expt.2:Impactof Architecture. Therearemanydifferent
raciesof97%and99.90%respectively. Thethirdnetwork ways in which a neural network can be compiled down
(nn-random) wastrainedonavariantof MNISTwhere intologicgates. In Expt. 1, wemadecertainarchitectural
the output labels in the training set are permuted pseudo- choicesforthecircuit, butwhatifwehadchosendifferently?
randomly and trained for 300 epochs to get to a training To evaluate that, here we replace the multipliers used in
accuracy of 91.27%.2 In all cases, we used the ADAM Expt. 1 with array multipliers (i.e., multipliers based on
optimizerwithdefaultparametersandbatchsizeof64. As the elementary algorithm for multiplication). Figure 2 b
expected, nn-real-2 is the least overfit and gets to a showstheresultant CFScurves (withdashedlines) aswell
validationsetaccuracyof97%(i.e., hasanegligiblegen- astheoriginalcurves (solidlines) forreference. Thecurves
eralizationgap), nn-real-100 ismoreoverfitgettingto donotcoincideindicatingthattheresultof CFSdepends
avalidationsetaccuracyof98.24%(agapof1.66%), and onthestructureofthecircuitandnotjustonthefunction
finally, the validation accuracy of nn-random is 9.73% implementedbythecircuit (sincethefunctionisthesame
(i.e., closetochance) confirmingthatitishorriblyoverfit. inbothcases). However, forthesamechoiceofarchitecture,
wefindthatthefalloffin CFScurvesareagainindicativeof
Conversionto Logic Circuits. Thisisdonebygenerating
thedegreeofoverfit.
logic subcircuits composed of 2-input And or Xor gates
and inverters for each of the operations in the neural net- Expt. 3: Impactof Choiceof Primitives. Evenatthelow-
work. Weights and activations are represented by signed estlevelofabstraction, wecanchoosewhatprimitivesto
8-bitand16-bitfixedpointnumbersrespectivelywith6 bits workwith. Toseehowthischoiceimpacts CFScurves, in
reservedforthefractionalpart. (Weightsfromtrainingare thisexperimentwedisallow Xorgatesasprimitives (thus
clamped to [−2.0,2.0) before conversion to fixed point.) requiringthatonly Andgatesandinvertersbeused). Fig-
Eachmultiply-accumulateunitmultipliesan8-bitconstant ure2 cshowstheresulting CFScurves (dashed) aswellas
(theweight) witha16-bitinput (theactivation) andaccumu- theoriginals (solid). Onceagain, weseethatthecurvesdo
latesin24 bitswithsaturation. Theconstantmultiplications notcoincideindicatingthatthechoiceofprimitivesmatters
are done by finding a minimal combination of bit-shifts but for the same choice of primitives, the falloff in CFS
(multiplicationsbypowersof2) andadditionsorsubtrac- curvesareagainindicativeofthedegreeofoverfit.
tions. Forexample,5×uisimplementedas4×u+uand
Expt. 4: Count of Rare Patterns. Figure 2 d shows for
11×uas16×u−4×u−u. Re LUsareimplementedwith
eachvalueofl, howmanyexampleshavenol-rarepatterns,
acomparatorandamultiplexer. Theoutputsofeachnet-
i.e., cannotbepossiblyaffectedbyl-CFS.Weobservein
workarethe10 signed16-bitactivationsbeforethesoftmax.
particular, that for nn-random, there are 54,094 exam-
Whenevaluatingaccuracy (with CFSorwithout) wepick
ples (about 90% of the training set) that have no 1-rare
theclasscorrespondingtothelargestofthe10 activations
patterns. Thisisinsharpcontrasttoasimplelookuptable
(top-1 accuracy). The resulting logic circuits have 35 to
whereeveryexamplewouldhavea1-rarepattern, andin
52 million Andor Xorgatesand5500 to6000 logiclevels.
factslightlymorethannn-real-100. Comparingthese
(Thesesizesalongwiththeneedtofitrandomdatadictated
curvesofcountstothe CFScurvesin Figure2 aindicates
thechoiceofarchitectureandbenchmark.)
thatperturbationisanimportantpartof CFSandthatrarity
Expt. 1: Effectof Simple CFS.Figure2 ashowsthetrain- byitselfisacrudermeasureofoverfittingsinceitmaynot
ingaccuraciesobtainedthroughl-CFSforeachofthethree beobservable.
networksaslvariesfrom1 to1024(whichisabout1.7%of
Expt. 5: Random Forests. Since CFS works on the cir-
thenumberoftrainingexamples). Wecalltheseplots CFS
cuit level, it can check random forests for overfit. Two
curves. Asl increases, i.e., asmorepatternsbecomerare
randomforestsweretrainedusingversion0.19.1 of Scikit-
and get perturbed, the accuracy falls eventually reaching
learn (Pedregosaetal.,2011). Eachforesthas10 treesand
chance. However, itisinterestingthatthedropinaccuracy
is trained using the default settings, except for bootstrap-
ishighestfornn-random (e.g., atl=64, thedropisabout
ping (toavoidnon-uniformweightsduringinference). The
45%), somewhat less for nn-real-100 (20%) and the
firstforest (rf-real) wastrainedon MNISTwhereasthe
leastfornn-real-2(1.4%). Thusthefalloffinaccuracy
secondforest (rf-random) wastrainedon MNISTwith
withlisanindicatorofthelevelofoverfitofanetwork.
theoutputlabelspseudo-randomlypermuted (asbeforewith
2 Whileevaluatingrandomfortrainingaccuracy (withorwith- nn-random). Bothforestsreachperfecttrainingaccuracy.
out CFS), thepermutedlabelsareused. rf-realgets95.58%validationaccuracywhereasasex-

### 8. Circuit-Based Intrinsic Methodsto Detect Overfitting
1.0
0.8
0.6
0.4
0.2
0.0
0 2...

Circuit-Based Intrinsic Methodsto Detect Overfitting
1.0
0.8
0.6
0.4
0.2
0.0
0 2 4 6 8 10
log2(l)
SFC-l
htiw
ycarucca
gniniart
1.0
0.8
0.6
0.4
nn-real-2 0.2
nn-real-100
nn-random
0.0
0 2 4 6 8 10
log2(l)
(a)
SFC-l
htiw
ycarucca
gniniart
nn-real-2
nn-real-2 w/ array
nn-real-100
nn-real-100 w/ array
nn-random
nn-random w/ array
(b)
1.0
0.8
0.6
0.4
0.2
0.0
0 2 4 6 8 10
log2(l)
SFC-l
htiw
ycarucca
gniniart
60000
50000
40000
30000
nn-real-2 20000
nn-real-2 w/o xors
nn-real-100 10000
nn-real-100 w/o xors
nn-random
nn-random w/o xors 0
0 1 2 3 4 5 6
log2(l)
(c)
snrettap
erar-l
on
/w
selpmaxe
.mun
nn-real-2
nn-real-100
nn-random
(d)
1.0
0.8
0.6
0.4
0.2
0.0
0 2 4 6 8 10 12
log2(l)
SFC-l
htiw
ycarucca
gniniart
1.0
0.8
0.6
0.4
rf-real
rf-random
nn-real-2 0.2
nn-real-100
nn-random
0.0
30 25 20 15 10 5
log2(p)
(e)
esion
teknalb
htiw
ycarucca
gniniart
rf-random
rf-real
nn-real-2
nn-real-100
nn-random
(f)
1.0
0.8
0.6
0.4
0.2
0.0
0 2 4 6 8 10 12
log2(l)
SFC-l
htiw
ycarucca
gniniart
1.0
epochs = 1
epochs = 2
epochs = 5 0.8
epochs = 10
epochs = 20
epochs = 50 0.6
epochs = 100
epochs = 200
0.4
0.2
0.0
30 28 26 24 22 20 18
log2(p)
(g)
esion
teknalb
htiw
ycarucca
gniniart
epochs = 1
epochs = 2
epochs = 5
epochs = 10
epochs = 20
epochs = 50
epochs = 100
epochs = 200
(h)
Figure2: Theresultsofexperimentsin Section3. Plot (a) showsthe CFScurvesfor3 networkswithdifferentamountsof
overfitting;(b) and (c) showtheimpactofthechoiceofmultipliersandofprimitivelogicgatesrespectivelyon CFScurves;
(d) showshowmanyexamplesareunaffectedbyl-CFSsincetheydonothaveanyl-rarepatterns;(e) shows CFScurvesfor
randomforests;(f) showstrainingaccuracieswhenasignalisrandomlyflippedwithprobabilityp;and (g) and (h) showthe
differencesbetween CFSandrandomflipsrespectivelyfor8 networkstrainedonadatasetwithhighlabelnoise.

### 9. Circuit-Based Intrinsic Methodsto Detect Overfitting
pectedrf-randomgetsnobetter...

Circuit-Based Intrinsic Methodsto Detect Overfitting
pectedrf-randomgetsnobetterthanchance.(rf-real
has about 14 K nodes per tree whereas rf-random has
about70 Knodespertree.)
Theforestsarecompileddowntocircuitsinastraightfor-
wardmanner. Eachtreeiscompiledseparatelyandproduces
1016-bitoutputs (oneperclass). Thecorrespondingoutputs
areaddedacrossall10 treesandtheclassoutputbythefor-
estcorrespondstotheclasswiththemaximumvalue. Each
internalnodeinatreemapstoamultiplexercontrolledbya
8-bitcomparatortoimplementthethreshold, andeachleaf
node corresponds to 10 16-bit constants representing the
numberofexamplesthatoccupyeachclassinthatleaf (thus
mostentriesarezero). Thecircuitforrf-realhasabout
700 Knodeswhereasrf-randomhas3 Mnodes. Bothare
lessthan250 logiclevelsdeep. (Thesearemuchsmaller
thanthecircuitsfortheneuralnetworks.)
Figure2 eshowsthe CFScurvesforthetworandomforests.
Onceagain, weseethattheoverfitmodel (rf-random)
degrades faster than the model with better generalization
(rf-real) confirmingthat CFSiseffectiveevenformod-
elsthatarefundamentallydifferentfromneuralnetworks.
However, itisinterestingtoseethatifwecompareacross
modelfamiliesi.e. betweentheneuralnetworksfrom Expt.
1(repeatedin Figure2 eforconvenience) andtherandom
forests, CFS is not effective at distinguishing overfit. In
particular, nn-real-2 whichisnotoverfitdegradesmore
rapidlythatrf-randomwhichishighlyoverfit. Wedis-
cussthisingreaterdetailin Section4.
Expt. 6: Blanket Noise. CFSmaybeseenasaddingatar-
getednoise. Here, instead, weaddblanketnoisebysimulat-
1.0
0.8
0.6
0.4
0.2
0.0
0 2 4 6 8 10 12
log2(l)
SFC-l
htiw
ycarucca
gniniart
ingthetrainingsetwhilerandomlyflippingthenodevalues
withprobabilityp. Aspvariesfrom2−30 to2−5, theresult-
ingnoisecurves (Figure2 f) aresimilartothe CFScurves
(Figure 2 e). However, the more overfit nn-real-100
does not fall faster than nn-real-2. With CFS, these
curvesarewellseparated, andthegapbetweenneuralnets
andforestsismuchlarger.
Expt. 7: Sensitivity. Tobetterunderstandthesensitivity
differencebetween CFSandblanketnoise, wetrained38
neuralnetworks (withthesamearchitectureasbeforebut
different number of epochs) on MNIST with exactly one
halfoflabelsrandomized (somaximumaccuracypossible
is about 55%). We show the CFS curves and the noise
curvesfor8 representativenetworksin Figures2 gand2 h
respectively. Note the crossover of the CFS curves that
indicatesalargerfalloffforoverfitnetworkscomparedto
the more uniform degradation of the noise curves. It is
fascinating that all the CFS curves cross over at a single
pointwithanaccuracyofabout50%. Thisisdiscussedin
Section4.
Expt. 8: Composite CFS. For completeness, Figure 3
shows the results of running Composite CFS instead of
Simple CFS in the setup of Expt. 7. We see that these
curvesareverysimilartothoseof Simple CFSin Figure2 g.
4.Discussion
Structure Dependence. Expt. 2 and3 showthattheresults
of Simple CFSdependnotjustonthefunctionbutonthe
structureofthecircuit. (Other CFSvariantsweinvestigated
showthisbehavioraswell.)Asmallexampleprovidessome
insight. Considerthe Booleanfunctionf (a, b, c)=aeval-
uatedonthetrainingsetcomprisingthefull Booleancube
(i.e., all 8 combinations of a, b, c ∈ {0,1}). In addition
to the direct implementation, f can also be implemented
epochs = 1 (redundantly) asa·b·c+a·¬b·c+a·b·¬c+a·¬b·¬c.
epochs = 2
epochs = 5 Itiseasytoseethatunder1-CFS, thedirectimplementation
epochs = 10
epochs = 20 is unchanged (there are no 1-rarepatterns) butthe redun-
epochs = 50
dantimplementationmapstoconstant0(theoutputofeach
epochs = 100
epochs = 200 conjunctionis1 onlyonce).
Althoughthisisnotaproblemwhenthecompilationpro-
cesscanbecontrolled, thisisbadnewsintheadversarial
setup. Agoodmodelwithapoorimplementationmayshow
steeperdegradationunder CFSthanamoreoverfitmodel
with better implementation (c.f. nn-real-2 with array
Figure 3: This plot shows the result of repeating Expt. 7 v/snn-real-100 without Xorsatl =256). Ideally, we
butusing Composite CFS(insteadof Simple CFS) where wouldliketofindavariantof CFSthatdoesnotdependon
wecomputeandperturbrarepatternsacrossallthesignals structurebutonlyonthefunction.3 Intheabsenceofthat
thatfeedintoalogicgate, i.e., acrossallthefaninsofagate
3 Inprinciple, onecouldcanonizethecircuitstructurebefore
(insteadofcomputingandperturbingrarepatternsforeach
applying CFS, saybybuilding Reduced Order Binary Decision Di-
signalorfaninindependently). Theresultswith Composite agrams (ROBDDs), butthatwouldbecomputationallyprohibitive.
CFSareverysimilartothoseof Simple CFS(Figure2 g). Alternatively, onecouldlightlyoptimizethecircuitbefore CFS

### 10. Circuit-Based Intrinsic Methodsto Detect Overfitting
1.0
0.8
0.6
0.4
0.2
0.0
0 2...

Circuit-Based Intrinsic Methodsto Detect Overfitting
1.0
0.8
0.6
0.4
0.2
0.0
0 2 4 6 8 10 12
log2(l)
SFC-l
htiw
ycarucca
gniniart
1.0
0.8
0.6
epochs = 1
epochs = 2
epochs = 5 0.4
epochs = 10
epochs = 20
epochs = 50 0.2
epochs = 100
epochs = 200
0.0
0 2 4 6 8 10 12
log2(l)
(a)
SFC-l
htiw
ycarucca
gniniart
epochs = 1
epochs = 2
epochs = 5
epochs = 10
epochs = 20
epochs = 50
epochs = 100
epochs = 200
(b)
Figure4: Theseplotsshowtheresultofrepeating Expt. 7 whenwecorruptonlyathirdofthelabelsin (a)MNISTandin (b)
Fashion-MNIST.Inbothcases, justasin Figure2 gweseethatallthecurvespassthroughapointclosetothemaximum
achievableaccuracy (around2/3). Thereasonforthisisdiscussedin Section4 under Generalizationin Deep Learning.
ideal, weviewthestructureofthecircuitasacertificateof thoughunbalancedhavenorarepatterns.
howwellthedatasetislearnt, andmakeit Merlin’srespon-
Thisexamplesuggestsawaytobreak CFS.If Merlinwanted
sibility to find and present the most convincing structure.
tobuildalookuptable, butprevent Arthurfromidentifying
Fromthisperspective, intheaboveexample, thedirectim-
it as such (by applying say 1-CFS), instead of building
plementationoff (whichisnotimpactedby1-CFS) ismore
thecircuitin Figure1, Merlincouldbuildadecision (i.e.,
convincingthantheredundantimplementationoff (which
multiplexer) treebyrecursivelysplittingononevariable (i.e.,
isseverelyimpacted).
onecomponentofx) atatimebypickingacomponentthatis
Adversarial Attack on CFS. Based on the discussion balanced (i.e., takeson0 and1 valuesroughlyequallyoften)
aboveitiseasytodesignawaytoarbitrarilydegradethe andleadstobothbranchesalsobeingbalanced (i.e., have
performance of a circuit under CFS. But is the opposite noclasswitheithertoofewortoomanyexamples, although
possible? Can Merlinfitanarbitraryfunctionontheinputs havingnoexamplesofaclassorhavingonlyexamplesof
but compile it down to a circuit which does not degrade oneclassisfineandexpectedattheleafnodes). Inother
under CFS? Expt. 5. offers a clue. The overfit model words, Merlin could use a procedure similar to the usual
rf-randomfittedonrandomlabelsfallsoffmoreslowly decisiontreeconstructionprocedures, butwiththeimportant
than nn-real-2 which generalizes well. What is go- differenceoffavoringbalancedsplitsinsteadofunbalanced
ing on? The short answer is that although each tree in splits.(Theunbalancedsplitsarelikelywhy CFSiseffective
rf-randomisextremelyoverfitwithmostleavescontain- totheextentitisonrandomforests.)
ingonlyasingleexample, thecircuitnodeshavefewrare
Finally, assuggestedbyareviewerofthispaper, itwouldbe
patternsduetotheobservabilitydon’tcaresintroducedby
interestingtoextendthisidea (ultimatelybasedon Shannon
themultiplexers.
decompositionandco-factoring) toanalgorithmthatcan
Againasimpleexampleisinstructive. Letf betheparity amplifythecountofrarepatternsinagivencircuitwithout
functiononnbits, i.e., f (x , x ,..., x )=x ⊕x ...⊕ changingitsfunctionality.
1 2 n 1 2
x . Consider an tree implementation of this function ob-
n Comparisonwith Blanket Noise. Basedon Expt. 6 and
tained using Shannon decomposition which has a multi-
7, webelievethatblanketnoiseislesssensitivethan CFS.
plexeratthetopcontrolledbyx andwith1⊕x ⊕x ...⊕
1 2 3 Our results here add to the extensive literature on noise,
x andx ⊕x ...⊕x asitsdatainputs. Ifthetraining
n 2 3 n generalizationandfaulttoleranceinneuralnetworks (e.g.,
set is the full Boolean cube, i.e., {0,1}n it is easy to see
see Bernieretal.(2001) andthereferencestherein) byex-
thattherearenol-raresignalsforl<<nsinceeachinput
tendingthemtothecircuitlevel (wheredistinctionsbetween
tothemultiplexerisbalanced, i.e., hasequalnumberof0 s
activationorweightnoise, oradditiveormultiplicativenoise
and1 s. Since Shannondecompositionisrecursive, asimilar
disappear) and to other model families such as random
argumentholdsforthelowerlevelsofthetreeuntilweget
forests. Furthermore, Expt. 6 presents a direct compari-
to the leaves which are constant 0 and constant 1 which
sonofthefault-toleranceofneuralnetsandforestswhere
butthatmaynotbeenough. forestsareseentobeabout1000 xmorefault-toleranttobit

### 11. Circuit-Based Intrinsic Methodsto Detect Overfitting
flips. Thisislikelyduetothe...

Circuit-Based Intrinsic Methodsto Detect Overfitting
flips. Thisislikelyduetotheredundancyfromensembling. obviouslyapply. Wehavenotstudiedifnormalizedmargin
Italsosuggeststhatnoise-basedintrinsicmethodscouldbe can be exploited by an adversary. Similarly, most mea-
easilyfooledbyanadversarybyaddingredundancy. suresbasedontheshallownessofminimaarenotadequate
sincetheyarenotscale-invariant (Dinhetal.,2017) andwe
Generalization in Deep Learning. Why do neural nets
havenotinvestigatedifmorerecentworkonscale-invariant
trainedwith SGDgeneralizewhentheyhavesufficient (ef-
measures (Rangamani et al., 2019) can be exploited. In
fective) capacity to memorize their training set? This is
comparison to these and other approaches for neural net-
anopenresearchquestion (Zhangetal.,2017;Arpitetal.,
works (Aroraetal.,2018;Neyshaburetal.,2018), CFSis
2017; Bartlett et al., 2017; Arora et al., 2018; Neyshabur
fundamentallymorediscrete, whichmakesitapplicableto
et al., 2018). Expt 5. shows that this question is not lim-
alargerclassofmodels. However, incontrasttotheother
itedtonets—thesamecouldbeaskedforrandomforests
approaches, wedonothaveanytheoreticalboundsyetwhile
aswell. One (informal) answerforforestsisthatdecision
ourresultsindicatethatwithoutfurtherrefinementsto CFS
treeconstructionprocedureslookforcommonpatternsbe-
itself, anygeneralizationboundsfroml-CFSwouldlikely
tweenexamples. Whenexamplessharecommonality, they
bevacuousinpractice.
arecombinedintocommonleafnodesandthemodelgen-
eralizes, whereas when there is little commonality, each
exampleisitsownleaf (sothetrainingsetisfitwell) butthe 5.Conclusionand Future Work
modelfailstogeneralize. Couldthesamethingbegoingon
Ourmainresultisthat CFSbasedonaddingsmallamounts
with SGDandnetworks? CFSonnn-randomand Expt4.
oftargetednoiseatthelogiccircuitlevelcandetectoverfit.
providedirectevidencethatevenonrandomdata, netsdo
This is remarkable because at this level of representation
not“brute-forcememorize”butidentifycommonpatterns
we have lost most aspects of the structure of the model,
in the data (a question raised in Zhang et al. (2017) and
such as the distinction between weights and activations,
discussedin Arpitetal.(2017)).
orevenwhetherthemodelisaneuralnetwork, arandom
Inthiscontext, itisinterestingtostudywhythe CFScurves forest, oralookuptable. Furthermore, variationssuchas
forthedifferentnetworksin Expt7. intersectatacommon perturbingonlyrarepatternsinsinglesignalsoracrossthe
pointcorrespondingapproximatelytotheachievableaccu- faninsofagateleadtoqualitativelysimilarresultsandthere
racy. Thisholdsforotheramountsoflabelrandomization arevariants (suchas Simple CFS) thatarenaturallyfreeof
e.g., see Figure4 forthe CFScurveswhenonethirdofthe hyper-parameters.
labelsarecorrupted. Roughlyhalfoftheexamplesareeasy
Bystudyingrarepatterns, wefindthat SGDdoesnotlead
sincetheyhavecorrectlabelsandarelearntinthefirstfew
to“bruteforce”memorization, butfindscommonpatterns
epochs. The remaining examples with corrupt labels are
(whethertrainingisdonewithrandomizedlabelsoractual
harder and learnt only in later epochs by the models that
labels), andneuralnetworksarenotunlikeforestsinthis
aretrainedlonger. With CFS, theaccuracyofthosemodels
regard. Byaddingblanketnoise, randomforestsarefoundto
breaksdownearliersincethehardexampleshavemorerare
beabout1000 xmoreresilienttonoisethanneuralnetworks
patterns than the easy examples, and the accuracy on the
whichcouldbeusefulwhenimplementingmachinelearning
easyexamplesthusformsalimitingcurveforallmodels.
systemswithunreliablelowlevelcomponents.
Thisprovidesmore (anddirect) evidencefortheclaimin
Arpit et al. (2017, §1) that “SGD learns simpler patterns Thereareseveraldirectionsforfuturework. Weanalyzeflat
firstbeforememorizing.” Furthermore, wecouldidentify circuits, but with a clever implementation that constructs
“simplerpatterns”asexamplesthathavefewerrarepatterns thecircuiton-the-flyfromahigherlevelspecification, the
and “memorizing” as what is required for examples that computations can scale to larger models. We could also
havemorerarepatterns. Thus, learningsimplerpatternsand apply CFSathigherlevelsofabstraction (perhapsaspartof
memorizationarenotfundamentallydifferentbutlieattwo themodelevaluationprocessinframeworks, suchas Scikit
endsofaspectrum. Learnand Tensorflow Estimators) thoughatthatlevelthere
are moredegrees of freedomin theimplementation (e.g.,
Related Work. Onemaybetemptedtoviewmarginasan
whatkindofnoisetoadd).
intrinsicmeasuretoestimatethegeneralizationofamodel.
However, when we have models with intermediate repre- Thenotionofrarityconsideredinthisworkmayberegarded
sentations, the notion of margin by itself is not adequate as a local notion, since we count the patterns at a signal
sinceanadversarycanoverfittoafavorableintermediate (or a group of signals in the case of Composite CFS) in
representationthatiseasilylinearlyseparated (butotherwise isolation. It is possible to extend this notion to a global
arbitrary). However, recentworkinthisarea (e.g., Bartlett notion of rarity by propagating occurrence counts along
etal.(2017)) hasfocusedonmarginsnormalizedbyspectral withsignalvaluesduringsimulation. Inthecorresponding
complexity (i.e., ameasurerelatedtothe Lipschitzconstant CFS, a pattern is perturbed when its global rarity drops
ofthenetwork) andinthatcasetheaboveargumentdoesnot

### 12. Circuit-Based Intrinsic Methodsto Detect Overfitting
Dwork, C., Feldman, V., Har...

Circuit-Based Intrinsic Methodsto Detect Overfitting
Dwork, C., Feldman, V., Hardt, M., Pitassi, T., Reingold, Rangamani, A., Nguyen, N.H., Kumar, A., Phan, D., Chin,
O., and Roth, A. Thereusableholdout: Preservingvalid- S.H., and Tran, T.D. AScale Invariant Flatness Mea-
ity in adaptive data analysis. Science, 349(6248):636– sure for Deep Network Minima. ar Xiv e-prints, art.
638, 2015. ISSN 0036-8075. doi: 10.1126/science. ar Xiv:1902.02434, Feb2019.
aaa9375. URL https://science.sciencemag.
Xiao, H., Rasul, K., and Vollgraf, R. Fashion-mnist: anovel
org/content/349/6248/636.
imagedatasetforbenchmarkingmachinelearningalgo-
Le Cun, Y.and Cortes, C.MNISThandwrittendigitdatabase. rithms. ar Xiv, 2017. URL https://arxiv.org/
http://yann.lecun.com/exdb/mnist/, 2010. URL http: abs/1708.07747.
//yann.lecun.com/exdb/mnist/.
Zhang, C., Bengio, S., Hardt, M., Recht, B., and Vinyals, O.
Neyshabur, B., Li, Z., Bhojanapalli, S., Le Cun, Y., and
Understandingdeeplearningrequiresrethinkinggeneral-
Srebro, N. Towards understanding the role of over-
ization. In Proceedingsofthe International Conference
parametrization in generalization of neural networks.
on Learning Representations ICLR,2017.
Co RR, abs/1805.12076,2018. URLhttp://arxiv.
org/abs/1805.12076. Zielinski, P., Krishnan, S., and Chatterjee, S. Weak and
stronggradientdirections:Explainingmemorization, gen-
Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V.,
eralization, and hardness of examples at scale. Ar Xiv,
Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P.,
abs/2003.07422,2020. URLhttps://arxiv.org/
Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cour-
abs/2003.07422.
napeau, D., Brucher, M., Perrot, M., and Duchesnay, E.
Scikit-learn: Machine learning in Python. Journal of
Machine Learning Research,12:2825–2830,2011.

### 13. Circuit-Based Intrinsic Methods to Detect Overfitting
Satrajit Chatterjee1 Alan ...

Circuit-Based Intrinsic Methods to Detect Overfitting
Satrajit Chatterjee1 Alan Mishchenko2
Abstract knowledge, suchas, theperformanceofthemodelonexam-
plesheldoutfromthetrainingprocess, detailsoftheprocess
Thefocusofthispaperisonintrinsicmethodsto
usedtofindthemodel (e.g., multiplehypothesistestingwith
detectoverfitting. Byintrinsicmethods, wemean
registration), orlimitationsofthefunctionfamilytowhich
methodsthatrelyonlyonthemodelandthetrain-
themodelbelongs (e.g., VCdimension, Rademachercom-
ingdata, asopposedtotraditionalmethods (we
plexity) orthoseofthesizeoftheparameterspaceofthe
callthemextrinsicmethods) thatrelyonperfor-
model (e.g., Akaike Information Criterion).
manceonatestsetoronboundsfrommodelcom-
plexity. Weproposeafamilyofintrinsicmethods Weclassifymethodsrelyingontheknowledgeofthefunc-
called Counterfactual Simulation (CFS) which tionfamilyasextrinsicbecauseoftentheinformationrel-
analyze the flow of training examples through evant to overfitting is not directly represented in a given
themodelbyidentifyingandperturbingrarepat- model. For example, a model could have been found by
terns. Byapplying CFStologiccircuitswegeta searchingamuchsmallerspacethanthatimpliednaïvelyby
methodthathasnohyper-parametersandworks thefunctionfamilytowhichitbelongs, duetoeitherexplicit
uniformlyacrossdifferenttypesofmodelssuch regularizationortheregularizationimplicitintheoptimiza-
as neural networks, random forests and lookup tionorsearchprocedure. Conversely, thegivenmodelcould
tables. Experimentally, CFScanseparatemodels havebeenfoundbysearchingamuchlargerspaceofmod-
with different levels of overfit using only their elsthroughhyper-parametersearch, orbypickingthebest
logiccircuitrepresentationswithoutanyaccess modelfromanumberofdifferentmodelfamilies, butthe
tothehighlevelstructure. Bycomparinglookup specificfinalmodelitselfdoesnotcarryanyvestigesofthe
tables, neural networks, and random forests us- largerspacethatwassearchedover. Bothofthesesituations
ing CFS, wegetinsightintowhyneuralnetworks arecommoninmodernmachinelearning.
generalize. Inparticular, wefindthatstochastic
Intrinsicmethodsareofpracticalinterestsincewithmodern
gradient descent in neural nets does not lead to
deeplearningmodels, wefindthatextrinsicestimatesbased
“bruteforce”memorization, butfindscommonpat-
onmodelcomplexityaretypicallyvacuoussincethesemod-
terns (whetherwetrainwithactualorrandomized
elsarepowerfulenoughtofitarbitrarydata (Zhangetal.,
labels), andneuralnetworksarenotunlikeforests
2017). Consequently, practitionersresorttostudyingperfor-
in this regard. Finally, we identify a limitation
manceonaholdoutdataset (orcrossvalidation), butthisis
withourproposalthatmakesitunsuitableinan
unsatisfactoryforacoupleofreasons. First, thismeansthat,
adversarialsetting, butpointsthewaytofuture
inalowdatasetting, wecannotuseallthedatafortraining,
workonrobustintrinsicmethods.
but have to keep significant portions aside for validation
(e.g., seediscussionin Dietterich (1998)). Second, itmay
1.Introduction bedifficulttoensureapristineholdoutthatisnottouched
duringtheresearchprocessparticularlyiftheprojectislong
Thispaperconsidersmethodstodetectoverfittingofamodel running. Even with a few queries to the hold out during
based only on the model and the training data. In termi- theresearchprocess, itispossibletostartfittingtothehold
nologythatweintroduce, wecallsuchmethodsintrinsic, out (Dworketal.,2015).
incontrasttoextrinsicmethods, whichrelyonadditional
Intrinsicmethodsarealsointerestingfromatheoreticalper-
1 Google, Mountain View, California, USA 2 Department of spective. Imaginethatwehavesufficientcomputingpower
EECS, Universityof California, Berkeley, California, USA.Corre- to, say, enumerateallneuralnetworks (andtheirweights)
spondenceto:Satrajit Chatterjee<schatter@google.com>, Alan
up to a certain size. Among all the networks that fit the
Mishchenko<alanmi@berkeley.edu>.
datawell, intrinsicmethodscoulddistinguishbetweenthose
Proceedings of the 37 th International Conference on Machine networksthatgeneralizewellfromthosethatdonot, andwe
Learning, Online, PMLR119,2020. Copyright2020 bytheau-
thor (s).
0202
gu A
5
]GL.sc[
2 v19910.7091:vi Xra

### 14. Circuit-Based Intrinsic Methodsto Detect Overfitting
belowthegiventhreshold. Pre...

Circuit-Based Intrinsic Methodsto Detect Overfitting
belowthegiventhreshold. Preliminaryexperimentsshow References
that this more stringent notion of rarity is more effective
Arora, S., Ge, R., Neyshabur, B., and Zhang, Y. Stronger
indetectingoverfitinthecaseofrandomforests, andmay
generalization bounds for deep nets via a compression
offerawaytopartiallymitigatetheadversarialattackon
approach. Co RR, abs/1802.05296, 2018. URL http:
CFSoutlinedintheprevioussection.
//arxiv.org/abs/1802.05296.
Finally, basedoninsightsfromouranalysisof Simple CFS,
Arpit, D., Jastrzebski, S.K., Ballas, N., Krueger, D., Bengio,
wewouldliketocontinuethesearchforanintrinsicmethod
E., Kanwal, M. S., Maharaj, T., Fischer, A., Courville,
thatdoesnotdependonthemodelstructureandisadversar-
A.C., Bengio, Y., and Lacoste-Julien, S. Acloserlook
iallyrobust, ortoshowthatsuchamethoddoesnotexist,
at memorization in deep networks. In Proceedings of
evenforlearningtasksofpracticalinterest. Asageneral-
the34 th International Conferenceon Machine Learning,
izationofthisidea, itisinterestingtocontemplatelearning
ICML2017, Sydney, NSW, Australia,6-11 August2017,
algorithmsthatproducecertificatesofgeneralization, much
pp. 233–242, 2017. URL http://proceedings.
likea Booleansatisfiabilitysolvercanproduceacertificate
mlr.press/v70/arpit17 a.html.
ofsatisfiabilityorofunsatisfiability.
Postscript: The Theoryof Coherent Gradients. Theob- Bartlett, P.L., Foster, D.J., and Telgarsky, M.J. Spectrally-
servations in thispaper from applying CFS toneural net- normalizedmarginboundsforneuralnetworks. In Guyon,
works—particularly, theabsenceof“bruteforce”memo- I., Luxburg, U.V., Bengio, S., Wallach, H., Fergus, R.,
rizationwithrandomlabels, andtheexistenceofeasyand Vishwanathan, S., and Garnett, R. (eds.), Advances in
hardexamplesasdiscussedin Section4—inspiredthede- Neural Information Processing Systems 30, pp. 6240–
velopmentofatheorycalled Coherent Gradients (CG) that 6249.Curran Associates, Inc.,2017.
providesasimpleandintuitiveexplanationofgeneralization
Bernier, J., Ortega, J., Ros Vidal, E., Rojas, I., and Pri-
innetworkstrainedwith (stochastic) gradientdescent.
eto, A. A quantitative study of fault tolerance, noise
Thekeyideain CGisasfollows. Sincetheoverallgradient immunity, and generalization ability of mlps. Neural
g istheaverageofthegradientsoftheindividualtraining Computation, 12:2941–2964, 01 2001. doi: 10.1162/
examples, theremaybecertaincomponents (directions) ofg 089976600300014782.
whicharesignificantlystrongerthanothercomponentsdue
Biere, A. Aigerlibraryandtools, 2007. URLhttp://
tothegradientsofmultipleexamplesbeinginagreement
fmv.jku.at/aiger. [Online;accessed1-July-2020].
onthosecomponents, andtherebyreinforcingeachother.
Since the changes to the trainable parameters of the net-
Chatterjee, S. On Algorithms for Technology Map-
workareproportionaltothegradient, thebiggestchangesto
ping. Ph D thesis, EECS Department, Univer-
theparametersarebiasedtobenefitmultipleexamples, and
sity of California, Berkeley, Aug 2007. URL
thereforelikelytogeneralizewell (basedonastabilityargu-
http://www2.eecs.berkeley.edu/Pubs/
ment).However, iftherearenosuchstrongcomponents, i.e.,
Tech Rpts/2007/EECS-2007-100.html.
alltheper-examplegradientsareroughlyorthogonal, then
eachexampleisfittedindependently, andthiscorresponds Chatterjee, S. Coherentgradients: Anapproachtounder-
tomemorization (andpoorgeneralization, againfromasta- standing generalization in gradient descent-based opti-
bilityargument). Thus CGprovidesanuniformexplanation mization. In Proceedingsofthe International Conference
ofbothgeneralizationandmemorizationinneuralnetworks. on Learning Representations ICLR,2020. URLhttps:
//openreview.net/forum?id=rye FY0 EFw S.
Pleasesee Chatterjee (2020) and Zielinskietal.(2020) for
amoredetaileddevelopmentofthisidea, includingananal- Dietterich, T. G. Approximate statistical tests for
ysisofeasyandhardexamplesfromthisperspective, anda comparing supervised classification learning algo-
naturalmodificationtogradientdescentthatsignificantlyre- rithms. Neural Comput., 10(7):1895–1923, Oc-
ducesoverfittingbysuppressingthosegradientcomponents tober 1998. ISSN 0899-7667. doi: 10.1162/
thatarenotcommontomanyexamples. 089976698300017197. URLhttp://dx.doi.org/
10.1162/089976698300017197.
Acknowledgments
Dinh, L., Pascanu, R., Bengio, S., and Bengio, Y.
Thefirstauthorthanks Michele Covell, Ali Rahimi, Alex Sharp minima can generalize for deep nets. Co RR,
Alemi, Shumeet Baluja, Sergey Ioffe, Tomas Izo, Shankar abs/1703.04933, 2017. URL http://arxiv.org/
Krishnan, Rahul Sukthankar, and Jay Yagnik for helpful abs/1703.04933.
discussions. The second author was supported in part by
SRCContract2867.001.


---

## Raw Markitdown Extraction (full text)

|     |     | Circuit-Based | Intrinsic           |     | Methods                                          | to Detect | Overfitting |     |     |
| --- | --- | ------------- | ------------------- | --- | ------------------------------------------------ | --------- | ----------- | --- | --- |
|     |     |               | SatrajitChatterjee1 |     | AlanMishchenko2                                  |           |             |     |     |
|     |     | Abstract      |                     |     | knowledge,suchas,theperformanceofthemodelonexam- |           |             |     |     |
plesheldoutfromthetrainingprocess,detailsoftheprocess
Thefocusofthispaperisonintrinsicmethodsto
usedtofindthemodel(e.g.,multiplehypothesistestingwith
| detectoverfitting. |     | Byintrinsicmethods,wemean |     |     |     |     |     |     |     |
| ------------------ | --- | ------------------------- | --- | --- | --- | --- | --- | --- | --- |
0202 guA 5  ]GL.sc[  2v19910.7091:viXra methodsthatrelyonlyonthemodelandthetrain- registration),orlimitationsofthefunctionfamilytowhich
themodelbelongs(e.g.,VCdimension,Rademachercom-
ingdata,asopposedtotraditionalmethods(we
plexity)orthoseofthesizeoftheparameterspaceofthe
callthemextrinsicmethods)thatrelyonperfor-
model(e.g.,AkaikeInformationCriterion).
manceonatestsetoronboundsfrommodelcom-
plexity. Weproposeafamilyofintrinsicmethods Weclassifymethodsrelyingontheknowledgeofthefunc-
| called | Counterfactual | Simulation | (CFS) | which |     |     |     |     |     |
| ------ | -------------- | ---------- | ----- | ----- | --- | --- | --- | --- | --- |
tionfamilyasextrinsicbecauseoftentheinformationrel-
analyze the flow of training examples through evant to overfitting is not directly represented in a given
themodelbyidentifyingandperturbingrarepat-
|     |     |     |     |     | model. | For example, | a model | could have been | found by |
| --- | --- | --- | --- | --- | ------ | ------------ | ------- | --------------- | -------- |
terns. ByapplyingCFStologiccircuitswegeta searchingamuchsmallerspacethanthatimpliednaïvelyby
methodthathasnohyper-parametersandworks thefunctionfamilytowhichitbelongs,duetoeitherexplicit
uniformlyacrossdifferenttypesofmodelssuch
regularizationortheregularizationimplicitintheoptimiza-
as neural networks, random forests and lookup tionorsearchprocedure. Conversely,thegivenmodelcould
| tables. | Experimentally,CFScanseparatemodels |     |     |     |     |     |     |     |     |
| ------- | ----------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- |
havebeenfoundbysearchingamuchlargerspaceofmod-
with different levels of overfit using only their elsthroughhyper-parametersearch,orbypickingthebest
logiccircuitrepresentationswithoutanyaccess modelfromanumberofdifferentmodelfamilies,butthe
| tothehighlevelstructure. |     |     | Bycomparinglookup |     |     |     |     |     |     |
| ------------------------ | --- | --- | ----------------- | --- | --- | --- | --- | --- | --- |
specificfinalmodelitselfdoesnotcarryanyvestigesofthe
tables, neural networks, and random forests us- largerspacethatwassearchedover. Bothofthesesituations
ingCFS,wegetinsightintowhyneuralnetworks arecommoninmodernmachinelearning.
| generalize. | Inparticular,wefindthatstochastic |     |     |     |     |     |     |     |     |
| ----------- | --------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- |
Intrinsicmethodsareofpracticalinterestsincewithmodern
| gradient | descent | in neural | nets does | not lead | to  |     |     |     |     |
| -------- | ------- | --------- | --------- | -------- | --- | --- | --- | --- | --- |
deeplearningmodels,wefindthatextrinsicestimatesbased
“bruteforce”memorization,butfindscommonpat-
onmodelcomplexityaretypicallyvacuoussincethesemod-
terns(whetherwetrainwithactualorrandomized
elsarepowerfulenoughtofitarbitrarydata(Zhangetal.,
labels),andneuralnetworksarenotunlikeforests
in this regard. Finally, we identify a limitation 2017). Consequently,practitionersresorttostudyingperfor-
manceonaholdoutdataset(orcrossvalidation),butthisis
withourproposalthatmakesitunsuitableinan
|                     |     |                         |     |     | unsatisfactoryforacoupleofreasons. |     |     | First,thismeansthat, |     |
| ------------------- | --- | ----------------------- | --- | --- | ---------------------------------- | --- | --- | -------------------- | --- |
| adversarialsetting, |     | butpointsthewaytofuture |     |     |                                    |     |     |                      |     |
inalowdatasetting,wecannotuseallthedatafortraining,
workonrobustintrinsicmethods.
|     |     |     |     |     | but                                     | have to keep | significant | portions aside | for validation |
| --- | --- | --- | --- | --- | --------------------------------------- | ------------ | ----------- | -------------- | -------------- |
|     |     |     |     |     | (e.g.,seediscussioninDietterich(1998)). |              |             | Second,itmay   |                |
1.Introduction bedifficulttoensureapristineholdoutthatisnottouched
duringtheresearchprocessparticularlyiftheprojectislong
Thispaperconsidersmethodstodetectoverfittingofamodel running. Even with a few queries to the hold out during
based only on the model and the training data. In termi- theresearchprocess,itispossibletostartfittingtothehold
nologythatweintroduce,wecallsuchmethodsintrinsic,
out(Dworketal.,2015).
incontrasttoextrinsicmethods,whichrelyonadditional
Intrinsicmethodsarealsointerestingfromatheoreticalper-
| 1Google, |     |     |     | 2Department |     |     |     |     |     |
| -------- | --- | --- | --- | ----------- | --- | --- | --- | --- | --- |
Mountain View, California, USA of spective. Imaginethatwehavesufficientcomputingpower
EECS,UniversityofCalifornia,Berkeley,California,USA.Corre- to,say,enumerateallneuralnetworks(andtheirweights)
spondenceto:SatrajitChatterjee<schatter@google.com>,Alan up to a certain size. Among all the networks that fit the
Mishchenko<alanmi@berkeley.edu>.
datawell,intrinsicmethodscoulddistinguishbetweenthose
|                               |        | 37th          |                       |            | networksthatgeneralizewellfromthosethatdonot,andwe |     |     |     |     |
| ----------------------------- | ------ | ------------- | --------------------- | ---------- | -------------------------------------------------- | --- | --- | --- | --- |
| Proceedings                   | of the | International | Conference            | on Machine |                                                    |     |     |     |     |
| Learning,Online,PMLR119,2020. |        |               | Copyright2020bytheau- |            |                                                    |     |     |     |     |
thor(s).

Circuit-BasedIntrinsicMethodstoDetectOverfitting
| couldviewthemodelasacertificateofgeneralization.1     |     |     |     | In  |     |     |     |     |     |     |     |
| ----------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| addition,iftheintrinsicmethodwasefficient,itwouldmean |     |     |     |     | x   |     |     |     |     |     |     |
thatsupervisedlearning(andnotjustfittingthetrainingdata)
isinNP.Intrinsicmethodscanalsohelpshedlightonwhy
|     |     |     |     |     |     | =   |     | =   |     | =   |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
neural networks trained with stochastic gradient descent x x x
|     |     |     |     |     |     | n -1 |     | 1   | 0   |     |     |
| --- | --- | --- | --- | --- | --- | ---- | --- | --- | --- | --- | --- |
...
| generalizeinspiteoftheirlargecapacity.              |     |     | Somerecentanal- |     |     |      |     |     |     |     |     |
| --------------------------------------------------- | --- | --- | --------------- | --- | --- | ---- | --- | --- | --- | --- | --- |
|                                                     |     |     |                 |     |     | s    |     | s   |     | s   |     |
| ysesbasedonnormalizedmargin,curvature,etc.(Bartlett |     |     |                 |     |     | n -1 |     | 1   |     | 0   |     |
et al., 2017; Rangamani et al., 2019; Arora et al., 2018; 1000000000 0 0 0
| Neyshaburetal.,2018)maybeseenasintrinsicestimates |     |     |     |     |     |     |     |     |     |     | y   |
| ------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
forgeneralizationalbeitspecializedtoneuralnetworks. y 1 y 1 y 1
|     |     |     |     |     |     | n -1 |     | 1   | 0   |     |     |
| --- | --- | --- | --- | --- | --- | ---- | --- | --- | --- | --- | --- |
Intrinsicmethodscanbeconsideredinthecontextofapro-
| tocolinvolvingtwoagents. |     | LetS | beapublicdatasetdrawn |     |     |     |     |     |     |     |     |
| ------------------------ | --- | ---- | --------------------- | --- | --- | --- | --- | --- | --- | --- | --- |
Figure1: Acircuitimplementingalookuptablethatmem-
| fromadistributionD |     | thatgeneratessamplesinfrequently |     |     |     |     |     |     |     |     |     |
| ------------------ | --- | -------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
(e.g.,quarterlyfinancialstatementsandstockmarketreturns orizes the training examples (x i ,y i ) for 0 ≤ i < n. We
|     |     |     |     |     | observe | that in | this extreme | case of | overfitting, | there | are |
| --- | --- | --- | --- | --- | ------- | ------- | ------------ | ------- | ------------ | ----- | --- |
ofpubliccompanies,orpublichealthdataontreatmentsand
|                   |                                 |     |     |     | signalsinthecircuit(thes |     |     | )thatidentifyspecifictraining |     |     |     |
| ----------------- | ------------------------------- | --- | --- | --- | ------------------------ | --- | --- | ----------------------------- | --- | --- | --- |
| patientoutcomes). | SupposeArthurwantstobuildamodel |     |     |     |                          |     |     | i                             |     |     |     |
fromS butinsteadofdoingsohimself,heoutsourcesitto examples. For e.g., s 0 is 1 only when x = x 0 and 0 for
|                              |     |     |                      |     | all | other x (assuming | the | x are distinct). |     | We say | that 1 |
| ---------------------------- | --- | --- | -------------------- | --- | --- | ----------------- | --- | ---------------- | --- | ------ | ------ |
| Merlin,anuntrustedadversary. |     |     | Merlincomesbackwitha |     |     | i                 |     | i                |     |        |        |
|                              |     |     |                      |     |     | rare pattern      |     | s                |     |        |        |
modelMbutdoesnotdiscloseanydetailsofhismodeling is a for the signal 0 . Based on this exam-
process. HowcanArthurconvincehimselfthatMisnot ple,weproposethattheoccurrenceofrarepatternsduring
simulationofatrainingsetthroughamodelindicatesover-
| horriblyoverfit? | Forexample,Mcouldsimplybealookup |     |     |     |     |     |     |     |     |     |     |
| ---------------- | -------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
table built from S. Normally Arthur would evaluate M fitting,andinthiswork,weexploretowhatextentsuchrare
patternscanbeusedtodetectoverfittinginmorecomplex
onnewsamplesfromD,butinoursetup,Arthurdoesnot
modelssuchasneuralnetworksandrandomforests.
| haveanysamplesotherthanthoseinS |                                  |     | sincealltheexisting |     |     |     |     |     |     |     |     |
| ------------------------------- | -------------------------------- | --- | ------------------- | --- | --- | --- | --- | --- | --- | --- | --- |
| dataispublic.                   | Now, ifArthuronlyhasaccesstoMasa |     |                     |     |     |     |     |     |     |     |     |
blackboxandhecanonlyevaluateMonelementsofS,it
appearsthereislittlehecandotodistinguishagoodmodel a directed acyclic graph (DAG) of fixed or floating point
adders,multipliers,andpointwisenon-linearities.Finally,at
| from a lookup | table. | But can | Arthur do better | if he has |     |     |     |     |     |     |     |
| ------------- | ------ | ------- | ---------------- | --------- | --- | --- | --- | --- | --- | --- | --- |
accesstotheinternalsignalsintheimplementationofM? thelowestlevelofabstraction,wecandescribethestructure
Thisisthecentralquestionofthispaper. ofMasaDAGofprimitivelogicgatessuchas2-inputAnd
|     |     |     |     |     | gates | and inverters, | i.e., | as a combinational |     | logic | circuit. |
| --- | --- | --- | --- | --- | ----- | -------------- | ----- | ------------------ | --- | ----- | -------- |
Wetakeafirststeptowardsansweringthisquestionbystudy-
Inoursetup,itisnaturaltoworkatthislowestlevel,i.e.,
inganaturally-motivatedfamilyofintrinsicmethods,called
logicgatessinceitallowsdifferentkindsofmodelssuch
CounterfactualSimulation(CFS),andevaluatingtheiref-
aslookuptables,randomforests,andneuralnetworksall
| ficacyexperimentallyonabenchmarkproblem. |     |     |     | Themain |                              |     |     |                    |     |     |     |
| ---------------------------------------- | --- | --- | --- | ------- | ---------------------------- | --- | --- | ------------------ | --- | --- | --- |
|                                          |     |     |     |         | tobemappedintothesameformat. |     |     | Thus,Merlinneednot |     |     |     |
ideabehindCFSistoanalyzetheflowofthetrainingexam- discloseevenwhattypeofmodelhehasbuilt,butsimply
plesinSthroughthestructureofM.Thisisonlyafirststep
providesArthurwithacombinationallogiccircuitforthe
sinceCFS,althoughpromisinginpractice,hassignificant
model.
| limitations. | Inparticular,ourexperimentsshowthateven |     |     |     |     |     |     |     |     |     |     |
| ------------ | --------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
ifwecouldproveboundsbasedonCFS,theywouldnotbe Tomakethisconcrete,considertheMNISTimageclassifi-
tightenoughinanadversarialsetting. However,wehope cationproblem(LeCun&Cortes,2010)whichwewilluse
thatthispaperencouragesresearchtoovercometheselimi- asarunningexample. Disthedistributionofhandwritten
tationsortoshowthatnosuchmethodcanexist,especially digits and their classes. S is a sample from D of 60,000
|                                      |     |     |     |     | imagesx                         | andtheircorrespondinglabelsy |     |     | (thus,0≤i< |               |     |
| ------------------------------------ | --- | --- | --- | --- | ------------------------------- | ---------------------------- | --- | --- | ---------- | ------------- | --- |
| forlearningtasksofpracticalinterest. |     |     |     |     |                                 | i                            |     |     | i          |               |     |
|                                      |     |     |     |     | 60000)i.e.,theMNISTtrainingset. |                              |     |     | Eachx      | i is6,272bits |     |
wide(correspondingto28×28pixels×8bitsperpixel),
2.CounterfactualSimulation(CFS)
y
|     |     |     |     |     | and | each i is 10 | bits wide | (for a 1-hot | representation |     | of  |
| --- | --- | --- | --- | --- | --- | ------------ | --------- | ------------ | -------------- | --- | --- |
ThestructureofMcanbedescribedatdifferentlevelsof the 10 possible classes). Therefore, a classifier to solve
thisproblemisacircuitwith6,272Booleaninputsand10
| abstraction.                                    | For instance, | if M         | is a fully connected | feed           |                 |     |     |     |     |     |     |
| ----------------------------------------------- | ------------- | ------------ | -------------------- | -------------- | --------------- | --- | --- | --- | --- | --- | --- |
| forwardneuralnetwork,wecandescribeitasasequence |               |              |                      |                | Booleanoutputs. |     |     |     |     |     |     |
| of layers.                                      | Going one     | level lower, | we can               | describe it as |                 |     |     |     |     |     |     |
SupposeMerlin’smodelfortheMNISTclassifierisasimple
1Keepingaholdoutsetwouldnothelpushere—whatwould lookuptable. Howwouldthecircuitforitlook? Figure1
thatevenmean? sketchesonepossibility. The6,272-bitinputxiscompared
|     |     |     |     |     | witheachoftheexamplesx |     |     | inturnandifthereisamatch, |     |     |     |
| --- | --- | --- | --- | --- | ---------------------- | --- | --- | ------------------------- | --- | --- | --- |
i

Circuit-BasedIntrinsicMethodstoDetectOverfitting
thecorresponding10-bitoutputy isselected.Ifnoexample OtherTypesofCFS.Thereareothervariantsofthepro-
i
matches,thenthemodel(arbitrarily)returnsthe1-hotvector ceduredescribedabove(whichwecallSimpleCFSorjust
representingclass‘0’. Now,ifwesimulatethiscircuiton CFS).OfparticularinterestisCompositeCFSwhichisuse-
examplesinS,wenoticethatthereareinternalsignalsin fulforcircuitswithgatesthathavemanyinputsorareat
thecircuitthatarecapableofidentifyingspecifictraining higherlevelsofabstraction. InCompositeCFS,welookat
examples. For example, the signal s (the output of the rarepatternsincombinationsofsignalsfeedingaparticular
0
x=x block)is1(true)forthetrainingexamplex and0 gateandperturbtheoutputofthatgate(byflippingit)when
| 0   |     |     |     |     |     | 0   |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
(false)forallothers. Inthiscase,wesay1isararepattern ararecombinationisseenattheinputs. Anotherpossibility
fors sinces rarelytakesonthevalue1onthetrainingset. istorandomizetheperturbationinsteadofalwaysflipping.
| 0   | 0   |     |     |     |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Formally,ifasignalsinMtakesonthevaluevatmostl Inourexperiments,wefoundthesevariantstoproducere-
timesonthetrainingsetS,wecallvanl-rarepatternfors. sults that are similar to those obtained from Simple CFS,
andsoweonlymentiontheminpassing.
| This observation | leads | to  | the first | of the | two main | ideas |     |     |     |     |     |     |     |     |
| ---------------- | ----- | --- | --------- | ------ | -------- | ----- | --- | --- | --- | --- | --- | --- | --- | --- |
behindCFS:Thepresenceofl-rarepatternssuggestsover-
fitting,and,therefore,poorgeneralizationsincetheyopen 3.ExperimentalResults
M
| up the possibility |     | that | has special | logic | to detect | and |     |     |     |     |     |     |     |     |
| ------------------ | --- | ---- | ----------- | ----- | --------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
CFSImplementation.Ourimplementationofl-CFSworks
| handlespecificexamples. |     |     | Acountofrarepatterns,however, |     |     |     |                          |     |     |     |                            |     |     |     |
| ----------------------- | --- | --- | ----------------------------- | --- | --- | --- | ------------------------ | --- | --- | --- | -------------------------- | --- | --- | --- |
|                         |     |     |                               |     |     |     | onadirectedacyclicgraphG |     |     |     | representingacombinational |     |     |     |
doesnotdirectlytranslateintoametricforgeneralization
|                                           |     |     |     |     |              |     | logic circuit | where  | each      | node | is either | the          | constant | 0, a  |
| ----------------------------------------- | --- | --- | --- | --- | ------------ | --- | ------------- | ------ | --------- | ---- | --------- | ------------ | -------- | ----- |
| (withoutbuildingapredictivemodelofthat!). |     |     |     |     | Furthermore, |     |               |        |           |      |           |              |          |       |
|                                           |     |     |     |     |              |     | primary       | input, | a 2-input | And  | gate,     | or a 2-input | Xor      | gate. |
althoughapatternmayberareitmayalsobeanobservabil-
|     |     |     |     |     |     |     | An edge | is either | a direct | connection |     | or an | inverter | and |
| --- | --- | --- | --- | --- | --- | --- | ------- | --------- | -------- | ---------- | --- | ----- | -------- | --- |
itydon’tcare(ODC),i.e.,itmayhavenoinfluenceonthe
representsaBooleanfunctionintermsoftheprimaryinputs.
| outputofthecircuit. |            | Forexample,ifthesignalwiththerare |          |       |             |     |         |           |       |              |     |       |         |       |
| ------------------- | ---------- | --------------------------------- | -------- | ----- | ----------- | --- | ------- | --------- | ----- | ------------ | --- | ----- | ------- | ----- |
|                     |            |                                   |          |       |             |     | This is | a variant | of an | And-Inverter |     | Graph | (Biere, | 2007; |
| pattern             | only feeds | into an                           | And gate | whose | other input | is  |         |           |       |              |     |       |         |       |
Chatterjee,2007),astandarddatastructureinmodernlogic
0whentherarepatternappears,thenthevalueoftherare
synthesisusedtohandlecircuitswithhundredsofmillions
patterndoesnotmatterindecidingtheoutputofthecircuit.
|     |     |     |     |     |     |     | ofnodes. | Whileconstructingthesegraphs, |     |     |     |     | wepropagate |     |
| --- | --- | --- | --- | --- | --- | --- | -------- | ----------------------------- | --- | --- | --- | --- | ----------- | --- |
Weaddressbothproblemswiththesecondmainideabehind constantsbutdonotextractcommonsub-expressions.
CFS:perturbedsimulationofatrainingexamplewherewe
|          |            |         |     |           |          |     | WemaketwopassesthroughthenodesofG |     |              |     |         |        | intopological |          |
| -------- | ---------- | ------- | --- | --------- | -------- | --- | --------------------------------- | --- | ------------ | --- | ------- | ------ | ------------- | -------- |
| simulate | an example | through | M   | as usual, | but when | we  |                                   |     |              |     |         |        |               |          |
|          |            |         |     |           |          |     | order starting                    |     | from primary |     | inputs. | In the | first         | pass, we |
encounteral-rarepattern,insteadofpropagatingittothe
|     |     |     |     |     |     |     | simulatethetrainingsetthroughG |     |     |     | toobtainthecountsof |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ------------------------------ | --- | --- | --- | ------------------- | --- | --- | --- |
fanouts(i.e.,gatesthatdependonthissignal),weperturbthe
|     |     |     |     |     |     |     | differentpatternsinthecircuit. |     |     |     | Inthesecondpass,weuse |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ------------------------------ | --- | --- | --- | --------------------- | --- | --- | --- |
patternandsimulatethefanoutswiththeperturbedpattern.
thecountsfromthefirstpasstoperturbthel-rarepatterns.
| A natural | perturbation | is       | to propagate | the     | opposite | value |           |      |      |            |     |           |         |      |
| --------- | ------------ | -------- | ------------ | ------- | -------- | ----- | --------- | ---- | ---- | ---------- | --- | --------- | ------- | ---- |
|           |              |          |              |         |          |       | In Simple | CFS, | this | boils down | to  | replacing | signals | that |
| instead   | of the rare  | pattern. | In our       | running | example, | this  |           |      |      |            |     |           |         |      |
takeonavalueof0onmostexampleswiththeconstant0
| corresponds                                     | to propagating |     | a 0 instead | of  | 1 for signal | s 0 |                                      |     |     |                           |     |                 |     |     |
| ----------------------------------------------- | -------------- | --- | ----------- | --- | ------------ | --- | ------------------------------------ | --- | --- | ------------------------- | --- | --------------- | --- | --- |
|                                                 |                |     |             |     |              |     | signalandlikewisefor1.               |     |     | CFSthusrunsinlineartimein |     |                 |     |     |
| tothemultiplexerwhensimulatingthetrainingimagex |                |     |             |     |              |     | .                                    |     |     |                           |     |                 |     |     |
|                                                 |                |     |             |     |              | 0   | thesizeofthegraphandthetrainingdata. |     |     |                           |     | Forperformance, |     |     |
Inthismanner,wepreventthemodelfromidentifyingx
0
|                           |     |     |                        |     |     |     | the simulations                |     | are done | in  | a bit parallel |                   | manner | for all |
| ------------------------- | --- | --- | ---------------------- | --- | --- | --- | ------------------------------ | --- | -------- | --- | -------------- | ----------------- | ------ | ------- |
| andweseethattheoutputforx |     |     | isnolongernecessarilyy |     |     |     | .                              |     |          |     |                |                   |        |         |
|                           |     |     | 0                      |     |     | 0   | trainingexamplesatthesametime. |     |          |     |                | Toavoidrunningout |        |         |
Wecallthismodifiedsimulationprocedurel-counterfactual
|     |     |     |     |     |     |     | of memory, | we  | use reference |     | counting | to  | recycle | storage |
| --- | --- | --- | --- | --- | --- | --- | ---------- | --- | ------------- | --- | -------- | --- | ------- | ------- |
simulationorl-CFSforshort.
forintermediatesimulatedvalueswhentheyarenolonger
Weperformperturbedsimulationforeachtrainingexample needed. A typical run of l-CFS in our experiments takes
inturnandmeasuretheresultingaverageaccuracyoverthe lessthan10minutesona3.7GHzXeonCPUandlessthan
| trainingset. | Wecallthisquantitythetrainingaccuracyob- |     |     |     |     |     | 2GBofRAM. |     |     |     |     |     |     |     |
| ------------ | ---------------------------------------- | --- | --- | --- | --- | --- | --------- | --- | --- | --- | --- | --- | --- | --- |
tainedthroughl-CFS.Inourrunningexampleofthelookup
|     |     |     |     |     |     |     | BenchmarkProblem. |     |     | Whilethediscussionfromthepre- |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ----------------- | --- | --- | ----------------------------- | --- | --- | --- | --- |
table, itiseasytoseethatthetrainingaccuracyobtained
|     |     |     |     |     |     |     | vious section |     | shows | how CFS | can | discover | overfit | when |
| --- | --- | --- | --- | --- | --- | --- | ------------- | --- | ----- | ------- | --- | -------- | ------- | ---- |
through1-CFSisnobetterthanrandomchance(sinceeach
|     |     |     |     |     |     |     | the model | is a | simple | lookup | table, | it is not | clear | if CFS |
| --- | --- | --- | --- | --- | --- | --- | --------- | ---- | ------ | ------ | ------ | --------- | ----- | ------ |
trainingexampleismappedtoclass‘0’under1-CFS).Now,
wouldbeeffectiveonneuralnetworkstrainedwithstochas-
| since random | chance | is what | one | would expect | to  | be the |              |         |     |        |           |      |           |     |
| ------------ | ------ | ------- | --- | ------------ | --- | ------ | ------------ | ------- | --- | ------ | --------- | ---- | --------- | --- |
|              |        |         |     |              |     |        | tic gradient | descent |     | (SGD). | To answer | this | question, | we  |
generalizationofthelookuptable(i.e.,itsaccuracyonD),
trained3neuralnetworksforMNISTinTensorFlow(Keras)
itistemptingtoconjecturethat1-CFStrainingaccuracyis
andcompiledthemdownintocombinationallogiccircuits.
| agoodestimateofaccuracyonD. |     |     |     | Althoughthatisnotthe |     |     |                                      |     |     |     |     |     |              |     |
| --------------------------- | --- | --- | --- | -------------------- | --- | --- | ------------------------------------ | --- | --- | --- | --- | --- | ------------ | --- |
|                             |     |     |     |                      |     |     | All3networkshavethesamearchitecture: |     |     |     |     |     | aninputlayer |     |
caseasweshallseeempiricallyinSection3,wefindthatthe
|     |     |     |     |     |     |     | of size | 784 (i.e., | 28  | × 28), | 3 fully | connected | ReLU | lay- |
| --- | --- | --- | --- | --- | --- | --- | ------- | ---------- | --- | ------ | ------- | --------- | ---- | ---- |
differenceintrainingaccuracybetweennormalsimulation
erswith256nodeseach,andafinalsoftmaxlayerwith10
andl-CFSisagoodmeasureofthedegreeofoverfitofM.
|     |     |     |     |     |     |     | outputs. | (Thus,thetotalnumberoftrainableparametersis |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | -------- | ------------------------------------------- | --- | --- | --- | --- | --- | --- |

Circuit-BasedIntrinsicMethodstoDetectOverfitting
335,114.) WealsoperformedsomeexperimentswithFash- Itisremarkablethatevenwhenaneuralnetworkisrepre-
ionMNIST(Xiaoetal.,2017)andtheresultsaresimilar. sentedataverylowlevelasalogiccircuit,relativeoverfit
canbedetectedusinganintuitivealgorithmwithnohyper-
Thefirsttwonetworks(nn-real-2andnn-real-100),
parameterstotune.
weretrainedontheMNISTtrainingsetfor2epochsand
100epochsrespectively. Theygettotraining(top-1)accu- Expt.2:ImpactofArchitecture.Therearemanydifferent
raciesof97%and99.90%respectively. Thethirdnetwork ways in which a neural network can be compiled down
(nn-random)wastrainedonavariantofMNISTwhere intologicgates. InExpt. 1,wemadecertainarchitectural
the output labels in the training set are permuted pseudo- choicesforthecircuit,butwhatifwehadchosendifferently?
randomly and trained for 300 epochs to get to a training To evaluate that, here we replace the multipliers used in
91.27%.2
accuracy of In all cases, we used the ADAM Expt. 1 with array multipliers (i.e., multipliers based on
optimizerwithdefaultparametersandbatchsizeof64. As the elementary algorithm for multiplication). Figure 2b
expected, nn-real-2 is the least overfit and gets to a showstheresultantCFScurves(withdashedlines)aswell
validationsetaccuracyof97%(i.e., hasanegligiblegen- astheoriginalcurves(solidlines)forreference. Thecurves
eralizationgap),nn-real-100ismoreoverfitgettingto donotcoincideindicatingthattheresultofCFSdepends
avalidationsetaccuracyof98.24%(agapof1.66%),and onthestructureofthecircuitandnotjustonthefunction
finally, the validation accuracy of nn-random is 9.73% implementedbythecircuit(sincethefunctionisthesame
(i.e.,closetochance)confirmingthatitishorriblyoverfit. inbothcases). However,forthesamechoiceofarchitecture,
wefindthatthefalloffinCFScurvesareagainindicativeof
| ConversiontoLogicCircuits. |     | Thisisdonebygenerating |     |     |     |     |     |
| -------------------------- | --- | ---------------------- | --- | --- | --- | --- | --- |
thedegreeofoverfit.
| logic subcircuits | composed | of 2-input | And or Xor gates |     |     |     |     |
| ----------------- | -------- | ---------- | ---------------- | --- | --- | --- | --- |
and inverters for each of the operations in the neural net- Expt. 3: ImpactofChoiceofPrimitives. Evenatthelow-
work. Weights and activations are represented by signed estlevelofabstraction,wecanchoosewhatprimitivesto
8-bitand16-bitfixedpointnumbersrespectivelywith6bits workwith. ToseehowthischoiceimpactsCFScurves,in
reservedforthefractionalpart. (Weightsfromtrainingare thisexperimentwedisallowXorgatesasprimitives(thus
clamped to [−2.0,2.0) before conversion to fixed point.) requiringthatonlyAndgatesandinvertersbeused). Fig-
Eachmultiply-accumulateunitmultipliesan8-bitconstant ure2cshowstheresultingCFScurves(dashed)aswellas
(theweight)witha16-bitinput(theactivation)andaccumu- theoriginals(solid). Onceagain,weseethatthecurvesdo
latesin24bitswithsaturation. Theconstantmultiplications notcoincideindicatingthatthechoiceofprimitivesmatters
are done by finding a minimal combination of bit-shifts but for the same choice of primitives, the falloff in CFS
(multiplicationsbypowersof2)andadditionsorsubtrac- curvesareagainindicativeofthedegreeofoverfit.
tions. Forexample,5×uisimplementedas4×u+uand
|                   |     |                         |     | Expt. 4: | Count of Rare Patterns. | Figure | 2d shows for |
| ----------------- | --- | ----------------------- | --- | -------- | ----------------------- | ------ | ------------ |
| 11×uas16×u−4×u−u. |     | ReLUsareimplementedwith |     |          |                         |        |              |
eachvalueofl,howmanyexampleshavenol-rarepatterns,
| acomparatorandamultiplexer. |     | Theoutputsofeachnet- |     |     |     |     |     |
| --------------------------- | --- | -------------------- | --- | --- | --- | --- | --- |
i.e.,cannotbepossiblyaffectedbyl-CFS.Weobservein
workarethe10signed16-bitactivationsbeforethesoftmax.
|     |     |     |     | particular, | that for nn-random, | there are | 54,094 exam- |
| --- | --- | --- | --- | ----------- | ------------------- | --------- | ------------ |
Whenevaluatingaccuracy(withCFSorwithout)wepick
|     |     |     |     | ples (about | 90% of the training | set) that have | no 1-rare |
| --- | --- | --- | --- | ----------- | ------------------- | -------------- | --------- |
theclasscorrespondingtothelargestofthe10activations
patterns. Thisisinsharpcontrasttoasimplelookuptable
| (top-1 accuracy). | The resulting | logic | circuits have 35 to |                                           |     |     |       |
| ----------------- | ------------- | ----- | ------------------- | ----------------------------------------- | --- | --- | ----- |
|                   |               |       |                     | whereeveryexamplewouldhavea1-rarepattern, |     |     | andin |
52millionAndorXorgatesand5500to6000logiclevels.
factslightlymorethannn-real-100.
Comparingthese
(Thesesizesalongwiththeneedtofitrandomdatadictated
curvesofcountstotheCFScurvesinFigure2aindicates
thechoiceofarchitectureandbenchmark.)
thatperturbationisanimportantpartofCFSandthatrarity
Expt. 1: EffectofSimpleCFS.Figure2ashowsthetrain- byitselfisacrudermeasureofoverfittingsinceitmaynot
| ingaccuraciesobtainedthroughl-CFSforeachofthethree |     |     |     | beobservable. |     |     |     |
| -------------------------------------------------- | --- | --- | --- | ------------- | --- | --- | --- |
networksaslvariesfrom1to1024(whichisabout1.7%of
|                               |                                         |                     |     | Expt. 5:    | Random Forests.     | Since CFS works | on the cir-  |
| ----------------------------- | --------------------------------------- | ------------------- | --- | ----------- | ------------------- | --------------- | ------------ |
| thenumberoftrainingexamples). |                                         | WecalltheseplotsCFS |     |             |                     |                 |              |
|                               |                                         |                     |     | cuit level, | it can check random | forests for     | overfit. Two |
| curves. Asl                   | increases,i.e.,asmorepatternsbecomerare |                     |     |             |                     |                 |              |
randomforestsweretrainedusingversion0.19.1ofScikit-
| and get perturbed, | the accuracy | falls eventually | reaching |                             |     |                         |     |
| ------------------ | ------------ | ---------------- | -------- | --------------------------- | --- | ----------------------- | --- |
|                    |              |                  |          | learn(Pedregosaetal.,2011). |     | Eachforesthas10treesand |     |
chance. However,itisinterestingthatthedropinaccuracy
|     |     |     |     | is trained | using the default settings, | except | for bootstrap- |
| --- | --- | --- | --- | ---------- | --------------------------- | ------ | -------------- |
ishighestfornn-random(e.g.,atl=64,thedropisabout
|                |                      |     |               | ping(toavoidnon-uniformweightsduringinference). |     |     | The |
| -------------- | -------------------- | --- | ------------- | ----------------------------------------------- | --- | --- | --- |
| 45%), somewhat | less for nn-real-100 |     | (20%) and the |                                                 |     |     |     |
firstforest(rf-real)wastrainedonMNISTwhereasthe
| leastfornn-real-2(1.4%). |     | Thusthefalloffinaccuracy |     |     |     |     |     |
| ------------------------ | --- | ------------------------ | --- | --- | --- | --- | --- |
secondforest(rf-random)wastrainedonMNISTwith
withlisanindicatorofthelevelofoverfitofanetwork.
theoutputlabelspseudo-randomlypermuted(asbeforewith
2Whileevaluatingrandomfortrainingaccuracy(withorwith- nn-random). Bothforestsreachperfecttrainingaccuracy.
outCFS),thepermutedlabelsareused. rf-realgets95.58%validationaccuracywhereasasex-

Circuit-BasedIntrinsicMethodstoDetectOverfitting
| 1.0                          |     |     |     | 1.0                          |     |     |     |     |
| ---------------------------- | --- | --- | --- | ---------------------------- | --- | --- | --- | --- |
| SFC-l htiw ycarucca gniniart |     |     |     | SFC-l htiw ycarucca gniniart |     |     |     |     |
| 0.8                          |     |     |     | 0.8                          |     |     |     |     |
| 0.6                          |     |     |     | 0.6                          |     |     |     |     |
| 0.4                          |     |     |     | 0.4 nn-real-2                |     |     |     |     |
nn-real-2 w/ array
nn-real-100
| 0.2 nn-real-2 |         |     |     | 0.2 nn-real-100 w/ array |     |         |     |     |
| ------------- | ------- | --- | --- | ------------------------ | --- | ------- | --- | --- |
| nn-real-100   |         |     |     | nn-random                |     |         |     |     |
| nn-random     |         |     |     | nn-random w/ array       |     |         |     |     |
| 0.0           |         |     |     | 0.0                      |     |         |     |     |
| 0 2           | 4       | 6 8 | 10  | 0                        | 2   | 4       | 6 8 | 10  |
|               | log2(l) |     |     |                          |     | log2(l) |     |     |
|               | (a)     |     |     |                          |     | (b)     |     |     |
60000
snrettap erar-l on /w selpmaxe .mun
| 1.0 |     |     |     |     |     |     | nn-real-2 |     |
| --- | --- | --- | --- | --- | --- | --- | --------- | --- |
nn-real-100
|     |     |     |     | 50000 |     |     | nn-random |     |
| --- | --- | --- | --- | ----- | --- | --- | --------- | --- |
SFC-l htiw ycarucca gniniart
0.8
40000
| 0.6            |     |     |     | 30000 |     |     |     |     |
| -------------- | --- | --- | --- | ----- | --- | --- | --- | --- |
| 0.4 nn-real-2  |     |     |     | 20000 |     |     |     |     |
nn-real-2 w/o xors
nn-real-100
10000
0.2 nn-real-100 w/o xors
nn-random
| nn-random w/o xors |     |     |     | 0   |     |         |     |     |
| ------------------ | --- | --- | --- | --- | --- | ------- | --- | --- |
|                    |     |     |     | 0   | 1   | 2 3     | 4 5 | 6   |
| 0.0 0 2            | 4   | 6 8 | 10  |     |     | log2(l) |     |     |
log2(l)
|     | (c) |     |     |                                          |     | (d) |     |     |
| --- | --- | --- | --- | ---------------------------------------- | --- | --- | --- | --- |
| 1.0 |     |     |     | esion teknalb htiw ycarucca gniniart 1.0 |     |     |     |     |
rf-random
rf-real
SFC-l htiw ycarucca gniniart
| 0.8 |     |     |     | 0.8 |     |     | nn-real-2 |     |
| --- | --- | --- | --- | --- | --- | --- | --------- | --- |
nn-real-100
nn-random
| 0.6 |     |     |     | 0.6 |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.4 |     |     |     | 0.4 |     |     |     |     |
rf-real
rf-random
| 0.2 |     |     |     | 0.2 |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
nn-real-2
nn-real-100
nn-random
| 0.0                          |         |              |     | 0.0                                      |     |         |              |     |
| ---------------------------- | ------- | ------------ | --- | ---------------------------------------- | --- | ------- | ------------ | --- |
| 0 2                          | 4 6     | 8 10         | 12  | 30 25                                    |     | 20      | 15 10        | 5   |
|                              | log2(l) |              |     |                                          |     | log2(p) |              |     |
|                              | (e)     |              |     |                                          |     | (f)     |              |     |
| 1.0                          |         |              |     | esion teknalb htiw ycarucca gniniart 1.0 |     |         |              |     |
|                              |         | epochs = 1   |     |                                          |     |         | epochs = 1   |     |
| SFC-l htiw ycarucca gniniart |         | epochs = 2   |     |                                          |     |         | epochs = 2   |     |
| 0.8                          |         | epochs = 5   |     | 0.8                                      |     |         | epochs = 5   |     |
|                              |         | epochs = 10  |     |                                          |     |         | epochs = 10  |     |
|                              |         | epochs = 20  |     |                                          |     |         | epochs = 20  |     |
| 0.6                          |         |              |     | 0.6                                      |     |         |              |     |
|                              |         | epochs = 50  |     |                                          |     |         | epochs = 50  |     |
|                              |         | epochs = 100 |     |                                          |     |         | epochs = 100 |     |
|                              |         | epochs = 200 |     |                                          |     |         | epochs = 200 |     |
| 0.4                          |         |              |     | 0.4                                      |     |         |              |     |
| 0.2                          |         |              |     | 0.2                                      |     |         |              |     |
| 0.0                          |         |              |     | 0.0                                      |     |         |              |     |
| 0 2                          | 4 6     | 8 10         | 12  | 30 28                                    | 26  | 24      | 22 20        | 18  |
|                              | log2(l) |              |     |                                          |     | log2(p) |              |     |
|                              | (g)     |              |     |                                          |     | (h)     |              |     |
Figure2: TheresultsofexperimentsinSection3. Plot(a)showstheCFScurvesfor3networkswithdifferentamountsof
overfitting;(b)and(c)showtheimpactofthechoiceofmultipliersandofprimitivelogicgatesrespectivelyonCFScurves;
(d)showshowmanyexamplesareunaffectedbyl-CFSsincetheydonothaveanyl-rarepatterns;(e)showsCFScurvesfor
randomforests;(f)showstrainingaccuracieswhenasignalisrandomlyflippedwithprobabilityp;and(g)and(h)showthe
differencesbetweenCFSandrandomflipsrespectivelyfor8networkstrainedonadatasetwithhighlabelnoise.

Circuit-BasedIntrinsicMethodstoDetectOverfitting
pectedrf-randomgetsnobetterthanchance.(rf-real ingthetrainingsetwhilerandomlyflippingthenodevalues
has about 14K nodes per tree whereas rf-random has withprobabilityp. Aspvariesfrom2−30to2−5,theresult-
about70Knodespertree.) ingnoisecurves(Figure2f)aresimilartotheCFScurves
|     |     |     |     |     |     | (Figure | 2e). However, | the | more overfit | nn-real-100 |     |
| --- | --- | --- | --- | --- | --- | ------- | ------------- | --- | ------------ | ----------- | --- |
Theforestsarecompileddowntocircuitsinastraightfor-
|     |     |     |     |     |     | does not | fall faster than | nn-real-2. |     | With | CFS, these |
| --- | --- | --- | --- | --- | --- | -------- | ---------------- | ---------- | --- | ---- | ---------- |
wardmanner.Eachtreeiscompiledseparatelyandproduces
curvesarewellseparated,andthegapbetweenneuralnets
| 1016-bitoutputs(oneperclass). |     |     | Thecorrespondingoutputs |     |     |     |     |     |     |     |     |
| ----------------------------- | --- | --- | ----------------------- | --- | --- | --- | --- | --- | --- | --- | --- |
andforestsismuchlarger.
areaddedacrossall10treesandtheclassoutputbythefor-
estcorrespondstotheclasswiththemaximumvalue. Each Expt. 7: Sensitivity. Tobetterunderstandthesensitivity
internalnodeinatreemapstoamultiplexercontrolledbya differencebetweenCFSandblanketnoise,wetrained38
8-bitcomparatortoimplementthethreshold,andeachleaf neuralnetworks(withthesamearchitectureasbeforebut
node corresponds to 10 16-bit constants representing the different number of epochs) on MNIST with exactly one
numberofexamplesthatoccupyeachclassinthatleaf(thus halfoflabelsrandomized(somaximumaccuracypossible
mostentriesarezero). Thecircuitforrf-realhasabout is about 55%). We show the CFS curves and the noise
700Knodeswhereasrf-randomhas3Mnodes. Bothare curvesfor8representativenetworksinFigures2gand2h
lessthan250logiclevelsdeep. (Thesearemuchsmaller respectively. Note the crossover of the CFS curves that
thanthecircuitsfortheneuralnetworks.) indicatesalargerfalloffforoverfitnetworkscomparedto
|     |     |     |     |     |     | the more | uniform degradation |     | of the | noise | curves. It is |
| --- | --- | --- | --- | --- | --- | -------- | ------------------- | --- | ------ | ----- | ------------- |
Figure2eshowstheCFScurvesforthetworandomforests.
|            |                                     |           |      |                       |     | fascinating                    | that all the | CFS | curves | cross over        | at a single |
| ---------- | ----------------------------------- | --------- | ---- | --------------------- | --- | ------------------------------ | ------------ | --- | ------ | ----------------- | ----------- |
| Onceagain, | weseethattheoverfitmodel(rf-random) |           |      |                       |     |                                |              |     |        |                   |             |
|            |                                     |           |      |                       |     | pointwithanaccuracyofabout50%. |              |     |        | Thisisdiscussedin |             |
| degrades   | faster than                         | the model | with | better generalization |     |                                |              |     |        |                   |             |
Section4.
(rf-real)confirmingthatCFSiseffectiveevenformod-
elsthatarefundamentallydifferentfromneuralnetworks. Expt. 8: Composite CFS. For completeness, Figure 3
However,itisinterestingtoseethatifwecompareacross shows the results of running Composite CFS instead of
modelfamiliesi.e. betweentheneuralnetworksfromExpt. Simple CFS in the setup of Expt. 7. We see that these
1(repeatedinFigure2eforconvenience)andtherandom curvesareverysimilartothoseofSimpleCFSinFigure2g.
| forests, | CFS is not | effective | at distinguishing |     | overfit. In |     |     |     |     |     |     |
| -------- | ---------- | --------- | ----------------- | --- | ----------- | --- | --- | --- | --- | --- | --- |
particular,nn-real-2whichisnotoverfitdegradesmore
4.Discussion
| rapidlythatrf-randomwhichishighlyoverfit. |     |     |     |     | Wedis- |     |     |     |     |     |     |
| ----------------------------------------- | --- | --- | --- | --- | ------ | --- | --- | --- | --- | --- | --- |
cussthisingreaterdetailinSection4. StructureDependence. Expt. 2and3showthattheresults
ofSimpleCFSdependnotjustonthefunctionbutonthe
| Expt.       | 6: BlanketNoise.                         | CFSmaybeseenasaddingatar- |     |     |     |                        |     |                                 |     |     |     |
| ----------- | ---------------------------------------- | ------------------------- | --- | --- | --- | ---------------------- | --- | ------------------------------- | --- | --- | --- |
|             |                                          |                           |     |     |     | structureofthecircuit. |     | (OtherCFSvariantsweinvestigated |     |     |     |
| getednoise. | Here,instead,weaddblanketnoisebysimulat- |                           |     |     |     |                        |     |                                 |     |     |     |
showthisbehavioraswell.)Asmallexampleprovidessome
insight. ConsidertheBooleanfunctionf(a,b,c)=aeval-
uatedonthetrainingsetcomprisingthefullBooleancube
|     |     |     |     |     |     | (i.e., all    | 8 combinations  | of  | a,b,c ∈ | {0,1}). | In addition |
| --- | --- | --- | --- | --- | --- | ------------- | --------------- | --- | ------- | ------- | ----------- |
|     |     |     |     |     |     | to the direct | implementation, |     | f can   | also be | implemented |
1.0
|     |                              |     |     | epochs = 1 |     | (redundantly)asa·b·c+a·¬b·c+a·b·¬c+a·¬b·¬c.         |     |     |     |     |     |
| --- | ---------------------------- | --- | --- | ---------- | --- | --------------------------------------------------- | --- | --- | --- | --- | --- |
|     | SFC-l htiw ycarucca gniniart |     |     | epochs = 2 |     |                                                     |     |     |     |     |     |
|     | 0.8                          |     |     |            |     | Itiseasytoseethatunder1-CFS,thedirectimplementation |     |     |     |     |     |
epochs = 5
epochs = 10
|     |     |     |     | epochs = 20 |     | is unchanged | (there | are no | 1-rarepatterns) |     | butthe redun- |
| --- | --- | --- | --- | ----------- | --- | ------------ | ------ | ------ | --------------- | --- | ------------- |
0.6
|     |     |     |     | epochs = 50 |     | dantimplementationmapstoconstant0(theoutputofeach |     |     |     |     |     |
| --- | --- | --- | --- | ----------- | --- | ------------------------------------------------- | --- | --- | --- | --- | --- |
epochs = 100
epochs = 200
|     | 0.4 |     |     |     |     | conjunctionis1onlyonce). |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | ------------------------ | --- | --- | --- | --- | --- |
Althoughthisisnotaproblemwhenthecompilationpro-
0.2
cesscanbecontrolled,thisisbadnewsintheadversarial
|     | 0.0 |     |     |      |     | setup. Agoodmodelwithapoorimplementationmayshow |     |     |     |     |     |
| --- | --- | --- | --- | ---- | --- | ----------------------------------------------- | --- | --- | --- | --- | --- |
|     | 0 2 | 4   | 6   | 8 10 | 12  |                                                 |     |     |     |     |     |
steeperdegradationunderCFSthanamoreoverfitmodel
log2(l)
|     |     |     |     |     |     | with better | implementation |     | (c.f. nn-real-2 |     | with array |
| --- | --- | --- | --- | --- | --- | ----------- | -------------- | --- | --------------- | --- | ---------- |
Figure 3: This plot shows the result of repeating Expt. 7 v/snn-real-100withoutXorsatl =256). Ideally,we
wouldliketofindavariantofCFSthatdoesnotdependon
butusingCompositeCFS(insteadofSimpleCFS)where
wecomputeandperturbrarepatternsacrossallthesignals structurebutonlyonthefunction.3 Intheabsenceofthat
thatfeedintoalogicgate,i.e.,acrossallthefaninsofagate 3Inprinciple,onecouldcanonizethecircuitstructurebefore
(insteadofcomputingandperturbingrarepatternsforeach
applyingCFS,saybybuildingReducedOrderBinaryDecisionDi-
| signalorfaninindependently). |     |     | TheresultswithComposite |     |     |     |     |     |     |     |     |
| ---------------------------- | --- | --- | ----------------------- | --- | --- | --- | --- | --- | --- | --- | --- |
agrams(ROBDDs),butthatwouldbecomputationallyprohibitive.
CFSareverysimilartothoseofSimpleCFS(Figure2g). Alternatively,onecouldlightlyoptimizethecircuitbeforeCFS

Circuit-BasedIntrinsicMethodstoDetectOverfitting
|     | 1.0 |     |     |     |     |     | 1.0 |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
epochs = 1
SFC-l htiw ycarucca gniniart SFC-l htiw ycarucca gniniart epochs = 2
|     | 0.8 |     |     |     |     |     | 0.8 |     |     |     | epochs = 5 |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---------- | --- |
epochs = 10
epochs = 20
|     | 0.6            |     |     |     |     |     | 0.6 |     |     |     | epochs = 50  |     |
| --- | -------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ------------ | --- |
|     | epochs = 1     |     |     |     |     |     |     |     |     |     | epochs = 100 |     |
|     | epochs = 2     |     |     |     |     |     |     |     |     |     | epochs = 200 |     |
|     | 0.4 epochs = 5 |     |     |     |     |     | 0.4 |     |     |     |              |     |
epochs = 10
epochs = 20
|     | 0.2 epochs = 50 |     |     |     |     |     | 0.2 |     |     |     |     |     |
| --- | --------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
epochs = 100
epochs = 200
|     | 0.0 |     |         |     |       |     | 0.0 |     |     |         |      |     |
| --- | --- | --- | ------- | --- | ----- | --- | --- | --- | --- | ------- | ---- | --- |
|     | 0 2 | 4   | 6       | 8   | 10 12 |     | 0   | 2   | 4   | 6       | 8 10 | 12  |
|     |     |     | log2(l) |     |       |     |     |     |     | log2(l) |      |     |
|     |     |     | (a)     |     |       |     |     |     |     | (b)     |      |     |
Figure4: TheseplotsshowtheresultofrepeatingExpt. 7whenwecorruptonlyathirdofthelabelsin(a)MNISTandin(b)
Fashion-MNIST.Inbothcases,justasinFigure2gweseethatallthecurvespassthroughapointclosetothemaximum
achievableaccuracy(around2/3). ThereasonforthisisdiscussedinSection4underGeneralizationinDeepLearning.
ideal,weviewthestructureofthecircuitasacertificateof thoughunbalancedhavenorarepatterns.
howwellthedatasetislearnt,andmakeitMerlin’srespon-
ThisexamplesuggestsawaytobreakCFS.IfMerlinwanted
| sibility | to find and present | the | most convincing |     | structure. |     |     |     |     |     |     |     |
| -------- | ------------------- | --- | --------------- | --- | ---------- | --- | --- | --- | --- | --- | --- | --- |
tobuildalookuptable,butpreventArthurfromidentifying
Fromthisperspective,intheaboveexample,thedirectim-
|     |     |     |     |     |     | it  | as such | (by applying |     | say 1-CFS), | instead | of building |
| --- | --- | --- | --- | --- | --- | --- | ------- | ------------ | --- | ----------- | ------- | ----------- |
plementationoff
(whichisnotimpactedby1-CFS)ismore
thecircuitinFigure1,Merlincouldbuildadecision(i.e.,
| convincingthantheredundantimplementationoff |     |     |     |     | (which |     |     |     |     |     |     |     |
| ------------------------------------------- | --- | --- | --- | --- | ------ | --- | --- | --- | --- | --- | --- | --- |
multiplexer)treebyrecursivelysplittingononevariable(i.e.,
isseverelyimpacted).
onecomponentofx)atatimebypickingacomponentthatis
Adversarial Attack on CFS. Based on the discussion balanced(i.e.,takeson0and1valuesroughlyequallyoften)
aboveitiseasytodesignawaytoarbitrarilydegradethe andleadstobothbranchesalsobeingbalanced(i.e.,have
performance of a circuit under CFS. But is the opposite noclasswitheithertoofewortoomanyexamples,although
possible? CanMerlinfitanarbitraryfunctionontheinputs havingnoexamplesofaclassorhavingonlyexamplesof
but compile it down to a circuit which does not degrade oneclassisfineandexpectedattheleafnodes). Inother
under CFS? Expt. 5. offers a clue. The overfit model words, Merlin could use a procedure similar to the usual
rf-randomfittedonrandomlabelsfallsoffmoreslowly decisiontreeconstructionprocedures,butwiththeimportant
than nn-real-2 which generalizes well. What is go- differenceoffavoringbalancedsplitsinsteadofunbalanced
ing on? The short answer is that although each tree in splits.(TheunbalancedsplitsarelikelywhyCFSiseffective
rf-randomisextremelyoverfitwithmostleavescontain- totheextentitisonrandomforests.)
ingonlyasingleexample,thecircuitnodeshavefewrare
Finally,assuggestedbyareviewerofthispaper,itwouldbe
patternsduetotheobservabilitydon’tcaresintroducedby
interestingtoextendthisidea(ultimatelybasedonShannon
themultiplexers.
decompositionandco-factoring)toanalgorithmthatcan
Againasimpleexampleisinstructive. Letf betheparity amplifythecountofrarepatternsinagivencircuitwithout
functiononnbits,i.e.,f(x 1 ,x 2 ,...,x n )=x 1 ⊕x 2 ...⊕ changingitsfunctionality.
| x . Consider | an tree       | implementation |     | of this | function ob- |                |     |     |               |     |         |            |
| ------------ | ------------- | -------------- | --- | ------- | ------------ | -------------- | --- | --- | ------------- | --- | ------- | ---------- |
| n            |               |                |     |         |              | Comparisonwith |     |     | BlanketNoise. |     | Basedon | Expt. 6and |
| tained       | using Shannon | decomposition  |     | which   | has a multi- |                |     |     |               |     |         |            |
7,webelievethatblanketnoiseislesssensitivethanCFS.
| plexeratthetopcontrolledbyx |          |                  | 1 andwith1⊕x |     | 2 ⊕x 3 ...⊕   |     |         |      |        |               |            |           |
| --------------------------- | -------- | ---------------- | ------------ | --- | ------------- | --- | ------- | ---- | ------ | ------------- | ---------- | --------- |
|                             |          |                  |              |     |               | Our | results | here | add to | the extensive | literature | on noise, |
| x andx                      | ⊕x ...⊕x | asitsdatainputs. |              |     | Ifthetraining |     |         |      |        |               |            |           |
n 2 3 n generalizationandfaulttoleranceinneuralnetworks(e.g.,
{0,1}n
| set is the | full Boolean | cube, | i.e., | it  | is easy to see |     |     |     |     |     |     |     |
| ---------- | ------------ | ----- | ----- | --- | -------------- | --- | --- | --- | --- | --- | --- | --- |
seeBernieretal.(2001)andthereferencestherein)byex-
thattherearenol-raresignalsforl<<nsinceeachinput
tendingthemtothecircuitlevel(wheredistinctionsbetween
tothemultiplexerisbalanced,i.e.,hasequalnumberof0s
activationorweightnoise,oradditiveormultiplicativenoise
and1s. SinceShannondecompositionisrecursive,asimilar
|     |     |     |     |     |     | disappear) |     | and to | other | model | families such | as random |
| --- | --- | --- | --- | --- | --- | ---------- | --- | ------ | ----- | ----- | ------------- | --------- |
argumentholdsforthelowerlevelsofthetreeuntilweget
|        |                  |          |       |          |         | forests. | Furthermore, |     | Expt. | 6   | presents a | direct compari- |
| ------ | ---------------- | -------- | ----- | -------- | ------- | -------- | ------------ | --- | ----- | --- | ---------- | --------------- |
| to the | leaves which are | constant | 0 and | constant | 1 which |          |              |     |       |     |            |                 |
sonofthefault-toleranceofneuralnetsandforestswhere
butthatmaynotbeenough. forestsareseentobeabout1000xmorefault-toleranttobit

Circuit-BasedIntrinsicMethodstoDetectOverfitting
flips. Thisislikelyduetotheredundancyfromensembling. obviouslyapply. Wehavenotstudiedifnormalizedmargin
Italsosuggeststhatnoise-basedintrinsicmethodscouldbe can be exploited by an adversary. Similarly, most mea-
easilyfooledbyanadversarybyaddingredundancy. suresbasedontheshallownessofminimaarenotadequate
sincetheyarenotscale-invariant(Dinhetal.,2017)andwe
| Generalization | in Deep | Learning. | Why | do  | neural nets |     |     |     |     |     |     |     |
| -------------- | ------- | --------- | --- | --- | ----------- | --- | --- | --- | --- | --- | --- | --- |
havenotinvestigatedifmorerecentworkonscale-invariant
trainedwithSGDgeneralizewhentheyhavesufficient(ef-
|                   |             |     |                |      |         | measures   | (Rangamani |           | et al., 2019) | can        | be exploited. | In   |
| ----------------- | ----------- | --- | -------------- | ---- | ------- | ---------- | ---------- | --------- | ------------- | ---------- | ------------- | ---- |
| fective) capacity | to memorize |     | their training | set? | This is |            |            |           |               |            |               |      |
|                   |             |     |                |      |         | comparison | to         | these and | other         | approaches | for neural    | net- |
anopenresearchquestion(Zhangetal.,2017;Arpitetal.,
works(Aroraetal.,2018;Neyshaburetal.,2018),CFSis
| 2017; Bartlett | et al., 2017; | Arora | et al., | 2018; | Neyshabur |     |     |     |     |     |     |     |
| -------------- | ------------- | ----- | ------- | ----- | --------- | --- | --- | --- | --- | --- | --- | --- |
fundamentallymorediscrete,whichmakesitapplicableto
| et al., 2018). | Expt 5. shows | that | this question |     | is not lim- |                       |     |     |                              |     |     |     |
| -------------- | ------------- | ---- | ------------- | --- | ----------- | --------------------- | --- | --- | ---------------------------- | --- | --- | --- |
|                |               |      |               |     |             | alargerclassofmodels. |     |     | However,incontrasttotheother |     |     |     |
itedtonets—thesamecouldbeaskedforrandomforests
approaches,wedonothaveanytheoreticalboundsyetwhile
aswell. One(informal)answerforforestsisthatdecision
ourresultsindicatethatwithoutfurtherrefinementstoCFS
treeconstructionprocedureslookforcommonpatternsbe-
itself,anygeneralizationboundsfroml-CFSwouldlikely
| tweenexamples. | Whenexamplessharecommonality,they |     |     |     |     |     |     |     |     |     |     |     |
| -------------- | --------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
bevacuousinpractice.
arecombinedintocommonleafnodesandthemodelgen-
| eralizes, | whereas when | there | is little | commonality, | each |     |     |     |     |     |     |     |
| --------- | ------------ | ----- | --------- | ------------ | ---- | --- | --- | --- | --- | --- | --- | --- |
5.ConclusionandFutureWork
exampleisitsownleaf(sothetrainingsetisfitwell)butthe
| modelfailstogeneralize. |     | Couldthesamethingbegoingon |     |     |     |     |     |     |     |     |     |     |
| ----------------------- | --- | -------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
OurmainresultisthatCFSbasedonaddingsmallamounts
| withSGDandnetworks? |     | CFSonnn-randomandExpt4. |     |     |     |     |     |     |     |     |     |     |
| ------------------- | --- | ----------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
oftargetednoiseatthelogiccircuitlevelcandetectoverfit.
providedirectevidencethatevenonrandomdata,netsdo
|     |     |     |     |     |     | This is | remarkable | because | at  | this level | of representation |     |
| --- | --- | --- | --- | --- | --- | ------- | ---------- | ------- | --- | ---------- | ----------------- | --- |
not“brute-forcememorize”butidentifycommonpatterns
|             |                    |        |          |        |            | we have | lost most       | aspects | of      | the structure | of the           | model, |
| ----------- | ------------------ | ------ | -------- | ------ | ---------- | ------- | --------------- | ------- | ------- | ------------- | ---------------- | ------ |
| in the data | (a question        | raised | in Zhang | et al. | (2017) and |         |                 |         |         |               |                  |        |
|             |                    |        |          |        |            | such as | the distinction |         | between | weights       | and activations, |        |
| discussedin | Arpitetal.(2017)). |        |          |        |            |         |                 |         |         |               |                  |        |
orevenwhetherthemodelisaneuralnetwork,arandom
Inthiscontext,itisinterestingtostudywhytheCFScurves forest,oralookuptable. Furthermore,variationssuchas
forthedifferentnetworksinExpt7. intersectatacommon perturbingonlyrarepatternsinsinglesignalsoracrossthe
pointcorrespondingapproximatelytotheachievableaccu- faninsofagateleadtoqualitativelysimilarresultsandthere
racy. Thisholdsforotheramountsoflabelrandomization arevariants(suchasSimpleCFS)thatarenaturallyfreeof
| e.g.,seeFigure4fortheCFScurveswhenonethirdofthe |                                 |     |     |     |     | hyper-parameters. |     |     |     |     |     |     |
| ----------------------------------------------- | ------------------------------- | --- | --- | --- | --- | ----------------- | --- | --- | --- | --- | --- | --- |
| labelsarecorrupted.                             | Roughlyhalfoftheexamplesareeasy |     |     |     |     |                   |     |     |     |     |     |     |
Bystudyingrarepatterns,wefindthatSGDdoesnotlead
sincetheyhavecorrectlabelsandarelearntinthefirstfew
to“bruteforce”memorization,butfindscommonpatterns
| epochs. | The remaining | examples | with | corrupt | labels are |     |     |     |     |     |     |     |
| ------- | ------------- | -------- | ---- | ------- | ---------- | --- | --- | --- | --- | --- | --- | --- |
(whethertrainingisdonewithrandomizedlabelsoractual
| harder and | learnt only | in later | epochs | by the models | that |     |     |     |     |     |     |     |
| ---------- | ----------- | -------- | ------ | ------------- | ---- | --- | --- | --- | --- | --- | --- | --- |
labels), andneuralnetworksarenotunlikeforestsinthis
| aretrainedlonger. | WithCFS,theaccuracyofthosemodels |     |     |     |     |     |     |     |     |     |     |     |
| ----------------- | -------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
regard.Byaddingblanketnoise,randomforestsarefoundto
breaksdownearliersincethehardexampleshavemorerare
beabout1000xmoreresilienttonoisethanneuralnetworks
| patterns | than the easy | examples, | and the | accuracy | on the |     |     |     |     |     |     |     |
| -------- | ------------- | --------- | ------- | -------- | ------ | --- | --- | --- | --- | --- | --- | --- |
whichcouldbeusefulwhenimplementingmachinelearning
easyexamplesthusformsalimitingcurveforallmodels.
systemswithunreliablelowlevelcomponents.
Thisprovidesmore(anddirect)evidencefortheclaimin
Arpit et al. (2017, §1) that “SGD learns simpler patterns Thereareseveraldirectionsforfuturework. Weanalyzeflat
firstbeforememorizing.” Furthermore,wecouldidentify circuits, but with a clever implementation that constructs
“simplerpatterns”asexamplesthathavefewerrarepatterns thecircuiton-the-flyfromahigherlevelspecification,the
| and “memorizing” | as what | is  | required | for examples | that |              |     |       |           |         |          |      |
| ---------------- | ------- | --- | -------- | ------------ | ---- | ------------ | --- | ----- | --------- | ------- | -------- | ---- |
|                  |         |     |          |              |      | computations | can | scale | to larger | models. | We could | also |
havemorerarepatterns. Thus,learningsimplerpatternsand applyCFSathigherlevelsofabstraction(perhapsaspartof
memorizationarenotfundamentallydifferentbutlieattwo themodelevaluationprocessinframeworks,suchasScikit
endsofaspectrum. LearnandTensorflowEstimators)thoughatthatlevelthere
|              |                                 |     |     |     |     | are moredegrees |     | of freedomin |     | theimplementation |     | (e.g., |
| ------------ | ------------------------------- | --- | --- | --- | --- | --------------- | --- | ------------ | --- | ----------------- | --- | ------ |
| RelatedWork. | Onemaybetemptedtoviewmarginasan |     |     |     |     |                 |     |              |     |                   |     |        |
whatkindofnoisetoadd).
intrinsicmeasuretoestimatethegeneralizationofamodel.
However, when we have models with intermediate repre- Thenotionofrarityconsideredinthisworkmayberegarded
sentations, the notion of margin by itself is not adequate as a local notion, since we count the patterns at a signal
sinceanadversarycanoverfittoafavorableintermediate (or a group of signals in the case of Composite CFS) in
representationthatiseasilylinearlyseparated(butotherwise isolation. It is possible to extend this notion to a global
arbitrary). However,recentworkinthisarea(e.g.,Bartlett notion of rarity by propagating occurrence counts along
etal.(2017))hasfocusedonmarginsnormalizedbyspectral withsignalvaluesduringsimulation. Inthecorresponding
complexity(i.e.,ameasurerelatedtotheLipschitzconstant CFS, a pattern is perturbed when its global rarity drops
ofthenetwork)andinthatcasetheaboveargumentdoesnot

Circuit-BasedIntrinsicMethodstoDetectOverfitting
| belowthegiventhreshold. |                | Preliminaryexperimentsshow |              |                | References                              |     |     |     |     |     |          |     |
| ----------------------- | -------------- | -------------------------- | ------------ | -------------- | --------------------------------------- | --- | --- | --- | --- | --- | -------- | --- |
| that this               | more stringent | notion                     | of rarity is | more effective |                                         |     |     |     |     |     |          |     |
|                         |                |                            |              |                | Arora,S.,Ge,R.,Neyshabur,B.,andZhang,Y. |     |     |     |     |     | Stronger |     |
indetectingoverfitinthecaseofrandomforests,andmay
|     |     |     |     |     | generalization |     | bounds | for deep | nets | via | a compression |     |
| --- | --- | --- | --- | --- | -------------- | --- | ------ | -------- | ---- | --- | ------------- | --- |
offerawaytopartiallymitigatetheadversarialattackon
|     |     |     |     |     | approach. | CoRR, | abs/1802.05296, |     |     | 2018. | URL | http: |
| --- | --- | --- | --- | --- | --------- | ----- | --------------- | --- | --- | ----- | --- | ----- |
CFSoutlinedintheprevioussection.
//arxiv.org/abs/1802.05296.
Finally,basedoninsightsfromouranalysisofSimpleCFS,
wewouldliketocontinuethesearchforanintrinsicmethod Arpit,D.,Jastrzebski,S.K.,Ballas,N.,Krueger,D.,Bengio,
|     |     |     |     |     | E., Kanwal, |     | M. S., | Maharaj, | T., Fischer, |     | A., Courville, |     |
| --- | --- | --- | --- | --- | ----------- | --- | ------ | -------- | ------------ | --- | -------------- | --- |
thatdoesnotdependonthemodelstructureandisadversar-
|     |     |     |     |     | A.C.,Bengio,Y.,andLacoste-Julien,S. |     |     |     |     |     | Acloserlook |     |
| --- | --- | --- | --- | --- | ----------------------------------- | --- | --- | --- | --- | --- | ----------- | --- |
iallyrobust,ortoshowthatsuchamethoddoesnotexist,
|                                          |     |     |     |             | at memorization |     | in  | deep networks. |     | In Proceedings |     | of  |
| ---------------------------------------- | --- | --- | --- | ----------- | --------------- | --- | --- | -------------- | --- | -------------- | --- | --- |
| evenforlearningtasksofpracticalinterest. |     |     |     | Asageneral- |                 |     |     |                |     |                |     |     |
the34thInternationalConferenceonMachineLearning,
izationofthisidea,itisinterestingtocontemplatelearning
ICML2017,Sydney,NSW,Australia,6-11August2017,
algorithmsthatproducecertificatesofgeneralization,much
|     |     |     |     |     | pp. 233–242, |     | 2017. | URL | http://proceedings. |     |     |     |
| --- | --- | --- | --- | --- | ------------ | --- | ----- | --- | ------------------- | --- | --- | --- |
likeaBooleansatisfiabilitysolvercanproduceacertificate
mlr.press/v70/arpit17a.html.
ofsatisfiabilityorofunsatisfiability.
|             |                               |     |     |        | Bartlett,P.L.,Foster,D.J.,andTelgarsky,M.J. |     |     |     |     |     | Spectrally- |     |
| ----------- | ----------------------------- | --- | --- | ------ | ------------------------------------------- | --- | --- | --- | --- | --- | ----------- | --- |
| Postscript: | TheTheoryofCoherentGradients. |     |     | Theob- |                                             |     |     |     |     |     |             |     |
servations in thispaper from applyingCFS toneural net- normalizedmarginboundsforneuralnetworks.InGuyon,
works—particularly,theabsenceof“bruteforce”memo- I.,Luxburg,U.V.,Bengio,S.,Wallach,H.,Fergus,R.,
|     |     |     |     |     | Vishwanathan, |     | S., and | Garnett, | R.  | (eds.), | Advances | in  |
| --- | --- | --- | --- | --- | ------------- | --- | ------- | -------- | --- | ------- | -------- | --- |
rizationwithrandomlabels,andtheexistenceofeasyand
hardexamplesasdiscussedinSection4—inspiredthede- Neural Information Processing Systems 30, pp. 6240–
6249.CurranAssociates,Inc.,2017.
velopmentofatheorycalledCoherentGradients(CG)that
providesasimpleandintuitiveexplanationofgeneralization
|     |     |     |     |     | Bernier, | J., Ortega, | J., | Ros Vidal, | E., | Rojas, | I., and | Pri- |
| --- | --- | --- | --- | --- | -------- | ----------- | --- | ---------- | --- | ------ | ------- | ---- |
innetworkstrainedwith(stochastic)gradientdescent.
|     |     |     |     |     | eto, A. | A   | quantitative | study | of fault | tolerance, |     | noise |
| --- | --- | --- | --- | --- | ------- | --- | ------------ | ----- | -------- | ---------- | --- | ----- |
ThekeyideainCGisasfollows. Sincetheoverallgradient immunity, and generalization ability of mlps. Neural
g istheaverageofthegradientsoftheindividualtraining Computation, 12:2941–2964, 01 2001. doi: 10.1162/
examples,theremaybecertaincomponents(directions)ofg 089976600300014782.
whicharesignificantlystrongerthanothercomponentsdue
|     |     |     |     |     | Biere, A. | Aigerlibraryandtools, |     |     | 2007. | URLhttp:// |     |     |
| --- | --- | --- | --- | --- | --------- | --------------------- | --- | --- | ----- | ---------- | --- | --- |
tothegradientsofmultipleexamplesbeinginagreement
|                    |            |                                 |            |             | fmv.jku.at/aiger. |     |     | [Online;accessed1-July-2020]. |     |            |     |      |
| ------------------ | ---------- | ------------------------------- | ---------- | ----------- | ----------------- | --- | --- | ----------------------------- | --- | ---------- | --- | ---- |
| onthosecomponents, |            | andtherebyreinforcingeachother. |            |             |                   |     |     |                               |     |            |     |      |
| Since the          | changes to | the trainable                   | parameters | of the net- |                   |     |     |                               |     |            |     |      |
|                    |            |                                 |            |             | Chatterjee,       | S.  | On  | Algorithms                    | for | Technology |     | Map- |
workareproportionaltothegradient,thebiggestchangesto
|     |     |     |     |     | ping. | PhD | thesis, | EECS | Department, |     | Univer- |     |
| --- | --- | --- | --- | --- | ----- | --- | ------- | ---- | ----------- | --- | ------- | --- |
theparametersarebiasedtobenefitmultipleexamples,and
|     |     |     |     |     | sity | of California, |     | Berkeley, | Aug | 2007. |     | URL |
| --- | --- | --- | --- | --- | ---- | -------------- | --- | --------- | --- | ----- | --- | --- |
thereforelikelytogeneralizewell(basedonastabilityargu- http://www2.eecs.berkeley.edu/Pubs/
ment).However,iftherearenosuchstrongcomponents,i.e.,
TechRpts/2007/EECS-2007-100.html.
alltheper-examplegradientsareroughlyorthogonal,then
eachexampleisfittedindependently,andthiscorresponds Chatterjee,S. Coherentgradients: Anapproachtounder-
tomemorization(andpoorgeneralization,againfromasta- standing generalization in gradient descent-based opti-
InProceedingsoftheInternationalConference
| bilityargument). | ThusCGprovidesanuniformexplanation |     |     |     | mization. |     |     |     |     |     |     |     |
| ---------------- | ---------------------------------- | --- | --- | --- | --------- | --- | --- | --- | --- | --- | --- | --- |
ofbothgeneralizationandmemorizationinneuralnetworks. onLearningRepresentationsICLR,2020. URLhttps:
//openreview.net/forum?id=ryeFY0EFwS.
PleaseseeChatterjee(2020)andZielinskietal.(2020)for
amoredetaileddevelopmentofthisidea,includingananal- Dietterich, T. G. Approximate statistical tests for
ysisofeasyandhardexamplesfromthisperspective,anda
|     |     |     |     |     | comparing |     | supervised | classification |     | learning |     | algo- |
| --- | --- | --- | --- | --- | --------- | --- | ---------- | -------------- | --- | -------- | --- | ----- |
naturalmodificationtogradientdescentthatsignificantlyre- rithms. Neural Comput., 10(7):1895–1923, Oc-
ducesoverfittingbysuppressingthosegradientcomponents tober 1998. ISSN 0899-7667. doi: 10.1162/
thatarenotcommontomanyexamples.
|     |     |     |     |     | 089976698300017197. |     |     | URLhttp://dx.doi.org/ |     |     |     |     |
| --- | --- | --- | --- | --- | ------------------- | --- | --- | --------------------- | --- | --- | --- | --- |
10.1162/089976698300017197.
Acknowledgments
|     |     |     |     |     | Dinh, L., | Pascanu, |     | R., Bengio, |     | S., and | Bengio, | Y.  |
| --- | --- | --- | --- | --- | --------- | -------- | --- | ----------- | --- | ------- | ------- | --- |
ThefirstauthorthanksMicheleCovell,AliRahimi,Alex Sharp minima can generalize for deep nets. CoRR,
Alemi,ShumeetBaluja,SergeyIoffe,TomasIzo,Shankar abs/1703.04933, 2017. URL http://arxiv.org/
abs/1703.04933.
| Krishnan,    | Rahul Sukthankar, | and    | Jay Yagnik    | for helpful |     |     |     |     |     |     |     |     |
| ------------ | ----------------- | ------ | ------------- | ----------- | --- | --- | --- | --- | --- | --- | --- | --- |
| discussions. | The second        | author | was supported | in part by  |     |     |     |     |     |     |     |     |
SRCContract2867.001.

Circuit-BasedIntrinsicMethodstoDetectOverfitting
Dwork, C., Feldman, V., Hardt, M., Pitassi, T., Reingold, Rangamani,A.,Nguyen,N.H.,Kumar,A.,Phan,D.,Chin,
O.,andRoth,A. Thereusableholdout: Preservingvalid- S.H.,andTran,T.D. AScaleInvariantFlatnessMea-
ity in adaptive data analysis. Science, 349(6248):636– sure for Deep Network Minima. arXiv e-prints, art.
638, 2015. ISSN 0036-8075. doi: 10.1126/science. arXiv:1902.02434,Feb2019.
| aaa9375. | URL https://science.sciencemag. |     |     |                                 |     |                |        |
| -------- | ------------------------------- | --- | --- | ------------------------------- | --- | -------------- | ------ |
|          |                                 |     |     | Xiao,H.,Rasul,K.,andVollgraf,R. |     | Fashion-mnist: | anovel |
org/content/349/6248/636.
imagedatasetforbenchmarkingmachinelearningalgo-
LeCun,Y.andCortes,C.MNISThandwrittendigitdatabase.
|     |     |     |     | rithms. arXiv, | 2017. URL | https://arxiv.org/ |     |
| --- | --- | --- | --- | -------------- | --------- | ------------------ | --- |
http://yann.lecun.com/exdb/mnist/, 2010. URL http: abs/1708.07747.
//yann.lecun.com/exdb/mnist/.
Zhang,C.,Bengio,S.,Hardt,M.,Recht,B.,andVinyals,O.
| Neyshabur, | B., Li, Z., | Bhojanapalli, | S., LeCun, | Y., and |     |     |     |
| ---------- | ----------- | ------------- | ---------- | ------- | --- | --- | --- |
Understandingdeeplearningrequiresrethinkinggeneral-
| Srebro, | N. Towards | understanding | the role | of over- |     |     |     |
| ------- | ---------- | ------------- | -------- | -------- | --- | --- | --- |
ization. InProceedingsoftheInternationalConference
parametrization in generalization of neural networks. onLearningRepresentationsICLR,2017.
| CoRR,abs/1805.12076,2018. |     | URLhttp://arxiv. |     |     |     |     |     |
| ------------------------- | --- | ---------------- | --- | --- | --- | --- | --- |
org/abs/1805.12076. Zielinski, P., Krishnan, S., and Chatterjee, S. Weak and
stronggradientdirections:Explainingmemorization,gen-
| Pedregosa, | F., Varoquaux, | G., Gramfort, | A., Michel,       | V.,          |              |             |                  |
| ---------- | -------------- | ------------- | ----------------- | ------------ | ------------ | ----------- | ---------------- |
|            |                |               |                   | eralization, | and hardness | of examples | at scale. ArXiv, |
| Thirion,   | B., Grisel,    | O., Blondel,  | M., Prettenhofer, | P.,          |              |             |                  |
abs/2003.07422,2020. URLhttps://arxiv.org/
Weiss,R.,Dubourg,V.,Vanderplas,J.,Passos,A.,Cour-
abs/2003.07422.
napeau,D.,Brucher,M.,Perrot,M.,andDuchesnay,E.
| Scikit-learn: | Machine | learning in | Python. Journal | of  |     |     |     |
| ------------- | ------- | ----------- | --------------- | --- | --- | --- | --- |
MachineLearningResearch,12:2825–2830,2011.
