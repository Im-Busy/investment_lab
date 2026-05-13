<!-- Page 1 -->

Deep Learning for Portfolio Optimization
Zihao Zhang, Stefan Zohren, Stephen Roberts
Oxford-Man Instituteof Quantitative Finance,
Universityof Oxford
Abstract
We adopt deep learning models to directly optimise the portfolio Sharpe ratio.
Theframeworkwepresentcircumventstherequirementsforforecastingexpected
returns and allows us to directly optimise portfolio weights by updating model
parameters. Insteadofselectingindividualassets, wetrade Exchange-Traded Funds
(ETFs) ofmarketindicestoformaportfolio. Indicesofdifferentassetclassesshow
robustcorrelationsandtradingthemsubstantiallyreducesthespectrumofavailable
assetstochoosefrom. Wecompareourmethodwithawiderangeofalgorithms
withresultsshowingthatourmodelobtainsthebestperformanceoverthetesting
period, from2011 totheendof April2020, includingthefinancialinstabilities
ofthefirstquarterof2020. Asensitivityanalysisisincludedtounderstandthe
relevanceofinputfeaturesandwefurtherstudytheperformanceofourapproach
underdifferentcostratesanddifferentrisklevelsviavolatilityscaling.
1 Introduction
Portfoliooptimisationisanessentialcomponentofatradingsystem. Theoptimisationaimstoselect
thebestassetdistributionwithinaportfolioinordertomaximisereturnsatagivenrisklevel. This
theorywaspioneeredin Markowitz’skeywork[20]andiswidelyknownasmodernportfoliotheory
(MPT).Themainbenefitofconstructingsuchaportfoliocomesfromthepromotionofdiversification
thatsmoothesouttheequitycurve, leadingtoahigherreturnperriskthantradinganindividualasset.
This observation has been proven (see e.g. [40]) showing that the risk (volatility) of a long-only
portfolioisalwayslowerthanthatofanindividualasset, foragivenexpectedreturn, aslongasassets
arenotperfectlycorrelated. Wenotethatthisisanaturalconsequenceof Jensen’sinequality[16].
Despitetheundeniablepowerofsuchdiversification, itisnotstraightforwardtoselectthe“right”
assetallocationsinaportfolio, asthedynamicsoffinancialmarketschangesignificantlyovertime.
Assetsthatexhibit, forexample, strongnegativecorrelationsinthepastcouldbepositivelycorrelated
inthefuture. Thisaddsextrarisktotheportfolioanddegradessubsequentperformance. Further, the
universeofavailableassetsforconstructingaportfolioisenormous. Takingthe USstockmarketsas
asingleexample, morethan5000 stocksareavailabletochoosefrom[34]. Indeed, awellrounded
portfolionotonlyconsistsofstocks, butalsoistypicallysupplementedwithbondsandcommodities,
furtherexpandingthespectrumofchoices.
Inthiswork, weconsiderdirectlyoptimisingaportfolio, utilisingdeeplearningmodels[18,12].
Unlikeclassicalmethods[20]whereexpectedreturnsarefirstpredicted (typicallythroughecono-
metricmodels), webypassthisforecastingsteptodirectlyobtainassetallocations. Severalworks
[25, 24, 39] have shown that the return forecasting approach is not guaranteed to maximise the
performanceofaportfolio, asthepredictionstepsattempttominimiseapredictionlosswhichisnot
theoverallrewardfromtheportfolio. Incontrast, ourapproachistodirectlyoptimisethe Sharpe
ratio[29], thusmaximisingreturnperunitofrisk. Ourframeworkstartsbyconcatenatingmultiple
featuresfromdifferentassetstoformasingleobservationandthenusesaneuralnetworktoextract
salientinformationandoutputportfolioweightssoastomaximisethe Sharperatio.
Website:zihao-z.com. Email:zihao@robots.ox.ac.uk
1202
na J
32
]MP.nif-q[
3 v56631.5002:vi Xra


<!-- Page 2 -->

0.8
S&B
0.6
S&V 0.4
0.2
S&C
0.0
B&V
-0.2
B&C -0.4
-0.6
V&C
-0.8
Figure1: Heatmapforrollingcorrelationsbetweendifferentindexpair. (S:stockindex, B:bond
index, C:commodityindexand V:volatilityindex.)
Insteadofchoosingindividualassets, Exchange-Traded Funds (ETFs)[11]ofmarketindicesare
selectedtoformaportfolio. Weusefourmarketindices: UStotalstockindex (VTI), USaggregate
bondindex (AGG), UScommodityindex (DBC) and Volatility Index (VIX).Alloftheseindicesare
popularlytraded ETFsthatofferhighliquidityandrelativelysmallexpenseratios. Tradingindices
substantiallyreducesthepossibleuniverseofassetchoicesandgainsexposuretomostsecurities.
Further, theseindicesaregenerallyuncorrelated, orevennegativelycorrelated, asshownin Figure1.
Individualinstrumentsinthesameassetclass, however, oftenexhibitstrongpositivecorrelations.
Forexample, morethan75%stocksarehighlycorrelatedwiththemarketindex[34], therebyadding
themtoaportfoliohelpslesswithdiversification.
Weareawarethatsubsectorindicescanbeincludedinaportfolio, ratherthanusingthetotalmarket
index, sincesub-industriesperformatdifferentlevelsandaweightingongoodperformanceina
sectorwouldthereforedeliverextrareturns. However, weseesubsectorindicesashighlycorrelated,
thusaddingthemagainprovidesminimaldiversificationfortheportfolio, andrisksloweringreturns
per unit risk. If higher returns are desired, we can use (e.g.) volatility scaling to upweight our
positionsandamplifyreturns. Wethereforedonotbelievethereisaneedtofindthebestperforming
sector. Instead, weaimtoprovideaportfoliothatdelivershighreturnperunitrisk, andallowsfor
volatilityscaling[26,13,19]toachievedesiredreturnlevels.
Outline: Theremainderofthepaperisstructuredasfollows. Weintroducerelevantliteraturein
Section2 andpresentourmethodologyin Section3. Section4 describesourexperimentsanddetails
theresultsofourmethodcomparedwitharangeofbaselinealgorithms. In Section5, wesummarise
ourfindingsanddiscusspossiblefuturework.
2 Literature Review
Inthissection, wereviewpopularportfoliooptimisationmethodsanddiscusshowdeeplearning
modelshavebeenappliedtothisfield. Thereisavastliteratureavailableonthistopic, soweaim
merelytohighlightkeyconcepts, popularintheindustryorinacademicstudy. Oneofthepopular
practicalapproachesisthereallocationstrategy[34]adoptedbymanypensionfunds (forexample,
Life Strategy Equity Fund, Vanguard). This approach constructs a portfolio by only investing in
stocks and bonds. A typical risk moderate portfolio would, for example, comprise 60% equities
and40%bondsandtheportfolioneedstobeonlyrebalancedsemi-annuallyorannuallytomaintain
thisallocationratio. Themethoddeliversgoodperformanceoverthelongterm, howeverthefixed
allocation ratio means that investors with preference for more weight on stocks need to tolerate
potentiallylargedrawdownsduringdullmarkets.
Mean-varianceanalysisor MPT[20]isusedformanyinstitutionalportfoliosthatsolvesaconstraint
optimisation problem to derive portfolio weights. Despite its popularity, the assumptions of the
theoryareundercriticismastheyareoftennotobeyedinrealfinancialmarkets. Inparticular, returns
areassumedtofollowa Gaussiandistributionin MPT, therefore, investorsonlyconsiderexpected
returnandvarianceoftheportfolioreturnstomakedecisions. However, itiswidelyaccepted (see
2


<!-- Page 3 -->

forinstance[4,38]) thatreturnstendtohavefattailsandextremelossesaremorelikelytooccurin
practice, leadingtoseveredrawdownsthatarenotbearable. The Maximum Diversification (MD)
portfolioisanotherpromisingmethodintroducedin[3]thataimstomaximisethediversificationof
aportfolio, therebyaimingtohaveminimallycorrelatedassetssotheportfoliocanachievehigher
returns (and lower risk) than other classical methods. We compare our model with both these
strategies, withtheresultssuggestingthatourmethodsdeliverbetterperformanceandtoleratelarger
transactioncoststhaneitherofthesebenchmarks.
Stochastic Portfolio Theory (SPT) was recently proposed in [7, 9]. Unlike other methods, SPT
aimstoachieverelativearbitragesmeaningtoselectportfoliosthatcanoutperformamarketindex
withprobabilityone. Suchinvestmentstrategieshavebeenstudiedin[5,6,27,36]. However, the
numberofrelativearbitragestrategiesremainssmall, astheorydoesnotsuggesthowtoconstructsuch
strategies. Wecancheckwhetheragivenstrategyisarelativearbitrage, butitisnon-trivialtodevelop
oneexante. Inthiswork, weincludeaparticularclassof SPTcalledfunctionallygeneratedportfolio
(FGP)[8]inourexperiment, buttheresultsuggeststhismethoddeliversinferiorperformancethan
otheralgorithmsandgenerateslargeturnovers, makingitunprofitableunderheavytransactioncosts.
Theideaofourend-to-endtrainingframeworkwasfirstinitiatedin[25,24]. However, theseworks
mainlyfocusonoptimisingtheperformanceforasingleassetsothereislittlediscussiononhow
portfoliosshouldbemaximised. Furthermore, theirtestingperiodisfrom1970 to1994, whereas
ourdatasetisuptodateandwestudythebehaviorofourstrategyunderthecurrentcrisisdueto
COVID-19.Wecanalsolinkourapproachtoreinforcementlearning (RL)[31,22,35]whereanagent
interactswithanenvironmenttomaximisecumulativerewards. Theworksof[1,15,39]havestudied
thisstreamandadopted RLtodesigntradingstrategies. However, thegoalof RListomaximise
expectedcumulativerewardssuchasprofitswhereas Sharperatiocannotbedirectlyoptimised.
3 Methodology
Inthissection, weintroduceourframeworkanddiscusshow Sharperatiocanbeoptimisedthrough
gradientascent. Wediscussthetypesofneuralnetworksusedanddetailthefunctionalityofeach
componentinourmethod.
3.1 Objective Function
The Sharperatioisusedtogaugethereturnperriskofaportfolioandisdefinedasexpectedreturn
overvolatility (excludingrisk-freerateforsimplicity):
E(R )
L= p (1)
Std (R )
p
where E(R ) and Std (R ) aretheestimatesofthemeanandstandarddeviationofportfolioreturns.
p p
Specifically, for a trading period of t = {1,··· , T}, we can maximise the following objective
function:
E(R )
L = p, t
T (cid:113)
E(R2 )−(E(R ))2
p, t p, t
(2)
T
1 (cid:88)
E(R )= R
p, t T p, t
t=1
where R isrealizedportfolioreturnovernassetsattimetdenotedas:
p, t
n
(cid:88)
R = w ·r (3)
p, t i, t−1 i, t
i=1
where r is the return of asset i with r = (p /p −1). We represent the allocation ratio
i, t i, t i, t i, t−1
(position) of asset i as w ∈ [0,1] and
(cid:80) nw
= 1. In our approach, a neural network f with
i, t i i, t
parametersθisadoptedtomodelw foralong-onlyportfolio:
i, t
w =f (θ|x ) (4)
i, t t
3


<!-- Page 4 -->

wherex representsthecurrentmarketinformationandwebypasstheclassicalforecastingstepby
t
linkingtheinputswithpositionstomaximisethe Sharpeovertradingperiod T, namely L . However,
T
along-onlyportfolioimposesconstraintsthatrequireweightstobepositiveandsummedtoone, we
usesoftmaxoutputstofulfilltheserequirements:
exp (w˜ )
w = i, t , wherew˜ aretherawweights. (5)
i, t (cid:80) nexp (w˜
)
i, t
j j, t
Suchaframeworkcanbeoptimisedusingunconstrainedoptimisationmethods. Particularly, weuse
gradientascenttomaximisethe Sharperatio. Thegradientof L withrespecttoparametersθ is
T
readilycalculable, withanexcellentderivationpresentedin[25,23]. Onceweobtain∂L /∂θ, we
T
canrepeatedlycomputethisvaluefromtrainingdataandupdatetheparametersbyusinggradient
ascent:
∂L
θ :=θ +α T (6)
new old ∂θ
whereαisthelearningrateandtheprocesscanberepeatedformanyepochsuntiltheconvergenceof
Sharperatioortheoptimisationofvalidationperformanceisachieved.
3.2 Model Architecture
Wedepictournetworkarchitecturein Figure2. Ourmodelconsistsofthreemainbuildingblocks:
inputlayer, neurallayerandoutputlayer. Theideaofthisdesignistouseneuralnetworkstoextract
cross-sectionalfeaturesfrominputassets. Featuresextractedfromdeeplearningmodelshavebeen
suggested to perform better than traditional hand-crafted features [39]. Once features have been
extracted, themodeloutputsportfolioweightsandweobtainrealisedreturnstomaximise Sharpe
ratio. Thefollowingdetailseachcomponentofourmethod.
Inputlayer Wedenoteeachassetas A andwehavenassetstoformaportfolio. Asingleinputis
i
preparedbyconcatenatinginformationfromallassets. Forexample, theinputfeaturesofoneasset
canbeitspastpricesandreturnswithadimensionof (k,2) wherekrepresentsthelookbackwindow.
Bystackingfeaturesacrossallassets, thedimensionoftheresultinginputwouldbe (k,2×n). We
canthenfeedthisinputtothenetworkandexpectnon-linearfeaturesbeingextracted.
Neurallayer Aseriesofhiddenlayerscanbestackedtoformanetwork, however, inpractice,
this part requires lots of experiments as there are plentiful ways of combining hidden layers and
theperformanceoftendependsonthedesignofarchitecture. Wehavetesteddeeplearningmodels
includingfullyconnectedneuralnetwork (FCN), convolutionalneuralnetwork (CNN) and Long
Short-Term Memory (LSTM)[14]. Overall, LSTMsdeliverthebestperformanceformodellingdaily
financialdataandanumberofworks[33,19,39]supportthisobservation.
Wenotetheproblemof FCNisitsproblemofsevereoverfitting. Asitassignsparameterstoeach
inputfeature, thisresultsinanexcessnumberofparameters. The LSTMoperateswithacellstructure
thathasgatemechanismstosummariseandfilterinformationfromitslonghistory, sothemodelends
upwithfewertrainableparametersandachievesbettergeneralisationresults. Incontrast, CNNswith
astrongsmoothing (typicaloflargeconvolutionalfilters) tendtohaveunderfittingproblems, such
thatoversmoothsolutionsareobtained. Duetothedesignofparametersharingandtheconvolution
operations, weexperience CNNstooverfiltertheinputs. However, wenotethat CNNsappeartobe
excellentcandidatesformodellinghigh-frequencyfinancialdatasuchaslimitorderbooks[37].
Ouputlayer Inordertoconstructalong-onlyportfolio, weusethesoftmaxactivationfunctionfor
theoutputlayer, whichnaturallyimposesconstraintstokeepportfolioweightspositiveandsumming
toone. Thenumberofoutputnodes (w ,··· , w ) isequaltothenumberofassetsinourportfolio,
1 n
andwecanmultiplytheseportfolioweightswithassociatedassets’returns (r ,··· , r ) tocalculate
1 n
realisedportfolioreturns (R ). Oncerealisedreturnsareobtained, wecanderivethe Sharperatio
p
andcalculatethegradientsofthe Sharperatiowithrespecttothemodelparametersandusegradient
ascenttoupdatetheparameters.
4


<!-- Page 5 -->

! ! !
" # $
Input layer
Hidden Layer
Neural layer
Hidden Layer
) ) )
" # $
Softmax
Output layer
% % %
" $&" $
'
(
Figure2: Modelarchitectureschematic. Overall, ourmodelcontainsthreemainbuildingblocks:
inputlayer, neurallayerandoutputlayer.
4 Experiments
4.1 Descriptionof Dataset
We use four market indices: US total stock index (VTI), US aggregate bond index (AGG), US
commodityindex (DBC) and Volatility Index (VIX).Thesearepopular Exchange-Traded Funds
(ETFs)[11]thathaveexistedformorethan15 years. Asdiscussedin Section1, tradingindicesoffers
advantagesovertradingindividualassetsbecausetheseindicesaregenerallyuncorrelatedresulting
indiversification. Adiversifiedportfoliodeliversahigherreturnperriskandtheideaofourstrategy
istohaveasystemthatdeliversgoodreward-to-riskratio. Ourdatasetrangesfrom2006 to2020 and
containsdailyobservations. Weretrainourmodelatevery2 yearsandusealldataavailableupto
thatpointtoupdateparameters. Overall, ourtestingperiodisfrom2011 totheendof April2020,
includingthemostrecentcrisisdueto COVID-19.
4.2 Baseline Algorithms
Wecompareourmethodwithagroupofbaselinealgorithms. Thefirstsetofbaselinemodelsare
reallocationstrategiesadoptedbymanypensionfunds. Thesestrategiesassignafixedallocation
ratio to relevant assets and rebalance portfolios annually to maintain these ratios. Investors can
selectaportfoliobasedontheirriskpreferences. Ingeneral, portfoliosweightedmoreonequities
woulddeliverbetterperformanceattheexpenseoflargervolatility. Inthiswork, weconsiderfour
suchstrategies: Allocation1(25%shares,25%bonds,25%commoditiesand25%volatilityindex),
Allocation2(50%shares,10%bonds,20%commodities, and20%volatilityindex), Allocation3
(10%shares,50%bonds,20%commodities, and20%volatilityindex), and Allocation4(40%shares,
40%bonds,10%commoditiesand10%volatilityindex).
The second set of comparison models are mean-variance optimisation (MV) [20] and maximum
diversification (MD)[32]. Weusemovingaverageswitharollingwindowof50 daystoestimate
theexpectedreturnsandcovariancematrix. Theportfolioweightsareupdatedatadailybasisand
weselectweightsthatmaximise Sharperatiofor MV.Thelastbaselinealgorithmisthediversity-
weightedportfolio (DWP) from Stochastic Portfolio Theorypresentedin[28]. The DWPrelates
5


<!-- Page 6 -->

portfolioweightstoassets’marketcapitalisationandithasbeensuggestedtobeabletooutperform
themarketindexwithcertainty[10].
4.3 Training Scheme
In this work, we use a single layer of LSTM connectivity, with 64 units, to model the portfolio
weightsandthencetooptimisethe Sharperatio. Wepurposelykeepournetworksimpletoindicate
the effectiveness of this end-to-end training pipeline instead of carefully fine-tuning the “right”
hyperparameters. Ourinputcontainsclosepricesanddailyreturnsforeachmarketindexandwe
takethepast50 daysoftheseobservationstoformasingleinput. Weareawarethatreturnscan
bederivedfromprices, butkeepingreturnshelpwiththeevaluationof Equation7 andwecanalso
treatthemasmomentumfeaturesin[26]. Asourfocusisnotonfeatureselection, wechoosethese
commonlyusedfeaturesinourwork. The Adamoptimiser[17]isusedfortrainingournetwork, and
themini-batchsizeis64. Wetake10%ofanytrainingdataasaseparatevalidation-settooptimise
hyperparametersandcontroloverfittingproblems. Anyhyperparameteroptimisationisdoneonthe
validationset, leavingthetestdataforthefinalperformanceevaluationandensuringthevalidityof
ourresults. Ingeneral, ourtrainingprocessstopsafter100 epochs.
4.4 Experimental Results
When reporting the test performance, we include transaction costs and use volatility scaling [26,
19,39]toscaleourpositionsbasedonmarketvolatility. Wecansetourownvolatilitytargetand
meetexpectationsofinvestorswithdifferentriskpreferences. Oncevolatilitiesareadjusted, our
investmentperformancesaremainlydrivenbystrategiesinsteadofbeingheavilyaffectedbymarkets.
Themodifiedportfolioreturncanbedefinedas:
R = (cid:88) n σ tgt w ·r −C· (cid:88) n (cid:12) (cid:12) σ tgt w − σ tgt w (cid:12) (cid:12) (7)
p, t σ i, t−1 i, t (cid:12)σ i, t−1 σ i, t−2(cid:12)
i, t−1 i, t−1 i, t−2
i i
whereσ isthevolatilitytargetandσ isanex-antevolatilityestimateofasseticalculated
tgt i, t−1
usinganexponentiallyweightedmovingstandarddeviationwitha50-daywindowonr . Weuse
i, t
dailychangesoftradedvalueofanassettorepresenttransactioncosts, whichiscalculatedbythe
secondtermin Equation7. C (=1 bs=0.0001) isthecostrateandwechangeittoreflecthowour
modelperformsunderdifferenttransactioncosts.
Toevaluatetheperformanceofourmethods, weutilisefollowingmetrics: expectedreturn (E(R)),
standarddeviationofreturn (Std (R)), Sharperatio[29], downsidedeviationofreturn (DD(R))[21],
and Sortinoratio[30].Allofthesemetricsareannualised, andwealsoreportonmaximumdrawdown
(MDD)[2], percentageofpositivereturn (%of+Ret) andtheratiobetweenpositiveandnegative
return (Ave. P/Ave. L).
Table1 presentstheresultsofourmodel (DLS) compardtootherbaselinealgorithms. Thetopofthe
tableshowstheresultswithoutusingvolatilityscaling, andwecanseethatourmodel (DLS) achieves
thebest Sharpe’sratioand Sortinoratio, deliveringthehighestreturnperrisk. However, giventhe
largedifferencesinvolatilities, wecannotdirectlycompareexpectedandcumulativereturnsfor
differentmethods, therebyvolatilityscalingalsohelpstomakefaircomparisons.
Oncevolatilitiesarescaled (showninthemiddleof Table1), DLSdeliversthebestperformance
acrossallevaluationmetricsexceptforaslightlylargerdrawdown. Ifwelookatthecumulative
returnsin Figure3, DLSexhibitsoutstandingperformanceoverthelonghaulandthemaximum
drawdownisreasonable, ensuringtheconfidenceofinvestorstoholdthroughhardtimes. Further, if
welookatthebottomof Table1 wherealargecostrate (C =0.1%) isused, ourmodel (DLS) stills
deliversthebestexpectedreturnandachievesthehighest Sharpeand Sortinoratios.
However, withahighercostrate, wecanseethatreallocationstrategiesworkwelland, inparticular,
Allocations3 and4 achievecomparableresultstoourmethod. Inordertoinvestigatewhyperformance
gapdiminisheswithahighercostrate, wepresenttheboxplotsforannualrealisedtradereturnsand
accumulatedcostsfordifferentassetsin Figure4. Overall, ourmodeldeliversbetterrealisedreturns
thanreallocationstrategies, butwealsoaccumulatemuchlargertransactioncostssinceourpositions
areadjustedonadailybasis, leadingtoahigherturnover.
6


<!-- Page 7 -->

Table1: Experimentresultsfordifferentalgorithms.
E(R) Std (R) Sharpe DD(R) Sortino MDD %of+Ret Ave. P
Ave. L
Novolatilityscalingand C =0.01%
Allocation1 0.282 0.303 0.929 0.136 2.065 0.142 0.479 1.193
Allocation2 0.249 0.212 1.173 0.095 2.616 0.097 0.483 1.254
Allocation3 0.228 0.256 0.890 0.116 1.962 0.122 0.476 1.183
Allocation4 0.152 0.123 1.228 0.052 2.932 0.081 0.505 1.349
MV 0.082 0.108 0.759 0.069 1.192 0.195 0.562 1.199
MD 0.462 0.523 0.882 0.239 1.931 0.273 0.473 1.182
DWP 0.051 0.102 0.493 0.067 0.740 0.179 0.549 1.107
DLS 0.313 0.168 1.858 0.099 3.135 0.102 0.537 1.518
Volatilityscaling (σ =0.10) and C =0.01%
tgt
Allocation1 0.160 0.105 1.526 0.061 2.629 0.111 0.554 1.289
Allocation2 0.123 0.106 1.146 0.065 1.861 0.127 0.549 1.211
Allocation3 0.145 0.105 1.383 0.061 2.396 0.105 0.542 1.259
Allocation4 0.164 0.104 1.579 0.064 2.588 0.112 0.565 1.303
MV 0.112 0.100 1.120 0.063 1.767 0.211 0.561 1.213
MD 0.157 0.106 1.484 0.065 2.414 0.125 0.565 1.297
DWP 0.089 0.109 0.818 0.069 1.291 0.115 0.556 1.148
DLS 0.206 0.105 1.962 0.062 3.322 0.123 0.559 1.375
Volatilityscaling (σ =0.10) and C =0.1%
tgt
Allocation1 0.133 0.105 1.274 0.061 2.172 0.113 0.548 1.236
Allocation2 0.105 0.107 0.986 0.066 1.590 0.244 0.547 1.179
Allocation3 0.117 0.105 1.110 0.061 1.903 0.107 0.538 1.203
Allocation4 0.135 0.104 1.299 0.064 2.108 0.114 0.559 1.244
MV 0.019 0.101 0.191 0.066 0.293 0.324 0.537 1.033
MD 0.095 0.106 0.899 0.066 1.431 0.145 0.549 1.171
DWP -0.083 0.110 -0.753 0.074 -1.129 0.627 0.508 0.880
DLS 0.148 0.105 1.403 0.063 2.327 0.125 0.547 1.272
Forreallocationstrategies, dailypositionchangesareonlyupdatedforvolatilityscaling. Otherwise,
weonlyactivelychangepositionsonceayeartorebalanceandmaintaintheallocationratio. Asa
result, reallocationstrategiesdeliverminimaltransactioncosts. Thisanalysisaimstoindicatethe
validityofourresultsandshowthatourmethodcanworkunderunfavorableconditions.
4.5 Model Performanceduring2020 Crisis
Due to the recent COVID-19 pandemic, global stock markets fell dramatically and experienced
extremevolatility. Thecrashstartedonthe24 th February2020 wheremarketsreportedtheirlargest
3.0 A A l l l l o o c c a a t t i i o o n n 1 2 1.75 A A l l l l o o c c a a t t i i o o n n 1 2 A A l l l l o o c c a a t t i i o o n n 1 2
Allocation 3 1.50 Allocation 3 1.0 Allocation 3
2.5 Allocation 4 Allocation 4 Allocation 4
2.0 M M D D W L V D S P 1 1 . . 0 2 0 5 M M D D W L V D S P 0.5 M M D D W L V D S P
1.5
0.75 0.0
1.0
0.50
0.5 0.25 0.5
0.0 0.00
1.0
2011201220132014201520162017201820192020 2011201220132014201520162017201820192020 2011201220132014201520162017201820192020
Figure 3: Cumulative returns (logarithmic scale) for Left: no volatility scaling and C = 0.01%;
Middle: volatilityscaling (σ =0.10) and C =0.01%;Right: volatilityscaling (σ =0.10) and
tgt tgt
C =0.1%.
7


<!-- Page 8 -->

0.4
0.2
0.0
0.2
0.4
Stock Bond Volatility Commodity
Asset Classes
snrute R
desilae R
Allocation 1
Allocation 2
Allocation 3
Allocation 4
MV
MD
DWP
DLS
0.010
0.008
0.006
0.004
0.002
0.000
Stock Bond Volatility Commodity
Asset Classes
stso C
launn A
Allocation 1
Allocation 2
Allocation 3
Allocation 4
MV
MD
DWP
DLS
Figure 4: Boxplot for Top: annual realised trade returns; Bottom: annual accumulated costs for
differentassetswithvolatilityscaling (σ =0.10) and C =0.01%.
tgt
one-weekdeclinessincethe2008 financialcrisis. Lateron, withanoilpricewarbetween Russiaand
the OPECcountries, marketsfurtherdampenedandencounteredthelargestsingle-daypercentage
dropsince Black Mondayin1987. Asof March2020, wehaveseenadownturnofatleast25%in
the USmarketsand30%inmost G20 countries. Thecrisisshatteredmanyinvestors’confidenceand
resultedinagreatlossoftheirwealths. However, italsoprovidesuswithagreatopportunitytostress
testourmethodandunderstandhowourmodelperformsduringthecrisis.
Inordertostudythemodelbehaviours, weplothowouralgorithmallocatedtheassetsfrom January
to April2020 in Figure5. Atthebeginningof2020, wecanseethatourmodelhadaquitediverse
holding. However, afterasmalldipinstockindexinearly February, wealmosthadonlybondsinour
portfolio. Thereweresomeequitypositionsleftbutverysmallpositionsforvolatilityandcommodity
indices. Whenthecrashstartedon24 th February, ourholdingswereconcentratedonthebondindex
whichisconsideredtobesafeassetsduringthecrisis. Interestingly, thebondindexalsofellthistime
(inthemiddleof March) althoughitreboundedquitequickly. Duringthebondfalling, ouroriginal
positionsdidnotchangemuchbutthescaledpositionsdecreasedalotforthebondindexduetoa
spikingvolatility, thereforeourdrawdownwassmall. Overall, wecanseethatourmodeldelivers
reasonableallocationsduringthecrisisandourpositionsareprotectedthroughvolatilityscaling.
4.6 Sensitivity Analysis
In order to understand how input features affect our decisions, we study the sensitivity analysis
presentedin[24]forourmethod. Theabsolutenormalisedsensitivityoffeaturex isdefinedas:
i
d L
S i = dx (cid:12) i (cid:12) (8)
max (cid:12) d L(cid:12)
j (cid:12) dxj (cid:12)
where L represents the objective function and S captures the relative sensitivity for feature x
i i
comparedwithotherfeatures. Weplotthetime-varyingsensitivitiesforallfeaturesin Figure6. The
y-axisindicatesthe400 featureswehavebecauseweuse4 indices (eachwithpricesandreturns) and
8


<!-- Page 9 -->

170 Stock Position 1.2 118 Bond Position
Scaled position Scaled position 6
1.0 116
160
5
0.8 114
150
4
140 0.6 112
3
130 0.4 110
2
120 0.2 108
1
110 0.0 106
2020-01-012020-01-15 2020-02-012020-02-15 2020-03-012020-03-15 2020-04-012020-04-15 2020-01-012020-01-15 2020-02-012020-02-15 2020-03-012020-03-15 2020-04-012020-04-15
0.25
80 Volatility Position 0.16 16 Commodity Position
Scale position Scaled position
0.14
70 0.20
15
0.12
60
0.10 14 0.15
50
0.08
40 0.06 13 0.10
30 0.04 12
0.05
20 0.02
11
10 0.00 0.00
2020-01-012020-01-15 2020-02-012020-02-15 2020-03-012020-03-15 2020-04-012020-04-15 2020-01-012020-01-15 2020-02-012020-02-15 2020-03-012020-03-15 2020-04-012020-04-15
Figure 5: Shifts of portfolio weights for our model (DLS) during the crisis of COVID-19 with
volatilityscaling (σ =0.10).
tgt
wetakeatimeframeofpast50 observationstoformasingleinputsothereare400 featuresintotal.
Therowlabeled“Sprice”representspricefeaturesforthestockindexandthebottomofrow“Sprice”
meansthemostrecentpriceforthatobservation. Sameconventionisusedforallotherfeatures.
The importance of features varies over the time, but the most recent features always make the
biggestcontributionsaswecanseethatthebottomofeachfeaturerowhasthehighestweight. This
observationmeetsourunderstandingas, fortime-series, recentobservationscarrymoreinformation.
Thefurtherawayfromthecurrentobservationpoint, thelessimportanceoffeaturesshowandwe
couldadjustfeaturesusedbasedonthisobservationsuchasusingasmalllookbackwindow.
0.5
0.4
0.3
0.2
0.1
Figure6: Sensitivityanalysisforinputfeaturesoverthetime.
9


<!-- Page 10 -->

5 Conclusion
Inthiswork, weadoptdeeplearningmodelstodirectlyoptimiseaportfolio’s Sharperatio. This
pipeline bypasses the traditional forecasting step and allows us to optimise portfolio weights by
updatingmodelparametersthroughgradientascent. Insteadofusingindividualassets, wefocuson
ETFsofmarketindicestoformaportfolio. Doingthissubstantiallyreducesthescopeofpossible
assetstochoosefrom, andtheseindiceshaveshownrobustcorrelations. Inthiswork, fourmarket
indiceshavebeenusedtoformaportfolio.
Wecompareourmethodwithawiderangeofpopularalgorithmsincludingreallocationstrategies,
classicalmean-varianceoptimisation, maximumdiversificationandstochasticportfoliotheorymodel.
Ourtestingperiodisfrom2011 tothe Aprilof2020, andincludetherecentcrisisdueto COVID-19.
Theresultsshowthatourmodeldeliversthebestperformanceandadetailedstudyofourmodel
performanceduringthecrisisshowstherationalityandpracticabilityofourmethod. Asensitivity
analysisisincludedtounderstandhowinputfeaturescontributetooutputsandtheobservationsmeet
oureconometricunderstanding, showingthemostrecentfeaturesaremostrelevant.
In subsequent continuation of this work, we aim to study portfolios performance under different
objectivefunctions. Giventheflexibleframeworkofourapproach, wecanmaximise Sortinoratioor
eventhediversificationdegreeofaportfolioaslongasfunctionsaredifferentiable. Wefurthernote
thatthevolatilityestimatesusedforscalingarelaggedestimatesthatdonotnecessarilyrepresent
currentmarketvolatilities. Weconsideranotherextensiontothisworktothusadaptthenetwork
architecturetoinfer (future) volatilityestimatesasapartofthetrainingprocess.
Acknowledgements
Theauthorswouldliketothankmembersof Machine Learning Research Groupatthe Universityof
Oxfordfortheirusefulcomments. Wearemostgratefultothe Oxford-Man Instituteof Quantitative
Financeforsupportanddataaccess.
References
[1] Francesco Bertoluzzoand Marco Corazza. Testingdifferentreinforcementlearningconfigura-
tionsforfinancialtrading: Introductionandapplications. Procedia Economicsand Finance,
3:68–77,2012.
[2] Alexei Chekhlov, Stanislav Uryasev, and Michael Zabarankin. Drawdownmeasureinportfolio
optimization. International Journalof Theoreticaland Applied Finance,8(01):13–58,2005.
[3] Yves Choueifatyand Yves Coignard. Towardmaximumdiversification. The Journalof Portfolio
Management,35(1):40–51,2008.
[4] Rama Contand De Nitions. Statisticalpropertiesoffinancialtimeseries. 1999.
[5] Daniel Fernholzand Ioannis Karatzas. Onoptimalarbitrage. The Annalsof Applied Probability,
pages1179–1204,2010.
[6] Daniel Fernholz, Ioannis Karatzas, etal. Optimalarbitrageundermodeluncertainty. The Annals
of Applied Probability,21(6):2191–2225,2011.
[7] ERobert Fernholz. Stochasticportfoliotheory. In Stochastic Portfolio Theory, pages1–24.
Springer,2002.
[8] Robert Fernholz. Portfoliogeneratingfunctions. In Quantitative Analysisin Financial Markets:
Collected Papersofthe New York University Mathematical Finance Seminar, pages344–367.
World Scientific,1999.
[9] Robert Fernholzand Ioannis Karatzas. Stochasticportfoliotheory: Anoverview. Handbookof
numericalanalysis,15:89–167,2009.
[10] Robert Fernholz, Ioannis Karatzas, and Constantinos Kardaras. Diversityandrelativearbitrage
inequitymarkets. Financeand Stochastics,9(1):1–27,2005.
[11] Gary LGastineau. Exchange-tradedfunds. Handbookoffinance,1,2008.
[12] Ian Goodfellow, Yoshua Bengio, and Aaron Courville. Deeplearning. MITpress,2016.
10


<!-- Page 11 -->

[13] Campbell RHarvey, Edward Hoyle, Russell Korgaonkar, Sandy Rattray, Matthew Sargaison,
and Otto Van Hemert. Theimpactofvolatilitytargeting. The Journalof Portfolio Management,
45(1):14–33,2018.
[14] Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural computation,
9(8):1735–1780,1997.
[15] Chien Yi Huang. Financialtradingasagame: Adeepreinforcementlearningapproach. ar Xiv
preprintar Xiv:1807.02787,2018.
[16] Johan Ludwig William Valdemar Jensenetal. Surlesfonctionsconvexesetlesinégalitésentre
lesvaleursmoyennes. Actamathematica,30:175–193,1906.
[17] Diederik PKingmaand Jimmy Ba. Adam: Amethodforstochasticoptimization. Proceedings
ofthe International Conferenceon Learning Representations,2015.
[18] Yann Le Cun, Yoshua Bengio, and Geoffrey Hinton. Deeplearning. Nature,521(7553):436–444,
2015.
[19] Bryan Lim, Stefan Zohren, and Stephen Roberts. Enhancingtime-seriesmomentumstrategies
usingdeepneuralnetworks. The Journalof Financial Data Science,1(4):19–38,2019.
[20] Harry Markowitz. Portfolioselection. Thejournaloffinance,7(1):77–91,1952.
[21] Alexander J Mc Neil, Rüdiger Frey, and Paul Embrechts. Quantitative risk management:
Concepts, techniquesandtools-revisededition. Princetonuniversitypress,2015.
[22] Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan
Wierstra, and Martin Riedmiller. Playing Atariwithdeepreinforcementlearning. NIPSDeep
Learning Workshop2013,2013.
[23] Gabriel Molina. Stocktradingwithrecurrentreinforcementlearning (RRL). CS229, nd Web,
15,2016.
[24] John Moodyand Matthew Saffell. Learningtotradeviadirectreinforcement. IEEEtransactions
onneural Networks,12(4):875–889,2001.
[25] John Moody, Lizhong Wu, Yuansong Liao, and Matthew Saffell. Performancefunctionsand
reinforcementlearningfortradingsystemsandportfolios. Journalof Forecasting,17(5-6):441–
470,1998.
[26] Tobias JMoskowitz, Yao Hua Ooi, and Lasse Heje Pedersen. Timeseriesmomentum. Journal
offinancialeconomics,104(2):228–250,2012.
[27] Johannes Ruf. Hedgingunderarbitrage. Mathematical Finance: An International Journalof
Mathematics, Statisticsand Financial Economics,23(2):297–317,2013.
[28] Yves-Laurent Kom Samoand Alexander Vervuurt. Stochasticportfoliotheory: Amachine
learning perspective. In Proceedings of the Thirty-Second Conference on Uncertainty in
Artificial Intelligence, pages657–665,2016.
[29] William FSharpe. Thesharperatio. Journalofportfoliomanagement,21(1):49–58,1994.
[30] Frank ASortinoand Lee NPrice. Performancemeasurementinadownsideriskframework.
the Journalof Investing,3(3):59–64,1994.
[31] Richard SSuttonand Andrew GBarto. Reinforcementlearning: Anintroduction. MITpress,
2018.
[32] Ludan Theronand Gary Van Vuuren. Themaximumdiversificationinvestmentstrategy: A
portfolioperformancecomparison. Cogent Economics&Finance,6(1):1427533,2018.
[33] Avraam Tsantekidis, Nikolaos Passalis, Anastasios Tefas, Juho Kanniainen, Moncef Gabbouj,
and Alexandros Iosifidis. Usingdeeplearningtodetectpricechangeindicationsinfinancial
markets. In201725 th European Signal Processing Conference (EUSIPCO), pages2511–2515.
IEEE,2017.
[34] Russell Wild. Index Investingfor Dummies. John Wiley&Sons,2008.
[35] Ronald JWilliams. Simplestatisticalgradient-followingalgorithmsforconnectionistreinforce-
mentlearning. Machinelearning,8(3-4):229–256,1992.
[36] Ting-Kam Leonard Wong. Optimizationofrelativearbitrage. Annalsof Finance,11(3-4):345–
382,2015.
11


<!-- Page 12 -->

[37] Zihao Zhang, Stefan Zohren, and Stephen Roberts. Deep LOB: Deep convolutional neural
networksforlimitorderbooks. IEEETransactionson Signal Processing,67(11):3001–3012,
2019.
[38] Zihao Zhang, Stefan Zohren, and Stephen Roberts. Extending deep learning models for
limitorderbookstoquantileregression. Proceedingsof Time Series Workshopofthe36 th
International Conference on Machine Learning, Long Beach, California, PMLR 97, 2019.,
2019.
[39] Zihao Zhang, Stefan Zohren, and Roberts Stephen. Deepreinforcementlearningfortrading.
The Journalof Financial Data Science,2020.
[40] Eric Zivot. Introductiontocomputationalfinanceandfinancialeconometrics. Chapman&Hall
Crc,2017.
12
