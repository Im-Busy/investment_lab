<!-- Page 1 -->

Backtest Overfitting in the Machine Learning Era:
A Comparison of Out-of-Sample Testing Methods in a Synthetic Controlled
Environment
Hamid Ariana,∗,1, Daniel Norouzi Mobarekehb,2 and Luis Secoc,3
c Universityof Toronto,40 St George St, Toronto, Ontario Canada M5 S2 E4
a York University,4700 Keele St, Toronto, Ontario, Canada M3 J1 P3
b Sharif Universityof Technology, Teymoori Sq, Tehran, Iran1459973941
ARTICLE INFO ABSTRACT
Keywords: Thisresearchexplorestheintegrationofadvancedstatisticalmodelsandmachinelearninginfinancial
Quantitative Finance analytics, representingashiftfromtraditionaltoadvanced, data-drivenmethods. Weaddressacritical
Machine Learning gapinquantitativefinance: theneedforrobustmodelevaluationandout-of-sampletestingmethod-
Cross-Validation ologies, particularlytailoredcross-validationtechniquesforfinancialmarkets. Wepresentacompre-
Probabilityof Backtest Overfitting hensiveframeworktoassessthesemethods, consideringtheuniquecharacteristicsoffinancialdata
likenon-stationarity, autocorrelation, andregimeshifts. Throughouranalysis, weunveilthemarked
superiorityofthe Combinatorial Purged (CPCV) methodinmitigatingoverfittingrisks, outperform-
ingtraditionalmethodslike K-Fold, Purged K-Fold, andespecially Walk-Forward, asevidencedby
itslower Probabilityof Backtest Overfitting (PBO) andsuperior Deflated Sharpe Ratio (DSR)Test
Statistic. Walk-Forward, bycontrast, exhibitsnotableshortcomingsinfalsediscoveryprevention,
characterizedbyincreasedtemporalvariabilityandweakerstationarity. Thiscontrastsstarklywith
CPCV’sdemonstrablestabilityandefficiency, confirmingitsreliabilityforfinancialstrategydevel-
opment. Theanalysisalsosuggeststhatchoosingbetween Purged K-Foldand K-Foldnecessitates
cautionduetotheircomparableperformanceandpotentialimpactontherobustnessoftrainingdata
inout-of-sampletesting. Ourinvestigationutilizesa Synthetic Controlled Environmentincorporat-
ingadvancedmodelslikethe Heston Stochastic Volatility, Merton Jump Diffusion, and Drift-Burst
Hypothesis, alongsideregime-switchingmodels. Thisapproachprovidesanuancedsimulationof
marketconditions, offeringnewinsightsintoevaluatingcross-validationtechniques. Ourstudyun-
derscoresthenecessityofspecializedvalidationmethodsinfinancialmodeling, especiallyintheface
ofgrowingregulatorydemandsandcomplexmarketdynamics. Itbridgestheoreticalandpractical
finance, offeringafreshoutlookonfinancialmodelvalidation. Highlightingthesignificanceofad-
vancedcross-validationtechniqueslike CPCV, ourresearchenhancesthereliabilityandapplicability
offinancialmodelsindecision-making.
advanced, data-driven approaches signifies a new epoch in
financial analysis. This new era is marked by the capabil-
ity to process and analyze extensive datasets, revealing in-
1. Introduction
tricate market patterns previously obscured by the limita-
1.1. Background tionsofconventionalmethods. Akeycatalystforthistrans-
The financial sector has witnessed a paradigmatic shift formationhasbeenthesignificantadvancementsincompu-
by integrating sophisticated statistical models and machine tational technology and the rise of high-frequency trading
learningtechniquesintoitsanalyticalframework. Thispiv- practices. These developments have led to a fundamental
otaltransitionfromtraditionalquantitativemethodstomore changeinmarketdynamics. Asaresult, theneedforrobust
andreliablemodelevaluationmethodologies, particularlyin
⋆Theauthorsarelistedinalphabeticalorder.
∗Correspondingauthor:Hamid Arian, Assistant Professorof Finance, cross-validation techniques, has gained unprecedented im-
York University, Toronto, Ontario, Canada, email:harian@yorku.ca portance. Suchmethodsareintegraltomaintainingthein-
ORCID:0000-0002-4624-9421 tegrityandeffectivenessoffinancialmodels, whicharees-
1 Assistant Professor of Finance, York University, Toronto, Ontario,
sentialinguidingdecision-makingprocessesacrossaspec-
Canada
trumoffinancialactivities, fromassetallocationtoriskman-
email:harian@yorku.ca
2 BScstudentof Applied Mathematics&Economics, Sharif University agement, inbothbuy-sideandsell-sideinstitutions.
of Technology, Tehran, Iran
email:norouzi@risklab.ai 1.2. Motivation
3 Professor, Universityof Toronto, Ontario, Canada
The impetus for our research stems from a pivotal ob-
email:luis.seco@utoronto.ca
Thepresentationslidesandacommentaryonthisarticleareavailable servation: despitesubstantialprogressinfinancialmodeling
on Risk Lab’swebsiteatthe Universityof Toronto:risklab.ca/backtesting. andanescalatingrelianceonmachinelearningalgorithms,
The architecture of the codes of this article is explained on risklab.ai/ thereisaglaringshortfallineffectivelyvalidatingthesemod-
backtesting, inboth Pythonand Juliaprogramminglanguages. Therepro-
elswithintheambitoffinancialmarkets. Thisresearchgap
ducibleresultsofthispaperarebasedonauthors’Pythonimplementation
on Risk Lab’s Git Hubpage:github.com/Risk Lab AI. becomesmorepronouncedwhenconsideringtheextensive
Arian, Norouzi, Seco Page1 of26
Electronic copy available at: https://ssrn.com/abstract=4686376


<!-- Page 2 -->

Backtest Overfitting in the Machine Learning Era
literature on predicting market factors. Yet, there is a con- 1.3.2. Rising Concernsover Backtest Overfittingand
spicuouslackofdiscussionontailoringcross-validational- False Discoveries
gorithmstoaccuratelyassessthesemodels (Lopezde Prado Theevolutionoffinancialmodelinghasnecessitatedad-
[2018, 2020]). Further complicating this landscape is the vancedmethodologiestoeffectivelyaddressthechallenges
paucityofresearchdedicatedtocriticallyevaluatingtheback- ofoverfittingandfalsediscoveriesinstrategyevaluation. Pi-
testingandcross-validationalgorithms. Wehypothesizethat oneeringcontributionsby Baileyetal.[2016]and Baileyand
the limited exploration in this domain is attributable to the López de Prado [2014 b], brought to the fore the need for
inherent complexities of financial datasets, which are typ- rigorous evaluation of trading strategies. They introduced
ically noisy, non-stationary, and characterized by intricate quantifiable metrics like the Probability of Backtest Over-
patternsshapedbyvariousvariables, frommacroeconomic fitting (PBO) and the Deflated Sharpe Ratio (DSR), which
shiftstomarketsentiments. Theseuniquedatasetattributes provided a statistical basis to assess the reliability of back-
oftenrendertraditionalcross-validationmethodsinsufficient testedstrategies. Despitetheseadvancements, asignificant
or misleading (Lopez de Prado [2018]). The grave conse- gapexistsintheliterature: acomprehensiveframeworklink-
quencesofmodelinaccuraciesinthiscontextcannotbeover- ingbacktestoverfittingassessmentwiththeeffectivenessof
stated, as they can lead to substantial financial losses and out-of-sample testing methodologies. Our study addresses
posesystemicrisks. Thishighlightsthecriticalneedtode- thisgapbyproposinganovelframeworkthatevaluatesout-
velopandrefinecross-validationmethodologiesfornavigat- of-sample testing techniques through the prism of backtest
ing financial data nuances. While hedge funds and invest- overfitting. By integrating key concepts such as PBO and
mentfirmsmighthavepracticalapproachestoaddressthese DSRintoouranalysis, weaimtoprovideaholisticevalua-
challenges, thereisastarksilenceintheacademicliterature tion of CV methods, ranging from traditional data science
onthisimperativeissue. Ourstudyseekstobridgethisgap, approaches to innovative financial models like those pro-
providinginsightsandmethodologiesvitalfortherigorous posedby Lopez De Prado. Thisapproachensuresfinancial
evaluationoffinancialmodels, therebycateringtofinance’s models’ robustness and predictive power, filling a critical
academicandpracticalrealms. voidinquantitativefinance.
1.3. Literature Review 1.3.3. Exploring Market Dynamicswith Synthetic
1.3.1. Evolutionof Backtestingon Out-of-Sample Controlled Environment
Data Advancementsinsyntheticdatagenerationwithinfinan-
Theevolutionofcross-validation (CV) methodologiesin cialanalysishaveseentheintegrationofsophisticatedmod-
quantitativefinancehasbeenmarkedbyasignificanttransi- els that adeptly replicate complex market dynamics. Our
tion from traditional data science approaches to more spe- study’s Synthetic Controlled Environmentembracesthiscom-
cializedtechniquestailoredforfinancialmarketdata. Con- plexitybymergingthe Heston Stochastic Volatility Model,
ventional methods like K-Fold Cross-Validation and Walk- characterized by the stochastic differential equation 𝑑𝑆 =
𝑡
Forward Cross-Validation, while effective in various ana- 𝜇𝑆𝑑𝑡+ √ 𝜈𝑆𝑑𝑊𝑆(Heston[1993]), withthe Merton Jump
𝑡 𝑡 𝑡 𝑡
lyticalcontexts, haveshownlimitationswhenappliedtofi- Diffusion Model (Merton [1976]), which introduces jumps
nancialmarketsduetotheirinabilitytoadequatelyaccount in asset prices through 𝑑𝑆 = 𝜇𝑆𝑑𝑡 + 𝜎𝑆𝑑𝑊 + 𝑆𝑑𝐽.
𝑡 𝑡 𝑡 𝑡 𝑡 𝑡
forthetemporaldependenciesandnon-stationarityinherent Following this, we explore the context of speculative mar-
in financial time series. Recognizing these shortcomings, ket bubbles. Schatz and Sornette [2020] categorizes bub-
Lopezde Pradointroducedadvanced CVtechniquesspecif- bles into Type-I and Type-II, with Type-I characterized by
ically designed for financial applications. Purged K-Fold an efficient full price process 𝑆 but inefficiencies in both
Cross-Validation, asoutlinedby Lopezde Prado[2018], en- pre-drawdown 𝑆̃ and drawdown 𝑋 processes, and Type-II
hancesthestandard K-Foldmethodbyincorporatinga’purg- byanefficientdrawdownprocess𝑋butanoverallinefficient
ing’mechanism, eliminatingdatafromthetrainingsetthat 𝑆. Thiscategorizationprovidesanuancedunderstandingof
couldinadvertentlyleakinformationaboutthetestset. This bubbledynamicsinfinancialmarkets. Buildingonthisfoun-
approachisparticularlycriticalinfinancialmodelingtopre- dation, ourenvironmentfurtherincorporatesthe Drift Burst
vent lookahead biases. Further advancing the field, Lopez Hypothesis (Christensen et al. [2022]), articulating short-
de Prado’s Combinatorial Purged Cross-Validation (CPCV) livedmarketanomaliesthroughequations𝜇 𝑡 db =𝑎 | 𝜏 db −𝑡 | −𝛼
methodoffersarobustsolutionforbacktestingtradingstrate- and𝜎
𝑡
vb =𝑏
|
𝜏
db
−𝑡
|
−𝛽, emphasizingthecriticalinterplaybe-
gies. Unliketraditional CVmethods, CPCVcreatesmultiple tween drift and volatility during such events. The addition
training and testing combinations, ensuring that each data of the Markov Chain model for regime transitions (Hamil-
segmentisusedfortrainingandvalidation, thusprovidinga ton[1994]), characterizedbyitstransitionmatrix𝑃 =[𝑃 ],
𝑖𝑗
morecomprehensiveassessmentofastrategy’sperformance enablesthesimulationtoadeptlymirrorfluidmarketstates
across various market scenarios. This method respects the adeptly, capturing the ephemeral nature of financial mar-
chronologicalorderingofdataandeffectivelyaddressesthe kets. This innovative amalgamation of stochastic volatil-
risk of overfitting, a prevalent issue in the development of ity, jump-diffusion, bubbledynamics, andregime-switching,
financialmodels. cohesively combined in our Synthetic Controlled Environ-
ment, setsagroundbreakingprecedentinthedomainoffi-
Arian, Norouzi, Seco Page 2 of 26
Electronic copy available at: https://ssrn.com/abstract=4686376


<!-- Page 3 -->

Backtest Overfitting in the Machine Learning Era
nancial model testing and validation, presenting a compre- 1.6. Contribution
hensiveframeworkforevaluatingout-of-sampletestingmethod- Thisresearchsignificantlycontributestoquantitativefi-
ologiesinthenuancedandintricateworldofquantitativefi- nancebypioneeringacomprehensiveframeworkforevalu-
nance. atingout-of-sampletestingmethodologies, particularlyinfi-
nancialmodeling. Webridgeanotablegapintheexistinglit-
1.4. Problem Statement eraturebylinkingtheconceptofbacktestoverfitting, asen-
Centraltoourstudyisaproblemofincreasingconcern capsulatedbymetricslikethe Probabilityof Backtest Over-
infinancialanalytics: developingarobustandreliableout- fitting (PBO) and the Deflated Sharpe Ratio (DSR), with
of-sample testing methodology congruent with the unique theefficacyofout-of-sampletestingmethods. Ourinnova-
attributes of financial time series data. This issue is mul- tiveapproachenhancestheaccuracyandreliabilityofcross-
tifaceted. Firstly, financial time series are characterized by validation techniques, addressing the challenges posed by
non-stationarity, autocorrelation, heteroskedasticity, andregime the temporal complexities and non-stationarity of financial
shifts, challengingtheapplicabilityofconventionalout-of- time series. We leverage the advanced statistical models
sample testing methods. Secondly, the temporal dynamics ofthe Heston Stochastic Volatility, Merton Jump Diffusion,
of financial data, with intricate lead-lag relationships and and Markov Chainforregimetransitions, combinedwithex-
evolutionarypatterns, demandanout-of-sampletestingap- ploring market dynamics through speculative bubbles and
proachthatpreservesthechronologicalsequenceofdatato the Drift Burst Hypothesis. Thissynthesisprovidesamore
avoidlook-aheadbiasandoverfitting, issuesfrequentlyen- nuancedsimulationofmarketconditions, offeringfreshin-
countered in applying machine learning models in finance. sightsandmethodologiesthatcansignificantlyimprovedecision-
Despitetheremarkableadvancementsinintegratingstatisti- makingprocessesinvariousfinancialapplications, fromrisk
cal models and machine learning techniques into financial managementtoalgorithmictrading. Ourworkadvancesthe
analysis, a significant gap persists in accurately assessing fieldbypresentinganovelandholisticperspectiveonmodel
these models, particularly under the challenges of backtest validationintheever-evolvingquantitativefinancedomain,
overfittingandthedynamicnatureoffinancialmarkets. Our thusenhancingbothfinancialpracticesandacademicresearch.
study specifically targets the inadequacy of existing cross-
validationtechniques, which, whilerobustintraditionaldata 1.7. Scopeand Limitations
science contexts, fall short of fully capturing the temporal Thisstudyevaluatescross-validationmethodswithinsyn-
dependencies and non-stationarity of financial data. This thetic market environments meticulously engineered to en-
gapisfurtherwidenedbythelackofacomprehensiveframe- compass diverse market conditions. Our research uses so-
work that integrates the assessment of backtest overfitting phisticatedstatisticalmodelstodissecttheintricaciesofback-
with the effectiveness of out-of-sample testing methodolo- test overfitting in these rigorously constructed settings. A
gies. Thesignificanceofthisproblemisnotlimitedtothe- notable limitation of our approach is the reliance on syn-
oreticalmodelingbuthasfar-reachingimplicationsinprac- theticdata, which, whileprovidingcontrolledexperimental
ticalaspectslikeriskmanagement, algorithmictrading, and conditions, mightnotfullycapturethecomplex, oftenunpre-
portfoliooptimization. Inaccuraciesinmodelvalidationcan dictable dynamics of real-world financial markets. Conse-
lead to substantial financial risks and losses, accentuating quently, extrapolatingourfindingstoactualmarketscenarios
theneedforrigorous, tailoredvalidationmethods, especially shouldbecautiouslyapproached, especiallywhenconsider-
underincreasingregulatoryscrutiny. ingapplicationsinlivetradingenvironmentsorriskmanage-
ment strategies. Moreover, while comprehensive, the spe-
1.5. Objectivesofthe Study cific choice of models and simulation parameters implies
Thecentralobjectiveofourstudyistodevelopacompre- certainconstraints. Thisnecessitatesfurtherempiricalval-
hensiveevaluationframeworkforcross-validationmethods idationindiverse, real-marketcontextstoenhancethegen-
infinancialmodeling, particularlyinthecontextofevolving eralizability of our results. Our study’s primary aim is to
market complexities and the challenges of backtest overfit- enrich the domain of financial model validation, striking a
ting. Byincorporatingkeyconceptslikethe Probabilityof crucialbalancebetweentheoreticaldepthandpracticalrel-
Backtest Overfitting (PBO) and the Deflated Sharpe Ratio evanceandpavingthewayforsubsequentresearchtobuild
(DSR), our framework holistically assesses various cross- uponthesefoundationalinsights.
validationapproaches, rangingfromtraditionaldatascience
1.8. Organizationofthe Paper
methods to more sophisticated financial models. The em-
phasisofthisstudyisnotinherentlyontheincorporationof Thispaperissystematicallystructuredtoexplorecross-
sophisticated methodologies; rather, it centers on a critical validationtechniquesinsyntheticmarketenvironmentscom-
assessment of the efficacy of these approaches when con- prehensively. Thepaperopenswith Introduction1, setting
sideringthedistinctiveattributesinherentinfinancialdata. thestagebydelineatingtheresearchbackground, objectives,
Weaimtobridgethegapbetweentheoreticalrobustnessand andthescopeofthestudy. Followingthis, the Methodology
practicalreliabilityinfinancialmodels, enhancingtheirap- section 2 explores the details of the statistical models and
plicabilityinhigh-stakesfinancialdecision-making, fromas- algorithmsemployed, outliningtheframeworkforsynthetic
setallocationtoriskmanagement. data generation and analysis. The Empirical Results sec-
tion3 thoroughlyexaminesourrigoroustestingandanalysis
Arian, Norouzi, Seco Page 3 of 26
Electronic copy available at: https://ssrn.com/abstract=4686376


<!-- Page 4 -->

Backtest Overfitting in the Machine Learning Era
findings, providinginsightsintotheperformanceandrobust- where the instantaneous variance, 𝜈, adheres to the Feller
𝑡
nessofvariouscross-validationmethods. Inthe Discussion square-rootor Cox-Ingersoll-Ross (CIR) process:
section4, weinterpretthesefindings, contextualizingthem
𝑑𝜈 =𝜅(𝜃−𝜈) d𝑡+𝜉 √ 𝜈d𝑊𝜈, (2.2)
withinthebroaderlandscapeofquantitativefinanceanddis- 𝑡 𝑡 𝑡 𝑡
cussing their implications. The paper culminates with the
with𝑊 and𝑊𝜈representing Wienerprocesses, exhibiting
Conclusion section 5, where we summarize the key take- 𝑡 𝑡
acorrelationof𝜌.
aways, acknowledgethelimitationsofourstudy, andsuggest
The model described in Eqn. (2.1) and Eqn. (2.2) uses
directionsforfutureresearch.
fourmainparameters. 𝜃 isthelong-termaveragevariance,
showingtheexpectedvariancethat𝜈 willapproachas𝑡in-
𝑡
2. Methodology creases. 𝜌describesthecorrelationbetweenthetwo Wiener
Themethodologysectionformsthebackboneofourre- processes in the model. 𝜅 shows how quickly 𝜈 𝑡 returns to
search, presentingacomprehensiveandsystematicapproach itslong-termaverage,𝜃. And𝜉isknownasthe’volatilityof
toexploringandanalyzingfinancialmarketdynamicsthrough volatility’, indicatinghowmuch𝜈 𝑡 canvary.
machinelearningandstatisticalmethods. Thissectionout- Asalientfeatureofthismodelisthe Fellercondition, ex-
lines the construction and utilization of a Synthetic Con- pressedas2𝜅𝜃 >𝜉2. Ensuringthisinequalityguaranteesthe
trolled Environment, whichintegratescomplexmarketmod- strictpositivityoftheprocess, ensuringnonegativevalues
elssuchasthe Heston Stochastic Volatilityand Merton Jump forvariance.
Diffusionmodelsandincorporatesregime-switchingdynam-
2.1.2. Jumps: The Merton Jump Diffusion Model
ics through Markov chains. Additionally, it addresses the
The Merton Jump Diffusion model by Merton [1976]
driftbursthypothesistomodelmarketanomalieslikespec-
enhances the geometric Brownian motion proposed by the
ulativebubblesandflashcrashes. Themethodologyelabo-
Black-Scholesmodelbyintegratingadiscretejumpcompo-
rates on developing and evaluating a prototypical financial
nent to capture abrupt stock price movements. The stock
machine-learningstrategy, encompassingevent-basedsam-
pricedynamicsaregivenby:
pling, trade directionality, bet sizing, and feature selection.
Crucially, themethodologyalsodelvesintoassessingback-
𝑑𝑆 =𝜇𝑆𝑑𝑡+𝜎𝑆𝑑𝑊 +𝑆𝑑𝐽, (2.3)
𝑡 𝑡 𝑡 𝑡 𝑡 𝑡
test overfitting through advanced statistical techniques, en-
suring the validity and robustness of the proposed trading In Eqn. (2.3), 𝜇𝑆𝑑𝑡 is the drift term that captures the ex-
𝑡
strategies. Themethodologiesaremeticulouslydesignedto pectedreturn,𝜎𝑆𝑑𝑊 embodiesthecontinuousrandomfluc-
𝑡 𝑡
capture the intricate nuances of financial markets, thereby tuationswith𝜎beingthestock’svolatility, and𝑑𝑊 thestan-
𝑡
enablingathoroughandaccurateanalysisoftradingstrate- dard Brownianmotionincrement, and𝑆𝑑𝐽 accountsforin-
𝑡 𝑡
gieswithinacontrolledyetrealisticmarketsimulation. stantaneousjumpsinthestockprice.
Thejumpprocess𝐽 in Eqn.(2.3) isdefinedas:
𝑡
2.1. Synthetic Controlled Environment
Infinancialanalysis, constructinga Synthetic Controlled 𝑁 ∑ (𝑡)
𝐽 = 𝑌, (2.4)
Environment is essential for thoroughly examining market 𝑡 𝑖
dynamics and validating theoretical models. This segment 𝑖=1
delineates an integrated simulation architecture synthesiz- where 𝑁(𝑡) is a Poisson process with intensity 𝜆, and 𝑌
𝑖
ingthe Hestonmodel’sstochasticvolatility, Merton’sjump- representslogarithmicjumpsizes, normallydistributedwith
diffusionframework, andthe Markovchains’regime-switching mean𝑚andstandarddeviation𝑠.
nuance. It also contemplates the drift burst hypothesis to Tosimulatepathsof𝑆, oneevolvesthestockpriceus-
𝑡
capturetransientmarketanomalies. Thesecomponentscon- ing the drift and diffusion terms, determines jumps based
structanuancedandcomprehensiveemulationofthefinan- on𝑁(𝑡), andadjuststhestockpriceaccordingtothemagni-
cial market’s complexity, serving as a critical substrate for tudefromthe𝑌 distribution. Bymergingcontinuousprice
𝑖
theexplorationandscrutinyofeconometrictheories. movementswithjumps, thismodelpotentiallyoffersamore
accurate representation of real-world stock price behaviors
2.1.1. Random Walk: The Heston Stochastic Volatility
thanmeregeometric Brownianmotion.
Model
Inmodelingthestochasticbehaviorofthemarketprice, 2.1.3. Speculative Bubbles&Flash Crashes: The
weemploythefoundational Hestonmodel, asarticulatedby Drift Burst Hypothesis
Heston[1993]. Thismodelprovidesaframeworkthatcap- Inthestudy Christensenetal.[2022], theauthorsintro-
turestheintrinsicvolatilitydynamicsofafinancialasset. duce the drift burst hypothesis to elucidate the short-lived
Attheheartofthe Hestonmodelliesthepremisethatthe flashcrashesevidentinhigh-frequencytickdata. Thismethod-
assetprice,𝑆 𝑡 , evolvesaccordingtothefollowingstochastic ologyzeroesinonthecomplexdancebetweendriftandvolatil-
differentialequation: ity. Theytheorizethatasuddenuptickindriftisonlyviable
if there’s a simultaneous surge in volatility. To articulate
𝑑𝑆 =𝜇𝑆d𝑡+ √ 𝜈𝑆d𝑊𝑆, (2.1)
𝑡 𝑡 𝑡 𝑡 𝑡 this, theyintroducethe"volatilityburst"concept, denotinga
rapidescalationinmarketvolatility.
Arian, Norouzi, Seco Page 4 of 26
Electronic copy available at: https://ssrn.com/abstract=4686376


<!-- Page 5 -->

Backtest Overfitting in the Machine Learning Era
Thedrift’ssuddenincreaseisconciselyencapsulatedin 2.1.5. Market Synthesis: Discrete Simulation
theequation: Inourstudy, weemployadiscretesimulationapproach
tomodelmarketdynamics, whichcanbeeffectivelyrepre-
𝜇 𝑡 db =𝑎 | 𝜏 db −𝑡 | −𝛼. (2.5) sented by the Euler-Maruyama method for stochastic dif-
ferential equations. This method provides a numerical ap-
In Eqn. (2.5), 𝜇 𝑡 db describes the drift at a given time 𝑡 ac- proximationofthecontinuousmarketprocessesinadiscrete
cordingtoitsdistancerelativetotheburstingtime𝜏 db . The framework. Byapplying Ito’s Lemma, theapproximationis
factor𝑎setsthescaleofthedrift, while 1 <𝛼 <1 measures givenby:
2
howintensethisdriftspikeis.
( ( ))
Similarly, the abrupt rise in volatility, or the "volatility 1 𝑣2
Δ𝑆 ≈ 𝜇− 𝜈 −𝜆 𝑚+ 𝑆Δ𝑡
burst", isrepresentedas: 𝑡 2 𝑡 2 𝑡
√ √
𝜎 𝑡 vb =𝑏 | 𝜏 db −𝑡 | −𝛽. (2.6) + 𝜈 𝑡 𝑆 𝑡 𝑍 Δ𝑡+𝑌Δ𝑁(𝑡). (2.9)
In Eqn.(2.9),Δ𝑆 isthechangeinassetprice,𝜇represents
𝑡
In Eqn.(2.6),𝜎 𝑡 vb indicatesthevolatilityattime𝑡. Thepa- the drift rate, and √ 𝜈 𝑡 is the volatility factor scaled by the
rameter𝑏quantifiesthesizeofthisvolatilitysurge, and0< standard normal random variable 𝑍. 𝑌 is a normally dis-
𝛽 < 1, gaugesitssharpness. tributedjumpsizewithmean𝑚andvariance𝑣2, andΔ𝑁(𝑡)
2
denotesthejumpprocessincrementscharacterizedbya Pois-
2.1.4. Regime Transitions: Markov Chain sondistributionwithintensity𝜆Δ𝑡.
Aregime-switchingtimeseriesmodelisappliedtosim- Thevariationininstantaneousvariance𝜈 iscapturedby
𝑡
ulatemarketdynamics, following Hamilton[1994]asmen- Eqn.(2.10):
tionedby Lopezde Prado[2020]. Themarketissegmented
intodiscreteregimes, eachwithuniquecharacteristics. The Δ𝜈 =𝜅(𝜃−𝜈)Δ𝑡+𝜉 √ 𝜈(𝜌 𝜖𝑃+ √ 1−𝜌2𝜖𝜈) √ Δ𝑡, (2.10)
𝑡 𝑡 𝑡 𝜖 𝑡 𝜖 𝑡
market’stransitionbetweentheseregimesatanygiventime𝑡
where𝜅 istherateatwhich𝜈 revertstoitslong-termmean
isdeterminedbya Markovchain, wherethetransitionprob- 𝑡
ability 𝑝 depends solely on the state immediately prior. 𝜃, and𝜉 measuresthevolatilityofthevariance. Thecorre-
𝑡,𝑛 latedstandardnormalwhitenoises𝜖𝜈 and𝜖𝑃 introduceran-
Thisapproachcapturesthefluidnatureoffinancialmarkets, 𝑡 𝑡
√
whichfluctuatebetweendifferentstates, reflectingshiftsin domnesswithacorrelationcoefficient𝜌 . Thefactor Δ𝑡is
𝜖
volatility and trends. By employing a Markov chain, these introducedtoscalethemodelappropriatelyinthediscrete-
transitions are modeled with mathematical precision while time setting, reflecting the properties of Brownian motion
maintaining economic plausibility, recognizing that finan- increments.
cialmarketstendtoexhibitamemoryofonlythemostrecent Incorporatingthe Markovchainregimetransitionmodel
events. into our discrete simulation, the constants 𝜇, 𝜃, 𝜉, 𝜌 , 𝜆,
𝜖
AMarkovchainisamathematicalsystemthattransitions 𝑚, and 𝑣2 are adjusted for each regime. The adjustment is
fromonestatetoanotherinastatespace. Itisdefinedbyits dictated by the state transitions determined by the Markov
set of states and the transition probabilities between these chain, whereeachstateencapsulatesadistinctmarketregime
states. Thefundamentalpropertyofa Markovchainisthat with its own parameter set. As the market transitions be-
theprobabilityofmovingtothenextstatedependsonlyon tweenregimes, theseparameterschangeaccordingly, align-
thepresentstateandnotonthesequenceofeventsthatpre- ingthesimulationwiththeunderlyingstochasticprocessthat
cededit. reflectsthedynamicfinancialmarketenvironment.
Givenafinitenumberofstates𝑆 = {𝑠 ,𝑠 ,…,𝑠 }, the
1 2 𝑛
2.2. Prototypical Financial Machine Learning
probability of transitioning from state 𝑠 to state 𝑠 in one
𝑖 𝑗
stepisdenotedby𝑃 : Strategy
𝑖𝑗
Developingacoherentmachine-learningstrategyinquan-
𝑃 𝑖𝑗 =𝑃(𝑋 𝑛+1 =𝑠 𝑗| 𝑋 𝑛 =𝑠 𝑖 ), (2.7) titativefinancenecessitatesameticulousfusionofstatistical
techniquesandmarketknowledge. Ourproposedmethodol-
where𝑋 𝑛 representsthestateattime𝑛, and𝑃 𝑖𝑗 istheentry ogyrigorouslycombinesevent-basedtriggers, trend-following
in the 𝑖-th row and 𝑗-th column of the transition matrix 𝑃. mechanisms, andriskassessmenttoolstoformulateaproto-
The matrix 𝑃 = [𝑃 𝑖𝑗 ] is called the transition matrix of the typical financial machine-learning strategy. It commences
Markovchain. Eachentry𝑃 𝑖𝑗 representstheone-steptran- with precisely identifying market events through CUSUM
sition probability from state 𝑠 𝑖 to state 𝑠 𝑗 as in Eqn. (2.7): filteringandprogressestoascertaintradedirectionalityvia
momentumanalysis. Thecoreofthestrategyharnessesmeta-
labeling to assess trade viability and employs an averaging
𝑃 𝑃 ⋯ 𝑃
⎡ 11 12 1𝑛⎤ approach to bet sizing sensitive to market conditions and
𝑃 = ⎢𝑃 21 𝑃 22 ⋯ 𝑃 2𝑛⎥ . (2.8) positionoverlap. Integratingfractionallydifferentiatedfea-
⎢ ⋮ ⋮ ⋱ ⋮ ⎥
⎢ ⎥ tures alongside traditional technical indicators forms a ro-
⎣ 𝑃 𝑛1 𝑃 𝑛2 ⋯ 𝑃 𝑛𝑛⎦
bust feature set, ensuring the preservation of temporal de-
pendenciesandadherencetostationarity—aprerequisitefor
Arian, Norouzi, Seco Page 5 of 26
Electronic copy available at: https://ssrn.com/abstract=4686376


<!-- Page 6 -->

Backtest Overfitting in the Machine Learning Era
thesuccessfulapplicationofpredictivemodelinginfinancial movingaveragesareformulatedasfollows:
contexts.
𝑁 𝑓𝑎𝑠𝑡−1
1 ∑
2.2.1. Sampling: CUSUMFiltering MA short (𝑦 𝑡 )= 𝑁 𝑦 𝑡−𝑖 ,
Portfolio management often relies on event-based trig- 𝑓𝑎𝑠𝑡 𝑖=0 (2.13)
gers for investment decisions. These events may include
1
𝑁 𝑠∑𝑙𝑜𝑤−1
structuralbreaks, signals, ormicrostructuralchanges, often MA long (𝑦 𝑡 )= 𝑁 𝑦 𝑡−𝑖 ,
promptedbymacroeconomicnews, volatilityshifts, orsig- 𝑠𝑙𝑜𝑤 𝑖=0
nificantpricedeviations. Inthiscontext, itiscrucialtoiden- where𝑁 and𝑁 representthenumberofperiodsfor
𝑓𝑎𝑠𝑡 𝑠𝑙𝑜𝑤
tifysucheventsaccurately, leveragingmachinelearning (ML) thefast (short-term) andslow (long-term) movingaverages,
toascertainthepotentialforreliablepredictivemodels. The respectively.
redefinitionofsignificanteventsortheenhancementoffea- A position is taken based on the relative positioning of
turesetsisacontinualprocessrefinedupondiscoveringnon- thesemovingaveragesposta CUSUMevent. Atradeisini-
predictivebehaviors. tiatedbasedontheseconditions:
We employ the Cumulative Sum (CUSUM) filter as an
1. Long Position:Triggeredwhen MA (𝑦) surpasses
event-basedsamplingtechniqueformethodologicalrigor, as short 𝑡
MA (𝑦), signalingupwardmarketmomentum.
mentionedby Lopezde Prado[2018]. Thismethoddetects long 𝑡
2. Short Position: Initiated when MA (𝑦) falls be-
deviationsinthemeanofaquantity, denotinganeventwhen short 𝑡
low MA (𝑦), indicatingdownwardmarketmomen-
a threshold is crossed. Given independent and identically long 𝑡
tum.
distributed (IID) observationsfromalocallystationarypro-
{ }
cess 𝑦 𝑡 𝑡=1,…,𝑇 , wedefinethe CUSUMas: Thestrategy, thus, alignsthepositionwiththecurrentmar-
kettrend, asindicatedbythemomentuminprices.
{ ( )}
𝑆 =max 0,𝑆 +𝑦 −𝔼 𝑦 , (2.11)
𝑡 𝑡−1 𝑡 𝑡−1 𝑡
2.2.3. Size Determination: Meta-Labelingvia
withtheinitialcondition𝑆 =0. Asignalforactionissug- Triple-Barrier Method
0
gestedatthesmallesttime𝑡where𝑆 ≥ℎ, withℎbeingthe Inourtradingframework, oncethesideofapositionis
𝑡
predefinedthresholdorfiltersize. It’snotablethat𝑆 isreset determinedthroughthemomentumstrategy, itundergoesa
tozeroif𝑦 ≤𝔼 ( 𝑦 ) −𝑆 , whichintentionally 𝑡 ignores rigorousevaluationviathetriple-barriermethodtoascertain
𝑡 𝑡−1 𝑡 𝑡−1
negativeshifts. itspotentialprofitability. Thisevaluationformsthebasisfor
Toencompassbothpositiveandnegativeshifts, weex- position sizing, leveraging a meta-labeling approach intro-
tendthistoasymmetric CUSUMfilter: ducedby Lopezde Prado[2018].
Uponidentificationofatrade’sdirection, thetriple-barrier
𝑆 𝑡 + =max { 0,𝑆 𝑡 + −1 +𝑦 𝑡 −𝔼 𝑡−1 ( 𝑦 𝑡 )} , 𝑆 0 + =0, methodappliesthreedistinctbarrierstodeterminetheout-
𝑆− =min { 0,𝑆− +𝑦 −𝔼 ( 𝑦 )} , 𝑆− =0, (2.12) come of the position. The horizontal barriers are set ac-
𝑡 𝑡−1 𝑡 𝑡−1 𝑡 0
𝑆 =max
{ 𝑆+,−𝑆−}
.
cordingtoadynamicvolatility-adjustedthresholdforprofit-
𝑡 𝑡 𝑡 takingandstop-loss, whiletheverticalbarrierisdefinedby
apredeterminedexpirationtime, denotedasℎ. Thelabelas-
Adopting Lam and Yam [1997]’s strategy, we generate al-
signment isas follows: hittingthe upperbarrier signifies a
ternatingbuy-sellsignalsuponobservingareturnℎrelative
successful trade, hence labeled 1; conversely, touching the
to a prior peak or trough, akin to the filter trading strategy
lowerbarrierfirstindicatesaloss, labeled−1. Iftheverti-
by Famaand Blume[1966]. Ourapplicationofthe CUSUM
caltimebarrierexpiresfirst, thelabelisdeterminedbythe
filter using Eqn. (2.12), however, is distinct; we only sam-
ple at bar 𝑡 if 𝑆 ≥ ℎ, subsequently resetting 𝑆 assuming signofthereturn, reflectingtheresultofthetradewithinthe
( ) 𝑡 𝑡 period[𝑡 ,𝑡 +ℎ].
𝔼 𝑦 = 𝑦 . We define 𝑦 as the natural logarithm of 𝑖,0 𝑖,0
𝑡−1 𝑡 𝑡−1 𝑡
Theroleofmeta-labelinginthiscontextistoscrutinize
the asset’s price to capture proportional price movements.
furtherthetradesindicatedbytheprimarymomentummodel.
Thethresholdℎisnotstatic;instead, itdynamicallyadjusts
Itconfirmsorrefutesthesuggestedpositions, effectivelyfil-
directlytothedailyvolatility, ensuringsensitivitytomarket
teringoutfalsepositivesandallowingforacalculateddeci-
conditions.
sionontheactualsizeoftheinvestment. Themeta-labeling
2.2.2. Side Determination: Momentum Strategy process directly informs the appropriate risk allocation for
Weemployamomentumstrategybasedonmovingav- eachpositionbyassigningaconfidenceleveltoeachpoten-
erages to determine the direction of trades signaled by the tialtrade. Thismethodologicalstepenhancestheprecision
event-based CUSUMfiltersampling. Specifically, wecalcu- of our strategy and ensures that position sizing is aligned
latetwomovingaveragesoftheprices, ashort-termmoving withtheevaluatedprofitabilityofthetrade, asindicatedby
average MA (𝑦) andalong-termmovingaverage MA (𝑦), theoutcomeofthetriple-barrierassessment.
short 𝑡 long 𝑡
toidentifytheprevailingtrend. Theshort-termmovingav-
2.2.4. Sample Weights: Label Uniqueness
erageisresponsivetorecentpricechanges, whilethelong-
Thevalidityofthe Independentand Identically Distributed
term moving average captures the underlying trend. These
(IID) assumptionisacommonshortfallinfinancialmachine
Arian, Norouzi, Seco Page 6 of 26
Electronic copy available at: https://ssrn.com/abstract=4686376


<!-- Page 7 -->

Backtest Overfitting in the Machine Learning Era
learning, astheoverlappingintervalsinthedataoftenvio- 2.2.5. Financial Features: Fractional Differentiation
lateit. Specifically, labels𝑦 and𝑦 maynotbe IIDifthere &Technical Analysis
𝑖 𝑗
isasharedinfluencefromacommonreturn𝑟 { }, In pursuing a robust financial machine-learning model,
𝑡
𝑗,0
, min 𝑡
𝑖,1
,𝑡
𝑗,1
where𝑡 >𝑡 forconsecutivelabels𝑖<𝑗. Toaddressthe ourmethodologyencompassesdiversefeaturesthatbalance
𝑖,1 𝑗,0
non-IID nature of financial datasets without compromising memorypreservationwiththenecessityforstationarity. Frac-
the model granularity, we utilize sample weights as intro- tional differentiation of log prices is employed to maintain
duced by Lopez de Prado [2018]. This method recognizes as much informative historical price behavior as possible
theinterconnectednessofdatapointsandadjuststheirinflu- whileensuringthedataadherestothestationaryrequirement
enceonthemodelaccordingly. Byweighingsamplesbased of predictive models (Lopez de Prado [2018]). Addition-
ontheiruniqueinformationandreturnimpact, weenhance ally, we incorporate exponentially weighted moving aver-
modelrobustness, enablingmoreaccurateanalysisoffinan- ages (EWMA) ofvolatility, capturingrecentmarketvolatil-
cialtimeseries. itytrendsandasuiteoftechnicalanalysisindicatorsthatpro-
We define concurrent labels at time 𝑡 as those that are videinsightsintomarketsentimentanddynamics. Techni-
bothinfluencedbyatleastonesharedreturn calanalysisfeaturesareextractedfromhistoricalpriceand
volume data and are widely used to capture market senti-
𝑝
𝑟 = 𝑡 −1. (2.14) mentandtrends, whichareindicativeoffuturepricemove-
𝑡−1,𝑡 𝑝 𝑡−1 ments and provide structured information from the other-
wisenoisymarketdata, aidingthemachinelearningmodel
The concurrency of labels 𝑦 and 𝑦 does not necessitate a
𝑖 𝑗 todiscernpatternsassociatedwithprofitabletradingoppor-
completeoverlapinperiod; rather, itissufficientthatthere
tunities. The features used for this problem are as follows:
isapartialtemporalintersectioninvolvingthereturnattime
𝑡.
Toquantifytheextentofoverlap, weconstructabinary 1. Frac Diff:Thefractionallydifferentiatedlogprice. Fi-
{ }
indicatorarray 1 foreachtime𝑡, where1 isset nancialtimeseriesarecharacterizedbyalowsignal-
𝑡,𝑖 𝑖=1,…,𝐼 𝑡,𝑖
to 1 if the interval [ 𝑡 ,𝑡 ] overlaps with [𝑡−1,𝑡], and 0 to-noiseratioandmemory, challengingtraditionalsta-
𝑖,0 𝑖,1
otherwise. Wethencalculatetheconcurrencycountattime tionarity transformations like integer differentiation,
𝑡, givenby which remove this memory and potentially valuable
predictive signals (Lopez De Prado [2015]). To ad-
𝐼 dressthis, fractionaldifferentiationisemployedtopre-
∑
𝑐 𝑡 = 1 𝑡,𝑖 . (2.15) servememorywhileensuringstationarity.
{ }
𝑖=1 Consider a time series 𝑋 and the backshift oper-
𝑡
ator 𝐵 such that 𝐵𝑘𝑋 = 𝑋 for any non-negative
Theuniquenessofalabelisinverselyproportionaltothe 𝑡 𝑡−𝑘
integer𝑘. Thebinomialtheoremappliedtoaninteger
number of labels concurrent with it (Eqn. (2.15)). Conse-
powercanbeextendedtorealpowersusingthebino-
quently, weassignsampleweightsbyinverselyscalingthem
mialseriesandappliedtothebackshiftoperator:
withtheconcurrencycountwhileconsideringthemagnitude
ofreturnsoverthelabel’slifespan. Forlabel𝑖, theprelimi-
naryweight𝑤̃ 𝑖 iscomputedasthenormofthesumofpro- (1−𝐵)𝑑 = ∑
∞ (
𝑑
)
(−𝐵)𝑘 = ∑
∞ ∏𝑘
𝑖=
−
0
1(𝑑−𝑖)
(−𝐵)𝑘.
portionallyattributedreturns: 𝑘 𝑘!
𝑘=0 𝑘=0
‖ ‖∑ 𝑡 𝑖,1 𝑟 𝑡−1,𝑡 ‖ ‖ (2.18)
𝑤̃ =‖ ‖. (2.16)
𝑖 ‖ 𝑐 ‖ Theexpansionin Eqn.(2.18) yieldsweights𝜔 , which
‖
‖
𝑡=𝑡
𝑖,0
𝑡 ‖
‖ areappliedtopastvaluesoftheseriestocom
𝑘
putethe
To facilitate a consistent scale for optimization algorithms fractionallydifferentiatedseries𝑋̃ :
𝑡
thatdefaulttoanassumptionofunitsampleweights, wenor-
malizethesepreliminaryweightscalculatedin Eqn.(2.16) to ∞ 𝑘−1
∑ ∏𝑑−𝑖
sumtothetotalnumberoflabels𝐼: 𝑋̃ = 𝜔 𝑋 , with𝜔 =(−1)𝑘 .
𝑡 𝑘 𝑡−𝑘 𝑘 𝑘!
𝑘=0 𝑖=0
𝑤̃
𝑤 = 𝑖 . (2.17) (2.19)
𝑖 ∑𝐼
𝑤̃
𝑗=1 𝑗
An approach to fractional differentiation employs a
Eqn.(2.17) ensuresthat
∑𝐼
𝑤 =𝐼. Throughthisweight-
fixed-width window by truncating the infinite series
𝑖=1 𝑖 based on a threshold criterion for the weights. The
ing scheme, we emphasize observations with greater abso-
fixed-width window approach can be formalized as
lutelogreturnsthatarelesscommon, therebyenhancingthe
follows: findthesmallest𝑙∗ suchthatthemodulusof
model’scapacitytolearnfromuniqueandsignificantmarket
theweights‖𝜔 ‖isnotlessthanthethreshold𝜏, and
events. ‖ 𝑙∗‖
‖𝜔 ‖ falls below 𝜏. The adjusted weights 𝜔̃ are
‖ 𝑙∗+1‖ 𝑘
Arian, Norouzi, Seco Page 7 of 26
Electronic copy available at: https://ssrn.com/abstract=4686376


<!-- Page 8 -->

Backtest Overfitting in the Machine Learning Era
thendefinedby: 10. ATR:The Average True Rangequantifiesmarketvolatil-
{ ity by averaging true ranges over a period, reflecting
𝜔̃ =
𝜔
𝑘
if𝑘≤𝑙∗,
(2.20) thedegreeofpricevolatility.
𝑘 0 if𝑘>𝑙∗. 11. Log DPO:Thelogarithmofthe Detrended Price Os-
cillatorcomparesrollingmeansatdifferentperiodsto
Applyingthesetruncatedweights, thefractionallydif-
identifycyclicalpatternsinthepricedata.
ferentiatedseries𝑋̃ isobtainedthroughafinitesum:
𝑡 12. MACDPosition:Indicatesthepositionofthe MACD
Histogramrelativetoitssignalline, withvaluesabove
∑
𝑙∗
zerosuggestingabullishcrossoverandbelowzeroa
𝑋̃ 𝑡 = 𝜔̃ 𝑘 𝑋 𝑡−𝑘 , for𝑡=𝑇−𝑙∗+1,…,𝑇. (2.21) bearishcrossover.
𝑘=0 13. ADXStrength: Reflectsthetrend’sstrengthasmea-
The resultant seriesin Eqn. (2.21) is a driftlessmix- sured by the ADX, categorizing trends as strong if
tureoftheoriginallevelandnoisecomponents, pro- aboveathresholdvalueandweakifbelow.
vidingastationaryseriesdespiteitsnon-Gaussiandis- 14. RSI Signal: Categorizes the RSI reading as signal-
tributionthatexhibitsmemory-inducedskewnessand ing overbought conditions above a high threshold or
kurtosis. oversoldconditionsbelowalowthreshold.
For a given time series {𝑋 𝑡 } 𝑡=1,…,𝑇 , the fixed-width 15. CCISignal: Providesasignalbasedonthe CCIread-
window fractional differentiation (FFD) approach is ing, indicatingoverboughtoroversoldconditionswhen
utilized to determine the order of differentiation 𝑑∗ crossingpredefinedthresholdlevels.
thatachievesstationarityintheseries{𝑋̃ 𝑡 } 𝑡=𝑙∗,…,𝑇 us- 16. Stochastic Signal:Generatesasignalfromthe Stochas-
ing ADFtests. Thevalueof𝑑∗indicatesthememory tic Oscillator, identifyingoverboughtoroversoldcon-
thatmustbeeliminatedtoattainstationarity. ditionsbasedonthresholdlevels.
2. Volatility:Volatilityisafundamentalfeaturethatcap-
17. ROCMomentum:Categorizesthemomentumbased
turesthemagnitudeofpricemovementsandiscritical
onthe ROC, withpositivevaluesindicatinganupward
formodelingriskandreturninfinancialmarkets. The
momentumandnegativevaluesadownwardmomen-
exponentially weighted moving average (EWMA) of
tum.
volatility gives more weight to recent observations,
18. Kumo Breakout: Identifiespricebreakoutsfromthe
makingitaresponsivemeasureofcurrentmarketcon-
Ichimoku Cloud, suggestingabullishbreakoutwhen
ditions. The EWMAvolatilityforagivenday𝑡iscal-
thepriceisabovethecloudandbearishwhenbelow.
culatedasfollows:
19. TK Position: Indicates the position of the Tenkan-
√
𝜎𝐸𝑊𝑀𝐴 = 𝜆𝜎2 +(1−𝜆)𝑟2, (2.22) senrelativetothe Kijun-seninthe Ichimoku Indicator,
𝑡 𝑡−1 𝑡 withvaluesaboveonesuggestingabullishcrossover
where𝑟 isthelogreturnattime𝑡, and𝜆isthedecay andbelowoneabearishcrossover.
𝑡
factorthatdeterminestheweightingofpastobserva- 20. Price Kumo Position: Categorizesthepriceposition
tions. relativetothe Ichimoku Cloud, suggestingbullishsen-
3. Z-Score: The Z-Scorestandardizesthelogpricesby timentwhenabovethecloudandbearishwhenbelow.
theirdeviationfromarollingmeanrelativetotherolling 21. Cloud Thickness:Measuresthethicknessofthe Ichimoku
standarddeviation, highlightingpriceanomalies. Cloud by taking the logarithm of the ratio between
4. Log MACDHistogram: Thedifferencebetweenthe thecloudspans, indicatingmarketvolatilityandsup-
logarithmically transformed MACD line and its cor- port/resistancestrength.
respondingsignallineindicatesmomentumshifts. 22. Momentum Confirmation:Confirmsthemomentum
5. ADX: The Average Directional Index measures the indicatedbythe Ichimoku Indicator, withthe Tenkan-
strength of a trend over a given period, with higher senabovethecloudsuggestingbullishmomentumand
valuesindicatingstrongertrends. belowsuggestingbearishmomentum.
6. RSI:The Relative Strength Indexidentifiesconditions
2.2.6. Bet Sizing: Averaging Active Bets
wheretheassetispotentiallyoverboughtoroversold,
Properbetsizingiscrucialinimplementingasuccessful
oftensignalingpossiblereversals.
investment strategy informed by machine learning predic-
7. CCI:The Commodity Channel Indexdetectscyclical
tions. Wedenoteby𝑝[𝑥]theprobabilityofalabel𝑥occur-
trends in asset prices, often used to spot impending
ring, where𝑥 ∈ {−1,1}. Todeterminetheappropriateness
marketreversals.
ofabet, wetestthenullhypothesis:
8. Stochastic: The Stochastic Oscillator compares the
closingpricetoitspricerangeoveraspecifiedperiod, Null Hypothesis1. 𝐻 ∶𝑝[𝑥=1]= 1.
indicatingmomentum. 0 2
9. ROC: The Rate of Change measures the velocity of Calculatingtheteststatistic:
pricechanges, withpositivevaluesindicatingupward
𝑝[𝑥=1]− 1
momentumandnegativevaluesindicatingdownward 𝑧= 2 ∼𝑍, (2.23)
√
momentum. 𝑝[𝑥=1](1−𝑝[𝑥=1])
Arian, Norouzi, Seco Page 8 of 26
Electronic copy available at: https://ssrn.com/abstract=4686376


<!-- Page 9 -->

Backtest Overfitting in the Machine Learning Era
where𝑧∈(−∞,+∞) and𝑍representsthestandardnormal to the number of neighbors chosen. By experiment-
distribution. Thebetsizeisthenderivedas ing with small numbers of neighbors, we expose the
modeltopotentialoverfitting, whereitmightrelytoo
𝑚=2𝑍[𝑧]−1, (2.24)
heavilyonimmediate, possiblynoisydatapoints. The
modelusedinourstudyisacustompipelineintegrat-
with𝑚∈[−1,1]and𝑍[⋅]beingthecumulativedistribution
ingstandardscalingwiththe KNeighbors Classifier.
function (CDF) of𝑍 for Eqn.(2.23). Thisformulationac-
2. Decision Tree: Decision Trees, while interpretable,
counts for predictions originating from both meta-labeling
caneasilyoverfitthetrainingdata, especiallywithout
andstandardlabelingestimators.
constraintsontreedepth. Ourconfigurationteststhe
The process of bet sizing involves determining the size
model in its most unconstrained form, providing in-
ofindividualbetsbasedontheprobabilityofoutcomesand
sights into its behavior without regularizing parame-
managing the aggregation of multiple bets that may be ac-
ters. Ourimplementationusesa Decision Tree Clas-
tiveconcurrently. Tomanagemultiple concurrentbets, we
{ } sifierwithapredefinedrandomstateforreproducibil-
defineabinaryindicator 1 foreachbet𝑖attime𝑡. This
𝑡,𝑖 ity. Theparametersincludethemaximumdepthofthe
indicatortakesthevalueof1 ifbet𝑖isactivewithinthein-
tree, theminimumnumberofsamplesrequiredtosplit
terval (𝑡−1,𝑡], and0 otherwise. Theaggregatebetsizeat
aninternalnode, andtheminimumnumberofsamples
time𝑡isthentheaverageofallactivebetsizesasshownin
requiredtobeataleafnode.
Eqn.(2.25):
3. XGBoost: XGBoost is an advanced implementation
∑𝐼 𝑚1 ofgradientboostingalgorithmsknownforitsefficiency,
𝑚 𝑡 = ∑ 𝑖= 𝐼 1 𝑖 𝑡,𝑖 , (2.25) flexibility, andportability. However, withexcessively
1
𝑖=1 𝑡,𝑖 highvaluesforparameterssuchasthenumberofes-
timators and learning rates, there is a risk of overfit-
where𝑚 istheindividualbetsize.
𝑖
ting, wherethemodelbecomesoverlytailoredtothe
2.3. Strategy Trials training data. It excels in handling sparse data and
Thissectionpresentsourstrategytrials, whichareinte- scaleseffectivelyacrossmultiplecores. Inoursetup,
graltoourfinancialmachine-learningresearch. Weemploy the XGBoost Classifierisemployedwithspecificpa-
acomprehensivemethodology, examiningmachinelearning rameterslikethenumberoftrees, maximumdepthof
modelslikek-Nearest Neighbors, Decision Trees, and XG- trees, learningrate, andsubsamplingratioofthetrain-
Boost, each with unique parameter settings. Our approach inginstances.
deliberately tests these models under conditions conducive Each model is exhaustively assessed across its param-
tooverfittingtoassesstheirrobustnessandadaptability. We eterspacetoevaluateitsefficacyandrobustnessinvarious
also introduce the Momentum Cross-Over Strategy, utiliz- marketscenarios. Thisextensiveparameterizationisadelib-
ingvariousmovingaveragewindowlengthstoaligntrades eratestrategytotestthemodels’susceptibilitytooverfitting,
withmarkettrends. Thiscombinationofdiversemodelsand acriticalconsiderationinfinancialmachine-learningappli-
adaptivestrategies, processedthroughasystematicpipeline cations.
that includes event-based sampling, meta-labeling, and it-
erative optimization, is designed to rigorously evaluate the 2.3.2. Momentum Cross-Over Strategy: An Overview
efficacy of trading strategies in complex market scenarios. The Momentum Cross-Over Strategy is a key element
Thetrialsaimtobalancetheexplorationofmachinelearn- of our strategy trials, aiming to align trade directions with
ing potentials in finance with the pragmatic challenges of markettrendsdetectedthroughmovingaverages. Thisstrat-
real-worldmarketconditions. egy’sadaptabilityliesinitsvariouscombinationsofwindow
lengthsforthemovingaverages, allowingittocapturemar-
2.3.1. Machine Learning Models: An Overview ketmomentumoverdifferenttimeframes. Byexperiment-
In our strategic analysis, we leverage various machine ingwithmultiplewindowlengthpairs, thestrategyadjuststo
learningmodels, eachwithadistinctsetofparameters. This variousmarketconditionsandintroducesflexibilitythatin-
approachisdesignedtorigorouslytestthemodelsundervary- creasesthelikelihoodofoverfitting. Thisapproachensures
ingconditions, potentiallyincreasingtheriskofoverfitting. athoroughexaminationofmarkettrends, aimingtooptimize
This methodological choice serves a dual purpose: firstly, tradepositionsinlinewiththeprevailingmarketdirection.
to rigorously challenge the robustness of the models under
extremeparameterconditions, andsecondly, toexaminethe 2.3.3. Trialson Synthesized Data: The Pipeline
models’performanceinscenariospronetooverfitting. This Ourstrategytrialsemployastreamlinedpipelinetoas-
deliberate stress testing provides valuable insights into the sess the potential for overfitting in various trading strate-
resilienceandadaptabilityofthealgorithmsincomplexfi- gies. Thepipelineintegratesevent-basedsampling, momen-
nancial environments. The following models and their re- tumstrategy, machinelearningmodels, andmeta-labelingto
spectiveparametersetsareintegraltothisanalysis: simulatediversemarketconditionsandteststrategyefficacy.
Thekeystepsofthispipelineare:
1. K-Nearest Neighbors (k-NN): The k-NN model is
predicatedonfeaturesimilarityandishighlysensitive 1. CUSUMSampling:Theprocessbeginswiththe CUSUM
Arian, Norouzi, Seco Page 9 of 26
Electronic copy available at: https://ssrn.com/abstract=4686376


<!-- Page 10 -->

Backtest Overfitting in the Machine Learning Era
filter, identifying significant market shifts based on formingtheanalysisononesubset (thetrainingset), andval-
deviations in log prices. This method generates sig- idatingtheanalysisontheothersubset (thetestset). Tore-
nalsforpotentialtradingopportunities. ducevariability, multipleroundsofcross-validationareper-
2. Momentum Cross-Over Strategy:Following CUSUM formedusingdifferentpartitions, andthevalidationresults
signals, the Momentum Cross-Over Strategyisapplied. areaveragedovertherounds.
Thisstepinvolveschoosingwindowsizesforcalculat- Infinancialmodeling, especiallyforbacktestingtrading
ingmovingaveragesanddeterminingthetradedirec- strategies, applying K-Foldcross-validationpresentsunique
tionbasedontheirrelativepositions. challenges. Financialdataaretypicallytime-seriesdatachar-
3. Machine Learning Model Selection:Amachinelearn- acterizedbytemporaldependenciesandnon-stationarity. These
ingmodel, suchask-NN, Decision Tree, or XGBoost, featuresoffinancialdataviolatethefundamentalassumption
is selected with specific parameters. This stage tests of traditional K-Fold cross-validation, which assumes that
model responses to trading signals, emphasizing the theobservationsareindependentandidenticallydistributed
analysis of overfitting risks under varying parameter (i.i.d).
settings. Theprocessof K-Foldcross-validationinfinancialback-
4. Meta-Labelingand Sample Weights: Tradesignals testinginvolvesthefollowingsteps:
areprocessedthroughmeta-labelingusingthe Triple- 1. Theentiredatasetisdividedinto𝑘consecutivefolds
Barrier Methodwhileconcurrentlyassigningsample
orsegments.
weightstotacklethenon-IIDnatureoffinancialdata,
2. Foreachiteration, adifferentfoldistreatedasthetest
thusenhancingthemodel’slearningefficacy.
set (or validation set), and the remaining 𝑘−1 folds
5. Model Fittingand Testing: Thechosenmodelisfit-
arecombinedtoformthetrainingset.
ted to the data, now with meta-labels and weights,
3. Themodelistrainedonthetrainingsetandvalidated
to evaluate its predictive accuracy under synthesized
onthetestset.
conditions.
4. The performance metric (e.g., Sharpe ratio, annual-
Thispipelineapproachcriticallyexaminestheinterplay izedreturn, drawdown) isrecordedforeachiteration.
betweendifferentcomponentsoftradingstrategies, focusing 5. Afteriteratingthroughallfolds, theperformancemet-
on the risk of overfitting. By simulating complex market ricsareaggregatedtoprovideanoverallperformance
scenarios, weaimtovalidatetherobustnessandadaptability estimate.
ofthesestrategiesforreal-worldapplication.
However, the temporal order of financial data necessi-
2.4. Backtestingon Out-of-Sample Data: tates careful handling. Shuffling or random data partition-
ing, as commonly done in other domains, can lead to sig-
Cross-Validation
nificantbiasesanderroneousconclusions. Forinstance, us-
Inquantitativefinance, therigorofatradingstrategyis
ing future data in constructing the training set, even inad-
often validated through backtesting on out-of-sample data.
vertently, introduceslookaheadbias, severelycompromising
This process involves assessing the strategy’s performance
themodel’svalidity.
usingdatanotemployedduringthemodel’strainingphase,
Moreover, financial markets are influenced by macroe-
providing insights into its real-world applicability. Cross-
conomic factors and market regimes, leading to structural
validation (CV) techniques are pivotal, offering structured
breaks. Thesefactorscanresultinmodelperformancethat
methodstoevaluatethestrategy’seffectivenessandrobust-
variessignificantlyacrossdifferentperiods, makingitdiffi-
ness under various market conditions. The methodologies
cult to generalize the results obtained from a conventional
forbacktestingrangefromconventionalapproacheslike K-
K-Foldcross-validationapproach.
Fold Cross-Validation, which divides the data into multi-
Despitetheselimitations, K-Foldcross-validationisof-
plesegmentsforiterativetesting, tomorespecializedmeth-
ten used in preliminary model assessments, given its sim-
odslike Walk-Forward Cross-Validationand Combinatorial
plicityandwidespreadunderstandinginthestatisticalcom-
Purged Cross-Validation. Each method has distinct char-
munity. However, researchers in quantitative finance must
acteristics in handling the data, particularly addressing the
supplement or replace this method with more appropriate
challengesposedbythetemporaldependenciesandnon-stationarity
techniques, suchas Combinatorial Purged Cross-Validation,
in financial time series. Understanding these methods’ nu-
thataccountforthepeculiaritiesoffinancialtimeseriesdata.
ancesinconstructingbacktestpathwaysiscrucialforaccu-
Itiscrucialtointerprettheresultsof K-Foldcross-validation
rate model validation and developing robust trading strate-
inthecontextoffinancialmarketswithcaution, understand-
gies.
ingthatitsassumptionsmaynotfullyalignwiththeunder-
2.4.1. Conventional Approach: K-Fold lyingdatacharacteristics.
Cross-Validation
2.4.2. Time-Consistent Validation: Walk-Forward
K-foldcross-validationisawidelyrecognizedstatistical
Cross-Validation
methodforvalidatingtheperformanceofpredictivemodels,
Walk-forwardcross-validation (WFCV) isamethodspecif-
particularly in machine learning contexts. It involves par-
ically tailored for time series data, addressing the unique
titioningasampleofdataintocomplementarysubsets, per-
Arian, Norouzi, Seco Page 10 of 26
Electronic copy available at: https://ssrn.com/abstract=4686376


<!-- Page 11 -->

Backtest Overfitting in the Machine Learning Era
challengesposedbyfinancialmarketdata’stemporaldepen- The Purged K-Foldprocessinvolvesseveralkeymodifi-
dencies and non-stationarity. Unlike conventional K-Fold cationstothestandard K-foldcross-validation:
cross-validation, which can inadvertently introduce looka-
1. The dataset is partitioned into 𝑘 folds, ensuring that
headbiasbyshufflingdata, WFCVrespectsthechronolog-
eachfoldisacontiguoussegmentoftimetomaintain
icalorderofobservations, ensuringamorerealisticandro-
thetemporalorderofobservations.
bustvalidationoftradingstrategies.
2. Eachfoldisusedonceasthevalidationset, whilethe
The WFCVprocessinvolvesthefollowingsteps:
remainingfoldsformthetrainingset. However, unlike
1. The dataset is divided into an initial training period
standard K-Foldcross-validation, a"purging"process
andasubsequenttestingperiod. Thesizeofthesepe-
isimplemented.
riodscanbefixedorexpanded.
3. The purging process involves removing observations
2. Themodelistrainedontheinitialtrainingsetandthen
from the training set that occur after the start of the
testedonthesubsequenttestingperiod.
validation period. This is done to eliminate the risk
3. Afterthefirstvalidation, thetrainingandtestingwin-
ofinformationleakagefromthefuture (validationpe-
dows are rolled forward. This means expanding or
riod) intothepast (trainingperiod).
shiftingthetrainingperiodandtestingonthenewsub-
4. Additionally, an"embargo"periodisappliedaftereach
sequentperiod.
trainingfoldendsandbeforethenextvalidationfold
4. Thisprocessisrepeateduntiltheentiredatasetistra-
starts. This embargo period serves as a buffer zone
versed, witheachiterationusinganewtestingperiod
tofurthermitigatetheriskofleakageduetotemporal
immediatelyfollowingthetrainingperiod.
dependenciesthatpurgingmightnotfullyaddress.
5. Performancemetricsarerecordedforeachtestingpe-
5. The model is trained on the purged and embargoed
riodandaggregatedtoevaluatethestrategy’soverall
trainingdataandthenvalidatedontheuntouchedval-
effectiveness.
idationfold.
WFCV’sprimaryadvantageliesinitsalignmentwiththe 6. Performance metrics are recorded for each fold and
practicalscenariosencounteredinlivetrading. Trainingand aggregatedtoprovideanoverallassessment.
testingonconsecutivedatasegmentscloselymimicthereal-
This methodology is particularly effective in financial
world situation where a model is trained on past data and
machinelearning, wheremodelsoftencapturetemporalre-
deployed on future, unseen data. This sequential approach
lationships, and even subtle information leakage can lead
helps understand how a strategy adapts to evolving market
to over-optimistic performance estimates. Purged K-Fold
conditionsandobjectivelyassessesitspredictivepowerand
Cross-Validation ensures a more robust and realistic evalu-
robustnessovertime.
ation of the model’s predictive power by incorporating the
However, WFCV has its limitations. The repetitive re-
purgingandembargomechanisms.
trainingprocesscanbecomputationallyintensive, especially
Purged K-Fold is especially relevant for strategies that
for large datasets and complex models. Additionally, the
relyonfeaturesextractedfromhistoricaldata, asitensures
choice of the size of the training and testing windows can
that the model is not inadvertently trained on future data.
significantlyimpacttheresults, requiringcarefulconsidera-
Thismethodisessentialforpreventingthecommonpitfalls
tionandsensitivityanalysis.
ofoverfittingandselectionbiasinfinancialmodeling.
WFCVisparticularlypertinentinfinancialmachinelearn-
While Purged K-Fold Cross-Validationofferssignificant
ing due to its ability to mitigate overfitting and model de-
advantagesinmaintainingdataintegrity, itrequirescareful
cay risks — common challenges in quantitative finance. It
consideration of the lengths of the purge and embargo pe-
ensuresthatmodelsarecontinuouslyupdatedandvalidated
riods, whichshouldbetailoredtothespecifictemporalde-
against the most recent data, reflecting the dynamic nature
pendenciesintheanalyzedfinancialdata.
offinancialmarkets.
Despite its advantages, WFCV should be employed as
2.4.4. Multi-Scenario, Leakage-Free Validation:
partofacomprehensivestrategyvalidationframework, along-
Combinatorial Purged Cross-Validation
sideothermethodslikecombinatorialpurgedcross-validation,
Combinatorial Purged Cross-Validation (CPCV) isintro-
tofullyaccountforthecomplexitiesoffinancialtimeseries
ducedby Lopezde Prado[2018]asaninnovativeapproach
toensurerobustmodelvalidation.
to address the limitations of single-path testing inherent in
2.4.3. Leakage-Resistant Validation: Purged K-Fold conventional Walk-Forwardand Cross-Validationmethods.
This method is specifically designed for the complex envi-
Purged K-Fold Cross-Validation is an advanced valida-
ronmentoffinancialmachinelearning, wheretemporalde-
tion technique developed by Lopez de Prado [2018] to ad-
pendencies and non-stationarity are prevalent. CPCV gen-
dress the issue of information leakage in financial time se-
erates multiple backtesting paths and integrates a purging
ries, a common pitfall in traditional cross-validation meth-
mechanismtoeliminatetheriskofinformationleakagefrom
ods. Thismethodisparticularlysuitedforvalidatingfinan-
trainingobservations.
cialmodelswheretheintegrityofthetemporalorderofdata
The CPCVmethodisimplementedasfollows:
iscrucialforpreventinglook-aheadbiasesandensuringre-
alisticperformanceestimation.
Arian, Norouzi, Seco Page 11 of 26
Electronic copy available at: https://ssrn.com/abstract=4686376


<!-- Page 12 -->

Backtest Overfitting in the Machine Learning Era
1. Thedataset, consistingof𝑇 observations, ispartitioned (c) This method does not account for the temporal
into𝑁 non-overlappinggroups. Thesegroupsmain- orderofdata, whichcanleadtounrealisticback-
tain the chronological order of data, where the first testpathsinfinancialtimeseriesduetopotential
𝑁 − 1 groups each have a size of ⌊ 𝑇∕𝑁 ⌋, and the informationleakageandautocorrelation.
𝑁-thgroupcontainstheremainingobservations. 2. Walk-Forward (WF)Validation:
2. For a selected size k of the testing set, CPCV calcu- (a) WFValidationinvolvesanexpandingandrolling
latesthenumberofpossibletraining/testingsplitsas window approach. The dataset is sequentially
( 𝑁 ) . Eachcombinationinvolveskgroupsfortest- dividedintoatrainingsetfollowedbyavalida-
𝑁−𝑘
ing, andthetotalnumberofgroupstestedis ( 𝑁 ) ×𝑘, tionset.
𝑁−𝑘 (b) Theuniqueaspectof WFisitschronologicalalign-
ensuringauniformdistributionacrossall𝑁 groups.
ment. The window rolls forward, ensuring the
3. Fromthecombinatorialsplits, eachgroupisuniformly
validation set always follows the training set in
includedinthetestingsets. Thisprocessresultsina
time.
comprehensive series of backtest paths, given by the (c) WF creates a single backtest path that closely
(𝑁)
combinatorialnumber . mimics real-world trading scenarios. However,
𝑘
4. Pathsaregeneratedbytrainingclassifiersonaportion itteststhestrategyonlyonce, providinglimited
𝑘
ofthedata, specifically1− , foreachcombination. insightintoitsrobustnessunderdifferentmarket
𝑁
Thealgorithmensuresthattheportionofdatainthe conditions.
training set is balanced against the number of paths 3. Combinatorial Purged Cross-Validation (CPCV):
andsizeofthetestingsets. (a) CPCVenhancesbacktestpathwaysbyintroduc-
5. The CPCVbacktestingalgorithminvolvespurgingand ingacombinatorialapproach. Thedatasetisdi-
embargoing as introduced before. Each path results vided into 𝑁 groups, from which 𝑘 groups are
from combining forecasts from different groups and selectedinvariouscombinationsfortrainingand
splitcombinations, ensuringacomprehensiveevalua- testing.
tionoftheclassifier’sperformance. (b) This method generates multiple backtest paths,
6. After processing all paths, the performance metrics eachrepresentingadifferentcombinationoftrain-
from each path are aggregated to assess the overall ingandvalidationsets. Itaddressestheissueof
effectivenessofthemodel, providinginsightsintoits single-path dependency seen in WF and tradi-
robustnessandconsistencyacrossvariousmarketcon- tional CV.
(c) CPCV also incorporates purging and embargo-
ditions.
ingtopreventinformationleakage, makingeach
CPCV’suniquecombinatorialapproachallowsforathor- pathmorerealisticandreducingtheriskofover-
ough evaluation of the model under diverse scenarios, ad- fitting.
dressingthecriticaloverfittingissue. Itprovidesamorenu- (d) Thekeyadvantageof CPCVisitsabilitytopro-
ancedandaccurateassessmentofamodel’spredictivecapa- videacomprehensiveviewofthestrategy’sper-
bilitiesinthedynamicfieldoffinancialmarkets. formanceacrossarangeofscenarios, unlikethe
While CPCVoffersanextensivevalidationframework, singlescenariotestedin WFandtraditional CV.
itscombinatorialnaturecanbecomputationallydemanding.
Each CVmethod’sapproachtoconstructingbacktestpath-
Therefore, it’sessentialtoconsidercomputationalresources
ways has implications for its utility in financial modeling.
andexecutiontime, particularlyforlargefinancialdatasets.
Traditional CV’sdisregardfortemporalorderlimitsitsap-
plicability for financial time series. WF’s single-path ap-
2.4.5. Scenario Creation: Constructing Backtest
proachoffersarealisticscenariobutlacksrobustnesstesting.
Pathways
CPCV, with its multiple, purged combinatorial paths, of-
The creation of backtest pathways varies significantly
fersacomprehensiveevaluationofastrategy’sperformance,
amongdifferentcross-validationmethods. Traditional Cross-
making it particularly suitable for complex financial mar-
Validation (CV), Walk-Forward (WF)Validation, and Com-
ketswheremultiplescenariosarecriticalforunderstanding
binatorial Purged Cross-Validation (CPCV) each have dis-
astrategy’seffectiveness.
tinctmethodologiesforgeneratingthesepaths. Understand-
ingthesedifferencesiscrucialforselectingtheappropriate
2.5. Assessmentof Backtest Overfitting
validationmethodinfinancialmodeling.
In the quest to develop robust trading strategies within
1. Traditional Cross-Validation (CV): quantitative finance, the assessment of backtest overfitting
emergesasacrucialfacet. Thissectiondelvesintothemethod-
(a) In traditional CV, the dataset is divided into 𝑘
ologiesdeployedtoevaluateandmitigatetheriskofoverfit-
folds. Each fold is a validation set once, while
ting, a common pitfall where strategies appear effective in
theremainingfoldsconstitutethetrainingset.
retrospectiveanalysesbutfalterinprospectiveapplications.
(b) Thebacktestpathin CVislinearandsequential.
Two pivotal concepts, the Probability of Backtest Overfit-
Eachfold’svalidationresultscontributetoasin-
ting (PBO) and the Deflated Sharpe Ratio (DSR), are har-
gleaggregatedperformancemetric.
nessed to scrutinize the reliability of backtested strategies.
Arian, Norouzi, Seco Page 12 of 26
Electronic copy available at: https://ssrn.com/abstract=4686376


<!-- Page 13 -->

Backtest Overfitting in the Machine Learning Era
PBOisgaugedthrough Combinatorially Symmetric Cross- 5. Finally, the PBOisestimatedbycalculatingthedistri-
Validation (CSCV), a technique that rigorously tests strat- butionofranksout-of-sample (OOS) andintegrating
egyperformanceacrossdiversemarketscenarios. Concur- theprobabilitydistributionfunction𝑓(𝜆) as:
rently, DSRoffersarefinedperspectiveonstrategyefficacy
0
byadjustingthe Probabilistic Sharpe Ratio (PSR) formulti-
PBO= 𝑓(𝜆)𝑑𝜆. (2.27)
pletrials, thusenhancingtheauthenticityofourbacktesting ∫
−∞
results. Together, thesemethodologiesfurnishacomprehen-
wherethe PBOrepresentstheprobabilityofin-sample
siveframeworkforevaluatingtheintegrityoftradingstrate-
optimalstrategiesunderperformingout-of-sample.
gies, ensuringthattheyarenotmerelyartifactsofhistorical
data but are genuinely predictive and robust against future Thisrigorousstatisticalapproachleadingto Eqn.(2.27)
marketconditions. allows us to evaluate the extent of overfitting in our strat-
egy development process, ensuring that selected strategies
2.5.1. Probabilityof Backtest Overfitting:
are robust and not merely tailored to historical market id-
Combinatorially Symmetric Cross-Validation
iosyncrasies.
Backtesttrialsarepivotalintherealmofquantitativefi-
nance, particularlyinthedevelopmentoftradingstrategies. 2.5.2. Probabilityof False Discovery: The Deflated
Utilizingthemethodologyoutlinedinprevioussections, we Sharpe Ratio
performmultiplebacktesttrials, ideallyselectingtheoptimal Inselectingtheoptimalstrategyfrommultiplebacktest
strategybasedonitsperformanceinthesetrials. However, trials, a key concern is the probability of false discovery,
this approach inherently risks backtest overfitting, where a whichreferstothelikelihoodthattheobservedperformance
strategy might show exceptional performance in a histori- ofastrategyisduetochanceratherthantruepredictivepower.
cal context but fails to generalize to new, unseen data. To To address this, we use the Deflated Sharpe Ratio (DSR),
quantitativelyassessandmitigatethisrisk, wecalculatethe whichextendsthe Probabilistic Sharpe Ratio (PSR) concept
Probabilityof Backtest Overfitting (PBO) usingthe Combi- toaccountforthemultiplicityoftrials.
natorially Symmetric Cross-Validation (CSCV) method as The PSR, as introduced by Bailey and Lopez de Prado
introducedby Baileyetal.[2016]. CSCVprovidesamore [2012], adjuststheobserved Sharpe Ratio (𝑆̂𝑅) byaccount-
robustmeasureofastrategy’seffectivenessbyexaminingits
ingforthedistributionalpropertiesofreturns, suchasskew-
performance across different segments of market data, al-
nessandkurtosis. Itiscalculatedas:
lowing us to evaluate the consistency of trial returns both
in-sampleandout-of-sample. ⎛ ⎞
√
The CSCVprocessisoutlinedinthefollowingsteps: ⎜ (𝑆̂𝑅−𝑆𝑅∗) 𝑇 −1 ⎟
𝑃̂𝑆𝑅(𝑆𝑅∗)=𝑍⎜ ⎟, (2.28)
√
1. F
w
o
h
r
e
m
re
at
e
io
ac
n
h
of
co
a
l
p
u
e
m
rf
n
or
r
m
ep
a
r
n
e
c
se
e
n
m
ts
at
t
r
h
i
e
x
l
𝑀
og
o
r
f
et
s
u
iz
r
e
ns
𝑇
s
×
er
𝑁
ies
, ⎜
⎜ ⎝ 1−𝛾̂ 3 𝑆̂𝑅+ 𝛾̂ 4 4 −1 𝑆̂𝑅
2⎟
⎟ ⎠
foraspecificmodelconfigurationover𝑇 timeobser-
vations. where𝑍[.]isthecumulativedistributionfunction (CDF) of
the standard Normal distribution, 𝑇 is the number of ob-
2. Partitioningof𝑀 into𝑆 disjointsubmatrices𝑀 of
𝑠
served returns, 𝛾̂ is the skewness of the returns, and 𝛾̂ is
equaldimensions, witheachsubmatrixbeingoforder 3 4
𝑇 ×𝑁. thekurtosisofthereturns. 𝑆𝑅∗isabenchmark Sharperatio
𝑆 againstwhich𝑆̂𝑅iscompared.
3. Formationofcombinations𝐶 ofthesesubmatrices,
𝑆
𝑆 The Deflated Sharpe Ratio (DSR), asintroducedby Bai-
takeningroupsofsize , yieldingatotalnumberof
2 ley and López de Prado [2014 b], refines the Probabilistic
combinationscalculatedas:
Sharpe Ratio (PSR) asgivenin Eqn.(2.28) byconsidering
( ) 𝑆∕2−1 the number of independent trials. This refinement yields a
𝑆 ∏ 𝑆−𝑖
= . (2.26) more precise measure of the probability of false discovery
𝑆∕2 𝑆∕2−𝑖
𝑖=0 when multiple strategies are tested. Specifically, the DSR
employs a benchmark Sharpe ratio (𝑆𝑅∗) which is calcu-
4. Foreachcombination𝑐 ∈𝐶 , thefollowingstepsare
𝑆 latedin Eqn.(2.29), thatisinfluencedbythevarianceofthe
carriedout:
estimated Sharpe Ratios (𝑆̂𝑅 ) fromthetrials, thenumberof
(a) Formation of the training set 𝐽 and the testing 𝑛
trials (𝑁), andincorporatesthe Euler-Mascheroniconstant
set𝐽̄.
(𝛾):
(b) Computationoftheperformancestatisticvectors
𝑅and𝑅̄ forthetrainingandtestingsets, respec- √
( )
tively. 𝑆𝑅∗ = 𝑉 {𝑆̂𝑅 }
𝑛
(c) Identificationoftheoptimalmodel𝑛∗inthetrain-
ingsetanddeterminationofitsrelativerank𝜔̄ 𝑐 ( (1−𝛾)𝑍−1 ( 1− 1 ) +𝛾𝑍−1 ( 1− 1 𝑒−1 )) ,
inthetestingset. 𝑁 𝑁
( 𝜔̄ ) (2.29)
(d) Definitionofthelogit𝜆 =log 𝑐 .
𝑐 1−𝜔̄
𝑐
Arian, Norouzi, Seco Page 13 of 26
Electronic copy available at: https://ssrn.com/abstract=4686376


<!-- Page 14 -->

Backtest Overfitting in the Machine Learning Era
where𝑍−1 istheinverseofthecumulativedistributionfunc-
tion (CDF) ofthestandardnormaldistribution𝑍. Thisad-
justment is based on the expectation of the maximum of a
sample of IID random variables from the standard normal
distribution, whichisdelineatedin Eqn.(2.30):
[ 1]
E[max{𝑥} ]≈(1−𝛾)𝑍−1 1−
𝑖 𝑖=1,…,𝑙 𝐼
[ 1 ] √
+𝛾𝑍−1 1− 𝑒−1 ≤ 2 log[𝐼], (2.30)
𝐼
with 𝛾 ≈ 0.57721566 representing the Euler-Mascheroni
constant, and 𝐼 ≫ 1 indicating a large number of trials.
This formulation, known as the "False Strategy Theorem"
Lopez de Prado [2020], informs the calculation of 𝑆𝑅∗ in
the DSRmethodology, providingabenchmarkagainstwhich
the observed Sharpe Ratios can be evaluated. The DSR,
computed using this adjusted 𝑆𝑅∗ within the PSR frame-
work, offersacomprehensiveassessmentofastrategy’strue
performancebycorrectingfortheinflationaryeffectofmul-
tipletestingandhelpsdistinguishgenuineskillfromstatis-
ticalflukes.
3. Empirical Results Figure 1: Flow Chart of the Empirical Results Simulation
Inthispivotalsectionofourstudy, wedelveintoacom-
prehensiveempiricalinvestigationdesignedtoperformacom-
parativeanalysisofcross-validationtechniqueswithinasyn- offerafaithfulrepresentationofhistoricalmarketbehavior,
thetic controlled environment. Our empirical endeavor is ensuringtherobustnessofoursimulation. Wedetailthecon-
meticulouslyconstructedtoevaluatetherobustnessofthese figuration of these models, their integration within a cohe-
techniques against backtest overfitting—a critical pitfall in siveframework, andtheparametersetsgoverningtheirbe-
thedevelopmentandvalidationoffinancialmodels. Theex- havior, whicharecrucialforcapturingthecomplexdynam-
ploration unfolds across a series of simulations replicating icsoffinancialmarkets.
market conditions, informed by sophisticated models like
3.1.1. Base Model: The Heston Stochastic Volatility
the Heston Stochastic Volatilityand Merton Jump Diffusion
Model
models, offeringarichtapestryoftranquilandturbulentmar-
The Heston Stochastic Volatility Model in our study is
ketscenarios. Theheartofourinquiryliesintherobustness
parameterizedusing"Calm"and"Volatile"marketregimes,
checkagainstbacktestoverfitting, ensuringthestrategieswe
basedontheempiricalanalysisofthe S&P500 during2008
assessarenotmerelyartifactsofhindsightbiasbutcanstand
and2011 by Papanicolaouand Sircar[2014]asshownin Ta-
the test of uncharted market dynamics. Through our Syn-
ble1. Theseparametersetsarechosenfortheirrobustnessin
thetic Controlled Market Environmentlens,28 strategictri-
replicatingthereal-worldmarketvolatilityobservedinthese
alsin1000 simulationsdissectthestrengthsandweaknesses
periods.
ofaspectrumofout-of-sampletestingmethodologies. Each
technique’s ability to identify and negate overfitting is rig-
3.1.2. Price Jumps: The Merton Jump Diffusion
orously examined, thereby serving as a crucible for deter-
Model
miningthemostreliableapproachtocross-validation. The
Incorporatingthe Merton Jump Diffusion Modelintoour
resultspresentedhereareatestamenttotheanalyticalrigor
simulation, we have calibrated parameters for "Calm" and
ofthestudybutalsoformacornerstonefortheselectionand
"Volatile"marketregimesbasedoninsightsfromthestudy
implementation of cross-validation techniques in the ever-
ofthe S&P500 marketby Hansonand Zhu[2004]asshown
evolvingdomainofquantitativefinance.
in Table 1. This parameterization is critical for accurately
replicating the jump behavior in asset prices characteristic
3.1. Implementationand Parameterizationof
ofvariedmarketvolatilityconditionsasobservedintheem-
Synthetic Data Models
piricaldata.
Thissubsectiondelineatesthedevelopmentofasynthetic
marketenvironment, utilizingthe Heston Stochastic Volatil-
3.1.3. Modeling Market Anomalies: The Drift Burst
ity Modelandthe Merton Jump Diffusion Model, withpa-
Hypothesis
rametersreflectingmarketconditionsduringtranquilandtu-
Adopting the Drift Burst Hypothesis model for simu-
multuoustimes. Parametersderivedfromempiricalstudies
latingmarketanomaliesandspeculativebubbles, ourstudy
Arian, Norouzi, Seco Page 14 of 26
Electronic copy available at: https://ssrn.com/abstract=4686376


<!-- Page 15 -->

Backtest Overfitting in the Machine Learning Era
Table 1
Parameterization of the Heston and Merton Jump Diffusion
Models for Calm and Volatile Market Regimes
Parameter Calm Regime Volatile Regime
Heston Stochastic Volatility
Expected Return (𝜇) 0.1 0.1
Mean Reversion Rate (𝜅) 3.98 3.81
Long-term Variance (𝜃) 0.029 0.25056
Volatilityof Variance (𝜉) 0.389645311 0.59176974
Correlation Coefficient (𝜌) -0.7 -0.7
Merton Jump Diffusion
Jump Intensity (𝜆) 121 121
Meanof Logarithmic Jump Size (𝑚) -0.000709 -0.000709
Varianceof Logarithmic Jump Size (𝑣) 0.0119 0.0119
alignswiththeparametersdelineatedinthefoundationalwork
by Christensenetal.[2022]asshownin Table2. Thismodel
uniquelyinfluencesthesyntheticmarketenvironmentbyim-
posingafixed-lengthregimecharacterizedbypredefinedar-
rays of drift and volatility values. During this regime, the
Heston Stochastic Volatility Modeloperatesunderthesecon-
ditions, featuringnon-stochasticvolatilityandtheabsenceof
jumps. Afterthedriftburstperiod, thesimulationmandates Figure2: Speculative Bubble Simulated Using Drift Burst Hy-
a transition to a different market regime, ensuring a realis- pothesis
tic representation of abrupt market transitions. To ensure
computational stability and circumvent potential zero divi-
Table 3
sion errors, the drift and volatility values are constant at a
Markov Chain Transition Matrix for Market Regimes
specificfractionoftheentireduration, correspondingtothe
explosionfilterwidth. From/To Calm Volatile Speculative Bubble
Calm 1−Δ𝑡 Δ𝑡−0.00001 0.00001
Volatile 20Δ𝑡 1−20Δ𝑡−0.00001 0.00001
Table 2
Speculative Bubble 1−Δ𝑡 Δ𝑡 0.0
Parameters for the Drift Burst Hypothesis Model
Parameter Value
Bubble Length (𝑇 ) 5×252 days
bubble
Pre-Burst Drift Parameter (𝑎 ) 0.35
before
Post-Burst Drift Parameter (𝑎 ) -0.35
after
Pre-Burst Volatility Parameter (𝑏 ) 0.458
before
Post-Burst Volatility Parameter (𝑏 ) 0.458
after
Drift Burst Intensity (𝛼) 0.75
Volatility Burst Intensity (𝛽) 0.225
Explosion Filter Width 0.1
3.1.4. Market Regime Dynamics: Markov Chain
Figure 3: Regime Transition Diagram
Transition Modeling
Inoursimulation, thetransitionsbetweenmarketregimes
are governed by a Markov Chain model, drawing insights
3.1.5. Putting Them All Together: Synthetic
from the works of Xie and Deng [2022] and Elliott et al.
Controlled Market Environment
[2016]onregime-switching Hestonmodelsasshownin Ta-
In this section, we present the integration of a compre-
ble 3. The transition matrix, pivotal to the Markov chain
hensive synthetic market environment, utilizing a blend of
model, ismeticulouslycalibratedbasedonthesereferences
the Heston Stochastic Volatilityand Merton Jump Diffusion
to represent regime shifts accurately, providing a realistic
models. Ourimplementationleveragesthe Pythonprogram-
portrayal of market regime dynamics within our synthetic
minglanguage, numpyfornumericalcomputations, the@ji
controlledenvironment. ⌋
tdecoratorforperformanceoptimization, andthe Quant E-
con library’s qe. Markov Chain for Markov chain generation.
Stochasticelements’reproducibilityisensuredthroughnp.
⌋
Arian, Norouzi, Seco Page 15 of 26
Electronic copy available at: https://ssrn.com/abstract=4686376


<!-- Page 16 -->

Backtest Overfitting in the Machine Learning Era
( ( ))
1 𝑣2
Δ𝑆 = 𝜇− 𝜈 −𝜆 𝑚+ 𝑆Δ𝑡
𝑡 2 𝑡 2 𝑡
√ √
+ 𝜈𝑆𝑍 Δ𝑡+𝑌Δ𝑁(𝑡),
𝑡 𝑡
Δ𝜈 =𝜅(𝜃−𝜈)Δ𝑡+𝜉 √ 𝜈(𝜌 𝜖𝑃 + √ 1−𝜌2𝜖𝜈) √ Δ𝑡.
𝑡 𝑡 𝑡 𝜖 𝑡 𝜖 𝑡
Figure 4: Price Series with Market Regimes
Figure 6: Simulated Log Prices
The synthesized log returns, encapsulating 1000 path-
waysover40 yearsofmarketdynamics, arecomprehensively
summarizedin Table4, presentingthemean, standarddevi-
ation, skewness, and excess kurtosis of returns across dif-
ferent market regimes. Across all regimes, the returns ex-
hibitaslightnegativeskewnessandanotableexcesskurto-
sis, suggesting a leptokurtic distribution more prone to ex-
tremeeventsthananormaldistribution. The’Calm’regime
presentsarelativelyhighermeanandlowervolatility, indi-
catingmorestablemarketconditions. Conversely, the’Volatile’
and’Bubble’regimesmanifestheightenedvolatilityandneg-
ative means, with the ’Bubble’ regime showing the largest
standard deviation and negative mean, characterizing peri-
Figure 5: Markov Regime Transition Matrix Heatmap From odsofsignificantmarketstressandpotentialdownturns.
Simulated Data The Q-Qplotsoflogreturnsin Figure7 illustrateregime-
specific distributions against a theoretical normal distribu-
tion. The ’All’ category shows significant tail deviations,
random.default_rng ().
indicatingoutlierpresence. The’Calm’regimealignsmore
1000 pricepathsaregenerated, eachsimulating40 years
closely with normality except in the tails, hinting at occa-
of market data, equivalent to 40×252 business days. The
sionalextremes. The’Volatile’regime’splotdivergesmore
time step for each simulation is Δ𝑡 = 1 . Initially, 1000 noticeably in the tails, typical of unstable market periods.
252
uniquerandomseedsaregenerated, whichisthefoundation The’Speculative Bubble’displayssteepslopesandmarked
forthepricepathsimulations. Thesimulationsadheretothe taildivergence, characteristicoftherapidpriceswingsdur-
followingequations, asdetailedin Eqn.(2.9) and Eqn.(2.10): ingspeculativephases. Theseplotsunderscorethedistinct
Arian, Norouzi, Seco Page 16 of 26
Electronic copy available at: https://ssrn.com/abstract=4686376


<!-- Page 17 -->

Backtest Overfitting in the Machine Learning Era
distributionalfeaturesofeachregime, fromrelativestability
in’Calm’conditionstothepronouncedtailrisksin’Volatile’
and’Speculative Bubble’scenarios.
Table 4
Descriptive Statistics of Log Returns Overall and For Each
Regime
Regimes Mean Std. Skewness Excess Kurtosis
All 0.000230 0.018216 −0.124832 5.164035
Calm 0.000306 0.014699 −0.096815 3.487078
Volatile−0.000131 0.033051 −0.011123 0.248862
Bubble−0.000753 0.040550 −0.054046 1.120036
Figure 8: Density Distribution of Log Returns Overall and for
Each Regime
betsizing, ismethodicallyimplementedandparameterized,
ensuringaharmoniousintegrationthatfortifiesourmodel’s
predictiveaccuracyandadaptabilitytothenuanceddynam-
icsoffinancialmarkets.
3.2.1. Volatility Assessmentand Event-Based
Sampling
Ourquantitativeanalysisadoptsan Exponentially Weighted
Moving Average (EWMA) approachtoassessdailyvolatil-
ity,𝜎. Utilizingpandas. Series.ewmwithaspanof100 days,
𝑡
we accurately capture the evolving market volatility. This
Figure 7: Q-Q Plot of Log Returns Overall and For Each calculated 𝜎 𝑡 forms the basis for our dynamic threshold in
Regime the symmetric Cumulative Sum (CUSUM) filter Lopez de
Prado[2018]appliedtologprices. Specifically, wesetthe
threshold at 1.8𝜎, which is instrumental in resampling the
𝑡
dataforidentifyingpositionopeningdays. Thismethodol-
3.2. Implementationofthe Financial Machine
ogyensuresadata-driven, responsivesamplingprocess, ef-
Learning Strategy Components
fectivelyaligningourtradingstrategywithprevailingmarket
Ourcomprehensivefinancialmachine-learningstrategy
volatilityandcapturingsignificantpricemovements.
systematically integrates analytical components for robust
and dynamic market analysis. This encompasses a metic-
3.2.2. Determining Trade Directionality
ulous assessment of market volatility, precise event-based
Inourtradingstrategy, tradedirectionalityisdetermined
sampling, strategicdeterminationoftradedirectionality, ap-
using a momentum strategy based on simple moving aver-
plication of advanced meta-labeling techniques, allocation
ages, calculatedviapandas. Series.rolling. Wedefineashort-
ofsampleweights, andselection offinancialfeatures. Our termmovingaverage MA (𝑦)= 1 ∑𝑁 𝑓𝑎𝑠𝑡−1 𝑦 and
methodologyincorporates Fractional Differentiationtoachieve short 𝑡 𝑁 𝑖=0 𝑡−𝑖
𝑓𝑎𝑠𝑡
stationarity without compromising memory, alongside the along-termmovingaverage MA (𝑦)= 1 ∑𝑁 𝑠𝑙𝑜𝑤−1 𝑦 ,
utilizationoftechnicalanalysisindicatorsforenhancedmar- long 𝑡 𝑁 𝑠𝑙𝑜𝑤 𝑖=0 𝑡−𝑖
where 𝑁 and 𝑁 represent the window sizes for the
ketinsight. Thestrategyintricatelybalancesriskandoppor- 𝑓𝑎𝑠𝑡 𝑠𝑙𝑜𝑤
respective averages. Trade positions are initiated based on
tunity through optimal bet sizing, grounded in probabilis-
thecrossoveroftheseaveragesposta CUSUMevent: along
tic assessments from meta-labeling and the uniqueness of
position when MA (𝑦) exceeds MA (𝑦), suggesting
tradelabels. Eachcomponent, fromvolatilityassessmentto short 𝑡 long 𝑡
upward momentum, anda short position when MA (𝑦)
short 𝑡
Arian, Norouzi, Seco Page 17 of 26
Electronic copy available at: https://ssrn.com/abstract=4686376


<!-- Page 18 -->

Backtest Overfitting in the Machine Learning Era
fallsbelow MA (𝑦), indicativeofdownwardmomentum. Thismethodicalapproachtofeatureselection, blending
long 𝑡
Thisapproachalignstradingactionswiththeprevailingmar- fractionaldifferentiationwithtechnicalanalysis, enablesour
kettrend, asreflectedinthepricemomentum. modeltocaptureintricatemarketdynamicseffectively. The
average Pearsoncorrelationbetweenthe22 featuresextracted
3.2.3. Meta-Labeling Strategy fromeachofthe1000 generatedpricepathwaysisdemon-
Incorporating Lopezde Prado[2018]’smeta-labelingwith stratedin Figure9.
thetriple-barriermethod, ourstrategyevaluatestradespost-
momentum-baseddirectiondetermination. Thisprocesscru-
ciallyinformspositionsizingdecisionsandenhancestrade
selection accuracy. The triple-barrier method applies two
horizontal barriers for profit-taking and stop-loss, set with
dynamicvolatility-adjustedthresholdsof0.5𝜎 and1.5𝜎 re-
𝑡 𝑡
spectively, and a vertical barrier with a 20 working days
expiration time. The outcome of a trade is determined as
follows: hitting the upper (profit-taking) barrier results in
a label of 1 for successful trades while reaching the lower
(stop-loss) barrierfirstassignsalabelof−1 forunsuccessful
trades. Ifneitherhorizontalbarrierishitwithinthevertical
time frame, the trade is evaluated based on the sign of the
returnattheendofthisperiod.
3.2.4. Sample Weight Allocation
Ourfinancialmodelcalculatessampleweightsbasedon
theuniquenessandmagnitudeoflogreturnswithinthemeta-
labeleddata. Foreachlabel𝑖, aconcurrencycount𝑐 anda
𝑡
binaryindicatorarray{1 }areusedtodetermineoverlap-
𝑡,𝑖
ping intervals. The preliminary weight 𝑤̃ is computed as
𝑖
‖ ‖ ∑𝑡 𝑖,1 𝑟 𝑡−1,𝑡 ‖ ‖. These weights are then normalized, giving
‖ ‖ 𝑡=𝑡 𝑖,0 𝑐 𝑡 ‖ ‖
𝑤 = 𝑤̃ 𝑖 , ensuringabalancedimpactofeachobserva- Figure 9: Average Feature Correlation Matrix
𝑖 ∑𝐼 𝑤̃
𝑗=1 𝑗
tiononthemodel. Thisapproachemphasizeslearningfrom
distinctmarketevents, thusrefiningthemodel’saccuracy.
3.2.6. Optimal Bet Sizing
3.2.5. Feature Selectionfor Financial Modeling Inourmodel, betsizingiscalibratedusingprobabilities
Wemeticulouslyselectfeaturesinourfinancialmodel- fromthemeta-labelingstrategyandtheuniquenessofeach
ingprocesstoensurerobustpredictivecapability. The Frac- label. Foreachlabel𝑥, wetestthehypothesis𝐻 0 ∶ 𝑝[𝑥 =
tional Differentiation (Frac Diff) featureispivotalinthisen- 1]= 1 usingthestatistic𝑧=
𝑝[𝑥=1]−1
2 . Thebetsize
√
deavor. We used the fixed-width window fractional differ- 2 𝑝[𝑥=1](1−𝑝[𝑥=1])
𝑚isdeterminedas𝑚=2𝑍[𝑧]−1, where𝑍[⋅]isthecumu-
entiation approach to set the weight-loss threshold at 0.01.
lativedistributionfunctionofthestandardnormaldistribu-
Thedifferentiationorderisincrementallydeterminedusing
tion. Tomitigatelook-aheadbias, weshiftthebetsizetime
steps of size 0.1, with a p-value threshold of 0.05 for the
seriesbyoneday, thencalculatethedailystrategyreturnby
ADFtestimplementedusingthestatsmodels.tsa.stattools
multiplying each bet size with the corresponding daily re-
module with a maximum lag of maxlag=1, balancing mem-
turn and position side. The bet size is then readjusted for
ory retention with the attainment of stationarity in the se-
thenexttradingdaybasedontheforthcomingmeta-labelor
ries. Weleveragetheta Pythonlibrarytoconstruct Techni-
liquidatedasneeded. Aggregatebetsizingatanytimestamp
cal Analysis features, utilizing its default configurations to
derive a spectrum of indicators. We apply specific thresh- 𝑡 is computed as 𝑚 =
∑𝐼
𝑖=1
𝑚 𝑖1𝑡,𝑖,
averaging all active bets
oldsforsomeindicatorstoenhancetheirinterpretability:
𝑡 ∑𝐼
𝑖=1
1𝑡,𝑖
at that time. This approach ensures dynamic adaptation of
1. ADX Strength: A threshold of 25 distinguishes be- betsizestotheevolvingmarketconditions, aligningwiththe
tweenstrongandweaktrends. probabilitiesanduniquenessoftradesignals.
2. RSI Signal: Thresholds of 30 and 70 identify over-
3.3. Designand Parameter Dynamicsof Trial
boughtandoversoldconditions.
Simulations
3. CCI Signal: Thresholds of -100 and 100 signal po-
tentialmarketreversals. Inourempiricalexploration, wemethodicallydesign28
strategytrialsbymanipulatingtwocorecomponents:thepa-
4. Stochastic Signal: Thresholdsof20 and80 indicate
rametersofthemomentumcross-overstrategyandthecon-
overboughtandoversoldconditions.
figurationsofvariousmachinelearningmodels. Wecreate
Arian, Norouzi, Seco Page 18 of 26
Electronic copy available at: https://ssrn.com/abstract=4686376


<!-- Page 19 -->

Backtest Overfitting in the Machine Learning Era
a comprehensive array of trials by alternating between dif- this meticulous approach ensures robustness and accuracy
ferent sets of rolling window sizes in the momentum strat- incomparingourout-of-sampletestingprocedures.
egyandadiverserangeofhyperparametersinthemachine
learningmodels. Thisapproachallowsustoassesstheper- 3.4.1. Implementationof K-Fold Cross-Validation
formance impact of these variables under varying market Ourimplementationof K-Fold Cross-Validation (KFold)
conditions and model specifications. Each trial represents infinancialmodelingutilizesthe KFoldclasswithinthe Cros
⌋
auniquecombinationoftheseconfigurations, providingus s Validator Controllerframework. Configuredwithn_split
⌋
withabroadspectrumofinsightsintothedynamicsofour s=4, thisapproachpartitionsthedatasetintofourdistinctseg-
financialstrategy. ments, adheringtotheconventionalmethodologyof KFold.
Eachsegmentsequentiallyservesasatestset, whilethere-
3.3.1. Momentum Cross-Over Strategy Variations maining data forms the training set. This structure is piv-
Ourfinancialmodelevaluatesmomentumcross-overstrat- otalinourfinancialtimeseriesanalysis, whereitiscrucial
egyvariationsbyalteringthemovingaverages’rollingwin- toavoidlook-aheadbiasandmaintainthechronologicalin-
dowsizes. Wetestfourdistinctconfigurations: (5,10),(20, tegrityofdata.
50), (50, 100), and (70, 140), representing various pairs of
Cross Validator Controller (
fast and slow-moving average window sizes. These trials
'kfold',
systematicallyexaminethestrategy’sperformanceunderdi-
n_splits=4,
versetemporaldynamics.
).cross_validator
3.3.2. Machine Learning Models Variations
Giventhenatureoffinancialdata, characterizedbytem-
Ourstrategyexploresvariousmachinelearningmodels poraldependencies, our KFoldimplementationistailoredto
topredictmeta-labels, eachwithspecificparameterconfigu- respectthesesequences, ensuringmoreaccurateandrealistic
rations. Certainhyperparametersareexploredwithasingle modelvalidation. Thisadherencetothetimeseriesstructure
candidatevalue, whileothersaretestedacrossmultipleval- inour KFoldsetupunderscoresourcommitmenttorigorous,
ues, ensuring all cases are comprehensively utilized in our temporally-awareanalyticalpracticesinfinancialmodeling.
strategy trials. The configurations are strategically chosen
toheightenthepotentialforoverfitting. Wehave 3.4.2. Implementationof Walk-Forward
Cross-Validation
1. k-Nearest Neighbors (k-NN):Implementedviaskl
⌋
Implementedusingthe Walk Forwardclass, our Walk-Forward
earn.neighbors. KNeighbors Classifier, withneighbors
Cross-Validation (WFCV) employsthe Cross Validator Cont
parameter varied as n_neighbors : [1, 2, 3]. The
roller with n_splits=4, indicating a division of the datase
⌋
t
dataisstandardizedusingsklearn.preprocessing. St
⌋ into four sequential segments. This ensures chronological
andard Scaler within a custom pipeline extending sk
⌋ trainingandtestingphases, whichiscrucialformaintaining
learn.pipeline. Pipeline, which incorporates sample
temporalintegrityinfinancialdataanalysis. Thisapproach,
weights.
emphasizingthesequenceandstructureofdata, mirrorsreal-
2. Decision Tree: Utilizedthroughsklearn.tree. Decis
world financial market dynamics and is key to achieving a
⌋
ion Tree Classifier, withparameterssettomin_sampl
realisticassessmentofmodelperformance. Thespecificpa-
⌋
es_split : [2]andmin_samples_leaf : [1].
rameterization of WFCV underscores our commitment to
3. XGBoost:Executedusingxgboost. XGBClassifier, with temporalconsistencyandrobustvalidationinfinancialmod-
parametersincludingn_estimators : [1000], max_de eling.
⌋
pth : [1000000000], learning_rate : [1, 10, 100],
Cross Validator Controller (
subsample : [1.0], andcolsample_bytree : [1.0].
'walkforward',
3.4. Out-of-Sample Testingvia Cross-Validation n_splits=4,
).cross_validator,
Inourquantitativefinanceframework, weapplyacom-
prehensive suite of cross-validation techniques to conduct
Byaligningmodelevaluationwiththechronologicalpro-
out-of-sampletesting, employingtherobust Cross Validator
gressionofmarketdata, thisconfigurationenhancesthere-
⌋
Controllerforinitializingdifferentvalidationmethods. This
liabilityandrelevanceofourstrategyassessments.
includes K-Fold, Walk-Forward, Purged K-Fold, and Com-
binatorial Purged Cross-Validation, eachspecificallyadapted 3.4.3. Implementationof Purged K-Fold
to the challenges of financial time series data. Utilizing Cross-Validation
⌋
Cross Validator.backtest_predictions, we generate backtest Theimplementationof Purged K-Fold Cross-Validation
paths for each cross-validation method, comprising prob- inourframeworkleveragesthe Purged KFoldclassthroughthe
abilities corresponding to the meta-labels. For labels en- Cross Validator Controller, specificallytailoredforfinancial
countered across multiple backtest paths, we average their time series data. Configured with n_splits=4, an embargo
probabilities, creating a consolidated measure that informs rate of embargo=0.02, and time-based partitioning, this ap-
subsequent strategy performance calculations. Integrating proachrigorouslymaintainstheintegrityofthetemporalor-
traditional and innovative cross-validation methodologies, der. Theinitializationparametersensurethatthedatasetis
Arian, Norouzi, Seco Page 19 of 26
Electronic copy available at: https://ssrn.com/abstract=4686376


<!-- Page 20 -->

Backtest Overfitting in the Machine Learning Era
dividedintofourcontiguoussegments, eachrepresentinga ourcollectionof28 strategytrials. Theanalysisismethod-
distinctperiodintime. ically structured to encompass a holistic evaluation of the
entire performance timeline and an annualized, segmented
Cross Validator Controller ( examination. Eachyearismeticulouslyanalyzed, consider-
'purgedkfold', ing252 tradingdayspersegment. Thisdual-facetedanalysis
n_splits=4, offersinsightsintothestrategies’overallandspecificyearly
times=times, performances and serves as a litmus test for the effective-
embargo=0.02 nessofdifferentout-of-sampletestingtechniquesincurbing
).cross_validator overfitting. Thescatterplotin Figure10 illustratesanegligi-
blecorrelationof-0.03 betweenthe Probabilityof Backtest
Thisstructureisinstrumentalformitigatinginformation
Overfitting (PBO) andthe Best Trial Deflated Sharpe Ratio
leakage and lookahead biases by purging training data that
(DSR) Test Statistic in the overall analysis, signaling their
overlapswiththevalidationperiodandimplementinganem-
independenceasevaluativetools. Theirindependenceisin-
bargo period. Such modifications are crucial in financial
strumental, asitimpliesamulti-facetedassessmentofback-
modeling, where the chronological sequence of data plays
testvalidity, combiningrobustnesschecksagainstoverfitting
a pivotal role in the validity and realism of backtesting re-
withadjustmentsformultiplehypothesistesting, therebyen-
sults. Our Purged K-Fold setup, therefore, ensures a more
richingthestrategyselectionprocesswithdiverseyetcom-
authenticandreliableassessmentofthemodel’spredictive
plementaryreliabilitymetrics.
capabilities.
3.4.4. Implementationof Combinatorial Purged
Cross-Validation
Ourimplementationof Combinatorial Purged Cross-Validation
(CPCV) isrealizedusingthe Combinatorial Purgedclass, or-
chestrated through the Cross Validator Controller. Tailored
forfinancialtimeseriesanalysis, CPCVisinitializedwith
⌋
n_splits=8 andn_test_groups=2, signifyingthatthedatasetis
dividedintoeightnon-overlappinggroupswithtwogroups
designatedfortestingineachcombinatorialsplit. Addition-
ally, anembargorateofembargo=0.02 isappliedtomitigate
information leakage further. This setup is encapsulated in
thefollowingconfiguration:
Cross Validator Controller (
'combinatorialpurged',
n_splits=8,
n_test_groups=2,
times=times,
embargo=0.02
).cross_validator,
The CPCVapproach, withitscombinatorialnature, en-
Figure 10: Probability of Backtest Overfitting vs Best Trial
suresathoroughanddiversifiedexaminationofthemodel’s
Deflated Sharpe Ratio Test Statistic with the Correlation of
performance across multiple backtest paths, effectively ad-
-0.03
dressing overfitting concerns prevalent in financial model-
ing. Theintegrationofpurgingandembargoingwithinthis
frameworkfurtherbolstersthetemporalintegrityoftheval-
3.5.1. Implementationof Combinatorially Symmetric
idation process, making CPCV a robust tool for assessing
Cross-Validation (CSCV)
predictive models in the dynamic environment of financial
Forassessingthe Probabilityof Backtest Overfitting (PBO)
markets. Theselectionofparametersinourimplementation
inour28 strategytrials, weimplementthe Combinatorially
reflects a deliberate balance between comprehensive back-
Symmetric Cross-Validation (CSCV) using Python’s numpy
testingandcomputationalfeasibility.
library. The CSCV method is applied to a matrix of strat-
egy returns, which represents the log returns for different
3.5. Comparative Assessmentof Out-of-Sample
modelconfigurationsacrossvarioustimeobservations. We
Testing Techniques
utilizen_partitions = 16 todividetheperformancematrix
Inourstudy, weconductadetailedcomparativeassess-
intoanequalnumberofdisjointsubmatrices, ensuringabal-
mentofvariousout-of-sampletestingmethodologies, focus-
ancedevaluationacrossmultipledatasegments. Ourevalua-
ingonreducingthelikelihoodofbacktestoverfittingwithin
tionmetric, the Sharperatio, iscomputedthroughacustom
Arian, Norouzi, Seco Page 20 of 26
Electronic copy available at: https://ssrn.com/abstract=4686376


<!-- Page 21 -->

Backtest Overfitting in the Machine Learning Era
function to measure the performance of strategies over an
annual risk-free rate of 0.05. The probability_of_backtes
⌋
t_overfittingfunctionsynthesizesthisdata, estimatingthe
PBOandproducinganarrayoflogitvalues. Thisprocedure
thoroughly compares each strategy’s in-sample and out-of-
sample performance, which is crucial for identifying over-
fittingandensuringtherobustnessofourtradingstrategies
againstfuturemarketscenarios.
3.5.2. Utilizationofthe Deflated Sharpe Ratioin False
Discovery Analysis
In our analysis of 28 trading strategy trials, we employ
the Deflated Sharpe Ratio (DSR) tocriticallyassessthelike-
lihoodoffalsediscoveries. Thisevaluationbeginswithcom-
putingthe Sharpe Ratiosforeachstrategy, usinganannual
risk-free rate of 0.05, to identify the best-performing trial.
Subsequently, wecalculatethe DSR, consideringtheskew-
ness, kurtosisoflogreturns, andthevarianceof Sharpe Ra-
tiosacrossalltrials. Crucially, wefocusontheteststatistic
derivedfromthe DSRcalculationratherthanitsvaluepost-
application in the normal cumulative distribution function
(CDF).Thisapproachallowsustomoreaccuratelydiscern
betweenstrategiesthatexhibitpredictiveskillandthosethat Figure 11: Distribution of Probability of Backtest Overfitting
mayhaveperformedwellbychance, ensuringamorerobust Values Across Simulations For Each Cross-Validation Method
andreliableselectionoftheoptimaltradingstrategy.
3.6. Analytical Approachesfor Out-of-Sample
Testing Results Evaluation
Ameticulousstatisticalexaminationofout-of-sampletest-
ingresultsisvitalforvalidatingtherobustnessofcross-validation
methods against overfitting. In this subsection, we imple-
mentacomprehensivesuiteofnon-parametricstatisticaltests,
augmented by multivariate analysis, to rigorously compare
thedistributionalcharacteristicsofmetricvaluesderivedfrom
different cross-validation techniques. Utilizing Python li-
braries scipy.stats, scikit_posthocs, and sklearn, we per-
form the Kruskal-Wallis H Test for a global understanding
of distributional differences, followed by pairwise compar-
isons via Dunn’s Test for specific methodological distinc-
tions. Furthermore, weconduct Principal Component Anal-
ysis (PCA) to assess the independence of simulation out-
puts, whichprovidesadeeperinsightintotheinterdependen-
ciesofthebacktestoverfittingmetricsacrossoursimulation
trials. Together, these analytical strategies ensure a robust
evaluation of the cross-validation methods’ stability, relia-
bility, andindependence, paintingacomprehensivepicture
oftheirperformanceinfinancialstrategyvalidation.
Figure 12: Distribution of Best Trial Deflated Sharpe Ra-
3.6.1. Assessing Distributional Variance Across
tio Test Statistic Values Across Simulations For Each Cross-
Methods Validation Method
The Kruskal-Wallis HTest, anon-parametricmethodfor
determining stochastic dominance among multiple groups,
evaluatesthenullhypothesisthatthedistributionsofmetric calculatetheeffectsizeusing𝜂2 = 𝐻−(𝑘−1), where𝐻 isthe
𝑁−𝑘
values across different cross-validation methods are identi- Kruskal-Wallisstatistic,𝑘representsthenumberofgroups,
cal. Unlikeparametriccounterparts, itdoesnotnecessitate and𝑁 isthetotalnumberofobservations. Thisstatisticde-
theassumptionofnormallydistributeddata. Significantre- lineatestheproportionoftotalvarianceinthemetricvalues
sultsfromthistestsuggestatleastonegroup’sdistribution thecross-validationmethodexplains, providinginsightsbe-
differs from others. Should the test yield significance, we
Arian, Norouzi, Seco Page 21 of 26
Electronic copy available at: https://ssrn.com/abstract=4686376


<!-- Page 22 -->

Backtest Overfitting in the Machine Learning Era
yondstatisticalsignificancetothemagnitudeofdifferences
observed.
3.6.2. Delineating Distinct Distributionsvia Pairwise
Comparisons
Weproceedwith Dunn’s Testforpairwisecomparisons
upondetectingsignificantvarianceindistributionswiththe
Kruskal-Wallis H Test. This method pinpoints the specific
cross-validationmethodswithstatisticallydiscerniblediffer-
ences. Dunn’s Test is adept for multiple comparisons, ap-
plying the Bonferroni correction to adjust the significance
threshold, thuscontrollingthefamily-wiseerrorrateandre-
inforcing the validity of the inferential distinctions among
pairs.
3.6.3. Principal Component Analysisfor Simulation
Independence
To evaluate the correlation and ascertain the indepen-
denceofour1000 Simulation, weconducteda Principal Com-
ponent Analysis (PCA) ontheannualizedmetricvalues, as-
sessing backtest overfitting. By examining the Cumulative
Explained Varianceby PCAComponentsforeach CVmethod,
wecoulddiscernthedegreeofcorrelationamongtrials. A Figure 13: Comparison of Overall Probability of Backtest
higherexplainedvariancebyfewerprincipalcomponentsin- Overfittingand Best Trial Deflated Sharpe Ratio Test Statistic
dicatesastrongercorrelationandlessindependencebetween Values Across Simulations For Each Cross-Validation Method
thetrials, whichiscriticalinunderstandingthediversifica-
tionbenefitsofourstrategyportfolio. Our PCAimplemen-
tationutilized Python’ssklearn.pipeline. Pipeline, incorpo- estmedian PBOvalueof0.451437, suggestingapotentially
ratingasklearn.preprocessing. Standard Scalertonormalize higherriskofoverfittingthanothermethods. The Kruskal-
thedata, asimpleaverageimputersklearn.impute. Simple I Wallistestindicatedsignificantdiscrepanciesamongthegroups
⌋
mputerforhandlingmissingvalues, andasklearn.decompo (𝑝 = 7.05×10−9, 𝜂2 = 0.01022), underscoring the pres-
⌋
sition. PCAtoperformthedecomposition, therebyproviding ence of at least one method with a distinct PBO distribu-
aquantitativeassessmentofthesimulations’interdependen- tion. Dunn’spairwisecomparison, aspresentedin Table5,
cies. further corroborated significant distinctions: ’Combinato-
rial Purged’demonstratedamarkedlylower PBOcompared
3.7. Disclosureof Empirical Findings toboth’K-Fold’(𝑝=4.20×10−6) and’Purged K-Fold’(𝑝=
In this subsection, we unveil the empirical findings de- 3.32×10−7), aswellasagainst’Walk-Forward’(𝑝=1.09×
rived from our meticulous analysis of backtest overfitting 10−6), implyingasuperiorefficacyinmitigatingtheriskof
withinvariouscross-validationframeworks. Ourinvestiga- overfitting. Meanwhile,’K-Fold’and’Purged K-Fold’were
tionmeticulouslyscrutinizesthe Probabilityof Backtest Over- statisticallyindistinguishablefrom’Walk-Forward’, suggest-
fitting (PBO) andthe Best Trial Deflated Sharpe Ratio (DSR) ingsimilar PBOprofilesbetweenthesemethods. Thesein-
Test Statistic, employing robust statistical methods to dis- sightsareessentialforstrategicallyselectingcross-validation
cernsignificantdisparitiesandtemporalvariabilitiesacross methodologiesinquantitativefinancemodels, aimingtore-
multiple validation techniques. The ensuing results illumi- ducetheprobabilityofoverfittingwhileensuringrobustpre-
natethecomparativeresilienceofthesemethodstotheper- dictiveperformance.
ilsofoverfittingandofferanuancedunderstandingoftheir Oursimulations’non-parametricanalysisofthe Best Trial
temporalbehavior, therebyguidingthestrategicselectionof Deflated Sharpe Ratio (DSR)Test Statisticvaluesrevealed
themostrobustandstablecross-validationapproachesinfi- distinct statistical characteristics among the various cross-
nancial strategy development. This disclosure is anchored validation methods. As illustrated in Figure 13, the distri-
inaprofoundcommitmenttoempiricalrigortobolsterthe butionof DSRvaluesforthe’Walk-Forward’methodology
integrity of model validation processes within quantitative markedly differed from those of other methods, with a no-
finance. tablylowermedianvalueof0.160818. The Kruskal-Wallis
testconfirmedsignificantdisparitiesacrossthedistributions
3.7.1. Overall Assessmentof Backtest Overfitting (𝑝 = 5.0367×10−15, 𝜂2 = 0.017), suggesting at least one
Ourcomparativeanalysisofthe Probabilityof Backtest method deviates from the others in terms of DSR values.
Overfitting (PBO) acrossvariouscross-validationtechniques Subsequentpairwisecomparisonsusing Dunn’s Test, detailed
revealed significant statistical differences, as visualized in in Table6, identified’Walk-Forward’assignificantlydiffer-
Figure13. The’Walk-Forward’approachexhibitedthehigh-
Arian, Norouzi, Seco Page 22 of 26
Electronic copy available at: https://ssrn.com/abstract=4686376


<!-- Page 23 -->

Backtest Overfitting in the Machine Learning Era
Table 5
Distributions Comparison for Probability of Backtest Over-
fitting Values Across Simulations For Each Cross-Validation
Method
Test P-Value Effect Size (𝜂2)
Kruskal Wallis 7.05 e-09 0.01022
Dunn’s Test p-Value Significant
Combinatorial Purgedvs. K-Fold 4.20 e-06 Yes
Combinatorial Purgedvs. Purged K-Fold 3.32 e-07 Yes
Combinatorial Purgedvs. Walk-Forward 1.09 e-06 Yes
K-Foldvs. Purged K-Fold 1.0 No
K-Foldvs. Walk-Forward 1.0 No
Purged K-Foldvs. Walk-Forward 1.0 No
ent from both ’K-Fold’ and ’Purged K-Fold’ (𝑝 = 8.32 ×
10−12 and 𝑝 = 1.42×10−11, respectively), while ’Combi-
natorial Purged’didnotexhibitsignificantdifferencesfrom
’K-Fold’and’Purged K-Fold’. Thesestatisticalinsightsare
crucialforrecognizingeachcross-validationmethod’srela-
tiveeffectivenessandcharacteristicsinminimizingoverfit-
tingwhilestrivingforoptimal Sharperatioperformance. Figure 14: Comparison of Temporal Probability of Backtest
Overfittingand Best Trial Deflated Sharpe Ratio Test Statistic
Efficiency Ratio Values Across Simulations For Each Cross-
Table 6
Validation Method
Distributions Comparison for Best Trial Deflated Sharpe Ra-
tio Test Statistic Values Across Simulations For Each Cross-
Validation Method
K-Fold’ and ’Walk-Forward’ (𝑝 = 9.60 × 10−54). These
Test P-Value Effect Size (𝜂2)
findingsunderscoretheimportanceofconsideringthe Effi-
Kruskal Wallis 5.0367 e-15 0.0174
ciency Ratiowhenevaluatingtheconsistencyof PBOover
Dunn’s Test p-Value Significant time, with’Walk-Forward’showingthegreatestvariability
Combinatorial Purgedvs. K-Fold 1.0 No and, thus, potentially, theleaststabilityin PBOvaluesyear
Combinatorial Purgedvs. Purged K-Fold 1.0 No overyear.
Combinatorial Purgedvs. Walk-Forward 3.21-09 Yes
K-Foldvs. Purged K-Fold 1.0 No
Table 7
K-Foldvs. Walk-Forward 8.32 e-12 Yes
Distributions Comparison for Probability of Backtest Over-
Purged K-Foldvs. Walk-Forward 1.42 e-11 Yes
fitting Efficiency Ratio Values Across Simulations For Each
Cross-Validation Method
Test P-Value Effect Size (𝜂2)
Kruskal Wallis 1.426 e-76 0.08876
3.7.2. Temporal Variabilityof Overfitting Assessment
𝜎2 Dunn’s Test p-Value Significant
The annual Efficiency Ratio, defined as , was calcu-
𝜇2
Combinatorial Purgedvs. K-Fold 8.14 e-23 Yes
latedforthe Probabilityof Backtest Overfitting (PBO) across
Combinatorial Purgedvs. Purged K-Fold 9.29 e-22 Yes
variouscross-validationmethodstoassesstherelativevari-
Combinatorial Purgedvs. Walk-Forward 1.06 e-07 Yes
abilityof PBOthroughtime. Asdepictedin Figure14,’Walk-
K-Foldvs. Purged K-Fold 1.0 No
Forward’displayedthehighestmedian Efficiency Ratiovalue K-Foldvs. Walk-Forward 2.15 e-54 Yes
of0.224821, suggestingahighervariancetothemean PBO Purged K-Foldvs. Walk-Forward 9.60 e-54 Yes
value than the other methods. The Kruskal-Wallis test re-
vealedhighlysignificantdifferencesinthe Efficiency Ratio
distributionsacrossmethods (𝑝=1.426×10−76,𝜂2 =0.09),
𝜎2
The annual Efficiency Ratio for the Best Trial De-
indicating varying levels of PBO stability. Dunn’s Test re- 𝜇2
sults, presented in Table 7, showed significant differences flated Sharpe Ratio (DSR) Test Statistic values was scruti-
between ’Combinatorial Purged’ and both ’K-Fold’ (𝑝 = nized to evaluate the variability of the DSR through time
8.14×10−23) and’Purged K-Fold’(𝑝=9.29×10−22), aswell for each cross-validation method. Figure 14 illustrates the
as’Walk-Forward’(𝑝=1.06×10−7). Additionally,’K-Fold’ distributions of these ratios, with ’Combinatorial Purged’
and’Walk-Forward’demonstratedasignificantvariancein showinganotablylowermedian Efficiency Ratioof34.30,
their Efficiency Ratios (𝑝 = 2.15×10−54), as did ’Purged suggestinggreaterefficiencyin DSRperformance. Instark
Arian, Norouzi, Seco Page 23 of 26
Electronic copy available at: https://ssrn.com/abstract=4686376


<!-- Page 24 -->

Backtest Overfitting in the Machine Learning Era
contrast,’K-Fold’and’Purged K-Fold’showedhigherme-
dianvaluesof84.36 and82.59, respectively, indicatingless
efficiency. The Kruskal-Wallis test underscored significant
differencesinefficiencyacrossmethods (𝑝 = 1.43×10−76,
𝜂2 =0.0888). Accordingto Dunn’s Testresultsshownin Ta-
ble8,’Combinatorial Purged’demonstratedstatisticallysig-
nificant higher efficiency when compared to both ’K-Fold’
(𝑝=8.14×10−23) and’Purged K-Fold’(𝑝=9.29×10−22),
aswellas’Walk-Forward’(𝑝 = 1.06×10−7). Conversely,
’K-Fold’and’Walk-Forward’showednosignificantdiffer-
enceintheirefficiency (𝑝=2.15×10−54), similarto’Purged
K-Fold’versus’Walk-Forward’(𝑝 = 9.60×10−54). These
findings are instrumental for discerning the most efficient
cross-validationmethodregarding DSRvariability, whichis
crucialforachievingstableperformanceinfinancialmachine-
learningapplications.
Table 8
Distributions Comparisonfor Best Trial Deflated Sharpe Ratio
Test Statistic Efficiency Ratio Values Across Simulations For
Each Cross-Validation Method
Test P-Value Effect Size (𝜂2)
Kruskal Wallis 1.43 e-76 0.0888 Figure 15: Comparison of Temporal Probability of Backtest
Overfittingand Best Trial Deflated Sharpe Ratio Test Statistic
Dunn’s Test p-Value Significant
ADFTest Statistic Values Across Simulations For Each Cross-
Combinatorial Purgedvs. K-Fold 8.14 e-23 Yes Validation Method
Combinatorial Purgedvs. Purged K-Fold 9.29 e-22 Yes
Combinatorial Purgedvs. Walk-Forward 1.06 e-07 Yes
K-Foldvs. Purged K-Fold 1.00 No Table 9
K-Foldvs. Walk-Forward 2.15 e-54 Yes Distributions Comparison for Probability of Backtest Overfit-
Purged K-Foldvs. Walk-Forward 9.60 e-54 Yes ting ADF Test Statistic Values Across Simulations For Each
Cross-Validation Method
Test P-Value Effect Size (𝜂2)
Kruskal Wallis 0.0 0.55106
3.7.3. Temporal Stationarityof Overfitting Assessment Dunn’s Test p-Value Significant
In our annual time series analysis of the Probability of Combinatorial Purgedvs. K-Fold 1.0 No
Backtest Overfitting (PBO), the Augmented Dickey-Fuller Combinatorial Purgedvs. Purged K-Fold 1.0 No
(ADF) teststatisticvalueswereutilizedtoexaminethesta- Combinatorial Purgedvs. Walk-Forward 0.0 Yes
tionarityofthe PBOthroughtime. Thesevaluesaredepicted K-Foldvs. Purged K-Fold 1.0 No
in Figure 15 and quantitatively analyzed in Table 9. The K-Foldvs. Walk-Forward 0.0 Yes
Purged K-Foldvs. Walk-Forward 0.0 Yes
’Walk-Forward’methodexhibitedamarkedlyhighermedian
ADFvalueof-2.41, indicatinglessstationarityandgreater
trendpresencethanothermethods. The Kruskal-Wallistest
yielded a significant result (𝑝 < 0.01, 𝜂2 = 0.55), imply-
ingsubstantialdifferencesinthetimeseriescharacteristics Table10. The’Walk-Forward’approachdemonstratedahigher
among the methods. Dunn’s Test revealed that the ’Walk- median ADFvalueof-3.86, suggestingaweakerpresenceof
Forward’method’s ADFvaluesweresignificantlydifferent stationarity compared to the more negative ADF values of
from those of ’K-Fold’, ’Purged K-Fold’, and ’Combinato- theothermethods, whichimpliesastrongerrejectionofthe
rial Purged’(allwith𝑝 = 0.0), underscoringitsdistinctbe- unitrootandthusastrongerindicationofstationarity. The
havior in terms of stationarity. These findings suggest that Kruskal-Wallistestprovidedextremelysignificantevidence
while’Walk-Forward’mightbemorepronetoexhibittrends ofdistributionaldifferencesamongthemethods (𝑝=2.01×
in PBO over time, the other methods did not show signifi- 10−50, 𝜂2 = 0.059). Dunn’s Test further identified signifi-
cantdifferencesamongthemselves, indicatingsimilarlevels cantdifferencesbetween’Walk-Forward’andallothermeth-
ofstationarityintheirrespective PBOvalues. ods, with’Walk-Forward’beinglessstationarycomparedto
Thestationarityoftheannual Best Trial Deflated Sharpe ’K-Fold’(𝑝 = 2.38×10−37),’Purged K-Fold’(𝑝 = 1.65×
Ratio (DSR)Test Statisticvalueswasassessedusingthe Aug- 10−31), and ’Combinatorial Purged’ (𝑝 = 9.81 × 10−36).
mented Dickey-Fuller (ADF) test, withthedistributionsvi- Theseresultsindicatethat’Walk-Forward’maybelesssuit-
sualizedin Figure15 andthestatisticalanalysisdetailedin able for strategies that require a consistent DSR over time,
Arian, Norouzi, Seco Page 24 of 26
Electronic copy available at: https://ssrn.com/abstract=4686376


<!-- Page 25 -->

Backtest Overfitting in the Machine Learning Era
whiletheothercross-validationmethodsdonotexhibitsig-
nificantdifferencesregardingstationarityintheir DSRval-
ues.
Table 10
Distributions Comparisonfor Best Trial Deflated Sharpe Ratio
Test Statistic ADF Test Statistic Values Across Simulations
For Each Cross-Validation Method
Test P-Value Effect Size (𝜂2)
Kruskal Wallis 2.01 e-50 0.05853
Dunn’s Test p-Value Significant
Combinatorial Purgedvs. K-Fold 1.0 No
Combinatorial Purgedvs. Purged K-Fold 1.0 No
Combinatorial Purgedvs. Walk-Forward 9.81 e-36 Yes
K-Foldvs. Purged K-Fold 1.0 No
K-Foldvs. Walk-Forward 2.38 e-37 Yes
Purged K-Foldvs. Walk-Forward 1.65 e-31 Yes
3.7.4. Correlationof Overfitting Assessments Across
Simulations
Figure 16: Temporal Probability of Backtest Overfitting and
Our Principal Component Analysis (PCA) investigation
Best Trial Deflated Sharpe Ratio Test Statistic Cumulative
intothecorrelationbetweendifferentoverfittingmetricsacross
Explained Variance by PCA Components For Each Cross-
simulationsrevealednotablepatternsofdependency. Asde-
Validation Method
pictedinthe PCAcumulativeexplainedvarianceplots (Fig-
ure16), boththe Probabilityof Backtest Overfitting (PBO)
andthe Best Trial Deflated Sharpe Ratio (DSR)Test Statistic of Backtest Overfitting (PBO) revealed ’Walk-Forward’ as
valuesforthe’Walk-Forward’methodarecharacterizedbya havingthehighestmedianvalue, indicatinggreatertempo-
higherexplainedvariancewithfewerprincipalcomponents. ralvariabilityandreducedstability. ’Combinatorial Purged’,
This pattern indicates a lower level of simulation result in- however, displayed a notably lower Efficiency Ratio, sug-
dependence, suggestingthattheperformancemetricsofthe gestingenhancedtemporalstabilityandconsistencyinper-
’Walk-Forward’methodaremoreinterrelatedthanthoseof formance. Whenevaluatingthe Efficiency Ratioforthe DSR
othercross-validationmethods. Suchacorrelationstructure Test Statistic, ’Combinatorial Purged’ exhibited a notably
withinthe’Walk-Forward’simulationsmayimplyaninher- lowermedianvalue, implyinggreaterefficiencyandstabil-
ent bias or systemic influence affecting the simulations, an ityinits DSRperformanceovertime. Thiscontrastedwith
essentialconsiderationforstrategyvalidationandtheselec- ’K-Fold’and’Purged K-Fold’, whichshowedhighermedian
tionofrobustcross-validationmethodologies. values, indicatingreducedefficiencyandpotentialvariabil-
ityin DSRperformance.
4. Discussion Thetemporalstationarityanalysisof PBO, usingthe Aug-
mented Dickey-Fuller (ADF) test, revealed that the ’Walk-
In assessing backtest overfitting, we observed notable
Forward’methodexhibitedlessstationarity, indicatingagreater
disparities across various cross-validation techniques. The
presenceoftrendsinits PBOovertime. Othermethods, in-
’Walk-Forward’approachexhibitedthehighest Probability
cluding ’Combinatorial Purged’, displayed more consistent
of Backtest Overfitting (PBO), signalingaheightenedriskof
stationaritylevels, suggestingmorereliableperformance. In
overfitting. Incontrast, the’Combinatorial Purged’method
assessing the stationarity of the DSR Test Statistic values,
significantlyoutperformedotherslike’K-Fold’and’Purged
’Walk-Forward’ demonstrated weaker stationarity, as indi-
K-Fold’, demonstratingitseffectivenessinreducingoverfit-
catedbyitshighermedian ADFvalue. Thiscontrastedwith
ting risks. The Deflated Sharpe Ratio (DSR) Test Statistic
othermethods, whichshowedstrongerindicationsofstation-
evaluationhighlighteddistinctperformancevariationsamong
arity, implyingamorestableandconsistentrejectionofunit
themethods. ’Walk-Forward’showedamarkedlylowerme-
rootintheir DSRvaluesovertime.
dian DSR, suggesting heightened false discovery probabil-
Our Principal Component Analysis (PCA) on the cor-
ity. Incomparison,’Combinatorial Purged’alignedclosely
relationbetweendifferentoverfittingmetricsacrosssimula-
with ’K-Fold’ and ’Purged K-Fold’, indicating a more bal-
tions highlighted a unique pattern for the ’Walk-Forward’
ancedapproachinachievingoptimalperformancewhilemit-
method, characterized by a higher explained variance with
igatingoverfitting.
fewer principal components. This pattern suggests a lower
Our analysis of the Efficiency Ratio for the Probability
level of result independence, indicating potential biases or
Arian, Norouzi, Seco Page 25 of 26
Electronic copy available at: https://ssrn.com/abstract=4686376


<!-- Page 26 -->

Backtest Overfitting in the Machine Learning Era
systemicinfluencesinthe’Walk-Forward’method. markettrading. The Journalof Business,39(1):226–241,1966.ISSN
00219398,15375374.
[12] Evelyn Fixand Joseph Lawson Hodges. Nonparametricdiscrimina-
5. Conclusions tion: consistencyproperties. Randolph Field, Texas, Project, pages
21–49,1951.
Ourinvestigationintocross-validationmethodologiesin
[13] James D.Hamilton. Time Series Analysis. Princeton University Press,
financialmodelinghasrevealedcriticalinsights, especially 1994.
thesuperiorityofthe’Combinatorial Purged’methodinmin- [14] Floyd BHansonand Zongwu Zhu. Comparisonofmarketparameters
imizing overfitting risks. This method outperforms tradi- forjump-diffusiondistributionsusingmultinomialmaximumlikeli-
hoodestimation. In200443 rd IEEEConferenceon Decisionand
tional approaches like ’K-Fold’, ’Purged K-Fold’, and no-
Control (CDC)(IEEECat. No.04 CH37601), volume4, pages3919–
tably’Walk-Forward’intermsofboththe Probabilityof Back-
3924.IEEE,2004.
test Overfitting (PBO) andthe Deflated Sharpe Ratio (DSR) [15] Steven L.Heston. Aclosed-formsolutionforoptionswithstochastic
Test Statistic. ’Walk-Forward’, in contrast, shows limita- volatilitywithapplicationstobondandcurrencyoptions. The Review
tionsinpreventingfalsediscoveryandexhibitsgreatertem- of Financial Studies,6(2):327–343,1993.
[16] Ulrich Hommand Jörg Breitung. Testingforspeculativebubblesin
poral variability and weaker stationarity from temporal as-
stockmarkets: acomparisonofalternativemethods. Journalof Fi-
sessmentofthesemethodologiesusingthe Efficiency Ratio
nancial Econometrics,10(1):198–231,2012.
and the Augmented Dickey-Fuller (ADF) test, raising con- [17] Kin Lamand HCYam. Cusumtechniquesfortechnicaltradingin
cerns about its reliability. On the other hand, ’Combinato- financialmarkets. Financial Engineeringandthe Japanese Markets,
rial Purged’demonstratesenhancedstabilityandefficiency, 4:257–274,1997.
[18] Marcos Lopez De Prado. Thefutureofempiricalfinance. Journalof
proving to be a more reliable choice for financial strategy
Portfolio Management,41(4),2015.
development. Thechoicebetween’Purged K-Fold’and’K-
[19] Marcos Lopezde Prado. Advancesinfinancialmachinelearning.
Fold’ requires caution, as they show no significant perfor- John Wiley&Sons,2018.
mance difference, and ’Purged K-Fold’ may reduce the ro- [20] Marcos Lopezde Prado. Machinelearningforassetmanagers. Cam-
bustness of training data for out-of-sample testing. These bridge University Press,2020.
[21] Robert C.Merton. Optionpricingwhenunderlyingstockreturnsare
findingssignificantlycontributetoquantitativefinance, pro-
discontinuous. Journalof Financial Economics,3(1):125–144,1976.
viding a robust framework for cross-validation that aligns
[22] Andrew Papanicolaouand Ronnie Sircar. Aregime-switchingheston
theoretical robustness with practical reliability. They un- modelforvixands&p500 impliedvolatilities. Quantitative Finance,
derscoretheneedfortailoredevaluationmethodsinanera 14(10):1811–1827,2014.
ofcomplexalgorithmsandlargedatasets, guidingdecision- [23] Michael Schatzand Didier Sornette. Inefficientbubblesandefficient
drawdownsinfinancialmarkets. International Journalof Theoretical
making in a data-driven financial world. Future research
and Applied Finance,23(07):2050047,2020.
shouldextendthesefindingstoreal-worldmarketconditions
[24] Laerd Statistics. Kruskal-wallishtestusingspssstatistics. Statistical
toenhancetheirapplicabilityandgeneralizability. tutorialsandsoftwareguides,2015.
[25] Yurong Xieand Guohe Deng. Vulnerableeuropeanoptionpricingin
amarkovregime-switchinghestonmodelwithstochasticinterestrate.
References Chaos, Solitons&Fractals,156:111896,2022.
[1] David HBaileyand Marcos Lopezde Prado. Thesharperatioefficient
frontier. Journalof Risk,15(2):13,2012.
[2] David HBaileyand Marcos Lópezde Prado. Thedeflatedsharpe
ratio: Correcting for selection bias, backtest overfitting and non-
normality. Journalof Portfolio Management,40(5):94–107,2014 b.
[3] David HBailey, Jonathan MBorwein, Marcos Lópezde Prado, and
Qiji Jim Zhu. Pseudomathematicsandfinancialcharlatanism: The
effectsofbacktestoverfittingonout-of-sampleperformance. Notices
ofthe AMS,61(5):458–471,2014 a.
[4] David H Bailey, Jonathan Borwein, Marcos Lopez de Prado, and
Qiji Jim Zhu. Theprobabilityofbacktestoverfitting. Journalof Com-
putational Finance, forthcoming,2016.
[5] Carlo Bonferroni. Teoriastatisticadelleclassiecalcolodelleproba-
bilita. Pubblicazionidel RIstituto Superioredi Scienze Economiche
e Commericialidi Firenze,8:3–62,1936.
[6] Leo Breiman. Classificationandregressiontrees. Routledge,2017.
[7] Tianqi Chenand Carlos Guestrin. Xgboost:Ascalabletreeboosting
system. In Proceedingsofthe22 ndacmsigkddinternationalconfer-
enceonknowledgediscoveryanddatamining, pages785–794,2016.
[8] Kim Christensen, Roel Oomen, and Roberto Renò. Thedriftburst
hypothesis. Journalof Econometrics,227(2):461–497,2022. ISSN
0304-4076.
[9] Olive Jean Dunn. Multiplecomparisonsusingranksums. Techno-
metrics,6(3):241–252,1964.
[10] Robert JElliott, Katsumasa Nishide, and Carlton-James UOsakwe.
Heston-type stochastic volatility with a markov switching regime.
Journalof Futures Markets,36(9):902–919,2016.
[11] Eugene F. Fama and Marshall E. Blume. Filter rules and stock-
Arian, Norouzi, Seco Page 26 of 26
Electronic copy available at: https://ssrn.com/abstract=4686376
