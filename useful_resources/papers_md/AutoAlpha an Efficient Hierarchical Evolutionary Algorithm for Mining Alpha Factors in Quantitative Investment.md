<!-- Page 1 -->

Auto Alpha: an Efficient Hierarchical Evolutionary Algorithm for Mining Alpha
Factors in Quantitative Investment
Tianping Zhang, Yuanqi Li, Yifei Jin, Jian Li
Institutefor Interdisciplinary Information Sciences (IIIS), Tsinghua University, China
ztp18@mails.tsinghua.edu.cn,{timezerolyq, yfjin1990}@gmail.com,
Abstract turns1. In quantitative trading practice, designing novel fac-
tors that can explain and predict future asset returns are of
The multi-factor model is a widely used model in
vitalimportancetotheprofitabilityofastrategy. Suchfactors
quantitative investment. The success of a multi-
areusuallycalledalphafactors, oralphasinshort.
factormodelislargelydeterminedbytheeffective-
In 2016, the quantitative investment management firm,
ness of the alpha factors used in the model. This
World Quant, made public 101 formulaic alpha factors in
paperproposesanewevolutionaryalgorithmcalled
[Kakushadze, 2016]. Since then, many quantitative trading
Auto Alpha to automatically generate effective for- methodshaveusedtheseformulaicalphasforstocktrendpre-
mulaicalphasfrommassivestockdatasets. Specif-
diction [Chen et al., 2019]. A formulaic alpha, as the name
ically, first we discover an inherent pattern of the
suggests, isakindofalphathatcanbepresentedasaformula
formulaic alphas and propose a hierarchical struc-
oramathematicalexpression.
ture to quickly locate the promising part of space
for search. Then we propose a new Quality Di-
Alpha#101=(close−open)/(high−low)
versity search based on the Principal Component
Analysis (PCA-QD) toguidethesearchawayfrom For example, the above formulaic alpha is one of the al-
thewell-exploredspaceformoredesirableresults. phafactorsfrom[Kakushadze,2016]andiscalculatedusing
Next, weutilizethewarmstartmethodandthere- theopenprice, thecloseprice, thehighestpriceandthelow-
placementmethodtopreventtheprematureconver- est price of stocks on each trading day. This alpha formula
genceproblem. Basedontheformulaicalphaswe reflects the momentum effect that has been observed in dif-
discover, weproposeanensemblelearning-to-rank ferent market (see e.g., [Jegadeesh and Titman, 1993]). For
modelforgeneratingtheportfolio. Thebacktestsin eachday, thealphagivesdifferentvaluesfordifferentstocks.
the Chinesestockmarketandthecomparisonswith Thehigherthevalue, itismorelikelythatthestockwillhave
severalbaselinesfurtherdemonstratetheeffective- relativelylargerreturnsinthefollowingdays.
ness of Auto Alpha in mining formulaic alphas for There are much more complicated formulaic alphas than
quantitativetrading. the one shown above. In Figure 2, we show two examples,
Alpha#71 and Alpha#72 in [Kakushadze, 2016]. The most
common way of producing new formulaic alphas is to have
1 Introduction
economistsorfinancialengineerstocomeupwithneweco-
Predictingthefuturereturnsofstocksisoneofthemostchal- nomical ideas, transform these ideas into formulas and then
lenging tasks in quantitative trading. Stock prices are af- validateitseffectivenessonthehistoricalstockdatasets. Itis
fected by many factors such as company performances, in- knownthat World Quanthasbeenemployingalargenumber
vestors’sentiment, andnewgovernmentpolicies, etc. Toex- offinancialengineersanddataminers (evenpart-timeonline
plain the fluctuation of stock markets, economists have es- users2) to design new alphas. This way of finding good al-
tablishedseveraltheoreticalmodels. Amongthemostpromi- phas requires tremendous human labor and expertise, which
nentones, the Capital Asset Pricing Model (CAPM)[Sharpe,
isnotrealisticforsmallfirmsorindividualinvestors. There-
1964] dictates that the expected return of a financial asset is fore, thereisanurgentneedtodeveloptoolsforminingnew
essentially determined by one factor, that is the market ex- effectivealphasfrommassivestockdatasetsautomatically.
cessreturn, whilethe Arbitrage Pricing Theory (APT)[Ross,
Ourgoalistofindasmanydiverseformulaicalphaswith
2013] models the return by a linear combination of differ- desirable performance as possible within limited computa-
entriskfactors. Sincethen, severalmulti-factormodelshave tionalresources. Unlikemanyoptimizationandsearchprob-
beenproposedandnumeroussuchfactors (alsocalledabnor- lems which aim at finding one single desirable solution, we
mal returns) have been found in the economics and finance
literature. For example, the celebrated Fama-French Three 1 https://en.wikipedia.org/wiki/Fama%E2%80%93 French three-
Factor Model[Famaand French,1993]discoveredthreeim- factor model
portant factors that can explain almost 90% of the stock re- 2 https://www.weareworldquant.com/en/home
0202
rp A
4
]PC.nif-q[
2 v54280.2002:vi Xra


<!-- Page 2 -->

operators factors QD) toguidethesearchawayfromtheexploredspacefor
+ - * / min max open close high Backtests morediverseandeffectiveformulaicalphas.
std mean tsrank low vwap volume
..... ..... • We introduce the warm start method at the initialization
/ step and the replacement method during reproduction to
Stock Ranking
— — prevent the premature convergence problem. This ad-
Auto Alpha Prediction
c o h l dresses Challenge3.
warm start method formula: (close-open) / (high-low)
Basedontheformulaicalphaswediscover, weproposean
replacement method Formulaic Prediction
PCA-QD depth+1 Alphas Model ensemble learning-to-rank model to predict the stocks rank-
ings and develop effective stock trading strategies. We per-
formbacktestsinthe Chinesestockmarketfordifferenthold-
Figure1:Theframeworkofourapproach.
ing periods. The backtesting results show that our method
consistentlyoutperformsseveralbaselinesandthemarketin-
dex.
2 Problem Statement
Miningformulaicalphascanberegardedasafeatureextrac-
tionproblem. Westartfromaninitialsetofbasicfactors (e.g.
Figure2:Formulas Alpha#71 and Alpha#72. open, close, volume, etc.) andoperators (e.g. +-*/, min, std,
etc.), andthenbuildformulaicalphasthatsatisfycertainper-
formancemeasurementcriterion, inordertorevealsomein-
prefertolookformultiplediversesolutionswithhighperfor-
herentpatternsofthestockmarket. Thebasicfactorsandthe
manceandlowcorrelation.
operators we use can be found in [Kakushadze, 2016]. The
Asshownin Figure3, aformulaicalphacanbeexpressed
data ispublic in Chinese stockmarkets and can beaccessed
asatreewheretheleavescorrespondtorawdataandtheinner
throughmultipleresources3. Inthissection, weformalizethe
nodescorrespondtovariousoperators. Asthediscretesearch
problemofminingformulaicalphas.
space is very large, it is natural to use genetic algorithms to
search for effective alphas in the form of trees. However,
2.1 Stock Returns
as we argue below, this is not straightforward and there are
severalchallengesweneedtoaddress. The return of a stock is generally determined by the close
Challenge 1: Quickly locate the promising search priceofthestockandtheholdingperiod. Foragivenstocks,
space. The vanilla genetic algorithm is generally inefficient agivendatetandagivenholdingperiodh, thereturnofthe
in mining effective formulaic alphas, due to the fact that ef- stockcanbecalculatedas:
fectivealphasaresparseinthehugesearchspace. Therefore, close −close
r (h) = t+h, s t, s
howtoquicklylocatethepromisingspaceforsearchbecomes t, s close
t, s
acriticalissue.
Challenge2: Guidethesearchawayfromtheexplored where close t, s is the close price of stock s at date t. As-
search space. In order to find many diverse and effective suming that there are n different stocks in the stock pool,
formulaicalphas, weneedtorunthegeneticalgorithmseveral the return vector at date t of holding period h is denoted as
timesformoreresults. However, thevanillageneticalgorithm r (h) =(r (h),..., r (h)).
t t,1 t, n
usuallyconvergestothesamelocalminima.
Challenge3: Preventtheprematureconvergenceprob- 2.2 Evaluation Metrics
lem. Theprematureconvergenceproblem[Guptaand Ghafir,
Foragivenformulaicalphai, itsvalueforthestocksatdate
2012]arisesingeneticalgorithmswhensometypeofeffec-
tivegenesdominatethewholepopulationanddestroythedi- t is defined as a t ( , i s ). We use the IC (information coefficient)
versityinthepopulation. Whenprematureconvergencehap- [Grinoldand Kahn,2000]toevaluatetheeffectivenessofan
pens, thepopulationstucksatasuboptimalstateandwecan alpha in the mining process. For a given formulaic alpha i
nolongerproduceoffspringwithhigherperformance. andagivenholdingperiodh, the ICcanbecalculatedasthe
In this paper, we propose a new model called Auto Alpha meanofthe ICarray:
toaddresstheabovechallengesinaunifiedframework. The T
technical contributions of this paper can be summarized as IC = 1 (cid:88) corr (a (i), r (h)) (1)
i T t t
follows: t=1
• We discover an inherent pattern of the formulaic alphas. wherea (i) = (a (i),..., a (i)) isthevaluevectorofformu-
t t,1 t, n
Basedonthis, wedesignahierarchicalstructuretoquickly laicalphaiatdatet, corr isthesample Pearson Correlation,
locatethepromisingspaceforsearch, whichaddress Chal- {corr (a (i), r (h))}T isthe ICarrayand Tisthenumberof
lenge1. t t t=1
tradingdaysinthetrainingperiod.
• For Challenge 2, we propose a new Quality Diversity
methodbasedonthe Principal Component Analysis (PCA- 3 http://tushare.org/#


<!-- Page 3 -->

root
operator /
gene2 gene3
op1 child1 / child2 op1
200
— — crossover op2 op3 — op2 — op3 175
gene1 150 125 c o h l f 1 f 2 c o f 1 h l f 2 100
75 formula: (close-open) / (high-low) 50
25
Figure 3: The demonstration of crossover. The leftmost tree
shows the tree representation of the formulaic alpha ’(close −
00.00 0.01 0.02 0.03 , &0.04 0.05 0.06 0.07
open)/(high−low)’. The trees on the right are two children af-
tercrossover.op:operator.f:factor.
The ICofanalphaindicatestherelevancebetweentheal-
phaandthestockreturns, andshouldbeashighaspossible4.
2.3 Similaritybetween Alphas
Thesimilaritybetweenthealphaiandjiscalculatedas:
T
sim (i, j)= 1 (cid:88) corr (a (i), a (j))
T t t
t=1
Agroupofalphasisdiverseifthesimilaritybetweenanytwo
alphasinthegroupislowerthan0.7.
In the process of mining formulaic alphas, our goal is to
findasmanydiverseformulaicalphaswithhigh ICaspossi-
blewithinlimitedcomputationalresources.
3 Auto Alpha
Auto Alphaisaframeworkbasedongeneticalgorithms[Whit-
ley, 1994]. Genetic algorithm is a kind of metaheuristic op-
timizationalgorithmwhichdrawsinspirationfrombiological
processthatproducesnewoffspringandevolveaspecies. The
vanillageneticalgorithmusesmechanismssuchasreproduc-
tion, crossover, mutation and selection to give birth to new
offspring. In each step of regeneration, it uses fitness func-
tion to select the best-fit individuals for reproduction. After
we give birth to new offspring through crossover and mu-
tation operations, we replace the least-fit individuals in the
populationwithnewindividualstorealizethemechanismof
eliminationthroughcompetition.
Inordertoapplythegeneticalgorithmforminingformu-
laicalphas, firstweneedtodefinethegeneticrepresentation
ofaformulaicalpha. Asshownintheleftmosttreeof Figure
3, aformulaicalphacanberepresentedasaformulaictree. It
would be much easier for us to carry out crossover and mu-
tation for trees. Figure 3 shows the crossover between two
formulaicalphasofdepth2. Weperformthecrossoverinthe
same depth level to prevent the depth from increasing. That
is, thecrossoverbetweengene1 andgene3 in Figure3 isnot
allowed. Thegene2 andgene3 arecalledrootgeneswhichare
directlyattachedtotheirrootoperatorswhilegene1 isnot.
3.1 Hierarchical Structure
Thesearchspaceoftreesishugeandtheeffectivealphasare
verysparse. Inourexperiment, wefindoutthatthestandard
geneticalgorithm (e.g., thatimplementedinpythonpackage
4 W.l.o.g., weassumethat IC ≥ 0 sincewecanmultiply−1 to
aformulaicalphawhichhasnegative IC.
 \ W L V Q H G
15 14
13 12
11 10 9 8 7
6 5 4 3
2
1
0
 \ F Q H X T H U I
250
200 150
100 50
00.00 0.01 0.02 0.03 , &0.04 0.05 0.06 0.07
(a) depth=2
 \ W L V Q H G
15 14
13 12
11 10 9 8 7
6 5 4 3
2
1
0
 \ F Q H X T H U I
(b) depth=3
Figure4:The ICoftherootgenesofthetop100 discoveredformu-
laicalphasanditsdistribution. Forexample, intheleftfigure, the
bluehistogramisthedensityplotof ICoftheformulaicalphasof
depth2(estimatedby20000 randomlygeneratedsamples).Andthe
redhistogramisthefrequencyplotof ICoftherootgenesofthetop
100 discoveredformulaicalphasofdepth3.
’gplearn’) is generally inefficient in initializing the popula-
tionandexploringthesearchspaceforminingformulaical-
phas (seetheresultsin Section4.4). Forremedy, Wepropose
anovelhierarchicalsearchstrategyforthegeneticalgorithm,
thatissignificantlymoreefficientintheinitializationandex-
plorationofthesearchspace.
Motivation
Intheearlystageofthisresearch, wehavebeenusingvanilla
geneticalgorithmsforminingformulaicalphas. Aninterest-
ing phenomenon occurs during the experiments that the al-
gorithm usually converges to similar formulaic alphas with
’vwap/close’ as a piece of its genes. ’vwap’ is the Volume
Weighted Average Priceofastock. Thegenevwap/closeit-
selfisalsoaneffectivealphawhichrelatestothephenomenon
of mean reversion. While vwap/close itself is an effective
formulaicalpha, theformulaicalphasofhigherdepthwhich
contain vwap/close as a piece of its genes usually combine
thismeanreversioninformationwithsomeotherinformation
andhavehighereffectiveness.
Based on such phenomenon, we propose a hypothesis
about the inherent pattern of the formulaic alphas, that is,
most of the effective alphas have at least one effective root
gene. Intuitively, ifwewanttoobtaintheformulaicalphasof
higherdepth, weshouldsearchnearbytheeffectivealphasof
lowerdepth. Wedesignanexperimenttoverifiesthehypoth-
esis. First, weusethevanillageneticalgorithmforevolving
formulaicalphas. Thenweselectthetop100 discoveredfor-
mulaicalphas. Foreachselectedalpha, wefurthercollectits
root gene with highest IC. We use the density plot to show
that those root genes are effective and are hard to obtain by
randomgeneration. Theresultsareshowninthe Figure4.
Basedontheanalysis, ifwemaintainapopulationwithdi-
verseandeffectiverootgenesinit, thecrossoveroperationat-
temptstosearchingnearbyeffectiveformulasoflowerdepth,
and thus improve the efficiency in obtaining diverse and ef-
fective alphas. In order to establish a diverse and effective
gene pool for initializing the population, we use the hierar-
chicalstructureastheframeworkfor Auto Alphaandgenerate
alphasfromlowertohigherdepthiteratively.


<!-- Page 4 -->

3.2 Guide The Search the genetic diversity. When premature happens, the whole
populationgetsstuckearlyinabadlocalminima. Thereare
Now the problem has turned to how to generate formulaic
many methods addressing the premature convergence prob-
alphas of each depth. Unlike many optimization and search
lem[Guptaand Ghafir, 2012], andweselecttwofromthem
problemswhichaimatfindingonesingledesirablesolution,
inparticularfor Auto Alpha.
ourgoalistolookforasmanyformulaicalphaswithhigh IC
Warm Start. Intheinitializationstep, insteadofrandomly
andlowcorrelationaspossible. Ifwehavealreadyobtaineda
generatingindividualstothesizeofthepopulation, wegen-
groupofformulaicalphas, wewouldliketoguidethesearch
erateindividuals K timesthesizeofthepopulationandthen
intounexploredspaceformorealphas.
select the individuals which rank at top 1 according to IC
However, thegeneticalgorithmsusuallyconvergeintothe K
intothepopulationasinitialization. Inthisway, weimprove
same local minima. In genetic algorithms community, one
the average effectiveness of the initialized individuals, and
methodtotacklesuchproblemisthe Quality Diversity (QD)
thusacceleratetheevolution. Thewarmstartmethodin Au-
search [Pugh et al., 2016], which seeks to find a maximally
to Alpha helps us filter out those genes that are not useful in
diverse collection of individuals where each individual has
constructingformulaicalphasofhigherdepth.
desirable performance. If we can calculate the similarity
Replacement Method. In the reproduction step, instead
(or distance) between individuals, then we can guide the
ofcomparingthenewindividualswiththeleast-fitindividu-
search by changing the objective landscape through penalty
als in the population, we compare the new individuals with
[Lehman and Stanley, 2011]. For example, if the similarity
their own parents. A pair of parents has two offspring af-
between the new alpha and any of the alphas in the record
tercrossover, andthetwooffspringcanreplacetheirparents
exceedsacertainthreshold, thenthefitnessofthenewalpha
whenthebestoftheirfitnessisgreaterthanthatoftheirpar-
ispenalizedtobe0(least-fit). However, calculatingthesim-
ents. In this way, all the genes in the population only have
ilaritiesisslow, especiallywhenthesizeoftherecordgrows
onepieceofcopyafterreproduction, whichhelpsusprevent
large. Assumingthatthesizeoftherecordisp, thencalculat-
theprematureconvergenceproblem.
ingthesimilaritiesbetweenanewalphaandthealphasinthe
recordis O(np T) where T isthenumberoftradingdaysand 3.4 Overall Algorithmof Auto Alpha
nisthenumberofstocks.
1. Weenumeratethealphasofdepth1 andselecttheeffective
In order to reduce the time complexity, we find a simpler
onestosetupthegenepool.
way to approximate the similarity and design our PCA-QD
search. Firstly, weusethefirstprincipalcomponentvectorto 2. We use the warm start method and the gene pool to ini-
represent the information of a formulaic alpha. Specifically, tialize the population of depth 2. Then we use crossover
thevaluesoftheformulaicalphaicanbepresentedasasam- and the replacement method for reproduction. If the new
plematrix A(i) =(a (i)) , whereeachcolumn (stock) can offspringhashigher ICthanitsparents, wewillcalculate
t, s T×n its PCA-similaritywiththealphasintherecordanddeter-
be treated as a feature and each row (date) can be treated as
minewhethertokeepitinthepopulation
a sample. Then we can calculate the first principal compo-
nentofthesamplematrix. Next, weusethe Pearson Correla- 3. We repeat step 2 for more alphas of depth 2 and update
tionbetweenthefirstprincipalcomponentsofthetwoalphas, thegenepoolandtherecord. Thenwecanstartgenerating
which we call PCA-similarity, to approximate the similarity alphasofdepth3 andsoforth.
betweenthetwoalphas. Thecalculationofthefirstprincipal We use the formulaic alphas as input features and we
componentusingpowermethodis O(n T +n2). Inthisway, use Light GBM [Ke et al., 2017] and XGBoost [Chen and
wereducethecomputationalcomplexityofcalculatingsimi- Guestrin, 2016] as learning algorithms to learn-to-rank the
laritiesfrom O(np T) to O(p T)(inexperiments, nis300 and stocks. Thenweensembletheresultsanduseitforstockin-
wehaven<T andn<p). vestment in the testing period. Due to space limit, we omit
We run the experiments to see how the approximation thedetailsofthetrainingandthechoiceofhyper-parameters.
method works by random sampling. The threshold for the
PCA-similarity is set to be 0.9. When the similarity is over 4 Experiments
0.7, the Mean Absolute Error (MAE) between the PCA-
4.1 Experimental Settings
similarity and the similarity is 0.092. And when the PCA-
similarity is over 0.9, the MAE is 0.125. Since we are us- Tasks. We conduct the experiments independently for the
ing the PCA-similarity instead of the similarity to guide the holdingperiodsof1 dayand5 days. Firstweuse Auto Alpha
search away from the explored area, we hope that this ap- togenerateagroupofeffectiveformulaicalphasforthegiven
proximation should be accurate when the similarity is high holding period. Then we use the ensemble model to learn
andwhenthe PCA-similarityishighaswell. to rank the stocks. Finally we use the predicted rankings to
construct the stock portfolio each day and provide backtests
3.3 Preventionof Premature Convergence toevaluatetheeffectivenessofthealphaswegenerate.
Datasets. We use the 300 stocks in the CSI 300 Index
Theprematureconvergenceproblemhasalwaysbeenacriti-
(hs300)5 as the stock pool when mining the formulaic al-
calissueinthegeneticalgorithms[Guptaand Ghafir,2012].
If we always replace the least-fit individuals with new indi- 5 CSI300 isacapitalization-weightedstockmarketindexrepli-
vidualsofgreaterfitness, itislikelythatsometypeofeffec- catingtheperformanceoftop300 stockstradedinthe Shanghaiand
tive genes may dominate the whole population and destroy Shenzhenstockexchanges.


<!-- Page 5 -->

Table1: Performanceofthetop5 formulaicalphasforholdingpe- • gplearn: ’gplearn’isapopularpythonpackagewhichper-
riod h = 1,5 in both training and testing period. The values out
forms the standard genetic algorithm. It can also be used
of/inthebracketsaretheresultsinthetraining/testingperiod.
togenerateformulaicalphas.
• SFM: [Zhang et al., 2017] proposed SFM (the State Fre-
h=1 h=5
Top-k Alpha
quency Memoryrecurrentnetwork) whichisanend-to-end
hs300 zz800 hs300 zz800
deeplearningmethodandappliedittothestockprediction
Top1 8.36%(7.10%) 8.41%(7.47%) 8.47%(6.15%) 8.82%(5.53%)
Top2 8.30%(6.07%) 8.17%(3.28%) 7.87%(6.52%) 7.80%(5.46%) tasks.
Top3 7.84%(5.55%) 7.24%(5.31%) 7.69%(5.61%) 7.90%(4.48%)
• Market: Themarketisrepresentedbythe CSI800 Index.
Top4 7.84%(6.64%) 7.64%(6.79%) 7.67%(7.30%) 7.88%(6.43%)
Top5 7.74%(6.66%) 8.02%(5.98%) 7.50%(5.53%) 7.40%(4.35%) Weshowthatourmethodisabletooutperformthemarket
signficantly.
Table 2: Comparison with gplearn and Alpha101. n: number of
diverseformulaicalphaswith IChigherthan0.05.avg IC:average 4.4 Resultsof Generated Formulaic Alphas
ICofthetop50 discoveredformulaicalphas. Table1 showsthe ICofthetop5 generatedformulaicalphas.
Weshowthatthealphascannotonlygeneralizeinthetesting
h=1 h=5 period, butalsogeneralizeinthedifferentstockpoolof CSI
Method
n avg IC n avg IC 800 Index. Table2 showsthecomparisonbetweenthealphas
generatedby Auto Alpha, thealphasgeneratedbygplearnand
Alpha101 0 1.02% 0 1.25%
the 101 alphas in [Kakushadze, 2016]. We use two metrics
gplearn 35 6.10% 7 3.35%
tocomparetheresultsfrombothquantitativeandqualitative
Auto Alpha 434 7.50% 415 6.71%
aspects. Oneisthenumberofdiverseformulaicalphaswith
IC higher than 0.05. Another is the average IC of the top
phasinthetrainingstage. Weperformbacktestsonthestock 50 diverse formulaic alphas. We use the stratified backtests
poolof CSI800 Index (zz800)6. Theoveralltestingperiodis to show the alphas’ capability in ranking stocks. The stock
from 20170901 to 20190731 and the training period is from poolisdividedequallyinto10 foldeachdayaccordingtothe
20100101 to20170831. ratingsthealphagivesandfold9 hasthestockswithhighest
ratings. Then the i-th strategy always buy the stocks in the
4.2 Evaluation Metrics i-thfoldeachday. Theresultsareshownin Figure5.
Wemeasuretheperformanceofthetradingstrategiesbystan-
dardmetricsintheliteraturesuchasannualizedreturn (AR), 2.0
fold_0 fold_6 fold_0 fold_6
annualizedvolatilityand Sharperatio (SR). 1.8 f f o o l l d d _ _ 1 2 f f o o l l d d _ _ 7 8 1.6 f f o o l l d d _ _ 1 2 f f o o l l d d _ _ 7 8
1.6 fold_3 fold_9 fold_3 fold_9
Annualizedreturn (AR).Annualizedreturncalculatesthe 1.4 f f o o l l d d _ _ 4 5 market 1.4 f f o o l l d d _ _ 4 5 market
rate of return for a given holding period scaled down to a 1.2 1.2
(cid:110) (cid:111) 1.0 1.0
12-monthperiod: AR=exp 365/T(cid:48) ×log (S T /S 0 ) −1, 0.8 0.8
0.6
where T(cid:48) isthenumberofdaysand T isthenumberoftrading 0.4 0.6
2017-102018-012018-042018-072018-102019-012019-042019-07 2017-102018-012018-042018-072018-102019-012019-042019-07
days. S denotesthetotalwealthattheendoft-thtradingday. (a) Alpha1 (b) Alpha2
t
Sharperatio (SR).Infinance, the Sharperatioisapopular
m s r a e t n r t e n a u a t u r s e n a u g l r o y i e z f : e o t d S h f R e v t o h p = l e o a r t r t ( i f i l R s o it k l p y i - , o a − , d w R j R h u f i s f c te h i ) s / d i σ t s h p p e e t , h r r w f e i o s h k r s e m t - a r f e a n re n d R e c a e r p r d a o i t s e d f 7 e t , a h v n a e ia n i a t d n i n o v σ n n e u p s o a t i m l f s iz t t e h h e n d e e t 1 1 1 1 . . . . 2 4 6 8 f f f f f f o o o o o o l l l l l l d d d d d d _ _ _ _ _ _ 0 1 2 3 4 5 f f f f m o o o o l l l l a d d d d r _ _ _ _ k 6 7 8 9 et 1 1 2 2 2 . . . . . 5 7 0 2 5 0 5 0 5 0 f f f f f f o o o o o o l l l l l l d d d d d d _ _ _ _ _ _ 0 1 2 3 4 5 f f f f m o o o o l l l l a d d d d r _ _ _ _ k 6 7 8 9 et
1.0 1.25
strategy’syearlylogarithmicreturns8.
0.8 1.00
0.6 0.75
4.3 Baselinesfor Comparison 0.4 0.50
2017-102018-012018-042018-072018-102019-012019-042019-07 2017-102018-012018-042018-072018-102019-012019-042019-07
(c) Alpha3 (d) Alpha4
To further demonstrate the effectiveness of our method, we
compareitwiththefollowingbaselines:
Figure5:Theresultsofstratifiedbacktestsofthetop4 alphas (h=1).
• Alpha101: We compare the formulaic alphas we au-
tomatically discover with the 101 formulaic alphas in
[Kakushadze, 2016]. We use the 101 alphas to train the 4.5 Backtesting Results
modelsandperformbacktestsunderthesamesettings.
Foreachholdingperiod, wecollectthetop150 formulaical-
phasgeneratedby Auto Alphawhichhavethehighest ICinthe
6 CSI800 Indexiscomprisedofalltheconstituentsof CSI300
training period for model training. Then we use the trained
Indexand CSI500 Index. Itisdesignedtoreflecttheoverallperfor-
modeltopredictthestockrankingsinthetestingperiod. We
manceofthelarge, middleandsmallcapitalstocksin Chinesestock
markets. ensuretheoverallproceduredoesnotusefutureinformation
7 Wesettherisk-freerateintheexperimentstobe0 forsimplicity, thatisnotavailableatthetradingtime. Weperformbacktests
sincethereisnoconsensusonthevalueofrisk-freerate. using historical data for holding periods h = 1 and 5. For
8 https://en.wikipedia.org/wiki/Volatility (finance) a given holding period h, the investor invests in the top 10


<!-- Page 6 -->

Table3: Resultsofbacktestsandcomparisonswithbaselines. Re-
35000 market:zz800
sultsrelativetothemarketareinthebrackets.
top10
30000 top20
top30
Method h=1 h=5 25000 top40
AR SR AR SR 20000 top50
market -4.1% -0.20 -6.4% -0.31 15000
SFM -60.0%(-58.5%) -2.05(-2.89%) -23.7%(-17.7%) -1.01(-1.53)
gplearn 61.8%(68.7%) 2.34(4.26) 16.9%(22.6%) 0.72(1.93) 10000
Alpha101 29.5%(35.3%) 1.06(2.02%) 16.3%(23.3%) 0.70(2.09)
2017-09 2017-12 2018-03 2018-06 2018-09 2018-12 2019-03 2019-06 2019-09
ourmethod 90.0%(98.2%) 3.39(6.02) 28.0%(34.0%) 1.20(3.05)
Figure7:Comparisononbacktestingstrategies (longingtop10, top
20,..., top50).Holdingperiodh=1.
stocks at the close price according to the predicted rankings
ineachday. Thenthestocksareheldforhdaysandsoldat 5 Related Work
theclosepriceattheendoftheholdingperiod. Thetransac-
Dataminingandmachinelearningtechniqueshavebeenused
tion cost is 0.3% as accustomed. The stock pool for trading
extensivelytoaddressavarietyofproblemsinfinance. Inpar-
simulationisthe CSI800 Index (zz800). Thebacktestingre-
ticular, ourworkiscloselyrelatedtofeatureextractioninma-
sultsareshowninthe Figure6,7 and Table3.
chinelearning (seee.g.,[Guyonetal.,2008]).Featureextrac-
tionhasbeenwidelyadoptedinfinancialprediction. [Zhang
35000 our method etal.,2018]extractedfeaturesfromthesemanticinformation
30000 market:zz800 instockcommenttextforreliabilitymodelingofstockcom-
alpha101
25000 gplearn ments. [Huang and Wu, 2008] proposed a GA-based model
20000 SFM to extract wavelet features for stock index prediction. Our
15000 work develops a substantially different GA-based algorithm
10000 andextractsgeneralformulaicalphafactors.
5000 Howtomaintaindiversityhasalwaysbeenacriticalissue
0 2017-09 2017-12 2018-03 2018-06 2018-09 2018-12 2019-03 2019-06 2019-09 ingeneticalgorithms[Guptaand Ghafir,2012]. Themethods
(a) holdingperiodh=1 toencouragediversityincludes Nitching[Sareniand Krahen-
buhl, 1998], Crowding[Mahfoud, 1992], Sharing[Goldberg
40000
our method et al., 1987], etc. The replacement method we use derives
35000 alpha101 from the Steady State Genetic Algorithms (SSGAs) [Engel-
30000 gplearn
SFM brecht,2007]. Anumberofreplacementmethodshavebeen
25000
20000 developed for SSGAs, including the parent-offspring com-
15000 petition we use [Smith and Vavak, 1999]. The Quality Di-
10000 versity (QD) is designed to illuminate the diverse individu-
5000 alswithhighperformance. Therearemanyqualitydiversity
0 2017-09 2017-12 2018-03 2018-06 2018-09 2018-12 2019-03 2019-06 2019-09 algorithms designated for different kinds of problems [Pugh
(b) holdingperiodh=1(relativetothemarket) et al., 2016]. Common examples include Novelty Search
with Local Competition (NSLC)[Lehmanand Stanley,2011]
our method and Multi-dimensional Archive of Phenotypic Elites (MAP-
16000 market:zz800
alpha101 Elites)[Mouretand Clune,2015].
gplearn
14000 SFM
12000 6 Conclusions
10000
In this paper, we propose Auto Alpha, an efficient algorithm
8000 that automatically discovers effective and diverse formulaic
6000 alphas for quantitative investment. We first propose a hier-
2017-09 2017-12 2018-03 2018-06 2018-09 2018-12 2019-03 2019-06 2019-09
(c) holdingperiodh=5 archical structure to quickly locate the promising space for
search. Secondly, weproposeanew PCA-QDsearchtoguide
18000 thesearchawayfromtheexploredareas. Thirdly, weutilize
our method
16000 alpha101 the warm start method and the parent-offspring replacement
gplearn
methodtopreventtheprematureconvergenceproblem. The
14000 SFM
backtestsandcomparisonswithseveralbaselinesshowtheef-
12000
fectivenessofourmethod. Finally, weremarkthat Auto Alpha
10000 canbealsoviewedasanapproachforautomaticfeatureex-
8000 traction. As the market becomes more efficient, discovering
alpha factors becomes more difficult and automatically ex-
2017-09 2017-12 2018-03 2018-06 2018-09 2018-12 2019-03 2019-06 2019-09
(d) holdingperiodh=5(relativetothemarket) tracting effective features is a promising future direction for
quantitativeinvestment.
Figure6:The Profit&Lossgraphofbacktests.


<!-- Page 7 -->

References [Mouretand Clune,2015] Jean-Baptiste Mouret and Jeff
Clune. Illuminating search spaces by mapping elites.
[Chenand Guestrin,2016] Tianqi Chenand Carlos Guestrin.
ar Xivpreprintar Xiv:1504.04909,2015.
Xgboost: Ascalabletreeboostingsystem. In Proceedings
ofthe22 ndacmsigkddinternationalconferenceonknowl- [Pughetal.,2016] Justin K Pugh, Lisa B Soros, and Ken-
edge discovery and data mining, pages 785–794. ACM, neth OStanley. Qualitydiversity: Anewfrontierforevo-
2016. lutionarycomputation. Frontiersin Roboticsand AI,3:40,
2016.
[Chenetal.,2019] Chi Chen, Li Zhao, Jiang Bian, Chunxiao
Xing, and Tie-Yan Liu. Investmentbehaviorscantellwhat [Ross,2013] Stephen ARoss. Thearbitragetheoryofcapi-
inside: Exploringstockintrinsicpropertiesforstocktrend talassetpricing. In HANDBOOKOFTHEFUNDAMEN-
prediction. In Proceedingsofthe25 th ACMSIGKDDIn- TALSOFFINANCIALDECISIONMAKING:Part I, pages
ternational Conference on Knowledge Discovery & Data 11–30.World Scientific,2013.
Mining, pages2376–2384.ACM,2019.
[Sareniand Krahenbuhl,1998] Bruno Sareni and Laurent
[Engelbrecht,2007] Andries PEngelbrecht. Computational Krahenbuhl. Fitness sharing and niching methods revis-
intelligence: anintroduction. John Wiley&Sons,2007. ited. IEEE transactions on Evolutionary Computation,
2(3):97–106,1998.
[Famaand French,1993] Eugene F Fama and Kenneth R
French. Commonriskfactorsinthereturnsonstocksand [Sharpe,1964] William F Sharpe. Capital asset prices: A
bonds. Journaloffinancialeconomics,33(1):3–56,1993. theoryofmarketequilibriumunderconditionsofrisk. The
journaloffinance,19(3):425–442,1964.
[Goldbergetal.,1987] David E Goldberg, Jon Richardson,
et al. Genetic algorithms with sharing for multimodal [Smithand Vavak,1999] Jim E Smith and Frantisek Vavak.
functionoptimization. In Geneticalgorithmsandtheirap- Replacementstrategiesinsteadystategeneticalgorithms:
plications: Proceedingsofthe Second International Con- dynamic environments. Journal of computing and infor-
ference on Genetic Algorithms, pages 41–49. Hillsdale, mationtechnology,7(1):49–59,1999.
NJ:Lawrence Erlbaum,1987. [Whitley,1994] Darrell Whitley. A genetic algorithm tuto-
[Grinoldand Kahn,2000] Richard CGrinoldand Ronald N rial. Statisticsandcomputing,4(2):65–85,1994.
Kahn. Activeportfoliomanagement. 2000. [Zhangetal.,2017] Liheng Zhang, Charu Aggarwal, and
[Guptaand Ghafir,2012] Deepti Gupta and Shabina Ghafir. Guo-Jun Qi. Stock price prediction via discovering
An overview of methods maintaining diversity in genetic multi-frequency trading patterns. In Proceedings of the
algorithms. Internationaljournalofemergingtechnology 23 rd ACM SIGKDD international conference on knowl-
andadvancedengineering,2(5):56–60,2012. edgediscoveryanddatamining, pages2141–2149.ACM,
2017.
[Guyonetal.,2008] Isabelle Guyon, Steve Gunn, Masoud
Nikravesh, and Lofti A Zadeh. Feature extraction: foun- [Zhangetal.,2018] Chen Zhang, Yijun Wang, Can Chen,
dationsandapplications, volume207. Springer,2008. Changying Du, Hongzhi Yin, and Hao Wang. Stockassis-
tant: Astockaiassistantforreliabilitymodelingofstock
[Huangand Wu,2008] Shian-Chang Huang and Tung-
comments. In Proceedingsofthe24 th ACMSIGKDDIn-
Kuang Wu. Integrating ga-based time-scale feature
ternational Conference on Knowledge Discovery & Data
extractionswithsvmsforstockindexforecasting. Expert
Mining, pages2710–2719.ACM,2018.
Systemswith Applications,35(4):2080–2088,2008.
[Jegadeeshand Titman,1993] Narasimhan Jegadeesh and
Sheridan Titman. Returns to buying winners and selling
losers: Implicationsforstockmarketefficiency. The Jour-
naloffinance,48(1):65–91,1993.
[Kakushadze,2016] Zura Kakushadze. 101 formulaic al-
phas. Wilmott,2016(84):72–81,2016.
[Keetal.,2017] Guolin Ke, Qi Meng, Thomas Finley,
Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, and
Tie-Yan Liu. Lightgbm: Ahighlyefficientgradientboost-
ingdecisiontree. In Advancesin Neural Information Pro-
cessing Systems, pages3146–3154,2017.
[Lehmanand Stanley,2011] Joel Lehman and Kenneth O
Stanley. Evolving a diversity of virtual creatures through
novelty search and local competition. In Proceedings of
the 13 th annual conference on Genetic and evolutionary
computation, pages211–218.ACM,2011.
[Mahfoud,1992] Samir W Mahfoud. Crowding and prese-
lectionrevisited. In PPSN, volume2, pages27–36,1992.
