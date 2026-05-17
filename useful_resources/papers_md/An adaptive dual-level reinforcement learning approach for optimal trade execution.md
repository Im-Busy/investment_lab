# An adaptive dual-level reinforcement learning approach for optimal trade execution

> *Source PDF: An adaptive dual-level reinforcement learning approach for optimal trade execution.pdf*
> *Extraction: Combined — previous structured extraction (base) + markitdown raw text*

---

# An adaptive dual-level reinforcement learning approach for optimal trade execution

> *Source PDF: An adaptive dual-level reinforcement learning approach for optimal trade execution.pdf*
> *Extraction: Combined — markitdown raw text (base) + previous extraction supplements*

---

ExpertSystemsWithApplications252(2024)124263
ContentslistsavailableatScienceDirect
ExpertSystemsWithApplications
journalhomepage:www.elsevier.com/locate/eswa
Anadaptivedual-levelreinforcementlearningapproachforoptimaltrade
execution
SoohanKimd,1,JimyeongKimc,1,HongKeeSula,1,YoungjoonHongb,1,∗
aDepartmentofFinance,Chung-AngUniversity,Seoul,RepublicofKorea
bDepartmentofMathematicalScience,KoreaAdvancedInstituteofScienceandTechnology(KAIST),Daejeon,RepublicofKorea
cStochasticAnalysisandApplicationResearchCenter(SAARC),KoreaAdvancedInstituteofScienceandTechnology(KAIST),Daejeon,RepublicofKorea
dNYUCourantInstituteofMathematicalSciences,NY,UnitedStatesofAmerica
A R T I C L E I N F O A B S T R A C T
Keywords: Thepurposeofthisresearchistodeviseatacticthatcancloselytrackthedailycumulativevolume-weighted
Volume-weightedaverageprice averageprice(VWAP)usingreinforcementlearningwhileminimizingthedeviationfromtheVWAP.Previous
Reinforcementlearning studies often choose a relatively short trading horizon to implement their models, making it difficult to
Optimaltradeexecution accurately track the daily cumulative VWAP since the stock price movement is often insignificant within
Proximalpolicyoptimization
theshorttradinghorizon.Ontheotherhand,trainingreinforcementlearningmodelsdirectlyoveralonger,
Markovdecisionprocess
dailyhorizonisburdensomeduetoextensivesequencelength.Hence,thereisaneedforamethodthatcan
dividethelongdailyhorizonintosmaller,moremanageablesegments.Weproposeamethodthatleverages
the U-shaped pattern of intraday stock trade volumes and uses Proximal Policy Optimization (PPO) as the
learningalgorithm.Ourmethodfollowsadual-levelapproach:aTransformermodelthatcapturestheoverall
(global)distributionofdailyvolumesinaU-shape,andaLSTMmodelthathandlesthedistributionoforders
withinsmaller(local)timeintervals.Theresultsfromourexperimentssuggestthatthisdual-levelarchitecture
improvescumulativeVWAPtrackingaccuracycomparedtopreviousreinforcementlearningapproaches.The
keyfindingisthatexplicitlyaccountingfortheU-shapedintradayvolumepatternleadstobetterperformance
inapproximatingthecumulativedailyVWAP.Thishasimplicationsfordevelopingtradingstrategiesthatneed
toefficientlytrackVWAPoverafulltradingday.
1. Introduction wantstoavoidmarketimpactandtheriskoffront-runnersexploiting
its strategy. To achieve this objective, the fund may outsource the
The optimal trade execution problem aims to find a strategy to trading to external brokerages or trading firms. In such a scenario,
optimallytradelargeorderswithinagivenperiodoftime.Oneofthe VWAP is commonly the most relevant metric for quantifying trading
mostcommonandpracticalmethodsthatpractitionersfrequentlyuseis performance. If the typical end-of-day market price were used as the
knownasVolumeWeightedAveragePrice(VWAP)trading(Madhavan, performance metric, traders might execute all orders at the market’s
2002). Research regarding the optimality of VWAP as an execution close. This could lead the fund to suffer from market impact, and
strategy is discussed in Kato (2015), which further highlights the theend-of-daymarketpricemightbecomemorevolatile.SinceVWAP
importanceoftheabilityofatradingmodeltotrackVWAPthroughout
reflectsastock’stransactionpriceovertime,itservesasamoresuitable
timeconsistently.VWAPiscalculatedbyaddingupthedollarstraded
metricfortradingperformance.UsingVWAPallowsthefundtoexecute
foreverytransaction(pricemultipliedbythenumberofsharestraded)
its strategy with minimal market impact, thereby avoiding potential
andthendividingbythetotalsharestradedfortheday.Thisgivesan
lossesfrommarketspeculationandfront-running.
averagepricethattakesintoaccountboththepriceandthevolumeof
InBertsimasandLo(1998),afundamentalworkforoptimaltrade
sharestraded.
execution is proposed, where the authors assume that market prices
Funds and traders often use VWAP as a benchmark to compare
followanarithmeticrandomwalk.Theyuseadynamicprogramming
the price at which they executed trades to the overall market price
principle to find an explicit closed-form solution. Building on this
for the security, in order to evaluate the performance of their trades.
work, Huberman and Stanzl (2005) and Almgren and Chriss (2001)
Suppose a fund intends to significantly change its stock holdings but
∗ Correspondingauthor.
E-mailaddresses: soo.han.kim@nyu.edu(S.Kim),jimyeongkim@kaist.ac.kr(J.Kim),hksul@cau.ac.kr(H.K.Sul),hongyj@kaist.ac.kr(Y.Hong).
1 Theseauthorscontributedequallytothiswork.
https://doi.org/10.1016/j.eswa.2024.124263
Received12May2023;Receivedinrevisedform7May2024;Accepted14May2024
Availableonline17May2024
0957-4174/©2024ElsevierLtd.Allrightsarereserved,includingthosefortextanddatamining,AItraining,andsimilartechnologies.

S.Kimetal.
ExpertSystemsWithApplications252(2024)124263
extendedtheresultinBertsimasandLo(1998)byincorporatingtrans-
| action costs, | more | complex | price | impact | functions, | and | risk aversion |     |     |     |     |     |     |     |
| ------------- | ---- | ------- | ----- | ------ | ---------- | --- | ------------- | --- | --- | --- | --- | --- | --- | --- |
parameters,undertheassumptionthatmarketpricesfollowaBrownian
| motion. These | dynamical | approaches, |     | however, |     | are difficult | to apply |     |     |     |     |     |     |     |
| ------------- | --------- | ----------- | --- | -------- | --- | ------------- | -------- | --- | --- | --- | --- | --- | --- | --- |
directlyintherealworldduetothediscrepancybetweentheirstrong
marketassumptionsandthecomplexityofactualmarketconditions.
Ontheotherhand,bothpractitionersandresearchershavewidely
usedthetime-weightedaverageprice(TWAP)strategyandthevolume-
| weighted | average | price (VWAP) |     | strategy | (Berkowitz | et  | al., 1988; |     |     |     |     |     |     |     |
| -------- | ------- | ------------ | --- | -------- | ---------- | --- | ---------- | --- | --- | --- | --- | --- | --- | --- |
Kakadeetal.,2004),whicharebasedoneitherpurerulesorstatistical
| rules. Especially | in  | Kakade | et al. | (2004), | the authors | used | historical |     |     |     |     |     |     |     |
| ----------------- | --- | ------ | ------ | ------- | ----------- | ---- | ---------- | --- | --- | --- | --- | --- | --- | --- |
datatoestimatetheaveragevolumetradedforeachtimeintervaland
| split the | order accordingly. |     | However, | this | strategy | is not | well-suited |     |     |     |     |     |     |     |
| --------- | ------------------ | --- | -------- | ---- | -------- | ------ | ----------- | --- | --- | --- | --- | --- | --- | --- |
forcapturingunexpectedvolatility.
Fromtheperspectivethatoptimalexecutionproblemsareatypeof
sequentialdecision-makingtask,RLhasbeencommonlyappliedinthis
fieldasitisatypeofstochasticdecision-makingprocessthatautomates
thepractitioner’staskofusingpastdatatomakedecisionsonwhento
| execute orders. | Since     | RL approaches |     | have            | many | advantages, | e.g. ca- |     |     |     |     |     |     |     |
| --------------- | --------- | ------------- | --- | --------------- | ---- | ----------- | -------- | --- | --- | --- | --- | --- | --- | --- |
| pable of        | capturing | the market’s  |     | microstructure, |      | the trading | results  |     |     |     |     |     |     |     |
conductedbyRLsolutionsoftenoutperformtraditionalVWAPtracking
approachessuchasBialkowskietal.(2008)andPodobniketal.(2009).
Tothebestofourknowledge,Nevmyvakaetal.(2006)isthefirstwork
toleverageRLframeworkssuchasQ-learning(Watkins&Dayan,1992)
| to optimal              | trade execution |     | problems.  | It is | important      | to note | that the  |     |     |     |     |     |     |     |
| ----------------------- | --------------- | --- | ---------- | ----- | -------------- | ------- | --------- | --- | --- | --- | --- | --- | --- | --- |
| curse of dimensionality |                 | in  | Q-learning | makes | it challenging |         | to handle |     |     |     |     |     |     |     |
high-dimensionaldata.While(Hendricks&Wilcox,2014)attemptsto
combineRLwiththeAlmgren–Chrissmodel,itisachallengingtaskas
themodeldependsoncertainassumptionsaboutthemarketdynamics.
ThankstotheadvancementsindeepRL,recentstudiessuchasMacri
| andLillo(2024), | Ningetal.(2021),andLinandBeling(2019)have |     |     |     |     |     |     |     |     |     |     |     |     |     |
| --------------- | ----------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
utilizedDeepQ-Networks(DQNs)(Mnihetal.,2013)foroptimaltrade
execution,addressingthechallengesofhigh-dimensionaldataandthe
| complexity   | of the   | financial | market  | without | relying | on     | any market  |     |     |     |     |     |     |     |
| ------------ | -------- | --------- | ------- | ------- | ------- | ------ | ----------- | --- | --- | --- | --- | --- | --- | --- |
| assumptions. | However, | these     | methods | require | the     | design | of specific |     |     |     |     |     |     |     |
attributes,whichcanbelabor-intensive.Recently,severalstudieshave
Fig.1. Tradevolumeratioovereach20-minuteperiodthroughouttheday.Thelines
investigatedtheuseofproximalpolicyoptimization(PPO)(Schulman represent the 1-year averages of the ratios and the shaded regions are drawn from
et al., 2017) based on optimal execution frameworks, which do not dailydeviationsfromtheaverages.
| require manually | designed |        | feature | engineering. |             | For instance, | Lin and     |     |     |     |     |     |     |     |
| ---------------- | -------- | ------ | ------- | ------------ | ----------- | ------------- | ----------- | --- | --- | --- | --- | --- | --- | --- |
| Beling (2020),   | Fang     | et al. | (2021), | Pan et       | al. (2022), | and           | Byun et al. |     |     |     |     |     |     |     |
(2023) have explored this approach in different scenarios. Specifi- will be executed in each interval. We propose two methods for the
cally,Panetal.(2022)focusedonoptimalexecutionwithlimitorders, firststage,thestatisticalU-shapemethodandtheU-shapeTransformer
while Lin and Beling (2020) and Fang et al. (2021) used market method.ThestatisticalU-shapemethodallocatesthevolumebasedon
orders.Amethodcombiningbothlimitandmarketorderswasproposed the historical average trade volumes. However, this approach cannot
byByunetal.(2023). fully capture the day-to-day variations (See Fig. 1). Alternatively, we
Prior literature often considers a relatively short trading horizon propose the U-shape Transformer for figuring out these variations.
ortheentiredayforRLimplementation.However,VWAPapproxima- In the next stage, we apply LSTM, trained by the RL framework to
tion becomes challenging when a short time frame is used since the distribute orders efficiently within each interval. It is important to
variations of financial data are often insignificant within short time note that there are numerous RL training algorithms, including Trust
intervals.Moreover,thenaiveapproachofconsideringtheentireday Region Policy Optimization (TRPO) (Schulman et al., 2015), Deep Q-
canputastrainonthenetworkarchitectureduetothelongsequence Networks(DQN)(Mnihetal.,2013),andProximalPolicyOptimization
|              |       |         |         |            |     |          |              | (PPO) (Schulman | et al., | 2017). | Among | these options, | we specifically |     |
| ------------ | ----- | ------- | ------- | ---------- | --- | -------- | ------------ | --------------- | ------- | ------ | ----- | -------------- | --------------- | --- |
| length. This | paper | aims to | develop | a strategy |     | that can | consistently |                 |         |        |       |                |                 |     |
choosePPOasourtrainingalgorithm.Foracomprehensiveunderstand-
trackthedailycumulativeVWAP.Toachievethisgoal,westartwith
ingofthesetechniquesandtheirapplications,werefertoSuttonand
| the observation | of  | an important | stock | trading |     | characteristic, | the U- |     |     |     |     |     |     |     |
| --------------- | --- | ------------ | ----- | ------- | --- | --------------- | ------ | --- | --- | --- | --- | --- | --- | --- |
Barto(2018).
| shaped intraday | trading | pattern, | documented |         | in           | Jain and | Joh (1988)    |           |           |            |          |          |              |     |
| --------------- | ------- | -------- | ---------- | ------- | ------------ | -------- | ------------- | --------- | --------- | ---------- | -------- | -------- | ------------ | --- |
|                 |         |          |            |         |              |          |               | We expect | that this | dual-level | approach | improves | the accuracy | of  |
| and Goodhart    | and     | O’Hara   | (1997).    | This is | a well-known |          | stylized fact |           |           |            |          |          |              |     |
approximatingthecumulativeVWAPbyproperlydistributingthetotal
statingthatastock’stradingvolumeoftenfollowsaU-shapedpattern
order.Ourcontributionsareasfollows:
throughouttheday.Thismeansthatvolumeishighestattheopening,
| falls rapidly  | to lower     | levels,       | and     | then rises | again      | towards    | the close   |               |              |             |               |               |                  |      |
| -------------- | ------------ | ------------- | ------- | ---------- | ---------- | ---------- | ----------- | ------------- | ------------ | ----------- | ------------- | ------------- | ---------------- | ---- |
|                |              |               |         |            |            |            |             | 1. We propose | a novel      | approach    | for           | optimal       | trade execution  | that |
| of the market, | but          | is relatively | low     | in the     | afternoon. | (See       | Fig. 1). In |               |              |             |               |               |                  |      |
|                |              |               |         |            |            |            |             | utilizes      | a dual-level | strategy.   | Orders        | are allocated | through          | two  |
| view of this   | observation, | we            | propose | a novel    | dual-level |            | approach to |               |              |             |               |               |                  |      |
|                |              |               |         |            |            |            |             | stages:       | in the       | first stage | we utilize    | the           | U-shaped pattern | of   |
| minimize       | the market   | impact        | and     | track the  | daily      | cumulative | VWAP        |               |              |             |               |               |                  |      |
|                |              |               |         |            |            |            |             | intraday      | volumes      | and         | in the second | stage         | we implement     | deep |
accuratelyandconsistently.Asthenamesuggests,ourmodelconsists reinforcementlearning.Sincethisnoveldualapproachsegments
of two stages. In the first stage, the model decides how much trade the data, we not only make reinforcement learning more com-
volume will be executed in each interval, using the U-shape property putationallytractablebutalsoimprovepredictionaccuracycom-
asaguideline.Inthesecondstage,theRLmodeldecideshowtheorders paredtoapplyingRLdirectlyonfullintradaysequences.
2

S.Kimetal. ExpertSystemsWithApplications252(2024)124263
Table1 objectiveistofindastrategythatmaximizesorminimizestheaverage
The table summarizes previous works on optimal trade execution utilizing the RL executionprice𝑃̄,whichiscalculatedas
framework. Note that DQN and PPO are learning algorithms of deep RL. Notably,
ourworkdistinguishesitselfbyfocusingonalong-termhorizon. 𝑃̄ = 𝑇 ∑ −1𝑂 𝑡𝑝.
RLmethod Horizon 𝑂 𝑡
𝑡=0
Nevmyvakaetal.(2006) Q-learning Short
HendricksandWilcox(2014) Q-learning Short While many financial firms pursue profits through trading, there are
LinandBeling(2019) DQN Short also numerous firms within the financial industry that prioritize con-
Ningetal.(2021) DQN Short sistentlytrackingtheVWAPovermaximizingtradingprofit.Thefocus
LinandBeling(2020) PPO Short
ofourpaperistailoredforthissecondgroupoftraders.TheVWAPis
Fangetal.(2021) PPO Short
calculatedas
Panetal.(2022) PPO Short
Thiswork PPO Long ∑𝑇−1𝑝𝑞
VWAP= 𝑡=0 𝑡 𝑡 ,
∑𝑇−1𝑞
𝑡=0 𝑡
where 𝑞 is the execution volume determined by the market. There-
2. We further propose the U-shape Transformer model to capture 𝑡
fore,theprimaryobjectiveofthispaperistominimizethedifference
theday-to-dayoscillationsoftheintradayU-shapedistribution. betweentheVWAPand𝑃̄,usingadual-levelapproach.
WhiletheU-shapedpatterniswell-known,relyingonfixedsta-
tisticalestimatesfailstoadapttosuddenfluctuationsorshiftsin
3. Methoddescription
marketconditions.TheU-shapeTransformerdirectlylearnsaro-
bustrepresentationfromrecentdata,leveragingself-attentionto
Inthissection,wedescribeourdual-levelapproachandformulation
automaticallyadapttosuddendynamicsintheintradayU-shape
oftheoptimaltradeexecutionproblemunderthelensofreinforcement
pattern.
learning.WealsoprovidedetailsonthearchitectureoftheTransformer
3. Weconductablationstudiestoshowthestrengthsofourmeth-
and LSTM neural networks and how they are integrated into our
odsinapproximatingthedailycumulativeVWAP.Furthermore,
proposedmethod.
we demonstrate that the methods employed to distribute vol-
umes from long trading horizons into short horizons can have 3.1. Dual-levelapproach
asignificantimpact.Inaddition,weintroducetheVAA,which
provides a more practical way to evaluate how well a trading Our approach involves two stages of distributing orders. In the
strategy tracks the daily VWAP. Our proposed dual-level ap- first stage, we allocate the total daily orders 𝑂 into 𝐿 intervals. In
proachcombiningU-shapeTransformersandRLachievesstate- thesecondstage,wefurtherdividetheordersfromeachintervalinto
of-the-artperformanceontheVAA(seeTable1). smaller executable orders using RL. Our approach begins by dividing
the market hours of a day into 𝐿 intervals. For each interval 𝑙, we
let 𝑂𝑙 as the total order to be executed, ensuring that the sum of all
2. Backgrounds
intervalordersequalsthetotaldailyorder,or
∑𝐿−1𝑂𝑙=𝑂.Wesuggest
𝑙=0
two methods for implementing the first stage. The naive approach
Inthissection,wediscussthebasicsofthelimitorderbook(LOB)
is to adhere to the statistical U-shape trade volume distribution for
and the optimal trade execution process. Most modern financial ex-
each interval obtained from past data (e.g. the historical average of
changes, including the Korea Stock Exchange (KRX), offer electronic
the past year). A more advanced method would be to use the output
tradingplatformswithaccesstolimitorderbooks.Tradersoftenrelyon
(predicted U-shape distribution) generated by a U-shape Transformer
theinformationprovidedbythelimitorderbooktodevelopsuccessful
model. In the latter case, 𝑂𝑙 is calculated progressively by referring
tradingstrategies.
only to market information from past days and previous intervals
within the day. For such an 𝑙th interval, which is comprised of 𝑇
2.1. Limitorderbook timesteps,theLSTMmodeloutputs𝑂𝑙,whichisthenumberoforders
𝑡
to be executed at timestep 𝑡 such that ∑𝑇−1𝑂𝑙 = 𝑂𝑙. Once 𝑂𝑙 is
Infinancialmarkets,alimitorderisanordertobuyorsellafixed 𝑡=0 𝑡 𝑡
decided, it is averaged over 𝐼 execution steps in the corresponding
number of shares at a specified price. This order becomes part of the
𝑡th subinterval. In other words, for each execution step, which is
limitorderbook,whichrecordsalllimitordersforaspecificstock.The set as every 5 s, the final number of orders executed is 𝑂𝑙∕𝐼. The
𝑡
bidpriceisthehighestpriceabuyeriswillingtopayforthesecurity,
choice of averaging is supported by our findings in Section 4.2 that
andtheaskpriceisthelowestpriceinwhichtheselleriswillingtosell.
there is no significant impact in approximating the VWAP since the
Amarketorder,bycontrast,isanordertobuyorsellafixednumber finalintervalistoosmall.Furthermore,inSection4.2,weverifythat
ofsharesatthecurrentmarketpricewithoutstatingtheprice.Whena incorporatinglong-termhorizonsyieldssuperiorefficiency.Therefore,
marketorderisplaced,itismatchedbythebestavailablelimitorders althoughadeepreinforcementlearningframeworkcanoffermarginal
intheLOB.TheLOBisimportantforoptimaltradeexecutionandhelps performanceimprovements,wehaveoptedforthesimplicityofusing
traderstracksupplyanddemandforastock,makingiteasiertoidentify theuniformdistributionforthefinalinterval.Aschematicillustration
thebesttimeandpricetobuyorsellshares. ofthisdesignisprovidedinFig.2.
2.2. Optimaltradeexecution 3.2. MDPformulationforoptimalexecution
Optimal trade execution refers to the process of buying or selling In this section, we explain our Markov Decision Process (MDP)
afinancialassetwhileachievingthedesiredobjective.Generally,this formulationofoptimaltradeexecution.AMDPistypicallyrepresented
problem formulates as follows: Within a timeframe of 𝑇 timesteps, using the tuple (, , , 𝑟, 𝛾), where  is the state space,  is the
0,1,…,𝑇 −1,atraderwhopossessesaninventoryof𝑂sharesmustbuy action space,  is the transition probability, 𝑟 is the reward function,
orselltheentiretyoftheinventory.Foreachtimestep𝑡∈{0,1,…,𝑇− and𝛾isthediscountfactor.Ourconfigurationsofthestatespaceand
1},thetraderdeterminesthenumberofsharestoorder𝑂 basedonthe actionspacearesimilartothatofLinandBeling(2020).Itisworth
𝑡
informationfromLOB,andcarriesoutthetradeatanexecutionprice notingthattheMDPimplementationinthispaperisappliedonaper
𝑝. For the group of traders seeking to maximize trading profit, their 𝑙thintervalbasisforagivenday.
𝑡
3

S.Kimetal.
ExpertSystemsWithApplications252(2024)124263
Fig.2. Aschematicillustrationofourdual-levelapproach.
3.2.1. State Ourrewardfunctioncompares𝑂𝑙∗ and𝑂𝑙 directlyforagiven𝑎𝑙 as
|                                                           |     |     |     |     |     |          |     |     | 𝑡   | 𝑡   | 𝑡   |
| --------------------------------------------------------- | --- | --- | --- | --- | --- | -------- | --- | --- | --- | --- | --- |
| Thestate𝑠𝑙∈consistsofbothpublicandprivateinformation.The |     |     |     |     |     | follows: |     |     |     |     |     |
𝑡
publicstateismadeupofthetop5bidandaskprices,alongwiththeir ⎧ 𝑀𝑙<0.01
|     |     |     |     |     |     |     | 1 𝑡 |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
associated volumes, while the private state includes the elapsed time 𝑟(𝑎𝑙)∶= ⎪ 1≤𝑀𝑙<0.05
|                                                         |           |        |       |           |                  | 𝑡         | ⎨ 0 0.0        | 𝑡   |     |     | (2) |
| ------------------------------------------------------- | --------- | ------ | ----- | --------- | ---------------- | --------- | -------------- | --- | --- | --- | --- |
| and the current                                         | remaining | volume | to be | executed. | The elapsed time |           |                |     |     |     |     |
|                                                         |           |        |       |           |                  |           | ⎪ −1 otherwise |     |     |     |     |
| rangesfrom0to𝑇−1andthecurrentremainingvolumeattimestep𝑡 |           |        |       |           |                  |           | ⎩              |     |     |     |     |
| isequalto𝑂𝑙−                                            | ∑𝑡 −      | 1 𝑂 𝑙. |       |           |                  |           | 𝑙−𝑂 𝑙∗         |     |     |     |     |
|                                                         | 𝑗         | = 0 𝑗  |       |           |                  | where𝑀𝑙∶= | |𝑂 𝑡 𝑡         | |.  |     |     |     |
|                                                         |           |        |       |           |                  |           | 𝑡 𝑂𝑙∗          |     |     |     |     |
𝑡
OnceMDPisdetermined,thegoalofpolicy-basedRListofindan
| 3.2.2. Action                    |     |     |     |                        |     | optimalpolicyparameter𝜃∗,i.e. |     |     |     |     |     |
| -------------------------------- | --- | --- | --- | ---------------------- | --- | ----------------------------- | --- | --- | --- | --- | --- |
| OurLSTMmodeloutputsapolicy𝜋(⋅|𝑠𝑙 |     |     |     | )∈R21,whichisadiscrete |     |                               |     |     |     |     |     |
|                                  |     |     |     | 𝑡                      |     | 𝜃∗=                           |     |     |     |     |     |
𝑙 
c a t e g o r i c a l p r o b ab ili t y d i s t r i b u t io n . F r o m t h is , a n a c t i o n 𝑎 ∈ ∶ = [ ]
|     |     |     |     |     | 𝑡   |     | ∑ 𝑇 |     |     |     | (3) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
{ 0 , 0 . 1 , 0 . 2 , … , 2 } i s s a m p l e d t o d e t e r m i n e t h e n u m b e r o f o r d e r s t o argmaxE 𝛾𝑡𝑟 | 𝑎 ∼𝜋 (⋅|𝑠 ),𝑠 ∼(⋅|𝑠 ,𝑎 ) ,
|     |     | 𝑙   |     |     |     |     | 𝑡 | | | 𝑡 𝜃 | 𝑡 𝑡+1 | 𝑡 𝑡 |     |
| --- | --- | --- | --- | --- | --- | --- | ----- | --- | ----- | --- | --- |
e x e c u t e , w it h 𝑂 𝑙 = 𝑎 𝑙 𝑂 . T o e n s u r e t h a t t h e to t a l n u m b e r o f o r de r s |
|                                               | 𝑡   | 𝑡 𝑇 |     |     |                 |        | 𝑡= 0    |     |     |     |     |
| --------------------------------------------- | --- | --- | --- | --- | --------------- | ------ | ------- | --- | --- | --- | --- |
| executedovertheentireperiodisequalto𝑂𝑙,thatis |     |     |     |     | ∑𝑇 − 1𝑂 𝑙=𝑂𝑙,we |        |         |     |     |     |     |
|                                               |     |     |     |     | 𝑡= 0 𝑡          | where𝑟 | =𝑟(𝑎 ). |     |     |     |     |
|                                               |     |     |     |     |                 |        | 𝑡 𝑡     |     |     |     |     |
imposethefollowingtworestrictions.
3.3. Neuralnetworkarchitectureandtraining
∑𝑗
| • If | 𝑂𝑙>𝑂𝑙 | forsome𝑗∈{0,1,2,…,𝑇 |     | −1},then |     |     |     |     |     |     |     |
| ---- | ----- | ------------------- | --- | -------- | --- | --- | --- | --- | --- | --- | --- |
| 𝑡=0  | 𝑡     |                     |     |          |     |     |     |     |     |     |     |
{ 𝑂𝑙− ∑𝑗−1𝑂𝑙, We now turn to the architectural details of the Transformer and
|     |     | 𝑡   | if𝑖=𝑗 |     |     |     |     |     |     |     |     |
| --- | --- | --- | ----- | --- | --- | --- | --- | --- | --- | --- | --- |
𝑂𝑙= 𝑡=0 LSTM models and elaborate on how they are used in our dual-level
| 𝑖   |     | 0, if𝑗<𝑖≤𝑇 | −1  |     |     |     |     |     |     |     |     |
| --- | --- | ---------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
approach.
| • 𝑂𝑙 | isequaltotheremainingvolumetobeexecutedatthelast |     |     |     |     |     |     |     |     |     |     |
| ---- | ------------------------------------------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
𝑇− 1
tim e stepforeachinterval(i.e.0≤𝑙≤𝐿−1) 3.3.1. Level1:TransformerforglobalU-shapeapproximation
TheobjectiveoftheTransformermodelistoprogressivelypredict
theU-shaperatioforeachintervalinagivenday.Weusethestructure
Theserestrictionsareinlinewiththefactthatitisessentialforbrokers
oftheTransformerEncoderandDecoderasproposedinVaswanietal.
topromptlyacquireorliquidateallsharesrequestedbytheircustomers.
(2017).TheoverallarchitectureoftheTransformermodelthatweuse
isdepictedinFig.3.
3.2.3. Reward U-shapeEncoder.TheU-shapeEncoder iscomposedofaTrans-
|            |          |             |     |           |                 | former | Encoder and | 𝐿 linear | layers, all | of which map | to the same |
| ---------- | -------- | ----------- | --- | --------- | --------------- | ------ | ----------- | -------- | ----------- | ------------ | ----------- |
| The reward | function | is designed | to  | encourage | the agent (LSTM |        |             |          |             |              |             |
model) to generate actions that result in orders that are close to the dimension.Thismoduleisresponsibleforlearningthegeneraldynam-
target volume 𝑂𝑙∗, in order to track the VWAP. This is achieved by ics of the U-shape distribution. The input 𝐸 𝑖𝑛 is a sequence of length
𝑡
usingthepricecalculatedfromtheexecutedorders.Here,𝑂𝑙∗ denotes 𝐿 of vectors each containing 𝑁 historical daily volume ratios of the
𝑡
|             |      |                   |      |         |                    | corresponding | interval | in the | sequence. | The days from | which these |
| ----------- | ---- | ----------------- | ---- | ------- | ------------------ | ------------- | -------- | ------ | --------- | ------------- | ----------- |
| the desired | VWAP | order at timestep | 𝑡 in | the 𝑙th | interval such that |               |          |        |           |               |             |
∑𝑇−1𝑂𝑙∗.ThedailyVWAPcanbeobtainedby ratios are drawn are randomly selected. Each vector in the sequence
𝑂𝑙=
𝑡=0 𝑡 is processed by a different linear layer that serves as an embedding
𝐿 −1𝑇 −1𝑂 𝑙∗ t o le a rn t h e u ni q u e f ea tu re s o f e a c h in t e rv a l. A f ter p a s s in g th r ou g h
∑ ∑ 𝑡 𝑝𝑙
𝑉𝑊𝐴𝑃 𝑑𝑎𝑦 = , (1) t he e m be d d in g l a y er , th e i np u t s e q ue n c e i s th e n pr o c e s se d b y t h e
𝑂 𝑡
|     | 𝑙=0 𝑡=0 |     |     |     |     | TransformerEncoder,resultingintheoutput𝐸 |     |     |     |     |     |
| --- | ------- | --- | --- | --- | --- | ---------------------------------------- | --- | --- | --- | --- | --- |
𝑜𝑢𝑡 .
| 𝑝𝑙  |     |     |     |     |     |     |     |     |    |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
where 𝑡 is the associated average market traded price of the corre- U-shape Decoder. While learns the general dynamics of the U-

sponding𝑡thsubinterval. shape distribution, the U-shape Decoder focuses on handling the
4

S.Kimetal. ExpertSystemsWithApplications252(2024)124263
3.3.3. Trainingprocedure
U-shape Transformer. Since the goal of the first level of order al-
locationistoapproximatetheU-shapedistributionwhilealsotracking
thedailycumulativeVWAP,thetrainingobjectiveoftheTransformer
modelistominimize
𝐽 ∶=E [ 𝑐 1 𝐿 ∑ −1( 𝑢𝑙 −𝑢𝑙 )2 +𝑐 𝑉𝐴𝐴 ] (6)
𝑇𝐹 𝑑𝑎𝑦𝑠 𝐿 𝑡𝑟𝑢𝑒 𝑝𝑟𝑒𝑑 2
𝑙=0
where𝑐 and𝑐 arecoefficientsand
1 2
VAA∶= | | | | | 𝑀𝑃 𝑑 𝑉 𝑎𝑦 𝑊 − 𝐴 𝑉 𝑃 𝑊 𝑑𝑎 𝐴 𝑦 𝑃 𝑑𝑎𝑦| | | | | (7)
denotestheVWAPApproximationAccuracy(VAA),whichwewillfur-
theruseasageneralmetricforperformanceevaluationinexperiments,
and
𝑀𝑃 ∶=
𝐿
∑
−1𝑇
∑
−1𝑂
𝑡
𝑙
𝑝𝑙
Fig. 3. The architecture of the U-shape Transformer. The U-shape Decoder in this 𝑑𝑎𝑦 𝑂 𝑡
𝑙=0 𝑡=0
figurerepresentsthe𝑙thdecodingstep.
is the price yielded by our model. This is one of the measures used
in practice to gauge how close a trader has traded closer to their
benchmark,theVWAP.Ifatraderhastradedatapricethatisexactly
day-to-dayvariationsofthedistribution.TheU-shapeDecoderiscom-
equal to the VWAP, then 𝑉𝐴𝐴 = 0. Thus lower VAA would be
posedofaTransformerDecoderand𝐿linearlayers,eachmappingto
better.Thefirstterminsidetheexpectationof(6)learnstomakeratio
differentdimensions.Incontrastto,theselinearlayersareappliedto
predictions closer to ground-truth values. The second term penalizes
theoutputoftheTransformerDecoder. theTransformermodelifthefinaldailyacquisitionpricegeneratedby
For every 𝑙th (𝑙 > 0) decoding step, both 𝐸 𝑜𝑢𝑡 and the cumulative thecombinationoftheTransformerandLSTMmodelsdeviatesfromthe
input sequence are fed as input to the Transformer Decoder, . We dailyVWAP.TheprimaryobjectiveoftheU-shapedtransformerinthe
denote the cumulative input sequence as 𝐷 𝑖 𝑙 𝑛 = (𝐷 𝑖 𝑙 𝑛,𝑗 ) 0≤𝑗≤𝑙−1 , where firstlevelistodistributethedailytotalorderwithprecision.However,
𝐷𝑙 ∶= CONCAT(𝐷𝑗 ,ℎ𝑗 ). Here, 𝐷𝑗 represents the output of  relyingsolelyonthefirsttermofthelossfunctionmaynotbeenough
at 𝑖𝑛 t , h 𝑗 e𝑗thstep,andℎ 𝑜𝑢 𝑗 𝑡 𝑇 is −1 thelastLST 𝑜𝑢 M 𝑡 hiddenvectorfromthe𝑗th to accomplish this objective. This is because 𝑢𝑙 𝑡𝑟𝑢𝑒 does not consider
𝑇−1 the local information, such as the LOB, which can pose difficulties
interval.TheoutputfromtheTransformerDecoderatthe𝑙thdecoding
in tracking the daily VWAP accurately. Furthermore, if only the first
step is a vector of dimension 𝐿+𝐻, which is then passed through a
termin(6)istakenintoaccount,thereinforcementlearningframework
linearlayerthatreducesitsdimensionto𝐿−𝑙.Thesoftmaxfunctionis
wouldnotinfluencethefirstlevel,leadingtoindependentoperationof
appliedtopredicttheratiosoftheremaining𝐿−𝑙intervals,andthefirst
thefirstandsecondlevels.Toensurethatthemicroinformationhasan
𝑙 components are zero-padded. Note that we use ℎ𝑗 𝑇−1 to capture the appropriate impact on the first level, we incorporate the second term
day-to-day variations in the U-shape distribution, as it holds interval- intothelossfunction.
specificinformationforthatparticularday.Toensure ∑𝐿 𝑙= − 0 1𝑢𝑙 𝑝𝑟𝑒𝑑 =1, LSTM.Tofindtheoptimalparameter𝜃fortheLSTMmodel,weuse
thefinal𝑙thintervalvolumeratiopredictioniscalculatedas an actor–critic style PPO algorithm (Schulman et al., 2017), which is
( 𝑙−1 ) oneofthemostpopularon-policyRLalgorithms,asourbaselearner.
𝑢𝑙 = 1− ∑ 𝑢𝑗 ⋅𝐷𝑙 [𝑙]. (4) One of the major advantages of the PPO in this task is its ability to
𝑝𝑟𝑒𝑑 𝑝𝑟𝑒𝑑 𝑜𝑢𝑡
𝑗=0 adapt to changing market conditions and effectively learn from noisy
andhigh-dimensionaldata.Thisalgorithmusestheactorloss𝐽𝐶𝐿𝐼𝑃(𝜃)
It is worth noting that 𝐷 𝑖 0 𝑛 is constructed by applying an additional andthecriticlossfunction𝐽𝑉𝐹(𝜃),whicharedefinedasfollow PP s O :
PPO
linear layer that transforms the dimension of the vector holding the
t w o i p th 5 t b h i e d g / i a v s e k n v d o a lu y m to e ta a l v o er r a d g e e r s d o e f no p t r e e d -m a a s r 𝑂 ke , t th d e at c a or to re 𝐿 sp + on 𝐻 d . in T g h i e n r t e e f r o v r a e l , 𝐽 ∶ P 𝐶 = P 𝐿 O E 𝐼𝑃( [ 𝜃 m ) in ( 𝑞(𝜃)𝐴̂,clip(𝑞(𝜃),1−𝜀,1+𝜀)𝐴̂)] , (8)
𝑡 𝑡 𝑡 𝑡 𝑡
order𝑂𝑙 canbedeterminedastheproductof𝑂and𝑢𝑙 ,i.e.
𝑝𝑟𝑒𝑑 and
[ ]
𝑂𝑙=𝑂×𝑢𝑙 𝑝𝑟𝑒𝑑 . (5) 𝐽 P 𝑉 P 𝐹 O (𝜃)=E 𝑡 (𝑉 𝑡 targ−𝑉 𝜃 (𝑠 𝑡 ))2 , (9)
where𝑞(𝜃)= 𝜋𝜃(𝑎𝑡|𝑠𝑡) ,
𝑡 𝜋𝜃old (𝑎𝑡|𝑠𝑡)
3.3.2. Level2:LSTMforlocalorderdistribution 𝑇−1
The objective of the LSTM model is to optimally allocate orders 𝑉 𝑡 targ= ∑ 𝛾𝑘−𝑡𝑟 𝑘 and𝐴̂ 𝑡 =𝑉 𝑡 targ−𝑉 𝜃 (𝑠 𝑡 ). (10)
withintheintervalsofagivenday.ThestructureofourLSTMmodel, 𝑘=𝑡
whichisoutlinedinFig.4,issimilartothatofLinandBeling(2020). PPO’sgoalistofindaparameter𝜃 thatmaximizesthemainobjective
Given𝑠𝑙 𝑡 asinput,itfirstpassesthroughtwolinearlayerswithhidden function𝐽 PPO (𝜃),whichisdefinedby
d T i h m e e r n e s s i u o l n tin 1 g 28 v a e n ct d or is is th t e h n en co p n r c o a c t e e s n s a e t d ed by wi a t n h 𝑎 L 𝑙 𝑡 S − T 1 M an C d e 𝑟 ll 𝑙 𝑡− w 1 i ∶ t = h h 𝑟( i 𝑎 d 𝑙 𝑡 d − e 1 ) n . 𝐽 PPO (𝜃)=𝐽 P 𝐶 P 𝐿 O 𝐼𝑃(𝜃)−𝑐 3 𝐽 P 𝑉 P 𝐹 O (𝜃)+𝑐 4 E 𝑡 [𝑆[𝜋 𝜃 ](𝑠 𝑡 )], (11)
dimension 𝐻 to obtain ℎ𝑙, which is then passed separately through where𝑐 3 ,𝑐 4 arecoefficients,andE 𝑡 [𝑆[𝜋 𝜃 ](𝑠 𝑡 )]denotesanentropyloss
𝑡
two linear layers to produce the policy 𝜋(⋅|𝑠𝑙
𝑡
) and the value 𝑉(𝑠𝑙
𝑡
) term,where𝑆[𝜋 𝜃 ](𝑠 𝑡 )isdefinedby
respectively.Wenotethatthevectorℎ𝑙 𝑇−1 whichisobtainedatthelast 𝑆[𝜋 𝜃 ](𝑠 𝑡 )∶=− ∑ 𝜋 𝜃 (𝑎|𝑠 𝑡 )log𝜋 𝜃 (𝑎|𝑠 𝑡 ). (12)
timestep is used as input to the Transformer Decoder. The action 𝑎𝑙
𝑡
, 𝑎∈
reward 𝑟𝑙 𝑡 , and the number of orders to execute 𝑂 𝑡 𝑙 is calculated as is By incorporating an entropy term, PPO can incentivize the agent to
describedinSection3.2.TheLSTMCellalsoreceivesℎ𝑙 and𝑐𝑙 as explorealternativeactions,preventgettingtrappedinsuboptimalpoli-
𝑡−1 𝑡−1
inputanditsoutputincludes𝑐𝑙 aswell. cies, and mitigate overfitting to the training data (Mnih et al., 2016;
𝑡
5

S.Kimetal.
ExpertSystemsWithApplications252(2024)124263
Fig.4. ThearchitectureoftheLSTMmodel.
Williams, 1992). To find the parameter 𝜃, PPO iteratively gathers Algorithm1Dual-levelNeuralNetwork
episodesandupdatestheparameter𝜃usingthefollowingscheme: Require: Number of 𝑜𝑢𝑡𝑒𝑟 𝑖𝑡𝑒𝑟𝑎𝑡𝑖𝑜𝑛𝑠 and 𝑛𝑢𝑚
𝑑𝑎𝑦𝑠
| 𝜃 =argmaxE | E   | [𝐽  | (𝜃)], |     |     |     | (13) |     |     |     |     |     |     |
| ---------- | --- | --- | ----- | --- | --- | --- | ---- | --- | --- | --- | --- | --- | --- |
𝑘+1 𝑠 𝑎∼𝜋𝜃𝑘 (⋅|𝑠) PPO Randomlyinitializelearnableparameters𝜃,𝜃,and𝜃
𝜃
where𝑘standsforthe𝑘thstep.The‘‘clip’’operatorin(8)allowsPPO Initializetrajectorybuffer
tolearnfrompreviousexperienceswithoutbecomingoverlydependent for𝑗=1to𝑜𝑢𝑡𝑒𝑟𝑖𝑡𝑒𝑟𝑎𝑡𝑖𝑜𝑛𝑠do
on them. Moreover, PPO can effectively retain knowledge from past for𝑑=1to𝑛𝑢𝑚𝑑𝑎𝑦𝑠do
experiences while also acquiring insights from new experiences. This Randomlychooseadateinthetrainingset
keyattributeofPPOisoneofitssignificantadvantages. 𝜇←theaverageofthedailyvolumeforthelast60days
𝜎 ←thestandarddeviationofthedailyvolumeforthelast60
4. Experiments
days
|     |     |     |     |     |     |     |     |  ← | ⋃   |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ---- | --- | --- | --- | --- | --- |
GatheringEpisodes(date,𝜇,𝜎)
| Our dual-level | approach |     | significantly | enhances | the | accuracy | and | endfor |     |     |     |     |     |
| -------------- | -------- | --- | ------------- | -------- | --- | -------- | --- | ------ | --- | --- | --- | --- | --- |
consistency of tracking the daily cumulative VWAP, despite the chal- Using , update ,  by optimizing the loss function 𝐽 in (5)
|           |           |                 |     |       |            |     |         | withrespectto𝜃 |     | and𝜃 |     |     | 𝑇𝐹  |
| --------- | --------- | --------------- | --- | ----- | ---------- | --- | ------- | --------------- | --- | ----- | --- | --- | --- |
| lenges in | improving | the performance |     | of RL | frameworks | for | optimal |                 |     |       |     |     |     |
trade execution in relatively short trading horizons. Furthermore, the Optimizethelossfunction𝐽 in(9)withrespectto𝜃
PPO
| U-shapeTransformerprovesefficientincapturingthedailyvariations |               |     |        |           |          |            |     |  ←∅   |     |     |     |     |     |
| -------------------------------------------------------------- | ------------- | --- | ------ | --------- | -------- | ---------- | --- | ------ | --- | --- | --- | --- | --- |
| of the U-shape                                                 | distribution. |     | In the | following | section, | we conduct | a   | endfor |     |     |     |     |     |
seriesofexperimentstoverifytheseassertions.
| 4.1. Experimentsettings |     |     |     |     |     |     |     | Algorithm2GatheringEpisodes |     |     |     |     |     |
| ----------------------- | --- | --- | --- | --- | --- | --- | --- | --------------------------- | --- | --- | --- | --- | --- |
Require:Thenumberofintervalsinaday𝐿
| 4.1.1. Implementation |     |     |     |     |     |     |     | Input:Date,𝜇,𝜎 |     |     |     |     |     |
| --------------------- | --- | --- | --- | --- | --- | --- | --- | -------------- | --- | --- | --- | --- | --- |
Output:Trajectorybufferof𝑖𝑡ℎday
Inallofourexperiments,thehyperparametersforthelossfunctions, 𝑖
| thelengthoftheintervals,thenumberoftotalorders,andthetrading |     |     |     |     |     |     |     | Initialize |     |     |     |     |     |
| ------------------------------------------------------------ | --- | --- | --- | --- | --- | --- | --- | ----------- | --- | --- | --- | --- | --- |
𝑖
horizonarefixedandsetaccordingtothefollowingspecifications: 𝑂∼(2.5×10−3𝜇,6.25×10−6𝜎2)
|                                  |     |     |           |                            |     |     |     | 𝐸 =(𝐸        | )   |                       |     |     |     |
| -------------------------------- | --- | --- | --------- | -------------------------- | --- | --- | --- | ------------- | --- | --------------------- | --- | --- | --- |
| - Thecoefficientsfor𝐽            |     |     | :𝑐 =0.5,𝑐 | =0.5                       |     |     |     | 𝑜𝑢𝑡           | 𝑖𝑛  |                       |     |     |     |
|                                  |     |     | TF 1      | 2                          |     |     |     | for𝑙=0to𝐿−1do |     |                       |     |     |     |
| - Thecoefficientsfor𝐽            |     |     | :𝑐 =1,𝑐   | =0.01                      |     |     |     |               |     |                       |     |     |     |
|                                  |     |     | PPO 3     | 4                          |     |     |     | if𝑙=0then     |     |                       |     |     |     |
| - Thelengthoftheintervals:𝐿=19,𝑇 |     |     |           | =20                        |     |     |     |               |     |                       |     |     |     |
|                                  |     |     |           |                            |     |     |     | Construct𝐷0   |     | fromrawpre-marketdata |     |     |     |
|                                  |     |     |           | ∼(2.5×10−3𝜇,6.25×10−6𝜎2), |     |     |     |               |     | 𝑖𝑛                    |     |     |     |
| - Thenumberoftotalorders:𝑂       |     |     |           |                            |     |     |     | else          |     |                       |     |     |     |
where𝜇and𝜎istheaverageandstandarddeviationoftheday 𝐷𝑙 ←𝐷𝑙−1⋃𝐶𝑜𝑛𝑐𝑎𝑡(𝐷𝑙−1,ℎ𝑙−1)
|     |     |     |     |     |     |     |     | 𝑖𝑛  | 𝑖𝑛  |     | 𝑜𝑢𝑡 𝑇−1 |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ------- | --- | --- |
totalvolumefortheprevioussixtydays,respectively.
endif
- Tradinghorizonofaday:380minutes(Fromwhenthemarket Obtain𝑢𝑙 and𝐷𝑙
byproceduredepictedinSection3.3.1
| opensat09:00:00towhenitclosesat15:20:00). |     |     |     |     |     |     |     |     | 𝑝𝑟𝑒𝑑 | 𝑜𝑢𝑡 |     |     |     |
| ----------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | ---- | --- | --- | --- | --- |
𝑂𝑙←𝑂⋅𝑢𝑙
𝑝𝑟𝑒𝑑
S in c e a t y pi c a l da y a l l o w s f o r 3 8 0 m i n o f tr ad in g, o u r m o d e l f ir st 𝑙 𝑙 𝑙 𝑙 𝑙 𝑙 𝑙} andℎ𝑙
|     |     |     |     |     |     |     |     | O b t a in | 𝜏 𝑙 = { 𝑠 | 𝑡 , 𝑎 𝑡 , 𝑟( 𝑎 𝑡 ), | 𝑉 ( 𝑠 𝑡 ), 𝜋 (𝑎 𝑡| 𝑠 𝑡 ),𝑂 𝑡 | 𝑡∈{0,1,2,⋯,𝑇−1} | by  |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------- | --------- | ------------------- | ---------------------------- | --------------- | --- |
take s t h e da i ly t o t al o rd e r , 𝑂 , a n d b r e ak s i td o w n to 𝑂 𝑙, 1 9 i nt e r v al s o f 𝑇−1
|                                                               |     |     |     |     |     |     |     | pr o c e du | r e d e p | i c t e d i n S | ec t i o n 3 .3 . 2 |     |     |
| ------------------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | ----------- | --------- | --------------- | ------------------- | --- | --- |
| 20mineach.(𝑙∈[0,18])WeeitherapplythestatisticalU-shapemethod  |     |     |     |     |     |     |     |  ←        | ⋃𝜏        |                 |                     |     |     |
|                                                               |     |     |     |     |     |     |     | 𝑖           | 𝑖 𝑙       |                 |                     |     |     |
| ortheTransformermodelforthisfirststageallocationprocess.Then, |     |     |     |     |     |     |     | endfor      |           |                 |                     |     |     |
withineachinterval,theLSTMmodelfurthersubdivides𝑂𝑙into20sets
of1-minutesubintervals,𝑂𝑙.(𝑡∈[0,19])
𝑡
| Ineachofthesesubintervals,𝑂𝑙 |     |     |     | isdividedintoequalportionsand |     |     |     |     |     |     |     |     |     |
| ---------------------------- | --- | --- | --- | ----------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
𝑡 stock,weaccountforthedifferencesintotalordersforvariousstocks,
| executed | over a span | of 12 | steps, assuming |     | that orders | are | executed |     |     |     |     |     |     |
| -------- | ----------- | ----- | --------------- | --- | ----------- | --- | -------- | --- | --- | --- | --- | --- | --- |
whichisoftenencounteredinreal-worldscenarios.Additionally,dur-
| every 5 s. | The method | and | hyperparameters |     | used to | train our | agents |     |     |     |     |     |     |
| ---------- | ---------- | --- | --------------- | --- | ------- | --------- | ------ | --- | --- | --- | --- | --- | --- |
ingtesting,weonlyusethedailygroundtruthU-shapevolumeratios
| for the experiment |     | are summarized |     | in Algorithms | 1,  | 2, and | Table 3 |             |              |      |                     |         |            |
| ------------------ | --- | -------------- | --- | ------------- | --- | ------ | ------- | ----------- | ------------ | ---- | ------------------- | ------- | ---------- |
|                    |     |                |     |               |     |        |         | of dates in | the training | data | for our Transformer | Encoder | to prevent |
whichsilocatedattheendofthepaper.
lookingaheadintothefuture.AsinpreviousworksofNevmyvakaetal.
| To conduct | realistic | simulations, |     | we determine | 𝑂   | in a | way that |     |     |     |     |     |     |
| ---------- | --------- | ------------ | --- | ------------ | --- | ---- | -------- | --- | --- | --- | --- | --- | --- |
(2006),HendricksandWilcox(2014),Ningetal.(2021),andLinand
| takes into | account | the volatility | of  | daily volume, | which | differs | from |     |     |     |     |     |     |
| ---------- | ------- | -------------- | --- | ------------- | ----- | ------- | ---- | --- | --- | --- | --- | --- | --- |
Beling(2020),wemakethefollowingassumptionsinourexperiments.
| most previous | works | where | it is fixed | to a certain | value. | This | choice |     |     |     |     |     |     |
| ------------- | ----- | ----- | ----------- | ------------ | ------ | ---- | ------ | --- | --- | --- | --- | --- | --- |
aims to simulate the fluctuations of total orders that financial firms 1. We assume that the actions taken by our model only affect
havetoexecuteinaday,takingintoconsiderationrecenttradevolume a temporary market, and that the market will recover to the
statistics for each stock. By considering the volume statistics for each equilibriumlevelatthenexttimestep.
6

S.Kimetal.
ExpertSystemsWithApplications252(2024)124263
Table2 theexperimentalsetupinNingetal.(2021).Thetotalorderfor
AveragevolumeforeachstockfromJanuary1st,2021toDecember31st,2021. thedayisthenequallydistributedamongsteachhour,andDQN
Stocks SE SH KC PH furtherallocatestheorderforfivesub-intervalsof12mineach.
Volume 17,000K 3800K 3900K 430K The orders are executed equally for each 5-second time step
withinthe12-minutesub-intervals.Toensurefaircomparisons,
weevaluatethedailypricegeneratedbyDQNagainstthedaily
cumulativeVWAPcalculatedbyexcludingthefinal20min.
- OPDisthemethodproposedinFangetal.(2021).Toimplement
|     |     |     |     |     |     |     |     | this | model, | we first | divide | a given day | into | ten intervals, | each |
| --- | --- | --- | --- | --- | --- | --- | --- | ---- | ------ | -------- | ------ | ----------- | ---- | -------------- | ---- |
spanning38min,andthenuseOPDtoallocateordersforeach
|     |     |     |     |     |     |     |     | interval. | Within | each | interval, | the | allocated | order | is equally |
| --- | --- | --- | --- | --- | --- | --- | --- | --------- | ------ | ---- | --------- | --- | --------- | ----- | ---------- |
distributedamongsteachminute.
- PPOisthemethodproposedinLinandBeling(2020).Toapply
|     |     |     |     |     |     |     |     | this   | model, | we equally | distribute | the     | day     | total order   | to each |
| --- | --- | --- | --- | --- | --- | --- | --- | ------ | ------ | ---------- | ---------- | ------- | ------- | ------------- | ------- |
|     |     |     |     |     |     |     |     |        |        |            | 𝑂          | = 𝑂∕380 |         | 𝑗 ∈ {1,…,380} |         |
|     |     |     |     |     |     |     |     | minute | in     | the day,   | i.e. 𝑗     |         | for all |               | as      |
inLinandBeling(2020).
𝑂𝑙,
|     |     |     |     |     |     |     |     | - (HU-)PPO |      | utilizes  | the statistical | U-shape   | to    | determine  | and     |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------- | ---- | --------- | --------------- | --------- | ----- | ---------- | ------- |
|     |     |     |     |     |     |     |     | within     | each | interval, | the             | allocated | order | is equally | divided |
amongeveryminute,i.e.𝑂𝑙=𝑂𝑙∕20forall𝑡∈{0,…,19},where
𝑡
Fig. 5. The VAA’s are depicted in dark blue for the best-case scenarios of uniform PPOisfinallyusedtodistributeorderswithintheminute.
distributionandU-shapedistributionorderallocationschemes,whiletheirworst-case - HULisourproposeddual-levelapproachwhichusesthestatis-
| counterparts | are shown | in brown. | The | uniform distribution |     | and U-shape | distribution |     |     |     |     |     |     |     |     |
| ------------ | --------- | --------- | --- | -------------------- | --- | ----------- | ------------ | --- | --- | --- | --- | --- | --- | --- | --- |
ticalU-shapetodetermine𝑂𝑙.
| allocate total | orders | to be | executed | per minute | in the same | manner | as PPO and |     |     |     |     |     |     |     |     |
| -------------- | ------ | ----- | -------- | ---------- | ----------- | ------ | ---------- | --- | --- | --- | --- | --- | --- | --- | --- |
- TULisourproposeddual-levelapproachthatemploystheTrans-
(HU-)PPO,respectively..
formermodeltodetermine𝑂𝑙.
- TURusesRNNstoallocateorderswithineachinterval(second
2. Commissionsandexchangefeesareignored. level) instead of LSTMs. Other configurations are the same as
| 3. Ourmodel’sordersaretradedimmediatelywithoutorderarrival |     |     |     |     |     |     |     | thatofTUL. |     |     |     |     |     |     |     |
| ---------------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | ---------- | --- | --- | --- | --- | --- | --- | --- |
delays.
Weevaluatetheperformanceofthesemethodsalongsideourproposed
Webelievethattheaboveassumptionsarereasonableasweconsider dual-levelapproachbasedontheVAAmetricin(7).Aswementioned
relatively small total orders in comparison to the daily total market inSection3.3.3,theVAAdirectlymeasurestheabilityofatradingstrat-
volumesofthestocks. egytocloselytrackandapproximatethedailyVWAP.Consequently,to
comprehensivelyassessabilityofeachmethodtoconsistentlytrackthe
4.1.2. Datasets daily VWAP over an extended period, we analyze both the mean and
Ourmillisecondtradeandlimitorderbook(LOB)tickdataisfrom variationoftheVAAmetric.
| the Korea | Exchange | (KRX). | We  | use the daily | trade | and LOB | data of |     |     |     |     |     |     |     |     |
| --------- | -------- | ------ | --- | ------------- | ----- | ------- | ------- | --- | --- | --- | --- | --- | --- | --- | --- |
Samsung Electronics (SE), SK Hynix (SH), Kia Corporation (KC) and 4.2. Experimentresults
| POSCO Holdings |        | Inc (PH) | from       | January 1st, | 2021  | to December | 31st,    |         |             |     |           |         |             |             |     |
| -------------- | ------ | -------- | ---------- | ------------ | ----- | ----------- | -------- | ------- | ----------- | --- | --------- | ------- | ----------- | ----------- | --- |
|                |        |          |            |              |       |             |          | We test | our models, |     | which are | trained | for 100,000 | iterations, | by  |
| 2021. Our      | choice | of the   | four firms | in the       | KOSPI | index is    | based on |         |             |     |           |         |             |             |     |
conductingtradingsimulationsinthebuyingdirection.Wefirstverify
| their liquidity | and | trade | volume | and variety | in  | market capitalization |     |     |     |     |     |     |     |     |     |
| --------------- | --- | ----- | ------ | ----------- | --- | --------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
andindustry.Inordertoevaluatetheeffectivenessofourapproachin the claim of the ineffectiveness of short trading horizons when order
relationtotradingvolume,wehaveoptedtoselectstockswitharange allocation does not consider the appropriate distribution method for
oftradevolumes.Specifically,weselectedstocksrepresentingdifferent volumesfromlongtradinghorizons.
tradevolumelevels—SEforlarge,SHandKCformedium,andPHfor Todothis,weexaminedtwodistinctscenarios:firstly,weuniformly
small volumes. The average daily trading volumes for each stock can distributed volumes from a long trading horizon into a short trading
be found in Table 2. We divide the data into two sets, using January horizon and secondly, we employed a statistical U-shape distribution
1st to September 30th as training data and October 1st to December to allocate volumes from a long trading horizon into a short trading
|     |     |     |     |     |     |     |     | horizon. Note | that | the | volumes | from a long | trading | horizon | mean |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------- | ---- | --- | ------- | ----------- | ------- | ------- | ---- |
31stastestdata.
dailytotalorders,whileashorttradinghorizonrepresentsa1-minute
| The millisecond |     | raw | data is | preprocessed | by  | first dividing | them |     |     |     |     |     |     |     |     |
| --------------- | --- | --- | ------- | ------------ | --- | -------------- | ---- | --- | --- | --- | --- | --- | --- | --- | --- |
tradinghorizon.Tocomparethebestandworstcases,wedeliberately
| into groups | of 5    | s, and   | extracting | data that   | represents | each     | 5-second |          |            |     |            |         |             |               |     |
| ----------- | ------- | -------- | ---------- | ----------- | ---------- | -------- | -------- | -------- | ---------- | --- | ---------- | ------- | ----------- | ------------- | --- |
|             |         |          |            |             |            |          |          | executed | trades for | the | lowest and | highest | buy orders, | respectively, |     |
| interval.   | We take | the last | LOB        | data of the | 5-second   | interval | to con-  |          |            |     |            |         |             |               |     |
structtheMDPstateandthe5-secondVWAPandtotaltradedvolume within the 1-minute trading horizon. Fig. 5 displays the comparison
for calculating the daily VWAP. The statistical U-shape statistics are oftheVAA’sonthetestdates.Wecanobservethattheworstcasefor
constructed by taking the year average of the U-shape ratios of the U-shapedistributionoutperformsthebestcaseforuniformdistribution.
20-minute intervals in the day. The data utilized in generating 𝐷0 Specifically,theVAA’smeanoftheworstcaseforU-shapedistribution
𝑖𝑛
showninSection3.3arethetop5bid/askvolumeaveragesoftheraw is10bpswhiletheVAA’smeanofthebestcaseforuniformdistribution
pre-marketdata(from08:30:00to09:00:00). is 11 bps. This result suggests that in the context of VWAP, the use
|     |     |     |     |     |     |     |     | of deep reinforcement |            |     | learning | does not    | significantly | improve | the       |
| --- | --- | --- | --- | --- | --- | --- | --- | --------------------- | ---------- | --- | -------- | ----------- | ------------- | ------- | --------- |
|     |     |     |     |     |     |     |     | approximation         | capability |     | without  | taking into | account       | the     | long-term |
4.1.3. Comparisonandablationstudy
|     |     |     |     |     |     |     |     | trading horizon. |     | This implies | that | in order | to effectively | leverage | the |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------------- | --- | ------------ | ---- | -------- | -------------- | -------- | --- |
Weprovideabriefexplanationofallmethodsusedforperformance
benefitsofdeepRLinthecontextofVWAPs,itshouldbeimplemented
comparisonduringourexperimentshere.Thehyperparametersforeach
onalongertradinghorizonsuchasasingletradingday
methodcanbefoundinTable3.
|     |     |     |     |     |     |     |     | Secondly, | we  | present | Table 4 | and Fig. | 6 to demonstrate |     | how our |
| --- | --- | --- | --- | --- | --- | --- | --- | --------- | --- | ------- | ------- | -------- | ---------------- | --- | ------- |
- DQN is the method proposed in Ning et al. (2021). To apply dual-levelapproachsignificantlyimprovesperformance.
this model, we begin by dividing a given day into six 1-hour Table 4 reports the mean and standard deviation values of VAA’s
intervals,excludingthefinal20minofmarkethourstoreplicate produced by DQN, OPD, PPO, HUL and TUL across the test dates,
7

S.Kimetal. ExpertSystemsWithApplications252(2024)124263
Fig.6. VAAforthetopthreemodelsaspertheevaluationsinTable4,overaperiodof60testdaysforselectedstocks.ThemodelsincludeTUL,HUL,andthebest-performing
thirdmodelspecifictoeachstock.Notably,TULdemonstratesmoreconsistentVAAwithfewerfluctuationscomparedtoothermodelsacrossallstocks.
Fig.7. U-shapeaveragesontestdaysforgroundtruthsandTransformer-predictedvalues.Theinnerplotsdepicttheabsoluteerrorvaluesofthepredictionsforeachstock..
and the percentage of them that fall under 10 bps. These methods aresignificantforstockswithsmallvolumes,itischallengingtoobtain
areusedbypractitioners,tomeasurehowthetraderswereperforming accurate predictions of the daily volume distribution by only using a
relativetotheVWAP.Ourproposeddual-levelapproachdemonstrates statistical U-shape distribution. However, U-shape Transformer in the
significant improvements compared to using short trading horizons firstlevelcanproperlyreflectthedailyvariation,andTULoutperforms
and single-level neural networks. Although HUL occasionally yields bothmodelsanddemonstratesthelowestmeanandstandarddeviation
betteraccuracy,TULgenerallyhasmuchlessdeviationfromthedaily values.ThehistogramsforVAAdistributionsgivenbyDQN,OPD,PPO,
cumulative VWAP than all other methods. This becomes evident in HULandTULoneachstocktestdataaredrawninFig.9,whichdepicts
the less liquid stock PH stock, where HUL lagged behind both DQN theclusteringtendencyofaccuraciesproducedbyTUL.
andPPO.Inthiscase,DQNandPPOexhibitedlowermeanandlower Thirdly,wecomparetheperformanceofTULandTURtoexamine
standarddeviationcomparedtoHUL.Sincethedailyvolumevariations which neural network architecture is optimal for order allocation in
8

S.Kimetal.
ExpertSystemsWithApplications252(2024)124263
Table3
Hyperparameters for DQN, OPD, PPO, HUL and TUL. Note that ↘ indicates a linearly annealing learning rate schedule. Network configurations of DQN and OPD follow Ning
etal.(2021)andFangetal.(2021),respectively.
| Hyperparameter       |     |     |     | DQN   |     | OPD   |     | PPO   |     |     | HUL   |     | TUL   |     |
| -------------------- | --- | --- | --- | ----- | --- | ----- | --- | ----- | --- | --- | ----- | --- | ----- | --- |
| Outeriterations      |     |     |     | 10000 |     | 10000 |     | 10000 |     |     | 10000 |     | 10000 |     |
| Inneriterations      |     |     |     | 20    |     | 10    |     | 10    |     |     | 10    |     | 10    |     |
| Batchsize            |     |     |     | 50    |     | 10    |     | 20    |     |     | 10    |     | 10    |     |
| PPOCLIP𝜀             |     |     |     | –     |     | 0.2   |     | 0.2   |     |     | 0.2   |     | 0.2   |     |
| Discountfactor𝛾      |     |     |     | –     |     | 1     |     | 1     |     |     | 1     |     | 1     |     |
| Numberoftrajectories |     |     |     | 1000  |     | 160   |     | 200   |     |     | 152   |     | 228   |     |
pereachouteriteration
| LSTMhiddendimension𝐻      |     |     |     | –    |     | –    |     | 128       |     |     | 128       |     | 129       |     |
| ------------------------- | --- | --- | --- | ---- | --- | ---- | --- | --------- | --- | --- | --------- | --- | --------- | --- |
| LSTMmodellearningrate     |     |     |     | –    |     | –    |     | 5e−5↘1e−5 |     |     | 5e−5↘1e−5 |     | 5e−5↘1e−5 |     |
| DQNorOPDmodellearningrate |     |     |     | 1e−4 |     | 1e−4 |     | –         |     |     | –         |     | –         |     |
| U-shapeEncoder            |     |     |     | –    |     | –    |     | –         |     |     | –         |     | 20        |     |
inputvectordimension𝑁
| U-shapeEncoder |     |     |     | –   |     | –   |     | –   |     |     | –   |     | 148 |     |
| -------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
embeddingdimension
| TransformerEncoder |     |     |     | –   |     | –   |     | –   |     |     | –   |     | 4   |     |
| ------------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
&Decodernumberofheads
| TransformerEncoder |     |     |     | –   |     | –   |     | –   |     |     | –   |     | 1   |     |
| ------------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
&Decodernumberoflayers
| TransformerEncoder |     |     |     | –   |     | –   |     | –   |     |     | –   |     | 128 |     |
| ------------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
&DecoderPFFNdimension
| Transformermodellearningrate |     |     |     | –   |     | –   |     | –   |     |     | –   |     | 1e−3↘2e−4 |     |
| ---------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --------- | --- |
approximatestheU-shapedistributionforthetestdatesaccuratelyand
|     |     |     |     |     |     |     | uses predicted | values      | for order | allocation. |     | Thus, TUL | becomes            | more |
| --- | --- | --- | --- | --- | --- | --- | -------------- | ----------- | --------- | ----------- | --- | --------- | ------------------ | ---- |
|     |     |     |     |     |     |     | adaptive       | in tracking | the daily | cumulative  |     | VWAP      | as it incorporates |      |
predictionsoffluctuationsthatmayappearintheU-shapedistribution
inthefuture.
WealsovalidatetheperformanceimprovementsofTULcompared
toothermodels,particularlyHUL,byexaminingTUL’sabilitytocap-
turedailyfluctuationsintheU-shape.Fig.8displaysthegroundtruth
U-shapeandTransformer-predicteddistributionsfortwosampledtest
|     |     |     |     |     |     |     | dates for   | the stock   | PH. We     | deliberately | choose        | PH             | since the    | perfor-   |
| --- | --- | --- | --- | --- | --- | --- | ----------- | ----------- | ---------- | ------------ | ------------- | -------------- | ------------ | --------- |
|     |     |     |     |     |     |     | mance gains | of TUL      | when       | compared     | to HUL        | were           | the largest  | for this  |
|     |     |     |     |     |     |     | particular  | stock,      | which has  | daily volume | distributions |                | that         | deviates  |
|     |     |     |     |     |     |     | the most    | from the    | average    | due to       | relatively    | low liquidity. |              | While the |
|     |     |     |     |     |     |     | U-shape     | Transformer | may        | occasionally | make          | erroneous      | predictions, |           |
|     |     |     |     |     |     |     | it quickly  | corrects    | itself and | adapts       | to changes    | in             | the daily    | volume    |
distribution,attemptingtoaccuratelyfollowratiochangesthroughout
theday.
5. Discussionandconclusion
|     |     |     |     |     |     |     | In this | study, | we propose | a dual-level |     | approach | to address | the |
| --- | --- | --- | --- | --- | --- | --- | ------- | ------ | ---------- | ------------ | --- | -------- | ---------- | --- |
problemoftrackingtheVWAPwithaccuracyandconsistency.Initially,
|     |     |     |     |     |     |     | as observed | in Fig. | 5, we | identified | that the | naive | uniform | distribu- |
| --- | --- | --- | --- | --- | --- | --- | ----------- | ------- | ----- | ---------- | -------- | ----- | ------- | --------- |
Fig.8. ThegroundtruthandTransformer-predictedU-shapeplotsfortwoselectedtest
datesforPH. Theinnerplotsdepicttheabsoluteerrorvaluesofthepredictions.. tion approach falls short in accurately tracking the daily VWAP. The
uniquenessofourmodelisinhowweimproveuponthisissuebytaking
|            |                            |        |       |                  |             |     | into account | the       | well-known     | U-shaped   | intraday | trading   | pattern. | In       |
| ---------- | -------------------------- | ------ | ----- | ---------------- | ----------- | --- | ------------ | --------- | -------------- | ---------- | -------- | --------- | -------- | -------- |
|            |                            |        |       |                  |             |     | the first    | stage, we | allocate       | the number | of       | orders to | execute  | in each  |
| the second | level. For SE, which       | is the | stock | with the highest | trading     |     |              |           |                |            |          |           |          |          |
|            |                            |        |       |                  |             |     | interval     | based     | on the U-shape | pattern.   | We       | consider  | two      | methods, |
| volume,    | TUR slightly underperforms |        | HUL   | and TUL and      | outperforms |     |              |           |                |            |          |           |          |          |
statisticalU-shapeandU-shapeTransformer,forimplementingthefirst
| all other | models. However, | it is evident | that | a significant | decline | in  |     |     |     |     |     |     |     |     |
| --------- | ---------------- | ------------- | ---- | ------------- | ------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
stage.ThenaiveapproachofusingthestatisticalU-shapedistribution
| performance | occurs for all | other stocks | with | lower trading | volumes. |     |     |     |     |     |     |     |     |     |
| ----------- | -------------- | ------------ | ---- | ------------- | -------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Thisperformancedegradationcanbeattributedtothelimitedcapabil- improvesperformance.However,ourexperimentsdemonstratethatthe
ity of the simpler RNN model when compared with the LSTM model U-shapeTransformerperformsevenbetter.Webelievethatthesuperior
to capture volume fluctuations in stocks that exhibit relatively lower performanceoftheU-shapeTransformerisduetoitsabilitytoadaptto
liquidity. unexpectedvariations,acapabilitythatthestatisticalU-shapemethod
Lastly, we demonstrate the effectiveness of the proposed U-shape lacks, as evidenced by Figs. 7 and 8. After the total daily orders are
TransformerusingFigs.7and8.Fig.7showstheplotsoftheground allocatedtoeachintervalinthefirststage,theLSTMmodelisusedin
truthU-shapeandTransformer-predictedU-shapevalues,wheretheU- thesecondstagetodeterminehowtheordersineachintervalshould
shaperatiovaluesforeach20-minuteintervalwereaveragedacrossall beexecuted.Oursimulationresultsshowthatthedual-levelapproach
consistentlyandaccuratelytracksthedailycumulativeVWAP.
testdates.WeobservethatourU-shapeTransformerisabletofollow
theU-shapepatternforthetestdatesonaverageforallstocks,which Whileourmethoddemonstratessuperiorperformancethroughex-
is in line with the comparable (and sometimes better) performance periments, a comprehensive mathematical understanding of its un-
of TUL against HUL. This is further supported by the fact that HUL derlying principles including uncertainty quantification remains to be
only utilizes the average U-shape ratio values of the past, while TUL explored. The limitations identified in our paper suggest potential
9

S.Kimetal.
ExpertSystemsWithApplications252(2024)124263
Fig.9. ThehistogramsoftheVAAdistributionsforeachmodelandstockonthetestdates.BothTULandHUL,employingtheU-shapeproperty,exhibitconsistentstabilityin
trackingtheVWAPforSE,SH,andKCstocks.Moreover,TULdemonstratesrobusttrackingforthePHstock,whichischaracterizedbysignificantdailyvolumefluctuations.
Table4
Table4summarizesthemean,standarddeviation,andpercentageofVAAswithintherange[0,10bps].TheMeanandStandardDeviationarerepresentedinunitsofbasispoints
(bps),andthe%in10bpsarerepresentedinunitsofpercentages.ForOPD,weexcludedVAAsabovethe80thpercentileforfaircomparisonssinceourtrainingdatasetwas
smallerthanthatusedinFangetal.(2021).ForTUR,weexcludedextremeoutliersforSH,KA,andPH.
| Stocks | Metric            | OPD   | DQN   | PPO   | HUL   | TUL   | TUR    |
| ------ | ----------------- | ----- | ----- | ----- | ----- | ----- | ------ |
|        | Mean              | 26.27 | 11.53 | 12.00 | 8.77  | 7.13  | 9.43   |
| SE     | StandardDeviation | 18.50 | 8.84  | 8.88  | 7.17  | 2.44  | 9.56   |
|        | %in10bps          | 25.00 | 57.38 | 49.18 | 65.57 | 88.52 | 65.57  |
|        | Mean              | 38.55 | 21.62 | 22.57 | 7.13  | 7.65  | 48.56  |
| SH     | StandardDeviation | 24.90 | 17.71 | 16.92 | 2.81  | 3.63  | 33.51  |
|        | %in10bps          | 10.42 | 30.00 | 25.00 | 91.67 | 88.33 | 5.45   |
|        | Mean              | 27.79 | 13.16 | 12.47 | 6.59  | 6.03  | 149.32 |
| KA     | StandardDeviation | 18.76 | 10.25 | 11.55 | 8.36  | 3.30  | 111.64 |
|        | %in10bps          | 18.75 | 46.67 | 56.67 | 90.00 | 86.67 | 9.43   |
|        | Mean              | 32.44 | 14.84 | 14.64 | 22.52 | 12.27 | 377.28 |
| PH     | StandardDeviation | 20.89 | 13.47 | 12.63 | 47.11 | 7.90  | 267.23 |
|        | %in10bps          | 18.75 | 40.98 | 40.98 | 32.79 | 44.26 | 1.85   |
directionsforfutureresearch.Additionally,wenotethatourproposed CRediTauthorshipcontributionstatement
methodentailsthefollowingethicalimplications.Firstly,ourapproach
providesinvestorswithnewoptionsfordealingwithmarketvolatility, Soohan Kim: Writing – original draft, Methodology, Software.
helpingtoavoidmarketimpactandtheriskoffront-runnersexploiting Jimyeong Kim: Writing – original draft, Data curation, Conceptual-
theirstrategies.Secondly,improvedabilityofourexecutionmodelto ization, Visualization. Hong Kee Sul: Writing – review & editing, In-
track the VWAP contributes to mitigating market impact, which can vestigation.YoungjoonHong:Writing–review&editing,Supervision,
potentiallyenhancemarkettransparencyandfairness. Projectadministration.
10

S.Kimetal.
ExpertSystemsWithApplications252(2024)124263
Declarationofcompetinginterest Kakade,S.M.,Kearns,M.,Mansour,Y.,&Ortiz,L.E.(2004).Competitivealgorithms
|     |     |     |     |     | for VWAP | and limit order trading. | In Proceedings of | the 5th ACM conference | on  |
| --- | --- | --- | --- | --- | -------- | ------------------------ | ----------------- | ---------------------- | --- |
Theauthorsdeclarethefollowingfinancialinterests/personalrela- electroniccommerce(pp.189–198).
Kato,T.(2015).VWAPexecutionasanoptimalstrategy.JSIAMLetters,7,33–36.
| tionships which | may be considered | as potential | competing | interests: |     |     |     |     |     |
| --------------- | ----------------- | ------------ | --------- | ---------- | --- | --- | --- | --- | --- |
Lin,S.,&Beling,P.A.(2019).Optimalliquidationwithdeepreinforcementlearning.
YoungjoonHongreportswasprovidedbySungkyunkwanUniversity.
InProceedingsofthe33rdconferenceonneuralinformationprocessingsystems,deep
reinforcementlearningworkshop.Vancouver,Canada.
Lin,S.,&Beling,P.A.(2020).Anend-to-endoptimaltradeexecutionframeworkbased
Dataavailability
onproximalpolicyoptimization..InIJCAI(pp.4548–4554).
|     |     |     |     |     | Macri, A., & | Lillo, F. (2024). Reinforcement | learning | for optimal execution | when |
| --- | --- | --- | --- | --- | ------------ | ------------------------------- | -------- | --------------------- | ---- |
Datawillbemadeavailableonrequest. liquidityistime-varying.arXivpreprintarXiv:2402.12049v2.
Madhavan,A.(2002).VWAPstrategies.Trading,1,32–39.
Acknowledgments Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T., Harley, T., Silver, D., &
|     |     |     |     |     | Kavukcuoglu, | K. (2016). Asynchronous | methods for | deep reinforcement | learning. |
| --- | --- | --- | --- | --- | ------------ | ----------------------- | ----------- | ------------------ | --------- |
InInternationalconferenceonmachinelearning(pp.1928–1937).PMLR.
TheworkofY.HongwassupportedbyBasicScienceResearchPro-
|     |     |     |     |     | Mnih, V., Kavukcuoglu, | K., Silver, D., | Graves, A., Antonoglou, | I., Wierstra, | D., & |
| --- | --- | --- | --- | --- | ---------------------- | --------------- | ----------------------- | ------------- | ----- |
gramthroughtheNationalResearchFoundationofKorea(NRF)funded
Riedmiller,M.(2013).Playingatariwithdeepreinforcementlearning.URL:http:
bytheMinistryofEducation,RepublicofKorea(NRF-2021R1A2C1093 //arxiv.org/abs/1312.5602. Cite arxiv:1312.5602Comment: NIPS Deep Learning
579)andbytheKoreagovernment(MSIT)(RS-2023-00219980). Workshop2013.
|            |     |     |     |     | Mnih, V., Kavukcuoglu,   | K., Silver, D.,    | Graves, A., Antonoglou,       | I., Wierstra, | D., & |
| ---------- | --- | --- | --- | --- | ------------------------ | ------------------ | ----------------------------- | ------------- | ----- |
|            |     |     |     |     | Riedmiller,              | M. (2013). Playing | atari with deep reinforcement | learning.     | arXiv |
| References |     |     |     |     | preprintarXiv:1312.5602. |                    |                               |               |       |
Nevmyvaka,Y.,Feng,Y.,&Kearns,M.(2006).Reinforcementlearningforoptimized
Almgren,R.,&Chriss,N.(2001).Optimalexecutionofportfoliotransactions.Journal trade execution. In Proceedings of the 23rd international conference on machine
| ofRisk,3,5–40. |     |     |     |     | learning(pp.673–680). |     |     |     |     |
| -------------- | --- | --- | --- | --- | --------------------- | --- | --- | --- | --- |
Berkowitz,S.A.,Logue,D.E.,&Noser,E.A.,Jr.(1988).Thetotalcostoftransactions Ning, B., Lin, F. H. T., & Jaimungal, S. (2021). Double deep q-learning for optimal
ontheNYSE.TheJournalofFinance,43(1),97–112. execution.AppliedMathematicalFinance,28(4),361–380.
Bertsimas, D., & Lo, A. W. (1998). Optimal control of execution costs. Journal of Pan,F.,Zhang,T.,Luo,L.,He,J.,&Liu,S.(2022).Learncontinuously,actdiscretely:
financialmarkets,1(1),1–50. Hybrid action-space reinforcement learning for optimal execution. In IJCAI (pp.
| Bialkowski,J.,Darolles,S.,&Fol,G.L.(2008).ImprovingVWAPstrategies:Adynamic |     |     |     |     | 3912–3918). |     |     |     |     |
| -------------------------------------------------------------------------- | --- | --- | --- | --- | ----------- | --- | --- | --- | --- |
volumeapproach.JournalofBanking&Finance,32,1709–1722. Podobnik,B.,Horvatic,D.,Petersen,A.M.,&Stanley,H.E.(2009).Cross-correlations
Byun,W.,Choi,B.,Kim,S.,&Jo,J.(2023).Practicalapplicationofdeepreinforcement betweenvolumechangeandpricechange.ProceedingsoftheNationalAcademyof
learningtooptimaltradeexecution.FinTech,2,414–429. Sciences,106(52),22079–22084.
Fang, Y., Ren, K., Liu, W., Zhou, D., Zhang, W., Bian, J., Yu, Y., & Liu, T.-Y. Schulman,J.,Levine,S.,Abbeel,P.,Jordan,M.,&Moritz,P.(2015).Trustregionpolicy
(2021). Universal trading for order execution with oracle policy distillation. 35, optimization.InInternationalconferenceonmachinelearning(pp.1889–1897).PMLR.
InProceedingsoftheAAAIconferenceonartificialintelligence(1),(pp.107–115). Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal
Goodhart,C.A.,&O’Hara,M.(1997).Highfrequencydatainfinancialmarkets:Issues policyoptimizationalgorithms.arXivpreprintarXiv:1707.06347.
andapplications.JournalofEmpiricalFinance,4,73–114. Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction. MIT
| Hendricks, D., | & Wilcox, D. (2014). | A reinforcement | learning | extension to the | Press. |     |     |     |     |
| -------------- | -------------------- | --------------- | -------- | ---------------- | ------ | --- | --- | --- | --- |
almgren-chrissframeworkforoptimaltradeexecution.In2014IEEEconferenceon Vaswani,A.,Shazeer,N.,Parmar,N.,Uszkoreit,J.,Jones,L.,Gomez,A.N.,Kaiser,Ł.,
computationalintelligenceforfinancialengineering&economics(CIFEr)(pp.457–464). &Polosukhin,I.(2017).Attentionisallyouneed.AdvancesinNeuralInformation
| IEEE. |     |     |     |     | ProcessingSystems,30. |     |     |     |     |
| ----- | --- | --- | --- | --- | --------------------- | --- | --- | --- | --- |
Huberman,G.,&Stanzl,W.(2005).Optimalliquiditytrading.ReviewofFinance,(2), Watkins,C.J.,&Dayan,P.(1992).Q-learning.MachineLearning,8(3),279–292.
165–200. Williams,R.J.(1992).Simplestatisticalgradient-followingalgorithmsforconnectionist
Jain,P.C.,&Joh,G.-H.(1988).Thedependencebetweenhourlypricesandtrading reinforcementlearning.ReinforcEmentLearning,5–32.
volume.TheJournalofFinancialandQuantitativeAnalysis,23(3),269–283.
11

---

## Content from Previous Extraction (not in markitdown output)

### Visual/Chart/Graph Descriptions

#### 1

S.Kimetal. Expert Systems With Applications252(2024)124263
Table1 objectiveistofindastrategythatmaximizesorminimizestheaverage
The table summarizes previous works on optimal trade execution utilizing the RL executionprice𝑃̄, whichiscalculatedas
framework. Note that DQN and PPO are learning algorithms of deep RL. Notably,
ourworkdistinguishesitselfbyfocusingonalong-termhorizon. 𝑃̄ = 𝑇 ∑ −1𝑂 𝑡𝑝.
RLmethod Horizon 𝑂 𝑡
𝑡=0
Nevmyvakaetal.(2006) Q-learning Short
Hendricksand Wilcox (2014) Q-learning Short While many financial firms pursue profits through trading, there are
Linand Beling (2019) DQN Short also numerous firms within the financial industry that prioritize con-
Ningetal.(2021) DQN Short sistentlytrackingthe VWAPovermaximizingtradingprofit. Thefocus
Linand Beling (2020) PPO Short
ofourpaperistailoredforthissecondgroupoftraders. The VWAPis
Fangetal.(2021) PPO Short
calculatedas
Panetal.(2022) PPO Short
Thiswork PPO Long ∑𝑇−1𝑝𝑞
VWAP= 𝑡=0 𝑡 𝑡 ,
∑𝑇−1𝑞
𝑡=0 𝑡
where 𝑞 is the execution volume determined by the market. There-
2. We further propose the U-shape Transformer model to capture 𝑡
fore, theprimaryobjectiveofthispaperistominimizethedifference
theday-to-dayoscillationsoftheintraday U-shapedistribution. betweenthe VWAPand𝑃̄, usingadual-levelapproach.
Whilethe U-shapedpatterniswell-known, relyingonfixedsta-
tisticalestimatesfailstoadapttosuddenfluctuationsorshiftsin
3. Methoddescription
marketconditions. The U-shape Transformerdirectlylearnsaro-
bustrepresentationfromrecentdata, leveragingself-attentionto
Inthissection, wedescribeourdual-levelapproachandformulation
automaticallyadapttosuddendynamicsintheintraday U-shape
oftheoptimaltradeexecutionproblemunderthelensofreinforcement
pattern.
learning. Wealsoprovidedetailsonthearchitectureofthe Transformer
3. Weconductablationstudiestoshowthestrengthsofourmeth-
and LSTM neural networks and how they are integrated into our
odsinapproximatingthedailycumulative VWAP.Furthermore,
proposedmethod.
we demonstrate that the methods employed to distribute vol-
umes from long trading horizons into short horizons can have 3.1. Dual-levelapproach
asignificantimpact. Inaddition, weintroducethe VAA, which
provides a more practical way to evaluate how well a trading Our approach involves two stages of distributing orders. In the
strategy tracks the daily VWAP. Our proposed dual-level ap- first stage, we allocate the total daily orders 𝑂 into 𝐿 intervals. In
proachcombining U-shape Transformersand RLachievesstate- thesecondstage, wefurtherdividetheordersfromeachintervalinto
of-the-artperformanceonthe VAA(see Table1). smaller executable orders using RL. Our approach begins by dividing
the market hours of a day into 𝐿 intervals. For each interval 𝑙, we
let 𝑂𝑙 as the total order to be executed, ensuring that the sum of all
2. Backgrounds
intervalordersequalsthetotaldailyorder, or
∑𝐿−1𝑂𝑙=𝑂.Wesuggest
𝑙=0
two methods for implementing the first stage. The naive approach
Inthissection, wediscussthebasicsofthelimitorderbook (LOB)
is to adhere to the statistical U-shape trade volume distribution for
and the optimal trade execution process. Most modern financial ex-
each interval obtained from past data (e.g. the historical average of
changes, including the Korea Stock Exchange (KRX), offer electronic
the past year). A more advanced method would be to use the output
tradingplatformswithaccesstolimitorderbooks. Tradersoftenrelyon
(predicted U-shape distribution) generated by a U-shape Transformer
theinformationprovidedbythelimitorderbooktodevelopsuccessful
model. In the latter case, 𝑂𝑙 is calculated progressively by referring
tradingstrategies.
only to market information from past days and previous intervals
within the day. For such an 𝑙th interval, which is comprised of 𝑇
2.1. Limitorderbook timesteps, the LSTMmodeloutputs𝑂𝑙, whichisthenumberoforders
𝑡
to be executed at timestep 𝑡 such that ∑𝑇−1𝑂𝑙 = 𝑂𝑙. Once 𝑂𝑙 is
Infinancialmarkets, alimitorderisanordertobuyorsellafixed 𝑡=0 𝑡 𝑡
decided, it is averaged over 𝐼 execution steps in the corresponding
number of shares at a specified price. This order becomes part of the
𝑡th subinterval. In other words, for each execution step, which is
limitorderbook, whichrecordsalllimitordersforaspecificstock. The set as every 5 s, the final number of orders executed is 𝑂𝑙∕𝐼. The
𝑡
bidpriceisthehighestpriceabuyeriswillingtopayforthesecurity,
choice of averaging is supported by our findings in Section 4.2 that
andtheaskpriceisthelowestpriceinwhichtheselleriswillingtosell.
there is no significant impact in approximating the VWAP since the
Amarketorder, bycontrast, isanordertobuyorsellafixednumber finalintervalistoosmall. Furthermore, in Section4.2, weverifythat
ofsharesatthecurrentmarketpricewithoutstatingtheprice. Whena incorporatinglong-termhorizonsyieldssuperiorefficiency. Therefore,
marketorderisplaced, itismatchedbythebestavailablelimitorders althoughadeepreinforcementlearningframeworkcanoffermarginal
inthe LOB.The LOBisimportantforoptimaltradeexecutionandhelps performanceimprovements, wehaveoptedforthesimplicityofusing
traderstracksupplyanddemandforastock, makingiteasiertoidentify theuniformdistributionforthefinalinterval. Aschematicillustration
thebesttimeandpricetobuyorsellshares. ofthisdesignisprovidedin Fig.2.
2.2. Optimaltradeexecution 3.2. MDPformulationforoptimalexecution
Optimal trade execution refers to the process of buying or selling In this section, we explain our Markov Decision Process (MDP)
afinancialassetwhileachievingthedesiredobjective. Generally, this formulationofoptimaltradeexecution. AMDPistypicallyrepresented
problem formulates as follows: Within a timeframe of 𝑇 timesteps, using the tuple (, , , 𝑟, 𝛾), where  is the state space,  is the
0,1,…,𝑇 −1, atraderwhopossessesaninventoryof𝑂sharesmustbuy action space,  is the transition probability, 𝑟 is the reward function,
orselltheentiretyoftheinventory. Foreachtimestep𝑡∈{0,1,…,𝑇− and𝛾isthediscountfactor. Ourconfigurationsofthestatespaceand
1}, thetraderdeterminesthenumberofsharestoorder𝑂 basedonthe actionspacearesimilartothatof Linand Beling (2020).Itisworth
𝑡
informationfrom LOB, andcarriesoutthetradeatanexecutionprice notingthatthe MDPimplementationinthispaperisappliedonaper
𝑝. For the group of traders seeking to maximize trading profit, their 𝑙thintervalbasisforagivenday.
𝑡
3

#### 2

S.Kimetal. Expert Systems With Applications252(2024)124263
Fig.2. Aschematicillustrationofourdual-levelapproach.
3.2.1. State Ourrewardfunctioncompares𝑂𝑙∗ and𝑂𝑙 directlyforagiven𝑎𝑙 as
𝑡 𝑡 𝑡
Thestate𝑠𝑙∈consistsofbothpublicandprivateinformation. The follows:
𝑡
publicstateismadeupofthetop5 bidandaskprices, alongwiththeir ⎧ 1 𝑀𝑙<0.01
associated volumes, while the private state includes the elapsed time 𝑟(𝑎𝑙)∶= ⎪ 0 0.0 𝑡 1≤𝑀𝑙<0.05 (2)
𝑡 ⎨ 𝑡
and the current remaining volume to be executed. The elapsed time ⎪ −1 otherwise
rangesfrom0 to𝑇−1 andthecurrentremainingvolumeattimestep𝑡 ⎩
isequalto𝑂𝑙− ∑𝑡 𝑗 − = 1 0 𝑂 𝑗 𝑙. where𝑀𝑙∶= |𝑂 𝑡 𝑙−𝑂 𝑡 𝑙∗ |.
𝑡 𝑂𝑙∗
𝑡
Once MDPisdetermined, thegoalofpolicy-based RListofindan
3.2.2. Action optimalpolicyparameter𝜃∗, i.e.
Our LSTMmodeloutputsapolicy𝜋(⋅|𝑠𝑙
𝑡
)∈R21, whichisadiscrete
𝜃∗=
c { e a 0 x t , e e 0 c g . u 1 o t , e r 0 i , . c 2 a w , l … it p h r , o 2 𝑂 b } 𝑡 𝑙 ab i = s ili s t 𝑎 a y 𝑙 𝑡 m 𝑂 𝑇 d p 𝑙 i . l s e t T r d i o b t u e o t n io s d u n e r . t e e F r t r m h o a i m n t e t t h h t e h is e , to a n t n a u l m a n c b u t e i m o r n b o e 𝑎 f r 𝑙 𝑡 o o ∈ r f d o e  r r s de ∶ t r = o s argmax E [ ∑ 𝑡= 𝑇 0 𝛾𝑡𝑟 𝑡 | | | | 𝑎 𝑡 ∼𝜋 𝜃 (⋅|𝑠 𝑡 ),𝑠 𝑡+1 ∼(⋅|𝑠 𝑡 ,𝑎 𝑡 ) ] , (3)
executedovertheentireperiodisequalto𝑂𝑙, thatis ∑𝑇 𝑡= − 0 1𝑂 𝑡 𝑙=𝑂𝑙, we where𝑟 𝑡 =𝑟(𝑎 𝑡 ).
imposethefollowingtworestrictions.
3.3. Neuralnetworkarchitectureandtraining
• If ∑𝑗 𝑂𝑙>𝑂𝑙 forsome𝑗∈{0,1,2,…,𝑇 −1}, then
𝑡=0 𝑡
{ 𝑂𝑙− ∑𝑗−1𝑂𝑙, if𝑖=𝑗 We now turn to the architectural details of the Transformer and
𝑂𝑙= 𝑡=0 𝑡 LSTM models and elaborate on how they are used in our dual-level
𝑖 0, if𝑗<𝑖≤𝑇 −1
approach.
• 𝑂𝑙 isequaltotheremainingvolumetobeexecutedatthelast
tim 𝑇− e 1 stepforeachinterval (i.e.0≤𝑙≤𝐿−1) 3.3.1. Level1:Transformerforglobal U-shapeapproximation
Theobjectiveofthe Transformermodelistoprogressivelypredict
the U-shaperatioforeachintervalinagivenday. Weusethestructure
Theserestrictionsareinlinewiththefactthatitisessentialforbrokers
ofthe Transformer Encoderand Decoderasproposedin Vaswanietal.
topromptlyacquireorliquidateallsharesrequestedbytheircustomers.
(2017).Theoverallarchitectureofthe Transformermodelthatweuse
isdepictedin Fig.3.
3.2.3. Reward U-shape Encoder. The U-shape Encoder iscomposedofa Trans-
The reward function is designed to encourage the agent (LSTM former Encoder and 𝐿 linear layers, all of which map to the same
model) to generate actions that result in orders that are close to the dimension. Thismoduleisresponsibleforlearningthegeneraldynam-
target volume 𝑂𝑙∗, in order to track the VWAP. This is achieved by ics of the U-shape distribution. The input 𝐸 𝑖𝑛 is a sequence of length
𝑡
usingthepricecalculatedfromtheexecutedorders. Here,𝑂𝑙∗ denotes 𝐿 of vectors each containing 𝑁 historical daily volume ratios of the
𝑡
corresponding interval in the sequence. The days from which these
the desired VWAP order at timestep 𝑡 in the 𝑙th interval such that
𝑂𝑙= ∑𝑇−1𝑂𝑙∗.Thedaily VWAPcanbeobtainedby ratios are drawn are randomly selected. Each vector in the sequence
𝑡=0 𝑡 is processed by a different linear layer that serves as an embedding
𝑉𝑊𝐴𝑃 𝑑𝑎𝑦 = 𝐿 ∑ −1𝑇 ∑ −1𝑂 𝑂 𝑡 𝑙∗ 𝑝𝑙 𝑡 , (1) t t o he le e a m rn be t d h d e in u g ni l q a u y e er f , ea th tu e re i s np o u f t e s a e c q h ue in n t c e e rv i a s l. th A e f n ter pr p o a c s e s s in se g d th b r y ou t g h h e
𝑙=0 𝑡=0 Transformer Encoder, resultingintheoutput𝐸 .
𝑜𝑢𝑡
where 𝑝𝑙 is the associated average market traded price of the corre- U-shape Decoder. While  learns the general dynamics of the U-
𝑡
sponding𝑡thsubinterval. shape distribution, the U-shape Decoder  focuses on handling the
4

#### 3

S.Kimetal. Expert Systems With Applications252(2024)124263
3.3.3. Trainingprocedure
U-shape Transformer. Since the goal of the first level of order al-
locationistoapproximatethe U-shapedistributionwhilealsotracking
thedailycumulative VWAP, thetrainingobjectiveofthe Transformer
modelistominimize
𝐽 ∶=E [ 𝑐 1 𝐿 ∑ −1( 𝑢𝑙 −𝑢𝑙 )2 +𝑐 𝑉𝐴𝐴 ] (6)
𝑇𝐹 𝑑𝑎𝑦𝑠 𝐿 𝑡𝑟𝑢𝑒 𝑝𝑟𝑒𝑑 2
𝑙=0
where𝑐 and𝑐 arecoefficientsand
1 2
VAA∶= | | | | | 𝑀𝑃 𝑑 𝑉 𝑎𝑦 𝑊 − 𝐴 𝑉 𝑃 𝑊 𝑑𝑎 𝐴 𝑦 𝑃 𝑑𝑎𝑦| | | | | (7)
denotesthe VWAPApproximation Accuracy (VAA), whichwewillfur-
theruseasageneralmetricforperformanceevaluationinexperiments,
and
𝑀𝑃 ∶=
𝐿
∑
−1𝑇
∑
−1𝑂
𝑡
𝑙
𝑝𝑙
Fig. 3. The architecture of the U-shape Transformer. The U-shape Decoder in this 𝑑𝑎𝑦 𝑂 𝑡
𝑙=0 𝑡=0
figurerepresentsthe𝑙thdecodingstep.
is the price yielded by our model. This is one of the measures used
in practice to gauge how close a trader has traded closer to their
benchmark, the VWAP.Ifatraderhastradedatapricethatisexactly
day-to-dayvariationsofthedistribution. The U-shape Decoderiscom-
equal to the VWAP, then 𝑉𝐴𝐴 = 0. Thus lower VAA would be
posedofa Transformer Decoderand𝐿linearlayers, eachmappingto
better. Thefirstterminsidetheexpectationof (6) learnstomakeratio
differentdimensions. Incontrastto, theselinearlayersareappliedto
predictions closer to ground-truth values. The second term penalizes
theoutputofthe Transformer Decoder. the Transformermodelifthefinaldailyacquisitionpricegeneratedby
For every 𝑙th (𝑙 > 0) decoding step, both 𝐸 𝑜𝑢𝑡 and the cumulative thecombinationofthe Transformerand LSTMmodelsdeviatesfromthe
input sequence are fed as input to the Transformer Decoder, . We daily VWAP.Theprimaryobjectiveofthe U-shapedtransformerinthe
denote the cumulative input sequence as 𝐷 𝑖 𝑙 𝑛 = (𝐷 𝑖 𝑙 𝑛,𝑗 ) 0≤𝑗≤𝑙−1 , where firstlevelistodistributethedailytotalorderwithprecision. However,
𝐷𝑙 ∶= CONCAT(𝐷𝑗 ,ℎ𝑗 ). Here, 𝐷𝑗 represents the output of  relyingsolelyonthefirsttermofthelossfunctionmaynotbeenough
at 𝑖𝑛 t , h 𝑗 e𝑗thstep, andℎ 𝑜𝑢 𝑗 𝑡 𝑇 is −1 thelast LST 𝑜𝑢 M 𝑡 hiddenvectorfromthe𝑗th to accomplish this objective. This is because 𝑢𝑙 𝑡𝑟𝑢𝑒 does not consider
𝑇−1 the local information, such as the LOB, which can pose difficulties
interval. Theoutputfromthe Transformer Decoderatthe𝑙thdecoding
in tracking the daily VWAP accurately. Furthermore, if only the first
step is a vector of dimension 𝐿+𝐻, which is then passed through a
termin (6) istakenintoaccount, thereinforcementlearningframework
linearlayerthatreducesitsdimensionto𝐿−𝑙.Thesoftmaxfunctionis
wouldnotinfluencethefirstlevel, leadingtoindependentoperationof
appliedtopredicttheratiosoftheremaining𝐿−𝑙intervals, andthefirst
thefirstandsecondlevels. Toensurethatthemicroinformationhasan
𝑙 components are zero-padded. Note that we use ℎ𝑗 𝑇−1 to capture the appropriate impact on the first level, we incorporate the second term
day-to-day variations in the U-shape distribution, as it holds interval- intothelossfunction.
specificinformationforthatparticularday. Toensure ∑𝐿 𝑙= − 0 1𝑢𝑙 𝑝𝑟𝑒𝑑 =1, LSTM.Tofindtheoptimalparameter𝜃forthe LSTMmodel, weuse
thefinal𝑙thintervalvolumeratiopredictioniscalculatedas an actor–critic style PPO algorithm (Schulman et al., 2017), which is
( 𝑙−1 ) oneofthemostpopularon-policy RLalgorithms, asourbaselearner.
𝑢𝑙 = 1− ∑ 𝑢𝑗 ⋅𝐷𝑙 [𝑙]. (4) One of the major advantages of the PPO in this task is its ability to
𝑝𝑟𝑒𝑑 𝑝𝑟𝑒𝑑 𝑜𝑢𝑡
𝑗=0 adapt to changing market conditions and effectively learn from noisy
andhigh-dimensionaldata. Thisalgorithmusestheactorloss𝐽𝐶𝐿𝐼𝑃(𝜃)
It is worth noting that 𝐷 𝑖 0 𝑛 is constructed by applying an additional andthecriticlossfunction𝐽𝑉𝐹(𝜃), whicharedefinedasfollow PP s O :
PPO
linear layer that transforms the dimension of the vector holding the
t w o i p th 5 t b h i e d g / i a v s e k n v d o a lu y m to e ta a l v o er r a d g e e r s d o e f no p t r e e d -m a a s r 𝑂 ke , t th d e at c a or to re 𝐿 sp + on 𝐻 d . in T g h i e n r t e e f r o v r a e l , 𝐽 ∶ P 𝐶 = P 𝐿 O E 𝐼𝑃( [ 𝜃 m ) in ( 𝑞(𝜃)𝐴̂, clip (𝑞(𝜃),1−𝜀,1+𝜀)𝐴̂)] , (8)
𝑡 𝑡 𝑡 𝑡 𝑡
order𝑂𝑙 canbedeterminedastheproductof𝑂and𝑢𝑙 , i.e.
𝑝𝑟𝑒𝑑 and
[ ]
𝑂𝑙=𝑂×𝑢𝑙 𝑝𝑟𝑒𝑑 . (5) 𝐽 P 𝑉 P 𝐹 O (𝜃)=E 𝑡 (𝑉 𝑡 targ−𝑉 𝜃 (𝑠 𝑡 ))2 , (9)
where𝑞(𝜃)= 𝜋𝜃(𝑎𝑡|𝑠𝑡) ,
𝑡 𝜋𝜃old (𝑎𝑡|𝑠𝑡)
3.3.2. Level2:LSTMforlocalorderdistribution 𝑇−1
The objective of the LSTM model is to optimally allocate orders 𝑉 𝑡 targ= ∑ 𝛾𝑘−𝑡𝑟 𝑘 and𝐴̂ 𝑡 =𝑉 𝑡 targ−𝑉 𝜃 (𝑠 𝑡 ). (10)
withintheintervalsofagivenday. Thestructureofour LSTMmodel, 𝑘=𝑡
whichisoutlinedin Fig.4, issimilartothatof Linand Beling (2020). PPO’sgoalistofindaparameter𝜃 thatmaximizesthemainobjective
Given𝑠𝑙 𝑡 asinput, itfirstpassesthroughtwolinearlayerswithhidden function𝐽 PPO (𝜃), whichisdefinedby
d T i h m e e r n e s s i u o l n tin 1 g 28 v a e n ct d or is is th t e h n en co p n r c o a c t e e s n s a e t d ed by wi a t n h 𝑎 L 𝑙 𝑡 S − T 1 M an C d e 𝑟 ll 𝑙 𝑡− w 1 i ∶ t = h h 𝑟( i 𝑎 d 𝑙 𝑡 d − e 1 ) n . 𝐽 PPO (𝜃)=𝐽 P 𝐶 P 𝐿 O 𝐼𝑃(𝜃)−𝑐 3 𝐽 P 𝑉 P 𝐹 O (𝜃)+𝑐 4 E 𝑡 [𝑆[𝜋 𝜃 ](𝑠 𝑡 )], (11)
dimension 𝐻 to obtain ℎ𝑙, which is then passed separately through where𝑐 3 ,𝑐 4 arecoefficients, and E 𝑡 [𝑆[𝜋 𝜃 ](𝑠 𝑡 )]denotesanentropyloss
𝑡
two linear layers to produce the policy 𝜋(⋅|𝑠𝑙
𝑡
) and the value 𝑉(𝑠𝑙
𝑡
) term, where𝑆[𝜋 𝜃 ](𝑠 𝑡 ) isdefinedby
respectively. Wenotethatthevectorℎ𝑙 𝑇−1 whichisobtainedatthelast 𝑆[𝜋 𝜃 ](𝑠 𝑡 )∶=− ∑ 𝜋 𝜃 (𝑎|𝑠 𝑡 ) log𝜋 𝜃 (𝑎|𝑠 𝑡 ). (12)
timestep is used as input to the Transformer Decoder. The action 𝑎𝑙
𝑡
, 𝑎∈
reward 𝑟𝑙 𝑡 , and the number of orders to execute 𝑂 𝑡 𝑙 is calculated as is By incorporating an entropy term, PPO can incentivize the agent to
describedin Section3.2.The LSTMCellalsoreceivesℎ𝑙 and𝑐𝑙 as explorealternativeactions, preventgettingtrappedinsuboptimalpoli-
𝑡−1 𝑡−1
inputanditsoutputincludes𝑐𝑙 aswell. cies, and mitigate overfitting to the training data (Mnih et al., 2016;
𝑡
5

#### 4

S.Kimetal. Expert Systems With Applications252(2024)124263
Fig.6. VAAforthetopthreemodelsaspertheevaluationsin Table4, overaperiodof60 testdaysforselectedstocks. Themodelsinclude TUL, HUL, andthebest-performing
thirdmodelspecifictoeachstock. Notably, TULdemonstratesmoreconsistent VAAwithfewerfluctuationscomparedtoothermodelsacrossallstocks.
Fig.7. U-shapeaveragesontestdaysforgroundtruthsand Transformer-predictedvalues. Theinnerplotsdepicttheabsoluteerrorvaluesofthepredictionsforeachstock..
and the percentage of them that fall under 10 bps. These methods aresignificantforstockswithsmallvolumes, itischallengingtoobtain
areusedbypractitioners, tomeasurehowthetraderswereperforming accurate predictions of the daily volume distribution by only using a
relativetothe VWAP.Ourproposeddual-levelapproachdemonstrates statistical U-shape distribution. However, U-shape Transformer in the
significant improvements compared to using short trading horizons firstlevelcanproperlyreflectthedailyvariation, and TULoutperforms
and single-level neural networks. Although HUL occasionally yields bothmodelsanddemonstratesthelowestmeanandstandarddeviation
betteraccuracy, TULgenerallyhasmuchlessdeviationfromthedaily values. Thehistogramsfor VAAdistributionsgivenby DQN, OPD, PPO,
cumulative VWAP than all other methods. This becomes evident in HULand TULoneachstocktestdataaredrawnin Fig.9, whichdepicts
the less liquid stock PH stock, where HUL lagged behind both DQN theclusteringtendencyofaccuraciesproducedby TUL.
and PPO.Inthiscase, DQNand PPOexhibitedlowermeanandlower Thirdly, wecomparetheperformanceof TULand TURtoexamine
standarddeviationcomparedto HUL.Sincethedailyvolumevariations which neural network architecture is optimal for order allocation in
8

#### 5

S.Kimetal. Expert Systems With Applications252(2024)124263
Table3
Hyperparameters for DQN, OPD, PPO, HUL and TUL. Note that ↘ indicates a linearly annealing learning rate schedule. Network configurations of DQN and OPD follow Ning
etal.(2021) and Fangetal.(2021), respectively.
Hyperparameter DQN OPD PPO HUL TUL
Outeriterations 10000 10000 10000 10000 10000
Inneriterations 20 10 10 10 10
Batchsize 50 10 20 10 10
PPOCLIP𝜀 – 0.2 0.2 0.2 0.2
Discountfactor𝛾 – 1 1 1 1
Numberoftrajectories 1000 160 200 152 228
pereachouteriteration
LSTMhiddendimension𝐻 – – 128 128 129
LSTMmodellearningrate – – 5 e−5↘1 e−5 5 e−5↘1 e−5 5 e−5↘1 e−5
DQNor OPDmodellearningrate 1 e−4 1 e−4 – – –
U-shape Encoder – – – – 20
inputvectordimension𝑁
U-shape Encoder – – – – 148
embeddingdimension
Transformer Encoder – – – – 4
&Decodernumberofheads
Transformer Encoder – – – – 1
&Decodernumberoflayers
Transformer Encoder – – – – 128
&Decoder PFFNdimension
Transformermodellearningrate – – – – 1 e−3↘2 e−4
approximatesthe U-shapedistributionforthetestdatesaccuratelyand
uses predicted values for order allocation. Thus, TUL becomes more
adaptive in tracking the daily cumulative VWAP as it incorporates
predictionsoffluctuationsthatmayappearinthe U-shapedistribution
inthefuture.
Wealsovalidatetheperformanceimprovementsof TULcompared
toothermodels, particularly HUL, byexamining TUL’sabilitytocap-
turedailyfluctuationsinthe U-shape. Fig.8 displaysthegroundtruth
U-shapeand Transformer-predicteddistributionsfortwosampledtest
dates for the stock PH. We deliberately choose PH since the perfor-
mance gains of TUL when compared to HUL were the largest for this
particular stock, which has daily volume distributions that deviates
the most from the average due to relatively low liquidity. While the
U-shape Transformer may occasionally make erroneous predictions,
it quickly corrects itself and adapts to changes in the daily volume
distribution, attemptingtoaccuratelyfollowratiochangesthroughout
theday.
5. Discussionandconclusion
In this study, we propose a dual-level approach to address the
problemoftrackingthe VWAPwithaccuracyandconsistency. Initially,
as observed in Fig. 5, we identified that the naive uniform distribu-
Fig.8. Thegroundtruthand Transformer-predicted U-shapeplotsfortwoselectedtest
datesfor PH. Theinnerplotsdepicttheabsoluteerrorvaluesofthepredictions.. tion approach falls short in accurately tracking the daily VWAP. The
uniquenessofourmodelisinhowweimproveuponthisissuebytaking
into account the well-known U-shaped intraday trading pattern. In
the second level. For SE, which is the stock with the highest trading the first stage, we allocate the number of orders to execute in each
volume, TUR slightly underperforms HUL and TUL and outperforms interval based on the U-shape pattern. We consider two methods,
all other models. However, it is evident that a significant decline in statistical U-shapeand U-shape Transformer, forimplementingthefirst
performance occurs for all other stocks with lower trading volumes. stage. Thenaiveapproachofusingthestatistical U-shapedistribution
Thisperformancedegradationcanbeattributedtothelimitedcapabil- improvesperformance. However, ourexperimentsdemonstratethatthe
ity of the simpler RNN model when compared with the LSTM model U-shape Transformerperformsevenbetter. Webelievethatthesuperior
to capture volume fluctuations in stocks that exhibit relatively lower performanceofthe U-shape Transformerisduetoitsabilitytoadaptto
liquidity. unexpectedvariations, acapabilitythatthestatistical U-shapemethod
Lastly, we demonstrate the effectiveness of the proposed U-shape lacks, as evidenced by Figs. 7 and 8. After the total daily orders are
Transformerusing Figs.7 and8.Fig.7 showstheplotsoftheground allocatedtoeachintervalinthefirststage, the LSTMmodelisusedin
truth U-shapeand Transformer-predicted U-shapevalues, wherethe U- thesecondstagetodeterminehowtheordersineachintervalshould
shaperatiovaluesforeach20-minuteintervalwereaveragedacrossall beexecuted. Oursimulationresultsshowthatthedual-levelapproach
testdates. Weobservethatour U-shape Transformerisabletofollow consistentlyandaccuratelytracksthedailycumulative VWAP.
the U-shapepatternforthetestdatesonaverageforallstocks, which Whileourmethoddemonstratessuperiorperformancethroughex-
is in line with the comparable (and sometimes better) performance periments, a comprehensive mathematical understanding of its un-
of TUL against HUL. This is further supported by the fact that HUL derlying principles including uncertainty quantification remains to be
only utilizes the average U-shape ratio values of the past, while TUL explored. The limitations identified in our paper suggest potential
9

### Additional Content

#### 1

Expert Systems With Applications252(2024)124263
Contentslistsavailableat Science Direct
Expert Systems With Applications
journalhomepage:www.elsevier.com/locate/eswa
Anadaptivedual-levelreinforcementlearningapproachforoptimaltrade
execution
Soohan Kimd,1, Jimyeong Kimc,1, Hong Kee Sula,1, Youngjoon Hongb,1,∗
a Departmentof Finance, Chung-Ang University, Seoul, Republicof Korea
b Departmentof Mathematical Science, Korea Advanced Instituteof Scienceand Technology (KAIST), Daejeon, Republicof Korea
c Stochastic Analysisand Application Research Center (SAARC), Korea Advanced Instituteof Scienceand Technology (KAIST), Daejeon, Republicof Korea
d NYUCourant Instituteof Mathematical Sciences, NY, United Statesof America
A R T I C L E I N F O A B S T R A C T
Keywords: Thepurposeofthisresearchistodeviseatacticthatcancloselytrackthedailycumulativevolume-weighted
Volume-weightedaverageprice averageprice (VWAP) usingreinforcementlearningwhileminimizingthedeviationfromthe VWAP.Previous
Reinforcementlearning studies often choose a relatively short trading horizon to implement their models, making it difficult to
Optimaltradeexecution accurately track the daily cumulative VWAP since the stock price movement is often insignificant within
Proximalpolicyoptimization
theshorttradinghorizon. Ontheotherhand, trainingreinforcementlearningmodelsdirectlyoveralonger,
Markovdecisionprocess
dailyhorizonisburdensomeduetoextensivesequencelength. Hence, thereisaneedforamethodthatcan
dividethelongdailyhorizonintosmaller, moremanageablesegments. Weproposeamethodthatleverages
the U-shaped pattern of intraday stock trade volumes and uses Proximal Policy Optimization (PPO) as the
learningalgorithm. Ourmethodfollowsadual-levelapproach:a Transformermodelthatcapturestheoverall
(global) distributionofdailyvolumesina U-shape, anda LSTMmodelthathandlesthedistributionoforders
withinsmaller (local) timeintervals. Theresultsfromourexperimentssuggestthatthisdual-levelarchitecture
improvescumulative VWAPtrackingaccuracycomparedtopreviousreinforcementlearningapproaches. The
keyfindingisthatexplicitlyaccountingforthe U-shapedintradayvolumepatternleadstobetterperformance
inapproximatingthecumulativedaily VWAP.Thishasimplicationsfordevelopingtradingstrategiesthatneed
toefficientlytrack VWAPoverafulltradingday.
1. Introduction wantstoavoidmarketimpactandtheriskoffront-runnersexploiting
its strategy. To achieve this objective, the fund may outsource the
The optimal trade execution problem aims to find a strategy to trading to external brokerages or trading firms. In such a scenario,
optimallytradelargeorderswithinagivenperiodoftime. Oneofthe VWAP is commonly the most relevant metric for quantifying trading
mostcommonandpracticalmethodsthatpractitionersfrequentlyuseis performance. If the typical end-of-day market price were used as the
knownas Volume Weighted Average Price (VWAP) trading (Madhavan, performance metric, traders might execute all orders at the market’s
2002). Research regarding the optimality of VWAP as an execution close. This could lead the fund to suffer from market impact, and
strategy is discussed in Kato (2015), which further highlights the theend-of-daymarketpricemightbecomemorevolatile. Since VWAP
importanceoftheabilityofatradingmodeltotrack VWAPthroughout
reflectsastock’stransactionpriceovertime, itservesasamoresuitable
timeconsistently. VWAPiscalculatedbyaddingupthedollarstraded
metricfortradingperformance. Using VWAPallowsthefundtoexecute
foreverytransaction (pricemultipliedbythenumberofsharestraded)
its strategy with minimal market impact, thereby avoiding potential
andthendividingbythetotalsharestradedfortheday. Thisgivesan
lossesfrommarketspeculationandfront-running.
averagepricethattakesintoaccountboththepriceandthevolumeof
In Bertsimasand Lo (1998), afundamentalworkforoptimaltrade
sharestraded.
execution is proposed, where the authors assume that market prices
Funds and traders often use VWAP as a benchmark to compare
followanarithmeticrandomwalk. Theyuseadynamicprogramming
the price at which they executed trades to the overall market price
principle to find an explicit closed-form solution. Building on this
for the security, in order to evaluate the performance of their trades.
work, Huberman and Stanzl (2005) and Almgren and Chriss (2001)
Suppose a fund intends to significantly change its stock holdings but
∗ Correspondingauthor.
E-mailaddresses: soo.han.kim@nyu.edu (S.Kim), jimyeongkim@kaist.ac.kr (J.Kim), hksul@cau.ac.kr (H.K.Sul), hongyj@kaist.ac.kr (Y.Hong).
1 Theseauthorscontributedequallytothiswork.
https://doi.org/10.1016/j.eswa.2024.124263
Received12 May2023;Receivedinrevisedform7 May2024;Accepted14 May2024
Availableonline17 May2024
0957-4174/©2024 Elsevier Ltd. Allrightsarereserved, includingthosefortextanddatamining, AItraining, andsimilartechnologies.

#### 2

S.Kimetal. Expert Systems With Applications252(2024)124263
extendedtheresultin Bertsimasand Lo (1998) byincorporatingtrans-
action costs, more complex price impact functions, and risk aversion
parameters, undertheassumptionthatmarketpricesfollowa Brownian
motion. These dynamical approaches, however, are difficult to apply
directlyintherealworldduetothediscrepancybetweentheirstrong
marketassumptionsandthecomplexityofactualmarketconditions.
Ontheotherhand, bothpractitionersandresearchershavewidely
usedthetime-weightedaverageprice (TWAP) strategyandthevolume-
weighted average price (VWAP) strategy (Berkowitz et al., 1988;
Kakadeetal.,2004), whicharebasedoneitherpurerulesorstatistical
rules. Especially in Kakade et al. (2004), the authors used historical
datatoestimatetheaveragevolumetradedforeachtimeintervaland
split the order accordingly. However, this strategy is not well-suited
forcapturingunexpectedvolatility.
Fromtheperspectivethatoptimalexecutionproblemsareatypeof
sequentialdecision-makingtask, RLhasbeencommonlyappliedinthis
fieldasitisatypeofstochasticdecision-makingprocessthatautomates
thepractitioner’staskofusingpastdatatomakedecisionsonwhento
execute orders. Since RL approaches have many advantages, e.g. ca-
pable of capturing the market’s microstructure, the trading results
conductedby RLsolutionsoftenoutperformtraditional VWAPtracking
approachessuchas Bialkowskietal.(2008) and Podobniketal.(2009).
Tothebestofourknowledge, Nevmyvakaetal.(2006) isthefirstwork
toleverage RLframeworkssuchas Q-learning (Watkins&Dayan,1992)
to optimal trade execution problems. It is important to note that the
curse of dimensionality in Q-learning makes it challenging to handle
high-dimensionaldata. While (Hendricks&Wilcox,2014) attemptsto
combine RLwiththe Almgren–Chrissmodel, itisachallengingtaskas
themodeldependsoncertainassumptionsaboutthemarketdynamics.
Thankstotheadvancementsindeep RL, recentstudiessuchas Macri
and Lillo (2024), Ningetal.(2021), and Linand Beling (2019) have
utilized Deep Q-Networks (DQNs)(Mnihetal.,2013) foroptimaltrade
execution, addressingthechallengesofhigh-dimensionaldataandthe
complexity of the financial market without relying on any market
assumptions. However, these methods require the design of specific
attributes, whichcanbelabor-intensive. Recently, severalstudieshave Fig.1. Tradevolumeratioovereach20-minuteperiodthroughouttheday. Thelines
investigatedtheuseofproximalpolicyoptimization (PPO)(Schulman represent the 1-year averages of the ratios and the shaded regions are drawn from
et al., 2017) based on optimal execution frameworks, which do not dailydeviationsfromtheaverages.
require manually designed feature engineering. For instance, Lin and
Beling (2020), Fang et al. (2021), Pan et al. (2022), and Byun et al.
(2023) have explored this approach in different scenarios. Specifi- will be executed in each interval. We propose two methods for the
cally, Panetal.(2022) focusedonoptimalexecutionwithlimitorders, firststage, thestatistical U-shapemethodandthe U-shape Transformer
while Lin and Beling (2020) and Fang et al. (2021) used market method. Thestatistical U-shapemethodallocatesthevolumebasedon
orders. Amethodcombiningbothlimitandmarketorderswasproposed the historical average trade volumes. However, this approach cannot
by Byunetal.(2023). fully capture the day-to-day variations (See Fig. 1). Alternatively, we
Prior literature often considers a relatively short trading horizon propose the U-shape Transformer for figuring out these variations.
ortheentiredayfor RLimplementation. However, VWAPapproxima- In the next stage, we apply LSTM, trained by the RL framework to
tion becomes challenging when a short time frame is used since the distribute orders efficiently within each interval. It is important to
variations of financial data are often insignificant within short time note that there are numerous RL training algorithms, including Trust
intervals. Moreover, thenaiveapproachofconsideringtheentireday Region Policy Optimization (TRPO) (Schulman et al., 2015), Deep Q-
canputastrainonthenetworkarchitectureduetothelongsequence Networks (DQN)(Mnihetal.,2013), and Proximal Policy Optimization
length. This paper aims to develop a strategy that can consistently (PPO) (Schulman et al., 2017). Among these options, we specifically
trackthedailycumulative VWAP.Toachievethisgoal, westartwith choose PPOasourtrainingalgorithm. Foracomprehensiveunderstand-
ingofthesetechniquesandtheirapplications, wereferto Suttonand
the observation of an important stock trading characteristic, the U-
Barto (2018).
shaped intraday trading pattern, documented in Jain and Joh (1988)
We expect that this dual-level approach improves the accuracy of
and Goodhart and O’Hara (1997). This is a well-known stylized fact
approximatingthecumulative VWAPbyproperlydistributingthetotal
statingthatastock’stradingvolumeoftenfollowsa U-shapedpattern
order. Ourcontributionsareasfollows:
throughouttheday. Thismeansthatvolumeishighestattheopening,
falls rapidly to lower levels, and then rises again towards the close 1. We propose a novel approach for optimal trade execution that
of the market, but is relatively low in the afternoon. (See Fig. 1). In utilizes a dual-level strategy. Orders are allocated through two
view of this observation, we propose a novel dual-level approach to stages: in the first stage we utilize the U-shaped pattern of
minimize the market impact and track the daily cumulative VWAP intraday volumes and in the second stage we implement deep
accuratelyandconsistently. Asthenamesuggests, ourmodelconsists reinforcementlearning. Sincethisnoveldualapproachsegments
of two stages. In the first stage, the model decides how much trade the data, we not only make reinforcement learning more com-
volume will be executed in each interval, using the U-shape property putationallytractablebutalsoimprovepredictionaccuracycom-
asaguideline. Inthesecondstage, the RLmodeldecideshowtheorders paredtoapplying RLdirectlyonfullintradaysequences.
2

#### 3

S.Kimetal. Expert Systems With Applications252(2024)124263
Fig.4. Thearchitectureofthe LSTMmodel.
Williams, 1992). To find the parameter 𝜃, PPO iteratively gathers Algorithm1 Dual-level Neural Network
episodesandupdatestheparameter𝜃usingthefollowingscheme: Require: Number of 𝑜𝑢𝑡𝑒𝑟 𝑖𝑡𝑒𝑟𝑎𝑡𝑖𝑜𝑛𝑠 and 𝑛𝑢𝑚
𝜃 =argmax E E [𝐽 (𝜃)], (13) 𝑑𝑎𝑦𝑠
𝑘+1 𝜃 𝑠 𝑎∼𝜋𝜃𝑘 (⋅|𝑠) PPO Randomlyinitializelearnableparameters𝜃,𝜃, and𝜃
where𝑘standsforthe𝑘thstep. The‘‘clip’’operatorin (8) allows PPO Initializetrajectorybuffer
tolearnfrompreviousexperienceswithoutbecomingoverlydependent for𝑗=1 to𝑜𝑢𝑡𝑒𝑟𝑖𝑡𝑒𝑟𝑎𝑡𝑖𝑜𝑛𝑠do
on them. Moreover, PPO can effectively retain knowledge from past for𝑑=1 to𝑛𝑢𝑚𝑑𝑎𝑦𝑠do
experiences while also acquiring insights from new experiences. This Randomlychooseadateinthetrainingset
keyattributeof PPOisoneofitssignificantadvantages. 𝜇←theaverageofthedailyvolumeforthelast60 days
𝜎 ←thestandarddeviationofthedailyvolumeforthelast60
4. Experiments days
 ← ⋃ Gathering Episodes (date,𝜇,𝜎)
Our dual-level approach significantly enhances the accuracy and endfor
consistency of tracking the daily cumulative VWAP, despite the chal- Using , update ,  by optimizing the loss function 𝐽 in (5)
𝑇𝐹
lenges in improving the performance of RL frameworks for optimal withrespectto𝜃 and𝜃
trade execution in relatively short trading horizons. Furthermore, the Optimizethelossfunction𝐽 in (9) withrespectto𝜃
PPO
U-shape Transformerprovesefficientincapturingthedailyvariations  ←∅
of the U-shape distribution. In the following section, we conduct a endfor
seriesofexperimentstoverifytheseassertions.
4.1. Experimentsettings Algorithm2 Gathering Episodes
Require:Thenumberofintervalsinaday𝐿
4.1.1. Implementation Input:Date,𝜇,𝜎
Inallofourexperiments, thehyperparametersforthelossfunctions, Output:Trajectorybufferof𝑖𝑡ℎday
𝑖
thelengthoftheintervals, thenumberoftotalorders, andthetrading Initialize
𝑖
horizonarefixedandsetaccordingtothefollowingspecifications: 𝑂∼(2.5×10−3𝜇,6.25×10−6𝜎2)
𝐸 =(𝐸 )
- Thecoefficientsfor𝐽 :𝑐 =0.5,𝑐 =0.5 𝑜𝑢𝑡 𝑖𝑛
TF 1 2 for𝑙=0 to𝐿−1 do
- Thecoefficientsfor𝐽 :𝑐 =1,𝑐 =0.01
PPO 3 4 if𝑙=0 then
- Thelengthoftheintervals:𝐿=19,𝑇 =20 Construct𝐷0 fromrawpre-marketdata
- Thenumberoftotalorders:𝑂 ∼(2.5×10−3𝜇,6.25×10−6𝜎2), 𝑖𝑛
else
where𝜇and𝜎istheaverageandstandarddeviationoftheday 𝐷𝑙 ←𝐷𝑙−1⋃𝐶𝑜𝑛𝑐𝑎𝑡(𝐷𝑙−1,ℎ𝑙−1)
totalvolumefortheprevioussixtydays, respectively. 𝑖𝑛 𝑖𝑛 𝑜𝑢𝑡 𝑇−1
endif
- Tradinghorizonofaday:380 minutes (Fromwhenthemarket Obtain𝑢𝑙 and𝐷𝑙 byproceduredepictedin Section3.3.1
opensat09:00:00 towhenitclosesat15:20:00). 𝑝𝑟𝑒𝑑 𝑜𝑢𝑡
𝑂𝑙←𝑂⋅𝑢𝑙
𝑝𝑟𝑒𝑑
take S s in t c h e e a da t i y ly pi t c o a t l al da o y rd a e l r l , o 𝑂 w , s a f n o d r b 3 r 8 e 0 ak m s i i n td o o f w tr n ad to in 𝑂 g, 𝑙, o 1 u 9 r i m nt o e d r e v l al f s ir o st f O pr b o t c a e in du 𝜏 r 𝑙 e = d { e 𝑠 p 𝑙 𝑡 i , c 𝑎 t 𝑙 𝑡 e , d 𝑟( i 𝑎 n 𝑙 𝑡 ), S 𝑉 ec ( t 𝑠 i 𝑙 𝑡 o ), n 𝜋 3 (𝑎 .3 𝑙 𝑡| . 𝑠 2 𝑙 𝑡 ),𝑂 𝑡 𝑙} 𝑡∈{0,1,2,⋯,𝑇−1} andℎ𝑙 𝑇−1 by
20 mineach.(𝑙∈[0,18])Weeitherapplythestatistical U-shapemethod  ← ⋃𝜏
𝑖 𝑖 𝑙
orthe Transformermodelforthisfirststageallocationprocess. Then, endfor
withineachinterval, the LSTMmodelfurthersubdivides𝑂𝑙into20 sets
of1-minutesubintervals,𝑂𝑙.(𝑡∈[0,19])
𝑡
Ineachofthesesubintervals,𝑂𝑙 isdividedintoequalportionsand
𝑡
stock, weaccountforthedifferencesintotalordersforvariousstocks,
executed over a span of 12 steps, assuming that orders are executed
whichisoftenencounteredinreal-worldscenarios. Additionally, dur-
every 5 s. The method and hyperparameters used to train our agents
ingtesting, weonlyusethedailygroundtruth U-shapevolumeratios
for the experiment are summarized in Algorithms 1, 2, and Table 3
of dates in the training data for our Transformer Encoder to prevent
whichsilocatedattheendofthepaper.
lookingaheadintothefuture. Asinpreviousworksof Nevmyvakaetal.
To conduct realistic simulations, we determine 𝑂 in a way that
(2006), Hendricksand Wilcox (2014), Ningetal.(2021), and Linand
takes into account the volatility of daily volume, which differs from
Beling (2020), wemakethefollowingassumptionsinourexperiments.
most previous works where it is fixed to a certain value. This choice
aims to simulate the fluctuations of total orders that financial firms 1. We assume that the actions taken by our model only affect
havetoexecuteinaday, takingintoconsiderationrecenttradevolume a temporary market, and that the market will recover to the
statistics for each stock. By considering the volume statistics for each equilibriumlevelatthenexttimestep.
6

#### 4

S.Kimetal. Expert Systems With Applications252(2024)124263
Table2 theexperimentalsetupin Ningetal.(2021).Thetotalorderfor
Averagevolumeforeachstockfrom January1 st,2021 to December31 st,2021. thedayisthenequallydistributedamongsteachhour, and DQN
Stocks SE SH KC PH furtherallocatestheorderforfivesub-intervalsof12 mineach.
Volume 17,000 K 3800 K 3900 K 430 K The orders are executed equally for each 5-second time step
withinthe12-minutesub-intervals. Toensurefaircomparisons,
weevaluatethedailypricegeneratedby DQNagainstthedaily
cumulative VWAPcalculatedbyexcludingthefinal20 min.
- OPDisthemethodproposedin Fangetal.(2021).Toimplement
this model, we first divide a given day into ten intervals, each
spanning38 min, andthenuse OPDtoallocateordersforeach
interval. Within each interval, the allocated order is equally
distributedamongsteachminute.
- PPOisthemethodproposedin Linand Beling (2020).Toapply
this model, we equally distribute the day total order to each
minute in the day, i.e. 𝑂 = 𝑂∕380 for all 𝑗 ∈ {1,…,380} as
𝑗
in Linand Beling (2020).
- (HU-)PPO utilizes the statistical U-shape to determine 𝑂𝑙, and
within each interval, the allocated order is equally divided
amongeveryminute, i.e.𝑂𝑙=𝑂𝑙∕20 forall𝑡∈{0,…,19}, where
𝑡
Fig. 5. The VAA’s are depicted in dark blue for the best-case scenarios of uniform PPOisfinallyusedtodistributeorderswithintheminute.
distributionand U-shapedistributionorderallocationschemes, whiletheirworst-case
- HULisourproposeddual-levelapproachwhichusesthestatis-
counterparts are shown in brown. The uniform distribution and U-shape distribution
tical U-shapetodetermine𝑂𝑙.
allocate total orders to be executed per minute in the same manner as PPO and
(HU-)PPO, respectively.. - TULisourproposeddual-levelapproachthatemploysthe Trans-
formermodeltodetermine𝑂𝑙.
- TURuses RNNstoallocateorderswithineachinterval (second
2. Commissionsandexchangefeesareignored. level) instead of LSTMs. Other configurations are the same as
3. Ourmodel’sordersaretradedimmediatelywithoutorderarrival thatof TUL.
delays.
Weevaluatetheperformanceofthesemethodsalongsideourproposed
Webelievethattheaboveassumptionsarereasonableasweconsider dual-levelapproachbasedonthe VAAmetricin (7).Aswementioned
relatively small total orders in comparison to the daily total market in Section3.3.3, the VAAdirectlymeasurestheabilityofatradingstrat-
volumesofthestocks. egytocloselytrackandapproximatethedaily VWAP.Consequently, to
comprehensivelyassessabilityofeachmethodtoconsistentlytrackthe
4.1.2. Datasets daily VWAP over an extended period, we analyze both the mean and
Ourmillisecondtradeandlimitorderbook (LOB) tickdataisfrom variationofthe VAAmetric.
the Korea Exchange (KRX). We use the daily trade and LOB data of
Samsung Electronics (SE), SK Hynix (SH), Kia Corporation (KC) and 4.2. Experimentresults
POSCO Holdings Inc (PH) from January 1 st, 2021 to December 31 st,
2021. Our choice of the four firms in the KOSPI index is based on We test our models, which are trained for 100,000 iterations, by
their liquidity and trade volume and variety in market capitalization conductingtradingsimulationsinthebuyingdirection. Wefirstverify
andindustry. Inordertoevaluatetheeffectivenessofourapproachin the claim of the ineffectiveness of short trading horizons when order
relationtotradingvolume, wehaveoptedtoselectstockswitharange allocation does not consider the appropriate distribution method for
oftradevolumes. Specifically, weselectedstocksrepresentingdifferent volumesfromlongtradinghorizons.
tradevolumelevels—SEforlarge, SHand KCformedium, and PHfor Todothis, weexaminedtwodistinctscenarios:firstly, weuniformly
small volumes. The average daily trading volumes for each stock can distributed volumes from a long trading horizon into a short trading
be found in Table 2. We divide the data into two sets, using January horizon and secondly, we employed a statistical U-shape distribution
1 st to September 30 th as training data and October 1 st to December to allocate volumes from a long trading horizon into a short trading
31 stastestdata. horizon. Note that the volumes from a long trading horizon mean
The millisecond raw data is preprocessed by first dividing them dailytotalorders, whileashorttradinghorizonrepresentsa1-minute
into groups of 5 s, and extracting data that represents each 5-second tradinghorizon. Tocomparethebestandworstcases, wedeliberately
interval. We take the last LOB data of the 5-second interval to con- executed trades for the lowest and highest buy orders, respectively,
structthe MDPstateandthe5-second VWAPandtotaltradedvolume within the 1-minute trading horizon. Fig. 5 displays the comparison
for calculating the daily VWAP. The statistical U-shape statistics are ofthe VAA’sonthetestdates. Wecanobservethattheworstcasefor
constructed by taking the year average of the U-shape ratios of the U-shapedistributionoutperformsthebestcaseforuniformdistribution.
20-minute intervals in the day. The data utilized in generating 𝐷0 Specifically, the VAA’smeanoftheworstcasefor U-shapedistribution
𝑖𝑛
shownin Section3.3 arethetop5 bid/askvolumeaveragesoftheraw is10 bpswhilethe VAA’smeanofthebestcaseforuniformdistribution
pre-marketdata (from08:30:00 to09:00:00). is 11 bps. This result suggests that in the context of VWAP, the use
of deep reinforcement learning does not significantly improve the
4.1.3. Comparisonandablationstudy approximation capability without taking into account the long-term
Weprovideabriefexplanationofallmethodsusedforperformance trading horizon. This implies that in order to effectively leverage the
comparisonduringourexperimentshere. Thehyperparametersforeach benefitsofdeep RLinthecontextof VWAPs, itshouldbeimplemented
methodcanbefoundin Table3. onalongertradinghorizonsuchasasingletradingday
Secondly, we present Table 4 and Fig. 6 to demonstrate how our
- DQN is the method proposed in Ning et al. (2021). To apply dual-levelapproachsignificantlyimprovesperformance.
this model, we begin by dividing a given day into six 1-hour Table 4 reports the mean and standard deviation values of VAA’s
intervals, excludingthefinal20 minofmarkethourstoreplicate produced by DQN, OPD, PPO, HUL and TUL across the test dates,
7

#### 5

S.Kimetal. Expert Systems With Applications252(2024)124263
Fig.9. Thehistogramsofthe VAAdistributionsforeachmodelandstockonthetestdates. Both TULand HUL, employingthe U-shapeproperty, exhibitconsistentstabilityin
trackingthe VWAPfor SE, SH, and KCstocks. Moreover, TULdemonstratesrobusttrackingforthe PHstock, whichischaracterizedbysignificantdailyvolumefluctuations.
Table4
Table4 summarizesthemean, standarddeviation, andpercentageof VAAswithintherange[0,10 bps].The Meanand Standard Deviationarerepresentedinunitsofbasispoints
(bps), andthe%in10 bpsarerepresentedinunitsofpercentages. For OPD, weexcluded VAAsabovethe80 thpercentileforfaircomparisonssinceourtrainingdatasetwas
smallerthanthatusedin Fangetal.(2021).For TUR, weexcludedextremeoutliersfor SH, KA, and PH.
Stocks Metric OPD DQN PPO HUL TUL TUR
Mean 26.27 11.53 12.00 8.77 7.13 9.43
SE Standard Deviation 18.50 8.84 8.88 7.17 2.44 9.56
%in10 bps 25.00 57.38 49.18 65.57 88.52 65.57
Mean 38.55 21.62 22.57 7.13 7.65 48.56
SH Standard Deviation 24.90 17.71 16.92 2.81 3.63 33.51
%in10 bps 10.42 30.00 25.00 91.67 88.33 5.45
Mean 27.79 13.16 12.47 6.59 6.03 149.32
KA Standard Deviation 18.76 10.25 11.55 8.36 3.30 111.64
%in10 bps 18.75 46.67 56.67 90.00 86.67 9.43
Mean 32.44 14.84 14.64 22.52 12.27 377.28
PH Standard Deviation 20.89 13.47 12.63 47.11 7.90 267.23
%in10 bps 18.75 40.98 40.98 32.79 44.26 1.85
directionsforfutureresearch. Additionally, wenotethatourproposed CRedi Tauthorshipcontributionstatement
methodentailsthefollowingethicalimplications. Firstly, ourapproach
providesinvestorswithnewoptionsfordealingwithmarketvolatility, Soohan Kim: Writing – original draft, Methodology, Software.
helpingtoavoidmarketimpactandtheriskoffront-runnersexploiting Jimyeong Kim: Writing – original draft, Data curation, Conceptual-
theirstrategies. Secondly, improvedabilityofourexecutionmodelto ization, Visualization. Hong Kee Sul: Writing – review & editing, In-
track the VWAP contributes to mitigating market impact, which can vestigation. Youngjoon Hong:Writing–review&editing, Supervision,
potentiallyenhancemarkettransparencyandfairness. Projectadministration.
10

#### 6

S.Kimetal. Expert Systems With Applications252(2024)124263
Declarationofcompetinginterest Kakade, S.M., Kearns, M., Mansour, Y.,&Ortiz, L.E.(2004).Competitivealgorithms
for VWAP and limit order trading. In Proceedings of the 5 th ACM conference on
Theauthorsdeclarethefollowingfinancialinterests/personalrela- electroniccommerce (pp.189–198).
Kato, T.(2015).VWAPexecutionasanoptimalstrategy. JSIAMLetters,7,33–36.
tionships which may be considered as potential competing interests:
Lin, S.,&Beling, P.A.(2019).Optimalliquidationwithdeepreinforcementlearning.
Youngjoon Hongreportswasprovidedby Sungkyunkwan University. In Proceedingsofthe33 rdconferenceonneuralinformationprocessingsystems, deep
reinforcementlearningworkshop. Vancouver, Canada.
Dataavailability Lin, S.,&Beling, P.A.(2020).Anend-to-endoptimaltradeexecutionframeworkbased
onproximalpolicyoptimization..In IJCAI(pp.4548–4554).
Macri, A., & Lillo, F. (2024). Reinforcement learning for optimal execution when
Datawillbemadeavailableonrequest.
liquidityistime-varying.ar Xivpreprintar Xiv:2402.12049 v2.
Madhavan, A.(2002).VWAPstrategies. Trading,1,32–39.
Acknowledgments Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T., Harley, T., Silver, D., &
Kavukcuoglu, K. (2016). Asynchronous methods for deep reinforcement learning.
In Internationalconferenceonmachinelearning (pp.1928–1937).PMLR.
Theworkof Y.Hongwassupportedby Basic Science Research Pro-
Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D., &
gramthroughthe National Research Foundationof Korea (NRF) funded Riedmiller, M.(2013).Playingatariwithdeepreinforcementlearning. URL:http:
bythe Ministryof Education, Republicof Korea (NRF-2021 R1 A2 C1093 //arxiv.org/abs/1312.5602. Cite arxiv:1312.5602 Comment: NIPS Deep Learning
579) andbythe Koreagovernment (MSIT)(RS-2023-00219980). Workshop2013.
Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D., &
Riedmiller, M. (2013). Playing atari with deep reinforcement learning. ar Xiv
References
preprintar Xiv:1312.5602.
Nevmyvaka, Y., Feng, Y.,&Kearns, M.(2006).Reinforcementlearningforoptimized
Almgren, R.,&Chriss, N.(2001).Optimalexecutionofportfoliotransactions. Journal trade execution. In Proceedings of the 23 rd international conference on machine
of Risk,3,5–40. learning (pp.673–680).
Berkowitz, S.A., Logue, D.E.,&Noser, E.A., Jr.(1988).Thetotalcostoftransactions Ning, B., Lin, F. H. T., & Jaimungal, S. (2021). Double deep q-learning for optimal
onthe NYSE.The Journalof Finance,43(1),97–112. execution. Applied Mathematical Finance,28(4),361–380.
Bertsimas, D., & Lo, A. W. (1998). Optimal control of execution costs. Journal of Pan, F., Zhang, T., Luo, L., He, J.,&Liu, S.(2022).Learncontinuously, actdiscretely:
financialmarkets,1(1),1–50. Hybrid action-space reinforcement learning for optimal execution. In IJCAI (pp.
Bialkowski, J., Darolles, S.,&Fol, G.L.(2008).Improving VWAPstrategies:Adynamic 3912–3918).
volumeapproach. Journalof Banking&Finance,32,1709–1722. Podobnik, B., Horvatic, D., Petersen, A.M.,&Stanley, H.E.(2009).Cross-correlations
Byun, W., Choi, B., Kim, S.,&Jo, J.(2023).Practicalapplicationofdeepreinforcement betweenvolumechangeandpricechange. Proceedingsofthe National Academyof
learningtooptimaltradeexecution. Fin Tech,2,414–429. Sciences,106(52),22079–22084.
Fang, Y., Ren, K., Liu, W., Zhou, D., Zhang, W., Bian, J., Yu, Y., & Liu, T.-Y. Schulman, J., Levine, S., Abbeel, P., Jordan, M.,&Moritz, P.(2015).Trustregionpolicy
(2021). Universal trading for order execution with oracle policy distillation. 35, optimization. In Internationalconferenceonmachinelearning (pp.1889–1897).PMLR.
In Proceedingsofthe AAAIconferenceonartificialintelligence (1),(pp.107–115). Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal
Goodhart, C.A.,&O’Hara, M.(1997).Highfrequencydatainfinancialmarkets:Issues policyoptimizationalgorithms.ar Xivpreprintar Xiv:1707.06347.
andapplications. Journalof Empirical Finance,4,73–114. Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction. MIT
Hendricks, D., & Wilcox, D. (2014). A reinforcement learning extension to the Press.
almgren-chrissframeworkforoptimaltradeexecution. In2014 IEEEconferenceon Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser,Ł.,
computationalintelligenceforfinancialengineering&economics (CIFEr)(pp.457–464). &Polosukhin, I.(2017).Attentionisallyouneed. Advancesin Neural Information
IEEE. Processing Systems,30.
Huberman, G.,&Stanzl, W.(2005).Optimalliquiditytrading. Reviewof Finance,(2), Watkins, C.J.,&Dayan, P.(1992).Q-learning. Machine Learning,8(3),279–292.
165–200. Williams, R.J.(1992).Simplestatisticalgradient-followingalgorithmsforconnectionist
Jain, P.C.,&Joh, G.-H.(1988).Thedependencebetweenhourlypricesandtrading reinforcementlearning. Reinforc Ement Learning,5–32.
volume. The Journalof Financialand Quantitative Analysis,23(3),269–283.
11


---

## Content Unique to Old Extraction (not found in markitdown output)

### 1. # An adaptive dual-level reinforcement learning approach for optimal trade execu...

# An adaptive dual-level reinforcement learning approach for optimal trade execution

### 2. > *Source PDF: An adaptive dual-level reinforcement learning approach for optima...

> *Source PDF: An adaptive dual-level reinforcement learning approach for optimal trade execution.pdf*
> *Extraction: Combined — markitdown raw text (base) + previous extraction supplements*

### 3. ## Content from Previous Extraction (not in markitdown output)...

## Content from Previous Extraction (not in markitdown output)

### 4. ### Visual/Chart/Graph Descriptions...

### Visual/Chart/Graph Descriptions

### 5. S.Kimetal. Expert Systems With Applications252(2024)124263
Table1 objectiveistof...

S.Kimetal. Expert Systems With Applications252(2024)124263
Table1 objectiveistofindastrategythatmaximizesorminimizestheaverage
The table summarizes previous works on optimal trade execution utilizing the RL executionprice𝑃̄, whichiscalculatedas
framework. Note that DQN and PPO are learning algorithms of deep RL. Notably,
ourworkdistinguishesitselfbyfocusingonalong-termhorizon. 𝑃̄ = 𝑇 ∑ −1𝑂 𝑡𝑝.
RLmethod Horizon 𝑂 𝑡
𝑡=0
Nevmyvakaetal.(2006) Q-learning Short
Hendricksand Wilcox (2014) Q-learning Short While many financial firms pursue profits through trading, there are
Linand Beling (2019) DQN Short also numerous firms within the financial industry that prioritize con-
Ningetal.(2021) DQN Short sistentlytrackingthe VWAPovermaximizingtradingprofit. Thefocus
Linand Beling (2020) PPO Short
ofourpaperistailoredforthissecondgroupoftraders. The VWAPis
Fangetal.(2021) PPO Short
calculatedas
Panetal.(2022) PPO Short
Thiswork PPO Long ∑𝑇−1𝑝𝑞
VWAP= 𝑡=0 𝑡 𝑡 ,
∑𝑇−1𝑞
𝑡=0 𝑡
where 𝑞 is the execution volume determined by the market. There-
2. We further propose the U-shape Transformer model to capture 𝑡
fore, theprimaryobjectiveofthispaperistominimizethedifference
theday-to-dayoscillationsoftheintraday U-shapedistribution. betweenthe VWAPand𝑃̄, usingadual-levelapproach.
Whilethe U-shapedpatterniswell-known, relyingonfixedsta-
tisticalestimatesfailstoadapttosuddenfluctuationsorshiftsin
3. Methoddescription
marketconditions. The U-shape Transformerdirectlylearnsaro-
bustrepresentationfromrecentdata, leveragingself-attentionto
Inthissection, wedescribeourdual-levelapproachandformulation
automaticallyadapttosuddendynamicsintheintraday U-shape
oftheoptimaltradeexecutionproblemunderthelensofreinforcement
pattern.
learning. Wealsoprovidedetailsonthearchitectureofthe Transformer
3. Weconductablationstudiestoshowthestrengthsofourmeth-
and LSTM neural networks and how they are integrated into our
odsinapproximatingthedailycumulative VWAP.Furthermore,
proposedmethod.
we demonstrate that the methods employed to distribute vol-
umes from long trading horizons into short horizons can have 3.1. Dual-levelapproach
asignificantimpact. Inaddition, weintroducethe VAA, which
provides a more practical way to evaluate how well a trading Our approach involves two stages of distributing orders. In the
strategy tracks the daily VWAP. Our proposed dual-level ap- first stage, we allocate the total daily orders 𝑂 into 𝐿 intervals. In
proachcombining U-shape Transformersand RLachievesstate- thesecondstage, wefurtherdividetheordersfromeachintervalinto
of-the-artperformanceonthe VAA(see Table1). smaller executable orders using RL. Our approach begins by dividing
the market hours of a day into 𝐿 intervals. For each interval 𝑙, we
let 𝑂𝑙 as the total order to be executed, ensuring that the sum of all
2. Backgrounds
intervalordersequalsthetotaldailyorder, or
∑𝐿−1𝑂𝑙=𝑂.Wesuggest
𝑙=0
two methods for implementing the first stage. The naive approach
Inthissection, wediscussthebasicsofthelimitorderbook (LOB)
is to adhere to the statistical U-shape trade volume distribution for
and the optimal trade execution process. Most modern financial ex-
each interval obtained from past data (e.g. the historical average of
changes, including the Korea Stock Exchange (KRX), offer electronic
the past year). A more advanced method would be to use the output
tradingplatformswithaccesstolimitorderbooks. Tradersoftenrelyon
(predicted U-shape distribution) generated by a U-shape Transformer
theinformationprovidedbythelimitorderbooktodevelopsuccessful
model. In the latter case, 𝑂𝑙 is calculated progressively by referring
tradingstrategies.
only to market information from past days and previous intervals
within the day. For such an 𝑙th interval, which is comprised of 𝑇
2.1. Limitorderbook timesteps, the LSTMmodeloutputs𝑂𝑙, whichisthenumberoforders
𝑡
to be executed at timestep 𝑡 such that ∑𝑇−1𝑂𝑙 = 𝑂𝑙. Once 𝑂𝑙 is
Infinancialmarkets, alimitorderisanordertobuyorsellafixed 𝑡=0 𝑡 𝑡
decided, it is averaged over 𝐼 execution steps in the corresponding
number of shares at a specified price. This order becomes part of the
𝑡th subinterval. In other words, for each execution step, which is
limitorderbook, whichrecordsalllimitordersforaspecificstock. The set as every 5 s, the final number of orders executed is 𝑂𝑙∕𝐼. The
𝑡
bidpriceisthehighestpriceabuyeriswillingtopayforthesecurity,
choice of averaging is supported by our findings in Section 4.2 that
andtheaskpriceisthelowestpriceinwhichtheselleriswillingtosell.
there is no significant impact in approximating the VWAP since the
Amarketorder, bycontrast, isanordertobuyorsellafixednumber finalintervalistoosmall. Furthermore, in Section4.2, weverifythat
ofsharesatthecurrentmarketpricewithoutstatingtheprice. Whena incorporatinglong-termhorizonsyieldssuperiorefficiency. Therefore,
marketorderisplaced, itismatchedbythebestavailablelimitorders althoughadeepreinforcementlearningframeworkcanoffermarginal
inthe LOB.The LOBisimportantforoptimaltradeexecutionandhelps performanceimprovements, wehaveoptedforthesimplicityofusing
traderstracksupplyanddemandforastock, makingiteasiertoidentify theuniformdistributionforthefinalinterval. Aschematicillustration
thebesttimeandpricetobuyorsellshares. ofthisdesignisprovidedin Fig.2.
2.2. Optimaltradeexecution 3.2. MDPformulationforoptimalexecution
Optimal trade execution refers to the process of buying or selling In this section, we explain our Markov Decision Process (MDP)
afinancialassetwhileachievingthedesiredobjective. Generally, this formulationofoptimaltradeexecution. AMDPistypicallyrepresented
problem formulates as follows: Within a timeframe of 𝑇 timesteps, using the tuple (, , , 𝑟, 𝛾), where  is the state space,  is the
0,1,…,𝑇 −1, atraderwhopossessesaninventoryof𝑂sharesmustbuy action space,  is the transition probability, 𝑟 is the reward function,
orselltheentiretyoftheinventory. Foreachtimestep𝑡∈{0,1,…,𝑇− and𝛾isthediscountfactor. Ourconfigurationsofthestatespaceand
1}, thetraderdeterminesthenumberofsharestoorder𝑂 basedonthe actionspacearesimilartothatof Linand Beling (2020).Itisworth
𝑡
informationfrom LOB, andcarriesoutthetradeatanexecutionprice notingthatthe MDPimplementationinthispaperisappliedonaper
𝑝. For the group of traders seeking to maximize trading profit, their 𝑙thintervalbasisforagivenday.
𝑡
3

### 6. S.Kimetal. Expert Systems With Applications252(2024)124263
Fig.2. Aschematicillu...

S.Kimetal. Expert Systems With Applications252(2024)124263
Fig.2. Aschematicillustrationofourdual-levelapproach.
3.2.1. State Ourrewardfunctioncompares𝑂𝑙∗ and𝑂𝑙 directlyforagiven𝑎𝑙 as
𝑡 𝑡 𝑡
Thestate𝑠𝑙∈consistsofbothpublicandprivateinformation. The follows:
𝑡
publicstateismadeupofthetop5 bidandaskprices, alongwiththeir ⎧ 1 𝑀𝑙<0.01
associated volumes, while the private state includes the elapsed time 𝑟(𝑎𝑙)∶= ⎪ 0 0.0 𝑡 1≤𝑀𝑙<0.05 (2)
𝑡 ⎨ 𝑡
and the current remaining volume to be executed. The elapsed time ⎪ −1 otherwise
rangesfrom0 to𝑇−1 andthecurrentremainingvolumeattimestep𝑡 ⎩
isequalto𝑂𝑙− ∑𝑡 𝑗 − = 1 0 𝑂 𝑗 𝑙. where𝑀𝑙∶= |𝑂 𝑡 𝑙−𝑂 𝑡 𝑙∗ |.
𝑡 𝑂𝑙∗
𝑡
Once MDPisdetermined, thegoalofpolicy-based RListofindan
3.2.2. Action optimalpolicyparameter𝜃∗, i.e.
Our LSTMmodeloutputsapolicy𝜋(⋅|𝑠𝑙
𝑡
)∈R21, whichisadiscrete
𝜃∗=
c { e a 0 x t , e e 0 c g . u 1 o t , e r 0 i , . c 2 a w , l … it p h r , o 2 𝑂 b } 𝑡 𝑙 ab i = s ili s t 𝑎 a y 𝑙 𝑡 m 𝑂 𝑇 d p 𝑙 i . l s e t T r d i o b t u e o t n io s d u n e r . t e e F r t r m h o a i m n t e t t h h t e h is e , to a n t n a u l m a n c b u t e i m o r n b o e 𝑎 f r 𝑙 𝑡 o o ∈ r f d o e  r r s de ∶ t r = o s argmax E [ ∑ 𝑡= 𝑇 0 𝛾𝑡𝑟 𝑡 | | | | 𝑎 𝑡 ∼𝜋 𝜃 (⋅|𝑠 𝑡 ),𝑠 𝑡+1 ∼(⋅|𝑠 𝑡 ,𝑎 𝑡 ) ] , (3)
executedovertheentireperiodisequalto𝑂𝑙, thatis ∑𝑇 𝑡= − 0 1𝑂 𝑡 𝑙=𝑂𝑙, we where𝑟 𝑡 =𝑟(𝑎 𝑡 ).
imposethefollowingtworestrictions.
3.3. Neuralnetworkarchitectureandtraining
• If ∑𝑗 𝑂𝑙>𝑂𝑙 forsome𝑗∈{0,1,2,…,𝑇 −1}, then
𝑡=0 𝑡
{ 𝑂𝑙− ∑𝑗−1𝑂𝑙, if𝑖=𝑗 We now turn to the architectural details of the Transformer and
𝑂𝑙= 𝑡=0 𝑡 LSTM models and elaborate on how they are used in our dual-level
𝑖 0, if𝑗<𝑖≤𝑇 −1
approach.
• 𝑂𝑙 isequaltotheremainingvolumetobeexecutedatthelast
tim 𝑇− e 1 stepforeachinterval (i.e.0≤𝑙≤𝐿−1) 3.3.1. Level1:Transformerforglobal U-shapeapproximation
Theobjectiveofthe Transformermodelistoprogressivelypredict
the U-shaperatioforeachintervalinagivenday. Weusethestructure
Theserestrictionsareinlinewiththefactthatitisessentialforbrokers
ofthe Transformer Encoderand Decoderasproposedin Vaswanietal.
topromptlyacquireorliquidateallsharesrequestedbytheircustomers.
(2017).Theoverallarchitectureofthe Transformermodelthatweuse
isdepictedin Fig.3.
3.2.3. Reward U-shape Encoder. The U-shape Encoder iscomposedofa Trans-
The reward function is designed to encourage the agent (LSTM former Encoder and 𝐿 linear layers, all of which map to the same
model) to generate actions that result in orders that are close to the dimension. Thismoduleisresponsibleforlearningthegeneraldynam-
target volume 𝑂𝑙∗, in order to track the VWAP. This is achieved by ics of the U-shape distribution. The input 𝐸 𝑖𝑛 is a sequence of length
𝑡
usingthepricecalculatedfromtheexecutedorders. Here,𝑂𝑙∗ denotes 𝐿 of vectors each containing 𝑁 historical daily volume ratios of the
𝑡
corresponding interval in the sequence. The days from which these
the desired VWAP order at timestep 𝑡 in the 𝑙th interval such that
𝑂𝑙= ∑𝑇−1𝑂𝑙∗.Thedaily VWAPcanbeobtainedby ratios are drawn are randomly selected. Each vector in the sequence
𝑡=0 𝑡 is processed by a different linear layer that serves as an embedding
𝑉𝑊𝐴𝑃 𝑑𝑎𝑦 = 𝐿 ∑ −1𝑇 ∑ −1𝑂 𝑂 𝑡 𝑙∗ 𝑝𝑙 𝑡 , (1) t t o he le e a m rn be t d h d e in u g ni l q a u y e er f , ea th tu e re i s np o u f t e s a e c q h ue in n t c e e rv i a s l. th A e f n ter pr p o a c s e s s in se g d th b r y ou t g h h e
𝑙=0 𝑡=0 Transformer Encoder, resultingintheoutput𝐸 .
𝑜𝑢𝑡
where 𝑝𝑙 is the associated average market traded price of the corre- U-shape Decoder. While  learns the general dynamics of the U-
𝑡
sponding𝑡thsubinterval. shape distribution, the U-shape Decoder  focuses on handling the
4

### 7. S.Kimetal. Expert Systems With Applications252(2024)124263
3.3.3. Trainingproced...

S.Kimetal. Expert Systems With Applications252(2024)124263
3.3.3. Trainingprocedure
U-shape Transformer. Since the goal of the first level of order al-
locationistoapproximatethe U-shapedistributionwhilealsotracking
thedailycumulative VWAP, thetrainingobjectiveofthe Transformer
modelistominimize
𝐽 ∶=E [ 𝑐 1 𝐿 ∑ −1( 𝑢𝑙 −𝑢𝑙 )2 +𝑐 𝑉𝐴𝐴 ] (6)
𝑇𝐹 𝑑𝑎𝑦𝑠 𝐿 𝑡𝑟𝑢𝑒 𝑝𝑟𝑒𝑑 2
𝑙=0
where𝑐 and𝑐 arecoefficientsand
1 2
VAA∶= | | | | | 𝑀𝑃 𝑑 𝑉 𝑎𝑦 𝑊 − 𝐴 𝑉 𝑃 𝑊 𝑑𝑎 𝐴 𝑦 𝑃 𝑑𝑎𝑦| | | | | (7)
denotesthe VWAPApproximation Accuracy (VAA), whichwewillfur-
theruseasageneralmetricforperformanceevaluationinexperiments,
and
𝑀𝑃 ∶=
𝐿
∑
−1𝑇
∑
−1𝑂
𝑡
𝑙
𝑝𝑙
Fig. 3. The architecture of the U-shape Transformer. The U-shape Decoder in this 𝑑𝑎𝑦 𝑂 𝑡
𝑙=0 𝑡=0
figurerepresentsthe𝑙thdecodingstep.
is the price yielded by our model. This is one of the measures used
in practice to gauge how close a trader has traded closer to their
benchmark, the VWAP.Ifatraderhastradedatapricethatisexactly
day-to-dayvariationsofthedistribution. The U-shape Decoderiscom-
equal to the VWAP, then 𝑉𝐴𝐴 = 0. Thus lower VAA would be
posedofa Transformer Decoderand𝐿linearlayers, eachmappingto
better. Thefirstterminsidetheexpectationof (6) learnstomakeratio
differentdimensions. Incontrastto, theselinearlayersareappliedto
predictions closer to ground-truth values. The second term penalizes
theoutputofthe Transformer Decoder. the Transformermodelifthefinaldailyacquisitionpricegeneratedby
For every 𝑙th (𝑙 > 0) decoding step, both 𝐸 𝑜𝑢𝑡 and the cumulative thecombinationofthe Transformerand LSTMmodelsdeviatesfromthe
input sequence are fed as input to the Transformer Decoder, . We daily VWAP.Theprimaryobjectiveofthe U-shapedtransformerinthe
denote the cumulative input sequence as 𝐷 𝑖 𝑙 𝑛 = (𝐷 𝑖 𝑙 𝑛,𝑗 ) 0≤𝑗≤𝑙−1 , where firstlevelistodistributethedailytotalorderwithprecision. However,
𝐷𝑙 ∶= CONCAT(𝐷𝑗 ,ℎ𝑗 ). Here, 𝐷𝑗 represents the output of  relyingsolelyonthefirsttermofthelossfunctionmaynotbeenough
at 𝑖𝑛 t , h 𝑗 e𝑗thstep, andℎ 𝑜𝑢 𝑗 𝑡 𝑇 is −1 thelast LST 𝑜𝑢 M 𝑡 hiddenvectorfromthe𝑗th to accomplish this objective. This is because 𝑢𝑙 𝑡𝑟𝑢𝑒 does not consider
𝑇−1 the local information, such as the LOB, which can pose difficulties
interval. Theoutputfromthe Transformer Decoderatthe𝑙thdecoding
in tracking the daily VWAP accurately. Furthermore, if only the first
step is a vector of dimension 𝐿+𝐻, which is then passed through a
termin (6) istakenintoaccount, thereinforcementlearningframework
linearlayerthatreducesitsdimensionto𝐿−𝑙.Thesoftmaxfunctionis
wouldnotinfluencethefirstlevel, leadingtoindependentoperationof
appliedtopredicttheratiosoftheremaining𝐿−𝑙intervals, andthefirst
thefirstandsecondlevels. Toensurethatthemicroinformationhasan
𝑙 components are zero-padded. Note that we use ℎ𝑗 𝑇−1 to capture the appropriate impact on the first level, we incorporate the second term
day-to-day variations in the U-shape distribution, as it holds interval- intothelossfunction.
specificinformationforthatparticularday. Toensure ∑𝐿 𝑙= − 0 1𝑢𝑙 𝑝𝑟𝑒𝑑 =1, LSTM.Tofindtheoptimalparameter𝜃forthe LSTMmodel, weuse
thefinal𝑙thintervalvolumeratiopredictioniscalculatedas an actor–critic style PPO algorithm (Schulman et al., 2017), which is
( 𝑙−1 ) oneofthemostpopularon-policy RLalgorithms, asourbaselearner.
𝑢𝑙 = 1− ∑ 𝑢𝑗 ⋅𝐷𝑙 [𝑙]. (4) One of the major advantages of the PPO in this task is its ability to
𝑝𝑟𝑒𝑑 𝑝𝑟𝑒𝑑 𝑜𝑢𝑡
𝑗=0 adapt to changing market conditions and effectively learn from noisy
andhigh-dimensionaldata. Thisalgorithmusestheactorloss𝐽𝐶𝐿𝐼𝑃(𝜃)
It is worth noting that 𝐷 𝑖 0 𝑛 is constructed by applying an additional andthecriticlossfunction𝐽𝑉𝐹(𝜃), whicharedefinedasfollow PP s O :
PPO
linear layer that transforms the dimension of the vector holding the
t w o i p th 5 t b h i e d g / i a v s e k n v d o a lu y m to e ta a l v o er r a d g e e r s d o e f no p t r e e d -m a a s r 𝑂 ke , t th d e at c a or to re 𝐿 sp + on 𝐻 d . in T g h i e n r t e e f r o v r a e l , 𝐽 ∶ P 𝐶 = P 𝐿 O E 𝐼𝑃( [ 𝜃 m ) in ( 𝑞(𝜃)𝐴̂, clip (𝑞(𝜃),1−𝜀,1+𝜀)𝐴̂)] , (8)
𝑡 𝑡 𝑡 𝑡 𝑡
order𝑂𝑙 canbedeterminedastheproductof𝑂and𝑢𝑙 , i.e.
𝑝𝑟𝑒𝑑 and
[ ]
𝑂𝑙=𝑂×𝑢𝑙 𝑝𝑟𝑒𝑑 . (5) 𝐽 P 𝑉 P 𝐹 O (𝜃)=E 𝑡 (𝑉 𝑡 targ−𝑉 𝜃 (𝑠 𝑡 ))2 , (9)
where𝑞(𝜃)= 𝜋𝜃(𝑎𝑡|𝑠𝑡) ,
𝑡 𝜋𝜃old (𝑎𝑡|𝑠𝑡)
3.3.2. Level2:LSTMforlocalorderdistribution 𝑇−1
The objective of the LSTM model is to optimally allocate orders 𝑉 𝑡 targ= ∑ 𝛾𝑘−𝑡𝑟 𝑘 and𝐴̂ 𝑡 =𝑉 𝑡 targ−𝑉 𝜃 (𝑠 𝑡 ). (10)
withintheintervalsofagivenday. Thestructureofour LSTMmodel, 𝑘=𝑡
whichisoutlinedin Fig.4, issimilartothatof Linand Beling (2020). PPO’sgoalistofindaparameter𝜃 thatmaximizesthemainobjective
Given𝑠𝑙 𝑡 asinput, itfirstpassesthroughtwolinearlayerswithhidden function𝐽 PPO (𝜃), whichisdefinedby
d T i h m e e r n e s s i u o l n tin 1 g 28 v a e n ct d or is is th t e h n en co p n r c o a c t e e s n s a e t d ed by wi a t n h 𝑎 L 𝑙 𝑡 S − T 1 M an C d e 𝑟 ll 𝑙 𝑡− w 1 i ∶ t = h h 𝑟( i 𝑎 d 𝑙 𝑡 d − e 1 ) n . 𝐽 PPO (𝜃)=𝐽 P 𝐶 P 𝐿 O 𝐼𝑃(𝜃)−𝑐 3 𝐽 P 𝑉 P 𝐹 O (𝜃)+𝑐 4 E 𝑡 [𝑆[𝜋 𝜃 ](𝑠 𝑡 )], (11)
dimension 𝐻 to obtain ℎ𝑙, which is then passed separately through where𝑐 3 ,𝑐 4 arecoefficients, and E 𝑡 [𝑆[𝜋 𝜃 ](𝑠 𝑡 )]denotesanentropyloss
𝑡
two linear layers to produce the policy 𝜋(⋅|𝑠𝑙
𝑡
) and the value 𝑉(𝑠𝑙
𝑡
) term, where𝑆[𝜋 𝜃 ](𝑠 𝑡 ) isdefinedby
respectively. Wenotethatthevectorℎ𝑙 𝑇−1 whichisobtainedatthelast 𝑆[𝜋 𝜃 ](𝑠 𝑡 )∶=− ∑ 𝜋 𝜃 (𝑎|𝑠 𝑡 ) log𝜋 𝜃 (𝑎|𝑠 𝑡 ). (12)
timestep is used as input to the Transformer Decoder. The action 𝑎𝑙
𝑡
, 𝑎∈
reward 𝑟𝑙 𝑡 , and the number of orders to execute 𝑂 𝑡 𝑙 is calculated as is By incorporating an entropy term, PPO can incentivize the agent to
describedin Section3.2.The LSTMCellalsoreceivesℎ𝑙 and𝑐𝑙 as explorealternativeactions, preventgettingtrappedinsuboptimalpoli-
𝑡−1 𝑡−1
inputanditsoutputincludes𝑐𝑙 aswell. cies, and mitigate overfitting to the training data (Mnih et al., 2016;
𝑡
5

### 8. S.Kimetal. Expert Systems With Applications252(2024)124263
Fig.6. VAAforthetopth...

S.Kimetal. Expert Systems With Applications252(2024)124263
Fig.6. VAAforthetopthreemodelsaspertheevaluationsin Table4, overaperiodof60 testdaysforselectedstocks. Themodelsinclude TUL, HUL, andthebest-performing
thirdmodelspecifictoeachstock. Notably, TULdemonstratesmoreconsistent VAAwithfewerfluctuationscomparedtoothermodelsacrossallstocks.
Fig.7. U-shapeaveragesontestdaysforgroundtruthsand Transformer-predictedvalues. Theinnerplotsdepicttheabsoluteerrorvaluesofthepredictionsforeachstock..
and the percentage of them that fall under 10 bps. These methods aresignificantforstockswithsmallvolumes, itischallengingtoobtain
areusedbypractitioners, tomeasurehowthetraderswereperforming accurate predictions of the daily volume distribution by only using a
relativetothe VWAP.Ourproposeddual-levelapproachdemonstrates statistical U-shape distribution. However, U-shape Transformer in the
significant improvements compared to using short trading horizons firstlevelcanproperlyreflectthedailyvariation, and TULoutperforms
and single-level neural networks. Although HUL occasionally yields bothmodelsanddemonstratesthelowestmeanandstandarddeviation
betteraccuracy, TULgenerallyhasmuchlessdeviationfromthedaily values. Thehistogramsfor VAAdistributionsgivenby DQN, OPD, PPO,
cumulative VWAP than all other methods. This becomes evident in HULand TULoneachstocktestdataaredrawnin Fig.9, whichdepicts
the less liquid stock PH stock, where HUL lagged behind both DQN theclusteringtendencyofaccuraciesproducedby TUL.
and PPO.Inthiscase, DQNand PPOexhibitedlowermeanandlower Thirdly, wecomparetheperformanceof TULand TURtoexamine
standarddeviationcomparedto HUL.Sincethedailyvolumevariations which neural network architecture is optimal for order allocation in
8

### 9. S.Kimetal. Expert Systems With Applications252(2024)124263
Table3
Hyperparameter...

S.Kimetal. Expert Systems With Applications252(2024)124263
Table3
Hyperparameters for DQN, OPD, PPO, HUL and TUL. Note that ↘ indicates a linearly annealing learning rate schedule. Network configurations of DQN and OPD follow Ning
etal.(2021) and Fangetal.(2021), respectively.
Hyperparameter DQN OPD PPO HUL TUL
Outeriterations 10000 10000 10000 10000 10000
Inneriterations 20 10 10 10 10
Batchsize 50 10 20 10 10
PPOCLIP𝜀 – 0.2 0.2 0.2 0.2
Discountfactor𝛾 – 1 1 1 1
Numberoftrajectories 1000 160 200 152 228
pereachouteriteration
LSTMhiddendimension𝐻 – – 128 128 129
LSTMmodellearningrate – – 5 e−5↘1 e−5 5 e−5↘1 e−5 5 e−5↘1 e−5
DQNor OPDmodellearningrate 1 e−4 1 e−4 – – –
U-shape Encoder – – – – 20
inputvectordimension𝑁
U-shape Encoder – – – – 148
embeddingdimension
Transformer Encoder – – – – 4
&Decodernumberofheads
Transformer Encoder – – – – 1
&Decodernumberoflayers
Transformer Encoder – – – – 128
&Decoder PFFNdimension
Transformermodellearningrate – – – – 1 e−3↘2 e−4
approximatesthe U-shapedistributionforthetestdatesaccuratelyand
uses predicted values for order allocation. Thus, TUL becomes more
adaptive in tracking the daily cumulative VWAP as it incorporates
predictionsoffluctuationsthatmayappearinthe U-shapedistribution
inthefuture.
Wealsovalidatetheperformanceimprovementsof TULcompared
toothermodels, particularly HUL, byexamining TUL’sabilitytocap-
turedailyfluctuationsinthe U-shape. Fig.8 displaysthegroundtruth
U-shapeand Transformer-predicteddistributionsfortwosampledtest
dates for the stock PH. We deliberately choose PH since the perfor-
mance gains of TUL when compared to HUL were the largest for this
particular stock, which has daily volume distributions that deviates
the most from the average due to relatively low liquidity. While the
U-shape Transformer may occasionally make erroneous predictions,
it quickly corrects itself and adapts to changes in the daily volume
distribution, attemptingtoaccuratelyfollowratiochangesthroughout
theday.
5. Discussionandconclusion
In this study, we propose a dual-level approach to address the
problemoftrackingthe VWAPwithaccuracyandconsistency. Initially,
as observed in Fig. 5, we identified that the naive uniform distribu-
Fig.8. Thegroundtruthand Transformer-predicted U-shapeplotsfortwoselectedtest
datesfor PH. Theinnerplotsdepicttheabsoluteerrorvaluesofthepredictions.. tion approach falls short in accurately tracking the daily VWAP. The
uniquenessofourmodelisinhowweimproveuponthisissuebytaking
into account the well-known U-shaped intraday trading pattern. In
the second level. For SE, which is the stock with the highest trading the first stage, we allocate the number of orders to execute in each
volume, TUR slightly underperforms HUL and TUL and outperforms interval based on the U-shape pattern. We consider two methods,
all other models. However, it is evident that a significant decline in statistical U-shapeand U-shape Transformer, forimplementingthefirst
performance occurs for all other stocks with lower trading volumes. stage. Thenaiveapproachofusingthestatistical U-shapedistribution
Thisperformancedegradationcanbeattributedtothelimitedcapabil- improvesperformance. However, ourexperimentsdemonstratethatthe
ity of the simpler RNN model when compared with the LSTM model U-shape Transformerperformsevenbetter. Webelievethatthesuperior
to capture volume fluctuations in stocks that exhibit relatively lower performanceofthe U-shape Transformerisduetoitsabilitytoadaptto
liquidity. unexpectedvariations, acapabilitythatthestatistical U-shapemethod
Lastly, we demonstrate the effectiveness of the proposed U-shape lacks, as evidenced by Figs. 7 and 8. After the total daily orders are
Transformerusing Figs.7 and8.Fig.7 showstheplotsoftheground allocatedtoeachintervalinthefirststage, the LSTMmodelisusedin
truth U-shapeand Transformer-predicted U-shapevalues, wherethe U- thesecondstagetodeterminehowtheordersineachintervalshould
shaperatiovaluesforeach20-minuteintervalwereaveragedacrossall beexecuted. Oursimulationresultsshowthatthedual-levelapproach
testdates. Weobservethatour U-shape Transformerisabletofollow consistentlyandaccuratelytracksthedailycumulative VWAP.
the U-shapepatternforthetestdatesonaverageforallstocks, which Whileourmethoddemonstratessuperiorperformancethroughex-
is in line with the comparable (and sometimes better) performance periments, a comprehensive mathematical understanding of its un-
of TUL against HUL. This is further supported by the fact that HUL derlying principles including uncertainty quantification remains to be
only utilizes the average U-shape ratio values of the past, while TUL explored. The limitations identified in our paper suggest potential
9

### 10. Expert Systems With Applications252(2024)124263
Contentslistsavailableat Science...

Expert Systems With Applications252(2024)124263
Contentslistsavailableat Science Direct
Expert Systems With Applications
journalhomepage:www.elsevier.com/locate/eswa
Anadaptivedual-levelreinforcementlearningapproachforoptimaltrade
execution
Soohan Kimd,1, Jimyeong Kimc,1, Hong Kee Sula,1, Youngjoon Hongb,1,∗
a Departmentof Finance, Chung-Ang University, Seoul, Republicof Korea
b Departmentof Mathematical Science, Korea Advanced Instituteof Scienceand Technology (KAIST), Daejeon, Republicof Korea
c Stochastic Analysisand Application Research Center (SAARC), Korea Advanced Instituteof Scienceand Technology (KAIST), Daejeon, Republicof Korea
d NYUCourant Instituteof Mathematical Sciences, NY, United Statesof America
A R T I C L E I N F O A B S T R A C T
Keywords: Thepurposeofthisresearchistodeviseatacticthatcancloselytrackthedailycumulativevolume-weighted
Volume-weightedaverageprice averageprice (VWAP) usingreinforcementlearningwhileminimizingthedeviationfromthe VWAP.Previous
Reinforcementlearning studies often choose a relatively short trading horizon to implement their models, making it difficult to
Optimaltradeexecution accurately track the daily cumulative VWAP since the stock price movement is often insignificant within
Proximalpolicyoptimization
theshorttradinghorizon. Ontheotherhand, trainingreinforcementlearningmodelsdirectlyoveralonger,
Markovdecisionprocess
dailyhorizonisburdensomeduetoextensivesequencelength. Hence, thereisaneedforamethodthatcan
dividethelongdailyhorizonintosmaller, moremanageablesegments. Weproposeamethodthatleverages
the U-shaped pattern of intraday stock trade volumes and uses Proximal Policy Optimization (PPO) as the
learningalgorithm. Ourmethodfollowsadual-levelapproach:a Transformermodelthatcapturestheoverall
(global) distributionofdailyvolumesina U-shape, anda LSTMmodelthathandlesthedistributionoforders
withinsmaller (local) timeintervals. Theresultsfromourexperimentssuggestthatthisdual-levelarchitecture
improvescumulative VWAPtrackingaccuracycomparedtopreviousreinforcementlearningapproaches. The
keyfindingisthatexplicitlyaccountingforthe U-shapedintradayvolumepatternleadstobetterperformance
inapproximatingthecumulativedaily VWAP.Thishasimplicationsfordevelopingtradingstrategiesthatneed
toefficientlytrack VWAPoverafulltradingday.
1. Introduction wantstoavoidmarketimpactandtheriskoffront-runnersexploiting
its strategy. To achieve this objective, the fund may outsource the
The optimal trade execution problem aims to find a strategy to trading to external brokerages or trading firms. In such a scenario,
optimallytradelargeorderswithinagivenperiodoftime. Oneofthe VWAP is commonly the most relevant metric for quantifying trading
mostcommonandpracticalmethodsthatpractitionersfrequentlyuseis performance. If the typical end-of-day market price were used as the
knownas Volume Weighted Average Price (VWAP) trading (Madhavan, performance metric, traders might execute all orders at the market’s
2002). Research regarding the optimality of VWAP as an execution close. This could lead the fund to suffer from market impact, and
strategy is discussed in Kato (2015), which further highlights the theend-of-daymarketpricemightbecomemorevolatile. Since VWAP
importanceoftheabilityofatradingmodeltotrack VWAPthroughout
reflectsastock’stransactionpriceovertime, itservesasamoresuitable
timeconsistently. VWAPiscalculatedbyaddingupthedollarstraded
metricfortradingperformance. Using VWAPallowsthefundtoexecute
foreverytransaction (pricemultipliedbythenumberofsharestraded)
its strategy with minimal market impact, thereby avoiding potential
andthendividingbythetotalsharestradedfortheday. Thisgivesan
lossesfrommarketspeculationandfront-running.
averagepricethattakesintoaccountboththepriceandthevolumeof
In Bertsimasand Lo (1998), afundamentalworkforoptimaltrade
sharestraded.
execution is proposed, where the authors assume that market prices
Funds and traders often use VWAP as a benchmark to compare
followanarithmeticrandomwalk. Theyuseadynamicprogramming
the price at which they executed trades to the overall market price
principle to find an explicit closed-form solution. Building on this
for the security, in order to evaluate the performance of their trades.
work, Huberman and Stanzl (2005) and Almgren and Chriss (2001)
Suppose a fund intends to significantly change its stock holdings but
∗ Correspondingauthor.
E-mailaddresses: soo.han.kim@nyu.edu (S.Kim), jimyeongkim@kaist.ac.kr (J.Kim), hksul@cau.ac.kr (H.K.Sul), hongyj@kaist.ac.kr (Y.Hong).
1 Theseauthorscontributedequallytothiswork.
https://doi.org/10.1016/j.eswa.2024.124263
Received12 May2023;Receivedinrevisedform7 May2024;Accepted14 May2024
Availableonline17 May2024
0957-4174/©2024 Elsevier Ltd. Allrightsarereserved, includingthosefortextanddatamining, AItraining, andsimilartechnologies.

### 11. S.Kimetal. Expert Systems With Applications252(2024)124263
extendedtheresultin B...

S.Kimetal. Expert Systems With Applications252(2024)124263
extendedtheresultin Bertsimasand Lo (1998) byincorporatingtrans-
action costs, more complex price impact functions, and risk aversion
parameters, undertheassumptionthatmarketpricesfollowa Brownian
motion. These dynamical approaches, however, are difficult to apply
directlyintherealworldduetothediscrepancybetweentheirstrong
marketassumptionsandthecomplexityofactualmarketconditions.
Ontheotherhand, bothpractitionersandresearchershavewidely
usedthetime-weightedaverageprice (TWAP) strategyandthevolume-
weighted average price (VWAP) strategy (Berkowitz et al., 1988;
Kakadeetal.,2004), whicharebasedoneitherpurerulesorstatistical
rules. Especially in Kakade et al. (2004), the authors used historical
datatoestimatetheaveragevolumetradedforeachtimeintervaland
split the order accordingly. However, this strategy is not well-suited
forcapturingunexpectedvolatility.
Fromtheperspectivethatoptimalexecutionproblemsareatypeof
sequentialdecision-makingtask, RLhasbeencommonlyappliedinthis
fieldasitisatypeofstochasticdecision-makingprocessthatautomates
thepractitioner’staskofusingpastdatatomakedecisionsonwhento
execute orders. Since RL approaches have many advantages, e.g. ca-
pable of capturing the market’s microstructure, the trading results
conductedby RLsolutionsoftenoutperformtraditional VWAPtracking
approachessuchas Bialkowskietal.(2008) and Podobniketal.(2009).
Tothebestofourknowledge, Nevmyvakaetal.(2006) isthefirstwork
toleverage RLframeworkssuchas Q-learning (Watkins&Dayan,1992)
to optimal trade execution problems. It is important to note that the
curse of dimensionality in Q-learning makes it challenging to handle
high-dimensionaldata. While (Hendricks&Wilcox,2014) attemptsto
combine RLwiththe Almgren–Chrissmodel, itisachallengingtaskas
themodeldependsoncertainassumptionsaboutthemarketdynamics.
Thankstotheadvancementsindeep RL, recentstudiessuchas Macri
and Lillo (2024), Ningetal.(2021), and Linand Beling (2019) have
utilized Deep Q-Networks (DQNs)(Mnihetal.,2013) foroptimaltrade
execution, addressingthechallengesofhigh-dimensionaldataandthe
complexity of the financial market without relying on any market
assumptions. However, these methods require the design of specific
attributes, whichcanbelabor-intensive. Recently, severalstudieshave Fig.1. Tradevolumeratioovereach20-minuteperiodthroughouttheday. Thelines
investigatedtheuseofproximalpolicyoptimization (PPO)(Schulman represent the 1-year averages of the ratios and the shaded regions are drawn from
et al., 2017) based on optimal execution frameworks, which do not dailydeviationsfromtheaverages.
require manually designed feature engineering. For instance, Lin and
Beling (2020), Fang et al. (2021), Pan et al. (2022), and Byun et al.
(2023) have explored this approach in different scenarios. Specifi- will be executed in each interval. We propose two methods for the
cally, Panetal.(2022) focusedonoptimalexecutionwithlimitorders, firststage, thestatistical U-shapemethodandthe U-shape Transformer
while Lin and Beling (2020) and Fang et al. (2021) used market method. Thestatistical U-shapemethodallocatesthevolumebasedon
orders. Amethodcombiningbothlimitandmarketorderswasproposed the historical average trade volumes. However, this approach cannot
by Byunetal.(2023). fully capture the day-to-day variations (See Fig. 1). Alternatively, we
Prior literature often considers a relatively short trading horizon propose the U-shape Transformer for figuring out these variations.
ortheentiredayfor RLimplementation. However, VWAPapproxima- In the next stage, we apply LSTM, trained by the RL framework to
tion becomes challenging when a short time frame is used since the distribute orders efficiently within each interval. It is important to
variations of financial data are often insignificant within short time note that there are numerous RL training algorithms, including Trust
intervals. Moreover, thenaiveapproachofconsideringtheentireday Region Policy Optimization (TRPO) (Schulman et al., 2015), Deep Q-
canputastrainonthenetworkarchitectureduetothelongsequence Networks (DQN)(Mnihetal.,2013), and Proximal Policy Optimization
length. This paper aims to develop a strategy that can consistently (PPO) (Schulman et al., 2017). Among these options, we specifically
trackthedailycumulative VWAP.Toachievethisgoal, westartwith choose PPOasourtrainingalgorithm. Foracomprehensiveunderstand-
ingofthesetechniquesandtheirapplications, wereferto Suttonand
the observation of an important stock trading characteristic, the U-
Barto (2018).
shaped intraday trading pattern, documented in Jain and Joh (1988)
We expect that this dual-level approach improves the accuracy of
and Goodhart and O’Hara (1997). This is a well-known stylized fact
approximatingthecumulative VWAPbyproperlydistributingthetotal
statingthatastock’stradingvolumeoftenfollowsa U-shapedpattern
order. Ourcontributionsareasfollows:
throughouttheday. Thismeansthatvolumeishighestattheopening,
falls rapidly to lower levels, and then rises again towards the close 1. We propose a novel approach for optimal trade execution that
of the market, but is relatively low in the afternoon. (See Fig. 1). In utilizes a dual-level strategy. Orders are allocated through two
view of this observation, we propose a novel dual-level approach to stages: in the first stage we utilize the U-shaped pattern of
minimize the market impact and track the daily cumulative VWAP intraday volumes and in the second stage we implement deep
accuratelyandconsistently. Asthenamesuggests, ourmodelconsists reinforcementlearning. Sincethisnoveldualapproachsegments
of two stages. In the first stage, the model decides how much trade the data, we not only make reinforcement learning more com-
volume will be executed in each interval, using the U-shape property putationallytractablebutalsoimprovepredictionaccuracycom-
asaguideline. Inthesecondstage, the RLmodeldecideshowtheorders paredtoapplying RLdirectlyonfullintradaysequences.
2

### 12. S.Kimetal. Expert Systems With Applications252(2024)124263
Fig.4. Thearchitectur...

S.Kimetal. Expert Systems With Applications252(2024)124263
Fig.4. Thearchitectureofthe LSTMmodel.
Williams, 1992). To find the parameter 𝜃, PPO iteratively gathers Algorithm1 Dual-level Neural Network
episodesandupdatestheparameter𝜃usingthefollowingscheme: Require: Number of 𝑜𝑢𝑡𝑒𝑟 𝑖𝑡𝑒𝑟𝑎𝑡𝑖𝑜𝑛𝑠 and 𝑛𝑢𝑚
𝜃 =argmax E E [𝐽 (𝜃)], (13) 𝑑𝑎𝑦𝑠
𝑘+1 𝜃 𝑠 𝑎∼𝜋𝜃𝑘 (⋅|𝑠) PPO Randomlyinitializelearnableparameters𝜃,𝜃, and𝜃
where𝑘standsforthe𝑘thstep. The‘‘clip’’operatorin (8) allows PPO Initializetrajectorybuffer
tolearnfrompreviousexperienceswithoutbecomingoverlydependent for𝑗=1 to𝑜𝑢𝑡𝑒𝑟𝑖𝑡𝑒𝑟𝑎𝑡𝑖𝑜𝑛𝑠do
on them. Moreover, PPO can effectively retain knowledge from past for𝑑=1 to𝑛𝑢𝑚𝑑𝑎𝑦𝑠do
experiences while also acquiring insights from new experiences. This Randomlychooseadateinthetrainingset
keyattributeof PPOisoneofitssignificantadvantages. 𝜇←theaverageofthedailyvolumeforthelast60 days
𝜎 ←thestandarddeviationofthedailyvolumeforthelast60
4. Experiments days
 ← ⋃ Gathering Episodes (date,𝜇,𝜎)
Our dual-level approach significantly enhances the accuracy and endfor
consistency of tracking the daily cumulative VWAP, despite the chal- Using , update ,  by optimizing the loss function 𝐽 in (5)
𝑇𝐹
lenges in improving the performance of RL frameworks for optimal withrespectto𝜃 and𝜃
trade execution in relatively short trading horizons. Furthermore, the Optimizethelossfunction𝐽 in (9) withrespectto𝜃
PPO
U-shape Transformerprovesefficientincapturingthedailyvariations  ←∅
of the U-shape distribution. In the following section, we conduct a endfor
seriesofexperimentstoverifytheseassertions.
4.1. Experimentsettings Algorithm2 Gathering Episodes
Require:Thenumberofintervalsinaday𝐿
4.1.1. Implementation Input:Date,𝜇,𝜎
Inallofourexperiments, thehyperparametersforthelossfunctions, Output:Trajectorybufferof𝑖𝑡ℎday
𝑖
thelengthoftheintervals, thenumberoftotalorders, andthetrading Initialize
𝑖
horizonarefixedandsetaccordingtothefollowingspecifications: 𝑂∼(2.5×10−3𝜇,6.25×10−6𝜎2)
𝐸 =(𝐸 )
- Thecoefficientsfor𝐽 :𝑐 =0.5,𝑐 =0.5 𝑜𝑢𝑡 𝑖𝑛
TF 1 2 for𝑙=0 to𝐿−1 do
- Thecoefficientsfor𝐽 :𝑐 =1,𝑐 =0.01
PPO 3 4 if𝑙=0 then
- Thelengthoftheintervals:𝐿=19,𝑇 =20 Construct𝐷0 fromrawpre-marketdata
- Thenumberoftotalorders:𝑂 ∼(2.5×10−3𝜇,6.25×10−6𝜎2), 𝑖𝑛
else
where𝜇and𝜎istheaverageandstandarddeviationoftheday 𝐷𝑙 ←𝐷𝑙−1⋃𝐶𝑜𝑛𝑐𝑎𝑡(𝐷𝑙−1,ℎ𝑙−1)
totalvolumefortheprevioussixtydays, respectively. 𝑖𝑛 𝑖𝑛 𝑜𝑢𝑡 𝑇−1
endif
- Tradinghorizonofaday:380 minutes (Fromwhenthemarket Obtain𝑢𝑙 and𝐷𝑙 byproceduredepictedin Section3.3.1
opensat09:00:00 towhenitclosesat15:20:00). 𝑝𝑟𝑒𝑑 𝑜𝑢𝑡
𝑂𝑙←𝑂⋅𝑢𝑙
𝑝𝑟𝑒𝑑
take S s in t c h e e a da t i y ly pi t c o a t l al da o y rd a e l r l , o 𝑂 w , s a f n o d r b 3 r 8 e 0 ak m s i i n td o o f w tr n ad to in 𝑂 g, 𝑙, o 1 u 9 r i m nt o e d r e v l al f s ir o st f O pr b o t c a e in du 𝜏 r 𝑙 e = d { e 𝑠 p 𝑙 𝑡 i , c 𝑎 t 𝑙 𝑡 e , d 𝑟( i 𝑎 n 𝑙 𝑡 ), S 𝑉 ec ( t 𝑠 i 𝑙 𝑡 o ), n 𝜋 3 (𝑎 .3 𝑙 𝑡| . 𝑠 2 𝑙 𝑡 ),𝑂 𝑡 𝑙} 𝑡∈{0,1,2,⋯,𝑇−1} andℎ𝑙 𝑇−1 by
20 mineach.(𝑙∈[0,18])Weeitherapplythestatistical U-shapemethod  ← ⋃𝜏
𝑖 𝑖 𝑙
orthe Transformermodelforthisfirststageallocationprocess. Then, endfor
withineachinterval, the LSTMmodelfurthersubdivides𝑂𝑙into20 sets
of1-minutesubintervals,𝑂𝑙.(𝑡∈[0,19])
𝑡
Ineachofthesesubintervals,𝑂𝑙 isdividedintoequalportionsand
𝑡
stock, weaccountforthedifferencesintotalordersforvariousstocks,
executed over a span of 12 steps, assuming that orders are executed
whichisoftenencounteredinreal-worldscenarios. Additionally, dur-
every 5 s. The method and hyperparameters used to train our agents
ingtesting, weonlyusethedailygroundtruth U-shapevolumeratios
for the experiment are summarized in Algorithms 1, 2, and Table 3
of dates in the training data for our Transformer Encoder to prevent
whichsilocatedattheendofthepaper.
lookingaheadintothefuture. Asinpreviousworksof Nevmyvakaetal.
To conduct realistic simulations, we determine 𝑂 in a way that
(2006), Hendricksand Wilcox (2014), Ningetal.(2021), and Linand
takes into account the volatility of daily volume, which differs from
Beling (2020), wemakethefollowingassumptionsinourexperiments.
most previous works where it is fixed to a certain value. This choice
aims to simulate the fluctuations of total orders that financial firms 1. We assume that the actions taken by our model only affect
havetoexecuteinaday, takingintoconsiderationrecenttradevolume a temporary market, and that the market will recover to the
statistics for each stock. By considering the volume statistics for each equilibriumlevelatthenexttimestep.
6

### 13. S.Kimetal. Expert Systems With Applications252(2024)124263
Table2 theexperimenta...

S.Kimetal. Expert Systems With Applications252(2024)124263
Table2 theexperimentalsetupin Ningetal.(2021).Thetotalorderfor
Averagevolumeforeachstockfrom January1 st,2021 to December31 st,2021. thedayisthenequallydistributedamongsteachhour, and DQN
Stocks SE SH KC PH furtherallocatestheorderforfivesub-intervalsof12 mineach.
Volume 17,000 K 3800 K 3900 K 430 K The orders are executed equally for each 5-second time step
withinthe12-minutesub-intervals. Toensurefaircomparisons,
weevaluatethedailypricegeneratedby DQNagainstthedaily
cumulative VWAPcalculatedbyexcludingthefinal20 min.
- OPDisthemethodproposedin Fangetal.(2021).Toimplement
this model, we first divide a given day into ten intervals, each
spanning38 min, andthenuse OPDtoallocateordersforeach
interval. Within each interval, the allocated order is equally
distributedamongsteachminute.
- PPOisthemethodproposedin Linand Beling (2020).Toapply
this model, we equally distribute the day total order to each
minute in the day, i.e. 𝑂 = 𝑂∕380 for all 𝑗 ∈ {1,…,380} as
𝑗
in Linand Beling (2020).
- (HU-)PPO utilizes the statistical U-shape to determine 𝑂𝑙, and
within each interval, the allocated order is equally divided
amongeveryminute, i.e.𝑂𝑙=𝑂𝑙∕20 forall𝑡∈{0,…,19}, where
𝑡
Fig. 5. The VAA’s are depicted in dark blue for the best-case scenarios of uniform PPOisfinallyusedtodistributeorderswithintheminute.
distributionand U-shapedistributionorderallocationschemes, whiletheirworst-case
- HULisourproposeddual-levelapproachwhichusesthestatis-
counterparts are shown in brown. The uniform distribution and U-shape distribution
tical U-shapetodetermine𝑂𝑙.
allocate total orders to be executed per minute in the same manner as PPO and
(HU-)PPO, respectively.. - TULisourproposeddual-levelapproachthatemploysthe Trans-
formermodeltodetermine𝑂𝑙.
- TURuses RNNstoallocateorderswithineachinterval (second
2. Commissionsandexchangefeesareignored. level) instead of LSTMs. Other configurations are the same as
3. Ourmodel’sordersaretradedimmediatelywithoutorderarrival thatof TUL.
delays.
Weevaluatetheperformanceofthesemethodsalongsideourproposed
Webelievethattheaboveassumptionsarereasonableasweconsider dual-levelapproachbasedonthe VAAmetricin (7).Aswementioned
relatively small total orders in comparison to the daily total market in Section3.3.3, the VAAdirectlymeasurestheabilityofatradingstrat-
volumesofthestocks. egytocloselytrackandapproximatethedaily VWAP.Consequently, to
comprehensivelyassessabilityofeachmethodtoconsistentlytrackthe
4.1.2. Datasets daily VWAP over an extended period, we analyze both the mean and
Ourmillisecondtradeandlimitorderbook (LOB) tickdataisfrom variationofthe VAAmetric.
the Korea Exchange (KRX). We use the daily trade and LOB data of
Samsung Electronics (SE), SK Hynix (SH), Kia Corporation (KC) and 4.2. Experimentresults
POSCO Holdings Inc (PH) from January 1 st, 2021 to December 31 st,
2021. Our choice of the four firms in the KOSPI index is based on We test our models, which are trained for 100,000 iterations, by
their liquidity and trade volume and variety in market capitalization conductingtradingsimulationsinthebuyingdirection. Wefirstverify
andindustry. Inordertoevaluatetheeffectivenessofourapproachin the claim of the ineffectiveness of short trading horizons when order
relationtotradingvolume, wehaveoptedtoselectstockswitharange allocation does not consider the appropriate distribution method for
oftradevolumes. Specifically, weselectedstocksrepresentingdifferent volumesfromlongtradinghorizons.
tradevolumelevels—SEforlarge, SHand KCformedium, and PHfor Todothis, weexaminedtwodistinctscenarios:firstly, weuniformly
small volumes. The average daily trading volumes for each stock can distributed volumes from a long trading horizon into a short trading
be found in Table 2. We divide the data into two sets, using January horizon and secondly, we employed a statistical U-shape distribution
1 st to September 30 th as training data and October 1 st to December to allocate volumes from a long trading horizon into a short trading
31 stastestdata. horizon. Note that the volumes from a long trading horizon mean
The millisecond raw data is preprocessed by first dividing them dailytotalorders, whileashorttradinghorizonrepresentsa1-minute
into groups of 5 s, and extracting data that represents each 5-second tradinghorizon. Tocomparethebestandworstcases, wedeliberately
interval. We take the last LOB data of the 5-second interval to con- executed trades for the lowest and highest buy orders, respectively,
structthe MDPstateandthe5-second VWAPandtotaltradedvolume within the 1-minute trading horizon. Fig. 5 displays the comparison
for calculating the daily VWAP. The statistical U-shape statistics are ofthe VAA’sonthetestdates. Wecanobservethattheworstcasefor
constructed by taking the year average of the U-shape ratios of the U-shapedistributionoutperformsthebestcaseforuniformdistribution.
20-minute intervals in the day. The data utilized in generating 𝐷0 Specifically, the VAA’smeanoftheworstcasefor U-shapedistribution
𝑖𝑛
shownin Section3.3 arethetop5 bid/askvolumeaveragesoftheraw is10 bpswhilethe VAA’smeanofthebestcaseforuniformdistribution
pre-marketdata (from08:30:00 to09:00:00). is 11 bps. This result suggests that in the context of VWAP, the use
of deep reinforcement learning does not significantly improve the
4.1.3. Comparisonandablationstudy approximation capability without taking into account the long-term
Weprovideabriefexplanationofallmethodsusedforperformance trading horizon. This implies that in order to effectively leverage the
comparisonduringourexperimentshere. Thehyperparametersforeach benefitsofdeep RLinthecontextof VWAPs, itshouldbeimplemented
methodcanbefoundin Table3. onalongertradinghorizonsuchasasingletradingday
Secondly, we present Table 4 and Fig. 6 to demonstrate how our
- DQN is the method proposed in Ning et al. (2021). To apply dual-levelapproachsignificantlyimprovesperformance.
this model, we begin by dividing a given day into six 1-hour Table 4 reports the mean and standard deviation values of VAA’s
intervals, excludingthefinal20 minofmarkethourstoreplicate produced by DQN, OPD, PPO, HUL and TUL across the test dates,
7

### 14. S.Kimetal. Expert Systems With Applications252(2024)124263
Fig.9. Thehistogramso...

S.Kimetal. Expert Systems With Applications252(2024)124263
Fig.9. Thehistogramsofthe VAAdistributionsforeachmodelandstockonthetestdates. Both TULand HUL, employingthe U-shapeproperty, exhibitconsistentstabilityin
trackingthe VWAPfor SE, SH, and KCstocks. Moreover, TULdemonstratesrobusttrackingforthe PHstock, whichischaracterizedbysignificantdailyvolumefluctuations.
Table4
Table4 summarizesthemean, standarddeviation, andpercentageof VAAswithintherange[0,10 bps].The Meanand Standard Deviationarerepresentedinunitsofbasispoints
(bps), andthe%in10 bpsarerepresentedinunitsofpercentages. For OPD, weexcluded VAAsabovethe80 thpercentileforfaircomparisonssinceourtrainingdatasetwas
smallerthanthatusedin Fangetal.(2021).For TUR, weexcludedextremeoutliersfor SH, KA, and PH.
Stocks Metric OPD DQN PPO HUL TUL TUR
Mean 26.27 11.53 12.00 8.77 7.13 9.43
SE Standard Deviation 18.50 8.84 8.88 7.17 2.44 9.56
%in10 bps 25.00 57.38 49.18 65.57 88.52 65.57
Mean 38.55 21.62 22.57 7.13 7.65 48.56
SH Standard Deviation 24.90 17.71 16.92 2.81 3.63 33.51
%in10 bps 10.42 30.00 25.00 91.67 88.33 5.45
Mean 27.79 13.16 12.47 6.59 6.03 149.32
KA Standard Deviation 18.76 10.25 11.55 8.36 3.30 111.64
%in10 bps 18.75 46.67 56.67 90.00 86.67 9.43
Mean 32.44 14.84 14.64 22.52 12.27 377.28
PH Standard Deviation 20.89 13.47 12.63 47.11 7.90 267.23
%in10 bps 18.75 40.98 40.98 32.79 44.26 1.85
directionsforfutureresearch. Additionally, wenotethatourproposed CRedi Tauthorshipcontributionstatement
methodentailsthefollowingethicalimplications. Firstly, ourapproach
providesinvestorswithnewoptionsfordealingwithmarketvolatility, Soohan Kim: Writing – original draft, Methodology, Software.
helpingtoavoidmarketimpactandtheriskoffront-runnersexploiting Jimyeong Kim: Writing – original draft, Data curation, Conceptual-
theirstrategies. Secondly, improvedabilityofourexecutionmodelto ization, Visualization. Hong Kee Sul: Writing – review & editing, In-
track the VWAP contributes to mitigating market impact, which can vestigation. Youngjoon Hong:Writing–review&editing, Supervision,
potentiallyenhancemarkettransparencyandfairness. Projectadministration.
10

### 15. S.Kimetal. Expert Systems With Applications252(2024)124263
Declarationofcompetin...

S.Kimetal. Expert Systems With Applications252(2024)124263
Declarationofcompetinginterest Kakade, S.M., Kearns, M., Mansour, Y.,&Ortiz, L.E.(2004).Competitivealgorithms
for VWAP and limit order trading. In Proceedings of the 5 th ACM conference on
Theauthorsdeclarethefollowingfinancialinterests/personalrela- electroniccommerce (pp.189–198).
Kato, T.(2015).VWAPexecutionasanoptimalstrategy. JSIAMLetters,7,33–36.
tionships which may be considered as potential competing interests:
Lin, S.,&Beling, P.A.(2019).Optimalliquidationwithdeepreinforcementlearning.
Youngjoon Hongreportswasprovidedby Sungkyunkwan University. In Proceedingsofthe33 rdconferenceonneuralinformationprocessingsystems, deep
reinforcementlearningworkshop. Vancouver, Canada.
Dataavailability Lin, S.,&Beling, P.A.(2020).Anend-to-endoptimaltradeexecutionframeworkbased
onproximalpolicyoptimization..In IJCAI(pp.4548–4554).
Macri, A., & Lillo, F. (2024). Reinforcement learning for optimal execution when
Datawillbemadeavailableonrequest.
liquidityistime-varying.ar Xivpreprintar Xiv:2402.12049 v2.
Madhavan, A.(2002).VWAPstrategies. Trading,1,32–39.
Acknowledgments Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T., Harley, T., Silver, D., &
Kavukcuoglu, K. (2016). Asynchronous methods for deep reinforcement learning.
In Internationalconferenceonmachinelearning (pp.1928–1937).PMLR.
Theworkof Y.Hongwassupportedby Basic Science Research Pro-
Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D., &
gramthroughthe National Research Foundationof Korea (NRF) funded Riedmiller, M.(2013).Playingatariwithdeepreinforcementlearning. URL:http:
bythe Ministryof Education, Republicof Korea (NRF-2021 R1 A2 C1093 //arxiv.org/abs/1312.5602. Cite arxiv:1312.5602 Comment: NIPS Deep Learning
579) andbythe Koreagovernment (MSIT)(RS-2023-00219980). Workshop2013.
Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D., &
Riedmiller, M. (2013). Playing atari with deep reinforcement learning. ar Xiv
References
preprintar Xiv:1312.5602.
Nevmyvaka, Y., Feng, Y.,&Kearns, M.(2006).Reinforcementlearningforoptimized
Almgren, R.,&Chriss, N.(2001).Optimalexecutionofportfoliotransactions. Journal trade execution. In Proceedings of the 23 rd international conference on machine
of Risk,3,5–40. learning (pp.673–680).
Berkowitz, S.A., Logue, D.E.,&Noser, E.A., Jr.(1988).Thetotalcostoftransactions Ning, B., Lin, F. H. T., & Jaimungal, S. (2021). Double deep q-learning for optimal
onthe NYSE.The Journalof Finance,43(1),97–112. execution. Applied Mathematical Finance,28(4),361–380.
Bertsimas, D., & Lo, A. W. (1998). Optimal control of execution costs. Journal of Pan, F., Zhang, T., Luo, L., He, J.,&Liu, S.(2022).Learncontinuously, actdiscretely:
financialmarkets,1(1),1–50. Hybrid action-space reinforcement learning for optimal execution. In IJCAI (pp.
Bialkowski, J., Darolles, S.,&Fol, G.L.(2008).Improving VWAPstrategies:Adynamic 3912–3918).
volumeapproach. Journalof Banking&Finance,32,1709–1722. Podobnik, B., Horvatic, D., Petersen, A.M.,&Stanley, H.E.(2009).Cross-correlations
Byun, W., Choi, B., Kim, S.,&Jo, J.(2023).Practicalapplicationofdeepreinforcement betweenvolumechangeandpricechange. Proceedingsofthe National Academyof
learningtooptimaltradeexecution. Fin Tech,2,414–429. Sciences,106(52),22079–22084.
Fang, Y., Ren, K., Liu, W., Zhou, D., Zhang, W., Bian, J., Yu, Y., & Liu, T.-Y. Schulman, J., Levine, S., Abbeel, P., Jordan, M.,&Moritz, P.(2015).Trustregionpolicy
(2021). Universal trading for order execution with oracle policy distillation. 35, optimization. In Internationalconferenceonmachinelearning (pp.1889–1897).PMLR.
In Proceedingsofthe AAAIconferenceonartificialintelligence (1),(pp.107–115). Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal
Goodhart, C.A.,&O’Hara, M.(1997).Highfrequencydatainfinancialmarkets:Issues policyoptimizationalgorithms.ar Xivpreprintar Xiv:1707.06347.
andapplications. Journalof Empirical Finance,4,73–114. Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction. MIT
Hendricks, D., & Wilcox, D. (2014). A reinforcement learning extension to the Press.
almgren-chrissframeworkforoptimaltradeexecution. In2014 IEEEconferenceon Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser,Ł.,
computationalintelligenceforfinancialengineering&economics (CIFEr)(pp.457–464). &Polosukhin, I.(2017).Attentionisallyouneed. Advancesin Neural Information
IEEE. Processing Systems,30.
Huberman, G.,&Stanzl, W.(2005).Optimalliquiditytrading. Reviewof Finance,(2), Watkins, C.J.,&Dayan, P.(1992).Q-learning. Machine Learning,8(3),279–292.
165–200. Williams, R.J.(1992).Simplestatisticalgradient-followingalgorithmsforconnectionist
Jain, P.C.,&Joh, G.-H.(1988).Thedependencebetweenhourlypricesandtrading reinforcementlearning. Reinforc Ement Learning,5–32.
volume. The Journalof Financialand Quantitative Analysis,23(3),269–283.
11


---

## Raw Markitdown Extraction (full text)

ExpertSystemsWithApplications252(2024)124263
ContentslistsavailableatScienceDirect
ExpertSystemsWithApplications
journalhomepage:www.elsevier.com/locate/eswa
Anadaptivedual-levelreinforcementlearningapproachforoptimaltrade
execution
SoohanKimd,1,JimyeongKimc,1,HongKeeSula,1,YoungjoonHongb,1,∗
aDepartmentofFinance,Chung-AngUniversity,Seoul,RepublicofKorea
bDepartmentofMathematicalScience,KoreaAdvancedInstituteofScienceandTechnology(KAIST),Daejeon,RepublicofKorea
cStochasticAnalysisandApplicationResearchCenter(SAARC),KoreaAdvancedInstituteofScienceandTechnology(KAIST),Daejeon,RepublicofKorea
dNYUCourantInstituteofMathematicalSciences,NY,UnitedStatesofAmerica
A R T I C L E I N F O A B S T R A C T
Keywords: Thepurposeofthisresearchistodeviseatacticthatcancloselytrackthedailycumulativevolume-weighted
Volume-weightedaverageprice averageprice(VWAP)usingreinforcementlearningwhileminimizingthedeviationfromtheVWAP.Previous
Reinforcementlearning studies often choose a relatively short trading horizon to implement their models, making it difficult to
Optimaltradeexecution accurately track the daily cumulative VWAP since the stock price movement is often insignificant within
Proximalpolicyoptimization
theshorttradinghorizon.Ontheotherhand,trainingreinforcementlearningmodelsdirectlyoveralonger,
Markovdecisionprocess
dailyhorizonisburdensomeduetoextensivesequencelength.Hence,thereisaneedforamethodthatcan
dividethelongdailyhorizonintosmaller,moremanageablesegments.Weproposeamethodthatleverages
the U-shaped pattern of intraday stock trade volumes and uses Proximal Policy Optimization (PPO) as the
learningalgorithm.Ourmethodfollowsadual-levelapproach:aTransformermodelthatcapturestheoverall
(global)distributionofdailyvolumesinaU-shape,andaLSTMmodelthathandlesthedistributionoforders
withinsmaller(local)timeintervals.Theresultsfromourexperimentssuggestthatthisdual-levelarchitecture
improvescumulativeVWAPtrackingaccuracycomparedtopreviousreinforcementlearningapproaches.The
keyfindingisthatexplicitlyaccountingfortheU-shapedintradayvolumepatternleadstobetterperformance
inapproximatingthecumulativedailyVWAP.Thishasimplicationsfordevelopingtradingstrategiesthatneed
toefficientlytrackVWAPoverafulltradingday.
1. Introduction wantstoavoidmarketimpactandtheriskoffront-runnersexploiting
its strategy. To achieve this objective, the fund may outsource the
The optimal trade execution problem aims to find a strategy to trading to external brokerages or trading firms. In such a scenario,
optimallytradelargeorderswithinagivenperiodoftime.Oneofthe VWAP is commonly the most relevant metric for quantifying trading
mostcommonandpracticalmethodsthatpractitionersfrequentlyuseis performance. If the typical end-of-day market price were used as the
knownasVolumeWeightedAveragePrice(VWAP)trading(Madhavan, performance metric, traders might execute all orders at the market’s
2002). Research regarding the optimality of VWAP as an execution close. This could lead the fund to suffer from market impact, and
strategy is discussed in Kato (2015), which further highlights the theend-of-daymarketpricemightbecomemorevolatile.SinceVWAP
importanceoftheabilityofatradingmodeltotrackVWAPthroughout
reflectsastock’stransactionpriceovertime,itservesasamoresuitable
timeconsistently.VWAPiscalculatedbyaddingupthedollarstraded
metricfortradingperformance.UsingVWAPallowsthefundtoexecute
foreverytransaction(pricemultipliedbythenumberofsharestraded)
its strategy with minimal market impact, thereby avoiding potential
andthendividingbythetotalsharestradedfortheday.Thisgivesan
lossesfrommarketspeculationandfront-running.
averagepricethattakesintoaccountboththepriceandthevolumeof
InBertsimasandLo(1998),afundamentalworkforoptimaltrade
sharestraded.
execution is proposed, where the authors assume that market prices
Funds and traders often use VWAP as a benchmark to compare
followanarithmeticrandomwalk.Theyuseadynamicprogramming
the price at which they executed trades to the overall market price
principle to find an explicit closed-form solution. Building on this
for the security, in order to evaluate the performance of their trades.
work, Huberman and Stanzl (2005) and Almgren and Chriss (2001)
Suppose a fund intends to significantly change its stock holdings but
∗ Correspondingauthor.
E-mailaddresses: soo.han.kim@nyu.edu(S.Kim),jimyeongkim@kaist.ac.kr(J.Kim),hksul@cau.ac.kr(H.K.Sul),hongyj@kaist.ac.kr(Y.Hong).
1 Theseauthorscontributedequallytothiswork.
https://doi.org/10.1016/j.eswa.2024.124263
Received12May2023;Receivedinrevisedform7May2024;Accepted14May2024
Availableonline17May2024
0957-4174/©2024ElsevierLtd.Allrightsarereserved,includingthosefortextanddatamining,AItraining,andsimilartechnologies.

S.Kimetal.
ExpertSystemsWithApplications252(2024)124263
extendedtheresultinBertsimasandLo(1998)byincorporatingtrans-
| action costs, | more | complex | price | impact | functions, | and | risk aversion |     |     |     |     |     |     |     |
| ------------- | ---- | ------- | ----- | ------ | ---------- | --- | ------------- | --- | --- | --- | --- | --- | --- | --- |
parameters,undertheassumptionthatmarketpricesfollowaBrownian
| motion. These | dynamical | approaches, |     | however, |     | are difficult | to apply |     |     |     |     |     |     |     |
| ------------- | --------- | ----------- | --- | -------- | --- | ------------- | -------- | --- | --- | --- | --- | --- | --- | --- |
directlyintherealworldduetothediscrepancybetweentheirstrong
marketassumptionsandthecomplexityofactualmarketconditions.
Ontheotherhand,bothpractitionersandresearchershavewidely
usedthetime-weightedaverageprice(TWAP)strategyandthevolume-
| weighted | average | price (VWAP) |     | strategy | (Berkowitz | et  | al., 1988; |     |     |     |     |     |     |     |
| -------- | ------- | ------------ | --- | -------- | ---------- | --- | ---------- | --- | --- | --- | --- | --- | --- | --- |
Kakadeetal.,2004),whicharebasedoneitherpurerulesorstatistical
| rules. Especially | in  | Kakade | et al. | (2004), | the authors | used | historical |     |     |     |     |     |     |     |
| ----------------- | --- | ------ | ------ | ------- | ----------- | ---- | ---------- | --- | --- | --- | --- | --- | --- | --- |
datatoestimatetheaveragevolumetradedforeachtimeintervaland
| split the | order accordingly. |     | However, | this | strategy | is not | well-suited |     |     |     |     |     |     |     |
| --------- | ------------------ | --- | -------- | ---- | -------- | ------ | ----------- | --- | --- | --- | --- | --- | --- | --- |
forcapturingunexpectedvolatility.
Fromtheperspectivethatoptimalexecutionproblemsareatypeof
sequentialdecision-makingtask,RLhasbeencommonlyappliedinthis
fieldasitisatypeofstochasticdecision-makingprocessthatautomates
thepractitioner’staskofusingpastdatatomakedecisionsonwhento
| execute orders. | Since     | RL approaches |     | have            | many | advantages, | e.g. ca- |     |     |     |     |     |     |     |
| --------------- | --------- | ------------- | --- | --------------- | ---- | ----------- | -------- | --- | --- | --- | --- | --- | --- | --- |
| pable of        | capturing | the market’s  |     | microstructure, |      | the trading | results  |     |     |     |     |     |     |     |
conductedbyRLsolutionsoftenoutperformtraditionalVWAPtracking
approachessuchasBialkowskietal.(2008)andPodobniketal.(2009).
Tothebestofourknowledge,Nevmyvakaetal.(2006)isthefirstwork
toleverageRLframeworkssuchasQ-learning(Watkins&Dayan,1992)
| to optimal              | trade execution |     | problems.  | It is | important      | to note | that the  |     |     |     |     |     |     |     |
| ----------------------- | --------------- | --- | ---------- | ----- | -------------- | ------- | --------- | --- | --- | --- | --- | --- | --- | --- |
| curse of dimensionality |                 | in  | Q-learning | makes | it challenging |         | to handle |     |     |     |     |     |     |     |
high-dimensionaldata.While(Hendricks&Wilcox,2014)attemptsto
combineRLwiththeAlmgren–Chrissmodel,itisachallengingtaskas
themodeldependsoncertainassumptionsaboutthemarketdynamics.
ThankstotheadvancementsindeepRL,recentstudiessuchasMacri
| andLillo(2024), | Ningetal.(2021),andLinandBeling(2019)have |     |     |     |     |     |     |     |     |     |     |     |     |     |
| --------------- | ----------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
utilizedDeepQ-Networks(DQNs)(Mnihetal.,2013)foroptimaltrade
execution,addressingthechallengesofhigh-dimensionaldataandthe
| complexity   | of the   | financial | market  | without | relying | on     | any market  |     |     |     |     |     |     |     |
| ------------ | -------- | --------- | ------- | ------- | ------- | ------ | ----------- | --- | --- | --- | --- | --- | --- | --- |
| assumptions. | However, | these     | methods | require | the     | design | of specific |     |     |     |     |     |     |     |
attributes,whichcanbelabor-intensive.Recently,severalstudieshave
Fig.1. Tradevolumeratioovereach20-minuteperiodthroughouttheday.Thelines
investigatedtheuseofproximalpolicyoptimization(PPO)(Schulman represent the 1-year averages of the ratios and the shaded regions are drawn from
et al., 2017) based on optimal execution frameworks, which do not dailydeviationsfromtheaverages.
| require manually | designed |        | feature | engineering. |             | For instance, | Lin and     |     |     |     |     |     |     |     |
| ---------------- | -------- | ------ | ------- | ------------ | ----------- | ------------- | ----------- | --- | --- | --- | --- | --- | --- | --- |
| Beling (2020),   | Fang     | et al. | (2021), | Pan et       | al. (2022), | and           | Byun et al. |     |     |     |     |     |     |     |
(2023) have explored this approach in different scenarios. Specifi- will be executed in each interval. We propose two methods for the
cally,Panetal.(2022)focusedonoptimalexecutionwithlimitorders, firststage,thestatisticalU-shapemethodandtheU-shapeTransformer
while Lin and Beling (2020) and Fang et al. (2021) used market method.ThestatisticalU-shapemethodallocatesthevolumebasedon
orders.Amethodcombiningbothlimitandmarketorderswasproposed the historical average trade volumes. However, this approach cannot
byByunetal.(2023). fully capture the day-to-day variations (See Fig. 1). Alternatively, we
Prior literature often considers a relatively short trading horizon propose the U-shape Transformer for figuring out these variations.
ortheentiredayforRLimplementation.However,VWAPapproxima- In the next stage, we apply LSTM, trained by the RL framework to
tion becomes challenging when a short time frame is used since the distribute orders efficiently within each interval. It is important to
variations of financial data are often insignificant within short time note that there are numerous RL training algorithms, including Trust
intervals.Moreover,thenaiveapproachofconsideringtheentireday Region Policy Optimization (TRPO) (Schulman et al., 2015), Deep Q-
canputastrainonthenetworkarchitectureduetothelongsequence Networks(DQN)(Mnihetal.,2013),andProximalPolicyOptimization
|              |       |         |         |            |     |          |              | (PPO) (Schulman | et al., | 2017). | Among | these options, | we specifically |     |
| ------------ | ----- | ------- | ------- | ---------- | --- | -------- | ------------ | --------------- | ------- | ------ | ----- | -------------- | --------------- | --- |
| length. This | paper | aims to | develop | a strategy |     | that can | consistently |                 |         |        |       |                |                 |     |
choosePPOasourtrainingalgorithm.Foracomprehensiveunderstand-
trackthedailycumulativeVWAP.Toachievethisgoal,westartwith
ingofthesetechniquesandtheirapplications,werefertoSuttonand
| the observation | of  | an important | stock | trading |     | characteristic, | the U- |     |     |     |     |     |     |     |
| --------------- | --- | ------------ | ----- | ------- | --- | --------------- | ------ | --- | --- | --- | --- | --- | --- | --- |
Barto(2018).
| shaped intraday | trading | pattern, | documented |         | in           | Jain and | Joh (1988)    |           |           |            |          |          |              |     |
| --------------- | ------- | -------- | ---------- | ------- | ------------ | -------- | ------------- | --------- | --------- | ---------- | -------- | -------- | ------------ | --- |
|                 |         |          |            |         |              |          |               | We expect | that this | dual-level | approach | improves | the accuracy | of  |
| and Goodhart    | and     | O’Hara   | (1997).    | This is | a well-known |          | stylized fact |           |           |            |          |          |              |     |
approximatingthecumulativeVWAPbyproperlydistributingthetotal
statingthatastock’stradingvolumeoftenfollowsaU-shapedpattern
order.Ourcontributionsareasfollows:
throughouttheday.Thismeansthatvolumeishighestattheopening,
| falls rapidly  | to lower     | levels,       | and     | then rises | again      | towards    | the close   |               |              |             |               |               |                  |      |
| -------------- | ------------ | ------------- | ------- | ---------- | ---------- | ---------- | ----------- | ------------- | ------------ | ----------- | ------------- | ------------- | ---------------- | ---- |
|                |              |               |         |            |            |            |             | 1. We propose | a novel      | approach    | for           | optimal       | trade execution  | that |
| of the market, | but          | is relatively | low     | in the     | afternoon. | (See       | Fig. 1). In |               |              |             |               |               |                  |      |
|                |              |               |         |            |            |            |             | utilizes      | a dual-level | strategy.   | Orders        | are allocated | through          | two  |
| view of this   | observation, | we            | propose | a novel    | dual-level |            | approach to |               |              |             |               |               |                  |      |
|                |              |               |         |            |            |            |             | stages:       | in the       | first stage | we utilize    | the           | U-shaped pattern | of   |
| minimize       | the market   | impact        | and     | track the  | daily      | cumulative | VWAP        |               |              |             |               |               |                  |      |
|                |              |               |         |            |            |            |             | intraday      | volumes      | and         | in the second | stage         | we implement     | deep |
accuratelyandconsistently.Asthenamesuggests,ourmodelconsists reinforcementlearning.Sincethisnoveldualapproachsegments
of two stages. In the first stage, the model decides how much trade the data, we not only make reinforcement learning more com-
volume will be executed in each interval, using the U-shape property putationallytractablebutalsoimprovepredictionaccuracycom-
asaguideline.Inthesecondstage,theRLmodeldecideshowtheorders paredtoapplyingRLdirectlyonfullintradaysequences.
2

S.Kimetal. ExpertSystemsWithApplications252(2024)124263
Table1 objectiveistofindastrategythatmaximizesorminimizestheaverage
The table summarizes previous works on optimal trade execution utilizing the RL executionprice𝑃̄,whichiscalculatedas
framework. Note that DQN and PPO are learning algorithms of deep RL. Notably,
ourworkdistinguishesitselfbyfocusingonalong-termhorizon. 𝑃̄ = 𝑇 ∑ −1𝑂 𝑡𝑝.
RLmethod Horizon 𝑂 𝑡
𝑡=0
Nevmyvakaetal.(2006) Q-learning Short
HendricksandWilcox(2014) Q-learning Short While many financial firms pursue profits through trading, there are
LinandBeling(2019) DQN Short also numerous firms within the financial industry that prioritize con-
Ningetal.(2021) DQN Short sistentlytrackingtheVWAPovermaximizingtradingprofit.Thefocus
LinandBeling(2020) PPO Short
ofourpaperistailoredforthissecondgroupoftraders.TheVWAPis
Fangetal.(2021) PPO Short
calculatedas
Panetal.(2022) PPO Short
Thiswork PPO Long ∑𝑇−1𝑝𝑞
VWAP= 𝑡=0 𝑡 𝑡 ,
∑𝑇−1𝑞
𝑡=0 𝑡
where 𝑞 is the execution volume determined by the market. There-
2. We further propose the U-shape Transformer model to capture 𝑡
fore,theprimaryobjectiveofthispaperistominimizethedifference
theday-to-dayoscillationsoftheintradayU-shapedistribution. betweentheVWAPand𝑃̄,usingadual-levelapproach.
WhiletheU-shapedpatterniswell-known,relyingonfixedsta-
tisticalestimatesfailstoadapttosuddenfluctuationsorshiftsin
3. Methoddescription
marketconditions.TheU-shapeTransformerdirectlylearnsaro-
bustrepresentationfromrecentdata,leveragingself-attentionto
Inthissection,wedescribeourdual-levelapproachandformulation
automaticallyadapttosuddendynamicsintheintradayU-shape
oftheoptimaltradeexecutionproblemunderthelensofreinforcement
pattern.
learning.WealsoprovidedetailsonthearchitectureoftheTransformer
3. Weconductablationstudiestoshowthestrengthsofourmeth-
and LSTM neural networks and how they are integrated into our
odsinapproximatingthedailycumulativeVWAP.Furthermore,
proposedmethod.
we demonstrate that the methods employed to distribute vol-
umes from long trading horizons into short horizons can have 3.1. Dual-levelapproach
asignificantimpact.Inaddition,weintroducetheVAA,which
provides a more practical way to evaluate how well a trading Our approach involves two stages of distributing orders. In the
strategy tracks the daily VWAP. Our proposed dual-level ap- first stage, we allocate the total daily orders 𝑂 into 𝐿 intervals. In
proachcombiningU-shapeTransformersandRLachievesstate- thesecondstage,wefurtherdividetheordersfromeachintervalinto
of-the-artperformanceontheVAA(seeTable1). smaller executable orders using RL. Our approach begins by dividing
the market hours of a day into 𝐿 intervals. For each interval 𝑙, we
let 𝑂𝑙 as the total order to be executed, ensuring that the sum of all
2. Backgrounds
intervalordersequalsthetotaldailyorder,or
∑𝐿−1𝑂𝑙=𝑂.Wesuggest
𝑙=0
two methods for implementing the first stage. The naive approach
Inthissection,wediscussthebasicsofthelimitorderbook(LOB)
is to adhere to the statistical U-shape trade volume distribution for
and the optimal trade execution process. Most modern financial ex-
each interval obtained from past data (e.g. the historical average of
changes, including the Korea Stock Exchange (KRX), offer electronic
the past year). A more advanced method would be to use the output
tradingplatformswithaccesstolimitorderbooks.Tradersoftenrelyon
(predicted U-shape distribution) generated by a U-shape Transformer
theinformationprovidedbythelimitorderbooktodevelopsuccessful
model. In the latter case, 𝑂𝑙 is calculated progressively by referring
tradingstrategies.
only to market information from past days and previous intervals
within the day. For such an 𝑙th interval, which is comprised of 𝑇
2.1. Limitorderbook timesteps,theLSTMmodeloutputs𝑂𝑙,whichisthenumberoforders
𝑡
to be executed at timestep 𝑡 such that ∑𝑇−1𝑂𝑙 = 𝑂𝑙. Once 𝑂𝑙 is
Infinancialmarkets,alimitorderisanordertobuyorsellafixed 𝑡=0 𝑡 𝑡
decided, it is averaged over 𝐼 execution steps in the corresponding
number of shares at a specified price. This order becomes part of the
𝑡th subinterval. In other words, for each execution step, which is
limitorderbook,whichrecordsalllimitordersforaspecificstock.The set as every 5 s, the final number of orders executed is 𝑂𝑙∕𝐼. The
𝑡
bidpriceisthehighestpriceabuyeriswillingtopayforthesecurity,
choice of averaging is supported by our findings in Section 4.2 that
andtheaskpriceisthelowestpriceinwhichtheselleriswillingtosell.
there is no significant impact in approximating the VWAP since the
Amarketorder,bycontrast,isanordertobuyorsellafixednumber finalintervalistoosmall.Furthermore,inSection4.2,weverifythat
ofsharesatthecurrentmarketpricewithoutstatingtheprice.Whena incorporatinglong-termhorizonsyieldssuperiorefficiency.Therefore,
marketorderisplaced,itismatchedbythebestavailablelimitorders althoughadeepreinforcementlearningframeworkcanoffermarginal
intheLOB.TheLOBisimportantforoptimaltradeexecutionandhelps performanceimprovements,wehaveoptedforthesimplicityofusing
traderstracksupplyanddemandforastock,makingiteasiertoidentify theuniformdistributionforthefinalinterval.Aschematicillustration
thebesttimeandpricetobuyorsellshares. ofthisdesignisprovidedinFig.2.
2.2. Optimaltradeexecution 3.2. MDPformulationforoptimalexecution
Optimal trade execution refers to the process of buying or selling In this section, we explain our Markov Decision Process (MDP)
afinancialassetwhileachievingthedesiredobjective.Generally,this formulationofoptimaltradeexecution.AMDPistypicallyrepresented
problem formulates as follows: Within a timeframe of 𝑇 timesteps, using the tuple (, , , 𝑟, 𝛾), where  is the state space,  is the
0,1,…,𝑇 −1,atraderwhopossessesaninventoryof𝑂sharesmustbuy action space,  is the transition probability, 𝑟 is the reward function,
orselltheentiretyoftheinventory.Foreachtimestep𝑡∈{0,1,…,𝑇− and𝛾isthediscountfactor.Ourconfigurationsofthestatespaceand
1},thetraderdeterminesthenumberofsharestoorder𝑂 basedonthe actionspacearesimilartothatofLinandBeling(2020).Itisworth
𝑡
informationfromLOB,andcarriesoutthetradeatanexecutionprice notingthattheMDPimplementationinthispaperisappliedonaper
𝑝. For the group of traders seeking to maximize trading profit, their 𝑙thintervalbasisforagivenday.
𝑡
3

S.Kimetal.
ExpertSystemsWithApplications252(2024)124263
Fig.2. Aschematicillustrationofourdual-levelapproach.
3.2.1. State Ourrewardfunctioncompares𝑂𝑙∗ and𝑂𝑙 directlyforagiven𝑎𝑙 as
|                                                           |     |     |     |     |     |          |     |     | 𝑡   | 𝑡   | 𝑡   |
| --------------------------------------------------------- | --- | --- | --- | --- | --- | -------- | --- | --- | --- | --- | --- |
| Thestate𝑠𝑙∈consistsofbothpublicandprivateinformation.The |     |     |     |     |     | follows: |     |     |     |     |     |
𝑡
publicstateismadeupofthetop5bidandaskprices,alongwiththeir ⎧ 𝑀𝑙<0.01
|     |     |     |     |     |     |     | 1 𝑡 |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
associated volumes, while the private state includes the elapsed time 𝑟(𝑎𝑙)∶= ⎪ 1≤𝑀𝑙<0.05
|                                                         |           |        |       |           |                  | 𝑡         | ⎨ 0 0.0        | 𝑡   |     |     | (2) |
| ------------------------------------------------------- | --------- | ------ | ----- | --------- | ---------------- | --------- | -------------- | --- | --- | --- | --- |
| and the current                                         | remaining | volume | to be | executed. | The elapsed time |           |                |     |     |     |     |
|                                                         |           |        |       |           |                  |           | ⎪ −1 otherwise |     |     |     |     |
| rangesfrom0to𝑇−1andthecurrentremainingvolumeattimestep𝑡 |           |        |       |           |                  |           | ⎩              |     |     |     |     |
| isequalto𝑂𝑙−                                            | ∑𝑡 −      | 1 𝑂 𝑙. |       |           |                  |           | 𝑙−𝑂 𝑙∗         |     |     |     |     |
|                                                         | 𝑗         | = 0 𝑗  |       |           |                  | where𝑀𝑙∶= | |𝑂 𝑡 𝑡         | |.  |     |     |     |
|                                                         |           |        |       |           |                  |           | 𝑡 𝑂𝑙∗          |     |     |     |     |
𝑡
OnceMDPisdetermined,thegoalofpolicy-basedRListofindan
| 3.2.2. Action                    |     |     |     |                        |     | optimalpolicyparameter𝜃∗,i.e. |     |     |     |     |     |
| -------------------------------- | --- | --- | --- | ---------------------- | --- | ----------------------------- | --- | --- | --- | --- | --- |
| OurLSTMmodeloutputsapolicy𝜋(⋅|𝑠𝑙 |     |     |     | )∈R21,whichisadiscrete |     |                               |     |     |     |     |     |
|                                  |     |     |     | 𝑡                      |     | 𝜃∗=                           |     |     |     |     |     |
𝑙 
c a t e g o r i c a l p r o b ab ili t y d i s t r i b u t io n . F r o m t h is , a n a c t i o n 𝑎 ∈ ∶ = [ ]
|     |     |     |     |     | 𝑡   |     | ∑ 𝑇 |     |     |     | (3) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
{ 0 , 0 . 1 , 0 . 2 , … , 2 } i s s a m p l e d t o d e t e r m i n e t h e n u m b e r o f o r d e r s t o argmaxE 𝛾𝑡𝑟 | 𝑎 ∼𝜋 (⋅|𝑠 ),𝑠 ∼(⋅|𝑠 ,𝑎 ) ,
|     |     | 𝑙   |     |     |     |     | 𝑡 | | | 𝑡 𝜃 | 𝑡 𝑡+1 | 𝑡 𝑡 |     |
| --- | --- | --- | --- | --- | --- | --- | ----- | --- | ----- | --- | --- |
e x e c u t e , w it h 𝑂 𝑙 = 𝑎 𝑙 𝑂 . T o e n s u r e t h a t t h e to t a l n u m b e r o f o r de r s |
|                                               | 𝑡   | 𝑡 𝑇 |     |     |                 |        | 𝑡= 0    |     |     |     |     |
| --------------------------------------------- | --- | --- | --- | --- | --------------- | ------ | ------- | --- | --- | --- | --- |
| executedovertheentireperiodisequalto𝑂𝑙,thatis |     |     |     |     | ∑𝑇 − 1𝑂 𝑙=𝑂𝑙,we |        |         |     |     |     |     |
|                                               |     |     |     |     | 𝑡= 0 𝑡          | where𝑟 | =𝑟(𝑎 ). |     |     |     |     |
|                                               |     |     |     |     |                 |        | 𝑡 𝑡     |     |     |     |     |
imposethefollowingtworestrictions.
3.3. Neuralnetworkarchitectureandtraining
∑𝑗
| • If | 𝑂𝑙>𝑂𝑙 | forsome𝑗∈{0,1,2,…,𝑇 |     | −1},then |     |     |     |     |     |     |     |
| ---- | ----- | ------------------- | --- | -------- | --- | --- | --- | --- | --- | --- | --- |
| 𝑡=0  | 𝑡     |                     |     |          |     |     |     |     |     |     |     |
{ 𝑂𝑙− ∑𝑗−1𝑂𝑙, We now turn to the architectural details of the Transformer and
|     |     | 𝑡   | if𝑖=𝑗 |     |     |     |     |     |     |     |     |
| --- | --- | --- | ----- | --- | --- | --- | --- | --- | --- | --- | --- |
𝑂𝑙= 𝑡=0 LSTM models and elaborate on how they are used in our dual-level
| 𝑖   |     | 0, if𝑗<𝑖≤𝑇 | −1  |     |     |     |     |     |     |     |     |
| --- | --- | ---------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
approach.
| • 𝑂𝑙 | isequaltotheremainingvolumetobeexecutedatthelast |     |     |     |     |     |     |     |     |     |     |
| ---- | ------------------------------------------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
𝑇− 1
tim e stepforeachinterval(i.e.0≤𝑙≤𝐿−1) 3.3.1. Level1:TransformerforglobalU-shapeapproximation
TheobjectiveoftheTransformermodelistoprogressivelypredict
theU-shaperatioforeachintervalinagivenday.Weusethestructure
Theserestrictionsareinlinewiththefactthatitisessentialforbrokers
oftheTransformerEncoderandDecoderasproposedinVaswanietal.
topromptlyacquireorliquidateallsharesrequestedbytheircustomers.
(2017).TheoverallarchitectureoftheTransformermodelthatweuse
isdepictedinFig.3.
3.2.3. Reward U-shapeEncoder.TheU-shapeEncoder iscomposedofaTrans-
|            |          |             |     |           |                 | former | Encoder and | 𝐿 linear | layers, all | of which map | to the same |
| ---------- | -------- | ----------- | --- | --------- | --------------- | ------ | ----------- | -------- | ----------- | ------------ | ----------- |
| The reward | function | is designed | to  | encourage | the agent (LSTM |        |             |          |             |              |             |
model) to generate actions that result in orders that are close to the dimension.Thismoduleisresponsibleforlearningthegeneraldynam-
target volume 𝑂𝑙∗, in order to track the VWAP. This is achieved by ics of the U-shape distribution. The input 𝐸 𝑖𝑛 is a sequence of length
𝑡
usingthepricecalculatedfromtheexecutedorders.Here,𝑂𝑙∗ denotes 𝐿 of vectors each containing 𝑁 historical daily volume ratios of the
𝑡
|             |      |                   |      |         |                    | corresponding | interval | in the | sequence. | The days from | which these |
| ----------- | ---- | ----------------- | ---- | ------- | ------------------ | ------------- | -------- | ------ | --------- | ------------- | ----------- |
| the desired | VWAP | order at timestep | 𝑡 in | the 𝑙th | interval such that |               |          |        |           |               |             |
∑𝑇−1𝑂𝑙∗.ThedailyVWAPcanbeobtainedby ratios are drawn are randomly selected. Each vector in the sequence
𝑂𝑙=
𝑡=0 𝑡 is processed by a different linear layer that serves as an embedding
𝐿 −1𝑇 −1𝑂 𝑙∗ t o le a rn t h e u ni q u e f ea tu re s o f e a c h in t e rv a l. A f ter p a s s in g th r ou g h
∑ ∑ 𝑡 𝑝𝑙
𝑉𝑊𝐴𝑃 𝑑𝑎𝑦 = , (1) t he e m be d d in g l a y er , th e i np u t s e q ue n c e i s th e n pr o c e s se d b y t h e
𝑂 𝑡
|     | 𝑙=0 𝑡=0 |     |     |     |     | TransformerEncoder,resultingintheoutput𝐸 |     |     |     |     |     |
| --- | ------- | --- | --- | --- | --- | ---------------------------------------- | --- | --- | --- | --- | --- |
𝑜𝑢𝑡 .
| 𝑝𝑙  |     |     |     |     |     |     |     |     |    |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
where 𝑡 is the associated average market traded price of the corre- U-shape Decoder. While learns the general dynamics of the U-

sponding𝑡thsubinterval. shape distribution, the U-shape Decoder focuses on handling the
4

S.Kimetal. ExpertSystemsWithApplications252(2024)124263
3.3.3. Trainingprocedure
U-shape Transformer. Since the goal of the first level of order al-
locationistoapproximatetheU-shapedistributionwhilealsotracking
thedailycumulativeVWAP,thetrainingobjectiveoftheTransformer
modelistominimize
𝐽 ∶=E [ 𝑐 1 𝐿 ∑ −1( 𝑢𝑙 −𝑢𝑙 )2 +𝑐 𝑉𝐴𝐴 ] (6)
𝑇𝐹 𝑑𝑎𝑦𝑠 𝐿 𝑡𝑟𝑢𝑒 𝑝𝑟𝑒𝑑 2
𝑙=0
where𝑐 and𝑐 arecoefficientsand
1 2
VAA∶= | | | | | 𝑀𝑃 𝑑 𝑉 𝑎𝑦 𝑊 − 𝐴 𝑉 𝑃 𝑊 𝑑𝑎 𝐴 𝑦 𝑃 𝑑𝑎𝑦| | | | | (7)
denotestheVWAPApproximationAccuracy(VAA),whichwewillfur-
theruseasageneralmetricforperformanceevaluationinexperiments,
and
𝑀𝑃 ∶=
𝐿
∑
−1𝑇
∑
−1𝑂
𝑡
𝑙
𝑝𝑙
Fig. 3. The architecture of the U-shape Transformer. The U-shape Decoder in this 𝑑𝑎𝑦 𝑂 𝑡
𝑙=0 𝑡=0
figurerepresentsthe𝑙thdecodingstep.
is the price yielded by our model. This is one of the measures used
in practice to gauge how close a trader has traded closer to their
benchmark,theVWAP.Ifatraderhastradedatapricethatisexactly
day-to-dayvariationsofthedistribution.TheU-shapeDecoderiscom-
equal to the VWAP, then 𝑉𝐴𝐴 = 0. Thus lower VAA would be
posedofaTransformerDecoderand𝐿linearlayers,eachmappingto
better.Thefirstterminsidetheexpectationof(6)learnstomakeratio
differentdimensions.Incontrastto,theselinearlayersareappliedto
predictions closer to ground-truth values. The second term penalizes
theoutputoftheTransformerDecoder. theTransformermodelifthefinaldailyacquisitionpricegeneratedby
For every 𝑙th (𝑙 > 0) decoding step, both 𝐸 𝑜𝑢𝑡 and the cumulative thecombinationoftheTransformerandLSTMmodelsdeviatesfromthe
input sequence are fed as input to the Transformer Decoder, . We dailyVWAP.TheprimaryobjectiveoftheU-shapedtransformerinthe
denote the cumulative input sequence as 𝐷 𝑖 𝑙 𝑛 = (𝐷 𝑖 𝑙 𝑛,𝑗 ) 0≤𝑗≤𝑙−1 , where firstlevelistodistributethedailytotalorderwithprecision.However,
𝐷𝑙 ∶= CONCAT(𝐷𝑗 ,ℎ𝑗 ). Here, 𝐷𝑗 represents the output of  relyingsolelyonthefirsttermofthelossfunctionmaynotbeenough
at 𝑖𝑛 t , h 𝑗 e𝑗thstep,andℎ 𝑜𝑢 𝑗 𝑡 𝑇 is −1 thelastLST 𝑜𝑢 M 𝑡 hiddenvectorfromthe𝑗th to accomplish this objective. This is because 𝑢𝑙 𝑡𝑟𝑢𝑒 does not consider
𝑇−1 the local information, such as the LOB, which can pose difficulties
interval.TheoutputfromtheTransformerDecoderatthe𝑙thdecoding
in tracking the daily VWAP accurately. Furthermore, if only the first
step is a vector of dimension 𝐿+𝐻, which is then passed through a
termin(6)istakenintoaccount,thereinforcementlearningframework
linearlayerthatreducesitsdimensionto𝐿−𝑙.Thesoftmaxfunctionis
wouldnotinfluencethefirstlevel,leadingtoindependentoperationof
appliedtopredicttheratiosoftheremaining𝐿−𝑙intervals,andthefirst
thefirstandsecondlevels.Toensurethatthemicroinformationhasan
𝑙 components are zero-padded. Note that we use ℎ𝑗 𝑇−1 to capture the appropriate impact on the first level, we incorporate the second term
day-to-day variations in the U-shape distribution, as it holds interval- intothelossfunction.
specificinformationforthatparticularday.Toensure ∑𝐿 𝑙= − 0 1𝑢𝑙 𝑝𝑟𝑒𝑑 =1, LSTM.Tofindtheoptimalparameter𝜃fortheLSTMmodel,weuse
thefinal𝑙thintervalvolumeratiopredictioniscalculatedas an actor–critic style PPO algorithm (Schulman et al., 2017), which is
( 𝑙−1 ) oneofthemostpopularon-policyRLalgorithms,asourbaselearner.
𝑢𝑙 = 1− ∑ 𝑢𝑗 ⋅𝐷𝑙 [𝑙]. (4) One of the major advantages of the PPO in this task is its ability to
𝑝𝑟𝑒𝑑 𝑝𝑟𝑒𝑑 𝑜𝑢𝑡
𝑗=0 adapt to changing market conditions and effectively learn from noisy
andhigh-dimensionaldata.Thisalgorithmusestheactorloss𝐽𝐶𝐿𝐼𝑃(𝜃)
It is worth noting that 𝐷 𝑖 0 𝑛 is constructed by applying an additional andthecriticlossfunction𝐽𝑉𝐹(𝜃),whicharedefinedasfollow PP s O :
PPO
linear layer that transforms the dimension of the vector holding the
t w o i p th 5 t b h i e d g / i a v s e k n v d o a lu y m to e ta a l v o er r a d g e e r s d o e f no p t r e e d -m a a s r 𝑂 ke , t th d e at c a or to re 𝐿 sp + on 𝐻 d . in T g h i e n r t e e f r o v r a e l , 𝐽 ∶ P 𝐶 = P 𝐿 O E 𝐼𝑃( [ 𝜃 m ) in ( 𝑞(𝜃)𝐴̂,clip(𝑞(𝜃),1−𝜀,1+𝜀)𝐴̂)] , (8)
𝑡 𝑡 𝑡 𝑡 𝑡
order𝑂𝑙 canbedeterminedastheproductof𝑂and𝑢𝑙 ,i.e.
𝑝𝑟𝑒𝑑 and
[ ]
𝑂𝑙=𝑂×𝑢𝑙 𝑝𝑟𝑒𝑑 . (5) 𝐽 P 𝑉 P 𝐹 O (𝜃)=E 𝑡 (𝑉 𝑡 targ−𝑉 𝜃 (𝑠 𝑡 ))2 , (9)
where𝑞(𝜃)= 𝜋𝜃(𝑎𝑡|𝑠𝑡) ,
𝑡 𝜋𝜃old (𝑎𝑡|𝑠𝑡)
3.3.2. Level2:LSTMforlocalorderdistribution 𝑇−1
The objective of the LSTM model is to optimally allocate orders 𝑉 𝑡 targ= ∑ 𝛾𝑘−𝑡𝑟 𝑘 and𝐴̂ 𝑡 =𝑉 𝑡 targ−𝑉 𝜃 (𝑠 𝑡 ). (10)
withintheintervalsofagivenday.ThestructureofourLSTMmodel, 𝑘=𝑡
whichisoutlinedinFig.4,issimilartothatofLinandBeling(2020). PPO’sgoalistofindaparameter𝜃 thatmaximizesthemainobjective
Given𝑠𝑙 𝑡 asinput,itfirstpassesthroughtwolinearlayerswithhidden function𝐽 PPO (𝜃),whichisdefinedby
d T i h m e e r n e s s i u o l n tin 1 g 28 v a e n ct d or is is th t e h n en co p n r c o a c t e e s n s a e t d ed by wi a t n h 𝑎 L 𝑙 𝑡 S − T 1 M an C d e 𝑟 ll 𝑙 𝑡− w 1 i ∶ t = h h 𝑟( i 𝑎 d 𝑙 𝑡 d − e 1 ) n . 𝐽 PPO (𝜃)=𝐽 P 𝐶 P 𝐿 O 𝐼𝑃(𝜃)−𝑐 3 𝐽 P 𝑉 P 𝐹 O (𝜃)+𝑐 4 E 𝑡 [𝑆[𝜋 𝜃 ](𝑠 𝑡 )], (11)
dimension 𝐻 to obtain ℎ𝑙, which is then passed separately through where𝑐 3 ,𝑐 4 arecoefficients,andE 𝑡 [𝑆[𝜋 𝜃 ](𝑠 𝑡 )]denotesanentropyloss
𝑡
two linear layers to produce the policy 𝜋(⋅|𝑠𝑙
𝑡
) and the value 𝑉(𝑠𝑙
𝑡
) term,where𝑆[𝜋 𝜃 ](𝑠 𝑡 )isdefinedby
respectively.Wenotethatthevectorℎ𝑙 𝑇−1 whichisobtainedatthelast 𝑆[𝜋 𝜃 ](𝑠 𝑡 )∶=− ∑ 𝜋 𝜃 (𝑎|𝑠 𝑡 )log𝜋 𝜃 (𝑎|𝑠 𝑡 ). (12)
timestep is used as input to the Transformer Decoder. The action 𝑎𝑙
𝑡
, 𝑎∈
reward 𝑟𝑙 𝑡 , and the number of orders to execute 𝑂 𝑡 𝑙 is calculated as is By incorporating an entropy term, PPO can incentivize the agent to
describedinSection3.2.TheLSTMCellalsoreceivesℎ𝑙 and𝑐𝑙 as explorealternativeactions,preventgettingtrappedinsuboptimalpoli-
𝑡−1 𝑡−1
inputanditsoutputincludes𝑐𝑙 aswell. cies, and mitigate overfitting to the training data (Mnih et al., 2016;
𝑡
5

S.Kimetal.
ExpertSystemsWithApplications252(2024)124263
Fig.4. ThearchitectureoftheLSTMmodel.
Williams, 1992). To find the parameter 𝜃, PPO iteratively gathers Algorithm1Dual-levelNeuralNetwork
episodesandupdatestheparameter𝜃usingthefollowingscheme: Require: Number of 𝑜𝑢𝑡𝑒𝑟 𝑖𝑡𝑒𝑟𝑎𝑡𝑖𝑜𝑛𝑠 and 𝑛𝑢𝑚
𝑑𝑎𝑦𝑠
| 𝜃 =argmaxE | E   | [𝐽  | (𝜃)], |     |     |     | (13) |     |     |     |     |     |     |
| ---------- | --- | --- | ----- | --- | --- | --- | ---- | --- | --- | --- | --- | --- | --- |
𝑘+1 𝑠 𝑎∼𝜋𝜃𝑘 (⋅|𝑠) PPO Randomlyinitializelearnableparameters𝜃,𝜃,and𝜃
𝜃
where𝑘standsforthe𝑘thstep.The‘‘clip’’operatorin(8)allowsPPO Initializetrajectorybuffer
tolearnfrompreviousexperienceswithoutbecomingoverlydependent for𝑗=1to𝑜𝑢𝑡𝑒𝑟𝑖𝑡𝑒𝑟𝑎𝑡𝑖𝑜𝑛𝑠do
on them. Moreover, PPO can effectively retain knowledge from past for𝑑=1to𝑛𝑢𝑚𝑑𝑎𝑦𝑠do
experiences while also acquiring insights from new experiences. This Randomlychooseadateinthetrainingset
keyattributeofPPOisoneofitssignificantadvantages. 𝜇←theaverageofthedailyvolumeforthelast60days
𝜎 ←thestandarddeviationofthedailyvolumeforthelast60
4. Experiments
days
|     |     |     |     |     |     |     |     |  ← | ⋃   |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ---- | --- | --- | --- | --- | --- |
GatheringEpisodes(date,𝜇,𝜎)
| Our dual-level | approach |     | significantly | enhances | the | accuracy | and | endfor |     |     |     |     |     |
| -------------- | -------- | --- | ------------- | -------- | --- | -------- | --- | ------ | --- | --- | --- | --- | --- |
consistency of tracking the daily cumulative VWAP, despite the chal- Using , update ,  by optimizing the loss function 𝐽 in (5)
|           |           |                 |     |       |            |     |         | withrespectto𝜃 |     | and𝜃 |     |     | 𝑇𝐹  |
| --------- | --------- | --------------- | --- | ----- | ---------- | --- | ------- | --------------- | --- | ----- | --- | --- | --- |
| lenges in | improving | the performance |     | of RL | frameworks | for | optimal |                 |     |       |     |     |     |
trade execution in relatively short trading horizons. Furthermore, the Optimizethelossfunction𝐽 in(9)withrespectto𝜃
PPO
| U-shapeTransformerprovesefficientincapturingthedailyvariations |               |     |        |           |          |            |     |  ←∅   |     |     |     |     |     |
| -------------------------------------------------------------- | ------------- | --- | ------ | --------- | -------- | ---------- | --- | ------ | --- | --- | --- | --- | --- |
| of the U-shape                                                 | distribution. |     | In the | following | section, | we conduct | a   | endfor |     |     |     |     |     |
seriesofexperimentstoverifytheseassertions.
| 4.1. Experimentsettings |     |     |     |     |     |     |     | Algorithm2GatheringEpisodes |     |     |     |     |     |
| ----------------------- | --- | --- | --- | --- | --- | --- | --- | --------------------------- | --- | --- | --- | --- | --- |
Require:Thenumberofintervalsinaday𝐿
| 4.1.1. Implementation |     |     |     |     |     |     |     | Input:Date,𝜇,𝜎 |     |     |     |     |     |
| --------------------- | --- | --- | --- | --- | --- | --- | --- | -------------- | --- | --- | --- | --- | --- |
Output:Trajectorybufferof𝑖𝑡ℎday
Inallofourexperiments,thehyperparametersforthelossfunctions, 𝑖
| thelengthoftheintervals,thenumberoftotalorders,andthetrading |     |     |     |     |     |     |     | Initialize |     |     |     |     |     |
| ------------------------------------------------------------ | --- | --- | --- | --- | --- | --- | --- | ----------- | --- | --- | --- | --- | --- |
𝑖
horizonarefixedandsetaccordingtothefollowingspecifications: 𝑂∼(2.5×10−3𝜇,6.25×10−6𝜎2)
|                                  |     |     |           |                            |     |     |     | 𝐸 =(𝐸        | )   |                       |     |     |     |
| -------------------------------- | --- | --- | --------- | -------------------------- | --- | --- | --- | ------------- | --- | --------------------- | --- | --- | --- |
| - Thecoefficientsfor𝐽            |     |     | :𝑐 =0.5,𝑐 | =0.5                       |     |     |     | 𝑜𝑢𝑡           | 𝑖𝑛  |                       |     |     |     |
|                                  |     |     | TF 1      | 2                          |     |     |     | for𝑙=0to𝐿−1do |     |                       |     |     |     |
| - Thecoefficientsfor𝐽            |     |     | :𝑐 =1,𝑐   | =0.01                      |     |     |     |               |     |                       |     |     |     |
|                                  |     |     | PPO 3     | 4                          |     |     |     | if𝑙=0then     |     |                       |     |     |     |
| - Thelengthoftheintervals:𝐿=19,𝑇 |     |     |           | =20                        |     |     |     |               |     |                       |     |     |     |
|                                  |     |     |           |                            |     |     |     | Construct𝐷0   |     | fromrawpre-marketdata |     |     |     |
|                                  |     |     |           | ∼(2.5×10−3𝜇,6.25×10−6𝜎2), |     |     |     |               |     | 𝑖𝑛                    |     |     |     |
| - Thenumberoftotalorders:𝑂       |     |     |           |                            |     |     |     | else          |     |                       |     |     |     |
where𝜇and𝜎istheaverageandstandarddeviationoftheday 𝐷𝑙 ←𝐷𝑙−1⋃𝐶𝑜𝑛𝑐𝑎𝑡(𝐷𝑙−1,ℎ𝑙−1)
|     |     |     |     |     |     |     |     | 𝑖𝑛  | 𝑖𝑛  |     | 𝑜𝑢𝑡 𝑇−1 |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ------- | --- | --- |
totalvolumefortheprevioussixtydays,respectively.
endif
- Tradinghorizonofaday:380minutes(Fromwhenthemarket Obtain𝑢𝑙 and𝐷𝑙
byproceduredepictedinSection3.3.1
| opensat09:00:00towhenitclosesat15:20:00). |     |     |     |     |     |     |     |     | 𝑝𝑟𝑒𝑑 | 𝑜𝑢𝑡 |     |     |     |
| ----------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | ---- | --- | --- | --- | --- |
𝑂𝑙←𝑂⋅𝑢𝑙
𝑝𝑟𝑒𝑑
S in c e a t y pi c a l da y a l l o w s f o r 3 8 0 m i n o f tr ad in g, o u r m o d e l f ir st 𝑙 𝑙 𝑙 𝑙 𝑙 𝑙 𝑙} andℎ𝑙
|     |     |     |     |     |     |     |     | O b t a in | 𝜏 𝑙 = { 𝑠 | 𝑡 , 𝑎 𝑡 , 𝑟( 𝑎 𝑡 ), | 𝑉 ( 𝑠 𝑡 ), 𝜋 (𝑎 𝑡| 𝑠 𝑡 ),𝑂 𝑡 | 𝑡∈{0,1,2,⋯,𝑇−1} | by  |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------- | --------- | ------------------- | ---------------------------- | --------------- | --- |
take s t h e da i ly t o t al o rd e r , 𝑂 , a n d b r e ak s i td o w n to 𝑂 𝑙, 1 9 i nt e r v al s o f 𝑇−1
|                                                               |     |     |     |     |     |     |     | pr o c e du | r e d e p | i c t e d i n S | ec t i o n 3 .3 . 2 |     |     |
| ------------------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | ----------- | --------- | --------------- | ------------------- | --- | --- |
| 20mineach.(𝑙∈[0,18])WeeitherapplythestatisticalU-shapemethod  |     |     |     |     |     |     |     |  ←        | ⋃𝜏        |                 |                     |     |     |
|                                                               |     |     |     |     |     |     |     | 𝑖           | 𝑖 𝑙       |                 |                     |     |     |
| ortheTransformermodelforthisfirststageallocationprocess.Then, |     |     |     |     |     |     |     | endfor      |           |                 |                     |     |     |
withineachinterval,theLSTMmodelfurthersubdivides𝑂𝑙into20sets
of1-minutesubintervals,𝑂𝑙.(𝑡∈[0,19])
𝑡
| Ineachofthesesubintervals,𝑂𝑙 |     |     |     | isdividedintoequalportionsand |     |     |     |     |     |     |     |     |     |
| ---------------------------- | --- | --- | --- | ----------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
𝑡 stock,weaccountforthedifferencesintotalordersforvariousstocks,
| executed | over a span | of 12 | steps, assuming |     | that orders | are | executed |     |     |     |     |     |     |
| -------- | ----------- | ----- | --------------- | --- | ----------- | --- | -------- | --- | --- | --- | --- | --- | --- |
whichisoftenencounteredinreal-worldscenarios.Additionally,dur-
| every 5 s. | The method | and | hyperparameters |     | used to | train our | agents |     |     |     |     |     |     |
| ---------- | ---------- | --- | --------------- | --- | ------- | --------- | ------ | --- | --- | --- | --- | --- | --- |
ingtesting,weonlyusethedailygroundtruthU-shapevolumeratios
| for the experiment |     | are summarized |     | in Algorithms | 1,  | 2, and | Table 3 |             |              |      |                     |         |            |
| ------------------ | --- | -------------- | --- | ------------- | --- | ------ | ------- | ----------- | ------------ | ---- | ------------------- | ------- | ---------- |
|                    |     |                |     |               |     |        |         | of dates in | the training | data | for our Transformer | Encoder | to prevent |
whichsilocatedattheendofthepaper.
lookingaheadintothefuture.AsinpreviousworksofNevmyvakaetal.
| To conduct | realistic | simulations, |     | we determine | 𝑂   | in a | way that |     |     |     |     |     |     |
| ---------- | --------- | ------------ | --- | ------------ | --- | ---- | -------- | --- | --- | --- | --- | --- | --- |
(2006),HendricksandWilcox(2014),Ningetal.(2021),andLinand
| takes into | account | the volatility | of  | daily volume, | which | differs | from |     |     |     |     |     |     |
| ---------- | ------- | -------------- | --- | ------------- | ----- | ------- | ---- | --- | --- | --- | --- | --- | --- |
Beling(2020),wemakethefollowingassumptionsinourexperiments.
| most previous | works | where | it is fixed | to a certain | value. | This | choice |     |     |     |     |     |     |
| ------------- | ----- | ----- | ----------- | ------------ | ------ | ---- | ------ | --- | --- | --- | --- | --- | --- |
aims to simulate the fluctuations of total orders that financial firms 1. We assume that the actions taken by our model only affect
havetoexecuteinaday,takingintoconsiderationrecenttradevolume a temporary market, and that the market will recover to the
statistics for each stock. By considering the volume statistics for each equilibriumlevelatthenexttimestep.
6

S.Kimetal.
ExpertSystemsWithApplications252(2024)124263
Table2 theexperimentalsetupinNingetal.(2021).Thetotalorderfor
AveragevolumeforeachstockfromJanuary1st,2021toDecember31st,2021. thedayisthenequallydistributedamongsteachhour,andDQN
Stocks SE SH KC PH furtherallocatestheorderforfivesub-intervalsof12mineach.
Volume 17,000K 3800K 3900K 430K The orders are executed equally for each 5-second time step
withinthe12-minutesub-intervals.Toensurefaircomparisons,
weevaluatethedailypricegeneratedbyDQNagainstthedaily
cumulativeVWAPcalculatedbyexcludingthefinal20min.
- OPDisthemethodproposedinFangetal.(2021).Toimplement
|     |     |     |     |     |     |     |     | this | model, | we first | divide | a given day | into | ten intervals, | each |
| --- | --- | --- | --- | --- | --- | --- | --- | ---- | ------ | -------- | ------ | ----------- | ---- | -------------- | ---- |
spanning38min,andthenuseOPDtoallocateordersforeach
|     |     |     |     |     |     |     |     | interval. | Within | each | interval, | the | allocated | order | is equally |
| --- | --- | --- | --- | --- | --- | --- | --- | --------- | ------ | ---- | --------- | --- | --------- | ----- | ---------- |
distributedamongsteachminute.
- PPOisthemethodproposedinLinandBeling(2020).Toapply
|     |     |     |     |     |     |     |     | this   | model, | we equally | distribute | the     | day     | total order   | to each |
| --- | --- | --- | --- | --- | --- | --- | --- | ------ | ------ | ---------- | ---------- | ------- | ------- | ------------- | ------- |
|     |     |     |     |     |     |     |     |        |        |            | 𝑂          | = 𝑂∕380 |         | 𝑗 ∈ {1,…,380} |         |
|     |     |     |     |     |     |     |     | minute | in     | the day,   | i.e. 𝑗     |         | for all |               | as      |
inLinandBeling(2020).
𝑂𝑙,
|     |     |     |     |     |     |     |     | - (HU-)PPO |      | utilizes  | the statistical | U-shape   | to    | determine  | and     |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------- | ---- | --------- | --------------- | --------- | ----- | ---------- | ------- |
|     |     |     |     |     |     |     |     | within     | each | interval, | the             | allocated | order | is equally | divided |
amongeveryminute,i.e.𝑂𝑙=𝑂𝑙∕20forall𝑡∈{0,…,19},where
𝑡
Fig. 5. The VAA’s are depicted in dark blue for the best-case scenarios of uniform PPOisfinallyusedtodistributeorderswithintheminute.
distributionandU-shapedistributionorderallocationschemes,whiletheirworst-case - HULisourproposeddual-levelapproachwhichusesthestatis-
| counterparts | are shown | in brown. | The | uniform distribution |     | and U-shape | distribution |     |     |     |     |     |     |     |     |
| ------------ | --------- | --------- | --- | -------------------- | --- | ----------- | ------------ | --- | --- | --- | --- | --- | --- | --- | --- |
ticalU-shapetodetermine𝑂𝑙.
| allocate total | orders | to be | executed | per minute | in the same | manner | as PPO and |     |     |     |     |     |     |     |     |
| -------------- | ------ | ----- | -------- | ---------- | ----------- | ------ | ---------- | --- | --- | --- | --- | --- | --- | --- | --- |
- TULisourproposeddual-levelapproachthatemploystheTrans-
(HU-)PPO,respectively..
formermodeltodetermine𝑂𝑙.
- TURusesRNNstoallocateorderswithineachinterval(second
2. Commissionsandexchangefeesareignored. level) instead of LSTMs. Other configurations are the same as
| 3. Ourmodel’sordersaretradedimmediatelywithoutorderarrival |     |     |     |     |     |     |     | thatofTUL. |     |     |     |     |     |     |     |
| ---------------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | ---------- | --- | --- | --- | --- | --- | --- | --- |
delays.
Weevaluatetheperformanceofthesemethodsalongsideourproposed
Webelievethattheaboveassumptionsarereasonableasweconsider dual-levelapproachbasedontheVAAmetricin(7).Aswementioned
relatively small total orders in comparison to the daily total market inSection3.3.3,theVAAdirectlymeasurestheabilityofatradingstrat-
volumesofthestocks. egytocloselytrackandapproximatethedailyVWAP.Consequently,to
comprehensivelyassessabilityofeachmethodtoconsistentlytrackthe
4.1.2. Datasets daily VWAP over an extended period, we analyze both the mean and
Ourmillisecondtradeandlimitorderbook(LOB)tickdataisfrom variationoftheVAAmetric.
| the Korea | Exchange | (KRX). | We  | use the daily | trade | and LOB | data of |     |     |     |     |     |     |     |     |
| --------- | -------- | ------ | --- | ------------- | ----- | ------- | ------- | --- | --- | --- | --- | --- | --- | --- | --- |
Samsung Electronics (SE), SK Hynix (SH), Kia Corporation (KC) and 4.2. Experimentresults
| POSCO Holdings |        | Inc (PH) | from       | January 1st, | 2021  | to December | 31st,    |         |             |     |           |         |             |             |     |
| -------------- | ------ | -------- | ---------- | ------------ | ----- | ----------- | -------- | ------- | ----------- | --- | --------- | ------- | ----------- | ----------- | --- |
|                |        |          |            |              |       |             |          | We test | our models, |     | which are | trained | for 100,000 | iterations, | by  |
| 2021. Our      | choice | of the   | four firms | in the       | KOSPI | index is    | based on |         |             |     |           |         |             |             |     |
conductingtradingsimulationsinthebuyingdirection.Wefirstverify
| their liquidity | and | trade | volume | and variety | in  | market capitalization |     |     |     |     |     |     |     |     |     |
| --------------- | --- | ----- | ------ | ----------- | --- | --------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
andindustry.Inordertoevaluatetheeffectivenessofourapproachin the claim of the ineffectiveness of short trading horizons when order
relationtotradingvolume,wehaveoptedtoselectstockswitharange allocation does not consider the appropriate distribution method for
oftradevolumes.Specifically,weselectedstocksrepresentingdifferent volumesfromlongtradinghorizons.
tradevolumelevels—SEforlarge,SHandKCformedium,andPHfor Todothis,weexaminedtwodistinctscenarios:firstly,weuniformly
small volumes. The average daily trading volumes for each stock can distributed volumes from a long trading horizon into a short trading
be found in Table 2. We divide the data into two sets, using January horizon and secondly, we employed a statistical U-shape distribution
1st to September 30th as training data and October 1st to December to allocate volumes from a long trading horizon into a short trading
|     |     |     |     |     |     |     |     | horizon. Note | that | the | volumes | from a long | trading | horizon | mean |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------- | ---- | --- | ------- | ----------- | ------- | ------- | ---- |
31stastestdata.
dailytotalorders,whileashorttradinghorizonrepresentsa1-minute
| The millisecond |     | raw | data is | preprocessed | by  | first dividing | them |     |     |     |     |     |     |     |     |
| --------------- | --- | --- | ------- | ------------ | --- | -------------- | ---- | --- | --- | --- | --- | --- | --- | --- | --- |
tradinghorizon.Tocomparethebestandworstcases,wedeliberately
| into groups | of 5    | s, and   | extracting | data that   | represents | each     | 5-second |          |            |     |            |         |             |               |     |
| ----------- | ------- | -------- | ---------- | ----------- | ---------- | -------- | -------- | -------- | ---------- | --- | ---------- | ------- | ----------- | ------------- | --- |
|             |         |          |            |             |            |          |          | executed | trades for | the | lowest and | highest | buy orders, | respectively, |     |
| interval.   | We take | the last | LOB        | data of the | 5-second   | interval | to con-  |          |            |     |            |         |             |               |     |
structtheMDPstateandthe5-secondVWAPandtotaltradedvolume within the 1-minute trading horizon. Fig. 5 displays the comparison
for calculating the daily VWAP. The statistical U-shape statistics are oftheVAA’sonthetestdates.Wecanobservethattheworstcasefor
constructed by taking the year average of the U-shape ratios of the U-shapedistributionoutperformsthebestcaseforuniformdistribution.
20-minute intervals in the day. The data utilized in generating 𝐷0 Specifically,theVAA’smeanoftheworstcaseforU-shapedistribution
𝑖𝑛
showninSection3.3arethetop5bid/askvolumeaveragesoftheraw is10bpswhiletheVAA’smeanofthebestcaseforuniformdistribution
pre-marketdata(from08:30:00to09:00:00). is 11 bps. This result suggests that in the context of VWAP, the use
|     |     |     |     |     |     |     |     | of deep reinforcement |            |     | learning | does not    | significantly | improve | the       |
| --- | --- | --- | --- | --- | --- | --- | --- | --------------------- | ---------- | --- | -------- | ----------- | ------------- | ------- | --------- |
|     |     |     |     |     |     |     |     | approximation         | capability |     | without  | taking into | account       | the     | long-term |
4.1.3. Comparisonandablationstudy
|     |     |     |     |     |     |     |     | trading horizon. |     | This implies | that | in order | to effectively | leverage | the |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------------- | --- | ------------ | ---- | -------- | -------------- | -------- | --- |
Weprovideabriefexplanationofallmethodsusedforperformance
benefitsofdeepRLinthecontextofVWAPs,itshouldbeimplemented
comparisonduringourexperimentshere.Thehyperparametersforeach
onalongertradinghorizonsuchasasingletradingday
methodcanbefoundinTable3.
|     |     |     |     |     |     |     |     | Secondly, | we  | present | Table 4 | and Fig. | 6 to demonstrate |     | how our |
| --- | --- | --- | --- | --- | --- | --- | --- | --------- | --- | ------- | ------- | -------- | ---------------- | --- | ------- |
- DQN is the method proposed in Ning et al. (2021). To apply dual-levelapproachsignificantlyimprovesperformance.
this model, we begin by dividing a given day into six 1-hour Table 4 reports the mean and standard deviation values of VAA’s
intervals,excludingthefinal20minofmarkethourstoreplicate produced by DQN, OPD, PPO, HUL and TUL across the test dates,
7

S.Kimetal. ExpertSystemsWithApplications252(2024)124263
Fig.6. VAAforthetopthreemodelsaspertheevaluationsinTable4,overaperiodof60testdaysforselectedstocks.ThemodelsincludeTUL,HUL,andthebest-performing
thirdmodelspecifictoeachstock.Notably,TULdemonstratesmoreconsistentVAAwithfewerfluctuationscomparedtoothermodelsacrossallstocks.
Fig.7. U-shapeaveragesontestdaysforgroundtruthsandTransformer-predictedvalues.Theinnerplotsdepicttheabsoluteerrorvaluesofthepredictionsforeachstock..
and the percentage of them that fall under 10 bps. These methods aresignificantforstockswithsmallvolumes,itischallengingtoobtain
areusedbypractitioners,tomeasurehowthetraderswereperforming accurate predictions of the daily volume distribution by only using a
relativetotheVWAP.Ourproposeddual-levelapproachdemonstrates statistical U-shape distribution. However, U-shape Transformer in the
significant improvements compared to using short trading horizons firstlevelcanproperlyreflectthedailyvariation,andTULoutperforms
and single-level neural networks. Although HUL occasionally yields bothmodelsanddemonstratesthelowestmeanandstandarddeviation
betteraccuracy,TULgenerallyhasmuchlessdeviationfromthedaily values.ThehistogramsforVAAdistributionsgivenbyDQN,OPD,PPO,
cumulative VWAP than all other methods. This becomes evident in HULandTULoneachstocktestdataaredrawninFig.9,whichdepicts
the less liquid stock PH stock, where HUL lagged behind both DQN theclusteringtendencyofaccuraciesproducedbyTUL.
andPPO.Inthiscase,DQNandPPOexhibitedlowermeanandlower Thirdly,wecomparetheperformanceofTULandTURtoexamine
standarddeviationcomparedtoHUL.Sincethedailyvolumevariations which neural network architecture is optimal for order allocation in
8

S.Kimetal.
ExpertSystemsWithApplications252(2024)124263
Table3
Hyperparameters for DQN, OPD, PPO, HUL and TUL. Note that ↘ indicates a linearly annealing learning rate schedule. Network configurations of DQN and OPD follow Ning
etal.(2021)andFangetal.(2021),respectively.
| Hyperparameter       |     |     |     | DQN   |     | OPD   |     | PPO   |     |     | HUL   |     | TUL   |     |
| -------------------- | --- | --- | --- | ----- | --- | ----- | --- | ----- | --- | --- | ----- | --- | ----- | --- |
| Outeriterations      |     |     |     | 10000 |     | 10000 |     | 10000 |     |     | 10000 |     | 10000 |     |
| Inneriterations      |     |     |     | 20    |     | 10    |     | 10    |     |     | 10    |     | 10    |     |
| Batchsize            |     |     |     | 50    |     | 10    |     | 20    |     |     | 10    |     | 10    |     |
| PPOCLIP𝜀             |     |     |     | –     |     | 0.2   |     | 0.2   |     |     | 0.2   |     | 0.2   |     |
| Discountfactor𝛾      |     |     |     | –     |     | 1     |     | 1     |     |     | 1     |     | 1     |     |
| Numberoftrajectories |     |     |     | 1000  |     | 160   |     | 200   |     |     | 152   |     | 228   |     |
pereachouteriteration
| LSTMhiddendimension𝐻      |     |     |     | –    |     | –    |     | 128       |     |     | 128       |     | 129       |     |
| ------------------------- | --- | --- | --- | ---- | --- | ---- | --- | --------- | --- | --- | --------- | --- | --------- | --- |
| LSTMmodellearningrate     |     |     |     | –    |     | –    |     | 5e−5↘1e−5 |     |     | 5e−5↘1e−5 |     | 5e−5↘1e−5 |     |
| DQNorOPDmodellearningrate |     |     |     | 1e−4 |     | 1e−4 |     | –         |     |     | –         |     | –         |     |
| U-shapeEncoder            |     |     |     | –    |     | –    |     | –         |     |     | –         |     | 20        |     |
inputvectordimension𝑁
| U-shapeEncoder |     |     |     | –   |     | –   |     | –   |     |     | –   |     | 148 |     |
| -------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
embeddingdimension
| TransformerEncoder |     |     |     | –   |     | –   |     | –   |     |     | –   |     | 4   |     |
| ------------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
&Decodernumberofheads
| TransformerEncoder |     |     |     | –   |     | –   |     | –   |     |     | –   |     | 1   |     |
| ------------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
&Decodernumberoflayers
| TransformerEncoder |     |     |     | –   |     | –   |     | –   |     |     | –   |     | 128 |     |
| ------------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
&DecoderPFFNdimension
| Transformermodellearningrate |     |     |     | –   |     | –   |     | –   |     |     | –   |     | 1e−3↘2e−4 |     |
| ---------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --------- | --- |
approximatestheU-shapedistributionforthetestdatesaccuratelyand
|     |     |     |     |     |     |     | uses predicted | values      | for order | allocation. |     | Thus, TUL | becomes            | more |
| --- | --- | --- | --- | --- | --- | --- | -------------- | ----------- | --------- | ----------- | --- | --------- | ------------------ | ---- |
|     |     |     |     |     |     |     | adaptive       | in tracking | the daily | cumulative  |     | VWAP      | as it incorporates |      |
predictionsoffluctuationsthatmayappearintheU-shapedistribution
inthefuture.
WealsovalidatetheperformanceimprovementsofTULcompared
toothermodels,particularlyHUL,byexaminingTUL’sabilitytocap-
turedailyfluctuationsintheU-shape.Fig.8displaysthegroundtruth
U-shapeandTransformer-predicteddistributionsfortwosampledtest
|     |     |     |     |     |     |     | dates for   | the stock   | PH. We     | deliberately | choose        | PH             | since the    | perfor-   |
| --- | --- | --- | --- | --- | --- | --- | ----------- | ----------- | ---------- | ------------ | ------------- | -------------- | ------------ | --------- |
|     |     |     |     |     |     |     | mance gains | of TUL      | when       | compared     | to HUL        | were           | the largest  | for this  |
|     |     |     |     |     |     |     | particular  | stock,      | which has  | daily volume | distributions |                | that         | deviates  |
|     |     |     |     |     |     |     | the most    | from the    | average    | due to       | relatively    | low liquidity. |              | While the |
|     |     |     |     |     |     |     | U-shape     | Transformer | may        | occasionally | make          | erroneous      | predictions, |           |
|     |     |     |     |     |     |     | it quickly  | corrects    | itself and | adapts       | to changes    | in             | the daily    | volume    |
distribution,attemptingtoaccuratelyfollowratiochangesthroughout
theday.
5. Discussionandconclusion
|     |     |     |     |     |     |     | In this | study, | we propose | a dual-level |     | approach | to address | the |
| --- | --- | --- | --- | --- | --- | --- | ------- | ------ | ---------- | ------------ | --- | -------- | ---------- | --- |
problemoftrackingtheVWAPwithaccuracyandconsistency.Initially,
|     |     |     |     |     |     |     | as observed | in Fig. | 5, we | identified | that the | naive | uniform | distribu- |
| --- | --- | --- | --- | --- | --- | --- | ----------- | ------- | ----- | ---------- | -------- | ----- | ------- | --------- |
Fig.8. ThegroundtruthandTransformer-predictedU-shapeplotsfortwoselectedtest
datesforPH. Theinnerplotsdepicttheabsoluteerrorvaluesofthepredictions.. tion approach falls short in accurately tracking the daily VWAP. The
uniquenessofourmodelisinhowweimproveuponthisissuebytaking
|            |                            |        |       |                  |             |     | into account | the       | well-known     | U-shaped   | intraday | trading   | pattern. | In       |
| ---------- | -------------------------- | ------ | ----- | ---------------- | ----------- | --- | ------------ | --------- | -------------- | ---------- | -------- | --------- | -------- | -------- |
|            |                            |        |       |                  |             |     | the first    | stage, we | allocate       | the number | of       | orders to | execute  | in each  |
| the second | level. For SE, which       | is the | stock | with the highest | trading     |     |              |           |                |            |          |           |          |          |
|            |                            |        |       |                  |             |     | interval     | based     | on the U-shape | pattern.   | We       | consider  | two      | methods, |
| volume,    | TUR slightly underperforms |        | HUL   | and TUL and      | outperforms |     |              |           |                |            |          |           |          |          |
statisticalU-shapeandU-shapeTransformer,forimplementingthefirst
| all other | models. However, | it is evident | that | a significant | decline | in  |     |     |     |     |     |     |     |     |
| --------- | ---------------- | ------------- | ---- | ------------- | ------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
stage.ThenaiveapproachofusingthestatisticalU-shapedistribution
| performance | occurs for all | other stocks | with | lower trading | volumes. |     |     |     |     |     |     |     |     |     |
| ----------- | -------------- | ------------ | ---- | ------------- | -------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Thisperformancedegradationcanbeattributedtothelimitedcapabil- improvesperformance.However,ourexperimentsdemonstratethatthe
ity of the simpler RNN model when compared with the LSTM model U-shapeTransformerperformsevenbetter.Webelievethatthesuperior
to capture volume fluctuations in stocks that exhibit relatively lower performanceoftheU-shapeTransformerisduetoitsabilitytoadaptto
liquidity. unexpectedvariations,acapabilitythatthestatisticalU-shapemethod
Lastly, we demonstrate the effectiveness of the proposed U-shape lacks, as evidenced by Figs. 7 and 8. After the total daily orders are
TransformerusingFigs.7and8.Fig.7showstheplotsoftheground allocatedtoeachintervalinthefirststage,theLSTMmodelisusedin
truthU-shapeandTransformer-predictedU-shapevalues,wheretheU- thesecondstagetodeterminehowtheordersineachintervalshould
shaperatiovaluesforeach20-minuteintervalwereaveragedacrossall beexecuted.Oursimulationresultsshowthatthedual-levelapproach
consistentlyandaccuratelytracksthedailycumulativeVWAP.
testdates.WeobservethatourU-shapeTransformerisabletofollow
theU-shapepatternforthetestdatesonaverageforallstocks,which Whileourmethoddemonstratessuperiorperformancethroughex-
is in line with the comparable (and sometimes better) performance periments, a comprehensive mathematical understanding of its un-
of TUL against HUL. This is further supported by the fact that HUL derlying principles including uncertainty quantification remains to be
only utilizes the average U-shape ratio values of the past, while TUL explored. The limitations identified in our paper suggest potential
9

S.Kimetal.
ExpertSystemsWithApplications252(2024)124263
Fig.9. ThehistogramsoftheVAAdistributionsforeachmodelandstockonthetestdates.BothTULandHUL,employingtheU-shapeproperty,exhibitconsistentstabilityin
trackingtheVWAPforSE,SH,andKCstocks.Moreover,TULdemonstratesrobusttrackingforthePHstock,whichischaracterizedbysignificantdailyvolumefluctuations.
Table4
Table4summarizesthemean,standarddeviation,andpercentageofVAAswithintherange[0,10bps].TheMeanandStandardDeviationarerepresentedinunitsofbasispoints
(bps),andthe%in10bpsarerepresentedinunitsofpercentages.ForOPD,weexcludedVAAsabovethe80thpercentileforfaircomparisonssinceourtrainingdatasetwas
smallerthanthatusedinFangetal.(2021).ForTUR,weexcludedextremeoutliersforSH,KA,andPH.
| Stocks | Metric            | OPD   | DQN   | PPO   | HUL   | TUL   | TUR    |
| ------ | ----------------- | ----- | ----- | ----- | ----- | ----- | ------ |
|        | Mean              | 26.27 | 11.53 | 12.00 | 8.77  | 7.13  | 9.43   |
| SE     | StandardDeviation | 18.50 | 8.84  | 8.88  | 7.17  | 2.44  | 9.56   |
|        | %in10bps          | 25.00 | 57.38 | 49.18 | 65.57 | 88.52 | 65.57  |
|        | Mean              | 38.55 | 21.62 | 22.57 | 7.13  | 7.65  | 48.56  |
| SH     | StandardDeviation | 24.90 | 17.71 | 16.92 | 2.81  | 3.63  | 33.51  |
|        | %in10bps          | 10.42 | 30.00 | 25.00 | 91.67 | 88.33 | 5.45   |
|        | Mean              | 27.79 | 13.16 | 12.47 | 6.59  | 6.03  | 149.32 |
| KA     | StandardDeviation | 18.76 | 10.25 | 11.55 | 8.36  | 3.30  | 111.64 |
|        | %in10bps          | 18.75 | 46.67 | 56.67 | 90.00 | 86.67 | 9.43   |
|        | Mean              | 32.44 | 14.84 | 14.64 | 22.52 | 12.27 | 377.28 |
| PH     | StandardDeviation | 20.89 | 13.47 | 12.63 | 47.11 | 7.90  | 267.23 |
|        | %in10bps          | 18.75 | 40.98 | 40.98 | 32.79 | 44.26 | 1.85   |
directionsforfutureresearch.Additionally,wenotethatourproposed CRediTauthorshipcontributionstatement
methodentailsthefollowingethicalimplications.Firstly,ourapproach
providesinvestorswithnewoptionsfordealingwithmarketvolatility, Soohan Kim: Writing – original draft, Methodology, Software.
helpingtoavoidmarketimpactandtheriskoffront-runnersexploiting Jimyeong Kim: Writing – original draft, Data curation, Conceptual-
theirstrategies.Secondly,improvedabilityofourexecutionmodelto ization, Visualization. Hong Kee Sul: Writing – review & editing, In-
track the VWAP contributes to mitigating market impact, which can vestigation.YoungjoonHong:Writing–review&editing,Supervision,
potentiallyenhancemarkettransparencyandfairness. Projectadministration.
10

S.Kimetal.
ExpertSystemsWithApplications252(2024)124263
Declarationofcompetinginterest Kakade,S.M.,Kearns,M.,Mansour,Y.,&Ortiz,L.E.(2004).Competitivealgorithms
|     |     |     |     |     | for VWAP | and limit order trading. | In Proceedings of | the 5th ACM conference | on  |
| --- | --- | --- | --- | --- | -------- | ------------------------ | ----------------- | ---------------------- | --- |
Theauthorsdeclarethefollowingfinancialinterests/personalrela- electroniccommerce(pp.189–198).
Kato,T.(2015).VWAPexecutionasanoptimalstrategy.JSIAMLetters,7,33–36.
| tionships which | may be considered | as potential | competing | interests: |     |     |     |     |     |
| --------------- | ----------------- | ------------ | --------- | ---------- | --- | --- | --- | --- | --- |
Lin,S.,&Beling,P.A.(2019).Optimalliquidationwithdeepreinforcementlearning.
YoungjoonHongreportswasprovidedbySungkyunkwanUniversity.
InProceedingsofthe33rdconferenceonneuralinformationprocessingsystems,deep
reinforcementlearningworkshop.Vancouver,Canada.
Lin,S.,&Beling,P.A.(2020).Anend-to-endoptimaltradeexecutionframeworkbased
Dataavailability
onproximalpolicyoptimization..InIJCAI(pp.4548–4554).
|     |     |     |     |     | Macri, A., & | Lillo, F. (2024). Reinforcement | learning | for optimal execution | when |
| --- | --- | --- | --- | --- | ------------ | ------------------------------- | -------- | --------------------- | ---- |
Datawillbemadeavailableonrequest. liquidityistime-varying.arXivpreprintarXiv:2402.12049v2.
Madhavan,A.(2002).VWAPstrategies.Trading,1,32–39.
Acknowledgments Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T., Harley, T., Silver, D., &
|     |     |     |     |     | Kavukcuoglu, | K. (2016). Asynchronous | methods for | deep reinforcement | learning. |
| --- | --- | --- | --- | --- | ------------ | ----------------------- | ----------- | ------------------ | --------- |
InInternationalconferenceonmachinelearning(pp.1928–1937).PMLR.
TheworkofY.HongwassupportedbyBasicScienceResearchPro-
|     |     |     |     |     | Mnih, V., Kavukcuoglu, | K., Silver, D., | Graves, A., Antonoglou, | I., Wierstra, | D., & |
| --- | --- | --- | --- | --- | ---------------------- | --------------- | ----------------------- | ------------- | ----- |
gramthroughtheNationalResearchFoundationofKorea(NRF)funded
Riedmiller,M.(2013).Playingatariwithdeepreinforcementlearning.URL:http:
bytheMinistryofEducation,RepublicofKorea(NRF-2021R1A2C1093 //arxiv.org/abs/1312.5602. Cite arxiv:1312.5602Comment: NIPS Deep Learning
579)andbytheKoreagovernment(MSIT)(RS-2023-00219980). Workshop2013.
|            |     |     |     |     | Mnih, V., Kavukcuoglu,   | K., Silver, D.,    | Graves, A., Antonoglou,       | I., Wierstra, | D., & |
| ---------- | --- | --- | --- | --- | ------------------------ | ------------------ | ----------------------------- | ------------- | ----- |
|            |     |     |     |     | Riedmiller,              | M. (2013). Playing | atari with deep reinforcement | learning.     | arXiv |
| References |     |     |     |     | preprintarXiv:1312.5602. |                    |                               |               |       |
Nevmyvaka,Y.,Feng,Y.,&Kearns,M.(2006).Reinforcementlearningforoptimized
Almgren,R.,&Chriss,N.(2001).Optimalexecutionofportfoliotransactions.Journal trade execution. In Proceedings of the 23rd international conference on machine
| ofRisk,3,5–40. |     |     |     |     | learning(pp.673–680). |     |     |     |     |
| -------------- | --- | --- | --- | --- | --------------------- | --- | --- | --- | --- |
Berkowitz,S.A.,Logue,D.E.,&Noser,E.A.,Jr.(1988).Thetotalcostoftransactions Ning, B., Lin, F. H. T., & Jaimungal, S. (2021). Double deep q-learning for optimal
ontheNYSE.TheJournalofFinance,43(1),97–112. execution.AppliedMathematicalFinance,28(4),361–380.
Bertsimas, D., & Lo, A. W. (1998). Optimal control of execution costs. Journal of Pan,F.,Zhang,T.,Luo,L.,He,J.,&Liu,S.(2022).Learncontinuously,actdiscretely:
financialmarkets,1(1),1–50. Hybrid action-space reinforcement learning for optimal execution. In IJCAI (pp.
| Bialkowski,J.,Darolles,S.,&Fol,G.L.(2008).ImprovingVWAPstrategies:Adynamic |     |     |     |     | 3912–3918). |     |     |     |     |
| -------------------------------------------------------------------------- | --- | --- | --- | --- | ----------- | --- | --- | --- | --- |
volumeapproach.JournalofBanking&Finance,32,1709–1722. Podobnik,B.,Horvatic,D.,Petersen,A.M.,&Stanley,H.E.(2009).Cross-correlations
Byun,W.,Choi,B.,Kim,S.,&Jo,J.(2023).Practicalapplicationofdeepreinforcement betweenvolumechangeandpricechange.ProceedingsoftheNationalAcademyof
learningtooptimaltradeexecution.FinTech,2,414–429. Sciences,106(52),22079–22084.
Fang, Y., Ren, K., Liu, W., Zhou, D., Zhang, W., Bian, J., Yu, Y., & Liu, T.-Y. Schulman,J.,Levine,S.,Abbeel,P.,Jordan,M.,&Moritz,P.(2015).Trustregionpolicy
(2021). Universal trading for order execution with oracle policy distillation. 35, optimization.InInternationalconferenceonmachinelearning(pp.1889–1897).PMLR.
InProceedingsoftheAAAIconferenceonartificialintelligence(1),(pp.107–115). Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal
Goodhart,C.A.,&O’Hara,M.(1997).Highfrequencydatainfinancialmarkets:Issues policyoptimizationalgorithms.arXivpreprintarXiv:1707.06347.
andapplications.JournalofEmpiricalFinance,4,73–114. Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction. MIT
| Hendricks, D., | & Wilcox, D. (2014). | A reinforcement | learning | extension to the | Press. |     |     |     |     |
| -------------- | -------------------- | --------------- | -------- | ---------------- | ------ | --- | --- | --- | --- |
almgren-chrissframeworkforoptimaltradeexecution.In2014IEEEconferenceon Vaswani,A.,Shazeer,N.,Parmar,N.,Uszkoreit,J.,Jones,L.,Gomez,A.N.,Kaiser,Ł.,
computationalintelligenceforfinancialengineering&economics(CIFEr)(pp.457–464). &Polosukhin,I.(2017).Attentionisallyouneed.AdvancesinNeuralInformation
| IEEE. |     |     |     |     | ProcessingSystems,30. |     |     |     |     |
| ----- | --- | --- | --- | --- | --------------------- | --- | --- | --- | --- |
Huberman,G.,&Stanzl,W.(2005).Optimalliquiditytrading.ReviewofFinance,(2), Watkins,C.J.,&Dayan,P.(1992).Q-learning.MachineLearning,8(3),279–292.
165–200. Williams,R.J.(1992).Simplestatisticalgradient-followingalgorithmsforconnectionist
Jain,P.C.,&Joh,G.-H.(1988).Thedependencebetweenhourlypricesandtrading reinforcementlearning.ReinforcEmentLearning,5–32.
volume.TheJournalofFinancialandQuantitativeAnalysis,23(3),269–283.
11
