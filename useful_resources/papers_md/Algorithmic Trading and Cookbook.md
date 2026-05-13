<!-- Page 1 -->

Algorithmic Trading and
Quantitative Strategies


<!-- Page 2 -->

Algorithmic Trading and
Quantitative Strategies
Raja Velu
Departmentof Finance
Whitman Schoolof Management
Syracuse University
Maxence Hardy
e Trading Quantitative Research
J.P.Morgan
Daniel Nehren
Statistical Modeling&Development
Barclays


<!-- Page 3 -->

Firsteditionpublished2020
by CRCPress
6000 Broken Sound Parkway NW, Suite300, Boca Raton, FL33487-2742
andby CRCPress
2 Park Square, Milton Park, Abingdon, Oxon, OX144 RN
©2020 Taylor&Francis Group, LLC
CRCPressisanimprintof Taylor&Francis Group, LLC
Reasonableeffortshavebeenmadetopublishreliabledataandinformation, buttheauthorandpublisher
cannotassumeresponsibilityforthevalidityofallmaterialsortheconsequencesoftheiruse. Theauthors
andpublishershaveattemptedtotracethecopyrightholdersofallmaterialreproducedinthispublication
andapologizetocopyrightholdersifpermissiontopublishinthisformhasnotbeenobtained. Ifany
copyrightmaterialhasnotbeenacknowledgedpleasewriteandletusknowsowemayrectifyinany
futurereprint.
Exceptaspermittedunder USCopyright Law, nopartofthisbookmaybereprinted, reproduced, trans-
mitted, orutilizedinanyformbyanyelectronic, mechanical, orothermeans, nowknownorhereafter
invented, includingphotocopying, microfilming, andrecording, orinanyinformationstorageorretrieval
system, withoutwrittenpermissionfromthepublishers.
Forpermissiontophotocopyorusematerialelectronicallyfromthiswork, accesswww.copyright.comor
contactthe Copyright Clearance Center, Inc.(CCC),222 Rosewood Drive, Danvers, MA01923,978-750-
8400.Forworksthatarenotavailableon CCCpleasecontactmpkbookspermissions@tandf.co.uk
Trademarknotice:Productorcorporatenamesmaybetrademarksorregisteredtrademarks, andareused
onlyforidentificationandexplanationwithoutintenttoinfringe.
Libraryof Congress Control Number:2020932899
ISBN:9781498737166(hbk)
ISBN:9780429183942(ebk)
Typesetin STIXGeneral
by Nova Techset Private Limited, Bengaluru&Chennai, India


<!-- Page 4 -->

Contents
Preface xi
I Introductionto Trading 1
1 Trading Fundamentals 3
1.1 ABrief Historyof Stock Trading . . . . . . . . . . . . . . . . . . 3
1.2 Market Structureand Trading Venues:AReview . . . . . . . . . . 7
1.2.1 Equity Markets Participants . . . . . . . . . . . . . . . . . 7
1.2.2 Watering Holesof Equity Markets . . . . . . . . . . . . . . 8
1.3 The Mechanicsof Trading . . . . . . . . . . . . . . . . . . . . . . 13
1.3.1 How Double Auction Markets Work . . . . . . . . . . . . . 13
1.3.2 The Open Auction . . . . . . . . . . . . . . . . . . . . . . 14
1.3.3 Continuous Trading . . . . . . . . . . . . . . . . . . . . . 19
1.3.4 The Closing Auction . . . . . . . . . . . . . . . . . . . . . 24
1.4 Taxonomyof Data Usedin Algorithmic Trading . . . . . . . . . . 26
1.4.1 Reference Data . . . . . . . . . . . . . . . . . . . . . . . . 26
1.4.2 Market Data . . . . . . . . . . . . . . . . . . . . . . . . . 35
1.4.3 Market Data Derived Statistics . . . . . . . . . . . . . . . . 39
1.4.4 Fundamental Dataand Other Data Sets . . . . . . . . . . . 42
1.5 Market Microstructure:Economic Fundamentalsof Trading . . . . 44
1.5.1 Liquidityand Market Making . . . . . . . . . . . . . . . . 45
II Foundations:Basic Modelsand Empirics 51
2 Univariate Time Series Models 53
2.1 Trades and Quotes Data and Their Aggregation: From Point
Processesto Discrete Time Series . . . . . . . . . . . . . . . . . . 54
2.2 Trading Decisionsas Short-Term Forecast Decisions . . . . . . . . 56
2.3 Stochastic Processes:Some Properties . . . . . . . . . . . . . . . . 57
2.4 Some Descriptive Toolsand Their Properties . . . . . . . . . . . . 61
2.5 Time Series Modelsfor Aggregated Data:Modelingthe Mean . . . 63
2.6 Key Stepsfor Model Building . . . . . . . . . . . . . . . . . . . . 70
2.7 Testing for Nonstationary (Unit Root) in ARIMA Models: To
Differenceor Not To . . . . . . . . . . . . . . . . . . . . . . . . . 77
2.8 Forecastingfor ARIMAProcesses . . . . . . . . . . . . . . . . . . 78
2.9 Stylized Modelsfor Asset Returns . . . . . . . . . . . . . . . . . . 82
2.10 Time Series Modelsfor Aggregated Data:Modelingthe Variance. . 84
v


<!-- Page 5 -->

vi Contents
2.11 Stylized Modelsfor Varianceof Asset Returns . . . . . . . . . . . 90
2.12 Exercises . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 92
3 Multivariate Time Series Models 95
3.1 Multivariate Regression . . . . . . . . . . . . . . . . . . . . . . . 96
3.2 Dimension-Reduction Methods . . . . . . . . . . . . . . . . . . . 99
3.3 Multiple Time Series Modeling . . . . . . . . . . . . . . . . . . . 104
3.4 Co-Integration, Co-Movementand Commonalityin Multiple Time
Series . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 106
3.5 Applicationsin Finance . . . . . . . . . . . . . . . . . . . . . . . 110
3.6 Multivariate GARCHModels . . . . . . . . . . . . . . . . . . . . 112
3.7 Illustrative Examples . . . . . . . . . . . . . . . . . . . . . . . . . 114
3.8 Exercises . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 122
4 Advanced Topics 125
4.1 State-Space Modeling . . . . . . . . . . . . . . . . . . . . . . . . 125
4.2 Regime Switchingand Change-Point Models . . . . . . . . . . . . 128
4.3 AModelfor Volume-Volatility Relationships . . . . . . . . . . . . 131
4.4 Modelsfor Point Processes . . . . . . . . . . . . . . . . . . . . . . 134
4.4.1 Stylized Modelsfor High Frequency Financial Data. . . . . 136
4.4.2 Modelsfor Multiple Assets:High Frequency Context . . . . 140
4.5 Analysisof Time Aggregated Data . . . . . . . . . . . . . . . . . 141
4.5.1 Realized Volatilityand Econometric Models . . . . . . . . 141
4.5.2 Volatilityand Price Bar Data . . . . . . . . . . . . . . . . . 143
4.6 Analyticsfrom Machine Learning Literature . . . . . . . . . . . . 146
4.6.1 Neural Networks . . . . . . . . . . . . . . . . . . . . . . . 147
4.6.2 Reinforcement Learning . . . . . . . . . . . . . . . . . . . 150
4.6.3 Multiple Indicatorsand Boosting Methods. . . . . . . . . . 152
4.7 Exercises . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 154
III Trading Algorithms 157
5 Statistical Trading Strategiesand Back-Testing 159
5.1 Introductionto Trading Strategies:Originand History . . . . . . . 159
5.2 Evaluationof Strategies:Various Measures . . . . . . . . . . . . . 160
5.3 Trading Rulesfor Time Aggregated Data . . . . . . . . . . . . . . 161
5.3.1 Filter Rules . . . . . . . . . . . . . . . . . . . . . . . . . . 162
5.3.2 Moving Average Variantsand Oscillators . . . . . . . . . . 164
5.4 Patterns Discoveryvia Non-Parametric Smoothing Methods . . . . 166
5.5 ADecomposition Algorithm . . . . . . . . . . . . . . . . . . . . 168
5.6 Fair Value Models . . . . . . . . . . . . . . . . . . . . . . . . . . 170
5.7 Back-Testingand Data Snooping:In-Sampleand Out-of-Sample
Performance Evaluation . . . . . . . . . . . . . . . . . . . . . . . 171
5.8 Pairs Trading . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 174
5.8.1 Distance-Based Algorithms . . . . . . . . . . . . . . . . . 175
5.8.2 Co-Integration . . . . . . . . . . . . . . . . . . . . . . . . 176


<!-- Page 6 -->

Contents vii
5.8.3 Some General Comments . . . . . . . . . . . . . . . . . . 180
5.8.4 Practical Considerations . . . . . . . . . . . . . . . . . . . 181
5.9 Cross-Sectional Momentum Strategies . . . . . . . . . . . . . . . . 184
5.10 Extraneous Signals:Trading Volume, Volatility, etc. . . . . . . . . 189
5.10.1 Filter Rules Basedon Returnand Volume . . . . . . . . . . 191
5.10.2 An Illustrative Example . . . . . . . . . . . . . . . . . . . 193
5.11 Tradingin Multiple Markets . . . . . . . . . . . . . . . . . . . . . 198
5.12 Other Topics:Trade Size, etc. . . . . . . . . . . . . . . . . . . . . 202
5.13 Machine Learning Methodsin Trading . . . . . . . . . . . . . . . 204
5.14 Exercises . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 207
6 Dynamic Portfolio Managementand Trading Strategies 215
6.1 Introductionto Modern Portfolio Theory . . . . . . . . . . . . . . 215
6.1.1 Mean-Variance Portfolio Theory . . . . . . . . . . . . . . . 215
6.1.2 Multifactor Models . . . . . . . . . . . . . . . . . . . . . . 219
6.1.3 Tests Relatedto CAPMand APT . . . . . . . . . . . . . . 219
6.1.4 An Illustrative Example . . . . . . . . . . . . . . . . . . . 222
6.1.5 Implicationsfor Investing . . . . . . . . . . . . . . . . . . 224
6.2 Statistical Underpinnings . . . . . . . . . . . . . . . . . . . . . . 226
6.2.1 Portfolio Allocation Using Regularization . . . . . . . . . . 230
6.2.2 Portfolio Strategies:Some General Findings . . . . . . . . . 232
6.3 Dynamic Portfolio Selection . . . . . . . . . . . . . . . . . . . . . 234
6.4 Portfolio Trackingand Rebalancing . . . . . . . . . . . . . . . . . 236
6.5 Transaction Costs, Shortingand Liquidity Constraints . . . . . . . 239
6.6 Portfolio Trading Strategies . . . . . . . . . . . . . . . . . . . . . 242
6.7 Exercises . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 244
7 News Analytics:From Market Attentionand Sentimentto Trading 251
7.1 Introductionto News Analytics:Behavioral Financeand Investor
Cognitive Biases . . . . . . . . . . . . . . . . . . . . . . . . . . . 251
7.2 Automated News Analysisand Market Sentiment . . . . . . . . . . 256
7.3 News Analyticsand Applicationsto Trading . . . . . . . . . . . . 258
7.4 Discussion / Future of Social Media and News in Algorithmic
Trading . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 267
IV Execution Algorithms 269
8 Modeling Trade Data 271
8.1 Normalizing Analytics . . . . . . . . . . . . . . . . . . . . . . . . 271
8.1.1 Order Size Normalization:ADV . . . . . . . . . . . . . . . 272
8.1.2 Time-Scale Normalization:Characteristic Time . . . . . . . 273
8.1.3 Intraday Return Normalization:Mid-Quote Volatility . . . . 275
8.1.4 Other Microstructure Normalizations . . . . . . . . . . . . 276
8.1.5 Intraday Normalization:Profiles . . . . . . . . . . . . . . . 277
8.1.6 Remainder (ofthe Day)Volume . . . . . . . . . . . . . . . 282
8.1.7 Auctions Volume . . . . . . . . . . . . . . . . . . . . . . . 283


<!-- Page 7 -->

viii Contents
8.2 Microstructure Signals . . . . . . . . . . . . . . . . . . . . . . . . 283
8.3 Limit Order Book (LOB):Studying Its Dynamics . . . . . . . . . . 286
8.3.1 LOBConstructionand Key Descriptives. . . . . . . . . . . 287
8.3.2 Modeling LOBDynamics . . . . . . . . . . . . . . . . . . 289
8.3.3 Models Basedon Hawkes Processes . . . . . . . . . . . . . 295
8.4 Modelsfor Hidden Liquidity . . . . . . . . . . . . . . . . . . . . . 306
8.5 Modeling LOB:Some Concluding Thoughts . . . . . . . . . . . . 310
9 Market Impact Models 313
9.1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 313
9.2 What Is Market Impact? . . . . . . . . . . . . . . . . . . . . . . . 314
9.3 Modeling Transaction Costs (TC) . . . . . . . . . . . . . . . . . . 315
9.4 Historical Reviewof Market Impact Research . . . . . . . . . . . . 318
9.5 Some Stylized Models . . . . . . . . . . . . . . . . . . . . . . . . 320
9.6 Price Impactinthe High Frequency Setting . . . . . . . . . . . . . 323
9.7 Models Basedon LOB . . . . . . . . . . . . . . . . . . . . . . . . 324
9.8 Empirical Estimationof Transaction Costs . . . . . . . . . . . . . 327
9.8.1 Reviewof Select Empirical Studies . . . . . . . . . . . . . 328
10 Execution Strategies 335
10.1 Execution Benchmarks:Practitioner’s View . . . . . . . . . . . . . 336
10.2 Evolutionof Execution Strategies . . . . . . . . . . . . . . . . . . 344
10.3 Layersofan Execution Strategy . . . . . . . . . . . . . . . . . . . 348
10.3.1 Scheduling Layer . . . . . . . . . . . . . . . . . . . . . . . 348
10.3.2 Order Placement . . . . . . . . . . . . . . . . . . . . . . . 352
10.3.3 Order Routing . . . . . . . . . . . . . . . . . . . . . . . . 353
10.4 Formal Descriptionof Some Execution Models . . . . . . . . . . . 359
10.4.1 First Generation Algorithms . . . . . . . . . . . . . . . . . 359
10.4.2 Second Generation Algorithms. . . . . . . . . . . . . . . . 361
10.5 Multiple Exchanges:Smart Order Routing Algorithm . . . . . . . . 367
10.6 Execution Algorithmsfor Multiple Assets . . . . . . . . . . . . . . 371
10.7 Extendingthe Algorithmsto Other Asset Classes . . . . . . . . . . 375
V Technology Considerations 381
11 The Technology Stack 383
11.1 From Client Instructionto Trade Reconciliation . . . . . . . . . . . 383
11.2 Algorithmic Trading Infrastructure . . . . . . . . . . . . . . . . . 387
11.3 HFTInfrastructure . . . . . . . . . . . . . . . . . . . . . . . . . . 394
11.4 ATSInfrastructure . . . . . . . . . . . . . . . . . . . . . . . . . . 395
11.4.1 Regulatory Considerations . . . . . . . . . . . . . . . . . . 395
11.4.2 Matching Engine . . . . . . . . . . . . . . . . . . . . . . . 397
11.4.3 Client Tieringand Other Rules . . . . . . . . . . . . . . . . 397


<!-- Page 8 -->

Contents ix
12 The Research Stack 401
12.1 Data Infrastructure . . . . . . . . . . . . . . . . . . . . . . . . . . 401
12.2 Calibration Infrastructure . . . . . . . . . . . . . . . . . . . . . . 403
12.3 Simulation Environment . . . . . . . . . . . . . . . . . . . . . . . 404
12.4 TCAEnvironment . . . . . . . . . . . . . . . . . . . . . . . . . . 408
12.5 Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 410
Bibliography 411
Subject Index 433


<!-- Page 9 -->

Preface
Algorithms have been around since the day trading has started. But they have
gainedimportancewiththeadventofcomputersandtheautomationoftrading. Effi-
ciency in execution has taken center stage and with that, speed and instantaneous
processingofassetrelatedinformationhavebecomeimportant. Inthisbook, wewill
focus on the methodology rooted in financial theory and demonstrate how relevant
data—both in the high frequency and in the low frequency spaces—can be mean-
ingfullyanalyzed. Theintentionistobringboththeacademicsandthepractitioners
together. Westrivetoachievewhat George Box (firstauthor’steacher) oncesaid:
“Oneimportantideaisthatscienceisameanswhereby
learningisachieved, notbymeretheoreticalspeculation
ontheonehand, norbytheundirectedaccumulationof
practicalfactsontheother, butratherbyamotivated
iterationbetweentheoryandpractice.”
Wehopethatweprovideaframeworkforrelevantinquiriesonthistopic. Toquote
Judea Pearl,
“Youcannotansweraquestionthatyoucannotask, and
youcannotaskaquestionthatyouhavenowordsfor.”
Theemphasisofthisbook, thereaderswillnotice, isondataanalysiswithguidance
fromappropriatemodels. As C.R.Rao (firstauthor’steacher) hasaptlyobserved:
Allknowledgeis, infinalanalysis, history.
All Sciencesare, intheabstract, mathematics.
Alljudgementsare, intherationale, statistics.
Thisbookgivesaninsidelookintothecurrentworldof Electronic Tradingand
Quantitative Strategies. We address actual challenges by presenting intuitive and
innovative ideas on how to approach them in the future. The subject is then aug-
mentedthroughamoreformaltreatmentofthenecessaryquantitativemethodswith
a targeted review of relevant academic literature. This dual approach is also reflec-
tiveofthedynamics, typicalofquantsworkingonatradingfloorwherecommercial
needs, such as time to market, often supersede consideration of rigorous models in
favorofintuitivelysimpleapproaches.
Our unique approach in this book is to provide the reader with hands-on tools.
Thisbookwillbeaccompaniedbyacollectionofpractical Jupyter Notebookswhere
selectmethodsareappliedtorealdata. Thiswillallowthereaderstogobeyondtheory
xi


<!-- Page 10 -->

xii Preface
intotheactualimplementation, whilefamiliarizingthemwiththelibrariesavailable.
Whereverpossiblethechartsandtablesinthebookcanbegenerateddirectlyfrom
thesenotebooksandthedatasetsprovidedbringfurtherlifetothetreatmentofthe
subject. Wealsoaddexercisestomostofthechapters, sothatthestudentscanwork
throughontheirown. Theseexerciseshavebeentestedoutbygraduatestudentsfrom
Stanfordand Singapore Management University. Thenotebooksaswellasthedata
and the exercises are made available on: https://github.com/Nehren D/algo_
trading_and_quant_strategies. This site will be updated on a periodic basis.
Whilereadingandworkingthroughthisbook, thereadershouldbeabletogaininsight
intohowthefieldof Electronic Tradingand Quantitative Strategies, oneofthemost
activeandexcitingspacesintheworldoffinance, hasevolved.
Thisbookisdividedintofiveparts.
• Part I sets the stage. We narrate the history and evolution of Equity Trading
anddelveintoareviewofthecurrentfeaturesofmodern Market Structure. This
givesthereaderscontextonthebusinessaspectsoftradinginorderforthemto
understandwhythingsworkastheydo. Thenextsectionwillprovideabriefhigh-
levelfoundationaloverviewofmarketmicrostructurewhichexplainsandmodels
the dynamics of a trading venue heavily influenced by the core mechanism of
howtradingtakesplace:theprice-timeprioritylimit-orderbookwithcontinuous
doubleauction. Thiswillsetthestagefortheintroductionofacriticalbutelusive
conceptintrading:Liquidity.
• Part II provides an overview of discrete time series models applied to equity
trading. Wewilladdressunivariateandmultivariatetimeseriesmodelsofboth
meanandvarianceofassetreturns, andotherassociatedquantitiessuchasvol-
ume. Whilesomewhatlessusedtodaybecauseofhighfrequencytrading, these
models are important as conceptual frameworks, and act as baselines for more
advancedmethods. Wealsocoversomeessentialconceptsin Point Processesas
theactualtradingdatacancomeatirregularintervals. Thelastchapterof Part II
willpresentmoreadvancedtopicslike State-Space Modelsandmodern Machine
Learningmethods.
• Part III dives into the broad topic of Quantitative Trading. Here we provide
thereaderwithatoolkittoconfidentlyapproachthesubject. Historicalperspec-
tivesfrom Alphagenerationtotheartofbacktestingarecoveredhere. Sincemost
quantitativestrategiesareportfolio-based, meaningthatalphasareusuallycom-
binedandoptimizedoverabasketofsecurities, wewillbrieflyintroducethetopic
of Active Portfolio Managementand Mean-Variance Optimization, andthemore
advanced topic of Dynamic Portfolio Selection. We conclude this section dis-
cussingasomewhatrecenttopic:Newsand Sentiment Analytics. Ourintentisto
alsoremindthereaderthatthefieldisnever“complete”asnewapproaches (such
as this from behavioral finance) are embraced by practitioners once the data is
available.
• Part IV covers Execution Algorithms, a sub-field of Quantitative Trading
which has evolved separately from simple mechanical workflow tools, into a


<!-- Page 11 -->

Preface xiii
multi-billiondollarbusiness. Webeginbyreviewingvariousapproachestomod-
elingtradedata, andthendiveintothefundamentalsubjectof Market Impact, a
complexandleastunderstoodconceptinfinance. Havingsetthestage, thefinal
sectionpresentsareviewoftheevolutionandthecurrentstateoftheartin Exe-
cution Algorithms.
• Finally, Part V deals with some technical aspects of developing both quanti-
tativetradingstrategiesandexecutionalgorithms. Tradinghasbecomeahighly
technologicalprocessthatrequirestheintegrationofnumeroustechnologiesand
systems, rangingfrommarketdatafeeds, toexchangeconnectivity, tolow-latency
networkingandco-location, toback-officebookingandreporting. Developinga
modernandhighperformingtradingplatformrequiresthoughtfulconsideration
and some compromise. In this part, we look at some important details in cre-
ating a full end-to-end technology stack for electronic trading. We also want to
emphasizethecriticalbutoftenignoredaspectofasuccessfultradingbusiness:
Research Environment.
Acknowledgments: The ideas for this book were planted ten years ago, while the
firstauthor, Raja Velu, wasvisitingthe Statistics Departmentat Stanfordattheinvi-
tationof Professor T.W.Anderson. Professor Tze-Leung Lai, whowasinchargeof
the Financial Mathematics program, suggested initiating a course on Algorithmic
Trading. Thiscoursewasdevelopedbythefirstauthorandtheotherauthors, Daniel
Nehren and Maxence Hardy, offered guest lectures to bring the practitioner’s view
totheclassroom. Thisbookinlargepartistheresultofthatinteraction. Wewantto
gratefullyacknowledgetheopportunitygivenbythe Stanford’s Statisticsdepartment
andby Professors Laiand Anderson.
Weoweapersonaldebttomanypeoplefortheirinvaluablecommentsandintel-
lectualcontributiontomanysourcesfromwhichthematerialforthisbookisdrawn.
The critical reviews by Professors Guofu Zhou and Ruey Tsay at various stages
of writing are gratefully acknowledged. Colleagues from Syracuse University, Jan
Ondrich, Ravi Shukla, David Weinbaum, Lai Xu, Suhasini Subba Rao from Texas
A&M, and Jeffrey Wurglerfrom New York University, allreadthroughvariousver-
sionsofthisbook. Theircommentshavehelpedtoimproveitscontentandthepre-
sentation. Studentswhotookthecourseat Stanford University, National University
of Singaporeand Singapore Management Universityandtheteachingassistantswere
instrumentalinshapingthestructureofthebook. Inparticular, wewanttorecognize
thehelpof Balakumar Balasubramaniam, whoofferedextensivecommentsonanear-
lierversion. Ontheintellectualside, wehavedrawnmaterialfromtheclassicbooks
by Tsay (2010);Box, Jenkins, Reinsel, Ljung (2015) onthemethodology;Campbell,
Loand Mac Kinlay (1996) onfinance;Friedman, Hastieand Tibshirani (2009) onsta-
tisticallearning. Inthetoneandsubstanceattimes, wecouldnotsaybetterthanwhat
isalreadysaidintheseclassicsandsothereadersmaynoticesomesimilarities.
Wehavereliedheavilyontheableassistanceof Caleb Mc Whorter, whohasput
thebooktogetherwithallthedemandsofhisowngraduatework. Wewanttothank


<!-- Page 12 -->

xiv Preface
ourdoctoralstudents, Kris Hermanand Zhaoque Zhou (Chosen) fortheirhelpatvari-
ousstagesofthebook. Thejointworkwiththemwasusefultodrawuponforcontent.
Asthefocusofthisbookisontheuseofrealdata, werelieduponseveralsourcesfor
help. Wewanttothank Professor Ravi Jagannathanforsharinghisthoughtsanddata
onpairstradingin Chapter5 and Kris Herman, whosenotesonthe Hawkesprocess
areusedin Chapter8.Thesentimentdatausedin Chapter7 wasprovidedbyi Sen-
tiumandthanksto Gautham Sastri. Scott Morris, William Douganand Peter Layton
at Blackthorne Inc, whosewillingnesstohelponashortnotice, onmattersrelatedto
dataandtradingstrategies isverymuchappreciated. We wanttoalsoacknowledge
editorialhelpfrom Claire Harshberberand Alyson Nehren.
Generoussupportwasprovidedbythe Whitman Schoolof Managementandthe
Departmentof Financefortheproductionofthebook. Raja Veluwouldliketothank
former Dean Kenneth Kavajecz and Professors Ravi Shukla and Peter Koveos who
serve (d) as department chairs and Professor Michel Benaroch, Associate Dean for
Research, fortheirencouragementandsupport.
Lastbutnotleasttheauthorsaregratefulandhumbledtohave Adam Hoganpro-
videtheartworkforthebookcover. Theoriginalpiecespecificallymadeforthisbook
isabeautifulexampleof Algorithmic Artrepresentingthetradingintensityofthe US
stockuniverseonthevarioustradingvenues. Wecannotthinkofamorefittingimage
forthisbook.
Finally, nowordswillsufficefortheloveandsupportofourfamilies.
Raja Velu
Maxence Hardy
Daniel Nehren


<!-- Page 13 -->

About the Authors
Raja Velu
Raja Veluisa Professorof Financeand Business Analyticsinthe Whitman School
of Management at Syracuse University. He obtained his Ph. D. in Business/Statis-
tics from University of Wisconsin-Madison in 1983. He served as a marketing fac-
ulty at the University of Wisconsin-Whitewater from 1984 to 1998 before moving
to Syracuse University. He was a Technical Architect at Yahoo! in the Sponsored
Search Division and was a visiting scientist at IBM-Almaden, Microsoft Research,
Googleand JPMC.Hehasalsoheldvisitingpositionsat Stanford’s Statistics Depart-
mentfrom2005 to2016 andwasavisitingfacultyatthe Indian Schoolof Business,
National University of Singapore and Singapore Management University. His cur-
rent research includes Modeling Big Chronological Data and Forecasting in High-
Dimensionalsettings. Hehaspublishedinleadingjournalssuchas Biometrika, Jour-
nalof Econometricsand Journalof Financialand Quantitative Analysis.
Maxence Hardy
Maxence Hardy is a Managing Director and the Head of e Trading Quantitative
Research for Equities and Futures at J.P. Morgan, based in New York. Mr. Hardy
isresponsibleforthedevelopmentofthealgorithmictradingstrategiesandmodels
underpinningtheagencyelectronicexecutionproductsforthe Equitiesand Futures
divisions globally. Prior to this role, he was the Asia Pacific Head of e Trading and
Systematic Trading Quantitative Researchforthreeyears, aswellas Asia Pacific Head
of Productforagencyelectronictrading, basedin Hong Kong. Mr. Hardyjoined J.P.
Morganin2010 from Societe Generalewherehewaspartofthealgoteamdevelop-
ing execution solutions for Program Trading. Mr. Hardy holds a master’s degree in
quantitativefinancefromthe University Paris IXDauphinein France.
Daniel Nehren
Daniel Nehren is a Managing Director and the Head of Statistical Modelling and
Developmentfor Equitiesat Barclays. Basedin New York, Mr. Nehrenisresponsible
forthedevelopmentofalgorithmictradingproductandmodel-basedbusinesslogic
forthe Equitiesdivisionglobally. Mr. Nehrenjoined Barclaysin2018 from Citadel,
wherehewastheheadof Equity Execution. Mr. Nehrenhasover16 years’experience
inthefinancialindustrywithafocusonglobalequitymarkets. Priorto Citadel, Mr.
Nehrenheldrolesat J.P.Morganasthe Global Headof Linear Quantitative Research,
Deutsche Bankasthe Directorand Co-Headof Delta One Quantitative Productsand
Goldman Sachs as the Executive Director of Equity Strategy. Mr. Nehren holds a
doctorateinelectricalengineeringfrom Politecnico Di Milanoin Italy.
xv


<!-- Page 14 -->

xvi Aboutthe Authors
Dedicatedto...
Yasodha, withoutherloveandsupport, thisisnotpossible.
R.V.
Melissa, forherpatienceeverystepoftheway, andhernever
endingsupport.
M.H.
Alyson, themuse, thepatientpartner, theinspirationforthiswork
andthenext. Andtheboxisstillnotfull.
D.N.


<!-- Page 15 -->

Part I
Introduction to Trading


<!-- Page 16 -->

2 Algorithmic Tradingand Quantitative Strategies
Weprovideabriefintroductiontomarketmicrostructureandtradingfromaprac-
titioner’spointofview. Thetermsusedinthispart, allcanbetracedbacktoacademic
literature;butthediscussioniskeptsimpleanddirect. Thedata, whichiscentralto
alltheanalysesandinferences, isthenintroduced. Thecomplexityofusingdatathat
canarise atirregularintervalscan bebetterunderstoodwith anexampleillustrated
here. Finally, thelastpartofthischaptercontainsabriefacademicreviewofmarket
microstructure—atopicaboutthemechanicsoftradingandhowthetradingcanbe
influencedbyvariousmarketdesigns. Thisisanevolvingfieldthatisofmuchinterest
toall:regulators, practitionersandacademics.


<!-- Page 17 -->

1
Trading Fundamentals
1.1 ABrief Historyof Stock Trading
Why We Trade:Companiesneedcapitaltooperateandexpandtheirbusinesses. To
raisecapital, theycaneitherborrowmoneythenpayitbackovertimewithinterest,
or they can sell a stake (equity) in the company to an investor. As part owner of
thecompany, theinvestorwouldthenreceiveaportionoftheprofitsintheformof
dividends. Equity and Debt, being scarce resources, have additional intrinsic value
thatchangeovertime;theirpricesareinfluencedbyfactorsrelatedtotheperformance
ofthecompany, existingmarketconditionsandinparticular, thefutureoutlookofthe
company, thesector, thedemandandsupplyofcapitalandtheeconomyasawhole.
Forinstance, ifinterestrateschargedtoborrowcapitalchange, thiswouldaffectthe
value of existing debt since its returns would be compared to the returns of similar
products/companiesthatofferhigher/lowerratesofreturn. Whenwediscusstrading
inthisbookwerefertotheactofbuyingandsellingdebtorequity (aswellasother
typesofinstruments) ofvariouscompaniesandinstitutionsamonginvestorswhohave
differentviewsoftheirintrinsicvalue.
Thesesecondarymarkettransactionsviatradingexchangesalsoservethepurpose
of“pricediscovery”(O’Hara (2003)[275]).Buyersandsellersmeetandagreeona
pricetoexchangeasecurity. Whenthattransactionismadepublic, itinturninforms
otherpotentialbuyersandsellersofthemostrecentmarketvaluationofthesecurity.
Theevolutionofthetradingprocessoverthelast200 yearsmakesforanincred-
ible tale of ingenuity, fierce competition and adept technology. To a large extent, it
continuestobedrivenbythepositive (andattimesnotsopositive) forcesofmaking
profit, creatingovertimeahighlycomplex, andamazinglyefficientmechanism, for
evaluatingtherealvalueofacompany.
The Originsof Equity Trading:Thetalebeginson May17,1792 whenagroupof
24 brokers signed the Buttonwood Agreement. This bound the group to trade only
with each other under specific rules. This agreement marked the birth of the New
York Stock Exchange (NYSE).Whilethe NYSEisnottheoldest Stock Exchangein
3


<!-- Page 18 -->

4 Algorithmic Tradingand Quantitative Strategies
the world,1 nor the oldest in the US,2 it is without a question the most historically
importantandundisputedsymbolofallfinancialmarkets. Thusinouropinion, itis
the most suitable place to start our discussion. The NYSE soon after moved their
operations to the nearby Tontine Coffee House and subsequently to various other
locations around the Wall Street area before settling in the current location on the
cornerof Wall St.and Broad St.in1865.
For the next almost 200 years, stock exchanges evolved in complexity and in
scope. They, however, conceptually remained unchanged, functioning as physical
locations where traders and stockbrokers met in person to buy and sell securities.
Most of the exchanges settled on an interaction system called Open Outcry where
new orders were communicated to the floor via hand signals and with a Market
Makerfacilitatingthetransactionsoftensteppingintoprovideshorttermliquidity.
Theadventofthetelegraphandsubsequentlythetelephonehaddramaticeffectsin
acceleratingthetradingprocessandthedisseminationofinformation, whileleaving
thefundamentalprocessoftradinguntouched.
Electronificationandthe Startof Fragmentation:Changescameinthelate1960 s
and early 1970 s. In 1971, the NASDAQ Stock Exchange launched as a completely
electronic system. Initially started as a quotation site, it soon turned into a full
exchange, quickly becoming the second largest US exchange by market capitaliza-
tion. In the meantime, another innovation was underway. In 1969, the Institutional
Networks Corporationlaunched Instinet, acomputerizedlinkbetweenbanks, mutual
fundcompanies, insurancecompaniessothattheycouldtradewitheachotherwith
immediacy, completely bypassing the NYSE. Instinet was the first example of an
Electronic Communication Network (ECN), an alternative approach to trading that
grew in popularity in the 80 s and 90 s with the launch of other notable venues like
Archipelagoand Island ECNs.
Thisevolutionstartedatrend (Liquidity Fragmentation) inmarketstructurethat
grewovertime. Interestforasecurityisnolongercentralizedbutratherdistributed
acrossmultiple“liquiditypools.”Thisdecentralizationofliquiditycreatedsignificant
challenges to the traditional approach of trading and accelerated the drive toward
electronification.
The Birthof High Frequency Tradingand Algorithmic Trading:Theyear2001
brought another momentous change in the structure of the market. On April 9 th,
the Securitiesand Exchange Commission (SEC)3 mandatedthattheminimumprice
incrementonanyexchangeshouldchangefrom1/16 thofadollar (≈ 6.25 cents) to
1 Thishonorsitswiththe Amsterdam Stock Exchangedatingbackto1602.
2 The Philadelphia Stock Exchangehasa2 yearheadstarthavingbeenestablishedin1790.
3 SECisanindependentfederalgovernmentagencyresponsibleforprotectinginvestors, andmaintain-
ingfairandorderlyfunctioningofthesecuritiesmarkethttp://www.sec.gov


<!-- Page 19 -->

Trading Fundamentals 5
1 cent.4 This seemingly minor rule change with the benign name of ‘Decimaliza-
tion’ (moving from fractions to decimal increments) had a dramatic effect, causing
the average spread to significantly drop and with that, the profits of market makers
and broker dealers also declined. The reduction in profit forced many market mak-
ing firms to exit the business which in turn reduced available market liquidity. To
bring liquidity back, exchanges introduced the Maker-Taker fee model. This model
compensated the traders providing liquidity (makers) in the form of rebates, while
continuingtochargeafeetotheconsumerofliquidity (takers).5
Themaker-takermodelcreatedunintentionalconsequences. Ifonecouldprovide
liquidity while limiting liquidity taking, one could make a small profit, due to the
rebate with minimal risk and capital. This process needs to be fairly automated as
the pertrade profitwould beminimal, requiringheavy tradingto generate realrev-
enue. Trading also needs to be very fast as position in the order book and speed of
cancellation of orders are both critical to profitability. This led to the explosion of
whatwetodaycall High Frequency Trading (HFT) andtothewildultra-lowlatency
technology arms race that has swept the industry over the past 15 years. HFT style
trading, formerlycalledopportunisticmarketmaking, alreadyexistedbutneverasa
significantportionofthemarket. Atitspeakitwasestimatedthatmorethan60%of
alltradingwasgeneratedby HFTs.
Thedecreaseinaveragetradingcost, aswellastheseculartrendofon-lineinvest-
ing led to a dramatic increase in trading volumes. On the other hand, the reduction
inper-tradecommissionandprofitabilityforcedbroker-dealerstobeginautomating
some of their more mundane trading activities. Simple workflow strategies slowly
evolvedintoafieldthatwenowcall Algorithmic Executionwhichisamaintopicof
thisbook.
Dark Poolsand Reg NMS:Inthemeantime, themarketstructurecontinuedtoevolve
andfragmentationcontinuedtoincrease. Inthelateeightiesandearlyninetiesanew
typeoftradingvenuesurfaced, withasomewhatdifferentvalueproposition:Allowing
traders to find a block of liquidity without having to display that information “out
loud”(i.e., onexchange).Thisapproachpromisedreducedriskofinformationleakage
asthesevenuesdonotpublishmarketdata, onlythenotificationofatradeisconveyed
afterthefact. Theseaptlybutominouslynamed Dark Poolshavebecomeastaplein
equitytrading, andnowrepresentanestimated30-40%ofallliquiditybeingtraded
in USEquities.
In2005, the Regulation National Market System (Reg NMS) wasintroducedinan
efforttoaddresstheincreaseintradingcomplexityandfragmentation. Additionally,
4 The1/16 thpriceincrementwasavestigeof Spanishmonetarystandardsofthe1600 swhendoubloons
wheredividedin2,4,8 parts.
5 Forsomereaderstheconceptsofprovidingandtakingliquiditymightbemurkyatbest. Donotfret,
thiswillallbecomeclearwhenwediscussthetradingprocessandintroduce Market Microstructure.


<!-- Page 20 -->

6 Algorithmic Tradingand Quantitative Strategies
itacceleratedthetransformationofmarketstructureinthe US.Throughitstwomain
rules, the intent of Reg NMS was to promote best price execution for investors by
encouraging competition among individual exchanges and individual orders. The
Access Rule promotes non-discriminatory access to quotes displayed by the var-
ious trading centers. It also had established a cap, limiting the fees that a trad-
ing center can charge for accessing a displayed quote. The Order Protection Rule
requires trading centers to obtain the best possible price for investors wherever it
is represented by an immediately accessible quote. The rule designates all regis-
tered exchanges as “protected” venues. It further mandates that apart from a few
exceptions, all market participants transact first at a price equal or better to the
best price available in these venues. This is known as the National Best Bid Offer
(NBBO).
The Order Protection Ruleinparticular, hadasignificanteffectonthe USmarket
structure. Bymakingallliquidityinprotectedvenuesofequalstatusitsignificantly
contributedtofurtherfragmentation. In2005 the NYSEmarketsharein NYSE-listed
stocks was still above 80%. By 2010, it had plunged to 25% and has not recovered
since.
Fragmentation continued to increase with new prominent entrants like BATS
Tradingand Direct Edge. Speedandtechnologyrapidlybecamemajordifferentiating
factors of success for market makers and other participants. The ability to process,
analyze, and react to market data faster than competing participants, meant captur-
ingfleetingopportunitiesandsuccessfullyavoidingadverseselection. Therefore, to
gainormaintainanedge, marketparticipantsheavilyinvestedintechnology, infaster
networks, andinstalledtheirserversinthesamedatacenters, asthevenuestheytrans-
actedon (co-location).
Conclusion:Thiswhirlwindtourofthehistoryoftradingwasprimarilytoreviewthe
background forces that led to the dizzying complexity of modern market structure,
that may be difficult to comprehend without the appropriate context. Although this
overviewwaslimitedtothe USAalone, itisnotmeanttoimplythattherestofthe
worldstayedstill. Europeand Asiabothprogressedalongsimilarlines, althoughata
slowerandmorecompressedpace.


<!-- Page 21 -->

Trading Fundamentals 7
1.2 Market Structureand Trading Venues:AReview
1.2.1 Equity Markets Participants
Withthecontextpresentedintheprevioussection, wenowprovideabriefreview
ofthestateofmodern Market Structure. Inordertogetasenseoftheunderpinnings
ofequitymarkets, itisimportanttounderstandwhoarethevariousparticipants, why
theytradeandwhattheirprincipalfocusis.
Long Only Asset Managers: These are the traditional Mutual Fund providers like
Vanguard, Fidelity, etc. Asizableportionof UShouseholdsinvestinmutualfundsas
partoftheircompany’spensionsfunds,401 K’s, andotherretirementvehicles. Some
ofthelargestprovidersareofconsiderablesizeandhaveaccumulatedmultipletril-
lionsofdollarsundertheirmanagement. Theyarecalled Long Onlyassetmanagers
becausetheyarerestrictedfromshortselling (whereyouborrowsharestosellinthe
marketinordertobenefitfromapricedrop).Theycanonlyprofitthroughdividends
and price appreciation. These firms need to trade frequently in order to re-balance
theirlargeportfolios, tomanagein-flows, out-flows, andtoachievethefund’sobjec-
tive. Theobjectivemaybetotrackandhopefullybeatacompetitivebenchmark. The
timehorizontheseinvestorscareaboutisingenerallong, frommonthstoyears. His-
torically, theseparticipantswerelessconcernedwithtransactioncostsbecausetheir
investmenttimescalesarelongandthereturnstheytargetdwarfthefewdozenbasis
pointsnormallyincurredintransactioncosts.6 Whattheyareparticularlyconcerned
about is information leakage. Many of these trades last from several days to weeks
andtheriskisthatotherparticipantsinthemarketrealizetheirintentionandprofit
fromtheinformationtothedetrimentofthefund’sinvestors.
Long-Short Asset Managers and Hedge Funds: These are large and small firms
cateringtoinstitutionalclientsandwealthyinvestors. Theyoftenrunmultiplestrate-
gies, most often market-neutral long-short strategies, holding long and short posi-
tions in order to have reduced market exposure and thus hopefully perform well in
bothraisingandfallingmarkets. Investorsofthistypeincludefirmslike Bridgewa-
ter, Renaissance Technologies, Citadel, Point 72 (former SAC Capital), and many
others. Theirtimehorizonisvaried, aswiththestrategiestheyemploy, butingeneral
theyareshorterthantheir Long Onlycounterparts. Theyalsotendtobeveryfocused
onminimizingtransactioncosts, becausetheyoftenmakemanymoresmallershort
terminvestmentsandthetransactioncostscanadduptobeasignificantfractionof
theirexpectedprofits.
6 Inrecentyearsthough, competitivepressureaswellas Best Executionregulatoryobligationshave
broughtalotoffocusontransactioncostsminimizationtothe Long Onlycommunityaswell.


<!-- Page 22 -->

8 Algorithmic Tradingand Quantitative Strategies
Broker-Dealers (a.k.a. Sell Side): These firms reside between the “Buy Side”
(generic term for asset management firms and hedge funds) and the various
exchanges.7 Theycanacteitheras Agentfortheclient (i.e., Broker) orprovideliq-
uidityas Principal (i.e., Dealer) fromtheirownaccounts. Theyhistoricallyalsohad
largeproprietarytradingdesksinvestingthefirm’scapitalusingstrategiesnotdissim-
ilartothestrategiesthathedgefundsuse. Sincetheintroductionofthe Dodd-Frank
Act,8 the amount of principal risk that a firm can carry has dramatically decreased
and banks had to shed their proprietary trading activities either shutting down the
desksorspinningthemoffintoindependenthedgefunds.
HFTs, ELPs (Electronic Liquidity Providers), and DMMs (Designated Market
Makers): These participants generate returns by acting as facilitators between the
above participants, providing liquidity and then unwinding it at a profit. They can
also act as aggregators of retail liquidity, e.g., individual investors who trade using
online providers like ETrade. Usually the most technology savvy operators in the
marketplace, theyleverageultra-low-latencyinfrastructureinordertobeextremely
nimble. Theygetinandoutofpositionsrapidly, takingadvantageoftinymis-pricing.
1.2.2 Watering Holesof Equity Markets
Nowthatweknowwhotheplayersare, webrieflyreviewvariouswaystheyaccess
liquidity.
Exchanges:Thisisstillthestandardapproachtotradingandaccountsforabout60–
70%ofallactivity. Thisiswhereaninvestorwillgoforimmediacyandasurerout-
come. Becausethefullorder-bookofanexchange, arrivals/cancellationsareallpub-
lished, thetraderknowsexactlytheliquiditythatisavailableandcanplanaccordingly.
Thisinformation, itshouldbekeptinmind, isknowntoallparticipants, especially
(duetotheiroftentechnologicaladvantages) theoneswhosestrategiesfocusonpat-
ternsoflargedirectionaltrades. Atraderthatneedstobuyorsellinlargesizewill
needtobecarefulabouthowmuchinformationtheirtradesdisseminateortheymay
pay dearly. The whole field of Algorithmic Execution evolved as an effort to trade
largepositionswhileminimizingmarketimpactandinformationalleakage.
Apartfromtheseconcerns, interactingwithexchangesisarguablythemostbasic
tasktotrading. Butitisnotstraightforward. Atthetimeofthiswriting, a USequity
trader can buy or sell stocks in 15 registered exchanges (13 actively trading).9 As
mentioned before when discussing Reg NMS, these exchanges are all “protected.”
Thus, the liquidity at the best price cannot be ignored. An exchange will need to
7 Notethatonlymemberfirmsareallowedtotradeonexchangeandmostassetmanagementfirmsare
non-membersthusneedanintermediarytotradeontheirbehalf.
8 https://www.cftc.gov/Law Regulation/Dodd Frank Act/index.htm
9 https://www.sec.gov/fast-answers/divisionsmarketregmrexchangesshtml.html


<!-- Page 23 -->

Trading Fundamentals 9
reroutetootherexchangeswherepriceisbetter (chargingafeeforit).Smart Order
Routershaveevolvedtomanagethiscomplexity.
Recentyearshaveseenaconsolidationoftheseexchangesinthehandsofmainly
threeplayers:ICE, NASDAQand CBOE.Hereisalistofvenuesoperatedbyeach,
respectively:
• NYSE, ARCA, MKT(former AMX, American Stock Exchange), NSX(former
National Stock Exchange), CHX(former Chicago Stock Exchange)
• NASDAQ, PHLX (former Philadelphia Stock Exchange), ISE (former Interna-
tional Securities Exchange), BX(former Boston Stock Exchange)
• CBOE, BZX(former BATS), BYX(former BATS-Y), EDGA, EDGX
Thenewestexchangeandasofnowtheonlyremainingindependentexchange, isthe
Investors Exchange (IEX).10 Wewilldiscussmoreaboutthislater.
Aninterestingobservationisthatthisconsolidationdidnothappenwithacontem-
poraneousreductioninfragmentation. Thesevenuescontinuetooperateasseparate
poolsofliquidity. Whiletheexchangeprovidersmakevalidargumentsthattheypro-
videdifferentbusinesspropositions, thereisagrowingconcernintheindustrythat
therevenuemodelfortheseexchangesisnowlargelycenteredaroundprovidingmar-
ketdataandchargingforexchangeconnectivityfees.11 Becausethevenuequotesare
protected, anyseriousoperatorneedstoconnectwiththeexchangesandleveragetheir
directmarketdatafeedsintheirtradingapplications. Withmoreexchanges, themore
connectionsandfeestheseexchangecancharge. Recentlyregulatorsarestartingto
weighinonthiscontentioustopic.12
It is also interesting to note that most of these exchanges are hosted in one of
fourdatacenterslocatedinthe New Jerseycountryside:Mahwah, Secaucus, Carteret,
Weehawken. Thedistancebetweenthesedatacentersaddssomelatencyinthedis-
seminationofinformationacrossexchangesandthuscreateslatencyarbitrageoppor-
tunities. Superfast HFTsco-locateineachdatacenterandleveragethebestavailable
technologysuchasmicrowaveandmorerecentlylasertechnologiestoconnectthem.
Theseoperatorscanseemarketdatachangesbeforeothersdo. Thentheyeithertrade
fasterorcanceltheirownquotestoavoidadverseselection (aphenomenonknownas
liquidityfading).
Allexchangeshavealmostexactlythesametradingmechanismandfromthetrad-
ingperspective, behaveexactlythesamewayduringthecontinuouspartofthetrading
dayexceptfortheopeningandclosingauctions. Theyprovidevisible (meaningthat
marketdataisdisseminated) orderbooksandoperateonaprice/timeprioritybasis.13
10 In2019, agroupoffinancialinstitutionsfiledanapplicationforamembersownedexchange, MEMX.
11 https://www.sifma.org/wp-content/uploads/2019/01/Expand-and-SIFMA-An-
Analysis-of-Market-Data-Fees-08-2018.pdf
12 https://www.sec.gov/tm/staff-guidance-sro-rule-filings-fees
13 Certain Futuresand Optionsexchangeshaveapro-ratamatchingmechanismwhichisdiscussedlater.


<!-- Page 24 -->

10 Algorithmic Tradingand Quantitative Strategies
Most of the exchanges use the maker-taker fee model that we discussed earlier but
some: BYX, EDGA, NSX, BX, have adopted an ‘inverted’ fee model, where post-
ingliquidityincursafeewhilearebateisprovidedfortakingliquidity. Thischange
causesthesevenuestodisplayamarkedlydifferentbehavior. Thecostforproviding
liquidity removes the incentive of rebate seeking HFTs; however these venues are
usedfirstwhenneedingimmediateliquidityasthetakeriscompensated. Thisinturn
brings in passive liquidity providers who are willing to pay for trading passively at
thatprice. Thisinterplayaddsasubtleandstillnotverywellunderstooddynamicto
analreadycomplexmarketstructure.
Finally, afewobservationsabout IEX.Itstartedasan Alternative Trading System
(ATS) whosemaininnovationisa38 milecoilofopticalfiberplacedinfrontofits
tradingengine. Thisintroducesa350 microseconddelayeachwayaptlynamedthe
“speedbump”thatismeanttoremovethespeedadvantagesof HFTs. Theintentisto
reducetheefficacyofthemorelatency-sensitivetacticsandthusprovidealiquidity
poolthatisless“toxic.”On June17,2016 IEXbecameafullfledgedexchangeaftera
verycontroversialprocess.14,15 Itwasmarketedasanexchangethatwouldbedifferent
and would attract substantial liquidity due to its innovative speed bump. But, as of
2019, thispotentialremainsstillsomewhatunrealizedand IEXhasnotmovedmuch
from the 2% to 3% market share range of which 80% is still in the form of hidden
liquidity. Thatbeingsaid IEXhasestablisheditselfasavocalcritic16 ofthecurrent
state of affairs continuing to shed light on some potential conflicts of interest that
have arisen in trading. At the time of this writing IEX does not charge any market
datafees.17
ATS/Dark Pools:Asdiscussedabove, tradinginlargeblocksinexchangesisnota
simplematterandrequiresadvancedalgorithmsforslicingtheblockordersandsmart
order routers for targeting the liquid exchanges. Even then the risk and the result-
ing costs due to information leakage can be significant. Dark Pools were invented
to counterbalance this situation. One investor can have a large order “sitting” in a
dark pool with no one knowing that it is there and would be able to find the other
sidewithoutshowinganysignalsoftheorder’spresence. Darkpoolsdonotdisplay
anyorderinformationandusethe NBBO(National Best Bid/Offer) asthereference
price. Inalmostallcases, toavoidaccessingprotectedvenues, thesepoolstradeonly
at the inside market (at or within the bid ask spread). Although, in order to maxi-
mizetheprobabilityoffindingliquiditymostoftheblockinteractionhappensatthe
mid-point. Off-exchangetradinghasgainedmoreandmoretractioninthelastfifteen
14 https://www.sec.gov/comments/10-222/10-222.shtml
15 https://www.bloomberg.com/news/articles/2016-06-14/sec-staff-recommends-
approving-iex-application-wsj-reports
16 https://iextrading.com/insights
17 https://iextrading.com/trading/market-data/


<!-- Page 25 -->

Trading Fundamentals 11
yearsandnowaccountsfor30–40%ofalltradedvolumeincertainmarketslikethe
US.ATS/Dark Poolvolumemakesuproughly30%ofthis USequitiesoff-exchange
volume. Clearly, thegrowthofdarkexecutionhasspurredalotofcompetitioninthe
market. Additionally, theclaimofreducedinformationleakageisprobablyoverstated
asitisstillpotentiallypossibletoidentifylargeblocksofliquidityby“pinging”the
pool at minimum lot size. In order to counteract this effect orders are usually sent
withaminimumfillquantitytagwhichallowstheblocktobetransparentfromsmall
pinging.
As of 2019, there are 33 different equity ATSs!18 All these venues compete on
pricing, availabilityofliquidity, systemperformance, andfunctionalitysuchashan-
dling of certain special order types, etc. Many of these ATS are run by the major
investmentbanksandtheyhavehistoricallydominatedthisspace. Asfarasoverall
liquiditygoes,10 darkpoolsaccountforabout75%ofall ATSvolume, andthetop
5 makeuproughly50%ofdarkliquidityinthe US(the UBSATSand Credit Suisse
Crossfinder are consistently at the top of the rankings19). Other venues were born
outofthefundamentaldesireforinvestmentfirmstotradedirectlywitheachother,
bypassing the intermediaries and thus reducing cost and information leakage. The
main problem encountered by these buy-side to buy-side pools is that many of the
trading strategies used by these firms tend to be highly correlated (e.g., two funds
trackingthesamebenchmark) andthustheliquidityisoftenonthesameside. There-
fore as a result, these venues have to find different approaches to leverage sell side
broker’sliquiditytosupplementtheirownsuchasaccessviaconditionalorders. BIDS
and Liquidnetarethebiggest ATSofthistype. BIDShadsignificantgrowthinrecent
yearswith Liquidnetlosingitsinitialdominance. Thisspaceisstillveryactivewith
newvenuescominguponaregularbasis.
Single Dealer Platform/Systematic Internalizers:Alargenumberoftradingfirms
inrecentyearsstartedprovidingdirectaccesstotheirinternalliquidity. Broker/Dealer
and other institutional clients connect to a Single Dealer Platform (SDP) directly.
These SDPs, alsocalled Systematic Internalizers, sendregular Indication Of Interest
(IOI) that are bespoke to a particular connection and the broker can respond when
thereisamatch. Thisapproachtotradingisalsogrowingfast. Because SDPsarenot
regulated ATS, theycanoffersomewhatuniqueproducts. Brokersthemselvesarenow
startingtoprovidetheirown SDPstoexposetheirinternalliquidity. Aquicklyevolv-
ing space that promises interesting innovations, but alas, it can also lead to further
complicationinanalreadycrowdedecosystem. Inthe US, theother70%ofnon-ATS
off-exchangevolumeiscomprisedof Retail Wholesalers, Market Makersand Single
Dealer Platforms, and Broker Dealers.
Auctions: Primary exchanges (exchanges where a particular instrument is listed)
beginandendthedaywitha (primary) auctionprocedure, thatleveragesspecialorder
18 http://www.finra.org/industry/equity-ats-firms
19 https://otctransparency.finra.org/otctransparency/Ats Data


<!-- Page 26 -->

12 Algorithmic Tradingand Quantitative Strategies
types to accumulate supply and demand and then run an algorithm that determines
thepricethatwouldatbestpairoffthemostvolume. The Closing Auctionisofpar-
ticular importance because many funds set their Net Asset Value (NAV) using the
officialclosingprice. Thisgenerallyleadstraderstotradeascloseaspossibletothe
closingtimeinordertooptimizethedualobjectiveofgettingthebestpricebutalso
notdeviatingtoomuchfromthecloseprice.
The Auction also represents an opportunity for active and passive investors to
exchangelargeamountsofshares (liquidity).Indexconstituentsgetupdatedonareg-
ular basis (additions, deletions, weight increase/decrease), and as they get updated,
passive investors need to update their holdings to reflect the optimal composition
of the benchmark that they track. In order to minimize the tracking error risk, that
updateneedstohappenclosetotheactualupdateoftheunderlyingbenchmark. Con-
sequently, most passive indexers tend to rebalance their portfolios on the same day
theunderlyingindexconstituentsareupdated, usingtheclosingauctionasareference
price, andthisresultsinsignificantflowsatthecloseauction. Theexplosivegrowthof
ETFs (Exchange Traded Funds) andotherpassivefundshaveexacerbatedthistrend
inrecentyears. Atthetimeofthiswriting, about10%ofthetotaldaily USvolumein
indexnamestradeattheclose. Recentmonthshavebroughtalotofmovementinthis
areawithbrokersand SDPstryingtoprovideuniquewaystoexposeinternalliquidity
markedfortheclosingauction.
Beyond the US: As previously mentioned the structure we presented above is not
uniquetothe US.Europeanand Asianexchangesthathadoperatedasasinglemar-
ketplace for longer than their US counterparts now offer a more diverse landscape.
The evolution of ATS/Dark Pools and other MTFs (Multilateral Trading Facilities)
havefollowedsuitbutinamoresubduedmanner. Tradinginthesemarketshasalways
been smaller and concentrated with fewer participants and there is not enough liq-
uiditytosupportalargenumberofvenues. Oftennewvenuescomeon-linebutare
quicklyabsorbedbyacompetitorwhentheyfailtomoveasignificantportionofthe
traded volume. Additional complexity arises with regulatory environments in these
variouscountrieslimitingcross-bordertrading. Asof December2018, fragmentation
in Europeanmarketsisstillquitelowerthaninthe US, with59%tradedonprimary
exchange,22%onlit MTF(thereareonly6, and4 ofthemtraderoughly98%ofthe
volume),6.5%ondark MTF(10 differentones), and6%tradedinsystematicinternal-
izers. In APAC, withtheexceptionof Australia, Hong Kongand Japan, mostcountries
onlyhaveoneprimaryexchangewherealltransactionstakeplace. Eveninthemore
developed market, Japan for instance, the Tokyo Stock Exchange still garners over
85%ofthetotalvolumetraded.
Summary: Modern market structure may appear to be a jumbled mess. Yes, it is.
Fully understanding the implications of these different methods of trading tied by
regulation, competition, behavioralidiosyncrasies (attimesduetoparticipantslack
ofunderstandingandpreconceivednotions) isadauntingtask. Itishoweveranenvi-
ronment that all practitioners have to navigate, and it remains difficult to formulate
theseissuesinmathematicalmodelsastheycouldbebasedonquestionableheuris-
tics to account for the residual complexity. But even with the dizzying complexity,


<!-- Page 27 -->

Trading Fundamentals 13
modernmarketstructureisafascinatingecosystem. Itisacontinuouslyevolvingsys-
temthroughtheforcesofingenuityandcompetition. Itprovidestoolsandservicesto
institutionalinvestorswhostrivetoreducethecostofexecution, aninvestmentman-
date. Wehopetheabovetreatmentprovidesthereaderwithatleastafootholdinthe
explorationofthisamazingsocialandfinancialexperimentation.
1.3 The Mechanicsof Trading
Inordertofullygraspthemaintopicsofthisbook, onerequiresatleastagood
understanding of the mechanics of trading. This process is somewhat complicated
andrequiressometechnicaldetailsandterminology. Inthissection, wewillstriveto
provideabriefbutfairlycompleteoverviewofthefundamentals. Thisshouldsuffice
forourpurposes. Foramorecompleteandthoroughtreatmentwereferthereadersto
theexistingliterature, mostnotably Harris (2003)[178].
1.3.1 How Double Auction Markets Work
Themostcommonapproachusedbymodernelectronicexchangescanbetermed,
as time/price priority, continuous double auction trading system. The term double
auction signifies that, unlike a common auction with one auctioneer dealing with
potential buyers, in this case there are multiple buyers and multiple sellers partic-
ipating in the process at the same time. These buyers and sellers interact with the
exchangebysendinginstructionselectronically, viaanetworkprotocoltoaspecial-
ized software and hardware infrastructure called: The Matching Engine. It has two
maincomponents:The Limit Order Bookandthe Matching Algorithm.
Limit Order Book (LOB):Itisacomplexdatastructurethatstoresallnon-executed
orderswithassociatedinstructions. Itishighlyspecializedsoastobeextremelyfast
to insert/update/delete orders and then able to sort them and to retrieve aggregated
information. Foranactivestock, the LOBcanbeupdatedandqueriedthousandsof
times every second, so it must be highly efficient and able to handle a high degree
of concurrency to ensure that the state is always correct. The LOB is comprised of
twocopiesofthecoredatastructure, onefor Buyordersandonefor Sellordersoften
referredtoasthetwo“sides”oftheorderbook. Thisstructureisthecoreabstraction
forallelectronicexchangesandsoitisveryimportanttounderstanditindetail.
The LOBsupportsthreebasicinstructions:Insert, cancel, andamend, withinsert
initiatinganeworder, cancelremovinganexistingorderfromthemarketandamend
modifying some of the parameters of the existing order. New orders must specify
“ordertype”andassociatedparametersnecessarytofullyencapsulatethetraderdeci-
sion. Wewillreviewordertypesinmoredetaillater, butwestartwiththetwomain
types: The limit order and the market order. The main difference between the two


<!-- Page 28 -->

14 Algorithmic Tradingand Quantitative Strategies
ordertypesisthatalimitorderhasapriceassociatedtoitwhileamarketorderdoes
not.
Accountingforlimitandmarketorders, thereareeighteventsatanygiventime,
fouroneitherside, thatcanalterthestateoftheorderbook:
• Limit Order Submission: A limit order is added to the queue at the specified
pricelevel.
• Limit Order Cancellation: An outstanding limit order is expired or canceled
andisthereforeremovedfromthe Limit Order Book (LOB).
• Limit Order Amendment:Anoutstandinglimitorderismodifiedbytheoriginal
sender (suchaschangingordersize).
• Execution:Buyandsellordersatappropriatepricesarepairedbythematching
algorithm (explainedbelow) intoabindingtransactionandareremovedfromthe
LOB
Matching Algorithm: This software component is responsible for interpreting the
variouseventstodetermineifanybuyandsellorderscanbematchedinanexecu-
tion. Whenmultipleorderscanbepairedthealgorithmusestheso-calledprice/time
prioritymeaningthatfirsttheorderwiththemostcompetitivepricesarematchedand
whenpricesareequaltheorderthatarrivedpriorischosen. Aswewillseeinlater
chaptersthisisonlyoneofthepossiblealgorithmsusedinpracticebutitisbyfarthe
mostcommon. Wewillgointomoredetailsinthenextsections.
The matching algorithm operates continuously throughout the trading hours. In
order to ensure an orderly start and end, this continuous session is usually comple-
mentedbyacoupleofdiscreteauctions. Thetradingdaygenerallystartswithanopen
auction, thenfollowedbythemaincontinuoussession, andendswithaclosingauc-
tion. Somemarketslike Japanalsohavealunchbreakwhichmightbeprecededbya
morningclosingauctionandfollowedbyanafternoonopeningauction. Wewillnow
discussthesemainmarketphasesinchronologicalorder.
1.3.2 The Open Auction
The Open Auction is only one type of call auction that is commonly held on
exchanges. Theterm“callauction”explainstheliquidity-aggregatingnatureofthis
event. Marketparticipantsare‘called’tosubmittheirquotestothemarketplacein
ordertodetermineamatchingpricethatwillmaximizetheamountofsharesthatcan
be transacted. To facilitate timely and orderly cross, auctions have strict order sub-
missionrules, includingspecifiedtimingforentries (see Table1.1) andinformation
disseminationtopreventwildpricefluctuationsandensurethattheprocessisefficient
forpricediscovery.
Mostexchangespublishorderimbalancethatexistsamongordersontheopening
orclosingbooks, alongwiththeindicativepriceandvolume. Forinstance, Nasdaq


<!-- Page 29 -->

Trading Fundamentals 15
Table1.1:Nasdaq Opening Cross
4:00 a.m. EST Extendedhourstradingandorderentrybegins.
9:25 a.m. EST Nasdaqentersquotesforparticipantswithnoopeninterest.
9:28 a.m. EST Disseminationoforderimbalanceinformationevery1 second.
Market-on-openordersmustbereceivedpriorto9:28 a.m.
9:30 a.m. EST Theopeningcrossoccurs.
publishes the following information20 between 9:28 a.m. EST and 9:30 a.m. EST,
every1 second, onitsmarketdatafeeds:
• Current Reference Price:Pricewithinthe Nasdaq Insideatwhichpairedshares
are maximized, the imbalance is minimized and the distance from the bid-ask
mid-pointisminimized, inthatorder.
• Near Indicative Clearing Price:Thecrossingpriceatwhichordersinthe Nas-
daqopening/closingbookandcontinuousbookwouldclearagainsteachother.
• Far Indicative Clearing Price:Thecrossingpriceatwhichordersinthe Nasdaq
opening/closingbookwouldclearagainsteachother.
• Numberof Paired Shares:Thenumberofon-openoron-closesharesthat Nas-
daqisabletopairoffatthecurrentreferenceprice.
• Imbalance Shares:Thenumberofopeningorclosingsharesthatwouldremain
unexecutedatthecurrentreferenceprice.
• Imbalance Side:Thesideoftheimbalance:B=buy-sideimbalance;S=sell-
sideimbalance;N=noimbalance;O=nomarketableon-openoron-closeorders.
In a double auction setup, the existence of multiple buyers and sellers requires
employing a matching algorithm to determine the actual opening price which we
will illustrate with a practical example. Table 1.2 gives an example of order book
submissionsforahypotheticalstock, whereordersarerankedbasedontheirarrival
time. Different exchanges around the world apply slightly different mechanisms to
theirauctions, butgenerallythefollowingrulesapplytomatchsupplyanddemand:
• Thecrossingpricemustmaximizethevolumetransacted.
• Ifseveralpricesresultinsimilarvolumetransacted, thecrossingpriceistheone
theclosestfromthelastprice.
• Thecrossingpriceisidenticalforallordersexecuted.
• Iftwoordersaresubmittedatthesameprice, theordersubmittedfirsthaspriority.
20 Source:Nasdaq Traderwebsite.


<!-- Page 30 -->

16 Algorithmic Tradingand Quantitative Strategies
Table1.2:Pre-Open Order Book Submissions
Timestamp Seq. Number Side Quantity Price
9:01:21 1 B 1500 12.10
9:02:36 2 S 1750 12.12
9:05:17 3 B 4500 12.17
9:06:22 4 S 1750 12.22
9:06:59 5 S 2500 12.11
9:07:33 6 B 1200 12.23
9:07:42 7 B 500 12.33
9:08:18 8 B 500 12.25
9:09:54 9 S 1930 12.30
9:09:55 10 B 1000 12.21
9:10:04 11 S 3500 12.05
9:10:39 12 B 2000 12.34
9:11:13 13 S 4750 12.25
9:11:46 14 B 2750 12.19
9:12:21 15 S 10000 12.33
9:12:48 16 B 3000 12.28
9:13:12 17 B 5500 12.35
9:14:51 18 S 1800 12.18
9:15:02 19 B 800 12.17
9:15:37 20 S 1200 12.19
9:16:42 21 S 5000 12.16
9:17:11 22 B 12500 12.15
9:18:27 23 S 450 12.23
9:19:13 24 S 3500 12.20
9:19:54 25 B 1120 12.16
• Itispossibleforanordertobepartiallyexecutediftheothersidequantityisnot
sufficient.
• “At Market” orders are executed against each other at the determined crossing
price, up to the available matching quantity on both sides, but generally do not
participateinthepriceformationprocess.
• Forthe Open Auction, unmatched“At Market”ordersareenteredintothecon-
tinuoussessionof LOBaslimitordersatthecrossingprice.
The first step is to organize orders by limit price, segregating buys and sells as
shownin Table1.3.Abuyordersubmittedwithalimitpriceof12.25 representsan
intenttoexecuteatanypricelowerorequalto12.25.Similarly, asellordersubmit-
ted at 12.25 represents an intent to sell at any price higher or equal to 12.25. For
eachpricelevel, wecanthendeterminethecumulativebuyinterestandsellinterest.
Thetheoreticalcrossquantityateachpricepointisthensimplytheminimumofthe
cumulativebuyinterestandthecumulativesellinterestasshownin Table1.4.The


<!-- Page 31 -->

Trading Fundamentals 17
Table1.3:Ranked Order Book Submissions
Timestamp Seq. Number Buy Price Buy Quantity Sell Quantity Sell Price
9:13:12 17 12.35 5500
9:10:39 12 12.34 2000
9:07:42 7 12.33 500
9:12:21 15 10000 12.33
9:09:54 9 1930 12.30
9:12:48 16 12.28 3000
9:08:18 8 12.25 500
9:11:13 13 4750 12.25
9:07:33 6 12.23 1200
9:18:27 23 450 12.23
9:06:22 4 1750 12.22
9:09:55 10 12.21 1000
9:19:13 24 3500 12.20
9:11:46 14 12.19 2750
9:15:37 20 1200 12.19
9:14:51 18 1800 12.18
9:05:17 3 12.17 4500
9:15:02 19 12.17 800
9:16:42 21 5000 12.16
9:19:54 25 12.16 1120
9:17:11 22 12.15 12500
9:02:36 2 1750 12.12
9:06:59 5 2500 12.11
9:01:21 1 12.10 1500
9:10:04 11 3500 12.05
crossingpriceisdeterminedasthepricethatwouldmaximizethecrossedquantity. In
ourexample, theopeningpricewillbe12.19, andtheopeningquantitywillbe15,750
shares.
Thelistofbuyordersexecutedduringtheauctionisshownin Table1.5 andthe
sell orders in Table 1.6. It is worth mentioning that the open auction tends to be
consideredasamajorpricediscoverymechanismgiventhefactthatitoccursaftera
periodofmarketinactivitywhenmarketparticipantswereunabletotransactevenif
theyhaveinformation. Allnewinformationaccumulatedovernightwillbereflected
inthefirstprintoftheday, matchingbuyingandsellinginterests.
Asmarketparticipantswithbetterinformationaremorelikelytobeparticipating
in the open auction with more aggressive orders in order to extract liquidity (and,
assuch, settingtheprice), thepricediscoverymechanismisoftenconsideredtobe
quite volatile and more suited for short-term alpha investors. Similarly, the period
immediatelyfollowingtheopenauctionalsotendstobemuchmorevolatilethanthe
rest of the day. As a result of which, most markets experience wider spreads while


<!-- Page 32 -->

18 Algorithmic Tradingand Quantitative Strategies
Table1.4:Cumulative Order Book Quantities
Sequence Cumulative Buy Sell Cumulative Quantity Crossed
Price
Number Buy Quantity Quantity Quantity Sell Quantity at Price
12.35 17 5500 5500 38130 5500
12.34 12 7500 2000 38130 7500
12.33 7 8000 500 38130 8000
12.33 15 8000 10000 38130 8000
12.30 9 8000 1930 28130 8000
12.28 16 11000 3000 26200 11000
12.25 8 11500 500 26200 11500
12.25 13 11500 4750 26200 11500
12.23 6 12700 1200 21450 12700
12.23 23 12700 450 21450 12700
12.22 4 12700 1750 21000 12700
12.21 10 13700 1000 19250 13700
12.20 24 13700 3500 19250 13700
12.19∗ 14 16450 2750 15750 15750
12.19 20 16450 1200 15750 15750
12.18 18 16450 1800 14550 14550
12.17 3 20950 4500 12750 12750
12.17 19 21750 800 12750 12750
12.16 21 21750 5000 12750 12750
12.16 25 22870 1120 7750 7750
12.15 22 35370 12500 7750 7750
12.12 2 35370 1750 7750 7750
12.11 5 35370 2500 6000 6000
12.10 1 36870 1500 3500 3500
12.05 11 36870 3500 3500 3500
marketmakerstrytoprotectthemselvesagainstinformationasymmetrybyquoting
wider bids and offers. The increased volatility and wider spreads might discourage
certaininvestorsfromparticipatinginthemarketattheopenauctionandintheperiod
immediatelyfollowingtheopen. Whilethisappearstobereasonablefromapricerisk
perspective, itisworthmentioningthatformanylessliquidstocks (inparticularsmall
andmidcapstocks), theopenauctioncanbeasignificantliquidityaggregationpoint
thatevensurpassesthecloseauction. In Australiaforinstance, thebottom50%less
liquidstockshavemorevolumetradedintheopenauctionthaninthecloseauction.
Similarlyin Japan, thelessliquidstockshavemorevolumetradedintheopenauction,
butalsointheafternoonopenauctionthatfollowsthemarketlunchbreak.
Fromanexecutionstandpoint, though, theusageoftheopenauctionhastobecon-
sidered carefully. While this represents a liquidity opportunity, the first print of the
daycanalsohaveasignificantanchoringeffectonthestockpricefortheremainder
oftheday. So, participatingintheopenshouldbeconsideredinlightoftheliquidity


<!-- Page 33 -->

Trading Fundamentals 19
Table1.5:Crossed Buy Orders
Price Seq. Number Buy Qty
12.35 17 5500
12.34 12 2000
12.33 7 500
12.28 16 3000
12.25 8 500
12.23 6 1200
12.21 10 1000
12.19∗ 14 2050
*Ordernumber14 wasfor2750 sharesbutdidnotgetfullyexecuted
as the bid quantity up to 12.19 exceeded the offered quantity at that
price. Thebalanceoforder14 willthenbepostedasalimitorderin
thecontinuoustradingsession
Table1.6:Crossed Sell Orders
Price Seq. Number Sell Qty
12.19 20 1200
12.18 18 1800
12.16 21 5000
12.12 2 1750
12.11 5 2500
12.05 11 3500
demandoftheorder:Ordersthataresmallenoughcanlikelydowithoutparticipating
intheopenauctionandtheperiodthatcontinuesimmediatelyfollowing, whilelarge
ordersthattrytoextractsignificantliquidityfromthemarketmightbenefitfrompar-
ticipating in the open auction. The market intraday momentum study by Gao, Han,
Liand Zhou (2018)[156]demonstrateshowthefirsthalf-hourreturnonthemarket,
asmeasuredfromthepreviousday’smarketclosepredictsthelasthalfhourreturn.
1.3.3 Continuous Trading
This refers to the main market phase between the auctions. During this market
sessionthestateoftheorderbookchangesquiterapidlyduetothemulti-agentnature
of financial markets and the prevalence of high frequency trading. Consequently, it
is important to understand the dynamics of the LOB before implementing trading
strategies. Thereexistsquiteadiversityofordertypesthataremostlyrelevanttothe
continuoustradingsession, butthetwomostbasiconesare:Limit Ordersand Market
Orders, whichwedescribebelow.
Alimitorderhasanassociatedside (Buyor Sell), aquantityandapricewhich
representthehighest (lowest) pricethetraderiswillingtobuy (sell).Aspreviously
discussed, oncealimitorderisreceivedbytheexchangeitisinsertedinadatastruc-
ture called a Limit Order Book (LOB) which contains two sub-structures, one per
side. Orders are inserted in this structure in price priority, higher prices for buys,


<!-- Page 34 -->

20 Algorithmic Tradingand Quantitative Strategies
lower prices for sells, and for orders at the same price the orders are stored in the
orderinwhichtheywerereceived. Thatiswhatismeantbyprice/timepriority.21 If
thepriceofanewlyarrivedorderoverlapswiththebestpriceavailableontheoppo-
siteside, theorderisexecutedeitherfullyoruptotheavailablequantityontheother
side. Theseordersaresaidtobe“matched”andagainthismatchinghappensinprice
andtimeprioritymeaningthatthebetterprices (higherforbuys, lowerforsells) are
executedfirstandordersthatarrivedbeforehandatthesamepricelevelareexecuted
first. Marketordersontheotherhanddonothaveapriceassociatedwiththemand
willimmediatelyexecuteagainsttheothersideandwillmatchwithmoreandmore
aggressivepricesuntilthefullorderisexecuted.
Orders on the buy side are called “bids” while those on the sell side are called
“asks.”Theaboveeventsareillustratedin Figure1.1 to Figure1.3.Whenamarket
(ormarketable) orderissubmitted, itdecreasesthenumberofoutstandingordersat
the opposite best price. For example, if a market bid order arrives, it will decrease
the number of outstanding asks at the best price. All unexecuted limit orders can
becanceled. Whenacancellationoccurs, itwilldecreasethenumberofoutstanding
ordersatthespecifiedpricelevel.
emulov
bids
60
asks
50
40
limitbid
30
20
10
0
73600 73800 74000 74200 74400 74600 74800 75000
price
Figure1.1:Limit Order Book—Limit Bid.
21 Note:Notallexchangesarematchingordersfollowingaprice/timepriorityalgorithm;akeychar-
acteristicofthe Futuresmarket, forinstance, istheexistenceofpro-ratamarketsforsomefixedincome
contracts, wherepassivechildordersreceivefillsfromaggressiveordersbasedontheirsizeasafraction
ofthetotalpassivepostedquantity.


<!-- Page 35 -->

Trading Fundamentals 21
emulov
bids
60
asks
50
40
30
20
marketbid
10
0
73600 73800 74000 74200 74400 74600 74800 75000
price
Figure1.2:Limit Order Book—Marketable Bid.
emulov
bids
60
asks askcancellation
50
40
30
20
10
0
73600 73800 74000 74200 74400 74600 74800 75000
price
Figure1.3:Limit Order Book—Ask Cancellation.


<!-- Page 36 -->

22 Algorithmic Tradingand Quantitative Strategies
Limitordersmakeupasignificantpercentage (70%) ofstockmarkettradingactiv-
ity. Themainadvantageofalimitorderisthatthereisnopriceriskassociatedtoit,
thatis, whentheorderisexecutedthelimitpriceisthemaximum (forabuyorder) or
minimum (forasellorder) pricethatwillbeachieved. Butifthelimitorderisnotmar-
ketable, theexecutionisnotguaranteedandthetimetogetanorderexecuteddepends
onvariousmarketfactors. Thetrade-offbetweenlimitordersandmarketableorders
dependsontheinvestor’sneedforimmediateliquidityandthefillprobabilityoflimit
orders. Thelimitpricechosen (howdeepintheorderbookistheorderplaced) aswell
astheamountofliquidityaheadofthesubmittedorder (howmanyshareswillneed
to tradebefore theorder getsexecuted following, forinstance, aprice/time priority
ordermatchingoftheexchange) affectboththeorderfillprobabilityanditsexpected
time to fill. These two metrics are of particular relevance for execution algorithms
andwillbestudiedinmoredepthlater.
The execution of limit orders does affect how the quotes are posted and are
updated. Ifthesizeofamarketorderexceedsthenumberofsharesavailableatthe
topofbook, itisusuallysplitandisexecutedatconsecutiveorderbooklevelsuntil
theorderisfilled. Marketordersareusuallyrestrictedtobe filledwithinadayand
ordersplacedafterthemarketsclosemightbeenteredthenextday.22
Order Types:Thediversityofordertypesisakeycomponentofcontinuousdouble
auctionelectronicmarkets. Ordertypesallowparticipantstoexpresspreciselytheir
intentionswithregardstotheirinteractionwiththemarketviathelimitorderbook.
Overtime, inanefforttocatertosophisticatedelectronictraders, exchangesaround
theworldhaveracedtoofferevermorecomplexordertypes. Here, wewilljustpro-
videabriefdescriptionofsome, besidesthemarketandlimitordersthatwerealready
mentioned:
• Peg:Specifyapricelevelatwhichtheordershouldbecontinuouslyandautomati-
callyrepriced. Forinstance, anorderpeggedtothebidpricewillbeautomatically
repricedasahigherpricelimitorder, eachtimethemarketbidpriceticksup. This
ordertypeisparticularlyusedformid-pointexecutionsinnon-displayedmarkets.
Onewouldthinkthatpeggingorderhastheadditionaladvantageofimprovingthe
queuepriorityoftheorderwhenitisrepricedsincethisprocessisdonedirectly
bytheexchange. Thatturnsoutnottobealwaystrue. Managingpegordersisthe
responsibilityofaseparatecomponentattheexchangeanditsinteractionspeed
withtheorderbookisusuallyslowerthanthatofultra-low-latencyoperators.
• Iceberg:Limitorderwithaspecifieddisplayquantity. Inordertopreventinfor-
mationleakagetoothermarketparticipants, atraderdesiringtobuyorsellalarge
quantity at a given price might elect to use an iceberg order with a small dis-
playsize. Forinstance, foranordertobuy100,000 sharesat$20 withadisplay
size of 2,000 shares. Only 2,000 shares would be displayed in the order book.
Once that quantity is executed, the order would automatically reload another
22 Dependingonthe Time-in-Forceselected.


<!-- Page 37 -->

Trading Fundamentals 23
2,000 shares at $20, and so on, until the full quantity is executed. Note that for
iceberg orders, only the visible quantity has time priority and once that quan-
tity has been executed the new tip of the iceberg will be placed at the back of
thequeue.
• Hidden:Whiletheyareavailabletotrade, theseordersarenotdirectlyvisibleto
othermarketparticipantsinthecentrallimitorderbook.
• Stop: These orders are also not visible, but additionally are not immediately
entered in the limit order book. They only become active once a certain price
(knownasthe Stop Price) isreachedorpassed. Theythenentertheorderbook
aseitherlimitormarketorderdependingontheusersetup.
• Trailing Stop: These orders function like stop orders, but the stop price is set
dynamicratherthanstatic (forinstance:−3%frompreviousclose).
• All-or-None:Specifically, requestafullexecutionoftheorder. Iftheorderisfor
1500 sharesbutonly1000 arebeingoffered, itwillnotbeexecuteduntilthefull
quantityisavailable.
• On-Open:Specifically, requestanexecutionattheopenprice. Itcanbelimit-on-
openormarket-on-open.
• On-Close: Specifically, request an execution at the close price. It can be limit-
on-closeormarket-on-close.
• Imbalance Only:Provideliquidityintendedtooffseton-open/on-closeorder/im-
balancesduringtheopening/closingcross. Thesegenerallyarelimitorders.
• D-Quote:Specialordertypeonthe NYSEmainlyusedduringthecloseauction
period.
• Funari: Special order type on the Tokyo Stock Exchange which allows limit
orders placed in the book during the continuous session to automatically enter
theclosingauctionasmarketorders.
Asdescribedabove, thereexistsawidevarietyoforderstypesofferedbydifferent
exchangestofacilitatevarioustypesoftradingactivities.
Validityof Instructions:Inadditiontoconditionsonprice, itispossibletoaddcon-
ditions on the life duration of the order known as Time-in-Force (TIF). The most
common types of TIF instructions include Day orders which are valid for the full
durationofthetradingsession, Extended Dayordersallowtradinginextendedhours,
and Good-Till-Cancel (GTC) orders will be placed again on the exchange the next
day with similar instructions if they were not completely filled. More sophisticated
marketparticipantsaimingatachievinggreatercontrolovertheirexecutionstendto


<!-- Page 38 -->

24 Algorithmic Tradingand Quantitative Strategies
alsofavor Immediate-or-Cancel (IOC) and Fill-or-Kill (FOK)Time-in-Forceinstruc-
tions. An IOCorderwillgetimmediatelycanceledbacktothesenderafterreaching
thematchingengineifitdoesnotgetanimmediatefill, andincaseofapartialfill, the
unfilledportionwillbecanceled, thuspreventingitfromcreatinganewpricelevelin
theorderbook. Ina Fill-or-Killscenario, theordergetseitherfilledinitsentiretyor
doesnotgetfilledatall. Thisinstructionisparticularlypopularwithhighfrequency
marketmakersandarbitrageursforwhichpartialfillsmightresultinunwantedleg-
gingriskasdiscussedin Chapter5 onpairstrading.
Finally, itisworthmentioningthatsomeexchangesaswellasalternativevenues
offertheabilityofspecifyingminimumfillsizes. Thismeansthatalimitorderwhich
might be eligible for a fill due to an incoming order at the same price level, only
receives a fill if the incoming order is larger than a pre-specified number of shares
ornotionalvalue. Thistypeofinstructionisusedbymarketparticipantsasawayof
minimizing the number of small fills which carry the risk of excessive information
dissemination. Thishappens, inparticular, indarkpools, wheretheycanbeusedto
detectthepresenceoflargerlimitordersthatwouldbeotherwisenotvisibletomarket
participants.
1.3.4 The Closing Auction
The Closing Auctiontendstobethemostpopularcallauctionforavarietyofrea-
sons. First, itisthelastopportunity (unlessoneengagesintheriskypracticeofoff-
hourstrading) formarketparticipantstotransactinarelativelyliquidenvironment,
before being exposed to the overnight period (when new information accumulates,
buttradescannoteasilytakeplace).Second, withtheincreaseinpassiveinvestment
strategies, providinginvestorswithreplicationofapredeterminedbenchmarkindex,
theclosingauctionhasbecomeaparticularlyrelevantpricesettingevent. Formost
passive funds, the net asset value (NAV) is based on close prices of the underly-
ingassets. Forthosereasons, theclosingauctionhasbecomeextremelyimportantto
manyinvestors. Fromanexecutionstandpoint, itisamajorliquidityeventthatmust
behandledcarefully.
Themechanicsoftheclosingauctionareinmostpartsimilartotheonesdescribed
above for the open auction. The major differences across countries (and sometimes
across exchanges within a country) are in the order submission times. Some coun-
tries, such as the US, have order submissions start and end before the continuous
session is over (see Table 1.7 and Table 1.8), while some other markets have two
non-overlappingcontinuousandcloseordersubmissionsessions.
Table1.7:Nasdaq Closing Cross
3:50 p.m. EST Cutoffforamend/cancelof MOC/LOCorders
3:55 p.m. EST Disseminationofimbalanceinformationbegins
3:55 p.m. EST Cutoffforentryof MOC/LOCorders
3:58 p.m. EST Freezeperiod-Late LOCorderscannotbeadded
OIordersoffsettingtheimbalancearestillaccepted
4:00 p.m. EST The Closing Crossoccurs


<!-- Page 39 -->

Trading Fundamentals 25
Table1.8:NYSEClosing Cross
3:50 p.m. EST Cutofffor MOC/LOCorderentryandmodifications
Disseminationofimbalanceinformationbegins
Closing Offsetorderscanbeentereduntil4:00 p.m.
3:55 p.m. EST Disseminationofd-quoteimbalanceinformation
3:58 p.m. EST Cutofffor MOC/LOCcancellationforlegitimateerror
3:59:50 p.m. EST Cutoffford-Quoteorderentryandmodification
4:00 p.m. EST The Closing Auctionstarts
4:02 p.m. EST DMMcanautomaticallyprocessauctionsnotyetcomplete
Table1.9:London Stock Exchange Sessions Times
7:00-7:50 a.m. GMT Pre-Trading
7:50-8:00*a.m. GMT Opening Auction Call
8:00-12:00 p.m. GMT Regular Trading
12:00-12:02*p.m. GMT Periodic Call Auction
12:02-4:30 p.m. GMT Regular Trading
4:30-4:35*p.m. GMT Closing Auction Call
4:35-4:40 p.m. GMT Closing Price Crossing
4:40-5:15 p.m. GMT Post-Close Trading
*Eachauctionendtimeissubjecttoarandom30 seconduncrossperiod.
Anadditionalintradayauctioncalltakesplaceevery3 rd Fridayofeachmonthforstocksunder-
lying FTSE100 indexoptions, andthe3 rd Fridayofeveryquarterforstocksunderlying FTSE
100/250 Index Futurestodeterminethe EDSP(Exchange Delivery Settlement Price).Theset-
tlementpriceisdeterminedastheindexvaluederivedfromtheindividualconstituentsintraday
auctiontakingplacebetween10 a.m.and10:15 a.m.andduringwhichtheelectroniccontinu-
oustradingissuspended. Usingacallauctionensuresthesettlementpriceforthesecontracts
ismorerepresentativeofafairmarketprice.
An aspect of the NYSE is the presence of floor brokers operating in an agency
capacity for their customers. They play a particular role during the close auction
thankstotheirabilitytohandlediscretionaryelectronicquoteorders (knownas“d-
Quote”orders) thatoffermoreflexibilitythantraditionalmarket-on-close (MOC) and
limit-on-close (LOC) orders. The main advantage of d-Quote orders is their ability
tobypassthe3:45 p.m.cutoffandbesubmittedorcanceleduntil3:59:50 p.m. This
allowslargeinstitutionalinvestorstoremainincontroloftheirordersalmostuntilthe
end of the continuous session, by delaying the decision of how much to allocate to
theauction. Theycanthereforereacttolargervolumeopportunitiesbasedonthepub-
lishedimbalance. Theycanalsominimizetheinformationleakagebynotbeingpart
oftheearlypublishedimbalancewhilethereisstillsignificanttimeinthecontinuous
sessionforotherparticipantstodrivethepriceaway. Sincethereisnorestrictionon
thesideofd-Quoteorderssubmission, itispossibletoseethetotalimbalancesignflip
once the d-Quotes are added to the publication at 3:55 p.m., creating opportunities


<!-- Page 40 -->

26 Algorithmic Tradingand Quantitative Strategies
forothermarketparticipantstoadjusttheirownclosetradingviad-Quotesorchange
theirpositioninginthecontinuoussession.
1.4 Taxonomyof Data Usedin Algorithmic Trading
Runningasuccessfultradingoperationrequiresavailabilityofdifferentdatasets.
Dataavailability, storage, management, andcleaningaresomeofthemostimportant
aspectsofafunctioningtradingbusinessandthecoreofanyresearchenvironment.
The amount of data and the complexity of maintaining such a “Data Lake” can be
daunting and very few excel at this aspect. In this section, we will review the most
importantdatasetsandtheirroleinalgorithmictradingresearch.
1.4.1 Reference Data
Whileoftenoverlooked, ormerelyconsideredasanafterthoughtinthedevelop-
mentofaresearchplatform,23 reliablereferencedataisthekeyfoundationofarobust
quantitativestrategydevelopment. Theexperiencedpractitionermaywanttoskipthis
section, howeverweencouragetheneophytetoreadthroughthetediousdetailstoget
abettergraspofthecomplexityathand.
• Trading Universe:Thefirstproblemforthefunctioningofatradingoperation
is knowing what instruments will be required to be traded on a particular day.
Thetradinguniverseisanevolvingentitythatchangesdailytoincorporatenew
listings (IPOs), de-listings, etc. Tobeabletojusttradenewinstruments, thereare
severalpiecesofinformationthatarerequiredtohaveinmultiplesystems. Market
Datamustbemadeavailable, andsomestaticdataneedstobesetorguessedto
workwithexistingcontrols, parametersforthevariousanalyticsneedtobemade
availableorsensiblydefaulted. Forresearch, inparticularquantitativestrategies,
knowingwhenaparticularstocknolongertradesisimportanttoavoidissueslike
survivorbias.
• Symbology Mapping:ISIN, SEDOL, RIC, Bloomberg Tickers,...Quantitative
strategiesoftenleveragedatafromavarietyofsources. Differentproviderskey
theirdatawithdifferentinstrumentidentifiersdependingonassetclassorregional
conventions, or sometimes use their own proprietary identifiers (e.g., Reuters
Identification Code—RIC, Bloomberg Ticker). Therefore, symbology mapping
isthefirststepinanydatamergingexercise. Suchdataisnotstatic. Oneofthe
symbolscanchangeonagivendayandothersremainunchangedforsometime,
complicatinghistoricaldatamerges.
23 Moredetailsonresearchplatformsarepresentedin Chapter12.


<!-- Page 41 -->

Trading Fundamentals 27
It is important to note that such mapping needs to persist as point-in-time data
and allow for historical “as of date” usage, requiring the implementation of a
bi-temporal data structure. Over the course of time, some instruments undergo
ticker changes (for example, from ABC to DEF on a later 𝑇 ) not necessarily
0
withoutanyparticularchangeontheunderlyingasset. Insuchcases, marketdata
recordedday-by-dayinatradedatabasewillchangefrombeingkeyedon ABC
tobeingkeyedon DEFafterthetickerchangedate𝑇 .Thishasimplicationsfor
0
practitioners working on data sets to build and backtest quantitative strategies.
Thesymbologymappingshouldallowforbothbackwardandforwardhandling
ofthechanges.
Forinstance, inthesimpleexamplementionedbelow, inordertoefficientlyback-
teststrategiesoveraperiodoftimespanning𝑇 , arobustmappingisneeded, so
0
thatitwillallowtoseamlesslyquerythedatafortheunderlyingassetinavariety
ofscenariossuchas:
– Signalgeneration:30-daybackwardclosetimeseriesasofdate𝑇 <𝑇 :
0
select close from data where date in [T-30, T], sym = ABC
– Signalgeneration:30-daybackwardclosetimeseriesasofdate𝑇 =𝑇 +10:
0
select close from data where date in [T -20, T +10], sym = DEF
0 0
– Positionholding:30-dayforwardclosetimeseriesasofdate𝑇 =𝑇 −10:
0
select close from data where date in [T -10, T +20], sym = ABC
0 0
• Ticker Changes: For comparable reasons as the ones described above in the
Symbology Mapping section, one needs to maintain a historical table of ticker
changesallowingtoseamlesslygoupanddowntimeseriesdata.
• Corporate Actions Calendars:Thiscategorycontainsstockandcashdividends
(bothannouncementdateandexecutiondate), stocksplits, reversesplits, rights
offer, mergersandacquisitions, spinoff, freefloatorsharesoutstandingadjust-
ments, quotationsuspension, etc.
Corporateactionsimpactthecontinuityofpriceandvolumetimeseriesand, as
such, mustberecordedinordertoproduceadjustedtimeseries. Themostcom-
moneventsaredividenddistributions. Onthedaythedividendispaid, thecor-
respondingamountisremovedfromthestockprice, creatingajumpintheprice
time series. The announcement date might also coincide with the stock experi-
encing more volatility as investors react to the news. Consequently, recording
theseeventsprovestobevaluableinthedesignofquantitativestrategiesasone
canassesstheeffectofdividendsannouncementorpaymentonperformance, and
decidetoeithernotholdasecuritythathasanupcomingdividendannouncement
or, conversely, tobuildstrategiesthatlooktobenefitfromtheaddedvolatility.
Anothertypeofcorporateeventsgeneratingdiscontinuityinhistoricaltimeseries
are stock splits or reverse splits and right offers. When the price of a stock
becomes too low or too high, a company may seek to split it to bring the price


<!-- Page 42 -->

28 Algorithmic Tradingand Quantitative Strategies
backtoalevelthatismoreconducivetoliquidtradingonexchanges.24 Whena
stockexperiencesa2:1 split, everythingelsebeingequal, itspricewillbehalved
andhence, itsvolumewilldouble. Inordertopreventthetimeseriesfromshow-
ingadiscontinuity, allhistoricaldatawillthenneedtobeadjustedbackwardto
reflectthesplit.
Mergers&Acquisitionsand Spin-offsarealsoregulareventsinthelifecycleof
corporations. Theirhistoryneedstoberecordedinordertoaccountfortheresult-
ingchangesinvaluationthatmightaffectagiventicker (s).Thesesituationscan
alsobeexploitedbytradingstrategiesknownas Merger Arbitrage.
Stockquotationscanbesuspendedasacoolingmechanism (oftenattherequest
of the underlying company) to prevent excess price volatility when significant
informationisabouttobereleasedtothemarket. Dependingonthecircumstance,
thesuspensioncanbetemporaryandintraday, orcanlastforextendedperiodsof
time if the market place allows it.25 Suspensions result in gaps in data and are
worthkeepingtrackof, astheycanimpactstrategiesinbacktesting (inabilityto
enterorexitaposition, uncertaintyinthepricingofcompositeassetsifagiven
stockhasasignificantweightin ETFsor Indexes, etc.).Somemarketswillalso
suspendtradingifthepriceswingsmorethanapredefinedamount (limitup/limit
down situations), either for a period of time or for the remainder of the trading
session.
• Static Data:Country, sector, primaryexchange, currencyandquotefactor. Static
dataisalsorelevantforthedevelopmentofquantitativetradingstrategies. Inpar-
ticular, country, currency and sector are useful to group instruments based on
theirfundamentalsimilarities. Awellknownexampleistheusageofsectorsto
groupstocksinordertocreatepairstradingstrategies. Itisworthnotingthatthere
existdifferenttypesofsectorclassifications (e.g., GICS®from S&P, ICB®from
FTSE) offering several levels of granularity,26 and that different classifications
mightbebettersuitedtodifferentassetclassesorcountries. Theconstituentsof
the Japanese index TOPIX, for instance, are classified into 33 sectors that are
thoughttobetterreflectthefundamentalstructureofthe Japaneseeconomyand
theexistenceoflargediversifiedconglomerates.
Maintaining a table of the quotation currency per instrument is also necessary
in order to aggregate positions at a portfolio level. Some exchanges allow the
quotationofpricesincurrenciesdifferentfromtheoneofthecountryinwhich
24 Averylowpricecreatestradingfrictionsastheminimumpriceincrementmightrepresentalargecost
relativetothestockprice. Averyhighpricemightalsodeterretailinvestorsfrominvestingintoasecurity
asitrequiresthemtodeploytoomuchcapitalperunit.
25 Forinstance, itwasthecaseforalargenumberofcompaniesin Chinain2016.
26 The Global Industry Classification Standard (GICS®) structureconsistsof11 sectors,24 industry
groups,68 industriesand157 sub-industries. Anexampleofthishierarchicalstructurewouldbe:Industrials
/Capital Goods/Machinery/Agricultural&Farm Machinery.


<!-- Page 43 -->

Trading Fundamentals 29
the exchange is located.27 Additionally, thus, the Quote Factor associated with
the quotation currency data needs to be stored. To account for the wide range
of currency values and preserve pricing precision, market data providers may
publish FXrateswithafactorof100 or1000.Hence, toconvertpricesto USDone
needstomultiplybythequotefactor:USDprice=localprice⋅fx⋅quotefactor.
Similarly, some exchanges quote prices in cents, and the associated quotation
currencyisreflectedwithasmallcapletter:GBP/GBp, ZAR/ZAr, ILS/ILs, etc.
• Exchange Specific Data: Despite the electronification of markets, individual
exchanges present a variety of differences that need to be accounted for when
designing trading strategies. The first group of information concerns the hours
anddatesofoperation:
– Holiday Calendar:Asnotallexchangesareclosedonthesameday, and
tradingdaystheyareoffdonotalwaysfullyfollowthecountry’spublicholi-
days, itisvaluabletorecordthem, inparticularintheinternationalcontext.
Strategies trading simultaneously in several markets and leveraging their
correlation, may not perform as expected if one of the markets is closed
whileothersareopen. Similarly, executionstrategiesinonemarketmight
beimpactedbytheabsenceoftradinginanothermarket (forinstance, Euro-
peanequitymarketsvolumetendstobe30%to40%lowerduring USmarket
holidays).
– Exchange Sessions Hours: These seemingly trivial data points can get
quite complex on a global scale. What are the different available sessions
(Pre-Market session, Continuous core session, After-Hour session, etc.)?
Whataretheauctiontimesaswellastheirrespectivecutofftimesfororder
submission?Istherealunchbreakrestrictingintradaytrading?Andifso,
arethereauctionsbeforeandafterthelunchbreak?Thetradingsessionsin
thefuturesmarketscanalsobequitecomplexwithmultiplephases, breaks,
aswellasofficialsettlementtimesthatmaydifferfromtheclosingtimeand
haveaneffectonliquidity.
In Indonesia, forinstance, marketshavedifferenttradinghourson Fridays.
Monday through Thursday the Indonesia Stock Exchange (IDX) is open
from 9:00 a.m. to 12:00 p.m., and then, from 1:30 p.m. to 4:00 p.m. On
Fridays, however, the lunch break is one hour longer and stretches from
11:30 a.m. to 2:00 p.m. This weekday effect is particularly important to
considerwhenbuildingvolumeprofilesasdiscussedin Chapter5.
Along with local times of operation, it is necessary to consider eventual
Daylight Saving Time (DST) adjustmentsthatmightaffecttherelativetrad-
inghoursofdifferentmarkets (somecountriesdonothave DSTadjustment
atall, whileforcountriesthatdohaveone, thedatesatwhichitappliesare
27 Forexample, Jardine Matheson Holdingsquotesin USDonthe Singaporeexchangewhilemostofthe
othersecuritiesquotein Singapore Dollars.


<!-- Page 44 -->

30 Algorithmic Tradingand Quantitative Strategies
notalwayscoordinated).Usually, USDSTstartsabouttwoweekspriorto
itsstartin Europe, bringingthetimedifferencebetween New Yorkand Lon-
dontofourhoursinsteadoffivehours. Thisresultsinthevolumespikein
Europeanequitiesassociatedtothe USmarketopenbeingonehourearlier,
requiringadjustmentofvolumeprofilesusedfortradingexecutions.
Some exchanges may also adjust the length of trading hours during the
course of the year. In Brazil for instance, the Bovespa continuous trading
hours are 10:00 a.m. to 5:55 p.m. from November to March, but an hour
shorter (10:00 a.m.to4:55 p.m.) from Aprilto Octobertobemoreconsis-
tentwith USmarkethours. Finally, withinonecountrytheremightalsoexist
differenttradinghoursbyvenuesasitisthecasein Japanwherethe Nagoya
Stock Exchangecloses30 minutesafterthemajor Tokyo Stock Exchange.
– Disrupted Days:Exchangeoutagesortradingdisruptions, aswellasmarket
dataissues, needtoberecordedsotheycanbefilteredoutwhenbuilding
ortestingstrategiesasthedifferenceinliquiditypatternsorthelackofdata
qualitymaylikelyimpacttheoveralloutcome.
Additionally, exchangesalsohavespecificrulesgoverningthemechanicsoftrad-
ing, suchas:
– Tick Size:Theminimumeligiblepriceincrement. Thiscanvarybyinstru-
ment, butalsochangedynamicallyasafunctionofthepriceoftheinstru-
ment (e.g., stocksunder$1 canquoteinincrementsof$0.0001, whileabove
thatpricetheminimumquoteincrementis$0.01).
– Tradeand Quote Lots:Similartoticksizes, certainexchangesrestrictthe
minimumsizeincrementforquotesortrades.
– Limit-Upand Limit-Down Constraints:Anumberofexchangesrestrict
themaximumdailyfluctuationsofsecurities. Usually, whensecuritiesreach
thesethresholds, theyeitherpausetradingorcanonlybetradedatabetter
pricethanthelimit-uplimit-downthreshold.
– Short Sell Restrictions: Some markets also impose execution level con-
straintsonshortsells (ontopofpotentiallocaterequirements).Forinstance,
whilealongsellordercantradeatanyprice, someexchangesrestrictshort
sellsnottotradeatapriceworsethanthelastpriceornottocreateanew
quotethatwouldbelowerthanthelowestprevailingquote. Theseconsider-
ationsareparticularlyimportanttokeepinmindforresearchersdeveloping
Long-Shortstrategiesasthisimpactstheabilitytosourceliquidity.
Becausethesevaluesandtheirpotentialactivationthresholdcanvaryovertime,
oneneedstomaintainhistoricvaluesaswell, inordertorunrealistichistorical
backtests.


<!-- Page 45 -->

Trading Fundamentals 31
• Market Data Condition Codes: With ever-growing complexity in market
microstructure, the dissemination of market data has grown complex as well.
While it is possible to store daily data as a single entry per day and per instru-
ment, investorsbuildingintradaystrategieslikelyneedtickbytickdataofallthe
events occurring in the market place. To help classify these events, exchanges
and market data providers attribute so-called condition codes to the trades and
quotestheypublish. Theseconditioncodesvaryperexchangeandperassetclass,
andeachmarketeventcanbeattributedtoseveralcodesatonce. So, toaggregate
intradaymarketdataproperlyandefficiently, anddecidewhicheventstokeepand
whichonestoexclude, itisnecessarytobuildamappingtableofthesecondition
codesandwhattheymean:Auctiontrade, litordarktrade, canceledorcorrected
trade, regulartrade, off-exchangetradereporting, block-sizetrade, tradeoriginat-
ingfromamulti-legordersuchasanoptionspreadtrade, etc.
Forinstance, inordertoassessaccessibleliquidityforatradingalgorithm, trades
thatarepublishedforreportingpurposes (e.g., negotiatedtransactionsthathap-
pened off-exchange) must be excluded. Thesetrades should alsonot beused to
updatesomeoftheaggregateddailydatausedintheconstructionoftradingstrate-
gies (dailyvolume, high, low,...).Executionalgorithmsalsoextensivelyleverage
thedistributionofintradayliquiditymetricstogaugetheirownparticipationin
auctionsandcontinuoussessions, orinlitversusdarkvenues, thereforerequiring
apreciseclassificationofintradaymarketdata.
• Special Day Calendars: Over the course of a year, some days present certain
distinct liquidity characteristicsthat need to be accounted forin both execution
strategiesandinthealphagenerationprocess. Thediversityofeventsacrossmar-
kets and asset classes can be quite challenging to handle. Among the irregular
events that affect liquidity in equity markets that need to be accounted for, we
canmentionthefollowingnon-exhaustivelistforillustrationpurposesonly:Half
tradingdayspreceding Christmasandfollowing Thanksgivinginthe USoronthe
Ramadanevein Turkey, Taiwanesemarketopeningontheweekendtomakeup
forlosttradingdaysduringholidayperiods, Koreanmarketchangingitstrading
hours on the day of the nationwide university entrance exam, Brazilian market
openinglateonthedayfollowingthe Carnival, etc.
There are also special days that are more regular and easier to handle. The last
tradingdaysofthemonthsandquarters, forinstance, tendtohaveadditionaltrad-
ingactivityasinvestorsrebalancetheirportfolios. Similarly, optionsandfutures
expiry days (quarterly/monthly expiry, ‘Triple Witching’28 in the US, Special
Quotationsin Japan, etc.) tendtoexperienceexcesstradingvolumeanddifferent
intradaypatternsresultingfromhedgingactivityandportfolioadjustments. Con-
sequently, theyneedtobehandledseparately, inparticular, whenmodelingtrad-
28 Triplewitchingdayshappenfourtimesayearonthethird Fridayof March, June, Septemberand
December. Onthesedays, thecontractsforstockindexfutures, stockindexoptionsandstockoptions
expireconcurrently.


<!-- Page 46 -->

32 Algorithmic Tradingand Quantitative Strategies
ingvolume. Asmostexecutionstrategiesmakeuseofrelativelyshortintervalvol-
umemetrics (e.g.,30-dayor60-day ADV), onesingledatapointcanimpactthe
overalllevelinferred. Similarlyautoregressivemodelsoflowordermayunderper-
formbothonspecialdaysandonthedaysfollowingthem. Asaresult, modelers
often remove special days and model normal days first. Then, special days are
modeledseparately, eitherindependentlyorusingthenormaldaysasabaseline.
Inordertoreflectchangesinthemarketandremainconsistentwithindexinclu-
sionrules, mostindicesneedtoundergoregularupdatesoftheconstituentsand
theirrespectiveweights. Thesignificantindicesdosoatregularintervals (annu-
ally, semi-annuallyorquarterly) onapre-announceddate. Atthecloseofbusiness
ofthatday, somestocksmightbeaddedtotheindexwhileothersareremoved,
ortheweightofeachstockintheindexmightbeincreasedordecreased. These
events, knownasindexrebalances, areparticularlyrelevanttopassiveinvestors
whoaretrackingtheindex. Inordertominimizethetrackingerrortothebench-
markindex, investorsneedtoalsoadjusttheirholdingsaccordingly. Additionally,
asmostfundsarebenchmarkedatthecloseprice, thereisanincentiveforfund
managerstotrytorebalancetheirholdingsatapriceascloseaspossibletothe
officialcloseprice, onthedaytheindexrebalancebecomeseffective. Asaresult,
onthesedays, intradayvolumedistributionissignificantlyskewedtowardtheend
ofdayandrequiressomeadjustmentintheexecutionstrategies.
• Futures-Specific Reference Data: Futures contracts present particular charac-
teristicsrequiringadditionalreferencedatatobecollected. Oneofthecoredif-
ferences of futures contracts compared to regular stocks is the fact that instru-
ments have an expiry date, after which the instrument ceases to exist. For the
purposeofbacktestingstrategies, itisnecessarytoknowwhichcontractwaslive
at any point in time through the use of an expiry calendar, but also which con-
tractwasthemostliquid. Forinstance, equityindexfuturestendtobethemost
liquid, forthefirstcontractavailable (alsoknownasfrontmonth), whileenergy
futuressuchasoiltendtobemoreliquidforthesecondcontract. Whilethismay
appeartobetrivial, whenbuildingatradingstrategyandmodelingpriceseries,
it is particularly important to know which contract carries the most significant
priceformationcharacteristicsandwhatisthetrueliquidityavailableinorderto
properlyestimatethemarketimpact.
The task of implementing a futures expiry calendar is further complicated by
thefactthereisnorealstandardizedfrequencythatappliesacrossmarkets. For
instance, Europeanequityindexfuturestendtoexpiremonthly, while USindex
futures expire quarterly. Some contracts even follow an irregular cycle through
the course of the year as it is the case for grain futures (e.g., wheat) that were


<!-- Page 47 -->

Trading Fundamentals 33
originallycreatedforhedgingpurposesandasaresulthaveexpirymonthsthat
followthecropcycle.29
The fact futures contracts expire on a regular basis has further implications in
terms of liquidity. If investors holding these contracts want to maintain their
exposureforlongerthanthelifespanofthecontract, theyneedtorollovertheir
positionsontothenextcontractwhichmighthaveanoticeablydifferentliquidity
level. Forinstance, thefrontcontractofthe S&P500(e.g., ESH8) tradesroughly
1.5 millioncontractsperdayintheweeksprecedingexpirywhilethenextmonth
contract (ESM8) only trades about 30,000 contracts per day. However, this liq-
uidity relationship will invert in the few days leading to the expiry of the front
contract as most investors roll their positions, and the most liquid contract will
becomethebackmonthcontract (see Figure1.4).Asaresult, whencomputing
rolling-windowmetrics (suchasaveragedailyvolumeforinstance), itisneces-
sarytoaccountforpotentialrolldatesthatmayhavehappenedduringthetime
span. Intheexampleabove, asimple60-dayaveragedailyvolumeon ESM8 taken
inearly April2018 wouldcapturealargenumberofdayswithverylowvolume
(January-March) owing to the fact the most liquid contract at the time was the
ESH8 contract, and would not accurately represent the volume activity of the
S&P500 futurescontract. Amoreappropriateaveragevolumemetrictobeused
asaforwardlookingvalueforexecutionpurposeswouldblendthevolumetime
seriesof ESH8 priortotherolldate, and ESM8 aftertherolldate.30 Additionally,
inordertoefficientlymergefuturespositionswithotherassetsinaninvestment
strategy, referencedatarelativetothequotationofthesecontracts, contractsize
(translation between the quotation points and the actual monetary value), cur-
rency, etc., mustbestored.
Finally, futures markets are characterized by the existence of different market
phases during the day, with significantly different liquidity characteristics. For
instance, equity index futures are much more liquid during the hours when the
corresponding equities markets are open. However, one can trade during the
overnightsessioniftheywantto. Theovernightsessionbeingmuchlessliquid,
the expected execution cost tends to be higher, and as such, the various market
datametrics (volumeprofile, averagespread, averagebid-asksizes,...) shouldbe
computed separately for each market phase, which requires maintaining a table
ofthestartandendtimesofeachsessionforeachcontract.
29 USWheat Futuresexpirein March, May, July, Septemberand December.
30 Itisworthnotingthatdifferentcontracts‘roll’atdifferentspeeds. Whileformonthlyexpirycontracts
itispossibletoseemostoftheopeninterestswitchfromthefrontmonthcontracttothebackmonthonthe
daypriortotheexpiry. Forquarterlycontractsitisnotuncommontoseetherollhappenoverthecourse
ofaweekormore, andthefrontmonthliquidityvanishseveraldaysaheadoftheactualexpiry. Careful
modelingisrecommendedonacasebycasebasis.


<!-- Page 48 -->

34 Algorithmic Tradingand Quantitative Strategies
Figure1.4:Futures Volume Rolling.
• Options-Specific Reference Data (Options Chain):Similartofuturescontracts,
optionscontractspresentacertainnumberofspecificitiesforwhichreferencedata
needtobecollected. Ontopofthesimilarfeatureofhavingaparticularexpiry
date, optionscontractsarealsodefinedbytheirstrikeprice. Thecombinationof
expiriesandstrikesisknownastheoptionchainforagivenunderlier. Theability
tomapequitytickerstooptiontickersandtheirrespectivestrikeandexpirydates
allows for the design of more complex investment and hedging strategies. For
instance, distancetostrike, changeinopeninterestofputsandcalls, etc., canall
beusedassignalsfortheunderlyingsecurityprice. Foraninterestingarticleon
howdeviationsinput-callparitycontainsinformationaboutfutureequityreturns,
referto Cremersand Weinbaum (2010)[95].
• Market-Moving News Releases: Macro-economic announcements are known
for their ability to move markets substantially. Consequently, it is necessary to
maintain a calendar of dates and times of their occurrences in order to assess
theirimpactonstrategiesanddecidehowbesttoreacttothem. Themostcom-
mon ones are central banks’ announcements or meeting minutes releases about
themajoreconomies (FED/FOMC, ECB, BOE, BOJ, SNB), Non-Farm Payrolls,
Purchasing Managers’ Index, Manufacturing Index, Crude Oil Inventories, etc.
While these news releases impact the broad market or some sectors, there are
also stock specific releases that need to be tracked: Earning calendars, special-
izedsectoreventssuchas FDAresultsforthehealthcareandbiotechsectors, etc.
• Related Tickers:Thereisawiderangeoftickersthatarerelatedtoeachother,
often because they fundamentally represent the same underlying asset. Main-
tainingaproperreferenceallowstoefficientlyexploitopportunitiesinthemar-
ket. Somenon-exhaustiveexamplesinclude:Primarytickerstocompositetickers
mapping (formarketswithfragmentedliquidity), duallisted/fungiblesecurities


<!-- Page 49 -->

Trading Fundamentals 35
in US and Canada, American Depository Receipt (ADR) or Global Depository
Receipt (GDR), localandforeignboardsin Thailand, etc.
• Composite Assets:Someinstrumentsrepresentseveralunderlyingassets (ETFs,
Indexes, Mutual Funds, ...). Their rise in popularity as investments over time
makesthemrelevantforquantitativestrategies. Theycanbeusedasefficientvehi-
clestoachievedesiredexposures (sectorandcountry ETFs, thematicfactor ETFs,
...), orascheaphedginginstruments, andtheycanprovidearbitrageopportuni-
tieswhentheydeviatefromtheir Net Asset Value (NAV).Inordertobeleveraged
inquantitativestrategies, oneneedstomaintainavarietyofinformationsuchasa
timeseriesoftheirconstituentsandthevalueofanycashcomponent, thedivisor
usedtotranslatethe NAVintothequotedprice, theconstituentweights.
• Latency Tables:Thislasttypeofdatawouldonlybeofinterestfordeveloping
strategies and for research in the higher frequency trading space. For these, it
mightberelevanttoknowthedistributionoflatencybetweendifferentdatacen-
tersastheycanbeusedformoreefficientorderroutingaswellasreorderingdata
thatmayhavebeenrecordedindifferentlocations.
Whiletheabovediscussionprovidesanon-exhaustivelistofissuesontherefer-
encedataavailabletobuildaquantitativeresearchplatform, theyhighlightthechal-
lengesthatmustbetakenintoaccountwhendesigningandimplementingalgorithmic
tradingstrategies. Onceinplace, apropersetofreferencedatawillallowthequan-
titative trader to systematically harness the actual content of various types of data
(describedlater) withoutbeingcaughtoff-guardbytheminutiaeoftrading.
1.4.2 Market Data
Whilehistoricallyalargeswathofmodelingfordevelopingstrategieswascarried
out on daily or minute bar data sets, the past fifteen years have seen a significant
rise in the usage of raw market data in an attempt to extract as much information
as possible, and act on it before the opportunity (or market inefficiency) dissipates.
Market data, itself, comes in various levels of granularity (and price!) and can be
subscribedtoeitherdirectlyfromexchanges (knownas“directfeeds”) orfromdata
vendorsaggregatinganddistributingit. Thelevelofdetailofthefeedsubscribedto
isgenerallydescribedas Level I, Level IIor Level IIImarketdata.
Level IData:Tradeand BBOQuotes:Level Imarketdataisthemostbasicform
oftick-by-tickdata. Historically, the Level Idatafeedwouldonlyrefertotradeinfor-
mation (Price, Size, Timeofeachtradereportedtothetape) buthasgrownovertime
intoatermgenerallyacceptedtomeanbothtradesandtopofbookquotes. Whilethe
tradefeedupdateswitheachtradeprintedonthetape, thequotefeedtendstoupdate
much more frequently each time liquidity is added to, or removed from, the top of
book (aroughestimateinliquidmarketsisthequotespresentmoreupdatesthanthe
tradesbyaboutanorderofmagnitude).


<!-- Page 50 -->

36 Algorithmic Tradingand Quantitative Strategies
Inordertobuildstrategies, thetimingofeacheventmustbeaspreciseaspossi-
ble. Whilethetimereportedfortradeandquotesisthematchingenginetimeatthe
exchange, mostdatabasesalsostoreareceptionorrecordtimetoreflectthepotential
latencybetweenthetradeeventandonebeingawarethatitdidhappenandbeingable
tostartmakingdecisionsonit. Accountingforthisreal-worldlatencyisanecessary
step for researchers building strategies on raw market data for which opportunities
maybeveryshortlivedandmaybeimpossibletoexploitbyparticipantswhoarenot
fastenough.
The Level I data is enough to reconstruct the Best Bid and Offer (BBO) of the
market. However, in fragmented markets, it is also useful to obtain an aggregated
consolidatedviewofallavailableliquidityatagivenpricelevelacrossallexchanges.
Marketdataaggregatorsusuallyprovidethisfunctionalityforuserswhodonotwish
toreconstructthefullorderbookthemselves.
Finally, it is worth noting that even Level I data contains significant additional
informationintheformoftradestatus (canceled, reportedlate, etc.) and, tradeand
quotequalifiers. Thesequalifiersprovidegranulardetailssuchaswhetheratradewas
anoddlot, anormaltrade, anauctiontrade, an Intermarket Sweep, anaverageprice
reporting, on which exchange it took place, etc. These details can be used to better
analyzethesequenceofeventsanddecideifagivenprintshouldbeusedtoupdate
thelastpriceandtotalvolumetradedatthatpointintimeornot. Forinstance, ifnot
processedappropriately, atradereportedtothetapeoutofsequencecouldresultina
largejumpinpricebecausethemarketmayhavesincemoved, andconsequentlythe
tradestatusisanindicationthattheprintshouldnotbeutilizedasitis.
Capturing raw market data, whether to build a research database or to process
itinreal-timetomaketradingdecisions, requiressignificantinvestmentsandexpert
knowledge to handle all the inherent complexity. Hence, one should carefully con-
sider the trade-off between the expected value that can be extracted from the extra
granularity, andtheadditionaloverheadcomparedtosimplersolutionssuchasusing
binneddata.
Level IIData:Market Depth:Level IImarketdatacontainsthesameinformationas
Level I, butwiththeadditionofquotedepthdata. Thequotefeeddisplaysalllitlimit
orderbookupdates (pricechanges, additionorremovalofsharesquoted) atanylevel
inthebook, andforallofthelitvenuesinfragmentedmarkets. Giventhevolumeof
datageneratedandthedecreasingactionablevalueofquoteupdatesastheirdistance
totopofbookincreases, someuserslimitthemselvestothetopfiveortenlevelsof
theorderbookwhentheycollectand/orprocessthedata.
Level III Data: Full Order View: Level III market data—also known as message
data—providesthemostgranularviewoftheactivityinthelimitorderbook. Each
orderarrivingisattributedaunique ID, whichallowsforitstrackingovertime, and
ispreciselyidentifiedwhenitisexecuted, canceledoramended. Similarto Level II,
thedatasetcontainsintradaydepthofbookactivityforallsecuritiesinanexchange.
Onceallmessagesfromdifferentexchangesareconsolidatedintoonesingledataset
orderedbytimestamp, itispossibletobuildafull (withnationaldepth) bookatany


<!-- Page 51 -->

Trading Fundamentals 37
momentintraday. Forillustrationpurposes, wetaketheexampleof USLevel IIIdata
andprovideashortdescriptionbelowin Table1.10.
Table1.10:Level IIIData
Variable: Description
Timestamp: Numberofmillisecondsafterthemidnight.
Ticker: Equitysymbol (upto8 characters)
Order: Uniqueorder ID.
T: Messagetype. Allowedvalues:
• “B”—Addbuyorder
• “S”—Addsellorder
• “E”—Executeoutstandingorderinpart
• “C”—Canceloutstandingorderinpart
• “F”—Executeoutstandingorderinfull
• “D”—Deleteoutstandingorderinfull
• “X”—Bulkvolumeforthecrossevent
• “T”—Executenon-displayedorder
Shares: Orderquantityforthe“B,”“S,”“E,”“X,”“C,”“T”messages. Zero
for“F”and“D”messages.
Price: Orderprice, availableforthe“B,”“S,”“X”and“T”messages.
Zeroforcancellationsandexecutions. Thelast4 digitsaredecimal
digits. The decimal portion is padded on the right with zeros. The
decimal point is implied by position; it does not appear inside the
pricefield. Divideby10000 toconvertintocurrencyvalue.
MPID: Market Participant IDassociatedwiththetransaction (4 characters)
MCID: Market Center Code (originatingexchange—1 character)
While the display and issuance of a new ID to the modified order varies from
exchangetoexchange, afewspecialtypesofordersareworthmentioning:
1. Ordersubjecttopricesliding:Theexecutionpricecouldbeonecentworsethanthe
displaypriceat NASDAQ;itisrankedatthelockingpriceasahiddenorder, and


<!-- Page 52 -->

38 Algorithmic Tradingand Quantitative Strategies
isdisplayedattheprice, oneminimumpricevariation (normally1 cent) inferior
tothelockingprice. Neworder IDwillbeusediftheorderisreplacedasadisplay
order. Atotherexchangestheoldorder IDwillbeused.
2. Pegged order: Based on NBBO, not routable, new timestamp given upon re-
pricing;displayrulesvaryoverexchanges.
3. Mid-pointpegorder:Non-displayed, canresultinhalf-pennyexecution.
4. Reserveorder:Displayedsizeisrankedasadisplayedlimitorderandthereserve
sizeisbehindnon-displayedordersandpeggedordersinpriority. Theminimum
displayquantityis100 andthisamountisreplenishedfromthereservesizewhen
itfallsbelow100 shares. Anewtimestampiscreatedandthedisplayedsizewill
bere-rankeduponreplenishment.
5. Discretionary order: Displayed at one price while passively trading at a more
aggressive discretionary price. The order becomes active when shares are avail-
ablewithinthediscretionarypricerange. Theorderisrankedlastinpriority. The
executionpricecouldbeworsethanthedisplayprice.
6. Intermarketsweeporder:Orderthatcanbeexecutedwithouttheneedforchecking
theprevailing NBBO.
Therichnessofthedatasetallowssophisticatedplayers, suchasmarketmakers,
toknownotonlythedepthofbookatagivenpriceandthedepthprofileofthebook
on both sides, but more importantly the relative position of an order from the top
positionoftheside (buyorsell).Amongthemostcommongranularmicrostructure
behaviorsstudiedwith Level IIIdata, wecanmention:
• Thepatternofinter-arrivaltimesofvariousevents.
• Arrivalandcancellationratesasafunctionofdistancefromnearesttouchprice.
• Arrivalandcancellationratesasafunctionofotheravailableinformation, such
asinthequeueoneithersideofthebook, orderbookimbalance, etc.
Oncemodeled, thesebehaviorscan, inturn, beemployedtodesignmoresophisticated
strategiesbyfocusingontradingrelatedquestions:
• Whatistheimpactofmarketorderonthelimitorderbook?
• Whatarethechancesforalimitordertomoveupthequeuefromagivenentry
position?
• Whatistheprobabilityofearningthespread?
• Whatistheexpecteddirectionofthepricemovementoverashorthorizon?


<!-- Page 53 -->

Trading Fundamentals 39
Asillustratedbythediversityandgranularityofinformationavailabletotraders,
markets mechanics have grown ever more complex over time, and a great deal of
peculiarities (in particular at the reference data level) need to be accounted for in
ordertodesignrobustandwellperformingstrategies. Theproverbialdevilisalways
in the details when it comes to quantitative trading but while it might be tempting
to go straight to the most granular source of data, in practice—with the exception
oftrulyhighfrequencystrategies—mostpractitionersbuildtheiralgorithmictrading
strategiesrelyingessentiallyonbinneddata. Thissimplifiesthedatacollectionand
handlingprocessesandalsogreatlyreducesthedimensionofthedatasetssoonecan
focusmoreonmodelingratherthandatawrangling. Mostofthetimeseriesmodels
andtechniquesdescribedinsubsequentchaptersaresuitedtodailyorbinneddata.
1.4.3 Market Data Derived Statistics
Most quantitative strategies, even the higher frequency strategies that leverage
more and more granular data also leverage derived statistics from the binned data,
suchasdailydata. Herewegivealistofthemostcommononesusedbypractitioners
andresearchersalike.
Daily Statistics
Thefirstgrouprepresentstheoveralltradingactivityintheinstrument:
• Open, High, Low, Close (OHLC) and Previous Close Price:The OHLCpro-
vides a good indication of the trading activity as well as the intraday volatility
experienced by the instrument. The distance traveled between the lowest and
highest point of the day usually gives a better indication of market sentiment
than the simple close-to-close return. Keeping the previous close value as part
ofthesametimeseriesisalsoagoodwaytoimprovecomputationefficiencyby
nothavingtomakeanadditionaldatabasequerytocomputethedailyreturnand
overnight gap. The previous close needs, however, to be properly adjusted for
corporateactionsanddividends.
• Last Trade before Close (Price/Size/Time): It is useful in determining how
muchtheclosepricemayhavejumpedinthefinalmomentsoftrading, andcon-
sequently, howstableitisasareferencevalueforthenextday.
• Volume: It is another valuable source of trading activity indicator, in particu-
larwhentheleveljumpsfromthelongtermaverage. Itisalsoworthcollecting
the volume breakdown between lit and dark venues, in particular for execution
strategies.
• Auctions Volumeand Price:Dependingontheexchange, therecanbemultiple
auctionsinaday (Open, Close, Morning Closeand Afternoon Open—formar-
ketswithalunchbreak—aswellasadhocliquidityorintradayauctions).Owing
totheirliquidityaggregationnature, theycanbeconsideredasavaluableprice
discoveryeventwhensignificantvolumeprintsoccur.


<!-- Page 54 -->

40 Algorithmic Tradingand Quantitative Strategies
• VWAP:Similartothe OHLC, theintraday VWAPpricegivesagoodindication
ofthetradingactivityontheday. Itisnotuncommontobuildtradingstrategies
using VWAPtimeseriesinsteadofjustclose-to-closeprices. Themainadvantage
beingthat VWAPpricesrepresentavalueoverthecourseofthedayand, assuch,
forlargerordersareeasiertoachievethroughalgorithmicexecutionthanasingle
print.
• Short Interest/Days-to-Cover/Utilization: This data set is a good proxy for
investorpositioning. Theshortpressuremightbeanindicationofupcomingshort
termmoves:Alargeshortinterestusuallyindicatesabearishviewfrominstitu-
tional investors. Similarly, the utilization level of available securities to borrow
(inordertoshort) givesanindicationofhowmuchroomisleftforfurthershort-
ing (when securities become “Hard to Borrow”, the cost of shorting becomes
significantly higher requiring short sellers to have strong enough beliefs in the
shorttermpricedirection).Finally, days-to-coverdataisalsovaluabletoassess
themagnitudeofapotentialshortsqueeze. Ifshortsellersneedtounwindtheir
positions, it is useful to know how much volume this represents as a fraction
ofavailabledailyliquidity. Thelargerthevalue, thelargerthepotentialsudden
upswingonheavilyshortedsecurities.
• Futures Data:Futuresmarketsprovideadditionalinsightintotheactivityoflarge
investorsthroughopeninterestdata, thatcanbeusefultodevelopalphastrategies.
Additionally, financialfuturesofferarbitrageopportunitiesiftheirbasisexhibits
mispricingcomparedtoone’sdividendestimates. Assuch, recordingthebasisof
futurescontractsisworthwhileevenforstrategiesthatdonotparticularlytarget
futures.
• Index-Level Data:Thisisalsoavaluabledatasettocollectasasourceofrel-
ativemeasuresforinstrumentspecificfeatures (Index OHLC, Volatility,...).In
particularfordispersionstrategies, normalizedfeatureshelpidentifyindividual
instrumentsdeviatingfromtheirbenchmarks.
• Options Data:Thederivativemarketisagoodsourceofinformationaboutthe
positioningoftradersthroughopeninterestand Greekssuchas Gammaand Vega.
How the broader market is pricing an instrument through implied volatility for
instanceisofinterest.
• Asset Class Specific:Thereisawealthofcross-assetinformationavailablewhen
buildingstrategies, inparticularin Fixed Income, FX, and Creditmarkets. Among
thebasicones, wewouldnote:
– Yield/benchmarkrates (repo,2 y,10 y,30 y)
– CDSSpreads
– USDollar Index
Thesecondgroupofdailydatarepresentsgranularintradaymicrostructureactiv-
ityandismostlyofinteresttointradayorexecutiontradingstrategies:


<!-- Page 55 -->

Trading Fundamentals 41
• Numberand Frequencyof Trades:Aproxyfortheactivitylevelofaninstru-
ment, and how continuous it is. Instruments with a low number of trades are
hardertoexecuteandcanbemorevolatile.
• Numberand Frequencyof Quote Updates:Similarproxyfortheactivitylevel.
• Topof Book Size:Aproxyforliquidityoftheinstrument (largertopofbooksize
makesitpossibletotradelargerordersizequasiimmediately, ifneeded).
• Depthof Book (priceandsize):Similarproxyforliquidity.
• Spread Size (average, median, timeweightedaverage):Thisprovidesaproxy
forcostoftrading. Aparametrizeddistributionofspreadsizecanbeusedtoiden-
tifyintradaytradingopportunitiesiftheyarecheaporexpensive.
• Trade Size (average, median):Similartospreadsize, tradesizesandtheirdis-
tribution are useful to identify intraday liquidity opportunities when examining
thevolumeavailableintheorderbook.
• Ticking Time (average, median):Thetickingtimeanditsdistributionisarep-
resentationofhowoften, oneshouldexpectchangesintheorderbookfirstlevel.
Thisisparticularlyhelpfulforexecutionalgorithmsforwhichthefrequencyof
updates (adding/canceling child orders, reevaluating decisions, etc.) should be
commensuratewiththecharacteristicsofthetradedinstrument.
Thedailydistributionsofthesemicrostructurevariablescanbeusedasstartof
day estimates in trading algorithms and be updated intraday as additional data
flowsinthroughonline Bayesianupdates.
Finally, thelastgroupofdailydatacanbederivedfromtheprevioustwogroups
throughaggregationbutisusuallystoredpre-computedinordertosavetimeduring
the research phase (e.g., X-day trailing data), or to be used as normalizing values
(e.g., sizeasapercentageof ADV, spreadinrelationtolong-termaverage,...).Some
commonexampleswouldbe:
• 𝑋-day Average Daily Volume (ADV)/Averageauctionvolume
• 𝑋-dayvolatility (close-to-close, open-to-close, etc.)
• Betawithrespecttoanindexorasector (plainbeta, orasymmetricup-days/down-
daysbeta)
• Correlationmatrix
Asaremindertothereader, aggregateddataneedtofullysupportthepeculiar-
itiesdescribedinthe Reference Datasection (forinstance:Theexistenceofspecial
eventdayswhich, ifincluded, cansignificantlyskewintradaydistributionofvalues;
mishandling of market asynchronicity resulting in inaccurate computations of key
quantitiessuchasbetaorcorrelation, etc.).


<!-- Page 56 -->

42 Algorithmic Tradingand Quantitative Strategies
Binned Data
Thefirstnaturalextensiontodailydatasetsisadiscretizationofthedayintobins
rangingfromafewsecondsto30 minutes. Thefeaturescollectedarecomparableto
theonesrelevantfordailydatasets (periodvolume, open, high, low, close, VWAP,
spread, etc.), but computed at higher frequency. It is worth mentioning that minute
bar data actually underpins the vast majority of microstructure models used in the
electronicexecutionspace. Volumeandspreadprofiles, forinstance, arerarelybuilt
withagranularityfinerthanoneminutetopreventintroducingexcessnoiseduepurely
to market frictions. The major advantage of binned-data is that discrete time series
methodscanbereadilyused.
These minute bar data sets are also quite popular for the backtesting of low-to-
mediumfrequencytradingstrategiestargetingintradayalpha (shortdurationmarket
neutrallong-shortbaskets, momentumandmean-reversionstrategies, etc.).Themain
benefit they provide is a significant dimension reduction compared to raw market
data (390 rowsperstockperdayinthe UScomparedtomillionsforrawdataforthe
liquidstocks), whichallowsresearcherstoperformrapidandefficientbacktestingas
theysearchforalpha. However, increasingdatafrequencyfromdailydatatointraday
minute bars also presents challenges. In particular, relationships that appear to be
stableusingclose-to-closevaluesbecomemuchnoisierasgranularityincreasesand
signals become harder to extract (e.g., drop in correlation between assets, pricing
inefficiency moves due to sudden liquidity demand, etc.). Similarly, for less liquid
assets with a low trade frequency, there may not be any trading activity for shorter
durations, resultinginemptybins.
1.4.4 Fundamental Dataand Other Data Sets
Averylargenumberofquantitativeinvestmentstrategiesarestillbasedonfunda-
mentaldata, andconsequentlycountlessresearchpapersareavailabletotheinterested
readerdescribingexamplesoftheirusage. Herewewillonlydescribethemainclasses
ofexistingfundamentaldata:
• Key Ratios: EPS (Earnings Per Share), P/E (Price-to-Earning), P/B (Price-to-
Book Value), .... These metrics represent a normalized view of the financials
ofcompaniesallowingforeasiercross-sectionalcomparisonofstocksandtheir
rankingovertime.
• Analyst Recommendations:Researchanalystsat Sell Sideinstitutionsspenda
greatdealofresourcesanalyzingcompaniestheycover, inordertoprovideinvest-
mentrecommendationsusuallyintheformofa Buy/Hold/Sellratingaccompa-
niedbyapricetarget. Whileindividualrecommendationsmightprovenoisy, the
aggregate values across a large number of institutions can be interpreted as a
consensusvaluationofagivenstock, andchangesinconsensuscanhaveadirect
impactonprice.
• Earnings Data:Similarlyresearchanalystsalsoprovidequarterlyearningesti-
matesthatcanbeusedasanindicationoftheperformanceofastockbeforethe


<!-- Page 57 -->

Trading Fundamentals 43
actualvaluegetspublishedbythecompany. Here, too, consensusvaluestendto
playalargerrole. Inparticular, whenthedifferencebetweentheforecastconsen-
susandtherealizedvalueislarge (knownasearningsurprise), asthestockmight
thenexperienceoutsizedreturnsinthefollowingdays. Thus, collectinganalysts’
forecastsaswellasrealizedvaluescanbeavaluablesourceofinformationinthe
designoftradingstrategies.
• Holders: In some markets, large institutional investors are required to disclose
theirholdingsonaregularbasis. Forinstance, inthe US, institutionalinvestment
managerswithover$100 millioninassetsmustreportquarterly, theirholdings,
tothe SECusing Form13 F.Theformsarethenpubliclyavailableviathe SEC’s
EDGARdatabase. Additionally, shareholdersmightberequiredtodisclosetheir
holdingsoncetheypasscertainownershipthresholds.31 Suddenchangesinsuch
ownershipmightindicatechangesinsentimentbysophisticatedinvestorandcan
haveasignificantimpactonstockperformance.
• Insiders Purchase/Sale:Insomemarkets, companydirectorsarerequiredbylaw
todisclosetheirholdingsofthecompanystockaswellasanyincreaseordecrease
ofsuchholdings.32 Thisisthoughttobeanindicatoroffuturestockpricemoves
fromthegroupofpeoplewhohaveaccesstothebestpossibleinformationabout
thecompany.
• Credit Ratings: Most companies issue both stocks and bonds to finance their
operations. Thecreditratingsofbondsandtheirchangesovertimeprovideaddi-
tionalinsightintothehealthofacompanyandareworthleveraging. Inparticular,
creditdowngradesresultinginhigherfundingcostsinthefuture, generallyhave
anegativeimpactonequityprices.
• Andmuch, muchmore:Recentyearshaveseentheemergenceofawidevariety
ofalternativedatasetsthatareavailabletoresearchersandpractitionersalike.33
Whileitisnotpossibletomakeacomprehensivelistofallthatareavailable, they
cangenerallybeclassifiedbasedontheircharacteristics:Frequencyofpublica-
tion, structuredorunstructured, andvelocityofdissemination. Thevalueofsuch
datadependsontheobjectivesandresourcesoftheuser (naturallanguagepro-
cessing or image recognition for unstructured data require significant time and
efforts), butalso—andmaybemoreimportantly—ontheuniquenessofthedata
31 Inthe US, Form13 Dmustbefiledwiththe SECwithin10 daysbyanyonewhoacquiresbeneficial
ownershipofmorethan5%ofanyclassofpubliclytradedsecuritiesinapubliccompany.
32 Inthe US, officersanddirectorsofpubliclytradedcompaniesarerequiredtodisclosetheirinitial
holdingsinthecompanybyfiling Form3 withthe SEC, aswellas Form4 within2 daysofanysubsequent
changes. Theformsarethenpubliclyavailableviathe SEC’s EDGARdatabase.
33 Forinstance, www.orbitalinsight.comoffersdailyretailtrafficanalytics, derivedfromsatellite
imageryanalysismonitoringover260,000 parkinglots, aswellasestimatesofoilinventoriesthrough
satellitemonitoringofoilstoragefacilities.


<!-- Page 58 -->

44 Algorithmic Tradingand Quantitative Strategies
set. Asmoreinvestorsgetaccesstoit, theharderitbecomestoextractmeaningful
alphafromit.
1.5 Market Microstructure:Economic Fundamentalsof Trading
Wefirstprovideabriefreviewofmarketmicrostructure, anareaoffinancethat
studieshowthesupplyanddemandofliquidityresultsinactualtransactionsandmod-
ifiesthesubsequentstateofthemarket, andthendelveintosomecriticaloperational
concepts. Wedrawuponkeyreviewpapersby Madhavan (2000)[254], Biais, Glosten
and Spatt (2005)[39]and O’Hara (2015)[276].Thecoreideaisthattheefficientmar-
ket hypothesis, which postulates that the equity price impounds all the information
abouttheequity, maynotholdduetomarketfrictions. Algorithmictradingessentially
exploitsthespeedwithwhichinvestorsacquiretheinformationandhowtheyuseit
alongwiththemarketfrictionsthatarisemainlyduetodemand-supplyimbalances.
Takinganinformationaleconomicsangle, Madhavan (2000)[254]providesamarket
microstructureanalysisframeworkfollowingthreemaincategories.
– Price Formationand Price Discovery:Howdopricesimpoundinformationover
timeandhowdothedeterminantsoftradingcostsvary?
– Market Design:Howdotradingrulesaffectpriceformation?
Market design generally refers to a set of rules that all players have to follow in
the trading process. These include the choice of tick size, circuit breakers which
can halt trading in the event of large price swings, the degree of anonymity and
the transparency of the information to market participants, etc. Markets around
the world and across asset classes can differ significantly in these types of rules,
creatingadiversesetofconstraintsandopportunityforalgorithmictraders. Some
earlyresearchontheeffectsofmarketdesignleadtofollowingbroadconclusions:
– Centralizedtradingviaonesinglemarkettendstoresultinmoreefficientprice
discoverywithsmallerbid-askspreads. Inthepresenceofmultiplemarkets, the
primarymarkets (suchas NYSE, NASDAQ) remainthemainsourcesofprice
discovery (see Hasbrouck (1995)[181]).
– Despite market participants’ preference for continuous, automated limit order
book markets, theoretical models suggest that multilateral trading approaches
such as single-price call auctions are the most efficient in processing diverse
information (See Mendelson (1982)[263]and Hoetal.(1985)[197]).
– Transparency:Howdothequantity, qualityandspeedofinformationprovidedto
marketparticipantsaffectthetradingprocess?
Transparencywhichisbroadlyclassifiedintopre-trade (litorderbook) andpost-
trade (tradereportingtothepublic, thoughatdifferenttimelags) isoftenatrade-off.


<!-- Page 59 -->

Trading Fundamentals 45
Whileintheorymoretransparencyshouldleadtobetterpricediscovery, thewide
disclosure of order book depth information can lead to thinner posted sizes and
widerbid-askspreadsifparticipantsfearrevealingtheirintentandpossiblytheir
inventorylevels, leadingthemtofavormoreoff-exchangeactivity.
Alltheabovepointscanbeanalyzedinlightoftherapidgrowthofhighfrequency
trading (HFT) describedearlierinthischapter. O’Hara (2015)[276]presentsissues
relatedtomicrostructureinthecontextof HFT.Whilethebasictenetthattradersmay
useprivateinformationorlearnfrommarketdatasuchasorders, tradesize, volume,
durationbetweensuccessivetrades, etc., hasremainedthesame, thetradingisnow
mostlyautomatedtofollowsomerules. Theserulesarebasedpartlyonpriorinforma-
tionandpartlyonchangingmarketconditions, monitoredthroughorderflows. With
thehighspeed, adverseselectionhastakenadifferentrole. Sometradersmayhave
accesstomarketdatamillisecondsbeforeothershaveitandthismayallowthemto
captureshorttermpricemovements. Thiswouldexpandthepoolofinformedtraders.
Herearesometopicsforresearchinmicrostructure:
– Withaparentorderslicedintoseveralchildordersthataresenttomarketforexe-
cution during the course of trading, it is difficult to discern who is the informed
trader. Informedtradersusesophisticateddynamicalgorithmstointeractwiththe
market. Retail (known in the literature as ‘uninformed’) trades usually cross the
spread.
– Moreworkneedstobedonetounderstandtradingintensityinshortintervals. Order
imbalanceisempiricallyshowntobeunrelatedtopricelevels.
– Informed traders may increasingly make use of hidden orders. How these orders
enterandexitthemarketsrequirefurtherstudies.
– Tradersrespondtochangingmarketconditionsbyrevisingtheirquotedprices. The
quotevolatilitycanprovidevaluableinformationabouttheperceiveduncertainty
inthemarket.
Although HFThasresultedinmoreefficientmarkets, withlowerbid-askspreads,
thedatarelatedtotradesequences, patternsofcancellationsacrossfragmentedmar-
ketsrequirenewtoolsforanalysisandformakingactionableinference. Inthisreview,
we do not present any models explicitly and these will be covered throughout this
book. Nowkeepinginlinewiththepracticalperspectivesofthisbook, wehighlight
somekeyconcepts.
1.5.1 Liquidityand Market Making
(a) ADefinitionof Liquidity:Financialmarketsarecommonlydescribedascarry-
ingthefunctionofefficientlydirectingtheflowofsavingsandinvestmentstothe
realeconomytoallowtheproductionofgoodsandservices. Animportantfactor
contributingtowell-developedfinancialmarketsisinfacilitatingliquiditywhich
enablesinvestorstodiversifytheirassetallocationandthetransfersofsecurities
atareasonabletransactioncost.


<!-- Page 60 -->

46 Algorithmic Tradingand Quantitative Strategies
Properlydescribing“liquidity”oftenprovestobeelusiveasthereisnocommonly
agreedupondefinition. Black (1971)[42]proposesarelativelyintuitivedescrip-
tionofaliquidmarket:“Themarketforastockisliquidifthefollowingconditions
hold:
• There are always bid and ask prices for the investor who wants to buy or
sellsmallamountsofstockimmediately.
• Thedifferencebetweenthebidandaskprices (thespread) isalwayssmall.
• Aninvestorwhoisbuyingorsellingalargeamountofstock, intheabsence
ofspecialinformation, canexpecttodosooveralongperiodoftimeata
pricenotverydifferent, onaverage, fromthecurrentmarketprice.
• An investor can buy or sell a large block of stock immediately, but at a
premium ordiscount thatdepends onthe sizeof theblock. The largerthe
block, thelargerthepremiumordiscount.”
In other words, Black defines a liquid market as a continuous market having
the characteristics of relatively tight spread, with enough depth on each side of
thelimitorderbooktoaccommodateinstantaneoustradingofsmallorders, and
whichisresilientenoughtoallowlargeorderstobetradedslowlywithoutsignif-
icantimpactonthepriceoftheasset. Thisgeneraldefinitionremainsparticularly
wellsuitedtotoday’smodernelectronicmarketsandcanbeusedbypractition-
erstoassessthedifferenceinliquiditybetweenvariousmarketswhenchoosing
wheretodeployastrategy.
(b) Modelfor Market Friction:Webeginwithamodeltoaccommodatethefriction
instockprice (𝑃).If𝑝∗ isthe (log) truevalueoftheassetwhichcanvaryover
𝑡 𝑡
timeduetoexpectedcashflowsorduetovariationinthediscountrate. Giventhe
publiclyavailableinformationandtheassumptionofmarketefficiency, wehave:
𝑝 =𝑝∗+𝑎 and𝑝∗ =𝑝∗ +𝜖.Thereforetheobservedreturn,𝑟 =𝑝 −𝑝 =
𝑡 𝑡 𝑡 𝑡 𝑡−1 𝑡 𝑡 𝑡 𝑡−1
𝜖 + (𝑎 − 𝑎 ) can exhibit some (negative) serial correlation, mainly a result
𝑡 𝑡 𝑡−1
of friction. The friction, ‘𝑎’ can be a function of a number of factors such as
𝑡
inventorycosts, riskaversion, bid-askspread, etc. When𝑎 =𝑐𝑠, where𝑠 isthe
𝑡 𝑡 𝑡
directionofthetradeand‘𝑐’isthehalf-spread, themodeliscalleda Rollmodel.
Thiscanexplainthestickinessinreturnsinsomecases.
(c) Different Styles of Market Participants: Liquidity Takers and Liquidity
Providers: Liquid financial markets carry their primary economic function by
facilitatingsavingsandinvestmentflowsaswellasallowinginvestorstoexchange
securities in the secondary markets. Economic models of financial markets
attempt to classify market participants into different categories. These can be
broadlydelineatedas:
Informed Traders, makingtradingdecisionsbasedonsuperiorinformationthat
isnotyetfullyreflectedintheassetprice. Thatknowledgecanbederivedfrom
either fundamental analysis or information not directly available nor known to
othermarketparticipants.


<!-- Page 61 -->

Trading Fundamentals 47
News Traders, making trading decisions based on market news or announce-
mentsandtryingtomakeprofitsbyanticipatingthemarket’sresponsetoapar-
ticularcatalyst. Theelectronificationofnewsdisseminationoffersnewopportuni-
tiesfordevelopingquantitativetradingstrategiesbyleveragingtext-miningtools
suchasnaturallanguageprocessingtointerpret, andtradeon, machinereadable
newsbeforeitisfullyreflectedinthemarketprices.
Noise Traders (as introduced by Kyle (1985) [234]), making trading decisions
withoutparticularinformationandatrandomtimesmainlyforliquidityreasons.
They can be seen as adding liquidity to the market through additional volume
transacted, butonlyhaveatemporaryeffectonpriceformation. Theirpresence
inthemarketallowsinformedtradersnottobeimmediatelydetectedwhenthey
starttransacting, asmarketmakerscannotnormallydistinguishtheoriginofthe
order flow between these two types of participants. In a market without noise
traders, beingfullyefficient, atequilibriumeachtradewouldberevealinginfor-
mation that would instantly be incorporated into prices, hereby removing any
profitopportunities.
Market Makers, providingliquiditytothemarketwiththeintentofcollecting
profits originating from trading frictions in the market place (bid-ask spread).
Risk-neutral market makers are exposed to adverse selection risk arising from
thepresenceofinformedtradersinthemarketplace, andthereforeestablishtheir
tradingdecisionsmostlybasedontheircurrentinventory. Assuch, theyareoften
consideredintheliteraturetodrivethedeterminationofefficientpricesbyacting
astherationalintermediaries.
Generalizing these concepts, market participants and trading strategies can be
separated between liquidity providing and liquidity seeking. The former being
essentiallythedomainofmarketmakerswhoselevelofactivity, proxiedbymar-
ket depth, is proportional to the amount of noise trading and inversely propor-
tional to the amount of informed trading (Kyle (1985) [234]). The latter being
thedomainofthevarietyofalgorithmictradingusers, describedbefore (mutual
funds, hedge funds, asset managers, etc.). Given the key role played by market
makersintheliquidityofelectronicmarkets, theyhavebeenthesubjectofalarge
corpusofacademicresearchfocusingontheiractivities.
(d) The Objectivesofthe Modern Market Maker:Atpresent, marketmakingcan
broadlybeseparatedintotwomaincategoriesbasedonthetradingcharacteris-
tics. Thefirstoneistheprovisionoflargeliquidity—knownasblocks—toinstitu-
tionalinvestors, andhastraditionallybeenintherealmofsell-sidebrokersacting
as intermediaries and maintaining significant inventories. Such market makers
usuallytransactthroughanon-continuous, negotiated, processbasedontheircur-
rentinventory, aswellastheirassessmentoftheriskinvolvedinliquidationofthe
positioninthefuture. Largerormorevolatilepositionsgenerallytendtocomeat
ahighercost, reflectingtheincreasedriskfortheintermediary. Buttheyprovide


<!-- Page 62 -->

48 Algorithmic Tradingand Quantitative Strategies
theendinvestorwithacertainpriceandanimmediateexecutionbearingnotim-
ingriskthatisassociatedwithexecutionovertime. Thesetransactions, because
they involve negotiations between two parties, still mostly happen in a manual
fashion or over the phone and then get reported to an appropriate exchange for
publicdissemination.
The second category of market making involves the provision of quasi-
continuous, immediately accessible quotes on an electronic venue. With the
advent of electronic trading described in the previous section, market makers
originallyseatedontheexchangefloorshaveprogressivelybeenreplacedbyelec-
tronic liquidity providers (ELP). The ELP leverage fast technology to dissemi-
nate timely quotes across multiple exchanges and develop automated quantita-
tive strategies to manage their inventory and the associated risk. As such, most
ELP can be classified as high frequency traders. They derive their profits from
three main sources: From liquidity rebates on exchanges that offer maker-taker
feestructure, fromspreadearnedwhensuccessfullybuyingonthebidandsell-
ingontheoffer, andfromshort-termpricemovesfavorabletotheirinventory.
Hendershott, Brogaardand Riordan (2014)[191]findthat HFTactivitytendsto
be concentrated in large liquid stocks and postulate that this can be attributed
toacombinationoflargerprofitopportunitiesemanatingfromtradeshappening
moreoften, andfromeasierriskmanagementduetolargerliquiditythatallows
foreasierexitofunfavorablepositionsatareasonablecost.
(e) Risk Management:Intheexistingliteratureoninformedtrading, itisobserved
that liquidity supplying risk-neutral market makers are adversely selected by
informed traders suddenly moving prices against them. For example, a market
maker buy quote tends to be executed when large sellers are pushing the price
down, resultinginevenlowerpricesinthenearterm. Thissignificantpotential
asymmetry of information at any point in time emphasizes the need for market
makerstoemployrobustriskmanagementtechniques, particularlyinthedomain
ofinventoryrisk. Foramarketmaker, riskmanagementisgenerallyaccomplished
byfirstadjustingmarketquotesupwardordownwardtoincreasethearrivalrate
ofsellersorbuyersandconsequentlyadjustingtheinventoryinthedesireddirec-
tion. If biasing the quotes does not result in a successful inventory adjustment,
themarketmakersgenerallyemploylimitorderstocrossthespread.
Hoand Stoll (1981)[196]introducedamarketmakingmodelinwhichthemarket
maker’sobjectiveistomaximizeprofitwhileminimizingtheprobabilityofruinby
determiningtheoptimalbid-askspreadtoquote. Theinventoryheldevolvesthrough
thearrivalofbidandaskorders, wherethearrivalrateistakentobeafunctionofbid
andaskprices. Theirmodelalsoincorporatestherelevantnotionofthesizedepen-
denceofspreadonthemarketmaker’stimehorizon. Thelongertheremainingtime,
the greater potential for adverse move risk for liquidity providers, and vice versa.
Thisisconsistentwithobservedspreads. Inmostmarkets, thespreadiswideratthe
beginningoftheday, narrowingtowardtheclose. Anadditionalreasonannotatedfor


<!-- Page 63 -->

Trading Fundamentals 49
thewiderspreadrightafterthebeginningofthetradingdayisduetotheexistence
ofpotentiallysignificantinformationasymmetryaccumulatedovernight. Asmarket
makersaredirectlyexposedtothatinformationasymmetry, theytendtoquotewider
spreadswhilethepricediscoveryprocessunfoldsfollowingtheopeningofcontinu-
oustrading, andprogressivelytightenthemasuncertaintyaboutthefairpriceforthe
assetdissipates.
Understandingthedynamicsofmarketmakersinventoryandriskmanagement,
andtheireffectonspreads, hasdirectimplicationsforthepractitionerswhointendto
deployalgorithmictradingstrategiesasthespreadpaidtoenterandexitpositionsisa
non-negligiblesourceofcostthatcanerodetheprofitabilityoflowalphaquantitative
strategies.
Hendershottand Seasholes (2007)[195]confirmsthatmarketmakers’inventories
arenegativelycorrelatedwithpreviouspricechangesandpositivelycorrelatedwith
subsequentchanges. Thisisconsistentwithmarketmakersfirstactingasadampener
of buying or selling pressure by bearing the risk of temporarily holding inventory
in return for earning the spread and thus potential price appreciation from market
reversal. Thismodeleasilylinksliquidityprovisionandthedynamicsofassetprices.
Finally, ithasbeenobservedthatthereisapositivecorrelationofmarketmakers
inventorywithsubsequentpriceschanges, inventoriescancomplementpastreturns
whenpredictingfuturereturns. Sinceinventoriesarenotpubliclyknown, marketpar-
ticipantsusedifferentproxiestoinfertheirvaluesthroughouttheday. Twocommonly
usedproxiesaretradeimbalances (thenetexcessofbuyorsellinitiatedtradevolume)
andspreads. Tradeimbalanceaimsatclassifyingtrades, eitherbuyinitiatedorsellini-
tiated, bycomparingtheirpricewiththeprevailingquote. Giventhatmarketmakers
trytominimizetheirdirectionalrisk, theycanonlyaccommodatealimitedamount
ofnon-diversifiedinventoryoverafiniteperiodoftime. Assuch, spreadsizes, and
moreparticularlytheirsuddenvariation, havealsobeenusedasproxiesfordetecting
excessinventoryforcingliquidityproviderstoadjusttheirpositions.


<!-- Page 64 -->

Bibliography
[1] F. Abdi and A. Ranaldo. A simple estimation of bid-ask spreads from daily
close, highandlowprices. The Reviewof Financial Studies,30:4437–4480,
2017.
[2] F.Abergeland A.Jedidi. Amathematicalapproachtoorderbookmodeling.
International Journalof Theoreticaland Applied Finance,16:1–40,2013.
[3] A.R.Admatiand P.Pfleiderer. Atheoryofintradaypatterns:Volumeandprice
variability. The Reviewof Financial Studies,1(1):3–40,1988.
[4] Y. Aït-Sahalia, P.A. Mykland, and L. Zhang. How often to sample a
continuous-timeprocessinthepresenceofmarketmicrostructurenoise. The
Reviewof Financial Studies,18(2):351–416,2005.
[5] H.Akaike. Anewlookatthestatisticalmodelidentification. IEEETransac-
tionson Automatic Control, AC–19:716–723,1974.
[6] S.S. Alexander. Price movements in speculative markets: Trends of random
walks. Industrial Management Review, pages7–26,1961.
[7] S.S. Alexander. Price movements in speculative markets: Trends of random
walks, no2. Industrial Management Review, pages25–46,1964.
[8] S. Alizadeh, M.W. Brandt, and F.X. Diebold. Range-based estimation of
stochasticvolatilitymodels. Journalof Finance,57:1047–1091,2002.
[9] R. Almgren. Execution costs. Encyclopedia of Quantitative Finance, pages
1–5,2008.
[10] R.Almgrenand N.Chriss. Optimalexecutionofportfoliotransactions. The
Journalof Risk,3:5–39,2000.
[11] R.Almgren, C.Thum, E.Hauptmann, and H.Li. Equitymarketimpact. Risk,
18(7):57–62,2005.
[12] R.F.Almgren. Optimalexecutionwithnonlinearimpactfunctionsandtrading
enhancedrisk. Applied Mathematical Finance,10:1–18,2003.
[13] N. Amenc, F. Goltz, A. Lodh, and L. Martellini. Diversifying the diversi-
fiersandtrackingthetrackingerror:Outperformingcap-weightedindiceswith
limited risk of underperformance. The Journal of Portfolio Management,
38(3):72–88,2012.
51


<!-- Page 65 -->

52 Bibliography
[14] S.Anatolyevand A.Gospodinov. Atradingapproachtotestingforpredictabil-
ity. Journalof Business&Economic Statistics,23:455–461,2005.
[15] S. Anatolyev and A. Gospodinov. Modeling financial return dynamics via
decomposition. Journalof Business&Economic Statistics,28:232–245,2010.
[16] T.Andersen, I.Archakov, G.Cebiroglu, and N.Hautsch. Volatilityinformation
feedbackandmarketmicrostructurenoise:Ataleoftworegimes. CFSWorking
Paper, Northwestern University,2017.
[17] T.G. Andersen. Return volatility and trading volume: An information flow
interpretationofstochasticvolatility. Journalof Finance,51:116–204,1996.
[18] T.G.Andersenand T.Bollerslev. Answeringtheskeptics:Yes, standardvolatil-
ity models do provide accurate forecasts. International Economic Review,
39:885–905,1998.
[19] T.W.Anderson. An Introductionto Multivariate Statistical Analysis. Second
Edition. Wiley, New York,1984.
[20] T.W.Andersonand A.M.Walker. Ontheasymptoticdistributionoftheauto-
correlationsofasamplefromalinearstochasticprocess. Annalsof Mathemat-
ical Statistics,35:1296–1303,1964.
[21] A.Angand A.Timmermann. Regimechangesandfinancialmarkets. Annual
Reviewof Financeand Economics,4:313–337,2012.
[22] W. Antweiler and M.Z. Frank. Is all that talk just noise? The information
content of internet stock message boards. Journal of Finance, 59(3):1259–
1294,2004.
[23] P.Asquith, R.Oman, and C.Safaya. Shortsalesandtradeclassificationalgo-
rithms. Journalof Financial Markets,13:157–173,2010.
[24] M. Avellaneda and J.H. Lee. Statistical arbitrage in the US equities market.
Quantitative Finance,10:761–782,2010.
[25] W.Bagehot. Theonlygameintown. Financial Analysis Journal,27:12–14,
1971.
[26] P. Bajgrowicz and O. Scaillet. Technical trading revisited: False discover-
ies, persistencetests, andtransactioncosts. Journalof Financial Economics,
106(3):473–491,2012.
[27] M. Baker and J. Wurgler. Investor sentiment and the cross-section of stock
returns. Journalof Finance,61(4):1645–1680,2006.
[28] M.Bakerand J.Wurgler. Investorsentimentinthestockmarket. Journalof
Economic Perspectives,21(2):129–151,2007.


<!-- Page 66 -->

Bibliography 53
[29] F.M.Bandiand J.R.Russell. Separatingmicrostructurenoisefromvolatility.
Journalof Financial Economics,79:655–692,2006.
[30] N.Barberis, A.Shleifer, and R.Vishny. Amodelofinvestorsentiment. Journal
of Financial Economics,49:307–343,1998.
[31] O.E. Barndorff-Nielsen and N. Shephard. Econometric analysis of realized
volatilityanditsuseinestimatingstochasticvolatilitymodels. Journalofthe
Royal Statistical Society:Series B(Statistical Methodology),64(2):253–280,
2002.
[32] L.Barras, O.Scaillet, and R.Wermers. Falsediscoveriesinmutualfundper-
formance:Measuringluckinestimatedalphas. Journalof Finance,65(1):179–
216,2010.
[33] R. Battalio, S.A. Corwin, and R. Jennings. Can brokers have it all? On the
relationbetweenmake-takefeesandlimitorderexecutionquality. Journalof
Finance,71:2193–2238,2016.
[34] L.Bauwens, S.Laurent, and J.V.K.Rombouts. Multivariate GARCHmodels:
Asurvey. The Journalof Applied Econometrics,21:79–109,2006.
[35] M.Bayraktar, I.Mashtaser, N.Meng, and S.Radchenko. Barravstotalmarket
equitytradingmodel, empiricalnotes. MSCIResearch,2015.
[36] P.Bertrandand C.Protopopescu. Thestatisticsoftheinformationratio. Inter-
national Journalof Business,15:71–86,2010.
[37] D.Bertsimasand A.W.Lo. Optimalcontrolofexecutionscosts. Journalof
Financial Markets,1:1–50,1998.
[38] H.Bessembinder, M.Panayides, and K.Venkataraman. Hiddenliquidity:An
analysis of order exposure strategies in electronic stock markets. Journal of
Financial Economics,94:361–383,2009.
[39] B.Biais, L.Glosten, and C.Spatt. Marketmicrostructure;asurveyofmicro-
foundations, empiricalresults, andpolicyimplications. Journalof Financial
Markets,8:217–264,2005.
[40] B.Biais, P.Hillion, and C.Spatt. Anempiricalanalysisofthelimitorderbook
and the order flow in the Paris bourse. Journal of Finance, 50:1655–1689,
1995.
[41] J.P.Bialkowski, S.Darolles, and Gaëlle G.Le Fol. Improving VWAP.strate-
gies: A dynamical volume approach. Journal of Banking and Finance, 32,
2006.
[42] F. Black. Towards a fully automated exchange, Part I. Financial Analysts
Journal,27:29–34,1971.


<!-- Page 67 -->

54 Bibliography
[43] F.Black. Capitalmarketequilibriumwithrestrictedborrowing. The Journal
of Business,45:444–454,1972.
[44] F.Blackand R.Litterman. Globalportfoliooptimization. Financial Analysts
Journal,48 No.5:28–43,1992.
[45] L.Blume, D.Easley, and M.O’Hara. Marketstatisticsandtechnicalanalysis:
Theroleofvolume. Journalof Finance,49:153–181,1994.
[46] T.Bollerslev. Generalizedautoregressiveconditionalheteroskedasticity. Jour-
nalof Econometrics,31:307–327,1986.
[47] M. Borkovec and H.G. Heidle. Building and evaluating a transaction cost
model:Aprimer. The Journalof Trading,5:57–77,2010.
[48] P.Bossaerts. Commonnonstationarycomponentsofassetprices. The Journal
of Economic Dynamicsand Control,12(2):347–364,1988.
[49] J.P. Bouchaud, J.D. Farmer, and F. Lillo. How Markets Digest Supply and
Demand and Slowly Incorporate Information into Prices. Academic Press,
2009.
[50] J.P.Bouchaud, Y.Gefen, M.Potters, and M.Wyart. Fluctuationsandresponse
infinancialmarkets:Thesubtlenatureof“random”pricechanges. Quantita-
tive Finance,4:176–190,2004.
[51] J.-P.Bouchaud, M.Mezard, and M.Potters. Statisticalpropertiesofstockorder
books:Empiricalresultsandmodels. Quantitative Finance,2:251–256,2002.
[52] D.Bowen, M.C.Hutchinson, and N.O’Sullivan. High-frequencyequitypairs
trading: Transaction costs, speed of execution and patterns in returns. The
Journalof Trading, Summer, pages31–38,2010.
[53] G.E.P.Box, G.M.Jenkins, G.C.Reinsel, and G.M.Ljung. Time Series Analy-
sis:Forecastingand Control,5 thedition. Wiley, New York,2015.
[54] G.E.P. Box and G.C. Tiao. A canonical analysis of multiple time series.
Biometrika,64:355–365,1977.
[55] P. Boyle, L. Garlappi, R. Uppal, and T. Wang. Keynes meets Markowitz:
The trade-off between familiarity and diversification. Management Science,
58:253–272,2012.
[56] M.W.Brandtand P.Santa-Clara. Dynamicportfolioselectionbyaugmenting
theassetspace. Journalof Finance,61:2187–2217,2006.
[57] L.Breiman. Baggingpredictors. Machine Learning,24(2):123–140,1996.
[58] L. Breiman. Prediction games and arcing algorithms. Neural Computation,
11(7):1493–1517,1999.


<!-- Page 68 -->

Bibliography 55
[59] D.R. Brillinger. Time Series: Data Analysis and Theory. Expanded edition.
Holden-Day, San Francisco,1981.
[60] W.Brock, J.Lakonishok, and B.Le Baron. Simpletechnicaltradingrulesand
thestochasticpropertiesofstockreturns. Journalof Finance,47:1731–1764,
1992.
[61] J. Brodie, I. Daubechies, C. De Mol, D. Giannone, and I. Loris. Sparse and
stable Markowitzportfolios. Proceedingsofthe National Academyof Sciences,
106:12267–12272,2009.
[62] C.Brownlees, F.Cipollini, and G.M.Gallo. Intra-dailyvolumemodelingand
prediction for algorithmic trading. The Journal of Financial Econometrics,
9:489–518,2011.
[63] B.Bruder, N.Gaussel, J.-C.Richard, and T.Roncalli. Regularizationofport-
folioallocation. White Paper Issue#10,2013.
[64] E.Bussetiand S.Boyd. Volume Weighted Average Price Optimal Execution.
unpublished, Stanford University,2015.
[65] J.Y. Campbell, S.J. Grossman, and J. Wang. Trading volume and serial cor-
relationinstockreturns. The Quarterly Journalof Economics,108:905–939,
1993.
[66] J.Y.Campbell, A.W.Lo, and A.C.Mac Kinlay. The Econometricsof Financial
Markets. Princeton University Press, New Jersey,1996.
[67] C. Cao, O. Hansch, and X. Wang. The information content of an open limit
orderbook. The Journalof Futures Markets,29:16–41,2009.
[68] M.M. Carhart. On persistence in mutual fund performance. Journal of
Finance,52(1):57–82,1997.
[69] M. Centoni and G. Cubadda. Modeling co-movements of economic time
series:Aselectivesurvey. Statistica,71:267–293,2011.
[70] A.P. Chaboud, B. Chiquoine, E. Hjalmarsson, and C. Vega. Rise of the
machines: Algorithmic trading in the foreign exchange market. Journal of
Finance,69:2045–2084,2014.
[71] B. Chakrabarty, P.C. Moulton, and A. Shkilko. Short sales, long sales, and
the Lee-Ready trade classification algorithm revisited. Journal of Financial
Markets,15(4):467–491,2012.
[72] K. Chan and W.-M. Fong. Trade size, order imbalance and the volatility-
volumerelation. Journalof Financial Economics,57:247–273,2000.
[73] L. Chan and J. Lakonishok. Institutional equity trading costs, NYSE versus
Nasdaq. Journalof Finance,52:713–735,1997.


<!-- Page 69 -->

56 Bibliography
[74] L.K.C.Chanand J.Lakonishok. Thebehaviorofstockpricesaroundinstitu-
tionaltrades. Journalof Finance,50:1147–1174,1995.
[75] L.K.C.Chanand J.Lakonishok. Institutionalequitytradingcosts:NYSEver-
sus Nasdaq. Journalof Finance,52(2):176–190,1997.
[76] N.F.Chen, R.Roll, and S.A.Ross. Economicforcesandthestockmarket. The
Journalof Business,59:383–403,1986.
[77] S.Chib. Estimationandcomparisonofmultiplechange-pointmodels. Journal
of Econometrics,86:221–241,1998.
[78] C.Chiyachantana, P.K.Jain, C.Jiang, and R.A.Wood. Internationalevidence
oninstitutionaltradingbehaviorandpriceimpact. Journalof Finance,59:869–
898,2004.
[79] T.Chordia, R.Roll, and A.Subrahmanyam. Commonalityinliquidity. Journal
of Financial Economics,56:3–28,2000.
[80] T.Chordia, R.Roll, and A.Subrahmanyam. Orderimbalance, liquidity, and
marketreturns. Journalof Financial Economics,65:111–130,2002.
[81] P.K.Clark. Asubordinatedstochasticprocessmodelwithfinitevariancefor
speculativeprices. Econometrica,41:135–155,1973.
[82] J. Conrad and G. Kaul. An anatomy of trading strategies. The Review of
Financial Studies,11:489–519,1998.
[83] J.Conradand S.Wahal. Thetermstructureofliquidityprovision. Journalof
Financial Economics,136:239–259,2020.
[84] J.S.Conrad, A.Hameed, and C.Niden. Volumeandautocovariancesinshort-
horizonindividualsecurityreturns. Journalof Finance,49:1305–1329,1994.
[85] A.Constantinos, J.A.Donkas, and A.Subrahmanyam. Cognitivedissonance,
sentiment and momentum. Journal of Financial and Quantitative Analysis,
46:245–275,2013.
[86] R. Cont and A. Kukanov. Optimal order placement in limit order markets.
Quantitative Finance,17:21–39,2017.
[87] R.Cont, A.Kukanov, and S.Stoikov. Thepriceimpactoforderbookevents.
The Journalof Financial Econometrics,12:47–88,2014.
[88] R.Cont, S.Stoikov, and R.Talreja. Astochasticmodelfororderbookdynam-
ics. Operations Research,58:549–563,2010.
[89] M. Cooper. Filter values based on price and volume in individual security
overreaction. The Reviewof Financial Studies,12:901–935,1999.


<!-- Page 70 -->

Bibliography 57
[90] S.A.Corwinand P.Schultz. Asimplewaytoestimatebid-askspreadsfrom
dailyhighandlowprices. Journalof Finance,67:719–759,2012.
[91] A.Cowles. Canstockmarketforecastersforecast. Econometrica,1:309–324,
1933.
[92] A.Cowles. Stockmarketforecasting. Econometrica,12:206–214,1944.
[93] D.R.Coxand P.A.W.Lewis. The Statistical Analysisof Seriesof Events. Chap-
manand Hall, London,1966.
[94] G. Creamer and Y. Freund. Automated trading with boosting and expert
weighting. Quantitative Finance,4:401–420,2010.
[95] M.Cremersand D.Weinbaum. Deviationsfromput-callparityandstockreturn
predictability. Journal of Financial and Quantitative Analysis, 45:335–367,
2010.
[96] D.M. Cutler, J.M. Poterba, and L.H. Summers. What Moves Stock Prices?,
volume15. National Bureauof Economic Research Cambridge, Mass., USA,
1989.
[97] S.Da, J.Engelberg, and P.Gao. Insearchofattention. Journalof Finance,
66(5):1461–1499,2011.
[98] Z.Da, J.Engelberg, and P.Gao. Thesumofallfearsinvestorsentimentand
assetprices. The Reviewof Financial Studies,28(1):1–32,2015.
[99] R.Dahlhausand S.Subba Rao. Statisticalinferencefortime-varying ARCH
processes. The Annalsof Statistics,34:1075–1114,2006.
[100] D.J. Daley and D. Vere-Jones. An Introduction to the Theory of Point Pro-
cesses, Volume I:Elementary Theoryand Methods. Springer, New York,2003.
[101] H.E.Daniels. Autocorrelationbetweenfirstdifferencesofmid-ranges. Econo-
metrica, pages215–219,1966.
[102] S.R.Dasand M.Y.Chen. Yahoo!for Amazon:Sentimentextractionfromsmall
talkontheweb. Management Science,53:1375–1388,2007.
[103] B.J.De Long, A.Shleifer, L.H.Summers, and R.J.Waldmann. Noisetraderrisk
infinancialmarkets. The Journalof Political Economy,98:703–738,1990.
[104] V.De Miguel, L.Garlappi, and R.Uppal. Optimalversusnaïvediversification:
Howinefficientisthe1∕𝑁portfoliostrategy?The Reviewof Financial Studies,
22:1915–1953,2009.
[105] A.P. Dempster, N.-M. Laird, and D.B. Rubin. Maximum likelihood from
incompletedataviathe EMalgorithm. Journalofthe Royal Statistical Society,
Series B,39:1–38,1977.


<!-- Page 71 -->

58 Bibliography
[106] B.Doand R.Faff. Doessimplepairstradingstillwork?The Financial Analysts
Journal,66(4):83–95,2010.
[107] I.Domowitz, J.Glen, and A.Madhavan. Liquidity, volatilityandequitytrading
costsacrosscountriesandovertime. International Finance,4:221–255,2001.
[108] I.Domowitzand H.Yegerman. Thecostofalgorithmictrading:Afirstlook
atcomparativeperformance. Algorithmic Trading:Precision, Control, Execu-
tion,2005.
[109] A. Dufour and R. Engle. Time and the price impact of a trade. Journal of
Finance,55(2):467–498,2000.
[110] D. Easley, M.L. De Prado, and M. O’Hara. Flow toxicity and liquidity in a
highfrequencyworld. The Reviewof Financial Studies,25:1457–1493,2012.
[111] D.Easley, M.L.De Prado, and M.O’Hara. Optimalexecutionhorizon. Math-
ematical Finance,25:640–672,2015.
[112] D.Easleyand M.O’Hara. Price, tradesize, andinformationinsecuritymar-
kets. Journalof Financial Economics,19:69–90,1987.
[113] D.Easleyand M.O’Hara. Timeandtheprocessofsecuritypriceadjustment.
Journalof Finance,19:69–90,1992.
[114] C.Eckartand G.Young. Theapproximationofonematrixbyanotheroflower
rank. Psychometrika,1:211–218,1936.
[115] B. Efron, T. Hastie, I. Johnstone, and R. Tibshirani. Least angle regression.
Annalsof Statistics,32:407–499,2004.
[116] J.Engelberg, P.Gao, and R.Jagannathan. Ananatomyofpairstrading:therole
ofidiosyncraticnews, commoninformationandliquidity. In Third Singapore
International Conferenceon Finance,2009.
[117] R.Engleand R.Ferstenberg. Executionrisk. The Journalof Portfolio Man-
agement,33(4):34–44,2007.
[118] R. Engle and F.K. Kroner. Multivariate simultaneous generalized ARCH.
Econometric Theory,11:122–150,1995.
[119] R. Engle and A. Lunde. Trades and quotes: A bivariate point process. The
Journalof Financial Economics,1:159–188,2003.
[120] R.F.Engle. Autoregressiveconditionalheteroscedasticitywithestimatesofthe
varianceof United Kingdominflations. Econometrica,50:987–1007,1982.
[121] R.F.Engle, R.Ferstenberg, and J.R.Russell. Measuringandmodelingexecu-
tioncostandrisk. The Journalof Portfolio Management,38(2):14–28,2012.


<!-- Page 72 -->

Bibliography 59
[122] R.F. Engle and C.W.J. Granger. Co-integration and error correction: Repre-
sentation, estimation, andtesting. Econometrica,55:251–276,1987.
[123] R.F.Engleand S.Kozicki. Testingforcommonfeatures. Journalof Business
&Economic Statistics,11(4):369–380,1993.
[124] R.F. Engle and D. Kraft. Multiperiod Forecast Error Variances of Inflation
Estimatedfrom ARCHModels, in:A.Zellner, ed:Applied Time Series Analysis
of Economic Data. Bureauofthe Census, Washington, DC,1983.
[125] R.F. Engle and J.R. Russell. Autoregressive conditioned duration: A new
modelforirregularlyspacedtransactiondata. Econometrica,66:1127–1162,
1998.
[126] R.F.Engleand R.Susmel. Commonvolatilityininternationalequitymarkets.
Journalof Business&Economic Statistics,11(2):167–176,1993.
[127] R.F.Engleand S.Kozicki. Testingforcommonfeatures. Journalof Business
&Economic Statistics,11:369–380,1993.
[128] R.F. Engle, V.K. Ng, and M. Rothschild. Asset pricing with a factor ARCH
covariancestructure:Empiricalestimatesfortreasurybills. Journalof Econo-
metrics,45:213–218,1990.
[129] T.W.Eppsand M.L.Epps. Thestochasticdependenceofsecuritypricechanges
andtransactionvolumes:Implicationsforthemixture-of-distributionhypoth-
esis. Econometrica,44:305–321,1976.
[130] T.W.Epps. Co-movementsinstockpricesintheveryshortrun. Journalofthe
American Statistical Association,74:291–298,1979.
[131] F.J. Fabozzi, S.M. Focardi, and P.N. Kolm. Quantitative Equity Investing:
Techniquesand Strategies. Wileyand Sons,2006.
[132] E.F.Famaand M.E.Blume. Filterrulesandstock-markettrading. The Journal
of Business,39:226–241,1966.
[133] E.F. Fama and K.R. French. A five-factor after pricing model. Journal of
Financial Economics,116:1–22,2015.
[134] E.F. Fama and K.R. French. International tests of a five-factor asset pricing
model. Fama-Miller Working Paper,2015.
[135] J.Fan, Y.Fan, and J.Lv. Highdimensionalcovariancematrixestimationusing
afactormodel. Journalof Econometrics,147:187–197,2008.
[136] J. Fan, F. Han, H. Liu, and B. Vickers. Robust inference of risks of large
portfolios. Journalof Econometrics,194:298–308,2016.


<!-- Page 73 -->

60 Bibliography
[137] J. Fan, J. Zhang, H. Liu, and K. Yu. Vast portfolio selection with cross-
exposure constraints. Journal of the American Statistical Association,
107:592–606,2012.
[138] A.Faragoand E.Hjalmarsson. Stockpriceco-movementandthefoundations
ofpairstrading. Journalof Financialand Quantitative Analysis,54:629–665,
2019.
[139] J.D. Farmer, A. Gerig, F. Lillo, and H. Waelbroeck. How efficiency shapes
marketimpact. Quantitative Finance,11:1743–1758,2013.
[140] J.D. Farmer, L. Gillemot, F. Lillo, S. Mike, and A. Sen. What really causes
largepricechanges? Quantitative Finance,4:383–397,2004.
[141] W.E.Fersonand A.F.Siegel. Theuseofconditioninginformationinportfolios.
Journalof Finance,56:967–982,2001.
[142] T. Foucault and A.J. Menkveld. Competition for order flow and smart order
routingsystems. Journalof Finance, pages119–157,2008.
[143] A.Frazzini, R.Israel, and T.J.Moskowitz. Tradingcosts (unpublished),2018.
[144] A. Frazzini and L.H. Pedersen. Betting against beta. Journal of Financial
Economics,111:1–25,2014.
[145] Y. Freund and R.E. Schapire. A decision-theoretic generalization of on-line
learningandanapplicationtoboosting. In European Conferenceon Compu-
tational Learning Theory, pages23–37.Springer,1995.
[146] Y.Freundand R.E.Schapire. Experimentswithanewboostingalgorithm. In
International Conference on Machine Learning, volume 96, pages 148–156.
Morgan Kaufmann Publishers Inc., San Francisco, CA,1996.
[147] Y. Freund and R.E. Schapire. A decision-theoretic generalization of on-line
learningandanapplicationtoboosting. In Journalof Computerand System
Sciences, volume55, pages119–139,1997.
[148] J. Friedman. Greedy function approximation: A gradient boosting machine.
Annalsof Statistics, pages1189–1232,2001.
[149] J.Friedman, T.Hastie, and R.Tibshirani. Additivelogisticregression:Asta-
tisticalviewofboosting (withdiscussionandarejoinderbytheauthors). The
Annalsof Statistics,28(2):337–407,2000.
[150] A. Frino, E. Jarnecic, and A. Lepone. The determinants of price impact of
blocktrades:Furtherevidence. Abacus,43(2):94–106,2007.
[151] K.A.Frootand E.M.Dabora. Howarestockpricesaffectedbythelocationof
trade? Journalof Financial Economics,53(2):189–216,1999.


<!-- Page 74 -->

Bibliography 61
[152] P.Fryzlewicz, T.Sapatinas, and S.Subba Rao. Normalizedleast-squaredesti-
mationintime-varying ARCHmodels. The Annalsof Statistics,36:742–786,
2008.
[153] W.A. Fuller. Introduction to Statistical Time Series, Second Edition. John
Wiley, New York,1996.
[154] L.Gagnonand G.A.Karolyi. Multi-markettradingandarbitrage. Journalof
Financial Economics,97:53–80,2010.
[155] A.R. Gallant, P.E. Rossi, and G. Tauchen. Nonlinear dynamic structures.
Econometrica,61:871–908,1993.
[156] L.Gao, Y.Han, S.Z.Li, and G.Zhou. Marketintradaymomentum. Journal
of Financial Economics,129:394–414,2018.
[157] N.Gârleanuand L.H.Pedersen. Dynamictradingwithpredictablereturnsand
transactioncosts. The Journalof Finance,68:2309–2340,2013.
[158] M.B.Garmanand M.J.Klass. Ontheestimationofsecuritypricevolatilities
fromhistoricaldata. The Journalof Business,53:67–78,1980.
[159] P.H. Garthwaite. An interpretation of partial least squares. Journal of the
American Statistical Association,89:122–127,1994.
[160] E.Gatev, W.N.Goetzmann, and R.G.Rouwenhorst. Pairstrading:Performance
ofarelativevaluearbitragerule. The Reviewof Financial Studies,19:797–827,
2006.
[161] J.Gatheral. No-dynamic-arbitrageandmarketimpact. Quantitative Finance,
10:769–759,2010.
[162] R.Gencay. Thepredictabilityofsecurityreturnswithsimpletechnicaltrading
rules. Journalof Empirical Finance,5:347–359,1998.
[163] S.Gervais, R.Kaniel, and D.H.Mingelgrin. Thehigh-volumereturnpremium.
Journalof Finance,56(3):877–919,2001.
[164] S.Gervais, R.Kaniel, and D.H.Mingelgrin. Thehigh-volumereturnpremium.
Journalof Finance,56:877–919,2001.
[165] E.Ghysels, C.Gourieroux, and J.Jasiak. Stochasticvolatilitydurationmodels.
Journalof Econometrics,119:413–433,2004.
[166] M.R.Gibbons, S.A.Ross, and J.Shanken. Atestoftheefficiencyofagiven
portfolio. Econometrica,57:1121–1152,1989.
[167] L.R.Glostenand P.R.Milgrom. Bid, askandtransactionpricesinaspecial-
istmarketwithheterogeneouslyinformedtraders. Journalof Financial Eco-
nomics,14:71–100,1985.


<!-- Page 75 -->

62 Bibliography
[168] I. Goodfellow, Y. Bengio, and A. Courville. Deep Learning. M.I.T. Press,
Boston,2016.
[169] R.C.Grinoldand R.N.Kahn. Active Portfolio Management, Second Edition.
Mc Graw-Hill,2000.
[170] A.Gross-Klussmannand N.Hautsch. Whenmachinesreadthenews:Using
automatedtextanalyticstoquantifyhighfrequencynews-impliedmarketreac-
tions. Journalof Empirical Finance,18:321–340,2011.
[171] A.D. Hall and N. Hautsch. Order aggressiveness and order book dynamics.
Empirical Economics,30:973–1005,2006.
[172] J.D.Hamilton. Anewapproachtotheeconomicanalysisofnonstationarytime
seriesandthebusinesscycle. Econometrica,57:357–384,1989.
[173] J.D.Hamilton. Analysisoftimeseriessubjecttochangesinregime. Journal
of Econometrics,45:39–70,1990.
[174] J.D.Hamiton. Macroeconomicregimesandregimeshifts. In J.B.Taylorand
H. Uhlig, editors, Handbook of Macroeconomics, chapter 3, pages 153–201.
Elsevier,2016.
[175] Y. Han, K. Yang, and G. Zhou. A new anomaly: The cross-sectional prof-
itabilityoftechnicalanalysis. Journalof Financialand Quantitative Analysis,
48:1433–1461,2013.
[176] P.R. Hansen. A test for superior predictive ability. Journal of Business &
Economic Statistics,23:365–380,2005.
[177] M. O’ Hara and M. Ye. Is market fragmentation harming market quality?
Journalof Financial Economics, pages454–474,2011.
[178] L.Harris. Tradingand Exchanges:Market Microstructurefor Practitioners.
Oxford University Press, New York,2003.
[179] A.C. Harvey. Forecasting, Structural Time Series Models and the Kalman
Filter. Cambridge University Press, Cambridge,1989.
[180] J.Hasbrouck. Measuringtheinformationcontentofstocktrades. Journalof
Financial Economics,46:179–207,1991.
[181] J. Hasbrouck. One security, many markets: Determining the contributions to
pricediscovery. Journalof Finance,50(4):1175–1199,1995.
[182] J. Hasbrouck and G. Saar. Technology and liquidity provision: The blurring
oftraditionaldefinitions. Journalof Financial Markets,12:143–172,2009.
[183] J.Hasbrouckand D.J.Seppi. Commonfactorsinprices, orderflows, andliq-
uidity. Journalof Financial Economics,59:383–411,2001.


<!-- Page 76 -->

Bibliography 63
[184] T.Hastie, R.Tibshirani, and J.Friedman. The Elementsof Statistical Learning;
Data Mining, Inferenceand Prediction, Second Edition. Springer-Verlag, New
York,2009.
[185] N.Hautschand R.Huang. Themarketimpactofalimitorder. The Journalof
Economic Dynamicsand Control,36:501–522,2012.
[186] A.G.Hawkes. Spectraofsomeself-excitingandmutuallyexcitingpointpro-
cesses. Biometrika,58:83–90,1971.
[187] A.G.Hawkes. Hawkesprocessesandtheirapplicationstofinance;areview.
Quantitative Finance,18:193–198,2018.
[188] X. He and R. Velu. Volume and volatility in a common-factor mixture of
distributionsmodel. Journalof Financialand Quantitative Analysis,49:33–
49,2014.
[189] I.S.Helland. Onthestructureofpartialleastsquaresregression. Communica-
tionsin Statistics, Simulationand Computation, B17:581–607,1988.
[190] I.S.Helland. Partialleastsquaresregressionandstatisticalmodels. Scandina-
vian Journalof Statistics,17:97–114,1990.
[191] T.Hendershott, J.Brogaard, and R.Riordon. Highfrequencytradingandprice
discovery. The Reviewof Financial Studies,27:2267–2306,2014.
[192] T. Hendershott, C.M. Jones, and A.J. Menkveld. Does algorithmic trading
improveliquidity? Journalof Finance,66(1):1–33,2011.
[193] T. Hendershott, C.M. Jones, and A.J. Menkveld. Does algorithmic trading
improveliquidity? Journalof Finance, pages1–3,2011.
[194] T. Hendershott and R. Riordan. Algorithmic trading and information.
http://faculty.haas.berkeley.edu/hender/ATInformation.pdf,2011.
[195] T.Hendershottand M.Seasholes. Marketmakerinventoriesandstockprices.
American Economic Review,97:210–214,2007.
[196] T. Ho and H.R. Stoll. Optimal dealer pricing under transactions and return
uncertainty. Journalof Financial Economics,9:47–73,1981.
[197] T.Ho, R.Schwartz, and D.Whitcomb. Thetradingdecisionandmarketclear-
ing under transaction price uncertainty. The Journal of Finance, 40:21–42,
1985.
[198] S. Hogan, R. Jarrow, M. Teo, and M. Warachka. Testing market efficiency
usingstatisticalarbitragewithapplicationtomomentumandvaluestrategies.
Journalof Financial Economics,73:525–565,2004.
[199] R.W.Holthausen, R.W.Leftwich, and D.Mayers. Large-blocktransactions, the
speedofresponse, andtemporaryandpermanentstock-priceeffect. Journal
of Financial Economics,26:71–95,1990.


<!-- Page 77 -->

64 Bibliography
[200] H.Hotelling. Analysisofacomplexofstatisticalvariablesintoprincipalcom-
ponents. Journalof Educational Psychology,4:417–441,498–520,1933.
[201] H.Hotelling. Themostpredictablecriterion. Journalof Educational Psychol-
ogy,26:139–142,1935.
[202] H. Hotelling. Relations between two sets of variables. Biometrika, 28:321–
322,1936.
[203] P.H.Hsu, Y.C.Hsu, and C.M.Kuan. Testingthe Predictive Abilityof Technical
Analysis Usinga New Stepwise Test Without Data Snooping Bias. Journalof
Empirical Finance,17:471–484,2010.
[204] Y.-P. Hu and R.S. Tsay. Principal volatility component analysis. Journal of
Business&Economic Statistics,32:153–164,2014.
[205] R.Huangand H.Stoll. Dealerversusauctionmarkets:Apairedcomparisonof
executioncostson NASDAQandthe NYSE.Journalof Financial Economics,
41:313–357,1996.
[206] W.Huang, C-A.Lehalle, and M.Rosenbaum. Simulatingandanalyzingorder
book data: The queue-reactive model. Journal of the American Statistical
Association,110:107–122,2015.
[207] D.Huang, F.Jiang, J.Tu, and G.Zhou. Investorsentimentaligned:Apowerful
predictorofstockreturns. The Reviewof Financial Studies,28:791–837,2015.
[208] G.Huberman. Familiaritybreedsinvestment. The Reviewof Financial Studies,
14:659–680,2001.
[209] G.Hubermanand W.Stanzl. Pricemanipulationandquasi-arbitrage. Econo-
metrica,72:1247–1275,2004.
[210] S.Hvidkjaer. Atrade-basedanalysisofmomentum. The Reviewof Financial
Studies,119:457–491,2006.
[211] R.Israel, T.Moskowitz, A.Ross, and L.Serban. Implementingmomentum:
Whathavewelearned. NBERWorking Paper,2017.
[212] R.Jagannathanand T.Ma. Riskreductioninlargeportfolios:Whyimposing
thewrongconstraintshelps. Journalof Finance,58:1651–1683,2003.
[213] C.M. Jarque and A.K. Bera. Efficient tests for normality, homoscedasticity
andserialindependenceofregressionresiduals. Economic Letters,6:255–259,
1980.
[214] N.Jegadeesh. Discussionof LMW(2000). Journalof Finance, pages1765–
1770,2000.
[215] N. Jegadeesh and S. Titman. Returns to buying winners and selling losers:
Implicationsforstockmarketefficiency. Journalof Finance,48:65–91,1993.


<!-- Page 78 -->

Bibliography 65
[216] N.Jegadeeshand S.Titman. Profitabilityofmomentumstrategies:Anevalu-
ationofalternativeexplanations. Journalof Finance,56:699–720,2001.
[217] N. Jegadeesh and S. Titman. Cross-sectional time series determinants of
momentumreturns. The Reviewof Financial Studies,15:143–157,2002.
[218] F.Jiang, J.Lee, X.Martin, and G.Zhou. Managersentimentandstockreturns.
Journalof Financial Economics,132:126–149,2019.
[219] W.Jiang, L.Shu, and D.W.Apley. Adaptive CUSUMprocedureswith EWMA-
basedshiftestimators. IIETransactions,40:992–1003,2008.
[220] J.D.Jobsonand B.Korkie. Estimationformarkowitzefficientportfolios. The
Journalof American Statistical Association,75:544–554,1980.
[221] S. Johansen. Statistical analysis of co-integration vectors. The Journal of
economicdynamicsandcontrol,12(2):231–254,1988.
[222] S. Johansen. Estimation and hypothesis testing of co-integration vectors in
Gaussianvectorautoregressivemodels. Econometrica:Journalofthe Econo-
metric Society, pages1551–1580,1991.
[223] C.Jones, G.Kaul, and M.Lipson. Information, tradingandvolatility. Journal
of Financial Economics,36:127–154,1994.
[224] L.P. Kaelbling, M.L. Littman, and A.W. Moore. Reinforcement learning: A
survey. Journalof Artificial Intelligence,4:237–285,1996.
[225] R.N.Kahnand M.Lemmon. Smartbeta:Theowner’smanual. The Journalof
Portfolio Management,41(2):76–83,2015.
[226] H.Kawakatsu. Directmultiperiodforecastingforalgorithmictrading. Journal
of Forecasting,37(1):83–101,2018.
[227] D.B.Keimand A.Madhavan. Theupstairsmarketforlarge-blocktransactions:
Analysisandmeasurementofpriceeffects. The Reviewof Financial Studies,
9:1–36,1996.
[228] D.BKeimand A.Madhavan. Transactioncostsandinvestmentstyle:Aninter-
exchange analysis of institutional equity trades. Journal of Financial Eco-
nomics,46:265–292,1997.
[229] J.L. Kelly. A new interpretation of information rate. Bell System Technical
Journal,35:917–926,1956.
[230] J.M. Keynes. The general theory of employment. The Quarterly Journal of
Economics, pages209–223,1937.
[231] P.D.Kochand T.W.Koch. Evolutionindynamiclinkagesacrossdailynational
stock indexes. Journal of International Money and Finance, 10.2:231–251,
1991.


<!-- Page 79 -->

66 Bibliography
[232] A. Kourtis. On the distribution and estimation of trading costs. Journal of
Empirical Finance,29:230–245,2014.
[233] P. Kratz and T. Schöneborn. Optimal liquidity in dark pools. Quantitative
Finance,14:1519–1539,2014.
[234] A. Kyle. Continuous time auctions and insider trading. Econometrics,
53:1315–1336,1985.
[235] T.L.Laiand H.Xing. Statistical Modelsand Methodsfor Financial Markets.
Springer,2008.
[236] T.L.Laiand H.Xing. Stochasticchange-point ARXGARCHmodelsandtheir
applications to econometric times series. Statistica Sinica, 23:1573–1594,
2013.
[237] T.L.Lai, H.Xing, and Z.Chen. Mean-varianceportfoliooptimizationwhen
meansandcovariancesareunknown. The Annalsof Applied Statistics,5:798–
823,2011.
[238] J.Lakonishok, A.Shleifer, and R.W.Vishny. Contrarianinvestment, extrapo-
lation, andrisk. Journalof Finance,49(5):1541–1578,1994.
[239] O. Ledoit and M. Wolf. Improved estimation of the covariance matrix of
stockreturnswithanapplicationtoportfolioselection. Journalof Empirical
Finance,10:603–621,2003.
[240] C.M.Leeand B.Swaminathan. Pricemomentumandtradingvolume. Journal
of Finance, LV:2017–2069,2000.
[241] C.M.Leeand M.Ready. Inferringtradedirectionfromintradaydata. Journal
of Finance,46:733–746,1991.
[242] J.Lewellen. Momentumandautocorrelationinstockreturns. The Reviewof
Financial Studies,15:65–91,2002.
[243] J.K.-S. Liew, S. Guo, and T. Zhang. Tweet sentiments and crowd-sourced
earningsestimatesasvaluablesourcesofinformationaroundearningsreleases.
The Journalof Alternative Investments, Winter Issue:1–20,2017.
[244] F.Lillo, J.D.Farmer, and R.Mantegna. Mastercurveforpriceimpactfunction.
Nature,421:129–130,2003.
[245] J.Lintner. Thevaluationofriskyassetsandtheselectionofriskyinvestment
in stock portfolios and capital budgets. Review of Economics and Statistics,
47:13–37,1965.
[246] G.Llorente, R.Michaely, G.Saar, and J.Wang. Dynamicvolume-returnrela-
tion of individual stocks. The Review of Financial Studies, 15:1005–1047,
2002.


<!-- Page 80 -->

Bibliography 67
[247] A.W. Lo, H. Mamaysky, and J. Wang. Foundation of technical analysis:
Computationalalgorithms, statisticalinferenceandempiricalimplementation.
Journalof Finance, LV(4):1705–1765,2000.
[248] A.W. Lo. The statistics of Sharpe ratios. Financial Analysis Journal, pages
36–52,2002.
[249] A.W.Loand A.C.Mac Kinlay. Whenarecontrarianprofitsduetostockmarket
overreaction? The Reviewof Financial Studies,3:175–205,1990.
[250] A.W.Lo, A.C.Mac Kinlay, and J.Zhang. Econometricmodelsoflimit-order
executions. Journalof Financial Economics,65:31–71,2002.
[251] T.F.Loeb. Tradingcost:Thecriticallinkbetweeninvestmentinformationand
results. Financial Analyst Journal,39(3):39–44,1983.
[252] M.Lopezde Prado. Advancesin Financial Machine Learning. John Wiley&
Sons, New Jersey,2018.
[253] J.Lorenzand R.Almgren. Mean-varianceoptimaladaptiveexecution. Applied
Mathematical Finance,18(4):395–422,2011.
[254] A.Madhavan. Marketmicrostructure. Journalof Financial Markets,3:205–
258,2000.
[255] A.Madhavan. VWAPstrategies. Trading,1:32–39,2002.
[256] A. Madhavan, M. Richardson, and M. Roomans. Why do security prices
change?Atransaction-levelanalysisof NYSEstocks. The Reviewof Financial
Studies,10(4):1035–1064,1997.
[257] C. Maglaras, C.C. Moallemi, and H. Zheng. Queueing dynamics and state
spacecollapseinfragmentedlimitorderbookmarkets. Operations Research
(toappear),2019.
[258] B.G. Malkiel. A Random Walk Down Wall Street: The Time-Tested Strategy
for Successful Investing,10 th Edition. Norton,2012.
[259] B. Mandelbrot. The variation of certain speculative prices. The Journal of
Business,36:294–319,1963.
[260] H. Markowitz. Portfolio Selection: Efficient Diversification of Investments.
Wiley:New York,1959.
[261] M. Martens and D. van Dijk. Measuring volatility with the realized range.
Journalof Econometrics,138:181–207,2007.
[262] R. Mc Culloch and R. Tsay. Nonlinearity in high-frequency financial data
and hierarchical models. Studies in Nonlinear Dynamics and Econometrics,
5:1067–1077,2001.


<!-- Page 81 -->

68 Bibliography
[263] H.Mendelson. Marketbehaviorinaclearinghouse. Econometrica:Journalof
the Econometric Society,1505–1524,1982.
[264] L.Menkhoff, L.Sarno, M.Schmeling, and A.Schrimpf. Currencymomentum
strategies. Journalof Financial Economics,106:660–684,2012.
[265] L.Menkhoffand M.P.Taylor. Theobstinatepassionofforeignexchangepro-
fessionals:Technicalanalysis. The Journalof Economic Literature, XLV:936–
972,2007.
[266] A.J.Menkveld, S.J.Koopman, and A.Lucas. Modelingaround-the-clockprice
discoveryforcross-listedstocksusingstatespacemethods. Journalof Busi-
ness&Economic Statistics,25:213–225,2007.
[267] R.C. Merton. On estimating the expected return on the market. Journal of
Financial Economics,8:323–336,1980.
[268] R.T. Merton. An intertemporal capital asset pricing model. Econometrica,
41:867–887,1973.
[269] R.O.Michaud. Efficient Asset Management. Harvard Business School Press,
Boston,1989.
[270] G. Mitra and L. Mitra. The Handbook of News Analytics in Finance (Ed).
Wiley Finance,2011.
[271] T. Moorman. An empirical investigation of methods to reduce transaction
costs. Journalof Empirical Finance,29:230–245,2014.
[272] E.Moro, J.Vicente, L.G.Moyano, A.Gerig, J.D.Farmer, G.Vaglica, F.Lillo,
and R.N. Mantegna. Market impact and trading profile of hidden orders in
stockmarkets. Physical Review E,80(6):066102,2009.
[273] T.J.Moskowitz, Y.H.Ooi, and L.H.Pedersen. Time Series Momentum. Jour-
nalof Financial Economics,104:228–250,2012.
[274] A.A. Obizhaeva and J. Wang. Optimal trading strategy and supply/demand
dynamics. Journalof Financial Markets,16:1–32,2013.
[275] M. O’Hara. Presidential address: Liquidity and price discovery. Journal of
Finance,58:1335–1354,2003.
[276] M.O’Hara. Highfrequencymarketmicrostructure. Journalof Financial Eco-
nomics,116:257–270,2015.
[277] J.Okunevand D.White. Domomentum-basedstrategiesstillworkinforeign
currency markets? Journal of Financial and Quantitative Analysis, 38:425–
447,2003.


<!-- Page 82 -->

Bibliography 69
[278] J.K.Ord, A.B.Koehler, and R.D.Snyder. Estimationandpredictionforaclass
of dynamic nonlinear statistical models. Journal of the American Statistical
Association,92(440):1621–1629,1997.
[279] D.A. Pachamanova and F.J. Fabozzi. Recent trends in equity portfolio con-
structionanalytics. The Journalof Portfolio Management,40:137–151,2014.
[280] A. Pardo and R. Pascual. On the hidden side of liquidity. The European
Journalof Finance,18:949–967,2012.
[281] C. Parlour and D. Seppi. Liquidity-based competition for order flow. The
Reviewof Financial Studies,16:301–343,2003.
[282] C.A.Parlourand D.J.Seppi. Limit Order Markets–ASurvey, In Handbook
of Financial Intermediationand Banking, Editedby A.Thakorand A.Boot.
Elsevier, Amsterdam,2008.
[283] L. Pastor and R.F. Stambough. The equity premium and structural breaks.
Journalof Finance,56:1207–1239,2001.
[284] A.J.Pattonand A.Timmermann. Monotonicityinassetreturns:Newtestswith
applicationstothetermstructure, the CAPM, andtheportfoliosorts. Journal
of Financial Economics,98:605–625,2010.
[285] R.L.Peterson. Tradingon Sentiment:The Powerof Mindsover Markets. Wiley
Finance,2016.
[286] M.J. Ready. Profits from technical trading rules. Financial Management,
Autumn:43–61,2002.
[287] G.C.Reinseland S.K.Ahn. Vectorautoregressivemodelswithunitrootsand
reducedrankstructure:estimation, likelihoodratiotest, andforecasting. The
Journalof Time Series Analysis,13(4):353–375,1992.
[288] G.C.Reinsel. Elementof Multivariate Time Series Analysis, Second Edition.
Springer-Verlag, New York,2002.
[289] G.C.Reinseland R.Velu. Multivariate Reduced-Rank Regression, Theoryand
Application. Springer-Verlag, New York,1998.
[290] L.C.G.Rogersand S.E.Satchell. Estimatingvariancefromhigh, lowandclos-
ingprices. Annalsof Applied Probability,1:504–512,1991.
[291] R.Roll. Asimplemodeloftheimplicitbid-askspreadinanefficientmarket.
Journalof Finance,39:1127–1139,1984.
[292] J.P.Romanoand M.Wolf. Stepwisemultipletestingasformalizeddatasnoop-
ing. Econometrica,73:1237–1282,2005.
[293] S.A.Ross. Thearbitragetheoryofcapitalassetpricing. The Journalof Eco-
nomic Theory,13:341–360,1976.


<!-- Page 83 -->

70 Bibliography
[294] I.Rosu. Adynamicmodelofthelimitorderbook. The Reviewof Financial
Studies,22:4601–4641,2009.
[295] T.H. Rydberg and N. Shephard. Dynamic trade-by-trade price movement:
Decompositionandmodels. Journalof Financial Econometrics,1:2–25,2003.
[296] S.Satchelland A.Snowcraft. Ademystificationofthe Black-Littermanmodel:
Managingquantitativeandtraditionalportfolioconstruction. Journalof Asset
Management,1:138–150,2000.
[297] V.Satish, A.Saxena, and M.Palmer. Predictingintradaytradingvolumeand
volumepercentages. The Journalof Trading,9:15–25,2014.
[298] M.Schneiderand F.Lillo. Cross-impactandno-dynamicsarbitrage. Quanti-
tative Finance,19:137–154,2019.
[299] G.Schwarz. Estimatingthedimensionofamodel. Annalsof Statistics,6:401–
404,1978.
[300] J.T.Scruggs. Noisetraderrisk:Evidencefromthesiamesetwins. Journalof
Financial Markets,10:76–105,2007.
[301] W.F.Sharpe. Capitalassetprices:Atheoryofmarketequilibriumundercon-
ditionsofrisk. Journalof Finance,19:425–442,1964.
[302] R.H.Shumwayand D.S.Stoffer. Arimamodels. In Time Series Analysisand
Its Applications, pages83–171.Springer,2011.
[303] D.Smith, N.Wang, Y.Wang, and E.J.Zychowicz. Sentimentandtheeffective-
nessof Technical Analysis:Evidencefromthe Hedge Fund Industry. Journal
of Financialand Quantitative Analysis,51:1991–2013,2016.
[304] E.Smith, D.J.Farmer, L.Gillemot, and S.Krishnamurthy. Statisticaltheory
ofthecontinuousdoubleauction. Quantitative Finance,3:481–514,2003.
[305] R. Stambaugh, J. Yu, and Y. Yuan. The short of it: Investor sentiment and
anomalies. Journalof Financial Economics,104:288–302,2012.
[306] M.Statman, S.Thorley, and K.Vorkink. Investoroverconfidenceandtrading
volume. The Reviewof Financial Studies,19(4):1531–1565,2006.
[307] S. Stoikov. The micro-price: A high-frequency estimator of future prices.
Quantitative Finance,18:1959–1966,2018.
[308] R.Sullivan, A.Timmermann, and H.White. Datasnooping, technicaltrading
ruleperformance, andthebootstrap. Journalof Finance,54:1647–1691,1999.
[309] R.S.Sultonand A.G.Barks. Reinforcement Learning, An Introduction, Second
Edition. MITPress, Boston,2018.


<!-- Page 84 -->

Bibliography 71
[310] R.J. Sweeney. Some new filter rule tests: Methods and results. Journal of
Financialand Quantitative Analysis,20(3):285–300,1988.
[311] G.E.Tauchenand M.Pitts. Thepricevariability-volumerelationshiponspec-
ulativemarkets. Econometrica,51:485–505,1983.
[312] P.C. Tetlock. Giving content to investor sentiment: The role of media in the
stockmarket. Journalof Finance,62(3):1139–1168,2007.
[313] I.M.Toke. Anintroductionto Hawkesprocesseswithapplicationstofinance,
2011. Lecture Notesfrom Ecole Centrale Paris, BNP, Paribas.
[314] B. Toth, Y. Lemperiere, C. Deremble, J. De Lataillade, J. Kockelkoren, and
J.-P.Bouchaud. Anomalouspriceimpactandthecriticalnatureofliquidityin
financialmarkets. Physical Review X,1(2):021006,2011.
[315] R.Tsay. Analysisof Financial Time Series, Third Edition. Wiley,2010.
[316] G.Tsoukalas, J.Wang, and K.Giesecke. Dynamicsportfolioexecution. Man-
agement Science,2019.
[317] J.Tuand G.Zhou. Markowitzmeetstalmud:Acombinationofsophisticated
andnaïvediversification. Journalof Financial Economics,99:204–215,2011.
[318] F.Vahidand R.F.Engle. Commontrendsandcommoncycles. The Journalof
Applied Econometrics,8:341–360,1993.
[319] R.Velu, A.Gretchika, M.Benaroch, D.Nehren, and K.Kuber. Market Impact:
To Trade Small or to Trade Seldom? Evidence from Algorithmic Execution
Data. Unpublished,2015.
[320] B.von Beschwitz, D.B.Keim, and M.Massa. Firstto“read”thenews:News
analytics and algorithmic trading. Review of Financial Studies (to appear),
2019.
[321] A.A.Weiss. ARMAmodelswith ARCHerrors. The Journalof Time Series
Analysis,5:129–143,1984.
[322] M.Westand J.Harrison. Bayesian Forecastingand Dynamic Models, Second
Edition. Springer-Verlag, New York,1997.
[323] H.White. Arealitycheckfordatasnooping. Econometrica,68:1097–1126,
1999.
[324] R.De Winneand C.D’Hondt. Hide-and-seekinthemarket:Placinganddetect-
inghiddenorder. Reviewof Finance,11:663–692,2007.
[325] H.Wold. PLSregression, volume6. Eds. N.L.Johnsonand S.Kotz,1984.
[326] H.Working. Noteonthecorrelationoffirstdifferencesofaveragesinarandom
chain. Econometrica: Journal of the Econometric Society, pages 916–918,
1960.


<!-- Page 85 -->

72 Bibliography
[327] D.Yangand Q.Zhang. Drift-independentvolatilityestimationbasedonhigh,
low, openandcloseprices. The Journalof Business,73:477–491,2000.
[328] J.W.Yang. Transactiondurationandasymmetricpriceimpactoftrades-Evi-
dencefrom Australia. Journalof Empirical Finance,18:91–102,2011.
[329] J.Yuand Y.Yuan. Investorsentimentandthemean–variancerelation. Journal
of Financial Economics,100:367–381,2011.
[330] E.Zarinelli, M.Treccani, J.D.Farmer, and F.Lillo. Beyondthesquareroot:
Evidenceforlogarithmicdependenceofmarketimpactonsizeandparticipa-
tionrate. Market Microstructureand Liquidity,1:1–31,2015.
[331] G. Zhou. Measuring investor sentiment. Annual Review of Financial Eco-
nomics,10:239–259,2018.
[332] Y.Zhuand G.Zhou. Technicalanalysis:Anassetallocationperspectiveonthe
useofmovingaverages. Journalof Financial Economics,92:519–544,2009.
