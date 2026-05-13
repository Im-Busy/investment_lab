<!-- Page 1 -->

Received June29,2021, accepted July27,2021, dateofpublication August3,2021, dateofcurrentversion August23,2021.
Digital Object Identifier10.1109/ACCESS.2021.3102495
Building a Calendar of Events Database by
Analyzing Financial Spikes
PRAKASHK.AITHAL ,(Member, IEEE), U.DINESHACHARYA , M.GEETHA ,(Member, IEEE),
ANDPARTHIVMENON
Departmentof Computer Scienceand Engineering, Manipal Instituteof Technology, Manipal Academyof Higher Education, Manipal576104, India
Correspondingauthor:M.Geetha (geetha.maiya@manipal.edu)
ABSTRACT An event is a piece of news that triggers a change in stock prices. Here, an event study is
undertakentocapturetheeffectofabnormalreturnsduetoanevent. Theeventcanaffectthestockmarketin
thelongtermorshortterm. Eventresearchisrelevanttoboththeefficientmarkethypothesisandbehavioral
finance. Inthisstudy, wecollecteddatafromwebsitesthatmanagefinancialandeconomicdata, performeda
sentimentanalysis, andcorrelatednewsarticledatawithchangesinaparticularcompany’sstockpricesinthe
stockmarket. Datawerecollectedfromtwowell-knownfinancialnewswebsites. Weobservedacorrelation
betweenstockpricesandnewsitems. Aneventperiodofonedaywasconsideredforthestudy. Theregression
equation determined the relationship between stock returns and polarity and subjectivity. Bayesian model
averagingwasperformedtoidentifytheeffectsofpolarityandsubjectivityonstockreturns. Time-seriesdata
were decomposed into components and detrended via regression. Prominent keywords and their polarity
values for a particular day were plotted. An event enhanced some stock returns while adversely affecting
otherstocks. Wefoundavariationrangeof-400 to200 fordifferentcompanystocksfortheselectedperiod.
INDEXTERMS Eventdatabase, socialmediadata, sentimentanalysis, dataanalytics, eventstudies.
I. INTRODUCTION decrease the value of a stock. Events have either short-term
Data Scienceinvolvesextractinginformationfromrawdata, consequencesoraffectstocksoverthelongrun. Eventsthat
converting information into knowledge and knowledge into affect the stock market for less than a week are considered
an actionable plan, which has broad applications for fields to be short-term events and that rest are long-term events.
rangingfromagricultureandspacesciencetoartsandfinance An event can affect either a particular stock or the entire
andwhichappliestoallfieldsinwhichdataareused. stockmarket. Thepresenteventstudyaimstodeterminethe
Finance is essential to a nation’s growth. Primarily two abnormalstockreturnattributabletonewinformationcarried
kindsoffinancialdataareused:quantitativedataonvariables byanevent.
such as stock prices, the volume of stock, and the PE ratio Financial events provide both quantitative and qualita-
and qualitative data drawn from material such as director tive data. Qualitative data such as financial news must be
reports, auditor reports, financial statements, and financial quantified. One can apply a sentiment analysis to quantify
news. Quantitative data reveal a stock’s movement, while financial news. Sentiment can be measured based on tone,
qualitativedataunearthinvestorsentiment, whichisthedriv- also known as polarity, or subjectivity. Polarity values vary
ingforcebehindstockmovement. from -1 to 1 where a value of -1 denotes extremely nega-
Stock prices fluctuate based on financial events captured tivesentimentand+1 denotestotalpositivesentiment. Sub-
byqualitativedata. Eventscanincludemanagementchanges, jectivity levels vary from 0 to 1. The lower the value of
declarations of dividends, festivals, natural calamities, pan- subjectivity is, the more meaning can be extracted from the
demics, recessions, budget and earnings releases, and quar- news.
terlyandyearlystatements. Someevents, suchasfestivalsand There is a correlation between financial news and stock
yearlystatements, occuronceayear. Changesinmanagement prices. If the correlation between news and the stock dif-
have either positive or adverse effects. Dividends always ference is greater, news influences the stock market. Event
studies assume the semistrong form of market efficiency.
The associate editor coordinating the review of this manuscript and The semistrong form of market efficiency applies two
approvingitforpublicationwas Francesco Benedetto . assumptions[1].
114192 Thisworkislicensedundera Creative Commons Attribution4.0 License. Formoreinformation, seehttps://creativecommons.org/licenses/by/4.0/ VOLUME9,2021


<!-- Page 2 -->

P.K.Aithaletal.:Building Calendarof Events Databaseby Analyzing Financial Spikes
• All publicly available information is reflected in stock
prices
• Stockpricesvarywhennewinformationarrives
Thelatterassumptionisutilizedtopredictthefuturevalueof
a stock in advance. An event changes a stock price, and an
eventstudycanthenbeutilizedtoanalyzefuturecashflows.
A. BAYESIANMODELAVERAGING
An analysis is conducted by first selecting the best model
based on a particular criterion and then learning about
the parameters of the chosen model. Any uncertainties of
the model selection process are ignored. Contrary to this
approach, Bayesian model averaging (BMA) learns param-
eters for all candidate models and allows one to select a
model with specific posterior probabilities. BMA offers the
followingadvantagesoverothermodels:
1) BMAcapturestheuncertaintyofamodel.
2) BMAreduceserror.
3) BMAretainsallmodeluncertaintyuntilthefinalstage.
4) BMAcontinuouslyadjustsmodelweights.
5) BMAusesasetofmodels, whichispreferabletousing
asinglemodel.
B. BAYESIANSTRUCTURALTIMESERIES
FIGURE1. Buildingandanalysisofcalendareventsdatabase.
The Bayesian Structural Time Series (BSTS) is a tool used
toidentifythecausalrelationship. Itutilizes GPUifpresent
Researchers look forward in time and predict the expected
else runs on CPU. It requires pre-period and post-period as
value a firm will derive from a corporate action announced
argumentsalongwiththedata.
to the public. This forward-looking focus of event studies
makes them more potent than other metrics such as return
C. TIMESERIESCOMPONENTS
oninvestment (ROI) andprofits. Investorsrespondpositively
It is helpful to decompose the time series into systematic
to product launches, sponsorships, and mergers, while they
andnonsystematiccomponents. Systematiccomponentscan
respondnegativelytoeventssuchasproductrecalls.[2].Ithas
bemodeledaseitherconsistentorrecurring. Nonsystematic
been observed that historical stock returns are affected by
elementsarerandomandcannotbemodeled. Thetimeseries
calendarevents. Wheneventstudieshavebeenconductedin
consistsofthefollowingcomponents:
developedcountries, littlerelatedliteratureisavailablefrom
1) Level developing nations. The stock market declines on Mondays
2) Trend andstrengthenson Friday. Thestartofanewmonthoryear
3) Seasonality alsoaffectsthestockmarket[3].Thepresenceoflong-term
4) Noise abnormalreturnscontradictstheefficientmarkethypothesis.
The level is the mean value of the time series. The trend is Differenttestingstrategiesareapplicabletoassetpricingand
eitherincreasingordecreasinginthetimeseries. Therepeat- buy-and-holdmethodologies[4].
ing short-term cycle is defined as seasonality. Randomness Toidentifytheabnormalreturnsoftheportfolioofstocks
present in the time series that cannot be modeled is defined affectedbyanevent, acontrolportfolioofstocksnotaffected
asnoise. by the event is considered as a benchmark, and the results
The rest of the paper is organized as follows. Section 2 ofbothportfoliosarecompared[5].Eventstudiescaneval-
reviews the related literature, section 3 elaborates on the uate a government’s policies, as markets react to new news
methodologyused, andsection4 presentstheresults. Conclu- immediatelyandadjusttheirvalue. Ifapolicyhasapositive
sionsandavenuesforfutureresearcharegiveninsection5. effectonthemarket, itwilleventuallysucceed. Speculation
onaneventalsohasamajorimpactonthestockmarket. The
II. LITERATURESURVEY stockmarketreactstospeculationandisaffectedlongbefore
Eventstudiesdetermineexcessreturnsattributabletoacor- an event is announced [6]. Event studies require the use of
porateevent. Eventscanrangefromanearningsreleasebya a window period under which an event is studied. An event
companyorbudgetannouncementtoanewproductlaunch, study evaluation is done based on the methodology utilized
announcement from a competing company, regular finan- andcaninvolvearegressionwithat-testortheuseofacapital
cialstatementorannouncementmadebyaregulatorybody. assetpricingmodel (CAPM) withat-test[7].
VOLUME9,2021 114193


<!-- Page 3 -->

P.K.Aithaletal.:Building Calendarof Events Databaseby Analyzing Financial Spikes
FIGURE2. Newsitemcollection.
An election is a major event that significantly influences arrives[12].Eventsdonothavestructureandcannotbestored
thestockmarket. Asetofstocksareaffectedbyanelection. inrelationaldatabases. Therefore, Not Only Structured Query
Electionshaveaneffectforapproximatelyfivemonthsbefore Language (No SQL) databases are utilized to store informa-
and after an event. The announcement of an election and tiononsuchevents. No SQLdatabasesrequirenodowntime
theelectionitselfgreatlyimpactthestockmarket[8].Event and are different from regular relational database manage-
studies have become the defacto standard for measuring ment systems (RDBMSs), as they can scale well at a lower
stockpricesduringandafteranevent. Abnormalreturnsare cost. A RDBMS has strict atomicity consistency isolation
determinedeitherastheresidualsfromstandardnormalized durability (ACID) properties, while No SQLdatabasesadhere
benchmarks or as a dummy variable in a regression equa- totheconsistencyavailabilityandpartitiontolerance (CAP)
tion [9]. News can be classified as repeating or surprising. theorem. Complexeventprocessingcannotbeachievedusing
Arepeatingeventcanbeeasilypredicted, andmarketsadjust a regular RDBMS, so one must in this case use No SQL
tosuchanevent. Asurprisingevent, ontheotherhand, leads databases for faster processing. Complex event processing
to abnormal returns. This is the case because such an event requireslowlatency, highthroughput, andtemporalandspa-
isunknown, andnoestimatehasbeenmade, whichresultsin tial event processing capabilities [13]. Stock prices move
riskthatmustbeaddressed[10]. with a correlated random walk rather than with an uncorre-
In[11], theauthorsconcludethatintradayabnormalreturns latedrandomwalk. Correlationscanbefoundthroughevent
are normally distributed, and Pattel test statistics function studies[14].
betterthanotherstatisticalmeasuresinmeasuringabnormal One comment posted on the Yahoo financial website
returns. The market responds to both good and bad news. almost crashed the Nevada company, after which the com-
Goodnewsreducesrisk, whileapieceofbadnewsincreases pany had to call a press conference to prove that its fun-
it. The efficient market hypothesis and behavioral finance damentals are strong. This incident demonstrates that event
applyinsomecasesandfailinothers. Bothhaveexplanatory studies are critical. The simplest measures used include the
power and must be explored. According to event studies, number of times a stock name is mentioned, the frequency
stocks will not alter prices unless and until a piece of news of certain keywords, and the sentiment of news about a
114194 VOLUME9,2021


<!-- Page 4 -->

P.K.Aithaletal.:Building Calendarof Events Databaseby Analyzing Financial Spikes
FIGURE3. Stockcollection.
givencompany[15].Therearetwotypesofevents, namely, change. Efficient market theory considers investors to be
simple events and complex events. A simple event occurs unemotional. Behavioral finance, on the other hand, treats
instantaneouslyandindependentofotherevents. Acomplex investors as an emotional entity and assumes news events
event is a set of events that are interrelated [16]. The tradi- to affect investments. Both the efficient market hypothesis
tionalinvestmentstrategytakesmonthsoryearstoarriveata and behavioral finance agree that news events are respon-
decision. Theadvancementofinformationtechnologyinthe sible for abnormal returns [18]. Text mining is used in
realmoffinancehasmadethedecision-makingprocessfaster. event studies, and it requires words to be segmented, and
Events can be analyzed in milliseconds using information spaces are used to separate words. Stop words need to
technology[17]. be removed. The sentiment of a sentence and frequency
The efficient market hypothesis states that market prices of words measure the quantum of the jump of stock
reflect current news and that as new news arrives, prices prices[19].
VOLUME9,2021 114195


<!-- Page 5 -->

P.K.Aithaletal.:Building Calendarof Events Databaseby Analyzing Financial Spikes
FIGURE4. Keywordsandtheirimpactoninvestorsentimentformeanpolarity.
Sentiment analysis is widely used in the field of finance. impactonstockmovement. Ifacompanycallsaconference
Themethodisalsoknownasopinionmining. Sentimentanal- call and the tone is positive, the stock of that company will
ysis is broadly classified into emotion recognition and tone movemultifoldinapositivedirection, resultinginanabnor-
analysisorpolarityanalysis. Opinionminingcanbeclassified malpositivereturn[23].Theglobalfinancialcrisisresultedin
based on whether a dictionary or machine learning is used governmentsinterveningandtakingregulatoryaction. Global
to arrive at an opinion. Machine learning-based approaches regulatory action is an event that affects the stock market
requiredomain-specificcontenttoworkcorrectly, whiledic- worldwide[24].
tionariesarecostly[20].Eventanalysismustbeconductedby The coronavirus pandemic has affected US asset prices.
batchprocessingofhistoricalevents. Theeventpatternmust The pandemic has resulted in high levels of volatility in
supportmathematical, logical, andstatisticalpatternsamong small and unstable markets. Stock markets are interrelated
events[21].Aneventstudyinvolvesseveralstepsandmany andsensitivetonews. Mediacoverageonthepandemichas
choicesmustbemadeateachstep. Thus, thereisnostandard resultedinanadversecrashinsomehighlyvolatilestocksand
meansofconductinganeventstudy. Thetimingofanevent unstable industries. The global stock markets are becoming
study and the event window size need to be defined by the interdependent. Evenwhenasinglestockmarketisaffected,
researcher. Surpriseeventsneedtobeevaluatedthroughevent thisimpactspreadstootherstockmarkets[25].[26]studied
studies[22]. corporate social responsibility with the help of a dictionary.
Ahistoricalanalysisof145100 newsarticleshasshowna Theauthorsshowedthatwhenafirmisnotresponsibleand
significantrelationshipbetweentoneandcompanies’perfor- thisresultsinanegativeevent, itsstockpricewillfalldrasti-
mance. However, asingleinstanceofnewsdoesnothavean cally. Eventandfirmperformancearedirectlyproportional.
114196 VOLUME9,2021


<!-- Page 6 -->

P.K.Aithaletal.:Building Calendarof Events Databaseby Analyzing Financial Spikes
FIGURE5. Keywordsandtheirimpactoninvestorsentimentformeansubjectivity.
Narrative economics has developed into a field of its own. Withtheemergenceofbigdatatechnologies, thestorageand
Anarrativeeventinfluencesastock, andnumerousexamples processingofqualitativedatahavebecomemoreaccessible.
of this phenomenon prove this fact [27]. The pandemic has Thefocusoftheresearcherhasthusswitchedfromtheuseof
causedalosstotheeconomy. Ahistoricalinfluenzarevealed quantitativedatatotheuseofqualitativedata[31].
thatapandemichaseffectssimilartothoseofinfluenzabut [32]calculateeconomicuncertaintybasedonnewspaper
onalargerscale. Whenapandemicaffectskeypersons, this coverage frequency and focus on monetary policy making.
willhaveatremendousimpactonthestockmarket[28]. Ontheotherhand, ourworkdealswitheventstudiesonhow
Event studies cover two time periods, pre-event and muchabnormalreturncanbeobtainedfromanevent.
postevent periods, and examine two groups, control and The above literature review that event studies require the
treatment groups. If the event period varies, its average can examination of an event period and that there are many
be taken as defined in [29]. Event information can be soft meansofperformingthem. Aneventcanbestudiedwithtwo
or hard. Hard information can be quantified and quickly groups, control and treatment groups, or the standard index
converted into a number. Soft information requires con- value. Thereareavarietyofwaystoconductaneventstudy.
text. When one separates soft information from its source, An event study must determine the period of research and
it becomes useless. Hard information is known as quanti- adoptaregressionequation.
tative, while soft information is known as qualitative [30].
Big data technologies can store qualitative data on events III. METHODOLOGY
rather than quantitative data along. The field of finance has Ourmethodologyispresentedinsubsections III-Aand III-B,
concentrated on the use of quantitative data for analyses. respectively, titled Toolsand Process.
VOLUME9,2021 114197


<!-- Page 7 -->

P.K.Aithaletal.:Building Calendarof Events Databaseby Analyzing Financial Spikes
FIGURE6. Meanpolarityinaday.
A. TOOLS Weselected Pythonasourlanguageofchoicefordatacol-
Theeconomictimes.comandthemoneycontrol.comarecho- lectionanditsrelatedmathematicalanalysisandweselected
sen for news article collection as these are the prominent Mongo DBasaredatabase. Mongo DBisascalableandfast
financial websites in India. A thorough review of the two No SQL database and is known for its efficient querying
websites used, namely, moneycontrol.com and economic- capabilities. Initially, Py Mongo, a native Python driver for
times.com, revealed which routes gave relevant data. Then, Mongo DB, was used to query the database. Subsequently,
thetypeofdatabaseanddatabaseschemawereselected. thecodewasmodifiedtousemongoengine—a Pythonlibrary
114198 VOLUME9,2021


<!-- Page 8 -->

P.K.Aithaletal.:Building Calendarof Events Databaseby Analyzing Financial Spikes
FIGURE7. Nifty50 variationinaday.
thatactsasanobjectdocumentmapperfor Mongo DB.Itthen such as sentiment analyses of text. The NLTK was also
waseasiertospecifyschemasandmanipulatedataobtained used later in the project to process text and extract relevant
fromthedatabaseintheformofobjectswitheachfieldofa information.
knowntype. Git (and Git Hub) wasusedforversioncontroloftheproject
A request library was used to retrieve content from web- files and folders. To make the code portable, environmental
sites. Beautiful Soup, a Python package for parsing Hyper variables, alongwiththedotenvmodule, wereusedtodynam-
Text Markup Language (HTML) and e Xtensible Markup icallyloadvariablesduringexecution.
Language (XML) documents, was used to analyze the data,
scrapewebpages, andparsethedataintousefuldata. Inaddi- B. PROCESS
tion, thepandaslibrarywasusedtomanipulate.csvfilesand Events were collected from moneycontrol.com and eco-
otherdataframestoefficientlystructurethepre-existingdata. nomictimes.com. The data are stored in the Mongo DB
Text Blob, a Python library built from the Natural Lan- database. Twocollectionsarestoredinthedatabase—detailed
guage Tool Kit (NLTK), provides user-friendly interfaces stockvaluesandeventsfromthewebsites. Thedatafieldsof
for performing natural language processing (NLP) tasks thesetwocollectionsareasshownbelow.
VOLUME9,2021 114199


<!-- Page 9 -->

P.K.Aithaletal.:Building Calendarof Events Databaseby Analyzing Financial Spikes
FIGURE9. Correlationbetweenstockreturnsandsubjectivity.
FIGURE8. Correlationbetweenstockreturnsandpolarity.
for every required instance from the Yahoo Finance
News Item (News,
website.
Date,
3) Thecorrelationbetweenthedifferenceinopeningand
Time,
closingstockvaluesandthevariousnewsarticlesofa
Company Name,
company is found using equations (2) and (3), where
Polarity,
C and C denotethecorrelationsofpolarityandsub-
P S
Subjectivity)
jectivity with returns, respectively. P denotes polarity,
Stocks (Company Symbol,
Rdenotesthereturn, and Sdenotesubjectivity, relating
Opening Value,
thenewstostockfluctuations.
Closing Value,
(cid:80) (P −mean (P))∗(R −mean (R))
Difference) C = i i (2)
Field news stores scrapped news from the two websites. P (cid:112)(cid:80) (P i −mean (P))2∗(cid:80) (R i −mean (R))2
Wheninformationispublished, thedateandtimearestored (cid:80) (S −mean (S))∗(R −mean (R))
C = i i (3)
inthedateandtimefields, respectively. Thecompanyname S (cid:112)(cid:80) (S −mean (S))2∗(cid:80) (R −mean (R))2
i i
is considered optional since news may affect multiple com-
panieswithoutcitingacompanyname. Polarityvaluesvary 4) Thefollowinggraphsareplotted
from -1 to 1. If TP represents the total number of positive • Keywords and their impact on investor sentiment
wordsand TNrepresentsthetotalnumberofnegativewords, formeanpolarity
polarityisdefinedasfollows (1). • Keywords and their impact on investor sentiment
formeansubjectivity
Polarity=(TP−TN)/(TP+TN) (1)
• Meanpolarityonaparticularday
Subjectivity is a type of probability that determines how • Variationin Nifty50 stockswithinadayduetoa
objective a sentence is. Subjectivity, as it is a probability budgetevent
value, variesfrom0 to1.Furtherdetailsoncomputingsub- • Correlationbetweenstockreturnsandpolarity
jectivitycanbefoundin[33]. • Correlationbetweenstockreturnsandsubjectivity
The company symbol includes a ticker symbol for the As the script is executed, the news item collection is
Nifty50 companies. The opening price is the stock price at updatedwithoutduplicates, andstockcollectionisrefreshed.
thebeginningoftheday. Theclosingpriceisthepriceofthe BMA utilizes a linear regression model for predictions,
stockattheendoftheday. Thedifferencefieldcontainsthe considers all parameters and applies linear regression mod-
differencebetweenclosingandopeningprices. els for all possible combinations of the parameters. Two
We collected and processed our data over the following independentvariablesareusedinthecurrentpaper, namely,
steps. polarityandsubjectivity, andonedependentvariable, return,
1) The date and time of the event and the actual news isalsoused. BMAapplieslinearregressioninthefollowing
articlewereextractedfromtheweb.
parametercombinations:
2) Theopeningandclosingpricesofvariouscompanies’ 1) Return
stockpriceswerecollected, andthedifferencebetween 2) Polarity
thesepriceswascomputed. Dataregardingcompanies 3) Subjectivity
andtheirsymbolswereobtainedfrom NSE.Thesedata 4) Returnand Polarity
werethenusedtodynamicallyobtainthestockvalues 5) Returnand Subjectivity
114200 VOLUME9,2021


<!-- Page 10 -->

P.K.Aithaletal.:Building Calendarof Events Databaseby Analyzing Financial Spikes
FIGURE10. Timecomponentsofreturn.
6) Polarityand Subjectivity IV. RESULTS
7) Return, Polarity, and Subjectivity To minimize the noise for collecting the news article, one
BMAfindsthelikelihoodofeachcombinationandthepos- has to scrape the HTML. The empty and paid articles are
terior probability of each parameter. A decision is made discardedwhilecollectingthenewsarticles. Eventswerecol-
basedontheposteriorprobabilityofthevariable. Theoverall lectedinthe Mongo DBdatabaseastwocollections, namely,
architectureisdepictedin Fig.1. news items and stocks. News articles were collected during
VOLUME9,2021 114201


<!-- Page 11 -->

P.K.Aithaletal.:Building Calendarof Events Databaseby Analyzing Financial Spikes
FIGURE11. Timecomponentsofpolarity.
Indianbudgetingtime, andthearticleshaveapositivetone. is not used on February 1 st, 2021, India’s budget day. The
The sampled database collections of news items and stocks word‘‘high‘‘isoftenusedinthenewsarticles.
are described in Fig.2 and Fig.3 respectively. We collected Themeanpolaritywithinadayisshownin Fig.6 During
information on the date and time of each news item, on the the Indian annual budget event held in Feb. 2021, the mean
articleitself, andondegreesofarticlepolarizationandsub- polarityvalueispositivefortheentireweekoftheevent.
jectivity. Forstocks, wecollectedstocksymbols, thedatesof Thevariationinthestockpricesof Nifty50 companiesis
closing and opening prices, the opening prices of particular shown in Fig. 7. From Fig.7, it is clear that due to budget
days, theclosingpricesofparticulardays, andthedifferences events, someindividualstocksarepositivelyaffected, while
betweenclosingandopeningprices. others are negatively affected. The variation ranges from -
Thekeywordsandtheirimpactsonthegiveninvestorfor 400 to200.
the given time period with respect to tone are presented Fig.8 and Fig.9 demonstrate the effect of polarity and
in Fig. 4, and the same results with respect to subjectivity subjectivityonthestockreturnsof Nifty50 companies. For
are shown in Fig. 5 From the figures, it is clear that news a given polarity and subjectivity value, the companies react
on budgets has a positive tone. The heat maps depicted differently. From Figures 8 and 9, the return is not approxi-
in Figures1 and2 show that the article keywords have a matelyzeroforallofthecompanies. Wealsofindvariation
positive tone, and words such as ‘‘high‘‘ are more popular rangingfrom-400 to200 duetotheevent. Certaincompanies
thanwordsuchas‘‘loss‘‘duringtheperiod. Theword‘‘loss‘‘ reacttospecificevents.
114202 VOLUME9,2021


<!-- Page 12 -->

P.K.Aithaletal.:Building Calendarof Events Databaseby Analyzing Financial Spikes
FIGURE12. Timecomponentsofsubjectivity.
The time components of return, polarity, and subjectivity Returns = 0.001−0.122∗Polarity
aregivenin Fig.10 to Fig.12, respectively. Thefiguresshow −0.025∗(Polarity)2
thattheseasonalitycomponentisnotpresentinthedata, and
−11.81∗(Polarity)3
theobservedcomponentreflectsatrend.
Theregressionequationofdegreeonewithreturnsasthe
+0.46∗(Polarity)4
dependent variable and polarity as an independent variable +241.89∗(Polarity)5 (6)
is illustrated by equation 4. The regression equation with Returns = −1.02−12.68∗Subjectivity
returns as a dependent variable and subjectivity as an inde-
−60.48∗(Subjectivity)2
pendentvariableispresentedasequation5.Thepolynomial
regressionequationofdegreefivewithreturnsasadependent
+138.82∗(Subjectivity)3
variable and polarity as an independent variable is given by −152.97∗(Subjectivity)4
equation 6. The polynomial regression equation of degree +64.47∗(Subjectivity)5 (7)
five with returns as a dependent variable and subjectivity
asanindependentvariableiswrittenasequation7.Thetop The BMAresultsaretabulatedintables1 and2 andshow
15 keywordsofthenewsitemsforthebudgetweekwereiden- that polarity and subjectivity improve the model’s perfor-
tified. Table3 lists the top 15 keywords in descending order mance. Polarityoutperformssubjectivity. Itisbettertocon-
offrequency. Thetableshowsthatthedevelopmentofcoron- siderpolarityandsubjectivityasmodelparameters. Wealso
avirusvaccineshasledtopositivesentimentamonginvestors. findthatcertainparametersinfluencethemodel, buttheyare
Pandemic-relatedwordsaccountfor5 ofthe15 mostfrequent notincludedinthemodel.
words. Spearmanrhoand Kendalltaurankcorrelationsarefound
aswellasaslightlypositivecorrelationbetweenpolarityand
Returns = 0.003+0.009∗Polarity (4) returns. Additionally, we find a slightly positive correlation
Returns = 0.0002+0.007∗Subjectivity (5) betweensubjectivityandreturns.
VOLUME9,2021 114203


<!-- Page 13 -->

P.K.Aithaletal.:Building Calendarof Events Databaseby Analyzing Financial Spikes
FIGURE13. Causalimpact.
TABLE1. Variablesandtheirlikelihoodvalues. TABLE2. Variablesandtheirposteriorprobabilityvalues.
a dependent variable. The pre and post-period are set equal
in the model. The result obtained with the BSTS model is
depictedin Fig.13 Theresultobtainedisstatisticallysignifi-
The polarity and subjectivity are given to the BSTS cantastheposteriortail-areaprobabilityis0.04 at95%causal
model as independent variables, and return is provided as impactlevel.
114204 VOLUME9,2021


<!-- Page 14 -->

P.K.Aithaletal.:Building Calendarof Events Databaseby Analyzing Financial Spikes
TABLE3. Keywordsandtheirfrequenciesofuseduringbudgetweek [5] C.Castro-Iragorri,‘‘Doesthemarketmodelprovideagoodcounterfac-
Feb.1 to Feb.7,2021. tualforeventstudiesinfinance?’’Financial Markets Portfolio Manage.,
vol.33, no.1, pp.71–91, Mar.2019.
[6] L.Raynaud.(2017).Evaluating Industrial Policies Using Event-Studies:
Evidence From Stock Market Reactions to Obama’s Stimulus Package.
[Online].Available:https://www.skidmore.edu
[7] K. Schmidheiny and S. Siegloch, ‘‘On event study designs and
distributed-lagmodels: Equivalence, generalizationand practicalimpli-
cations,’’ CESifo Working Paper, IZA, Germany, Tech. Rep. 12079,
2019.
[8] I.P.A.Arganthaand I.M.S.N.Sudirman,‘‘Stockmarketreactiontothe
eventof Indonesia’sgeneralelectioneventsin2019,’’Amer. J.Hum. Social
Sci. Res., vol.4, no.1, pp.202–208,2019.
[9] J.Binder,‘‘Theeventstudymethodologysince1969,’’Rev. Quant. Finance
Accounting, vol.11, no.2, pp.111–137,1998.
[10] N. Bohn, F. A. Rabhi, D. Kundisch, L. Yao, and T. Mutter, ‘‘Towards
automated event studies using high frequency news and trading data,’’
in Proc. Int. Workshop Enterprise Appl. Services Finance Ind. Spain:
Springer,2012, pp.20–41.
[11] J.Affleck-Graves, C.M.Callahan, and R.Ramanan,‘‘Detectingabnor-
malbid-askspread:Acomparisonofeventstudymethods,’’Rev. Quant.
Finance Accounting, vol.14, no.1, pp.45–65,2000.
[12] G.Bird, W.Du, and T.Willett,‘‘Behavioralfinanceandefficientmarkets:
What does the euro crisis tell us?’’ Open Econ. Rev., vol. 28, no. 2,
pp.273–295, Apr.2017.
[13] B. Fang and P. Zhang, ‘‘Big data in finance,’’ in Big Data
V. CONCLUSIONANDFUTUREWORK Concepts, Theories, and Applications. Australia: Springer, 2016,
Two prominent financial websites, namely, moneycon- pp.391–412.
[14] H.Stanley, X.Gabaix, P.Gopikrishnan, and V.Plerou,‘‘Correlatedran-
trol.comandeconomictimes.com, werescrapedtobuildacal-
domness:Rareandnot-so-rareeventsinfinance,’’in Practical Fruitsof
endarofeventsdatabase. The Pythonlanguagewasutilized Econophysics. Japan:Springer,2006, pp.2–18.
to scrape thewebsites. Time components weredecomposed [15] D.Shenand S.-H.Chen,‘‘Bigdatafinanceandfinancialmarkets,’’in Big
Datain Computational Social Scienceand Humanities. Taiwan:Springer,
from data, and we found the only trend is present in the
2018, pp.235–248.
data, whichwasdetrendedviaregression. BMAwasapplied
[16] Z. Milosevic, A. Berry, W. Chen, and F. A. Rabhi, ‘‘An event-based
with linear regression, and we found polarity to be a more modeltosupportdistributedreal-timeanalytics:Financecasestudy,’’in
significantparameterthansubjectivity. Wealsoshowthatthe Proc. IEEE19 th Int. Enterprise Distrib. Object Comput. Conf., Sep.2015,
pp.122–127.
discoveryofcoronavirusvaccinesispositivelyaffectingthe
[17] C.-F.Huang, H.-C.Li, and B.-R.Chang,‘‘Anintelligentcomputational
Indianstockmarket. Tosummarize, thisarticlecontributesto financemodelformicrostructure-basedtradingsystem,’’in Proc. Int. Conf.
theexistingbodyofknowledgeinthefollowingways: Appl. Syst. Innov.(ICASI), May2016, pp.1–4.
[18] Q.Li, J.Tan, J.Wang, and H.Chen,‘‘Amultimodalevent-driven LSTM
1) Acalendarofeventsdatabaseiscreated. modelforstockpredictionusingonlinenews,’’IEEETrans. Knowl. Data
2) Insights are derived from the events database with Eng., earlyaccess, Jan.23,2020, doi:10.1109/TKDE.2020.2968894.
[19] Z.Li, Y.Cai, and S.Hu,‘‘Researchonsystemicfinancialriskmeasurement
quantitativeandqualitativedata.
basedon HMMandtextmining:Acaseof Chinafinancialmarket,’’IEEE
3) Wefindthattheinventionofcoronavirusvaccineshas Access, vol.9, pp.22171–22185,2021.
impactedthestockmarket. [20] A.Gupta, V.Dengre, H.A.Kheruwala, and M.Shah,‘‘Comprehensive
reviewoftext-miningapplicationsinfinance,’’Financial Innov., vol.6,
Event studies are a useful means of analyzing the stock no.1, pp.1–25, Dec.2020.
market, as stock markets are sensitive to events. Portfo- [21] Z.Milosevic, W.Chen, A.Berry, and F.A.Rabhi,‘‘Anopenarchitec-
lio optimization via event studies can be considered in tureforevent-basedanalytics,’’Int. J.Data Sci. Anal., vol.2, nos.1–2,
pp.13–27, Dec.2016.
futureenhancements. Furtherworkcanalsocategorizeevents
[22] J.Distler,‘‘Theeventstudyasaresearchmethodformeasuring M&A
into the following categories: global, national, sectoral, and performance,’’in Acquisitionsby Emerging Multinational Corporations.
company-specificevents. Onecanscrapetheotherfinancial Germany:Springer,2018, pp.221–268.
[23] B.R.Upreti, P.M.Back, P.Malo, O.Ahlgren, and A.Sinha,‘‘Knowledge-
websites in the future along with moneycontrol.com and
drivenapproachesforfinancialnewsanalytics,’’in Network Theoryand
economictimes.comwebsites. Agent-Based Modelingin Economicsand Finance. Singapore:Springer,
2019, pp.375–404.
[24] M.Hoesli, S.Milcheva, and A.Moss,‘‘Isfinancialregulationgoodorbad
REFERENCES
forrealestatecompanies?—Aneventstudy,’’J.Real Estate Finance Econ.,
[1] B.G.Malkiel,‘‘Theefficientmarkethypothesisanditscritics,’’J.Econ. vol.61, pp.369–407, Oct.2017.
Perspect., vol.17, no.1, pp.59–82,2003. [25] H.Liu, A.Manzoor, C.Wang, L.Zhang, and Z.Manzoor,‘‘The COVID-19
[2] A.Sorescu, N.L.Warren, and L.Ertekin,‘‘Eventstudymethodologyin outbreakandaffectedcountriesstockmarketsresponse,’’Int. J.Environ.
themarketingliterature:Anoverview,’’J.Acad. Marketing Sci., vol.45, Res. Public Health, vol.17, no.8, p.2800, Apr.2020.
no.2, pp.186–207, Mar.2017. [26] G.Capelle-Blancardand A.Petit,‘‘Everylittlehelps?ESGnewsandstock
[3] D.Winkelriedand L.A.Iberico,‘‘Calendareffectsin Latin Americanstock marketreaction,’’J.Bus. Ethics, vol.157, no.2, pp.543–565, Jun.2019.
markets,’’Empirical Econ., vol.54, no.3, pp.1215–1235, May2018. [27] R.J.Shiller, Narrative Economics:How Storiesgo Viraland Drive Major
[4] J.S.Angand S.Zhang,‘‘Evaluatinglong-horizoneventstudymethod- Economic Events. Princeton, NJ, USA:Princeton Univ. Press,2020.
ology,’’in Handbookof Financial Econometricsand Statistics, C.-F.Lee [28] V.Y.Fan, D.T.Jamison, and L.H.Summers,‘‘Pandemicrisk:Howlarge
and J.C.Lee, Eds. New York, NY, USA:Springer,2015, pp.383–411, doi: aretheexpectedlosses?’’Bull. World Health Org., vol.96, no.2, p.129,
10.1007/978-1-4614-7750-1_14. 2018.
VOLUME9,2021 114205


<!-- Page 15 -->

P.K.Aithaletal.:Building Calendarof Events Databaseby Analyzing Financial Spikes
[29] A. Goodman-Bacon, ‘‘Difference-in-differences with variation in treat- M.GEETHA(Member, IEEE) receivedthe Ph. D.
menttiming,’’J.Econometrics,2021.[Online].Available:https://www. degreefrom NITK, Surathkal, in2010.Sheiscur-
sciencedirect.com/science/article/pii/S0304407621001445, doi: 10.1016/ rentlya Professorwiththe Departmentof Com-
j.jeconom.2021.03.014. puter Scienceand Engineering, Manipal Institute
[30] J. M. Liberti and M. A. Petersen, ‘‘Information: Hard and soft,’’ Rev. of Technology, Manipal Academyof Higher Edu-
Corporate Finance Stud., vol.8, no.1, pp.1–41,2019. cation, Manipal, India. Shehaspresentedseveral
[31] M.P.Bach,Ž.Krstić, S.Seljan, and L.Turulja,‘‘Textminingforbigdata
papers at national and international conferences,
analysisinfinancialsector:Aliteraturereview,’’Sustainability, vol.11,
and her work has been published in several
no.5, p.1277, Feb.2019.
internationaljournals. Hercurrentresearchinter-
[32] S. J. Davis, S. R. Baker, and N. Bloom, ‘‘Measuring economic policy
ests include data mining and text mining in the
uncertainity,’’Nat. Bur. Econ. Res., Cambridge, MA, USA, Tech. Rep.
healthcareandfinancialsectors.
21633,2015, pp.1–75.
[33] A.Kamal,‘‘Subjectivityclassificationusingmachinelearningtechniques
for mining feature-opinion pairs from web opinion sources,’’ 2013,
ar Xiv:1312.6962.[Online].Available:http://arxiv.org/abs/1312.6962
PRAKASH K. AITHAL (Member, IEEE) iscur-
rently pursuing the Ph. D. degree with Manipal
Academy of Higher Education, Manipal, India.
He is currently an Assistant Professor with the
Department of Computer Science and Engineer-
ing, Manipal Institute of Technology, Manipal
Academyof Higher Education. Hehaspresented
severalpapersatnationalandinternationalconfer-
ences, andhisworkhasbeenpublishedinseveral
internationaljournals. Hiscurrentresearchinterest
includesdataanalyticsinthefinancialdomain.
U.DINESHACHARYAreceivedthe Ph. D.degree
from Manipal Academy of Higher Education, PARTHIV MENON is currently pursuing the
Manipal, India, in 2008. He is currently a Pro- B.Tech.degreewiththe Departmentof Computer
fessorwiththe Departmentof Computer Science Science and Engineering, Manipal Institute of
and Engineering, Manipal Instituteof Technology, Technology, Manipal Academyof Higher Educa-
Manipal Academy of Higher Education. He has tion, Manipal, India. Hisresearchinterestincludes
presentedseveralpapersatnationalandinterna- dataanalytics.
tional conferences, and his work has been pub-
lishedinseveralinternationaljournals. Hiscurrent
research interests include data analytics in the
healthcare, agriculture, andfinancialsectors.
114206 VOLUME9,2021
