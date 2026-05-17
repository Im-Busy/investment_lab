# Building a Calendar of Events Database by Analyzing Financial Spikes

> *Source PDF: Building a Calendar of Events Database by Analyzing Financial Spikes.pdf*
> *Extraction: Combined — previous structured extraction (base) + markitdown raw text*

---

# Building a Calendar of Events Database by Analyzing Financial Spikes

> *Source PDF: Building a Calendar of Events Database by Analyzing Financial Spikes.pdf*
> *Extraction: Combined — markitdown raw text (base) + previous extraction supplements*

---

Received June 29, 2021, accepted July 27, 2021, date of publication August 3, 2021, date of current version August 23, 2021.

Digital Object Identifier 10.1109/ACCESS.2021.3102495

Building a Calendar of Events Database by
Analyzing Financial Spikes

PRAKASH K. AITHAL , (Member, IEEE), U. DINESH ACHARYA , M. GEETHA , (Member, IEEE),
AND PARTHIV MENON
Department of Computer Science and Engineering, Manipal Institute of Technology, Manipal Academy of Higher Education, Manipal 576104, India

Corresponding author: M. Geetha (geetha.maiya@ manipal.edu)

ABSTRACT An event is a piece of news that triggers a change in stock prices. Here, an event study is
undertaken to capture the effect of abnormal returns due to an event. The event can affect the stock market in
the long term or short term. Event research is relevant to both the efﬁcient market hypothesis and behavioral
ﬁnance. In this study, we collected data from websites that manage ﬁnancial and economic data, performed a
sentiment analysis, and correlated news article data with changes in a particular company’s stock prices in the
stock market. Data were collected from two well-known ﬁnancial news websites. We observed a correlation
between stock prices and news items. An event period of one day was considered for the study. The regression
equation determined the relationship between stock returns and polarity and subjectivity. Bayesian model
averaging was performed to identify the effects of polarity and subjectivity on stock returns. Time-series data
were decomposed into components and detrended via regression. Prominent keywords and their polarity
values for a particular day were plotted. An event enhanced some stock returns while adversely affecting
other stocks. We found a variation range of -400 to 200 for different company stocks for the selected period.

INDEX TERMS Event database, social media data, sentiment analysis, data analytics, event studies.

I. INTRODUCTION
Data Science involves extracting information from raw data,
converting information into knowledge and knowledge into
an actionable plan, which has broad applications for ﬁelds
ranging from agriculture and space science to arts and ﬁnance
and which applies to all ﬁelds in which data are used.

Finance is essential to a nation’s growth. Primarily two
kinds of ﬁnancial data are used: quantitative data on variables
such as stock prices, the volume of stock, and the PE ratio
and qualitative data drawn from material such as director
reports, auditor reports, ﬁnancial statements, and ﬁnancial
news. Quantitative data reveal a stock’s movement, while
qualitative data unearth investor sentiment, which is the driv-
ing force behind stock movement.

Stock prices ﬂuctuate based on ﬁnancial events captured
by qualitative data. Events can include management changes,
declarations of dividends, festivals, natural calamities, pan-
demics, recessions, budget and earnings releases, and quar-
terly and yearly statements. Some events, such as festivals and
yearly statements, occur once a year. Changes in management
have either positive or adverse effects. Dividends always

The associate editor coordinating the review of this manuscript and

approving it for publication was Francesco Benedetto

.

decrease the value of a stock. Events have either short-term
consequences or affect stocks over the long run. Events that
affect the stock market for less than a week are considered
to be short-term events and that rest are long-term events.
An event can affect either a particular stock or the entire
stock market. The present event study aims to determine the
abnormal stock return attributable to new information carried
by an event.

Financial events provide both quantitative and qualita-
tive data. Qualitative data such as ﬁnancial news must be
quantiﬁed. One can apply a sentiment analysis to quantify
ﬁnancial news. Sentiment can be measured based on tone,
also known as polarity, or subjectivity. Polarity values vary
from -1 to 1 where a value of -1 denotes extremely nega-
tive sentiment and +1 denotes total positive sentiment. Sub-
jectivity levels vary from 0 to 1. The lower the value of
subjectivity is, the more meaning can be extracted from the
news.

There is a correlation between ﬁnancial news and stock
prices. If the correlation between news and the stock dif-
ference is greater, news inﬂuences the stock market. Event
studies assume the semistrong form of market efﬁciency.
The semistrong form of market efﬁciency applies two
assumptions [1].

114192

This work is licensed under a Creative Commons Attribution 4.0 License. For more information, see https://creativecommons.org/licenses/by/4.0/

VOLUME 9, 2021

P. K. Aithal et al.: Building Calendar of Events Database by Analyzing Financial Spikes

• All publicly available information is reﬂected in stock

prices

• Stock prices vary when new information arrives

The latter assumption is utilized to predict the future value of
a stock in advance. An event changes a stock price, and an
event study can then be utilized to analyze future cashﬂows.

A. BAYESIAN MODEL AVERAGING
An analysis is conducted by ﬁrst selecting the best model
based on a particular criterion and then learning about
the parameters of the chosen model. Any uncertainties of
the model selection process are ignored. Contrary to this
approach, Bayesian model averaging (BMA) learns param-
eters for all candidate models and allows one to select a
model with speciﬁc posterior probabilities. BMA offers the
following advantages over other models:

1) BMA captures the uncertainty of a model.
2) BMA reduces error.
3) BMA retains all model uncertainty until the ﬁnal stage.
4) BMA continuously adjusts model weights.
5) BMA uses a set of models, which is preferable to using

a single model.

B. BAYESIAN STRUCTURAL TIME SERIES
The Bayesian Structural Time Series (BSTS) is a tool used
to identify the causal relationship. It utilizes GPU if present
else runs on CPU. It requires pre-period and post-period as
arguments along with the data.

C. TIME SERIES COMPONENTS
It is helpful to decompose the time series into systematic
and nonsystematic components. Systematic components can
be modeled as either consistent or recurring. Nonsystematic
elements are random and cannot be modeled. The time series
consists of the following components:

1) Level
2) Trend
3) Seasonality
4) Noise

The level is the mean value of the time series. The trend is
either increasing or decreasing in the time series. The repeat-
ing short-term cycle is deﬁned as seasonality. Randomness
present in the time series that cannot be modeled is deﬁned
as noise.

The rest of the paper is organized as follows. Section 2
reviews the related literature, section 3 elaborates on the
methodology used, and section 4 presents the results. Conclu-
sions and avenues for future research are given in section 5.

II. LITERATURE SURVEY
Event studies determine excess returns attributable to a cor-
porate event. Events can range from an earnings release by a
company or budget announcement to a new product launch,
announcement from a competing company, regular ﬁnan-
cial statement or announcement made by a regulatory body.

FIGURE 1. Building and analysis of calendar events database.

Researchers look forward in time and predict the expected
value a ﬁrm will derive from a corporate action announced
to the public. This forward-looking focus of event studies
makes them more potent than other metrics such as return
on investment (ROI) and proﬁts. Investors respond positively
to product launches, sponsorships, and mergers, while they
respond negatively to events such as product recalls. [2]. It has
been observed that historical stock returns are affected by
calendar events. When event studies have been conducted in
developed countries, little related literature is available from
developing nations. The stock market declines on Mondays
and strengthens on Friday. The start of a new month or year
also affects the stock market [3]. The presence of long-term
abnormal returns contradicts the efﬁcient market hypothesis.
Different testing strategies are applicable to asset pricing and
buy-and-hold methodologies [4].

To identify the abnormal returns of the portfolio of stocks
affected by an event, a control portfolio of stocks not affected
by the event is considered as a benchmark, and the results
of both portfolios are compared [5]. Event studies can eval-
uate a government’s policies, as markets react to new news
immediately and adjust their value. If a policy has a positive
effect on the market, it will eventually succeed. Speculation
on an event also has a major impact on the stock market. The
stock market reacts to speculation and is affected long before
an event is announced [6]. Event studies require the use of
a window period under which an event is studied. An event
study evaluation is done based on the methodology utilized
and can involve a regression with a t-test or the use of a capital
asset pricing model (CAPM) with a t-test [7].

VOLUME 9, 2021

114193

P. K. Aithal et al.: Building Calendar of Events Database by Analyzing Financial Spikes

FIGURE 2. News item collection.

An election is a major event that signiﬁcantly inﬂuences
the stock market. A set of stocks are affected by an election.
Elections have an effect for approximately ﬁve months before
and after an event. The announcement of an election and
the election itself greatly impact the stock market [8]. Event
studies have become the defacto standard for measuring
stock prices during and after an event. Abnormal returns are
determined either as the residuals from standard normalized
benchmarks or as a dummy variable in a regression equa-
tion [9]. News can be classiﬁed as repeating or surprising.
A repeating event can be easily predicted, and markets adjust
to such an event. A surprising event, on the other hand, leads
to abnormal returns. This is the case because such an event
is unknown, and no estimate has been made, which results in
risk that must be addressed [10].

In [11], the authors conclude that intraday abnormal returns
are normally distributed, and Pattel test statistics function
better than other statistical measures in measuring abnormal
returns. The market responds to both good and bad news.
Good news reduces risk, while a piece of bad news increases
it. The efﬁcient market hypothesis and behavioral ﬁnance
apply in some cases and fail in others. Both have explanatory
power and must be explored. According to event studies,
stocks will not alter prices unless and until a piece of news

arrives [12]. Events do not have structure and cannot be stored
in relational databases. Therefore, Not Only Structured Query
Language (NoSQL) databases are utilized to store informa-
tion on such events. NoSQL databases require no downtime
and are different from regular relational database manage-
ment systems (RDBMSs), as they can scale well at a lower
cost. A RDBMS has strict atomicity consistency isolation
durability (ACID) properties, while NoSQL databases adhere
to the consistency availability and partition tolerance (CAP)
theorem. Complex event processing cannot be achieved using
a regular RDBMS, so one must in this case use NoSQL
databases for faster processing. Complex event processing
requires low latency, high throughput, and temporal and spa-
tial event processing capabilities [13]. Stock prices move
with a correlated random walk rather than with an uncorre-
lated random walk. Correlations can be found through event
studies [14].

One comment posted on the Yahoo ﬁnancial website
almost crashed the Nevada company, after which the com-
pany had to call a press conference to prove that its fun-
damentals are strong. This incident demonstrates that event
studies are critical. The simplest measures used include the
number of times a stock name is mentioned, the frequency
of certain keywords, and the sentiment of news about a

114194

VOLUME 9, 2021

P. K. Aithal et al.: Building Calendar of Events Database by Analyzing Financial Spikes

FIGURE 3. Stock collection.

given company [15]. There are two types of events, namely,
simple events and complex events. A simple event occurs
instantaneously and independent of other events. A complex
event is a set of events that are interrelated [16]. The tradi-
tional investment strategy takes months or years to arrive at a
decision. The advancement of information technology in the
realm of ﬁnance has made the decision-making process faster.
Events can be analyzed in milliseconds using information
technology [17].

The efﬁcient market hypothesis states that market prices
reﬂect current news and that as new news arrives, prices

change. Efﬁcient market theory considers investors to be
unemotional. Behavioral ﬁnance, on the other hand, treats
investors as an emotional entity and assumes news events
to affect investments. Both the efﬁcient market hypothesis
and behavioral ﬁnance agree that news events are respon-
sible for abnormal returns [18]. Text mining is used in
event studies, and it requires words to be segmented, and
spaces are used to separate words. Stop words need to
be removed. The sentiment of a sentence and frequency
of words measure the quantum of the jump of stock
prices [19].

VOLUME 9, 2021

114195

P. K. Aithal et al.: Building Calendar of Events Database by Analyzing Financial Spikes

FIGURE 4. Keywords and their impact on investor sentiment for mean polarity.

Sentiment analysis is widely used in the ﬁeld of ﬁnance.
The method is also known as opinion mining. Sentiment anal-
ysis is broadly classiﬁed into emotion recognition and tone
analysis or polarity analysis. Opinion mining can be classiﬁed
based on whether a dictionary or machine learning is used
to arrive at an opinion. Machine learning-based approaches
require domain-speciﬁc content to work correctly, while dic-
tionaries are costly [20]. Event analysis must be conducted by
batch processing of historical events. The event pattern must
support mathematical, logical, and statistical patterns among
events [21]. An event study involves several steps and many
choices must be made at each step. Thus, there is no standard
means of conducting an event study. The timing of an event
study and the event window size need to be deﬁned by the
researcher. Surprise events need to be evaluated through event
studies [22].

A historical analysis of 145100 news articles has shown a
signiﬁcant relationship between tone and companies’ perfor-
mance. However, a single instance of news does not have an

impact on stock movement. If a company calls a conference
call and the tone is positive, the stock of that company will
move multifold in a positive direction, resulting in an abnor-
mal positive return [23]. The global ﬁnancial crisis resulted in
governments intervening and taking regulatory action. Global
regulatory action is an event that affects the stock market
worldwide [24].

The coronavirus pandemic has affected US asset prices.
The pandemic has resulted in high levels of volatility in
small and unstable markets. Stock markets are interrelated
and sensitive to news. Media coverage on the pandemic has
resulted in an adverse crash in some highly volatile stocks and
unstable industries. The global stock markets are becoming
interdependent. Even when a single stock market is affected,
this impact spreads to other stock markets [25]. [26] studied
corporate social responsibility with the help of a dictionary.
The authors showed that when a ﬁrm is not responsible and
this results in a negative event, its stock price will fall drasti-
cally. Event and ﬁrm performance are directly proportional.

114196

VOLUME 9, 2021

P. K. Aithal et al.: Building Calendar of Events Database by Analyzing Financial Spikes

FIGURE 5. Keywords and their impact on investor sentiment for mean subjectivity.

Narrative economics has developed into a ﬁeld of its own.
A narrative event inﬂuences a stock, and numerous examples
of this phenomenon prove this fact [27]. The pandemic has
caused a loss to the economy. A historical inﬂuenza revealed
that a pandemic has effects similar to those of inﬂuenza but
on a larger scale. When a pandemic affects key persons, this
will have a tremendous impact on the stock market [28].

Event studies cover two time periods, pre-event and
postevent periods, and examine two groups, control and
treatment groups. If the event period varies, its average can
be taken as deﬁned in [29]. Event information can be soft
or hard. Hard information can be quantiﬁed and quickly
converted into a number. Soft information requires con-
text. When one separates soft information from its source,
it becomes useless. Hard information is known as quanti-
tative, while soft information is known as qualitative [30].
Big data technologies can store qualitative data on events
rather than quantitative data along. The ﬁeld of ﬁnance has
concentrated on the use of quantitative data for analyses.

With the emergence of big data technologies, the storage and
processing of qualitative data have become more accessible.
The focus of the researcher has thus switched from the use of
quantitative data to the use of qualitative data [31].

[32] calculate economic uncertainty based on newspaper
coverage frequency and focus on monetary policy making.
On the other hand, our work deals with event studies on how
much abnormal return can be obtained from an event.

The above literature review that event studies require the
examination of an event period and that there are many
means of performing them. An event can be studied with two
groups, control and treatment groups, or the standard index
value. There are a variety of ways to conduct an event study.
An event study must determine the period of research and
adopt a regression equation.

III. METHODOLOGY
Our methodology is presented in subsections III-A and III-B,
respectively, titled Tools and Process.

VOLUME 9, 2021

114197

P. K. Aithal et al.: Building Calendar of Events Database by Analyzing Financial Spikes

FIGURE 6. Mean polarity in a day.

A. TOOLS
The economictimes.com and the moneycontrol.com are cho-
sen for news article collection as these are the prominent
ﬁnancial websites in India. A thorough review of the two
websites used, namely, moneycontrol.com and economic-
times.com, revealed which routes gave relevant data. Then,
the type of database and database schema were selected.

We selected Python as our language of choice for data col-
lection and its related mathematical analysis and we selected
MongoDB as are database. MongoDB is a scalable and fast
NoSQL database and is known for its efﬁcient querying
capabilities. Initially, PyMongo, a native Python driver for
MongoDB, was used to query the database. Subsequently,
the code was modiﬁed to use mongoengine—a Python library

114198

VOLUME 9, 2021

P. K. Aithal et al.: Building Calendar of Events Database by Analyzing Financial Spikes

FIGURE 7. Nifty 50 variation in a day.

that acts as an object document mapper for MongoDB. It then
was easier to specify schemas and manipulate data obtained
from the database in the form of objects with each ﬁeld of a
known type.

A request library was used to retrieve content from web-
sites. BeautifulSoup, a Python package for parsing Hyper
Text Markup Language (HTML) and eXtensible Markup
Language (XML) documents, was used to analyze the data,
scrape webpages, and parse the data into useful data. In addi-
tion, the pandas library was used to manipulate.csv ﬁles and
other data frames to efﬁciently structure the pre-existing data.
TextBlob, a Python library built from the Natural Lan-
guage Tool Kit (NLTK), provides user-friendly interfaces
for performing natural language processing (NLP) tasks

such as sentiment analyses of text. The NLTK was also
used later in the project to process text and extract relevant
information.

Git (and GitHub) was used for version control of the project
ﬁles and folders. To make the code portable, environmental
variables, along with the dotenv module, were used to dynam-
ically load variables during execution.

B. PROCESS
Events were collected from moneycontrol.com and eco-
nomictimes.com. The data are stored in the MongoDB
database. Two collections are stored in the database—detailed
stock values and events from the websites. The data ﬁelds of
these two collections are as shown below.

VOLUME 9, 2021

114199

P. K. Aithal et al.: Building Calendar of Events Database by Analyzing Financial Spikes

FIGURE 8. Correlation between stock returns and polarity.

FIGURE 9. Correlation between stock returns and subjectivity.

NewsItem(News,
Date,
Time,
CompanyName,
Polarity,
Subjectivity)
Stocks(CompanySymbol,
OpeningValue,
ClosingValue,
Difference)
Field news stores scrapped news from the two websites.
When information is published, the date and time are stored
in the date and time ﬁelds, respectively. The company name
is considered optional since news may affect multiple com-
panies without citing a company name. Polarity values vary
from -1 to 1. If TP represents the total number of positive
words and TN represents the total number of negative words,
polarity is deﬁned as follows (1).

Polarity = (TP − TN )/(TP + TN )

(1)

Subjectivity is a type of probability that determines how
objective a sentence is. Subjectivity, as it is a probability
value, varies from 0 to 1. Further details on computing sub-
jectivity can be found in [33].

The company symbol includes a ticker symbol for the
Nifty 50 companies. The opening price is the stock price at
the beginning of the day. The closing price is the price of the
stock at the end of the day. The difference ﬁeld contains the
difference between closing and opening prices.

We collected and processed our data over the following

steps.

1) The date and time of the event and the actual news

article were extracted from the web.

2) The opening and closing prices of various companies’
stock prices were collected, and the difference between
these prices was computed. Data regarding companies
and their symbols were obtained from NSE. These data
were then used to dynamically obtain the stock values

for every required instance from the Yahoo Finance
website.

3) The correlation between the difference in opening and
closing stock values and the various news articles of a
company is found using equations (2) and (3), where
CP and CS denote the correlations of polarity and sub-
jectivity with returns, respectively. P denotes polarity,
R denotes the return, and S denote subjectivity, relating
the news to stock ﬂuctuations.

CP =

CS =

(cid:80)(Pi − mean(P)) ∗ (Ri − mean(R))
(cid:112)(cid:80)(Pi − mean(P))2 ∗ (cid:80)(Ri − mean(R))2
(cid:80)(Si − mean(S)) ∗ (Ri − mean(R))
(cid:112)(cid:80)(Si − mean(S))2 ∗ (cid:80)(Ri − mean(R))2

(2)

(3)

4) The following graphs are plotted

• Keywords and their impact on investor sentiment

for mean polarity

• Keywords and their impact on investor sentiment

for mean subjectivity

• Mean polarity on a particular day
• Variation in Nifty 50 stocks within a day due to a

budget event

• Correlation between stock returns and polarity
• Correlation between stock returns and subjectivity
As the script is executed, the news item collection is
updated without duplicates, and stock collection is refreshed.
BMA utilizes a linear regression model for predictions,
considers all parameters and applies linear regression mod-
els for all possible combinations of the parameters. Two
independent variables are used in the current paper, namely,
polarity and subjectivity, and one dependent variable, return,
is also used. BMA applies linear regression in the following
parameter combinations:

1) Return
2) Polarity
3) Subjectivity
4) Return and Polarity
5) Return and Subjectivity

114200

VOLUME 9, 2021

P. K. Aithal et al.: Building Calendar of Events Database by Analyzing Financial Spikes

FIGURE 10. Time components of return.

6) Polarity and Subjectivity
7) Return, Polarity, and Subjectivity

BMA ﬁnds the likelihood of each combination and the pos-
terior probability of each parameter. A decision is made
based on the posterior probability of the variable. The overall
architecture is depicted in Fig. 1.

IV. RESULTS
To minimize the noise for collecting the news article, one
has to scrape the HTML. The empty and paid articles are
discarded while collecting the news articles. Events were col-
lected in the MongoDB database as two collections, namely,
news items and stocks. News articles were collected during

VOLUME 9, 2021

114201

P. K. Aithal et al.: Building Calendar of Events Database by Analyzing Financial Spikes

FIGURE 11. Time components of polarity.

Indian budgeting time, and the articles have a positive tone.
The sampled database collections of news items and stocks
are described in Fig.2 and Fig.3 respectively. We collected
information on the date and time of each news item, on the
article itself, and on degrees of article polarization and sub-
jectivity. For stocks, we collected stock symbols, the dates of
closing and opening prices, the opening prices of particular
days, the closing prices of particular days, and the differences
between closing and opening prices.

The keywords and their impacts on the given investor for
the given time period with respect to tone are presented
in Fig. 4, and the same results with respect to subjectivity
are shown in Fig. 5 From the ﬁgures, it is clear that news
on budgets has a positive tone. The heat maps depicted
in Figures 1 and 2 show that the article keywords have a
positive tone, and words such as ‘‘high‘‘ are more popular
than word such as ‘‘loss‘‘ during the period. The word ‘‘loss‘‘

is not used on February 1st, 2021, India’s budget day. The
word ‘‘high‘‘ is often used in the news articles.

The mean polarity within a day is shown in Fig. 6 During
the Indian annual budget event held in Feb. 2021, the mean
polarity value is positive for the entire week of the event.

The variation in the stock prices of Nifty 50 companies is
shown in Fig. 7. From Fig.7, it is clear that due to budget
events, some individual stocks are positively affected, while
others are negatively affected. The variation ranges from -
400 to 200.

Fig.8 and Fig.9 demonstrate the effect of polarity and
subjectivity on the stock returns of Nifty 50 companies. For
a given polarity and subjectivity value, the companies react
differently. From Figures 8 and 9, the return is not approxi-
mately zero for all of the companies. We also ﬁnd variation
ranging from -400 to 200 due to the event. Certain companies
react to speciﬁc events.

114202

VOLUME 9, 2021

P. K. Aithal et al.: Building Calendar of Events Database by Analyzing Financial Spikes

FIGURE 12. Time components of subjectivity.

The time components of return, polarity, and subjectivity
are given in Fig. 10 to Fig. 12, respectively. The ﬁgures show
that the seasonality component is not present in the data, and
the observed component reﬂects a trend.

The regression equation of degree one with returns as the
dependent variable and polarity as an independent variable
is illustrated by equation 4. The regression equation with
returns as a dependent variable and subjectivity as an inde-
pendent variable is presented as equation 5. The polynomial
regression equation of degree ﬁve with returns as a dependent
variable and polarity as an independent variable is given by
equation 6. The polynomial regression equation of degree
ﬁve with returns as a dependent variable and subjectivity
as an independent variable is written as equation 7. The top
15 keywords of the news items for the budget week were iden-
tiﬁed. Table3 lists the top 15 keywords in descending order
of frequency. The table shows that the development of coron-
avirus vaccines has led to positive sentiment among investors.
Pandemic-related words account for 5 of the 15 most frequent
words.

Returns = 0.003 + 0.009 ∗ Polarity
Returns = 0.0002 + 0.007 ∗ Subjectivity

(4)
(5)

Returns = 0.001 − 0.122 ∗ Polarity

−0.025 ∗ (Polarity)2
−11.81 ∗ (Polarity)3
+0.46 ∗ (Polarity)4
+241.89 ∗ (Polarity)5

Returns = −1.02 − 12.68 ∗ Subjectivity

−60.48 ∗ (Subjectivity)2
+138.82 ∗ (Subjectivity)3
−152.97 ∗ (Subjectivity)4
+64.47 ∗ (Subjectivity)5

(6)

(7)

The BMA results are tabulated in tables 1 and 2 and show
that polarity and subjectivity improve the model’s perfor-
mance. Polarity outperforms subjectivity. It is better to con-
sider polarity and subjectivity as model parameters. We also
ﬁnd that certain parameters inﬂuence the model, but they are
not included in the model.

Spearman rho and Kendall tau rank correlations are found
as well as a slightly positive correlation between polarity and
returns. Additionally, we ﬁnd a slightly positive correlation
between subjectivity and returns.

VOLUME 9, 2021

114203

P. K. Aithal et al.: Building Calendar of Events Database by Analyzing Financial Spikes

FIGURE 13. Causal impact.

TABLE 1. Variables and their likelihood values.

TABLE 2. Variables and their posterior probability values.

The polarity and subjectivity are given to the BSTS
model as independent variables, and return is provided as

a dependent variable. The pre and post-period are set equal
in the model. The result obtained with the BSTS model is
depicted in Fig. 13 The result obtained is statistically signiﬁ-
cant as the posterior tail-area probability is 0.04 at 95% causal
impact level.

114204

VOLUME 9, 2021

P. K. Aithal et al.: Building Calendar of Events Database by Analyzing Financial Spikes

TABLE 3. Keywords and their frequencies of use during budget week
Feb. 1 to Feb. 7, 2021.

[5] C. Castro-Iragorri, ‘‘Does the market model provide a good counterfac-
tual for event studies in ﬁnance?’’ Financial Markets Portfolio Manage.,
vol. 33, no. 1, pp. 71–91, Mar. 2019.

[6] L. Raynaud. (2017). Evaluating Industrial Policies Using Event-Studies:
Evidence From Stock Market Reactions to Obama’s Stimulus Package.
[Online]. Available: https://www.skidmore.edu

[7] K. Schmidheiny and S. Siegloch,

‘‘On event study designs and
distributed-lag models: Equivalence, generalization and practical impli-
cations,’’ CESifo Working Paper, IZA, Germany, Tech. Rep. 12079,
2019.

[8] I. P. A. Argantha and I. M. S. N. Sudirman, ‘‘Stock market reaction to the
event of Indonesia’s general election events in 2019,’’ Amer. J. Hum. Social
Sci. Res., vol. 4, no. 1, pp. 202–208, 2019.

[9] J. Binder, ‘‘The event study methodology since 1969,’’ Rev. Quant. Finance

Accounting, vol. 11, no. 2, pp. 111–137, 1998.

[10] N. Bohn, F. A. Rabhi, D. Kundisch, L. Yao, and T. Mutter, ‘‘Towards
automated event studies using high frequency news and trading data,’’
in Proc. Int. Workshop Enterprise Appl. Services Finance Ind. Spain:
Springer, 2012, pp. 20–41.

[11] J. Afﬂeck-Graves, C. M. Callahan, and R. Ramanan, ‘‘Detecting abnor-
mal bid-ask spread: A comparison of event study methods,’’ Rev. Quant.
Finance Accounting, vol. 14, no. 1, pp. 45–65, 2000.

V. CONCLUSION AND FUTURE WORK
Two prominent ﬁnancial websites, namely, moneycon-
trol.com and economictimes.com, were scraped to build a cal-
endar of events database. The Python language was utilized
to scrape the websites. Time components were decomposed
from data, and we found the only trend is present in the
data, which was detrended via regression. BMA was applied
with linear regression, and we found polarity to be a more
signiﬁcant parameter than subjectivity. We also show that the
discovery of coronavirus vaccines is positively affecting the
Indian stock market. To summarize, this article contributes to
the existing body of knowledge in the following ways:

1) A calendar of events database is created.
2) Insights are derived from the events database with

quantitative and qualitative data.

3) We ﬁnd that the invention of coronavirus vaccines has

impacted the stock market.

Event studies are a useful means of analyzing the stock
market, as stock markets are sensitive to events. Portfo-
lio optimization via event studies can be considered in
future enhancements. Further work can also categorize events
into the following categories: global, national, sectoral, and
company-speciﬁc events. One can scrape the other ﬁnancial
websites in the future along with moneycontrol.com and
economictimes.com websites.

REFERENCES
[1] B. G. Malkiel, ‘‘The efﬁcient market hypothesis and its critics,’’ J. Econ.

Perspect., vol. 17, no. 1, pp. 59–82, 2003.

[2] A. Sorescu, N. L. Warren, and L. Ertekin, ‘‘Event study methodology in
the marketing literature: An overview,’’ J. Acad. Marketing Sci., vol. 45,
no. 2, pp. 186–207, Mar. 2017.

[3] D. Winkelried and L. A. Iberico, ‘‘Calendar effects in Latin American stock
markets,’’ Empirical Econ., vol. 54, no. 3, pp. 1215–1235, May 2018.
[4] J. S. Ang and S. Zhang, ‘‘Evaluating long-horizon event study method-
ology,’’ in Handbook of Financial Econometrics and Statistics, C.-F. Lee
and J. C. Lee, Eds. New York, NY, USA: Springer, 2015, pp. 383–411, doi:
10.1007/978-1-4614-7750-1_14.

[12] G. Bird, W. Du, and T. Willett, ‘‘Behavioral ﬁnance and efﬁcient markets:
What does the euro crisis tell us?’’ Open Econ. Rev., vol. 28, no. 2,
pp. 273–295, Apr. 2017.
[13] B. Fang and P. Zhang,

in Big Data
Concepts, Theories, and Applications. Australia: Springer, 2016,
pp. 391–412.

‘‘Big data in ﬁnance,’’

[14] H. Stanley, X. Gabaix, P. Gopikrishnan, and V. Plerou, ‘‘Correlated ran-
domness: Rare and not-so-rare events in ﬁnance,’’ in Practical Fruits of
Econophysics. Japan: Springer, 2006, pp. 2–18.

[15] D. Shen and S.-H. Chen, ‘‘Big data ﬁnance and ﬁnancial markets,’’ in Big
Data in Computational Social Science and Humanities. Taiwan: Springer,
2018, pp. 235–248.

[16] Z. Milosevic, A. Berry, W. Chen, and F. A. Rabhi, ‘‘An event-based
model to support distributed real-time analytics: Finance case study,’’ in
Proc. IEEE 19th Int. Enterprise Distrib. Object Comput. Conf., Sep. 2015,
pp. 122–127.

[17] C.-F. Huang, H.-C. Li, and B.-R. Chang, ‘‘An intelligent computational
ﬁnance model for microstructure-based trading system,’’ in Proc. Int. Conf.
Appl. Syst. Innov. (ICASI), May 2016, pp. 1–4.

[18] Q. Li, J. Tan, J. Wang, and H. Chen, ‘‘A multimodal event-driven LSTM
model for stock prediction using online news,’’ IEEE Trans. Knowl. Data
Eng., early access, Jan. 23, 2020, doi: 10.1109/TKDE.2020.2968894.
[19] Z. Li, Y. Cai, and S. Hu, ‘‘Research on systemic ﬁnancial risk measurement
based on HMM and text mining: A case of China ﬁnancial market,’’ IEEE
Access, vol. 9, pp. 22171–22185, 2021.

[20] A. Gupta, V. Dengre, H. A. Kheruwala, and M. Shah, ‘‘Comprehensive
review of text-mining applications in ﬁnance,’’ Financial Innov., vol. 6,
no. 1, pp. 1–25, Dec. 2020.

[21] Z. Milosevic, W. Chen, A. Berry, and F. A. Rabhi, ‘‘An open architec-
ture for event-based analytics,’’ Int. J. Data Sci. Anal., vol. 2, nos. 1–2,
pp. 13–27, Dec. 2016.

[22] J. Distler, ‘‘The event study as a research method for measuring M&A
performance,’’ in Acquisitions by Emerging Multinational Corporations.
Germany: Springer, 2018, pp. 221–268.

[23] B. R. Upreti, P. M. Back, P. Malo, O. Ahlgren, and A. Sinha, ‘‘Knowledge-
driven approaches for ﬁnancial news analytics,’’ in Network Theory and
Agent-Based Modeling in Economics and Finance. Singapore: Springer,
2019, pp. 375–404.

[24] M. Hoesli, S. Milcheva, and A. Moss, ‘‘Is ﬁnancial regulation good or bad
for real estate companies?—An event study,’’ J. Real Estate Finance Econ.,
vol. 61, pp. 369–407, Oct. 2017.

[25] H. Liu, A. Manzoor, C. Wang, L. Zhang, and Z. Manzoor, ‘‘The COVID-19
outbreak and affected countries stock markets response,’’ Int. J. Environ.
Res. Public Health, vol. 17, no. 8, p. 2800, Apr. 2020.

[26] G. Capelle-Blancard and A. Petit, ‘‘Every little helps? ESG news and stock
market reaction,’’ J. Bus. Ethics, vol. 157, no. 2, pp. 543–565, Jun. 2019.
[27] R. J. Shiller, Narrative Economics: How Stories go Viral and Drive Major
Economic Events. Princeton, NJ, USA: Princeton Univ. Press, 2020.
[28] V. Y. Fan, D. T. Jamison, and L. H. Summers, ‘‘Pandemic risk: How large
are the expected losses?’’ Bull. World Health Org., vol. 96, no. 2, p. 129,
2018.

VOLUME 9, 2021

114205

P. K. Aithal et al.: Building Calendar of Events Database by Analyzing Financial Spikes

[29] A. Goodman-Bacon, ‘‘Difference-in-differences with variation in treat-
ment timing,’’ J. Econometrics, 2021. [Online]. Available: https://www.
sciencedirect.com/science/article/pii/S0304407621001445, doi: 10.1016/
j.jeconom.2021.03.014.

[30] J. M. Liberti and M. A. Petersen, ‘‘Information: Hard and soft,’’ Rev.

Corporate Finance Stud., vol. 8, no. 1, pp. 1–41, 2019.

[31] M. P. Bach, Ž. Krstić, S. Seljan, and L. Turulja, ‘‘Text mining for big data
analysis in ﬁnancial sector: A literature review,’’ Sustainability, vol. 11,
no. 5, p. 1277, Feb. 2019.

[32] S. J. Davis, S. R. Baker, and N. Bloom, ‘‘Measuring economic policy
uncertainity,’’ Nat. Bur. Econ. Res., Cambridge, MA, USA, Tech. Rep.
21633, 2015, pp. 1–75.

[33] A. Kamal, ‘‘Subjectivity classiﬁcation using machine learning techniques
for mining feature-opinion pairs from web opinion sources,’’ 2013,
arXiv:1312.6962. [Online]. Available: http://arxiv.org/abs/1312.6962

PRAKASH K. AITHAL (Member, IEEE) is cur-
rently pursuing the Ph.D. degree with Manipal
Academy of Higher Education, Manipal, India.
He is currently an Assistant Professor with the
Department of Computer Science and Engineer-
ing, Manipal Institute of Technology, Manipal
Academy of Higher Education. He has presented
several papers at national and international confer-
ences, and his work has been published in several
international journals. His current research interest

includes data analytics in the ﬁnancial domain.

U. DINESH ACHARYA received the Ph.D. degree
from Manipal Academy of Higher Education,
Manipal, India, in 2008. He is currently a Pro-
fessor with the Department of Computer Science
and Engineering, Manipal Institute of Technology,
Manipal Academy of Higher Education. He has
presented several papers at national and interna-
tional conferences, and his work has been pub-
lished in several international journals. His current
research interests include data analytics in the

healthcare, agriculture, and ﬁnancial sectors.

M. GEETHA (Member, IEEE) received the Ph.D.
degree from NITK, Surathkal, in 2010. She is cur-
rently a Professor with the Department of Com-
puter Science and Engineering, Manipal Institute
of Technology, Manipal Academy of Higher Edu-
cation, Manipal, India. She has presented several
papers at national and international conferences,
and her work has been published in several
international journals. Her current research inter-
ests include data mining and text mining in the

healthcare and ﬁnancial sectors.

PARTHIV MENON is currently pursuing the
B.Tech. degree with the Department of Computer
Science and Engineering, Manipal Institute of
Technology, Manipal Academy of Higher Educa-
tion, Manipal, India. His research interest includes
data analytics.

114206

VOLUME 9, 2021



---

## Content from Previous Extraction (not in markitdown output)

### Visual/Chart/Graph Descriptions

#### 1

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

#### 2

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

#### 3

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

#### 4

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

#### 5

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

#### 6

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

#### 7

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

#### 8

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

#### 9

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

#### 10

P.K.Aithaletal.:Building Calendarof Events Databaseby Analyzing Financial Spikes
FIGURE10. Timecomponentsofreturn.
6) Polarityand Subjectivity IV. RESULTS
7) Return, Polarity, and Subjectivity To minimize the noise for collecting the news article, one
BMAfindsthelikelihoodofeachcombinationandthepos- has to scrape the HTML. The empty and paid articles are
terior probability of each parameter. A decision is made discardedwhilecollectingthenewsarticles. Eventswerecol-
basedontheposteriorprobabilityofthevariable. Theoverall lectedinthe Mongo DBdatabaseastwocollections, namely,
architectureisdepictedin Fig.1. news items and stocks. News articles were collected during
VOLUME9,2021 114201

#### 11

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

#### 12

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

#### 13

P.K.Aithaletal.:Building Calendarof Events Databaseby Analyzing Financial Spikes
FIGURE13. Causalimpact.
TABLE1. Variablesandtheirlikelihoodvalues. TABLE2. Variablesandtheirposteriorprobabilityvalues.
a dependent variable. The pre and post-period are set equal
in the model. The result obtained with the BSTS model is
depictedin Fig.13 Theresultobtainedisstatisticallysignifi-
The polarity and subjectivity are given to the BSTS cantastheposteriortail-areaprobabilityis0.04 at95%causal
model as independent variables, and return is provided as impactlevel.
114204 VOLUME9,2021

### Additional Content

#### 1

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

#### 2

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


---

## Content Unique to Old Extraction (not found in markitdown output)

### 1. # Building a Calendar of Events Database by Analyzing Financial Spikes...

# Building a Calendar of Events Database by Analyzing Financial Spikes

### 2. > *Source PDF: Building a Calendar of Events Database by Analyzing Financial Spi...

> *Source PDF: Building a Calendar of Events Database by Analyzing Financial Spikes.pdf*
> *Extraction: Combined — markitdown raw text (base) + previous extraction supplements*

### 3. ## Content from Previous Extraction (not in markitdown output)...

## Content from Previous Extraction (not in markitdown output)

### 4. ### Visual/Chart/Graph Descriptions...

### Visual/Chart/Graph Descriptions

### 5. Received June29,2021, accepted July27,2021, dateofpublication August3,2021, date...

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

### 6. P.K.Aithaletal.:Building Calendarof Events Databaseby Analyzing Financial Spikes...

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

### 7. P.K.Aithaletal.:Building Calendarof Events Databaseby Analyzing Financial Spikes...

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

### 8. P.K.Aithaletal.:Building Calendarof Events Databaseby Analyzing Financial Spikes...

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

### 9. P.K.Aithaletal.:Building Calendarof Events Databaseby Analyzing Financial Spikes...

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

### 10. P.K.Aithaletal.:Building Calendarof Events Databaseby Analyzing Financial Spikes...

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

### 11. P.K.Aithaletal.:Building Calendarof Events Databaseby Analyzing Financial Spikes...

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

### 12. P.K.Aithaletal.:Building Calendarof Events Databaseby Analyzing Financial Spikes...

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

### 13. P.K.Aithaletal.:Building Calendarof Events Databaseby Analyzing Financial Spikes...

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

### 14. P.K.Aithaletal.:Building Calendarof Events Databaseby Analyzing Financial Spikes...

P.K.Aithaletal.:Building Calendarof Events Databaseby Analyzing Financial Spikes
FIGURE10. Timecomponentsofreturn.
6) Polarityand Subjectivity IV. RESULTS
7) Return, Polarity, and Subjectivity To minimize the noise for collecting the news article, one
BMAfindsthelikelihoodofeachcombinationandthepos- has to scrape the HTML. The empty and paid articles are
terior probability of each parameter. A decision is made discardedwhilecollectingthenewsarticles. Eventswerecol-
basedontheposteriorprobabilityofthevariable. Theoverall lectedinthe Mongo DBdatabaseastwocollections, namely,
architectureisdepictedin Fig.1. news items and stocks. News articles were collected during
VOLUME9,2021 114201

### 15. P.K.Aithaletal.:Building Calendarof Events Databaseby Analyzing Financial Spikes...

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

### 16. P.K.Aithaletal.:Building Calendarof Events Databaseby Analyzing Financial Spikes...

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

### 17. P.K.Aithaletal.:Building Calendarof Events Databaseby Analyzing Financial Spikes...

P.K.Aithaletal.:Building Calendarof Events Databaseby Analyzing Financial Spikes
FIGURE13. Causalimpact.
TABLE1. Variablesandtheirlikelihoodvalues. TABLE2. Variablesandtheirposteriorprobabilityvalues.
a dependent variable. The pre and post-period are set equal
in the model. The result obtained with the BSTS model is
depictedin Fig.13 Theresultobtainedisstatisticallysignifi-
The polarity and subjectivity are given to the BSTS cantastheposteriortail-areaprobabilityis0.04 at95%causal
model as independent variables, and return is provided as impactlevel.
114204 VOLUME9,2021

### 18. P.K.Aithaletal.:Building Calendarof Events Databaseby Analyzing Financial Spikes...

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

### 19. P.K.Aithaletal.:Building Calendarof Events Databaseby Analyzing Financial Spikes...

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


---

## Raw Markitdown Extraction (full text)

Received June 29, 2021, accepted July 27, 2021, date of publication August 3, 2021, date of current version August 23, 2021.

Digital Object Identifier 10.1109/ACCESS.2021.3102495

Building a Calendar of Events Database by
Analyzing Financial Spikes

PRAKASH K. AITHAL , (Member, IEEE), U. DINESH ACHARYA , M. GEETHA , (Member, IEEE),
AND PARTHIV MENON
Department of Computer Science and Engineering, Manipal Institute of Technology, Manipal Academy of Higher Education, Manipal 576104, India

Corresponding author: M. Geetha (geetha.maiya@ manipal.edu)

ABSTRACT An event is a piece of news that triggers a change in stock prices. Here, an event study is
undertaken to capture the effect of abnormal returns due to an event. The event can affect the stock market in
the long term or short term. Event research is relevant to both the efﬁcient market hypothesis and behavioral
ﬁnance. In this study, we collected data from websites that manage ﬁnancial and economic data, performed a
sentiment analysis, and correlated news article data with changes in a particular company’s stock prices in the
stock market. Data were collected from two well-known ﬁnancial news websites. We observed a correlation
between stock prices and news items. An event period of one day was considered for the study. The regression
equation determined the relationship between stock returns and polarity and subjectivity. Bayesian model
averaging was performed to identify the effects of polarity and subjectivity on stock returns. Time-series data
were decomposed into components and detrended via regression. Prominent keywords and their polarity
values for a particular day were plotted. An event enhanced some stock returns while adversely affecting
other stocks. We found a variation range of -400 to 200 for different company stocks for the selected period.

INDEX TERMS Event database, social media data, sentiment analysis, data analytics, event studies.

I. INTRODUCTION
Data Science involves extracting information from raw data,
converting information into knowledge and knowledge into
an actionable plan, which has broad applications for ﬁelds
ranging from agriculture and space science to arts and ﬁnance
and which applies to all ﬁelds in which data are used.

Finance is essential to a nation’s growth. Primarily two
kinds of ﬁnancial data are used: quantitative data on variables
such as stock prices, the volume of stock, and the PE ratio
and qualitative data drawn from material such as director
reports, auditor reports, ﬁnancial statements, and ﬁnancial
news. Quantitative data reveal a stock’s movement, while
qualitative data unearth investor sentiment, which is the driv-
ing force behind stock movement.

Stock prices ﬂuctuate based on ﬁnancial events captured
by qualitative data. Events can include management changes,
declarations of dividends, festivals, natural calamities, pan-
demics, recessions, budget and earnings releases, and quar-
terly and yearly statements. Some events, such as festivals and
yearly statements, occur once a year. Changes in management
have either positive or adverse effects. Dividends always

The associate editor coordinating the review of this manuscript and

approving it for publication was Francesco Benedetto

.

decrease the value of a stock. Events have either short-term
consequences or affect stocks over the long run. Events that
affect the stock market for less than a week are considered
to be short-term events and that rest are long-term events.
An event can affect either a particular stock or the entire
stock market. The present event study aims to determine the
abnormal stock return attributable to new information carried
by an event.

Financial events provide both quantitative and qualita-
tive data. Qualitative data such as ﬁnancial news must be
quantiﬁed. One can apply a sentiment analysis to quantify
ﬁnancial news. Sentiment can be measured based on tone,
also known as polarity, or subjectivity. Polarity values vary
from -1 to 1 where a value of -1 denotes extremely nega-
tive sentiment and +1 denotes total positive sentiment. Sub-
jectivity levels vary from 0 to 1. The lower the value of
subjectivity is, the more meaning can be extracted from the
news.

There is a correlation between ﬁnancial news and stock
prices. If the correlation between news and the stock dif-
ference is greater, news inﬂuences the stock market. Event
studies assume the semistrong form of market efﬁciency.
The semistrong form of market efﬁciency applies two
assumptions [1].

114192

This work is licensed under a Creative Commons Attribution 4.0 License. For more information, see https://creativecommons.org/licenses/by/4.0/

VOLUME 9, 2021

P. K. Aithal et al.: Building Calendar of Events Database by Analyzing Financial Spikes

• All publicly available information is reﬂected in stock

prices

• Stock prices vary when new information arrives

The latter assumption is utilized to predict the future value of
a stock in advance. An event changes a stock price, and an
event study can then be utilized to analyze future cashﬂows.

A. BAYESIAN MODEL AVERAGING
An analysis is conducted by ﬁrst selecting the best model
based on a particular criterion and then learning about
the parameters of the chosen model. Any uncertainties of
the model selection process are ignored. Contrary to this
approach, Bayesian model averaging (BMA) learns param-
eters for all candidate models and allows one to select a
model with speciﬁc posterior probabilities. BMA offers the
following advantages over other models:

1) BMA captures the uncertainty of a model.
2) BMA reduces error.
3) BMA retains all model uncertainty until the ﬁnal stage.
4) BMA continuously adjusts model weights.
5) BMA uses a set of models, which is preferable to using

a single model.

B. BAYESIAN STRUCTURAL TIME SERIES
The Bayesian Structural Time Series (BSTS) is a tool used
to identify the causal relationship. It utilizes GPU if present
else runs on CPU. It requires pre-period and post-period as
arguments along with the data.

C. TIME SERIES COMPONENTS
It is helpful to decompose the time series into systematic
and nonsystematic components. Systematic components can
be modeled as either consistent or recurring. Nonsystematic
elements are random and cannot be modeled. The time series
consists of the following components:

1) Level
2) Trend
3) Seasonality
4) Noise

The level is the mean value of the time series. The trend is
either increasing or decreasing in the time series. The repeat-
ing short-term cycle is deﬁned as seasonality. Randomness
present in the time series that cannot be modeled is deﬁned
as noise.

The rest of the paper is organized as follows. Section 2
reviews the related literature, section 3 elaborates on the
methodology used, and section 4 presents the results. Conclu-
sions and avenues for future research are given in section 5.

II. LITERATURE SURVEY
Event studies determine excess returns attributable to a cor-
porate event. Events can range from an earnings release by a
company or budget announcement to a new product launch,
announcement from a competing company, regular ﬁnan-
cial statement or announcement made by a regulatory body.

FIGURE 1. Building and analysis of calendar events database.

Researchers look forward in time and predict the expected
value a ﬁrm will derive from a corporate action announced
to the public. This forward-looking focus of event studies
makes them more potent than other metrics such as return
on investment (ROI) and proﬁts. Investors respond positively
to product launches, sponsorships, and mergers, while they
respond negatively to events such as product recalls. [2]. It has
been observed that historical stock returns are affected by
calendar events. When event studies have been conducted in
developed countries, little related literature is available from
developing nations. The stock market declines on Mondays
and strengthens on Friday. The start of a new month or year
also affects the stock market [3]. The presence of long-term
abnormal returns contradicts the efﬁcient market hypothesis.
Different testing strategies are applicable to asset pricing and
buy-and-hold methodologies [4].

To identify the abnormal returns of the portfolio of stocks
affected by an event, a control portfolio of stocks not affected
by the event is considered as a benchmark, and the results
of both portfolios are compared [5]. Event studies can eval-
uate a government’s policies, as markets react to new news
immediately and adjust their value. If a policy has a positive
effect on the market, it will eventually succeed. Speculation
on an event also has a major impact on the stock market. The
stock market reacts to speculation and is affected long before
an event is announced [6]. Event studies require the use of
a window period under which an event is studied. An event
study evaluation is done based on the methodology utilized
and can involve a regression with a t-test or the use of a capital
asset pricing model (CAPM) with a t-test [7].

VOLUME 9, 2021

114193

P. K. Aithal et al.: Building Calendar of Events Database by Analyzing Financial Spikes

FIGURE 2. News item collection.

An election is a major event that signiﬁcantly inﬂuences
the stock market. A set of stocks are affected by an election.
Elections have an effect for approximately ﬁve months before
and after an event. The announcement of an election and
the election itself greatly impact the stock market [8]. Event
studies have become the defacto standard for measuring
stock prices during and after an event. Abnormal returns are
determined either as the residuals from standard normalized
benchmarks or as a dummy variable in a regression equa-
tion [9]. News can be classiﬁed as repeating or surprising.
A repeating event can be easily predicted, and markets adjust
to such an event. A surprising event, on the other hand, leads
to abnormal returns. This is the case because such an event
is unknown, and no estimate has been made, which results in
risk that must be addressed [10].

In [11], the authors conclude that intraday abnormal returns
are normally distributed, and Pattel test statistics function
better than other statistical measures in measuring abnormal
returns. The market responds to both good and bad news.
Good news reduces risk, while a piece of bad news increases
it. The efﬁcient market hypothesis and behavioral ﬁnance
apply in some cases and fail in others. Both have explanatory
power and must be explored. According to event studies,
stocks will not alter prices unless and until a piece of news

arrives [12]. Events do not have structure and cannot be stored
in relational databases. Therefore, Not Only Structured Query
Language (NoSQL) databases are utilized to store informa-
tion on such events. NoSQL databases require no downtime
and are different from regular relational database manage-
ment systems (RDBMSs), as they can scale well at a lower
cost. A RDBMS has strict atomicity consistency isolation
durability (ACID) properties, while NoSQL databases adhere
to the consistency availability and partition tolerance (CAP)
theorem. Complex event processing cannot be achieved using
a regular RDBMS, so one must in this case use NoSQL
databases for faster processing. Complex event processing
requires low latency, high throughput, and temporal and spa-
tial event processing capabilities [13]. Stock prices move
with a correlated random walk rather than with an uncorre-
lated random walk. Correlations can be found through event
studies [14].

One comment posted on the Yahoo ﬁnancial website
almost crashed the Nevada company, after which the com-
pany had to call a press conference to prove that its fun-
damentals are strong. This incident demonstrates that event
studies are critical. The simplest measures used include the
number of times a stock name is mentioned, the frequency
of certain keywords, and the sentiment of news about a

114194

VOLUME 9, 2021

P. K. Aithal et al.: Building Calendar of Events Database by Analyzing Financial Spikes

FIGURE 3. Stock collection.

given company [15]. There are two types of events, namely,
simple events and complex events. A simple event occurs
instantaneously and independent of other events. A complex
event is a set of events that are interrelated [16]. The tradi-
tional investment strategy takes months or years to arrive at a
decision. The advancement of information technology in the
realm of ﬁnance has made the decision-making process faster.
Events can be analyzed in milliseconds using information
technology [17].

The efﬁcient market hypothesis states that market prices
reﬂect current news and that as new news arrives, prices

change. Efﬁcient market theory considers investors to be
unemotional. Behavioral ﬁnance, on the other hand, treats
investors as an emotional entity and assumes news events
to affect investments. Both the efﬁcient market hypothesis
and behavioral ﬁnance agree that news events are respon-
sible for abnormal returns [18]. Text mining is used in
event studies, and it requires words to be segmented, and
spaces are used to separate words. Stop words need to
be removed. The sentiment of a sentence and frequency
of words measure the quantum of the jump of stock
prices [19].

VOLUME 9, 2021

114195

P. K. Aithal et al.: Building Calendar of Events Database by Analyzing Financial Spikes

FIGURE 4. Keywords and their impact on investor sentiment for mean polarity.

Sentiment analysis is widely used in the ﬁeld of ﬁnance.
The method is also known as opinion mining. Sentiment anal-
ysis is broadly classiﬁed into emotion recognition and tone
analysis or polarity analysis. Opinion mining can be classiﬁed
based on whether a dictionary or machine learning is used
to arrive at an opinion. Machine learning-based approaches
require domain-speciﬁc content to work correctly, while dic-
tionaries are costly [20]. Event analysis must be conducted by
batch processing of historical events. The event pattern must
support mathematical, logical, and statistical patterns among
events [21]. An event study involves several steps and many
choices must be made at each step. Thus, there is no standard
means of conducting an event study. The timing of an event
study and the event window size need to be deﬁned by the
researcher. Surprise events need to be evaluated through event
studies [22].

A historical analysis of 145100 news articles has shown a
signiﬁcant relationship between tone and companies’ perfor-
mance. However, a single instance of news does not have an

impact on stock movement. If a company calls a conference
call and the tone is positive, the stock of that company will
move multifold in a positive direction, resulting in an abnor-
mal positive return [23]. The global ﬁnancial crisis resulted in
governments intervening and taking regulatory action. Global
regulatory action is an event that affects the stock market
worldwide [24].

The coronavirus pandemic has affected US asset prices.
The pandemic has resulted in high levels of volatility in
small and unstable markets. Stock markets are interrelated
and sensitive to news. Media coverage on the pandemic has
resulted in an adverse crash in some highly volatile stocks and
unstable industries. The global stock markets are becoming
interdependent. Even when a single stock market is affected,
this impact spreads to other stock markets [25]. [26] studied
corporate social responsibility with the help of a dictionary.
The authors showed that when a ﬁrm is not responsible and
this results in a negative event, its stock price will fall drasti-
cally. Event and ﬁrm performance are directly proportional.

114196

VOLUME 9, 2021

P. K. Aithal et al.: Building Calendar of Events Database by Analyzing Financial Spikes

FIGURE 5. Keywords and their impact on investor sentiment for mean subjectivity.

Narrative economics has developed into a ﬁeld of its own.
A narrative event inﬂuences a stock, and numerous examples
of this phenomenon prove this fact [27]. The pandemic has
caused a loss to the economy. A historical inﬂuenza revealed
that a pandemic has effects similar to those of inﬂuenza but
on a larger scale. When a pandemic affects key persons, this
will have a tremendous impact on the stock market [28].

Event studies cover two time periods, pre-event and
postevent periods, and examine two groups, control and
treatment groups. If the event period varies, its average can
be taken as deﬁned in [29]. Event information can be soft
or hard. Hard information can be quantiﬁed and quickly
converted into a number. Soft information requires con-
text. When one separates soft information from its source,
it becomes useless. Hard information is known as quanti-
tative, while soft information is known as qualitative [30].
Big data technologies can store qualitative data on events
rather than quantitative data along. The ﬁeld of ﬁnance has
concentrated on the use of quantitative data for analyses.

With the emergence of big data technologies, the storage and
processing of qualitative data have become more accessible.
The focus of the researcher has thus switched from the use of
quantitative data to the use of qualitative data [31].

[32] calculate economic uncertainty based on newspaper
coverage frequency and focus on monetary policy making.
On the other hand, our work deals with event studies on how
much abnormal return can be obtained from an event.

The above literature review that event studies require the
examination of an event period and that there are many
means of performing them. An event can be studied with two
groups, control and treatment groups, or the standard index
value. There are a variety of ways to conduct an event study.
An event study must determine the period of research and
adopt a regression equation.

III. METHODOLOGY
Our methodology is presented in subsections III-A and III-B,
respectively, titled Tools and Process.

VOLUME 9, 2021

114197

P. K. Aithal et al.: Building Calendar of Events Database by Analyzing Financial Spikes

FIGURE 6. Mean polarity in a day.

A. TOOLS
The economictimes.com and the moneycontrol.com are cho-
sen for news article collection as these are the prominent
ﬁnancial websites in India. A thorough review of the two
websites used, namely, moneycontrol.com and economic-
times.com, revealed which routes gave relevant data. Then,
the type of database and database schema were selected.

We selected Python as our language of choice for data col-
lection and its related mathematical analysis and we selected
MongoDB as are database. MongoDB is a scalable and fast
NoSQL database and is known for its efﬁcient querying
capabilities. Initially, PyMongo, a native Python driver for
MongoDB, was used to query the database. Subsequently,
the code was modiﬁed to use mongoengine—a Python library

114198

VOLUME 9, 2021

P. K. Aithal et al.: Building Calendar of Events Database by Analyzing Financial Spikes

FIGURE 7. Nifty 50 variation in a day.

that acts as an object document mapper for MongoDB. It then
was easier to specify schemas and manipulate data obtained
from the database in the form of objects with each ﬁeld of a
known type.

A request library was used to retrieve content from web-
sites. BeautifulSoup, a Python package for parsing Hyper
Text Markup Language (HTML) and eXtensible Markup
Language (XML) documents, was used to analyze the data,
scrape webpages, and parse the data into useful data. In addi-
tion, the pandas library was used to manipulate.csv ﬁles and
other data frames to efﬁciently structure the pre-existing data.
TextBlob, a Python library built from the Natural Lan-
guage Tool Kit (NLTK), provides user-friendly interfaces
for performing natural language processing (NLP) tasks

such as sentiment analyses of text. The NLTK was also
used later in the project to process text and extract relevant
information.

Git (and GitHub) was used for version control of the project
ﬁles and folders. To make the code portable, environmental
variables, along with the dotenv module, were used to dynam-
ically load variables during execution.

B. PROCESS
Events were collected from moneycontrol.com and eco-
nomictimes.com. The data are stored in the MongoDB
database. Two collections are stored in the database—detailed
stock values and events from the websites. The data ﬁelds of
these two collections are as shown below.

VOLUME 9, 2021

114199

P. K. Aithal et al.: Building Calendar of Events Database by Analyzing Financial Spikes

FIGURE 8. Correlation between stock returns and polarity.

FIGURE 9. Correlation between stock returns and subjectivity.

NewsItem(News,
Date,
Time,
CompanyName,
Polarity,
Subjectivity)
Stocks(CompanySymbol,
OpeningValue,
ClosingValue,
Difference)
Field news stores scrapped news from the two websites.
When information is published, the date and time are stored
in the date and time ﬁelds, respectively. The company name
is considered optional since news may affect multiple com-
panies without citing a company name. Polarity values vary
from -1 to 1. If TP represents the total number of positive
words and TN represents the total number of negative words,
polarity is deﬁned as follows (1).

Polarity = (TP − TN )/(TP + TN )

(1)

Subjectivity is a type of probability that determines how
objective a sentence is. Subjectivity, as it is a probability
value, varies from 0 to 1. Further details on computing sub-
jectivity can be found in [33].

The company symbol includes a ticker symbol for the
Nifty 50 companies. The opening price is the stock price at
the beginning of the day. The closing price is the price of the
stock at the end of the day. The difference ﬁeld contains the
difference between closing and opening prices.

We collected and processed our data over the following

steps.

1) The date and time of the event and the actual news

article were extracted from the web.

2) The opening and closing prices of various companies’
stock prices were collected, and the difference between
these prices was computed. Data regarding companies
and their symbols were obtained from NSE. These data
were then used to dynamically obtain the stock values

for every required instance from the Yahoo Finance
website.

3) The correlation between the difference in opening and
closing stock values and the various news articles of a
company is found using equations (2) and (3), where
CP and CS denote the correlations of polarity and sub-
jectivity with returns, respectively. P denotes polarity,
R denotes the return, and S denote subjectivity, relating
the news to stock ﬂuctuations.

CP =

CS =

(cid:80)(Pi − mean(P)) ∗ (Ri − mean(R))
(cid:112)(cid:80)(Pi − mean(P))2 ∗ (cid:80)(Ri − mean(R))2
(cid:80)(Si − mean(S)) ∗ (Ri − mean(R))
(cid:112)(cid:80)(Si − mean(S))2 ∗ (cid:80)(Ri − mean(R))2

(2)

(3)

4) The following graphs are plotted

• Keywords and their impact on investor sentiment

for mean polarity

• Keywords and their impact on investor sentiment

for mean subjectivity

• Mean polarity on a particular day
• Variation in Nifty 50 stocks within a day due to a

budget event

• Correlation between stock returns and polarity
• Correlation between stock returns and subjectivity
As the script is executed, the news item collection is
updated without duplicates, and stock collection is refreshed.
BMA utilizes a linear regression model for predictions,
considers all parameters and applies linear regression mod-
els for all possible combinations of the parameters. Two
independent variables are used in the current paper, namely,
polarity and subjectivity, and one dependent variable, return,
is also used. BMA applies linear regression in the following
parameter combinations:

1) Return
2) Polarity
3) Subjectivity
4) Return and Polarity
5) Return and Subjectivity

114200

VOLUME 9, 2021

P. K. Aithal et al.: Building Calendar of Events Database by Analyzing Financial Spikes

FIGURE 10. Time components of return.

6) Polarity and Subjectivity
7) Return, Polarity, and Subjectivity

BMA ﬁnds the likelihood of each combination and the pos-
terior probability of each parameter. A decision is made
based on the posterior probability of the variable. The overall
architecture is depicted in Fig. 1.

IV. RESULTS
To minimize the noise for collecting the news article, one
has to scrape the HTML. The empty and paid articles are
discarded while collecting the news articles. Events were col-
lected in the MongoDB database as two collections, namely,
news items and stocks. News articles were collected during

VOLUME 9, 2021

114201

P. K. Aithal et al.: Building Calendar of Events Database by Analyzing Financial Spikes

FIGURE 11. Time components of polarity.

Indian budgeting time, and the articles have a positive tone.
The sampled database collections of news items and stocks
are described in Fig.2 and Fig.3 respectively. We collected
information on the date and time of each news item, on the
article itself, and on degrees of article polarization and sub-
jectivity. For stocks, we collected stock symbols, the dates of
closing and opening prices, the opening prices of particular
days, the closing prices of particular days, and the differences
between closing and opening prices.

The keywords and their impacts on the given investor for
the given time period with respect to tone are presented
in Fig. 4, and the same results with respect to subjectivity
are shown in Fig. 5 From the ﬁgures, it is clear that news
on budgets has a positive tone. The heat maps depicted
in Figures 1 and 2 show that the article keywords have a
positive tone, and words such as ‘‘high‘‘ are more popular
than word such as ‘‘loss‘‘ during the period. The word ‘‘loss‘‘

is not used on February 1st, 2021, India’s budget day. The
word ‘‘high‘‘ is often used in the news articles.

The mean polarity within a day is shown in Fig. 6 During
the Indian annual budget event held in Feb. 2021, the mean
polarity value is positive for the entire week of the event.

The variation in the stock prices of Nifty 50 companies is
shown in Fig. 7. From Fig.7, it is clear that due to budget
events, some individual stocks are positively affected, while
others are negatively affected. The variation ranges from -
400 to 200.

Fig.8 and Fig.9 demonstrate the effect of polarity and
subjectivity on the stock returns of Nifty 50 companies. For
a given polarity and subjectivity value, the companies react
differently. From Figures 8 and 9, the return is not approxi-
mately zero for all of the companies. We also ﬁnd variation
ranging from -400 to 200 due to the event. Certain companies
react to speciﬁc events.

114202

VOLUME 9, 2021

P. K. Aithal et al.: Building Calendar of Events Database by Analyzing Financial Spikes

FIGURE 12. Time components of subjectivity.

The time components of return, polarity, and subjectivity
are given in Fig. 10 to Fig. 12, respectively. The ﬁgures show
that the seasonality component is not present in the data, and
the observed component reﬂects a trend.

The regression equation of degree one with returns as the
dependent variable and polarity as an independent variable
is illustrated by equation 4. The regression equation with
returns as a dependent variable and subjectivity as an inde-
pendent variable is presented as equation 5. The polynomial
regression equation of degree ﬁve with returns as a dependent
variable and polarity as an independent variable is given by
equation 6. The polynomial regression equation of degree
ﬁve with returns as a dependent variable and subjectivity
as an independent variable is written as equation 7. The top
15 keywords of the news items for the budget week were iden-
tiﬁed. Table3 lists the top 15 keywords in descending order
of frequency. The table shows that the development of coron-
avirus vaccines has led to positive sentiment among investors.
Pandemic-related words account for 5 of the 15 most frequent
words.

Returns = 0.003 + 0.009 ∗ Polarity
Returns = 0.0002 + 0.007 ∗ Subjectivity

(4)
(5)

Returns = 0.001 − 0.122 ∗ Polarity

−0.025 ∗ (Polarity)2
−11.81 ∗ (Polarity)3
+0.46 ∗ (Polarity)4
+241.89 ∗ (Polarity)5

Returns = −1.02 − 12.68 ∗ Subjectivity

−60.48 ∗ (Subjectivity)2
+138.82 ∗ (Subjectivity)3
−152.97 ∗ (Subjectivity)4
+64.47 ∗ (Subjectivity)5

(6)

(7)

The BMA results are tabulated in tables 1 and 2 and show
that polarity and subjectivity improve the model’s perfor-
mance. Polarity outperforms subjectivity. It is better to con-
sider polarity and subjectivity as model parameters. We also
ﬁnd that certain parameters inﬂuence the model, but they are
not included in the model.

Spearman rho and Kendall tau rank correlations are found
as well as a slightly positive correlation between polarity and
returns. Additionally, we ﬁnd a slightly positive correlation
between subjectivity and returns.

VOLUME 9, 2021

114203

P. K. Aithal et al.: Building Calendar of Events Database by Analyzing Financial Spikes

FIGURE 13. Causal impact.

TABLE 1. Variables and their likelihood values.

TABLE 2. Variables and their posterior probability values.

The polarity and subjectivity are given to the BSTS
model as independent variables, and return is provided as

a dependent variable. The pre and post-period are set equal
in the model. The result obtained with the BSTS model is
depicted in Fig. 13 The result obtained is statistically signiﬁ-
cant as the posterior tail-area probability is 0.04 at 95% causal
impact level.

114204

VOLUME 9, 2021

P. K. Aithal et al.: Building Calendar of Events Database by Analyzing Financial Spikes

TABLE 3. Keywords and their frequencies of use during budget week
Feb. 1 to Feb. 7, 2021.

[5] C. Castro-Iragorri, ‘‘Does the market model provide a good counterfac-
tual for event studies in ﬁnance?’’ Financial Markets Portfolio Manage.,
vol. 33, no. 1, pp. 71–91, Mar. 2019.

[6] L. Raynaud. (2017). Evaluating Industrial Policies Using Event-Studies:
Evidence From Stock Market Reactions to Obama’s Stimulus Package.
[Online]. Available: https://www.skidmore.edu

[7] K. Schmidheiny and S. Siegloch,

‘‘On event study designs and
distributed-lag models: Equivalence, generalization and practical impli-
cations,’’ CESifo Working Paper, IZA, Germany, Tech. Rep. 12079,
2019.

[8] I. P. A. Argantha and I. M. S. N. Sudirman, ‘‘Stock market reaction to the
event of Indonesia’s general election events in 2019,’’ Amer. J. Hum. Social
Sci. Res., vol. 4, no. 1, pp. 202–208, 2019.

[9] J. Binder, ‘‘The event study methodology since 1969,’’ Rev. Quant. Finance

Accounting, vol. 11, no. 2, pp. 111–137, 1998.

[10] N. Bohn, F. A. Rabhi, D. Kundisch, L. Yao, and T. Mutter, ‘‘Towards
automated event studies using high frequency news and trading data,’’
in Proc. Int. Workshop Enterprise Appl. Services Finance Ind. Spain:
Springer, 2012, pp. 20–41.

[11] J. Afﬂeck-Graves, C. M. Callahan, and R. Ramanan, ‘‘Detecting abnor-
mal bid-ask spread: A comparison of event study methods,’’ Rev. Quant.
Finance Accounting, vol. 14, no. 1, pp. 45–65, 2000.

V. CONCLUSION AND FUTURE WORK
Two prominent ﬁnancial websites, namely, moneycon-
trol.com and economictimes.com, were scraped to build a cal-
endar of events database. The Python language was utilized
to scrape the websites. Time components were decomposed
from data, and we found the only trend is present in the
data, which was detrended via regression. BMA was applied
with linear regression, and we found polarity to be a more
signiﬁcant parameter than subjectivity. We also show that the
discovery of coronavirus vaccines is positively affecting the
Indian stock market. To summarize, this article contributes to
the existing body of knowledge in the following ways:

1) A calendar of events database is created.
2) Insights are derived from the events database with

quantitative and qualitative data.

3) We ﬁnd that the invention of coronavirus vaccines has

impacted the stock market.

Event studies are a useful means of analyzing the stock
market, as stock markets are sensitive to events. Portfo-
lio optimization via event studies can be considered in
future enhancements. Further work can also categorize events
into the following categories: global, national, sectoral, and
company-speciﬁc events. One can scrape the other ﬁnancial
websites in the future along with moneycontrol.com and
economictimes.com websites.

REFERENCES
[1] B. G. Malkiel, ‘‘The efﬁcient market hypothesis and its critics,’’ J. Econ.

Perspect., vol. 17, no. 1, pp. 59–82, 2003.

[2] A. Sorescu, N. L. Warren, and L. Ertekin, ‘‘Event study methodology in
the marketing literature: An overview,’’ J. Acad. Marketing Sci., vol. 45,
no. 2, pp. 186–207, Mar. 2017.

[3] D. Winkelried and L. A. Iberico, ‘‘Calendar effects in Latin American stock
markets,’’ Empirical Econ., vol. 54, no. 3, pp. 1215–1235, May 2018.
[4] J. S. Ang and S. Zhang, ‘‘Evaluating long-horizon event study method-
ology,’’ in Handbook of Financial Econometrics and Statistics, C.-F. Lee
and J. C. Lee, Eds. New York, NY, USA: Springer, 2015, pp. 383–411, doi:
10.1007/978-1-4614-7750-1_14.

[12] G. Bird, W. Du, and T. Willett, ‘‘Behavioral ﬁnance and efﬁcient markets:
What does the euro crisis tell us?’’ Open Econ. Rev., vol. 28, no. 2,
pp. 273–295, Apr. 2017.
[13] B. Fang and P. Zhang,

in Big Data
Concepts, Theories, and Applications. Australia: Springer, 2016,
pp. 391–412.

‘‘Big data in ﬁnance,’’

[14] H. Stanley, X. Gabaix, P. Gopikrishnan, and V. Plerou, ‘‘Correlated ran-
domness: Rare and not-so-rare events in ﬁnance,’’ in Practical Fruits of
Econophysics. Japan: Springer, 2006, pp. 2–18.

[15] D. Shen and S.-H. Chen, ‘‘Big data ﬁnance and ﬁnancial markets,’’ in Big
Data in Computational Social Science and Humanities. Taiwan: Springer,
2018, pp. 235–248.

[16] Z. Milosevic, A. Berry, W. Chen, and F. A. Rabhi, ‘‘An event-based
model to support distributed real-time analytics: Finance case study,’’ in
Proc. IEEE 19th Int. Enterprise Distrib. Object Comput. Conf., Sep. 2015,
pp. 122–127.

[17] C.-F. Huang, H.-C. Li, and B.-R. Chang, ‘‘An intelligent computational
ﬁnance model for microstructure-based trading system,’’ in Proc. Int. Conf.
Appl. Syst. Innov. (ICASI), May 2016, pp. 1–4.

[18] Q. Li, J. Tan, J. Wang, and H. Chen, ‘‘A multimodal event-driven LSTM
model for stock prediction using online news,’’ IEEE Trans. Knowl. Data
Eng., early access, Jan. 23, 2020, doi: 10.1109/TKDE.2020.2968894.
[19] Z. Li, Y. Cai, and S. Hu, ‘‘Research on systemic ﬁnancial risk measurement
based on HMM and text mining: A case of China ﬁnancial market,’’ IEEE
Access, vol. 9, pp. 22171–22185, 2021.

[20] A. Gupta, V. Dengre, H. A. Kheruwala, and M. Shah, ‘‘Comprehensive
review of text-mining applications in ﬁnance,’’ Financial Innov., vol. 6,
no. 1, pp. 1–25, Dec. 2020.

[21] Z. Milosevic, W. Chen, A. Berry, and F. A. Rabhi, ‘‘An open architec-
ture for event-based analytics,’’ Int. J. Data Sci. Anal., vol. 2, nos. 1–2,
pp. 13–27, Dec. 2016.

[22] J. Distler, ‘‘The event study as a research method for measuring M&A
performance,’’ in Acquisitions by Emerging Multinational Corporations.
Germany: Springer, 2018, pp. 221–268.

[23] B. R. Upreti, P. M. Back, P. Malo, O. Ahlgren, and A. Sinha, ‘‘Knowledge-
driven approaches for ﬁnancial news analytics,’’ in Network Theory and
Agent-Based Modeling in Economics and Finance. Singapore: Springer,
2019, pp. 375–404.

[24] M. Hoesli, S. Milcheva, and A. Moss, ‘‘Is ﬁnancial regulation good or bad
for real estate companies?—An event study,’’ J. Real Estate Finance Econ.,
vol. 61, pp. 369–407, Oct. 2017.

[25] H. Liu, A. Manzoor, C. Wang, L. Zhang, and Z. Manzoor, ‘‘The COVID-19
outbreak and affected countries stock markets response,’’ Int. J. Environ.
Res. Public Health, vol. 17, no. 8, p. 2800, Apr. 2020.

[26] G. Capelle-Blancard and A. Petit, ‘‘Every little helps? ESG news and stock
market reaction,’’ J. Bus. Ethics, vol. 157, no. 2, pp. 543–565, Jun. 2019.
[27] R. J. Shiller, Narrative Economics: How Stories go Viral and Drive Major
Economic Events. Princeton, NJ, USA: Princeton Univ. Press, 2020.
[28] V. Y. Fan, D. T. Jamison, and L. H. Summers, ‘‘Pandemic risk: How large
are the expected losses?’’ Bull. World Health Org., vol. 96, no. 2, p. 129,
2018.

VOLUME 9, 2021

114205

P. K. Aithal et al.: Building Calendar of Events Database by Analyzing Financial Spikes

[29] A. Goodman-Bacon, ‘‘Difference-in-differences with variation in treat-
ment timing,’’ J. Econometrics, 2021. [Online]. Available: https://www.
sciencedirect.com/science/article/pii/S0304407621001445, doi: 10.1016/
j.jeconom.2021.03.014.

[30] J. M. Liberti and M. A. Petersen, ‘‘Information: Hard and soft,’’ Rev.

Corporate Finance Stud., vol. 8, no. 1, pp. 1–41, 2019.

[31] M. P. Bach, Ž. Krstić, S. Seljan, and L. Turulja, ‘‘Text mining for big data
analysis in ﬁnancial sector: A literature review,’’ Sustainability, vol. 11,
no. 5, p. 1277, Feb. 2019.

[32] S. J. Davis, S. R. Baker, and N. Bloom, ‘‘Measuring economic policy
uncertainity,’’ Nat. Bur. Econ. Res., Cambridge, MA, USA, Tech. Rep.
21633, 2015, pp. 1–75.

[33] A. Kamal, ‘‘Subjectivity classiﬁcation using machine learning techniques
for mining feature-opinion pairs from web opinion sources,’’ 2013,
arXiv:1312.6962. [Online]. Available: http://arxiv.org/abs/1312.6962

PRAKASH K. AITHAL (Member, IEEE) is cur-
rently pursuing the Ph.D. degree with Manipal
Academy of Higher Education, Manipal, India.
He is currently an Assistant Professor with the
Department of Computer Science and Engineer-
ing, Manipal Institute of Technology, Manipal
Academy of Higher Education. He has presented
several papers at national and international confer-
ences, and his work has been published in several
international journals. His current research interest

includes data analytics in the ﬁnancial domain.

U. DINESH ACHARYA received the Ph.D. degree
from Manipal Academy of Higher Education,
Manipal, India, in 2008. He is currently a Pro-
fessor with the Department of Computer Science
and Engineering, Manipal Institute of Technology,
Manipal Academy of Higher Education. He has
presented several papers at national and interna-
tional conferences, and his work has been pub-
lished in several international journals. His current
research interests include data analytics in the

healthcare, agriculture, and ﬁnancial sectors.

M. GEETHA (Member, IEEE) received the Ph.D.
degree from NITK, Surathkal, in 2010. She is cur-
rently a Professor with the Department of Com-
puter Science and Engineering, Manipal Institute
of Technology, Manipal Academy of Higher Edu-
cation, Manipal, India. She has presented several
papers at national and international conferences,
and her work has been published in several
international journals. Her current research inter-
ests include data mining and text mining in the

healthcare and ﬁnancial sectors.

PARTHIV MENON is currently pursuing the
B.Tech. degree with the Department of Computer
Science and Engineering, Manipal Institute of
Technology, Manipal Academy of Higher Educa-
tion, Manipal, India. His research interest includes
data analytics.

114206

VOLUME 9, 2021
