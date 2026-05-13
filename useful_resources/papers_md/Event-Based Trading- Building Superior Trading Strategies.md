<!-- Page 1 -->

Event-Based Trading: Building Superior Trading Strategies
with State-of-the-Art Information Extraction Tools
Complete Research Paper
Zvi Ben-Ami Ronen Feldman
The Hebrew University The Hebrew University
zvi.benami@mail.huji.ac.il ronen.feldman@huji.ac.il
Abstract
Advanced sentiment analysis systems, which typically produce a positive or negative
signal on firms for financial traders, consider financial events when calculating aggregate
sentiment scores. However, these systems typically produce overly noisy signals for traders
because they aggregate the information culled from the events rather than address each
distinct event-type differently (all event-types receive the same weight in a text’s final
sentiment score). We demonstrate that event-types have different informative values for
traders, and specific event-types may be particularly relevant for specific investment
periods. In this work we evaluate and compare the effects of event-type signals and
sentiment analysis signals, automatically extracted from a given text, on stock prices in
different investment periods. We show that traders using trading strategies based on event-
types and sentiment analysis obtain superior performance compared to acceptable
benchmarks.
Introduction
Current technological advancements in data science open new avenues to financial traders,
who are constantly looking for new strategies, methods and tools to assist them in making
better investment decisions. Powerful computers regularly monitor and analyze structured
1
Electronic copy available at: https://ssrn.com/abstract=2907600


<!-- Page 2 -->

Event-Based Trading
data to provide signals to traders, and High Frequency Trading algorithms even execute
trades on their behalf, with the goal of generating arbitrage profits from short-term
fluctuations (Terrence and Riordan 2013; Treleaven et al. 2013). Although traders
commonly use computers to process and analyze financial structured data, they less
frequently rely on computational abilities to process and analyze unstructured data.
Technological advancements in the field of Natural Language Processing (NLP) offer
traders with tools necessary to analyze unstructured financial data. Sentiment analysis, for
example, which has captured the attention of financial traders around the world, offers
great value in determining the tone of large amounts of textual data about firms, markets,
commodities, and other financial instruments (Feldman 2013; Loughran and Mcdonald
2016). By providing additional signals to traders that can be considered when devising their
trading strategies, sentiment analysis can beneficially support trading decisions. Most
existing studies on sentiment analysis in the field of finance look at the aggregate sentiment
score assigned to firms after analyzing texts from trade or social digital media (Bollen et
al. 2011; Feldman et al. 2010; Tetlock et al. 2008). Irrespective of the accuracy of the
applied sentiment analysis method, most systems tend to miss one crucial factor in the
analysis, which is context. By counting positive and negative words, or even by performing
a flawless sentiment analysis, one can at most obtain the overall tone of the article, which,
we argue here, can be a very noisy signal to traders. In this work, we suggest that one
should not treat all sentiment expressions evenly, and different event-types that embed
sentiments should be considered and used as trading signals relevant for specific
investment periods.
2
Electronic copy available at: https://ssrn.com/abstract=2907600


<!-- Page 3 -->

Event-Based Trading
News of financial events is presumed to affect stock prices. There are different opinions in
the literature on the extent to which news is reflected in stock prices and the time lag
involved before such news is reflected. These questions are fundamentally tied to Eugene
Fama’s Efficient Market Hypothesis (EMH) (Fama 1970). Many studies have since
attempted to identify methods in which publicly available information can, nevertheless,
be used to obtain excess returns (Antweiler and Frank 2004; Davis et al. 2012; Finnerty
1976; Grossman and Stiglitz 1980). These studies argue against Fama’s hypothesis that
states that it is impossible to beat the market because prices already incorporate and reflect
all relevant information.
In this work, we analyze which events reported on digital media can be valuable signals
for traders in making trading decisions and whether they enhance or diminish the value of
sentiment analysis signals. Subsequent questions are whether events should be considered
separately or aggregately, and for which periods relevant securities should be held after
obtaining such signals. To answer these questions, we used VIP (Visual Information
Extraction Platform), an automatic information extraction platform, for the text analysis
and the Quantopian Research Platform for the performance analysis. VIP processes,
analyzes, and converts unstructured financial data into structured data sets of stock-specific
sentiments tied to business events. This tool allows us to identify various sentiments and
events that are related to public companies. Using Quantopian, one can create trading
algorithms and test them against historical data (a process known as backtesting). We also
used Alphalens, a component of the Quantopian Research Platform, which was designed
to analyze alpha factors.
3
Electronic copy available at: https://ssrn.com/abstract=2907600


<!-- Page 4 -->

Event-Based Trading
Related work
The question of how to exploit unstructured firm-related digital data has occupied many
financial economists and computer scientists. Early work argues that by merely observing
the tone of the text about a particular firm on digital media, traders can obtain a valuable
signal. For example, Tetlock et al. (2008) found that the occurrence of negative words in
news articles about firms could function as a valuable signal for traders. With advances in
the field of NLP, researchers have adopted more sophisticated approaches to sentiment
analysis and information extraction, and applied them on financial texts (Davis et al. 2012;
Heston and Sinha 2015; Li et al. 2014; Schumaker and Chen 2009; Zhang and Skiena
2010). Boudoukh et al. (2016) showed that public firm-level news is a meaningful
component of stock return variance. Distinguishing between information that is relevant to
the firm’s fundamentals (e.g., analyst recommendations, acquisitions, products, etc.) and
irrelevant information (e.g. executive changes, dividends, etc.), they argued that only
certain events should be considered relevant signals. This level of granularity is, however,
inadequate on two counts. First, after distinguishing between events that are likely to
impact stock prices and those that are not, these researchers treated all relevant events
equally in their analysis. Rather than considering the individual impact of every event -
type, the researchers looked at the aggregate effect of all relevant events. Second, we argue
that events should be further classified to distinguish between subsets of events. For
example, product events might involve product trials or product recalls. Each event-type
would obviously have a very different impact on stock prices. Boudoukh et al. ( 2016) do
not make such distinction. Ding et al. (2014) used a deep neural network model in order to
4
Electronic copy available at: https://ssrn.com/abstract=2907600


<!-- Page 5 -->

Event-Based Trading
predict stock price movements, but similarly to Boudoukh et al. (2016), they aggregated
events’ sentiments to determine the polarity of the news.
To the best of our knowledge, no other study considers the impact of individual events,
identified by an automatic Information extraction platform, and compares the excess return
generated when such signals are used separately or together with the sentiment scores
(tones), over various time horizons.
Data and data processing
To test which event-types on digital media might be valuable signals for investors making
trading decisions, whether events should be used as isolated or aggregate signals, and to
compare event signals to sentiment signals we obtained three corpora:
● Corpus A: 6,750 articles1 on various firms and indices
● Corpus B: 1,000,000 articles 2on various firms
● Corpus C: 1,567,260 Dow Jones newswires articles3.
We processed these corpora using VIP to extract sentiments and events related to different
firms. VIP processes, analyzes, and converts unstructured financial data into structured
data sets of stock-specific sentiments tied to business events. VIP uses state-of-the-art NLP
1 downloaded from Yahoo Finance (https://finance.yahoo.com) and Google Finance
(https://www.google.com/finance)
2 downloaded from Yahoo Finance (https://finance.yahoo.com) and Google Finance
(https://www.google.com/finance)
3 Downloaded from the Dow Jones newswires website (http://www.dowjones.com/products/dj-
news). The articles were published between March 2011 and March 2016 and discuss 613 different
firms traded on the NYSE, NASDAQ and AMEX that are mostly included in the S&P 500 index.
5
Electronic copy available at: https://ssrn.com/abstract=2907600


<!-- Page 6 -->

Event-Based Trading
techniques, specifically a proprietary, leading-edge pattern language designed to identify
relevant linguistic patterns in context-free grammar involving sentiments or business
events. The core of the VIP system is an integrated sentiment analysis engine that
incorporates knowledge from multiple sources, and performs a relevance analysis to
determine sentiment and event scores of financial content of potential significance for
financial traders. Using VIP, we generated the following outputs for every company: Daily
𝑃−𝑁
sentiment score – ranges between -1 and 1 and is calculated as: 𝑆𝑐𝑜𝑟𝑒 = (where P
𝑃+𝑁+3
and N represent the number of positive and negative sentiment expressions, respectively);
Time-weighted sentiment score – the sum of all sentiment scores of a particular ticker,
while weighting older articles with a decay factor of a half-life of 10 days.; and Daily event
count – number of positive and negative mentions of events of a particular type on a certain
traded company, e.g., number of positive or negative Financial Results mentioned on a
given company in a given day.
To conduct a financial analysis of VIP’s output, we followed a few preprocessing steps:
we removed tickers that did not exist on the first date of the analysis or that since changed.
(this step is required for mapping symbols to security objects in Quantopian); we calculated
the daily event score (positives count minus negatives count) for each event-type; we
calculated the aggregated daily event score for all event-types, for each ticker; we
normalized the event score and the aggregated daily event scores by log(n+1) to handle
skewness of the data. We ended up with the following daily signals for each company in
our corpora: Daily Sentiment Score; Time-weighted Sentiment Score; Normalized
Aggregated Event Score; and Normalized Event Score by event-type.
6
Electronic copy available at: https://ssrn.com/abstract=2907600


<!-- Page 7 -->

Event-Based Trading
The Quantopian Research Platform
The Quantopian Research platform4 is an IPython notebook environment that is used for
research and data analysis during algorithm creation. It is also used for analyzing the past
performance of algorithms. The IDE (interactive development environment) is used for
writing algorithms and kicking off backtests using historical data. We uploaded the signals
described in the previous section to the Quantopian Research platform and used the
Alphalens Python package for the analysis. Alphalens5 is a Python package for
performance analysis of alpha factors, which can be used to create cross-sectional equity
algorithms. Building a rigorous workflow with Alphalens enhances the robustness of
backtesting strategies and reduces proneness to overfitting, both of which are important
parameters in algorithm evaluation.
For each signal, we conducted the Alphalens analysis, as follows: Eliminate all nulls; Map
symbols to security objects in Quantopian; Index the dataframe properly, such that it is
similar to pipeline output; Convert the dates to UTC; Obtain pricing data from the earliest
date up to a month after the final date; Use the signal of t-1 and trade on the open date of
t; Create the factor tear sheet using the signal and six periods (1, 5, 10, 15, 20 and 30 days).
Experiments
To evaluate the economic value of events and sentiment signals generated from an
automated information extraction platform, we conducted a set of experiments. First, we
4 https://www.quantopian.com/research
5 http://quantopian.github.io/alphalens/
7
Electronic copy available at: https://ssrn.com/abstract=2907600


<!-- Page 8 -->

Event-Based Trading
tested whether the polarity of specific events can be used as a signal for financial traders.
More specifically, we identified which positive (negative) events were followed by an
increase (decrease) in stock prices and after what interval. Second, we looked at whether
buying specific stocks when certain events appear in digital media outperforms buying a
benchmark portfolio for a given period. Third, we tested whether event-types perform
differently in different sectors. Finally, using the Quantopian Research Platform, we
analyzed event-type signals extracted from a large scaled dataset, and determined the
event-type signals with the potential to generate superior alphas for traders, and in which
holding periods.
Experiment I – Event Polarity as a Signal
In the first experiment, we evaluated whether event polarity could be used as an indicator
of future positive and negative returns, in different holding periods. For this experiment,
we used Corpus A. We tested the correlation of the polarity of these events with future
stock prices, that is, whether an event classified as positive (negative) was followed by an
increase (decline) in stock price. We looked at the changes in stock prices in time lags of
1, 2, 5, 7, 10, 15, and 20 trading days. The results of this experiment for the 10 most
frequent events in our corpus are presented in Figure 1.
The results of this experiment clearly show that events have different effects over different
time periods. For example, the effect of negative financial results seems to be strongest
after one trading day, and fades off over time, while positive financial results seem to have
a weaker effect after one trading day, but this effect becomes stronger over time
8
Electronic copy available at: https://ssrn.com/abstract=2907600


<!-- Page 9 -->

Event-Based Trading
80%
t1 t2 t5 t7 t10 t15 t20
60%
40%
20%
0%
Financial Financial Dividend Analyst New Merger - Forecast Service - Alliance Asset Sale
Results Results POS Rating POS Product Acquisition POS Product POS POS
POS NEG POS POS Deal POS
Figure 1: Predicting stock returns using event polarity, by event, by time period. The Figure shows
the correlation between the events’ polarity and the corresponding stock return after t-days.
Experiment II - Comparing performance to benchmarks
In the third experiment, we explored whether trading specific stocks when certain events
appear in digital media outperforms trading a benchmark portfolio (Russel, 2000). We
processed Corpus B with VIP , and simulated a situation in which traders buy stocks (sell
short) when positive (negative) events occur, and hold them for different time periods.
We recorded cases when trading according to events significantly outperformed trading the
benchmark portfolio. A Benjamini-Hochberg correction was applied to control for the fact
that small p-values (less than 5%) may occur by chance, and lead to the incorrect rejection
of true null hypotheses. The results are presented in Figure 3 below.
Negative Inside Sell Purchase (n=831,783) + + + + +
Positive Alliance (n=821,837) + +
Positive New Product (n=814,790) +
Positive Financial Results (n=786,557) + + + + + + + + + + + + + + + + + + + + + + + +
Positive Merger and Acquisition (n=661,737) + + + +
Positive Contract - Agreement - Deal (n=579,513) + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +
Negative Financial Results (n=391,096) + +
Positive Forecast (n=390,819) + + + + + + + + + + + + + + + + + + + + + + + + + + + + +
Positive Dividend (n=374,032) + + + +
Positive Facilities (n=371,548) + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +
Positive Investment (n=335,243) + + + + + + + + + + + + + + + + + + + + + + + + + +
Positive Executive Change (n=334,256) + + + + + + + + + + +
Positive Asset Sale (n=292,888) + + + +
Positive Analyst Rating (n=264,927) + + + + + + + + + + + + + + +
Positive Inside Sell Purchase (n=255,826) + + + + + + + + + + + + + + + + + + + + + + + + + +
Positive Asset Acquisition (n=248,505) + + + + + + + + + + + + + + + + + + + + + + + + + + +
Positive Service Product Deal (n=226,043)
Negative Forecast (n=226,016) + +
Negative Lawsuit (n=209,818) + + +
Positive Stock Buyback (n=172,491) + + + + + + +
t0t1t2t3t4t5t6t7t8t9t10t11t12t13t14t15t16t17t18t19t20t21t22t23t24t25t26t27t28t29t30
Figure 3: Trading based on events compared to trading a benchmark portfolio. (+) signifies that
trading on event signals significantly outperforms trading the benchmark portfolio.
9
Electronic copy available at: https://ssrn.com/abstract=2907600


<!-- Page 10 -->

Event-Based Trading
Experiment III – Events in different sectors
In the fourth experiment, we tested whether similar event-types in different sectors have
different economic value for traders. In particular we looked at the following nine sectors:
SnPEnergy, SnPTech, Industrials, Staples, SnPMaterials, SnPFinance, TelecomUtilities,
HealthCare, Discretionary, and a set comprising all nine sectors.
For this experiment, we processed Corpus B with VIP and classified the companies by
sector. We looked at three short-term trading periods of 1, 2, and 5 days. The effects of
similar event-type in different sectors showed no significant differences for a trading period
of one day. The adjacency matrices in Figure 4 below outline the significant differences
between similar events in different sectors for trading horizons of 2 and 5 days
T=2 T-5
SnPEnergy SnPEnergy
SnPTech 0.04 SnPTech 0.035
Industrials Industrials
Staples 0.04 0.05 0.04 Staples
SnPMaterials SnPMaterials 0.035 0.019 0.0270.018
SnPFinance 0.05 SnPFinance 0.019
TelecomUtilities TelecomUtilities
HealthCare 0.04 HealthCare 0.027
Discretionary Discretionary 0.018
All sectors All sectors
Sn
P E n e rgy
Sn
P T e ch
In
d u stria ls
Sta
p le s
Sn
P M a te ria
ls
Sn
P Fin a n ce
T
e le co m U tilitie
s
H
e a lth C a re
D
iscre tio n a
ry
A
ll Se cto rs
Sn
P E n e rgy
Sn
P T e ch
In
d u stria ls
Sta
p le s
Sn
P M a te ria
ls
Sn
P Fin a n ce
T
e le co m U tilitie
s
H
e a lth C a re
D
iscre tio n a
ry
A
ll Se cto rs
Figure 4: Adjacency matrix of significant differences between similar events in different sectors
(p-values)
The results indicate that for a time period of two days, similar event-type signals generated
significantly different effects in Staples, SnPFinance, and SnPTech sectors. For a five-day
period, event-type signals generated significantly different effects in SnPMaterials and
Health Care as well as SnPFinance and SnPTech. In general, each event-type signal
10
Electronic copy available at: https://ssrn.com/abstract=2907600


<!-- Page 11 -->

Event-Based Trading
generated a similar effect in most sectors, which led us to believe that sectors should not
play an important role when considering event signals in short-term trading periods.
Experiment IV – Developing superior trading strategies using event detection
For our final experiment, we used Alphalens and the Quantopian Research Platform to
conduct a return analysis when investing according to combinations of sentiment signals
and various event-type signals, which were identified with VIP after processing Corpus C.
We first compared between the alphas and betas resulting from investing according to a
Daily Sentiment Score signal, Time-weighted Sentiment Score; signal and the Aggregated
Event Score signal in various investment periods ranging from one day to one month. The
results are presented in Table 1 below. Cells highlighted in green represent excess return
compared to the returns based on the Daily Sentiment Score signal, and cells highlighted
in red represent inferior return compared to returns based on the Daily Sentiment Score
signal.
Table 1: Alphas and betas resulting from investing according to the Daily Sentiment Score signal, Time-
weighted Sentiment Score signal and Aggregated Event Score signal. Annual Alpha (Beta).
1 5 10 15 20 30
Daily Sentiment
0.296 (-0.024) 0.065 (-0.02) 0.03 (-0.018) 0.019 (-0.015) 0.015 (-0.017) 0.011 (-0.025)
Score signal
Time-weighted
0.228 (-0.061) 0.056 (-0.049) 0.032 (-0.056) 0.027 (-0.06) 0.027 (-0.065) 0.023 (-0.077)
Sentiment Score
Aggregated
0.272 (0.001) 0.054 (-0.011) 0.019 (-0.005) 0.01 (-0.008) 0.01 (-0.013) 0.01 (-0.003)
Event Score
Of these three signals, the best alpha for the short run (shorter than 10 days) was obtained
using the Daily Sentiment Score signal, which outperformed the alphas of Time-weighted
Sentiment Score and Aggregated Event Score signals. In the longer run (10 days and
11
Electronic copy available at: https://ssrn.com/abstract=2907600


<!-- Page 12 -->

Event-Based Trading
longer), however, the alpha of the Time-weighted Sentiment Score signal outperformed the
alpha of Daily Sentiment Score and Aggregated Event Score signals.
Since specific event-types do not occur on most days, it is not practical to use these event-
types as daily trading signals, unlike the Daily Sentiment Score, Time-weighted Sentiment
score and Aggregated Event Score signals. Therefore, to test their economic trading value
of the last three signals, we test whether returns increase or decline when specific event-
type signals are added to the Daily Sentiment Score signal. The results for the Daily
Sentiment Score signal together with the 25 most frequent event-types in our corpus are
presented in Table 2 below. When the signal from a particular event-types improves the
economic value of the Daily Sentiment Score signal, the corresponding cell is highlighted
in green. When the signal reduces the economic value of the Daily Sentiment Score signal,
the corresponding cell is highlighted in red.
Several insights emerge from this experiment: First, signals based on events mentioned in
digital media can be used as valuable signals for investors when devising their trading
strategy, and all event-based signals used in this study seem to generate a positive alpha.
Second, we classify several event-types by their impact on trading returns: events that have
a positive long-term effect (e.g., “Lawsuits” and “New Products”), events with a positive
short-term effect (e.g., “Dividends”), events that seem to have a positive effect for all time
periods examined in this study (e.g., “Financial Results” and “Forecast”), and events that
do not seem to have a positive effect in any time period (e.g., “Asset Sales” and “Workforce
Changes”). Third, of the event-type signals that we tested, the “Financial Results” signal
seems to have the most positive impact on the Daily Sentiment Score signal. Fourth, the
12
Electronic copy available at: https://ssrn.com/abstract=2907600


<!-- Page 13 -->

Event-Based Trading
low betas of all tested event-types indicate that event-based trading generates less volatile
results compared to a benchmark.
Table 2: Alphas and betas resulting from an investment strategy based on Daily Sentiment Score and the
25 most frequent event-types in our corpus – annual Alphas (Betas).
1 5 10 15 20 30
Financial Results 0.447 (-0.018) 0.093 (-0.02) 0.044 (-0.018) 0.029 (-0.02) 0.029 (-0.026) 0.029 (-0.034)
Inside Sell
0.272 (-0.014) 0.062 (-0.012) 0.026 (-0.008) 0.016 (-0.004) 0.016 (-0.006) 0.016 (-0.012)
Purchase
New Product 0.283 (-0.027) 0.063 (-0.022) 0.028 (-0.02) 0.019 (-0.018) 0.019 (-0.021) 0.019 (-0.028)
Alliance 0.279 (-0.025) 0.063 (-0.023) 0.029 (-0.021) 0.018 (-0.019) 0.018 (-0.021) 0.018 (-0.026)
Merger and
0.287 (-0.025) 0.062 (-0.024) 0.028 (-0.021) 0.017 (-0.018) 0.017 (-0.019) 0.017 (-0.024)
Acquisition
Forecast 0.435 (-0.016) 0.083 (-0.013) 0.036 (-0.001) 0.022 (-0.007) 0.022 (-0.003) 0.022 (-0.012)
Contract -
0.285 (-0.022) 0.06 (-0.022) 0.027 (-0.021) 0.018 (-0.017) 0.018 (-0.018) 0.018 (-0.024)
Agreement - Deal
Analyst Rating 0.314 (-0.025) 0.069 (-0.021) 0.031 (-0.017) 0.021 (-0.016) 0.021 (-0.02) 0.021 (-0.028)
Executive Change 0.286 (-0.021) 0.061 (-0.017) 0.027 (-0.014) 0.016 (-0.012) 0.016 (-0.014) 0.016 (-0.021)
Dividend 0.3 (-0.025) 0.07 (-0.022) 0.028 (-0.022) 0.017 (-0.02) 0.013 (-0.022) 0.009 (-0.028)
Facilities 0.282 (-0.027) 0.062 (-0.022) 0.027 (-0.02) 0.018 (-0.018) 0.018 (-0.02) 0.018 (-0.028)
Investment 0.279 (0.023) 0.062 (-0.02) 0.027 (-0.017) 0.017 (-0.016) 0.017 (-0.018) 0.017 (-0.025)
Asset Sale 0.278 (-0.02) 0.06 (-0.018) 0.026 (-0.016) 0.016 (-0.011) 0.012 (-0.011) 0.008 (-0.02)
Asset
0.29 (-0.022) 0.064 (-0.017) 0.028 (-0.015) 0.018 (-0.013) 0.014 (-0.015) 0.01 (-0.023)
Acquisition
Lawsuit 0.29 (-0.024) 0.064 (-0.018) 0.029 (-0.018) 0.019 (-0.016) 0.019 (-0.017) 0.019 (-0.024)
Service Product
0.295 (-0.022) 0.064 (-0.019) 0.028 (-0.017) 0.019 (-0.014) 0.015 (-0.015) 0.01 (-0.023)
Deal
Stock Buyback 0.301 (-0.024) 0.068 (-0.02) 0.032 (-0.019) 0.021 (-0.016) 0.016 (-0.019) 0.012 (-0.026)
Senior Executive
0.291 (-0.023) 0.064 (-0.018) 0.029 (-0.017) 0.019 (-0.016) 0.015 (-0.019) 0.011 (-0.02)
Change
Credit Debt
0.295 (-0.025) 0.066 (-0.021) 0.03 (-0.019) 0.019 (-0.016) 0.016 (-0.019) 0.011 (-0.026)
Rating
Board Change 0.293 (-0.024) 0.065 (-0.021) 0.03 (-0.02) 0.019 (-0.015) 0.015 (-0.017) 0.011 (-0.024)
Price Target 0.312 (-0.025) 0.068 (-0.021) 0.031 (-0.019) 0.02 (-0.015) 0.017 (-0.018) 0.012 (-0.026)
Workforce
0.296 (-0.022) 0.065 (-0.018) 0.03 (-0.017) 0.019 (-0.014) 0.015 (-0.016) 0.011 (-0.024)
Change
Settlement 0.293 (-0.021) 0.063 (-0.017) 0.028 (-0.017) 0.018 (-0.013) 0.014 (-0.015) 0.01 (-0.021)
Investigation 0.297 (-0.023) 0.065 (-0.018) 0.029 (-0.016) 0.019 (-0.016) 0.015 (-0.019) 0.011 (-0.026)
Product Update 0.295 (-0.025) 0.065 (-0.021) 0.03 (-0.019) 0.019 (-0.016) 0.016 (-0.017) 0.011 (-0.026)
Conclusions and future work
In this paper, we evaluated whether financial events extracted from digital media using an
automatic information extraction platform constitute valuable trading signals. The effect
of different events was determined in different time lags. We established that the polarity
13
Electronic copy available at: https://ssrn.com/abstract=2907600


<!-- Page 14 -->

Event-Based Trading
of different event-types is, indeed, reflected in the stock price in different time periods. We
concluded that in contrast to aggregate polarity of events, which is typically used in
sentiment analysis systems, event-types should be used as isolated signals. We further used
a benchmark portfolio and demonstrated that superior returns can be obtained when
considering different event-types for different investment periods. We found that both
sentiments and events can generate positive alphas. We further found that Daily Sentiment
Score signal is a better signal than Time-weighted Sentiment Score signal in the short term,
and vice versa in the long term. When used aggregately, event-related information
produces a noisy signal that does not improve the alpha of the Daily Sentiment Score signal.
Using specific event-types, however, enhances the results of a strategy based on the Daily
Sentiment Score signal. Event-types impact the Daily Sentiment Score signal differently
in different time periods. We also compared the effect of the same event-type in different
sectors, and in most cases, we found that the same event-types had a similar impact in all
sectors. Consequently, we believe that sector-based information is less important in
developing event-based trading strategies.
Future work might further investigate how to use automated information extraction
platforms to further enhance event-based trading strategies, or might focus on determining
the effect of events in non-financial domains.
References
Antweiler, W., and Frank, M. Z. 2004. “Is All That Talk Just Noise ? The Information
Content of Internet Stock Message Boards,” (LIX:3).
Bollen, J., Mao, H., and Zeng, X. 2011. “Twitter mood predicts the stock market,” Journal
of Computational Science (2:1), Elsevier B.V., pp. 1–8 (doi:
14
Electronic copy available at: https://ssrn.com/abstract=2907600


<!-- Page 15 -->

Event-Based Trading
10.1016/j.jocs.2010.12.007).
Boudoukh, J., Feldman, R., Kogan, S., and Richardson, M. P. 2016. “Information, Trading,
and Volatility: Evidence from Firm-Specific New,” (available at
https://ssrn.com/abstract=2193667 or http://dx.doi.org/10.2139/ssrn.2193667).
Davis, A. K., Piger, J. M., and Sedor, L. M. 2012. “Beyond the Numbers: Measuring the
Information Content of Earnings Press Release Language,” Contemporary
Accounting Research (29:3), Blackwell Publishing Ltd, pp. 845–868 (doi:
10.1111/j.1911-3846.2011.01130.x).
Ding, X., Zhang, Y., Liu, T., and Duan, J. 2014. “Using Structured Events to Predict Stock
Price Movement: An Empirical Investigation.,” in EMNLP, pp. 1415–1425.
Fama, E. F. 1970. “Efficient Markets: A Review of Theory and Emirical Work,” The
Journal of Finance (25:2), pp. 383–417.
Feldman, R. 2013. “Techniques and Applications for Sentiment Analysis,”
Communications of the ACM (56:4), p. 82 (doi: 10.1145/2436256.2436274).
Feldman, R., Rosenfeld, B., Bar-haim, R., and Fresko, M. 2010. “The Stock Sonar —
Sentiment Analysis of Stocks Based on a Hybrid Approach,” in Proceedings of the
Twenty-Third Innovative Applications of Artificial Intelligence Conference, pp. 1642–
1647.
Finnerty, J. E. 1976. “Insiders and Market Efficiency,” The Journal of Finance (31:4), pp.
1141–1148 (doi: 10.1111/j.1540-6261.1976.tb01965.x).
Grossman, S. J., and Stiglitz, J. E. 1980. “On the impossibility of informationally efficient
markets,” The American economic review (70:3), JSTOR, pp. 393–408.
Heston, S. L., and Sinha, N. R. 2015. “News versus Sentiment: Predicting Stock Returns
from News Stories.,”
Li, X., Xie, H., Chen, L., Wang, J., and Deng, X. 2014. “News Impact on Stock Price
Return via Sentiment Analysis,” Knowledge-Based Systems (69), pp. 14–23 (doi:
10.1016/j.knosys.2014.04.022).
Loughran, T., and Mcdonald, B. 2016. “Textual Analysis in Accounting and Finance: A
Survey,” Journal of Accounting Research (54:4), pp. 1187–1230 (doi: 10.1111/1475-
679X.12123).
Schumaker, R. P., and Chen, H. 2009. “Textual analysis of stock market prediction using
breaking financial news,” ACM Transactions on Information Systems (27:2), pp. 1–
19 (doi: 10.1145/1462198.1462204).
Terrence, H., and Riordan, R. 2013. “Algorithmic Trading and the Market for Liquidity,”
Journal of Financial and Quantitative Analysis (48:4), Cambridge University Press,
pp. 1001–1024 (doi: 10.1017/S0022109013000471).
Tetlock, P. C., Saar-Tsechansky, M., and Macskassy, S. 2008. “More Than Words:
Quantifying Language to Measure Firms’ Fundamentals.,” The Journal of Finance
(63), pp. 1437–1467.
Treleaven, P., Galas, M., and Lalchand, V. 2013. “Algorithmic Trading Review,”
Communications of the ACM (56:11), pp. 76–85 (doi: 10.1145/2500117).
Zhang, W., and Skiena, S. 2010. “Trading Strategies to Exploit Blog and News Sentiment,”
in Proceedings of the Fourth International AAAI Conference on Weblogs and Social
Media (Vol. d), pp. 375–378.
15
Electronic copy available at: https://ssrn.com/abstract=2907600
