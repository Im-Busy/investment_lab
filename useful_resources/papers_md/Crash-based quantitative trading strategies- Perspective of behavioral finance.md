# Crash-based quantitative trading strategies- Perspective of behavioral finance

> *Source PDF: Crash-based quantitative trading strategies- Perspective of behavioral finance.pdf*
> *Extraction: Combined — previous structured extraction (base) + markitdown raw text*

---

# Crash-based quantitative trading strategies- Perspective of behavioral finance

> *Source PDF: Crash-based quantitative trading strategies- Perspective of behavioral finance.pdf*
> *Extraction: Combined — markitdown raw text (base) + previous extraction supplements*

---

Contents lists available at ScienceDirect

Finance Research Letters

journal homepage: www.elsevier.com/locate/frl

Crash-based quantitative trading strategies: Perspective of
behavioral finance

Yan Fang a, Jie Yuan a, J. Jimmy Yang b,*, Shangjun Ying a, 1
a Shanghai University of International Business and Economics
b Oregon State University, School of Accounting, Finance, and Information Systems, College of Business, 426 Austin Hall, Corvallis, OR,
97331United States

A R T I C L E  I N F O

A B S T R A C T

Inspired by the studies on stock market crashes, we use documented indicators from behavioral
finance to construct two quantitative trading strategies, i.e., Crash + Timing Strategy and Crash
+ Momentum-Reversal Strategy. Empirical analyses show that both strategies are effective and
robust. Behavioral factors can be  beneficial to investors  when they are  incorporated into  their
trading strategies.

JEL codes:
G1
C22

Keywords:
Behavioral finance
Crash factor
Market timing
Momentum-reversal strategy

1. Introduction

With the development of the behavioral finance, many scholars seek to explain the causes of stock market crashes using behavioral
theories. For instance, Avery and Zemsky (1998) point out that herding can lead to mispricing of assets and price bubbles. Kim et al.
(2016) suggest that firms with overconfident CEOs have higher stock price crash risk than other firms. Jang and Kang (2019) find that
price  crashes  could  be  caused  by  the  overpricing  driven  by  institutional  investors.  While  existing  studies  focus  on  exploring  the
influential factors that may lead to stock market crashes or predicting crash events in stock markets, we investigate how investors can
profit from those events.

Based on the aforementioned studies, a price crash can be caused by investors’ cognitive biases, such as herding bias and over-
confidence bias. Even though such cognitive biases can result in mispricing of securities, investors may theoretically expect excess
returns generated from a stock market crash as long as they rationally use certain behavioral deviations (Conrad et al., 2014; Jang and
Kang, 2019; Pelster, 2020). Therefore, it is highly possible to realize excess returns from a stock market crash.

During a stock market crash, stock prices typically move in a similar pattern. First, stock prices continue to rise for an extended
period until the bubble reaches its peak. Then, the bubble bursts and the market collapses with a sudden slump. If one wants to profit
from a stock market crash, the ability to select stocks and set the timing is critical. Accordingly, this paper adopts the behavioral
finance approach to propose two quantitative trading strategies based on a documented crash factor. One strategy is built on the crash
factor  and  market  timing  (i.e.,  Crash  + Timing  Strategy,  CTS),  while  the  other  is  constructed  by  applying  the  crash  factor  and
momentum-reversal strategy (i.e., Crash + Momentum-Reversal Strategy, CMRS). Our empirical analyses present stable returns and

* Corresponding author.

E-mail address: jimmy.yang@bus.oregonstate.edu (J.J. Yang).

1  Ying acknowledges financial support from the National Natural Science Foundation of China (71571116).

https://doi.org/10.1016/j.frl.2021.102185
Received 10 June 2020; Received in revised form 11 April 2021; Accepted 26 May 2021

FinanceResearchLetters45(2022)102185Availableonline30May20211544-6123/©2021ElsevierInc.Allrightsreserved.Y. Fang et al.

confirm the effectiveness and robustness of these market crash-based trading strategies.

2. Construction of trading strategies

2.1. Crash + timing strategy

According to the concept of mean reversion and the gambler’s fallacy (Tversky and Kahneman, 1971), stock prices can rebound
from the current crash and possibly experience another crash. Investors can generate profits by buying the dips during the crash and
selling the rips during the consequent rebound. We use the crash factor2 of Jang and Kang (2019) to screen stocks. Specifically, top 10%
stocks at the end of the previous month (t - 1) and the current month (t) are sorted out by the crash factor, and the top 10% stocks at the
end of the current month are regarded as stocks whose bubbles just burst. However, stocks selected solely based on crash factor could
again take a nosedive, so they cannot be directly put into quantitative investing.3  We consider appropriate timing for buying and
selling (i.e. market timing) by constructing the crash + timing strategy.

Due to the self-serving memory bias, investors generally repeat their previous behaviors for a risky stock position (G¨odker et al.,
2020). Therefore, we assume that before a stock market crashes, stocks will display characteristics similar to those perceived during the
last crash. The strength/weakness of buying and selling on long and short positions before and after a crash would be the best indicator
for market timing measurement. Consequently, we use the relative strength index (RSI) (Wilder, 1978) to identify the best times to buy
stocks, and apply a momentum factor, which is defined as the corresponding 5-day momentum at the peak from last month, to capture
the sell signals. During an observation period, we sell stocks when the momentum factor is similar to the peak momentum factor in the
previous month (t-1). The specific process of CTS is given in Algorithm 1.

2.2. Crash + momentum-reversal strategy

Momentum (Jegadeesh and Titman, 1993) and reversal are two well-known anomalies within the field of behavioral finance.
However, momentum strategies suffer from occasional large crashes (i.e. momentum crash) (Barroso and Santa-Clara, 2015; Daniel
and Moskowitz, 2016). During a momentum crash, momentum often tends to reverse. That is, the portfolio with the lowest momentum
becomes the best performer, and the momentum-reversal strategy realizes excess returns. Therefore, the momentum-reversal strategy
can considerably enhance investment efficiency.

The  second  strategy  is  to  combine  the  crash  factor  with  momentum-reversal  strategy.  We  first  add  the  crash  factor  to  the
momentum-reversal strategy, adopt a scoring method to construct a new comprehensive scoring factor, and use this new factor to
screen stocks. The stocks selected are featured with crashes in current month and expected to rebound later. That is, underperformed
stocks will likely outperform in the future (Bali et al., 2011). The process of CMRS is shown in Algorithm 2.

3. Empirical analyses

We perform empirical analyses using daily stock data from January 1, 2018 to December 31, 2018 and the Center for Research in
Security Prices (CRSP) value-weighted index as the market index. The sample includes all stocks with share codes 10 and 11 from
NYSE, AMEX and NASDAQ. All data are from the CRSP database. The data processing and calculation of crash factor are conducted
based on Jang and Kang (2019) (See Appendix 1).

3.1. Performance of CTS

To determine when to buy stocks, we choose the 14-day RSI4  within the oversold level as the buy limit price. An RSI below 30
(Wilder, 1978) would suggest a buy. Since a rebound may happen or continue beyond the current month, we use the next two months
as the observation period. In addition, we consider the following four methods to determine the sell threshold.

Method 1: Use RSI as the benchmark and the overbought level (RSI=70) as a threshold;
Method 2: Adopt the pre-set take profit level (i.e., 3%)5 as a threshold;
Method 3: Utilize the highest cumulative return over the last month captured by the momentum factor as a threshold;
Method 4: Consider the highest cumulative return over the last month captured by the momentum-information discreteness (Da
et al., 2014) as a threshold.

When adopting different methods to identify the best times to sell stocks, the stop-loss order is executed at 2.5% below the current
stock price (Han et al., 2016). Evaluating the effectiveness of a strategy is essential for an investment. We evaluate our strategies based

2  The crash factor is defined as the probability in the event of log returns lower than (cid:0) 70% over the next 12 months.
3  We do not discuss profit taking from short selling herein due to multiple short-selling bans in the world stock markets.
4  In order to choose the appropriate timeframe, a comparison is conducted among the respective performance of 14-day RSI, 24-day RSI and

market index, seeking the timeframe that outperforms the market index.

5  This is how to avoid emotional investing and selling into corrections within the field of behavioral finance.

FinanceResearchLetters45(2022)1021852Y. Fang et al.

Table 1
Performance of CTS with different methods in 2018.

Annualized return
Win rate

Method 1

(cid:0) 7.145%
30.113%

Method 2

3.296%
56.000%

Method 3

8.464%
37.113%

Method 4

5.515%
45.455%

Market Index

(cid:0) 10.017%
—

Note: The sell signal in Method 1 is 14-day RSI above 70. The sell signal in Method 2 is when the cumulative return is 3% above the purchase price.
The sell signals in Method 3 and Method 4 are the 5-day momentum value and the 5-day information discreteness that are equal to or above the
highest cumulative return over the last month, respectively.

Fig. 1. Performance of CMRS from quintile 1 to 5 for 2018, and the benchmark refers to the cumulative return of the market index.

Table 2
This table reports the annualized return, Sharp ratio, maximum drawdown and win rate on quintile portfolios sorted on the comprehensive scoring
factors. At the end of each month i, quintile portfolios are constructed by sorting stocks based on the comprehensive scoring factors. Column quintile 1
is for the highest comprehensive scoring factors quintile, while column quintile 5 shows the results for the lowest comprehensive factors quintile.
Column CRSP is for the performance of CRSP value-weighted index.

Panel 1: CMRS

Annualized return
Sharp ratio
Max. drawdown
Win rate

Quintile 1

(cid:0) 71.570%
(cid:0) 0.361775
(cid:0) 74.372%
13.878%

Panel 2: Momentum-reversal strategy

Annualized return
Sharp ratio
Max. drawdown
Win rate

Quintile 1

(cid:0) 80.798%
(cid:0) 0.501626
(cid:0) 82.073%
4.147%

Quintile 2

(cid:0) 36.926%
(cid:0) 0.151652
(cid:0) 45.374%
17.809%

Quintile 2

(cid:0) 41.240%
(cid:0) 0.183588
(cid:0) 48.318%
9.704%

Quintile 3

17.761%
0.064567
(cid:0) 21.431%
26.712%

Quintile 3

(cid:0) 7.102%
(cid:0) 0.022715
(cid:0) 24.038%
27.437%

Quintile 4

39.228%
0.138178
(cid:0) 14.036%
31.133%

Quintile 4

46.513%
0.156668
(cid:0) 13.444%
38.024%

Quintile 5

104.276%
0.328911
(cid:0) 9.426%
40.288%

Quintile 5

287.544%
0.517607
(cid:0) 7.187%
45.663%

CRSP

(cid:0) 6.012%
(cid:0) 0.018106
(cid:0) 20.960%
—

CRSP

(cid:0) 6.012%
(cid:0) 0.018106
(cid:0) 20.960%
—

on the annualized return and win rate, where the win rate refers to the ratio of the number of stocks with a positive return to the
number of stocks with non-zero returns. The results are shown in Table 1, where the assessments for Methods 1–4 are given in columns
2–5 and the results of market index is given in column 6.

In Table 1, we can see that 1) the market return in 2018 is negative and all four Methods outperform the market index; 2) the
annualized  return  in  Method  1  is  negative,  and  its  win  rate  is  the  lowest,  which  implies  that  the  market  timing  strategy  solely
depending on RSI is insufficient6; 3) Method 2 has the highest win rate, which indicates that the concept of take-profit can effectively
avoid making wrong investment decisions, and achieves good performance in quantitative investing; 4) the annualized returns in both
Methods 3 and 4 are positive and are substantially higher than the market return, which suggests that the momentum factor is a

6  In  addition  to  the  sell  threshold  of  RSI=70  suggested  by  Wilder  (1978),  we  also  use  RSI=80  as  the  sell  threshold  and  RSI=20  as  the  buy
threshold.  The  annualized  return  and  win  rate  of  Method  1  are  both  improved.  Specifically,  the  annualized  return  improves  from  (cid:0) 7.145%  to
7.465%, while the win rate increases from 30.113% to 34.408%.

FinanceResearchLetters45(2022)1021853Y. Fang et al.

Algorithm 1
The process for the CTS.

1  Start with time t,

a)  Pick up top 10% stocks based on crash factors at the end of month t and month t (cid:0) 1 and name them as Set 1 and Set 2, respectively;
b)  Remove stocks which appear on both month t (cid:0) 1 and month t from set 1, and use the remaining stocks to create a new portfolio.
c)  For stocks listed in our new portfolio,

i  we will buy when they reach the buy limit order during our observation period;
ii  we will sell when they reach the sell limit order or when the stop-loss price has been reached during our observation period; otherwise, we

will sell them at the end of the observation period.

2  2. Repeat Step 1.

Algorithm 2
The process of CMRS.

1  Start with time t,

a)  Buy the top 20% of stocks sorted by the new comprehensive factor at the end of month t;
b)  Hold stocks till the end of month t + 1 and sell them.

2  Repeat Step 1.

workable market timing indicator.

3.2. Performance of CMRS

In order to evaluate the performance of CMRS, all momentum factors and crash factors at the end of the month (t) are sorted in
ascending and descending order, respectively, and the comprehensive scoring factors are calculated according to the scoring method.7
The comprehensive scoring factors are sorted in descending order, and divided into five groups (Fama and French, 1993). We buy the
stocks in these five groups in month t and sell them at the end of the month (t + 1). Finally, we calculate the cumulative return in each
group. The results are shown in Fig. 1, where each curve represents the cumulative log return of a quintile regression group.

Despite of some fluctuations in the market of 2018, the cumulative return of each group shows a clear monotonic pattern. Among
them, the cumulative returns of quintiles 3 to 5 are all higher than the cumulative return of the market index, which indicates that the
trading strategy brings investors an excess return. We also compare and analyze the performance of both CMRS and the traditional
momentum-reversal strategy by using four indicators, i.e., annualized return, Sharpe ratio, maximum drawdown and win rate. The
results are presented in Table 2.

Both  strategies  show  strong  portfolio  performance.  All  indicators  in  quintiles  1–5  show  clear  monotonic  patterns.  CMRS  out-
performs the momentum-reversal strategy in quintiles 1–3. CMRS also outperforms the market index in quintiles 3–5. Although the
best performer Quintile 5 of CMRS falls behind the control group of the momentum-reversal strategy, this trading strategy is overall an
effective stock selection strategy.

Generally speaking, momentum crash is the most important factor that restricts application of the momentum strategy, and it can
sometimes cause the momentum factor to fail. Although the market fluctuated in 2018, there was no momentum crashes. That is the
main reason that we cannot find an obvious advantage of our proposed strategy in Table 2. To show that CMRS is effective during a
momentum crash, we examine the data from January 1, 2015 to December 31, 2015 (momentum crashes happened over this period) to
test the robustness of our trading strategies. The results show that both CTS and CMRS achieve favorable results, but the momentum-
reversal strategy generally fails. Please see Appendix 2 for details.

In addition, we examine an up market period from January 1, 2019 to July 31, 2019 to test whether our trading strategies would
work when the overall market return is positive. We find that the CTS strategy performs even better in the up market as annualized
returns and win rates in 2019 are higher than those in 2018 and 2015. Results from the CMRS strategy are similar to those in 2018.
Those results are not tabulated, but they are available on request.

4. Conclusion

Inspired by the latest studies on stock market crashes, this paper constructs two quantitative trading strategies and tests their
performances. Empirical analyses show that 1) the crash factor cannot directly function as a momentum factor, but when the crash
factor combines with an appropriate timing indicator, it could generate excess returns for investors; 2) the momentum factor is useful
in a market timing approach; 3) during a volatile market, a momentum-reversal strategy may fail while the CMRS proposed in this
paper is still effective.

Unlike previous studies, this article explores quantitative trading strategies from the perspective of behavioral finance, considering

7  The so-called scoring method is a method where stocks’  crash factors are ranked in ascending order, their momentum factors are ranked in

descending order, and the average of the two ranks is defined as the comprehensive scoring factor.

FinanceResearchLetters45(2022)1021854Y. Fang et al.

Table A.1
Performance of CTS with different methods in 2015.

Annualized return
Win rate

Method 1

(cid:0) 4.749%
28.049%

Method 2

9.497%
57.927%

Method 3

6.332%
46.154%

Method 4

0.712%
43.357%

Market Index

(cid:0) 6.989%
—

the effect from stock market crash, momentum factor, and RSI. This approach provides investors useful trading strategies to potentially
exploit behavioral bias in stock markets. However, there is room for future studies to further explore and extend these strategies. First,
more studies are needed in the application of those strategies. For instance, potential profits from long-short portfolios in the context of
the second strategy can be explored. Second, the construction of the two strategies only takes into consideration the behavior of short-
term investors. A long-term approach can provide a complete picture and strengthen our understanding of market crash-based trading
strategies. Third, we use the market index as the benchmark to measure performance of our trading strategies, but risk-adjusted returns
can be used to appropriately account for risks involved in the trading strategies.

Declarations of Competing Interest

None

Appendix 1. Crash factor

Crash factor is defined as:

Crash factor =

exp(C)
1 + exp(C) + exp(J)

where

C = α0 + α1RM12 + α2EXRET12 + α3TVOL + α4TSKEW + α5SIZE + α6DTURN + α7AGE + α8TANG + α9SALESG

and

J = β0 + β1RM12 + β2EXRET12 + β3TVOL + β4TSKEW + β5SIZE + β6DTURN + β7AGE + β8TANG + β9SALESG

Variable RM12 is the log return of the CRSP value-weighted index over the past 12 months; EXRET12 is the log return over the past
12 months in excess of RM12; TVOL and TSKEW are the standard deviation and skewness, respectively, of daily log returns over the
past 6 months; SIZE is the log of market capitalization; DTURN is the detrended turnover, defined by the six-month average of monthly
share turnover minus the prior 18-month average; AGE is the number of years since the firm’s first appearance on the CRSP monthly
stock file; TANG is tangible assets divided by total asset; and SALESG is the sales growth over the prior year. Parameters αi and βi (i = 0,
1, ⋅⋅⋅, 9) are from Jang and Kang (2019).

Appendix 2. Robustness testing for quantitative trading strategies

What  does  the  advantage  of  the  proposed  trading  strategies  lie  in,  chance  or  strength?  We  further  test  their  reliability  and
robustness. Data from CRSP database over the period from January 1, 2015 to December 31, 2015 are used for the robustness testing.

1. Robustness Test for CTS

To identify the best times for selling stocks, the 24-day RSI within the overbought level is used as the buy limit price, and the stop-
loss order is set to be 2.5% below the current stock price. The strategy’s performances for four methods defined in Section 3 are given in
Table A.1.

As we can see, 1) over the period, the market return is negative and all the Methods 1–4 outperform the market index; 2) the
annualized return in Method 1 is negative and its win rate is the lowest, indicating that the market timing strategy solely depending on
RSI is insufficient; 3) Method 2 has the highest annualized return and win rate, implying the take-profit strategy can effectively avoid

FinanceResearchLetters45(2022)1021855Y. Fang et al.

Fig. A.1. Performance of CMRS from quintile 1 to 5 for 2015 CMRS. Note: Benchmark refers to the cumulative return of the market index. The
vertical line represents the date of July 31, 2015.

Table A.2
Performance of CMRS and Momentum-reversal Strategy in 2015.

Panel 1: CMRS

Annualized return
Sharp ratio
Max. drawdown
Win rate

Quintile 1

(cid:0) 23.039%
(cid:0) 0.073
(cid:0) 36.575%
45.117%

Panel 2: Momentum-reversal strategy

Annualized return
Sharp ratio
Max. drawdown
Win rate

Quintile 1

(cid:0) 20.144%
(cid:0) 0.064
(cid:0) 34.123%
44.826%

Quintile 2

(cid:0) 20.981%
(cid:0) 0.082
(cid:0) 31.096%
22.472%

Quintile 2

(cid:0) 17.747%
(cid:0) 0.070
(cid:0) 27.822%
22.170%

Quintile 3

(cid:0) 19.836%
(cid:0) 0.078
(cid:0) 29.028%
22.532%

Quintile 3

(cid:0) 15.981%
(cid:0) 0.065
(cid:0) 24.707%
23.241%

Quintile 4

(cid:0) 12.608%
(cid:0) 0.052
(cid:0) 21.128%
23.088%

Quintile 4

(cid:0) 15.022%
(cid:0) 0.064
(cid:0) 22.425%
22.951%

Quintile 5

(cid:0) 13.447%
(cid:0) 0.062
(cid:0) 19.734%
22.298%

Quintile 5

(cid:0) 21.312%
(cid:0) 0.089
(cid:0) 27.636%
22.182%

RM

(cid:0) 7.620%
(cid:0) 0.027
(cid:0) 17.486%
—

RM

(cid:0) 7.620%
(cid:0) 0.027
(cid:0) 17.486%
—

making wrong investment decisions; 4) the results in both Methods 3 and 4 considerably outperform the market index, which indicates
that the momentum factor is a workable market timing indicator. In sum, this trading strategy seems robust.

2. Robustness Test for CMRS

Data in 2015 are divided into groups, since there exists a crash. The cumulative return of each group is shown in Fig. A.1.
In a relatively stable market (before August 2015), the cumulative return of each group has a clear monotonic pattern. The cu-
mulative returns of quintiles 1 to 3 are higher than the cumulative return of the market index.8 When the market experiences fluc-
tuation (since August 2015), the cumulative return of each group shows no clear monotonic pattern. The performances of both CMRS
and the momentum-reversal strategy are shown in Table A.2.

Although all groups underperform the market index, the annualized returns and maximum drawdowns of CMRS maintain their
monotonicity. In this case, investors can long the best performer and short the worst performer so as to achieve excess returns. All
indicators of the momentum-reversal strategy fail to maintain their monotonic patterns, indicating that this strategy is ineffective. The
CMRS contains certain robustness as it achieves favorable results during a momentum crash.

8  The empirical results observed between Jan. 2015 and Aug. 2015 are opposite to those observed in 2018 although both periods are relatively
stable. A potential reason is the existence of bubbles in 2015. The theory of rational bubbles (Blanchard and Watson, 1982) implies that investors’
irrational  behavior  and  expectations  induce  the  stock  price  to  deviate  from  its  true  value,  giving  rise  to  a  bubble.  Once  the  bubble  bursts,  it
automatically  leads  to  a  crash  in  the  market.  Scheinkman  and  Xiong  (2003)  also  point  out  that  heterogeneous  beliefs  generated  by  investors’
overconfidence bring bubbles to the market and then increase the possibility of its crash. Heterogeneous beliefs from overconfident investors would
result in prices that deviate from the true value, and then lead to a momentum reversal effect in the market. As a result, the order of cumulative
returns from quintiles 1 to 5 in Figure A.1 before the crash in August is different from that in Figure 1.

FinanceResearchLetters45(2022)1021856Y. Fang et al.

Reference

Avery, C., Zemsky, P., 1998. Multidimensional uncertainty and herd behavior in financial markets. Am. Econ. Rev. 724–748.
Bali, T.G., Cakici, N., Whitelaw, R.F., 2011. Maxing out: stocks as lotteries and the cross-section of expected returns. J. Financ. Econ. 99 (2), 427–446.
Barroso, P., Santa-Clara, P., 2015. Momentum has its moments. J. Financ. Econ. 116 (1), 111–120.
Blanchard, O.J., Watson, M.W., 1982. Bubbles, Rational Expectations and Financial Markets. NBER working paper, (w0945).
Conrad, J., Kapadia, N., Xing, Y., 2014. Death and jackpot: why do individual investors hold overpriced stocks? J. Financ. Econ. 113 (3), 455–475.
Daniel, K., Moskowitz, T.J., 2016. Momentum crashes. J. Financ. Econ. 122 (2), 221–247.
Da, Z., Gurun, U.G., Warachka, M., 2014. Frog in the pan: continuous information and momentum. Rev. Financ. Stud. 27 (7), 2171–2218.
Fama, E.F., French, K.R., 1993. Common risk factors in the returns on stocks and bonds. J. Financ. Econ. 33, 3–56.
G¨odker, K., Peiran Jiao, R., Smeets, R., 2020. In: Investor Memory. 2020 AFA Annual Meeting Working Paper. https://papers.ssrn.com/sol3/papers.cfm?abstract_

id=3348315.

Han, Y., Zhou, G., Zhu, Y., 2016. Taming Momentum crashes: A simple Stop-Loss Strategy. Available at SSRN 2407199.
Kim, J.B., Zhang, L., 2016. CEO Overconfidence and Stock Price Crash Risk. Contemp. Account. Res. 33 (4), 1720–1749.
Jang, J., Kang, J., 2019. Probability of price crashes, rational speculative bubbles, and the cross-section of stock returns. J. Financ. Econ. 132 (1), 222–247.
Jegadeesh, N., Titman, S., 1993. Returns to buying winners and selling losers: implications for stock market efficiency. J. Finance 48 (1), 65–91.
Pelster, M., 2020. The gambler’s and hot-hand fallacies: empirical evidence from trading data. Econ. Lett. 187, 108887.
Scheinkman, J.A., Xiong, W., 2003. Overconfidence and speculative bubbles. J. Political Econ. 111 (6), 1183–1220.
Tversky, A., Kahneman, D., 1971. Belief in the law of small numbers. Psychol. Bull. 76 (2), 105–110.
Wilder Jr., J.W., 1978. New Concepts in Technical Trading Systems. Hunter Publishing Company, Greensboro, N.C.

FinanceResearchLetters45(2022)1021857

---

## Content from Previous Extraction (not in markitdown output)

### Visual/Chart/Graph Descriptions

#### 1

Y. Fang et al. F i n a n c e R e s e a r c h L e t t e r s 45(2022)102185
Fig. A.1. Performance of CMRS from quintile 1 to 5 for 2015 CMRS. Note: Benchmark refers to the cumulative return of the market index. The
vertical line represents the date of July 31, 2015.
Table A.2
Performance of CMRS and Momentum-reversal Strategy in 2015.
Panel 1: CMRS
Quintile 1 Quintile 2 Quintile 3 Quintile 4 Quintile 5 RM
Annualized return (cid:0) 23.039% (cid:0) 20.981% (cid:0) 19.836% (cid:0) 12.608% (cid:0) 13.447% (cid:0) 7.620%
Sharp ratio (cid:0) 0.073 (cid:0) 0.082 (cid:0) 0.078 (cid:0) 0.052 (cid:0) 0.062 (cid:0) 0.027
Max. drawdown (cid:0) 36.575% (cid:0) 31.096% (cid:0) 29.028% (cid:0) 21.128% (cid:0) 19.734% (cid:0) 17.486%
Win rate 45.117% 22.472% 22.532% 23.088% 22.298% —
Panel 2: Momentum-reversal strategy
Quintile 1 Quintile 2 Quintile 3 Quintile 4 Quintile 5 RM
Annualized return (cid:0) 20.144% (cid:0) 17.747% (cid:0) 15.981% (cid:0) 15.022% (cid:0) 21.312% (cid:0) 7.620%
Sharp ratio (cid:0) 0.064 (cid:0) 0.070 (cid:0) 0.065 (cid:0) 0.064 (cid:0) 0.089 (cid:0) 0.027
Max. drawdown (cid:0) 34.123% (cid:0) 27.822% (cid:0) 24.707% (cid:0) 22.425% (cid:0) 27.636% (cid:0) 17.486%
Win rate 44.826% 22.170% 23.241% 22.951% 22.182% —
making wrong investment decisions; 4) the results in both Methods 3 and 4 considerably outperform the market index, which indicates
that the momentum factor is a workable market timing indicator. In sum, this trading strategy seems robust.
2. Robustness Test for CMRS
Data in 2015 are divided into groups, since there exists a crash. The cumulative return of each group is shown in Fig. A.1.
In a relatively stable market (before August 2015), the cumulative return of each group has a clear monotonic pattern. The cu-
mulative returns of quintiles 1 to 3 are higher than the cumulative return of the market index.8 When the market experiences fluc-
tuation (since August 2015), the cumulative return of each group shows no clear monotonic pattern. The performances of both CMRS
and the momentum-reversal strategy are shown in Table A.2.
Although all groups underperform the market index, the annualized returns and maximum drawdowns of CMRS maintain their
monotonicity. In this case, investors can long the best performer and short the worst performer so as to achieve excess returns. All
indicators of the momentum-reversal strategy fail to maintain their monotonic patterns, indicating that this strategy is ineffective. The
CMRS contains certain robustness as it achieves favorable results during a momentum crash.
8 The empirical results observed between Jan. 2015 and Aug. 2015 are opposite to those observed in 2018 although both periods are relatively
stable. A potential reason is the existence of bubbles in 2015. The theory of rational bubbles (Blanchard and Watson, 1982) implies that investors’
irrational behavior and expectations induce the stock price to deviate from its true value, giving rise to a bubble. Once the bubble bursts, it
automatically leads to a crash in the market. Scheinkman and Xiong (2003) also point out that heterogeneous beliefs generated by investors’
overconfidence bring bubbles to the market and then increase the possibility of its crash. Heterogeneous beliefs from overconfident investors would
result in prices that deviate from the true value, and then lead to a momentum reversal effect in the market. As a result, the order of cumulative
returns from quintiles 1 to 5 in Figure A.1 before the crash in August is different from that in Figure 1.
6

### Additional Content

#### 1

FinanceResearchLetters45(2022)102185
Contents lists available at ScienceDirect
Finance Research Letters
journal homepage: www.elsevier.com/locate/frl
Crash-based quantitative trading strategies: Perspective of
behavioral finance
Yan Fanga, Jie Yuana, J. Jimmy Yangb,*, Shangjun Yinga,1
aShanghai University of International Business and Economics
bOregon State University, School of Accounting, Finance, and Information Systems, College of Business, 426 Austin Hall, Corvallis, OR,
97331United States
A R T I C L E I N F O A B S T R A C T
JEL codes: Inspired by the studies on stock market crashes, we use documented indicators from behavioral
G1 finance to construct two quantitative trading strategies, i.e., Crash +Timing Strategy and Crash
C22 +Momentum-Reversal Strategy. Empirical analyses show that both strategies are effective and
Keywords: robust. Behavioral factors can be beneficial to investors when they are incorporated into their
Behavioral finance trading strategies.
Crash factor
Market timing
Momentum-reversal strategy
1. Introduction
With the development of the behavioral finance, many scholars seek to explain the causes of stock market crashes using behavioral
theories. For instance, Avery and Zemsky (1998) point out that herding can lead to mispricing of assets and price bubbles. Kim et al.
(2016) suggest that firms with overconfident CEOs have higher stock price crash risk than other firms. Jang and Kang (2019) find that
price crashes could be caused by the overpricing driven by institutional investors. While existing studies focus on exploring the
influential factors that may lead to stock market crashes or predicting crash events in stock markets, we investigate how investors can
profit from those events.
Based on the aforementioned studies, a price crash can be caused by investors’ cognitive biases, such as herding bias and over-
confidence bias. Even though such cognitive biases can result in mispricing of securities, investors may theoretically expect excess
returns generated from a stock market crash as long as they rationally use certain behavioral deviations (Conrad et al., 2014; Jang and
Kang, 2019; Pelster, 2020). Therefore, it is highly possible to realize excess returns from a stock market crash.
During a stock market crash, stock prices typically move in a similar pattern. First, stock prices continue to rise for an extended
period until the bubble reaches its peak. Then, the bubble bursts and the market collapses with a sudden slump. If one wants to profit
from a stock market crash, the ability to select stocks and set the timing is critical. Accordingly, this paper adopts the behavioral
finance approach to propose two quantitative trading strategies based on a documented crash factor. One strategy is built on the crash
factor and market timing (i.e., Crash + Timing Strategy, CTS), while the other is constructed by applying the crash factor and
momentum-reversal strategy (i.e., Crash +Momentum-Reversal Strategy, CMRS). Our empirical analyses present stable returns and
* Corresponding author.
E-mail address: jimmy.yang@bus.oregonstate.edu (J.J. Yang).
1 Ying acknowledges financial support from the National Natural Science Foundation of China (71571116).
https://doi.org/10.1016/j.frl.2021.102185
Received 10 June 2020; Received in revised form 11 April 2021; Accepted 26 May 2021
Availableonline30May2021
1544-6123/©2021ElsevierInc.Allrightsreserved.

#### 2

Y. Fang et al. F i n a n c e R e s e a r c h L e t t e r s 45(2022)102185
confirm the effectiveness and robustness of these market crash-based trading strategies.
2. Construction of trading strategies
2.1. Crash +timing strategy
According to the concept of mean reversion and the gambler’s fallacy (Tversky and Kahneman, 1971), stock prices can rebound
from the current crash and possibly experience another crash. Investors can generate profits by buying the dips during the crash and
selling the rips during the consequent rebound. We use the crash factor2 of Jang and Kang (2019) to screen stocks. Specifically, top 10%
stocks at the end of the previous month (t - 1) and the current month (t) are sorted out by the crash factor, and the top 10% stocks at the
end of the current month are regarded as stocks whose bubbles just burst. However, stocks selected solely based on crash factor could
again take a nosedive, so they cannot be directly put into quantitative investing.3 We consider appropriate timing for buying and
selling (i.e. market timing) by constructing the crash +timing strategy.
Due to the self-serving memory bias, investors generally repeat their previous behaviors for a risky stock position (Go¨dker et al.,
2020). Therefore, we assume that before a stock market crashes, stocks will display characteristics similar to those perceived during the
last crash. The strength/weakness of buying and selling on long and short positions before and after a crash would be the best indicator
for market timing measurement. Consequently, we use the relative strength index (RSI) (Wilder, 1978) to identify the best times to buy
stocks, and apply a momentum factor, which is defined as the corresponding 5-day momentum at the peak from last month, to capture
the sell signals. During an observation period, we sell stocks when the momentum factor is similar to the peak momentum factor in the
previous month (t-1). The specific process of CTS is given in Algorithm 1.
2.2. Crash +momentum-reversal strategy
Momentum (Jegadeesh and Titman, 1993) and reversal are two well-known anomalies within the field of behavioral finance.
However, momentum strategies suffer from occasional large crashes (i.e. momentum crash) (Barroso and Santa-Clara, 2015; Daniel
and Moskowitz, 2016). During a momentum crash, momentum often tends to reverse. That is, the portfolio with the lowest momentum
becomes the best performer, and the momentum-reversal strategy realizes excess returns. Therefore, the momentum-reversal strategy
can considerably enhance investment efficiency.
The second strategy is to combine the crash factor with momentum-reversal strategy. We first add the crash factor to the
momentum-reversal strategy, adopt a scoring method to construct a new comprehensive scoring factor, and use this new factor to
screen stocks. The stocks selected are featured with crashes in current month and expected to rebound later. That is, underperformed
stocks will likely outperform in the future (Bali et al., 2011). The process of CMRS is shown in Algorithm 2.
3. Empirical analyses
We perform empirical analyses using daily stock data from January 1, 2018 to December 31, 2018 and the Center for Research in
Security Prices (CRSP) value-weighted index as the market index. The sample includes all stocks with share codes 10 and 11 from
NYSE, AMEX and NASDAQ. All data are from the CRSP database. The data processing and calculation of crash factor are conducted
based on Jang and Kang (2019) (See Appendix 1).
3.1. Performance of CTS
To determine when to buy stocks, we choose the 14-day RSI4 within the oversold level as the buy limit price. An RSI below 30
(Wilder, 1978) would suggest a buy. Since a rebound may happen or continue beyond the current month, we use the next two months
as the observation period. In addition, we consider the following four methods to determine the sell threshold.
Method 1: Use RSI as the benchmark and the overbought level (RSI=70) as a threshold;
Method 2: Adopt the pre-set take profit level (i.e., 3%)5 as a threshold;
Method 3: Utilize the highest cumulative return over the last month captured by the momentum factor as a threshold;
Method 4: Consider the highest cumulative return over the last month captured by the momentum-information discreteness (Da
et al., 2014) as a threshold.
When adopting different methods to identify the best times to sell stocks, the stop-loss order is executed at 2.5% below the current
stock price (Han et al., 2016). Evaluating the effectiveness of a strategy is essential for an investment. We evaluate our strategies based
2 The crash factor is defined as the probability in the event of log returns lower than (cid:0) 70% over the next 12 months.
3 We do not discuss profit taking from short selling herein due to multiple short-selling bans in the world stock markets.
4 In order to choose the appropriate timeframe, a comparison is conducted among the respective performance of 14-day RSI, 24-day RSI and
market index, seeking the timeframe that outperforms the market index.
5 This is how to avoid emotional investing and selling into corrections within the field of behavioral finance.
2

#### 3

Y. Fang et al. F i n a n c e R e s e a r c h L e t t e r s 45(2022)102185
Table 1
Performance of CTS with different methods in 2018.
Method 1 Method 2 Method 3 Method 4 Market Index
Annualized return (cid:0) 7.145% 3.296% 8.464% 5.515% (cid:0) 10.017%
Win rate 30.113% 56.000% 37.113% 45.455% —
Note: The sell signal in Method 1 is 14-day RSI above 70. The sell signal in Method 2 is when the cumulative return is 3% above the purchase price.
The sell signals in Method 3 and Method 4 are the 5-day momentum value and the 5-day information discreteness that are equal to or above the
highest cumulative return over the last month, respectively.
Fig. 1. Performance of CMRS from quintile 1 to 5 for 2018, and the benchmark refers to the cumulative return of the market index.
Table 2
This table reports the annualized return, Sharp ratio, maximum drawdown and win rate on quintile portfolios sorted on the comprehensive scoring
factors. At the end of each month i, quintile portfolios are constructed by sorting stocks based on the comprehensive scoring factors. Column quintile 1
is for the highest comprehensive scoring factors quintile, while column quintile 5 shows the results for the lowest comprehensive factors quintile.
Column CRSP is for the performance of CRSP value-weighted index.
Panel 1: CMRS
Quintile 1 Quintile 2 Quintile 3 Quintile 4 Quintile 5 CRSP
Annualized return (cid:0) 71.570% (cid:0) 36.926% 17.761% 39.228% 104.276% (cid:0) 6.012%
Sharp ratio (cid:0) 0.361775 (cid:0) 0.151652 0.064567 0.138178 0.328911 (cid:0) 0.018106
Max. drawdown (cid:0) 74.372% (cid:0) 45.374% (cid:0) 21.431% (cid:0) 14.036% (cid:0) 9.426% (cid:0) 20.960%
Win rate 13.878% 17.809% 26.712% 31.133% 40.288% —
Panel 2: Momentum-reversal strategy
Quintile 1 Quintile 2 Quintile 3 Quintile 4 Quintile 5 CRSP
Annualized return (cid:0) 80.798% (cid:0) 41.240% (cid:0) 7.102% 46.513% 287.544% (cid:0) 6.012%
Sharp ratio (cid:0) 0.501626 (cid:0) 0.183588 (cid:0) 0.022715 0.156668 0.517607 (cid:0) 0.018106
Max. drawdown (cid:0) 82.073% (cid:0) 48.318% (cid:0) 24.038% (cid:0) 13.444% (cid:0) 7.187% (cid:0) 20.960%
Win rate 4.147% 9.704% 27.437% 38.024% 45.663% —
on the annualized return and win rate, where the win rate refers to the ratio of the number of stocks with a positive return to the
number of stocks with non-zero returns. The results are shown in Table 1, where the assessments for Methods 1–4 are given in columns
2–5 and the results of market index is given in column 6.
In Table 1, we can see that 1) the market return in 2018 is negative and all four Methods outperform the market index; 2) the
annualized return in Method 1 is negative, and its win rate is the lowest, which implies that the market timing strategy solely
depending on RSI is insufficient6; 3) Method 2 has the highest win rate, which indicates that the concept of take-profit can effectively
avoid making wrong investment decisions, and achieves good performance in quantitative investing; 4) the annualized returns in both
Methods 3 and 4 are positive and are substantially higher than the market return, which suggests that the momentum factor is a
6 In addition to the sell threshold of RSI=70 suggested by Wilder (1978), we also use RSI=80 as the sell threshold and RSI=20 as the buy
threshold. The annualized return and win rate of Method 1 are both improved. Specifically, the annualized return improves from (cid:0) 7.145% to
7.465%, while the win rate increases from 30.113% to 34.408%.
3

#### 4

Y. Fang et al. F i n a n c e R e s e a r c h L e t t e r s 45(2022)102185
Algorithm 1
The process for the CTS.
1 Start with time t,
a) Pick up top 10% stocks based on crash factors at the end of month t and month t (cid:0) 1 and name them as Set 1 and Set 2, respectively;
b) Remove stocks which appear on both month t (cid:0) 1 and month t from set 1, and use the remaining stocks to create a new portfolio.
c) For stocks listed in our new portfolio,
i we will buy when they reach the buy limit order during our observation period;
ii we will sell when they reach the sell limit order or when the stop-loss price has been reached during our observation period; otherwise, we
will sell them at the end of the observation period.
2 2. Repeat Step 1.
Algorithm 2
The process of CMRS.
1 Start with time t,
a) Buy the top 20% of stocks sorted by the new comprehensive factor at the end of month t;
b) Hold stocks till the end of month t +1 and sell them.
2 Repeat Step 1.
workable market timing indicator.
3.2. Performance of CMRS
In order to evaluate the performance of CMRS, all momentum factors and crash factors at the end of the month (t) are sorted in
ascending and descending order, respectively, and the comprehensive scoring factors are calculated according to the scoring method.7
The comprehensive scoring factors are sorted in descending order, and divided into five groups (Fama and French, 1993). We buy the
stocks in these five groups in month t and sell them at the end of the month (t +1). Finally, we calculate the cumulative return in each
group. The results are shown in Fig. 1, where each curve represents the cumulative log return of a quintile regression group.
Despite of some fluctuations in the market of 2018, the cumulative return of each group shows a clear monotonic pattern. Among
them, the cumulative returns of quintiles 3 to 5 are all higher than the cumulative return of the market index, which indicates that the
trading strategy brings investors an excess return. We also compare and analyze the performance of both CMRS and the traditional
momentum-reversal strategy by using four indicators, i.e., annualized return, Sharpe ratio, maximum drawdown and win rate. The
results are presented in Table 2.
Both strategies show strong portfolio performance. All indicators in quintiles 1–5 show clear monotonic patterns. CMRS out-
performs the momentum-reversal strategy in quintiles 1–3. CMRS also outperforms the market index in quintiles 3–5. Although the
best performer Quintile 5 of CMRS falls behind the control group of the momentum-reversal strategy, this trading strategy is overall an
effective stock selection strategy.
Generally speaking, momentum crash is the most important factor that restricts application of the momentum strategy, and it can
sometimes cause the momentum factor to fail. Although the market fluctuated in 2018, there was no momentum crashes. That is the
main reason that we cannot find an obvious advantage of our proposed strategy in Table 2. To show that CMRS is effective during a
momentum crash, we examine the data from January 1, 2015 to December 31, 2015 (momentum crashes happened over this period) to
test the robustness of our trading strategies. The results show that both CTS and CMRS achieve favorable results, but the momentum-
reversal strategy generally fails. Please see Appendix 2 for details.
In addition, we examine an up market period from January 1, 2019 to July 31, 2019 to test whether our trading strategies would
work when the overall market return is positive. We find that the CTS strategy performs even better in the up market as annualized
returns and win rates in 2019 are higher than those in 2018 and 2015. Results from the CMRS strategy are similar to those in 2018.
Those results are not tabulated, but they are available on request.
4. Conclusion
Inspired by the latest studies on stock market crashes, this paper constructs two quantitative trading strategies and tests their
performances. Empirical analyses show that 1) the crash factor cannot directly function as a momentum factor, but when the crash
factor combines with an appropriate timing indicator, it could generate excess returns for investors; 2) the momentum factor is useful
in a market timing approach; 3) during a volatile market, a momentum-reversal strategy may fail while the CMRS proposed in this
paper is still effective.
Unlike previous studies, this article explores quantitative trading strategies from the perspective of behavioral finance, considering
7 The so-called scoring method is a method where stocks’ crash factors are ranked in ascending order, their momentum factors are ranked in
descending order, and the average of the two ranks is defined as the comprehensive scoring factor.
4

#### 5

Y. Fang et al. F i n a n c e R e s e a r c h L e t t e r s 45(2022)102185
Table A.1
Performance of CTS with different methods in 2015.
Method 1 Method 2 Method 3 Method 4 Market Index
Annualized return (cid:0) 4.749% 9.497% 6.332% 0.712% (cid:0) 6.989%
Win rate 28.049% 57.927% 46.154% 43.357% —
the effect from stock market crash, momentum factor, and RSI. This approach provides investors useful trading strategies to potentially
exploit behavioral bias in stock markets. However, there is room for future studies to further explore and extend these strategies. First,
more studies are needed in the application of those strategies. For instance, potential profits from long-short portfolios in the context of
the second strategy can be explored. Second, the construction of the two strategies only takes into consideration the behavior of short-
term investors. A long-term approach can provide a complete picture and strengthen our understanding of market crash-based trading
strategies. Third, we use the market index as the benchmark to measure performance of our trading strategies, but risk-adjusted returns
can be used to appropriately account for risks involved in the trading strategies.
Declarations of Competing Interest
None
Appendix 1. Crash factor
Crash factor is defined as:
exp(C)
Crashfactor=
1+exp(C)+exp(J)
where
C=α +α RM12+α EXRET12+α TVOL+α TSKEW+α SIZE+α DTURN+α AGE+α TANG+α SALESG
0 1 2 3 4 5 6 7 8 9
and
J=β +β RM12+β EXRET12+β TVOL+β TSKEW+β SIZE+β DTURN+β AGE+β TANG+β SALESG
0 1 2 3 4 5 6 7 8 9
Variable RM12 is the log return of the CRSP value-weighted index over the past 12 months; EXRET12 is the log return over the past
12 months in excess of RM12; TVOL and TSKEW are the standard deviation and skewness, respectively, of daily log returns over the
past 6 months; SIZE is the log of market capitalization; DTURN is the detrended turnover, defined by the six-month average of monthly
share turnover minus the prior 18-month average; AGE is the number of years since the firm’s first appearance on the CRSP monthly
stock file; TANG is tangible assets divided by total asset; and SALESG is the sales growth over the prior year. Parameters α i and β i (i =0,
1, ⋅⋅⋅, 9) are from Jang and Kang (2019).
Appendix 2. Robustness testing for quantitative trading strategies
What does the advantage of the proposed trading strategies lie in, chance or strength? We further test their reliability and
robustness. Data from CRSP database over the period from January 1, 2015 to December 31, 2015 are used for the robustness testing.
1. Robustness Test for CTS
To identify the best times for selling stocks, the 24-day RSI within the overbought level is used as the buy limit price, and the stop-
loss order is set to be 2.5% below the current stock price. The strategy’s performances for four methods defined in Section 3 are given in
Table A.1.
As we can see, 1) over the period, the market return is negative and all the Methods 1–4 outperform the market index; 2) the
annualized return in Method 1 is negative and its win rate is the lowest, indicating that the market timing strategy solely depending on
RSI is insufficient; 3) Method 2 has the highest annualized return and win rate, implying the take-profit strategy can effectively avoid
5

#### 6

Y. Fang et al. F i n a n c e R e s e a r c h L e t t e r s 45(2022)102185
Reference
Avery, C., Zemsky, P., 1998. Multidimensional uncertainty and herd behavior in financial markets. Am. Econ. Rev. 724–748.
Bali, T.G., Cakici, N., Whitelaw, R.F., 2011. Maxing out: stocks as lotteries and the cross-section of expected returns. J. Financ. Econ. 99 (2), 427–446.
Barroso, P., Santa-Clara, P., 2015. Momentum has its moments. J. Financ. Econ. 116 (1), 111–120.
Blanchard, O.J., Watson, M.W., 1982. Bubbles, Rational Expectations and Financial Markets. NBER working paper, (w0945).
Conrad, J., Kapadia, N., Xing, Y., 2014. Death and jackpot: why do individual investors hold overpriced stocks? J. Financ. Econ. 113 (3), 455–475.
Daniel, K., Moskowitz, T.J., 2016. Momentum crashes. J. Financ. Econ. 122 (2), 221–247.
Da, Z., Gurun, U.G., Warachka, M., 2014. Frog in the pan: continuous information and momentum. Rev. Financ. Stud. 27 (7), 2171–2218.
Fama, E.F., French, K.R., 1993. Common risk factors in the returns on stocks and bonds. J. Financ. Econ. 33, 3–56.
Go¨dker, K., Peiran Jiao, R., Smeets, R., 2020. In: Investor Memory. 2020 AFA Annual Meeting Working Paper. https://papers.ssrn.com/sol3/papers.cfm?abstract_
id=3348315.
Han, Y., Zhou, G., Zhu, Y., 2016. Taming Momentum crashes: A simple Stop-Loss Strategy. Available at SSRN 2407199.
Kim, J.B., Zhang, L., 2016. CEO Overconfidence and Stock Price Crash Risk. Contemp. Account. Res. 33 (4), 1720–1749.
Jang, J., Kang, J., 2019. Probability of price crashes, rational speculative bubbles, and the cross-section of stock returns. J. Financ. Econ. 132 (1), 222–247.
Jegadeesh, N., Titman, S., 1993. Returns to buying winners and selling losers: implications for stock market efficiency. J. Finance 48 (1), 65–91.
Pelster, M., 2020. The gambler’s and hot-hand fallacies: empirical evidence from trading data. Econ. Lett. 187, 108887.
Scheinkman, J.A., Xiong, W., 2003. Overconfidence and speculative bubbles. J. Political Econ. 111 (6), 1183–1220.
Tversky, A., Kahneman, D., 1971. Belief in the law of small numbers. Psychol. Bull. 76 (2), 105–110.
Wilder Jr., J.W., 1978. New Concepts in Technical Trading Systems. Hunter Publishing Company, Greensboro, N.C.
7


---

## Content Unique to Old Extraction (not found in markitdown output)

### 1. # Crash-based quantitative trading strategies- Perspective of behavioral finance...

# Crash-based quantitative trading strategies- Perspective of behavioral finance

### 2. > *Source PDF: Crash-based quantitative trading strategies- Perspective of behav...

> *Source PDF: Crash-based quantitative trading strategies- Perspective of behavioral finance.pdf*
> *Extraction: Combined — markitdown raw text (base) + previous extraction supplements*

### 3. ## Content from Previous Extraction (not in markitdown output)...

## Content from Previous Extraction (not in markitdown output)

### 4. ### Visual/Chart/Graph Descriptions...

### Visual/Chart/Graph Descriptions

### 5. Y. Fang et al. F i n a n c e R e s e a r c h L e t t e r s 45(2022)102185
Fig. A...

Y. Fang et al. F i n a n c e R e s e a r c h L e t t e r s 45(2022)102185
Fig. A.1. Performance of CMRS from quintile 1 to 5 for 2015 CMRS. Note: Benchmark refers to the cumulative return of the market index. The
vertical line represents the date of July 31, 2015.
Table A.2
Performance of CMRS and Momentum-reversal Strategy in 2015.
Panel 1: CMRS
Quintile 1 Quintile 2 Quintile 3 Quintile 4 Quintile 5 RM
Annualized return (cid:0) 23.039% (cid:0) 20.981% (cid:0) 19.836% (cid:0) 12.608% (cid:0) 13.447% (cid:0) 7.620%
Sharp ratio (cid:0) 0.073 (cid:0) 0.082 (cid:0) 0.078 (cid:0) 0.052 (cid:0) 0.062 (cid:0) 0.027
Max. drawdown (cid:0) 36.575% (cid:0) 31.096% (cid:0) 29.028% (cid:0) 21.128% (cid:0) 19.734% (cid:0) 17.486%
Win rate 45.117% 22.472% 22.532% 23.088% 22.298% —
Panel 2: Momentum-reversal strategy
Quintile 1 Quintile 2 Quintile 3 Quintile 4 Quintile 5 RM
Annualized return (cid:0) 20.144% (cid:0) 17.747% (cid:0) 15.981% (cid:0) 15.022% (cid:0) 21.312% (cid:0) 7.620%
Sharp ratio (cid:0) 0.064 (cid:0) 0.070 (cid:0) 0.065 (cid:0) 0.064 (cid:0) 0.089 (cid:0) 0.027
Max. drawdown (cid:0) 34.123% (cid:0) 27.822% (cid:0) 24.707% (cid:0) 22.425% (cid:0) 27.636% (cid:0) 17.486%
Win rate 44.826% 22.170% 23.241% 22.951% 22.182% —
making wrong investment decisions; 4) the results in both Methods 3 and 4 considerably outperform the market index, which indicates
that the momentum factor is a workable market timing indicator. In sum, this trading strategy seems robust.
2. Robustness Test for CMRS
Data in 2015 are divided into groups, since there exists a crash. The cumulative return of each group is shown in Fig. A.1.
In a relatively stable market (before August 2015), the cumulative return of each group has a clear monotonic pattern. The cu-
mulative returns of quintiles 1 to 3 are higher than the cumulative return of the market index.8 When the market experiences fluc-
tuation (since August 2015), the cumulative return of each group shows no clear monotonic pattern. The performances of both CMRS
and the momentum-reversal strategy are shown in Table A.2.
Although all groups underperform the market index, the annualized returns and maximum drawdowns of CMRS maintain their
monotonicity. In this case, investors can long the best performer and short the worst performer so as to achieve excess returns. All
indicators of the momentum-reversal strategy fail to maintain their monotonic patterns, indicating that this strategy is ineffective. The
CMRS contains certain robustness as it achieves favorable results during a momentum crash.
8 The empirical results observed between Jan. 2015 and Aug. 2015 are opposite to those observed in 2018 although both periods are relatively
stable. A potential reason is the existence of bubbles in 2015. The theory of rational bubbles (Blanchard and Watson, 1982) implies that investors’
irrational behavior and expectations induce the stock price to deviate from its true value, giving rise to a bubble. Once the bubble bursts, it
automatically leads to a crash in the market. Scheinkman and Xiong (2003) also point out that heterogeneous beliefs generated by investors’
overconfidence bring bubbles to the market and then increase the possibility of its crash. Heterogeneous beliefs from overconfident investors would
result in prices that deviate from the true value, and then lead to a momentum reversal effect in the market. As a result, the order of cumulative
returns from quintiles 1 to 5 in Figure A.1 before the crash in August is different from that in Figure 1.
6

### 6. FinanceResearchLetters45(2022)102185
Contents lists available at ScienceDirect
F...

FinanceResearchLetters45(2022)102185
Contents lists available at ScienceDirect
Finance Research Letters
journal homepage: www.elsevier.com/locate/frl
Crash-based quantitative trading strategies: Perspective of
behavioral finance
Yan Fanga, Jie Yuana, J. Jimmy Yangb,*, Shangjun Yinga,1
aShanghai University of International Business and Economics
bOregon State University, School of Accounting, Finance, and Information Systems, College of Business, 426 Austin Hall, Corvallis, OR,
97331United States
A R T I C L E I N F O A B S T R A C T
JEL codes: Inspired by the studies on stock market crashes, we use documented indicators from behavioral
G1 finance to construct two quantitative trading strategies, i.e., Crash +Timing Strategy and Crash
C22 +Momentum-Reversal Strategy. Empirical analyses show that both strategies are effective and
Keywords: robust. Behavioral factors can be beneficial to investors when they are incorporated into their
Behavioral finance trading strategies.
Crash factor
Market timing
Momentum-reversal strategy
1. Introduction
With the development of the behavioral finance, many scholars seek to explain the causes of stock market crashes using behavioral
theories. For instance, Avery and Zemsky (1998) point out that herding can lead to mispricing of assets and price bubbles. Kim et al.
(2016) suggest that firms with overconfident CEOs have higher stock price crash risk than other firms. Jang and Kang (2019) find that
price crashes could be caused by the overpricing driven by institutional investors. While existing studies focus on exploring the
influential factors that may lead to stock market crashes or predicting crash events in stock markets, we investigate how investors can
profit from those events.
Based on the aforementioned studies, a price crash can be caused by investors’ cognitive biases, such as herding bias and over-
confidence bias. Even though such cognitive biases can result in mispricing of securities, investors may theoretically expect excess
returns generated from a stock market crash as long as they rationally use certain behavioral deviations (Conrad et al., 2014; Jang and
Kang, 2019; Pelster, 2020). Therefore, it is highly possible to realize excess returns from a stock market crash.
During a stock market crash, stock prices typically move in a similar pattern. First, stock prices continue to rise for an extended
period until the bubble reaches its peak. Then, the bubble bursts and the market collapses with a sudden slump. If one wants to profit
from a stock market crash, the ability to select stocks and set the timing is critical. Accordingly, this paper adopts the behavioral
finance approach to propose two quantitative trading strategies based on a documented crash factor. One strategy is built on the crash
factor and market timing (i.e., Crash + Timing Strategy, CTS), while the other is constructed by applying the crash factor and
momentum-reversal strategy (i.e., Crash +Momentum-Reversal Strategy, CMRS). Our empirical analyses present stable returns and
* Corresponding author.
E-mail address: jimmy.yang@bus.oregonstate.edu (J.J. Yang).
1 Ying acknowledges financial support from the National Natural Science Foundation of China (71571116).
https://doi.org/10.1016/j.frl.2021.102185
Received 10 June 2020; Received in revised form 11 April 2021; Accepted 26 May 2021
Availableonline30May2021
1544-6123/©2021ElsevierInc.Allrightsreserved.

### 7. Y. Fang et al. F i n a n c e R e s e a r c h L e t t e r s 45(2022)102185
confir...

Y. Fang et al. F i n a n c e R e s e a r c h L e t t e r s 45(2022)102185
confirm the effectiveness and robustness of these market crash-based trading strategies.
2. Construction of trading strategies
2.1. Crash +timing strategy
According to the concept of mean reversion and the gambler’s fallacy (Tversky and Kahneman, 1971), stock prices can rebound
from the current crash and possibly experience another crash. Investors can generate profits by buying the dips during the crash and
selling the rips during the consequent rebound. We use the crash factor2 of Jang and Kang (2019) to screen stocks. Specifically, top 10%
stocks at the end of the previous month (t - 1) and the current month (t) are sorted out by the crash factor, and the top 10% stocks at the
end of the current month are regarded as stocks whose bubbles just burst. However, stocks selected solely based on crash factor could
again take a nosedive, so they cannot be directly put into quantitative investing.3 We consider appropriate timing for buying and
selling (i.e. market timing) by constructing the crash +timing strategy.
Due to the self-serving memory bias, investors generally repeat their previous behaviors for a risky stock position (Go¨dker et al.,
2020). Therefore, we assume that before a stock market crashes, stocks will display characteristics similar to those perceived during the
last crash. The strength/weakness of buying and selling on long and short positions before and after a crash would be the best indicator
for market timing measurement. Consequently, we use the relative strength index (RSI) (Wilder, 1978) to identify the best times to buy
stocks, and apply a momentum factor, which is defined as the corresponding 5-day momentum at the peak from last month, to capture
the sell signals. During an observation period, we sell stocks when the momentum factor is similar to the peak momentum factor in the
previous month (t-1). The specific process of CTS is given in Algorithm 1.
2.2. Crash +momentum-reversal strategy
Momentum (Jegadeesh and Titman, 1993) and reversal are two well-known anomalies within the field of behavioral finance.
However, momentum strategies suffer from occasional large crashes (i.e. momentum crash) (Barroso and Santa-Clara, 2015; Daniel
and Moskowitz, 2016). During a momentum crash, momentum often tends to reverse. That is, the portfolio with the lowest momentum
becomes the best performer, and the momentum-reversal strategy realizes excess returns. Therefore, the momentum-reversal strategy
can considerably enhance investment efficiency.
The second strategy is to combine the crash factor with momentum-reversal strategy. We first add the crash factor to the
momentum-reversal strategy, adopt a scoring method to construct a new comprehensive scoring factor, and use this new factor to
screen stocks. The stocks selected are featured with crashes in current month and expected to rebound later. That is, underperformed
stocks will likely outperform in the future (Bali et al., 2011). The process of CMRS is shown in Algorithm 2.
3. Empirical analyses
We perform empirical analyses using daily stock data from January 1, 2018 to December 31, 2018 and the Center for Research in
Security Prices (CRSP) value-weighted index as the market index. The sample includes all stocks with share codes 10 and 11 from
NYSE, AMEX and NASDAQ. All data are from the CRSP database. The data processing and calculation of crash factor are conducted
based on Jang and Kang (2019) (See Appendix 1).
3.1. Performance of CTS
To determine when to buy stocks, we choose the 14-day RSI4 within the oversold level as the buy limit price. An RSI below 30
(Wilder, 1978) would suggest a buy. Since a rebound may happen or continue beyond the current month, we use the next two months
as the observation period. In addition, we consider the following four methods to determine the sell threshold.
Method 1: Use RSI as the benchmark and the overbought level (RSI=70) as a threshold;
Method 2: Adopt the pre-set take profit level (i.e., 3%)5 as a threshold;
Method 3: Utilize the highest cumulative return over the last month captured by the momentum factor as a threshold;
Method 4: Consider the highest cumulative return over the last month captured by the momentum-information discreteness (Da
et al., 2014) as a threshold.
When adopting different methods to identify the best times to sell stocks, the stop-loss order is executed at 2.5% below the current
stock price (Han et al., 2016). Evaluating the effectiveness of a strategy is essential for an investment. We evaluate our strategies based
2 The crash factor is defined as the probability in the event of log returns lower than (cid:0) 70% over the next 12 months.
3 We do not discuss profit taking from short selling herein due to multiple short-selling bans in the world stock markets.
4 In order to choose the appropriate timeframe, a comparison is conducted among the respective performance of 14-day RSI, 24-day RSI and
market index, seeking the timeframe that outperforms the market index.
5 This is how to avoid emotional investing and selling into corrections within the field of behavioral finance.
2

### 8. Y. Fang et al. F i n a n c e R e s e a r c h L e t t e r s 45(2022)102185
Table ...

Y. Fang et al. F i n a n c e R e s e a r c h L e t t e r s 45(2022)102185
Table 1
Performance of CTS with different methods in 2018.
Method 1 Method 2 Method 3 Method 4 Market Index
Annualized return (cid:0) 7.145% 3.296% 8.464% 5.515% (cid:0) 10.017%
Win rate 30.113% 56.000% 37.113% 45.455% —
Note: The sell signal in Method 1 is 14-day RSI above 70. The sell signal in Method 2 is when the cumulative return is 3% above the purchase price.
The sell signals in Method 3 and Method 4 are the 5-day momentum value and the 5-day information discreteness that are equal to or above the
highest cumulative return over the last month, respectively.
Fig. 1. Performance of CMRS from quintile 1 to 5 for 2018, and the benchmark refers to the cumulative return of the market index.
Table 2
This table reports the annualized return, Sharp ratio, maximum drawdown and win rate on quintile portfolios sorted on the comprehensive scoring
factors. At the end of each month i, quintile portfolios are constructed by sorting stocks based on the comprehensive scoring factors. Column quintile 1
is for the highest comprehensive scoring factors quintile, while column quintile 5 shows the results for the lowest comprehensive factors quintile.
Column CRSP is for the performance of CRSP value-weighted index.
Panel 1: CMRS
Quintile 1 Quintile 2 Quintile 3 Quintile 4 Quintile 5 CRSP
Annualized return (cid:0) 71.570% (cid:0) 36.926% 17.761% 39.228% 104.276% (cid:0) 6.012%
Sharp ratio (cid:0) 0.361775 (cid:0) 0.151652 0.064567 0.138178 0.328911 (cid:0) 0.018106
Max. drawdown (cid:0) 74.372% (cid:0) 45.374% (cid:0) 21.431% (cid:0) 14.036% (cid:0) 9.426% (cid:0) 20.960%
Win rate 13.878% 17.809% 26.712% 31.133% 40.288% —
Panel 2: Momentum-reversal strategy
Quintile 1 Quintile 2 Quintile 3 Quintile 4 Quintile 5 CRSP
Annualized return (cid:0) 80.798% (cid:0) 41.240% (cid:0) 7.102% 46.513% 287.544% (cid:0) 6.012%
Sharp ratio (cid:0) 0.501626 (cid:0) 0.183588 (cid:0) 0.022715 0.156668 0.517607 (cid:0) 0.018106
Max. drawdown (cid:0) 82.073% (cid:0) 48.318% (cid:0) 24.038% (cid:0) 13.444% (cid:0) 7.187% (cid:0) 20.960%
Win rate 4.147% 9.704% 27.437% 38.024% 45.663% —
on the annualized return and win rate, where the win rate refers to the ratio of the number of stocks with a positive return to the
number of stocks with non-zero returns. The results are shown in Table 1, where the assessments for Methods 1–4 are given in columns
2–5 and the results of market index is given in column 6.
In Table 1, we can see that 1) the market return in 2018 is negative and all four Methods outperform the market index; 2) the
annualized return in Method 1 is negative, and its win rate is the lowest, which implies that the market timing strategy solely
depending on RSI is insufficient6; 3) Method 2 has the highest win rate, which indicates that the concept of take-profit can effectively
avoid making wrong investment decisions, and achieves good performance in quantitative investing; 4) the annualized returns in both
Methods 3 and 4 are positive and are substantially higher than the market return, which suggests that the momentum factor is a
6 In addition to the sell threshold of RSI=70 suggested by Wilder (1978), we also use RSI=80 as the sell threshold and RSI=20 as the buy
threshold. The annualized return and win rate of Method 1 are both improved. Specifically, the annualized return improves from (cid:0) 7.145% to
7.465%, while the win rate increases from 30.113% to 34.408%.
3

### 9. Y. Fang et al. F i n a n c e R e s e a r c h L e t t e r s 45(2022)102185
Algori...

Y. Fang et al. F i n a n c e R e s e a r c h L e t t e r s 45(2022)102185
Algorithm 1
The process for the CTS.
1 Start with time t,
a) Pick up top 10% stocks based on crash factors at the end of month t and month t (cid:0) 1 and name them as Set 1 and Set 2, respectively;
b) Remove stocks which appear on both month t (cid:0) 1 and month t from set 1, and use the remaining stocks to create a new portfolio.
c) For stocks listed in our new portfolio,
i we will buy when they reach the buy limit order during our observation period;
ii we will sell when they reach the sell limit order or when the stop-loss price has been reached during our observation period; otherwise, we
will sell them at the end of the observation period.
2 2. Repeat Step 1.
Algorithm 2
The process of CMRS.
1 Start with time t,
a) Buy the top 20% of stocks sorted by the new comprehensive factor at the end of month t;
b) Hold stocks till the end of month t +1 and sell them.
2 Repeat Step 1.
workable market timing indicator.
3.2. Performance of CMRS
In order to evaluate the performance of CMRS, all momentum factors and crash factors at the end of the month (t) are sorted in
ascending and descending order, respectively, and the comprehensive scoring factors are calculated according to the scoring method.7
The comprehensive scoring factors are sorted in descending order, and divided into five groups (Fama and French, 1993). We buy the
stocks in these five groups in month t and sell them at the end of the month (t +1). Finally, we calculate the cumulative return in each
group. The results are shown in Fig. 1, where each curve represents the cumulative log return of a quintile regression group.
Despite of some fluctuations in the market of 2018, the cumulative return of each group shows a clear monotonic pattern. Among
them, the cumulative returns of quintiles 3 to 5 are all higher than the cumulative return of the market index, which indicates that the
trading strategy brings investors an excess return. We also compare and analyze the performance of both CMRS and the traditional
momentum-reversal strategy by using four indicators, i.e., annualized return, Sharpe ratio, maximum drawdown and win rate. The
results are presented in Table 2.
Both strategies show strong portfolio performance. All indicators in quintiles 1–5 show clear monotonic patterns. CMRS out-
performs the momentum-reversal strategy in quintiles 1–3. CMRS also outperforms the market index in quintiles 3–5. Although the
best performer Quintile 5 of CMRS falls behind the control group of the momentum-reversal strategy, this trading strategy is overall an
effective stock selection strategy.
Generally speaking, momentum crash is the most important factor that restricts application of the momentum strategy, and it can
sometimes cause the momentum factor to fail. Although the market fluctuated in 2018, there was no momentum crashes. That is the
main reason that we cannot find an obvious advantage of our proposed strategy in Table 2. To show that CMRS is effective during a
momentum crash, we examine the data from January 1, 2015 to December 31, 2015 (momentum crashes happened over this period) to
test the robustness of our trading strategies. The results show that both CTS and CMRS achieve favorable results, but the momentum-
reversal strategy generally fails. Please see Appendix 2 for details.
In addition, we examine an up market period from January 1, 2019 to July 31, 2019 to test whether our trading strategies would
work when the overall market return is positive. We find that the CTS strategy performs even better in the up market as annualized
returns and win rates in 2019 are higher than those in 2018 and 2015. Results from the CMRS strategy are similar to those in 2018.
Those results are not tabulated, but they are available on request.
4. Conclusion
Inspired by the latest studies on stock market crashes, this paper constructs two quantitative trading strategies and tests their
performances. Empirical analyses show that 1) the crash factor cannot directly function as a momentum factor, but when the crash
factor combines with an appropriate timing indicator, it could generate excess returns for investors; 2) the momentum factor is useful
in a market timing approach; 3) during a volatile market, a momentum-reversal strategy may fail while the CMRS proposed in this
paper is still effective.
Unlike previous studies, this article explores quantitative trading strategies from the perspective of behavioral finance, considering
7 The so-called scoring method is a method where stocks’ crash factors are ranked in ascending order, their momentum factors are ranked in
descending order, and the average of the two ranks is defined as the comprehensive scoring factor.
4

### 10. Y. Fang et al. F i n a n c e R e s e a r c h L e t t e r s 45(2022)102185
Table ...

Y. Fang et al. F i n a n c e R e s e a r c h L e t t e r s 45(2022)102185
Table A.1
Performance of CTS with different methods in 2015.
Method 1 Method 2 Method 3 Method 4 Market Index
Annualized return (cid:0) 4.749% 9.497% 6.332% 0.712% (cid:0) 6.989%
Win rate 28.049% 57.927% 46.154% 43.357% —
the effect from stock market crash, momentum factor, and RSI. This approach provides investors useful trading strategies to potentially
exploit behavioral bias in stock markets. However, there is room for future studies to further explore and extend these strategies. First,
more studies are needed in the application of those strategies. For instance, potential profits from long-short portfolios in the context of
the second strategy can be explored. Second, the construction of the two strategies only takes into consideration the behavior of short-
term investors. A long-term approach can provide a complete picture and strengthen our understanding of market crash-based trading
strategies. Third, we use the market index as the benchmark to measure performance of our trading strategies, but risk-adjusted returns
can be used to appropriately account for risks involved in the trading strategies.
Declarations of Competing Interest
None
Appendix 1. Crash factor
Crash factor is defined as:
exp(C)
Crashfactor=
1+exp(C)+exp(J)
where
C=α +α RM12+α EXRET12+α TVOL+α TSKEW+α SIZE+α DTURN+α AGE+α TANG+α SALESG
0 1 2 3 4 5 6 7 8 9
and
J=β +β RM12+β EXRET12+β TVOL+β TSKEW+β SIZE+β DTURN+β AGE+β TANG+β SALESG
0 1 2 3 4 5 6 7 8 9
Variable RM12 is the log return of the CRSP value-weighted index over the past 12 months; EXRET12 is the log return over the past
12 months in excess of RM12; TVOL and TSKEW are the standard deviation and skewness, respectively, of daily log returns over the
past 6 months; SIZE is the log of market capitalization; DTURN is the detrended turnover, defined by the six-month average of monthly
share turnover minus the prior 18-month average; AGE is the number of years since the firm’s first appearance on the CRSP monthly
stock file; TANG is tangible assets divided by total asset; and SALESG is the sales growth over the prior year. Parameters α i and β i (i =0,
1, ⋅⋅⋅, 9) are from Jang and Kang (2019).
Appendix 2. Robustness testing for quantitative trading strategies
What does the advantage of the proposed trading strategies lie in, chance or strength? We further test their reliability and
robustness. Data from CRSP database over the period from January 1, 2015 to December 31, 2015 are used for the robustness testing.
1. Robustness Test for CTS
To identify the best times for selling stocks, the 24-day RSI within the overbought level is used as the buy limit price, and the stop-
loss order is set to be 2.5% below the current stock price. The strategy’s performances for four methods defined in Section 3 are given in
Table A.1.
As we can see, 1) over the period, the market return is negative and all the Methods 1–4 outperform the market index; 2) the
annualized return in Method 1 is negative and its win rate is the lowest, indicating that the market timing strategy solely depending on
RSI is insufficient; 3) Method 2 has the highest annualized return and win rate, implying the take-profit strategy can effectively avoid
5

### 11. Y. Fang et al. F i n a n c e R e s e a r c h L e t t e r s 45(2022)102185
Refere...

Y. Fang et al. F i n a n c e R e s e a r c h L e t t e r s 45(2022)102185
Reference
Avery, C., Zemsky, P., 1998. Multidimensional uncertainty and herd behavior in financial markets. Am. Econ. Rev. 724–748.
Bali, T.G., Cakici, N., Whitelaw, R.F., 2011. Maxing out: stocks as lotteries and the cross-section of expected returns. J. Financ. Econ. 99 (2), 427–446.
Barroso, P., Santa-Clara, P., 2015. Momentum has its moments. J. Financ. Econ. 116 (1), 111–120.
Blanchard, O.J., Watson, M.W., 1982. Bubbles, Rational Expectations and Financial Markets. NBER working paper, (w0945).
Conrad, J., Kapadia, N., Xing, Y., 2014. Death and jackpot: why do individual investors hold overpriced stocks? J. Financ. Econ. 113 (3), 455–475.
Daniel, K., Moskowitz, T.J., 2016. Momentum crashes. J. Financ. Econ. 122 (2), 221–247.
Da, Z., Gurun, U.G., Warachka, M., 2014. Frog in the pan: continuous information and momentum. Rev. Financ. Stud. 27 (7), 2171–2218.
Fama, E.F., French, K.R., 1993. Common risk factors in the returns on stocks and bonds. J. Financ. Econ. 33, 3–56.
Go¨dker, K., Peiran Jiao, R., Smeets, R., 2020. In: Investor Memory. 2020 AFA Annual Meeting Working Paper. https://papers.ssrn.com/sol3/papers.cfm?abstract_
id=3348315.
Han, Y., Zhou, G., Zhu, Y., 2016. Taming Momentum crashes: A simple Stop-Loss Strategy. Available at SSRN 2407199.
Kim, J.B., Zhang, L., 2016. CEO Overconfidence and Stock Price Crash Risk. Contemp. Account. Res. 33 (4), 1720–1749.
Jang, J., Kang, J., 2019. Probability of price crashes, rational speculative bubbles, and the cross-section of stock returns. J. Financ. Econ. 132 (1), 222–247.
Jegadeesh, N., Titman, S., 1993. Returns to buying winners and selling losers: implications for stock market efficiency. J. Finance 48 (1), 65–91.
Pelster, M., 2020. The gambler’s and hot-hand fallacies: empirical evidence from trading data. Econ. Lett. 187, 108887.
Scheinkman, J.A., Xiong, W., 2003. Overconfidence and speculative bubbles. J. Political Econ. 111 (6), 1183–1220.
Tversky, A., Kahneman, D., 1971. Belief in the law of small numbers. Psychol. Bull. 76 (2), 105–110.
Wilder Jr., J.W., 1978. New Concepts in Technical Trading Systems. Hunter Publishing Company, Greensboro, N.C.
7


---

## Raw Markitdown Extraction (full text)

Contents lists available at ScienceDirect

Finance Research Letters

journal homepage: www.elsevier.com/locate/frl

Crash-based quantitative trading strategies: Perspective of
behavioral finance

Yan Fang a, Jie Yuan a, J. Jimmy Yang b,*, Shangjun Ying a, 1
a Shanghai University of International Business and Economics
b Oregon State University, School of Accounting, Finance, and Information Systems, College of Business, 426 Austin Hall, Corvallis, OR,
97331United States

A R T I C L E  I N F O

A B S T R A C T

Inspired by the studies on stock market crashes, we use documented indicators from behavioral
finance to construct two quantitative trading strategies, i.e., Crash + Timing Strategy and Crash
+ Momentum-Reversal Strategy. Empirical analyses show that both strategies are effective and
robust. Behavioral factors can be  beneficial to investors  when they are  incorporated into  their
trading strategies.

JEL codes:
G1
C22

Keywords:
Behavioral finance
Crash factor
Market timing
Momentum-reversal strategy

1. Introduction

With the development of the behavioral finance, many scholars seek to explain the causes of stock market crashes using behavioral
theories. For instance, Avery and Zemsky (1998) point out that herding can lead to mispricing of assets and price bubbles. Kim et al.
(2016) suggest that firms with overconfident CEOs have higher stock price crash risk than other firms. Jang and Kang (2019) find that
price  crashes  could  be  caused  by  the  overpricing  driven  by  institutional  investors.  While  existing  studies  focus  on  exploring  the
influential factors that may lead to stock market crashes or predicting crash events in stock markets, we investigate how investors can
profit from those events.

Based on the aforementioned studies, a price crash can be caused by investors’ cognitive biases, such as herding bias and over-
confidence bias. Even though such cognitive biases can result in mispricing of securities, investors may theoretically expect excess
returns generated from a stock market crash as long as they rationally use certain behavioral deviations (Conrad et al., 2014; Jang and
Kang, 2019; Pelster, 2020). Therefore, it is highly possible to realize excess returns from a stock market crash.

During a stock market crash, stock prices typically move in a similar pattern. First, stock prices continue to rise for an extended
period until the bubble reaches its peak. Then, the bubble bursts and the market collapses with a sudden slump. If one wants to profit
from a stock market crash, the ability to select stocks and set the timing is critical. Accordingly, this paper adopts the behavioral
finance approach to propose two quantitative trading strategies based on a documented crash factor. One strategy is built on the crash
factor  and  market  timing  (i.e.,  Crash  + Timing  Strategy,  CTS),  while  the  other  is  constructed  by  applying  the  crash  factor  and
momentum-reversal strategy (i.e., Crash + Momentum-Reversal Strategy, CMRS). Our empirical analyses present stable returns and

* Corresponding author.

E-mail address: jimmy.yang@bus.oregonstate.edu (J.J. Yang).

1  Ying acknowledges financial support from the National Natural Science Foundation of China (71571116).

https://doi.org/10.1016/j.frl.2021.102185
Received 10 June 2020; Received in revised form 11 April 2021; Accepted 26 May 2021

FinanceResearchLetters45(2022)102185Availableonline30May20211544-6123/©2021ElsevierInc.Allrightsreserved.Y. Fang et al.

confirm the effectiveness and robustness of these market crash-based trading strategies.

2. Construction of trading strategies

2.1. Crash + timing strategy

According to the concept of mean reversion and the gambler’s fallacy (Tversky and Kahneman, 1971), stock prices can rebound
from the current crash and possibly experience another crash. Investors can generate profits by buying the dips during the crash and
selling the rips during the consequent rebound. We use the crash factor2 of Jang and Kang (2019) to screen stocks. Specifically, top 10%
stocks at the end of the previous month (t - 1) and the current month (t) are sorted out by the crash factor, and the top 10% stocks at the
end of the current month are regarded as stocks whose bubbles just burst. However, stocks selected solely based on crash factor could
again take a nosedive, so they cannot be directly put into quantitative investing.3  We consider appropriate timing for buying and
selling (i.e. market timing) by constructing the crash + timing strategy.

Due to the self-serving memory bias, investors generally repeat their previous behaviors for a risky stock position (G¨odker et al.,
2020). Therefore, we assume that before a stock market crashes, stocks will display characteristics similar to those perceived during the
last crash. The strength/weakness of buying and selling on long and short positions before and after a crash would be the best indicator
for market timing measurement. Consequently, we use the relative strength index (RSI) (Wilder, 1978) to identify the best times to buy
stocks, and apply a momentum factor, which is defined as the corresponding 5-day momentum at the peak from last month, to capture
the sell signals. During an observation period, we sell stocks when the momentum factor is similar to the peak momentum factor in the
previous month (t-1). The specific process of CTS is given in Algorithm 1.

2.2. Crash + momentum-reversal strategy

Momentum (Jegadeesh and Titman, 1993) and reversal are two well-known anomalies within the field of behavioral finance.
However, momentum strategies suffer from occasional large crashes (i.e. momentum crash) (Barroso and Santa-Clara, 2015; Daniel
and Moskowitz, 2016). During a momentum crash, momentum often tends to reverse. That is, the portfolio with the lowest momentum
becomes the best performer, and the momentum-reversal strategy realizes excess returns. Therefore, the momentum-reversal strategy
can considerably enhance investment efficiency.

The  second  strategy  is  to  combine  the  crash  factor  with  momentum-reversal  strategy.  We  first  add  the  crash  factor  to  the
momentum-reversal strategy, adopt a scoring method to construct a new comprehensive scoring factor, and use this new factor to
screen stocks. The stocks selected are featured with crashes in current month and expected to rebound later. That is, underperformed
stocks will likely outperform in the future (Bali et al., 2011). The process of CMRS is shown in Algorithm 2.

3. Empirical analyses

We perform empirical analyses using daily stock data from January 1, 2018 to December 31, 2018 and the Center for Research in
Security Prices (CRSP) value-weighted index as the market index. The sample includes all stocks with share codes 10 and 11 from
NYSE, AMEX and NASDAQ. All data are from the CRSP database. The data processing and calculation of crash factor are conducted
based on Jang and Kang (2019) (See Appendix 1).

3.1. Performance of CTS

To determine when to buy stocks, we choose the 14-day RSI4  within the oversold level as the buy limit price. An RSI below 30
(Wilder, 1978) would suggest a buy. Since a rebound may happen or continue beyond the current month, we use the next two months
as the observation period. In addition, we consider the following four methods to determine the sell threshold.

Method 1: Use RSI as the benchmark and the overbought level (RSI=70) as a threshold;
Method 2: Adopt the pre-set take profit level (i.e., 3%)5 as a threshold;
Method 3: Utilize the highest cumulative return over the last month captured by the momentum factor as a threshold;
Method 4: Consider the highest cumulative return over the last month captured by the momentum-information discreteness (Da
et al., 2014) as a threshold.

When adopting different methods to identify the best times to sell stocks, the stop-loss order is executed at 2.5% below the current
stock price (Han et al., 2016). Evaluating the effectiveness of a strategy is essential for an investment. We evaluate our strategies based

2  The crash factor is defined as the probability in the event of log returns lower than (cid:0) 70% over the next 12 months.
3  We do not discuss profit taking from short selling herein due to multiple short-selling bans in the world stock markets.
4  In order to choose the appropriate timeframe, a comparison is conducted among the respective performance of 14-day RSI, 24-day RSI and

market index, seeking the timeframe that outperforms the market index.

5  This is how to avoid emotional investing and selling into corrections within the field of behavioral finance.

FinanceResearchLetters45(2022)1021852Y. Fang et al.

Table 1
Performance of CTS with different methods in 2018.

Annualized return
Win rate

Method 1

(cid:0) 7.145%
30.113%

Method 2

3.296%
56.000%

Method 3

8.464%
37.113%

Method 4

5.515%
45.455%

Market Index

(cid:0) 10.017%
—

Note: The sell signal in Method 1 is 14-day RSI above 70. The sell signal in Method 2 is when the cumulative return is 3% above the purchase price.
The sell signals in Method 3 and Method 4 are the 5-day momentum value and the 5-day information discreteness that are equal to or above the
highest cumulative return over the last month, respectively.

Fig. 1. Performance of CMRS from quintile 1 to 5 for 2018, and the benchmark refers to the cumulative return of the market index.

Table 2
This table reports the annualized return, Sharp ratio, maximum drawdown and win rate on quintile portfolios sorted on the comprehensive scoring
factors. At the end of each month i, quintile portfolios are constructed by sorting stocks based on the comprehensive scoring factors. Column quintile 1
is for the highest comprehensive scoring factors quintile, while column quintile 5 shows the results for the lowest comprehensive factors quintile.
Column CRSP is for the performance of CRSP value-weighted index.

Panel 1: CMRS

Annualized return
Sharp ratio
Max. drawdown
Win rate

Quintile 1

(cid:0) 71.570%
(cid:0) 0.361775
(cid:0) 74.372%
13.878%

Panel 2: Momentum-reversal strategy

Annualized return
Sharp ratio
Max. drawdown
Win rate

Quintile 1

(cid:0) 80.798%
(cid:0) 0.501626
(cid:0) 82.073%
4.147%

Quintile 2

(cid:0) 36.926%
(cid:0) 0.151652
(cid:0) 45.374%
17.809%

Quintile 2

(cid:0) 41.240%
(cid:0) 0.183588
(cid:0) 48.318%
9.704%

Quintile 3

17.761%
0.064567
(cid:0) 21.431%
26.712%

Quintile 3

(cid:0) 7.102%
(cid:0) 0.022715
(cid:0) 24.038%
27.437%

Quintile 4

39.228%
0.138178
(cid:0) 14.036%
31.133%

Quintile 4

46.513%
0.156668
(cid:0) 13.444%
38.024%

Quintile 5

104.276%
0.328911
(cid:0) 9.426%
40.288%

Quintile 5

287.544%
0.517607
(cid:0) 7.187%
45.663%

CRSP

(cid:0) 6.012%
(cid:0) 0.018106
(cid:0) 20.960%
—

CRSP

(cid:0) 6.012%
(cid:0) 0.018106
(cid:0) 20.960%
—

on the annualized return and win rate, where the win rate refers to the ratio of the number of stocks with a positive return to the
number of stocks with non-zero returns. The results are shown in Table 1, where the assessments for Methods 1–4 are given in columns
2–5 and the results of market index is given in column 6.

In Table 1, we can see that 1) the market return in 2018 is negative and all four Methods outperform the market index; 2) the
annualized  return  in  Method  1  is  negative,  and  its  win  rate  is  the  lowest,  which  implies  that  the  market  timing  strategy  solely
depending on RSI is insufficient6; 3) Method 2 has the highest win rate, which indicates that the concept of take-profit can effectively
avoid making wrong investment decisions, and achieves good performance in quantitative investing; 4) the annualized returns in both
Methods 3 and 4 are positive and are substantially higher than the market return, which suggests that the momentum factor is a

6  In  addition  to  the  sell  threshold  of  RSI=70  suggested  by  Wilder  (1978),  we  also  use  RSI=80  as  the  sell  threshold  and  RSI=20  as  the  buy
threshold.  The  annualized  return  and  win  rate  of  Method  1  are  both  improved.  Specifically,  the  annualized  return  improves  from  (cid:0) 7.145%  to
7.465%, while the win rate increases from 30.113% to 34.408%.

FinanceResearchLetters45(2022)1021853Y. Fang et al.

Algorithm 1
The process for the CTS.

1  Start with time t,

a)  Pick up top 10% stocks based on crash factors at the end of month t and month t (cid:0) 1 and name them as Set 1 and Set 2, respectively;
b)  Remove stocks which appear on both month t (cid:0) 1 and month t from set 1, and use the remaining stocks to create a new portfolio.
c)  For stocks listed in our new portfolio,

i  we will buy when they reach the buy limit order during our observation period;
ii  we will sell when they reach the sell limit order or when the stop-loss price has been reached during our observation period; otherwise, we

will sell them at the end of the observation period.

2  2. Repeat Step 1.

Algorithm 2
The process of CMRS.

1  Start with time t,

a)  Buy the top 20% of stocks sorted by the new comprehensive factor at the end of month t;
b)  Hold stocks till the end of month t + 1 and sell them.

2  Repeat Step 1.

workable market timing indicator.

3.2. Performance of CMRS

In order to evaluate the performance of CMRS, all momentum factors and crash factors at the end of the month (t) are sorted in
ascending and descending order, respectively, and the comprehensive scoring factors are calculated according to the scoring method.7
The comprehensive scoring factors are sorted in descending order, and divided into five groups (Fama and French, 1993). We buy the
stocks in these five groups in month t and sell them at the end of the month (t + 1). Finally, we calculate the cumulative return in each
group. The results are shown in Fig. 1, where each curve represents the cumulative log return of a quintile regression group.

Despite of some fluctuations in the market of 2018, the cumulative return of each group shows a clear monotonic pattern. Among
them, the cumulative returns of quintiles 3 to 5 are all higher than the cumulative return of the market index, which indicates that the
trading strategy brings investors an excess return. We also compare and analyze the performance of both CMRS and the traditional
momentum-reversal strategy by using four indicators, i.e., annualized return, Sharpe ratio, maximum drawdown and win rate. The
results are presented in Table 2.

Both  strategies  show  strong  portfolio  performance.  All  indicators  in  quintiles  1–5  show  clear  monotonic  patterns.  CMRS  out-
performs the momentum-reversal strategy in quintiles 1–3. CMRS also outperforms the market index in quintiles 3–5. Although the
best performer Quintile 5 of CMRS falls behind the control group of the momentum-reversal strategy, this trading strategy is overall an
effective stock selection strategy.

Generally speaking, momentum crash is the most important factor that restricts application of the momentum strategy, and it can
sometimes cause the momentum factor to fail. Although the market fluctuated in 2018, there was no momentum crashes. That is the
main reason that we cannot find an obvious advantage of our proposed strategy in Table 2. To show that CMRS is effective during a
momentum crash, we examine the data from January 1, 2015 to December 31, 2015 (momentum crashes happened over this period) to
test the robustness of our trading strategies. The results show that both CTS and CMRS achieve favorable results, but the momentum-
reversal strategy generally fails. Please see Appendix 2 for details.

In addition, we examine an up market period from January 1, 2019 to July 31, 2019 to test whether our trading strategies would
work when the overall market return is positive. We find that the CTS strategy performs even better in the up market as annualized
returns and win rates in 2019 are higher than those in 2018 and 2015. Results from the CMRS strategy are similar to those in 2018.
Those results are not tabulated, but they are available on request.

4. Conclusion

Inspired by the latest studies on stock market crashes, this paper constructs two quantitative trading strategies and tests their
performances. Empirical analyses show that 1) the crash factor cannot directly function as a momentum factor, but when the crash
factor combines with an appropriate timing indicator, it could generate excess returns for investors; 2) the momentum factor is useful
in a market timing approach; 3) during a volatile market, a momentum-reversal strategy may fail while the CMRS proposed in this
paper is still effective.

Unlike previous studies, this article explores quantitative trading strategies from the perspective of behavioral finance, considering

7  The so-called scoring method is a method where stocks’  crash factors are ranked in ascending order, their momentum factors are ranked in

descending order, and the average of the two ranks is defined as the comprehensive scoring factor.

FinanceResearchLetters45(2022)1021854Y. Fang et al.

Table A.1
Performance of CTS with different methods in 2015.

Annualized return
Win rate

Method 1

(cid:0) 4.749%
28.049%

Method 2

9.497%
57.927%

Method 3

6.332%
46.154%

Method 4

0.712%
43.357%

Market Index

(cid:0) 6.989%
—

the effect from stock market crash, momentum factor, and RSI. This approach provides investors useful trading strategies to potentially
exploit behavioral bias in stock markets. However, there is room for future studies to further explore and extend these strategies. First,
more studies are needed in the application of those strategies. For instance, potential profits from long-short portfolios in the context of
the second strategy can be explored. Second, the construction of the two strategies only takes into consideration the behavior of short-
term investors. A long-term approach can provide a complete picture and strengthen our understanding of market crash-based trading
strategies. Third, we use the market index as the benchmark to measure performance of our trading strategies, but risk-adjusted returns
can be used to appropriately account for risks involved in the trading strategies.

Declarations of Competing Interest

None

Appendix 1. Crash factor

Crash factor is defined as:

Crash factor =

exp(C)
1 + exp(C) + exp(J)

where

C = α0 + α1RM12 + α2EXRET12 + α3TVOL + α4TSKEW + α5SIZE + α6DTURN + α7AGE + α8TANG + α9SALESG

and

J = β0 + β1RM12 + β2EXRET12 + β3TVOL + β4TSKEW + β5SIZE + β6DTURN + β7AGE + β8TANG + β9SALESG

Variable RM12 is the log return of the CRSP value-weighted index over the past 12 months; EXRET12 is the log return over the past
12 months in excess of RM12; TVOL and TSKEW are the standard deviation and skewness, respectively, of daily log returns over the
past 6 months; SIZE is the log of market capitalization; DTURN is the detrended turnover, defined by the six-month average of monthly
share turnover minus the prior 18-month average; AGE is the number of years since the firm’s first appearance on the CRSP monthly
stock file; TANG is tangible assets divided by total asset; and SALESG is the sales growth over the prior year. Parameters αi and βi (i = 0,
1, ⋅⋅⋅, 9) are from Jang and Kang (2019).

Appendix 2. Robustness testing for quantitative trading strategies

What  does  the  advantage  of  the  proposed  trading  strategies  lie  in,  chance  or  strength?  We  further  test  their  reliability  and
robustness. Data from CRSP database over the period from January 1, 2015 to December 31, 2015 are used for the robustness testing.

1. Robustness Test for CTS

To identify the best times for selling stocks, the 24-day RSI within the overbought level is used as the buy limit price, and the stop-
loss order is set to be 2.5% below the current stock price. The strategy’s performances for four methods defined in Section 3 are given in
Table A.1.

As we can see, 1) over the period, the market return is negative and all the Methods 1–4 outperform the market index; 2) the
annualized return in Method 1 is negative and its win rate is the lowest, indicating that the market timing strategy solely depending on
RSI is insufficient; 3) Method 2 has the highest annualized return and win rate, implying the take-profit strategy can effectively avoid

FinanceResearchLetters45(2022)1021855Y. Fang et al.

Fig. A.1. Performance of CMRS from quintile 1 to 5 for 2015 CMRS. Note: Benchmark refers to the cumulative return of the market index. The
vertical line represents the date of July 31, 2015.

Table A.2
Performance of CMRS and Momentum-reversal Strategy in 2015.

Panel 1: CMRS

Annualized return
Sharp ratio
Max. drawdown
Win rate

Quintile 1

(cid:0) 23.039%
(cid:0) 0.073
(cid:0) 36.575%
45.117%

Panel 2: Momentum-reversal strategy

Annualized return
Sharp ratio
Max. drawdown
Win rate

Quintile 1

(cid:0) 20.144%
(cid:0) 0.064
(cid:0) 34.123%
44.826%

Quintile 2

(cid:0) 20.981%
(cid:0) 0.082
(cid:0) 31.096%
22.472%

Quintile 2

(cid:0) 17.747%
(cid:0) 0.070
(cid:0) 27.822%
22.170%

Quintile 3

(cid:0) 19.836%
(cid:0) 0.078
(cid:0) 29.028%
22.532%

Quintile 3

(cid:0) 15.981%
(cid:0) 0.065
(cid:0) 24.707%
23.241%

Quintile 4

(cid:0) 12.608%
(cid:0) 0.052
(cid:0) 21.128%
23.088%

Quintile 4

(cid:0) 15.022%
(cid:0) 0.064
(cid:0) 22.425%
22.951%

Quintile 5

(cid:0) 13.447%
(cid:0) 0.062
(cid:0) 19.734%
22.298%

Quintile 5

(cid:0) 21.312%
(cid:0) 0.089
(cid:0) 27.636%
22.182%

RM

(cid:0) 7.620%
(cid:0) 0.027
(cid:0) 17.486%
—

RM

(cid:0) 7.620%
(cid:0) 0.027
(cid:0) 17.486%
—

making wrong investment decisions; 4) the results in both Methods 3 and 4 considerably outperform the market index, which indicates
that the momentum factor is a workable market timing indicator. In sum, this trading strategy seems robust.

2. Robustness Test for CMRS

Data in 2015 are divided into groups, since there exists a crash. The cumulative return of each group is shown in Fig. A.1.
In a relatively stable market (before August 2015), the cumulative return of each group has a clear monotonic pattern. The cu-
mulative returns of quintiles 1 to 3 are higher than the cumulative return of the market index.8 When the market experiences fluc-
tuation (since August 2015), the cumulative return of each group shows no clear monotonic pattern. The performances of both CMRS
and the momentum-reversal strategy are shown in Table A.2.

Although all groups underperform the market index, the annualized returns and maximum drawdowns of CMRS maintain their
monotonicity. In this case, investors can long the best performer and short the worst performer so as to achieve excess returns. All
indicators of the momentum-reversal strategy fail to maintain their monotonic patterns, indicating that this strategy is ineffective. The
CMRS contains certain robustness as it achieves favorable results during a momentum crash.

8  The empirical results observed between Jan. 2015 and Aug. 2015 are opposite to those observed in 2018 although both periods are relatively
stable. A potential reason is the existence of bubbles in 2015. The theory of rational bubbles (Blanchard and Watson, 1982) implies that investors’
irrational  behavior  and  expectations  induce  the  stock  price  to  deviate  from  its  true  value,  giving  rise  to  a  bubble.  Once  the  bubble  bursts,  it
automatically  leads  to  a  crash  in  the  market.  Scheinkman  and  Xiong  (2003)  also  point  out  that  heterogeneous  beliefs  generated  by  investors’
overconfidence bring bubbles to the market and then increase the possibility of its crash. Heterogeneous beliefs from overconfident investors would
result in prices that deviate from the true value, and then lead to a momentum reversal effect in the market. As a result, the order of cumulative
returns from quintiles 1 to 5 in Figure A.1 before the crash in August is different from that in Figure 1.

FinanceResearchLetters45(2022)1021856Y. Fang et al.

Reference

Avery, C., Zemsky, P., 1998. Multidimensional uncertainty and herd behavior in financial markets. Am. Econ. Rev. 724–748.
Bali, T.G., Cakici, N., Whitelaw, R.F., 2011. Maxing out: stocks as lotteries and the cross-section of expected returns. J. Financ. Econ. 99 (2), 427–446.
Barroso, P., Santa-Clara, P., 2015. Momentum has its moments. J. Financ. Econ. 116 (1), 111–120.
Blanchard, O.J., Watson, M.W., 1982. Bubbles, Rational Expectations and Financial Markets. NBER working paper, (w0945).
Conrad, J., Kapadia, N., Xing, Y., 2014. Death and jackpot: why do individual investors hold overpriced stocks? J. Financ. Econ. 113 (3), 455–475.
Daniel, K., Moskowitz, T.J., 2016. Momentum crashes. J. Financ. Econ. 122 (2), 221–247.
Da, Z., Gurun, U.G., Warachka, M., 2014. Frog in the pan: continuous information and momentum. Rev. Financ. Stud. 27 (7), 2171–2218.
Fama, E.F., French, K.R., 1993. Common risk factors in the returns on stocks and bonds. J. Financ. Econ. 33, 3–56.
G¨odker, K., Peiran Jiao, R., Smeets, R., 2020. In: Investor Memory. 2020 AFA Annual Meeting Working Paper. https://papers.ssrn.com/sol3/papers.cfm?abstract_

id=3348315.

Han, Y., Zhou, G., Zhu, Y., 2016. Taming Momentum crashes: A simple Stop-Loss Strategy. Available at SSRN 2407199.
Kim, J.B., Zhang, L., 2016. CEO Overconfidence and Stock Price Crash Risk. Contemp. Account. Res. 33 (4), 1720–1749.
Jang, J., Kang, J., 2019. Probability of price crashes, rational speculative bubbles, and the cross-section of stock returns. J. Financ. Econ. 132 (1), 222–247.
Jegadeesh, N., Titman, S., 1993. Returns to buying winners and selling losers: implications for stock market efficiency. J. Finance 48 (1), 65–91.
Pelster, M., 2020. The gambler’s and hot-hand fallacies: empirical evidence from trading data. Econ. Lett. 187, 108887.
Scheinkman, J.A., Xiong, W., 2003. Overconfidence and speculative bubbles. J. Political Econ. 111 (6), 1183–1220.
Tversky, A., Kahneman, D., 1971. Belief in the law of small numbers. Psychol. Bull. 76 (2), 105–110.
Wilder Jr., J.W., 1978. New Concepts in Technical Trading Systems. Hunter Publishing Company, Greensboro, N.C.

FinanceResearchLetters45(2022)1021857
