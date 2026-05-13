<!-- Page 1 -->

Risk Management for Event-Driven Funds
Philippe Jorion
Paul Merage School of Business, University of California at Irvine, and
Pacific Alternative Asset Management Company (PAAMCO)
This version: September 2007
DRAFT
The author wishes to thank Jim Berens, Jane Buchan, Mayer Cherem, Rob Dudley, Judy
Posnikoff, and Sydney Zhang for useful discussions on this topic.
Correspondence can be addressed to:
Philippe Jorion
Paul Merage School of Business PAAMCO
University of California at Irvine 19540 Jamboree Road
Irvine, CA 92697-3125 Irvine, CA 92612
Phone: (949) 824-5245 Phone: (949) 261-4964
E-mail: pjorion@uci.edu E-mail: pjorion@paamco.com


<!-- Page 2 -->

Risk Management for Event-Driven Funds
Abstract
Many portfolio strategies are “event-driven,” i.e., try to benefit from price
movements caused by corporate events such as restructurings, bankruptcies, mergers,
acquisitions, or other special situations. Such trading strategies involve payoffs that
have discontinuous and skewed distributions that cannot be measured well with
conventional risk methods. This paper develops methods to measure the forward-looking
risk of portfolios exposed to such discrete events, based on current positions. When
events are independent, the portfolio follows a binomial distribution. This approach is
extended to the more realistic case where events are not independent. For mergers and
acquisitions, empirical estimates of deal break correlations are positive but relatively
low, which implies that most event risk is idiosyncratic and diversifiable. This
methodology can be used to evaluate the risk and return of different portfolio structures.
JEL Classifications: G11 (portfolio choice), G23 (private financial institutions), G32 (financial risk
management)
Keywords: merger arbitrage, risk management, hedge funds, value at risk


<!-- Page 3 -->

P. Jorion - Risk Management for Event-Driven Funds 1
Introduction
Many hedge fund strategies are “event-driven,” i.e., try to benefit from price movements
caused by corporate events such as restructurings, bankruptcies, mergers, acquisitions, or
other special situations. Such trading strategies involve payoffs that have discontinuous
distributions. Either the event happens or not, which is a binary distribution. These
distributions pose a particular challenge to risk management because of their
discontinuous nature, as well as the asymmetry of their payoffs. In addition, the history
of price movements may not be relevant for measuring risk.
Consider, for example, a portfolio manager positioned to take advantage of a
proposed merger and acquisition (M&A). The acquirer makes an offer that exceeds the
current market price, by an amount known as the “merger premium.” Offers can take the
form of cash or stock of the bidding company. Right after the deal announcement, the
target company’s price jumps up, in proportion to this premium. There is still
uncertainty, however, as to the success of the deal, which explains why the price of the
target company generally trades at a discount to the offer price, known as the “arbitrage
spread.” The typical position for risk arbitrageurs is to buy the target company and, in
the case of stock deals, short the acquirer. If a deal succeeds, the position generally
creates a modest gain.1 In contrast, in the case of deal failure, the target company’s price
can fall back to its pre-announcement level, which is a big loss.2 Thus the position is
exposed to an asymmetric “deal risk.” Such risk, however, cannot be measured from
1 Success, however, does not simply mean that the original bidder takes control of the target. Other bidders
can intervene in the meantime. Success can be defined as a situation where the price of the target goes to or
above the offer price due to any acquisition. Equivalently, failure is the case when the target remains
independent.
2 The drop can be more than the initial run-up in price if the deal reveals negative information about the
target. The payoff on the strategy also depends on the price movement in the acquirer’s stock in the case of
positions that involve both the target and the acquirer.


<!-- Page 4 -->

P. Jorion - Risk Management for Event-Driven Funds 2
historical data. The goal of this paper is to develop methods to measure the risk of
portfolios exposed to such discrete events.
Current industry methods for managing the risk of such positions are rather
elementary. Moore et al. (2006) describe a recent survey of risk management methods
used by 21 risk arbitrageurs in mergers and acquisitions. The primary method used to
control event risk is position limits. The first type is a limit on the fraction of the
portfolio invested in any deal, which is typically 10% of the portfolio value. This implies
a minimum of 10 positions. In the survey sample, the number of deals varies between 20
and 100, with a median of 30. The second type is a limit on the maximum loss of any
position, e.g. 5% of the portfolio value if the deal fails. This loss represents a fall in the
stock price back to its pre-announcement price. These two types of constraints are
largely overlapping.
Such single name limits help to diversify the portfolio but are totally ad hoc. This
is because, as revealed in the Moore survey, M&A arbitrageurs do not measure their total
portfolio risk. Portfolio risk can be summarized, however, with a single measure of
downside risk such as Value at Risk (VAR).3
The construction of portfolio risk measures, however, cannot be based on
traditional risk models, which do not take into account the uncertainty generated by the
currently unfolding events. Indeed, Schachter (2006) states that VAR “is not well suited”
in these cases. This is certainly the case for conventional VAR measures based on recent
historical data. Nevertheless, we show that it is perfectly feasible to construct a forward-
3 See Jorion (2006) for an exposition of VAR.


<!-- Page 5 -->

P. Jorion - Risk Management for Event-Driven Funds 3
looking distribution of portfolio profits and losses based on current positions and to
summarize it using a VAR measure.
Such measure would allow risk arbitrageurs to gauge optimum diversification
levels, which depend on the number of deals, the expected profit on each deal, and the
contribution of each deal to portfolio risk. Also, VAR can used to determine the optimal
amount of economic capital required to support the portfolio, or equivalently, the optimal
amount of leverage. Moore et al. (2006) report that about two-thirds of arbitrageurs use
leverage.
To our knowledge, this paper is the first to provide quantitative tools to measure
the portfolio risk of event-driven funds. The methodology proposed here expands on
methods developed for credit portfolios, which also involve discrete events, i.e., defaults.
In the case of cash deals, a position in the target firm is similar to an unsecured bond
issued by the acquirer that matures on the closing date. The methodology further
demonstrates the benefits of using position information to manage risks.
The paper proceeds in two steps. Section I describes drawbacks of conventional
risk measures for event-driven funds. Section II then derives a new methodology for
measuring portfolio risk when events are independent. Realistically, however, there may
be dependencies across events. Section III provides some empirical evidence on
probabilities and correlations of deal breaks. Section IV then analyzes portfolio risk with
correlated events and shows how correlations affect risk measures for event-driven
portfolios. Section V draws implications for assessing the amount of economic capital
required to support the portfolio as well as acceptable levels of leverage. Finally, Section
VI contains some conclusions.


<!-- Page 6 -->

P. Jorion - Risk Management for Event-Driven Funds 4
I. Drawbacks of Conventional Risk Measures
Typical risk measures are returns-based. One example is the volatility of
historical returns for the fund over a recent window. This approach is simple but has
severe shortcomings. It is unable to react quickly to changes in the portfolio composition
or in the trading strategy. In addition, it makes it difficult to understand the structural
drivers of risk and hence to manage the risk profile of the portfolio.
More recently, position-based risk measures such as VAR have become widely
used. Conventional VAR measures combine current position information with the recent
history of risk factors, assuming that this history can be applied to the future. Current
positions are taken as fixed over the risk horizon.
This approach could be defined as “conventional position-based.” Generally, this
is a vast improvement over returns-based methods because it can handle changes in
portfolio composition and can be used to manage portfolio risk.
In the case of event-driven funds, however, the conventional application of
position-based VAR methods can be misleading for a number of reasons. First, the event
itself changes the nature of the stochastic process, from a random walk to a discrete-
outcome process. Thus, historical data cannot be used in the usual fashion. As an
example, consider the data provided by Mitchell and Pulvino (2001). They report that,
over the period 1963 to 1998, a value-weighted portfolio of M&A deals had an annual
realized volatility of 9.3%. This number represents the actual risk of a portfolio invested
in an average of 31 M&A positions taken after the announcement. They do not provide
the volatility of the same portfolio before the announcement, but surely this should be
less than that of a value-weighted market index because the latter is better diversified.


<!-- Page 7 -->

P. Jorion - Risk Management for Event-Driven Funds 5
Over the same period, this index had a volatility of 15.1%. This number can be taken as
a lower limit for the position-based risk of the M&A portfolio. In this case, the actual
risk is (15.1-9.3)/15.1=38% lower than the conventional position-based risk, which is
substantial.
Another problem is that the recent history includes the announcement of the event
itself. This creates a large jump in the price of the target, which artificially increases the
risk measure. The shorter is the window, the larger is the effect. As an example, take an
unleveraged M&A hedge fund with more than 150 positions. From May 2005 to April
2007, this fund had a return-based volatility of 2.2% pa. Conventional risk forecasts
were also measured for this fund due to the availability of position data, applied each
month to a 4-year window. Over the same period, these monthly volatility forecasts
averaged to 9.8%. Here again, the bias is substantial. Actual risk is about (9.8-
2.2)/9.8=78% less than the conventional position-based risk measure.
Perhaps simple rules of thumb could be developed for correcting conventional
position-based risk measures. Such adjustments cannot be universal, however. This is
because the relationship between a conventional position-based risk measure and its
actual counterpart depends on the structure of the portfolio, which involves the number of
positions, the distribution of deal payoffs, as well as historical risk measures. Note, for
instance, that the actual risk of the first M&A portfolio was 9.3%, which is much higher
than the risk of the second M&A portfolio, which was 2.2%. This is because the first
portfolio was much less diversified, with 31 positions against 150 for the second.


<!-- Page 8 -->

P. Jorion - Risk Management for Event-Driven Funds 6
Instead, position-based information can be used directly to construct forward-
looking measures of risk that account for the discrete-outcome processes that characterize
event-driven funds. This is the main purpose of this paper.
II. Measuring Risk from Position Information
Suppose we are investing in a stock that is currently in play as a takeover target at
a fixed offer price. The distribution of payoffs is characterized by two states of the
world. Table 1 gives an example. We evaluate a cash deal where the stock of the target
currently trades at $100, post-announcement. We ignore directional market risk and
focus on event risk only. The event indicator is b. In the case of failure or deal break,
b=1, with probability p=0.15; the dollar return R is an absolute loss AL of - $15. In case
of success, b=0, the absolute profit is AP=+$5.
Table 1: Single-Deal Distribution
Failure (b=1) Success (b=0)
Probability 15% 85%
Payoff (R) - $15 +$5
This example assumes a probability of failure of E[b] = p = 15%, which is typical
of empirical studies of M&A.4 More generally, the portfolio manager is supposed to
assess the probabilities and payoffs of each deal and to add value through skillful
selection of deals. In this example, the expected profit is positive, at E[R] = 15%×(- $15)
+85%× ($5) = $2.00. For simplicity, this is assumed to be over the risk-free rate. The
4 See for instance Baker and Savasoglu (2002).


<!-- Page 9 -->

P. Jorion - Risk Management for Event-Driven Funds 7
volatility for this Bernoulli distribution is s [b] = p(1- p). The distribution, however,
is strongly skewed in the direction of losses.
Here, the horizon is assumed to be the average time needed for the resolution of a
deal. Mitchell and Pulvino (2001) indicate that the average horizon of a deal is 59
trading days, or about three calendar months. In practice, this horizon is uncertain. The
distribution of payoffs could be extended to include an intermediate case where the deal
is still pending. Extrapolating the quarterly horizon to a year, the expected rate of return
is ($2· 4)/$100 = 8 percent per annum.5
Note that this example is calibrated to be realistic in terms of expected returns:
Mitchell and Pulvino (2001) find that simulated excess returns from risk arbitrage
strategies average 9.25% per annum, which is equivalent to the 2% per quarter assumed
here.6 These numbers represent pure alpha because they adjust for the risk-free rate and
the market beta. The methodology presented here, however, does not depend on these
numbers.
The next issue is the aggregation at the portfolio level. This type of problem is
similar to building a distribution of portfolio credit losses. To simplify, define Las a
positive number representing the loss relative to the success case, e.g., $20 for Table 1.
Assume first a homogeneous portfolio: All deals have equal size L and equal failure
probability p. If the events are independent, it is straightforward to build this distribution.
5 A variety of explanations have been advanced for this “risk arbitrage” premium. One is that risk
arbitrageurs need compensation for bearing event risk. This information-rich environment requires
specialized expertise, e.g. trading and legal, which needs compensation. Another is that some investors
may be forced to sell because of investment guidelines that would be breached after a takeover. For
instance, a minimum level of portfolio yield could be breached if the merged company pays no dividend.
Finally, these strategies have non-linear exposures to the market, because the break probability increases
when the market falls. This is akin to a short put position, which should command a risk premium.
6 After costs, however, they report an average return of about 3.5%.


<!-- Page 10 -->

P. Jorion - Risk Management for Event-Driven Funds 8
The total number of deals is N. The total number of failures k then follows the binomial
distribution f:
 N 
f (k ; N , p ) =   p k (1 - p ) N - k (1)
 k 
The distribution of total portfolio losses TL = k · L is given directly by Equation
(1). This is also a binomial distribution, which can be summarized by a dispersion
measure, such as VAR, or the lowest quantile at a specified confidence level.
We can now explore the effect of changing the number of deals N on the shape of
the distribution. To maintain comparability across portfolios, the total dollar exposure is
kept fixed, say at $100. Hence, the size of each deal L is inversely related to N across
portfolios. We characterize the distribution by its VAR, say at the 95 percent confidence
level. Because the distribution is discrete, we report the loss associated with a confidence
level at least equal to 95 percent. Table 2 reports the VAR along with the associated
actual confidence level and number of failures. VAR is measured as the negative of the
dollar loss over the period, and is generally a positive number. The distribution of losses
is described in Figure 1 for N=5, 10, and 100.
Table 2: Distribution of Independent Deals (Constant Exposure of $100)
VAR(‡ 95%) VAR(‡ 99.9%)
Number Actual Number Actual Number
of Expected Confidence of Confidence of
Deals Profit VAR L e v el Failures VAR Level Failures
1 +$2.00 $15.00 100.0% 1 $15.00 100.0% 1
5 +$2.00 $3.00 97.3% 2 $11.00 99.99% 4
10 +$2.00 $1.00 95.0% 3 $7.00 99.99% 6
20 +$2.00 $1.00 97.8% 6 $4.00 99.98% 9
30 +$2.00 $0.33 97.2% 8 $2.33 99.92% 11
40 +$2.00 $0.00 97.1% 10 $2.00 99.96% 14
50 +$2.00 - $0.20 97.0% 12 $1.40 99.93% 16
100 +$2.00 - $0.80 96.1% 21 $0.40 99.94% 27


<!-- Page 11 -->

P. Jorion - Risk Management for Event-Driven Funds 9
Table 2 shows that the expected dollar profit is independent of the number of
deals, which simply reflects the fixed total exposure of the portfolio. VAR, however,
does change with the number of deals. With one deal only, the minimum 95% VAR
implies a loss of $15, which actually corresponds to a 100 percent confidence level, or
one failure. With 10 independent deals, the 95% VAR goes down to $1.00. With 40
deals, the 95% VAR is zero. In other words, there is a low probability (less than 5%) of
losing any money on this portfolio. This is because the expected profit is positive and the
dispersion of payoffs shrinks rapidly as N increases. As the number of deals further
increases to 100, VAR becomes - $0.80, which means that there is a low probability of
making a profit less than $0.80. Thus, increasing the number of independent deals
shrinks the spread of the distribution rather rapidly, as illustrated in Figure 1. As N
increases, the binomial distribution converges to a Poisson distribution.
It should also be noted that, because of the discrete nature of outcomes, VAR does
not always change smoothly when changing parameters. Consider for instance VAR at
the minimum 95 percent confidence level. This stays at $1.00 for N=10 and 20. For
N=10, the actual confidence level is just at its required minimum of 95.0%. At N=20, the
actual confidence level increases to 97.8%. This corresponds to 6 deal failures; the
confidence level for 5 deal failures is still below 95%. As N increases further, VAR
eventually drops, but in discrete steps. Thus VAR is not a smooth function.


<!-- Page 12 -->

P. Jorion - Risk Management for Event-Driven Funds 10
Fig. 1. Distribution of Portfolio Losses with Independence
50%
40%
30%
20%
10%
0%
0.5$- 0.4$- 0.3$- 0.2$- 0.1$- 0.0$ 0.1$ 0.2$ 0.3$ 0.4$ 0.5$ 0.6$ 0.7$ 0.8$ 0.9$ 0.01$ 0.11$ 0.21$ 0.31$ 0.41$ 0.51$
Probability
N=5
N=10
N=100
Loss
This distribution could be used to infer the maximum amount of leverage that the
fund could safely take. Assuming the fund wants to maintain a single-A credit rating, the
confidence level can be taken as one minus the default probability over a year. Credit
rating agencies report a historical default rate of approximately 0.1 percent for A-rated
companies. This implies a confidence level of 99.9 percent.7 This VAR measure, which
can be interpreted as “economic capital,” is listed in the last column of Table 2.
For one deal, the 99.9% VAR is $15. If the notional value of the position is $100,
the portfolio can support a maximum leverage of $100/$15, or 6.7. For 10 deals,
economic capital shrinks to $7.0. Thus leverage could go to $100/$7.0, or 14.3. This
7 Under the newly established Basel II rules, commercial banks need to maintain regulatory capital in
excess of their 99.9 percent credit VAR over a horizon of one year. In our case, the horizon is one quarter,
so that the annual default rate is even lower than that for A-rated credits.


<!-- Page 13 -->

P. Jorion - Risk Management for Event-Driven Funds 11
would generate a rate of return on economic capital of $2.0/$7.0, or 28.6% per quarter.8
With 100 deals, economic capital is $0.4. Because this number is so low, leverage can be
even higher, generating a rate of return of 500%. Of course, this very high number
cannot be realistic because it assumes independent deals. The effect of dependencies is
examined next.
8 Normally, financing costs should be taken into account. The estimate of 2% expected return, however, is
already in excess of the risk-free rate, so there is no need for further adjustment. The only portion of
financing costs that is ignored is the spread between the borrowing and deposit rate.


<!-- Page 14 -->

P. Jorion - Risk Management for Event-Driven Funds 12
III. Empirical Estimation of Event Correlations
In practice, the events in the portfolio may not be independent. A large fall in the
market could lead to many deals breaking during the same year. Mitchell and Pulvino
(2001), for example, show that a 20 percent decrease in the market return increases the
probability of failure by about 9 percent. This contemporaneous dependency on the
market is akin to positive correlations across events.
This section provides empirical estimates of the deal-break probability and the
average break correlation. This study considers a large sample of 1,765 M&A deals
involving at least one North American company and concluded between 1997 and 2006.
This sample only includes deals with an announced total value over $500 million. The
data were collected from Bloomberg and are summarized in Table 3. Deals are tabulated
by their outcome every quarter.
Table 3: Description of M&A Deals: 1997-2006
Number of
Deals Average Value Mean
Completed of Deals ($ Completion
per Quarter million) Days for Deals
Failure 5.3 $5,377 138
Success 38.8 $3,385 137
Total 44.1
In this 40-quarter period, we observe 212 deal breaks and 1,553 completed deals,
which gives a 12% failure rate. The average number of deals completed in a quarter is
44.1, with a mean completion time of 138 calendar days, or about 4.5 months.
To estimate correlations across deal failures, we use a standard methodology
developed for credit events.9 Initially, each deal is assumed to have the same
9 See for example Bahar and Regal (2001) and de Servigny and Renault (2002).


<!-- Page 15 -->

P. Jorion - Risk Management for Event-Driven Funds 13
unconditional probability of failure p, and all pairwise correlation coefficients r are
assumed equal. This is extended later.
Suppose that we observe a number F of failures during a period t, out of a total of
t
N observations. The probability of failure is then estimated by
t
F
p = E[b]= ∑T w t (2)
t=1 t N
t
where w is the weight assigned to each observation, here taken as (1/T). The probability
t
p of a joint deal break can be estimated from the ratio of the total number of pairs
12
breaking during the period, or F(F- 1)/2, to the total number of pairs combinations, or
t t
N(N- 1)/2. De Servigny and Renault (2002) show via simulations that small-sample
t t
properties of the joint default probability estimator are slightly better for
(F )2
p = E[bb ]= ∑T w t (3)
12 1 2 t=1 t (N )2
t
The deal break correlation can then be constructed from the covariance between the two
events
Cov(bb ) = r · s (b)s (b) = E[b - p][b - p]= E[bb ]- p2 = p - p2 (4)
1 2 1 2 1 2 12
Replacing s (b) by the standard deviation of a Bernoulli variable, the correlation can be
written as
p - p· p
r = 12 (5)
p(1- p)· p(1- p)
Note that the deal break correlation is also related to the variance of the time series of
fraction of breaks during each period. Define this fraction as x=F/N. If x fluctuates
t t t t
sharply over time, this implies clustering of failures, or positive correlation. Using


<!-- Page 16 -->

P. Jorion - Risk Management for Event-Driven Funds 14
Equations (2), (3) and (4), and ignoring the degree of freedom adjustment, the variance of
x can be written as
1 1
s 2 = ∑T (x - x)2 = ( ∑T x2)- x2 = p - p2 = r · p(1- p) (6)
T t=1 t T t=1 t 12
So, a high variance is equivalent to high failure correlation.
Figure 2 plots the quarterly fraction of deals failing. Using Equation (2), the
average break probability is estimated to be 12%. The figure, however, reveals some
variation in the fraction of deals failing over time. This reached a peak of 27% in the first
quarter of 2003. The fraction was zero for two quarters (the first quarters of 1997 and
2004.) This variation can be interpreted in terms of clustering of failures, or correlations
across deal breaks.
Fig. 2. Fraction of Deals Failing: 1997-2006, Quarterly
30%
25%
20%
15%
10%
5%
0%
1Q7991 1Q8991 1Q9991 1Q0002 1Q1002 1Q2002 1Q3002 1Q4002 1Q5002 1Q6002


<!-- Page 17 -->

P. Jorion - Risk Management for Event-Driven Funds 15
Using Equation (3) to estimate p and Equation (5), the break correlation is
12
estimated at 0.03. The failure probability is 0.12. To assess the reliability of these
estimates, we simulate their distribution by bootstrapping the sample of 40 quarterly
fractions. Based on 1000 replications, the 95% significance interval for the break
probability is (0.102, 0.137), and that for the break correlation is (0.017, 0.045).
Therefore, the break correlation is positive and significantly different from zero. This
justifies the use of an average break correlation of 0.03 in the following section.
More generally, the portfolio manager could model the probability of failure as a
function of a number of deal variables.10 Here, we fit the aggregate fraction of breaks
using stock market returns and bond yields as independent variables. We can distinguish
between “predictive regressions,” which model predictable time-variation in the failure
probability, and “explanatory regressions,” which model correlations through the
influence of common risk factors. Results are presented in Table 4.
The first column in Table 4 shows a negative coefficient on the stock market
return during the same quarter. This means that the fraction of breaks increases when the
stock market falls during the same quarter. Splitting up market returns into positive and
negative values shows that the effect is asymmetric. Positive returns are basically
uncorrelated with deal breaks. Large negative returns, however, are significantly
associated with a greater proportion of deal breaks. This is as expected: A large drop in
the stock market reduces the value of the target and makes it more likely for the acquirer
to walk away from the deal.
10 Mitchell and Pulvino (2001) estimate the probability of a deal breaking from individual deal
characteristics and general market factors. For example, “hostility” is the most important indicator,
increasing the probability of failure by 12.8 percent. They also find that the S&P beta on the downside is
greater than on the upside.


<!-- Page 18 -->

P. Jorion - Risk Management for Event-Driven Funds 16
The - 0.36 coefficient means that when the stock market drops by 20%, the break
probability increases by 7%. This conditional dependence on the market explains why
we observe positive correlations across deal breaks.11 After conditioning on this common
factor, the break correlation becomes unimportant. In practice, using a simple measure of
unconditional positive correlation is sufficient because market downturns cannot be
predicted.
Table 4: Explaining and Predicting Fraction of Breaks for M&A Deals
This table presents regressions of the fraction of breaks on various contemporaneous and predictive variables. RM is the total
return on the S&P500 index; HY is the yield to maturity on the Lehman high-yield corporate bond index. Sample period is
1997-2006, or 40 quarters. Standard errors are below the estimated coefficients. The superscripts **, and * indicate
significance at two-tailed 5% and 10% levels, respectively.
Independent Explanatory Explanatory Predictive Predictive Predictive
Variables Model 1 Model 2 Model 1 Model 2 Model 3
Constant 0.12 0.10 0.13 0.02 0.05
(0.01) (0.01) (0.01) (0.04) (0.04)
RM(t) -0.08
(0.10)
RM(t)>0 0.13
(0.18)
RM(t)<0 -0.36*
(0.22)
RM(t-1) -0.26** -0.19*
(0.10) (0.10)
HY(t-1) 1.00** 0.68
(0.40) (0.42)
R-square (%) 1.67 7.14 14.30 15.54 21.04
Focusing next on predictive regressions, the coefficient on the previous stock
market return is negative and strongly significant, indicating that falling stock prices
increase the fraction of breaks the next quarter. The explanation is the same as before. In
addition, the coefficient on the bond yield at the start of the quarter is positive and highly
significant. As expected, higher financing costs lead to a greater fraction of deals failing.
11 Note that a similar factor structure is used for credit portfolio models, where correlations across defaults
are induced by a 1-factor structure.


<!-- Page 19 -->

P. Jorion - Risk Management for Event-Driven Funds 17
This leads to a predictable break probability that varies between 9% and 18% over this
period.
It should be noted that this calibration exercise is limited to large North American
deals. To lower the failure correlation, a portfolio manager could include in the portfolio
M&A deals from smaller firms and from outside North America. More generally, the
methodology described next can be used to illustrate how to assess the benefits from
searching for lower correlations.


<!-- Page 20 -->

P. Jorion - Risk Management for Event-Driven Funds 18
IV. Building Portfolio Distributions with Correlated Events
Section I has shown that the dispersion of the distribution of losses shrinks rapidly
as the number of independent deals increases. In practice, however, positive correlations
temper this risk reduction effect. The question is how to adjust the measure of portfolio
risk. Because M&A events are similar in structure to credit events, the methodology
behind recent credit risk models can be used to measure event risk.
Various methods have been proposed to measure the risk of credit portfolios and
can be applied to this case. One approach is the “Binomial Expansion Technique” (BET)
advocated by Moody’s (1996) to evaluate Collateralized Debt Obligations (CDOs). This
technique approximates the actual distribution of N correlated events with a simpler
binomial distribution for another number D of uncorrelated events. D is called the
“diversity” score. It is computed for each deal i, from event probabilities (p) deal size
(L), and bivariate correlations (r ):
( )( )
∑N ∑N -
p L (1 p )L
= i=1 i i i=1 i i
D
∑N ∑N r - - (7)
p (1 p )p (1 p ) L L
i=1 j=1 ij i i j j i j
This expression is obtained by matching the first and second moments of the true
distribution with the binomial distribution f(k; D, p). This is therefore an approximation
only. As before, VAR can be computed for this new distribution.
To illustrate, assume that all probabilities, all losses, and all correlations are equal,
even though this need not be the case. The diversity score then reduces to
(N pL)(N (1 - p)L) N2
= =
D
( N +r (N - 1)N ) p(1 - p)p - (1 p) LL ( N + r (N - 1)N ) (8)


<!-- Page 21 -->

P. Jorion - Risk Management for Event-Driven Funds 19
If the events are independent, all correlations are zero, and Equation (7) reduces to D=N,
as we would expect. More generally, a positive correlation leads to D<N, which implies
that the distribution has longer tails than with independent events.
Table 5 compares the distributions of portfolio of N=30 deals for different
correlation coefficients. The zero correlation case is the same as before, with a 95%
VAR of $0.33. The diversity score decreases with the correlation. For example, when
the average correlation increases to 0.03, the diversity score falls from 30 to 16.
Accordingly, the 95% VAR increases sharply, from $0.33 to $1.25. With a correlation of
0.10, the 95% VAR moves to $3.57. As expected, higher correlations lead to greater
risk. When the correlation is 0.50 or above, VAR moves to $15, which is the same as for
a portfolio of one deal. In this case, there is no diversification benefit.
Table 5: Distribution of Dependent Deals Using the BET Approach (N=30)
Correlation Diversity Expected VAR (‡ 95% ) VAR (‡ 99.9%)
Score Profit [Actual Confidence Level]
0.00 30.0 +$2.00 $0.33 $2.33 [99.92%]
0.03 16.0 +$2.00 $1.25 $5.00 [99.98%]
0.05 12.2 +$2.00 $1.67 $5.00 [99.93%]
0.10 7.7 +$2.00 $3.57 $9.29 [99.99%]
0.20 4.4 +$2.00 $5.00 $10.00 [99.95%]
0.50 1.9 +$2.00 $15.00 $15.00 [100.00%]
Figure 3 compares the distribution of losses with N=30 in two cases, r =0 and
r =0.05. The distributions are heavily skewed. In each case the expected return is $2.0.
With zero correlation, the right tail falls off quickly. With positive correlation, the tail is
much longer. Thus, a modest amount of correlation has a substantial impact on portfolio
risk.


<!-- Page 22 -->

P. Jorion - Risk Management for Event-Driven Funds 20
Fig. 3. Distribution of Portfolio Losses with Various Correlations: N=30
25%
20%
15%
10%
5%
0%
0.5$- 0.1$- 0.3$ 0.7$ 0.11$ 0.51$
VAR=$5.0
Correlation = 0
VAR=$2.3
Correlation = 0.05
Loss
More general methods can be used instead of the Binomial Expansion Technique
approach. The distribution of events can be constructed from Monte Carlo simulations.
In this framework, each event is driven by a latent random variable such that a drop
below some cutoff point simulates a break with a specified probability. Correlations can
be induced across these random variables so as to generate specified default
correlations.12 Such bottom-up approach, which is similar to structural models of credit
risk, can be used to derive a more general distribution of gains and losses for the event-
driven portfolio. This will be used for results in the next section.
As an illustration of the effect of number of deals on the risk profile of the
portfolio, a simple experiment can be performed based on the actual number of deals
12 In our example with p=15%, we need a correlation of latent variables calibrated at 0.068 to obtain the
observed default correlation of 0.03 using multivariate normal random variables.


<!-- Page 23 -->

P. Jorion - Risk Management for Event-Driven Funds 21
breaks and successes over the period 1997 to 2006. The average number of deals per
quarter is 44. The first portfolio is equally-invested in all deals. Consistently with the
previous analysis, a success is assigned a return of +5%; a failure has a return of - 15%.
Figure 4 displays the hypothetical time series of returns on this portfolio. During the first
quarter, for example, the failure rate is exactly 0%. The return is then computed as R =
0%(-15%) + (1-0%)(+5%) = 5%, which is the highest feasible value. Over the entire
sample, the average return is close to 2%. The all-deal portfolio has very low volatility.
In this experiment, it loses money in one quarter out of 40 only.
Fig. 4. Time-Series of Portfolio Returns for Different Portfolio Sizes
10%
8%
6%
4%
2%
0%
-2%
-4%
-6%
-8%
-10%
1Q7991 1Q8991 1Q9991 1Q0002 1Q1002 1Q2002 1Q3002 1Q4002 1Q5002 1Q6002
All Deals
5 Deals
The second portfolio consists of five deals only. Each quarter, the outcome is
selected randomly, using the deal break fraction in that same quarter as probability of
failure. This approach preserves the pattern of correlations in the data. Figure 4 shows a


<!-- Page 24 -->

P. Jorion - Risk Management for Event-Driven Funds 22
typical outcome. This 5-deal portfolio is much more volatile than the all-deal one even
though it has the same mean. It suffers losses during eight quarters. The worst loss is
- 7%. Clearly, the portfolio with a greater number of deals is the safer one.
Alternatively, this first portfolio could be leveraged three times and have the same level
of risk as the 5-deal portfolio, while delivering three times the expected return.


<!-- Page 25 -->

P. Jorion - Risk Management for Event-Driven Funds 23
V. Implications for Assessing Economic Capital and Leverage
This methodology can be used to infer economic capital levels, or conversely the
number of deals required to keep a fixed level of risk for event-driven portfolios.
Leverage can be assessed from economic capital. Recall that with one deal only,
economic capital is $15, implying a maximum leverage of $100/$15.0=6.7 at which point
the rate of return on economic capital is $2.0/$15.0=13.3% per quarter.
For a 10-deal portfolio with a correlation of 0.03, VAR is $9.0, at a minimum
99.9% confidence level. As a result, the portfolio could be leveraged by a factor of
$100/$9.0, or about 10 times the original equity investment, generating an expected
return of $2.0/$9.0=22%.
This analysis shows that increasing the number of deals substantially lowers risk,
because deal risk is largely idiosyncratic. When N increases to 30 and 100, economic
capital drops to $5.00 and $3.80, respectively. Figure 5 illustrates how economic capital
decreases with the number of deals. Lower risk allows greater leverage, increasing
expected returns.


<!-- Page 26 -->

P. Jorion - Risk Management for Event-Driven Funds 24
Fig. 5. Economic Capital and Portfolio Size (Break Correlation=0.03)
VAR
$50
$45
Leverage=3
$40
Leverage=2
$35
Leverage=1
$30
$25
$20
$15
$10
$5
$0
01 10 20 30 40 50 60 70 80 90 100
Number of Deals
This powerful diversification effect is driven by the low correlation, taken as 0.03
here. Next, Figure 6 examines how correlations affect economic capital, for various
portfolio sizes. The effect is rather dramatic. For N=100, increasing the correlation from
0.00 to 0.10 pushes economic capital from $0.4 to $9.0. Beyond a correlation of 0.30,
there is basically no diversification effect: VAR converges to the worst loss of $15.
Average correlations are very unlikely to reach such values, however, from historical
experience. The previously computed 95% confidence interval for r ranges from 0.017
to 0.045. This translates into a confidence interval of $2.3 to $5.2 for VAR with N=100.
So, even in a worst-case scenario for the break correlation, there is still a fair amount of
diversification.


<!-- Page 27 -->

P. Jorion - Risk Management for Event-Driven Funds 25
Fig. 6. Economic Capital and Break Correlation (Break Probability=15%)
VAR
$16
$14
$12
$10
$8
N=10
$6
N=30
N=100
$4
$2
$0
0.00 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50
Correlation
Economic capital is also sensitive to the break probability, but less so. Figure 7
shows that if this changes from the base number of 15% to 20%, economic capital
increases from $3.8 to $5.0%, when N=100. The 95% confidence interval for the
estimated break correlation was (10.2%, 13.7%) for this sample. So, the worst increase
from the mid-point probability is only 2%. On the other hand, increases in the break
probability are important because they directly reduce the expected return on the
portfolio.


<!-- Page 28 -->

P. Jorion - Risk Management for Event-Driven Funds 26
Fig. 7. Economic Capital and Break Probability (Break Correlation=0.03)
VAR
$16
$14
$12
$10
$8
$6
$4
N=10
N=30
$2
N=100
$0
0.00 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40
Break Probability
Figure 8 illustrates the trade-off between expected return and risk for various
levels of leverage and portfolio size, assuming a baseline correlation of 0.03. Different
levels of leverage are represented by horizontal lines. For a given portfolio size N,
expected (excess) returns and risk increase linearly with leverage. A portfolio with
N=100 can be 3 times leveraged and still have a level of risk similar to a portfolio with
N=10, while delivering three times the expected return.
Apparently, this analysis would lead to the conclusion that having a larger number
of deals is always an improvement. This is because we assumed that the expected profit
per deal remains constant at $2. In practice, portfolio managers have expertise in
evaluating some types of deals. When we increase the number of deals to evaluate, we
would expect that the profit per deal should decrease, reflecting acquisition information
costs. Thus, there should be an optimal number of deals, reflecting the tradeoff between


<!-- Page 29 -->

P. Jorion - Risk Management for Event-Driven Funds 27
risk and profits. Without a measure of portfolio risk, however, such assessment is not
even possible.
Fig. 8. Trade-Off between Expected Return and Economic Capital
(Break Correlation=0.03)
Expected Return
$8
N=100
N=30
N=10 Leverage = 3
$6
Leverage = 2
$4
Leverage = 1
$2
$0
$0 $5 $10 $15 $20 $25 $30
VAR
Finally, we should note that the analysis of economic capital should also account
for non-event risks, including market risk and some idiosyncratic risk.13 These additional
risks can be incorporated in an enlarged model. Stress tests should also be performed to
evaluate the sensitivity of the results to changes in input parameters, including break
probabilities and correlations.
13 There is also horizon risk, which is the risk of a lengthening of the horizon for the deal resolution. This
happened, for instance, during 2004, when the Sarbanes-Oxley Act temporarily slowed down deal activity.


<!-- Page 30 -->

P. Jorion - Risk Management for Event-Driven Funds 28
VI. Conclusions
Event-driven portfolio strategies face discrete deal risk. In contrast with other
types of market risks, where distributions are continuous, deal risk is represented by a
binary variable, which is the success or failure of a deal. This type of risk cannot be
captured by conventional position-based risk measures.
Current positions, however, can be used to construct the distribution of portfolio
returns. To do so, the risk manager needs estimates of the probability of success for each
deal, of the payoffs from success and failure, and of joint correlations across deals. This
paper develops new methods to construct quantitative measures of portfolio risk for
event-driven portfolios.
Once constructed, this distribution can be summarized by the usual risk measures,
such as Value at Risk, which can be used to assess acceptable levels of leverage. This
paper finds that, in the case of independent deals, the economic capital required to
support a portfolio of 30 independent deals is six times lower than for a single-deal
portfolio. Thus, this diversified portfolio could be levered six times more than a single-
deal position and still maintain the same level of risk. In the meantime, this higher
leverage implies a higher return on capital.
In practice, however, these diversification benefits are limited because of
clustering across deal breaks. This paper provides empirical estimates indicating that the
average deal-break correlation is around 0.03 for large North American deals. This
positive correlation decreases the benefits from diversification and increases the required
economic capital.


<!-- Page 31 -->

P. Jorion - Risk Management for Event-Driven Funds 29
This methodology can be used to improve current risk management practices for
event-driven portfolios. Apparently, portfolio managers primarily rely on position limits
to control risk. Similar methods were used to manage credit risk before the generalized
application of modern risk measurement methods.
Without such quantitative methods, it seems difficult to assess the optimal risk-
return profile of the portfolio. With limited expertise or resources, increasing the number
of deals is costly, because it decreases the average expected return per deal. On the other
hand, increasing the number of deals does provide diversification benefits. The optimal
portfolio represents the point where the marginal cost equates the marginal benefit from
increasing deals. Without a formal measure of the marginal benefit, however, this
optimal position cannot be identified.
Armed with these portfolio measurement tools, the portfolio manager should
search for deals that are profitable, yet not too correlated across each other. This paper
provides the means to assess the costs and benefits of different structures for event-driven
portfolios.


<!-- Page 32 -->

P. Jorion - Risk Management for Event-Driven Funds 30
REFERENCES
Bahar, Reza and Krishnan Regal, 2001, “Measuring Default Correlation,” Risk (March),
129—132.
Baker, Malcolm and Serkan Savasoglu, 2002, “Limited Arbitrage in Mergers and
Acquisitions,” Journal of Financial Economics 64, 91-115.
De Servigny, Arnaud and Olivier Renault, 2002, “Default Correlation: Empirical
Evidence,” Working Paper, Standard and Poor’s.
Jorion, Philippe, 2006, “Value at Risk: The New Benchmark to Manage Financial Risk,”
McGrawHill.
Mitchell, Mark and Todd Pulvino, 2001, “Characteristics of Risk and Return in Risk
Arbitrage,” Journal of Finance 56, 2135-2175.
Mitchell, Mark, Todd Pulvino, and Erik Stafford, 2004, “Price Pressure around Mergers,”
Journal of Finance 59, 31-63.
Moody’s, 1996, “The Binomial Expansion Technique Applied to CBO/CLO Analysis,”
Moody’s Special Report.
Moore, Keith, Gene Lai, and Henry Oppenheimer, 2006, “The Behavior of Risk
Arbitrageurs in Mergers and Acquisitions,” Journal of Alternative Investments
(Summer), 19-29.
Schachter, Barry, 2006, “Limits on far VAR,” Risk (March).
