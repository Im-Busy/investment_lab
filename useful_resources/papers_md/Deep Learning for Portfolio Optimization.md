# Deep Learning for Portfolio Optimization

> *Source PDF: Deep Learning for Portfolio Optimization.pdf*
> *Extraction: Combined — previous structured extraction (base) + markitdown raw text*

---

# Deep Learning for Portfolio Optimization

> *Source PDF: Deep Learning for Portfolio Optimization.pdf*
> *Extraction: Combined — markitdown raw text (base) + previous extraction supplements*

---

1
2
0
2

n
a
J

3
2

]

M
P
.
n
i
f
-
q
[

3
v
5
6
6
3
1
.
5
0
0
2
:
v
i
X
r
a

Deep Learning for Portfolio Optimization

Zihao Zhang, Stefan Zohren, Stephen Roberts
Oxford-Man Institute of Quantitative Finance,
University of Oxford

Abstract

We adopt deep learning models to directly optimise the portfolio Sharpe ratio.
The framework we present circumvents the requirements for forecasting expected
returns and allows us to directly optimise portfolio weights by updating model
parameters. Instead of selecting individual assets, we trade Exchange-Traded Funds
(ETFs) of market indices to form a portfolio. Indices of different asset classes show
robust correlations and trading them substantially reduces the spectrum of available
assets to choose from. We compare our method with a wide range of algorithms
with results showing that our model obtains the best performance over the testing
period, from 2011 to the end of April 2020, including the ﬁnancial instabilities
of the ﬁrst quarter of 2020. A sensitivity analysis is included to understand the
relevance of input features and we further study the performance of our approach
under different cost rates and different risk levels via volatility scaling.

1

Introduction

Portfolio optimisation is an essential component of a trading system. The optimisation aims to select
the best asset distribution within a portfolio in order to maximise returns at a given risk level. This
theory was pioneered in Markowitz’s key work [20] and is widely known as modern portfolio theory
(MPT). The main beneﬁt of constructing such a portfolio comes from the promotion of diversiﬁcation
that smoothes out the equity curve, leading to a higher return per risk than trading an individual asset.
This observation has been proven (see e.g. [40]) showing that the risk (volatility) of a long-only
portfolio is always lower than that of an individual asset, for a given expected return, as long as assets
are not perfectly correlated. We note that this is a natural consequence of Jensen’s inequality [16].

Despite the undeniable power of such diversiﬁcation, it is not straightforward to select the “right”
asset allocations in a portfolio, as the dynamics of ﬁnancial markets change signiﬁcantly over time.
Assets that exhibit, for example, strong negative correlations in the past could be positively correlated
in the future. This adds extra risk to the portfolio and degrades subsequent performance. Further, the
universe of available assets for constructing a portfolio is enormous. Taking the US stock markets as
a single example, more than 5000 stocks are available to choose from [34]. Indeed, a well rounded
portfolio not only consists of stocks, but also is typically supplemented with bonds and commodities,
further expanding the spectrum of choices.

In this work, we consider directly optimising a portfolio, utilising deep learning models [18, 12].
Unlike classical methods [20] where expected returns are ﬁrst predicted (typically through econo-
metric models), we bypass this forecasting step to directly obtain asset allocations. Several works
[25, 24, 39] have shown that the return forecasting approach is not guaranteed to maximise the
performance of a portfolio, as the prediction steps attempt to minimise a prediction loss which is not
the overall reward from the portfolio. In contrast, our approach is to directly optimise the Sharpe
ratio [29], thus maximising return per unit of risk. Our framework starts by concatenating multiple
features from different assets to form a single observation and then uses a neural network to extract
salient information and output portfolio weights so as to maximise the Sharpe ratio.

Website: zihao-z.com. Email: zihao@robots.ox.ac.uk

Figure 1: Heatmap for rolling correlations between different index pair. (S: stock index, B: bond
index, C: commodity index and V: volatility index.)

Instead of choosing individual assets, Exchange-Traded Funds (ETFs) [11] of market indices are
selected to form a portfolio. We use four market indices: US total stock index (VTI), US aggregate
bond index (AGG), US commodity index (DBC) and Volatility Index (VIX). All of these indices are
popularly traded ETFs that offer high liquidity and relatively small expense ratios. Trading indices
substantially reduces the possible universe of asset choices and gains exposure to most securities.
Further, these indices are generally uncorrelated, or even negatively correlated, as shown in Figure 1.
Individual instruments in the same asset class, however, often exhibit strong positive correlations.
For example, more than 75% stocks are highly correlated with the market index [34], thereby adding
them to a portfolio helps less with diversiﬁcation.

We are aware that subsector indices can be included in a portfolio, rather than using the total market
index, since sub-industries perform at different levels and a weighting on good performance in a
sector would therefore deliver extra returns. However, we see subsector indices as highly correlated,
thus adding them again provides minimal diversiﬁcation for the portfolio, and risks lowering returns
per unit risk. If higher returns are desired, we can use (e.g.) volatility scaling to upweight our
positions and amplify returns. We therefore do not believe there is a need to ﬁnd the best performing
sector. Instead, we aim to provide a portfolio that delivers high return per unit risk, and allows for
volatility scaling [26, 13, 19] to achieve desired return levels.

Outline: The remainder of the paper is structured as follows. We introduce relevant literature in
Section 2 and present our methodology in Section 3. Section 4 describes our experiments and details
the results of our method compared with a range of baseline algorithms. In Section 5, we summarise
our ﬁndings and discuss possible future work.

2 Literature Review

In this section, we review popular portfolio optimisation methods and discuss how deep learning
models have been applied to this ﬁeld. There is a vast literature available on this topic, so we aim
merely to highlight key concepts, popular in the industry or in academic study. One of the popular
practical approaches is the reallocation strategy [34] adopted by many pension funds (for example,
LifeStrategy Equity Fund, Vanguard). This approach constructs a portfolio by only investing in
stocks and bonds. A typical risk moderate portfolio would, for example, comprise 60% equities
and 40% bonds and the portfolio needs to be only rebalanced semi-annually or annually to maintain
this allocation ratio. The method delivers good performance over the long term, however the ﬁxed
allocation ratio means that investors with preference for more weight on stocks need to tolerate
potentially large drawdowns during dull markets.

Mean-variance analysis or MPT [20] is used for many institutional portfolios that solves a constraint
optimisation problem to derive portfolio weights. Despite its popularity, the assumptions of the
theory are under criticism as they are often not obeyed in real ﬁnancial markets. In particular, returns
are assumed to follow a Gaussian distribution in MPT, therefore, investors only consider expected
return and variance of the portfolio returns to make decisions. However, it is widely accepted (see

2

S&BS&VS&CB&VB&CV&C0.80.60.40.20.0-0.2-0.6-0.4-0.8for instance [4, 38]) that returns tend to have fat tails and extreme losses are more likely to occur in
practice, leading to severe drawdowns that are not bearable. The Maximum Diversiﬁcation (MD)
portfolio is another promising method introduced in [3] that aims to maximise the diversiﬁcation of
a portfolio, thereby aiming to have minimally correlated assets so the portfolio can achieve higher
returns (and lower risk) than other classical methods. We compare our model with both these
strategies, with the results suggesting that our methods deliver better performance and tolerate larger
transaction costs than either of these benchmarks.

Stochastic Portfolio Theory (SPT) was recently proposed in [7, 9]. Unlike other methods, SPT
aims to achieve relative arbitrages meaning to select portfolios that can outperform a market index
with probability one. Such investment strategies have been studied in [5, 6, 27, 36]. However, the
number of relative arbitrage strategies remains small, as theory does not suggest how to construct such
strategies. We can check whether a given strategy is a relative arbitrage, but it is non-trivial to develop
one ex ante. In this work, we include a particular class of SPT called functionally generated portfolio
(FGP) [8] in our experiment, but the result suggests this method delivers inferior performance than
other algorithms and generates large turnovers, making it unproﬁtable under heavy transaction costs.

The idea of our end-to-end training framework was ﬁrst initiated in [25, 24]. However, these works
mainly focus on optimising the performance for a single asset so there is little discussion on how
portfolios should be maximised. Furthermore, their testing period is from 1970 to 1994, whereas
our dataset is up to date and we study the behavior of our strategy under the current crisis due to
COVID-19. We can also link our approach to reinforcement learning (RL) [31, 22, 35] where an agent
interacts with an environment to maximise cumulative rewards. The works of [1, 15, 39] have studied
this stream and adopted RL to design trading strategies. However, the goal of RL is to maximise
expected cumulative rewards such as proﬁts whereas Sharpe ratio can not be directly optimised.

3 Methodology

In this section, we introduce our framework and discuss how Sharpe ratio can be optimised through
gradient ascent. We discuss the types of neural networks used and detail the functionality of each
component in our method.

3.1 Objective Function

The Sharpe ratio is used to gauge the return per risk of a portfolio and is deﬁned as expected return
over volatility (excluding risk-free rate for simplicity):

L =

E(Rp)
Std(Rp)

(1)

where E(Rp) and Std(Rp) are the estimates of the mean and standard deviation of portfolio returns.
Speciﬁcally, for a trading period of t = {1, · · · , T }, we can maximise the following objective
function:

LT =

(cid:113)

E(Rp,t)

E(R2

p,t) − (E(Rp,t))2

E(Rp,t) =

1
T

T
(cid:88)

t=1

Rp,t

where Rp,t is realized portfolio return over n assets at time t denoted as:

Rp,t =

n
(cid:88)

i=1

wi,t−1 · ri,t

(2)

(3)

where ri,t is the return of asset i with ri,t = (pi,t/pi,t−1 − 1). We represent the allocation ratio
(position) of asset i as wi,t ∈ [0, 1] and (cid:80)n
i wi,t = 1. In our approach, a neural network f with
parameters θ is adopted to model wi,t for a long-only portfolio:

wi,t = f (θ|xt)

3

(4)

where xt represents the current market information and we bypass the classical forecasting step by
linking the inputs with positions to maximise the Sharpe over trading period T , namely LT . However,
a long-only portfolio imposes constraints that require weights to be positive and summed to one, we
use softmax outputs to fulﬁll these requirements:

wi,t =

exp( ˜wi,t)
j exp( ˜wj,t)

(cid:80)n

, where ˜wi,t are the raw weights.

(5)

Such a framework can be optimised using unconstrained optimisation methods. Particularly, we use
gradient ascent to maximise the Sharpe ratio. The gradient of LT with respect to parameters θ is
readily calculable, with an excellent derivation presented in [25, 23]. Once we obtain ∂LT /∂θ, we
can repeatedly compute this value from training data and update the parameters by using gradient
ascent:

θnew := θold + α

∂LT
∂θ

(6)

where α is the learning rate and the process can be repeated for many epochs until the convergence of
Sharpe ratio or the optimisation of validation performance is achieved.

3.2 Model Architecture

We depict our network architecture in Figure 2. Our model consists of three main building blocks:
input layer, neural layer and output layer. The idea of this design is to use neural networks to extract
cross-sectional features from input assets. Features extracted from deep learning models have been
suggested to perform better than traditional hand-crafted features [39]. Once features have been
extracted, the model outputs portfolio weights and we obtain realised returns to maximise Sharpe
ratio. The following details each component of our method.

Input layer We denote each asset as Ai and we have n assets to form a portfolio. A single input is
prepared by concatenating information from all assets. For example, the input features of one asset
can be its past prices and returns with a dimension of (k, 2) where k represents the lookback window.
By stacking features across all assets, the dimension of the resulting input would be (k, 2 × n). We
can then feed this input to the network and expect non-linear features being extracted.

Neural layer A series of hidden layers can be stacked to form a network, however, in practice,
this part requires lots of experiments as there are plentiful ways of combining hidden layers and
the performance often depends on the design of architecture. We have tested deep learning models
including fully connected neural network (FCN), convolutional neural network (CNN) and Long
Short-Term Memory (LSTM) [14]. Overall, LSTMs deliver the best performance for modelling daily
ﬁnancial data and a number of works [33, 19, 39] support this observation.

We note the problem of FCN is its problem of severe overﬁtting. As it assigns parameters to each
input feature, this results in an excess number of parameters. The LSTM operates with a cell structure
that has gate mechanisms to summarise and ﬁlter information from its long history, so the model ends
up with fewer trainable parameters and achieves better generalisation results. In contrast, CNNs with
a strong smoothing (typical of large convolutional ﬁlters) tend to have underﬁtting problems, such
that oversmooth solutions are obtained. Due to the design of parameter sharing and the convolution
operations, we experience CNNs to overﬁlter the inputs. However, we note that CNNs appear to be
excellent candidates for modelling high-frequency ﬁnancial data such as limit order books [37].

In order to construct a long-only portfolio, we use the softmax activation function for
Ouput layer
the output layer, which naturally imposes constraints to keep portfolio weights positive and summing
to one. The number of output nodes (w1, · · · , wn) is equal to the number of assets in our portfolio,
and we can multiply these portfolio weights with associated assets’ returns (r1, · · · , rn) to calculate
realised portfolio returns (Rp). Once realised returns are obtained, we can derive the Sharpe ratio
and calculate the gradients of the Sharpe ratio with respect to the model parameters and use gradient
ascent to update the parameters.

4

Figure 2: Model architecture schematic. Overall, our model contains three main building blocks:
input layer, neural layer and output layer.

4 Experiments

4.1 Description of Dataset

We use four market indices: US total stock index (VTI), US aggregate bond index (AGG), US
commodity index (DBC) and Volatility Index (VIX). These are popular Exchange-Traded Funds
(ETFs) [11] that have existed for more than 15 years. As discussed in Section 1, trading indices offers
advantages over trading individual assets because these indices are generally uncorrelated resulting
in diversiﬁcation. A diversiﬁed portfolio delivers a higher return per risk and the idea of our strategy
is to have a system that delivers good reward-to-risk ratio. Our dataset ranges from 2006 to 2020 and
contains daily observations. We retrain our model at every 2 years and use all data available up to
that point to update parameters. Overall, our testing period is from 2011 to the end of April 2020,
including the most recent crisis due to COVID-19.

4.2 Baseline Algorithms

We compare our method with a group of baseline algorithms. The ﬁrst set of baseline models are
reallocation strategies adopted by many pension funds. These strategies assign a ﬁxed allocation
ratio to relevant assets and rebalance portfolios annually to maintain these ratios. Investors can
select a portfolio based on their risk preferences. In general, portfolios weighted more on equities
would deliver better performance at the expense of larger volatility. In this work, we consider four
such strategies: Allocation 1 (25% shares, 25% bonds, 25% commodities and 25% volatility index),
Allocation 2 (50% shares, 10% bonds, 20% commodities, and 20% volatility index), Allocation 3
(10% shares, 50% bonds, 20% commodities, and 20% volatility index), and Allocation 4 (40% shares,
40% bonds, 10% commodities and 10% volatility index).

The second set of comparison models are mean-variance optimisation (MV) [20] and maximum
diversiﬁcation (MD) [32]. We use moving averages with a rolling window of 50 days to estimate
the expected returns and covariance matrix. The portfolio weights are updated at a daily basis and
we select weights that maximise Sharpe ratio for MV. The last baseline algorithm is the diversity-
weighted portfolio (DWP) from Stochastic Portfolio Theory presented in [28]. The DWP relates

5

!"!#!$Hidden LayerHidden Layer%"%$&"%$'(Output layerNeural layerInput layer)"Softmax)#)$portfolio weights to assets’ market capitalisation and it has been suggested to be able to outperform
the market index with certainty [10].

4.3 Training Scheme

In this work, we use a single layer of LSTM connectivity, with 64 units, to model the portfolio
weights and thence to optimise the Sharpe ratio. We purposely keep our network simple to indicate
the effectiveness of this end-to-end training pipeline instead of carefully ﬁne-tuning the “right”
hyperparameters. Our input contains close prices and daily returns for each market index and we
take the past 50 days of these observations to form a single input. We are aware that returns can
be derived from prices, but keeping returns help with the evaluation of Equation 7 and we can also
treat them as momentum features in [26]. As our focus is not on feature selection, we choose these
commonly used features in our work. The Adam optimiser [17] is used for training our network, and
the mini-batch size is 64. We take 10% of any training data as a separate validation-set to optimise
hyperparameters and control overﬁtting problems. Any hyperparameter optimisation is done on the
validation set, leaving the test data for the ﬁnal performance evaluation and ensuring the validity of
our results. In general, our training process stops after 100 epochs.

4.4 Experimental Results

When reporting the test performance, we include transaction costs and use volatility scaling [26,
19, 39] to scale our positions based on market volatility. We can set our own volatility target and
meet expectations of investors with different risk preferences. Once volatilities are adjusted, our
investment performances are mainly driven by strategies instead of being heavily affected by markets.
The modiﬁed portfolio return can be deﬁned as:

Rp,t =

n
(cid:88)

i

σtgt
σi,t−1

wi,t−1 · ri,t − C ·

n
(cid:88)

i

(cid:12)
(cid:12)
(cid:12)

σtgt
σi,t−1

wi,t−1 −

σtgt
σi,t−2

wi,t−2

(cid:12)
(cid:12)
(cid:12)

(7)

where σtgt is the volatility target and σi,t−1 is an ex-ante volatility estimate of asset i calculated
using an exponentially weighted moving standard deviation with a 50-day window on ri,t. We use
daily changes of traded value of an asset to represent transaction costs, which is calculated by the
second term in Equation 7. C (=1bs=0.0001) is the cost rate and we change it to reﬂect how our
model performs under different transaction costs.

To evaluate the performance of our methods, we utilise following metrics: expected return (E(R)),
standard deviation of return (Std(R)), Sharpe ratio [29], downside deviation of return (DD(R)) [21],
and Sortino ratio [30]. All of these metrics are annualised, and we also report on maximum drawdown
(MDD) [2], percentage of positive return (% of + Ret) and the ratio between positive and negative
return (Ave. P / Ave. L).

Table 1 presents the results of our model (DLS) compard to other baseline algorithms. The top of the
table shows the results without using volatility scaling, and we can see that our model (DLS) achieves
the best Sharpe’s ratio and Sortino ratio, delivering the highest return per risk. However, given the
large differences in volatilities, we can not directly compare expected and cumulative returns for
different methods, thereby volatility scaling also helps to make fair comparisons.

Once volatilities are scaled (shown in the middle of Table 1), DLS delivers the best performance
across all evaluation metrics except for a slightly larger drawdown. If we look at the cumulative
returns in Figure 3, DLS exhibits outstanding performance over the long haul and the maximum
drawdown is reasonable, ensuring the conﬁdence of investors to hold through hard times. Further, if
we look at the bottom of Table 1 where a large cost rate (C = 0.1%) is used, our model (DLS) stills
delivers the best expected return and achieves the highest Sharpe and Sortino ratios.

However, with a higher cost rate, we can see that reallocation strategies work well and, in particular,
Allocations 3 and 4 achieve comparable results to our method. In order to investigate why performance
gap diminishes with a higher cost rate, we present the boxplots for annual realised trade returns and
accumulated costs for different assets in Figure 4. Overall, our model delivers better realised returns
than reallocation strategies, but we also accumulate much larger transaction costs since our positions
are adjusted on a daily basis, leading to a higher turnover.

6

Table 1: Experiment results for different algorithms.

E(R)

Std(R)

Sharpe DD(R)

Sortino MDD % of + Ret

Allocation 1
Allocation 2
Allocation 3
Allocation 4
MV
MD
DWP
DLS

Allocation 1
Allocation 2
Allocation 3
Allocation 4
MV
MD
DWP
DLS

0.282
0.249
0.228
0.152
0.082
0.462
0.051
0.313

0.160
0.123
0.145
0.164
0.112
0.157
0.089
0.206

No volatility scaling and C = 0.01%

0.303
0.212
0.256
0.123
0.108
0.523
0.102
0.168

0.929
1.173
0.890
1.228
0.759
0.882
0.493
1.858

0.136
0.095
0.116
0.052
0.069
0.239
0.067
0.099

2.065
2.616
1.962
2.932
1.192
1.931
0.740
3.135

0.142
0.097
0.122
0.081
0.195
0.273
0.179
0.102

Volatility scaling (σtgt = 0.10) and C = 0.01%

0.105
0.106
0.105
0.104
0.100
0.106
0.109
0.105

1.526
1.146
1.383
1.579
1.120
1.484
0.818
1.962

0.061
0.065
0.061
0.064
0.063
0.065
0.069
0.062

2.629
1.861
2.396
2.588
1.767
2.414
1.291
3.322

0.111
0.127
0.105
0.112
0.211
0.125
0.115
0.123

Volatility scaling (σtgt = 0.10) and C = 0.1%

Allocation 1
Allocation 2
Allocation 3
Allocation 4
MV
MD
DWP
DLS

0.133
0.105
0.117
0.135
0.019
0.095
-0.083
0.148

0.105
0.107
0.105
0.104
0.101
0.106
0.110
0.105

1.274
0.986
1.110
1.299
0.191
0.899
-0.753
1.403

0.061
0.066
0.061
0.064
0.066
0.066
0.074
0.063

2.172
1.590
1.903
2.108
0.293
1.431
-1.129
2.327

0.113
0.244
0.107
0.114
0.324
0.145
0.627
0.125

0.479
0.483
0.476
0.505
0.562
0.473
0.549
0.537

0.554
0.549
0.542
0.565
0.561
0.565
0.556
0.559

0.548
0.547
0.538
0.559
0.537
0.549
0.508
0.547

Ave. P
Ave. L

1.193
1.254
1.183
1.349
1.199
1.182
1.107
1.518

1.289
1.211
1.259
1.303
1.213
1.297
1.148
1.375

1.236
1.179
1.203
1.244
1.033
1.171
0.880
1.272

For reallocation strategies, daily position changes are only updated for volatility scaling. Otherwise,
we only actively change positions once a year to rebalance and maintain the allocation ratio. As a
result, reallocation strategies deliver minimal transaction costs. This analysis aims to indicate the
validity of our results and show that our method can work under unfavorable conditions.

4.5 Model Performance during 2020 Crisis

Due to the recent COVID-19 pandemic, global stock markets fell dramatically and experienced
extreme volatility. The crash started on the 24th February 2020 where markets reported their largest

Figure 3: Cumulative returns (logarithmic scale) for Left: no volatility scaling and C = 0.01%;
Middle: volatility scaling (σtgt = 0.10) and C = 0.01%; Right: volatility scaling (σtgt = 0.10) and
C = 0.1%.

7

20112012201320142015201620172018201920200.00.51.01.52.02.53.0Allocation 1Allocation 2Allocation 3Allocation 4MVMDDWPDLS20112012201320142015201620172018201920200.000.250.500.751.001.251.501.75Allocation 1Allocation 2Allocation 3Allocation 4MVMDDWPDLS20112012201320142015201620172018201920201.00.50.00.51.0Allocation 1Allocation 2Allocation 3Allocation 4MVMDDWPDLSFigure 4: Boxplot for Top: annual realised trade returns; Bottom: annual accumulated costs for
different assets with volatility scaling (σtgt = 0.10) and C = 0.01%.

one-week declines since the 2008 ﬁnancial crisis. Later on, with an oil price war between Russia and
the OPEC countries, markets further dampened and encountered the largest single-day percentage
drop since Black Monday in 1987. As of March 2020, we have seen a downturn of at least 25% in
the US markets and 30% in most G20 countries. The crisis shattered many investors’ conﬁdence and
resulted in a great loss of their wealths. However, it also provides us with a great opportunity to stress
test our method and understand how our model performs during the crisis.

In order to study the model behaviours, we plot how our algorithm allocated the assets from January
to April 2020 in Figure 5. At the beginning of 2020, we can see that our model had a quite diverse
holding. However, after a small dip in stock index in early February, we almost had only bonds in our
portfolio. There were some equity positions left but very small positions for volatility and commodity
indices. When the crash started on 24th February, our holdings were concentrated on the bond index
which is considered to be safe assets during the crisis. Interestingly, the bond index also fell this time
(in the middle of March) although it rebounded quite quickly. During the bond falling, our original
positions did not change much but the scaled positions decreased a lot for the bond index due to a
spiking volatility, therefore our drawdown was small. Overall, we can see that our model delivers
reasonable allocations during the crisis and our positions are protected through volatility scaling.

4.6 Sensitivity Analysis

In order to understand how input features affect our decisions, we study the sensitivity analysis
presented in [24] for our method. The absolute normalised sensitivity of feature xi is deﬁned as:

Si =

dL
dxi
(cid:12)
(cid:12)
(cid:12)

maxj

(cid:12)
(cid:12)
(cid:12)

(8)

dL
dxj

where L represents the objective function and Si captures the relative sensitivity for feature xi
compared with other features. We plot the time-varying sensitivities for all features in Figure 6. The
y-axis indicates the 400 features we have because we use 4 indices (each with prices and returns) and

8

StockBondVolatilityCommodityAsset Classes0.40.20.00.20.4Realised ReturnsAllocation 1Allocation 2Allocation 3Allocation 4MVMDDWPDLSStockBondVolatilityCommodityAsset Classes0.0000.0020.0040.0060.0080.010Annual CostsAllocation 1Allocation 2Allocation 3Allocation 4MVMDDWPDLSFigure 5: Shifts of portfolio weights for our model (DLS) during the crisis of COVID-19 with
volatility scaling (σtgt = 0.10).

we take a timeframe of past 50 observations to form a single input so there are 400 features in total.
The row labeled “Sprice” represents price features for the stock index and the bottom of row “Sprice”
means the most recent price for that observation. Same convention is used for all other features.

The importance of features varies over the time, but the most recent features always make the
biggest contributions as we can see that the bottom of each feature row has the highest weight. This
observation meets our understanding as, for time-series, recent observations carry more information.
The further away from the current observation point, the less importance of features show and we
could adjust features used based on this observation such as using a small lookback window.

Figure 6: Sensitivity analysis for input features over the time.

9

2020-01-012020-01-152020-02-012020-02-152020-03-012020-03-152020-04-012020-04-15110120130140150160170Stock0.00.20.40.60.81.01.2PositionScaled position2020-01-012020-01-152020-02-012020-02-152020-03-012020-03-152020-04-012020-04-15106108110112114116118Bond123456PositionScaled position2020-01-012020-01-152020-02-012020-02-152020-03-012020-03-152020-04-012020-04-151020304050607080Volatility0.000.020.040.060.080.100.120.140.16PositionScale position2020-01-012020-01-152020-02-012020-02-152020-03-012020-03-152020-04-012020-04-15111213141516Commodity0.000.050.100.150.200.25PositionScaled position0.50.40.30.20.15 Conclusion

In this work, we adopt deep learning models to directly optimise a portfolio’s Sharpe ratio. This
pipeline bypasses the traditional forecasting step and allows us to optimise portfolio weights by
updating model parameters through gradient ascent. Instead of using individual assets, we focus on
ETFs of market indices to form a portfolio. Doing this substantially reduces the scope of possible
assets to choose from, and these indices have shown robust correlations. In this work, four market
indices have been used to form a portfolio.

We compare our method with a wide range of popular algorithms including reallocation strategies,
classical mean-variance optimisation, maximum diversiﬁcation and stochastic portfolio theory model.
Our testing period is from 2011 to the April of 2020, and include the recent crisis due to COVID-19.
The results show that our model delivers the best performance and a detailed study of our model
performance during the crisis shows the rationality and practicability of our method. A sensitivity
analysis is included to understand how input features contribute to outputs and the observations meet
our econometric understanding, showing the most recent features are most relevant.

In subsequent continuation of this work, we aim to study portfolios performance under different
objective functions. Given the ﬂexible framework of our approach, we can maximise Sortino ratio or
even the diversiﬁcation degree of a portfolio as long as functions are differentiable. We further note
that the volatility estimates used for scaling are lagged estimates that do not necessarily represent
current market volatilities. We consider another extension to this work to thus adapt the network
architecture to infer (future) volatility estimates as a part of the training process.

Acknowledgements

The authors would like to thank members of Machine Learning Research Group at the University of
Oxford for their useful comments. We are most grateful to the Oxford-Man Institute of Quantitative
Finance for support and data access.

References

[1] Francesco Bertoluzzo and Marco Corazza. Testing different reinforcement learning conﬁgura-
tions for ﬁnancial trading: Introduction and applications. Procedia Economics and Finance,
3:68–77, 2012.

[2] Alexei Chekhlov, Stanislav Uryasev, and Michael Zabarankin. Drawdown measure in portfolio
optimization. International Journal of Theoretical and Applied Finance, 8(01):13–58, 2005.
[3] Yves Choueifaty and Yves Coignard. Toward maximum diversiﬁcation. The Journal of Portfolio

Management, 35(1):40–51, 2008.

[4] Rama Cont and De Nitions. Statistical properties of ﬁnancial time series. 1999.
[5] Daniel Fernholz and Ioannis Karatzas. On optimal arbitrage. The Annals of Applied Probability,

pages 1179–1204, 2010.

[6] Daniel Fernholz, Ioannis Karatzas, et al. Optimal arbitrage under model uncertainty. The Annals

of Applied Probability, 21(6):2191–2225, 2011.

[7] E Robert Fernholz. Stochastic portfolio theory. In Stochastic Portfolio Theory, pages 1–24.

Springer, 2002.

[8] Robert Fernholz. Portfolio generating functions. In Quantitative Analysis in Financial Markets:
Collected Papers of the New York University Mathematical Finance Seminar, pages 344–367.
World Scientiﬁc, 1999.

[9] Robert Fernholz and Ioannis Karatzas. Stochastic portfolio theory: An overview. Handbook of

numerical analysis, 15:89–167, 2009.

[10] Robert Fernholz, Ioannis Karatzas, and Constantinos Kardaras. Diversity and relative arbitrage

in equity markets. Finance and Stochastics, 9(1):1–27, 2005.

[11] Gary L Gastineau. Exchange-traded funds. Handbook of ﬁnance, 1, 2008.
[12] Ian Goodfellow, Yoshua Bengio, and Aaron Courville. Deep learning. MIT press, 2016.

10

[13] Campbell R Harvey, Edward Hoyle, Russell Korgaonkar, Sandy Rattray, Matthew Sargaison,
and Otto Van Hemert. The impact of volatility targeting. The Journal of Portfolio Management,
45(1):14–33, 2018.

[14] Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural computation,

9(8):1735–1780, 1997.

[15] Chien Yi Huang. Financial trading as a game: A deep reinforcement learning approach. arXiv

preprint arXiv:1807.02787, 2018.

[16] Johan Ludwig William Valdemar Jensen et al. Sur les fonctions convexes et les inégalités entre

les valeurs moyennes. Acta mathematica, 30:175–193, 1906.

[17] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. Proceedings

of the International Conference on Learning Representations, 2015.

[18] Yann LeCun, Yoshua Bengio, and Geoffrey Hinton. Deep learning. Nature, 521(7553):436–444,

2015.

[19] Bryan Lim, Stefan Zohren, and Stephen Roberts. Enhancing time-series momentum strategies
using deep neural networks. The Journal of Financial Data Science, 1(4):19–38, 2019.

[20] Harry Markowitz. Portfolio selection. The journal of ﬁnance, 7(1):77–91, 1952.
[21] Alexander J McNeil, Rüdiger Frey, and Paul Embrechts. Quantitative risk management:

Concepts, techniques and tools-revised edition. Princeton university press, 2015.

[22] Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan
Wierstra, and Martin Riedmiller. Playing Atari with deep reinforcement learning. NIPS Deep
Learning Workshop 2013, 2013.

[23] Gabriel Molina. Stock trading with recurrent reinforcement learning (RRL). CS229, nd Web,

15, 2016.

[24] John Moody and Matthew Saffell. Learning to trade via direct reinforcement. IEEE transactions

on neural Networks, 12(4):875–889, 2001.

[25] John Moody, Lizhong Wu, Yuansong Liao, and Matthew Saffell. Performance functions and
reinforcement learning for trading systems and portfolios. Journal of Forecasting, 17(5-6):441–
470, 1998.

[26] Tobias J Moskowitz, Yao Hua Ooi, and Lasse Heje Pedersen. Time series momentum. Journal

of ﬁnancial economics, 104(2):228–250, 2012.

[27] Johannes Ruf. Hedging under arbitrage. Mathematical Finance: An International Journal of

Mathematics, Statistics and Financial Economics, 23(2):297–317, 2013.

[28] Yves-Laurent Kom Samo and Alexander Vervuurt. Stochastic portfolio theory: A machine
In Proceedings of the Thirty-Second Conference on Uncertainty in

learning perspective.
Artiﬁcial Intelligence, pages 657–665, 2016.

[29] William F Sharpe. The sharpe ratio. Journal of portfolio management, 21(1):49–58, 1994.
[30] Frank A Sortino and Lee N Price. Performance measurement in a downside risk framework.

the Journal of Investing, 3(3):59–64, 1994.

[31] Richard S Sutton and Andrew G Barto. Reinforcement learning: An introduction. MIT press,

2018.

[32] Ludan Theron and Gary Van Vuuren. The maximum diversiﬁcation investment strategy: A
portfolio performance comparison. Cogent Economics & Finance, 6(1):1427533, 2018.
[33] Avraam Tsantekidis, Nikolaos Passalis, Anastasios Tefas, Juho Kanniainen, Moncef Gabbouj,
and Alexandros Iosiﬁdis. Using deep learning to detect price change indications in ﬁnancial
markets. In 2017 25th European Signal Processing Conference (EUSIPCO), pages 2511–2515.
IEEE, 2017.

[34] Russell Wild. Index Investing for Dummies. John Wiley & Sons, 2008.
[35] Ronald J Williams. Simple statistical gradient-following algorithms for connectionist reinforce-

ment learning. Machine learning, 8(3-4):229–256, 1992.

[36] Ting-Kam Leonard Wong. Optimization of relative arbitrage. Annals of Finance, 11(3-4):345–

382, 2015.

11

[37] Zihao Zhang, Stefan Zohren, and Stephen Roberts. DeepLOB: Deep convolutional neural
networks for limit order books. IEEE Transactions on Signal Processing, 67(11):3001–3012,
2019.

[38] Zihao Zhang, Stefan Zohren, and Stephen Roberts. Extending deep learning models for
limit order books to quantile regression. Proceedings of Time Series Workshop of the 36 th
International Conference on Machine Learning, Long Beach, California, PMLR 97, 2019.,
2019.

[39] Zihao Zhang, Stefan Zohren, and Roberts Stephen. Deep reinforcement learning for trading.

The Journal of Financial Data Science, 2020.

[40] Eric Zivot. Introduction to computational ﬁnance and ﬁnancial econometrics. Chapman & Hall

Crc, 2017.

12



---

## Content from Previous Extraction (not in markitdown output)

### Visual/Chart/Graph Descriptions

#### 1

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

#### 2

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

#### 3

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

#### 4

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

#### 5

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

#### 6

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

#### 7

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

### Additional Content

#### 1

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

#### 2

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

#### 3

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

#### 4

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

#### 5

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


---

## Content Unique to Old Extraction (not found in markitdown output)

### 1. # Deep Learning for Portfolio Optimization...

# Deep Learning for Portfolio Optimization

### 2. > *Source PDF: Deep Learning for Portfolio Optimization.pdf*
> *Extraction: Comb...

> *Source PDF: Deep Learning for Portfolio Optimization.pdf*
> *Extraction: Combined — markitdown raw text (base) + previous extraction supplements*

### 3. ## Content from Previous Extraction (not in markitdown output)...

## Content from Previous Extraction (not in markitdown output)

### 4. ### Visual/Chart/Graph Descriptions...

### Visual/Chart/Graph Descriptions

### 5. 0.8
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
Figure1: Heatmap...

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

### 6. wherex representsthecurrentmarketinformationandwebypasstheclassicalforecastingst...

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

### 7. ! ! !
" # $
Input layer
Hidden Layer
Neural layer
Hidden Layer
) ) )
" # $
Softm...

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

### 8. portfolioweightstoassets’marketcapitalisationandithasbeensuggestedtobeabletooutp...

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

### 9. Table1: Experimentresultsfordifferentalgorithms.
E(R) Std (R) Sharpe DD(R) Sorti...

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

### 10. 0.4
0.2
0.0
0.2
0.4
Stock Bond Volatility Commodity
Asset Classes
snrute R
desil...

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

### 11. 170 Stock Position 1.2 118 Bond Position
Scaled position Scaled position 6
1.0 1...

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

### 12. Deep Learning for Portfolio Optimization
Zihao Zhang, Stefan Zohren, Stephen Rob...

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

### 13. forinstance[4,38]) thatreturnstendtohavefattailsandextremelossesaremorelikelytoo...

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

### 14. 5 Conclusion
Inthiswork, weadoptdeeplearningmodelstodirectlyoptimiseaportfolio’s...

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

### 15. [13] Campbell RHarvey, Edward Hoyle, Russell Korgaonkar, Sandy Rattray, Matthew ...

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

### 16. [37] Zihao Zhang, Stefan Zohren, and Stephen Roberts. Deep LOB: Deep convolution...

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


---

## Raw Markitdown Extraction (full text)

1
2
0
2

n
a
J

3
2

]

M
P
.
n
i
f
-
q
[

3
v
5
6
6
3
1
.
5
0
0
2
:
v
i
X
r
a

Deep Learning for Portfolio Optimization

Zihao Zhang, Stefan Zohren, Stephen Roberts
Oxford-Man Institute of Quantitative Finance,
University of Oxford

Abstract

We adopt deep learning models to directly optimise the portfolio Sharpe ratio.
The framework we present circumvents the requirements for forecasting expected
returns and allows us to directly optimise portfolio weights by updating model
parameters. Instead of selecting individual assets, we trade Exchange-Traded Funds
(ETFs) of market indices to form a portfolio. Indices of different asset classes show
robust correlations and trading them substantially reduces the spectrum of available
assets to choose from. We compare our method with a wide range of algorithms
with results showing that our model obtains the best performance over the testing
period, from 2011 to the end of April 2020, including the ﬁnancial instabilities
of the ﬁrst quarter of 2020. A sensitivity analysis is included to understand the
relevance of input features and we further study the performance of our approach
under different cost rates and different risk levels via volatility scaling.

1

Introduction

Portfolio optimisation is an essential component of a trading system. The optimisation aims to select
the best asset distribution within a portfolio in order to maximise returns at a given risk level. This
theory was pioneered in Markowitz’s key work [20] and is widely known as modern portfolio theory
(MPT). The main beneﬁt of constructing such a portfolio comes from the promotion of diversiﬁcation
that smoothes out the equity curve, leading to a higher return per risk than trading an individual asset.
This observation has been proven (see e.g. [40]) showing that the risk (volatility) of a long-only
portfolio is always lower than that of an individual asset, for a given expected return, as long as assets
are not perfectly correlated. We note that this is a natural consequence of Jensen’s inequality [16].

Despite the undeniable power of such diversiﬁcation, it is not straightforward to select the “right”
asset allocations in a portfolio, as the dynamics of ﬁnancial markets change signiﬁcantly over time.
Assets that exhibit, for example, strong negative correlations in the past could be positively correlated
in the future. This adds extra risk to the portfolio and degrades subsequent performance. Further, the
universe of available assets for constructing a portfolio is enormous. Taking the US stock markets as
a single example, more than 5000 stocks are available to choose from [34]. Indeed, a well rounded
portfolio not only consists of stocks, but also is typically supplemented with bonds and commodities,
further expanding the spectrum of choices.

In this work, we consider directly optimising a portfolio, utilising deep learning models [18, 12].
Unlike classical methods [20] where expected returns are ﬁrst predicted (typically through econo-
metric models), we bypass this forecasting step to directly obtain asset allocations. Several works
[25, 24, 39] have shown that the return forecasting approach is not guaranteed to maximise the
performance of a portfolio, as the prediction steps attempt to minimise a prediction loss which is not
the overall reward from the portfolio. In contrast, our approach is to directly optimise the Sharpe
ratio [29], thus maximising return per unit of risk. Our framework starts by concatenating multiple
features from different assets to form a single observation and then uses a neural network to extract
salient information and output portfolio weights so as to maximise the Sharpe ratio.

Website: zihao-z.com. Email: zihao@robots.ox.ac.uk

Figure 1: Heatmap for rolling correlations between different index pair. (S: stock index, B: bond
index, C: commodity index and V: volatility index.)

Instead of choosing individual assets, Exchange-Traded Funds (ETFs) [11] of market indices are
selected to form a portfolio. We use four market indices: US total stock index (VTI), US aggregate
bond index (AGG), US commodity index (DBC) and Volatility Index (VIX). All of these indices are
popularly traded ETFs that offer high liquidity and relatively small expense ratios. Trading indices
substantially reduces the possible universe of asset choices and gains exposure to most securities.
Further, these indices are generally uncorrelated, or even negatively correlated, as shown in Figure 1.
Individual instruments in the same asset class, however, often exhibit strong positive correlations.
For example, more than 75% stocks are highly correlated with the market index [34], thereby adding
them to a portfolio helps less with diversiﬁcation.

We are aware that subsector indices can be included in a portfolio, rather than using the total market
index, since sub-industries perform at different levels and a weighting on good performance in a
sector would therefore deliver extra returns. However, we see subsector indices as highly correlated,
thus adding them again provides minimal diversiﬁcation for the portfolio, and risks lowering returns
per unit risk. If higher returns are desired, we can use (e.g.) volatility scaling to upweight our
positions and amplify returns. We therefore do not believe there is a need to ﬁnd the best performing
sector. Instead, we aim to provide a portfolio that delivers high return per unit risk, and allows for
volatility scaling [26, 13, 19] to achieve desired return levels.

Outline: The remainder of the paper is structured as follows. We introduce relevant literature in
Section 2 and present our methodology in Section 3. Section 4 describes our experiments and details
the results of our method compared with a range of baseline algorithms. In Section 5, we summarise
our ﬁndings and discuss possible future work.

2 Literature Review

In this section, we review popular portfolio optimisation methods and discuss how deep learning
models have been applied to this ﬁeld. There is a vast literature available on this topic, so we aim
merely to highlight key concepts, popular in the industry or in academic study. One of the popular
practical approaches is the reallocation strategy [34] adopted by many pension funds (for example,
LifeStrategy Equity Fund, Vanguard). This approach constructs a portfolio by only investing in
stocks and bonds. A typical risk moderate portfolio would, for example, comprise 60% equities
and 40% bonds and the portfolio needs to be only rebalanced semi-annually or annually to maintain
this allocation ratio. The method delivers good performance over the long term, however the ﬁxed
allocation ratio means that investors with preference for more weight on stocks need to tolerate
potentially large drawdowns during dull markets.

Mean-variance analysis or MPT [20] is used for many institutional portfolios that solves a constraint
optimisation problem to derive portfolio weights. Despite its popularity, the assumptions of the
theory are under criticism as they are often not obeyed in real ﬁnancial markets. In particular, returns
are assumed to follow a Gaussian distribution in MPT, therefore, investors only consider expected
return and variance of the portfolio returns to make decisions. However, it is widely accepted (see

2

S&BS&VS&CB&VB&CV&C0.80.60.40.20.0-0.2-0.6-0.4-0.8for instance [4, 38]) that returns tend to have fat tails and extreme losses are more likely to occur in
practice, leading to severe drawdowns that are not bearable. The Maximum Diversiﬁcation (MD)
portfolio is another promising method introduced in [3] that aims to maximise the diversiﬁcation of
a portfolio, thereby aiming to have minimally correlated assets so the portfolio can achieve higher
returns (and lower risk) than other classical methods. We compare our model with both these
strategies, with the results suggesting that our methods deliver better performance and tolerate larger
transaction costs than either of these benchmarks.

Stochastic Portfolio Theory (SPT) was recently proposed in [7, 9]. Unlike other methods, SPT
aims to achieve relative arbitrages meaning to select portfolios that can outperform a market index
with probability one. Such investment strategies have been studied in [5, 6, 27, 36]. However, the
number of relative arbitrage strategies remains small, as theory does not suggest how to construct such
strategies. We can check whether a given strategy is a relative arbitrage, but it is non-trivial to develop
one ex ante. In this work, we include a particular class of SPT called functionally generated portfolio
(FGP) [8] in our experiment, but the result suggests this method delivers inferior performance than
other algorithms and generates large turnovers, making it unproﬁtable under heavy transaction costs.

The idea of our end-to-end training framework was ﬁrst initiated in [25, 24]. However, these works
mainly focus on optimising the performance for a single asset so there is little discussion on how
portfolios should be maximised. Furthermore, their testing period is from 1970 to 1994, whereas
our dataset is up to date and we study the behavior of our strategy under the current crisis due to
COVID-19. We can also link our approach to reinforcement learning (RL) [31, 22, 35] where an agent
interacts with an environment to maximise cumulative rewards. The works of [1, 15, 39] have studied
this stream and adopted RL to design trading strategies. However, the goal of RL is to maximise
expected cumulative rewards such as proﬁts whereas Sharpe ratio can not be directly optimised.

3 Methodology

In this section, we introduce our framework and discuss how Sharpe ratio can be optimised through
gradient ascent. We discuss the types of neural networks used and detail the functionality of each
component in our method.

3.1 Objective Function

The Sharpe ratio is used to gauge the return per risk of a portfolio and is deﬁned as expected return
over volatility (excluding risk-free rate for simplicity):

L =

E(Rp)
Std(Rp)

(1)

where E(Rp) and Std(Rp) are the estimates of the mean and standard deviation of portfolio returns.
Speciﬁcally, for a trading period of t = {1, · · · , T }, we can maximise the following objective
function:

LT =

(cid:113)

E(Rp,t)

E(R2

p,t) − (E(Rp,t))2

E(Rp,t) =

1
T

T
(cid:88)

t=1

Rp,t

where Rp,t is realized portfolio return over n assets at time t denoted as:

Rp,t =

n
(cid:88)

i=1

wi,t−1 · ri,t

(2)

(3)

where ri,t is the return of asset i with ri,t = (pi,t/pi,t−1 − 1). We represent the allocation ratio
(position) of asset i as wi,t ∈ [0, 1] and (cid:80)n
i wi,t = 1. In our approach, a neural network f with
parameters θ is adopted to model wi,t for a long-only portfolio:

wi,t = f (θ|xt)

3

(4)

where xt represents the current market information and we bypass the classical forecasting step by
linking the inputs with positions to maximise the Sharpe over trading period T , namely LT . However,
a long-only portfolio imposes constraints that require weights to be positive and summed to one, we
use softmax outputs to fulﬁll these requirements:

wi,t =

exp( ˜wi,t)
j exp( ˜wj,t)

(cid:80)n

, where ˜wi,t are the raw weights.

(5)

Such a framework can be optimised using unconstrained optimisation methods. Particularly, we use
gradient ascent to maximise the Sharpe ratio. The gradient of LT with respect to parameters θ is
readily calculable, with an excellent derivation presented in [25, 23]. Once we obtain ∂LT /∂θ, we
can repeatedly compute this value from training data and update the parameters by using gradient
ascent:

θnew := θold + α

∂LT
∂θ

(6)

where α is the learning rate and the process can be repeated for many epochs until the convergence of
Sharpe ratio or the optimisation of validation performance is achieved.

3.2 Model Architecture

We depict our network architecture in Figure 2. Our model consists of three main building blocks:
input layer, neural layer and output layer. The idea of this design is to use neural networks to extract
cross-sectional features from input assets. Features extracted from deep learning models have been
suggested to perform better than traditional hand-crafted features [39]. Once features have been
extracted, the model outputs portfolio weights and we obtain realised returns to maximise Sharpe
ratio. The following details each component of our method.

Input layer We denote each asset as Ai and we have n assets to form a portfolio. A single input is
prepared by concatenating information from all assets. For example, the input features of one asset
can be its past prices and returns with a dimension of (k, 2) where k represents the lookback window.
By stacking features across all assets, the dimension of the resulting input would be (k, 2 × n). We
can then feed this input to the network and expect non-linear features being extracted.

Neural layer A series of hidden layers can be stacked to form a network, however, in practice,
this part requires lots of experiments as there are plentiful ways of combining hidden layers and
the performance often depends on the design of architecture. We have tested deep learning models
including fully connected neural network (FCN), convolutional neural network (CNN) and Long
Short-Term Memory (LSTM) [14]. Overall, LSTMs deliver the best performance for modelling daily
ﬁnancial data and a number of works [33, 19, 39] support this observation.

We note the problem of FCN is its problem of severe overﬁtting. As it assigns parameters to each
input feature, this results in an excess number of parameters. The LSTM operates with a cell structure
that has gate mechanisms to summarise and ﬁlter information from its long history, so the model ends
up with fewer trainable parameters and achieves better generalisation results. In contrast, CNNs with
a strong smoothing (typical of large convolutional ﬁlters) tend to have underﬁtting problems, such
that oversmooth solutions are obtained. Due to the design of parameter sharing and the convolution
operations, we experience CNNs to overﬁlter the inputs. However, we note that CNNs appear to be
excellent candidates for modelling high-frequency ﬁnancial data such as limit order books [37].

In order to construct a long-only portfolio, we use the softmax activation function for
Ouput layer
the output layer, which naturally imposes constraints to keep portfolio weights positive and summing
to one. The number of output nodes (w1, · · · , wn) is equal to the number of assets in our portfolio,
and we can multiply these portfolio weights with associated assets’ returns (r1, · · · , rn) to calculate
realised portfolio returns (Rp). Once realised returns are obtained, we can derive the Sharpe ratio
and calculate the gradients of the Sharpe ratio with respect to the model parameters and use gradient
ascent to update the parameters.

4

Figure 2: Model architecture schematic. Overall, our model contains three main building blocks:
input layer, neural layer and output layer.

4 Experiments

4.1 Description of Dataset

We use four market indices: US total stock index (VTI), US aggregate bond index (AGG), US
commodity index (DBC) and Volatility Index (VIX). These are popular Exchange-Traded Funds
(ETFs) [11] that have existed for more than 15 years. As discussed in Section 1, trading indices offers
advantages over trading individual assets because these indices are generally uncorrelated resulting
in diversiﬁcation. A diversiﬁed portfolio delivers a higher return per risk and the idea of our strategy
is to have a system that delivers good reward-to-risk ratio. Our dataset ranges from 2006 to 2020 and
contains daily observations. We retrain our model at every 2 years and use all data available up to
that point to update parameters. Overall, our testing period is from 2011 to the end of April 2020,
including the most recent crisis due to COVID-19.

4.2 Baseline Algorithms

We compare our method with a group of baseline algorithms. The ﬁrst set of baseline models are
reallocation strategies adopted by many pension funds. These strategies assign a ﬁxed allocation
ratio to relevant assets and rebalance portfolios annually to maintain these ratios. Investors can
select a portfolio based on their risk preferences. In general, portfolios weighted more on equities
would deliver better performance at the expense of larger volatility. In this work, we consider four
such strategies: Allocation 1 (25% shares, 25% bonds, 25% commodities and 25% volatility index),
Allocation 2 (50% shares, 10% bonds, 20% commodities, and 20% volatility index), Allocation 3
(10% shares, 50% bonds, 20% commodities, and 20% volatility index), and Allocation 4 (40% shares,
40% bonds, 10% commodities and 10% volatility index).

The second set of comparison models are mean-variance optimisation (MV) [20] and maximum
diversiﬁcation (MD) [32]. We use moving averages with a rolling window of 50 days to estimate
the expected returns and covariance matrix. The portfolio weights are updated at a daily basis and
we select weights that maximise Sharpe ratio for MV. The last baseline algorithm is the diversity-
weighted portfolio (DWP) from Stochastic Portfolio Theory presented in [28]. The DWP relates

5

!"!#!$Hidden LayerHidden Layer%"%$&"%$'(Output layerNeural layerInput layer)"Softmax)#)$portfolio weights to assets’ market capitalisation and it has been suggested to be able to outperform
the market index with certainty [10].

4.3 Training Scheme

In this work, we use a single layer of LSTM connectivity, with 64 units, to model the portfolio
weights and thence to optimise the Sharpe ratio. We purposely keep our network simple to indicate
the effectiveness of this end-to-end training pipeline instead of carefully ﬁne-tuning the “right”
hyperparameters. Our input contains close prices and daily returns for each market index and we
take the past 50 days of these observations to form a single input. We are aware that returns can
be derived from prices, but keeping returns help with the evaluation of Equation 7 and we can also
treat them as momentum features in [26]. As our focus is not on feature selection, we choose these
commonly used features in our work. The Adam optimiser [17] is used for training our network, and
the mini-batch size is 64. We take 10% of any training data as a separate validation-set to optimise
hyperparameters and control overﬁtting problems. Any hyperparameter optimisation is done on the
validation set, leaving the test data for the ﬁnal performance evaluation and ensuring the validity of
our results. In general, our training process stops after 100 epochs.

4.4 Experimental Results

When reporting the test performance, we include transaction costs and use volatility scaling [26,
19, 39] to scale our positions based on market volatility. We can set our own volatility target and
meet expectations of investors with different risk preferences. Once volatilities are adjusted, our
investment performances are mainly driven by strategies instead of being heavily affected by markets.
The modiﬁed portfolio return can be deﬁned as:

Rp,t =

n
(cid:88)

i

σtgt
σi,t−1

wi,t−1 · ri,t − C ·

n
(cid:88)

i

(cid:12)
(cid:12)
(cid:12)

σtgt
σi,t−1

wi,t−1 −

σtgt
σi,t−2

wi,t−2

(cid:12)
(cid:12)
(cid:12)

(7)

where σtgt is the volatility target and σi,t−1 is an ex-ante volatility estimate of asset i calculated
using an exponentially weighted moving standard deviation with a 50-day window on ri,t. We use
daily changes of traded value of an asset to represent transaction costs, which is calculated by the
second term in Equation 7. C (=1bs=0.0001) is the cost rate and we change it to reﬂect how our
model performs under different transaction costs.

To evaluate the performance of our methods, we utilise following metrics: expected return (E(R)),
standard deviation of return (Std(R)), Sharpe ratio [29], downside deviation of return (DD(R)) [21],
and Sortino ratio [30]. All of these metrics are annualised, and we also report on maximum drawdown
(MDD) [2], percentage of positive return (% of + Ret) and the ratio between positive and negative
return (Ave. P / Ave. L).

Table 1 presents the results of our model (DLS) compard to other baseline algorithms. The top of the
table shows the results without using volatility scaling, and we can see that our model (DLS) achieves
the best Sharpe’s ratio and Sortino ratio, delivering the highest return per risk. However, given the
large differences in volatilities, we can not directly compare expected and cumulative returns for
different methods, thereby volatility scaling also helps to make fair comparisons.

Once volatilities are scaled (shown in the middle of Table 1), DLS delivers the best performance
across all evaluation metrics except for a slightly larger drawdown. If we look at the cumulative
returns in Figure 3, DLS exhibits outstanding performance over the long haul and the maximum
drawdown is reasonable, ensuring the conﬁdence of investors to hold through hard times. Further, if
we look at the bottom of Table 1 where a large cost rate (C = 0.1%) is used, our model (DLS) stills
delivers the best expected return and achieves the highest Sharpe and Sortino ratios.

However, with a higher cost rate, we can see that reallocation strategies work well and, in particular,
Allocations 3 and 4 achieve comparable results to our method. In order to investigate why performance
gap diminishes with a higher cost rate, we present the boxplots for annual realised trade returns and
accumulated costs for different assets in Figure 4. Overall, our model delivers better realised returns
than reallocation strategies, but we also accumulate much larger transaction costs since our positions
are adjusted on a daily basis, leading to a higher turnover.

6

Table 1: Experiment results for different algorithms.

E(R)

Std(R)

Sharpe DD(R)

Sortino MDD % of + Ret

Allocation 1
Allocation 2
Allocation 3
Allocation 4
MV
MD
DWP
DLS

Allocation 1
Allocation 2
Allocation 3
Allocation 4
MV
MD
DWP
DLS

0.282
0.249
0.228
0.152
0.082
0.462
0.051
0.313

0.160
0.123
0.145
0.164
0.112
0.157
0.089
0.206

No volatility scaling and C = 0.01%

0.303
0.212
0.256
0.123
0.108
0.523
0.102
0.168

0.929
1.173
0.890
1.228
0.759
0.882
0.493
1.858

0.136
0.095
0.116
0.052
0.069
0.239
0.067
0.099

2.065
2.616
1.962
2.932
1.192
1.931
0.740
3.135

0.142
0.097
0.122
0.081
0.195
0.273
0.179
0.102

Volatility scaling (σtgt = 0.10) and C = 0.01%

0.105
0.106
0.105
0.104
0.100
0.106
0.109
0.105

1.526
1.146
1.383
1.579
1.120
1.484
0.818
1.962

0.061
0.065
0.061
0.064
0.063
0.065
0.069
0.062

2.629
1.861
2.396
2.588
1.767
2.414
1.291
3.322

0.111
0.127
0.105
0.112
0.211
0.125
0.115
0.123

Volatility scaling (σtgt = 0.10) and C = 0.1%

Allocation 1
Allocation 2
Allocation 3
Allocation 4
MV
MD
DWP
DLS

0.133
0.105
0.117
0.135
0.019
0.095
-0.083
0.148

0.105
0.107
0.105
0.104
0.101
0.106
0.110
0.105

1.274
0.986
1.110
1.299
0.191
0.899
-0.753
1.403

0.061
0.066
0.061
0.064
0.066
0.066
0.074
0.063

2.172
1.590
1.903
2.108
0.293
1.431
-1.129
2.327

0.113
0.244
0.107
0.114
0.324
0.145
0.627
0.125

0.479
0.483
0.476
0.505
0.562
0.473
0.549
0.537

0.554
0.549
0.542
0.565
0.561
0.565
0.556
0.559

0.548
0.547
0.538
0.559
0.537
0.549
0.508
0.547

Ave. P
Ave. L

1.193
1.254
1.183
1.349
1.199
1.182
1.107
1.518

1.289
1.211
1.259
1.303
1.213
1.297
1.148
1.375

1.236
1.179
1.203
1.244
1.033
1.171
0.880
1.272

For reallocation strategies, daily position changes are only updated for volatility scaling. Otherwise,
we only actively change positions once a year to rebalance and maintain the allocation ratio. As a
result, reallocation strategies deliver minimal transaction costs. This analysis aims to indicate the
validity of our results and show that our method can work under unfavorable conditions.

4.5 Model Performance during 2020 Crisis

Due to the recent COVID-19 pandemic, global stock markets fell dramatically and experienced
extreme volatility. The crash started on the 24th February 2020 where markets reported their largest

Figure 3: Cumulative returns (logarithmic scale) for Left: no volatility scaling and C = 0.01%;
Middle: volatility scaling (σtgt = 0.10) and C = 0.01%; Right: volatility scaling (σtgt = 0.10) and
C = 0.1%.

7

20112012201320142015201620172018201920200.00.51.01.52.02.53.0Allocation 1Allocation 2Allocation 3Allocation 4MVMDDWPDLS20112012201320142015201620172018201920200.000.250.500.751.001.251.501.75Allocation 1Allocation 2Allocation 3Allocation 4MVMDDWPDLS20112012201320142015201620172018201920201.00.50.00.51.0Allocation 1Allocation 2Allocation 3Allocation 4MVMDDWPDLSFigure 4: Boxplot for Top: annual realised trade returns; Bottom: annual accumulated costs for
different assets with volatility scaling (σtgt = 0.10) and C = 0.01%.

one-week declines since the 2008 ﬁnancial crisis. Later on, with an oil price war between Russia and
the OPEC countries, markets further dampened and encountered the largest single-day percentage
drop since Black Monday in 1987. As of March 2020, we have seen a downturn of at least 25% in
the US markets and 30% in most G20 countries. The crisis shattered many investors’ conﬁdence and
resulted in a great loss of their wealths. However, it also provides us with a great opportunity to stress
test our method and understand how our model performs during the crisis.

In order to study the model behaviours, we plot how our algorithm allocated the assets from January
to April 2020 in Figure 5. At the beginning of 2020, we can see that our model had a quite diverse
holding. However, after a small dip in stock index in early February, we almost had only bonds in our
portfolio. There were some equity positions left but very small positions for volatility and commodity
indices. When the crash started on 24th February, our holdings were concentrated on the bond index
which is considered to be safe assets during the crisis. Interestingly, the bond index also fell this time
(in the middle of March) although it rebounded quite quickly. During the bond falling, our original
positions did not change much but the scaled positions decreased a lot for the bond index due to a
spiking volatility, therefore our drawdown was small. Overall, we can see that our model delivers
reasonable allocations during the crisis and our positions are protected through volatility scaling.

4.6 Sensitivity Analysis

In order to understand how input features affect our decisions, we study the sensitivity analysis
presented in [24] for our method. The absolute normalised sensitivity of feature xi is deﬁned as:

Si =

dL
dxi
(cid:12)
(cid:12)
(cid:12)

maxj

(cid:12)
(cid:12)
(cid:12)

(8)

dL
dxj

where L represents the objective function and Si captures the relative sensitivity for feature xi
compared with other features. We plot the time-varying sensitivities for all features in Figure 6. The
y-axis indicates the 400 features we have because we use 4 indices (each with prices and returns) and

8

StockBondVolatilityCommodityAsset Classes0.40.20.00.20.4Realised ReturnsAllocation 1Allocation 2Allocation 3Allocation 4MVMDDWPDLSStockBondVolatilityCommodityAsset Classes0.0000.0020.0040.0060.0080.010Annual CostsAllocation 1Allocation 2Allocation 3Allocation 4MVMDDWPDLSFigure 5: Shifts of portfolio weights for our model (DLS) during the crisis of COVID-19 with
volatility scaling (σtgt = 0.10).

we take a timeframe of past 50 observations to form a single input so there are 400 features in total.
The row labeled “Sprice” represents price features for the stock index and the bottom of row “Sprice”
means the most recent price for that observation. Same convention is used for all other features.

The importance of features varies over the time, but the most recent features always make the
biggest contributions as we can see that the bottom of each feature row has the highest weight. This
observation meets our understanding as, for time-series, recent observations carry more information.
The further away from the current observation point, the less importance of features show and we
could adjust features used based on this observation such as using a small lookback window.

Figure 6: Sensitivity analysis for input features over the time.

9

2020-01-012020-01-152020-02-012020-02-152020-03-012020-03-152020-04-012020-04-15110120130140150160170Stock0.00.20.40.60.81.01.2PositionScaled position2020-01-012020-01-152020-02-012020-02-152020-03-012020-03-152020-04-012020-04-15106108110112114116118Bond123456PositionScaled position2020-01-012020-01-152020-02-012020-02-152020-03-012020-03-152020-04-012020-04-151020304050607080Volatility0.000.020.040.060.080.100.120.140.16PositionScale position2020-01-012020-01-152020-02-012020-02-152020-03-012020-03-152020-04-012020-04-15111213141516Commodity0.000.050.100.150.200.25PositionScaled position0.50.40.30.20.15 Conclusion

In this work, we adopt deep learning models to directly optimise a portfolio’s Sharpe ratio. This
pipeline bypasses the traditional forecasting step and allows us to optimise portfolio weights by
updating model parameters through gradient ascent. Instead of using individual assets, we focus on
ETFs of market indices to form a portfolio. Doing this substantially reduces the scope of possible
assets to choose from, and these indices have shown robust correlations. In this work, four market
indices have been used to form a portfolio.

We compare our method with a wide range of popular algorithms including reallocation strategies,
classical mean-variance optimisation, maximum diversiﬁcation and stochastic portfolio theory model.
Our testing period is from 2011 to the April of 2020, and include the recent crisis due to COVID-19.
The results show that our model delivers the best performance and a detailed study of our model
performance during the crisis shows the rationality and practicability of our method. A sensitivity
analysis is included to understand how input features contribute to outputs and the observations meet
our econometric understanding, showing the most recent features are most relevant.

In subsequent continuation of this work, we aim to study portfolios performance under different
objective functions. Given the ﬂexible framework of our approach, we can maximise Sortino ratio or
even the diversiﬁcation degree of a portfolio as long as functions are differentiable. We further note
that the volatility estimates used for scaling are lagged estimates that do not necessarily represent
current market volatilities. We consider another extension to this work to thus adapt the network
architecture to infer (future) volatility estimates as a part of the training process.

Acknowledgements

The authors would like to thank members of Machine Learning Research Group at the University of
Oxford for their useful comments. We are most grateful to the Oxford-Man Institute of Quantitative
Finance for support and data access.

References

[1] Francesco Bertoluzzo and Marco Corazza. Testing different reinforcement learning conﬁgura-
tions for ﬁnancial trading: Introduction and applications. Procedia Economics and Finance,
3:68–77, 2012.

[2] Alexei Chekhlov, Stanislav Uryasev, and Michael Zabarankin. Drawdown measure in portfolio
optimization. International Journal of Theoretical and Applied Finance, 8(01):13–58, 2005.
[3] Yves Choueifaty and Yves Coignard. Toward maximum diversiﬁcation. The Journal of Portfolio

Management, 35(1):40–51, 2008.

[4] Rama Cont and De Nitions. Statistical properties of ﬁnancial time series. 1999.
[5] Daniel Fernholz and Ioannis Karatzas. On optimal arbitrage. The Annals of Applied Probability,

pages 1179–1204, 2010.

[6] Daniel Fernholz, Ioannis Karatzas, et al. Optimal arbitrage under model uncertainty. The Annals

of Applied Probability, 21(6):2191–2225, 2011.

[7] E Robert Fernholz. Stochastic portfolio theory. In Stochastic Portfolio Theory, pages 1–24.

Springer, 2002.

[8] Robert Fernholz. Portfolio generating functions. In Quantitative Analysis in Financial Markets:
Collected Papers of the New York University Mathematical Finance Seminar, pages 344–367.
World Scientiﬁc, 1999.

[9] Robert Fernholz and Ioannis Karatzas. Stochastic portfolio theory: An overview. Handbook of

numerical analysis, 15:89–167, 2009.

[10] Robert Fernholz, Ioannis Karatzas, and Constantinos Kardaras. Diversity and relative arbitrage

in equity markets. Finance and Stochastics, 9(1):1–27, 2005.

[11] Gary L Gastineau. Exchange-traded funds. Handbook of ﬁnance, 1, 2008.
[12] Ian Goodfellow, Yoshua Bengio, and Aaron Courville. Deep learning. MIT press, 2016.

10

[13] Campbell R Harvey, Edward Hoyle, Russell Korgaonkar, Sandy Rattray, Matthew Sargaison,
and Otto Van Hemert. The impact of volatility targeting. The Journal of Portfolio Management,
45(1):14–33, 2018.

[14] Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural computation,

9(8):1735–1780, 1997.

[15] Chien Yi Huang. Financial trading as a game: A deep reinforcement learning approach. arXiv

preprint arXiv:1807.02787, 2018.

[16] Johan Ludwig William Valdemar Jensen et al. Sur les fonctions convexes et les inégalités entre

les valeurs moyennes. Acta mathematica, 30:175–193, 1906.

[17] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. Proceedings

of the International Conference on Learning Representations, 2015.

[18] Yann LeCun, Yoshua Bengio, and Geoffrey Hinton. Deep learning. Nature, 521(7553):436–444,

2015.

[19] Bryan Lim, Stefan Zohren, and Stephen Roberts. Enhancing time-series momentum strategies
using deep neural networks. The Journal of Financial Data Science, 1(4):19–38, 2019.

[20] Harry Markowitz. Portfolio selection. The journal of ﬁnance, 7(1):77–91, 1952.
[21] Alexander J McNeil, Rüdiger Frey, and Paul Embrechts. Quantitative risk management:

Concepts, techniques and tools-revised edition. Princeton university press, 2015.

[22] Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan
Wierstra, and Martin Riedmiller. Playing Atari with deep reinforcement learning. NIPS Deep
Learning Workshop 2013, 2013.

[23] Gabriel Molina. Stock trading with recurrent reinforcement learning (RRL). CS229, nd Web,

15, 2016.

[24] John Moody and Matthew Saffell. Learning to trade via direct reinforcement. IEEE transactions

on neural Networks, 12(4):875–889, 2001.

[25] John Moody, Lizhong Wu, Yuansong Liao, and Matthew Saffell. Performance functions and
reinforcement learning for trading systems and portfolios. Journal of Forecasting, 17(5-6):441–
470, 1998.

[26] Tobias J Moskowitz, Yao Hua Ooi, and Lasse Heje Pedersen. Time series momentum. Journal

of ﬁnancial economics, 104(2):228–250, 2012.

[27] Johannes Ruf. Hedging under arbitrage. Mathematical Finance: An International Journal of

Mathematics, Statistics and Financial Economics, 23(2):297–317, 2013.

[28] Yves-Laurent Kom Samo and Alexander Vervuurt. Stochastic portfolio theory: A machine
In Proceedings of the Thirty-Second Conference on Uncertainty in

learning perspective.
Artiﬁcial Intelligence, pages 657–665, 2016.

[29] William F Sharpe. The sharpe ratio. Journal of portfolio management, 21(1):49–58, 1994.
[30] Frank A Sortino and Lee N Price. Performance measurement in a downside risk framework.

the Journal of Investing, 3(3):59–64, 1994.

[31] Richard S Sutton and Andrew G Barto. Reinforcement learning: An introduction. MIT press,

2018.

[32] Ludan Theron and Gary Van Vuuren. The maximum diversiﬁcation investment strategy: A
portfolio performance comparison. Cogent Economics & Finance, 6(1):1427533, 2018.
[33] Avraam Tsantekidis, Nikolaos Passalis, Anastasios Tefas, Juho Kanniainen, Moncef Gabbouj,
and Alexandros Iosiﬁdis. Using deep learning to detect price change indications in ﬁnancial
markets. In 2017 25th European Signal Processing Conference (EUSIPCO), pages 2511–2515.
IEEE, 2017.

[34] Russell Wild. Index Investing for Dummies. John Wiley & Sons, 2008.
[35] Ronald J Williams. Simple statistical gradient-following algorithms for connectionist reinforce-

ment learning. Machine learning, 8(3-4):229–256, 1992.

[36] Ting-Kam Leonard Wong. Optimization of relative arbitrage. Annals of Finance, 11(3-4):345–

382, 2015.

11

[37] Zihao Zhang, Stefan Zohren, and Stephen Roberts. DeepLOB: Deep convolutional neural
networks for limit order books. IEEE Transactions on Signal Processing, 67(11):3001–3012,
2019.

[38] Zihao Zhang, Stefan Zohren, and Stephen Roberts. Extending deep learning models for
limit order books to quantile regression. Proceedings of Time Series Workshop of the 36 th
International Conference on Machine Learning, Long Beach, California, PMLR 97, 2019.,
2019.

[39] Zihao Zhang, Stefan Zohren, and Roberts Stephen. Deep reinforcement learning for trading.

The Journal of Financial Data Science, 2020.

[40] Eric Zivot. Introduction to computational ﬁnance and ﬁnancial econometrics. Chapman & Hall

Crc, 2017.

12
