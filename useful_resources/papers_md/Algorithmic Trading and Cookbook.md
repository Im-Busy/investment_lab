# Algorithmic Trading and Cookbook

> *Source PDF: Algorithmic Trading and Cookbook.pdf*
> *Extraction: Combined — previous structured extraction (base) + markitdown raw text*

---

# Algorithmic Trading and Cookbook

> *Source PDF: Algorithmic Trading and Cookbook.pdf*
> *Extraction: Combined — markitdown raw text (base) + previous extraction supplements*

---

Algorithmic Trading and
Quantitative Strategies

Algorithmic Trading and
Quantitative Strategies

Raja Velu
Department of Finance
Whitman School of Management
Syracuse University

Maxence Hardy
eTrading Quantitative Research
J.P. Morgan

Daniel Nehren
Statistical Modeling & Development
Barclays

First edition published 2020
by CRC Press
6000 Broken Sound Parkway NW, Suite 300, Boca Raton, FL 33487-2742

and by CRC Press
2 Park Square, Milton Park, Abingdon, Oxon, OX14 4RN

© 2020 Taylor & Francis Group, LLC

CRC Press is an imprint of Taylor & Francis Group, LLC

Reasonable eﬀorts have been made to publish reliable data and information, but the author and publisher
cannot assume responsibility for the validity of all materials or the consequences of their use. The authors
and publishers have attempted to trace the copyright holders of all material reproduced in this publication
and apologize to copyright holders if permission to publish in this form has not been obtained. If any
copyright material has not been acknowledged please write and let us know so we may rectify in any
future reprint.

Except as permitted under US Copyright Law, no part of this book may be reprinted, reproduced, trans-
mitted, or utilized in any form by any electronic, mechanical, or other means, now known or hereafter
invented, including photocopying, microﬁlming, and recording, or in any information storage or retrieval
system, without written permission from the publishers.

For permission to photocopy or use material electronically from this work, access www.copyright.com or
contact the Copyright Clearance Center, Inc. (CCC), 222 Rosewood Drive, Danvers, MA 01923, 978-750-
8400. For works that are not available on CCC please contact mpkbookspermissions@tandf.co.uk

Trademark notice: Product or corporate names may be trademarks or registered trademarks, and are used
only for identiﬁcation and explanation without intent to infringe.

Library of Congress Control Number: 2020932899

ISBN: 9781498737166 (hbk)
ISBN: 9780429183942 (ebk)

Typeset in STIXGeneral
by Nova Techset Private Limited, Bengaluru & Chennai, India

Contents

Preface

I

Introduction to Trading

1 Trading Fundamentals

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

1.3 The Mechanics of Trading .

.
1.2.1 Equity Markets Participants
1.2.2 Watering Holes of Equity Markets .
.

.
.
.
1.3.1 How Double Auction Markets Work .
.
.
1.3.2 The Open Auction .
.
1.3.3 Continuous Trading .
.
1.3.4 The Closing Auction .

.
.
.
.
.
.
.
1.4 Taxonomy of Data Used in Algorithmic Trading
.
.
.

1.1 A Brief History of Stock Trading
.
1.2 Market Structure and Trading Venues: A Review .
.
.
.
.
.
.
.
.
.
.
.
.

.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
Fundamental Data and Other Data Sets
1.5 Market Microstructure: Economic Fundamentals of Trading .
.

1.4.1 Reference Data .
.
.
.
1.4.2 Market Data
1.4.3 Market Data Derived Statistics .
1.4.4

.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
. .
.
.

1.5.1 Liquidity and Market Making .

.
.
.
.
.
.
.
.
.
.
.
.
.
.

.
.
.
.
.
.
.
.
.
.
.
.
.
.

.
.
.
.
.
.
.
.
.
.
.
.
.
.

.
.
.
.
.
.
.

.
.
.
.

.
.
.

.
.
.

.
.
.

.
.
.

.
.
.

.
.
.

.
.
.

.
.
.

.
.
.

.
.

.
.

.
.

.
.

.
.

.
.

.
.

.

.

.

.

.

.

.

.

.

.

.

.

.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.

.
.
.
.
.
.
. .
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.

II

Foundations: Basic Models and Empirics

2 Univariate Time Series Models

.

.

.

.

.

.

.

.

.

Processes to Discrete Time Series

2.1 Trades and Quotes Data and Their Aggregation: From Point
.
.
.
.

.
. .
2.2 Trading Decisions as Short-Term Forecast Decisions
.
.
.
2.3 Stochastic Processes: Some Properties .
.
.
.
. .
.
2.4 Some Descriptive Tools and Their Properties
2.5 Time Series Models for Aggregated Data: Modeling the Mean .
2.6 Key Steps for Model Building .
.
.
2.7 Testing for Nonstationary (Unit Root) in ARIMA Models: To
.
.
.
.

.
.
.
2.8 Forecasting for ARIMA Processes .
2.9 Stylized Models for Asset Returns .
.
2.10 Time Series Models for Aggregated Data: Modeling the Variance .

Diﬀerence or Not To .

.
.
.
.
. .
.
.

.
.
.
.
.
.

.
.
.
.

.
.
.

.
.
.

.
.
.

.
.
.

.
.
.

.
.
.

.
.
.

.
.
.

.
.
.

.
.
.

.
.
.

.
.
.

.
.
.

.
.
.

.
.

.
.

.
.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.
.
.
.
.
.

.
.
.
.

xi

1

3
3
7
7
8
13
13
14
19
24
26
26
35
39
42
44
45

51

53

54
56
57
61
63
70

77
78
82
84

v

vi

2.11 Stylized Models for Variance of Asset Returns
.
.
.
2.12 Exercises

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

Contents

.
.

.
.

.
.

.
.

.
.

.
.

.
.

.
.

.
.

.
.

.
.

.

90
92

.

.

.
.
.

.
.
.

3 Multivariate Time Series Models
.
.
.
3.1 Multivariate Regression
.
3.2 Dimension-Reduction Methods
3.3 Multiple Time Series Modeling
.
3.4 Co-Integration, Co-Movement and Commonality in Multiple Time
.
.
.
.
.
.
.

.
.
.
3.5 Applications in Finance
.
3.6 Multivariate GARCH Models
.
Illustrative Examples .
3.7
.
.
.
3.8 Exercises

.
.
.
.
.
.
.
.
. .

.
.
. .
.
.
.
.
.
.

.
.
. .
.
.

Series

.
.
.
.
.

.
.
.
.
.

.
.
.
.
.

.
.
.
.
.

.
.
.
.
.

.
.
.
.
.

.
.
.
.
.

.
.
.
.
.

.
.
.
.
.

.
.
.
.
.

.
.
.
.
.

.
.
.
.
.

.
.
.
.
.

.
.
.
.
.

.
.
.

.
.
.

.
.
.

.
.
.

.
.
.

.
.
.

.
.
.

.
.
.

.
.
.

.
.
.

.
.
.

.
.
.

.
.
.

.
.

.
.

.
.

.
.

.

.

.

.

.

.

.

.

.

.

.

.

4 Advanced Topics

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.
.
.
.

.
.
.
.

.
.
.
.

.
.
.
.

.
.
.
.

.
.
. .
.
.
.
.

4.5 Analysis of Time Aggregated Data

4.1 State-Space Modeling
.
4.2 Regime Switching and Change-Point Models
4.3 A Model for Volume-Volatility Relationships
.
4.4 Models for Point Processes .

.
.
.
.
.
Stylized Models for High Frequency Financial Data .

.
.
.
.
4.4.1
.
4.4.2 Models for Multiple Assets: High Frequency Context .
.
.
.
.
.
.
.
.

.
4.5.1 Realized Volatility and Econometric Models
.
4.5.2 Volatility and Price Bar Data .
.
.
4.6 Analytics from Machine Learning Literature
.
.
.
4.6.1 Neural Networks .
4.6.2 Reinforcement Learning .
.
.
4.6.3 Multiple Indicators and Boosting Methods .
.
.
.

4.7 Exercises

.
.
.
.
.
.
.
.

.
.
.
.
.
.
.
.

.
.
.
.
.
.
.
.

.
.
.
.
.
.
.
.

.
.
.
.
.
.

.
.
.
.

.
.
.
.

. .

.
.

.
.

.
.

.
.

.
.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
. .
.
.
.
.

III Trading Algorithms

5 Statistical Trading Strategies and Back-Testing

.

.

.

.

.

.

.

.

.
.

.
.
.

.
.
.

.
.
.

.
.
.
.

.
.
.
.
.

.
.
.
.
.

Filter Rules .

5.1
5.2 Evaluation of Strategies: Various Measures
5.3 Trading Rules for Time Aggregated Data
.

.
.
5.3.1
.
.
.
5.3.2 Moving Average Variants and Oscillators .

Introduction to Trading Strategies: Origin and History
.
.
.
.

.
.
.
.
.
.
.
.
.
.
.
5.4 Patterns Discovery via Non-Parametric Smoothing Methods
.
.
5.5 A Decomposition Algorithm .
5.6 Fair Value Models
.
.
.
.
.
5.7 Back-Testing and Data Snooping: In-Sample and Out-of-Sample
.
.
.
.
.
.
.
.

.
.
.
5.8.1 Distance-Based Algorithms
.
5.8.2 Co-Integration .

Performance Evaluation
.

5.8 Pairs Trading .

.
.
.
.
.
.
.
.

.
.
.
.

.
.
.
.

.
.
.
.

.
.
.
.

.
.
.
.

.
.
.
.

.
.
.
.

.
.
.
.

.
.
.
.

.
.
.
.

.
.
.
.

.
.
.
.

.
.
.
.

.
.
.
.

.
.

.
.

.
.

.
.

.
.

.
.

.
.

.
.

.
.

.
.

.
.

.
.

.
.

.
.

.
.

.
.

.
.

.

.

.

.

.

.

.

.

.

.

.

.

.
.
.
.
.
.
.
.

.
.
.
.

95
96
.
.
99
. 104

. 106
. 110
. 112
. 114
. 122

125
. 125
. 128
. 131
. 134
. 136
. 140
. 141
. 141
. 143
. 146
. 147
. 150
. 152
. 154

157

159
. 159
. 160
. 161
. 162
. 164
. 166
. 168
. 170

. 171
. 174
. 175
. 176

Contents

.
.

.
.

.
.
.

.
.
.

.
.
.

.
.
.

.
.
.

5.8.3
5.8.4

Some General Comments
Practical Considerations .

.
.
.
.
5.9 Cross-Sectional Momentum Strategies .
.
5.10 Extraneous Signals: Trading Volume, Volatility, etc.
5.10.1 Filter Rules Based on Return and Volume .
.
5.10.2 An Illustrative Example
.
.
.
.
.

.
.
5.11 Trading in Multiple Markets
5.12 Other Topics: Trade Size, etc.
.
5.13 Machine Learning Methods in Trading
.
.
.
5.14 Exercises

.
.
.
.
.

.
.
.
.
.

.
.
.
.
.

.
.
.
.
.

.
.
.
.
.

.
.
.

.
.
.

.
.
.

.
.

.

.

.

.

.

.

.

.

.

.

.

.

.

.
.
.

.
.
.
.
.
.

.
.
.
.
.
.
.
.
.
.

.

.

.

.

.

.

6.1

.
.
.
.
.
.
.

.
Implications for Investing .
.

.
Introduction to Modern Portfolio Theory
.
6.1.1 Mean-Variance Portfolio Theory .
6.1.2 Multifactor Models .
.
.
6.1.3 Tests Related to CAPM and APT .
.
6.1.4 An Illustrative Example
.
6.1.5
.

6 Dynamic Portfolio Management and Trading Strategies
.
.
.
. .
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
Portfolio Allocation Using Regularization .
.
.
Portfolio Strategies: Some General Findings .
.
.
6.3 Dynamic Portfolio Selection .
.
6.4 Portfolio Tracking and Rebalancing .
.
.
6.5 Transaction Costs, Shorting and Liquidity Constraints
.
6.6 Portfolio Trading Strategies
.
.
.
6.7 Exercises

6.2 Statistical Underpinnings

6.2.1
6.2.2

.
.
.
.
.
.
.

.
.
.
.
.
.
.

.
.
.

.
.
.

.
.
.

.
.

.
.

.
.

.
.

.
.

.
.

.
.

.
.

.
.

.
.

.
.

.
.

.
.

.
.

.
.

.
.

.
.

.
.

.
.

.
.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.
.
.
.
.
.
.
.
.
.

.
.
.
.
.
.
.
.
.
.
.
.
.
.

.
.
.
.
.
.
.
.
.
.

.
.
.
.
.
.
.
.
.
.
.
.
.
.

.
.
.
.
.
.
.
.
.
.

.
.
.
.
.
.
.
.
.
.

.
.
.
.
.
.
.
.
. .
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.

.
.
.
.
.
.
.
.
.
.

.
.
.
.
.
.
.
.
.
.
.
.
.
.

7.1

7 News Analytics: From Market Attention and Sentiment to Trading
Introduction to News Analytics: Behavioral Finance and Investor
.
.
Cognitive Biases
.
.

.
.
.
.
7.2 Automated News Analysis and Market Sentiment
7.3 News Analytics and Applications to Trading
.
.
7.4 Discussion / Future of Social Media and News in Algorithmic
.
.

Trading .

.
.
.

.
.
.

.
.
.

.
.
.

.
.
.

.
.
.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

IV Execution Algorithms

8 Modeling Trade Data

8.1 Normalizing Analytics

.

.

.

.

.

.

.

.

.

.
.

.
.

.
.

.
.

.
.

.
.

.
.
8.1.1 Order Size Normalization: ADV .
.
8.1.2 Time-Scale Normalization: Characteristic Time .
8.1.3
8.1.4 Other Microstructure Normalizations
.
8.1.5
Intraday Normalization: Proﬁles .
.
8.1.6 Remainder (of the Day) Volume .
.
.
8.1.7 Auctions Volume .

.
.
.
Intraday Return Normalization: Mid-Quote Volatility .
.
.
.
.

.
.
.
.

.
.
.
.

.
.
.
.

.
.
.
.

.
.
.
.

.
.
.
.

.
.
.
.

.
.
.
.

. .

.
.
.

.
.
.

.
.
.

.
.

.

.

.

.

.

vii

. 180
. 181
. 184
. 189
. 191
. 193
. 198
. 202
. 204
. 207

215
. 215
. 215
. 219
. 219
. 222
. 224
. 226
. 230
. 232
. 234
. 236
. 239
. 242
. 244

251

. 251
. 256
. 258

. 267

269

271
. 271
. 272
. 273
. 275
. 276
. 277
. 282
. 283

.
.
.
.
.
.
.
.
.
.

.
.
.
.
.
.
.
.
.
.
.
.
.
.

.
.
.

.

.
.
.
.
.
.
.
.

.

.
.
.
.
.
.
.
.

viii

.

.

.

.

.

.

.

.

.

.

.

.

.

8.2 Microstructure Signals
.
8.3 Limit Order Book (LOB): Studying Its Dynamics .
.
.
.
.
.

8.3.1 LOB Construction and Key Descriptives .
.
.
8.3.2 Modeling LOB Dynamics
.
8.3.3 Models Based on Hawkes Processes .
.
.
.

8.4 Models for Hidden Liquidity .
.
8.5 Modeling LOB: Some Concluding Thoughts

.
.
.
.

.

.

.

.

.

.

.

.

.

.

.

.

9 Market Impact Models
.

Introduction .

.

.

.

.

.
.

.
.

.
.

.
.

.
.
.

.
.
.

.
.
.

.
.
.

.
.
.

.
.
.

.
.
9.1
.
.
9.2 What Is Market Impact?
.
9.3 Modeling Transaction Costs (TC)
.
9.4 Historical Review of Market Impact Research .
.
9.5 Some Stylized Models
.
.
9.6 Price Impact in the High Frequency Setting .
.
.
9.7 Models Based on LOB .
.
.
.
9.8 Empirical Estimation of Transaction Costs
.
.
9.8.1 Review of Select Empirical Studies

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.
.
.
.
.
.
.
.
.

.
.
.
.
.
.
.
.
.

.
.
.
.
.
.
.

.
.
.
.
.
.
.
.
.

.
.
.
.
.
.
.

.
.
.
.
.
.
.
.
.

10 Execution Strategies

.
.
.
.
.

.
.
.
.
.
.

10.3.1 Scheduling Layer .
.
10.3.2 Order Placement
.
10.3.3 Order Routing .

10.1 Execution Benchmarks: Practitioner’s View .
.
.
10.2 Evolution of Execution Strategies
.
10.3 Layers of an Execution Strategy . .
.
.
.
.
.
.

.
.
.
.
.
.
10.4 Formal Description of Some Execution Models .
.
.

.
.
.
.
.
.
.
.
.
10.5 Multiple Exchanges: Smart Order Routing Algorithm .
.
10.6 Execution Algorithms for Multiple Assets .
.
.
10.7 Extending the Algorithms to Other Asset Classes .

10.4.1 First Generation Algorithms .
.
10.4.2 Second Generation Algorithms .

.
.
.
.
. .

.
.
.
.
.
.
.
.
.

.
.
.
.
.
.
.
.
.

.
.
.
.
.

.
.
.
.
.

.
.
.
.
.

.
.
.

.
.
.

.
.

.
.

.
.

.
.

.
.

.

.

.

V Technology Considerations

11 The Technology Stack

11.1 From Client Instruction to Trade Reconciliation .
.
11.2 Algorithmic Trading Infrastructure
.
.
11.3 HFT Infrastructure .
.
.
.
11.4 ATS Infrastructure
.
.
.

.
.
.
.
.
.
11.4.1 Regulatory Considerations .
11.4.2 Matching Engine .
.
.
11.4.3 Client Tiering and Other Rules .

.
.
. .
.
.
.
.
.
.
.
.

.
.
.
.
.
.

.
.
.
.
.
.

.
.
.
.
.

.
.

.
.

.
.

.
.

.
.

.
.

.

.

.

.

.
.
.
.
.
.
.

.
.
.
.
.
.
.

.
.
.
.
.
.
.

Contents

.
.
.
.
.
.
.

.
.
.
.
.
.
.
.
.

.
.
.
.
.
.
.
.
.
.
.
.

.
.
.
.
.
.
.

.
.
.
.
.
.
.

.
.
.
.
.
.
.
.
.

.
.
.
.
.
.
.

.
.
.
.
.
.
.
.
.

.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
. .
.
.

.
.
.
.
.
.
.

.
.
.
.
.
.
.
.
.

.
.
.
.
.
.
.
.
.
.
.
.

.
.
.
.
.
.
.

.
.
.
.
.
.
. .
.
.
.
.
.
.

.
.
.
.
.
.
.

.
.
.
.
.
.
.
.
.

.
.
.
.
.
.
.
.
.
.
.
.

.
.
.
.
.
.
.

.
.
.
.
.
.
.

.
.
.
.
.
.
.
.
.

.
.
.
.
.
.
.
.
.
.
.
.

.
.
.
.
.
.
.

. 283
. 286
. 287
. 289
. 295
. 306
. 310

313
. 313
. 314
. 315
. 318
. 320
. 323
. 324
. 327
. 328

335
. 336
. 344
. 348
. 348
. 352
. 353
. 359
. 359
. 361
. 367
. 371
. 375

381

383
. 383
. 387
. 394
. 395
. 395
. 397
. 397

Contents

12 The Research Stack

.

12.1 Data Infrastructure .
.
12.2 Calibration Infrastructure
12.3 Simulation Environment
.
12.4 TCA Environment
.
.
12.5 Conclusion .

.
.

.
.

.

.

Bibliography

Subject Index

.

.
.
.

.
.
.
.
.

.
.
.
.
.

.
.
.
.
.

.
.
.
.
.

.
.
.
.
.

.
.
.
.
.

.
.
.
.
.

.
.
.
.
.

.
.
.
.
.

.
.
.
.
.

.
.
.
.
.

.
.
.
.
.

.
.
.
.
.

.
.
.
.
.

.
.
.
.
.

.
.
.
.
.

.
.
.
.
.

.
.
.
.
.

.
.
.
.
.

.
.
.
.
.

.
.
.
.
.

ix

401
. 401
. 403
. 404
. 408
. 410

411

433

Preface

Algorithms have been around since the day trading has started. But they have
gained importance with the advent of computers and the automation of trading. Eﬃ-
ciency in execution has taken center stage and with that, speed and instantaneous
processing of asset related information have become important. In this book, we will
focus on the methodology rooted in ﬁnancial theory and demonstrate how relevant
data—both in the high frequency and in the low frequency spaces—can be mean-
ingfully analyzed. The intention is to bring both the academics and the practitioners
together. We strive to achieve what George Box (ﬁrst author’s teacher) once said:

“One important idea is that science is a means whereby
learning is achieved, not by mere theoretical speculation
on the one hand, nor by the undirected accumulation of
practical facts on the other, but rather by a motivated
iteration between theory and practice.”

We hope that we provide a framework for relevant inquiries on this topic. To quote

Judea Pearl,

“You cannot answer a question that you cannot ask, and
you cannot ask a question that you have no words for.”

The emphasis of this book, the readers will notice, is on data analysis with guidance
from appropriate models. As C.R. Rao (ﬁrst author’s teacher) has aptly observed:

All knowledge is, in ﬁnal analysis, history.
All Sciences are, in the abstract, mathematics.
All judgements are, in the rationale, statistics.

This book gives an inside look into the current world of Electronic Trading and
Quantitative Strategies. We address actual challenges by presenting intuitive and
innovative ideas on how to approach them in the future. The subject is then aug-
mented through a more formal treatment of the necessary quantitative methods with
a targeted review of relevant academic literature. This dual approach is also reﬂec-
tive of the dynamics, typical of quants working on a trading ﬂoor where commercial
needs, such as time to market, often supersede consideration of rigorous models in
favor of intuitively simple approaches.

Our unique approach in this book is to provide the reader with hands-on tools.
This book will be accompanied by a collection of practical Jupyter Notebooks where
select methods are applied to real data. This will allow the readers to go beyond theory

xi

xii

Preface

into the actual implementation, while familiarizing them with the libraries available.
Wherever possible the charts and tables in the book can be generated directly from
these notebooks and the data sets provided bring further life to the treatment of the
subject. We also add exercises to most of the chapters, so that the students can work
through on their own. These exercises have been tested out by graduate students from
Stanford and Singapore Management University. The notebooks as well as the data
and the exercises are made available on: https://github.com/NehrenD/algo_
trading_and_quant_strategies. This site will be updated on a periodic basis.
While reading and working through this book, the reader should be able to gain insight
into how the ﬁeld of Electronic Trading and Quantitative Strategies, one of the most
active and exciting spaces in the world of ﬁnance, has evolved.

This book is divided into ﬁve parts.

• Part

I sets the stage. We narrate the history and evolution of Equity Trading
and delve into a review of the current features of modern Market Structure. This
gives the readers context on the business aspects of trading in order for them to
understand why things work as they do. The next section will provide a brief high-
level foundational overview of market microstructure which explains and models
the dynamics of a trading venue heavily inﬂuenced by the core mechanism of
how trading takes place: the price-time priority limit-order book with continuous
double auction. This will set the stage for the introduction of a critical but elusive
concept in trading: Liquidity.

• Part II provides an overview of discrete time series models applied to equity
trading. We will address univariate and multivariate time series models of both
mean and variance of asset returns, and other associated quantities such as vol-
ume. While somewhat less used today because of high frequency trading, these
models are important as conceptual frameworks, and act as baselines for more
advanced methods. We also cover some essential concepts in Point Processes as
the actual trading data can come at irregular intervals. The last chapter of Part II
will present more advanced topics like State-Space Models and modern Machine
Learning methods.

• Part

III dives into the broad topic of Quantitative Trading. Here we provide
the reader with a toolkit to conﬁdently approach the subject. Historical perspec-
tives from Alpha generation to the art of backtesting are covered here. Since most
quantitative strategies are portfolio-based, meaning that alphas are usually com-
bined and optimized over a basket of securities, we will brieﬂy introduce the topic
of Active Portfolio Management and Mean-Variance Optimization, and the more
advanced topic of Dynamic Portfolio Selection. We conclude this section dis-
cussing a somewhat recent topic: News and Sentiment Analytics. Our intent is to
also remind the reader that the ﬁeld is never “complete” as new approaches (such
as this from behavioral ﬁnance) are embraced by practitioners once the data is
available.

• Part

IV covers Execution Algorithms, a sub-ﬁeld of Quantitative Trading
which has evolved separately from simple mechanical workﬂow tools, into a

Preface

xiii

multi-billion dollar business. We begin by reviewing various approaches to mod-
eling trade data, and then dive into the fundamental subject of Market Impact, a
complex and least understood concept in ﬁnance. Having set the stage, the ﬁnal
section presents a review of the evolution and the current state of the art in Exe-
cution Algorithms.

• Finally, Part V deals with some technical aspects of developing both quanti-
tative trading strategies and execution algorithms. Trading has become a highly
technological process that requires the integration of numerous technologies and
systems, ranging from market data feeds, to exchange connectivity, to low-latency
networking and co-location, to back-oﬃce booking and reporting. Developing a
modern and high performing trading platform requires thoughtful consideration
and some compromise. In this part, we look at some important details in cre-
ating a full end-to-end technology stack for electronic trading. We also want to
emphasize the critical but often ignored aspect of a successful trading business:
Research Environment.

Acknowledgments: The ideas for this book were planted ten years ago, while the
ﬁrst author, Raja Velu, was visiting the Statistics Department at Stanford at the invi-
tation of Professor T.W. Anderson. Professor Tze-Leung Lai, who was in charge of
the Financial Mathematics program, suggested initiating a course on Algorithmic
Trading. This course was developed by the ﬁrst author and the other authors, Daniel
Nehren and Maxence Hardy, oﬀered guest lectures to bring the practitioner’s view
to the classroom. This book in large part is the result of that interaction. We want to
gratefully acknowledge the opportunity given by the Stanford’s Statistics department
and by Professors Lai and Anderson.

We owe a personal debt to many people for their invaluable comments and intel-
lectual contribution to many sources from which the material for this book is drawn.
The critical reviews by Professors Guofu Zhou and Ruey Tsay at various stages
of writing are gratefully acknowledged. Colleagues from Syracuse University, Jan
Ondrich, Ravi Shukla, David Weinbaum, Lai Xu, Suhasini Subba Rao from Texas
A&M, and Jeﬀrey Wurgler from New York University, all read through various ver-
sions of this book. Their comments have helped to improve its content and the pre-
sentation. Students who took the course at Stanford University, National University
of Singapore and Singapore Management University and the teaching assistants were
instrumental in shaping the structure of the book. In particular, we want to recognize
the help of Balakumar Balasubramaniam, who oﬀered extensive comments on an ear-
lier version. On the intellectual side, we have drawn material from the classic books
by Tsay (2010); Box, Jenkins, Reinsel, Ljung (2015) on the methodology; Campbell,
Lo and MacKinlay (1996) on ﬁnance; Friedman, Hastie and Tibshirani (2009) on sta-
tistical learning. In the tone and substance at times, we could not say better than what
is already said in these classics and so the readers may notice some similarities.

We have relied heavily on the able assistance of Caleb McWhorter, who has put
the book together with all the demands of his own graduate work. We want to thank

xiv

Preface

our doctoral students, Kris Herman and Zhaoque Zhou (Chosen) for their help at vari-
ous stages of the book. The joint work with them was useful to draw upon for content.
As the focus of this book is on the use of real data, we relied upon several sources for
help. We want to thank Professor Ravi Jagannathan for sharing his thoughts and data
on pairs trading in Chapter 5 and Kris Herman, whose notes on the Hawkes process
are used in Chapter 8. The sentiment data used in Chapter 7 was provided by iSen-
tium and thanks to Gautham Sastri. Scott Morris, William Dougan and Peter Layton
at Blackthorne Inc, whose willingness to help on a short notice, on matters related to
data and trading strategies is very much appreciated. We want to also acknowledge
editorial help from Claire Harshberber and Alyson Nehren.

Generous support was provided by the Whitman School of Management and the
Department of Finance for the production of the book. Raja Velu would like to thank
former Dean Kenneth Kavajecz and Professors Ravi Shukla and Peter Koveos who
serve(d) as department chairs and Professor Michel Benaroch, Associate Dean for
Research, for their encouragement and support.

Last but not least the authors are grateful and humbled to have Adam Hogan pro-
vide the artwork for the book cover. The original piece speciﬁcally made for this book
is a beautiful example of Algorithmic Art representing the trading intensity of the US
stock universe on the various trading venues. We cannot think of a more ﬁtting image
for this book.

Finally, no words will suﬃce for the love and support of our families.

Raja Velu
Maxence Hardy
Daniel Nehren

About the Authors

Raja Velu
Raja Velu is a Professor of Finance and Business Analytics in the Whitman School
of Management at Syracuse University. He obtained his Ph.D. in Business/Statis-
tics from University of Wisconsin-Madison in 1983. He served as a marketing fac-
ulty at the University of Wisconsin-Whitewater from 1984 to 1998 before moving
to Syracuse University. He was a Technical Architect at Yahoo! in the Sponsored
Search Division and was a visiting scientist at IBM-Almaden, Microsoft Research,
Google and JPMC. He has also held visiting positions at Stanford’s Statistics Depart-
ment from 2005 to 2016 and was a visiting faculty at the Indian School of Business,
National University of Singapore and Singapore Management University. His cur-
rent research includes Modeling Big Chronological Data and Forecasting in High-
Dimensional settings. He has published in leading journals such as Biometrika, Jour-
nal of Econometrics and Journal of Financial and Quantitative Analysis.

Maxence Hardy
Maxence Hardy is a Managing Director and the Head of eTrading Quantitative
Research for Equities and Futures at J.P. Morgan, based in New York. Mr. Hardy
is responsible for the development of the algorithmic trading strategies and models
underpinning the agency electronic execution products for the Equities and Futures
divisions globally. Prior to this role, he was the Asia Paciﬁc Head of eTrading and
Systematic Trading Quantitative Research for three years, as well as Asia Paciﬁc Head
of Product for agency electronic trading, based in Hong Kong. Mr. Hardy joined J.P.
Morgan in 2010 from Societe Generale where he was part of the algo team develop-
ing execution solutions for Program Trading. Mr. Hardy holds a master’s degree in
quantitative ﬁnance from the University Paris IX Dauphine in France.

Daniel Nehren
Daniel Nehren is a Managing Director and the Head of Statistical Modelling and
Development for Equities at Barclays. Based in New York, Mr. Nehren is responsible
for the development of algorithmic trading product and model-based business logic
for the Equities division globally. Mr. Nehren joined Barclays in 2018 from Citadel,
where he was the head of Equity Execution. Mr. Nehren has over 16 years’ experience
in the ﬁnancial industry with a focus on global equity markets. Prior to Citadel, Mr.
Nehren held roles at J.P. Morgan as the Global Head of Linear Quantitative Research,
Deutsche Bank as the Director and Co-Head of Delta One Quantitative Products and
Goldman Sachs as the Executive Director of Equity Strategy. Mr. Nehren holds a
doctorate in electrical engineering from Politecnico Di Milano in Italy.

xv

xvi

Dedicated to. . .

Yasodha, without her love and support, this is not possible.

Melissa, for her patience every step of the way, and her never
ending support.

About the Authors

R.V.

M.H.

Alyson, the muse, the patient partner, the inspiration for this work
and the next. And the box is still not full.

D.N.

Part I

Introduction to Trading

2

Algorithmic Trading and Quantitative Strategies

We provide a brief introduction to market microstructure and trading from a prac-
titioner’s point of view. The terms used in this part, all can be traced back to academic
literature; but the discussion is kept simple and direct. The data, which is central to
all the analyses and inferences, is then introduced. The complexity of using data that
can arise at irregular intervals can be better understood with an example illustrated
here. Finally, the last part of this chapter contains a brief academic review of market
microstructure—a topic about the mechanics of trading and how the trading can be
inﬂuenced by various market designs. This is an evolving ﬁeld that is of much interest
to all: regulators, practitioners and academics.

1

Trading Fundamentals

1.1 A Brief History of Stock Trading
Why We Trade: Companies need capital to operate and expand their businesses. To
raise capital, they can either borrow money then pay it back over time with interest,
or they can sell a stake (equity) in the company to an investor. As part owner of
the company, the investor would then receive a portion of the proﬁts in the form of
dividends. Equity and Debt, being scarce resources, have additional intrinsic value
that change over time; their prices are inﬂuenced by factors related to the performance
of the company, existing market conditions and in particular, the future outlook of the
company, the sector, the demand and supply of capital and the economy as a whole.
For instance, if interest rates charged to borrow capital change, this would aﬀect the
value of existing debt since its returns would be compared to the returns of similar
products/companies that oﬀer higher/lower rates of return. When we discuss trading
in this book we refer to the act of buying and selling debt or equity (as well as other
types of instruments) of various companies and institutions among investors who have
diﬀerent views of their intrinsic value.

These secondary market transactions via trading exchanges also serve the purpose
of “price discovery” (O’Hara (2003) [275]). Buyers and sellers meet and agree on a
price to exchange a security. When that transaction is made public, it in turn informs
other potential buyers and sellers of the most recent market valuation of the security.
The evolution of the trading process over the last 200 years makes for an incred-
ible tale of ingenuity, ﬁerce competition and adept technology. To a large extent, it
continues to be driven by the positive (and at times not so positive) forces of making
proﬁt, creating over time a highly complex, and amazingly eﬃcient mechanism, for
evaluating the real value of a company.

The Origins of Equity Trading: The tale begins on May 17, 1792 when a group of
24 brokers signed the Buttonwood Agreement. This bound the group to trade only
with each other under speciﬁc rules. This agreement marked the birth of the New
York Stock Exchange (NYSE). While the NYSE is not the oldest Stock Exchange in

3

4

Algorithmic Trading and Quantitative Strategies

the world,1 nor the oldest in the US,2 it is without a question the most historically
important and undisputed symbol of all ﬁnancial markets. Thus in our opinion, it is
the most suitable place to start our discussion. The NYSE soon after moved their
operations to the nearby Tontine Coﬀee House and subsequently to various other
locations around the Wall Street area before settling in the current location on the
corner of Wall St. and Broad St. in 1865.

For the next almost 200 years, stock exchanges evolved in complexity and in
scope. They, however, conceptually remained unchanged, functioning as physical
locations where traders and stockbrokers met in person to buy and sell securities.
Most of the exchanges settled on an interaction system called Open Outcry where
new orders were communicated to the ﬂoor via hand signals and with a Market
Maker facilitating the transactions often stepping in to provide short term liquidity.
The advent of the telegraph and subsequently the telephone had dramatic eﬀects in
accelerating the trading process and the dissemination of information, while leaving
the fundamental process of trading untouched.

Electroniﬁcation and the Start of Fragmentation: Changes came in the late 1960s
and early 1970s. In 1971, the NASDAQ Stock Exchange launched as a completely
electronic system. Initially started as a quotation site, it soon turned into a full
exchange, quickly becoming the second largest US exchange by market capitaliza-
tion. In the meantime, another innovation was underway. In 1969, the Institutional
Networks Corporation launched Instinet, a computerized link between banks, mutual
fund companies, insurance companies so that they could trade with each other with
immediacy, completely bypassing the NYSE. Instinet was the ﬁrst example of an
Electronic Communication Network (ECN), an alternative approach to trading that
grew in popularity in the 80s and 90s with the launch of other notable venues like
Archipelago and Island ECNs.

This evolution started a trend (Liquidity Fragmentation) in market structure that
grew over time. Interest for a security is no longer centralized but rather distributed
across multiple “liquidity pools.” This decentralization of liquidity created signiﬁcant
challenges to the traditional approach of trading and accelerated the drive toward
electroniﬁcation.

The Birth of High Frequency Trading and Algorithmic Trading: The year 2001
brought another momentous change in the structure of the market. On April 9th,
the Securities and Exchange Commission (SEC)3 mandated that the minimum price
increment on any exchange should change from 1/16th of a dollar (≈ 6.25 cents) to

1This honor sits with the Amsterdam Stock Exchange dating back to 1602.

2The Philadelphia Stock Exchange has a 2 year head start having been established in 1790.

3SEC is an independent federal government agency responsible for protecting investors, and maintain-

ing fair and orderly functioning of the securities market http://www.sec.gov

Trading Fundamentals

5

1 cent.4 This seemingly minor rule change with the benign name of ‘Decimaliza-
tion’ (moving from fractions to decimal increments) had a dramatic eﬀect, causing
the average spread to signiﬁcantly drop and with that, the proﬁts of market makers
and broker dealers also declined. The reduction in proﬁt forced many market mak-
ing ﬁrms to exit the business which in turn reduced available market liquidity. To
bring liquidity back, exchanges introduced the Maker-Taker fee model. This model
compensated the traders providing liquidity (makers) in the form of rebates, while
continuing to charge a fee to the consumer of liquidity (takers).5

The maker-taker model created unintentional consequences. If one could provide
liquidity while limiting liquidity taking, one could make a small proﬁt, due to the
rebate with minimal risk and capital. This process needs to be fairly automated as
the per trade proﬁt would be minimal, requiring heavy trading to generate real rev-
enue. Trading also needs to be very fast as position in the order book and speed of
cancellation of orders are both critical to proﬁtability. This led to the explosion of
what we today call High Frequency Trading (HFT) and to the wild ultra-low latency
technology arms race that has swept the industry over the past 15 years. HFT style
trading, formerly called opportunistic market making, already existed but never as a
signiﬁcant portion of the market. At its peak it was estimated that more than 60% of
all trading was generated by HFTs.

The decrease in average trading cost, as well as the secular trend of on-line invest-
ing led to a dramatic increase in trading volumes. On the other hand, the reduction
in per-trade commission and proﬁtability forced broker-dealers to begin automating
some of their more mundane trading activities. Simple workﬂow strategies slowly
evolved into a ﬁeld that we now call Algorithmic Execution which is a main topic of
this book.

Dark Pools and Reg NMS: In the meantime, the market structure continued to evolve
and fragmentation continued to increase. In the late eighties and early nineties a new
type of trading venue surfaced, with a somewhat diﬀerent value proposition: Allowing
traders to ﬁnd a block of liquidity without having to display that information “out
loud” (i.e., on exchange). This approach promised reduced risk of information leakage
as these venues do not publish market data, only the notiﬁcation of a trade is conveyed
after the fact. These aptly but ominously named Dark Pools have become a staple in
equity trading, and now represent an estimated 30-40% of all liquidity being traded
in US Equities.

In 2005, the Regulation National Market System (Reg NMS) was introduced in an
eﬀort to address the increase in trading complexity and fragmentation. Additionally,

4The 1/16th price increment was a vestige of Spanish monetary standards of the 1600s when doubloons

where divided in 2, 4, 8 parts.

5For some readers the concepts of providing and taking liquidity might be murky at best. Do not fret,

this will all become clear when we discuss the trading process and introduce Market Microstructure.

6

Algorithmic Trading and Quantitative Strategies

it accelerated the transformation of market structure in the US. Through its two main
rules, the intent of Reg NMS was to promote best price execution for investors by
encouraging competition among individual exchanges and individual orders. The
Access Rule promotes non-discriminatory access to quotes displayed by the var-
ious trading centers. It also had established a cap, limiting the fees that a trad-
ing center can charge for accessing a displayed quote. The Order Protection Rule
requires trading centers to obtain the best possible price for investors wherever it
is represented by an immediately accessible quote. The rule designates all regis-
tered exchanges as “protected” venues. It further mandates that apart from a few
exceptions, all market participants transact ﬁrst at a price equal or better to the
best price available in these venues. This is known as the National Best Bid Oﬀer
(NBBO).

The Order Protection Rule in particular, had a signiﬁcant eﬀect on the US market
structure. By making all liquidity in protected venues of equal status it signiﬁcantly
contributed to further fragmentation. In 2005 the NYSE market share in NYSE-listed
stocks was still above 80%. By 2010, it had plunged to 25% and has not recovered
since.

Fragmentation continued to increase with new prominent entrants like BATS
Trading and Direct Edge. Speed and technology rapidly became major diﬀerentiating
factors of success for market makers and other participants. The ability to process,
analyze, and react to market data faster than competing participants, meant captur-
ing ﬂeeting opportunities and successfully avoiding adverse selection. Therefore, to
gain or maintain an edge, market participants heavily invested in technology, in faster
networks, and installed their servers in the same data centers, as the venues they trans-
acted on (co-location).

Conclusion: This whirlwind tour of the history of trading was primarily to review the
background forces that led to the dizzying complexity of modern market structure,
that may be diﬃcult to comprehend without the appropriate context. Although this
overview was limited to the USA alone, it is not meant to imply that the rest of the
world stayed still. Europe and Asia both progressed along similar lines, although at a
slower and more compressed pace.

Trading Fundamentals

7

1.2 Market Structure and Trading Venues: A Review

1.2.1 Equity Markets Participants

With the context presented in the previous section, we now provide a brief review
of the state of modern Market Structure. In order to get a sense of the underpinnings
of equity markets, it is important to understand who are the various participants, why
they trade and what their principal focus is.

Long Only Asset Managers: These are the traditional Mutual Fund providers like
Vanguard, Fidelity, etc. A sizable portion of US households invest in mutual funds as
part of their company’s pensions funds, 401K’s, and other retirement vehicles. Some
of the largest providers are of considerable size and have accumulated multiple tril-
lions of dollars under their management. They are called Long Only asset managers
because they are restricted from short selling (where you borrow shares to sell in the
market in order to beneﬁt from a price drop). They can only proﬁt through dividends
and price appreciation. These ﬁrms need to trade frequently in order to re-balance
their large portfolios, to manage in-ﬂows, out-ﬂows, and to achieve the fund’s objec-
tive. The objective may be to track and hopefully beat a competitive benchmark. The
time horizon these investors care about is in general long, from months to years. His-
torically, these participants were less concerned with transaction costs because their
investment time scales are long and the returns they target dwarf the few dozen basis
points normally incurred in transaction costs.6 What they are particularly concerned
about is information leakage. Many of these trades last from several days to weeks
and the risk is that other participants in the market realize their intention and proﬁt
from the information to the detriment of the fund’s investors.

Long-Short Asset Managers and Hedge Funds: These are large and small ﬁrms
catering to institutional clients and wealthy investors. They often run multiple strate-
gies, most often market-neutral long-short strategies, holding long and short posi-
tions in order to have reduced market exposure and thus hopefully perform well in
both raising and falling markets. Investors of this type include ﬁrms like Bridgewa-
ter, Renaissance Technologies, Citadel, Point 72 (former SAC Capital), and many
others. Their time horizon is varied, as with the strategies they employ, but in general
they are shorter than their Long Only counterparts. They also tend to be very focused
on minimizing transaction costs, because they often make many more smaller short
term investments and the transaction costs can add up to be a signiﬁcant fraction of
their expected proﬁts.

6In recent years though, competitive pressure as well as Best Execution regulatory obligations have

brought a lot of focus on transaction costs minimization to the Long Only community as well.

8

Algorithmic Trading and Quantitative Strategies

Broker-Dealers (a.k.a. Sell Side): These ﬁrms reside between the “Buy Side”
(generic term for asset management ﬁrms and hedge funds) and the various
exchanges.7 They can act either as Agent for the client (i.e., Broker) or provide liq-
uidity as Principal (i.e., Dealer) from their own accounts. They historically also had
large proprietary trading desks investing the ﬁrm’s capital using strategies not dissim-
ilar to the strategies that hedge funds use. Since the introduction of the Dodd-Frank
Act,8 the amount of principal risk that a ﬁrm can carry has dramatically decreased
and banks had to shed their proprietary trading activities either shutting down the
desks or spinning them oﬀ into independent hedge funds.

HFTs, ELPs (Electronic Liquidity Providers), and DMMs (Designated Market
Makers): These participants generate returns by acting as facilitators between the
above participants, providing liquidity and then unwinding it at a proﬁt. They can
also act as aggregators of retail liquidity, e.g., individual investors who trade using
online providers like ETrade. Usually the most technology savvy operators in the
market place, they leverage ultra-low-latency infrastructure in order to be extremely
nimble. They get in and out of positions rapidly, taking advantage of tiny mis-pricing.

1.2.2 Watering Holes of Equity Markets

Now that we know who the players are, we brieﬂy review various ways they access

liquidity.

Exchanges: This is still the standard approach to trading and accounts for about 60–
70% of all activity. This is where an investor will go for immediacy and a surer out-
come. Because the full order-book of an exchange, arrivals/cancellations are all pub-
lished, the trader knows exactly the liquidity that is available and can plan accordingly.
This information, it should be kept in mind, is known to all participants, especially
(due to their often technological advantages) the ones whose strategies focus on pat-
terns of large directional trades. A trader that needs to buy or sell in large size will
need to be careful about how much information their trades disseminate or they may
pay dearly. The whole ﬁeld of Algorithmic Execution evolved as an eﬀort to trade
large positions while minimizing market impact and informational leakage.

Apart from these concerns, interacting with exchanges is arguably the most basic
task to trading. But it is not straightforward. At the time of this writing, a US equity
trader can buy or sell stocks in 15 registered exchanges (13 actively trading).9 As
mentioned before when discussing Reg NMS, these exchanges are all “protected.”
Thus, the liquidity at the best price cannot be ignored. An exchange will need to

7Note that only member ﬁrms are allowed to trade on exchange and most asset management ﬁrms are

non-members thus need an intermediary to trade on their behalf.

8https://www.cftc.gov/LawRegulation/DoddFrankAct/index.htm

9https://www.sec.gov/fast-answers/divisionsmarketregmrexchangesshtml.html

Trading Fundamentals

9

reroute to other exchanges where price is better (charging a fee for it). Smart Order
Routers have evolved to manage this complexity.

Recent years have seen a consolidation of these exchanges in the hands of mainly
three players: ICE, NASDAQ and CBOE. Here is a list of venues operated by each,
respectively:

• NYSE, ARCA, MKT (former AMX, American Stock Exchange), NSX (former

National Stock Exchange), CHX (former Chicago Stock Exchange)

• NASDAQ, PHLX (former Philadelphia Stock Exchange), ISE (former Interna-

tional Securities Exchange), BX (former Boston Stock Exchange)

• CBOE, BZX (former BATS), BYX (former BATS-Y), EDGA, EDGX

The newest exchange and as of now the only remaining independent exchange, is the
Investors Exchange (IEX).10 We will discuss more about this later.

An interesting observation is that this consolidation did not happen with a contem-
poraneous reduction in fragmentation. These venues continue to operate as separate
pools of liquidity. While the exchange providers make valid arguments that they pro-
vide diﬀerent business propositions, there is a growing concern in the industry that
the revenue model for these exchanges is now largely centered around providing mar-
ket data and charging for exchange connectivity fees.11 Because the venue quotes are
protected, any serious operator needs to connect with the exchanges and leverage their
direct market data feeds in their trading applications. With more exchanges, the more
connections and fees these exchange can charge. Recently regulators are starting to
weigh in on this contentious topic.12

It is also interesting to note that most of these exchanges are hosted in one of
four data centers located in the New Jersey countryside: Mahwah, Secaucus, Carteret,
Weehawken. The distance between these data centers adds some latency in the dis-
semination of information across exchanges and thus creates latency arbitrage oppor-
tunities. Super fast HFTs co-locate in each data center and leverage the best available
technology such as microwave and more recently laser technologies to connect them.
These operators can see market data changes before others do. Then they either trade
faster or cancel their own quotes to avoid adverse selection (a phenomenon known as
liquidity fading).

All exchanges have almost exactly the same trading mechanism and from the trad-
ing perspective, behave exactly the same way during the continuous part of the trading
day except for the opening and closing auctions. They provide visible (meaning that
market data is disseminated) order books and operate on a price/time priority basis.13

10In 2019, a group of ﬁnancial institutions ﬁled an application for a members owned exchange, MEMX.

11https://www.sifma.org/wp-content/uploads/2019/01/Expand-and-SIFMA-An-

Analysis-of-Market-Data-Fees-08-2018.pdf

12https://www.sec.gov/tm/staff-guidance-sro-rule-filings-fees

13Certain Futures and Options exchanges have a pro-rata matching mechanism which is discussed later.

10

Algorithmic Trading and Quantitative Strategies

Most of the exchanges use the maker-taker fee model that we discussed earlier but
some: BYX, EDGA, NSX, BX, have adopted an ‘inverted’ fee model, where post-
ing liquidity incurs a fee while a rebate is provided for taking liquidity. This change
causes these venues to display a markedly diﬀerent behavior. The cost for providing
liquidity removes the incentive of rebate seeking HFTs; however these venues are
used ﬁrst when needing immediate liquidity as the taker is compensated. This in turn
brings in passive liquidity providers who are willing to pay for trading passively at
that price. This interplay adds a subtle and still not very well understood dynamic to
an already complex market structure.

Finally, a few observations about IEX. It started as an Alternative Trading System
(ATS) whose main innovation is a 38 mile coil of optical ﬁber placed in front of its
trading engine. This introduces a 350 microsecond delay each way aptly named the
“speed bump” that is meant to remove the speed advantages of HFTs. The intent is to
reduce the eﬃcacy of the more latency-sensitive tactics and thus provide a liquidity
pool that is less “toxic.” On June 17, 2016 IEX became a full ﬂedged exchange after a
very controversial process.14,15 It was marketed as an exchange that would be diﬀerent
and would attract substantial liquidity due to its innovative speed bump. But, as of
2019, this potential remains still somewhat unrealized and IEX has not moved much
from the 2% to 3% market share range of which 80% is still in the form of hidden
liquidity. That being said IEX has established itself as a vocal critic16 of the current
state of aﬀairs continuing to shed light on some potential conﬂicts of interest that
have arisen in trading. At the time of this writing IEX does not charge any market
data fees.17

ATS/Dark Pools: As discussed above, trading in large blocks in exchanges is not a
simple matter and requires advanced algorithms for slicing the block orders and smart
order routers for targeting the liquid exchanges. Even then the risk and the result-
ing costs due to information leakage can be signiﬁcant. Dark Pools were invented
to counterbalance this situation. One investor can have a large order “sitting” in a
dark pool with no one knowing that it is there and would be able to ﬁnd the other
side without showing any signals of the order’s presence. Dark pools do not display
any order information and use the NBBO (National Best Bid/Oﬀer) as the reference
price. In almost all cases, to avoid accessing protected venues, these pools trade only
at the inside market (at or within the bid ask spread). Although, in order to maxi-
mize the probability of ﬁnding liquidity most of the block interaction happens at the
mid-point. Oﬀ-exchange trading has gained more and more traction in the last ﬁfteen

14https://www.sec.gov/comments/10-222/10-222.shtml

15https://www.bloomberg.com/news/articles/2016-06-14/sec-staff-recommends-

approving-iex-application-wsj-reports

16https://iextrading.com/insights

17https://iextrading.com/trading/market-data/

Trading Fundamentals

11

years and now accounts for 30–40% of all traded volume in certain markets like the
US. ATS/Dark Pool volume makes up roughly 30% of this US equities oﬀ-exchange
volume. Clearly, the growth of dark execution has spurred a lot of competition in the
market. Additionally, the claim of reduced information leakage is probably overstated
as it is still potentially possible to identify large blocks of liquidity by “pinging” the
pool at minimum lot size. In order to counteract this eﬀect orders are usually sent
with a minimum ﬁll quantity tag which allows the block to be transparent from small
pinging.

As of 2019, there are 33 diﬀerent equity ATSs!18 All these venues compete on
pricing, availability of liquidity, system performance, and functionality such as han-
dling of certain special order types, etc. Many of these ATS are run by the major
investment banks and they have historically dominated this space. As far as overall
liquidity goes, 10 dark pools account for about 75% of all ATS volume, and the top
5 make up roughly 50% of dark liquidity in the US (the UBS ATS and Credit Suisse
Crossﬁnder are consistently at the top of the rankings19). Other venues were born
out of the fundamental desire for investment ﬁrms to trade directly with each other,
bypassing the intermediaries and thus reducing cost and information leakage. The
main problem encountered by these buy-side to buy-side pools is that many of the
trading strategies used by these ﬁrms tend to be highly correlated (e.g., two funds
tracking the same benchmark) and thus the liquidity is often on the same side. There-
fore as a result, these venues have to ﬁnd diﬀerent approaches to leverage sell side
broker’s liquidity to supplement their own such as access via conditional orders. BIDS
and Liquidnet are the biggest ATS of this type. BIDS had signiﬁcant growth in recent
years with Liquidnet losing its initial dominance. This space is still very active with
new venues coming up on a regular basis.

Single Dealer Platform/Systematic Internalizers: A large number of trading ﬁrms
in recent years started providing direct access to their internal liquidity. Broker/Dealer
and other institutional clients connect to a Single Dealer Platform (SDP) directly.
These SDPs, also called Systematic Internalizers, send regular Indication Of Interest
(IOI) that are bespoke to a particular connection and the broker can respond when
there is a match. This approach to trading is also growing fast. Because SDPs are not
regulated ATS, they can oﬀer somewhat unique products. Brokers themselves are now
starting to provide their own SDPs to expose their internal liquidity. A quickly evolv-
ing space that promises interesting innovations, but alas, it can also lead to further
complication in an already crowded ecosystem. In the US, the other 70% of non-ATS
oﬀ-exchange volume is comprised of Retail Wholesalers, Market Makers and Single
Dealer Platforms, and Broker Dealers.

Auctions: Primary exchanges (exchanges where a particular instrument is listed)
begin and end the day with a (primary) auction procedure, that leverages special order

18http://www.finra.org/industry/equity-ats-firms

19https://otctransparency.finra.org/otctransparency/AtsData

12

Algorithmic Trading and Quantitative Strategies

types to accumulate supply and demand and then run an algorithm that determines
the price that would at best pair oﬀ the most volume. The Closing Auction is of par-
ticular importance because many funds set their Net Asset Value (NAV) using the
oﬃcial closing price. This generally leads traders to trade as close as possible to the
closing time in order to optimize the dual objective of getting the best price but also
not deviating too much from the close price.

The Auction also represents an opportunity for active and passive investors to
exchange large amounts of shares (liquidity). Index constituents get updated on a reg-
ular basis (additions, deletions, weight increase/decrease), and as they get updated,
passive investors need to update their holdings to reﬂect the optimal composition
of the benchmark that they track. In order to minimize the tracking error risk, that
update needs to happen close to the actual update of the underlying benchmark. Con-
sequently, most passive indexers tend to rebalance their portfolios on the same day
the underlying index constituents are updated, using the closing auction as a reference
price, and this results in signiﬁcant ﬂows at the close auction. The explosive growth of
ETFs (Exchange Traded Funds) and other passive funds have exacerbated this trend
in recent years. At the time of this writing, about 10% of the total daily US volume in
index names trade at the close. Recent months have brought a lot of movement in this
area with brokers and SDPs trying to provide unique ways to expose internal liquidity
marked for the closing auction.

Beyond the US: As previously mentioned the structure we presented above is not
unique to the US. European and Asian exchanges that had operated as a single mar-
ketplace for longer than their US counterparts now oﬀer a more diverse landscape.
The evolution of ATS/Dark Pools and other MTFs (Multilateral Trading Facilities)
have followed suit but in a more subdued manner. Trading in these markets has always
been smaller and concentrated with fewer participants and there is not enough liq-
uidity to support a large number of venues. Often new venues come on-line but are
quickly absorbed by a competitor when they fail to move a signiﬁcant portion of the
traded volume. Additional complexity arises with regulatory environments in these
various countries limiting cross-border trading. As of December 2018, fragmentation
in European markets is still quite lower than in the US, with 59% traded on primary
exchange, 22% on lit MTF (there are only 6, and 4 of them trade roughly 98% of the
volume), 6.5% on dark MTF (10 diﬀerent ones), and 6% traded in systematic internal-
izers. In APAC, with the exception of Australia, Hong Kong and Japan, most countries
only have one primary exchange where all transactions take place. Even in the more
developed market, Japan for instance, the Tokyo Stock Exchange still garners over
85% of the total volume traded.

Summary: Modern market structure may appear to be a jumbled mess. Yes, it is.
Fully understanding the implications of these diﬀerent methods of trading tied by
regulation, competition, behavioral idiosyncrasies (at times due to participants lack
of understanding and preconceived notions) is a daunting task. It is however an envi-
ronment that all practitioners have to navigate, and it remains diﬃcult to formulate
these issues in mathematical models as they could be based on questionable heuris-
tics to account for the residual complexity. But even with the dizzying complexity,

Trading Fundamentals

13

modern market structure is a fascinating ecosystem. It is a continuously evolving sys-
tem through the forces of ingenuity and competition. It provides tools and services to
institutional investors who strive to reduce the cost of execution, an investment man-
date. We hope the above treatment provides the reader with at least a foothold in the
exploration of this amazing social and ﬁnancial experimentation.

1.3 The Mechanics of Trading

In order to fully grasp the main topics of this book, one requires at least a good
understanding of the mechanics of trading. This process is somewhat complicated
and requires some technical details and terminology. In this section, we will strive to
provide a brief but fairly complete overview of the fundamentals. This should suﬃce
for our purposes. For a more complete and thorough treatment we refer the readers to
the existing literature, most notably Harris (2003) [178].

1.3.1 How Double Auction Markets Work

The most common approach used by modern electronic exchanges can be termed,
as time/price priority, continuous double auction trading system. The term double
auction signiﬁes that, unlike a common auction with one auctioneer dealing with
potential buyers, in this case there are multiple buyers and multiple sellers partic-
ipating in the process at the same time. These buyers and sellers interact with the
exchange by sending instructions electronically, via a network protocol to a special-
ized software and hardware infrastructure called: The Matching Engine. It has two
main components: The Limit Order Book and the Matching Algorithm.

Limit Order Book (LOB): It is a complex data structure that stores all non-executed
orders with associated instructions. It is highly specialized so as to be extremely fast
to insert/update/delete orders and then able to sort them and to retrieve aggregated
information. For an active stock, the LOB can be updated and queried thousands of
times every second, so it must be highly eﬃcient and able to handle a high degree
of concurrency to ensure that the state is always correct. The LOB is comprised of
two copies of the core data structure, one for Buy orders and one for Sell orders often
referred to as the two “sides” of the order book. This structure is the core abstraction
for all electronic exchanges and so it is very important to understand it in detail.

The LOB supports three basic instructions: Insert, cancel, and amend, with insert
initiating a new order, cancel removing an existing order from the market and amend
modifying some of the parameters of the existing order. New orders must specify
“order type” and associated parameters necessary to fully encapsulate the trader deci-
sion. We will review order types in more detail later, but we start with the two main
types: The limit order and the market order. The main diﬀerence between the two

14

Algorithmic Trading and Quantitative Strategies

order types is that a limit order has a price associated to it while a market order does
not.

Accounting for limit and market orders, there are eight events at any given time,

four on either side, that can alter the state of the order book:

• Limit Order Submission: A limit order is added to the queue at the speciﬁed

price level.

• Limit Order Cancellation: An outstanding limit order is expired or canceled

and is therefore removed from the Limit Order Book (LOB).

• Limit Order Amendment: An outstanding limit order is modiﬁed by the original

sender (such as changing order size).

• Execution: Buy and sell orders at appropriate prices are paired by the matching
algorithm (explained below) into a binding transaction and are removed from the
LOB

Matching Algorithm: This software component is responsible for interpreting the
various events to determine if any buy and sell orders can be matched in an execu-
tion. When multiple orders can be paired the algorithm uses the so-called price/time
priority meaning that ﬁrst the order with the most competitive prices are matched and
when prices are equal the order that arrived prior is chosen. As we will see in later
chapters this is only one of the possible algorithms used in practice but it is by far the
most common. We will go into more details in the next sections.

The matching algorithm operates continuously throughout the trading hours. In
order to ensure an orderly start and end, this continuous session is usually comple-
mented by a couple of discrete auctions. The trading day generally starts with an open
auction, then followed by the main continuous session, and ends with a closing auc-
tion. Some markets like Japan also have a lunch break which might be preceded by a
morning closing auction and followed by an afternoon opening auction. We will now
discuss these main market phases in chronological order.

1.3.2 The Open Auction

The Open Auction is only one type of call auction that is commonly held on
exchanges. The term “call auction” explains the liquidity-aggregating nature of this
event. Market participants are ‘called’ to submit their quotes to the market place in
order to determine a matching price that will maximize the amount of shares that can
be transacted. To facilitate timely and orderly cross, auctions have strict order sub-
mission rules, including speciﬁed timing for entries (see Table 1.1) and information
dissemination to prevent wild price ﬂuctuations and ensure that the process is eﬃcient
for price discovery.

Most exchanges publish order imbalance that exists among orders on the opening
or closing books, along with the indicative price and volume. For instance, Nasdaq

Trading Fundamentals

15

Table 1.1: Nasdaq Opening Cross

4:00 a.m. EST Extended hours trading and order entry begins.
9:25 a.m. EST Nasdaq enters quotes for participants with no open interest.
9:28 a.m. EST Dissemination of order imbalance information every 1 second.

Market-on-open orders must be received prior to 9:28 a.m.

9:30 a.m. EST The opening cross occurs.

publishes the following information20 between 9:28 a.m. EST and 9:30 a.m. EST,
every 1 second, on its market data feeds:

• Current Reference Price: Price within the Nasdaq Inside at which paired shares
are maximized, the imbalance is minimized and the distance from the bid-ask
mid-point is minimized, in that order.

• Near Indicative Clearing Price: The crossing price at which orders in the Nas-
daq opening / closing book and continuous book would clear against each other.

• Far Indicative Clearing Price: The crossing price at which orders in the Nasdaq

opening / closing book would clear against each other.

• Number of Paired Shares: The number of on-open or on-close shares that Nas-

daq is able to pair oﬀ at the current reference price.

• Imbalance Shares: The number of opening or closing shares that would remain

unexecuted at the current reference price.

• Imbalance Side: The side of the imbalance: B = buy-side imbalance; S = sell-
side imbalance; N = no imbalance; O = no marketable on-open or on-close orders.

In a double auction setup, the existence of multiple buyers and sellers requires
employing a matching algorithm to determine the actual opening price which we
will illustrate with a practical example. Table 1.2 gives an example of order book
submissions for a hypothetical stock, where orders are ranked based on their arrival
time. Diﬀerent exchanges around the world apply slightly diﬀerent mechanisms to
their auctions, but generally the following rules apply to match supply and demand:

• The crossing price must maximize the volume transacted.

• If several prices result in similar volume transacted, the crossing price is the one

the closest from the last price.

• The crossing price is identical for all orders executed.

• If two orders are submitted at the same price, the order submitted ﬁrst has priority.

20Source: Nasdaq Trader website.

16

Algorithmic Trading and Quantitative Strategies

Table 1.2: Pre-Open Order Book Submissions

Timestamp Seq. Number Side Quantity
1500
1750
4500
1750
2500
1200
500
500
1930
1000
3500
2000
4750
2750
10000
3000
5500
1800
800
1200
5000
12500
450
3500
1120

9:01:21
9:02:36
9:05:17
9:06:22
9:06:59
9:07:33
9:07:42
9:08:18
9:09:54
9:09:55
9:10:04
9:10:39
9:11:13
9:11:46
9:12:21
9:12:48
9:13:12
9:14:51
9:15:02
9:15:37
9:16:42
9:17:11
9:18:27
9:19:13
9:19:54

1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25

B
S
B
S
S
B
B
B
S
B
S
B
S
B
S
B
B
S
B
S
S
B
S
S
B

Price
12.10
12.12
12.17
12.22
12.11
12.23
12.33
12.25
12.30
12.21
12.05
12.34
12.25
12.19
12.33
12.28
12.35
12.18
12.17
12.19
12.16
12.15
12.23
12.20
12.16

• It is possible for an order to be partially executed if the other side quantity is not

suﬃcient.

• “At Market” orders are executed against each other at the determined crossing
price, up to the available matching quantity on both sides, but generally do not
participate in the price formation process.

• For the Open Auction, unmatched “At Market” orders are entered into the con-

tinuous session of LOB as limit orders at the crossing price.

The ﬁrst step is to organize orders by limit price, segregating buys and sells as
shown in Table 1.3. A buy order submitted with a limit price of 12.25 represents an
intent to execute at any price lower or equal to 12.25. Similarly, a sell order submit-
ted at 12.25 represents an intent to sell at any price higher or equal to 12.25. For
each price level, we can then determine the cumulative buy interest and sell interest.
The theoretical cross quantity at each price point is then simply the minimum of the
cumulative buy interest and the cumulative sell interest as shown in Table 1.4. The

Trading Fundamentals

17

Table 1.3: Ranked Order Book Submissions

Timestamp Seq. Number Buy Price Buy Quantity Sell Quantity Sell Price

9:13:12
9:10:39
9:07:42
9:12:21
9:09:54
9:12:48
9:08:18
9:11:13
9:07:33
9:18:27
9:06:22
9:09:55
9:19:13
9:11:46
9:15:37
9:14:51
9:05:17
9:15:02
9:16:42
9:19:54
9:17:11
9:02:36
9:06:59
9:01:21
9:10:04

12.35
12.34
12.33

12.28
12.25

12.23

12.21

12.19

12.17
12.17

12.16
12.15

5500
2000
500

3000
500

1200

1000

2750

4500
800

1120
12500

12.10

1500

17
12
7
15
9
16
8
13
6
23
4
10
24
14
20
18
3
19
21
25
22
2
5
1
11

10000
1930

12.33
12.30

4750

12.25

450
1750

12.23
12.22

3500

12.20

1200
1800

12.19
12.18

5000

12.16

1750
2500

12.12
12.11

3500

12.05

crossing price is determined as the price that would maximize the crossed quantity. In
our example, the opening price will be 12.19, and the opening quantity will be 15,750
shares.

The list of buy orders executed during the auction is shown in Table 1.5 and the
sell orders in Table 1.6. It is worth mentioning that the open auction tends to be
considered as a major price discovery mechanism given the fact that it occurs after a
period of market inactivity when market participants were unable to transact even if
they have information. All new information accumulated overnight will be reﬂected
in the ﬁrst print of the day, matching buying and selling interests.

As market participants with better information are more likely to be participating
in the open auction with more aggressive orders in order to extract liquidity (and,
as such, setting the price), the price discovery mechanism is often considered to be
quite volatile and more suited for short-term alpha investors. Similarly, the period
immediately following the open auction also tends to be much more volatile than the
rest of the day. As a result of which, most markets experience wider spreads while

18

Price

12.35
12.34
12.33
12.33
12.30
12.28
12.25
12.25
12.23
12.23
12.22
12.21
12.20
12.19∗
12.19
12.18
12.17
12.17
12.16
12.16
12.15
12.12
12.11
12.10
12.05

Algorithmic Trading and Quantitative Strategies

Table 1.4: Cumulative Order Book Quantities

Sequence
Number
17
12
7
15
9
16
8
13
6
23
4
10
24
14
20
18
3
19
21
25
22
2
5
1
11

Cumulative
Buy Quantity
5500
7500
8000
8000
8000
11000
11500
11500
12700
12700
12700
13700
13700
16450
16450
16450
20950
21750
21750
22870
35370
35370
35370
36870
36870

Sell
Quantity

Buy
Quantity
5500
2000
500

10000
1930

3000
500

1200

1000

2750

4500
800

1120
12500

1500

4750

450
1750

3500

1200
1800

5000

1750
2500

3500

Cumulative
Sell Quantity
38130
38130
38130
38130
28130
26200
26200
26200
21450
21450
21000
19250
19250
15750
15750
14550
12750
12750
12750
7750
7750
7750
6000
3500
3500

Quantity Crossed
at Price

5500
7500
8000
8000
8000
11000
11500
11500
12700
12700
12700
13700
13700
15750
15750
14550
12750
12750
12750
7750
7750
7750
6000
3500
3500

market makers try to protect themselves against information asymmetry by quoting
wider bids and oﬀers. The increased volatility and wider spreads might discourage
certain investors from participating in the market at the open auction and in the period
immediately following the open. While this appears to be reasonable from a price risk
perspective, it is worth mentioning that for many less liquid stocks (in particular small
and mid cap stocks), the open auction can be a signiﬁcant liquidity aggregation point
that even surpasses the close auction. In Australia for instance, the bottom 50% less
liquid stocks have more volume traded in the open auction than in the close auction.
Similarly in Japan, the less liquid stocks have more volume traded in the open auction,
but also in the afternoon open auction that follows the market lunch break.

From an execution standpoint, though, the usage of the open auction has to be con-
sidered carefully. While this represents a liquidity opportunity, the ﬁrst print of the
day can also have a signiﬁcant anchoring eﬀect on the stock price for the remainder
of the day. So, participating in the open should be considered in light of the liquidity

Trading Fundamentals

19

Table 1.5: Crossed Buy Orders

Price
12.35
12.34
12.33
12.28
12.25
12.23
12.21
12.19∗

Seq. Number Buy Qty
5500
2000
500
3000
500
1200
1000
2050

17
12
7
16
8
6
10
14
*Order number 14 was for 2750 shares but did not get fully executed
as the bid quantity up to 12.19 exceeded the oﬀered quantity at that
price. The balance of order 14 will then be posted as a limit order in
the continuous trading session

Table 1.6: Crossed Sell Orders

Price Seq. Number Sell Qty
12.19
12.18
12.16
12.12
12.11
12.05

1200
1800
5000
1750
2500
3500

20
18
21
2
5
11

demand of the order: Orders that are small enough can likely do without participating
in the open auction and the period that continues immediately following, while large
orders that try to extract signiﬁcant liquidity from the market might beneﬁt from par-
ticipating in the open auction. The market intraday momentum study by Gao, Han,
Li and Zhou (2018) [156] demonstrates how the ﬁrst half-hour return on the market,
as measured from the previous day’s market close predicts the last half hour return.

1.3.3 Continuous Trading

This refers to the main market phase between the auctions. During this market
session the state of the order book changes quite rapidly due to the multi-agent nature
of ﬁnancial markets and the prevalence of high frequency trading. Consequently, it
is important to understand the dynamics of the LOB before implementing trading
strategies. There exists quite a diversity of order types that are mostly relevant to the
continuous trading session, but the two most basic ones are: Limit Orders and Market
Orders, which we describe below.

A limit order has an associated side (Buy or Sell), a quantity and a price which
represent the highest (lowest) price the trader is willing to buy (sell). As previously
discussed, once a limit order is received by the exchange it is inserted in a data struc-
ture called a Limit Order Book (LOB) which contains two sub-structures, one per
side. Orders are inserted in this structure in price priority, higher prices for buys,

20

Algorithmic Trading and Quantitative Strategies

lower prices for sells, and for orders at the same price the orders are stored in the
order in which they were received. That is what is meant by price/time priority.21 If
the price of a newly arrived order overlaps with the best price available on the oppo-
site side, the order is executed either fully or up to the available quantity on the other
side. These orders are said to be “matched” and again this matching happens in price
and time priority meaning that the better prices (higher for buys, lower for sells) are
executed ﬁrst and orders that arrived beforehand at the same price level are executed
ﬁrst. Market orders on the other hand do not have a price associated with them and
will immediately execute against the other side and will match with more and more
aggressive prices until the full order is executed.

Orders on the buy side are called “bids” while those on the sell side are called
“asks.” The above events are illustrated in Figure 1.1 to Figure 1.3. When a market
(or marketable) order is submitted, it decreases the number of outstanding orders at
the opposite best price. For example, if a market bid order arrives, it will decrease
the number of outstanding asks at the best price. All unexecuted limit orders can
be canceled. When a cancellation occurs, it will decrease the number of outstanding
orders at the speciﬁed price level.

bids
asks

limit bid

e
m
u
l
o
v

60

50

40

30

20

10

0

73600

73800

74000

74200

74400

74600

74800

75000

price

Figure 1.1: Limit Order Book—Limit Bid.

21Note: Not all exchanges are matching orders following a price/time priority algorithm; a key char-
acteristic of the Futures market, for instance, is the existence of pro-rata markets for some ﬁxed income
contracts, where passive child orders receive ﬁlls from aggressive orders based on their size as a fraction
of the total passive posted quantity.

Trading Fundamentals

21

e
m
u
l
o
v

e
m
u
l
o
v

60

50

40

30

20

10

0

60

50

40

30

20

10

0

bids
asks

market bid

73600

73800

74000

74200

74400

74600

74800

75000

price

Figure 1.2: Limit Order Book—Marketable Bid.

bids
asks

ask cancellation

73600

73800

74000

74200

74400

74600

74800

75000

price

Figure 1.3: Limit Order Book—Ask Cancellation.

22

Algorithmic Trading and Quantitative Strategies

Limit orders make up a signiﬁcant percentage (70%) of stock market trading activ-
ity. The main advantage of a limit order is that there is no price risk associated to it,
that is, when the order is executed the limit price is the maximum (for a buy order) or
minimum (for a sell order) price that will be achieved. But if the limit order is not mar-
ketable, the execution is not guaranteed and the time to get an order executed depends
on various market factors. The trade-oﬀ between limit orders and marketable orders
depends on the investor’s need for immediate liquidity and the ﬁll probability of limit
orders. The limit price chosen (how deep in the order book is the order placed) as well
as the amount of liquidity ahead of the submitted order (how many shares will need
to trade before the order gets executed following, for instance, a price/time priority
order matching of the exchange) aﬀect both the order ﬁll probability and its expected
time to ﬁll. These two metrics are of particular relevance for execution algorithms
and will be studied in more depth later.

The execution of limit orders does aﬀect how the quotes are posted and are
updated. If the size of a market order exceeds the number of shares available at the
top of book, it is usually split and is executed at consecutive order book levels until
the order is ﬁlled. Market orders are usually restricted to be ﬁlled within a day and
orders placed after the markets close might be entered the next day.22

Order Types: The diversity of order types is a key component of continuous double
auction electronic markets. Order types allow participants to express precisely their
intentions with regards to their interaction with the market via the limit order book.
Over time, in an eﬀort to cater to sophisticated electronic traders, exchanges around
the world have raced to oﬀer ever more complex order types. Here, we will just pro-
vide a brief description of some, besides the market and limit orders that were already
mentioned:

• Peg: Specify a price level at which the order should be continuously and automati-
cally repriced. For instance, an order pegged to the bid price will be automatically
repriced as a higher price limit order, each time the market bid price ticks up. This
order type is particularly used for mid-point executions in non-displayed markets.
One would think that pegging order has the additional advantage of improving the
queue priority of the order when it is repriced since this process is done directly
by the exchange. That turns out not to be always true. Managing peg orders is the
responsibility of a separate component at the exchange and its interaction speed
with the order book is usually slower than that of ultra-low-latency operators.

• Iceberg: Limit order with a speciﬁed display quantity. In order to prevent infor-
mation leakage to other market participants, a trader desiring to buy or sell a large
quantity at a given price might elect to use an iceberg order with a small dis-
play size. For instance, for an order to buy 100,000 shares at $20 with a display
size of 2,000 shares. Only 2,000 shares would be displayed in the order book.
Once that quantity is executed, the order would automatically reload another

22Depending on the Time-in-Force selected.

Trading Fundamentals

23

2,000 shares at $20, and so on, until the full quantity is executed. Note that for
iceberg orders, only the visible quantity has time priority and once that quan-
tity has been executed the new tip of the iceberg will be placed at the back of
the queue.

• Hidden: While they are available to trade, these orders are not directly visible to

other market participants in the central limit order book.

• Stop: These orders are also not visible, but additionally are not immediately
entered in the limit order book. They only become active once a certain price
(known as the Stop Price) is reached or passed. They then enter the order book
as either limit or market order depending on the user setup.

• Trailing Stop: These orders function like stop orders, but the stop price is set

dynamic rather than static (for instance: −3% from previous close).

• All-or-None: Speciﬁcally, request a full execution of the order. If the order is for
1500 shares but only 1000 are being oﬀered, it will not be executed until the full
quantity is available.

• On-Open: Speciﬁcally, request an execution at the open price. It can be limit-on-

open or market-on-open.

• On-Close: Speciﬁcally, request an execution at the close price. It can be limit-

on-close or market-on-close.

• Imbalance Only: Provide liquidity intended to oﬀset on-open/on-close order/im-
balances during the opening/closing cross. These generally are limit orders.

• D-Quote: Special order type on the NYSE mainly used during the close auction

period.

• Funari: Special order type on the Tokyo Stock Exchange which allows limit
orders placed in the book during the continuous session to automatically enter
the closing auction as market orders.

As described above, there exists a wide variety of orders types oﬀered by diﬀerent

exchanges to facilitate various types of trading activities.

Validity of Instructions: In addition to conditions on price, it is possible to add con-
ditions on the life duration of the order known as Time-in-Force (TIF). The most
common types of TIF instructions include Day orders which are valid for the full
duration of the trading session, Extended Day orders allow trading in extended hours,
and Good-Till-Cancel (GTC) orders will be placed again on the exchange the next
day with similar instructions if they were not completely ﬁlled. More sophisticated
market participants aiming at achieving greater control over their executions tend to

24

Algorithmic Trading and Quantitative Strategies

also favor Immediate-or-Cancel (IOC) and Fill-or-Kill (FOK) Time-in-Force instruc-
tions. An IOC order will get immediately canceled back to the sender after reaching
the matching engine if it does not get an immediate ﬁll, and in case of a partial ﬁll, the
unﬁlled portion will be canceled, thus preventing it from creating a new price level in
the order book. In a Fill-or-Kill scenario, the order gets either ﬁlled in its entirety or
does not get ﬁlled at all. This instruction is particularly popular with high frequency
market makers and arbitrageurs for which partial ﬁlls might result in unwanted leg-
ging risk as discussed in Chapter 5 on pairs trading.

Finally, it is worth mentioning that some exchanges as well as alternative venues
oﬀer the ability of specifying minimum ﬁll sizes. This means that a limit order which
might be eligible for a ﬁll due to an incoming order at the same price level, only
receives a ﬁll if the incoming order is larger than a pre-speciﬁed number of shares
or notional value. This type of instruction is used by market participants as a way of
minimizing the number of small ﬁlls which carry the risk of excessive information
dissemination. This happens, in particular, in dark pools, where they can be used to
detect the presence of larger limit orders that would be otherwise not visible to market
participants.

1.3.4 The Closing Auction

The Closing Auction tends to be the most popular call auction for a variety of rea-
sons. First, it is the last opportunity (unless one engages in the risky practice of oﬀ-
hours trading) for market participants to transact in a relatively liquid environment,
before being exposed to the overnight period (when new information accumulates,
but trades cannot easily take place). Second, with the increase in passive investment
strategies, providing investors with replication of a predetermined benchmark index,
the closing auction has become a particularly relevant price setting event. For most
passive funds, the net asset value (NAV) is based on close prices of the underly-
ing assets. For those reasons, the closing auction has become extremely important to
many investors. From an execution standpoint, it is a major liquidity event that must
be handled carefully.

The mechanics of the closing auction are in most part similar to the ones described
above for the open auction. The major diﬀerences across countries (and sometimes
across exchanges within a country) are in the order submission times. Some coun-
tries, such as the US, have order submissions start and end before the continuous
session is over (see Table 1.7 and Table 1.8), while some other markets have two
non-overlapping continuous and close order submission sessions.

Table 1.7: Nasdaq Closing Cross

3:50 p.m. EST Cutoﬀ for amend/cancel of MOC/LOC orders
3:55 p.m. EST Dissemination of imbalance information begins
3:55 p.m. EST Cutoﬀ for entry of MOC/LOC orders
3:58 p.m. EST Freeze period - Late LOC orders cannot be added

OI orders oﬀsetting the imbalance are still accepted

4:00 p.m. EST The Closing Cross occurs

Trading Fundamentals

25

Table 1.8: NYSE Closing Cross

3:50 p.m. EST

Cutoﬀ for MOC/LOC order entry and modiﬁcations
Dissemination of imbalance information begins
Closing Oﬀset orders can be entered until 4:00 p.m.
Dissemination of d-quote imbalance information
Cutoﬀ for MOC/LOC cancellation for legitimate error

3:55 p.m. EST
3:58 p.m. EST
3:59:50 p.m. EST Cutoﬀ for d-Quote order entry and modiﬁcation
The Closing Auction starts
4:00 p.m. EST
DMM can automatically process auctions not yet complete
4:02 p.m. EST

Table 1.9: London Stock Exchange Sessions Times

Pre-Trading
Opening Auction Call
Regular Trading

7:00-7:50 a.m. GMT
7:50-8:00* a.m. GMT
8:00-12:00 p.m. GMT
12:00-12:02* p.m. GMT Periodic Call Auction
12:02-4:30 p.m. GMT
4:30-4:35* p.m. GMT
4:35-4:40 p.m. GMT
4:40-5:15 p.m. GMT

Regular Trading
Closing Auction Call
Closing Price Crossing
Post-Close Trading

*Each auction end time is subject to a random 30 second uncross period.
An additional intraday auction call takes place every 3rd Friday of each month for stocks under-
lying FTSE 100 index options, and the 3rd Friday of every quarter for stocks underlying FTSE
100/250 Index Futures to determine the EDSP (Exchange Delivery Settlement Price). The set-
tlement price is determined as the index value derived from the individual constituents intraday
auction taking place between 10 a.m. and 10:15 a.m. and during which the electronic continu-
ous trading is suspended. Using a call auction ensures the settlement price for these contracts
is more representative of a fair market price.

An aspect of the NYSE is the presence of ﬂoor brokers operating in an agency
capacity for their customers. They play a particular role during the close auction
thanks to their ability to handle discretionary electronic quote orders (known as “d-
Quote” orders) that oﬀer more ﬂexibility than traditional market-on-close (MOC) and
limit-on-close (LOC) orders. The main advantage of d-Quote orders is their ability
to bypass the 3:45 p.m. cutoﬀ and be submitted or canceled until 3:59:50 p.m. This
allows large institutional investors to remain in control of their orders almost until the
end of the continuous session, by delaying the decision of how much to allocate to
the auction. They can therefore react to larger volume opportunities based on the pub-
lished imbalance. They can also minimize the information leakage by not being part
of the early published imbalance while there is still signiﬁcant time in the continuous
session for other participants to drive the price away. Since there is no restriction on
the side of d-Quote orders submission, it is possible to see the total imbalance sign ﬂip
once the d-Quotes are added to the publication at 3:55 p.m., creating opportunities

26

Algorithmic Trading and Quantitative Strategies

for other market participants to adjust their own close trading via d-Quotes or change
their positioning in the continuous session.

1.4 Taxonomy of Data Used in Algorithmic Trading

Running a successful trading operation requires availability of diﬀerent data sets.
Data availability, storage, management, and cleaning are some of the most important
aspects of a functioning trading business and the core of any research environment.
The amount of data and the complexity of maintaining such a “Data Lake” can be
daunting and very few excel at this aspect. In this section, we will review the most
important data sets and their role in algorithmic trading research.

1.4.1 Reference Data

While often overlooked, or merely considered as an afterthought in the develop-
ment of a research platform,23 reliable reference data is the key foundation of a robust
quantitative strategy development. The experienced practitioner may want to skip this
section, however we encourage the neophyte to read through the tedious details to get
a better grasp of the complexity at hand.

• Trading Universe: The ﬁrst problem for the functioning of a trading operation
is knowing what instruments will be required to be traded on a particular day.
The trading universe is an evolving entity that changes daily to incorporate new
listings (IPOs), de-listings, etc. To be able to just trade new instruments, there are
several pieces of information that are required to have in multiple systems. Market
Data must be made available, and some static data needs to be set or guessed to
work with existing controls, parameters for the various analytics need to be made
available or sensibly defaulted. For research, in particular quantitative strategies,
knowing when a particular stock no longer trades is important to avoid issues like
survivor bias.

• Symbology Mapping: ISIN, SEDOL, RIC, Bloomberg Tickers, . . . Quantitative
strategies often leverage data from a variety of sources. Diﬀerent providers key
their data with diﬀerent instrument identiﬁers depending on asset class or regional
conventions, or sometimes use their own proprietary identiﬁers (e.g., Reuters
Identiﬁcation Code—RIC, Bloomberg Ticker). Therefore, symbology mapping
is the ﬁrst step in any data merging exercise. Such data is not static. One of the
symbols can change on a given day and others remain unchanged for some time,
complicating historical data merges.

23More details on research platforms are presented in Chapter 12.

Trading Fundamentals

27

It is important to note that such mapping needs to persist as point-in-time data
and allow for historical “as of date” usage, requiring the implementation of a
bi-temporal data structure. Over the course of time, some instruments undergo
) not necessarily
ticker changes (for example, from ABC to DEF on a later 𝑇0
without any particular change on the underlying asset. In such cases, market data
recorded day-by-day in a trade database will change from being keyed on ABC
to being keyed on DEF after the ticker change date 𝑇0
. This has implications for
practitioners working on data sets to build and backtest quantitative strategies.
The symbology mapping should allow for both backward and forward handling
of the changes.

For instance, in the simple example mentioned below, in order to eﬃciently back-
, a robust mapping is needed, so
test strategies over a period of time spanning 𝑇0
that it will allow to seamlessly query the data for the underlying asset in a variety
of scenarios such as:

– Signal generation: 30-day backward close time series as of date 𝑇 < 𝑇0
select close from data where date in [T-30, T], sym = ABC

:

– Signal generation: 30-day backward close time series as of date 𝑇 = 𝑇0+10:
select close from data where date in [T0-20, T0+10], sym = DEF
– Position holding: 30-day forward close time series as of date 𝑇 = 𝑇0 − 10:
select close from data where date in [T0-10, T0+20], sym = ABC

• Ticker Changes: For comparable reasons as the ones described above in the
Symbology Mapping section, one needs to maintain a historical table of ticker
changes allowing to seamlessly go up and down time series data.

• Corporate Actions Calendars: This category contains stock and cash dividends
(both announcement date and execution date), stock splits, reverse splits, rights
oﬀer, mergers and acquisitions, spin oﬀ, free ﬂoat or shares outstanding adjust-
ments, quotation suspension, etc.

Corporate actions impact the continuity of price and volume time series and, as
such, must be recorded in order to produce adjusted time series. The most com-
mon events are dividend distributions. On the day the dividend is paid, the cor-
responding amount is removed from the stock price, creating a jump in the price
time series. The announcement date might also coincide with the stock experi-
encing more volatility as investors react to the news. Consequently, recording
these events proves to be valuable in the design of quantitative strategies as one
can assess the eﬀect of dividends announcement or payment on performance, and
decide to either not hold a security that has an upcoming dividend announcement
or, conversely, to build strategies that look to beneﬁt from the added volatility.

Another type of corporate events generating discontinuity in historical time series
are stock splits or reverse splits and right oﬀers. When the price of a stock
becomes too low or too high, a company may seek to split it to bring the price

28

Algorithmic Trading and Quantitative Strategies

back to a level that is more conducive to liquid trading on exchanges.24 When a
stock experiences a 2:1 split, everything else being equal, its price will be halved
and hence, its volume will double. In order to prevent the time series from show-
ing a discontinuity, all historical data will then need to be adjusted backward to
reﬂect the split.

Mergers & Acquisitions and Spin-oﬀs are also regular events in the lifecycle of
corporations. Their history needs to be recorded in order to account for the result-
ing changes in valuation that might aﬀect a given ticker(s). These situations can
also be exploited by trading strategies known as Merger Arbitrage.

Stock quotations can be suspended as a cooling mechanism (often at the request
of the underlying company) to prevent excess price volatility when signiﬁcant
information is about to be released to the market. Depending on the circumstance,
the suspension can be temporary and intraday, or can last for extended periods of
time if the market place allows it.25 Suspensions result in gaps in data and are
worth keeping track of, as they can impact strategies in backtesting (inability to
enter or exit a position, uncertainty in the pricing of composite assets if a given
stock has a signiﬁcant weight in ETFs or Indexes, etc.). Some markets will also
suspend trading if the price swings more than a predeﬁned amount (limit up / limit
down situations), either for a period of time or for the remainder of the trading
session.

• Static Data: Country, sector, primary exchange, currency and quote factor. Static
data is also relevant for the development of quantitative trading strategies. In par-
ticular, country, currency and sector are useful to group instruments based on
their fundamental similarities. A well known example is the usage of sectors to
group stocks in order to create pairs trading strategies. It is worth noting that there
exist diﬀerent types of sector classiﬁcations (e.g., GICS®from S&P, ICB®from
FTSE) oﬀering several levels of granularity,26 and that diﬀerent classiﬁcations
might be better suited to diﬀerent asset classes or countries. The constituents of
the Japanese index TOPIX, for instance, are classiﬁed into 33 sectors that are
thought to better reﬂect the fundamental structure of the Japanese economy and
the existence of large diversiﬁed conglomerates.

Maintaining a table of the quotation currency per instrument is also necessary
in order to aggregate positions at a portfolio level. Some exchanges allow the
quotation of prices in currencies diﬀerent from the one of the country in which

24A very low price creates trading frictions as the minimum price increment might represent a large cost
relative to the stock price. A very high price might also deter retail investors from investing into a security
as it requires them to deploy too much capital per unit.

25For instance, it was the case for a large number of companies in China in 2016.

26The Global Industry Classiﬁcation Standard (GICS®) structure consists of 11 sectors, 24 industry
groups, 68 industries and 157 sub-industries. An example of this hierarchical structure would be: Industrials
/ Capital Goods / Machinery / Agricultural & Farm Machinery.

Trading Fundamentals

29

the exchange is located.27 Additionally, thus, the Quote Factor associated with
the quotation currency data needs to be stored. To account for the wide range
of currency values and preserve pricing precision, market data providers may
publish FX rates with a factor of 100 or 1000. Hence, to convert prices to USD one
needs to multiply by the quote factor: USD price = local price ⋅ fx ⋅ quote factor.
Similarly, some exchanges quote prices in cents, and the associated quotation
currency is reﬂected with a small cap letter: GBP/GBp, ZAR/ZAr, ILS/ILs, etc.

• Exchange Speciﬁc Data: Despite the electroniﬁcation of markets, individual
exchanges present a variety of diﬀerences that need to be accounted for when
designing trading strategies. The ﬁrst group of information concerns the hours
and dates of operation:

– Holiday Calendar: As not all exchanges are closed on the same day, and
trading days they are oﬀ do not always fully follow the country’s public holi-
days, it is valuable to record them, in particular in the international context.
Strategies trading simultaneously in several markets and leveraging their
correlation, may not perform as expected if one of the markets is closed
while others are open. Similarly, execution strategies in one market might
be impacted by the absence of trading in another market (for instance, Euro-
pean equity markets volume tends to be 30% to 40% lower during US market
holidays).

– Exchange Sessions Hours: These seemingly trivial data points can get
quite complex on a global scale. What are the diﬀerent available sessions
(Pre-Market session, Continuous core session, After-Hour session, etc.)?
What are the auction times as well as their respective cutoﬀ times for order
submission? Is there a lunch break restricting intraday trading? And if so,
are there auctions before and after the lunch break? The trading sessions in
the futures markets can also be quite complex with multiple phases, breaks,
as well as oﬃcial settlement times that may diﬀer from the closing time and
have an eﬀect on liquidity.
In Indonesia, for instance, markets have diﬀerent trading hours on Fridays.
Monday through Thursday the Indonesia Stock Exchange (IDX) is open
from 9:00 a.m. to 12:00 p.m., and then, from 1:30 p.m. to 4:00 p.m. On
Fridays, however, the lunch break is one hour longer and stretches from
11:30 a.m. to 2:00 p.m. This weekday eﬀect is particularly important to
consider when building volume proﬁles as discussed in Chapter 5.
Along with local times of operation, it is necessary to consider eventual
Daylight Saving Time (DST) adjustments that might aﬀect the relative trad-
ing hours of diﬀerent markets (some countries do not have DST adjustment
at all, while for countries that do have one, the dates at which it applies are

27For example, Jardine Matheson Holdings quotes in USD on the Singapore exchange while most of the

other securities quote in Singapore Dollars.

30

Algorithmic Trading and Quantitative Strategies

not always coordinated). Usually, US DST starts about two weeks prior to
its start in Europe, bringing the time diﬀerence between New York and Lon-
don to four hours instead of ﬁve hours. This results in the volume spike in
European equities associated to the US market open being one hour earlier,
requiring adjustment of volume proﬁles used for trading executions.
Some exchanges may also adjust the length of trading hours during the
course of the year. In Brazil for instance, the Bovespa continuous trading
hours are 10:00 a.m. to 5:55 p.m. from November to March, but an hour
shorter (10:00 a.m. to 4:55 p.m.) from April to October to be more consis-
tent with US market hours. Finally, within one country there might also exist
diﬀerent trading hours by venues as it is the case in Japan where the Nagoya
Stock Exchange closes 30 minutes after the major Tokyo Stock Exchange.
– Disrupted Days: Exchange outages or trading disruptions, as well as market
data issues, need to be recorded so they can be ﬁltered out when building
or testing strategies as the diﬀerence in liquidity patterns or the lack of data
quality may likely impact the overall outcome.

Additionally, exchanges also have speciﬁc rules governing the mechanics of trad-
ing, such as:

– Tick Size: The minimum eligible price increment. This can vary by instru-
ment, but also change dynamically as a function of the price of the instru-
ment (e.g., stocks under $1 can quote in increments of $0.0001, while above
that price the minimum quote increment is $0.01).

– Trade and Quote Lots: Similar to tick sizes, certain exchanges restrict the

minimum size increment for quotes or trades.

– Limit-Up and Limit-Down Constraints: A number of exchanges restrict
the maximum daily ﬂuctuations of securities. Usually, when securities reach
these thresholds, they either pause trading or can only be traded at a better
price than the limit-up limit-down threshold.

– Short Sell Restrictions: Some markets also impose execution level con-
straints on short sells (on top of potential locate requirements). For instance,
while a long sell order can trade at any price, some exchanges restrict short
sells not to trade at a price worse than the last price or not to create a new
quote that would be lower than the lowest prevailing quote. These consider-
ations are particularly important to keep in mind for researchers developing
Long-Short strategies as this impacts the ability to source liquidity.

Because these values and their potential activation threshold can vary over time,
one needs to maintain historic values as well, in order to run realistic historical
backtests.

Trading Fundamentals

31

• Market Data Condition Codes: With ever-growing complexity in market
microstructure, the dissemination of market data has grown complex as well.
While it is possible to store daily data as a single entry per day and per instru-
ment, investors building intraday strategies likely need tick by tick data of all the
events occurring in the market place. To help classify these events, exchanges
and market data providers attribute so-called condition codes to the trades and
quotes they publish. These condition codes vary per exchange and per asset class,
and each market event can be attributed to several codes at once. So, to aggregate
intraday market data properly and eﬃciently, and decide which events to keep and
which ones to exclude, it is necessary to build a mapping table of these condition
codes and what they mean: Auction trade, lit or dark trade, canceled or corrected
trade, regular trade, oﬀ-exchange trade reporting, block-size trade, trade originat-
ing from a multi-leg order such as an option spread trade, etc.

For instance, in order to assess accessible liquidity for a trading algorithm, trades
that are published for reporting purposes (e.g., negotiated transactions that hap-
pened oﬀ-exchange) must be excluded. These trades should also not be used to
update some of the aggregated daily data used in the construction of trading strate-
gies (daily volume, high, low, . . . ). Execution algorithms also extensively leverage
the distribution of intraday liquidity metrics to gauge their own participation in
auctions and continuous sessions, or in lit versus dark venues, therefore requiring
a precise classiﬁcation of intraday market data.

• Special Day Calendars: Over the course of a year, some days present certain
distinct liquidity characteristics that need to be accounted for in both execution
strategies and in the alpha generation process. The diversity of events across mar-
kets and asset classes can be quite challenging to handle. Among the irregular
events that aﬀect liquidity in equity markets that need to be accounted for, we
can mention the following non-exhaustive list for illustration purposes only: Half
trading days preceding Christmas and following Thanksgiving in the US or on the
Ramadan eve in Turkey, Taiwanese market opening on the weekend to make up
for lost trading days during holiday periods, Korean market changing its trading
hours on the day of the nationwide university entrance exam, Brazilian market
opening late on the day following the Carnival, etc.

There are also special days that are more regular and easier to handle. The last
trading days of the months and quarters, for instance, tend to have additional trad-
ing activity as investors rebalance their portfolios. Similarly, options and futures
expiry days (quarterly/monthly expiry, ‘Triple Witching’28 in the US, Special
Quotations in Japan, etc.) tend to experience excess trading volume and diﬀerent
intraday patterns resulting from hedging activity and portfolio adjustments. Con-
sequently, they need to be handled separately, in particular, when modeling trad-

28Triple witching days happen four times a year on the third Friday of March, June, September and
December. On these days, the contracts for stock index futures, stock index options and stock options
expire concurrently.

32

Algorithmic Trading and Quantitative Strategies

ing volume. As most execution strategies make use of relatively short interval vol-
ume metrics (e.g., 30-day or 60-day ADV), one single data point can impact the
overall level inferred. Similarly autoregressive models of low order may underper-
form both on special days and on the days following them. As a result, modelers
often remove special days and model normal days ﬁrst. Then, special days are
modeled separately, either independently or using the normal days as a baseline.

In order to reﬂect changes in the market and remain consistent with index inclu-
sion rules, most indices need to undergo regular updates of the constituents and
their respective weights. The signiﬁcant indices do so at regular intervals (annu-
ally, semi-annually or quarterly) on a pre-announced date. At the close of business
of that day, some stocks might be added to the index while others are removed,
or the weight of each stock in the index might be increased or decreased. These
events, known as index rebalances, are particularly relevant to passive investors
who are tracking the index. In order to minimize the tracking error to the bench-
mark index, investors need to also adjust their holdings accordingly. Additionally,
as most funds are benchmarked at the close price, there is an incentive for fund
managers to try to rebalance their holdings at a price as close as possible to the
oﬃcial close price, on the day the index rebalance becomes eﬀective. As a result,
on these days, intraday volume distribution is signiﬁcantly skewed toward the end
of day and requires some adjustment in the execution strategies.

• Futures-Speciﬁc Reference Data: Futures contracts present particular charac-
teristics requiring additional reference data to be collected. One of the core dif-
ferences of futures contracts compared to regular stocks is the fact that instru-
ments have an expiry date, after which the instrument ceases to exist. For the
purpose of backtesting strategies, it is necessary to know which contract was live
at any point in time through the use of an expiry calendar, but also which con-
tract was the most liquid. For instance, equity index futures tend to be the most
liquid, for the ﬁrst contract available (also known as front month), while energy
futures such as oil tend to be more liquid for the second contract. While this may
appear to be trivial, when building a trading strategy and modeling price series,
it is particularly important to know which contract carries the most signiﬁcant
price formation characteristics and what is the true liquidity available in order to
properly estimate the market impact.

The task of implementing a futures expiry calendar is further complicated by
the fact there is no real standardized frequency that applies across markets. For
instance, European equity index futures tend to expire monthly, while US index
futures expire quarterly. Some contracts even follow an irregular cycle through
the course of the year as it is the case for grain futures (e.g., wheat) that were

Trading Fundamentals

33

originally created for hedging purposes and as a result have expiry months that
follow the crop cycle.29

The fact futures contracts expire on a regular basis has further implications in
terms of liquidity. If investors holding these contracts want to maintain their
exposure for longer than the lifespan of the contract, they need to roll over their
positions onto the next contract which might have a noticeably diﬀerent liquidity
level. For instance, the front contract of the S&P 500 (e.g., ESH8) trades roughly
1.5 million contracts per day in the weeks preceding expiry while the next month
contract (ESM8) only trades about 30,000 contracts per day. However, this liq-
uidity relationship will invert in the few days leading to the expiry of the front
contract as most investors roll their positions, and the most liquid contract will
become the back month contract (see Figure 1.4). As a result, when computing
rolling-window metrics (such as average daily volume for instance), it is neces-
sary to account for potential roll dates that may have happened during the time
span. In the example above, a simple 60-day average daily volume on ESM8 taken
in early April 2018 would capture a large number of days with very low volume
(January-March) owing to the fact the most liquid contract at the time was the
ESH8 contract, and would not accurately represent the volume activity of the
S&P 500 futures contract. A more appropriate average volume metric to be used
as a forward looking value for execution purposes would blend the volume time
series of ESH8 prior to the roll date, and ESM8 after the roll date.30 Additionally,
in order to eﬃciently merge futures positions with other assets in an investment
strategy, reference data relative to the quotation of these contracts, contract size
(translation between the quotation points and the actual monetary value), cur-
rency, etc., must be stored.

Finally, futures markets are characterized by the existence of diﬀerent market
phases during the day, with signiﬁcantly diﬀerent liquidity characteristics. For
instance, equity index futures are much more liquid during the hours when the
corresponding equities markets are open. However, one can trade during the
overnight session if they want to. The overnight session being much less liquid,
the expected execution cost tends to be higher, and as such, the various market
data metrics (volume proﬁle, average spread, average bid-ask sizes, ...) should be
computed separately for each market phase, which requires maintaining a table
of the start and end times of each session for each contract.

29US Wheat Futures expire in March, May, July, September and December.

30It is worth noting that diﬀerent contracts ‘roll’ at diﬀerent speeds. While for monthly expiry contracts
it is possible to see most of the open interest switch from the front month contract to the back month on the
day prior to the expiry. For quarterly contracts it is not uncommon to see the roll happen over the course
of a week or more, and the front month liquidity vanish several days ahead of the actual expiry. Careful
modeling is recommended on a case by case basis.

34

Algorithmic Trading and Quantitative Strategies

Figure 1.4: Futures Volume Rolling.

• Options-Speciﬁc Reference Data (Options Chain): Similar to futures contracts,
options contracts present a certain number of speciﬁcities for which reference data
need to be collected. On top of the similar feature of having a particular expiry
date, options contracts are also deﬁned by their strike price. The combination of
expiries and strikes is known as the option chain for a given underlier. The ability
to map equity tickers to option tickers and their respective strike and expiry dates
allows for the design of more complex investment and hedging strategies. For
instance, distance to strike, change in open interest of puts and calls, etc., can all
be used as signals for the underlying security price. For an interesting article on
how deviations in put-call parity contains information about future equity returns,
refer to Cremers and Weinbaum (2010) [95].

• Market-Moving News Releases: Macro-economic announcements are known
for their ability to move markets substantially. Consequently, it is necessary to
maintain a calendar of dates and times of their occurrences in order to assess
their impact on strategies and decide how best to react to them. The most com-
mon ones are central banks’ announcements or meeting minutes releases about
the major economies (FED/FOMC, ECB, BOE, BOJ, SNB), Non-Farm Payrolls,
Purchasing Managers’ Index, Manufacturing Index, Crude Oil Inventories, etc.
While these news releases impact the broad market or some sectors, there are
also stock speciﬁc releases that need to be tracked: Earning calendars, special-
ized sector events such as FDA results for the healthcare and biotech sectors, etc.

• Related Tickers: There is a wide range of tickers that are related to each other,
often because they fundamentally represent the same underlying asset. Main-
taining a proper reference allows to eﬃciently exploit opportunities in the mar-
ket. Some non-exhaustive examples include: Primary tickers to composite tickers
mapping (for markets with fragmented liquidity), dual listed/fungible securities

Trading Fundamentals

35

in US and Canada, American Depository Receipt (ADR) or Global Depository
Receipt (GDR), local and foreign boards in Thailand, etc.

• Composite Assets: Some instruments represent several underlying assets (ETFs,
Indexes, Mutual Funds, . . . ). Their rise in popularity as investments over time
makes them relevant for quantitative strategies. They can be used as eﬃcient vehi-
cles to achieve desired exposures (sector and country ETFs, thematic factor ETFs,
. . . ), or as cheap hedging instruments, and they can provide arbitrage opportuni-
ties when they deviate from their Net Asset Value (NAV). In order to be leveraged
in quantitative strategies, one needs to maintain a variety of information such as a
time series of their constituents and the value of any cash component, the divisor
used to translate the NAV into the quoted price, the constituent weights.

• Latency Tables: This last type of data would only be of interest for developing
strategies and for research in the higher frequency trading space. For these, it
might be relevant to know the distribution of latency between diﬀerent data cen-
ters as they can be used for more eﬃcient order routing as well as reordering data
that may have been recorded in diﬀerent locations.

While the above discussion provides a non-exhaustive list of issues on the refer-
ence data available to build a quantitative research platform, they highlight the chal-
lenges that must be taken into account when designing and implementing algorithmic
trading strategies. Once in place, a proper set of reference data will allow the quan-
titative trader to systematically harness the actual content of various types of data
(described later) without being caught oﬀ-guard by the minutiae of trading.

1.4.2 Market Data

While historically a large swath of modeling for developing strategies was carried
out on daily or minute bar data sets, the past ﬁfteen years have seen a signiﬁcant
rise in the usage of raw market data in an attempt to extract as much information
as possible, and act on it before the opportunity (or market ineﬃciency) dissipates.
Market data, itself, comes in various levels of granularity (and price!) and can be
subscribed to either directly from exchanges (known as “direct feeds”) or from data
vendors aggregating and distributing it. The level of detail of the feed subscribed to
is generally described as Level I, Level II or Level III market data.

Level I Data: Trade and BBO Quotes: Level I market data is the most basic form
of tick-by-tick data. Historically, the Level I data feed would only refer to trade infor-
mation (Price, Size, Time of each trade reported to the tape) but has grown over time
into a term generally accepted to mean both trades and top of book quotes. While the
trade feed updates with each trade printed on the tape, the quote feed tends to update
much more frequently each time liquidity is added to, or removed from, the top of
book (a rough estimate in liquid markets is the quotes present more updates than the
trades by about an order of magnitude).

36

Algorithmic Trading and Quantitative Strategies

In order to build strategies, the timing of each event must be as precise as possi-
ble. While the time reported for trade and quotes is the matching engine time at the
exchange, most databases also store a reception or record time to reﬂect the potential
latency between the trade event and one being aware that it did happen and being able
to start making decisions on it. Accounting for this real-world latency is a necessary
step for researchers building strategies on raw market data for which opportunities
may be very short lived and may be impossible to exploit by participants who are not
fast enough.

The Level I data is enough to reconstruct the Best Bid and Oﬀer (BBO) of the
market. However, in fragmented markets, it is also useful to obtain an aggregated
consolidated view of all available liquidity at a given price level across all exchanges.
Market data aggregators usually provide this functionality for users who do not wish
to reconstruct the full order book themselves.

Finally, it is worth noting that even Level I data contains signiﬁcant additional
information in the form of trade status (canceled, reported late, etc.) and, trade and
quote qualiﬁers. These qualiﬁers provide granular details such as whether a trade was
an odd lot, a normal trade, an auction trade, an Intermarket Sweep, an average price
reporting, on which exchange it took place, etc. These details can be used to better
analyze the sequence of events and decide if a given print should be used to update
the last price and total volume traded at that point in time or not. For instance, if not
processed appropriately, a trade reported to the tape out of sequence could result in a
large jump in price because the market may have since moved, and consequently the
trade status is an indication that the print should not be utilized as it is.

Capturing raw market data, whether to build a research database or to process
it in real-time to make trading decisions, requires signiﬁcant investments and expert
knowledge to handle all the inherent complexity. Hence, one should carefully con-
sider the trade-oﬀ between the expected value that can be extracted from the extra
granularity, and the additional overhead compared to simpler solutions such as using
binned data.

Level II Data: Market Depth: Level II market data contains the same information as
Level I, but with the addition of quote depth data. The quote feed displays all lit limit
order book updates (price changes, addition or removal of shares quoted) at any level
in the book, and for all of the lit venues in fragmented markets. Given the volume of
data generated and the decreasing actionable value of quote updates as their distance
to top of book increases, some users limit themselves to the top ﬁve or ten levels of
the order book when they collect and/or process the data.

Level III Data: Full Order View: Level III market data—also known as message
data—provides the most granular view of the activity in the limit order book. Each
order arriving is attributed a unique ID, which allows for its tracking over time, and
is precisely identiﬁed when it is executed, canceled or amended. Similar to Level II,
the data set contains intraday depth of book activity for all securities in an exchange.
Once all messages from diﬀerent exchanges are consolidated into one single data set
ordered by timestamp, it is possible to build a full (with national depth) book at any

Trading Fundamentals

37

moment intraday. For illustration purposes, we take the example of US Level III data
and provide a short description below in Table 1.10.

Table 1.10: Level III Data

Variable:

Description

Timestamp: Number of milliseconds after the midnight.

Ticker:

Equity symbol (up to 8 characters)

Order:

Unique order ID.

T:

Message type. Allowed values:

• “B”—Add buy order

• “S”—Add sell order

• “E”—Execute outstanding order in part

• “C”—Cancel outstanding order in part

• “F”—Execute outstanding order in full

• “D”—Delete outstanding order in full

• “X”—Bulk volume for the cross event

• “T”—Execute non-displayed order

Shares:

Price:

Order quantity for the “B,” “S,” “E,” “X,” “C,” “T” messages. Zero
for “F” and “D” messages.

Order price, available for the “B,” “S,” “X” and “T” messages.
Zero for cancellations and executions. The last 4 digits are decimal
digits. The decimal portion is padded on the right with zeros. The
decimal point is implied by position; it does not appear inside the
price ﬁeld. Divide by 10000 to convert into currency value.

MPID:

Market Participant ID associated with the transaction (4 characters)

MCID:

Market Center Code (originating exchange—1 character)

While the display and issuance of a new ID to the modiﬁed order varies from

exchange to exchange, a few special types of orders are worth mentioning:

1. Order subject to price sliding: The execution price could be one cent worse than the
display price at NASDAQ; it is ranked at the locking price as a hidden order, and

38

Algorithmic Trading and Quantitative Strategies

is displayed at the price, one minimum price variation (normally 1 cent) inferior
to the locking price. New order ID will be used if the order is replaced as a display
order. At other exchanges the old order ID will be used.

2. Pegged order: Based on NBBO, not routable, new timestamp given upon re-

pricing; display rules vary over exchanges.

3. Mid-point peg order: Non-displayed, can result in half-penny execution.

4. Reserve order: Displayed size is ranked as a displayed limit order and the reserve
size is behind non-displayed orders and pegged orders in priority. The minimum
display quantity is 100 and this amount is replenished from the reserve size when
it falls below 100 shares. A new timestamp is created and the displayed size will
be re-ranked upon replenishment.

5. Discretionary order: Displayed at one price while passively trading at a more
aggressive discretionary price. The order becomes active when shares are avail-
able within the discretionary price range. The order is ranked last in priority. The
execution price could be worse than the display price.

6. Intermarket sweep order: Order that can be executed without the need for checking

the prevailing NBBO.

The richness of the data set allows sophisticated players, such as market makers,
to know not only the depth of book at a given price and the depth proﬁle of the book
on both sides, but more importantly the relative position of an order from the top
position of the side (buy or sell). Among the most common granular microstructure
behaviors studied with Level III data, we can mention:

• The pattern of inter-arrival times of various events.

• Arrival and cancellation rates as a function of distance from nearest touch price.

• Arrival and cancellation rates as a function of other available information, such

as in the queue on either side of the book, order book imbalance, etc.

Once modeled, these behaviors can, in turn, be employed to design more sophisticated
strategies by focusing on trading related questions:

• What is the impact of market order on the limit order book?

• What are the chances for a limit order to move up the queue from a given entry

position?

• What is the probability of earning the spread?

• What is the expected direction of the price movement over a short horizon?

Trading Fundamentals

39

As illustrated by the diversity and granularity of information available to traders,
markets mechanics have grown ever more complex over time, and a great deal of
peculiarities (in particular at the reference data level) need to be accounted for in
order to design robust and well performing strategies. The proverbial devil is always
in the details when it comes to quantitative trading but while it might be tempting
to go straight to the most granular source of data, in practice—with the exception
of truly high frequency strategies—most practitioners build their algorithmic trading
strategies relying essentially on binned data. This simpliﬁes the data collection and
handling processes and also greatly reduces the dimension of the data sets so one can
focus more on modeling rather than data wrangling. Most of the time series models
and techniques described in subsequent chapters are suited to daily or binned data.

1.4.3 Market Data Derived Statistics

Most quantitative strategies, even the higher frequency strategies that leverage
more and more granular data also leverage derived statistics from the binned data,
such as daily data. Here we give a list of the most common ones used by practitioners
and researchers alike.

Daily Statistics

The ﬁrst group represents the overall trading activity in the instrument:

• Open, High, Low, Close (OHLC) and Previous Close Price: The OHLC pro-
vides a good indication of the trading activity as well as the intraday volatility
experienced by the instrument. The distance traveled between the lowest and
highest point of the day usually gives a better indication of market sentiment
than the simple close-to-close return. Keeping the previous close value as part
of the same time series is also a good way to improve computation eﬃciency by
not having to make an additional database query to compute the daily return and
overnight gap. The previous close needs, however, to be properly adjusted for
corporate actions and dividends.

• Last Trade before Close (Price/Size/Time): It is useful in determining how
much the close price may have jumped in the ﬁnal moments of trading, and con-
sequently, how stable it is as a reference value for the next day.

• Volume: It is another valuable source of trading activity indicator, in particu-
lar when the level jumps from the long term average. It is also worth collecting
the volume breakdown between lit and dark venues, in particular for execution
strategies.

• Auctions Volume and Price: Depending on the exchange, there can be multiple
auctions in a day (Open, Close, Morning Close and Afternoon Open—for mar-
kets with a lunch break—as well as ad hoc liquidity or intraday auctions). Owing
to their liquidity aggregation nature, they can be considered as a valuable price
discovery event when signiﬁcant volume prints occur.

40

Algorithmic Trading and Quantitative Strategies

• VWAP: Similar to the OHLC, the intraday VWAP price gives a good indication
of the trading activity on the day. It is not uncommon to build trading strategies
using VWAP time series instead of just close-to-close prices. The main advantage
being that VWAP prices represent a value over the course of the day and, as such,
for larger orders are easier to achieve through algorithmic execution than a single
print.

• Short Interest/Days-to-Cover/Utilization: This data set is a good proxy for
investor positioning. The short pressure might be an indication of upcoming short
term moves: A large short interest usually indicates a bearish view from institu-
tional investors. Similarly, the utilization level of available securities to borrow
(in order to short) gives an indication of how much room is left for further short-
ing (when securities become “Hard to Borrow”, the cost of shorting becomes
signiﬁcantly higher requiring short sellers to have strong enough beliefs in the
short term price direction). Finally, days-to-cover data is also valuable to assess
the magnitude of a potential short squeeze. If short sellers need to unwind their
positions, it is useful to know how much volume this represents as a fraction
of available daily liquidity. The larger the value, the larger the potential sudden
upswing on heavily shorted securities.

• Futures Data: Futures markets provide additional insight into the activity of large
investors through open interest data, that can be useful to develop alpha strategies.
Additionally, ﬁnancial futures oﬀer arbitrage opportunities if their basis exhibits
mispricing compared to one’s dividend estimates. As such, recording the basis of
futures contracts is worthwhile even for strategies that do not particularly target
futures.

• Index-Level Data: This is also a valuable data set to collect as a source of rel-
ative measures for instrument speciﬁc features (Index OHLC, Volatility, . . . ). In
particular for dispersion strategies, normalized features help identify individual
instruments deviating from their benchmarks.

• Options Data: The derivative market is a good source of information about the
positioning of traders through open interest and Greeks such as Gamma and Vega.
How the broader market is pricing an instrument through implied volatility for
instance is of interest.

• Asset Class Speciﬁc: There is a wealth of cross-asset information available when
building strategies, in particular in Fixed Income, FX, and Credit markets. Among
the basic ones, we would note:

– Yield / benchmark rates (repo, 2y, 10y, 30y)
– CDS Spreads
– US Dollar Index

The second group of daily data represents granular intraday microstructure activ-

ity and is mostly of interest to intraday or execution trading strategies:

Trading Fundamentals

41

• Number and Frequency of Trades: A proxy for the activity level of an instru-
ment, and how continuous it is. Instruments with a low number of trades are
harder to execute and can be more volatile.

• Number and Frequency of Quote Updates: Similar proxy for the activity level.

• Top of Book Size: A proxy for liquidity of the instrument (larger top of book size

makes it possible to trade larger order size quasi immediately, if needed).

• Depth of Book (price and size): Similar proxy for liquidity.

• Spread Size (average, median, time weighted average): This provides a proxy
for cost of trading. A parametrized distribution of spread size can be used to iden-
tify intraday trading opportunities if they are cheap or expensive.

• Trade Size (average, median): Similar to spread size, trade sizes and their dis-
tribution are useful to identify intraday liquidity opportunities when examining
the volume available in the order book.

• Ticking Time (average, median): The ticking time and its distribution is a rep-
resentation of how often, one should expect changes in the order book ﬁrst level.
This is particularly helpful for execution algorithms for which the frequency of
updates (adding/canceling child orders, reevaluating decisions, etc.) should be
commensurate with the characteristics of the traded instrument.

The daily distributions of these microstructure variables can be used as start of
day estimates in trading algorithms and be updated intraday as additional data
ﬂows in through online Bayesian updates.

Finally, the last group of daily data can be derived from the previous two groups
through aggregation but is usually stored pre-computed in order to save time during
the research phase (e.g., X-day trailing data), or to be used as normalizing values
(e.g., size as a percentage of ADV, spread in relation to long-term average, . . . ). Some
common examples would be:

• 𝑋-day Average Daily Volume (ADV) / Average auction volume

• 𝑋-day volatility (close-to-close, open-to-close, etc.)

• Beta with respect to an index or a sector (plain beta, or asymmetric up-days/down-

days beta)

• Correlation matrix

As a reminder to the reader, aggregated data need to fully support the peculiar-
ities described in the Reference Data section (for instance: The existence of special
event days which, if included, can signiﬁcantly skew intraday distribution of values;
mishandling of market asynchronicity resulting in inaccurate computations of key
quantities such as beta or correlation, etc.).

42

Binned Data

Algorithmic Trading and Quantitative Strategies

The ﬁrst natural extension to daily data sets is a discretization of the day into bins
ranging from a few seconds to 30 minutes. The features collected are comparable to
the ones relevant for daily data sets (period volume, open, high, low, close, VWAP,
spread, etc.), but computed at higher frequency. It is worth mentioning that minute
bar data actually underpins the vast majority of microstructure models used in the
electronic execution space. Volume and spread proﬁles, for instance, are rarely built
with a granularity ﬁner than one minute to prevent introducing excess noise due purely
to market frictions. The major advantage of binned-data is that discrete time series
methods can be readily used.

These minute bar data sets are also quite popular for the backtesting of low-to-
medium frequency trading strategies targeting intraday alpha (short duration market
neutral long-short baskets, momentum and mean-reversion strategies, etc.). The main
beneﬁt they provide is a signiﬁcant dimension reduction compared to raw market
data (390 rows per stock per day in the US compared to millions for raw data for the
liquid stocks), which allows researchers to perform rapid and eﬃcient backtesting as
they search for alpha. However, increasing data frequency from daily data to intraday
minute bars also presents challenges. In particular, relationships that appear to be
stable using close-to-close values become much noisier as granularity increases and
signals become harder to extract (e.g., drop in correlation between assets, pricing
ineﬃciency moves due to sudden liquidity demand, etc.). Similarly, for less liquid
assets with a low trade frequency, there may not be any trading activity for shorter
durations, resulting in empty bins.

1.4.4 Fundamental Data and Other Data Sets

A very large number of quantitative investment strategies are still based on funda-
mental data, and consequently countless research papers are available to the interested
reader describing examples of their usage. Here we will only describe the main classes
of existing fundamental data:

• Key Ratios: EPS (Earnings Per Share), P/E (Price-to-Earning), P/B (Price-to-
Book Value), . . . . These metrics represent a normalized view of the ﬁnancials
of companies allowing for easier cross-sectional comparison of stocks and their
ranking over time.

• Analyst Recommendations: Research analysts at Sell Side institutions spend a
great deal of resources analyzing companies they cover, in order to provide invest-
ment recommendations usually in the form of a Buy/Hold/Sell rating accompa-
nied by a price target. While individual recommendations might prove noisy, the
aggregate values across a large number of institutions can be interpreted as a
consensus valuation of a given stock, and changes in consensus can have a direct
impact on price.

• Earnings Data: Similarly research analysts also provide quarterly earning esti-
mates that can be used as an indication of the performance of a stock before the

Trading Fundamentals

43

actual value gets published by the company. Here, too, consensus values tend to
play a larger role. In particular, when the diﬀerence between the forecast consen-
sus and the realized value is large (known as earning surprise), as the stock might
then experience outsized returns in the following days. Thus, collecting analysts’
forecasts as well as realized values can be a valuable source of information in the
design of trading strategies.

• Holders: In some markets, large institutional investors are required to disclose
their holdings on a regular basis. For instance, in the US, institutional investment
managers with over $100 million in assets must report quarterly, their holdings,
to the SEC using Form 13F. The forms are then publicly available via the SEC’s
EDGAR database. Additionally, shareholders might be required to disclose their
holdings once they pass certain ownership thresholds.31 Sudden changes in such
ownership might indicate changes in sentiment by sophisticated investor and can
have a signiﬁcant impact on stock performance.

• Insiders Purchase/Sale: In some markets, company directors are required by law
to disclose their holdings of the company stock as well as any increase or decrease
of such holdings.32 This is thought to be an indicator of future stock price moves
from the group of people who have access to the best possible information about
the company.

• Credit Ratings: Most companies issue both stocks and bonds to ﬁnance their
operations. The credit ratings of bonds and their changes over time provide addi-
tional insight into the health of a company and are worth leveraging. In particular,
credit downgrades resulting in higher funding costs in the future, generally have
a negative impact on equity prices.

• And much, much more: Recent years have seen the emergence of a wide variety
of alternative data sets that are available to researchers and practitioners alike.33
While it is not possible to make a comprehensive list of all that are available, they
can generally be classiﬁed based on their characteristics: Frequency of publica-
tion, structured or unstructured, and velocity of dissemination. The value of such
data depends on the objectives and resources of the user (natural language pro-
cessing or image recognition for unstructured data require signiﬁcant time and
eﬀorts), but also—and maybe more importantly—on the uniqueness of the data

31In the US, Form 13D must be ﬁled with the SEC within 10 days by anyone who acquires beneﬁcial

ownership of more than 5% of any class of publicly traded securities in a public company.

32In the US, oﬃcers and directors of publicly traded companies are required to disclose their initial
holdings in the company by ﬁling Form 3 with the SEC, as well as Form 4 within 2 days of any subsequent
changes. The forms are then publicly available via the SEC’s EDGAR database.

33For instance, www.orbitalinsight.com oﬀers daily retail traﬃc analytics, derived from satellite
imagery analysis monitoring over 260,000 parking lots, as well as estimates of oil inventories through
satellite monitoring of oil storage facilities.

44

Algorithmic Trading and Quantitative Strategies

set. As more investors get access to it, the harder it becomes to extract meaningful
alpha from it.

1.5 Market Microstructure: Economic Fundamentals of Trading
We ﬁrst provide a brief review of market microstructure, an area of ﬁnance that
studies how the supply and demand of liquidity results in actual transactions and mod-
iﬁes the subsequent state of the market, and then delve into some critical operational
concepts. We draw upon key review papers by Madhavan (2000) [254], Biais, Glosten
and Spatt (2005) [39] and O’Hara (2015) [276]. The core idea is that the eﬃcient mar-
ket hypothesis, which postulates that the equity price impounds all the information
about the equity, may not hold due to market frictions. Algorithmic trading essentially
exploits the speed with which investors acquire the information and how they use it
along with the market frictions that arise mainly due to demand-supply imbalances.
Taking an informational economics angle, Madhavan (2000) [254] provides a market
microstructure analysis framework following three main categories.

– Price Formation and Price Discovery: How do prices impound information over

time and how do the determinants of trading costs vary?

– Market Design: How do trading rules aﬀect price formation?

Market design generally refers to a set of rules that all players have to follow in
the trading process. These include the choice of tick size, circuit breakers which
can halt trading in the event of large price swings, the degree of anonymity and
the transparency of the information to market participants, etc. Markets around
the world and across asset classes can diﬀer signiﬁcantly in these types of rules,
creating a diverse set of constraints and opportunity for algorithmic traders. Some
early research on the eﬀects of market design lead to following broad conclusions:

– Centralized trading via one single market tends to result in more eﬃcient price
discovery with smaller bid-ask spreads. In the presence of multiple markets, the
primary markets (such as NYSE, NASDAQ) remain the main sources of price
discovery (see Hasbrouck (1995) [181]).

– Despite market participants’ preference for continuous, automated limit order
book markets, theoretical models suggest that multilateral trading approaches
such as single-price call auctions are the most eﬃcient in processing diverse
information (See Mendelson (1982) [263] and Ho et al. (1985) [197]).

– Transparency: How do the quantity, quality and speed of information provided to

market participants aﬀect the trading process?

Transparency which is broadly classiﬁed into pre-trade (lit order book) and post-
trade (trade reporting to the public, though at diﬀerent time lags) is often a trade-oﬀ.

Trading Fundamentals

45

While in theory more transparency should lead to better price discovery, the wide
disclosure of order book depth information can lead to thinner posted sizes and
wider bid-ask spreads if participants fear revealing their intent and possibly their
inventory levels, leading them to favor more oﬀ-exchange activity.

All the above points can be analyzed in light of the rapid growth of high frequency
trading (HFT) described earlier in this chapter. O’Hara (2015) [276] presents issues
related to microstructure in the context of HFT. While the basic tenet that traders may
use private information or learn from market data such as orders, trade size, volume,
duration between successive trades, etc., has remained the same, the trading is now
mostly automated to follow some rules. These rules are based partly on prior informa-
tion and partly on changing market conditions, monitored through order ﬂows. With
the high speed, adverse selection has taken a diﬀerent role. Some traders may have
access to market data milliseconds before others have it and this may allow them to
capture short term price movements. This would expand the pool of informed traders.
Here are some topics for research in microstructure:

– With a parent order sliced into several child orders that are sent to market for exe-
cution during the course of trading, it is diﬃcult to discern who is the informed
trader. Informed traders use sophisticated dynamic algorithms to interact with the
market. Retail (known in the literature as ‘uninformed’) trades usually cross the
spread.

– More work needs to be done to understand trading intensity in short intervals. Order

imbalance is empirically shown to be unrelated to price levels.

– Informed traders may increasingly make use of hidden orders. How these orders

enter and exit the markets require further studies.

– Traders respond to changing market conditions by revising their quoted prices. The
quote volatility can provide valuable information about the perceived uncertainty
in the market.

Although HFT has resulted in more eﬃcient markets, with lower bid-ask spreads,
the data related to trade sequences, patterns of cancellations across fragmented mar-
kets require new tools for analysis and for making actionable inference. In this review,
we do not present any models explicitly and these will be covered throughout this
book. Now keeping in line with the practical perspectives of this book, we highlight
some key concepts.

1.5.1 Liquidity and Market Making
(a) A Deﬁnition of Liquidity: Financial markets are commonly described as carry-
ing the function of eﬃciently directing the ﬂow of savings and investments to the
real economy to allow the production of goods and services. An important factor
contributing to well-developed ﬁnancial markets is in facilitating liquidity which
enables investors to diversify their asset allocation and the transfers of securities
at a reasonable transaction cost.

46

Algorithmic Trading and Quantitative Strategies

Properly describing “liquidity” often proves to be elusive as there is no commonly
agreed upon deﬁnition. Black (1971) [42] proposes a relatively intuitive descrip-
tion of a liquid market: “The market for a stock is liquid if the following conditions
hold:

• There are always bid and ask prices for the investor who wants to buy or

sell small amounts of stock immediately.

• The diﬀerence between the bid and ask prices (the spread) is always small.
• An investor who is buying or selling a large amount of stock, in the absence
of special information, can expect to do so over a long period of time at a
price not very diﬀerent, on average, from the current market price.

• An investor can buy or sell a large block of stock immediately, but at a
premium or discount that depends on the size of the block. The larger the
block, the larger the premium or discount.”

In other words, Black deﬁnes a liquid market as a continuous market having
the characteristics of relatively tight spread, with enough depth on each side of
the limit order book to accommodate instantaneous trading of small orders, and
which is resilient enough to allow large orders to be traded slowly without signif-
icant impact on the price of the asset. This general deﬁnition remains particularly
well suited to today’s modern electronic markets and can be used by practition-
ers to assess the diﬀerence in liquidity between various markets when choosing
where to deploy a strategy.

). If 𝑝∗
𝑡

(b) Model for Market Friction: We begin with a model to accommodate the friction
in stock price (𝑃𝑡
is the (log) true value of the asset which can vary over
time due to expected cash ﬂows or due to variation in the discount rate. Given the
publicly available information and the assumption of market eﬃciency, we have:
. Therefore the observed return, 𝑟𝑡 = 𝑝𝑡 − 𝑝𝑡−1 =
𝑝𝑡 = 𝑝∗
𝜖𝑡 + (𝑎𝑡 − 𝑎𝑡−1) can exhibit some (negative) serial correlation, mainly a result
’ can be a function of a number of factors such as
of friction. The friction, ‘𝑎𝑡
inventory costs, risk aversion, bid-ask spread, etc. When 𝑎𝑡 = 𝑐𝑠𝑡
is the
direction of the trade and ‘𝑐’ is the half-spread, the model is called a Roll model.
This can explain the stickiness in returns in some cases.

, where 𝑠𝑡

𝑡−1 + 𝜖𝑡

𝑡 = 𝑝∗

and 𝑝∗

𝑡 + 𝑎𝑡

(c) Diﬀerent Styles of Market Participants: Liquidity Takers and Liquidity
Providers: Liquid ﬁnancial markets carry their primary economic function by
facilitating savings and investment ﬂows as well as allowing investors to exchange
securities in the secondary markets. Economic models of ﬁnancial markets
attempt to classify market participants into diﬀerent categories. These can be
broadly delineated as:

Informed Traders, making trading decisions based on superior information that
is not yet fully reﬂected in the asset price. That knowledge can be derived from
either fundamental analysis or information not directly available nor known to
other market participants.

Trading Fundamentals

47

News Traders, making trading decisions based on market news or announce-
ments and trying to make proﬁts by anticipating the market’s response to a par-
ticular catalyst. The electroniﬁcation of news dissemination oﬀers new opportuni-
ties for developing quantitative trading strategies by leveraging text-mining tools
such as natural language processing to interpret, and trade on, machine readable
news before it is fully reﬂected in the market prices.

Noise Traders (as introduced by Kyle (1985) [234]), making trading decisions
without particular information and at random times mainly for liquidity reasons.
They can be seen as adding liquidity to the market through additional volume
transacted, but only have a temporary eﬀect on price formation. Their presence
in the market allows informed traders not to be immediately detected when they
start transacting, as market makers cannot normally distinguish the origin of the
order ﬂow between these two types of participants. In a market without noise
traders, being fully eﬃcient, at equilibrium each trade would be revealing infor-
mation that would instantly be incorporated into prices, hereby removing any
proﬁt opportunities.

Market Makers, providing liquidity to the market with the intent of collecting
proﬁts originating from trading frictions in the market place (bid-ask spread).
Risk-neutral market makers are exposed to adverse selection risk arising from
the presence of informed traders in the marketplace, and therefore establish their
trading decisions mostly based on their current inventory. As such, they are often
considered in the literature to drive the determination of eﬃcient prices by acting
as the rational intermediaries.

Generalizing these concepts, market participants and trading strategies can be
separated between liquidity providing and liquidity seeking. The former being
essentially the domain of market makers whose level of activity, proxied by mar-
ket depth, is proportional to the amount of noise trading and inversely propor-
tional to the amount of informed trading (Kyle (1985) [234]). The latter being
the domain of the variety of algorithmic trading users, described before (mutual
funds, hedge funds, asset managers, etc.). Given the key role played by market
makers in the liquidity of electronic markets, they have been the subject of a large
corpus of academic research focusing on their activities.

(d) The Objectives of the Modern Market Maker: At present, market making can
broadly be separated into two main categories based on the trading characteris-
tics. The ﬁrst one is the provision of large liquidity—known as blocks—to institu-
tional investors, and has traditionally been in the realm of sell-side brokers acting
as intermediaries and maintaining signiﬁcant inventories. Such market makers
usually transact through a non-continuous, negotiated, process based on their cur-
rent inventory, as well as their assessment of the risk involved in liquidation of the
position in the future. Larger or more volatile positions generally tend to come at
a higher cost, reﬂecting the increased risk for the intermediary. But they provide

48

Algorithmic Trading and Quantitative Strategies

the end investor with a certain price and an immediate execution bearing no tim-
ing risk that is associated with execution over time. These transactions, because
they involve negotiations between two parties, still mostly happen in a manual
fashion or over the phone and then get reported to an appropriate exchange for
public dissemination.

The second category of market making involves the provision of quasi-
continuous, immediately accessible quotes on an electronic venue. With the
advent of electronic trading described in the previous section, market makers
originally seated on the exchange ﬂoors have progressively been replaced by elec-
tronic liquidity providers (ELP). The ELP leverage fast technology to dissemi-
nate timely quotes across multiple exchanges and develop automated quantita-
tive strategies to manage their inventory and the associated risk. As such, most
ELP can be classiﬁed as high frequency traders. They derive their proﬁts from
three main sources: From liquidity rebates on exchanges that oﬀer maker-taker
fee structure, from spread earned when successfully buying on the bid and sell-
ing on the oﬀer, and from short-term price moves favorable to their inventory.

Hendershott, Brogaard and Riordan (2014) [191] ﬁnd that HFT activity tends to
be concentrated in large liquid stocks and postulate that this can be attributed
to a combination of larger proﬁt opportunities emanating from trades happening
more often, and from easier risk management due to larger liquidity that allows
for easier exit of unfavorable positions at a reasonable cost.

(e) Risk Management: In the existing literature on informed trading, it is observed
that liquidity supplying risk-neutral market makers are adversely selected by
informed traders suddenly moving prices against them. For example, a market
maker buy quote tends to be executed when large sellers are pushing the price
down, resulting in even lower prices in the near term. This signiﬁcant potential
asymmetry of information at any point in time emphasizes the need for market
makers to employ robust risk management techniques, particularly in the domain
of inventory risk. For a market maker, risk management is generally accomplished
by ﬁrst adjusting market quotes upward or downward to increase the arrival rate
of sellers or buyers and consequently adjusting the inventory in the desired direc-
tion. If biasing the quotes does not result in a successful inventory adjustment,
the market makers generally employ limit orders to cross the spread.

Ho and Stoll (1981) [196] introduced a market making model in which the market
maker’s objective is to maximize proﬁt while minimizing the probability of ruin by
determining the optimal bid-ask spread to quote. The inventory held evolves through
the arrival of bid and ask orders, where the arrival rate is taken to be a function of bid
and ask prices. Their model also incorporates the relevant notion of the size depen-
dence of spread on the market maker’s time horizon. The longer the remaining time,
the greater potential for adverse move risk for liquidity providers, and vice versa.
This is consistent with observed spreads. In most markets, the spread is wider at the
beginning of the day, narrowing toward the close. An additional reason annotated for

Trading Fundamentals

49

the wider spread right after the beginning of the trading day is due to the existence
of potentially signiﬁcant information asymmetry accumulated overnight. As market
makers are directly exposed to that information asymmetry, they tend to quote wider
spreads while the price discovery process unfolds following the opening of continu-
ous trading, and progressively tighten them as uncertainty about the fair price for the
asset dissipates.

Understanding the dynamics of market makers inventory and risk management,
and their eﬀect on spreads, has direct implications for the practitioners who intend to
deploy algorithmic trading strategies as the spread paid to enter and exit positions is a
non-negligible source of cost that can erode the proﬁtability of low alpha quantitative
strategies.

Hendershott and Seasholes (2007) [195] conﬁrms that market makers’ inventories
are negatively correlated with previous price changes and positively correlated with
subsequent changes. This is consistent with market makers ﬁrst acting as a dampener
of buying or selling pressure by bearing the risk of temporarily holding inventory
in return for earning the spread and thus potential price appreciation from market
reversal. This model easily links liquidity provision and the dynamics of asset prices.
Finally, it has been observed that there is a positive correlation of market makers
inventory with subsequent prices changes, inventories can complement past returns
when predicting future returns. Since inventories are not publicly known, market par-
ticipants use diﬀerent proxies to infer their values throughout the day. Two commonly
used proxies are trade imbalances (the net excess of buy or sell initiated trade volume)
and spreads. Trade imbalance aims at classifying trades, either buy initiated or sell ini-
tiated, by comparing their price with the prevailing quote. Given that market makers
try to minimize their directional risk, they can only accommodate a limited amount
of non-diversiﬁed inventory over a ﬁnite period of time. As such, spread sizes, and
more particularly their sudden variation, have also been used as proxies for detecting
excess inventory forcing liquidity providers to adjust their positions.

Bibliography

[1] F. Abdi and A. Ranaldo. A simple estimation of bid-ask spreads from daily
close, high and low prices. The Review of Financial Studies, 30:4437–4480,
2017.

[2] F. Abergel and A. Jedidi. A mathematical approach to order book modeling.
International Journal of Theoretical and Applied Finance, 16:1–40, 2013.

[3] A.R. Admati and P. Pﬂeiderer. A theory of intraday patterns: Volume and price

variability. The Review of Financial Studies, 1(1):3–40, 1988.

[4] Y. Aït-Sahalia, P.A. Mykland, and L. Zhang. How often to sample a
continuous-time process in the presence of market microstructure noise. The
Review of Financial Studies, 18(2):351–416, 2005.

[5] H. Akaike. A new look at the statistical model identiﬁcation. IEEE Transac-

tions on Automatic Control, AC–19:716–723, 1974.

[6] S.S. Alexander. Price movements in speculative markets: Trends of random

walks. Industrial Management Review, pages 7–26, 1961.

[7] S.S. Alexander. Price movements in speculative markets: Trends of random

walks, no 2. Industrial Management Review, pages 25–46, 1964.

[8] S. Alizadeh, M.W. Brandt, and F.X. Diebold. Range-based estimation of
stochastic volatility models. Journal of Finance, 57:1047–1091, 2002.

[9] R. Almgren. Execution costs. Encyclopedia of Quantitative Finance, pages

1–5, 2008.

[10] R. Almgren and N. Chriss. Optimal execution of portfolio transactions. The

Journal of Risk, 3:5–39, 2000.

[11] R. Almgren, C. Thum, E. Hauptmann, and H. Li. Equity market impact. Risk,

18(7):57–62, 2005.

[12] R.F. Almgren. Optimal execution with nonlinear impact functions and trading

enhanced risk. Applied Mathematical Finance, 10:1–18, 2003.

[13] N. Amenc, F. Goltz, A. Lodh, and L. Martellini. Diversifying the diversi-
ﬁers and tracking the tracking error: Outperforming cap-weighted indices with
limited risk of underperformance. The Journal of Portfolio Management,
38(3):72–88, 2012.

51

52

Bibliography

[14] S. Anatolyev and A. Gospodinov. A trading approach to testing for predictabil-

ity. Journal of Business & Economic Statistics, 23:455–461, 2005.

[15] S. Anatolyev and A. Gospodinov. Modeling ﬁnancial return dynamics via
decomposition. Journal of Business & Economic Statistics, 28:232–245, 2010.

[16] T. Andersen, I. Archakov, G. Cebiroglu, and N. Hautsch. Volatility information
feedback and market microstructure noise: A tale of two regimes. CFS Working
Paper, Northwestern University, 2017.

[17] T.G. Andersen. Return volatility and trading volume: An information ﬂow
interpretation of stochastic volatility. Journal of Finance, 51:116–204, 1996.

[18] T.G. Andersen and T. Bollerslev. Answering the skeptics: Yes, standard volatil-
International Economic Review,

ity models do provide accurate forecasts.
39:885–905, 1998.

[19] T.W. Anderson. An Introduction to Multivariate Statistical Analysis. Second

Edition. Wiley, New York, 1984.

[20] T.W. Anderson and A.M. Walker. On the asymptotic distribution of the auto-
correlations of a sample from a linear stochastic process. Annals of Mathemat-
ical Statistics, 35:1296–1303, 1964.

[21] A. Ang and A. Timmermann. Regime changes and ﬁnancial markets. Annual

Review of Finance and Economics, 4:313–337, 2012.

[22] W. Antweiler and M.Z. Frank.

Is all that talk just noise? The information
content of internet stock message boards. Journal of Finance, 59(3):1259–
1294, 2004.

[23] P. Asquith, R. Oman, and C. Safaya. Short sales and trade classiﬁcation algo-

rithms. Journal of Financial Markets, 13:157–173, 2010.

[24] M. Avellaneda and J.H. Lee. Statistical arbitrage in the US equities market.

Quantitative Finance, 10:761–782, 2010.

[25] W. Bagehot. The only game in town. Financial Analysis Journal, 27:12–14,

1971.

[26] P. Bajgrowicz and O. Scaillet. Technical trading revisited: False discover-
ies, persistence tests, and transaction costs. Journal of Financial Economics,
106(3):473–491, 2012.

[27] M. Baker and J. Wurgler.

Investor sentiment and the cross-section of stock

returns. Journal of Finance, 61(4):1645–1680, 2006.

[28] M. Baker and J. Wurgler. Investor sentiment in the stock market. Journal of

Economic Perspectives, 21(2):129–151, 2007.

Bibliography

53

[29] F.M. Bandi and J.R. Russell. Separating microstructure noise from volatility.

Journal of Financial Economics, 79:655–692, 2006.

[30] N. Barberis, A. Shleifer, and R. Vishny. A model of investor sentiment. Journal

of Financial Economics, 49:307–343, 1998.

[31] O.E. Barndorﬀ-Nielsen and N. Shephard. Econometric analysis of realized
volatility and its use in estimating stochastic volatility models. Journal of the
Royal Statistical Society: Series B (Statistical Methodology), 64(2):253–280,
2002.

[32] L. Barras, O. Scaillet, and R. Wermers. False discoveries in mutual fund per-
formance: Measuring luck in estimated alphas. Journal of Finance, 65(1):179–
216, 2010.

[33] R. Battalio, S.A. Corwin, and R. Jennings. Can brokers have it all? On the
relation between make-take fees and limit order execution quality. Journal of
Finance, 71:2193–2238, 2016.

[34] L. Bauwens, S. Laurent, and J.V.K. Rombouts. Multivariate GARCH models:

A survey. The Journal of Applied Econometrics, 21:79–109, 2006.

[35] M. Bayraktar, I. Mashtaser, N. Meng, and S. Radchenko. Barra vs total market

equity trading model, empirical notes. MSCI Research, 2015.

[36] P. Bertrand and C. Protopopescu. The statistics of the information ratio. Inter-

national Journal of Business, 15:71–86, 2010.

[37] D. Bertsimas and A.W. Lo. Optimal control of executions costs. Journal of

Financial Markets, 1:1–50, 1998.

[38] H. Bessembinder, M. Panayides, and K. Venkataraman. Hidden liquidity: An
analysis of order exposure strategies in electronic stock markets. Journal of
Financial Economics, 94:361–383, 2009.

[39] B. Biais, L. Glosten, and C. Spatt. Market microstructure; a survey of micro-
foundations, empirical results, and policy implications. Journal of Financial
Markets, 8:217–264, 2005.

[40] B. Biais, P. Hillion, and C. Spatt. An empirical analysis of the limit order book
and the order ﬂow in the Paris bourse. Journal of Finance, 50:1655–1689,
1995.

[41] J.P. Bialkowski, S. Darolles, and Gaëlle G. Le Fol. Improving VWAP. strate-
gies: A dynamical volume approach. Journal of Banking and Finance, 32,
2006.

[42] F. Black. Towards a fully automated exchange, Part I. Financial Analysts

Journal, 27:29–34, 1971.

54

Bibliography

[43] F. Black. Capital market equilibrium with restricted borrowing. The Journal

of Business, 45:444–454, 1972.

[44] F. Black and R. Litterman. Global portfolio optimization. Financial Analysts

Journal, 48 No. 5:28–43, 1992.

[45] L. Blume, D. Easley, and M. O’Hara. Market statistics and technical analysis:

The role of volume. Journal of Finance, 49:153–181, 1994.

[46] T. Bollerslev. Generalized autoregressive conditional heteroskedasticity. Jour-

nal of Econometrics, 31:307–327, 1986.

[47] M. Borkovec and H.G. Heidle. Building and evaluating a transaction cost

model: A primer. The Journal of Trading, 5:57–77, 2010.

[48] P. Bossaerts. Common nonstationary components of asset prices. The Journal

of Economic Dynamics and Control, 12(2):347–364, 1988.

[49] J.P. Bouchaud, J.D. Farmer, and F. Lillo. How Markets Digest Supply and
Demand and Slowly Incorporate Information into Prices. Academic Press,
2009.

[50] J.P. Bouchaud, Y. Gefen, M. Potters, and M. Wyart. Fluctuations and response
in ﬁnancial markets: The subtle nature of “random” price changes. Quantita-
tive Finance, 4:176–190, 2004.

[51] J.-P. Bouchaud, M. Mezard, and M. Potters. Statistical properties of stock order
books: Empirical results and models. Quantitative Finance, 2:251–256, 2002.

[52] D. Bowen, M.C. Hutchinson, and N. O’Sullivan. High-frequency equity pairs
trading: Transaction costs, speed of execution and patterns in returns. The
Journal of Trading, Summer, pages 31–38, 2010.

[53] G.E.P. Box, G.M. Jenkins, G.C. Reinsel, and G.M. Ljung. Time Series Analy-

sis: Forecasting and Control, 5th edition. Wiley, New York, 2015.

[54] G.E.P. Box and G.C. Tiao. A canonical analysis of multiple time series.

Biometrika, 64:355–365, 1977.

[55] P. Boyle, L. Garlappi, R. Uppal, and T. Wang. Keynes meets Markowitz:
The trade-oﬀ between familiarity and diversiﬁcation. Management Science,
58:253–272, 2012.

[56] M.W. Brandt and P. Santa-Clara. Dynamic portfolio selection by augmenting

the asset space. Journal of Finance, 61:2187–2217, 2006.

[57] L. Breiman. Bagging predictors. Machine Learning, 24(2):123–140, 1996.

[58] L. Breiman. Prediction games and arcing algorithms. Neural Computation,

11(7):1493–1517, 1999.

Bibliography

55

[59] D.R. Brillinger. Time Series: Data Analysis and Theory. Expanded edition.

Holden-Day, San Francisco, 1981.

[60] W. Brock, J. Lakonishok, and B. LeBaron. Simple technical trading rules and
the stochastic properties of stock returns. Journal of Finance, 47:1731–1764,
1992.

[61] J. Brodie, I. Daubechies, C. De Mol, D. Giannone, and I. Loris. Sparse and
stable Markowitz portfolios. Proceedings of the National Academy of Sciences,
106:12267–12272, 2009.

[62] C. Brownlees, F. Cipollini, and G.M. Gallo. Intra-daily volume modeling and
prediction for algorithmic trading. The Journal of Financial Econometrics,
9:489–518, 2011.

[63] B. Bruder, N. Gaussel, J.-C. Richard, and T. Roncalli. Regularization of port-

folio allocation. White Paper Issue #10, 2013.

[64] E. Busseti and S. Boyd. Volume Weighted Average Price Optimal Execution.

unpublished, Stanford University, 2015.

[65] J.Y. Campbell, S.J. Grossman, and J. Wang. Trading volume and serial cor-
relation in stock returns. The Quarterly Journal of Economics, 108:905–939,
1993.

[66] J.Y. Campbell, A.W. Lo, and A.C. MacKinlay. The Econometrics of Financial

Markets. Princeton University Press, New Jersey, 1996.

[67] C. Cao, O. Hansch, and X. Wang. The information content of an open limit

order book. The Journal of Futures Markets, 29:16–41, 2009.

[68] M.M. Carhart. On persistence in mutual fund performance.

Finance, 52(1):57–82, 1997.

Journal of

[69] M. Centoni and G. Cubadda. Modeling co-movements of economic time

series: A selective survey. Statistica, 71:267–293, 2011.

[70] A.P. Chaboud, B. Chiquoine, E. Hjalmarsson, and C. Vega. Rise of the
machines: Algorithmic trading in the foreign exchange market. Journal of
Finance, 69:2045–2084, 2014.

[71] B. Chakrabarty, P.C. Moulton, and A. Shkilko. Short sales, long sales, and
the Lee-Ready trade classiﬁcation algorithm revisited. Journal of Financial
Markets, 15(4):467–491, 2012.

[72] K. Chan and W.-M. Fong. Trade size, order imbalance and the volatility-
volume relation. Journal of Financial Economics, 57:247–273, 2000.

[73] L. Chan and J. Lakonishok. Institutional equity trading costs, NYSE versus

Nasdaq. Journal of Finance, 52:713–735, 1997.

56

Bibliography

[74] L.K.C. Chan and J. Lakonishok. The behavior of stock prices around institu-

tional trades. Journal of Finance, 50:1147–1174, 1995.

[75] L.K.C. Chan and J. Lakonishok. Institutional equity trading costs: NYSE ver-

sus Nasdaq. Journal of Finance, 52(2):176–190, 1997.

[76] N.F. Chen, R. Roll, and S.A. Ross. Economic forces and the stock market. The

Journal of Business, 59:383–403, 1986.

[77] S. Chib. Estimation and comparison of multiple change-point models. Journal

of Econometrics, 86:221–241, 1998.

[78] C. Chiyachantana, P.K. Jain, C. Jiang, and R.A. Wood. International evidence
on institutional trading behavior and price impact. Journal of Finance, 59:869–
898, 2004.

[79] T. Chordia, R. Roll, and A. Subrahmanyam. Commonality in liquidity. Journal

of Financial Economics, 56:3–28, 2000.

[80] T. Chordia, R. Roll, and A. Subrahmanyam. Order imbalance, liquidity, and

market returns. Journal of Financial Economics, 65:111–130, 2002.

[81] P.K. Clark. A subordinated stochastic process model with ﬁnite variance for

speculative prices. Econometrica, 41:135–155, 1973.

[82] J. Conrad and G. Kaul. An anatomy of trading strategies. The Review of

Financial Studies, 11:489–519, 1998.

[83] J. Conrad and S. Wahal. The term structure of liquidity provision. Journal of

Financial Economics, 136:239–259, 2020.

[84] J.S. Conrad, A. Hameed, and C. Niden. Volume and autocovariances in short-
horizon individual security returns. Journal of Finance, 49:1305–1329, 1994.

[85] A. Constantinos, J.A. Donkas, and A. Subrahmanyam. Cognitive dissonance,
sentiment and momentum. Journal of Financial and Quantitative Analysis,
46:245–275, 2013.

[86] R. Cont and A. Kukanov. Optimal order placement in limit order markets.

Quantitative Finance, 17:21–39, 2017.

[87] R. Cont, A. Kukanov, and S. Stoikov. The price impact of order book events.

The Journal of Financial Econometrics, 12:47–88, 2014.

[88] R. Cont, S. Stoikov, and R. Talreja. A stochastic model for order book dynam-

ics. Operations Research, 58:549–563, 2010.

[89] M. Cooper. Filter values based on price and volume in individual security

overreaction. The Review of Financial Studies, 12:901–935, 1999.

Bibliography

57

[90] S.A. Corwin and P. Schultz. A simple way to estimate bid-ask spreads from

daily high and low prices. Journal of Finance, 67:719–759, 2012.

[91] A. Cowles. Can stock market forecasters forecast. Econometrica, 1:309–324,

1933.

[92] A. Cowles. Stock market forecasting. Econometrica, 12:206–214, 1944.

[93] D.R. Cox and P.A.W. Lewis. The Statistical Analysis of Series of Events. Chap-

man and Hall, London, 1966.

[94] G. Creamer and Y. Freund. Automated trading with boosting and expert

weighting. Quantitative Finance, 4:401–420, 2010.

[95] M. Cremers and D. Weinbaum. Deviations from put-call parity and stock return
predictability. Journal of Financial and Quantitative Analysis, 45:335–367,
2010.

[96] D.M. Cutler, J.M. Poterba, and L.H. Summers. What Moves Stock Prices?,
volume 15. National Bureau of Economic Research Cambridge, Mass., USA,
1989.

[97] S. Da, J. Engelberg, and P. Gao. In search of attention. Journal of Finance,

66(5):1461–1499, 2011.

[98] Z. Da, J. Engelberg, and P. Gao. The sum of all fears investor sentiment and

asset prices. The Review of Financial Studies, 28(1):1–32, 2015.

[99] R. Dahlhaus and S. Subba Rao. Statistical inference for time-varying ARCH

processes. The Annals of Statistics, 34:1075–1114, 2006.

[100] D.J. Daley and D. Vere-Jones. An Introduction to the Theory of Point Pro-
cesses, Volume I: Elementary Theory and Methods. Springer, New York, 2003.

[101] H.E. Daniels. Autocorrelation between ﬁrst diﬀerences of mid-ranges. Econo-

metrica, pages 215–219, 1966.

[102] S.R. Das and M.Y. Chen. Yahoo! for Amazon: Sentiment extraction from small

talk on the web. Management Science, 53:1375–1388, 2007.

[103] B.J. DeLong, A. Shleifer, L.H. Summers, and R.J. Waldmann. Noise trader risk

in ﬁnancial markets. The Journal of Political Economy, 98:703–738, 1990.

[104] V. DeMiguel, L. Garlappi, and R. Uppal. Optimal versus naïve diversiﬁcation:
How ineﬃcient is the 1∕𝑁 portfolio strategy? The Review of Financial Studies,
22:1915–1953, 2009.

[105] A.P. Dempster, N.-M. Laird, and D.B. Rubin. Maximum likelihood from
incomplete data via the EM algorithm. Journal of the Royal Statistical Society,
Series B, 39:1–38, 1977.

58

Bibliography

[106] B. Do and R. Faﬀ. Does simple pairs trading still work? The Financial Analysts

Journal, 66(4):83–95, 2010.

[107] I. Domowitz, J. Glen, and A. Madhavan. Liquidity, volatility and equity trading
costs across countries and over time. International Finance, 4:221–255, 2001.

[108] I. Domowitz and H. Yegerman. The cost of algorithmic trading: A ﬁrst look
at comparative performance. Algorithmic Trading: Precision, Control, Execu-
tion, 2005.

[109] A. Dufour and R. Engle. Time and the price impact of a trade. Journal of

Finance, 55(2):467–498, 2000.

[110] D. Easley, M.L. De Prado, and M. O’Hara. Flow toxicity and liquidity in a
high frequency world. The Review of Financial Studies, 25:1457–1493, 2012.

[111] D. Easley, M.L. De Prado, and M. O’Hara. Optimal execution horizon. Math-

ematical Finance, 25:640–672, 2015.

[112] D. Easley and M. O’ Hara. Price, trade size, and information in security mar-

kets. Journal of Financial Economics, 19:69–90, 1987.

[113] D. Easley and M. O’ Hara. Time and the process of security price adjustment.

Journal of Finance, 19:69–90, 1992.

[114] C. Eckart and G. Young. The approximation of one matrix by another of lower

rank. Psychometrika, 1:211–218, 1936.

[115] B. Efron, T. Hastie, I. Johnstone, and R. Tibshirani. Least angle regression.

Annals of Statistics, 32:407–499, 2004.

[116] J. Engelberg, P. Gao, and R. Jagannathan. An anatomy of pairs trading: the role
of idiosyncratic news, common information and liquidity. In Third Singapore
International Conference on Finance, 2009.

[117] R. Engle and R. Ferstenberg. Execution risk. The Journal of Portfolio Man-

agement, 33(4):34–44, 2007.

[118] R. Engle and F.K. Kroner. Multivariate simultaneous generalized ARCH.

Econometric Theory, 11:122–150, 1995.

[119] R. Engle and A. Lunde. Trades and quotes: A bivariate point process. The

Journal of Financial Economics, 1:159–188, 2003.

[120] R.F. Engle. Autoregressive conditional heteroscedasticity with estimates of the

variance of United Kingdom inﬂations. Econometrica, 50:987–1007, 1982.

[121] R.F. Engle, R. Ferstenberg, and J.R. Russell. Measuring and modeling execu-
tion cost and risk. The Journal of Portfolio Management, 38(2):14–28, 2012.

Bibliography

59

[122] R.F. Engle and C.W.J. Granger. Co-integration and error correction: Repre-
sentation, estimation, and testing. Econometrica, 55:251–276, 1987.

[123] R.F. Engle and S. Kozicki. Testing for common features. Journal of Business

& Economic Statistics, 11(4):369–380, 1993.

[124] R.F. Engle and D. Kraft. Multiperiod Forecast Error Variances of Inﬂation
Estimated from ARCH Models, in: A. Zellner, ed: Applied Time Series Analysis
of Economic Data. Bureau of the Census, Washington, DC, 1983.

[125] R.F. Engle and J.R. Russell. Autoregressive conditioned duration: A new
model for irregularly spaced transaction data. Econometrica, 66:1127–1162,
1998.

[126] R.F. Engle and R. Susmel. Common volatility in international equity markets.

Journal of Business & Economic Statistics, 11(2):167–176, 1993.

[127] R.F. Engle and S. Kozicki. Testing for common features. Journal of Business

& Economic Statistics, 11:369–380, 1993.

[128] R.F. Engle, V.K. Ng, and M. Rothschild. Asset pricing with a factor ARCH
covariance structure: Empirical estimates for treasury bills. Journal of Econo-
metrics, 45:213–218, 1990.

[129] T.W. Epps and M.L. Epps. The stochastic dependence of security price changes
and transaction volumes: Implications for the mixture-of-distribution hypoth-
esis. Econometrica, 44:305–321, 1976.

[130] T.W. Epps. Co-movements in stock prices in the very short run. Journal of the

American Statistical Association, 74:291–298, 1979.

[131] F.J. Fabozzi, S.M. Focardi, and P.N. Kolm. Quantitative Equity Investing:

Techniques and Strategies. Wiley and Sons, 2006.

[132] E.F. Fama and M.E. Blume. Filter rules and stock-market trading. The Journal

of Business, 39:226–241, 1966.

[133] E.F. Fama and K.R. French. A ﬁve-factor after pricing model. Journal of

Financial Economics, 116:1–22, 2015.

[134] E.F. Fama and K.R. French. International tests of a ﬁve-factor asset pricing

model. Fama-Miller Working Paper, 2015.

[135] J. Fan, Y. Fan, and J. Lv. High dimensional covariance matrix estimation using

a factor model. Journal of Econometrics, 147:187–197, 2008.

[136] J. Fan, F. Han, H. Liu, and B. Vickers. Robust inference of risks of large

portfolios. Journal of Econometrics, 194:298–308, 2016.

60

Bibliography

[137] J. Fan, J. Zhang, H. Liu, and K. Yu. Vast portfolio selection with cross-
the American Statistical Association,

Journal of

exposure constraints.
107:592–606, 2012.

[138] A. Farago and E. Hjalmarsson. Stock price co-movement and the foundations
of pairs trading. Journal of Financial and Quantitative Analysis, 54:629–665,
2019.

[139] J.D. Farmer, A. Gerig, F. Lillo, and H. Waelbroeck. How eﬃciency shapes

market impact. Quantitative Finance, 11:1743–1758, 2013.

[140] J.D. Farmer, L. Gillemot, F. Lillo, S. Mike, and A. Sen. What really causes

large price changes? Quantitative Finance, 4:383–397, 2004.

[141] W.E. Ferson and A.F. Siegel. The use of conditioning information in portfolios.

Journal of Finance, 56:967–982, 2001.

[142] T. Foucault and A.J. Menkveld. Competition for order ﬂow and smart order

routing systems. Journal of Finance, pages 119–157, 2008.

[143] A. Frazzini, R. Israel, and T.J. Moskowitz. Trading costs (unpublished), 2018.

[144] A. Frazzini and L.H. Pedersen. Betting against beta. Journal of Financial

Economics, 111:1–25, 2014.

[145] Y. Freund and R.E. Schapire. A decision-theoretic generalization of on-line
learning and an application to boosting. In European Conference on Compu-
tational Learning Theory, pages 23–37. Springer, 1995.

[146] Y. Freund and R.E. Schapire. Experiments with a new boosting algorithm. In
International Conference on Machine Learning, volume 96, pages 148–156.
Morgan Kaufmann Publishers Inc., San Francisco, CA, 1996.

[147] Y. Freund and R.E. Schapire. A decision-theoretic generalization of on-line
learning and an application to boosting. In Journal of Computer and System
Sciences, volume 55, pages 119–139, 1997.

[148] J. Friedman. Greedy function approximation: A gradient boosting machine.

Annals of Statistics, pages 1189–1232, 2001.

[149] J. Friedman, T. Hastie, and R. Tibshirani. Additive logistic regression: A sta-
tistical view of boosting (with discussion and a rejoinder by the authors). The
Annals of Statistics, 28(2):337–407, 2000.

[150] A. Frino, E. Jarnecic, and A. Lepone. The determinants of price impact of

block trades: Further evidence. Abacus, 43(2):94–106, 2007.

[151] K.A. Froot and E.M. Dabora. How are stock prices aﬀected by the location of

trade? Journal of Financial Economics, 53(2):189–216, 1999.

Bibliography

61

[152] P. Fryzlewicz, T. Sapatinas, and S. Subba Rao. Normalized least-squared esti-
mation in time-varying ARCH models. The Annals of Statistics, 36:742–786,
2008.

[153] W.A. Fuller.

Introduction to Statistical Time Series, Second Edition. John

Wiley, New York, 1996.

[154] L. Gagnon and G.A. Karolyi. Multi-market trading and arbitrage. Journal of

Financial Economics, 97:53–80, 2010.

[155] A.R. Gallant, P.E. Rossi, and G. Tauchen. Nonlinear dynamic structures.

Econometrica, 61:871–908, 1993.

[156] L. Gao, Y. Han, S.Z. Li, and G. Zhou. Market intraday momentum. Journal

of Financial Economics, 129:394–414, 2018.

[157] N. Gârleanu and L.H. Pedersen. Dynamic trading with predictable returns and

transaction costs. The Journal of Finance, 68:2309–2340, 2013.

[158] M.B. Garman and M.J. Klass. On the estimation of security price volatilities

from historical data. The Journal of Business, 53:67–78, 1980.

[159] P.H. Garthwaite. An interpretation of partial least squares. Journal of the

American Statistical Association, 89:122–127, 1994.

[160] E. Gatev, W.N. Goetzmann, and R.G. Rouwenhorst. Pairs trading: Performance
of a relative value arbitrage rule. The Review of Financial Studies, 19:797–827,
2006.

[161] J. Gatheral. No-dynamic-arbitrage and market impact. Quantitative Finance,

10:769–759, 2010.

[162] R. Gencay. The predictability of security returns with simple technical trading

rules. Journal of Empirical Finance, 5:347–359, 1998.

[163] S. Gervais, R. Kaniel, and D.H. Mingelgrin. The high-volume return premium.

Journal of Finance, 56(3):877–919, 2001.

[164] S. Gervais, R. Kaniel, and D.H. Mingelgrin. The high-volume return premium.

Journal of Finance, 56:877–919, 2001.

[165] E. Ghysels, C. Gourieroux, and J. Jasiak. Stochastic volatility duration models.

Journal of Econometrics, 119:413–433, 2004.

[166] M.R. Gibbons, S.A. Ross, and J. Shanken. A test of the eﬃciency of a given

portfolio. Econometrica, 57:1121–1152, 1989.

[167] L.R. Glosten and P.R. Milgrom. Bid, ask and transaction prices in a special-
ist market with heterogeneously informed traders. Journal of Financial Eco-
nomics, 14:71–100, 1985.

62

Bibliography

[168] I. Goodfellow, Y. Bengio, and A. Courville. Deep Learning. M.I.T. Press,

Boston, 2016.

[169] R.C. Grinold and R.N. Kahn. Active Portfolio Management, Second Edition.

McGraw-Hill, 2000.

[170] A. Gross-Klussmann and N. Hautsch. When machines read the news: Using
automated text analytics to quantify high frequency news-implied market reac-
tions. Journal of Empirical Finance, 18:321–340, 2011.

[171] A.D. Hall and N. Hautsch. Order aggressiveness and order book dynamics.

Empirical Economics, 30:973–1005, 2006.

[172] J.D. Hamilton. A new approach to the economic analysis of nonstationary time

series and the business cycle. Econometrica, 57:357–384, 1989.

[173] J.D. Hamilton. Analysis of time series subject to changes in regime. Journal

of Econometrics, 45:39–70, 1990.

[174] J.D. Hamiton. Macroeconomic regimes and regime shifts. In J.B. Taylor and
H. Uhlig, editors, Handbook of Macroeconomics, chapter 3, pages 153–201.
Elsevier, 2016.

[175] Y. Han, K. Yang, and G. Zhou. A new anomaly: The cross-sectional prof-
itability of technical analysis. Journal of Financial and Quantitative Analysis,
48:1433–1461, 2013.

[176] P.R. Hansen. A test for superior predictive ability. Journal of Business &

Economic Statistics, 23:365–380, 2005.

[177] M. O’ Hara and M. Ye.

Is market fragmentation harming market quality?

Journal of Financial Economics, pages 454–474, 2011.

[178] L. Harris. Trading and Exchanges: Market Microstructure for Practitioners.

Oxford University Press, New York, 2003.

[179] A.C. Harvey. Forecasting, Structural Time Series Models and the Kalman

Filter. Cambridge University Press, Cambridge, 1989.

[180] J. Hasbrouck. Measuring the information content of stock trades. Journal of

Financial Economics, 46:179–207, 1991.

[181] J. Hasbrouck. One security, many markets: Determining the contributions to

price discovery. Journal of Finance, 50(4):1175–1199, 1995.

[182] J. Hasbrouck and G. Saar. Technology and liquidity provision: The blurring

of traditional deﬁnitions. Journal of Financial Markets, 12:143–172, 2009.

[183] J. Hasbrouck and D.J. Seppi. Common factors in prices, order ﬂows, and liq-

uidity. Journal of Financial Economics, 59:383–411, 2001.

Bibliography

63

[184] T. Hastie, R. Tibshirani, and J. Friedman. The Elements of Statistical Learning;
Data Mining, Inference and Prediction, Second Edition. Springer-Verlag, New
York, 2009.

[185] N. Hautsch and R. Huang. The market impact of a limit order. The Journal of

Economic Dynamics and Control, 36:501–522, 2012.

[186] A.G. Hawkes. Spectra of some self-exciting and mutually exciting point pro-

cesses. Biometrika, 58:83–90, 1971.

[187] A.G. Hawkes. Hawkes processes and their applications to ﬁnance; a review.

Quantitative Finance, 18:193–198, 2018.

[188] X. He and R. Velu. Volume and volatility in a common-factor mixture of
distributions model. Journal of Financial and Quantitative Analysis, 49:33–
49, 2014.

[189] I.S. Helland. On the structure of partial least squares regression. Communica-
tions in Statistics, Simulation and Computation, B17:581–607, 1988.

[190] I.S. Helland. Partial least squares regression and statistical models. Scandina-

vian Journal of Statistics, 17:97–114, 1990.

[191] T. Hendershott, J. Brogaard, and R. Riordon. High frequency trading and price

discovery. The Review of Financial Studies, 27:2267–2306, 2014.

[192] T. Hendershott, C.M. Jones, and A.J. Menkveld. Does algorithmic trading

improve liquidity? Journal of Finance, 66(1):1–33, 2011.

[193] T. Hendershott, C.M. Jones, and A.J. Menkveld. Does algorithmic trading

improve liquidity? Journal of Finance, pages 1–3, 2011.

[194] T. Hendershott and R. Riordan.

Algorithmic trading and information.

http://faculty.haas.berkeley.edu/hender/ATInformation.pdf, 2011.

[195] T. Hendershott and M. Seasholes. Market maker inventories and stock prices.

American Economic Review, 97:210–214, 2007.

[196] T. Ho and H.R. Stoll. Optimal dealer pricing under transactions and return

uncertainty. Journal of Financial Economics, 9:47–73, 1981.

[197] T. Ho, R. Schwartz, and D. Whitcomb. The trading decision and market clear-
ing under transaction price uncertainty. The Journal of Finance, 40:21–42,
1985.

[198] S. Hogan, R. Jarrow, M. Teo, and M. Warachka. Testing market eﬃciency
using statistical arbitrage with application to momentum and value strategies.
Journal of Financial Economics, 73:525–565, 2004.

[199] R.W. Holthausen, R.W. Leftwich, and D. Mayers. Large-block transactions, the
speed of response, and temporary and permanent stock-price eﬀect. Journal
of Financial Economics, 26:71–95, 1990.

64

Bibliography

[200] H. Hotelling. Analysis of a complex of statistical variables into principal com-
ponents. Journal of Educational Psychology, 4:417–441, 498–520, 1933.

[201] H. Hotelling. The most predictable criterion. Journal of Educational Psychol-

ogy, 26:139–142, 1935.

[202] H. Hotelling. Relations between two sets of variables. Biometrika, 28:321–

322, 1936.

[203] P.H. Hsu, Y.C. Hsu, and C.M. Kuan. Testing the Predictive Ability of Technical
Analysis Using a New Stepwise Test Without Data Snooping Bias. Journal of
Empirical Finance, 17:471–484, 2010.

[204] Y.-P. Hu and R.S. Tsay. Principal volatility component analysis. Journal of

Business & Economic Statistics, 32:153–164, 2014.

[205] R. Huang and H. Stoll. Dealer versus auction markets: A paired comparison of
execution costs on NASDAQ and the NYSE. Journal of Financial Economics,
41:313–357, 1996.

[206] W. Huang, C-A. Lehalle, and M. Rosenbaum. Simulating and analyzing order
book data: The queue-reactive model. Journal of the American Statistical
Association, 110:107–122, 2015.

[207] D. Huang, F. Jiang, J. Tu, and G. Zhou. Investor sentiment aligned: A powerful
predictor of stock returns. The Review of Financial Studies, 28:791–837, 2015.

[208] G. Huberman. Familiarity breeds investment. The Review of Financial Studies,

14:659–680, 2001.

[209] G. Huberman and W. Stanzl. Price manipulation and quasi-arbitrage. Econo-

metrica, 72:1247–1275, 2004.

[210] S. Hvidkjaer. A trade-based analysis of momentum. The Review of Financial

Studies, 119:457–491, 2006.

[211] R. Israel, T. Moskowitz, A. Ross, and L. Serban. Implementing momentum:

What have we learned. NBER Working Paper, 2017.

[212] R. Jagannathan and T. Ma. Risk reduction in large portfolios: Why imposing
the wrong constraints helps. Journal of Finance, 58:1651–1683, 2003.

[213] C.M. Jarque and A.K. Bera. Eﬃcient tests for normality, homoscedasticity
and serial independence of regression residuals. Economic Letters, 6:255–259,
1980.

[214] N. Jegadeesh. Discussion of LMW (2000). Journal of Finance, pages 1765–

1770, 2000.

[215] N. Jegadeesh and S. Titman. Returns to buying winners and selling losers:
Implications for stock market eﬃciency. Journal of Finance, 48:65–91, 1993.

Bibliography

65

[216] N. Jegadeesh and S. Titman. Proﬁtability of momentum strategies: An evalu-
ation of alternative explanations. Journal of Finance, 56:699–720, 2001.

[217] N. Jegadeesh and S. Titman. Cross-sectional time series determinants of
momentum returns. The Review of Financial Studies, 15:143–157, 2002.

[218] F. Jiang, J. Lee, X. Martin, and G. Zhou. Manager sentiment and stock returns.

Journal of Financial Economics, 132:126–149, 2019.

[219] W. Jiang, L. Shu, and D.W. Apley. Adaptive CUSUM procedures with EWMA-

based shift estimators. IIE Transactions, 40:992–1003, 2008.

[220] J.D. Jobson and B. Korkie. Estimation for markowitz eﬃcient portfolios. The

Journal of American Statistical Association, 75:544–554, 1980.

[221] S. Johansen. Statistical analysis of co-integration vectors. The Journal of

economic dynamics and control, 12(2):231–254, 1988.

[222] S. Johansen. Estimation and hypothesis testing of co-integration vectors in
Gaussian vector autoregressive models. Econometrica: Journal of the Econo-
metric Society, pages 1551–1580, 1991.

[223] C. Jones, G. Kaul, and M. Lipson. Information, trading and volatility. Journal

of Financial Economics, 36:127–154, 1994.

[224] L.P. Kaelbling, M.L. Littman, and A.W. Moore. Reinforcement learning: A

survey. Journal of Artiﬁcial Intelligence, 4:237–285, 1996.

[225] R.N. Kahn and M. Lemmon. Smart beta: The owner’s manual. The Journal of

Portfolio Management, 41(2):76–83, 2015.

[226] H. Kawakatsu. Direct multiperiod forecasting for algorithmic trading. Journal

of Forecasting, 37(1):83–101, 2018.

[227] D.B. Keim and A. Madhavan. The upstairs market for large-block transactions:
Analysis and measurement of price eﬀects. The Review of Financial Studies,
9:1–36, 1996.

[228] D.B Keim and A. Madhavan. Transaction costs and investment style: An inter-
exchange analysis of institutional equity trades. Journal of Financial Eco-
nomics, 46:265–292, 1997.

[229] J.L. Kelly. A new interpretation of information rate. Bell System Technical

Journal, 35:917–926, 1956.

[230] J.M. Keynes. The general theory of employment. The Quarterly Journal of

Economics, pages 209–223, 1937.

[231] P.D. Koch and T.W. Koch. Evolution in dynamic linkages across daily national
stock indexes. Journal of International Money and Finance, 10.2:231–251,
1991.

66

Bibliography

[232] A. Kourtis. On the distribution and estimation of trading costs. Journal of

Empirical Finance, 29:230–245, 2014.

[233] P. Kratz and T. Schöneborn. Optimal liquidity in dark pools. Quantitative

Finance, 14:1519–1539, 2014.

[234] A. Kyle. Continuous time auctions and insider trading. Econometrics,

53:1315–1336, 1985.

[235] T.L. Lai and H. Xing. Statistical Models and Methods for Financial Markets.

Springer, 2008.

[236] T.L. Lai and H. Xing. Stochastic change-point ARX GARCH models and their
applications to econometric times series. Statistica Sinica, 23:1573–1594,
2013.

[237] T.L. Lai, H. Xing, and Z. Chen. Mean-variance portfolio optimization when
means and covariances are unknown. The Annals of Applied Statistics, 5:798–
823, 2011.

[238] J. Lakonishok, A. Shleifer, and R.W. Vishny. Contrarian investment, extrapo-

lation, and risk. Journal of Finance, 49(5):1541–1578, 1994.

[239] O. Ledoit and M. Wolf.

Improved estimation of the covariance matrix of
stock returns with an application to portfolio selection. Journal of Empirical
Finance, 10:603–621, 2003.

[240] C.M. Lee and B. Swaminathan. Price momentum and trading volume. Journal

of Finance, LV:2017–2069, 2000.

[241] C.M. Lee and M. Ready. Inferring trade direction from intraday data. Journal

of Finance, 46:733–746, 1991.

[242] J. Lewellen. Momentum and autocorrelation in stock returns. The Review of

Financial Studies, 15:65–91, 2002.

[243] J.K.-S. Liew, S. Guo, and T. Zhang. Tweet sentiments and crowd-sourced
earnings estimates as valuable sources of information around earnings releases.
The Journal of Alternative Investments, Winter Issue:1–20, 2017.

[244] F. Lillo, J.D. Farmer, and R. Mantegna. Master curve for price impact function.

Nature, 421:129–130, 2003.

[245] J. Lintner. The valuation of risky assets and the selection of risky investment
in stock portfolios and capital budgets. Review of Economics and Statistics,
47:13–37, 1965.

[246] G. Llorente, R. Michaely, G. Saar, and J. Wang. Dynamic volume-return rela-
tion of individual stocks. The Review of Financial Studies, 15:1005–1047,
2002.

Bibliography

67

[247] A.W. Lo, H. Mamaysky, and J. Wang. Foundation of technical analysis:
Computational algorithms, statistical inference and empirical implementation.
Journal of Finance, LV(4):1705–1765, 2000.

[248] A.W. Lo. The statistics of Sharpe ratios. Financial Analysis Journal, pages

36–52, 2002.

[249] A.W. Lo and A.C. MacKinlay. When are contrarian proﬁts due to stock market

overreaction? The Review of Financial Studies, 3:175–205, 1990.

[250] A.W. Lo, A.C. MacKinlay, and J. Zhang. Econometric models of limit-order

executions. Journal of Financial Economics, 65:31–71, 2002.

[251] T.F. Loeb. Trading cost: The critical link between investment information and

results. Financial Analyst Journal, 39(3):39–44, 1983.

[252] M. Lopez de Prado. Advances in Financial Machine Learning. John Wiley &

Sons, New Jersey, 2018.

[253] J. Lorenz and R. Almgren. Mean-variance optimal adaptive execution. Applied

Mathematical Finance, 18(4):395–422, 2011.

[254] A. Madhavan. Market microstructure. Journal of Financial Markets, 3:205–

258, 2000.

[255] A. Madhavan. VWAP strategies. Trading, 1:32–39, 2002.

[256] A. Madhavan, M. Richardson, and M. Roomans. Why do security prices
change? A transaction-level analysis of NYSE stocks. The Review of Financial
Studies, 10(4):1035–1064, 1997.

[257] C. Maglaras, C.C. Moallemi, and H. Zheng. Queueing dynamics and state
space collapse in fragmented limit order book markets. Operations Research
(to appear), 2019.

[258] B.G. Malkiel. A Random Walk Down Wall Street: The Time-Tested Strategy

for Successful Investing, 10th Edition. Norton, 2012.

[259] B. Mandelbrot. The variation of certain speculative prices. The Journal of

Business, 36:294–319, 1963.

[260] H. Markowitz. Portfolio Selection: Eﬃcient Diversiﬁcation of Investments.

Wiley: New York, 1959.

[261] M. Martens and D. van Dijk. Measuring volatility with the realized range.

Journal of Econometrics, 138:181–207, 2007.

[262] R. McCulloch and R. Tsay. Nonlinearity in high-frequency ﬁnancial data
and hierarchical models. Studies in Nonlinear Dynamics and Econometrics,
5:1067–1077, 2001.

68

Bibliography

[263] H. Mendelson. Market behavior in a clearing house. Econometrica: Journal of

the Econometric Society, 1505–1524, 1982.

[264] L. Menkhoﬀ, L. Sarno, M. Schmeling, and A. Schrimpf. Currency momentum

strategies. Journal of Financial Economics, 106:660–684, 2012.

[265] L. Menkhoﬀ and M.P. Taylor. The obstinate passion of foreign exchange pro-
fessionals: Technical analysis. The Journal of Economic Literature, XLV:936–
972, 2007.

[266] A.J. Menkveld, S.J. Koopman, and A. Lucas. Modeling around-the-clock price
discovery for cross-listed stocks using state space methods. Journal of Busi-
ness & Economic Statistics, 25:213–225, 2007.

[267] R.C. Merton. On estimating the expected return on the market. Journal of

Financial Economics, 8:323–336, 1980.

[268] R.T. Merton. An intertemporal capital asset pricing model. Econometrica,

41:867–887, 1973.

[269] R.O. Michaud. Eﬃcient Asset Management. Harvard Business School Press,

Boston, 1989.

[270] G. Mitra and L. Mitra. The Handbook of News Analytics in Finance (Ed).

Wiley Finance, 2011.

[271] T. Moorman. An empirical investigation of methods to reduce transaction

costs. Journal of Empirical Finance, 29:230–245, 2014.

[272] E. Moro, J. Vicente, L.G. Moyano, A. Gerig, J.D. Farmer, G. Vaglica, F. Lillo,
and R.N. Mantegna. Market impact and trading proﬁle of hidden orders in
stock markets. Physical Review E, 80(6):066102, 2009.

[273] T.J. Moskowitz, Y.H. Ooi, and L.H. Pedersen. Time Series Momentum. Jour-

nal of Financial Economics, 104:228–250, 2012.

[274] A.A. Obizhaeva and J. Wang. Optimal trading strategy and supply/demand

dynamics. Journal of Financial Markets, 16:1–32, 2013.

[275] M. O’Hara. Presidential address: Liquidity and price discovery. Journal of

Finance, 58:1335–1354, 2003.

[276] M. O’Hara. High frequency market microstructure. Journal of Financial Eco-

nomics, 116:257–270, 2015.

[277] J. Okunev and D. White. Do momentum - based strategies still work in foreign
currency markets? Journal of Financial and Quantitative Analysis, 38:425–
447, 2003.

Bibliography

69

[278] J.K. Ord, A.B. Koehler, and R.D. Snyder. Estimation and prediction for a class
of dynamic nonlinear statistical models. Journal of the American Statistical
Association, 92(440):1621–1629, 1997.

[279] D.A. Pachamanova and F.J. Fabozzi. Recent trends in equity portfolio con-
struction analytics. The Journal of Portfolio Management, 40:137–151, 2014.

[280] A. Pardo and R. Pascual. On the hidden side of liquidity. The European

Journal of Finance, 18:949–967, 2012.

[281] C. Parlour and D. Seppi. Liquidity-based competition for order ﬂow. The

Review of Financial Studies, 16:301–343, 2003.

[282] C.A. Parlour and D.J. Seppi. Limit Order Markets – A Survey, In Handbook
of Financial Intermediation and Banking, Edited by A. Thakor and A. Boot.
Elsevier, Amsterdam, 2008.

[283] L. Pastor and R.F. Stambough. The equity premium and structural breaks.

Journal of Finance, 56:1207–1239, 2001.

[284] A.J. Patton and A. Timmermann. Monotonicity in asset returns: New tests with
applications to the term structure, the CAPM, and the portfolio sorts. Journal
of Financial Economics, 98:605–625, 2010.

[285] R.L. Peterson. Trading on Sentiment: The Power of Minds over Markets. Wiley

Finance, 2016.

[286] M.J. Ready. Proﬁts from technical trading rules. Financial Management,

Autumn:43–61, 2002.

[287] G.C. Reinsel and S.K. Ahn. Vector autoregressive models with unit roots and
reduced rank structure: estimation, likelihood ratio test, and forecasting. The
Journal of Time Series Analysis, 13(4):353–375, 1992.

[288] G.C. Reinsel. Element of Multivariate Time Series Analysis, Second Edition.

Springer-Verlag, New York, 2002.

[289] G.C. Reinsel and R. Velu. Multivariate Reduced-Rank Regression, Theory and

Application. Springer-Verlag, New York, 1998.

[290] L.C.G. Rogers and S.E. Satchell. Estimating variance from high, low and clos-

ing prices. Annals of Applied Probability, 1:504–512, 1991.

[291] R. Roll. A simple model of the implicit bid-ask spread in an eﬃcient market.

Journal of Finance, 39:1127–1139, 1984.

[292] J.P. Romano and M. Wolf. Stepwise multiple testing as formalized data snoop-

ing. Econometrica, 73:1237–1282, 2005.

[293] S.A. Ross. The arbitrage theory of capital asset pricing. The Journal of Eco-

nomic Theory, 13:341–360, 1976.

70

Bibliography

[294] I. Rosu. A dynamic model of the limit order book. The Review of Financial

Studies, 22:4601–4641, 2009.

[295] T.H. Rydberg and N. Shephard. Dynamic trade-by-trade price movement:
Decomposition and models. Journal of Financial Econometrics, 1:2–25, 2003.

[296] S. Satchell and A. Snowcraft. A demystiﬁcation of the Black-Litterman model:
Managing quantitative and traditional portfolio construction. Journal of Asset
Management, 1:138–150, 2000.

[297] V. Satish, A. Saxena, and M. Palmer. Predicting intraday trading volume and

volume percentages. The Journal of Trading, 9:15–25, 2014.

[298] M. Schneider and F. Lillo. Cross-impact and no-dynamics arbitrage. Quanti-

tative Finance, 19:137–154, 2019.

[299] G. Schwarz. Estimating the dimension of a model. Annals of Statistics, 6:401–

404, 1978.

[300] J.T. Scruggs. Noise trader risk: Evidence from the siamese twins. Journal of

Financial Markets, 10:76–105, 2007.

[301] W.F. Sharpe. Capital asset prices: A theory of market equilibrium under con-

ditions of risk. Journal of Finance, 19:425–442, 1964.

[302] R.H. Shumway and D.S. Stoﬀer. Arima models. In Time Series Analysis and

Its Applications, pages 83–171. Springer, 2011.

[303] D. Smith, N. Wang, Y. Wang, and E.J. Zychowicz. Sentiment and the eﬀective-
ness of Technical Analysis: Evidence from the Hedge Fund Industry. Journal
of Financial and Quantitative Analysis, 51:1991–2013, 2016.

[304] E. Smith, D.J. Farmer, L. Gillemot, and S. Krishnamurthy. Statistical theory

of the continuous double auction. Quantitative Finance, 3:481–514, 2003.

[305] R. Stambaugh, J. Yu, and Y. Yuan. The short of it: Investor sentiment and

anomalies. Journal of Financial Economics, 104:288–302, 2012.

[306] M. Statman, S. Thorley, and K. Vorkink. Investor overconﬁdence and trading

volume. The Review of Financial Studies, 19(4):1531–1565, 2006.

[307] S. Stoikov. The micro-price: A high-frequency estimator of future prices.

Quantitative Finance, 18:1959–1966, 2018.

[308] R. Sullivan, A. Timmermann, and H. White. Data snooping, technical trading
rule performance, and the bootstrap. Journal of Finance, 54:1647–1691, 1999.

[309] R.S. Sulton and A.G. Barks. Reinforcement Learning, An Introduction, Second

Edition. MIT Press, Boston, 2018.

Bibliography

71

[310] R.J. Sweeney. Some new ﬁlter rule tests: Methods and results. Journal of

Financial and Quantitative Analysis, 20(3):285–300, 1988.

[311] G.E. Tauchen and M. Pitts. The price variability-volume relationship on spec-

ulative markets. Econometrica, 51:485–505, 1983.

[312] P.C. Tetlock. Giving content to investor sentiment: The role of media in the

stock market. Journal of Finance, 62(3):1139–1168, 2007.

[313] I.M. Toke. An introduction to Hawkes processes with applications to ﬁnance,

2011. Lecture Notes from Ecole Centrale Paris, BNP, Paribas.

[314] B. Toth, Y. Lemperiere, C. Deremble, J. De Lataillade, J. Kockelkoren, and
J.-P. Bouchaud. Anomalous price impact and the critical nature of liquidity in
ﬁnancial markets. Physical Review X, 1(2):021006, 2011.

[315] R. Tsay. Analysis of Financial Time Series, Third Edition. Wiley, 2010.

[316] G. Tsoukalas, J. Wang, and K. Giesecke. Dynamics portfolio execution. Man-

agement Science, 2019.

[317] J. Tu and G. Zhou. Markowitz meets talmud: A combination of sophisticated
and naïve diversiﬁcation. Journal of Financial Economics, 99:204–215, 2011.

[318] F. Vahid and R.F. Engle. Common trends and common cycles. The Journal of

Applied Econometrics, 8:341–360, 1993.

[319] R. Velu, A. Gretchika, M. Benaroch, D. Nehren, and K. Kuber. Market Impact:
To Trade Small or to Trade Seldom? Evidence from Algorithmic Execution
Data. Unpublished, 2015.

[320] B. von Beschwitz, D.B. Keim, and M. Massa. First to “read” the news: News
analytics and algorithmic trading. Review of Financial Studies (to appear),
2019.

[321] A.A. Weiss. ARMA models with ARCH errors. The Journal of Time Series

Analysis, 5:129–143, 1984.

[322] M. West and J. Harrison. Bayesian Forecasting and Dynamic Models, Second

Edition. Springer-Verlag, New York, 1997.

[323] H. White. A reality check for data snooping. Econometrica, 68:1097–1126,

1999.

[324] R. De Winne and C. D’Hondt. Hide-and-seek in the market: Placing and detect-

ing hidden order. Review of Finance, 11:663–692, 2007.

[325] H. Wold. PLS regression, volume 6. Eds. N.L. Johnson and S. Kotz, 1984.

[326] H. Working. Note on the correlation of ﬁrst diﬀerences of averages in a random
chain. Econometrica: Journal of the Econometric Society, pages 916–918,
1960.

72

Bibliography

[327] D. Yang and Q. Zhang. Drift-independent volatility estimation based on high,
low, open and close prices. The Journal of Business, 73:477–491, 2000.

[328] J.W. Yang. Transaction duration and asymmetric price impact of trades - Evi-
dence from Australia. Journal of Empirical Finance, 18:91–102, 2011.

[329] J. Yu and Y. Yuan. Investor sentiment and the mean–variance relation. Journal

of Financial Economics, 100:367–381, 2011.

[330] E. Zarinelli, M. Treccani, J.D. Farmer, and F. Lillo. Beyond the square root:
Evidence for logarithmic dependence of market impact on size and participa-
tion rate. Market Microstructure and Liquidity, 1:1–31, 2015.

[331] G. Zhou. Measuring investor sentiment. Annual Review of Financial Eco-

nomics, 10:239–259, 2018.

[332] Y. Zhu and G. Zhou. Technical analysis: An asset allocation perspective on the
use of moving averages. Journal of Financial Economics, 92:519–544, 2009.



---

## Content from Previous Extraction (not in markitdown output)

### Visual/Chart/Graph Descriptions

#### 1

Contents ix
12 The Research Stack 401
12.1 Data Infrastructure . . . . . . . . . . . . . . . . . . . . . . . . . . 401
12.2 Calibration Infrastructure . . . . . . . . . . . . . . . . . . . . . . 403
12.3 Simulation Environment . . . . . . . . . . . . . . . . . . . . . . . 404
12.4 TCAEnvironment . . . . . . . . . . . . . . . . . . . . . . . . . . 408
12.5 Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 410
Bibliography 411
Subject Index 433

#### 2

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

#### 3

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

#### 4

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

#### 5

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

#### 6

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

#### 7

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

#### 8

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

#### 9

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

#### 10

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

#### 11

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

#### 12

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

#### 13

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

#### 14

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

#### 15

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

#### 16

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

#### 17

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

#### 18

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

#### 19

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

#### 20

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

#### 21

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

#### 22

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

#### 23

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

#### 24

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

#### 25

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

#### 26

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

#### 27

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

#### 28

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

#### 29

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

#### 30

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

### Additional Content

#### 1

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

#### 2

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

#### 3

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

#### 4

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

#### 5

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

#### 6

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

#### 7

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

#### 8

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

#### 9

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

#### 10

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

#### 11

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

#### 12

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

#### 13

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

#### 14

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

#### 15

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

#### 16

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

#### 17

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

#### 18

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

#### 19

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

#### 20

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


---

## Content Unique to Old Extraction (not found in markitdown output)

### 1. # Algorithmic Trading and Cookbook...

# Algorithmic Trading and Cookbook

### 2. > *Source PDF: Algorithmic Trading and Cookbook.pdf*
> *Extraction: Combined — m...

> *Source PDF: Algorithmic Trading and Cookbook.pdf*
> *Extraction: Combined — markitdown raw text (base) + previous extraction supplements*

### 3. ## Content from Previous Extraction (not in markitdown output)...

## Content from Previous Extraction (not in markitdown output)

### 4. ### Visual/Chart/Graph Descriptions...

### Visual/Chart/Graph Descriptions

### 5. Contents ix
12 The Research Stack 401
12.1 Data Infrastructure . . . . . . . . ....

Contents ix
12 The Research Stack 401
12.1 Data Infrastructure . . . . . . . . . . . . . . . . . . . . . . . . . . 401
12.2 Calibration Infrastructure . . . . . . . . . . . . . . . . . . . . . . 403
12.3 Simulation Environment . . . . . . . . . . . . . . . . . . . . . . . 404
12.4 TCAEnvironment . . . . . . . . . . . . . . . . . . . . . . . . . . 408
12.5 Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 410
Bibliography 411
Subject Index 433

### 6. xii Preface
intotheactualimplementation, whilefamiliarizingthemwiththelibrariesa...

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

### 7. xiv Preface
ourdoctoralstudents, Kris Hermanand Zhaoque Zhou (Chosen) fortheirhe...

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

### 8. 1
Trading Fundamentals
1.1 ABrief Historyof Stock Trading
Why We Trade:Companies...

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

### 9. 4 Algorithmic Tradingand Quantitative Strategies
the world,1 nor the oldest in t...

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

### 10. 20 Algorithmic Tradingand Quantitative Strategies
lower prices for sells, and fo...

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

### 11. Trading Fundamentals 21
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
73600 738...

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

### 12. Trading Fundamentals 33
originallycreatedforhedgingpurposesandasaresulthaveexpir...

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

### 13. 34 Algorithmic Tradingand Quantitative Strategies
Figure1.4:Futures Volume Rolli...

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

### 14. Trading Fundamentals 37
momentintraday. Forillustrationpurposes, wetaketheexampl...

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

### 15. Trading Fundamentals 43
actualvaluegetspublishedbythecompany. Here, too, consens...

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

### 16. Bibliography
[1] F. Abdi and A. Ranaldo. A simple estimation of bid-ask spreads ...

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

### 17. 52 Bibliography
[14] S.Anatolyevand A.Gospodinov. Atradingapproachtotestingforpr...

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

### 18. Bibliography 53
[29] F.M.Bandiand J.R.Russell. Separatingmicrostructurenoisefrom...

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

### 19. 54 Bibliography
[43] F.Black. Capitalmarketequilibriumwithrestrictedborrowing. T...

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

### 20. 56 Bibliography
[74] L.K.C.Chanand J.Lakonishok. Thebehaviorofstockpricesaroundi...

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

### 21. Bibliography 57
[90] S.A.Corwinand P.Schultz. Asimplewaytoestimatebid-askspreads...

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

### 22. 58 Bibliography
[106] B.Doand R.Faff. Doessimplepairstradingstillwork?The Financ...

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

### 23. 60 Bibliography
[137] J. Fan, J. Zhang, H. Liu, and K. Yu. Vast portfolio select...

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

### 24. Bibliography 61
[152] P.Fryzlewicz, T.Sapatinas, and S.Subba Rao. Normalizedleas...

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

### 25. 62 Bibliography
[168] I. Goodfellow, Y. Bengio, and A. Courville. Deep Learning....

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

### 26. Bibliography 63
[184] T.Hastie, R.Tibshirani, and J.Friedman. The Elementsof Sta...

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

### 27. 64 Bibliography
[200] H.Hotelling. Analysisofacomplexofstatisticalvariablesintop...

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

### 28. Bibliography 65
[216] N.Jegadeeshand S.Titman. Profitabilityofmomentumstrategies...

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

### 29. 66 Bibliography
[232] A. Kourtis. On the distribution and estimation of trading ...

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

### 30. Bibliography 67
[247] A.W. Lo, H. Mamaysky, and J. Wang. Foundation of technical...

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


---

## Raw Markitdown Extraction (full text)

Algorithmic Trading and
Quantitative Strategies

Algorithmic Trading and
Quantitative Strategies

Raja Velu
Department of Finance
Whitman School of Management
Syracuse University

Maxence Hardy
eTrading Quantitative Research
J.P. Morgan

Daniel Nehren
Statistical Modeling & Development
Barclays

First edition published 2020
by CRC Press
6000 Broken Sound Parkway NW, Suite 300, Boca Raton, FL 33487-2742

and by CRC Press
2 Park Square, Milton Park, Abingdon, Oxon, OX14 4RN

© 2020 Taylor & Francis Group, LLC

CRC Press is an imprint of Taylor & Francis Group, LLC

Reasonable eﬀorts have been made to publish reliable data and information, but the author and publisher
cannot assume responsibility for the validity of all materials or the consequences of their use. The authors
and publishers have attempted to trace the copyright holders of all material reproduced in this publication
and apologize to copyright holders if permission to publish in this form has not been obtained. If any
copyright material has not been acknowledged please write and let us know so we may rectify in any
future reprint.

Except as permitted under US Copyright Law, no part of this book may be reprinted, reproduced, trans-
mitted, or utilized in any form by any electronic, mechanical, or other means, now known or hereafter
invented, including photocopying, microﬁlming, and recording, or in any information storage or retrieval
system, without written permission from the publishers.

For permission to photocopy or use material electronically from this work, access www.copyright.com or
contact the Copyright Clearance Center, Inc. (CCC), 222 Rosewood Drive, Danvers, MA 01923, 978-750-
8400. For works that are not available on CCC please contact mpkbookspermissions@tandf.co.uk

Trademark notice: Product or corporate names may be trademarks or registered trademarks, and are used
only for identiﬁcation and explanation without intent to infringe.

Library of Congress Control Number: 2020932899

ISBN: 9781498737166 (hbk)
ISBN: 9780429183942 (ebk)

Typeset in STIXGeneral
by Nova Techset Private Limited, Bengaluru & Chennai, India

Contents

Preface

I

Introduction to Trading

1 Trading Fundamentals

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

1.3 The Mechanics of Trading .

.
1.2.1 Equity Markets Participants
1.2.2 Watering Holes of Equity Markets .
.

.
.
.
1.3.1 How Double Auction Markets Work .
.
.
1.3.2 The Open Auction .
.
1.3.3 Continuous Trading .
.
1.3.4 The Closing Auction .

.
.
.
.
.
.
.
1.4 Taxonomy of Data Used in Algorithmic Trading
.
.
.

1.1 A Brief History of Stock Trading
.
1.2 Market Structure and Trading Venues: A Review .
.
.
.
.
.
.
.
.
.
.
.
.

.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
Fundamental Data and Other Data Sets
1.5 Market Microstructure: Economic Fundamentals of Trading .
.

1.4.1 Reference Data .
.
.
.
1.4.2 Market Data
1.4.3 Market Data Derived Statistics .
1.4.4

.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
. .
.
.

1.5.1 Liquidity and Market Making .

.
.
.
.
.
.
.
.
.
.
.
.
.
.

.
.
.
.
.
.
.
.
.
.
.
.
.
.

.
.
.
.
.
.
.
.
.
.
.
.
.
.

.
.
.
.
.
.
.

.
.
.
.

.
.
.

.
.
.

.
.
.

.
.
.

.
.
.

.
.
.

.
.
.

.
.
.

.
.
.

.
.

.
.

.
.

.
.

.
.

.
.

.
.

.

.

.

.

.

.

.

.

.

.

.

.

.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.

.
.
.
.
.
.
. .
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.

II

Foundations: Basic Models and Empirics

2 Univariate Time Series Models

.

.

.

.

.

.

.

.

.

Processes to Discrete Time Series

2.1 Trades and Quotes Data and Their Aggregation: From Point
.
.
.
.

.
. .
2.2 Trading Decisions as Short-Term Forecast Decisions
.
.
.
2.3 Stochastic Processes: Some Properties .
.
.
.
. .
.
2.4 Some Descriptive Tools and Their Properties
2.5 Time Series Models for Aggregated Data: Modeling the Mean .
2.6 Key Steps for Model Building .
.
.
2.7 Testing for Nonstationary (Unit Root) in ARIMA Models: To
.
.
.
.

.
.
.
2.8 Forecasting for ARIMA Processes .
2.9 Stylized Models for Asset Returns .
.
2.10 Time Series Models for Aggregated Data: Modeling the Variance .

Diﬀerence or Not To .

.
.
.
.
. .
.
.

.
.
.
.
.
.

.
.
.
.

.
.
.

.
.
.

.
.
.

.
.
.

.
.
.

.
.
.

.
.
.

.
.
.

.
.
.

.
.
.

.
.
.

.
.
.

.
.
.

.
.
.

.
.

.
.

.
.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.
.
.
.
.
.

.
.
.
.

xi

1

3
3
7
7
8
13
13
14
19
24
26
26
35
39
42
44
45

51

53

54
56
57
61
63
70

77
78
82
84

v

vi

2.11 Stylized Models for Variance of Asset Returns
.
.
.
2.12 Exercises

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

Contents

.
.

.
.

.
.

.
.

.
.

.
.

.
.

.
.

.
.

.
.

.
.

.

90
92

.

.

.
.
.

.
.
.

3 Multivariate Time Series Models
.
.
.
3.1 Multivariate Regression
.
3.2 Dimension-Reduction Methods
3.3 Multiple Time Series Modeling
.
3.4 Co-Integration, Co-Movement and Commonality in Multiple Time
.
.
.
.
.
.
.

.
.
.
3.5 Applications in Finance
.
3.6 Multivariate GARCH Models
.
Illustrative Examples .
3.7
.
.
.
3.8 Exercises

.
.
.
.
.
.
.
.
. .

.
.
. .
.
.
.
.
.
.

.
.
. .
.
.

Series

.
.
.
.
.

.
.
.
.
.

.
.
.
.
.

.
.
.
.
.

.
.
.
.
.

.
.
.
.
.

.
.
.
.
.

.
.
.
.
.

.
.
.
.
.

.
.
.
.
.

.
.
.
.
.

.
.
.
.
.

.
.
.
.
.

.
.
.
.
.

.
.
.

.
.
.

.
.
.

.
.
.

.
.
.

.
.
.

.
.
.

.
.
.

.
.
.

.
.
.

.
.
.

.
.
.

.
.
.

.
.

.
.

.
.

.
.

.

.

.

.

.

.

.

.

.

.

.

.

4 Advanced Topics

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.
.
.
.

.
.
.
.

.
.
.
.

.
.
.
.

.
.
.
.

.
.
. .
.
.
.
.

4.5 Analysis of Time Aggregated Data

4.1 State-Space Modeling
.
4.2 Regime Switching and Change-Point Models
4.3 A Model for Volume-Volatility Relationships
.
4.4 Models for Point Processes .

.
.
.
.
.
Stylized Models for High Frequency Financial Data .

.
.
.
.
4.4.1
.
4.4.2 Models for Multiple Assets: High Frequency Context .
.
.
.
.
.
.
.
.

.
4.5.1 Realized Volatility and Econometric Models
.
4.5.2 Volatility and Price Bar Data .
.
.
4.6 Analytics from Machine Learning Literature
.
.
.
4.6.1 Neural Networks .
4.6.2 Reinforcement Learning .
.
.
4.6.3 Multiple Indicators and Boosting Methods .
.
.
.

4.7 Exercises

.
.
.
.
.
.
.
.

.
.
.
.
.
.
.
.

.
.
.
.
.
.
.
.

.
.
.
.
.
.
.
.

.
.
.
.
.
.

.
.
.
.

.
.
.
.

. .

.
.

.
.

.
.

.
.

.
.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
. .
.
.
.
.

III Trading Algorithms

5 Statistical Trading Strategies and Back-Testing

.

.

.

.

.

.

.

.

.
.

.
.
.

.
.
.

.
.
.

.
.
.
.

.
.
.
.
.

.
.
.
.
.

Filter Rules .

5.1
5.2 Evaluation of Strategies: Various Measures
5.3 Trading Rules for Time Aggregated Data
.

.
.
5.3.1
.
.
.
5.3.2 Moving Average Variants and Oscillators .

Introduction to Trading Strategies: Origin and History
.
.
.
.

.
.
.
.
.
.
.
.
.
.
.
5.4 Patterns Discovery via Non-Parametric Smoothing Methods
.
.
5.5 A Decomposition Algorithm .
5.6 Fair Value Models
.
.
.
.
.
5.7 Back-Testing and Data Snooping: In-Sample and Out-of-Sample
.
.
.
.
.
.
.
.

.
.
.
5.8.1 Distance-Based Algorithms
.
5.8.2 Co-Integration .

Performance Evaluation
.

5.8 Pairs Trading .

.
.
.
.
.
.
.
.

.
.
.
.

.
.
.
.

.
.
.
.

.
.
.
.

.
.
.
.

.
.
.
.

.
.
.
.

.
.
.
.

.
.
.
.

.
.
.
.

.
.
.
.

.
.
.
.

.
.
.
.

.
.
.
.

.
.

.
.

.
.

.
.

.
.

.
.

.
.

.
.

.
.

.
.

.
.

.
.

.
.

.
.

.
.

.
.

.
.

.

.

.

.

.

.

.

.

.

.

.

.

.
.
.
.
.
.
.
.

.
.
.
.

95
96
.
.
99
. 104

. 106
. 110
. 112
. 114
. 122

125
. 125
. 128
. 131
. 134
. 136
. 140
. 141
. 141
. 143
. 146
. 147
. 150
. 152
. 154

157

159
. 159
. 160
. 161
. 162
. 164
. 166
. 168
. 170

. 171
. 174
. 175
. 176

Contents

.
.

.
.

.
.
.

.
.
.

.
.
.

.
.
.

.
.
.

5.8.3
5.8.4

Some General Comments
Practical Considerations .

.
.
.
.
5.9 Cross-Sectional Momentum Strategies .
.
5.10 Extraneous Signals: Trading Volume, Volatility, etc.
5.10.1 Filter Rules Based on Return and Volume .
.
5.10.2 An Illustrative Example
.
.
.
.
.

.
.
5.11 Trading in Multiple Markets
5.12 Other Topics: Trade Size, etc.
.
5.13 Machine Learning Methods in Trading
.
.
.
5.14 Exercises

.
.
.
.
.

.
.
.
.
.

.
.
.
.
.

.
.
.
.
.

.
.
.
.
.

.
.
.

.
.
.

.
.
.

.
.

.

.

.

.

.

.

.

.

.

.

.

.

.

.
.
.

.
.
.
.
.
.

.
.
.
.
.
.
.
.
.
.

.

.

.

.

.

.

6.1

.
.
.
.
.
.
.

.
Implications for Investing .
.

.
Introduction to Modern Portfolio Theory
.
6.1.1 Mean-Variance Portfolio Theory .
6.1.2 Multifactor Models .
.
.
6.1.3 Tests Related to CAPM and APT .
.
6.1.4 An Illustrative Example
.
6.1.5
.

6 Dynamic Portfolio Management and Trading Strategies
.
.
.
. .
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
Portfolio Allocation Using Regularization .
.
.
Portfolio Strategies: Some General Findings .
.
.
6.3 Dynamic Portfolio Selection .
.
6.4 Portfolio Tracking and Rebalancing .
.
.
6.5 Transaction Costs, Shorting and Liquidity Constraints
.
6.6 Portfolio Trading Strategies
.
.
.
6.7 Exercises

6.2 Statistical Underpinnings

6.2.1
6.2.2

.
.
.
.
.
.
.

.
.
.
.
.
.
.

.
.
.

.
.
.

.
.
.

.
.

.
.

.
.

.
.

.
.

.
.

.
.

.
.

.
.

.
.

.
.

.
.

.
.

.
.

.
.

.
.

.
.

.
.

.
.

.
.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.
.
.
.
.
.
.
.
.
.

.
.
.
.
.
.
.
.
.
.
.
.
.
.

.
.
.
.
.
.
.
.
.
.

.
.
.
.
.
.
.
.
.
.
.
.
.
.

.
.
.
.
.
.
.
.
.
.

.
.
.
.
.
.
.
.
.
.

.
.
.
.
.
.
.
.
. .
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.

.
.
.
.
.
.
.
.
.
.

.
.
.
.
.
.
.
.
.
.
.
.
.
.

7.1

7 News Analytics: From Market Attention and Sentiment to Trading
Introduction to News Analytics: Behavioral Finance and Investor
.
.
Cognitive Biases
.
.

.
.
.
.
7.2 Automated News Analysis and Market Sentiment
7.3 News Analytics and Applications to Trading
.
.
7.4 Discussion / Future of Social Media and News in Algorithmic
.
.

Trading .

.
.
.

.
.
.

.
.
.

.
.
.

.
.
.

.
.
.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

IV Execution Algorithms

8 Modeling Trade Data

8.1 Normalizing Analytics

.

.

.

.

.

.

.

.

.

.
.

.
.

.
.

.
.

.
.

.
.

.
.
8.1.1 Order Size Normalization: ADV .
.
8.1.2 Time-Scale Normalization: Characteristic Time .
8.1.3
8.1.4 Other Microstructure Normalizations
.
8.1.5
Intraday Normalization: Proﬁles .
.
8.1.6 Remainder (of the Day) Volume .
.
.
8.1.7 Auctions Volume .

.
.
.
Intraday Return Normalization: Mid-Quote Volatility .
.
.
.
.

.
.
.
.

.
.
.
.

.
.
.
.

.
.
.
.

.
.
.
.

.
.
.
.

.
.
.
.

.
.
.
.

. .

.
.
.

.
.
.

.
.
.

.
.

.

.

.

.

.

vii

. 180
. 181
. 184
. 189
. 191
. 193
. 198
. 202
. 204
. 207

215
. 215
. 215
. 219
. 219
. 222
. 224
. 226
. 230
. 232
. 234
. 236
. 239
. 242
. 244

251

. 251
. 256
. 258

. 267

269

271
. 271
. 272
. 273
. 275
. 276
. 277
. 282
. 283

.
.
.
.
.
.
.
.
.
.

.
.
.
.
.
.
.
.
.
.
.
.
.
.

.
.
.

.

.
.
.
.
.
.
.
.

.

.
.
.
.
.
.
.
.

viii

.

.

.

.

.

.

.

.

.

.

.

.

.

8.2 Microstructure Signals
.
8.3 Limit Order Book (LOB): Studying Its Dynamics .
.
.
.
.
.

8.3.1 LOB Construction and Key Descriptives .
.
.
8.3.2 Modeling LOB Dynamics
.
8.3.3 Models Based on Hawkes Processes .
.
.
.

8.4 Models for Hidden Liquidity .
.
8.5 Modeling LOB: Some Concluding Thoughts

.
.
.
.

.

.

.

.

.

.

.

.

.

.

.

.

9 Market Impact Models
.

Introduction .

.

.

.

.

.
.

.
.

.
.

.
.

.
.
.

.
.
.

.
.
.

.
.
.

.
.
.

.
.
.

.
.
9.1
.
.
9.2 What Is Market Impact?
.
9.3 Modeling Transaction Costs (TC)
.
9.4 Historical Review of Market Impact Research .
.
9.5 Some Stylized Models
.
.
9.6 Price Impact in the High Frequency Setting .
.
.
9.7 Models Based on LOB .
.
.
.
9.8 Empirical Estimation of Transaction Costs
.
.
9.8.1 Review of Select Empirical Studies

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.
.
.
.
.
.
.
.
.

.
.
.
.
.
.
.
.
.

.
.
.
.
.
.
.

.
.
.
.
.
.
.
.
.

.
.
.
.
.
.
.

.
.
.
.
.
.
.
.
.

10 Execution Strategies

.
.
.
.
.

.
.
.
.
.
.

10.3.1 Scheduling Layer .
.
10.3.2 Order Placement
.
10.3.3 Order Routing .

10.1 Execution Benchmarks: Practitioner’s View .
.
.
10.2 Evolution of Execution Strategies
.
10.3 Layers of an Execution Strategy . .
.
.
.
.
.
.

.
.
.
.
.
.
10.4 Formal Description of Some Execution Models .
.
.

.
.
.
.
.
.
.
.
.
10.5 Multiple Exchanges: Smart Order Routing Algorithm .
.
10.6 Execution Algorithms for Multiple Assets .
.
.
10.7 Extending the Algorithms to Other Asset Classes .

10.4.1 First Generation Algorithms .
.
10.4.2 Second Generation Algorithms .

.
.
.
.
. .

.
.
.
.
.
.
.
.
.

.
.
.
.
.
.
.
.
.

.
.
.
.
.

.
.
.
.
.

.
.
.
.
.

.
.
.

.
.
.

.
.

.
.

.
.

.
.

.
.

.

.

.

V Technology Considerations

11 The Technology Stack

11.1 From Client Instruction to Trade Reconciliation .
.
11.2 Algorithmic Trading Infrastructure
.
.
11.3 HFT Infrastructure .
.
.
.
11.4 ATS Infrastructure
.
.
.

.
.
.
.
.
.
11.4.1 Regulatory Considerations .
11.4.2 Matching Engine .
.
.
11.4.3 Client Tiering and Other Rules .

.
.
. .
.
.
.
.
.
.
.
.

.
.
.
.
.
.

.
.
.
.
.
.

.
.
.
.
.

.
.

.
.

.
.

.
.

.
.

.
.

.

.

.

.

.
.
.
.
.
.
.

.
.
.
.
.
.
.

.
.
.
.
.
.
.

Contents

.
.
.
.
.
.
.

.
.
.
.
.
.
.
.
.

.
.
.
.
.
.
.
.
.
.
.
.

.
.
.
.
.
.
.

.
.
.
.
.
.
.

.
.
.
.
.
.
.
.
.

.
.
.
.
.
.
.

.
.
.
.
.
.
.
.
.

.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
. .
.
.

.
.
.
.
.
.
.

.
.
.
.
.
.
.
.
.

.
.
.
.
.
.
.
.
.
.
.
.

.
.
.
.
.
.
.

.
.
.
.
.
.
. .
.
.
.
.
.
.

.
.
.
.
.
.
.

.
.
.
.
.
.
.
.
.

.
.
.
.
.
.
.
.
.
.
.
.

.
.
.
.
.
.
.

.
.
.
.
.
.
.

.
.
.
.
.
.
.
.
.

.
.
.
.
.
.
.
.
.
.
.
.

.
.
.
.
.
.
.

. 283
. 286
. 287
. 289
. 295
. 306
. 310

313
. 313
. 314
. 315
. 318
. 320
. 323
. 324
. 327
. 328

335
. 336
. 344
. 348
. 348
. 352
. 353
. 359
. 359
. 361
. 367
. 371
. 375

381

383
. 383
. 387
. 394
. 395
. 395
. 397
. 397

Contents

12 The Research Stack

.

12.1 Data Infrastructure .
.
12.2 Calibration Infrastructure
12.3 Simulation Environment
.
12.4 TCA Environment
.
.
12.5 Conclusion .

.
.

.
.

.

.

Bibliography

Subject Index

.

.
.
.

.
.
.
.
.

.
.
.
.
.

.
.
.
.
.

.
.
.
.
.

.
.
.
.
.

.
.
.
.
.

.
.
.
.
.

.
.
.
.
.

.
.
.
.
.

.
.
.
.
.

.
.
.
.
.

.
.
.
.
.

.
.
.
.
.

.
.
.
.
.

.
.
.
.
.

.
.
.
.
.

.
.
.
.
.

.
.
.
.
.

.
.
.
.
.

.
.
.
.
.

.
.
.
.
.

ix

401
. 401
. 403
. 404
. 408
. 410

411

433

Preface

Algorithms have been around since the day trading has started. But they have
gained importance with the advent of computers and the automation of trading. Eﬃ-
ciency in execution has taken center stage and with that, speed and instantaneous
processing of asset related information have become important. In this book, we will
focus on the methodology rooted in ﬁnancial theory and demonstrate how relevant
data—both in the high frequency and in the low frequency spaces—can be mean-
ingfully analyzed. The intention is to bring both the academics and the practitioners
together. We strive to achieve what George Box (ﬁrst author’s teacher) once said:

“One important idea is that science is a means whereby
learning is achieved, not by mere theoretical speculation
on the one hand, nor by the undirected accumulation of
practical facts on the other, but rather by a motivated
iteration between theory and practice.”

We hope that we provide a framework for relevant inquiries on this topic. To quote

Judea Pearl,

“You cannot answer a question that you cannot ask, and
you cannot ask a question that you have no words for.”

The emphasis of this book, the readers will notice, is on data analysis with guidance
from appropriate models. As C.R. Rao (ﬁrst author’s teacher) has aptly observed:

All knowledge is, in ﬁnal analysis, history.
All Sciences are, in the abstract, mathematics.
All judgements are, in the rationale, statistics.

This book gives an inside look into the current world of Electronic Trading and
Quantitative Strategies. We address actual challenges by presenting intuitive and
innovative ideas on how to approach them in the future. The subject is then aug-
mented through a more formal treatment of the necessary quantitative methods with
a targeted review of relevant academic literature. This dual approach is also reﬂec-
tive of the dynamics, typical of quants working on a trading ﬂoor where commercial
needs, such as time to market, often supersede consideration of rigorous models in
favor of intuitively simple approaches.

Our unique approach in this book is to provide the reader with hands-on tools.
This book will be accompanied by a collection of practical Jupyter Notebooks where
select methods are applied to real data. This will allow the readers to go beyond theory

xi

xii

Preface

into the actual implementation, while familiarizing them with the libraries available.
Wherever possible the charts and tables in the book can be generated directly from
these notebooks and the data sets provided bring further life to the treatment of the
subject. We also add exercises to most of the chapters, so that the students can work
through on their own. These exercises have been tested out by graduate students from
Stanford and Singapore Management University. The notebooks as well as the data
and the exercises are made available on: https://github.com/NehrenD/algo_
trading_and_quant_strategies. This site will be updated on a periodic basis.
While reading and working through this book, the reader should be able to gain insight
into how the ﬁeld of Electronic Trading and Quantitative Strategies, one of the most
active and exciting spaces in the world of ﬁnance, has evolved.

This book is divided into ﬁve parts.

• Part

I sets the stage. We narrate the history and evolution of Equity Trading
and delve into a review of the current features of modern Market Structure. This
gives the readers context on the business aspects of trading in order for them to
understand why things work as they do. The next section will provide a brief high-
level foundational overview of market microstructure which explains and models
the dynamics of a trading venue heavily inﬂuenced by the core mechanism of
how trading takes place: the price-time priority limit-order book with continuous
double auction. This will set the stage for the introduction of a critical but elusive
concept in trading: Liquidity.

• Part II provides an overview of discrete time series models applied to equity
trading. We will address univariate and multivariate time series models of both
mean and variance of asset returns, and other associated quantities such as vol-
ume. While somewhat less used today because of high frequency trading, these
models are important as conceptual frameworks, and act as baselines for more
advanced methods. We also cover some essential concepts in Point Processes as
the actual trading data can come at irregular intervals. The last chapter of Part II
will present more advanced topics like State-Space Models and modern Machine
Learning methods.

• Part

III dives into the broad topic of Quantitative Trading. Here we provide
the reader with a toolkit to conﬁdently approach the subject. Historical perspec-
tives from Alpha generation to the art of backtesting are covered here. Since most
quantitative strategies are portfolio-based, meaning that alphas are usually com-
bined and optimized over a basket of securities, we will brieﬂy introduce the topic
of Active Portfolio Management and Mean-Variance Optimization, and the more
advanced topic of Dynamic Portfolio Selection. We conclude this section dis-
cussing a somewhat recent topic: News and Sentiment Analytics. Our intent is to
also remind the reader that the ﬁeld is never “complete” as new approaches (such
as this from behavioral ﬁnance) are embraced by practitioners once the data is
available.

• Part

IV covers Execution Algorithms, a sub-ﬁeld of Quantitative Trading
which has evolved separately from simple mechanical workﬂow tools, into a

Preface

xiii

multi-billion dollar business. We begin by reviewing various approaches to mod-
eling trade data, and then dive into the fundamental subject of Market Impact, a
complex and least understood concept in ﬁnance. Having set the stage, the ﬁnal
section presents a review of the evolution and the current state of the art in Exe-
cution Algorithms.

• Finally, Part V deals with some technical aspects of developing both quanti-
tative trading strategies and execution algorithms. Trading has become a highly
technological process that requires the integration of numerous technologies and
systems, ranging from market data feeds, to exchange connectivity, to low-latency
networking and co-location, to back-oﬃce booking and reporting. Developing a
modern and high performing trading platform requires thoughtful consideration
and some compromise. In this part, we look at some important details in cre-
ating a full end-to-end technology stack for electronic trading. We also want to
emphasize the critical but often ignored aspect of a successful trading business:
Research Environment.

Acknowledgments: The ideas for this book were planted ten years ago, while the
ﬁrst author, Raja Velu, was visiting the Statistics Department at Stanford at the invi-
tation of Professor T.W. Anderson. Professor Tze-Leung Lai, who was in charge of
the Financial Mathematics program, suggested initiating a course on Algorithmic
Trading. This course was developed by the ﬁrst author and the other authors, Daniel
Nehren and Maxence Hardy, oﬀered guest lectures to bring the practitioner’s view
to the classroom. This book in large part is the result of that interaction. We want to
gratefully acknowledge the opportunity given by the Stanford’s Statistics department
and by Professors Lai and Anderson.

We owe a personal debt to many people for their invaluable comments and intel-
lectual contribution to many sources from which the material for this book is drawn.
The critical reviews by Professors Guofu Zhou and Ruey Tsay at various stages
of writing are gratefully acknowledged. Colleagues from Syracuse University, Jan
Ondrich, Ravi Shukla, David Weinbaum, Lai Xu, Suhasini Subba Rao from Texas
A&M, and Jeﬀrey Wurgler from New York University, all read through various ver-
sions of this book. Their comments have helped to improve its content and the pre-
sentation. Students who took the course at Stanford University, National University
of Singapore and Singapore Management University and the teaching assistants were
instrumental in shaping the structure of the book. In particular, we want to recognize
the help of Balakumar Balasubramaniam, who oﬀered extensive comments on an ear-
lier version. On the intellectual side, we have drawn material from the classic books
by Tsay (2010); Box, Jenkins, Reinsel, Ljung (2015) on the methodology; Campbell,
Lo and MacKinlay (1996) on ﬁnance; Friedman, Hastie and Tibshirani (2009) on sta-
tistical learning. In the tone and substance at times, we could not say better than what
is already said in these classics and so the readers may notice some similarities.

We have relied heavily on the able assistance of Caleb McWhorter, who has put
the book together with all the demands of his own graduate work. We want to thank

xiv

Preface

our doctoral students, Kris Herman and Zhaoque Zhou (Chosen) for their help at vari-
ous stages of the book. The joint work with them was useful to draw upon for content.
As the focus of this book is on the use of real data, we relied upon several sources for
help. We want to thank Professor Ravi Jagannathan for sharing his thoughts and data
on pairs trading in Chapter 5 and Kris Herman, whose notes on the Hawkes process
are used in Chapter 8. The sentiment data used in Chapter 7 was provided by iSen-
tium and thanks to Gautham Sastri. Scott Morris, William Dougan and Peter Layton
at Blackthorne Inc, whose willingness to help on a short notice, on matters related to
data and trading strategies is very much appreciated. We want to also acknowledge
editorial help from Claire Harshberber and Alyson Nehren.

Generous support was provided by the Whitman School of Management and the
Department of Finance for the production of the book. Raja Velu would like to thank
former Dean Kenneth Kavajecz and Professors Ravi Shukla and Peter Koveos who
serve(d) as department chairs and Professor Michel Benaroch, Associate Dean for
Research, for their encouragement and support.

Last but not least the authors are grateful and humbled to have Adam Hogan pro-
vide the artwork for the book cover. The original piece speciﬁcally made for this book
is a beautiful example of Algorithmic Art representing the trading intensity of the US
stock universe on the various trading venues. We cannot think of a more ﬁtting image
for this book.

Finally, no words will suﬃce for the love and support of our families.

Raja Velu
Maxence Hardy
Daniel Nehren

About the Authors

Raja Velu
Raja Velu is a Professor of Finance and Business Analytics in the Whitman School
of Management at Syracuse University. He obtained his Ph.D. in Business/Statis-
tics from University of Wisconsin-Madison in 1983. He served as a marketing fac-
ulty at the University of Wisconsin-Whitewater from 1984 to 1998 before moving
to Syracuse University. He was a Technical Architect at Yahoo! in the Sponsored
Search Division and was a visiting scientist at IBM-Almaden, Microsoft Research,
Google and JPMC. He has also held visiting positions at Stanford’s Statistics Depart-
ment from 2005 to 2016 and was a visiting faculty at the Indian School of Business,
National University of Singapore and Singapore Management University. His cur-
rent research includes Modeling Big Chronological Data and Forecasting in High-
Dimensional settings. He has published in leading journals such as Biometrika, Jour-
nal of Econometrics and Journal of Financial and Quantitative Analysis.

Maxence Hardy
Maxence Hardy is a Managing Director and the Head of eTrading Quantitative
Research for Equities and Futures at J.P. Morgan, based in New York. Mr. Hardy
is responsible for the development of the algorithmic trading strategies and models
underpinning the agency electronic execution products for the Equities and Futures
divisions globally. Prior to this role, he was the Asia Paciﬁc Head of eTrading and
Systematic Trading Quantitative Research for three years, as well as Asia Paciﬁc Head
of Product for agency electronic trading, based in Hong Kong. Mr. Hardy joined J.P.
Morgan in 2010 from Societe Generale where he was part of the algo team develop-
ing execution solutions for Program Trading. Mr. Hardy holds a master’s degree in
quantitative ﬁnance from the University Paris IX Dauphine in France.

Daniel Nehren
Daniel Nehren is a Managing Director and the Head of Statistical Modelling and
Development for Equities at Barclays. Based in New York, Mr. Nehren is responsible
for the development of algorithmic trading product and model-based business logic
for the Equities division globally. Mr. Nehren joined Barclays in 2018 from Citadel,
where he was the head of Equity Execution. Mr. Nehren has over 16 years’ experience
in the ﬁnancial industry with a focus on global equity markets. Prior to Citadel, Mr.
Nehren held roles at J.P. Morgan as the Global Head of Linear Quantitative Research,
Deutsche Bank as the Director and Co-Head of Delta One Quantitative Products and
Goldman Sachs as the Executive Director of Equity Strategy. Mr. Nehren holds a
doctorate in electrical engineering from Politecnico Di Milano in Italy.

xv

xvi

Dedicated to. . .

Yasodha, without her love and support, this is not possible.

Melissa, for her patience every step of the way, and her never
ending support.

About the Authors

R.V.

M.H.

Alyson, the muse, the patient partner, the inspiration for this work
and the next. And the box is still not full.

D.N.

Part I

Introduction to Trading

2

Algorithmic Trading and Quantitative Strategies

We provide a brief introduction to market microstructure and trading from a prac-
titioner’s point of view. The terms used in this part, all can be traced back to academic
literature; but the discussion is kept simple and direct. The data, which is central to
all the analyses and inferences, is then introduced. The complexity of using data that
can arise at irregular intervals can be better understood with an example illustrated
here. Finally, the last part of this chapter contains a brief academic review of market
microstructure—a topic about the mechanics of trading and how the trading can be
inﬂuenced by various market designs. This is an evolving ﬁeld that is of much interest
to all: regulators, practitioners and academics.

1

Trading Fundamentals

1.1 A Brief History of Stock Trading
Why We Trade: Companies need capital to operate and expand their businesses. To
raise capital, they can either borrow money then pay it back over time with interest,
or they can sell a stake (equity) in the company to an investor. As part owner of
the company, the investor would then receive a portion of the proﬁts in the form of
dividends. Equity and Debt, being scarce resources, have additional intrinsic value
that change over time; their prices are inﬂuenced by factors related to the performance
of the company, existing market conditions and in particular, the future outlook of the
company, the sector, the demand and supply of capital and the economy as a whole.
For instance, if interest rates charged to borrow capital change, this would aﬀect the
value of existing debt since its returns would be compared to the returns of similar
products/companies that oﬀer higher/lower rates of return. When we discuss trading
in this book we refer to the act of buying and selling debt or equity (as well as other
types of instruments) of various companies and institutions among investors who have
diﬀerent views of their intrinsic value.

These secondary market transactions via trading exchanges also serve the purpose
of “price discovery” (O’Hara (2003) [275]). Buyers and sellers meet and agree on a
price to exchange a security. When that transaction is made public, it in turn informs
other potential buyers and sellers of the most recent market valuation of the security.
The evolution of the trading process over the last 200 years makes for an incred-
ible tale of ingenuity, ﬁerce competition and adept technology. To a large extent, it
continues to be driven by the positive (and at times not so positive) forces of making
proﬁt, creating over time a highly complex, and amazingly eﬃcient mechanism, for
evaluating the real value of a company.

The Origins of Equity Trading: The tale begins on May 17, 1792 when a group of
24 brokers signed the Buttonwood Agreement. This bound the group to trade only
with each other under speciﬁc rules. This agreement marked the birth of the New
York Stock Exchange (NYSE). While the NYSE is not the oldest Stock Exchange in

3

4

Algorithmic Trading and Quantitative Strategies

the world,1 nor the oldest in the US,2 it is without a question the most historically
important and undisputed symbol of all ﬁnancial markets. Thus in our opinion, it is
the most suitable place to start our discussion. The NYSE soon after moved their
operations to the nearby Tontine Coﬀee House and subsequently to various other
locations around the Wall Street area before settling in the current location on the
corner of Wall St. and Broad St. in 1865.

For the next almost 200 years, stock exchanges evolved in complexity and in
scope. They, however, conceptually remained unchanged, functioning as physical
locations where traders and stockbrokers met in person to buy and sell securities.
Most of the exchanges settled on an interaction system called Open Outcry where
new orders were communicated to the ﬂoor via hand signals and with a Market
Maker facilitating the transactions often stepping in to provide short term liquidity.
The advent of the telegraph and subsequently the telephone had dramatic eﬀects in
accelerating the trading process and the dissemination of information, while leaving
the fundamental process of trading untouched.

Electroniﬁcation and the Start of Fragmentation: Changes came in the late 1960s
and early 1970s. In 1971, the NASDAQ Stock Exchange launched as a completely
electronic system. Initially started as a quotation site, it soon turned into a full
exchange, quickly becoming the second largest US exchange by market capitaliza-
tion. In the meantime, another innovation was underway. In 1969, the Institutional
Networks Corporation launched Instinet, a computerized link between banks, mutual
fund companies, insurance companies so that they could trade with each other with
immediacy, completely bypassing the NYSE. Instinet was the ﬁrst example of an
Electronic Communication Network (ECN), an alternative approach to trading that
grew in popularity in the 80s and 90s with the launch of other notable venues like
Archipelago and Island ECNs.

This evolution started a trend (Liquidity Fragmentation) in market structure that
grew over time. Interest for a security is no longer centralized but rather distributed
across multiple “liquidity pools.” This decentralization of liquidity created signiﬁcant
challenges to the traditional approach of trading and accelerated the drive toward
electroniﬁcation.

The Birth of High Frequency Trading and Algorithmic Trading: The year 2001
brought another momentous change in the structure of the market. On April 9th,
the Securities and Exchange Commission (SEC)3 mandated that the minimum price
increment on any exchange should change from 1/16th of a dollar (≈ 6.25 cents) to

1This honor sits with the Amsterdam Stock Exchange dating back to 1602.

2The Philadelphia Stock Exchange has a 2 year head start having been established in 1790.

3SEC is an independent federal government agency responsible for protecting investors, and maintain-

ing fair and orderly functioning of the securities market http://www.sec.gov

Trading Fundamentals

5

1 cent.4 This seemingly minor rule change with the benign name of ‘Decimaliza-
tion’ (moving from fractions to decimal increments) had a dramatic eﬀect, causing
the average spread to signiﬁcantly drop and with that, the proﬁts of market makers
and broker dealers also declined. The reduction in proﬁt forced many market mak-
ing ﬁrms to exit the business which in turn reduced available market liquidity. To
bring liquidity back, exchanges introduced the Maker-Taker fee model. This model
compensated the traders providing liquidity (makers) in the form of rebates, while
continuing to charge a fee to the consumer of liquidity (takers).5

The maker-taker model created unintentional consequences. If one could provide
liquidity while limiting liquidity taking, one could make a small proﬁt, due to the
rebate with minimal risk and capital. This process needs to be fairly automated as
the per trade proﬁt would be minimal, requiring heavy trading to generate real rev-
enue. Trading also needs to be very fast as position in the order book and speed of
cancellation of orders are both critical to proﬁtability. This led to the explosion of
what we today call High Frequency Trading (HFT) and to the wild ultra-low latency
technology arms race that has swept the industry over the past 15 years. HFT style
trading, formerly called opportunistic market making, already existed but never as a
signiﬁcant portion of the market. At its peak it was estimated that more than 60% of
all trading was generated by HFTs.

The decrease in average trading cost, as well as the secular trend of on-line invest-
ing led to a dramatic increase in trading volumes. On the other hand, the reduction
in per-trade commission and proﬁtability forced broker-dealers to begin automating
some of their more mundane trading activities. Simple workﬂow strategies slowly
evolved into a ﬁeld that we now call Algorithmic Execution which is a main topic of
this book.

Dark Pools and Reg NMS: In the meantime, the market structure continued to evolve
and fragmentation continued to increase. In the late eighties and early nineties a new
type of trading venue surfaced, with a somewhat diﬀerent value proposition: Allowing
traders to ﬁnd a block of liquidity without having to display that information “out
loud” (i.e., on exchange). This approach promised reduced risk of information leakage
as these venues do not publish market data, only the notiﬁcation of a trade is conveyed
after the fact. These aptly but ominously named Dark Pools have become a staple in
equity trading, and now represent an estimated 30-40% of all liquidity being traded
in US Equities.

In 2005, the Regulation National Market System (Reg NMS) was introduced in an
eﬀort to address the increase in trading complexity and fragmentation. Additionally,

4The 1/16th price increment was a vestige of Spanish monetary standards of the 1600s when doubloons

where divided in 2, 4, 8 parts.

5For some readers the concepts of providing and taking liquidity might be murky at best. Do not fret,

this will all become clear when we discuss the trading process and introduce Market Microstructure.

6

Algorithmic Trading and Quantitative Strategies

it accelerated the transformation of market structure in the US. Through its two main
rules, the intent of Reg NMS was to promote best price execution for investors by
encouraging competition among individual exchanges and individual orders. The
Access Rule promotes non-discriminatory access to quotes displayed by the var-
ious trading centers. It also had established a cap, limiting the fees that a trad-
ing center can charge for accessing a displayed quote. The Order Protection Rule
requires trading centers to obtain the best possible price for investors wherever it
is represented by an immediately accessible quote. The rule designates all regis-
tered exchanges as “protected” venues. It further mandates that apart from a few
exceptions, all market participants transact ﬁrst at a price equal or better to the
best price available in these venues. This is known as the National Best Bid Oﬀer
(NBBO).

The Order Protection Rule in particular, had a signiﬁcant eﬀect on the US market
structure. By making all liquidity in protected venues of equal status it signiﬁcantly
contributed to further fragmentation. In 2005 the NYSE market share in NYSE-listed
stocks was still above 80%. By 2010, it had plunged to 25% and has not recovered
since.

Fragmentation continued to increase with new prominent entrants like BATS
Trading and Direct Edge. Speed and technology rapidly became major diﬀerentiating
factors of success for market makers and other participants. The ability to process,
analyze, and react to market data faster than competing participants, meant captur-
ing ﬂeeting opportunities and successfully avoiding adverse selection. Therefore, to
gain or maintain an edge, market participants heavily invested in technology, in faster
networks, and installed their servers in the same data centers, as the venues they trans-
acted on (co-location).

Conclusion: This whirlwind tour of the history of trading was primarily to review the
background forces that led to the dizzying complexity of modern market structure,
that may be diﬃcult to comprehend without the appropriate context. Although this
overview was limited to the USA alone, it is not meant to imply that the rest of the
world stayed still. Europe and Asia both progressed along similar lines, although at a
slower and more compressed pace.

Trading Fundamentals

7

1.2 Market Structure and Trading Venues: A Review

1.2.1 Equity Markets Participants

With the context presented in the previous section, we now provide a brief review
of the state of modern Market Structure. In order to get a sense of the underpinnings
of equity markets, it is important to understand who are the various participants, why
they trade and what their principal focus is.

Long Only Asset Managers: These are the traditional Mutual Fund providers like
Vanguard, Fidelity, etc. A sizable portion of US households invest in mutual funds as
part of their company’s pensions funds, 401K’s, and other retirement vehicles. Some
of the largest providers are of considerable size and have accumulated multiple tril-
lions of dollars under their management. They are called Long Only asset managers
because they are restricted from short selling (where you borrow shares to sell in the
market in order to beneﬁt from a price drop). They can only proﬁt through dividends
and price appreciation. These ﬁrms need to trade frequently in order to re-balance
their large portfolios, to manage in-ﬂows, out-ﬂows, and to achieve the fund’s objec-
tive. The objective may be to track and hopefully beat a competitive benchmark. The
time horizon these investors care about is in general long, from months to years. His-
torically, these participants were less concerned with transaction costs because their
investment time scales are long and the returns they target dwarf the few dozen basis
points normally incurred in transaction costs.6 What they are particularly concerned
about is information leakage. Many of these trades last from several days to weeks
and the risk is that other participants in the market realize their intention and proﬁt
from the information to the detriment of the fund’s investors.

Long-Short Asset Managers and Hedge Funds: These are large and small ﬁrms
catering to institutional clients and wealthy investors. They often run multiple strate-
gies, most often market-neutral long-short strategies, holding long and short posi-
tions in order to have reduced market exposure and thus hopefully perform well in
both raising and falling markets. Investors of this type include ﬁrms like Bridgewa-
ter, Renaissance Technologies, Citadel, Point 72 (former SAC Capital), and many
others. Their time horizon is varied, as with the strategies they employ, but in general
they are shorter than their Long Only counterparts. They also tend to be very focused
on minimizing transaction costs, because they often make many more smaller short
term investments and the transaction costs can add up to be a signiﬁcant fraction of
their expected proﬁts.

6In recent years though, competitive pressure as well as Best Execution regulatory obligations have

brought a lot of focus on transaction costs minimization to the Long Only community as well.

8

Algorithmic Trading and Quantitative Strategies

Broker-Dealers (a.k.a. Sell Side): These ﬁrms reside between the “Buy Side”
(generic term for asset management ﬁrms and hedge funds) and the various
exchanges.7 They can act either as Agent for the client (i.e., Broker) or provide liq-
uidity as Principal (i.e., Dealer) from their own accounts. They historically also had
large proprietary trading desks investing the ﬁrm’s capital using strategies not dissim-
ilar to the strategies that hedge funds use. Since the introduction of the Dodd-Frank
Act,8 the amount of principal risk that a ﬁrm can carry has dramatically decreased
and banks had to shed their proprietary trading activities either shutting down the
desks or spinning them oﬀ into independent hedge funds.

HFTs, ELPs (Electronic Liquidity Providers), and DMMs (Designated Market
Makers): These participants generate returns by acting as facilitators between the
above participants, providing liquidity and then unwinding it at a proﬁt. They can
also act as aggregators of retail liquidity, e.g., individual investors who trade using
online providers like ETrade. Usually the most technology savvy operators in the
market place, they leverage ultra-low-latency infrastructure in order to be extremely
nimble. They get in and out of positions rapidly, taking advantage of tiny mis-pricing.

1.2.2 Watering Holes of Equity Markets

Now that we know who the players are, we brieﬂy review various ways they access

liquidity.

Exchanges: This is still the standard approach to trading and accounts for about 60–
70% of all activity. This is where an investor will go for immediacy and a surer out-
come. Because the full order-book of an exchange, arrivals/cancellations are all pub-
lished, the trader knows exactly the liquidity that is available and can plan accordingly.
This information, it should be kept in mind, is known to all participants, especially
(due to their often technological advantages) the ones whose strategies focus on pat-
terns of large directional trades. A trader that needs to buy or sell in large size will
need to be careful about how much information their trades disseminate or they may
pay dearly. The whole ﬁeld of Algorithmic Execution evolved as an eﬀort to trade
large positions while minimizing market impact and informational leakage.

Apart from these concerns, interacting with exchanges is arguably the most basic
task to trading. But it is not straightforward. At the time of this writing, a US equity
trader can buy or sell stocks in 15 registered exchanges (13 actively trading).9 As
mentioned before when discussing Reg NMS, these exchanges are all “protected.”
Thus, the liquidity at the best price cannot be ignored. An exchange will need to

7Note that only member ﬁrms are allowed to trade on exchange and most asset management ﬁrms are

non-members thus need an intermediary to trade on their behalf.

8https://www.cftc.gov/LawRegulation/DoddFrankAct/index.htm

9https://www.sec.gov/fast-answers/divisionsmarketregmrexchangesshtml.html

Trading Fundamentals

9

reroute to other exchanges where price is better (charging a fee for it). Smart Order
Routers have evolved to manage this complexity.

Recent years have seen a consolidation of these exchanges in the hands of mainly
three players: ICE, NASDAQ and CBOE. Here is a list of venues operated by each,
respectively:

• NYSE, ARCA, MKT (former AMX, American Stock Exchange), NSX (former

National Stock Exchange), CHX (former Chicago Stock Exchange)

• NASDAQ, PHLX (former Philadelphia Stock Exchange), ISE (former Interna-

tional Securities Exchange), BX (former Boston Stock Exchange)

• CBOE, BZX (former BATS), BYX (former BATS-Y), EDGA, EDGX

The newest exchange and as of now the only remaining independent exchange, is the
Investors Exchange (IEX).10 We will discuss more about this later.

An interesting observation is that this consolidation did not happen with a contem-
poraneous reduction in fragmentation. These venues continue to operate as separate
pools of liquidity. While the exchange providers make valid arguments that they pro-
vide diﬀerent business propositions, there is a growing concern in the industry that
the revenue model for these exchanges is now largely centered around providing mar-
ket data and charging for exchange connectivity fees.11 Because the venue quotes are
protected, any serious operator needs to connect with the exchanges and leverage their
direct market data feeds in their trading applications. With more exchanges, the more
connections and fees these exchange can charge. Recently regulators are starting to
weigh in on this contentious topic.12

It is also interesting to note that most of these exchanges are hosted in one of
four data centers located in the New Jersey countryside: Mahwah, Secaucus, Carteret,
Weehawken. The distance between these data centers adds some latency in the dis-
semination of information across exchanges and thus creates latency arbitrage oppor-
tunities. Super fast HFTs co-locate in each data center and leverage the best available
technology such as microwave and more recently laser technologies to connect them.
These operators can see market data changes before others do. Then they either trade
faster or cancel their own quotes to avoid adverse selection (a phenomenon known as
liquidity fading).

All exchanges have almost exactly the same trading mechanism and from the trad-
ing perspective, behave exactly the same way during the continuous part of the trading
day except for the opening and closing auctions. They provide visible (meaning that
market data is disseminated) order books and operate on a price/time priority basis.13

10In 2019, a group of ﬁnancial institutions ﬁled an application for a members owned exchange, MEMX.

11https://www.sifma.org/wp-content/uploads/2019/01/Expand-and-SIFMA-An-

Analysis-of-Market-Data-Fees-08-2018.pdf

12https://www.sec.gov/tm/staff-guidance-sro-rule-filings-fees

13Certain Futures and Options exchanges have a pro-rata matching mechanism which is discussed later.

10

Algorithmic Trading and Quantitative Strategies

Most of the exchanges use the maker-taker fee model that we discussed earlier but
some: BYX, EDGA, NSX, BX, have adopted an ‘inverted’ fee model, where post-
ing liquidity incurs a fee while a rebate is provided for taking liquidity. This change
causes these venues to display a markedly diﬀerent behavior. The cost for providing
liquidity removes the incentive of rebate seeking HFTs; however these venues are
used ﬁrst when needing immediate liquidity as the taker is compensated. This in turn
brings in passive liquidity providers who are willing to pay for trading passively at
that price. This interplay adds a subtle and still not very well understood dynamic to
an already complex market structure.

Finally, a few observations about IEX. It started as an Alternative Trading System
(ATS) whose main innovation is a 38 mile coil of optical ﬁber placed in front of its
trading engine. This introduces a 350 microsecond delay each way aptly named the
“speed bump” that is meant to remove the speed advantages of HFTs. The intent is to
reduce the eﬃcacy of the more latency-sensitive tactics and thus provide a liquidity
pool that is less “toxic.” On June 17, 2016 IEX became a full ﬂedged exchange after a
very controversial process.14,15 It was marketed as an exchange that would be diﬀerent
and would attract substantial liquidity due to its innovative speed bump. But, as of
2019, this potential remains still somewhat unrealized and IEX has not moved much
from the 2% to 3% market share range of which 80% is still in the form of hidden
liquidity. That being said IEX has established itself as a vocal critic16 of the current
state of aﬀairs continuing to shed light on some potential conﬂicts of interest that
have arisen in trading. At the time of this writing IEX does not charge any market
data fees.17

ATS/Dark Pools: As discussed above, trading in large blocks in exchanges is not a
simple matter and requires advanced algorithms for slicing the block orders and smart
order routers for targeting the liquid exchanges. Even then the risk and the result-
ing costs due to information leakage can be signiﬁcant. Dark Pools were invented
to counterbalance this situation. One investor can have a large order “sitting” in a
dark pool with no one knowing that it is there and would be able to ﬁnd the other
side without showing any signals of the order’s presence. Dark pools do not display
any order information and use the NBBO (National Best Bid/Oﬀer) as the reference
price. In almost all cases, to avoid accessing protected venues, these pools trade only
at the inside market (at or within the bid ask spread). Although, in order to maxi-
mize the probability of ﬁnding liquidity most of the block interaction happens at the
mid-point. Oﬀ-exchange trading has gained more and more traction in the last ﬁfteen

14https://www.sec.gov/comments/10-222/10-222.shtml

15https://www.bloomberg.com/news/articles/2016-06-14/sec-staff-recommends-

approving-iex-application-wsj-reports

16https://iextrading.com/insights

17https://iextrading.com/trading/market-data/

Trading Fundamentals

11

years and now accounts for 30–40% of all traded volume in certain markets like the
US. ATS/Dark Pool volume makes up roughly 30% of this US equities oﬀ-exchange
volume. Clearly, the growth of dark execution has spurred a lot of competition in the
market. Additionally, the claim of reduced information leakage is probably overstated
as it is still potentially possible to identify large blocks of liquidity by “pinging” the
pool at minimum lot size. In order to counteract this eﬀect orders are usually sent
with a minimum ﬁll quantity tag which allows the block to be transparent from small
pinging.

As of 2019, there are 33 diﬀerent equity ATSs!18 All these venues compete on
pricing, availability of liquidity, system performance, and functionality such as han-
dling of certain special order types, etc. Many of these ATS are run by the major
investment banks and they have historically dominated this space. As far as overall
liquidity goes, 10 dark pools account for about 75% of all ATS volume, and the top
5 make up roughly 50% of dark liquidity in the US (the UBS ATS and Credit Suisse
Crossﬁnder are consistently at the top of the rankings19). Other venues were born
out of the fundamental desire for investment ﬁrms to trade directly with each other,
bypassing the intermediaries and thus reducing cost and information leakage. The
main problem encountered by these buy-side to buy-side pools is that many of the
trading strategies used by these ﬁrms tend to be highly correlated (e.g., two funds
tracking the same benchmark) and thus the liquidity is often on the same side. There-
fore as a result, these venues have to ﬁnd diﬀerent approaches to leverage sell side
broker’s liquidity to supplement their own such as access via conditional orders. BIDS
and Liquidnet are the biggest ATS of this type. BIDS had signiﬁcant growth in recent
years with Liquidnet losing its initial dominance. This space is still very active with
new venues coming up on a regular basis.

Single Dealer Platform/Systematic Internalizers: A large number of trading ﬁrms
in recent years started providing direct access to their internal liquidity. Broker/Dealer
and other institutional clients connect to a Single Dealer Platform (SDP) directly.
These SDPs, also called Systematic Internalizers, send regular Indication Of Interest
(IOI) that are bespoke to a particular connection and the broker can respond when
there is a match. This approach to trading is also growing fast. Because SDPs are not
regulated ATS, they can oﬀer somewhat unique products. Brokers themselves are now
starting to provide their own SDPs to expose their internal liquidity. A quickly evolv-
ing space that promises interesting innovations, but alas, it can also lead to further
complication in an already crowded ecosystem. In the US, the other 70% of non-ATS
oﬀ-exchange volume is comprised of Retail Wholesalers, Market Makers and Single
Dealer Platforms, and Broker Dealers.

Auctions: Primary exchanges (exchanges where a particular instrument is listed)
begin and end the day with a (primary) auction procedure, that leverages special order

18http://www.finra.org/industry/equity-ats-firms

19https://otctransparency.finra.org/otctransparency/AtsData

12

Algorithmic Trading and Quantitative Strategies

types to accumulate supply and demand and then run an algorithm that determines
the price that would at best pair oﬀ the most volume. The Closing Auction is of par-
ticular importance because many funds set their Net Asset Value (NAV) using the
oﬃcial closing price. This generally leads traders to trade as close as possible to the
closing time in order to optimize the dual objective of getting the best price but also
not deviating too much from the close price.

The Auction also represents an opportunity for active and passive investors to
exchange large amounts of shares (liquidity). Index constituents get updated on a reg-
ular basis (additions, deletions, weight increase/decrease), and as they get updated,
passive investors need to update their holdings to reﬂect the optimal composition
of the benchmark that they track. In order to minimize the tracking error risk, that
update needs to happen close to the actual update of the underlying benchmark. Con-
sequently, most passive indexers tend to rebalance their portfolios on the same day
the underlying index constituents are updated, using the closing auction as a reference
price, and this results in signiﬁcant ﬂows at the close auction. The explosive growth of
ETFs (Exchange Traded Funds) and other passive funds have exacerbated this trend
in recent years. At the time of this writing, about 10% of the total daily US volume in
index names trade at the close. Recent months have brought a lot of movement in this
area with brokers and SDPs trying to provide unique ways to expose internal liquidity
marked for the closing auction.

Beyond the US: As previously mentioned the structure we presented above is not
unique to the US. European and Asian exchanges that had operated as a single mar-
ketplace for longer than their US counterparts now oﬀer a more diverse landscape.
The evolution of ATS/Dark Pools and other MTFs (Multilateral Trading Facilities)
have followed suit but in a more subdued manner. Trading in these markets has always
been smaller and concentrated with fewer participants and there is not enough liq-
uidity to support a large number of venues. Often new venues come on-line but are
quickly absorbed by a competitor when they fail to move a signiﬁcant portion of the
traded volume. Additional complexity arises with regulatory environments in these
various countries limiting cross-border trading. As of December 2018, fragmentation
in European markets is still quite lower than in the US, with 59% traded on primary
exchange, 22% on lit MTF (there are only 6, and 4 of them trade roughly 98% of the
volume), 6.5% on dark MTF (10 diﬀerent ones), and 6% traded in systematic internal-
izers. In APAC, with the exception of Australia, Hong Kong and Japan, most countries
only have one primary exchange where all transactions take place. Even in the more
developed market, Japan for instance, the Tokyo Stock Exchange still garners over
85% of the total volume traded.

Summary: Modern market structure may appear to be a jumbled mess. Yes, it is.
Fully understanding the implications of these diﬀerent methods of trading tied by
regulation, competition, behavioral idiosyncrasies (at times due to participants lack
of understanding and preconceived notions) is a daunting task. It is however an envi-
ronment that all practitioners have to navigate, and it remains diﬃcult to formulate
these issues in mathematical models as they could be based on questionable heuris-
tics to account for the residual complexity. But even with the dizzying complexity,

Trading Fundamentals

13

modern market structure is a fascinating ecosystem. It is a continuously evolving sys-
tem through the forces of ingenuity and competition. It provides tools and services to
institutional investors who strive to reduce the cost of execution, an investment man-
date. We hope the above treatment provides the reader with at least a foothold in the
exploration of this amazing social and ﬁnancial experimentation.

1.3 The Mechanics of Trading

In order to fully grasp the main topics of this book, one requires at least a good
understanding of the mechanics of trading. This process is somewhat complicated
and requires some technical details and terminology. In this section, we will strive to
provide a brief but fairly complete overview of the fundamentals. This should suﬃce
for our purposes. For a more complete and thorough treatment we refer the readers to
the existing literature, most notably Harris (2003) [178].

1.3.1 How Double Auction Markets Work

The most common approach used by modern electronic exchanges can be termed,
as time/price priority, continuous double auction trading system. The term double
auction signiﬁes that, unlike a common auction with one auctioneer dealing with
potential buyers, in this case there are multiple buyers and multiple sellers partic-
ipating in the process at the same time. These buyers and sellers interact with the
exchange by sending instructions electronically, via a network protocol to a special-
ized software and hardware infrastructure called: The Matching Engine. It has two
main components: The Limit Order Book and the Matching Algorithm.

Limit Order Book (LOB): It is a complex data structure that stores all non-executed
orders with associated instructions. It is highly specialized so as to be extremely fast
to insert/update/delete orders and then able to sort them and to retrieve aggregated
information. For an active stock, the LOB can be updated and queried thousands of
times every second, so it must be highly eﬃcient and able to handle a high degree
of concurrency to ensure that the state is always correct. The LOB is comprised of
two copies of the core data structure, one for Buy orders and one for Sell orders often
referred to as the two “sides” of the order book. This structure is the core abstraction
for all electronic exchanges and so it is very important to understand it in detail.

The LOB supports three basic instructions: Insert, cancel, and amend, with insert
initiating a new order, cancel removing an existing order from the market and amend
modifying some of the parameters of the existing order. New orders must specify
“order type” and associated parameters necessary to fully encapsulate the trader deci-
sion. We will review order types in more detail later, but we start with the two main
types: The limit order and the market order. The main diﬀerence between the two

14

Algorithmic Trading and Quantitative Strategies

order types is that a limit order has a price associated to it while a market order does
not.

Accounting for limit and market orders, there are eight events at any given time,

four on either side, that can alter the state of the order book:

• Limit Order Submission: A limit order is added to the queue at the speciﬁed

price level.

• Limit Order Cancellation: An outstanding limit order is expired or canceled

and is therefore removed from the Limit Order Book (LOB).

• Limit Order Amendment: An outstanding limit order is modiﬁed by the original

sender (such as changing order size).

• Execution: Buy and sell orders at appropriate prices are paired by the matching
algorithm (explained below) into a binding transaction and are removed from the
LOB

Matching Algorithm: This software component is responsible for interpreting the
various events to determine if any buy and sell orders can be matched in an execu-
tion. When multiple orders can be paired the algorithm uses the so-called price/time
priority meaning that ﬁrst the order with the most competitive prices are matched and
when prices are equal the order that arrived prior is chosen. As we will see in later
chapters this is only one of the possible algorithms used in practice but it is by far the
most common. We will go into more details in the next sections.

The matching algorithm operates continuously throughout the trading hours. In
order to ensure an orderly start and end, this continuous session is usually comple-
mented by a couple of discrete auctions. The trading day generally starts with an open
auction, then followed by the main continuous session, and ends with a closing auc-
tion. Some markets like Japan also have a lunch break which might be preceded by a
morning closing auction and followed by an afternoon opening auction. We will now
discuss these main market phases in chronological order.

1.3.2 The Open Auction

The Open Auction is only one type of call auction that is commonly held on
exchanges. The term “call auction” explains the liquidity-aggregating nature of this
event. Market participants are ‘called’ to submit their quotes to the market place in
order to determine a matching price that will maximize the amount of shares that can
be transacted. To facilitate timely and orderly cross, auctions have strict order sub-
mission rules, including speciﬁed timing for entries (see Table 1.1) and information
dissemination to prevent wild price ﬂuctuations and ensure that the process is eﬃcient
for price discovery.

Most exchanges publish order imbalance that exists among orders on the opening
or closing books, along with the indicative price and volume. For instance, Nasdaq

Trading Fundamentals

15

Table 1.1: Nasdaq Opening Cross

4:00 a.m. EST Extended hours trading and order entry begins.
9:25 a.m. EST Nasdaq enters quotes for participants with no open interest.
9:28 a.m. EST Dissemination of order imbalance information every 1 second.

Market-on-open orders must be received prior to 9:28 a.m.

9:30 a.m. EST The opening cross occurs.

publishes the following information20 between 9:28 a.m. EST and 9:30 a.m. EST,
every 1 second, on its market data feeds:

• Current Reference Price: Price within the Nasdaq Inside at which paired shares
are maximized, the imbalance is minimized and the distance from the bid-ask
mid-point is minimized, in that order.

• Near Indicative Clearing Price: The crossing price at which orders in the Nas-
daq opening / closing book and continuous book would clear against each other.

• Far Indicative Clearing Price: The crossing price at which orders in the Nasdaq

opening / closing book would clear against each other.

• Number of Paired Shares: The number of on-open or on-close shares that Nas-

daq is able to pair oﬀ at the current reference price.

• Imbalance Shares: The number of opening or closing shares that would remain

unexecuted at the current reference price.

• Imbalance Side: The side of the imbalance: B = buy-side imbalance; S = sell-
side imbalance; N = no imbalance; O = no marketable on-open or on-close orders.

In a double auction setup, the existence of multiple buyers and sellers requires
employing a matching algorithm to determine the actual opening price which we
will illustrate with a practical example. Table 1.2 gives an example of order book
submissions for a hypothetical stock, where orders are ranked based on their arrival
time. Diﬀerent exchanges around the world apply slightly diﬀerent mechanisms to
their auctions, but generally the following rules apply to match supply and demand:

• The crossing price must maximize the volume transacted.

• If several prices result in similar volume transacted, the crossing price is the one

the closest from the last price.

• The crossing price is identical for all orders executed.

• If two orders are submitted at the same price, the order submitted ﬁrst has priority.

20Source: Nasdaq Trader website.

16

Algorithmic Trading and Quantitative Strategies

Table 1.2: Pre-Open Order Book Submissions

Timestamp Seq. Number Side Quantity
1500
1750
4500
1750
2500
1200
500
500
1930
1000
3500
2000
4750
2750
10000
3000
5500
1800
800
1200
5000
12500
450
3500
1120

9:01:21
9:02:36
9:05:17
9:06:22
9:06:59
9:07:33
9:07:42
9:08:18
9:09:54
9:09:55
9:10:04
9:10:39
9:11:13
9:11:46
9:12:21
9:12:48
9:13:12
9:14:51
9:15:02
9:15:37
9:16:42
9:17:11
9:18:27
9:19:13
9:19:54

1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25

B
S
B
S
S
B
B
B
S
B
S
B
S
B
S
B
B
S
B
S
S
B
S
S
B

Price
12.10
12.12
12.17
12.22
12.11
12.23
12.33
12.25
12.30
12.21
12.05
12.34
12.25
12.19
12.33
12.28
12.35
12.18
12.17
12.19
12.16
12.15
12.23
12.20
12.16

• It is possible for an order to be partially executed if the other side quantity is not

suﬃcient.

• “At Market” orders are executed against each other at the determined crossing
price, up to the available matching quantity on both sides, but generally do not
participate in the price formation process.

• For the Open Auction, unmatched “At Market” orders are entered into the con-

tinuous session of LOB as limit orders at the crossing price.

The ﬁrst step is to organize orders by limit price, segregating buys and sells as
shown in Table 1.3. A buy order submitted with a limit price of 12.25 represents an
intent to execute at any price lower or equal to 12.25. Similarly, a sell order submit-
ted at 12.25 represents an intent to sell at any price higher or equal to 12.25. For
each price level, we can then determine the cumulative buy interest and sell interest.
The theoretical cross quantity at each price point is then simply the minimum of the
cumulative buy interest and the cumulative sell interest as shown in Table 1.4. The

Trading Fundamentals

17

Table 1.3: Ranked Order Book Submissions

Timestamp Seq. Number Buy Price Buy Quantity Sell Quantity Sell Price

9:13:12
9:10:39
9:07:42
9:12:21
9:09:54
9:12:48
9:08:18
9:11:13
9:07:33
9:18:27
9:06:22
9:09:55
9:19:13
9:11:46
9:15:37
9:14:51
9:05:17
9:15:02
9:16:42
9:19:54
9:17:11
9:02:36
9:06:59
9:01:21
9:10:04

12.35
12.34
12.33

12.28
12.25

12.23

12.21

12.19

12.17
12.17

12.16
12.15

5500
2000
500

3000
500

1200

1000

2750

4500
800

1120
12500

12.10

1500

17
12
7
15
9
16
8
13
6
23
4
10
24
14
20
18
3
19
21
25
22
2
5
1
11

10000
1930

12.33
12.30

4750

12.25

450
1750

12.23
12.22

3500

12.20

1200
1800

12.19
12.18

5000

12.16

1750
2500

12.12
12.11

3500

12.05

crossing price is determined as the price that would maximize the crossed quantity. In
our example, the opening price will be 12.19, and the opening quantity will be 15,750
shares.

The list of buy orders executed during the auction is shown in Table 1.5 and the
sell orders in Table 1.6. It is worth mentioning that the open auction tends to be
considered as a major price discovery mechanism given the fact that it occurs after a
period of market inactivity when market participants were unable to transact even if
they have information. All new information accumulated overnight will be reﬂected
in the ﬁrst print of the day, matching buying and selling interests.

As market participants with better information are more likely to be participating
in the open auction with more aggressive orders in order to extract liquidity (and,
as such, setting the price), the price discovery mechanism is often considered to be
quite volatile and more suited for short-term alpha investors. Similarly, the period
immediately following the open auction also tends to be much more volatile than the
rest of the day. As a result of which, most markets experience wider spreads while

18

Price

12.35
12.34
12.33
12.33
12.30
12.28
12.25
12.25
12.23
12.23
12.22
12.21
12.20
12.19∗
12.19
12.18
12.17
12.17
12.16
12.16
12.15
12.12
12.11
12.10
12.05

Algorithmic Trading and Quantitative Strategies

Table 1.4: Cumulative Order Book Quantities

Sequence
Number
17
12
7
15
9
16
8
13
6
23
4
10
24
14
20
18
3
19
21
25
22
2
5
1
11

Cumulative
Buy Quantity
5500
7500
8000
8000
8000
11000
11500
11500
12700
12700
12700
13700
13700
16450
16450
16450
20950
21750
21750
22870
35370
35370
35370
36870
36870

Sell
Quantity

Buy
Quantity
5500
2000
500

10000
1930

3000
500

1200

1000

2750

4500
800

1120
12500

1500

4750

450
1750

3500

1200
1800

5000

1750
2500

3500

Cumulative
Sell Quantity
38130
38130
38130
38130
28130
26200
26200
26200
21450
21450
21000
19250
19250
15750
15750
14550
12750
12750
12750
7750
7750
7750
6000
3500
3500

Quantity Crossed
at Price

5500
7500
8000
8000
8000
11000
11500
11500
12700
12700
12700
13700
13700
15750
15750
14550
12750
12750
12750
7750
7750
7750
6000
3500
3500

market makers try to protect themselves against information asymmetry by quoting
wider bids and oﬀers. The increased volatility and wider spreads might discourage
certain investors from participating in the market at the open auction and in the period
immediately following the open. While this appears to be reasonable from a price risk
perspective, it is worth mentioning that for many less liquid stocks (in particular small
and mid cap stocks), the open auction can be a signiﬁcant liquidity aggregation point
that even surpasses the close auction. In Australia for instance, the bottom 50% less
liquid stocks have more volume traded in the open auction than in the close auction.
Similarly in Japan, the less liquid stocks have more volume traded in the open auction,
but also in the afternoon open auction that follows the market lunch break.

From an execution standpoint, though, the usage of the open auction has to be con-
sidered carefully. While this represents a liquidity opportunity, the ﬁrst print of the
day can also have a signiﬁcant anchoring eﬀect on the stock price for the remainder
of the day. So, participating in the open should be considered in light of the liquidity

Trading Fundamentals

19

Table 1.5: Crossed Buy Orders

Price
12.35
12.34
12.33
12.28
12.25
12.23
12.21
12.19∗

Seq. Number Buy Qty
5500
2000
500
3000
500
1200
1000
2050

17
12
7
16
8
6
10
14
*Order number 14 was for 2750 shares but did not get fully executed
as the bid quantity up to 12.19 exceeded the oﬀered quantity at that
price. The balance of order 14 will then be posted as a limit order in
the continuous trading session

Table 1.6: Crossed Sell Orders

Price Seq. Number Sell Qty
12.19
12.18
12.16
12.12
12.11
12.05

1200
1800
5000
1750
2500
3500

20
18
21
2
5
11

demand of the order: Orders that are small enough can likely do without participating
in the open auction and the period that continues immediately following, while large
orders that try to extract signiﬁcant liquidity from the market might beneﬁt from par-
ticipating in the open auction. The market intraday momentum study by Gao, Han,
Li and Zhou (2018) [156] demonstrates how the ﬁrst half-hour return on the market,
as measured from the previous day’s market close predicts the last half hour return.

1.3.3 Continuous Trading

This refers to the main market phase between the auctions. During this market
session the state of the order book changes quite rapidly due to the multi-agent nature
of ﬁnancial markets and the prevalence of high frequency trading. Consequently, it
is important to understand the dynamics of the LOB before implementing trading
strategies. There exists quite a diversity of order types that are mostly relevant to the
continuous trading session, but the two most basic ones are: Limit Orders and Market
Orders, which we describe below.

A limit order has an associated side (Buy or Sell), a quantity and a price which
represent the highest (lowest) price the trader is willing to buy (sell). As previously
discussed, once a limit order is received by the exchange it is inserted in a data struc-
ture called a Limit Order Book (LOB) which contains two sub-structures, one per
side. Orders are inserted in this structure in price priority, higher prices for buys,

20

Algorithmic Trading and Quantitative Strategies

lower prices for sells, and for orders at the same price the orders are stored in the
order in which they were received. That is what is meant by price/time priority.21 If
the price of a newly arrived order overlaps with the best price available on the oppo-
site side, the order is executed either fully or up to the available quantity on the other
side. These orders are said to be “matched” and again this matching happens in price
and time priority meaning that the better prices (higher for buys, lower for sells) are
executed ﬁrst and orders that arrived beforehand at the same price level are executed
ﬁrst. Market orders on the other hand do not have a price associated with them and
will immediately execute against the other side and will match with more and more
aggressive prices until the full order is executed.

Orders on the buy side are called “bids” while those on the sell side are called
“asks.” The above events are illustrated in Figure 1.1 to Figure 1.3. When a market
(or marketable) order is submitted, it decreases the number of outstanding orders at
the opposite best price. For example, if a market bid order arrives, it will decrease
the number of outstanding asks at the best price. All unexecuted limit orders can
be canceled. When a cancellation occurs, it will decrease the number of outstanding
orders at the speciﬁed price level.

bids
asks

limit bid

e
m
u
l
o
v

60

50

40

30

20

10

0

73600

73800

74000

74200

74400

74600

74800

75000

price

Figure 1.1: Limit Order Book—Limit Bid.

21Note: Not all exchanges are matching orders following a price/time priority algorithm; a key char-
acteristic of the Futures market, for instance, is the existence of pro-rata markets for some ﬁxed income
contracts, where passive child orders receive ﬁlls from aggressive orders based on their size as a fraction
of the total passive posted quantity.

Trading Fundamentals

21

e
m
u
l
o
v

e
m
u
l
o
v

60

50

40

30

20

10

0

60

50

40

30

20

10

0

bids
asks

market bid

73600

73800

74000

74200

74400

74600

74800

75000

price

Figure 1.2: Limit Order Book—Marketable Bid.

bids
asks

ask cancellation

73600

73800

74000

74200

74400

74600

74800

75000

price

Figure 1.3: Limit Order Book—Ask Cancellation.

22

Algorithmic Trading and Quantitative Strategies

Limit orders make up a signiﬁcant percentage (70%) of stock market trading activ-
ity. The main advantage of a limit order is that there is no price risk associated to it,
that is, when the order is executed the limit price is the maximum (for a buy order) or
minimum (for a sell order) price that will be achieved. But if the limit order is not mar-
ketable, the execution is not guaranteed and the time to get an order executed depends
on various market factors. The trade-oﬀ between limit orders and marketable orders
depends on the investor’s need for immediate liquidity and the ﬁll probability of limit
orders. The limit price chosen (how deep in the order book is the order placed) as well
as the amount of liquidity ahead of the submitted order (how many shares will need
to trade before the order gets executed following, for instance, a price/time priority
order matching of the exchange) aﬀect both the order ﬁll probability and its expected
time to ﬁll. These two metrics are of particular relevance for execution algorithms
and will be studied in more depth later.

The execution of limit orders does aﬀect how the quotes are posted and are
updated. If the size of a market order exceeds the number of shares available at the
top of book, it is usually split and is executed at consecutive order book levels until
the order is ﬁlled. Market orders are usually restricted to be ﬁlled within a day and
orders placed after the markets close might be entered the next day.22

Order Types: The diversity of order types is a key component of continuous double
auction electronic markets. Order types allow participants to express precisely their
intentions with regards to their interaction with the market via the limit order book.
Over time, in an eﬀort to cater to sophisticated electronic traders, exchanges around
the world have raced to oﬀer ever more complex order types. Here, we will just pro-
vide a brief description of some, besides the market and limit orders that were already
mentioned:

• Peg: Specify a price level at which the order should be continuously and automati-
cally repriced. For instance, an order pegged to the bid price will be automatically
repriced as a higher price limit order, each time the market bid price ticks up. This
order type is particularly used for mid-point executions in non-displayed markets.
One would think that pegging order has the additional advantage of improving the
queue priority of the order when it is repriced since this process is done directly
by the exchange. That turns out not to be always true. Managing peg orders is the
responsibility of a separate component at the exchange and its interaction speed
with the order book is usually slower than that of ultra-low-latency operators.

• Iceberg: Limit order with a speciﬁed display quantity. In order to prevent infor-
mation leakage to other market participants, a trader desiring to buy or sell a large
quantity at a given price might elect to use an iceberg order with a small dis-
play size. For instance, for an order to buy 100,000 shares at $20 with a display
size of 2,000 shares. Only 2,000 shares would be displayed in the order book.
Once that quantity is executed, the order would automatically reload another

22Depending on the Time-in-Force selected.

Trading Fundamentals

23

2,000 shares at $20, and so on, until the full quantity is executed. Note that for
iceberg orders, only the visible quantity has time priority and once that quan-
tity has been executed the new tip of the iceberg will be placed at the back of
the queue.

• Hidden: While they are available to trade, these orders are not directly visible to

other market participants in the central limit order book.

• Stop: These orders are also not visible, but additionally are not immediately
entered in the limit order book. They only become active once a certain price
(known as the Stop Price) is reached or passed. They then enter the order book
as either limit or market order depending on the user setup.

• Trailing Stop: These orders function like stop orders, but the stop price is set

dynamic rather than static (for instance: −3% from previous close).

• All-or-None: Speciﬁcally, request a full execution of the order. If the order is for
1500 shares but only 1000 are being oﬀered, it will not be executed until the full
quantity is available.

• On-Open: Speciﬁcally, request an execution at the open price. It can be limit-on-

open or market-on-open.

• On-Close: Speciﬁcally, request an execution at the close price. It can be limit-

on-close or market-on-close.

• Imbalance Only: Provide liquidity intended to oﬀset on-open/on-close order/im-
balances during the opening/closing cross. These generally are limit orders.

• D-Quote: Special order type on the NYSE mainly used during the close auction

period.

• Funari: Special order type on the Tokyo Stock Exchange which allows limit
orders placed in the book during the continuous session to automatically enter
the closing auction as market orders.

As described above, there exists a wide variety of orders types oﬀered by diﬀerent

exchanges to facilitate various types of trading activities.

Validity of Instructions: In addition to conditions on price, it is possible to add con-
ditions on the life duration of the order known as Time-in-Force (TIF). The most
common types of TIF instructions include Day orders which are valid for the full
duration of the trading session, Extended Day orders allow trading in extended hours,
and Good-Till-Cancel (GTC) orders will be placed again on the exchange the next
day with similar instructions if they were not completely ﬁlled. More sophisticated
market participants aiming at achieving greater control over their executions tend to

24

Algorithmic Trading and Quantitative Strategies

also favor Immediate-or-Cancel (IOC) and Fill-or-Kill (FOK) Time-in-Force instruc-
tions. An IOC order will get immediately canceled back to the sender after reaching
the matching engine if it does not get an immediate ﬁll, and in case of a partial ﬁll, the
unﬁlled portion will be canceled, thus preventing it from creating a new price level in
the order book. In a Fill-or-Kill scenario, the order gets either ﬁlled in its entirety or
does not get ﬁlled at all. This instruction is particularly popular with high frequency
market makers and arbitrageurs for which partial ﬁlls might result in unwanted leg-
ging risk as discussed in Chapter 5 on pairs trading.

Finally, it is worth mentioning that some exchanges as well as alternative venues
oﬀer the ability of specifying minimum ﬁll sizes. This means that a limit order which
might be eligible for a ﬁll due to an incoming order at the same price level, only
receives a ﬁll if the incoming order is larger than a pre-speciﬁed number of shares
or notional value. This type of instruction is used by market participants as a way of
minimizing the number of small ﬁlls which carry the risk of excessive information
dissemination. This happens, in particular, in dark pools, where they can be used to
detect the presence of larger limit orders that would be otherwise not visible to market
participants.

1.3.4 The Closing Auction

The Closing Auction tends to be the most popular call auction for a variety of rea-
sons. First, it is the last opportunity (unless one engages in the risky practice of oﬀ-
hours trading) for market participants to transact in a relatively liquid environment,
before being exposed to the overnight period (when new information accumulates,
but trades cannot easily take place). Second, with the increase in passive investment
strategies, providing investors with replication of a predetermined benchmark index,
the closing auction has become a particularly relevant price setting event. For most
passive funds, the net asset value (NAV) is based on close prices of the underly-
ing assets. For those reasons, the closing auction has become extremely important to
many investors. From an execution standpoint, it is a major liquidity event that must
be handled carefully.

The mechanics of the closing auction are in most part similar to the ones described
above for the open auction. The major diﬀerences across countries (and sometimes
across exchanges within a country) are in the order submission times. Some coun-
tries, such as the US, have order submissions start and end before the continuous
session is over (see Table 1.7 and Table 1.8), while some other markets have two
non-overlapping continuous and close order submission sessions.

Table 1.7: Nasdaq Closing Cross

3:50 p.m. EST Cutoﬀ for amend/cancel of MOC/LOC orders
3:55 p.m. EST Dissemination of imbalance information begins
3:55 p.m. EST Cutoﬀ for entry of MOC/LOC orders
3:58 p.m. EST Freeze period - Late LOC orders cannot be added

OI orders oﬀsetting the imbalance are still accepted

4:00 p.m. EST The Closing Cross occurs

Trading Fundamentals

25

Table 1.8: NYSE Closing Cross

3:50 p.m. EST

Cutoﬀ for MOC/LOC order entry and modiﬁcations
Dissemination of imbalance information begins
Closing Oﬀset orders can be entered until 4:00 p.m.
Dissemination of d-quote imbalance information
Cutoﬀ for MOC/LOC cancellation for legitimate error

3:55 p.m. EST
3:58 p.m. EST
3:59:50 p.m. EST Cutoﬀ for d-Quote order entry and modiﬁcation
The Closing Auction starts
4:00 p.m. EST
DMM can automatically process auctions not yet complete
4:02 p.m. EST

Table 1.9: London Stock Exchange Sessions Times

Pre-Trading
Opening Auction Call
Regular Trading

7:00-7:50 a.m. GMT
7:50-8:00* a.m. GMT
8:00-12:00 p.m. GMT
12:00-12:02* p.m. GMT Periodic Call Auction
12:02-4:30 p.m. GMT
4:30-4:35* p.m. GMT
4:35-4:40 p.m. GMT
4:40-5:15 p.m. GMT

Regular Trading
Closing Auction Call
Closing Price Crossing
Post-Close Trading

*Each auction end time is subject to a random 30 second uncross period.
An additional intraday auction call takes place every 3rd Friday of each month for stocks under-
lying FTSE 100 index options, and the 3rd Friday of every quarter for stocks underlying FTSE
100/250 Index Futures to determine the EDSP (Exchange Delivery Settlement Price). The set-
tlement price is determined as the index value derived from the individual constituents intraday
auction taking place between 10 a.m. and 10:15 a.m. and during which the electronic continu-
ous trading is suspended. Using a call auction ensures the settlement price for these contracts
is more representative of a fair market price.

An aspect of the NYSE is the presence of ﬂoor brokers operating in an agency
capacity for their customers. They play a particular role during the close auction
thanks to their ability to handle discretionary electronic quote orders (known as “d-
Quote” orders) that oﬀer more ﬂexibility than traditional market-on-close (MOC) and
limit-on-close (LOC) orders. The main advantage of d-Quote orders is their ability
to bypass the 3:45 p.m. cutoﬀ and be submitted or canceled until 3:59:50 p.m. This
allows large institutional investors to remain in control of their orders almost until the
end of the continuous session, by delaying the decision of how much to allocate to
the auction. They can therefore react to larger volume opportunities based on the pub-
lished imbalance. They can also minimize the information leakage by not being part
of the early published imbalance while there is still signiﬁcant time in the continuous
session for other participants to drive the price away. Since there is no restriction on
the side of d-Quote orders submission, it is possible to see the total imbalance sign ﬂip
once the d-Quotes are added to the publication at 3:55 p.m., creating opportunities

26

Algorithmic Trading and Quantitative Strategies

for other market participants to adjust their own close trading via d-Quotes or change
their positioning in the continuous session.

1.4 Taxonomy of Data Used in Algorithmic Trading

Running a successful trading operation requires availability of diﬀerent data sets.
Data availability, storage, management, and cleaning are some of the most important
aspects of a functioning trading business and the core of any research environment.
The amount of data and the complexity of maintaining such a “Data Lake” can be
daunting and very few excel at this aspect. In this section, we will review the most
important data sets and their role in algorithmic trading research.

1.4.1 Reference Data

While often overlooked, or merely considered as an afterthought in the develop-
ment of a research platform,23 reliable reference data is the key foundation of a robust
quantitative strategy development. The experienced practitioner may want to skip this
section, however we encourage the neophyte to read through the tedious details to get
a better grasp of the complexity at hand.

• Trading Universe: The ﬁrst problem for the functioning of a trading operation
is knowing what instruments will be required to be traded on a particular day.
The trading universe is an evolving entity that changes daily to incorporate new
listings (IPOs), de-listings, etc. To be able to just trade new instruments, there are
several pieces of information that are required to have in multiple systems. Market
Data must be made available, and some static data needs to be set or guessed to
work with existing controls, parameters for the various analytics need to be made
available or sensibly defaulted. For research, in particular quantitative strategies,
knowing when a particular stock no longer trades is important to avoid issues like
survivor bias.

• Symbology Mapping: ISIN, SEDOL, RIC, Bloomberg Tickers, . . . Quantitative
strategies often leverage data from a variety of sources. Diﬀerent providers key
their data with diﬀerent instrument identiﬁers depending on asset class or regional
conventions, or sometimes use their own proprietary identiﬁers (e.g., Reuters
Identiﬁcation Code—RIC, Bloomberg Ticker). Therefore, symbology mapping
is the ﬁrst step in any data merging exercise. Such data is not static. One of the
symbols can change on a given day and others remain unchanged for some time,
complicating historical data merges.

23More details on research platforms are presented in Chapter 12.

Trading Fundamentals

27

It is important to note that such mapping needs to persist as point-in-time data
and allow for historical “as of date” usage, requiring the implementation of a
bi-temporal data structure. Over the course of time, some instruments undergo
) not necessarily
ticker changes (for example, from ABC to DEF on a later 𝑇0
without any particular change on the underlying asset. In such cases, market data
recorded day-by-day in a trade database will change from being keyed on ABC
to being keyed on DEF after the ticker change date 𝑇0
. This has implications for
practitioners working on data sets to build and backtest quantitative strategies.
The symbology mapping should allow for both backward and forward handling
of the changes.

For instance, in the simple example mentioned below, in order to eﬃciently back-
, a robust mapping is needed, so
test strategies over a period of time spanning 𝑇0
that it will allow to seamlessly query the data for the underlying asset in a variety
of scenarios such as:

– Signal generation: 30-day backward close time series as of date 𝑇 < 𝑇0
select close from data where date in [T-30, T], sym = ABC

:

– Signal generation: 30-day backward close time series as of date 𝑇 = 𝑇0+10:
select close from data where date in [T0-20, T0+10], sym = DEF
– Position holding: 30-day forward close time series as of date 𝑇 = 𝑇0 − 10:
select close from data where date in [T0-10, T0+20], sym = ABC

• Ticker Changes: For comparable reasons as the ones described above in the
Symbology Mapping section, one needs to maintain a historical table of ticker
changes allowing to seamlessly go up and down time series data.

• Corporate Actions Calendars: This category contains stock and cash dividends
(both announcement date and execution date), stock splits, reverse splits, rights
oﬀer, mergers and acquisitions, spin oﬀ, free ﬂoat or shares outstanding adjust-
ments, quotation suspension, etc.

Corporate actions impact the continuity of price and volume time series and, as
such, must be recorded in order to produce adjusted time series. The most com-
mon events are dividend distributions. On the day the dividend is paid, the cor-
responding amount is removed from the stock price, creating a jump in the price
time series. The announcement date might also coincide with the stock experi-
encing more volatility as investors react to the news. Consequently, recording
these events proves to be valuable in the design of quantitative strategies as one
can assess the eﬀect of dividends announcement or payment on performance, and
decide to either not hold a security that has an upcoming dividend announcement
or, conversely, to build strategies that look to beneﬁt from the added volatility.

Another type of corporate events generating discontinuity in historical time series
are stock splits or reverse splits and right oﬀers. When the price of a stock
becomes too low or too high, a company may seek to split it to bring the price

28

Algorithmic Trading and Quantitative Strategies

back to a level that is more conducive to liquid trading on exchanges.24 When a
stock experiences a 2:1 split, everything else being equal, its price will be halved
and hence, its volume will double. In order to prevent the time series from show-
ing a discontinuity, all historical data will then need to be adjusted backward to
reﬂect the split.

Mergers & Acquisitions and Spin-oﬀs are also regular events in the lifecycle of
corporations. Their history needs to be recorded in order to account for the result-
ing changes in valuation that might aﬀect a given ticker(s). These situations can
also be exploited by trading strategies known as Merger Arbitrage.

Stock quotations can be suspended as a cooling mechanism (often at the request
of the underlying company) to prevent excess price volatility when signiﬁcant
information is about to be released to the market. Depending on the circumstance,
the suspension can be temporary and intraday, or can last for extended periods of
time if the market place allows it.25 Suspensions result in gaps in data and are
worth keeping track of, as they can impact strategies in backtesting (inability to
enter or exit a position, uncertainty in the pricing of composite assets if a given
stock has a signiﬁcant weight in ETFs or Indexes, etc.). Some markets will also
suspend trading if the price swings more than a predeﬁned amount (limit up / limit
down situations), either for a period of time or for the remainder of the trading
session.

• Static Data: Country, sector, primary exchange, currency and quote factor. Static
data is also relevant for the development of quantitative trading strategies. In par-
ticular, country, currency and sector are useful to group instruments based on
their fundamental similarities. A well known example is the usage of sectors to
group stocks in order to create pairs trading strategies. It is worth noting that there
exist diﬀerent types of sector classiﬁcations (e.g., GICS®from S&P, ICB®from
FTSE) oﬀering several levels of granularity,26 and that diﬀerent classiﬁcations
might be better suited to diﬀerent asset classes or countries. The constituents of
the Japanese index TOPIX, for instance, are classiﬁed into 33 sectors that are
thought to better reﬂect the fundamental structure of the Japanese economy and
the existence of large diversiﬁed conglomerates.

Maintaining a table of the quotation currency per instrument is also necessary
in order to aggregate positions at a portfolio level. Some exchanges allow the
quotation of prices in currencies diﬀerent from the one of the country in which

24A very low price creates trading frictions as the minimum price increment might represent a large cost
relative to the stock price. A very high price might also deter retail investors from investing into a security
as it requires them to deploy too much capital per unit.

25For instance, it was the case for a large number of companies in China in 2016.

26The Global Industry Classiﬁcation Standard (GICS®) structure consists of 11 sectors, 24 industry
groups, 68 industries and 157 sub-industries. An example of this hierarchical structure would be: Industrials
/ Capital Goods / Machinery / Agricultural & Farm Machinery.

Trading Fundamentals

29

the exchange is located.27 Additionally, thus, the Quote Factor associated with
the quotation currency data needs to be stored. To account for the wide range
of currency values and preserve pricing precision, market data providers may
publish FX rates with a factor of 100 or 1000. Hence, to convert prices to USD one
needs to multiply by the quote factor: USD price = local price ⋅ fx ⋅ quote factor.
Similarly, some exchanges quote prices in cents, and the associated quotation
currency is reﬂected with a small cap letter: GBP/GBp, ZAR/ZAr, ILS/ILs, etc.

• Exchange Speciﬁc Data: Despite the electroniﬁcation of markets, individual
exchanges present a variety of diﬀerences that need to be accounted for when
designing trading strategies. The ﬁrst group of information concerns the hours
and dates of operation:

– Holiday Calendar: As not all exchanges are closed on the same day, and
trading days they are oﬀ do not always fully follow the country’s public holi-
days, it is valuable to record them, in particular in the international context.
Strategies trading simultaneously in several markets and leveraging their
correlation, may not perform as expected if one of the markets is closed
while others are open. Similarly, execution strategies in one market might
be impacted by the absence of trading in another market (for instance, Euro-
pean equity markets volume tends to be 30% to 40% lower during US market
holidays).

– Exchange Sessions Hours: These seemingly trivial data points can get
quite complex on a global scale. What are the diﬀerent available sessions
(Pre-Market session, Continuous core session, After-Hour session, etc.)?
What are the auction times as well as their respective cutoﬀ times for order
submission? Is there a lunch break restricting intraday trading? And if so,
are there auctions before and after the lunch break? The trading sessions in
the futures markets can also be quite complex with multiple phases, breaks,
as well as oﬃcial settlement times that may diﬀer from the closing time and
have an eﬀect on liquidity.
In Indonesia, for instance, markets have diﬀerent trading hours on Fridays.
Monday through Thursday the Indonesia Stock Exchange (IDX) is open
from 9:00 a.m. to 12:00 p.m., and then, from 1:30 p.m. to 4:00 p.m. On
Fridays, however, the lunch break is one hour longer and stretches from
11:30 a.m. to 2:00 p.m. This weekday eﬀect is particularly important to
consider when building volume proﬁles as discussed in Chapter 5.
Along with local times of operation, it is necessary to consider eventual
Daylight Saving Time (DST) adjustments that might aﬀect the relative trad-
ing hours of diﬀerent markets (some countries do not have DST adjustment
at all, while for countries that do have one, the dates at which it applies are

27For example, Jardine Matheson Holdings quotes in USD on the Singapore exchange while most of the

other securities quote in Singapore Dollars.

30

Algorithmic Trading and Quantitative Strategies

not always coordinated). Usually, US DST starts about two weeks prior to
its start in Europe, bringing the time diﬀerence between New York and Lon-
don to four hours instead of ﬁve hours. This results in the volume spike in
European equities associated to the US market open being one hour earlier,
requiring adjustment of volume proﬁles used for trading executions.
Some exchanges may also adjust the length of trading hours during the
course of the year. In Brazil for instance, the Bovespa continuous trading
hours are 10:00 a.m. to 5:55 p.m. from November to March, but an hour
shorter (10:00 a.m. to 4:55 p.m.) from April to October to be more consis-
tent with US market hours. Finally, within one country there might also exist
diﬀerent trading hours by venues as it is the case in Japan where the Nagoya
Stock Exchange closes 30 minutes after the major Tokyo Stock Exchange.
– Disrupted Days: Exchange outages or trading disruptions, as well as market
data issues, need to be recorded so they can be ﬁltered out when building
or testing strategies as the diﬀerence in liquidity patterns or the lack of data
quality may likely impact the overall outcome.

Additionally, exchanges also have speciﬁc rules governing the mechanics of trad-
ing, such as:

– Tick Size: The minimum eligible price increment. This can vary by instru-
ment, but also change dynamically as a function of the price of the instru-
ment (e.g., stocks under $1 can quote in increments of $0.0001, while above
that price the minimum quote increment is $0.01).

– Trade and Quote Lots: Similar to tick sizes, certain exchanges restrict the

minimum size increment for quotes or trades.

– Limit-Up and Limit-Down Constraints: A number of exchanges restrict
the maximum daily ﬂuctuations of securities. Usually, when securities reach
these thresholds, they either pause trading or can only be traded at a better
price than the limit-up limit-down threshold.

– Short Sell Restrictions: Some markets also impose execution level con-
straints on short sells (on top of potential locate requirements). For instance,
while a long sell order can trade at any price, some exchanges restrict short
sells not to trade at a price worse than the last price or not to create a new
quote that would be lower than the lowest prevailing quote. These consider-
ations are particularly important to keep in mind for researchers developing
Long-Short strategies as this impacts the ability to source liquidity.

Because these values and their potential activation threshold can vary over time,
one needs to maintain historic values as well, in order to run realistic historical
backtests.

Trading Fundamentals

31

• Market Data Condition Codes: With ever-growing complexity in market
microstructure, the dissemination of market data has grown complex as well.
While it is possible to store daily data as a single entry per day and per instru-
ment, investors building intraday strategies likely need tick by tick data of all the
events occurring in the market place. To help classify these events, exchanges
and market data providers attribute so-called condition codes to the trades and
quotes they publish. These condition codes vary per exchange and per asset class,
and each market event can be attributed to several codes at once. So, to aggregate
intraday market data properly and eﬃciently, and decide which events to keep and
which ones to exclude, it is necessary to build a mapping table of these condition
codes and what they mean: Auction trade, lit or dark trade, canceled or corrected
trade, regular trade, oﬀ-exchange trade reporting, block-size trade, trade originat-
ing from a multi-leg order such as an option spread trade, etc.

For instance, in order to assess accessible liquidity for a trading algorithm, trades
that are published for reporting purposes (e.g., negotiated transactions that hap-
pened oﬀ-exchange) must be excluded. These trades should also not be used to
update some of the aggregated daily data used in the construction of trading strate-
gies (daily volume, high, low, . . . ). Execution algorithms also extensively leverage
the distribution of intraday liquidity metrics to gauge their own participation in
auctions and continuous sessions, or in lit versus dark venues, therefore requiring
a precise classiﬁcation of intraday market data.

• Special Day Calendars: Over the course of a year, some days present certain
distinct liquidity characteristics that need to be accounted for in both execution
strategies and in the alpha generation process. The diversity of events across mar-
kets and asset classes can be quite challenging to handle. Among the irregular
events that aﬀect liquidity in equity markets that need to be accounted for, we
can mention the following non-exhaustive list for illustration purposes only: Half
trading days preceding Christmas and following Thanksgiving in the US or on the
Ramadan eve in Turkey, Taiwanese market opening on the weekend to make up
for lost trading days during holiday periods, Korean market changing its trading
hours on the day of the nationwide university entrance exam, Brazilian market
opening late on the day following the Carnival, etc.

There are also special days that are more regular and easier to handle. The last
trading days of the months and quarters, for instance, tend to have additional trad-
ing activity as investors rebalance their portfolios. Similarly, options and futures
expiry days (quarterly/monthly expiry, ‘Triple Witching’28 in the US, Special
Quotations in Japan, etc.) tend to experience excess trading volume and diﬀerent
intraday patterns resulting from hedging activity and portfolio adjustments. Con-
sequently, they need to be handled separately, in particular, when modeling trad-

28Triple witching days happen four times a year on the third Friday of March, June, September and
December. On these days, the contracts for stock index futures, stock index options and stock options
expire concurrently.

32

Algorithmic Trading and Quantitative Strategies

ing volume. As most execution strategies make use of relatively short interval vol-
ume metrics (e.g., 30-day or 60-day ADV), one single data point can impact the
overall level inferred. Similarly autoregressive models of low order may underper-
form both on special days and on the days following them. As a result, modelers
often remove special days and model normal days ﬁrst. Then, special days are
modeled separately, either independently or using the normal days as a baseline.

In order to reﬂect changes in the market and remain consistent with index inclu-
sion rules, most indices need to undergo regular updates of the constituents and
their respective weights. The signiﬁcant indices do so at regular intervals (annu-
ally, semi-annually or quarterly) on a pre-announced date. At the close of business
of that day, some stocks might be added to the index while others are removed,
or the weight of each stock in the index might be increased or decreased. These
events, known as index rebalances, are particularly relevant to passive investors
who are tracking the index. In order to minimize the tracking error to the bench-
mark index, investors need to also adjust their holdings accordingly. Additionally,
as most funds are benchmarked at the close price, there is an incentive for fund
managers to try to rebalance their holdings at a price as close as possible to the
oﬃcial close price, on the day the index rebalance becomes eﬀective. As a result,
on these days, intraday volume distribution is signiﬁcantly skewed toward the end
of day and requires some adjustment in the execution strategies.

• Futures-Speciﬁc Reference Data: Futures contracts present particular charac-
teristics requiring additional reference data to be collected. One of the core dif-
ferences of futures contracts compared to regular stocks is the fact that instru-
ments have an expiry date, after which the instrument ceases to exist. For the
purpose of backtesting strategies, it is necessary to know which contract was live
at any point in time through the use of an expiry calendar, but also which con-
tract was the most liquid. For instance, equity index futures tend to be the most
liquid, for the ﬁrst contract available (also known as front month), while energy
futures such as oil tend to be more liquid for the second contract. While this may
appear to be trivial, when building a trading strategy and modeling price series,
it is particularly important to know which contract carries the most signiﬁcant
price formation characteristics and what is the true liquidity available in order to
properly estimate the market impact.

The task of implementing a futures expiry calendar is further complicated by
the fact there is no real standardized frequency that applies across markets. For
instance, European equity index futures tend to expire monthly, while US index
futures expire quarterly. Some contracts even follow an irregular cycle through
the course of the year as it is the case for grain futures (e.g., wheat) that were

Trading Fundamentals

33

originally created for hedging purposes and as a result have expiry months that
follow the crop cycle.29

The fact futures contracts expire on a regular basis has further implications in
terms of liquidity. If investors holding these contracts want to maintain their
exposure for longer than the lifespan of the contract, they need to roll over their
positions onto the next contract which might have a noticeably diﬀerent liquidity
level. For instance, the front contract of the S&P 500 (e.g., ESH8) trades roughly
1.5 million contracts per day in the weeks preceding expiry while the next month
contract (ESM8) only trades about 30,000 contracts per day. However, this liq-
uidity relationship will invert in the few days leading to the expiry of the front
contract as most investors roll their positions, and the most liquid contract will
become the back month contract (see Figure 1.4). As a result, when computing
rolling-window metrics (such as average daily volume for instance), it is neces-
sary to account for potential roll dates that may have happened during the time
span. In the example above, a simple 60-day average daily volume on ESM8 taken
in early April 2018 would capture a large number of days with very low volume
(January-March) owing to the fact the most liquid contract at the time was the
ESH8 contract, and would not accurately represent the volume activity of the
S&P 500 futures contract. A more appropriate average volume metric to be used
as a forward looking value for execution purposes would blend the volume time
series of ESH8 prior to the roll date, and ESM8 after the roll date.30 Additionally,
in order to eﬃciently merge futures positions with other assets in an investment
strategy, reference data relative to the quotation of these contracts, contract size
(translation between the quotation points and the actual monetary value), cur-
rency, etc., must be stored.

Finally, futures markets are characterized by the existence of diﬀerent market
phases during the day, with signiﬁcantly diﬀerent liquidity characteristics. For
instance, equity index futures are much more liquid during the hours when the
corresponding equities markets are open. However, one can trade during the
overnight session if they want to. The overnight session being much less liquid,
the expected execution cost tends to be higher, and as such, the various market
data metrics (volume proﬁle, average spread, average bid-ask sizes, ...) should be
computed separately for each market phase, which requires maintaining a table
of the start and end times of each session for each contract.

29US Wheat Futures expire in March, May, July, September and December.

30It is worth noting that diﬀerent contracts ‘roll’ at diﬀerent speeds. While for monthly expiry contracts
it is possible to see most of the open interest switch from the front month contract to the back month on the
day prior to the expiry. For quarterly contracts it is not uncommon to see the roll happen over the course
of a week or more, and the front month liquidity vanish several days ahead of the actual expiry. Careful
modeling is recommended on a case by case basis.

34

Algorithmic Trading and Quantitative Strategies

Figure 1.4: Futures Volume Rolling.

• Options-Speciﬁc Reference Data (Options Chain): Similar to futures contracts,
options contracts present a certain number of speciﬁcities for which reference data
need to be collected. On top of the similar feature of having a particular expiry
date, options contracts are also deﬁned by their strike price. The combination of
expiries and strikes is known as the option chain for a given underlier. The ability
to map equity tickers to option tickers and their respective strike and expiry dates
allows for the design of more complex investment and hedging strategies. For
instance, distance to strike, change in open interest of puts and calls, etc., can all
be used as signals for the underlying security price. For an interesting article on
how deviations in put-call parity contains information about future equity returns,
refer to Cremers and Weinbaum (2010) [95].

• Market-Moving News Releases: Macro-economic announcements are known
for their ability to move markets substantially. Consequently, it is necessary to
maintain a calendar of dates and times of their occurrences in order to assess
their impact on strategies and decide how best to react to them. The most com-
mon ones are central banks’ announcements or meeting minutes releases about
the major economies (FED/FOMC, ECB, BOE, BOJ, SNB), Non-Farm Payrolls,
Purchasing Managers’ Index, Manufacturing Index, Crude Oil Inventories, etc.
While these news releases impact the broad market or some sectors, there are
also stock speciﬁc releases that need to be tracked: Earning calendars, special-
ized sector events such as FDA results for the healthcare and biotech sectors, etc.

• Related Tickers: There is a wide range of tickers that are related to each other,
often because they fundamentally represent the same underlying asset. Main-
taining a proper reference allows to eﬃciently exploit opportunities in the mar-
ket. Some non-exhaustive examples include: Primary tickers to composite tickers
mapping (for markets with fragmented liquidity), dual listed/fungible securities

Trading Fundamentals

35

in US and Canada, American Depository Receipt (ADR) or Global Depository
Receipt (GDR), local and foreign boards in Thailand, etc.

• Composite Assets: Some instruments represent several underlying assets (ETFs,
Indexes, Mutual Funds, . . . ). Their rise in popularity as investments over time
makes them relevant for quantitative strategies. They can be used as eﬃcient vehi-
cles to achieve desired exposures (sector and country ETFs, thematic factor ETFs,
. . . ), or as cheap hedging instruments, and they can provide arbitrage opportuni-
ties when they deviate from their Net Asset Value (NAV). In order to be leveraged
in quantitative strategies, one needs to maintain a variety of information such as a
time series of their constituents and the value of any cash component, the divisor
used to translate the NAV into the quoted price, the constituent weights.

• Latency Tables: This last type of data would only be of interest for developing
strategies and for research in the higher frequency trading space. For these, it
might be relevant to know the distribution of latency between diﬀerent data cen-
ters as they can be used for more eﬃcient order routing as well as reordering data
that may have been recorded in diﬀerent locations.

While the above discussion provides a non-exhaustive list of issues on the refer-
ence data available to build a quantitative research platform, they highlight the chal-
lenges that must be taken into account when designing and implementing algorithmic
trading strategies. Once in place, a proper set of reference data will allow the quan-
titative trader to systematically harness the actual content of various types of data
(described later) without being caught oﬀ-guard by the minutiae of trading.

1.4.2 Market Data

While historically a large swath of modeling for developing strategies was carried
out on daily or minute bar data sets, the past ﬁfteen years have seen a signiﬁcant
rise in the usage of raw market data in an attempt to extract as much information
as possible, and act on it before the opportunity (or market ineﬃciency) dissipates.
Market data, itself, comes in various levels of granularity (and price!) and can be
subscribed to either directly from exchanges (known as “direct feeds”) or from data
vendors aggregating and distributing it. The level of detail of the feed subscribed to
is generally described as Level I, Level II or Level III market data.

Level I Data: Trade and BBO Quotes: Level I market data is the most basic form
of tick-by-tick data. Historically, the Level I data feed would only refer to trade infor-
mation (Price, Size, Time of each trade reported to the tape) but has grown over time
into a term generally accepted to mean both trades and top of book quotes. While the
trade feed updates with each trade printed on the tape, the quote feed tends to update
much more frequently each time liquidity is added to, or removed from, the top of
book (a rough estimate in liquid markets is the quotes present more updates than the
trades by about an order of magnitude).

36

Algorithmic Trading and Quantitative Strategies

In order to build strategies, the timing of each event must be as precise as possi-
ble. While the time reported for trade and quotes is the matching engine time at the
exchange, most databases also store a reception or record time to reﬂect the potential
latency between the trade event and one being aware that it did happen and being able
to start making decisions on it. Accounting for this real-world latency is a necessary
step for researchers building strategies on raw market data for which opportunities
may be very short lived and may be impossible to exploit by participants who are not
fast enough.

The Level I data is enough to reconstruct the Best Bid and Oﬀer (BBO) of the
market. However, in fragmented markets, it is also useful to obtain an aggregated
consolidated view of all available liquidity at a given price level across all exchanges.
Market data aggregators usually provide this functionality for users who do not wish
to reconstruct the full order book themselves.

Finally, it is worth noting that even Level I data contains signiﬁcant additional
information in the form of trade status (canceled, reported late, etc.) and, trade and
quote qualiﬁers. These qualiﬁers provide granular details such as whether a trade was
an odd lot, a normal trade, an auction trade, an Intermarket Sweep, an average price
reporting, on which exchange it took place, etc. These details can be used to better
analyze the sequence of events and decide if a given print should be used to update
the last price and total volume traded at that point in time or not. For instance, if not
processed appropriately, a trade reported to the tape out of sequence could result in a
large jump in price because the market may have since moved, and consequently the
trade status is an indication that the print should not be utilized as it is.

Capturing raw market data, whether to build a research database or to process
it in real-time to make trading decisions, requires signiﬁcant investments and expert
knowledge to handle all the inherent complexity. Hence, one should carefully con-
sider the trade-oﬀ between the expected value that can be extracted from the extra
granularity, and the additional overhead compared to simpler solutions such as using
binned data.

Level II Data: Market Depth: Level II market data contains the same information as
Level I, but with the addition of quote depth data. The quote feed displays all lit limit
order book updates (price changes, addition or removal of shares quoted) at any level
in the book, and for all of the lit venues in fragmented markets. Given the volume of
data generated and the decreasing actionable value of quote updates as their distance
to top of book increases, some users limit themselves to the top ﬁve or ten levels of
the order book when they collect and/or process the data.

Level III Data: Full Order View: Level III market data—also known as message
data—provides the most granular view of the activity in the limit order book. Each
order arriving is attributed a unique ID, which allows for its tracking over time, and
is precisely identiﬁed when it is executed, canceled or amended. Similar to Level II,
the data set contains intraday depth of book activity for all securities in an exchange.
Once all messages from diﬀerent exchanges are consolidated into one single data set
ordered by timestamp, it is possible to build a full (with national depth) book at any

Trading Fundamentals

37

moment intraday. For illustration purposes, we take the example of US Level III data
and provide a short description below in Table 1.10.

Table 1.10: Level III Data

Variable:

Description

Timestamp: Number of milliseconds after the midnight.

Ticker:

Equity symbol (up to 8 characters)

Order:

Unique order ID.

T:

Message type. Allowed values:

• “B”—Add buy order

• “S”—Add sell order

• “E”—Execute outstanding order in part

• “C”—Cancel outstanding order in part

• “F”—Execute outstanding order in full

• “D”—Delete outstanding order in full

• “X”—Bulk volume for the cross event

• “T”—Execute non-displayed order

Shares:

Price:

Order quantity for the “B,” “S,” “E,” “X,” “C,” “T” messages. Zero
for “F” and “D” messages.

Order price, available for the “B,” “S,” “X” and “T” messages.
Zero for cancellations and executions. The last 4 digits are decimal
digits. The decimal portion is padded on the right with zeros. The
decimal point is implied by position; it does not appear inside the
price ﬁeld. Divide by 10000 to convert into currency value.

MPID:

Market Participant ID associated with the transaction (4 characters)

MCID:

Market Center Code (originating exchange—1 character)

While the display and issuance of a new ID to the modiﬁed order varies from

exchange to exchange, a few special types of orders are worth mentioning:

1. Order subject to price sliding: The execution price could be one cent worse than the
display price at NASDAQ; it is ranked at the locking price as a hidden order, and

38

Algorithmic Trading and Quantitative Strategies

is displayed at the price, one minimum price variation (normally 1 cent) inferior
to the locking price. New order ID will be used if the order is replaced as a display
order. At other exchanges the old order ID will be used.

2. Pegged order: Based on NBBO, not routable, new timestamp given upon re-

pricing; display rules vary over exchanges.

3. Mid-point peg order: Non-displayed, can result in half-penny execution.

4. Reserve order: Displayed size is ranked as a displayed limit order and the reserve
size is behind non-displayed orders and pegged orders in priority. The minimum
display quantity is 100 and this amount is replenished from the reserve size when
it falls below 100 shares. A new timestamp is created and the displayed size will
be re-ranked upon replenishment.

5. Discretionary order: Displayed at one price while passively trading at a more
aggressive discretionary price. The order becomes active when shares are avail-
able within the discretionary price range. The order is ranked last in priority. The
execution price could be worse than the display price.

6. Intermarket sweep order: Order that can be executed without the need for checking

the prevailing NBBO.

The richness of the data set allows sophisticated players, such as market makers,
to know not only the depth of book at a given price and the depth proﬁle of the book
on both sides, but more importantly the relative position of an order from the top
position of the side (buy or sell). Among the most common granular microstructure
behaviors studied with Level III data, we can mention:

• The pattern of inter-arrival times of various events.

• Arrival and cancellation rates as a function of distance from nearest touch price.

• Arrival and cancellation rates as a function of other available information, such

as in the queue on either side of the book, order book imbalance, etc.

Once modeled, these behaviors can, in turn, be employed to design more sophisticated
strategies by focusing on trading related questions:

• What is the impact of market order on the limit order book?

• What are the chances for a limit order to move up the queue from a given entry

position?

• What is the probability of earning the spread?

• What is the expected direction of the price movement over a short horizon?

Trading Fundamentals

39

As illustrated by the diversity and granularity of information available to traders,
markets mechanics have grown ever more complex over time, and a great deal of
peculiarities (in particular at the reference data level) need to be accounted for in
order to design robust and well performing strategies. The proverbial devil is always
in the details when it comes to quantitative trading but while it might be tempting
to go straight to the most granular source of data, in practice—with the exception
of truly high frequency strategies—most practitioners build their algorithmic trading
strategies relying essentially on binned data. This simpliﬁes the data collection and
handling processes and also greatly reduces the dimension of the data sets so one can
focus more on modeling rather than data wrangling. Most of the time series models
and techniques described in subsequent chapters are suited to daily or binned data.

1.4.3 Market Data Derived Statistics

Most quantitative strategies, even the higher frequency strategies that leverage
more and more granular data also leverage derived statistics from the binned data,
such as daily data. Here we give a list of the most common ones used by practitioners
and researchers alike.

Daily Statistics

The ﬁrst group represents the overall trading activity in the instrument:

• Open, High, Low, Close (OHLC) and Previous Close Price: The OHLC pro-
vides a good indication of the trading activity as well as the intraday volatility
experienced by the instrument. The distance traveled between the lowest and
highest point of the day usually gives a better indication of market sentiment
than the simple close-to-close return. Keeping the previous close value as part
of the same time series is also a good way to improve computation eﬃciency by
not having to make an additional database query to compute the daily return and
overnight gap. The previous close needs, however, to be properly adjusted for
corporate actions and dividends.

• Last Trade before Close (Price/Size/Time): It is useful in determining how
much the close price may have jumped in the ﬁnal moments of trading, and con-
sequently, how stable it is as a reference value for the next day.

• Volume: It is another valuable source of trading activity indicator, in particu-
lar when the level jumps from the long term average. It is also worth collecting
the volume breakdown between lit and dark venues, in particular for execution
strategies.

• Auctions Volume and Price: Depending on the exchange, there can be multiple
auctions in a day (Open, Close, Morning Close and Afternoon Open—for mar-
kets with a lunch break—as well as ad hoc liquidity or intraday auctions). Owing
to their liquidity aggregation nature, they can be considered as a valuable price
discovery event when signiﬁcant volume prints occur.

40

Algorithmic Trading and Quantitative Strategies

• VWAP: Similar to the OHLC, the intraday VWAP price gives a good indication
of the trading activity on the day. It is not uncommon to build trading strategies
using VWAP time series instead of just close-to-close prices. The main advantage
being that VWAP prices represent a value over the course of the day and, as such,
for larger orders are easier to achieve through algorithmic execution than a single
print.

• Short Interest/Days-to-Cover/Utilization: This data set is a good proxy for
investor positioning. The short pressure might be an indication of upcoming short
term moves: A large short interest usually indicates a bearish view from institu-
tional investors. Similarly, the utilization level of available securities to borrow
(in order to short) gives an indication of how much room is left for further short-
ing (when securities become “Hard to Borrow”, the cost of shorting becomes
signiﬁcantly higher requiring short sellers to have strong enough beliefs in the
short term price direction). Finally, days-to-cover data is also valuable to assess
the magnitude of a potential short squeeze. If short sellers need to unwind their
positions, it is useful to know how much volume this represents as a fraction
of available daily liquidity. The larger the value, the larger the potential sudden
upswing on heavily shorted securities.

• Futures Data: Futures markets provide additional insight into the activity of large
investors through open interest data, that can be useful to develop alpha strategies.
Additionally, ﬁnancial futures oﬀer arbitrage opportunities if their basis exhibits
mispricing compared to one’s dividend estimates. As such, recording the basis of
futures contracts is worthwhile even for strategies that do not particularly target
futures.

• Index-Level Data: This is also a valuable data set to collect as a source of rel-
ative measures for instrument speciﬁc features (Index OHLC, Volatility, . . . ). In
particular for dispersion strategies, normalized features help identify individual
instruments deviating from their benchmarks.

• Options Data: The derivative market is a good source of information about the
positioning of traders through open interest and Greeks such as Gamma and Vega.
How the broader market is pricing an instrument through implied volatility for
instance is of interest.

• Asset Class Speciﬁc: There is a wealth of cross-asset information available when
building strategies, in particular in Fixed Income, FX, and Credit markets. Among
the basic ones, we would note:

– Yield / benchmark rates (repo, 2y, 10y, 30y)
– CDS Spreads
– US Dollar Index

The second group of daily data represents granular intraday microstructure activ-

ity and is mostly of interest to intraday or execution trading strategies:

Trading Fundamentals

41

• Number and Frequency of Trades: A proxy for the activity level of an instru-
ment, and how continuous it is. Instruments with a low number of trades are
harder to execute and can be more volatile.

• Number and Frequency of Quote Updates: Similar proxy for the activity level.

• Top of Book Size: A proxy for liquidity of the instrument (larger top of book size

makes it possible to trade larger order size quasi immediately, if needed).

• Depth of Book (price and size): Similar proxy for liquidity.

• Spread Size (average, median, time weighted average): This provides a proxy
for cost of trading. A parametrized distribution of spread size can be used to iden-
tify intraday trading opportunities if they are cheap or expensive.

• Trade Size (average, median): Similar to spread size, trade sizes and their dis-
tribution are useful to identify intraday liquidity opportunities when examining
the volume available in the order book.

• Ticking Time (average, median): The ticking time and its distribution is a rep-
resentation of how often, one should expect changes in the order book ﬁrst level.
This is particularly helpful for execution algorithms for which the frequency of
updates (adding/canceling child orders, reevaluating decisions, etc.) should be
commensurate with the characteristics of the traded instrument.

The daily distributions of these microstructure variables can be used as start of
day estimates in trading algorithms and be updated intraday as additional data
ﬂows in through online Bayesian updates.

Finally, the last group of daily data can be derived from the previous two groups
through aggregation but is usually stored pre-computed in order to save time during
the research phase (e.g., X-day trailing data), or to be used as normalizing values
(e.g., size as a percentage of ADV, spread in relation to long-term average, . . . ). Some
common examples would be:

• 𝑋-day Average Daily Volume (ADV) / Average auction volume

• 𝑋-day volatility (close-to-close, open-to-close, etc.)

• Beta with respect to an index or a sector (plain beta, or asymmetric up-days/down-

days beta)

• Correlation matrix

As a reminder to the reader, aggregated data need to fully support the peculiar-
ities described in the Reference Data section (for instance: The existence of special
event days which, if included, can signiﬁcantly skew intraday distribution of values;
mishandling of market asynchronicity resulting in inaccurate computations of key
quantities such as beta or correlation, etc.).

42

Binned Data

Algorithmic Trading and Quantitative Strategies

The ﬁrst natural extension to daily data sets is a discretization of the day into bins
ranging from a few seconds to 30 minutes. The features collected are comparable to
the ones relevant for daily data sets (period volume, open, high, low, close, VWAP,
spread, etc.), but computed at higher frequency. It is worth mentioning that minute
bar data actually underpins the vast majority of microstructure models used in the
electronic execution space. Volume and spread proﬁles, for instance, are rarely built
with a granularity ﬁner than one minute to prevent introducing excess noise due purely
to market frictions. The major advantage of binned-data is that discrete time series
methods can be readily used.

These minute bar data sets are also quite popular for the backtesting of low-to-
medium frequency trading strategies targeting intraday alpha (short duration market
neutral long-short baskets, momentum and mean-reversion strategies, etc.). The main
beneﬁt they provide is a signiﬁcant dimension reduction compared to raw market
data (390 rows per stock per day in the US compared to millions for raw data for the
liquid stocks), which allows researchers to perform rapid and eﬃcient backtesting as
they search for alpha. However, increasing data frequency from daily data to intraday
minute bars also presents challenges. In particular, relationships that appear to be
stable using close-to-close values become much noisier as granularity increases and
signals become harder to extract (e.g., drop in correlation between assets, pricing
ineﬃciency moves due to sudden liquidity demand, etc.). Similarly, for less liquid
assets with a low trade frequency, there may not be any trading activity for shorter
durations, resulting in empty bins.

1.4.4 Fundamental Data and Other Data Sets

A very large number of quantitative investment strategies are still based on funda-
mental data, and consequently countless research papers are available to the interested
reader describing examples of their usage. Here we will only describe the main classes
of existing fundamental data:

• Key Ratios: EPS (Earnings Per Share), P/E (Price-to-Earning), P/B (Price-to-
Book Value), . . . . These metrics represent a normalized view of the ﬁnancials
of companies allowing for easier cross-sectional comparison of stocks and their
ranking over time.

• Analyst Recommendations: Research analysts at Sell Side institutions spend a
great deal of resources analyzing companies they cover, in order to provide invest-
ment recommendations usually in the form of a Buy/Hold/Sell rating accompa-
nied by a price target. While individual recommendations might prove noisy, the
aggregate values across a large number of institutions can be interpreted as a
consensus valuation of a given stock, and changes in consensus can have a direct
impact on price.

• Earnings Data: Similarly research analysts also provide quarterly earning esti-
mates that can be used as an indication of the performance of a stock before the

Trading Fundamentals

43

actual value gets published by the company. Here, too, consensus values tend to
play a larger role. In particular, when the diﬀerence between the forecast consen-
sus and the realized value is large (known as earning surprise), as the stock might
then experience outsized returns in the following days. Thus, collecting analysts’
forecasts as well as realized values can be a valuable source of information in the
design of trading strategies.

• Holders: In some markets, large institutional investors are required to disclose
their holdings on a regular basis. For instance, in the US, institutional investment
managers with over $100 million in assets must report quarterly, their holdings,
to the SEC using Form 13F. The forms are then publicly available via the SEC’s
EDGAR database. Additionally, shareholders might be required to disclose their
holdings once they pass certain ownership thresholds.31 Sudden changes in such
ownership might indicate changes in sentiment by sophisticated investor and can
have a signiﬁcant impact on stock performance.

• Insiders Purchase/Sale: In some markets, company directors are required by law
to disclose their holdings of the company stock as well as any increase or decrease
of such holdings.32 This is thought to be an indicator of future stock price moves
from the group of people who have access to the best possible information about
the company.

• Credit Ratings: Most companies issue both stocks and bonds to ﬁnance their
operations. The credit ratings of bonds and their changes over time provide addi-
tional insight into the health of a company and are worth leveraging. In particular,
credit downgrades resulting in higher funding costs in the future, generally have
a negative impact on equity prices.

• And much, much more: Recent years have seen the emergence of a wide variety
of alternative data sets that are available to researchers and practitioners alike.33
While it is not possible to make a comprehensive list of all that are available, they
can generally be classiﬁed based on their characteristics: Frequency of publica-
tion, structured or unstructured, and velocity of dissemination. The value of such
data depends on the objectives and resources of the user (natural language pro-
cessing or image recognition for unstructured data require signiﬁcant time and
eﬀorts), but also—and maybe more importantly—on the uniqueness of the data

31In the US, Form 13D must be ﬁled with the SEC within 10 days by anyone who acquires beneﬁcial

ownership of more than 5% of any class of publicly traded securities in a public company.

32In the US, oﬃcers and directors of publicly traded companies are required to disclose their initial
holdings in the company by ﬁling Form 3 with the SEC, as well as Form 4 within 2 days of any subsequent
changes. The forms are then publicly available via the SEC’s EDGAR database.

33For instance, www.orbitalinsight.com oﬀers daily retail traﬃc analytics, derived from satellite
imagery analysis monitoring over 260,000 parking lots, as well as estimates of oil inventories through
satellite monitoring of oil storage facilities.

44

Algorithmic Trading and Quantitative Strategies

set. As more investors get access to it, the harder it becomes to extract meaningful
alpha from it.

1.5 Market Microstructure: Economic Fundamentals of Trading
We ﬁrst provide a brief review of market microstructure, an area of ﬁnance that
studies how the supply and demand of liquidity results in actual transactions and mod-
iﬁes the subsequent state of the market, and then delve into some critical operational
concepts. We draw upon key review papers by Madhavan (2000) [254], Biais, Glosten
and Spatt (2005) [39] and O’Hara (2015) [276]. The core idea is that the eﬃcient mar-
ket hypothesis, which postulates that the equity price impounds all the information
about the equity, may not hold due to market frictions. Algorithmic trading essentially
exploits the speed with which investors acquire the information and how they use it
along with the market frictions that arise mainly due to demand-supply imbalances.
Taking an informational economics angle, Madhavan (2000) [254] provides a market
microstructure analysis framework following three main categories.

– Price Formation and Price Discovery: How do prices impound information over

time and how do the determinants of trading costs vary?

– Market Design: How do trading rules aﬀect price formation?

Market design generally refers to a set of rules that all players have to follow in
the trading process. These include the choice of tick size, circuit breakers which
can halt trading in the event of large price swings, the degree of anonymity and
the transparency of the information to market participants, etc. Markets around
the world and across asset classes can diﬀer signiﬁcantly in these types of rules,
creating a diverse set of constraints and opportunity for algorithmic traders. Some
early research on the eﬀects of market design lead to following broad conclusions:

– Centralized trading via one single market tends to result in more eﬃcient price
discovery with smaller bid-ask spreads. In the presence of multiple markets, the
primary markets (such as NYSE, NASDAQ) remain the main sources of price
discovery (see Hasbrouck (1995) [181]).

– Despite market participants’ preference for continuous, automated limit order
book markets, theoretical models suggest that multilateral trading approaches
such as single-price call auctions are the most eﬃcient in processing diverse
information (See Mendelson (1982) [263] and Ho et al. (1985) [197]).

– Transparency: How do the quantity, quality and speed of information provided to

market participants aﬀect the trading process?

Transparency which is broadly classiﬁed into pre-trade (lit order book) and post-
trade (trade reporting to the public, though at diﬀerent time lags) is often a trade-oﬀ.

Trading Fundamentals

45

While in theory more transparency should lead to better price discovery, the wide
disclosure of order book depth information can lead to thinner posted sizes and
wider bid-ask spreads if participants fear revealing their intent and possibly their
inventory levels, leading them to favor more oﬀ-exchange activity.

All the above points can be analyzed in light of the rapid growth of high frequency
trading (HFT) described earlier in this chapter. O’Hara (2015) [276] presents issues
related to microstructure in the context of HFT. While the basic tenet that traders may
use private information or learn from market data such as orders, trade size, volume,
duration between successive trades, etc., has remained the same, the trading is now
mostly automated to follow some rules. These rules are based partly on prior informa-
tion and partly on changing market conditions, monitored through order ﬂows. With
the high speed, adverse selection has taken a diﬀerent role. Some traders may have
access to market data milliseconds before others have it and this may allow them to
capture short term price movements. This would expand the pool of informed traders.
Here are some topics for research in microstructure:

– With a parent order sliced into several child orders that are sent to market for exe-
cution during the course of trading, it is diﬃcult to discern who is the informed
trader. Informed traders use sophisticated dynamic algorithms to interact with the
market. Retail (known in the literature as ‘uninformed’) trades usually cross the
spread.

– More work needs to be done to understand trading intensity in short intervals. Order

imbalance is empirically shown to be unrelated to price levels.

– Informed traders may increasingly make use of hidden orders. How these orders

enter and exit the markets require further studies.

– Traders respond to changing market conditions by revising their quoted prices. The
quote volatility can provide valuable information about the perceived uncertainty
in the market.

Although HFT has resulted in more eﬃcient markets, with lower bid-ask spreads,
the data related to trade sequences, patterns of cancellations across fragmented mar-
kets require new tools for analysis and for making actionable inference. In this review,
we do not present any models explicitly and these will be covered throughout this
book. Now keeping in line with the practical perspectives of this book, we highlight
some key concepts.

1.5.1 Liquidity and Market Making
(a) A Deﬁnition of Liquidity: Financial markets are commonly described as carry-
ing the function of eﬃciently directing the ﬂow of savings and investments to the
real economy to allow the production of goods and services. An important factor
contributing to well-developed ﬁnancial markets is in facilitating liquidity which
enables investors to diversify their asset allocation and the transfers of securities
at a reasonable transaction cost.

46

Algorithmic Trading and Quantitative Strategies

Properly describing “liquidity” often proves to be elusive as there is no commonly
agreed upon deﬁnition. Black (1971) [42] proposes a relatively intuitive descrip-
tion of a liquid market: “The market for a stock is liquid if the following conditions
hold:

• There are always bid and ask prices for the investor who wants to buy or

sell small amounts of stock immediately.

• The diﬀerence between the bid and ask prices (the spread) is always small.
• An investor who is buying or selling a large amount of stock, in the absence
of special information, can expect to do so over a long period of time at a
price not very diﬀerent, on average, from the current market price.

• An investor can buy or sell a large block of stock immediately, but at a
premium or discount that depends on the size of the block. The larger the
block, the larger the premium or discount.”

In other words, Black deﬁnes a liquid market as a continuous market having
the characteristics of relatively tight spread, with enough depth on each side of
the limit order book to accommodate instantaneous trading of small orders, and
which is resilient enough to allow large orders to be traded slowly without signif-
icant impact on the price of the asset. This general deﬁnition remains particularly
well suited to today’s modern electronic markets and can be used by practition-
ers to assess the diﬀerence in liquidity between various markets when choosing
where to deploy a strategy.

). If 𝑝∗
𝑡

(b) Model for Market Friction: We begin with a model to accommodate the friction
in stock price (𝑃𝑡
is the (log) true value of the asset which can vary over
time due to expected cash ﬂows or due to variation in the discount rate. Given the
publicly available information and the assumption of market eﬃciency, we have:
. Therefore the observed return, 𝑟𝑡 = 𝑝𝑡 − 𝑝𝑡−1 =
𝑝𝑡 = 𝑝∗
𝜖𝑡 + (𝑎𝑡 − 𝑎𝑡−1) can exhibit some (negative) serial correlation, mainly a result
’ can be a function of a number of factors such as
of friction. The friction, ‘𝑎𝑡
inventory costs, risk aversion, bid-ask spread, etc. When 𝑎𝑡 = 𝑐𝑠𝑡
is the
direction of the trade and ‘𝑐’ is the half-spread, the model is called a Roll model.
This can explain the stickiness in returns in some cases.

, where 𝑠𝑡

𝑡−1 + 𝜖𝑡

𝑡 = 𝑝∗

and 𝑝∗

𝑡 + 𝑎𝑡

(c) Diﬀerent Styles of Market Participants: Liquidity Takers and Liquidity
Providers: Liquid ﬁnancial markets carry their primary economic function by
facilitating savings and investment ﬂows as well as allowing investors to exchange
securities in the secondary markets. Economic models of ﬁnancial markets
attempt to classify market participants into diﬀerent categories. These can be
broadly delineated as:

Informed Traders, making trading decisions based on superior information that
is not yet fully reﬂected in the asset price. That knowledge can be derived from
either fundamental analysis or information not directly available nor known to
other market participants.

Trading Fundamentals

47

News Traders, making trading decisions based on market news or announce-
ments and trying to make proﬁts by anticipating the market’s response to a par-
ticular catalyst. The electroniﬁcation of news dissemination oﬀers new opportuni-
ties for developing quantitative trading strategies by leveraging text-mining tools
such as natural language processing to interpret, and trade on, machine readable
news before it is fully reﬂected in the market prices.

Noise Traders (as introduced by Kyle (1985) [234]), making trading decisions
without particular information and at random times mainly for liquidity reasons.
They can be seen as adding liquidity to the market through additional volume
transacted, but only have a temporary eﬀect on price formation. Their presence
in the market allows informed traders not to be immediately detected when they
start transacting, as market makers cannot normally distinguish the origin of the
order ﬂow between these two types of participants. In a market without noise
traders, being fully eﬃcient, at equilibrium each trade would be revealing infor-
mation that would instantly be incorporated into prices, hereby removing any
proﬁt opportunities.

Market Makers, providing liquidity to the market with the intent of collecting
proﬁts originating from trading frictions in the market place (bid-ask spread).
Risk-neutral market makers are exposed to adverse selection risk arising from
the presence of informed traders in the marketplace, and therefore establish their
trading decisions mostly based on their current inventory. As such, they are often
considered in the literature to drive the determination of eﬃcient prices by acting
as the rational intermediaries.

Generalizing these concepts, market participants and trading strategies can be
separated between liquidity providing and liquidity seeking. The former being
essentially the domain of market makers whose level of activity, proxied by mar-
ket depth, is proportional to the amount of noise trading and inversely propor-
tional to the amount of informed trading (Kyle (1985) [234]). The latter being
the domain of the variety of algorithmic trading users, described before (mutual
funds, hedge funds, asset managers, etc.). Given the key role played by market
makers in the liquidity of electronic markets, they have been the subject of a large
corpus of academic research focusing on their activities.

(d) The Objectives of the Modern Market Maker: At present, market making can
broadly be separated into two main categories based on the trading characteris-
tics. The ﬁrst one is the provision of large liquidity—known as blocks—to institu-
tional investors, and has traditionally been in the realm of sell-side brokers acting
as intermediaries and maintaining signiﬁcant inventories. Such market makers
usually transact through a non-continuous, negotiated, process based on their cur-
rent inventory, as well as their assessment of the risk involved in liquidation of the
position in the future. Larger or more volatile positions generally tend to come at
a higher cost, reﬂecting the increased risk for the intermediary. But they provide

48

Algorithmic Trading and Quantitative Strategies

the end investor with a certain price and an immediate execution bearing no tim-
ing risk that is associated with execution over time. These transactions, because
they involve negotiations between two parties, still mostly happen in a manual
fashion or over the phone and then get reported to an appropriate exchange for
public dissemination.

The second category of market making involves the provision of quasi-
continuous, immediately accessible quotes on an electronic venue. With the
advent of electronic trading described in the previous section, market makers
originally seated on the exchange ﬂoors have progressively been replaced by elec-
tronic liquidity providers (ELP). The ELP leverage fast technology to dissemi-
nate timely quotes across multiple exchanges and develop automated quantita-
tive strategies to manage their inventory and the associated risk. As such, most
ELP can be classiﬁed as high frequency traders. They derive their proﬁts from
three main sources: From liquidity rebates on exchanges that oﬀer maker-taker
fee structure, from spread earned when successfully buying on the bid and sell-
ing on the oﬀer, and from short-term price moves favorable to their inventory.

Hendershott, Brogaard and Riordan (2014) [191] ﬁnd that HFT activity tends to
be concentrated in large liquid stocks and postulate that this can be attributed
to a combination of larger proﬁt opportunities emanating from trades happening
more often, and from easier risk management due to larger liquidity that allows
for easier exit of unfavorable positions at a reasonable cost.

(e) Risk Management: In the existing literature on informed trading, it is observed
that liquidity supplying risk-neutral market makers are adversely selected by
informed traders suddenly moving prices against them. For example, a market
maker buy quote tends to be executed when large sellers are pushing the price
down, resulting in even lower prices in the near term. This signiﬁcant potential
asymmetry of information at any point in time emphasizes the need for market
makers to employ robust risk management techniques, particularly in the domain
of inventory risk. For a market maker, risk management is generally accomplished
by ﬁrst adjusting market quotes upward or downward to increase the arrival rate
of sellers or buyers and consequently adjusting the inventory in the desired direc-
tion. If biasing the quotes does not result in a successful inventory adjustment,
the market makers generally employ limit orders to cross the spread.

Ho and Stoll (1981) [196] introduced a market making model in which the market
maker’s objective is to maximize proﬁt while minimizing the probability of ruin by
determining the optimal bid-ask spread to quote. The inventory held evolves through
the arrival of bid and ask orders, where the arrival rate is taken to be a function of bid
and ask prices. Their model also incorporates the relevant notion of the size depen-
dence of spread on the market maker’s time horizon. The longer the remaining time,
the greater potential for adverse move risk for liquidity providers, and vice versa.
This is consistent with observed spreads. In most markets, the spread is wider at the
beginning of the day, narrowing toward the close. An additional reason annotated for

Trading Fundamentals

49

the wider spread right after the beginning of the trading day is due to the existence
of potentially signiﬁcant information asymmetry accumulated overnight. As market
makers are directly exposed to that information asymmetry, they tend to quote wider
spreads while the price discovery process unfolds following the opening of continu-
ous trading, and progressively tighten them as uncertainty about the fair price for the
asset dissipates.

Understanding the dynamics of market makers inventory and risk management,
and their eﬀect on spreads, has direct implications for the practitioners who intend to
deploy algorithmic trading strategies as the spread paid to enter and exit positions is a
non-negligible source of cost that can erode the proﬁtability of low alpha quantitative
strategies.

Hendershott and Seasholes (2007) [195] conﬁrms that market makers’ inventories
are negatively correlated with previous price changes and positively correlated with
subsequent changes. This is consistent with market makers ﬁrst acting as a dampener
of buying or selling pressure by bearing the risk of temporarily holding inventory
in return for earning the spread and thus potential price appreciation from market
reversal. This model easily links liquidity provision and the dynamics of asset prices.
Finally, it has been observed that there is a positive correlation of market makers
inventory with subsequent prices changes, inventories can complement past returns
when predicting future returns. Since inventories are not publicly known, market par-
ticipants use diﬀerent proxies to infer their values throughout the day. Two commonly
used proxies are trade imbalances (the net excess of buy or sell initiated trade volume)
and spreads. Trade imbalance aims at classifying trades, either buy initiated or sell ini-
tiated, by comparing their price with the prevailing quote. Given that market makers
try to minimize their directional risk, they can only accommodate a limited amount
of non-diversiﬁed inventory over a ﬁnite period of time. As such, spread sizes, and
more particularly their sudden variation, have also been used as proxies for detecting
excess inventory forcing liquidity providers to adjust their positions.

Bibliography

[1] F. Abdi and A. Ranaldo. A simple estimation of bid-ask spreads from daily
close, high and low prices. The Review of Financial Studies, 30:4437–4480,
2017.

[2] F. Abergel and A. Jedidi. A mathematical approach to order book modeling.
International Journal of Theoretical and Applied Finance, 16:1–40, 2013.

[3] A.R. Admati and P. Pﬂeiderer. A theory of intraday patterns: Volume and price

variability. The Review of Financial Studies, 1(1):3–40, 1988.

[4] Y. Aït-Sahalia, P.A. Mykland, and L. Zhang. How often to sample a
continuous-time process in the presence of market microstructure noise. The
Review of Financial Studies, 18(2):351–416, 2005.

[5] H. Akaike. A new look at the statistical model identiﬁcation. IEEE Transac-

tions on Automatic Control, AC–19:716–723, 1974.

[6] S.S. Alexander. Price movements in speculative markets: Trends of random

walks. Industrial Management Review, pages 7–26, 1961.

[7] S.S. Alexander. Price movements in speculative markets: Trends of random

walks, no 2. Industrial Management Review, pages 25–46, 1964.

[8] S. Alizadeh, M.W. Brandt, and F.X. Diebold. Range-based estimation of
stochastic volatility models. Journal of Finance, 57:1047–1091, 2002.

[9] R. Almgren. Execution costs. Encyclopedia of Quantitative Finance, pages

1–5, 2008.

[10] R. Almgren and N. Chriss. Optimal execution of portfolio transactions. The

Journal of Risk, 3:5–39, 2000.

[11] R. Almgren, C. Thum, E. Hauptmann, and H. Li. Equity market impact. Risk,

18(7):57–62, 2005.

[12] R.F. Almgren. Optimal execution with nonlinear impact functions and trading

enhanced risk. Applied Mathematical Finance, 10:1–18, 2003.

[13] N. Amenc, F. Goltz, A. Lodh, and L. Martellini. Diversifying the diversi-
ﬁers and tracking the tracking error: Outperforming cap-weighted indices with
limited risk of underperformance. The Journal of Portfolio Management,
38(3):72–88, 2012.

51

52

Bibliography

[14] S. Anatolyev and A. Gospodinov. A trading approach to testing for predictabil-

ity. Journal of Business & Economic Statistics, 23:455–461, 2005.

[15] S. Anatolyev and A. Gospodinov. Modeling ﬁnancial return dynamics via
decomposition. Journal of Business & Economic Statistics, 28:232–245, 2010.

[16] T. Andersen, I. Archakov, G. Cebiroglu, and N. Hautsch. Volatility information
feedback and market microstructure noise: A tale of two regimes. CFS Working
Paper, Northwestern University, 2017.

[17] T.G. Andersen. Return volatility and trading volume: An information ﬂow
interpretation of stochastic volatility. Journal of Finance, 51:116–204, 1996.

[18] T.G. Andersen and T. Bollerslev. Answering the skeptics: Yes, standard volatil-
International Economic Review,

ity models do provide accurate forecasts.
39:885–905, 1998.

[19] T.W. Anderson. An Introduction to Multivariate Statistical Analysis. Second

Edition. Wiley, New York, 1984.

[20] T.W. Anderson and A.M. Walker. On the asymptotic distribution of the auto-
correlations of a sample from a linear stochastic process. Annals of Mathemat-
ical Statistics, 35:1296–1303, 1964.

[21] A. Ang and A. Timmermann. Regime changes and ﬁnancial markets. Annual

Review of Finance and Economics, 4:313–337, 2012.

[22] W. Antweiler and M.Z. Frank.

Is all that talk just noise? The information
content of internet stock message boards. Journal of Finance, 59(3):1259–
1294, 2004.

[23] P. Asquith, R. Oman, and C. Safaya. Short sales and trade classiﬁcation algo-

rithms. Journal of Financial Markets, 13:157–173, 2010.

[24] M. Avellaneda and J.H. Lee. Statistical arbitrage in the US equities market.

Quantitative Finance, 10:761–782, 2010.

[25] W. Bagehot. The only game in town. Financial Analysis Journal, 27:12–14,

1971.

[26] P. Bajgrowicz and O. Scaillet. Technical trading revisited: False discover-
ies, persistence tests, and transaction costs. Journal of Financial Economics,
106(3):473–491, 2012.

[27] M. Baker and J. Wurgler.

Investor sentiment and the cross-section of stock

returns. Journal of Finance, 61(4):1645–1680, 2006.

[28] M. Baker and J. Wurgler. Investor sentiment in the stock market. Journal of

Economic Perspectives, 21(2):129–151, 2007.

Bibliography

53

[29] F.M. Bandi and J.R. Russell. Separating microstructure noise from volatility.

Journal of Financial Economics, 79:655–692, 2006.

[30] N. Barberis, A. Shleifer, and R. Vishny. A model of investor sentiment. Journal

of Financial Economics, 49:307–343, 1998.

[31] O.E. Barndorﬀ-Nielsen and N. Shephard. Econometric analysis of realized
volatility and its use in estimating stochastic volatility models. Journal of the
Royal Statistical Society: Series B (Statistical Methodology), 64(2):253–280,
2002.

[32] L. Barras, O. Scaillet, and R. Wermers. False discoveries in mutual fund per-
formance: Measuring luck in estimated alphas. Journal of Finance, 65(1):179–
216, 2010.

[33] R. Battalio, S.A. Corwin, and R. Jennings. Can brokers have it all? On the
relation between make-take fees and limit order execution quality. Journal of
Finance, 71:2193–2238, 2016.

[34] L. Bauwens, S. Laurent, and J.V.K. Rombouts. Multivariate GARCH models:

A survey. The Journal of Applied Econometrics, 21:79–109, 2006.

[35] M. Bayraktar, I. Mashtaser, N. Meng, and S. Radchenko. Barra vs total market

equity trading model, empirical notes. MSCI Research, 2015.

[36] P. Bertrand and C. Protopopescu. The statistics of the information ratio. Inter-

national Journal of Business, 15:71–86, 2010.

[37] D. Bertsimas and A.W. Lo. Optimal control of executions costs. Journal of

Financial Markets, 1:1–50, 1998.

[38] H. Bessembinder, M. Panayides, and K. Venkataraman. Hidden liquidity: An
analysis of order exposure strategies in electronic stock markets. Journal of
Financial Economics, 94:361–383, 2009.

[39] B. Biais, L. Glosten, and C. Spatt. Market microstructure; a survey of micro-
foundations, empirical results, and policy implications. Journal of Financial
Markets, 8:217–264, 2005.

[40] B. Biais, P. Hillion, and C. Spatt. An empirical analysis of the limit order book
and the order ﬂow in the Paris bourse. Journal of Finance, 50:1655–1689,
1995.

[41] J.P. Bialkowski, S. Darolles, and Gaëlle G. Le Fol. Improving VWAP. strate-
gies: A dynamical volume approach. Journal of Banking and Finance, 32,
2006.

[42] F. Black. Towards a fully automated exchange, Part I. Financial Analysts

Journal, 27:29–34, 1971.

54

Bibliography

[43] F. Black. Capital market equilibrium with restricted borrowing. The Journal

of Business, 45:444–454, 1972.

[44] F. Black and R. Litterman. Global portfolio optimization. Financial Analysts

Journal, 48 No. 5:28–43, 1992.

[45] L. Blume, D. Easley, and M. O’Hara. Market statistics and technical analysis:

The role of volume. Journal of Finance, 49:153–181, 1994.

[46] T. Bollerslev. Generalized autoregressive conditional heteroskedasticity. Jour-

nal of Econometrics, 31:307–327, 1986.

[47] M. Borkovec and H.G. Heidle. Building and evaluating a transaction cost

model: A primer. The Journal of Trading, 5:57–77, 2010.

[48] P. Bossaerts. Common nonstationary components of asset prices. The Journal

of Economic Dynamics and Control, 12(2):347–364, 1988.

[49] J.P. Bouchaud, J.D. Farmer, and F. Lillo. How Markets Digest Supply and
Demand and Slowly Incorporate Information into Prices. Academic Press,
2009.

[50] J.P. Bouchaud, Y. Gefen, M. Potters, and M. Wyart. Fluctuations and response
in ﬁnancial markets: The subtle nature of “random” price changes. Quantita-
tive Finance, 4:176–190, 2004.

[51] J.-P. Bouchaud, M. Mezard, and M. Potters. Statistical properties of stock order
books: Empirical results and models. Quantitative Finance, 2:251–256, 2002.

[52] D. Bowen, M.C. Hutchinson, and N. O’Sullivan. High-frequency equity pairs
trading: Transaction costs, speed of execution and patterns in returns. The
Journal of Trading, Summer, pages 31–38, 2010.

[53] G.E.P. Box, G.M. Jenkins, G.C. Reinsel, and G.M. Ljung. Time Series Analy-

sis: Forecasting and Control, 5th edition. Wiley, New York, 2015.

[54] G.E.P. Box and G.C. Tiao. A canonical analysis of multiple time series.

Biometrika, 64:355–365, 1977.

[55] P. Boyle, L. Garlappi, R. Uppal, and T. Wang. Keynes meets Markowitz:
The trade-oﬀ between familiarity and diversiﬁcation. Management Science,
58:253–272, 2012.

[56] M.W. Brandt and P. Santa-Clara. Dynamic portfolio selection by augmenting

the asset space. Journal of Finance, 61:2187–2217, 2006.

[57] L. Breiman. Bagging predictors. Machine Learning, 24(2):123–140, 1996.

[58] L. Breiman. Prediction games and arcing algorithms. Neural Computation,

11(7):1493–1517, 1999.

Bibliography

55

[59] D.R. Brillinger. Time Series: Data Analysis and Theory. Expanded edition.

Holden-Day, San Francisco, 1981.

[60] W. Brock, J. Lakonishok, and B. LeBaron. Simple technical trading rules and
the stochastic properties of stock returns. Journal of Finance, 47:1731–1764,
1992.

[61] J. Brodie, I. Daubechies, C. De Mol, D. Giannone, and I. Loris. Sparse and
stable Markowitz portfolios. Proceedings of the National Academy of Sciences,
106:12267–12272, 2009.

[62] C. Brownlees, F. Cipollini, and G.M. Gallo. Intra-daily volume modeling and
prediction for algorithmic trading. The Journal of Financial Econometrics,
9:489–518, 2011.

[63] B. Bruder, N. Gaussel, J.-C. Richard, and T. Roncalli. Regularization of port-

folio allocation. White Paper Issue #10, 2013.

[64] E. Busseti and S. Boyd. Volume Weighted Average Price Optimal Execution.

unpublished, Stanford University, 2015.

[65] J.Y. Campbell, S.J. Grossman, and J. Wang. Trading volume and serial cor-
relation in stock returns. The Quarterly Journal of Economics, 108:905–939,
1993.

[66] J.Y. Campbell, A.W. Lo, and A.C. MacKinlay. The Econometrics of Financial

Markets. Princeton University Press, New Jersey, 1996.

[67] C. Cao, O. Hansch, and X. Wang. The information content of an open limit

order book. The Journal of Futures Markets, 29:16–41, 2009.

[68] M.M. Carhart. On persistence in mutual fund performance.

Finance, 52(1):57–82, 1997.

Journal of

[69] M. Centoni and G. Cubadda. Modeling co-movements of economic time

series: A selective survey. Statistica, 71:267–293, 2011.

[70] A.P. Chaboud, B. Chiquoine, E. Hjalmarsson, and C. Vega. Rise of the
machines: Algorithmic trading in the foreign exchange market. Journal of
Finance, 69:2045–2084, 2014.

[71] B. Chakrabarty, P.C. Moulton, and A. Shkilko. Short sales, long sales, and
the Lee-Ready trade classiﬁcation algorithm revisited. Journal of Financial
Markets, 15(4):467–491, 2012.

[72] K. Chan and W.-M. Fong. Trade size, order imbalance and the volatility-
volume relation. Journal of Financial Economics, 57:247–273, 2000.

[73] L. Chan and J. Lakonishok. Institutional equity trading costs, NYSE versus

Nasdaq. Journal of Finance, 52:713–735, 1997.

56

Bibliography

[74] L.K.C. Chan and J. Lakonishok. The behavior of stock prices around institu-

tional trades. Journal of Finance, 50:1147–1174, 1995.

[75] L.K.C. Chan and J. Lakonishok. Institutional equity trading costs: NYSE ver-

sus Nasdaq. Journal of Finance, 52(2):176–190, 1997.

[76] N.F. Chen, R. Roll, and S.A. Ross. Economic forces and the stock market. The

Journal of Business, 59:383–403, 1986.

[77] S. Chib. Estimation and comparison of multiple change-point models. Journal

of Econometrics, 86:221–241, 1998.

[78] C. Chiyachantana, P.K. Jain, C. Jiang, and R.A. Wood. International evidence
on institutional trading behavior and price impact. Journal of Finance, 59:869–
898, 2004.

[79] T. Chordia, R. Roll, and A. Subrahmanyam. Commonality in liquidity. Journal

of Financial Economics, 56:3–28, 2000.

[80] T. Chordia, R. Roll, and A. Subrahmanyam. Order imbalance, liquidity, and

market returns. Journal of Financial Economics, 65:111–130, 2002.

[81] P.K. Clark. A subordinated stochastic process model with ﬁnite variance for

speculative prices. Econometrica, 41:135–155, 1973.

[82] J. Conrad and G. Kaul. An anatomy of trading strategies. The Review of

Financial Studies, 11:489–519, 1998.

[83] J. Conrad and S. Wahal. The term structure of liquidity provision. Journal of

Financial Economics, 136:239–259, 2020.

[84] J.S. Conrad, A. Hameed, and C. Niden. Volume and autocovariances in short-
horizon individual security returns. Journal of Finance, 49:1305–1329, 1994.

[85] A. Constantinos, J.A. Donkas, and A. Subrahmanyam. Cognitive dissonance,
sentiment and momentum. Journal of Financial and Quantitative Analysis,
46:245–275, 2013.

[86] R. Cont and A. Kukanov. Optimal order placement in limit order markets.

Quantitative Finance, 17:21–39, 2017.

[87] R. Cont, A. Kukanov, and S. Stoikov. The price impact of order book events.

The Journal of Financial Econometrics, 12:47–88, 2014.

[88] R. Cont, S. Stoikov, and R. Talreja. A stochastic model for order book dynam-

ics. Operations Research, 58:549–563, 2010.

[89] M. Cooper. Filter values based on price and volume in individual security

overreaction. The Review of Financial Studies, 12:901–935, 1999.

Bibliography

57

[90] S.A. Corwin and P. Schultz. A simple way to estimate bid-ask spreads from

daily high and low prices. Journal of Finance, 67:719–759, 2012.

[91] A. Cowles. Can stock market forecasters forecast. Econometrica, 1:309–324,

1933.

[92] A. Cowles. Stock market forecasting. Econometrica, 12:206–214, 1944.

[93] D.R. Cox and P.A.W. Lewis. The Statistical Analysis of Series of Events. Chap-

man and Hall, London, 1966.

[94] G. Creamer and Y. Freund. Automated trading with boosting and expert

weighting. Quantitative Finance, 4:401–420, 2010.

[95] M. Cremers and D. Weinbaum. Deviations from put-call parity and stock return
predictability. Journal of Financial and Quantitative Analysis, 45:335–367,
2010.

[96] D.M. Cutler, J.M. Poterba, and L.H. Summers. What Moves Stock Prices?,
volume 15. National Bureau of Economic Research Cambridge, Mass., USA,
1989.

[97] S. Da, J. Engelberg, and P. Gao. In search of attention. Journal of Finance,

66(5):1461–1499, 2011.

[98] Z. Da, J. Engelberg, and P. Gao. The sum of all fears investor sentiment and

asset prices. The Review of Financial Studies, 28(1):1–32, 2015.

[99] R. Dahlhaus and S. Subba Rao. Statistical inference for time-varying ARCH

processes. The Annals of Statistics, 34:1075–1114, 2006.

[100] D.J. Daley and D. Vere-Jones. An Introduction to the Theory of Point Pro-
cesses, Volume I: Elementary Theory and Methods. Springer, New York, 2003.

[101] H.E. Daniels. Autocorrelation between ﬁrst diﬀerences of mid-ranges. Econo-

metrica, pages 215–219, 1966.

[102] S.R. Das and M.Y. Chen. Yahoo! for Amazon: Sentiment extraction from small

talk on the web. Management Science, 53:1375–1388, 2007.

[103] B.J. DeLong, A. Shleifer, L.H. Summers, and R.J. Waldmann. Noise trader risk

in ﬁnancial markets. The Journal of Political Economy, 98:703–738, 1990.

[104] V. DeMiguel, L. Garlappi, and R. Uppal. Optimal versus naïve diversiﬁcation:
How ineﬃcient is the 1∕𝑁 portfolio strategy? The Review of Financial Studies,
22:1915–1953, 2009.

[105] A.P. Dempster, N.-M. Laird, and D.B. Rubin. Maximum likelihood from
incomplete data via the EM algorithm. Journal of the Royal Statistical Society,
Series B, 39:1–38, 1977.

58

Bibliography

[106] B. Do and R. Faﬀ. Does simple pairs trading still work? The Financial Analysts

Journal, 66(4):83–95, 2010.

[107] I. Domowitz, J. Glen, and A. Madhavan. Liquidity, volatility and equity trading
costs across countries and over time. International Finance, 4:221–255, 2001.

[108] I. Domowitz and H. Yegerman. The cost of algorithmic trading: A ﬁrst look
at comparative performance. Algorithmic Trading: Precision, Control, Execu-
tion, 2005.

[109] A. Dufour and R. Engle. Time and the price impact of a trade. Journal of

Finance, 55(2):467–498, 2000.

[110] D. Easley, M.L. De Prado, and M. O’Hara. Flow toxicity and liquidity in a
high frequency world. The Review of Financial Studies, 25:1457–1493, 2012.

[111] D. Easley, M.L. De Prado, and M. O’Hara. Optimal execution horizon. Math-

ematical Finance, 25:640–672, 2015.

[112] D. Easley and M. O’ Hara. Price, trade size, and information in security mar-

kets. Journal of Financial Economics, 19:69–90, 1987.

[113] D. Easley and M. O’ Hara. Time and the process of security price adjustment.

Journal of Finance, 19:69–90, 1992.

[114] C. Eckart and G. Young. The approximation of one matrix by another of lower

rank. Psychometrika, 1:211–218, 1936.

[115] B. Efron, T. Hastie, I. Johnstone, and R. Tibshirani. Least angle regression.

Annals of Statistics, 32:407–499, 2004.

[116] J. Engelberg, P. Gao, and R. Jagannathan. An anatomy of pairs trading: the role
of idiosyncratic news, common information and liquidity. In Third Singapore
International Conference on Finance, 2009.

[117] R. Engle and R. Ferstenberg. Execution risk. The Journal of Portfolio Man-

agement, 33(4):34–44, 2007.

[118] R. Engle and F.K. Kroner. Multivariate simultaneous generalized ARCH.

Econometric Theory, 11:122–150, 1995.

[119] R. Engle and A. Lunde. Trades and quotes: A bivariate point process. The

Journal of Financial Economics, 1:159–188, 2003.

[120] R.F. Engle. Autoregressive conditional heteroscedasticity with estimates of the

variance of United Kingdom inﬂations. Econometrica, 50:987–1007, 1982.

[121] R.F. Engle, R. Ferstenberg, and J.R. Russell. Measuring and modeling execu-
tion cost and risk. The Journal of Portfolio Management, 38(2):14–28, 2012.

Bibliography

59

[122] R.F. Engle and C.W.J. Granger. Co-integration and error correction: Repre-
sentation, estimation, and testing. Econometrica, 55:251–276, 1987.

[123] R.F. Engle and S. Kozicki. Testing for common features. Journal of Business

& Economic Statistics, 11(4):369–380, 1993.

[124] R.F. Engle and D. Kraft. Multiperiod Forecast Error Variances of Inﬂation
Estimated from ARCH Models, in: A. Zellner, ed: Applied Time Series Analysis
of Economic Data. Bureau of the Census, Washington, DC, 1983.

[125] R.F. Engle and J.R. Russell. Autoregressive conditioned duration: A new
model for irregularly spaced transaction data. Econometrica, 66:1127–1162,
1998.

[126] R.F. Engle and R. Susmel. Common volatility in international equity markets.

Journal of Business & Economic Statistics, 11(2):167–176, 1993.

[127] R.F. Engle and S. Kozicki. Testing for common features. Journal of Business

& Economic Statistics, 11:369–380, 1993.

[128] R.F. Engle, V.K. Ng, and M. Rothschild. Asset pricing with a factor ARCH
covariance structure: Empirical estimates for treasury bills. Journal of Econo-
metrics, 45:213–218, 1990.

[129] T.W. Epps and M.L. Epps. The stochastic dependence of security price changes
and transaction volumes: Implications for the mixture-of-distribution hypoth-
esis. Econometrica, 44:305–321, 1976.

[130] T.W. Epps. Co-movements in stock prices in the very short run. Journal of the

American Statistical Association, 74:291–298, 1979.

[131] F.J. Fabozzi, S.M. Focardi, and P.N. Kolm. Quantitative Equity Investing:

Techniques and Strategies. Wiley and Sons, 2006.

[132] E.F. Fama and M.E. Blume. Filter rules and stock-market trading. The Journal

of Business, 39:226–241, 1966.

[133] E.F. Fama and K.R. French. A ﬁve-factor after pricing model. Journal of

Financial Economics, 116:1–22, 2015.

[134] E.F. Fama and K.R. French. International tests of a ﬁve-factor asset pricing

model. Fama-Miller Working Paper, 2015.

[135] J. Fan, Y. Fan, and J. Lv. High dimensional covariance matrix estimation using

a factor model. Journal of Econometrics, 147:187–197, 2008.

[136] J. Fan, F. Han, H. Liu, and B. Vickers. Robust inference of risks of large

portfolios. Journal of Econometrics, 194:298–308, 2016.

60

Bibliography

[137] J. Fan, J. Zhang, H. Liu, and K. Yu. Vast portfolio selection with cross-
the American Statistical Association,

Journal of

exposure constraints.
107:592–606, 2012.

[138] A. Farago and E. Hjalmarsson. Stock price co-movement and the foundations
of pairs trading. Journal of Financial and Quantitative Analysis, 54:629–665,
2019.

[139] J.D. Farmer, A. Gerig, F. Lillo, and H. Waelbroeck. How eﬃciency shapes

market impact. Quantitative Finance, 11:1743–1758, 2013.

[140] J.D. Farmer, L. Gillemot, F. Lillo, S. Mike, and A. Sen. What really causes

large price changes? Quantitative Finance, 4:383–397, 2004.

[141] W.E. Ferson and A.F. Siegel. The use of conditioning information in portfolios.

Journal of Finance, 56:967–982, 2001.

[142] T. Foucault and A.J. Menkveld. Competition for order ﬂow and smart order

routing systems. Journal of Finance, pages 119–157, 2008.

[143] A. Frazzini, R. Israel, and T.J. Moskowitz. Trading costs (unpublished), 2018.

[144] A. Frazzini and L.H. Pedersen. Betting against beta. Journal of Financial

Economics, 111:1–25, 2014.

[145] Y. Freund and R.E. Schapire. A decision-theoretic generalization of on-line
learning and an application to boosting. In European Conference on Compu-
tational Learning Theory, pages 23–37. Springer, 1995.

[146] Y. Freund and R.E. Schapire. Experiments with a new boosting algorithm. In
International Conference on Machine Learning, volume 96, pages 148–156.
Morgan Kaufmann Publishers Inc., San Francisco, CA, 1996.

[147] Y. Freund and R.E. Schapire. A decision-theoretic generalization of on-line
learning and an application to boosting. In Journal of Computer and System
Sciences, volume 55, pages 119–139, 1997.

[148] J. Friedman. Greedy function approximation: A gradient boosting machine.

Annals of Statistics, pages 1189–1232, 2001.

[149] J. Friedman, T. Hastie, and R. Tibshirani. Additive logistic regression: A sta-
tistical view of boosting (with discussion and a rejoinder by the authors). The
Annals of Statistics, 28(2):337–407, 2000.

[150] A. Frino, E. Jarnecic, and A. Lepone. The determinants of price impact of

block trades: Further evidence. Abacus, 43(2):94–106, 2007.

[151] K.A. Froot and E.M. Dabora. How are stock prices aﬀected by the location of

trade? Journal of Financial Economics, 53(2):189–216, 1999.

Bibliography

61

[152] P. Fryzlewicz, T. Sapatinas, and S. Subba Rao. Normalized least-squared esti-
mation in time-varying ARCH models. The Annals of Statistics, 36:742–786,
2008.

[153] W.A. Fuller.

Introduction to Statistical Time Series, Second Edition. John

Wiley, New York, 1996.

[154] L. Gagnon and G.A. Karolyi. Multi-market trading and arbitrage. Journal of

Financial Economics, 97:53–80, 2010.

[155] A.R. Gallant, P.E. Rossi, and G. Tauchen. Nonlinear dynamic structures.

Econometrica, 61:871–908, 1993.

[156] L. Gao, Y. Han, S.Z. Li, and G. Zhou. Market intraday momentum. Journal

of Financial Economics, 129:394–414, 2018.

[157] N. Gârleanu and L.H. Pedersen. Dynamic trading with predictable returns and

transaction costs. The Journal of Finance, 68:2309–2340, 2013.

[158] M.B. Garman and M.J. Klass. On the estimation of security price volatilities

from historical data. The Journal of Business, 53:67–78, 1980.

[159] P.H. Garthwaite. An interpretation of partial least squares. Journal of the

American Statistical Association, 89:122–127, 1994.

[160] E. Gatev, W.N. Goetzmann, and R.G. Rouwenhorst. Pairs trading: Performance
of a relative value arbitrage rule. The Review of Financial Studies, 19:797–827,
2006.

[161] J. Gatheral. No-dynamic-arbitrage and market impact. Quantitative Finance,

10:769–759, 2010.

[162] R. Gencay. The predictability of security returns with simple technical trading

rules. Journal of Empirical Finance, 5:347–359, 1998.

[163] S. Gervais, R. Kaniel, and D.H. Mingelgrin. The high-volume return premium.

Journal of Finance, 56(3):877–919, 2001.

[164] S. Gervais, R. Kaniel, and D.H. Mingelgrin. The high-volume return premium.

Journal of Finance, 56:877–919, 2001.

[165] E. Ghysels, C. Gourieroux, and J. Jasiak. Stochastic volatility duration models.

Journal of Econometrics, 119:413–433, 2004.

[166] M.R. Gibbons, S.A. Ross, and J. Shanken. A test of the eﬃciency of a given

portfolio. Econometrica, 57:1121–1152, 1989.

[167] L.R. Glosten and P.R. Milgrom. Bid, ask and transaction prices in a special-
ist market with heterogeneously informed traders. Journal of Financial Eco-
nomics, 14:71–100, 1985.

62

Bibliography

[168] I. Goodfellow, Y. Bengio, and A. Courville. Deep Learning. M.I.T. Press,

Boston, 2016.

[169] R.C. Grinold and R.N. Kahn. Active Portfolio Management, Second Edition.

McGraw-Hill, 2000.

[170] A. Gross-Klussmann and N. Hautsch. When machines read the news: Using
automated text analytics to quantify high frequency news-implied market reac-
tions. Journal of Empirical Finance, 18:321–340, 2011.

[171] A.D. Hall and N. Hautsch. Order aggressiveness and order book dynamics.

Empirical Economics, 30:973–1005, 2006.

[172] J.D. Hamilton. A new approach to the economic analysis of nonstationary time

series and the business cycle. Econometrica, 57:357–384, 1989.

[173] J.D. Hamilton. Analysis of time series subject to changes in regime. Journal

of Econometrics, 45:39–70, 1990.

[174] J.D. Hamiton. Macroeconomic regimes and regime shifts. In J.B. Taylor and
H. Uhlig, editors, Handbook of Macroeconomics, chapter 3, pages 153–201.
Elsevier, 2016.

[175] Y. Han, K. Yang, and G. Zhou. A new anomaly: The cross-sectional prof-
itability of technical analysis. Journal of Financial and Quantitative Analysis,
48:1433–1461, 2013.

[176] P.R. Hansen. A test for superior predictive ability. Journal of Business &

Economic Statistics, 23:365–380, 2005.

[177] M. O’ Hara and M. Ye.

Is market fragmentation harming market quality?

Journal of Financial Economics, pages 454–474, 2011.

[178] L. Harris. Trading and Exchanges: Market Microstructure for Practitioners.

Oxford University Press, New York, 2003.

[179] A.C. Harvey. Forecasting, Structural Time Series Models and the Kalman

Filter. Cambridge University Press, Cambridge, 1989.

[180] J. Hasbrouck. Measuring the information content of stock trades. Journal of

Financial Economics, 46:179–207, 1991.

[181] J. Hasbrouck. One security, many markets: Determining the contributions to

price discovery. Journal of Finance, 50(4):1175–1199, 1995.

[182] J. Hasbrouck and G. Saar. Technology and liquidity provision: The blurring

of traditional deﬁnitions. Journal of Financial Markets, 12:143–172, 2009.

[183] J. Hasbrouck and D.J. Seppi. Common factors in prices, order ﬂows, and liq-

uidity. Journal of Financial Economics, 59:383–411, 2001.

Bibliography

63

[184] T. Hastie, R. Tibshirani, and J. Friedman. The Elements of Statistical Learning;
Data Mining, Inference and Prediction, Second Edition. Springer-Verlag, New
York, 2009.

[185] N. Hautsch and R. Huang. The market impact of a limit order. The Journal of

Economic Dynamics and Control, 36:501–522, 2012.

[186] A.G. Hawkes. Spectra of some self-exciting and mutually exciting point pro-

cesses. Biometrika, 58:83–90, 1971.

[187] A.G. Hawkes. Hawkes processes and their applications to ﬁnance; a review.

Quantitative Finance, 18:193–198, 2018.

[188] X. He and R. Velu. Volume and volatility in a common-factor mixture of
distributions model. Journal of Financial and Quantitative Analysis, 49:33–
49, 2014.

[189] I.S. Helland. On the structure of partial least squares regression. Communica-
tions in Statistics, Simulation and Computation, B17:581–607, 1988.

[190] I.S. Helland. Partial least squares regression and statistical models. Scandina-

vian Journal of Statistics, 17:97–114, 1990.

[191] T. Hendershott, J. Brogaard, and R. Riordon. High frequency trading and price

discovery. The Review of Financial Studies, 27:2267–2306, 2014.

[192] T. Hendershott, C.M. Jones, and A.J. Menkveld. Does algorithmic trading

improve liquidity? Journal of Finance, 66(1):1–33, 2011.

[193] T. Hendershott, C.M. Jones, and A.J. Menkveld. Does algorithmic trading

improve liquidity? Journal of Finance, pages 1–3, 2011.

[194] T. Hendershott and R. Riordan.

Algorithmic trading and information.

http://faculty.haas.berkeley.edu/hender/ATInformation.pdf, 2011.

[195] T. Hendershott and M. Seasholes. Market maker inventories and stock prices.

American Economic Review, 97:210–214, 2007.

[196] T. Ho and H.R. Stoll. Optimal dealer pricing under transactions and return

uncertainty. Journal of Financial Economics, 9:47–73, 1981.

[197] T. Ho, R. Schwartz, and D. Whitcomb. The trading decision and market clear-
ing under transaction price uncertainty. The Journal of Finance, 40:21–42,
1985.

[198] S. Hogan, R. Jarrow, M. Teo, and M. Warachka. Testing market eﬃciency
using statistical arbitrage with application to momentum and value strategies.
Journal of Financial Economics, 73:525–565, 2004.

[199] R.W. Holthausen, R.W. Leftwich, and D. Mayers. Large-block transactions, the
speed of response, and temporary and permanent stock-price eﬀect. Journal
of Financial Economics, 26:71–95, 1990.

64

Bibliography

[200] H. Hotelling. Analysis of a complex of statistical variables into principal com-
ponents. Journal of Educational Psychology, 4:417–441, 498–520, 1933.

[201] H. Hotelling. The most predictable criterion. Journal of Educational Psychol-

ogy, 26:139–142, 1935.

[202] H. Hotelling. Relations between two sets of variables. Biometrika, 28:321–

322, 1936.

[203] P.H. Hsu, Y.C. Hsu, and C.M. Kuan. Testing the Predictive Ability of Technical
Analysis Using a New Stepwise Test Without Data Snooping Bias. Journal of
Empirical Finance, 17:471–484, 2010.

[204] Y.-P. Hu and R.S. Tsay. Principal volatility component analysis. Journal of

Business & Economic Statistics, 32:153–164, 2014.

[205] R. Huang and H. Stoll. Dealer versus auction markets: A paired comparison of
execution costs on NASDAQ and the NYSE. Journal of Financial Economics,
41:313–357, 1996.

[206] W. Huang, C-A. Lehalle, and M. Rosenbaum. Simulating and analyzing order
book data: The queue-reactive model. Journal of the American Statistical
Association, 110:107–122, 2015.

[207] D. Huang, F. Jiang, J. Tu, and G. Zhou. Investor sentiment aligned: A powerful
predictor of stock returns. The Review of Financial Studies, 28:791–837, 2015.

[208] G. Huberman. Familiarity breeds investment. The Review of Financial Studies,

14:659–680, 2001.

[209] G. Huberman and W. Stanzl. Price manipulation and quasi-arbitrage. Econo-

metrica, 72:1247–1275, 2004.

[210] S. Hvidkjaer. A trade-based analysis of momentum. The Review of Financial

Studies, 119:457–491, 2006.

[211] R. Israel, T. Moskowitz, A. Ross, and L. Serban. Implementing momentum:

What have we learned. NBER Working Paper, 2017.

[212] R. Jagannathan and T. Ma. Risk reduction in large portfolios: Why imposing
the wrong constraints helps. Journal of Finance, 58:1651–1683, 2003.

[213] C.M. Jarque and A.K. Bera. Eﬃcient tests for normality, homoscedasticity
and serial independence of regression residuals. Economic Letters, 6:255–259,
1980.

[214] N. Jegadeesh. Discussion of LMW (2000). Journal of Finance, pages 1765–

1770, 2000.

[215] N. Jegadeesh and S. Titman. Returns to buying winners and selling losers:
Implications for stock market eﬃciency. Journal of Finance, 48:65–91, 1993.

Bibliography

65

[216] N. Jegadeesh and S. Titman. Proﬁtability of momentum strategies: An evalu-
ation of alternative explanations. Journal of Finance, 56:699–720, 2001.

[217] N. Jegadeesh and S. Titman. Cross-sectional time series determinants of
momentum returns. The Review of Financial Studies, 15:143–157, 2002.

[218] F. Jiang, J. Lee, X. Martin, and G. Zhou. Manager sentiment and stock returns.

Journal of Financial Economics, 132:126–149, 2019.

[219] W. Jiang, L. Shu, and D.W. Apley. Adaptive CUSUM procedures with EWMA-

based shift estimators. IIE Transactions, 40:992–1003, 2008.

[220] J.D. Jobson and B. Korkie. Estimation for markowitz eﬃcient portfolios. The

Journal of American Statistical Association, 75:544–554, 1980.

[221] S. Johansen. Statistical analysis of co-integration vectors. The Journal of

economic dynamics and control, 12(2):231–254, 1988.

[222] S. Johansen. Estimation and hypothesis testing of co-integration vectors in
Gaussian vector autoregressive models. Econometrica: Journal of the Econo-
metric Society, pages 1551–1580, 1991.

[223] C. Jones, G. Kaul, and M. Lipson. Information, trading and volatility. Journal

of Financial Economics, 36:127–154, 1994.

[224] L.P. Kaelbling, M.L. Littman, and A.W. Moore. Reinforcement learning: A

survey. Journal of Artiﬁcial Intelligence, 4:237–285, 1996.

[225] R.N. Kahn and M. Lemmon. Smart beta: The owner’s manual. The Journal of

Portfolio Management, 41(2):76–83, 2015.

[226] H. Kawakatsu. Direct multiperiod forecasting for algorithmic trading. Journal

of Forecasting, 37(1):83–101, 2018.

[227] D.B. Keim and A. Madhavan. The upstairs market for large-block transactions:
Analysis and measurement of price eﬀects. The Review of Financial Studies,
9:1–36, 1996.

[228] D.B Keim and A. Madhavan. Transaction costs and investment style: An inter-
exchange analysis of institutional equity trades. Journal of Financial Eco-
nomics, 46:265–292, 1997.

[229] J.L. Kelly. A new interpretation of information rate. Bell System Technical

Journal, 35:917–926, 1956.

[230] J.M. Keynes. The general theory of employment. The Quarterly Journal of

Economics, pages 209–223, 1937.

[231] P.D. Koch and T.W. Koch. Evolution in dynamic linkages across daily national
stock indexes. Journal of International Money and Finance, 10.2:231–251,
1991.

66

Bibliography

[232] A. Kourtis. On the distribution and estimation of trading costs. Journal of

Empirical Finance, 29:230–245, 2014.

[233] P. Kratz and T. Schöneborn. Optimal liquidity in dark pools. Quantitative

Finance, 14:1519–1539, 2014.

[234] A. Kyle. Continuous time auctions and insider trading. Econometrics,

53:1315–1336, 1985.

[235] T.L. Lai and H. Xing. Statistical Models and Methods for Financial Markets.

Springer, 2008.

[236] T.L. Lai and H. Xing. Stochastic change-point ARX GARCH models and their
applications to econometric times series. Statistica Sinica, 23:1573–1594,
2013.

[237] T.L. Lai, H. Xing, and Z. Chen. Mean-variance portfolio optimization when
means and covariances are unknown. The Annals of Applied Statistics, 5:798–
823, 2011.

[238] J. Lakonishok, A. Shleifer, and R.W. Vishny. Contrarian investment, extrapo-

lation, and risk. Journal of Finance, 49(5):1541–1578, 1994.

[239] O. Ledoit and M. Wolf.

Improved estimation of the covariance matrix of
stock returns with an application to portfolio selection. Journal of Empirical
Finance, 10:603–621, 2003.

[240] C.M. Lee and B. Swaminathan. Price momentum and trading volume. Journal

of Finance, LV:2017–2069, 2000.

[241] C.M. Lee and M. Ready. Inferring trade direction from intraday data. Journal

of Finance, 46:733–746, 1991.

[242] J. Lewellen. Momentum and autocorrelation in stock returns. The Review of

Financial Studies, 15:65–91, 2002.

[243] J.K.-S. Liew, S. Guo, and T. Zhang. Tweet sentiments and crowd-sourced
earnings estimates as valuable sources of information around earnings releases.
The Journal of Alternative Investments, Winter Issue:1–20, 2017.

[244] F. Lillo, J.D. Farmer, and R. Mantegna. Master curve for price impact function.

Nature, 421:129–130, 2003.

[245] J. Lintner. The valuation of risky assets and the selection of risky investment
in stock portfolios and capital budgets. Review of Economics and Statistics,
47:13–37, 1965.

[246] G. Llorente, R. Michaely, G. Saar, and J. Wang. Dynamic volume-return rela-
tion of individual stocks. The Review of Financial Studies, 15:1005–1047,
2002.

Bibliography

67

[247] A.W. Lo, H. Mamaysky, and J. Wang. Foundation of technical analysis:
Computational algorithms, statistical inference and empirical implementation.
Journal of Finance, LV(4):1705–1765, 2000.

[248] A.W. Lo. The statistics of Sharpe ratios. Financial Analysis Journal, pages

36–52, 2002.

[249] A.W. Lo and A.C. MacKinlay. When are contrarian proﬁts due to stock market

overreaction? The Review of Financial Studies, 3:175–205, 1990.

[250] A.W. Lo, A.C. MacKinlay, and J. Zhang. Econometric models of limit-order

executions. Journal of Financial Economics, 65:31–71, 2002.

[251] T.F. Loeb. Trading cost: The critical link between investment information and

results. Financial Analyst Journal, 39(3):39–44, 1983.

[252] M. Lopez de Prado. Advances in Financial Machine Learning. John Wiley &

Sons, New Jersey, 2018.

[253] J. Lorenz and R. Almgren. Mean-variance optimal adaptive execution. Applied

Mathematical Finance, 18(4):395–422, 2011.

[254] A. Madhavan. Market microstructure. Journal of Financial Markets, 3:205–

258, 2000.

[255] A. Madhavan. VWAP strategies. Trading, 1:32–39, 2002.

[256] A. Madhavan, M. Richardson, and M. Roomans. Why do security prices
change? A transaction-level analysis of NYSE stocks. The Review of Financial
Studies, 10(4):1035–1064, 1997.

[257] C. Maglaras, C.C. Moallemi, and H. Zheng. Queueing dynamics and state
space collapse in fragmented limit order book markets. Operations Research
(to appear), 2019.

[258] B.G. Malkiel. A Random Walk Down Wall Street: The Time-Tested Strategy

for Successful Investing, 10th Edition. Norton, 2012.

[259] B. Mandelbrot. The variation of certain speculative prices. The Journal of

Business, 36:294–319, 1963.

[260] H. Markowitz. Portfolio Selection: Eﬃcient Diversiﬁcation of Investments.

Wiley: New York, 1959.

[261] M. Martens and D. van Dijk. Measuring volatility with the realized range.

Journal of Econometrics, 138:181–207, 2007.

[262] R. McCulloch and R. Tsay. Nonlinearity in high-frequency ﬁnancial data
and hierarchical models. Studies in Nonlinear Dynamics and Econometrics,
5:1067–1077, 2001.

68

Bibliography

[263] H. Mendelson. Market behavior in a clearing house. Econometrica: Journal of

the Econometric Society, 1505–1524, 1982.

[264] L. Menkhoﬀ, L. Sarno, M. Schmeling, and A. Schrimpf. Currency momentum

strategies. Journal of Financial Economics, 106:660–684, 2012.

[265] L. Menkhoﬀ and M.P. Taylor. The obstinate passion of foreign exchange pro-
fessionals: Technical analysis. The Journal of Economic Literature, XLV:936–
972, 2007.

[266] A.J. Menkveld, S.J. Koopman, and A. Lucas. Modeling around-the-clock price
discovery for cross-listed stocks using state space methods. Journal of Busi-
ness & Economic Statistics, 25:213–225, 2007.

[267] R.C. Merton. On estimating the expected return on the market. Journal of

Financial Economics, 8:323–336, 1980.

[268] R.T. Merton. An intertemporal capital asset pricing model. Econometrica,

41:867–887, 1973.

[269] R.O. Michaud. Eﬃcient Asset Management. Harvard Business School Press,

Boston, 1989.

[270] G. Mitra and L. Mitra. The Handbook of News Analytics in Finance (Ed).

Wiley Finance, 2011.

[271] T. Moorman. An empirical investigation of methods to reduce transaction

costs. Journal of Empirical Finance, 29:230–245, 2014.

[272] E. Moro, J. Vicente, L.G. Moyano, A. Gerig, J.D. Farmer, G. Vaglica, F. Lillo,
and R.N. Mantegna. Market impact and trading proﬁle of hidden orders in
stock markets. Physical Review E, 80(6):066102, 2009.

[273] T.J. Moskowitz, Y.H. Ooi, and L.H. Pedersen. Time Series Momentum. Jour-

nal of Financial Economics, 104:228–250, 2012.

[274] A.A. Obizhaeva and J. Wang. Optimal trading strategy and supply/demand

dynamics. Journal of Financial Markets, 16:1–32, 2013.

[275] M. O’Hara. Presidential address: Liquidity and price discovery. Journal of

Finance, 58:1335–1354, 2003.

[276] M. O’Hara. High frequency market microstructure. Journal of Financial Eco-

nomics, 116:257–270, 2015.

[277] J. Okunev and D. White. Do momentum - based strategies still work in foreign
currency markets? Journal of Financial and Quantitative Analysis, 38:425–
447, 2003.

Bibliography

69

[278] J.K. Ord, A.B. Koehler, and R.D. Snyder. Estimation and prediction for a class
of dynamic nonlinear statistical models. Journal of the American Statistical
Association, 92(440):1621–1629, 1997.

[279] D.A. Pachamanova and F.J. Fabozzi. Recent trends in equity portfolio con-
struction analytics. The Journal of Portfolio Management, 40:137–151, 2014.

[280] A. Pardo and R. Pascual. On the hidden side of liquidity. The European

Journal of Finance, 18:949–967, 2012.

[281] C. Parlour and D. Seppi. Liquidity-based competition for order ﬂow. The

Review of Financial Studies, 16:301–343, 2003.

[282] C.A. Parlour and D.J. Seppi. Limit Order Markets – A Survey, In Handbook
of Financial Intermediation and Banking, Edited by A. Thakor and A. Boot.
Elsevier, Amsterdam, 2008.

[283] L. Pastor and R.F. Stambough. The equity premium and structural breaks.

Journal of Finance, 56:1207–1239, 2001.

[284] A.J. Patton and A. Timmermann. Monotonicity in asset returns: New tests with
applications to the term structure, the CAPM, and the portfolio sorts. Journal
of Financial Economics, 98:605–625, 2010.

[285] R.L. Peterson. Trading on Sentiment: The Power of Minds over Markets. Wiley

Finance, 2016.

[286] M.J. Ready. Proﬁts from technical trading rules. Financial Management,

Autumn:43–61, 2002.

[287] G.C. Reinsel and S.K. Ahn. Vector autoregressive models with unit roots and
reduced rank structure: estimation, likelihood ratio test, and forecasting. The
Journal of Time Series Analysis, 13(4):353–375, 1992.

[288] G.C. Reinsel. Element of Multivariate Time Series Analysis, Second Edition.

Springer-Verlag, New York, 2002.

[289] G.C. Reinsel and R. Velu. Multivariate Reduced-Rank Regression, Theory and

Application. Springer-Verlag, New York, 1998.

[290] L.C.G. Rogers and S.E. Satchell. Estimating variance from high, low and clos-

ing prices. Annals of Applied Probability, 1:504–512, 1991.

[291] R. Roll. A simple model of the implicit bid-ask spread in an eﬃcient market.

Journal of Finance, 39:1127–1139, 1984.

[292] J.P. Romano and M. Wolf. Stepwise multiple testing as formalized data snoop-

ing. Econometrica, 73:1237–1282, 2005.

[293] S.A. Ross. The arbitrage theory of capital asset pricing. The Journal of Eco-

nomic Theory, 13:341–360, 1976.

70

Bibliography

[294] I. Rosu. A dynamic model of the limit order book. The Review of Financial

Studies, 22:4601–4641, 2009.

[295] T.H. Rydberg and N. Shephard. Dynamic trade-by-trade price movement:
Decomposition and models. Journal of Financial Econometrics, 1:2–25, 2003.

[296] S. Satchell and A. Snowcraft. A demystiﬁcation of the Black-Litterman model:
Managing quantitative and traditional portfolio construction. Journal of Asset
Management, 1:138–150, 2000.

[297] V. Satish, A. Saxena, and M. Palmer. Predicting intraday trading volume and

volume percentages. The Journal of Trading, 9:15–25, 2014.

[298] M. Schneider and F. Lillo. Cross-impact and no-dynamics arbitrage. Quanti-

tative Finance, 19:137–154, 2019.

[299] G. Schwarz. Estimating the dimension of a model. Annals of Statistics, 6:401–

404, 1978.

[300] J.T. Scruggs. Noise trader risk: Evidence from the siamese twins. Journal of

Financial Markets, 10:76–105, 2007.

[301] W.F. Sharpe. Capital asset prices: A theory of market equilibrium under con-

ditions of risk. Journal of Finance, 19:425–442, 1964.

[302] R.H. Shumway and D.S. Stoﬀer. Arima models. In Time Series Analysis and

Its Applications, pages 83–171. Springer, 2011.

[303] D. Smith, N. Wang, Y. Wang, and E.J. Zychowicz. Sentiment and the eﬀective-
ness of Technical Analysis: Evidence from the Hedge Fund Industry. Journal
of Financial and Quantitative Analysis, 51:1991–2013, 2016.

[304] E. Smith, D.J. Farmer, L. Gillemot, and S. Krishnamurthy. Statistical theory

of the continuous double auction. Quantitative Finance, 3:481–514, 2003.

[305] R. Stambaugh, J. Yu, and Y. Yuan. The short of it: Investor sentiment and

anomalies. Journal of Financial Economics, 104:288–302, 2012.

[306] M. Statman, S. Thorley, and K. Vorkink. Investor overconﬁdence and trading

volume. The Review of Financial Studies, 19(4):1531–1565, 2006.

[307] S. Stoikov. The micro-price: A high-frequency estimator of future prices.

Quantitative Finance, 18:1959–1966, 2018.

[308] R. Sullivan, A. Timmermann, and H. White. Data snooping, technical trading
rule performance, and the bootstrap. Journal of Finance, 54:1647–1691, 1999.

[309] R.S. Sulton and A.G. Barks. Reinforcement Learning, An Introduction, Second

Edition. MIT Press, Boston, 2018.

Bibliography

71

[310] R.J. Sweeney. Some new ﬁlter rule tests: Methods and results. Journal of

Financial and Quantitative Analysis, 20(3):285–300, 1988.

[311] G.E. Tauchen and M. Pitts. The price variability-volume relationship on spec-

ulative markets. Econometrica, 51:485–505, 1983.

[312] P.C. Tetlock. Giving content to investor sentiment: The role of media in the

stock market. Journal of Finance, 62(3):1139–1168, 2007.

[313] I.M. Toke. An introduction to Hawkes processes with applications to ﬁnance,

2011. Lecture Notes from Ecole Centrale Paris, BNP, Paribas.

[314] B. Toth, Y. Lemperiere, C. Deremble, J. De Lataillade, J. Kockelkoren, and
J.-P. Bouchaud. Anomalous price impact and the critical nature of liquidity in
ﬁnancial markets. Physical Review X, 1(2):021006, 2011.

[315] R. Tsay. Analysis of Financial Time Series, Third Edition. Wiley, 2010.

[316] G. Tsoukalas, J. Wang, and K. Giesecke. Dynamics portfolio execution. Man-

agement Science, 2019.

[317] J. Tu and G. Zhou. Markowitz meets talmud: A combination of sophisticated
and naïve diversiﬁcation. Journal of Financial Economics, 99:204–215, 2011.

[318] F. Vahid and R.F. Engle. Common trends and common cycles. The Journal of

Applied Econometrics, 8:341–360, 1993.

[319] R. Velu, A. Gretchika, M. Benaroch, D. Nehren, and K. Kuber. Market Impact:
To Trade Small or to Trade Seldom? Evidence from Algorithmic Execution
Data. Unpublished, 2015.

[320] B. von Beschwitz, D.B. Keim, and M. Massa. First to “read” the news: News
analytics and algorithmic trading. Review of Financial Studies (to appear),
2019.

[321] A.A. Weiss. ARMA models with ARCH errors. The Journal of Time Series

Analysis, 5:129–143, 1984.

[322] M. West and J. Harrison. Bayesian Forecasting and Dynamic Models, Second

Edition. Springer-Verlag, New York, 1997.

[323] H. White. A reality check for data snooping. Econometrica, 68:1097–1126,

1999.

[324] R. De Winne and C. D’Hondt. Hide-and-seek in the market: Placing and detect-

ing hidden order. Review of Finance, 11:663–692, 2007.

[325] H. Wold. PLS regression, volume 6. Eds. N.L. Johnson and S. Kotz, 1984.

[326] H. Working. Note on the correlation of ﬁrst diﬀerences of averages in a random
chain. Econometrica: Journal of the Econometric Society, pages 916–918,
1960.

72

Bibliography

[327] D. Yang and Q. Zhang. Drift-independent volatility estimation based on high,
low, open and close prices. The Journal of Business, 73:477–491, 2000.

[328] J.W. Yang. Transaction duration and asymmetric price impact of trades - Evi-
dence from Australia. Journal of Empirical Finance, 18:91–102, 2011.

[329] J. Yu and Y. Yuan. Investor sentiment and the mean–variance relation. Journal

of Financial Economics, 100:367–381, 2011.

[330] E. Zarinelli, M. Treccani, J.D. Farmer, and F. Lillo. Beyond the square root:
Evidence for logarithmic dependence of market impact on size and participa-
tion rate. Market Microstructure and Liquidity, 1:1–31, 2015.

[331] G. Zhou. Measuring investor sentiment. Annual Review of Financial Eco-

nomics, 10:239–259, 2018.

[332] Y. Zhu and G. Zhou. Technical analysis: An asset allocation perspective on the
use of moving averages. Journal of Financial Economics, 92:519–544, 2009.
