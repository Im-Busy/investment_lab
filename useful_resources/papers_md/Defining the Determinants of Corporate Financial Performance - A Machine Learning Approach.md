# Defining the Determinants of Corporate Financial Performance - A Machine Learning Approach

> *Source PDF: Defining the Determinants of Corporate Financial Performance - A Machine Learning Approach.pdf*
> *Extraction: Combined — markitdown raw text (base) + previous extraction supplements*

---

Available online at www.ijournalse.org

Emerging Science Journal
(ISSN: 2610-9182)

Vol. 9, No. 4, August, 2025

Defining the Determinants of Corporate Financial Performance:
A Machine Learning Approach

Idelia R. Badykova 1* , Aliya A. Dinmukhametova 1

1 Department of Business Statistics and Economics, Kazan National Research Technological University, Kazan, Russia.

Abstract

This study investigates the determinants of corporate financial performance (CFP) among Russian
enterprises (2012–2023) through the lens of geopolitical disruptions, employing ensemble machine
learning  (ML)  to  address  methodological  gaps  in  modeling  non-linear  institutional  interactions.
Using  data  from  25  large  non-financial  firms,  we  analyze  sectoral,  organizational,  and  strategic
drivers, integrating train-test splits (75%/25%) and 10-fold cross-validation to mitigate overfitting.
Results  reveal  that  industry  affiliation,  initially  dominant  (28%  explanatory  power  pre-2022),
declined sharply post-sanctions (15%), reflecting vulnerabilities in globally integrated sectors like
manufacturing  and  extractives.  Organizational  size  exhibited  a  nonlinear  relationship  with  CFP,
favoring  comparatively  smaller  firms’  agility  over  larger  enterprises’  rigidity,  consistent  with
transaction  cost  economics.  Strategic  investments  in  corporate  social  responsibility  (CSR)  and
research  and  development  (R&D)  diminished  post-2022  as  firms  prioritized  liquidity  and
operational  stability,  aligning  with  resource-based  view  principles.  Methodologically,  Shapley
Additive Explanations (SHAP) clarified threshold effects in CSR returns and innovation’s reduced
role  under  sanctions.  The  study  innovates  by  applying  ensemble  machine  learning  to  sanction-
affected emerging markets, challenging linear econometric assumptions and advancing institutional
theory through a crisis-contextualized framework of resource dependence and stakeholder salience.
Findings underscore the fragility of intangible assets under systemic shocks and advocate adaptive
resource allocation frameworks to balance short-term survival with long-term resilience. This work
provides  policymakers  and  managers  actionable  insights  for  fostering  operational  agility  and
strategic foresight in volatile institutional environments.

Keywords:

Corporate Financial Effectiveness;

Ensemble Machine Learning;

Geopolitical Risk;

Sanctions; Resource Dependence Theory;

Resource-Based View (RBV);

Shapley Additive Explanations (SHAP);

Russian Enterprises;

Emerging Markets.

Article History:

Received:

Revised:

Accepted:

Published:

25

10

17

01

April

July

July

August

2025

2025

2025

2025

1- Introduction

Corporate  Financial  Performance  (CFP)  remains  a  pivotal  indicator  of  organizational  resilience,  synthesizing
operational  efficiency,  governance  quality,  and  macroeconomic  adaptability  [1,  2].  Grounded  in  foundational
principles of financial analysis [3] and capital structure theory [4], CFP evaluation has evolved to incorporate dynamic
capabilities  [5]  and  institutional  transitions  [6].  While  existing  studies  have  mapped  key  CFP  determinants  using
traditional  econometric  methods  [7,  8],  including  capital  structure  dynamics  [9],  the  application  of  advanced
computational  techniques  –  particularly  ensemble  machine  learning  (ML)  –  to  dissect  complex,  non-linear
relationships in crisis contexts remains nascent [10, 11]. Recent sector-specific applications, such as Yıldırım et al.
(2024)  [11],  demonstrate  ML’s  efficacy  in  ranking  financial  performance  determinants  within  energy  sectors,
highlighting the method’s potential for domain-specific insights. Theoretical frameworks such as agency theory [12]
and  stakeholder  theory  [13,  14]  underscore  the  multifaceted  nature  of  CFP  determinants,  with  evolutionary
perspectives highlighting adaptive processes [15].

* CONTACT: badykovair@corp.knrtu.ru

DOI: http://dx.doi.org/10.28991/ESJ-2025-09-04-01
©  2025  by  the  authors.  Licensee  ESJ,  Italy.  This  is  an  open  access  article  under  the  terms  and  conditions  of  the  Creative
Commons Attribution (CC-BY) license (https://creativecommons.org/licenses/by/4.0/).

Page | 1764

Emerging Science Journal | Vol. 9, No. 4

Prior work relies heavily on linear regression, which struggles to capture non-linear interactions and threshold effects
– critical in sanction-driven economies where variables like research and development (R&D) spending may exhibit
discontinuous impacts [16, 17]. For example, the resource-based view (RBV) [18] highlights R&D’s role in competitive
advantage,  but  liquidity  constraints  under  sanctions  may  distort  this  relationship  [19],  complicating  the  curvilinear
association between social investments and  financial outcomes [20]. Emerging methodologies, such as deep learning
architectures integrating environmental, social, and governance (ESG) data [21], offer novel pathways to quantify these
interactions,  particularly  in  forecasting  performance  under volatility. Frequently  traditional  frameworks  inadequately
model how internal factors (e.g., liquidity) and external shocks (e.g., sanctions) interactively amplify or mitigate risks.
For  instance,  corporate  social  responsibility’s  (CSR)  role  may  shift  from  profit-enhancing  to  survival-critical  under
sanctions [22, 23], a transition stakeholder theory attributes to shifting salience among competing claims [14] as firms
prioritize short-term legitimacy over long-term ethics [24] – a dynamic linear models fail to disentangle. Aksoy and
Yilmaz (2024) [23] further emphasize the moderating role of sustainability performance in crisis resilience, revealing
that firms with robust ESG practices mitigated pandemic-induced financial losses more effectively.

While  sector-specific  vulnerabilities  are  acknowledged  [25],  predictive  analytics  under  systemic  shocks  are  rare.
Institutional theory [6, 26] suggests firms in crisis adopt isomorphic survival strategies during institutional transitions,
yet  such  adaptations  remain  poorly  quantified.  Russia’s  post-2022  environment,  marked  by  abrupt  supply  chain
disruptions and ownership restructuring [27, 28], demands tools that handle high-dimensional, noisy data [29]. Despite
ML’s rise in finance, few studies employ ensemble techniques to improve robustness in CFP prediction, particularly in
sanction-affected  contexts  where  evolutionary  adaptations  occur. Hsu  et  al.  (2025)  [21]  advance  this  discourse  by
integrating ESG metrics into deep learning frameworks, demonstrating enhanced predictive accuracy for firms operating
in  regulated  industries.  Stewardship  theory  [30]  posits  that  managerial  autonomy  enhances  resilience,  but  existing
models  lack  the  granularity  to  test  this  in  high-stakes  environments.  Ensemble  models,  which  aggregate  multiple
algorithms (e.g., Random Forests, Gradient Boosting), remain untapped for isolating key drivers of financial resilience.

The aim of this paper is twofold: First, to empirically identify the non-linear, crisis-driven interactions between CFP
determinants – including R&D, CSR, leverage, and industry dynamics – in Russia’s post-2022 sanction context, using
ensemble  ML  to  overcome  limitations  of  traditional  econometrics.  Second,  to  theorize  how  institutional  upheavals
reshape the relevance of foundational frameworks (agency, stakeholder, RBV) in emerging markets, particularly through
the  lens  of  dynamic  capabilities  [5].  By  bridging  the  post-2022  empirical  void,  this  study  advances  CFP  theory  by
contextualizing universal determinants within institutional upheavals [1, 2]. Recent empirical work highlights the duality
of firm-specific factors: larger firms benefit from economies of scale [31] but face innovation inertia as they age [32],
while liquidity management balances growth opportunities against debt risks [19]. Industry dynamics further modulate
outcomes – emerging sectors thrive on disruption through dynamic capabilities [5], whereas regulated industries face
structural constraints [33].

Theoretical synthesis reveals competing priorities: agency theory advocates governance controls to curb managerial
excess [34], while stewardship theory emphasizes trust-based autonomy [35]. Stakeholder theory bridges profit motives
with  ESG  considerations  [14,  36],  yet  CSR’s  financial  returns  remain  contested  [10,  37,  38].  These  tensions  are
magnified in crises, where survival imperatives reshape strategic investments [31].

The remainder of this paper proceeds as follows: Section 2 outlines the data and methodology, Section 3 presents the

empirical results and discussion, and Section 4 concludes with theoretical and policy implications.

2- Data and Methodology

To  conduct  the  empirical  analysis,  data  spanning  2012–2023  were  collected  from  official  corporate  reports  and
websites of 25 large Russian non-financial companies, yielding 300 firm-year observations (see Table 1). The restricted
sample size reflects the voluntary disclosure practices governing CSR and innovation metrics in Russia, where dominant
market actors disproportionately publish such data. While this limitation introduces potential selection bias, the dataset
was retained to align with the study’s focus on CSR and innovation as critical determinants of firm performance. Despite
the modest sample, ensemble machine learning (ML) methods remain methodologically appropriate due to their inherent
compatibility with bootstrap resampling techniques, which mitigate overfitting risks in small-n contexts.

The market-to-book (MTB) ratio was selected as the dependent variable, operationalizing firm value as the ratio of
market capitalization to book equity. This metric was prioritized over profitability-based indicators (e.g., ROA, ROE)
due to its relative stability during macroeconomic volatility, as it captures investor expectations of future growth rather
than transient accounting performance [3, 39].

Explanatory variables encompass both financial and non-financial dimensions: EBITDA margin, company size, age,

industry affiliation, financial leverage, innovation activity, CSR expenditures, and observation year (2012–2023).

Control variables such as industry and year were included to isolate firm-specific effects from sectoral trends and
temporal shocks. Financial leverage and EBITDA margin account for capital structure and operational efficiency, while
innovation and CSR expenditures test hypotheses derived from resource-based and stakeholder theories.

Page | 1765

Table 1. Descriptive statistics of data for 25 large Russian enterprises from 2012 to 2023

Description

Designation
on the graph

Mean

std

Min

Median

Max

Company affiliation

Company

13.039

7.187

1.000

13.000

25.000

Emerging Science Journal | Vol. 9, No. 4

Indicators

Company

Year

Innovations

R&D expenses / Revenue

Number of the year

Year

Innov

2017.4

3.470

2012.0

2017.0

2023.0

0.002

0.005

0.000

0.001

0.043

EBITDA margin

EBITDA / Revenue

Ebitda Margin

0.268

0.136

0.000

0.262

0.647

Size

CSR

Natural logarithm of revenue

CSR expenses / Revenue

Size

Csr

12.098

3.106

0.000

12.609

16.273

0.015

0.159

0.000

0.002

2.667

CSR (staff)

Education Expenditures / Wage Fund

Csrstaff

0.005

0.017

0.000

0.002

0.196

M/B

Age

Capitalization / Book value

Number of years since the company's registration

Leverage

Ratio of the sum of short-term and long-term loans
to the sum of equity and borrowed funds

m/b

Age

Lev

1.109

0.908

0.000

0.915

3.980

13.227

9.993

0.000

16.000

28.000

0.383

0.235

0.000

0.367

0.928

Belonging to the sector:

Industry
characteristic

1: Extractive;

2: Manufacturing;

3: Transport and telecommunications

Field

1.461

0.827

0.000

1.000

3.000

Note: Compiled by the authors based on calculations obtained using the Python programming language.

The present investigation focused on two key variables within the framework of CSR. The first variable analyzed was
the proportion of CSR expenditures relative to enterprise revenue, a metric that provides foundational insight into the
organization’s commitment to social responsibility initiatives. The second variable examined the relationship between
employee  training  costs  and  the  labor  remuneration  fund,  grounded  in  the  premise  that  investments  in  workforce
development represent a critical dimension of CSR with the potential to enhance organizational efficiency. Additionally,
innovation was operationalized through the ratio of R&D expenditures to revenue, a widely recognized indicator for
assessing  technological  advancement  and  its  contribution  to  economic  growth.  These  metrics  collectively  aimed  to
evaluate the interplay between CSR practices, human capital investment, and innovation in driving organizational and
economic outcomes.

The statistical analysis reveals that the innovation, CSR, and CSRstaff variables exhibit mean values exceeding their
respective medians, consistent with a positively skewed distribution characterized by a concentration of outliers in the
upper  range.  This  asymmetry  underscores  the  presence  of  firms  with  disproportionately  high  innovation  and  social
responsibility investments within the sample.

To address overfitting risks inherent in machine learning applications, two methodological strategies were employed:
First,  a  train-test  split  partitioned  the  dataset  into  a  training  subset  (n  =  225  observations,  75%  of  the  data)  and  an
independent  test  subset  (n  =  75  observations,  25%),  ensuring  that  model  performance  could  be  rigorously  assessed
against  unseen  data  to  evaluate  generalizability.  Second,  10-fold  cross-validation  was  applied  to  the  training  data,
iteratively dividing it into 10 mutually exclusive folds. For each iteration, one-fold served as a validation subset while
the remaining nine were used for model training. This protocol facilitated hyperparameter optimization while providing
a systematic evaluation of model stability across heterogeneous data partitions.

Figure 1 provides a schematic overview of the ensemble methodology.

Data → Train-Test Split → CV on Train → Ensemble Training → Test Evaluation

Figure 1. Flowchart of the ensemble methods process

The empirical analysis employed a multi-model framework to evaluate the hypotheses, with Table 2 summarizing
performance  metrics  –  including  mean  absolute  error  (MAE),  mean  squared  error  (MSE),  root  mean  squared  error
(RMSE), R-squared (R²), root mean squared logarithmic error (RMSLE), and mean absolute percentage error (MAPE)
– for all algorithms. Models were ranked by adjusted R² to prioritize explanatory power while penalizing overfitting, a
critical consideration given the study’s limited sample size.

Page | 1766

Emerging Science Journal | Vol. 9, No. 4

Table 2. Models and their quality metrics for the period of 2012-2023

Model

MAE

MSE

RMSE

R2

RMSLE  MAPE

Extra Trees Regressor

0.3970

0.3910

0.5932

0.4395

0.2438

0.3760

Random Forest Regressor

0.4424

0.4418

0.6397

0.3934

0.2662

0.4566

Light Gradient Boosting Machine

0.4425

0.4355

0.6311

0.3815

0.2621

0.4702

Extreme Gradient Boosting

0.4417

0.4462

0.6356

0.3561

0.2626

0.4342

Gradient Boosting Regressor

0.4518

0.4712

0.6511

0.2992

0.2698

0.5077

AdaBoost Regressor

0.5452

0.5254

0.7017

0.2457

0.3141

0.7427

K Neighbors Regressor

0.6055

0.6499

0.7883

0.1643

0.3446

0.7781

Ridge Regression

Decision Tree Regressor

Elastic Net

Lasso Regression

Lasso Least Angle Regression

Linear Regression

0.6315

0.4942

0.7167

0.7226

0.7226

0.7257

0.6806

0.5865

0.8394

0.8597

0.8597

2.4714

0.8068

0.7437

0.8851

0.8970

0.8970

0.1602

0.1419

0.0302

0.0051

0.0051

0.3612

0.3111

0.3951

0.4010

0.4010

1.1785

-4.2882

0.3937

0.8778

0.4986

0.9837

1.0157

1.0157

1.0056

s
d
o
h
t
e

M

e
l
b
m
e
s
n
E

s
d
o
h
t
e

M

l
a
n
o
i
t
i
d
a
r
T

Note: Calculated by the authors using the Python programming language

The top-performing cohort comprised six ensemble machine learning models, categorized into two methodological

groups:

1. Bagging algorithms: Extra Trees Regressor and Random Forest Regressor.

2. Boosting  algorithms:  Light  Gradient  Boosting  Machine  (LGBM),  Extreme  Gradient  Boosting  (XGBoost),
Gradient  Boosting  Regressor,  and  AdaBoost  Regressor,  which  iteratively  minimize  bias  via  error-correcting
ensembles.

Boosting  methods  train  homogeneous  models  sequentially,  with  each  iteration  addressing  residuals  from  prior
models. In contrast, Random Forest – a bagging technique – aggregates predictions from an ensemble of decision trees,
averaging  outputs  to  stabilize  results.  The  Extra  Trees  Regressor  diverges  from  Random  Forest  by  selecting  splits
randomly  rather  than  algorithmically,  accelerating  model  training  without  significant  performance  degradation.  This
feature renders it advantageous for large-scale datasets, though it typically omits bootstrapping.

Classical machine learning models were evaluated for benchmarking, including:

1. Linear regression variants: Ordinary Least Squares (OLS), Lasso, Least Angle Regression (LARS), Elastic Net,

and Ridge Regression.

2. Non-parametric methods: Decision Tree Regressor and K-Nearest Neighbors (KNN) Regressor.

Ensemble  methods  consistently  outperformed  classical  approaches  across  metrics  (Table  2),  though  all  models
exhibited modest predictive accuracy in the post-2022 sanction context. This contrasts with pre-2022 data (2012–2021),
where ensemble models achieved robust performance (Random Forest: adjusted R² = 0.73; Extra Trees: adjusted R² =
0.72;  Table  3).  The  decline  in  post-2022  explanatory  power  likely  reflects  exogenous  geopolitical  shocks,  which
introduced unobserved determinants of financial efficiency.

Table 3. Models and their quality metrics for the period of 2012-2021

Model

MAE

MSE

RMSE

R2

RMSLE  MAPE

Random Forest Regressor

0.3337

0.2277

0.4595

0.7275

0.1735

0.3478

Extra Trees Regressor

0.3314

0.2369

0.4680

0.7236

0.1700

0.3241

Extreme Gradient Boosting

0.3192

0.2257

0.4534

0.7209

0.1745

0.3300

Light Gradient Boosting Machine

Gradient Boosting Regressor

AdaBoost Regressor

0.3530

0.3476

0.4149

0.2285

0.2532

0.2699

0.4673

0.4831

0.5107

0.7047

0.6888

0.6573

0.1822

0.1831

0.2222

0.3420

0.3415

0.6432

Linear Regression

0.5088

0.4211

0.6409

0.4436

0.2756

0.6610

Decision Tree Regressor

0.4481

0.4906

0.6664

0.3834

0.2409

0.4428

Ridge Regression

0.5779

0.5172

0.7065

0.3461

0.3024

0.7571

K Neighbors Regressor

Elastic Net

Lasso Regression

Lasso Least Angle Regression

0.5482

0.7471

0.7516

0.7516

0.5097

0.8559

0.8844

0.8844

0.7020

0.2900

0.2801

0.9092

-0.0635

0.3719

0.9220

-0.0831

0.3758

0.9220

-0.0831

0.3758

0.7051

1.0715

1.1304

1.1304

s
d
o
h
t
e

M

e
l
b
m
e
s
n
E

s
d
o
h
t
e

M

l
a
n
o
i
t
i
d
a
r
T

Note: Calculated by the authors using the Python programming language

Page | 1767

Emerging Science Journal | Vol. 9, No. 4

Validation results revealed critical insights:

•  2012–2023: Cross-validation R² = 0.44 (Figure 2, right) versus test set R² = 0.72 (training R² = 1.00), indicating

pronounced overfitting (Figure 3, right).

•  2012–2021: Cross-validation R² = 0.73 (Figure 2, left) versus test R² = 0.81, (training R² = 0.97), suggesting mild

overfitting (Figure 3, left).

Figure 2. Validation Curves for Random Forest Regressor (2012-2021) and Extra Trees Regressor (2012-2023)

Figure 3. Residuals’ graphs for Random Forest Regressor (2012-2021) and Extra Trees Regressor (2012-2023)

These disparities underscore the necessity of cross-validation to mitigate overfitting and balance model complexity

with generalizability.

The comparative analysis of pre- and post-2022 periods highlights the methodological necessity of contextualizing

model performance within institutional upheavals.

3- Results and Discussion

A  previously  noted  limitation  of  the  model  is  its  interpretational  complexity.  To  contextualize  this  challenge,  the
Scikit-Learn  library’s  feature  importance  methodology  can  be  examined  (Figure  4).  Feature  importance  values  are
computed as the mean and standard deviation of accumulated impurity reduction across all decision trees within the
ensemble, reflecting each variable’s contribution to predictive accuracy.

Industry affiliation emerged as a substantively weighted factor in modeling financial efficiency, accounting for 15%
of  explanatory power  in  the 2012–2023 period  and 28%  in  2012–2021.  Comparative  analysis  between  the  extended
(2012–2023) and initial (2012–2021) periods reveals shifts in determinant significance: firm size increased from 14% to
21%, EBITDA margin from 6% to 11%, and the individual firm effect (captured by company-specific fixed factors) rose
from 4% to 11%. Conversely, CSR metrics exhibited  minimal variation, with general CSR expenditures contributing
5%–6% and employee-focused CSR 7.5%–9% across both intervals. Age (5% to 8%) and financial leverage (4% to 8%)
also gained marginal relevance.

Page | 1768

Emerging Science Journal | Vol. 9, No. 4

Figure 4. Feature importance plot for the periods of 2012-2021 and 2012-2023

The most pronounced shift occurred in innovation’s explanatory role, which declined from 22% (2012–2021) to 5%
(2012–2023). This suggests firms under geopolitical strain  – particularly post-2022  –  may reallocate resources from
innovation (a capital-intensive endeavor) toward operational stabilization.

To  enhance  interpretability,  the  Shapley  Additive  Explanations  (SHAP)  method  offers  a  theoretically  grounded
alternative. Rooted in cooperative game theory, SHAP quantifies feature contributions by decomposing predictions into
Shapley values, which assign marginal impact scores to each variable based on its synergistic role across all possible
feature combinations.

The SHAP summary plot (Figure 5) ranks features by their absolute impact on the target variable for both periods,
with  vertical  positioning  indicating  effect  magnitude.  Features  higher  on  the  y-axis  exert  greater  influence,  while
horizontal dispersion reflects the directionality (positive/negative) and consistency of their effects across observations.

Figure 5. Importance of factors influencing CFO for the periods of 2012-2021 and 2012-2023

The prominence of the extractive and manufacturing sectors in driving financial performance – compared to transport
and telecommunications – reflects structural differences in capital intensity, global market integration, and regulatory
exposure. Extractive industries, such as oil and mining, often benefit from commodity price cycles and export-driven
revenue models, while manufacturing sectors leverage economies of scale and supply chain control, as Chandler (1990)
[40] observed in his analysis of industrial evolution. The 28% to 15% decline in sectoral factor importance post-2022
(Figure  3)  suggests  disruptions  tied  to  geopolitical  events,  notably  sanctions  on  Russian  exports.  For  instance,
manufacturing  sectors  reliant  on  imported  technologies  faced  supply  chain  bottlenecks,  while  extractive  industries
contended with redirected trade flows and price volatility. These dynamics align with resource dependence theory, which
emphasizes how external shocks disproportionately affect industries with high dependency on global markets or foreign
inputs.

The increased significance of organizational size during the 2012–2023 period underscores its nonlinear relationship
with financial outcomes, a phenomenon consistent with transaction cost economics, first theorized by Williamson (1981)
[41]. Large enterprises – though not the largest – demonstrated stable positive impacts, likely due to economies of scale,
financing  access,  and  risk  diversification.  However,  the  largest  firms  exhibited  variable  effects,  potentially  due  to
bureaucratic inertia or overextension in turbulent markets. Sanctions may have amplified this dichotomy: smaller firms
within the “large” category adapted nimbly to regulatory changes, while mega-corporations faced operational rigidity or
geopolitical targeting, such as asset freezes. This mirrors the threshold effect described by Coase (1937) [42], wherein
organizational size optimizes efficiency until coordination costs dominate.

Page | 1769

Emerging Science Journal | Vol. 9, No. 4

Economic performance, financial leverage, and firm age influenced financial efficiency in ways resembling direct
relationships. Economic performance, measured by EBITDA margins, enhanced financial efficiency by freeing capital
for reinvestment and debt servicing, a principle foundational to Modigliani & Miller’s (1958) [4] propositions on capital
structure. Financial leverage exhibited a dual role: moderate debt amplified returns through tax shields and low-cost
capital, but excessive leverage increased vulnerability to interest rate hikes or revenue shocks – a trend exacerbated by
post-2022  macroeconomic  instability,  as  Baker  &  Wurgler (2002) [9]  theorized.  Firm  age  correlated  positively  with
efficiency, reflecting accumulated expertise and stakeholder trust, a pattern Nelson & Winter (1982) [15] attributed to
evolutionary organizational learning. However, older firms risked path dependency, hindering innovation during crises
– a tension central to dynamic capability theory, as articulated by Teece et al. (1997) [5].

The  reduced importance of CSR and R&D expenditures post-2022 signals a strategic pivot toward survival amid
sanctions. Companies deprioritized CSR initiatives, such as employee welfare and environmental programs, to conserve
liquidity – a decision aligned with the RBV principles outlined by Barney (1991) [18]. Similarly, R&D’s diminished
role highlights a shift from long-term innovation to short-term operational stability. Notably, the study’s observation that
R&D  quality  –  not  expenditure  –  drove  pre-2022  success  aligns  with  Cohen  &  Levinthal’s  (1990)  [43]  absorptive
capacity theory, which emphasizes the role of prior knowledge in leveraging R&D investments.

Figure 5 reveals that low CSR expenditures correlated with negative returns, while high CSR yielded positive – albeit
infrequent – outcomes. This suggests a threshold effect: minimal CSR may harm reputation and employee morale, but
beyond  a  baseline,  marginal  returns  diminish,  a  phenomenon  Barnett  &  Salomon  (2012)  [20]  documented  in  their
analysis  of  CSR-performance  relationships.  Post-2022,  the  rarity  of  high  CSR  investments  reflects  firms’  triage  of
stakeholder commitments. For example, companies maintained essential employee welfare programs to retain talent but
curtailed community initiatives.  This selective retention mirrors stakeholder salience theory, wherein firms prioritize
stakeholders critical to immediate survival, such as employees and regulators, as Mitchell et al. (1997) [14] proposed.

The post-2022 resurgence of traditional factors (e.g., size, profitability) and decline in strategic investments (e.g.,
R&D,  CSR)  underscore  the  RBV’s  applicability  in  emerging  markets  under  stress,  a  framework  Peng  (2003)  [6]
advanced in his work on transitional economies. Sanctions forced firms to rely on tangible resources, such as physical
assets and liquidity, rather than intangible capabilities. These findings also resonate with institutional theory, as abrupt
regulatory changes disrupted norms, compelling firms to adopt conservative, compliance-driven strategies, a process
DiMaggio & Powell (1983) [26] termed “coercive isomorphism.”

The  methodological  rigor  of  the  study  is  reinforced  by  its  use  of  SHAP  values,  a  technique  used  to  enhance  the
interpretability  of  machine  learning  models.  While  SHAP’s  ability  to  quantify  each  factor’s  marginal  contribution
strengthens causal inference, its sensitivity to temporal data shifts – such as the 2022 sanctions – necessitates cautious
interpretation, as abrupt changes may reflect outlier events rather than systemic trends. Extending the analysis to 2023
captures acute crisis responses but risks overlooking long-term adaptations, a limitation that future studies could address
by incorporating longitudinal data.

These findings also highlight critical avenues for further inquiry. First, longitudinal research could assess whether
short-term reductions in CSR and R&D undermine organizational resilience in subsequent crises, testing the enduring
implications of resource reallocation. Second, cross-country comparisons may reveal whether sanctions uniformly erode
strategic  investments  or  if  institutional  contexts  mediate  firm  responses.  Third,  sectoral  analyses  could  explore  how
extractive and manufacturing firms navigated sanctions differently, particularly examining whether vertical integration
or  domestic  supplier  networks  mitigated  risks.  Such  investigations  would  deepen  understanding  of  how  geopolitical
shocks interact with industry-specific dynamics to reshape financial efficiency.

4- Conclusion

This analysis identifies sectoral, organizational, and strategic determinants of corporate financial performance (CFP)
across  Russian  enterprises  from  2012  to  2023,  contextualized  within  geopolitical  disruptions.  Industry  affiliation
emerged  as  the  most  consequential  predictor,  with  extractive  and  manufacturing  sectors  demonstrating  outsized
influence relative to transport and telecommunications due to capital intensity and global market integration. However,
the explanatory weight of this factor declined sharply post-2022 (28% to 15%), reflecting sector-specific vulnerabilities
to  sanctions,  including  supply  chain  bottlenecks  and  commodity  price  volatility  –  a  pattern  aligned  with  resource
dependence  theory.  Organizational  size  exhibited  a  nonlinear  relationship  with  CFP:  mid-sized  firms  capitalized  on
economies of scale and adaptability, while the largest enterprises grappled with bureaucratic inertia, echoing transaction
cost  economics’  emphasis  on  coordination  thresholds.  Traditional  drivers,  such  as  economic  performance  (EBITDA
margins) and firm age, retained stable impacts, though excessive financial leverage amplified post-2022 risks. Notably,
strategic investments in CSR and R&D – critical for long-term competitiveness – were deprioritized during sanctions as
firms reallocated resources toward short-term survival, a strategic shift consistent with the RBV. These results illustrate
how geopolitical shocks reconfigure corporate priorities, eroding the value of intangible assets like innovation pipelines
while amplifying reliance on tangible resources and operational efficiency.

Page | 1770

Emerging Science Journal | Vol. 9, No. 4

The  findings  highlight  the  RBV’s  applicability  in  emerging  markets  under  systemic  stress,  where  institutional
turbulence  compels  firms  to  prioritize  static,  defensive  strategies  over  dynamic  capabilities.  However,  the  observed
retrenchment  from  innovation  and  stakeholder  engagement  raises  concerns  about  long-term  resilience,  as  these
investments are pivotal for post-crisis recovery. Policymakers must address this tension by incentivizing sustained R&D
and CSR continuity through mechanisms such as tax relief or targeted subsidies. Corporate leaders, conversely, should
balance  operational  agility  with  strategic  foresight,  fostering  absorptive  capacity  to  mitigate  external  shocks.  Future
research  should  investigate  cross-country  variations  in  sanction  responses,  sector-specific  adaptation  pathways  (e.g.,
vertical  integration  in  manufacturing),  and  the  longitudinal  consequences  of  CSR/R&D  divestment.  Ultimately,  the
analysis reveals the fragility of competitive advantages in volatile markets and underscores the imperative of adaptive
resource allocation frameworks to sustain growth amid geopolitical uncertainty. By situating universal determinants like
size and profitability within institutional upheavals, this work advances CFP theory while offering pragmatic insights
for firms and regulators navigating contested economic landscapes.

5- Declarations

5-1- Author Contributions

Conceptualization, I.B. and A.D.; methodology, I.B.; software, I.B.; validation, I.B. and A.D.; formal analysis, I.B.
and A.D.; investigation, I.B. and A.D.; resources, I.B. and A.D.; data curation, I.B. and A.D.; writing—original draft
preparation,  I.B.;  writing—review  and  editing,  A.D.;  visualization,  I.B.  and  A.D.;  supervision,  I.B.;  project
administration, I.B. and A.D.; funding acquisition,  I.B. and A.D. All authors have read and agreed to the published
version of the manuscript.

5-2- Data Availability Statement

The data presented in this study are available on request from the corresponding author.

5-3- Funding

The authors received no financial support for the research, authorship, and/or publication of this article.

5-4- Institutional Review Board Statement

Not applicable.

5-5- Informed Consent Statement

Not applicable.

5-6- Conflicts of Interest

The authors declare that there is no conflict of interest regarding the publication of this manuscript. In addition, the
ethical  issues,  including  plagiarism,  informed  consent,  misconduct,  data  fabrication  and/or  falsification,  double
publication and/or submission, and redundancies have been completely observed by the authors.

6- References

[1] Khanna, T., & Palepu, K. G. (2010). Winning in Emerging Markets: A Road Map for Strategy and Execution. NHRD Network

Journal, 3(3), 75–75. doi:10.1177/0974173920100316.

[2] Meyer, K. E., & Peng, M. W. (2020). Theoretical foundations of emerging economy business research. Multinational Enterprises

and Emerging Economies, 47(1), 24–43. doi:10.4337/9781788978927.00008.

[3] Graham, B., & Dodd, D. L. F. (1934). Security Analysis. McGraw-Hill Education, New York, United States

[4] Modigliani, F., & Miller, M. H. (1958).  The cost of capital, corporation finance and the theory of investment. The American

economic review, 48(3), 261-297.

[5] Teece, D. J., Pisano, G., & Shuen, A. (1997). Dynamic capabilities and strategic management. Strategic Management Journal,

18(7), 509–533. doi:10.1002/(sici)1097-0266(199708)18:7<509::aid-smj882>3.0.co;2-z.

[6] Peng,  M.  W.  (2003).  Institutional  Transitions  and  Strategic  Choices.  The  Academy  of  Management  Review,  28(2),  275.

doi:10.2307/30040713.

[7] Orlitzky, M., Schmidt, F. L., & Rynes, S. L. (2003). Corporate social and financial performance: A meta-analysis. Organization

Studies, 24(3), 403–441. doi:10.1177/0170840603024003910.

[8] Friedman, M. (1970). The social responsibility of business is to increase its profits. New York Times Magazine, New York, United
https://www.nytimes.com/1970/09/13/archives/a-friedman-doctrine-the-social-responsibility-of-

States.  Available  online:
business-is-to.html (accessed on July 2025).

Page | 1771

Emerging Science Journal | Vol. 9, No. 4

[9] Baker,  M.,  &  Wurgler,  J.  (2002).  Market  timing  and  capital  structure.  Journal  of  Finance,  57(1),  1–32.  doi:10.1111/1540-

6261.00414.

[10] Lessmann, S., Baesens, B., Seow, H. V., & Thomas, L. C. (2015). Benchmarking state-of-the-art classification algorithms for
credit scoring: An update of research. European Journal of Operational Research, 247(1), 124–136. doi:10.1016/j.ejor.2015.05.030.

[11] Yıldırım, H. H., Rençber, Ö. F., & Yüksel Yıldırım, C. (2024). Ranking the determinants of financial performance using machine
learning  methods:  an  application  to  BIST  energy  companies.  Mathematical  Modelling  and  Numerical  Simulation  with
Applications, 4(5-Special Issue: ICAME'24), 165–186. doi:10.53391/mmnsa.1594426.

[12] Jensen,  M.  C.,  &  Meckling,  W. H.  (1976). Theory  of the  firm: Managerial  behavior,  agency  costs  and  ownership  structure.

Journal of Financial Economics, 3(4), 305–360. doi:10.1016/0304-405X(76)90026-X.

[13] Freeman,  R.  E.  (2010).  Strategic  management:  A  stakeholder  approach.  Cambridge  university  press,  Cambridge,  United

Kingdom. doi:10.1017/cbo9781139192675.

[14] Mitchell, R. K., Agle, B. R., & Wood, D. J. (1997). Toward a theory of stakeholder identification and salience: Defining the
853–886.
counts.  Academy

of  Management  Review,

of  who

22(4),

really

principle
and  what
doi:10.5465/AMR.1997.9711022105.

[15] Nelson, R. R., & Winter, S. G. (1985). An evolutionary theory of economic change. Harvard University Press, Cambridge, United

States.

[16] Tyagi,  S.,  Nauriyal,  D.  K.,  &  Gulati,  R.  (2018).  Firm  level  R&D  intensity:  evidence  from  Indian  drugs  and  pharmaceutical

industry. Review of Managerial Science, 12(1), 167–202. doi:10.1007/s11846-016-0218-8.

[17] Bouaziz, Z. (2016). The Impact of R&D Expenses on Firm Performance: Empirical Witness from the Bist Technology Index.

Journal of Business Theory and Practice, 4(1), 51. doi:10.22158/jbtp.v4n1p51.

[18] Barney,  J.  (1991).  Firm  Resources  and  Sustained  Competitive  Advantage.  Journal  of  Management,  17(1),  99–120.

doi:10.1177/014920639101700108.

[19] Alshakhanbeh, M., Pitchay, A. A., Mia, M. A., Nairoukh, R., & Abd Jalil, M. I. (2024). The Impact of Liquidity and Leverage
on Financial Performance of Public Listed Firms in Jordan. International Journal of Academic Research in Accounting, Finance
and Management Sciences, 14(4), 1276–1299. doi:10.6007/ijarafms/v14-i4/23843.

[20] Barnett, M. L., & Salomon, R. M. (2012). Does it pay to be really good? addressing the shape of the relationship between social

and financial performance. Strategic Management Journal, 33(11), 1304–1320. doi:10.1002/smj.1980.

[21] Hsu, W. L., Lin, Y. L., Lai, J. P., Liu, Y. H., & Pai, P. F. (2025).  Forecasting Corporate Financial Performance Using Deep
417.

and  Governance  Data.  Electronics

(Switzerland),

Social,

14(3),

Learning  with  Environmental,
doi:10.3390/electronics14030417.

[22] Lins, K. V., Servaes, H., & Tamayo, A. (2017). Social Capital, Trust, and Firm Performance: The Value of Corporate Social

Responsibility during the Financial Crisis. Journal of Finance, 72(4), 1785–1824. doi:10.1111/jofi.12505.

[23] Aksoy, M., & Yilmaz, M. K. (2024). Financial performance of emerging market companies during the COVID-19 pandemic:
moderating role of sustainability performance. Journal of Global Responsibility, 16(3), 520–541. doi:10.1108/jgr-11-2023-0174.

[24] Donaldson,  T.,  &  Preston,  L.  E. (1995).  The Stakeholder Theory of  the  Corporation:  Concepts,  Evidence,  and  Implications.

Academy of Management Review, 20(1), 65–91. doi:10.5465/amr.1995.9503271992.

[25] Claessens, S., & Kodres, L. (2014). The Regulatory Responses to the Global Financial Crisis: Some Uncomfortable Questions.

IMF Working Papers, 14(46), 1. doi:10.5089/9781484335970.001.

[26] Dimaggio,  P.  J.,  &  Powell, W. W.  (2021).  the  Iron  Cage  Revisited:  Institutional  Isomorphism  and  Collective  Rationality  in

Organizational Fields. The New Economic Sociology: A Reader, 48(2), 111–134. doi:10.2307/2095101.

[27] Hufbauer, G., Schott, J., Elliott, K., & Oegg, B. (2007). Economic sanctions reconsidered. Peterson Institute for international

economics, Washington, United States.

[28] Бадыкова, И. Р. (2024). The issue of increasing the efficiency of economic activity of Russian enterprises: an empirical analysis

of panel data. Leadership and Management, 11(1), 295–316. doi:10.18334/lim.11.1.119953. (In Russian).

[29] Bühlmann, P., & van de Geer, S. (2011). Statistics for High-Dimensional Data. In Springer Series in Statistics. Springer Berlin,

Germany. doi:10.1007/978-3-642-20192-9.

[30] Davis, J. H., Schoorman, F. D., & Donaldson, L. (1997). Toward a stewardship theory of management. Academy of Management

Review, 22(1), 20–47. doi:10.5465/AMR.1997.9707180258.

[31] Golubeva, O. (2021). Firms’ performance during the COVID-19 outbreak: international evidence from 13 countries. Corporate

Governance (Bingley), 21(6), 1011–1027. doi:10.1108/CG-09-2020-0405.

Page | 1772

Emerging Science Journal | Vol. 9, No. 4

[32] Thi Thuy, D. (2023). Literature review on Factors Affecting Financial Performance of firms. International Journal of Business

and Management Invention, 12(6), 181–188. doi:10.35629/8028-1206181188.

[33] Beaumier, G., & Gjesvik, L. (2025). Digital Governance in a Rubber Band: Structural Constraints in Governing a Global Digital

Economy. Global Studies Quarterly, 5(2). doi:10.1093/isagsq/ksaf043.

[34] Fama, E. F., & Jensen, M. C. (1983). Separation of Ownership and Control. The Journal of Law and Economics, 26(2), 301–

325. doi:10.1086/467037.

[35] Hernandez, M. (2012). Toward an understanding of the psychology of stewardship. Academy of Management Review, 37(2),

172–193. doi:10.5465/amr.2010.0363.

[36] Friede, G., Busch, T., & Bassen, A. (2015). ESG and financial performance: aggregated evidence from more than 2000 empirical

studies. Journal of Sustainable Finance & Investment, 5(4), 210–233. doi:10.1080/20430795.2015.1118917.

[37] Contini, M., Annunziata, E., Rizzi, F., & Frey, M. (2020). Exploring the influence of Corporate Social Responsibility (CSR)
in  BRICS  countries.  Journal  of  Cleaner  Production,  247.

loyalty:  An  experiment

domains  on  consumers’
doi:10.1016/j.jclepro.2019.119158.

[38] Volkova, O., & Kuznetsova, A. (2022). The Impact of Corporate Social Responsibility on Corporate Financial Performance:
Evidence from Russian and Dutch Companies. Journal of Corporate Finance Research, 16(2), 15–31. doi:10.17323/j.jcfr.2073-
0438.16.2.2022.15-31.

[39] Fama, E. F., & French, K. R. (1992). The Cross‐Section of Expected Stock Returns. The Journal of Finance, 47(2), 427–465.

doi:10.1111/j.1540-6261.1992.tb04398.x.

[40] Chandler, A. D. (1990). Scale and Scope: The Dynamics of Industrial Capitalism. Harvard University Press, Cambridge, United

States.

[41] Williamson, O. E. (1981). The Economics of Organization: The Transaction Cost Approach. American Journal of Sociology,

87(3), 548–577. doi:10.1086/227496.

[42] Coase, R. H. (1937). The Nature of the Firm. Economica, 4(16), 386–405. doi:10.1111/j.1468-0335.1937.tb00002.x.

[43] Cohen, W. M., & Levinthal, D. A. (1990). Absorptive Capacity: A New Perspective on Learning and Innovation. Administrative

Science Quarterly, 35(1), 128. doi:10.2307/2393553.

Page | 1773



---

## Content from Previous Extraction (not in markitdown output)

### Visual/Chart/Graph Descriptions

#### 1

### Visual/Chart/Graph Descriptions

#### 2

Designation
      Indicators                         Description                            Mean      std     Min    Median    Max
                                                              on the graph

### Additional Content

#### 1

# Defining the Determinants of Corporate Financial Performance - A Machine Learning Approach

#### 2

> *Source PDF: Defining the Determinants of Corporate Financial Performance - A Machine Learning Approach.pdf*
> *Extraction: Combined — markitdown raw text (base) + previous extraction supplements*

#### 3

## Content from Previous Extraction (not in markitdown output)

#### 4

# Defining the Determinants of Corporate Financial Performance - A Machine Learning Approach

#### 5

Keywords:
 Abstract
                                                                                              Corporate Financial Effectiveness;
  This study investigates the determinants of corporate financial performance (CFP) among Russian                                                                                    Ensemble Machine Learning;
  enterprises (2012–2023) through the lens of geopolitical disruptions, employing ensemble machine
                                                                                                     Geopolitical Risk;  learning (ML) to address methodological gaps in modeling non-linear institutional interactions.
 Using data from 25 large non-financial firms, we analyze sectoral, organizational, and strategic   Sanctions; Resource Dependence Theory;
  drivers, integrating train-test splits (75%/25%) and 10-fold cross-validation to mitigate overfitting.   Resource-Based View (RBV);
  Results reveal that industry affiliation, initially dominant (28% explanatory power pre-2022),
                                                                                           Shapley Additive Explanations (SHAP);  declined sharply post-sanctions (15%), reflecting vulnerabilities in globally integrated sectors like
  manufacturing and extractives. Organizational size exhibited a nonlinear relationship with CFP,   Russian Enterprises;
  favoring comparatively smaller firms’ agility over larger enterprises’ rigidity, consistent with   Emerging Markets.
  transaction cost economics. Strategic investments in corporate social responsibility (CSR) and
  research and development (R&D) diminished post-2022  as  firms  prioritized  liquidity and
  operational stability, aligning with resource-based view principles. Methodologically, Shapley
  Additive Explanations (SHAP) clarified threshold effects in CSR returns and innovation’s reduced
  role under sanctions. The study innovates by applying ensemble machine learning to sanction-   Article History:
  affected emerging markets, challenging linear econometric assumptions and advancing institutional
  theory through a crisis-contextualized framework of resource dependence and stakeholder salience.   Received:      25    April      2025
  Findings underscore the fragility of intangible assets under systemic shocks and advocate adaptive   Revised:       10    July       2025
  resource allocation frameworks to balance short-term survival with long-term resilience. This work
  provides policymakers and managers actionable insights for fostering operational agility and   Accepted:      17    July       2025
  strategic foresight in volatile institutional environments.                                                                                           Published:     01    August    2025

#### 6

Company                 Company affiliation                Company      13.039    7.187     1.000     13.000     25.000

#### 7

Year                    Number of the year                    Year         2017.4    3.470    2012.0     2017.0     2023.0

#### 8

Innovations           R&D expenses / Revenue                  Innov         0.002     0.005     0.000      0.001      0.043

#### 9

Size                     Natural logarithm of revenue                   Size         12.098    3.106     0.000     12.609     16.273

#### 10

CSR               CSR expenses / Revenue                    Csr          0.015     0.159     0.000      0.002      2.667

#### 11

M/B                       Capitalization / Book value                 m/b          1.109     0.908     0.000      0.915      3.980

#### 12

Age         Number of years since the company's registration       Age         13.227    9.993     0.000     16.000     28.000

#### 13

Ratio of the sum of short-term and long-term loans
      Leverage                                                      Lev          0.383     0.235     0.000      0.367      0.928
                                to the sum of equity and borrowed funds

#### 14

Belonging to the sector:
       Industry          1: Extractive;
                                                                                    Field         1.461     0.827     0.000      1.000      3.000
     characteristic       2: Manufacturing;
                           3: Transport and telecommunications

#### 15

Random Forest Regressor          0.4424     0.4418     0.6397     0.3934     0.2662     0.4566                                          Methods       Light Gradient Boosting Machine       0.4425     0.4355     0.6311     0.3815     0.2621     0.4702

#### 16

Gradient Boosting Regressor         0.4518     0.4712     0.6511     0.2992     0.2698     0.5077                                                Ensemble
                           AdaBoost Regressor             0.5452     0.5254     0.7017     0.2457     0.3141     0.7427

#### 17

Ridge Regression              0.6315     0.6806     0.8068     0.1602     0.3612     0.8778
                                          Methods           Decision Tree Regressor           0.4942     0.5865     0.7437     0.1419     0.3111     0.4986
                                        Elastic Net                 0.7167     0.8394     0.8851     0.0302     0.3951     0.9837

#### 18

Lasso Regression              0.7226     0.8597     0.8970     0.0051     0.4010     1.0157                                                                   Traditional        Lasso Least Angle Regression        0.7226     0.8597     0.8970     0.0051     0.4010     1.0157
                                Linear Regression              0.7257     2.4714     1.1785     -4.2882     0.3937     1.0056

#### 19

Extra Trees Regressor           0.3314     0.2369     0.4680     0.7236     0.1700     0.3241                                          Methods          Extreme Gradient Boosting         0.3192     0.2257     0.4534     0.7209     0.1745     0.3300

#### 20

Light Gradient Boosting Machine      0.3530     0.2285     0.4673     0.7047     0.1822     0.3420
