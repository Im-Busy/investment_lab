# Profit Mirage - Revisiting Information Leakage in LLM-based Financial Agents

> *Source PDF: Profit Mirage - Revisiting Information Leakage in LLM-based Financial Agents.pdf*
> *Extraction: Combined — previous structured extraction (base) + markitdown raw text*

---

# Profit Mirage - Revisiting Information Leakage in LLM-based Financial Agents

        Profit Mirage: Revisiting Information Leakage in LLM-based
                             Financial Agents

                  Xiangyu Li                 Yawen Zeng                   Xiaofen Xing
                 65603605 lxy@gmail.com            yawenzeng11@gmail.com                xfxing@scut.edu.cn
            South China University of Technology             Byte Dance              South China University of Technology
                   Guangzhou, China                         Beijing, China                    Guangzhou, China

                                            Jin Xu                   Xiangmin Xu∗
                                        jinxu@scut.edu.cn                  xmxu@scut.edu.cn
                              South China University of Technology   South China University of Technology
                                      Pazhou Lab                     Guangzhou, China
                                    Guangzhou, China

                                                    Why does this mirage arise? We show that the culprit is not2025   Abstract
          LLM-based financial agents have attracted widespread excitement      flawed risk management or noisy market data, but information
            for their ability to trade like human experts. However, most systems     leakage baked into the LLM itself. Modern foundation mod-
                                                                                           els ingest web-scale corpora that contain post-hoc explanations Oct    exhibit a “profit mirage”: dazzling back-tested returns evaporate
          once the model’s knowledge window ends, because of the inherent      of past price movements—“NVIDIA surged 190% in 2023 on AI
9           information leakage in LLMs. In this paper, we systematically quan-     boom”—alongside contemporaneous news. When these snippets
              tify this leakage issue across four dimensions and release Fin Lake-     appear in the training set, the model does not learn why prices
         Bench, a leakage-robust evaluation benchmark. Furthermore, to     move; it learns that they already moved, and simply recites the an-
           mitigate this issue, we introduce Fact Fin, a framework that ap-     swer during back-testing. In fact, this “pre-training contamination”
             plies counterfactual perturbations to compel LLM-based agents to        is lethal in finance area.
           learn causal drivers instead of memorized outcomes. Fact Fin inte-        Furthermore, we formalize this concern and demonstrate its[cs. AI]           grates four core components: Strategy Code Generator, Retrieval-      empirical prevalence through four experiments:
         Augmented Generation, Monte Carlo Tree Search, and Counter-
            factual Simulator. Extensive experiments show that our method
           surpasses all baselines in out-of-sample generalization, delivering                                                                     • 1) Back-testing versus generalization (Section 2.1). By rolling
           superior risk-adjusted performance.
                                                                                   the calendar forward we reveal that almost every published LLM-
                                                                            based agent fails to beat a random baseline once its knowledge
       Keywords                                                                               cutoffis passed.
           Quantitative Finance, Large Language Model, LLM-based Agent       • 2) Counterfactual evaluation (Section 2.2). We feed models
                                                                                         carefully crafted counterfactual prompts by perturbing key mar-
        1  Introduction                                                  ket inputs. Results show high prediction consistency, with the
                                                                          worst model maintaining 82.13% of predictions unchanged de-
         The advent of large language models (LLMs) has precipitated a
                                                                                         spite significant input alterations. This proves that agents are
           paradigmatic shift in quantitative finance. Representative systems
                                                                                 primarily reciting memorized patterns rather than analyzing
           include Fin GPT [30], Fin Mem [33], Fin Report[12] and the multi-
                                                                                     tradable information.
           agent architecture Hedge-Agents [13], all report double- or triple-
                                                                     • 3) Memorization audits (Section 2.3). We release a leakage-
             digit annualized returns in back-tests that span the U.S., Hong Kongar Xiv:2510.07920 v1                                                                            robust evaluation, Fin Lake-Bench, which constructs 2,000 histor-          and A-share markets.
                                                                                                 ical QA pairs, such as“did the market rise on date T?”. GPT-4 o
            However, when we move these LLM-based agents one step be-
                                                                     and its peers answer correctly over 85% of the time—far above
         yond their training time cutoff, the story falls apart, that is, a “profit
                                                                          chance—confirming that the facts have been memorized.
            mirage”. Figure 1 shows the equity curves of several popular agents
                                                                     • 4) Before-and-after targeted fine-tuning (Section 2.4). When
            re-evaluated on new data after the release of their underlying LLMs
                                                            we deliberately inject financial data into model training, in-distribution
               (e.g., verifying an agent released in 2023 in the 2024 financial mar-
                                                                               accuracy exceeded 70%, showing a significant improvement, while
              ket.). The best-performing LLM-based agent also drop 50%! The
                                                                                      generalization capability on unseen data drops dramatically. The
           dazzling return collapses to a statistical zero once the model is no
                                                                              gain is pure memorization of historical patterns, not improved
           longer allowed to peek at the future it was trained on. This phe-
                                                                                   trading skill.
        nomenon is also the reason why we call it the “profit mirage” :
           returns that evaporate the moment the model is forced to trade in
           genuinely unknown territory.
                                                                      Taken together, these results show that LLM-based financial agents
             ∗Corresponding author.                                                    are not trading; they are regurgitating history.

, ,                                                                                                                                                                                 Li et al.


   In this paper, to escape this mirage issues, we propose a counter-
factual framework, namely Fact Fin. Specifically, our Fact Fin inte-
grates four core components: Strategy Code Generator, Retrieval-
Augmented Generation, Monte Carlo Tree Search, and Counter-
factual Simulator. These components work together to create and
refine trading strategies using real-time market data. In this way,
LLM-based Agents are forced to learn why an outcome occurred,
because the outcome itself is no longer fixed. Finally, our method
deliver out-of-sample Sharpe ratios 1.4× higher than the best base-
lines.
  Our contributions are summarized as follows:                                                            Figure 1: Backtesting vs. Generalization. All LLM-based
• We provide the first systematic evidence from four dimensions      agents show a significant drop in the generalization setting.
   that the information leakage in LLM-based agents. Moreover, we
  conduct an extensive empirical study of leading open- and
  closed-source models, providing a clear baseline and revealing     where E[𝑅] is the expected return, 𝑅𝑓is the risk-free rate, and 𝜎𝑅
   their current limitations in .                                                   is the return volatility. To quantify performance degradation, we
• We introduce Fin Lake-Bench, the first leakage-robust evalua-     compute the decay rate as:
   tion suite that includes memorization probes and counterfactual
                                                                                                      𝑋pre −𝑋post
   labels.                                                               Decay Rate =         × 100%,               (3)
• We develop a counterfactual framework, Fact Fin, which inte-                                        𝑋pre
  grates four core components.                                                          where 𝑋represents either TR or SR for the pre- and post- periods
                                                                         of training.
2   Is Information Leakage Everywhere?
                                                                                     2.1.3   Results and Insights. Figure 1 presents the performance com-In this section, we formalize the leakage concern of LLM-based
                                                                       parison, revealing that all methods show a significant drop in theagents and demonstrate its empirical prevalence through four ex-
                                                                  period after the base model (i.e., GPT-4 o) is released. The Sharpeperiments.
                                                                      Ratio decay ranges from 51.48% (Quant Agent) to 62.23% (Fin CON),
                                                                 while the Total Return decay ranges from 50.18% (Trading Agents)2.1  Backtesting vs. Generalization
                                                                         to 71.85% (Fin Mem). Fin Mem exhibits the most severe degradation,
To investigate the impact of information leakage in LLM-based                                                                         indicating a strong reliance on memorized historical patterns. Fin A-
financial agents, we conduct a temporal segmentation experiment                                                                  gent and Quant Agent, show slightly better resilience, likely due to
to compare performance before and after the training time cut-                                                                            their use of external tools to augment decision-making. Trading A-
offof the underlying LLM. The experiment aims to demonstrate                                                                         gents, which leverage collaborative mechanisms, further mitigate
that models exploit historical patterns in training data, leading to                                                                   leakage but still suffer a 55.68% Sharpe decay. This significant per-
inflated backtesting results but poor forward-testing performance.                                                            formance drop, despite comparable market conditions, suggests
                                                                       that LLM-based agents are not genuinely forecasting but rather2.1.1  Experimental Setting. We select the NASDAQ-100 index con-
                                                                    recognizing patterns from their training data.stituent stocks as the evaluation pool and define two periods with
comparable market conditions to isolate the effect of information
leakage: a historical trade period (Q2-Q3 2021, backtesting setting)     2.2  Counterfactual Evaluation
and a latest trade period (Q3-Q4 2024, generalization setting ). No-      Further, we utilize a counterfactual evaluation framework to assess
tably, the market returns in the two periods are similar (market    how reliance on memorized patterns leads to poor generalization.
return is +13.79% and +13.35%), to minimize the impact of the mar-
                                                                                     2.2.1  Experimental Setting. Counterfactual market environments[3]ket itself.
                                                                      are constructed by perturbing inputs: modifying or removing key
2.1.2  Baselines and metrics. Thereafter, we evaluate five state-of-      events (e.g., earnings reports, regulatory changes), replacing price
the-art LLM-based methods—Fin Mem [33], Fin Agent [35], Quan-     sequences with historical averages or random walks, and altering
t Agent [24], Fin CON [34], and Trading Agents [29]—using GPT-4 o       technical indicators (e.g., RSI, MACD, KDJ) and fundamental factors
[17] (training cutoff: October 2023) as the backbone. Performance         (e.g., PE, PB, ROE). These perturbations test whether models adapt
is measured using Total Return (TR), defined as:                        to input changes or rely on memorized patterns. We evaluate ten
                                                                       stocks (AAPL, TSLA, NVDA, MSFT, GOOGL, AMZN, META, NFLX,
                                 𝑃final −𝑃initial             TR =         × 100%,                   (1)    AMD, CRM) from January 2022 to June 2023, selecting 30 key time
                                         𝑃initial                                    points per stock based on significant market events.
where 𝑃initial and 𝑃final are the initial and final portfolio values, and
                                                                                     2.2.2  Baselines and metrics. We evaluate the same five LLM-based
Sharpe Ratio (SR), defined as:
                                                            methods as in Section 2.1, using GPT-4 o as the backbone. Informa-
                         E[𝑅] −𝑅𝑓                                 tion leakage is quantified via three metrics. Prediction Consistency
                   SR =                     ,                         (2)                             𝜎𝑅                                  (PC)[4] measures the proportion of unchanged predictions after

Profit Mirage: Revisiting Information Leakage in LLM-based Financial Agents                                                                                                                                                                                   , ,


       Table 1: Counterfactual evaluation of agents.


               Methods      PC ↓    CI ↓   IDS ↑

               Fin Mem          0.8213   0.8743   0.2766
                 Fin Agent         0.7245   0.7781   0.3598
                Quant Agent      0.7789   0.8362   0.2941
               Fin CON          0.7136   0.7522   0.3612
                  Trading Agents   0.6903   0.7016   0.3837


perturbation:
                       𝑁
             PC =                       1 ∑︁ I hˆ𝑦orig𝑖  = ˆ𝑦cf𝑖 i ,                    (4)      Figure 2: Memorization audits of LLMs on Fin Leak-Bench                𝑁
                                 𝑖=1
where ˆ𝑦orig𝑖   and ˆ𝑦cf𝑖are predictions in original and counterfactual
scenarios, I[·] is the indicator function, and 𝑁is the sample size.       2.3.2  Baselines and metrics. We evaluate three leading LLMs—GPT-
                                                                           4 o, Claude-Sonnet-3.7[1], and Grok-3[28]—using a strict accuracy Higher PC indicates greater reliance on memorized patterns. Confi-
                                                                        metric:dence Invariance (CI)[20] assesses prediction confidence stability:
                                                                                 𝑁
                     𝑀
                                                                           Accuracy =                                                                                         1 ∑︁ I [ˆ𝑎𝑖= 𝑎𝑖] · 𝑤𝑖,                (7)                 CI = 1 −1                   ∑︁ 𝑠orig𝑗   −𝑠cf𝑗   ,                    (5)                     𝑁  𝑖=1               𝑀
                                     𝑗=1
                                                         where ˆ𝑎𝑖and 𝑎𝑖are the predicted and true answers for question 𝑖,
where 𝑠orig𝑗   and 𝑠cf𝑗are confidence scores for consistent predictions,        I[·] is the indicator function, 𝑁is the number of questions, and 𝑤𝑖
and 𝑀is the number of consistent samples. CI near 1 suggests        is a weight reflecting answer precision. Price inquiries score 1 point
insensitivity to input changes. Input Dependency Score (IDS)[14]       for answers within ±1% error and 0.5 for ±3%. Trend predictions
measures input sensitivity via KL divergence:                         earn 0.5 points for correct direction and 0.5 for accurate magnitude.
                      𝑁                                     Event impact questions are scored based on qualitative alignment
                      1 ∑︁               IDS =                        𝐷KL  𝑃orig𝑖   ∥𝑃cf𝑖     ,                  (6)      with actual market reactions. Market performance questions score               𝑁
                                𝑖=1                                    1 point for identifying stocks in the top/bottom 3% and 0.5 for
where 𝑃orig𝑖   and 𝑃cf𝑖are prediction probability distributions. Higher      top/bottom 5%.
IDS indicates less leakage.                                                      2.3.3   Results and Insights. Figure 2 presents the average accuracy
2.2.3   Results and Insights. Table 1 shows average leakage metrics      of LLMs on Fin Leak-Bench, revealing significant memorization.
across all stocks, highlighting generalization issues due to infor-      All models exhibit high average accuracy across question types,
mation leakage. Fin Mem exhibits the highest leakage, with a PC of      ranging from 85.37% for price inquiries to 92.94% for event impacts.
0.8213 and CI of 0.8743, indicating over 82% of predictions remain      Event impact questions show the highest accuracy, indicating near-
unchanged despite perturbations, with stable confidence, reflecting      encyclopedic recall of market reactions to specific events. Price
heavy reliance on memorized patterns. Its low IDS (0.2766) con-      inquiries (85.37%) and market performance (89.83%) accuracies sug-
firms limited input sensitivity, limiting generalization. Single-agent       gest precise memory of individual data points and relative rankings.
models like Fin Agent and Quant Agent show moderate improve-     Most critically, the 90.23% accuracy on trend prediction questions,
ments via tool-augmented decision-making. Moreover, multi-agent     which require recalling temporal sequences, confirms that models
systems, such as Fin CON and Trading Agents, exhibit the lowest     memorize complete market movement patterns, enabling them to
leakage, benefiting from collaborative verification, but PC above      “predict” historical outcomes with high fidelity. This memorization
0.69 indicates persistent leakage. High PC and CI across methods       directly contributes to information leakage in financial forecast-
confirm reliance on memorized patterns over input-driven forecast-      ing, as models rely on recalled patterns rather than input-driven
ing, causing poor generalization.                                      reasoning, undermining generalization.

2.3  Memorization Audits                             2.4  Before-and-after Fine-tuning
Furthermore, we explore the memory capacity of LLMs.                  Finally, we assess the extent to which the acquired knowledge via
                                                                        training affects the performance.
2.3.1  Fin Lake-Bench. Fin Leak-Bench comprises 2,000 financial
question-answer pairs spanning January 2022 to June 2023, cov-       2.4.1  Experimental Setting. We fine-tune two base models, Qwen2.5-
ering four categories: price inquiries (e.g., NVIDIA’s closing price      7 B-Instruct [23] and Llama-3.1-8 B-Instruct [15], using Lo RA [6]
on a specific date), trend predictions (e.g., Apple’s stock movement      (rank=64, alpha=16, dropout=0.1, batch size=16, gradient accumula-
over two months), event impacts (e.g., Tesla’s stock response to ma-      tion steps=4, learning rate=2 e-4, warmup ratio=0.03) on the FNSPID
jor corporate announcements), and market performance (e.g., top-      dataset [2], comprising Dow Jones Industrial Average (DJIA) con-
performing NASDAQ-100 stocks on a given date). Representative       stituent stocks (30 stocks) from January 2020 to December 2022. The
examples and their corresponding scoring criteria are summarized       test set includes market data for these DJIA stocks from January to
in Table 2.                                                       June 2022.

, ,                                                                                                                                                                                 Li et al.


                          Table 2: Examples of Fin Leak-Bench Questions and Scoring Criteria


    Category      Question                  Ground Truth Answer      Scoring Criteria

     Price Inquiry    What was NVIDIA’s closing price   $229.73                      1 point if answer ∈[227.43, 232.03]; 0.5 point
                  on March 15, 2022?                                                                    if ∈[222.84, 236.62]; otherwise 0.
                What was Tesla’s opening price   $122.56                      1 point if answer ∈[121.33, 123.79]; 0.5 point
                  on January 12, 2023?                                                                  if ∈[118.88, 126.24]; otherwise 0.
    Event Impact    What was the impact of Musk’s   Negative impact: Tesla’s stock   Full point for responses mentioning >15%
                      Twitter acquisition on October 27,   fell over 15%, significantly un-  drop, with reasons such as Musk’s distrac-
                       2022, on Tesla’s stock price?      derperforming the market.      tion, forced share sale, or Twitter-related
                                                                                     negative sentiment.
               How did Silicon Valley Bank’s col-  Complex impact: Initial sector  1 point for answers covering both initial
                      lapse on March 10, 2023, affect  drop of 2%, followed by large-  decline and rebound; 0.5 point if only partial
                      tech stocks?                     cap tech rebound; startup pres-   effects are mentioned.
                                                        sures persisted.

    Trend  Predic-  What was Amazon’s price trend  Steady decline with minor re-  0.5 point for correct trend direction; +0.5 if
     tion            from April 1, 2022, over the fol-  bounds, cumulative drop of  magnitude is within ±5% error margin.
                    lowing 4 weeks?                  11.91%.
               How did Apple’s stock perform in  Consistent upward trend, cu-  0.5 point for correct trend direction; +0.5 if
                      the 2 months following February  mulative gain of 7.49%.        magnitude is within ±5% error margin.
                         15, 2023?

    Market Perfor- How did the energy sector in the  Weak performance: sustained  1 point if answer ∈[-2.28%, -1.68%]; 0.5 point
    mance        S&P 500 perform on August 15,  decline with average drop of    if ∈[-2.48%, -1.48%]; otherwise 0.
                     2022?                             1.98%.
                 Which Dow Jones component  MSFT                       1 point for MSFT; 0.5 points for UNH or
                  had the largest decline on Janu-                           AXP.
                     ary 5, 2023?


                                                         where pmodel𝑡    and phist𝑡   are the model’s prediction and historical
                                                                     pattern vectors at time 𝑡, and 𝑇is the number of time points. The
                                                                     Generalization Change[8] assesses the relative change in accuracy
                                                        on unseen data:
                                                                                                Accpostunseen −Accpreunseen                                                                           ΔGen =               × 100%,           (10)
                                                                                                  Accpreunseen

                                                         where Accpreunseen and Accpostunseen are pre-fine-tuning and post-fine-
                                                                  tuning accuracies on unseen data.
 Figure 3: Before-and-after fine-tuning on model behavior.
                                                                                  2.4.3  Results and Insights. Figure 3 illustrates the impact of fine-
                                                                   tuning on model behavior. Training markedly improves in-distribution
                                                                     accuracy, with Qwen2.5-7 B-Instruct increasing from 51.61% to
2.4.2  Baselines and metrics. We evaluate the impact of fine-tuning
                                                                  72.16% and Llama-3.1-8 B-Instruct from 54.73% to 76.52%. However,
using three metrics. The Bias Score[7] quantifies prediction bias
                                                                               this performance leap is accompanied by increased prediction bias,
toward frequently observed stocks:
                                                              with Bias Scores of 0.2895 and 0.3122, indicating that models no
                                                                  longer                                                                                    treat all stocks                                                                                                  impartially,                                                                                                    favoring                                                                                                               frequently                                                                                                                observed                            𝑆   𝑓train (𝑠) · 𝑝score (𝑠)                    1 ∑︁                          −1              Bias =                                                                     ones. Moreover,                                                                      High Memory                                                                                                Scores                                                                                                              of 0.7759                                                                                                 and 0.8113                                                                                                  show                    𝑆                                              𝑆,              (8)                           𝑠=1  Í𝑆𝑠′=1 𝑓train (𝑠′)                                                                       that models closely replicate historical patterns from the train-
where 𝑓train (𝑠) is the frequency of stock 𝑠in the training data,      ing data, suggesting they memorize future outcomes rather than
𝑝score (𝑠) is the model’s prediction score for stock 𝑠, and 𝑆is the      learning robust forecasting principles. Critically, generalization ca-
number of stocks. The Memory Score measures alignment with       pability declines substantially, with reductions of 21.53% and 18.06%
historical patterns:                                                     in accuracy on unseen data, confirming that fine-tuning leads to
                                                                          overfitting on leaked data. This behavior, driven by training data,
                        1 𝑇∑︁           Memory =                              cos  pmodel𝑡       , phist𝑡       ,               (9)      exacerbates information leakage risks as models reproduce memo-                    𝑇                                𝑡=1                                        rized patterns.

Profit Mirage: Revisiting Information Leakage in LLM-based Financial Agents                                                                                                                                                                                   , ,


  These findings highlight that fine-tuned models are not smarter                                                                      Market
but rather memorize historical patterns!                                    Information Σμ          Retrieval-Augmented Generation
3  Our Approach: A Counterfactual Evolution                                        Factor Extraction           Enhanced Structured
                                                                                                                                    Market Features St                                                                                                                                                                                                                                                                                                                                         ′   Framework
                                                                                                      Price Data Pt                Strategy Code Generator
3.1  Preliminaries
To escape the mirage issues, we propose Fact Fin, an external frame-                                                                                    Initial Strategy υ
work that mitigates leakage by using LLMs as strategy generators                                                                                     Market Factors Ft
rather than direct decision-makers, leveraging counterfactual rea-                               Monte Carlo Tree Search
soning and strategy evolution. We define the market state at time 𝑡
as 𝑆𝑡= {𝑃𝑡, 𝐹𝑡, 𝑁𝑡}, where 𝑃𝑡∈R𝑑is price data, 𝐹𝑡∈R𝑚denotes                                                            Optimized Strategy C‘
market factors, and 𝑁𝑡∈R𝑛represents factorized news. A trading             Market News Nt                  Strategy Performance
strategy𝐶: 𝑆𝑡→𝐴𝑡maps 𝑆𝑡to actions 𝐴𝑡∈{buy, sell, hold}. Thus,                                               Evaluation ϔ
LLM-based Agents are forced to learn why an outcome occurred,                                                                    Convergence                                                                             Counterfactual
because the outcome itself is no longer fixed.                                                                                                               Failure                                                                              Simulator                      Strategy Leakage
                                                                                                           Assessment ϔ푐Ϣ
3.2  Framework Overview                                                                                                                                                            Final Strategy C∗
                                                                                      Counterfactual As depicted in Figure 4, our Fact Fin is an external framework that
                                                                                     Scenario Dcf                   Trade Executionmitigates information leakage in LLM-based financial prediction
by using LLMs as strategy generators, coupled with counterfac-
tual reasoning and strategy evolution. Fact Fin integrates four core           Figure 4: The overall architecture of our Fact Fin.
components: Strategy Code Generator (SCG), Retrieval-Augmented
Generation (RAG), Monte Carlo Tree Search (MCTS), and Counter-
                                                         where 𝑆′𝑡includes processed prices, factors, and news. For newsfactual Simulator (CS). These components synergistically form a
                                                                           data, RAG extracts quantified features:pipeline that generates and optimizes trading strategies driven by
real-time market inputs, ensuring robust predictions for LLM-based                         𝑁′𝑡= 𝜓(𝑁𝑡,𝑡),                        (13)
financial forecasting methods.
                                                         where 𝜓converts news at time 𝑡into sentiment scores and topic
                                                                          distributions in R𝑛. This approach "factorizes" unstructured infor-
3.3  Strategy Code Generator
                                                                  mation, making it more suitable for systematic trading strategies
The Strategy Code Generator (SCG) transforms financial prediction      while reducing the risk of the model falling back on memorized
into a code generation task, leveraging LLMs to produce executable      outcomes.
trading strategy code based on market state 𝑆𝑡. SCG generates a
strategy 𝐶: 𝑆𝑡→𝐴𝑡, defined as:                                                         3.5 MCTS for Strategy Evolution
             𝐶= SCG(𝑆𝑡, 𝑃),                       (11)     The Monte Carlo Tree Search (MCTS)[21] component optimizes
                                                                          strategies generated by the SCG, ensuring adaptation to market
where 𝑃is a prompt template. By focusing on generating systematic                                                                            factors in 𝑆𝑡while avoiding reliance on memorized patterns. MCTS
strategies rather than specific predictions, we reduce the model’s                                                                produces an optimized strategy 𝐶∗: 𝑆𝑡→𝐴𝑡, defined as:
reliance on memorized historical price movements. A simplified
                                                           𝐶∗= MCTS(𝐶, 𝐷),                      (14) prompt is as follows1:
 [Prompt Template]                                          where 𝐶is the initial strategy and 𝐷represents evaluation datasets.
 Given market state with {Prices}, {Factors}, and {News}, generate     MCTS explores the strategy space by selecting nodes using the
  executable trading strategy code, e.g., {Examples}, using only       Upper Confidence Bound:
  provided inputs.                                                                                   √︄                                                                                     𝑤(𝑠)       ln 𝑁(𝑠)
                                                                         UCB(𝑠) =    + 𝑐                   ,               (15)
                                                                                                 𝑛(𝑠)        𝑛(𝑠)
3.4 RAG for Market Factors
                                                          where 𝑤(𝑠) is the cumulative reward, 𝑛(𝑠) is the visit count, 𝑁(𝑠)The Retrieval-Augmented Generation (RAG)[11] component en-
                                                                                         is the parent’s visit count, and 𝑐is an exploration parameter. Newhances the Strategy Code Generator by retrieving and processing
                                                                       strategy variants are generated via:real-time market factors within 𝑆𝑡, ensuring strategies rely on cur-
rent inputs rather than memorized data. RAG transforms the market                      𝐶new = SCG(𝐹𝑡, 𝑃modify,𝐶),                 (16)
state 𝑆𝑡= {𝑃𝑡, 𝐹𝑡, 𝑁𝑡} into structured features 𝑆′𝑡, defined as:
                                                           and evaluated as:
                           𝑆′𝑡= RAG(𝑆𝑡),                        (12)                                                  𝑅= Evaluate (𝐶new, 𝐷).                   (17)
1 The templates will change based on different market conditions, fully disclosed in     Node statistics are updated as 𝑤(𝑠) = 𝑤(𝑠) + 𝑅and 𝑛(𝑠) = 𝑛(𝑠) + 1.
the Appendix.                                              By iteratively refining strategies based on real-time market inputs,

, ,                                                                                                                                                                                 Li et al.


Algorithm 1: Workflow of Our Fact Fin                                  in Section 2.2, which measure stability, sensitivity, and data depen-
Require: Market state 𝑆𝑡= {𝑃𝑡, 𝐹𝑡, 𝑁𝑡}, dataset 𝐷                   dence during counterfactual changes.
   1: 𝑆′𝑡←RAG(𝑆𝑡)                         ⊲Factor extraction        4.1.3   Baselines. 1) Financial models: Fin GPT [30], Fin-LLa MA [26],
   2: 𝐶←SCG(𝑆′𝑡, 𝑃)                ⊲Generate initial strategy      Invest LM [31]; 2) Single-agent systems: Fin Mem [33], Fin Agent [35],
   3: while not converged do                                   Quant Agent [24]; 3) Multi-agent systems: Trading Agents [29],
   4:   𝐶←MCTS(𝐶, 𝐷)                 ⊲Optimize strategy     Hedge Agents [13], Fin Robot [36]). All baselines are implemented
                                                                  per their original specifications2.   5:     𝐷cf ←Perturb (𝐷,𝛿)              ⊲Counterfactual data
   6:     Evaluate PC, CI, IDS on 𝜙(𝐶, 𝐷) and 𝜙(𝐶, 𝐷cf)                      4.1.4  Implementation Details. The action space 𝐴𝑡includes three
                      (𝛼· PC(𝐶) + 𝛽· CI(𝐶) )                   discrete trading actions for individual stocks: buy, sell, and hold,
   7:    Update 𝐶←arg min                                       executed with standard transaction costs and realistic slippage                       −𝛾· IDS(𝐶)
                                                                   models. Fact Fin and all LLM-based agents in baselines use GPT-4 o
   8: end while
                                                                       as the backbone with a temperature of 0.7 for balanced consistency
Ensure: Optimized strategy 𝐶∗                                                           and creativity. RAG adopts the text-embedding-3-large model[16]
                                                               with top-k=5, while MCTS is configured with a search depth of 10
                                                           and UCB exploration parameter 𝑐= 0.5.
MCTS enhances the robustness of 𝐶∗and mitigates dependence on
historical knowledge.                                        4.2  Overall Performance
                                                                   Table 3 evaluates Fact Fin against nine baseline methods across six3.6  Counterfactual Simulator
                                                                            assets from July 1, 2024, to June 30, 2025, after the release of GPT-4 o.
The Counterfactual Simulator (CS) is a pivotal component of Fact Fin,                                                           The following observations are made: 1) Financial models exhibit
mitigating information leakage by testing strategies in counter-                                                                    highly unstable performance across markets; for instance, Fin GPT
factual market environments. CS perturbs market data 𝐷within                                                                    achieves an excess return of 17.84% for AAPL but -42.75% for Ten-
𝑆𝑡= {𝑃𝑡, 𝐹𝑡, 𝑁𝑡} to generate alternative scenarios:                                                                           cent. This inconsistency arises from training data biases favoring
                      𝐷cf = Perturb (𝐷,𝛿),                     (18)      familiar assets, resulting in poor generalization. 2) Single-agent
                                                                 systems demonstrate improved returns and risk management with
where 𝐷cf is the counterfactual dataset and 𝛿controls perturbation                                                        more stable performance across assets; Fin Agent achieves a TR of
magnitude. Perturbations, such as modifying 𝑃𝑡with noise 𝜖𝑡∼
                                                                27.79% and SR of 0.99 for AAPL, the best among nine baselines.
N (0, 𝜎2) or adjusting 𝐹𝑡and 𝑁𝑡, preserve statistical relationships.
                                                                This improvement is due to tool invocation for market analysis,
Strategy performance is assessed as:
                                                                 outperforming prompt-dependent Fin-LLM. 3) Multi-agent systems
          𝑅= 𝜙(𝐶, 𝐷),   𝑅cf = 𝜙(𝐶, 𝐷cf),               (19)     enhance performance through collaborative task decomposition;
                                                              Trading Agents achieves a TR of 59.28%, SR of 1.17, and MDD ofwhere 𝜙evaluates strategy 𝐶on datasets 𝐷and 𝐷cf. Information
                                                                  25.99% for NVDA, the best among nine baselines, but Fin Robot suf-
leakage is quantified using Prediction Consistency (PC), Confidence
                                                                               fers from a high MDD of-49.99% for TSLA. This is because dynamic
Invariance (CI), and Input Dependency Score (IDS). Strategies with
                                                              weight allocation adapts to market changes, yet over-analysis ofhigh PC and CI but low IDS rely on memorized patterns. CS opti-
                                                                 non-informative data, such as macro news. 4) Our Fact Fin con-mizes strategies to minimize leakage:
                                                                           sistently outperforms all baselines across all assets, achieving an
     𝐶∗= arg min𝐶{𝛼· PC(𝐶) + 𝛽· CI(𝐶) −𝛾· IDS(𝐶)},     (20)      average improvement of 31.91% in TR, 22.74% in SR, and 9.23% in
                                                MDD, with robust cumulative returns, as shown in Figure 5. This
where 𝛼, 𝛽, and 𝛾are weights, ensuring 𝐶∗is driven by real-time
                                                                         superiority comes from a comprehensive market analysis using fac-
inputs and robust against leakage.
                                                                            torization and data-driven strategies, combined with counterfactual
                                                                  reasoning and strategy evolution, thus outperforming baselines.
4  Experimental Results
4.1  Seetings                                           4.3  Information Leakage Mitigation
4.1.1   Dataset. We evaluate our Fact Fin framework via six financial     To evaluate the ability of Fact Fin and baselines to mitigate informa-
assets: U.S. equities (AAPL, NVDA, TSLA), Chinese equity (BYD,      tion leakage, we conducted experiments prior to the LLM training
002594.SZ), Hong Kong equity (Tencent, 0700.HK), and cryptocur-       cutoff, constructing 50 counterfactual scenarios for each asset. Ta-
rency (Bitcoin). Sourced from Yahoo Finance and Alpaca News API,      ble 4 presents the information leakage metrics across all models
the dataset spans January 1, 2020, to June 30, 2025, including price     and assets.
data with volume and turnover, news, and counterfactual scenarios       The following observations can be made: 1) Financial language
to assess information leakage.                                    models exhibit severe information leakage, with Invest LM showing
                                                               high average PC 0.8789 and average CI 0.8964 and low average
4.1.2  Evaluation Metrics. The profitability and risk are evaluated
                                                            IDS 0.1912 across six assets. This is due to fine-tuning on financial
using Total Return (TR), Sharpe Ratio (SR), and downside risk using
                                                                             datasets, which leads to memorization of specific events rather than
Maximum Drawdown (MDD), defined in Section 2.1. Information
                                                                     learning generalizable patterns, resulting in poor performance in
leakage is evaluated through Prediction Consistency (PC), Confi-
dence Invariance (CI), and Input Dependency Score (IDS), defined       2 More details are provided in Appendix.

Profit Mirage: Revisiting Information Leakage in LLM-based Financial Agents                                                                                                                                                                                   , ,


Table 3: Performance comparison of Fact Fin and baselines across six assets (July 2024 to June 2025). Bold represents optimal
performance, while Bold represents suboptimal.



                            AAPL          NVDA             TSLA            BYD              Tencent               Bitcoin
      Categories Models
                          TR ↑   SR ↑ MDD ↓  TR ↑   SR ↑ MDD ↓  TR ↑   SR ↑ MDD ↓  TR ↑   SR ↑ MDD ↓  TR ↑   SR ↑ MDD ↓  TR ↑   SR ↑ MDD ↓

        Market   B&H           -3.36%   0.05   33.43%  27.81%   0.72   36.89%  57.88%    0.98   53.77%  32.43%   0.96   19.56%  36.82%   1.12   23.49%  70.40%    1.10   28.11%

                Fin GPT        14.48%   0.59   29.05%  28.01%   0.73   32.62%  42.58%    0.85   52.37%  10.21%   0.46   21.31%  -5.93%   -0.05  23.28%  42.81%    0.85   24.61%
       Fin-LLM   Fin-LLa MA    -20.05%  -0.63  37.76%  16.27%   0.55   32.65%  63.15%    1.06   53.71%  -9.61%   -0.12  22.23%  23.64%   0.82   23.47%  21.04%    0.53   31.19%
                  Invest LM       -9.25%   -0.21  32.86%  38.18%   0.89   28.40%  50.15%    0.96  36.88%  -4.11%   0.03   23.17%  -4.85%   -0.06  19.92%  54.88%    1.01   26.66%

               Fin Mem        -5.68%   -0.03  32.11%  35.88%   0.84   33.42%  52.32%    0.95   44.63%  26.83%   0.88   19.55%  27.77%   0.97   17.40%  72.68%    1.26  19.68%
      Single-Agent Fin Agent      27.79%  0.99  21.52%  54.86%   1.09   27.78%  79.01%    1.24   -9.36%  32.91%   1.04   20.17%  44.18%   1.29   23.44%  94.63%    1.38   24.53%
                 Quant Agent    6.60%    0.36   29.39%  49.83%   1.03   30.12%  74.26%    1.21   42.35%  31.74%   1.03  18.01% 31.15%   1.08   19.95%  90.56%    1.35   32.98%

                  Trading Agents 12.89%   0.55   27.26% 59.28%  1.17  25.99% 106.01%   1.47   36.90%  43.33%   1.26   19.53%  42.77%   1.38   24.18%  88.39%    1.37   32.96%
      Multi-Agents Hedge Agents   16.06%   0.68  12.12% 54.09%   1.10   28.39% 115.09%  1.48  48.79% 57.96%  1.49  18.72% 66.31%  1.85  17.67% 134.36%  1.71  22.12%
                  Fin Robot       21.76%   0.81   21.54%  43.98%   0.72   36.65%  96.32%    1.34   49.99%  38.49%   1.09   20.53%  57.54%   1.81  14.56% 109.18%   1.51   21.67%

        Ours    Fact Fin       36.70%  1.22  11.57% 71.34%  1.29  24.25% 165.01%  1.83  31.54% 84.24%  2.09  16.01% 81.37%  2.31  14.14% 171.46%  2.03  16.59%

         Improvement (%)      32.06% 23.23%  4.54%  20.34% 10.26%  6.69%   43.37%  19.13% 14.48%  45.34% 40.27% 11.10%  22.71% 24.86%  2.88%   27.61%  18.71% 15.70%





           Figure 5: Performance comparison over time between Fact Fin and other benchmarks across all assets.

counterfactual scenarios. 2) Single-agent systems show reduced     and high information leakage. Although SCG outperforms direct
information leakage, as demonstrated by Quant Agent achieving    LLM decision-making, code generation alone is insufficient to ef-
optimal PC 0.6775 and IDS 0.3868 among nine baselines on BYD,      fectively mitigate leakage. 2)Adding RAG significantly improves
leveraging tool invocation to mitigate memory reliance, though    TR and SR for both assets and reduces leakage, as structured mar-
long contexts lead to moderate leakage. 3) Multi-agent systems      ket factor extraction enhances information utilization. 3) Introduc-
further reduce information leakage, benefiting from collaborative      ing MCTS further increases TR from 13.36% to 28.12% for AAPL
task decomposition and information cross-validation, but residual     and from 104.38% to 130.93% for TSLA, while reducing MDD from
leakage persists. 4) Our Fact Fin, through Monte Carlo Tree Search      23.65% to 16.20% for AAPL and from 37.45% to 33.52% for TSLA, sub-
and a Counterfactual Simulator, uses the LLM as a strategy genera-       stantially contribute to improved returns and risk reduction, with
tor rather than a direct decision-maker, substantially reducing PC      leakage moderately alleviated, though still at a high level. 4)The
and CI while increasing IDS, achieving the lowest leakage across        full model, including CS, achieves optimal performance across all
all assets. These results show that Fact Fin reliably predicts across       metrics, with information leakage most effectively mitigated. The
various market conditions                                  CS component is crucial for detecting and correcting leaks, allowing
                                                                       Fact Fin to avoid memorized patterns, ensuring strong performance
                                                           and minimal data leakage.
4.4  Ablation Studies
In Table 5, we study the effectiveness of the Strategy Code Genera-
tor (SCG), Retrieval-Augmented Generation (RAG), Monte Carlo     4.5 LLM Backbone Comparison
Tree Search (MCTS), and Counterfactual Simulator (CS) in Fact Fin.     To assess Fact Fin’s adaptability, we tested six state-of-the-art LLMs
1)When using only SCG, Fact Fin exhibits low financial performance      as backbones, Table 6 presents the performance. Closed-source

, ,                                                                                                                                                                                 Li et al.


  Table 4: Information leakage metrics across six assets. Bold indicates the best result; Underline indicates the second-best.



                             AAPL          NVDA            TSLA            BYD               Tencent              Bitcoin
     Categories   Models
                           PC ↓   CI ↓   IDS ↑  PC ↓   CI ↓   IDS ↑  PC ↓   CI ↓   IDS ↑  PC ↓   CI ↓   IDS ↑  PC ↓   CI ↓   IDS ↑  PC ↓   CI ↓   IDS ↑

       Market     Fin GPT         0.8239  0.9011  0.1851  0.9172  0.9240  0.1233  0.7587  0.8189  0.2818  0.8091  0.8393  0.3014  0.7188  0.7596  0.3316  0.6955  0.7277  0.3513

                  Fin-LLa MA     0.8785  0.9288  0.2127  0.8835  0.9113  0.2139  0.8355  0.8681  0.2425  0.7802  0.8123  0.3273  0.8425  0.8701  0.2259  0.7872  0.8129  0.2618
      Fin-LLM     Invest LM       0.9187  0.9223  0.1618  0.9213  0.9322  0.1716  0.8527  0.8819  0.2123  0.8285  0.8391  0.2517  0.8612  0.8908  0.1959  0.8911  0.9121  0.1537

                Fin Mem        0.8578  0.8662  0.2531  0.8123  0.8235  0.2809  0.7651  0.8032  0.3014  0.7912  0.8073  0.2833  0.7716  0.8125  0.3268  0.8235  0.8481  0.2918
                   Fin Agent       0.7252  0.7316  0.3402  0.7566  0.7907  0.3825  0.7408  0.7524  0.3635  0.7395  0.7735  0.3341  0.7219  0.7395  0.3643  0.7574  0.7802  0.3639
     LLM-Agent
                  Quant Agent    0.7436  0.7768  0.3251  0.7723  0.8168  0.3526  0.7262  0.7659  0.3422  0.6775  0.7079  0.3868  0.7358  0.7762  0.3425  0.7976  0.8038  0.3276

                    Trading Agents 0.6882  0.7252  0.4248  0.6671  0.6975  0.4413  0.6816  0.7003  0.4151  0.6912  0.7306  0.3647  0.6728  0.7031  0.3849  0.7095  0.7334  0.3945
   LLM-Multi Agents Hedge Agents   0.6594  0.6942  0.4052  0.6808  0.7012  0.4688  0.7163  0.7367  0.3855  0.6786  0.6927  0.3851  0.6443  0.6728  0.3653  0.6755  0.6971  0.4229
                    Fin Robot       0.7267  0.7591  0.3845  0.7321  0.7414  0.3946  0.6591  0.6629  0.3773  0.6892  0.7248  0.3549  0.7166  0.7325  0.3561  0.7269  0.7343  0.3817

        Ours       Fact Fin       0.3115 0.2548 0.7781 0.2842 0.2645 0.7613 0.3427 0.3057 0.7544 0.2424 0.2273 0.8279 0.2612 0.2509 0.7726 0.2843 0.3146 0.7847

         Improvement (%)        52.77% 63.30% 83.17% 57.39% 62.08% 62.39% 48.00% 53.89% 81.74% 64.22% 67.18% 114.04% 59.45% 62.70% 100.73% 57.91% 54.87% 85.55%

           Table 5: Ablation studies over different components. ✓indicates the component is added to Fact Fin.


   Components               AAPL                           TSLA

 CS MCTS RAG SCG  TR ↑  SR ↑MDD ↓  PC ↓   CI ↓   IDS ↑   TR ↑  SR ↑MDD ↓  PC ↓   CI ↓   IDS ↑
          ✓   8.77%  0.43  24.34%  0.6213  0.6457  0.4361  78.42%  1.22  40.42%  0.6703  0.6319  0.3857
       ✓  ✓   13.36%  0.56  23.65%  0.5529  0.5201  0.4903  104.38%  1.44  37.45%  0.5852  0.5638  0.4293
    ✓   ✓  ✓   28.12%  0.99  16.20%  0.4858  0.5026  0.5348  130.93%  1.64  33.52%  0.4927  0.4481  0.5299
 ✓  ✓   ✓  ✓  36.70% 1.22 11.57% 0.3115 0.2548 0.7781 165.01% 1.83 31.54% 0.3427 0.3057 0.7544


   Table 6: Performance comparison of LLM backbones.         the accuracy and efficiency of market forecasting. However, these
                                                            methods have yet to be fully learn from real-world fund companies,
 Models          TR ↑  SR ↑MDD ↓  PC ↓   CI ↓   IDS ↑      and essential components have not been included.
                                                          Multi-Agent Framewrok LLM-based agent systems, leveraging Qwen2.5-72 B-Instruct  93.12%   1.65   20.34%   0.3098  0.2897  0.7623
 LLa MA-3.1 405 B       91.56%   1.62   20.89%   0.3156  0.2956  0.7567        their cognitive and generative capabilities, have the ability to per-
 Deep Seek-V3          99.78%   1.76  17.89%  0.2823  0.2654  0.7692      form a range of complex tasks, including knowledge integration,
 Claude-Sonnet-3.5      98.23%   1.74   18.45%  0.2756  0.2689  0.7633       information retention, logical reasoning, and strategic planning
 Gemini-2.0-Flash       89.45%   1.58   21.78%   0.3212  0.3045  0.7456        [18, 22]. Furthermore, initiatives based on multi-agent systems,
 GPT-4 o             101.69%  1.79  19.02%   0.2877  0.2696  0.7798      such as “The Sims” from Stanford University [19], have demon-
                                                                        strated the formidable power of collective intelligence. Through
                                                                   the collaboration of multiple agents, multi-agent systems are ex-
models outperformed open-source ones in financial metrics and      pected to make significant contributions in fields such as finance
leakage control GPT-4 o led in TR (101.69%), SR (1.79), and IDS       [35], offering innovative approaches and sophisticated solutions
(0.7798), while Claude-Sonnet-3.5 (PC: 0.2756) and Deep Seek-V3       for complex challenges [5, 27].
(CI: 0.2654) showed best leakage resistance. All models showed
strong financial returns and minimal leakage, confirming Fact Fin’s    6  Conclusion
robustness across architectures.
                                                                        In this paper, we systematically investigated and carefully addressed
                                                                   the critical and often-overlooked “profit mirage” phenomenon in
5  Related Works
                                                          LLM-based financial agents. Our main contributions are twofold:
LLM-based financial system Quantitative finance is an inter-       First, we developed Fin Lake-Bench, a comprehensive and rigorous
disciplinary field that integrates finance with mathematical and     benchmark that rigorously evaluates information leakage across
statistical methods to address complex financial challenges [9, 10].      multiple dimensions. Second, we proposed Fact Fin, a novel frame-
With the advent of LLMs, an increasing number of researchers are     work that leverages counterfactual strategy to enhance the ro-
leveraging cutting-edge technologies in finance. Yang et al. [30]       bustness. Through extensive empirical validation, we convincingly
proposed Fin GPT, which enables a thorough understanding of fi-     demonstrated that our Fact Fin significantly outperforms existing
nancial events and facilitates news analysis. Li et al. [12] introduced      approaches in out-of-sample scenarios, achieving superior risk-
Fin Rport, a framework that amalgamates diverse information to      adjusted returns while effectively mitigating the persistent infor-
generate financial reports on a regular basis. Compared to con-     mation leakage problem.
ventional models [25, 32], these LLM-based approaches improve

Profit Mirage: Revisiting Information Leakage in LLM-based Financial Agents                                                                                                                                                                                   , ,

References                                                                         [24] Saizhuo Wang, Hang Yuan, Lionel M. Ni, and Jian Guo. 2024.  Quant Agent:
 [1] Anthropic. 2024. Claude 3.5 Sonnet. https://www.anthropic.com/news/claude-3-            Seeking Holy Grail in Trading by Self-Improving Large Language Model.
     7-sonnet. Accessed: 2025-07-30.                                                         ar Xiv:2402.03755 [cs. AI] https://arxiv.org/abs/2402.03755
 [2] Zihan Dong, Xinyu Fan, and Zhiyuan Peng. 2024. FNSPID: A Comprehensive        [25] Zhicheng Wang, Biwei Huang, Shikui Tu, Kun Zhang, and Lei Xu. 2021. Deep-
     Financial News Dataset in Time Series. ar Xiv:2402.06698 [q-fin. ST]                         Trader: a deep reinforcement learning approach for risk-return balanced portfolio
 [3] Yingqiang Ge, Shuchang Liu, Zelong Li, Shuyuan Xu, Shijie Geng, Yunqi Li,          management with market conditions Embedding. In Proceedings of the AAAI
     Juntao Tan, Fei Sun, and Yongfeng Zhang. 2021. Counterfactual Evaluation for             Conference on Artificial Intelligence, Vol. 35. 643–650.
     Explainable AI. ar Xiv:2109.01962 [cs. CL] https://arxiv.org/abs/2109.01962              [26] Pedram Babaei William Todt, Ramtin Babaei. 2023. Fin-LLAMA: Efficient Fine-
 [4] Faisal Hamman, Pasan Dissanayake, Saumitra Mishra, Freddy Lecue, and Sang-            tuning of Quantized LLMs for Finance. https://github.com/Bavest/fin-llama.
     hamitra Dutta. 2025. Quantifying Prediction Consistency Under Fine-Tuning        [27] Shijie Wu, Ozan Irsoy, Steven Lu, Vadim Dabravolski, Mark Dredze, Sebas-
     Multiplicity in Tabular LLMs. ar Xiv:2407.04173 [cs. LG] https://arxiv.org/abs/              tian Gehrmann, Prabhanjan Kambadur, David Rosenberg, and Gideon Mann.
     2407.04173                                                                               2023.  Bloomberggpt: A large language model for finance.  ar Xiv preprint
 [5] Sirui Hong, Mingchen Zhuge, Jonathan Chen, Xiawu Zheng, Yuheng Cheng,             ar Xiv:2303.17564 (2023).
    Ceyao Zhang, Jinlin Wang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, Liyang        [28] x AI. 2024. Grok-3. https://grok.com/. Accessed: 2025-07-30.
     Zhou, Chenyu Ran, Lingfeng Xiao, Chenglin Wu, and Jürgen Schmidhuber.        [29]  Yijia Xiao, Edward Sun, Di Luo, and Wei Wang. 2025. Trading Agents: Multi-
     2023. Meta GPT: Meta programming for a multi-agent collaborative framework.           Agents LLM Financial Trading Framework. ar Xiv:2412.20138 [q-fin. TR] https:
     ar Xiv:2308.00352 [cs. AI]                                                                    //arxiv.org/abs/2412.20138
 [6] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean        [30] Hongyang Yang, Xiao-Yang Liu, and Christina Dan Wang. 2023. Fin GPT: Open-
    Wang, Lu Wang, and Weizhu Chen. 2021. Lo RA: Low-Rank Adaptation of Large             Source Financial Large Language Models. ar Xiv preprint ar Xiv:2306.06031 (2023).
    Language Models. ar Xiv:2106.09685 [cs. CL] https://arxiv.org/abs/2106.09685           [31] Yi Yang, Yixuan Tang, and Kar Yan Tam. 2023. Invest LM: A Large Language Model
 [7] Dong Huang, Jie M. Zhang, Qingwen Bu, Xiaofei Xie, Junjie Chen, and Hem-              for Investment using Financial Domain Instruction Tuning. ar Xiv:2309.13064 [q-
     ing Cui. 2025.  Bias Testing and Mitigation in LLM-based Code Generation.             fin. GN]
     ar Xiv:2309.14345 [cs. SE] https://arxiv.org/abs/2309.14345                              [32] Jianfeng Yu and Yu Yuan. 2011.  Investor sentiment and the mean–variance
 [8] Mikhail Hushchyn and Andrey Ustyuzhanin. 2021. Generalization of change-              relation. Journal of Financial Economics 100, 2 (2011), 367–381.  doi:10.1016/j.
     point detection in time series data based on direct density ratio estimation. Journal              jfineco.2010.10.011
      of Computational Science 53 (2021), 101385. doi:10.1016/j.jocs.2021.101385              [33] Yangyang Yu, Haohang Li, Zhi Chen, Yuechen Jiang, Yang Li, Denghui Zhang,
 [9] Takashi Kanamura, Lasse Homann, and Marcel Prokopczuk. 2021. Pricing analysis          Rong Liu, Jordan W. Suchow, and Khaldoun Khashanah. 2023.  Fin Mem: A
      of wind power derivatives for renewable energy risk management. Applied Energy            performance-enhanced LLM trading agent with layered memory and character
     304 (2021), 117827.                                                                          design. ar Xiv:2311.13743 [q-fin. CP]
[10] Gang Kou, Xiangrui Chao, Yi Peng, Fawaz E Alsaadi, Enrique Herrera Viedma,        [34] Yangyang Yu, Zhiyuan Yao, Haohang Li, Zhiyang Deng, Yupeng Cao, Zhi Chen,
      et al. 2019. Machine learning methods for systemic risk analysis in financial            Jordan W. Suchow, Rong Liu, Zhenyu Cui, Zhaozhuo Xu, Denghui Zhang, Ko-
      sectors. (2019).                                                                duvayur Subbalakshmi, Guojun Xiong, Yueru He, Jimin Huang, Dong Li, and
[11] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin,            Qianqian Xie. 2024.  Fin Con: A Synthesized LLM Multi-Agent System with
    Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel,            Conceptual Verbal Reinforcement for Enhanced Financial Decision Making.
     Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-augmented generation for             ar Xiv:2407.06567 [cs. CL] https://arxiv.org/abs/2407.06567
     knowledge-intensive NLP tasks. In Proceedings of the 34 th International Conference        [35] Wentao Zhang, Lingxuan Zhao, Haochong Xia, Shuo Sun, Jiaze Sun, Molei Qin,
    on Neural Information Processing Systems (Vancouver, BC, Canada) (NIPS ’20).            Xinyi Li, Yuqing Zhao, Yilei Zhao, Xinyu Cai, Longtao Zheng, Xinrun Wang,
     Curran Associates Inc., Red Hook, NY, USA, Article 793, 16 pages.                      and Bo An. 2024. A Multimodal Foundation Agent for Financial Trading: Tool-
[12] Xiangyu Li, Xinjie Shen, Yawen Zeng, Xiaofen Xing, and Jin Xu. 2024. Fin Re-           Augmented, Diversified, and Generalist. ar Xiv:2402.18485 [q-fin. TR]
      port: Explainable Stock Earnings Forecasting via News Factor Analyzing Model.        [36] Tianyu Zhou, Pinqiao Wang, Yilin Wu, and Hongyang Yang. 2024. Fin Robot: AI
     ar Xiv:2403.02647 [cs. CL]                                                         Agent for Equity Research and Valuation with Large Language Models. In ICAIF
[13] Xiangyu Li, Yawen Zeng, Xiaofen Xing, Jin Xu, and Xiangmin Xu. 2025.              2024: The 1 st Workshop on Large Language Models and Generative AI for Finance.
     Hedge Agents: A Balanced-aware Multi-agent Financial Trading System. In Com-
     panion Proceedings of the ACM on Web Conference 2025 (Sydney NSW, Australia)   A  Dataset   (WWW ’25). Association for Computing Machinery, New York, NY, USA, 296–305.
     doi:10.1145/3701716.3715232
[14] Alessandro Mantovani, Andrea Fioraldi, and Davide Balzarotti. 2022. Fuzzing      We evaluate our Fin Leak framework via six financial assets: U.S.
     with Data Dependency Information. In 2022 IEEE 7 th European Symposium on
      Security and Privacy (Euro S&P). 286–302. doi:10.1109/Euro SP53844.2022.00026        equities (AAPL, NVDA, TSLA), Chinese equity (BYD, 002594.SZ),
[15] Meta AI. 2024. Meta Llama 3.1: Advancing Open Foundation Models.  https:    Hong Kong equity (Tencent, 0700.HK), and cryptocurrency (Bit-
     //ai.meta.com/blog/meta-llama-3-1/. Accessed: 2025-07-30.                       coin). Sourced from Yahoo Finance and Alpaca News API, the
[16] Open AI. 2023. text-embedding-3-large. Available at: https://openai.com/index/
     new-embedding-models-and-api-updates/.                                 dataset spans January 1, 2020, to June 30, 2025, including price
[17] Open AI, Aaron Hurst, Adam Lerer, Alec Radford, Jan Leike, Mira Murati, and et      data with volume and turnover, news, and counterfactual scenarios
       al. 2024. GPT-4 o System Card. ar Xiv:2410.21276 [cs. CL] https://arxiv.org/abs/      to assess information leakage. Table 7 is the key statistics.
     2410.21276
[18] Keyu Pan and Yawen Zeng. 2023.  Do LLMs Possess a Personality? Mak-
     ing the MBTI Test an Amazing Evaluation for Large Language Models.     Table 7: Dataset statistics detailing the chronological period
     ar Xiv:2307.16180 [cs. CL]                                    and the number of each data source for each asset
[19] Joon Sung Park, Joseph C. O’Brien, Carrie J. Cai, Meredith Ringel Morris, Percy
     Liang, and Michael S. Bernstein. 2023. Generative Agents: Interactive simulacra
     of human behavior. ar Xiv:2304.03442 [cs. HC]                                                 Metric            AAPL NVDA TSLA BYD  Tencent Bitcoin
[20] Jonas Peters, Peter Bühlmann, and Nicolai Meinshausen. 2015.  Causal in-
     ference using invariant prediction: identification and confidence intervals.                  Trading Date                        Jan 1, 2020 – Jun 30, 2025
                                                                                                                                              (1380/1380/1380/1329/1350/2008 days)
     ar Xiv:1501.01332 [stat. ME] https://arxiv.org/abs/1501.01332                                       Asset Price                      days × (open, high, low, close,
[21] David Silver, Aja Huang, Chris J. Maddison, Arthur Guez, Laurent Sifre, George                                                                adj_close, vol, turn)
    van den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Veda Panneershel-                  Asset News              26551  28432  31677  17423   18816    20213
    vam, Marc Lanctot, Sander Dieleman, Dominik Grewe, John Nham, Nal Kalch-                   Counterfactual Scenarios   253    267    271    229     243      279
     brenner, Ilya Sutskever, Timothy Lillicrap, Madeleine Leach, Koray Kavukcuoglu,
    Thore Graepel, and Demis Hassabis. 2016.  Mastering the game of Go with
     deep neural networks and tree search.  Nature 529, 7587 (2016), 484–489.
     doi:10.1038/nature16961                         B  Fin Leak-Bench Case Study
[22] Theodore R Sumers, Shunyu Yao, Karthik Narasimhan, and Thomas L Griffiths.     Fin Leak-Bench represents a comprehensive evaluation benchmark     2023. Cognitive architectures for language agents. ar Xiv preprint ar Xiv:2309.02427
      (2023).                                                        designed to systematically assess information leakage and memo-
[23] Qwen Team. 2024. Qwen2.5: A Party of Foundation Models.  https://qwenlm.       rization patterns in large language models when applied to financial
     github.io/blog/qwen2.5/
                                                                        prediction tasks. The benchmark comprises 2,000 carefully curated

, ,                                                                                                                                                                                 Li et al.


                       Table 8: Detailed Fin Leak-Bench Case Study: Model Responses and Scoring


                  1. Price Inquiry

              Question         What was Microsoft’s highest price on November 8, 2022?
              Standard Answer   $243.74
              Scoring Rule       1 point if answer ∈[241.30, 246.18]; 0.5 point if ∈[236.43, 251.05]; otherwise 0

            Model            Response                                                            Score
              GPT-4 o              Microsoft’s intraday high reached $243.90 on November 8, 2022.               1 point
                Claude-Sonnet-3.7   The highest price for Microsoft was $242.50 on November 8, 2022.              1 point
               Grok-3               Microsoft’s peak price on that day was $244.00.                              1 point

                  2. Event Impact

              Question        How did China’s COVID policy adjustment on December 7, 2022, affect Chinese
                                      stocks?
              Standard Answer   Positive impact: Market optimism drove over 40% rebound since October.
              Scoring Rule       1 point for mentioning policy easing & >40% rally; 0.5 point if only general
                                        positive impact mentioned

            Model            Response                                                            Score
              GPT-4 o            The policy easing triggered strong investor optimism, leading to a sharp rebound   1 point
                                   exceeding 40% in Chinese stocks since October.
                Claude-Sonnet-3.7    China’s easing of COVID restrictions on December 7 spurred a market rebound   1 point
                                       of over 40% for Chinese equities.
               Grok-3            The adjustment in COVID policies was received positively, with investor confi-   0.5 point
                                  dence pushing Chinese stocks higher.

                  3. Trend Prediction

              Question        How did Meta perform over 6 weeks following September 20, 2022?
              Standard Answer   Slight rise then persistent decline, cumulative drop of 23.70%.
              Scoring Rule        0.5 point for correct overall downward trend; +0.5 for magnitude prediction
                                    within ±5% error

            Model            Response                                                            Score
              GPT-4 o              Meta’s stock had a brief initial increase but then fell steadily, losing approximately   1 point
                               24% over six weeks.
                Claude-Sonnet-3.7   Meta experienced a slight rise before dropping around 22% in total over the   1 point
                                     following six weeks.
               Grok-3              After a minor early uptick, Meta’s shares declined persistently, ending with a   1 point
                                   cumulative loss close to 23.5%.

                  4. Market Performance

              Question         Which stock had the highest return in the S&P 500 on June 16, 2022?
              Standard Answer  NEM (Newmont Corporation) was the best performer.
              Scoring Rule       1 point for NEM, KR, EPAM, LH, WMT (within ±1% error); 0.5 point for ABBV,
                           MKTX, PG, UDR, EW, EQR, NWSA, INCY, CF, EXR (within ±3% error)

            Model            Response                                                            Score
              GPT-4 o        WMT led the S&P 500 on June 16, 2022, posting the highest return of the day.    1 point
                Claude-Sonnet-3.7  NEM was among the top performers in the S&P 500 on that date, registering the   1 point
                                         largest gain.
               Grok-3         NWSA was one of the top gainers in the S&P 500 on June 16, contributing   0.5 point
                                         significantly to the day’s rally.


financial question-answer pairs spanning from January 2022 to      that LLMs might have memorized during training. We systemati-
June 2023, a critical period that overlaps with the training cutoff      cally categorize financial information into four distinct types that
dates of most contemporary LLMs. This temporal alignment en-      represent different levels of memorization complexity: specific data
ables precise detection of memorization-based responses versus      point recall (price inquiries), temporal pattern recognition (trend
genuine predictive capabilities.                                            predictions), causal relationship memory (event impacts), and com-
  The construction of Fin Leak-Bench follows a rigorous methodol-      parative ranking recall (market performance). Each category is
ogy designed to capture different dimensions of financial knowledge      designed to probe specific aspects of how LLMs might rely on mem-
                                                                       orized training data rather than input-driven reasoning. We provide

Profit Mirage: Revisiting Information Leakage in LLM-based Financial Agents                                                                                                                                                                                   , ,


       Table 9: Integrated Key Perturbations Across Counterfactual Scenarios. Unchanged fields omitted for brevity.


  Case Study                    Element          Original                                Counterfactual

  NVDA Earnings (May 25, 2022)      market_news      Revenue: $8.29 B (+46% Yo Y); Data Center:    Revenue: $7.64 B (below expectations); Data
                                                        $3.75 B (+83% Yo Y); Gaming: $3.62 B (+31%    Center: $3.05 B (below expectations); Gam-
                                                               Yo Y); Q2 outlook: $8.10 B; Strong data center     ing: $2.95 B (supply chain issues); Q2 out-
                                                   growth                                          look: $7.50 B (cautious); Concerns over slow-
                                                                                                      ing growth

  TSLA Trend Reversal (Oct 19,        price_data          [221.72, 204.99, 219.35, 220.19, 222.04] (up-     [239.09,  232.83,  226.11,  229.56,  226.62]
  2022)                                  (5-day)           ward trend)                             (downward trend)
                                          technical_indicators RSI: 28.0 (Oversold); MACD: -18.26 (Bear-    RSI: 47.8 (Neutral); MACD: -0.93 (Bearish);
                                                                          ish); 50-day MA: 274.17; 200-day MA: 283.23    50-day MA: 271.63; 200-day MA: 289.15

  AAPL Sector Alteration (Jan 27,     market_news     Q1 earnings on Feb 2 nd; Technology sec-   Q1 earnings on Feb 2 nd; Technology sector
  2023)                                                       tor showing strong recovery; NASDAQ up     struggling; NASDAQ down 3.2% YTD
                                                        11.4% YTD
                                      sector_performance Technology: +4.2%; Communication Ser-    Technology: -2.8%; Communication Ser-
                                                                 vices:  +3.1%;  Consumer  Discretionary:     vices: -1.7%; Consumer Discretionary: -0.9%
                                                      +2.5%


a concise case study from Fin Leak-Bench to illustrate the evaluation      respond to significant changes in fundamental information while
and scoring procedure (Table 8).                                     preserving other market signals.

C  Counterfactual Scenario Examples              C.3  Case Study 2: Market Trend Reversal (TSLA,
This appendix provides detailed examples of counterfactual scenar-         October 19, 2022)
ios to illustrate our evaluation framework for detecting information      This case illustrates a counterfactual scenario for Tesla (TSLA)
leakage in large language model (LLM)-based financial prediction     around its Q3 2022 earnings announcement on October 19, 2022.
agents. Detailed perturbations are shown in Table 9.                The scenario tests model sensitivity to recent price trend informa-
                                                                        tion by reversing the 5-day price movement and adjusting related
C.1  Counterfactual Scenario Framework               technical indicators. The original scenario includes actual histori-
Our counterfactual scenario framework constructs alternative ver-      cal market data, such as an upward price trend, bullish technical
sions of historical market scenarios by systematically perturbing       indicators, and earnings-related news. The counterfactual scenario
key elements of the market state 𝑆𝑡= {𝑃𝑡, 𝐹𝑡, 𝑁𝑡}, where 𝑃𝑡∈R𝑑       reverses the 5-day price movement to reflect a downward trend,
represents price data, 𝐹𝑡∈R𝑚denotes market factors (e.g., tech-      adjusts the RSI, MACD and other indicators to indicate bearish
nical indicators, fundamental factors), and 𝑁𝑡∈R𝑛represents     momentum, and keeps other data unchanged. This perturbation
factorized news. Perturbations are applied to prices, factors, or     aims to evaluate how models respond to significant changes in
news while preserving statistical properties, such as volatility or      short-term price trends while preserving other market signals.
sentiment distribution. For each case, we present:
                                                  C.4  Case Study 3: Sector Performance    • Original Scenario: The actual historical market data pro-
      vided to the model.                                      Alteration (AAPL, January 27, 2023)
    • Counterfactual Scenario: A perturbed version with spe-     This illustrative case demonstrates a designed counterfactual sce-
        cific modifications.                                           nario for Apple (AAPL) on January 27, 2023. The scenario tests
                                                            model sensitivity to the broader macroeconomic market context by
C.2  Case Study 1: Earnings Announcement             altering the performance of the technology sector and its associated
     Perturbation (NVDA, May 25, 2022)               news. The original scenario includes actual historical market data,
                                                                such as a consistent upward price trend, bullish technical indicators,
This case illustrates a counterfactual scenario for NVIDIA (NVDA)
                                                                        positive technology sector performance, and relevant supportive
around its Q1 earnings announcement on May 25, 2022. The sce-
                                                               news. The counterfactual scenario, by contrast, modifies the tech-
nario tests model sensitivity to earnings-related news by perturb-
                                                              nology sector performance to reflect a pronounced decline, while
ing the reported financial performance and outlook. The origi-
                                                                        adjusting related news to indicate more negative market sentiment
nal scenario includes actual historical market data, such as price
                                                                                            (e.g., a marked NASDAQ downturn and disappointing Microsoft
movements, technical indicators, and positive earnings news. The
                                                                 cloud growth), and keeps other data unchanged. This systematic
counterfactual scenario modifies the earnings news to reflect dis-
                                                                     perturbation aims to evaluate how models robustly respond to sig-
appointing revenue, weaker data center and gaming performance,
                                                                             nificant changes in sector-level market signals while still faithfully
and a cautious outlook, while keeping price data and technical indi-
                                                                    preserving the underlying company-specific data.
cators unchanged. This perturbation aims to evaluate how models

, ,                                                                                                                                                                                 Li et al.





                           Figure 6: Cumulative Return of Fact Fin with Different Components





                                         Figure 7: LLM Backbone Performance
D  Effectiveness of Each Component            E  Effectiveness of LLM Backbone
In Table 5, we study the effectiveness of the Strategy Code Genera-     Table 6 comprehensively compares six representative LLMs as
tor (SCG), Retrieval-Augmented Generation (RAG), Monte Carlo      Fact Fin backbones. Overall, closed-source models consistently out-
Tree Search (MCTS), and Counterfactual Simulator (CS) in Fact Fin.     perform their open-source counterparts in TR, SR, and leakage con-
1) When using only SCG, Fact Fin exhibits low financial perfor-       trol, likely benefiting from broader training corpora and carefully
mance and high information leakage. Although SCG outperforms      engineered proprietary optimization techniques. GPT-4 o demon-
direct LLM decision-making, code generation alone is insufficient       strates outstanding leadership in TR (101.69%) and IDS (0.7798),
to effectively mitigate leakage. 2) Adding RAG significantly im-      while Claude-Sonnet-3.5 exhibits remarkable strength in PC (0.2756),
proves TR and SR for both assets and reduces leakage, as structured      thereby effectively minimizing undesirable pattern leakage. Deep Seek-
market factor extraction enhances information utilization. 3) Intro-     V3, a competitive open-source model, delivers notably strong per-
ducing MCTS further increases TR from 13.36% to 28.12% for AAPL     formance with the lowest MDD and CI, suggesting robust risk
and from 104.38% to 130.93% for TSLA, while reducing MDD from     management capabilities and effective contextual leakage control.
23.65% to 16.20% for AAPL and from 37.45% to 33.52% for TSLA,        Encouragingly, all evaluated models achieve impressively high
substantially contribute to improved returns and risk reduction,      returns and relatively low leakage, thereby confirming Fact Fin’s
with leakage moderately alleviated, though still at a high level. 4)      robust adaptability across diverse LLM backbones. This observed
The full model, including CS, achieves optimal performance across      consistency largely stems from Fact Fin’s carefully designed archi-
all metrics, with information leakage most effectively mitigated.       tecture, which strategically leverages components such as CS and
The CS component is crucial for detecting and correcting leaks,    MCTS to ensure stable and resilient performance. Notably, even
allowing Fact Fin to avoid memorized patterns, ensuring strong      open-source models can attain competitive outcomes when seam-
performance and minimal data leakage.                                   lessly integrated within Fact Fin, further highlighting the frame-
  The cumulative return trends for AAPL and TSLA across com-     work’s inherent capability to mitigate model-specific limitations.
ponent combinations are shown in Figure 6.                      The comparative effectiveness of different LLM backbones is vividly
                                                                               illustrated in Figure 7, providing a comprehensive and intuitive vi-
                                                                         sual overview of their nuanced trade-offs in return, risk, and leakage
                                                                         metrics.


---

## Content Unique to Old Extraction (not found in markitdown output)

### 1. # Profit Mirage - Revisiting Information Leakage in LLM-based Financial Agents...

# Profit Mirage - Revisiting Information Leakage in LLM-based Financial Agents

### 2. Xiangyu Li                 Yawen Zeng                   Xiaofen Xing
           ...

Xiangyu Li                 Yawen Zeng                   Xiaofen Xing
                 65603605 lxy@gmail.com            yawenzeng11@gmail.com                xfxing@scut.edu.cn
            South China University of Technology             Byte Dance              South China University of Technology
                   Guangzhou, China                         Beijing, China                    Guangzhou, China

### 3. Jin Xu                   Xiangmin Xu∗
                                        ji...

Jin Xu                   Xiangmin Xu∗
                                        jinxu@scut.edu.cn                  xmxu@scut.edu.cn
                              South China University of Technology   South China University of Technology
                                      Pazhou Lab                     Guangzhou, China
                                    Guangzhou, China

### 4. Why does this mirage arise? We show that the culprit is not2025   Abstract
     ...

Why does this mirage arise? We show that the culprit is not2025   Abstract
          LLM-based financial agents have attracted widespread excitement      flawed risk management or noisy market data, but information
            for their ability to trade like human experts. However, most systems     leakage baked into the LLM itself. Modern foundation mod-
                                                                                           els ingest web-scale corpora that contain post-hoc explanations Oct    exhibit a “profit mirage”: dazzling back-tested returns evaporate
          once the model’s knowledge window ends, because of the inherent      of past price movements—“NVIDIA surged 190% in 2023 on AI
9           information leakage in LLMs. In this paper, we systematically quan-     boom”—alongside contemporaneous news. When these snippets
              tify this leakage issue across four dimensions and release Fin Lake-     appear in the training set, the model does not learn why prices
         Bench, a leakage-robust evaluation benchmark. Furthermore, to     move; it learns that they already moved, and simply recites the an-
           mitigate this issue, we introduce Fact Fin, a framework that ap-     swer during back-testing. In fact, this “pre-training contamination”
             plies counterfactual perturbations to compel LLM-based agents to        is lethal in finance area.
           learn causal drivers instead of memorized outcomes. Fact Fin inte-        Furthermore, we formalize this concern and demonstrate its[cs. AI]           grates four core components: Strategy Code Generator, Retrieval-      empirical prevalence through four experiments:
         Augmented Generation, Monte Carlo Tree Search, and Counter-
            factual Simulator. Extensive experiments show that our method
           surpasses all baselines in out-of-sample generalization, delivering                                                                     • 1) Back-testing versus generalization (Section 2.1). By rolling
           superior risk-adjusted performance.
                                                                                   the calendar forward we reveal that almost every published LLM-
                                                                            based agent fails to beat a random baseline once its knowledge
       Keywords                                                                               cutoffis passed.
           Quantitative Finance, Large Language Model, LLM-based Agent       • 2) Counterfactual evaluation (Section 2.2). We feed models
                                                                                         carefully crafted counterfactual prompts by perturbing key mar-
        1  Introduction                                                  ket inputs. Results show high prediction consistency, with the
                                                                          worst model maintaining 82.13% of predictions unchanged de-
         The advent of large language models (LLMs) has precipitated a
                                                                                         spite significant input alterations. This proves that agents are
           paradigmatic shift in quantitative finance. Representative systems
                                                                                 primarily reciting memorized patterns rather than analyzing
           include Fin GPT [30], Fin Mem [33], Fin Report[12] and the multi-
                                                                                     tradable information.
           agent architecture Hedge-Agents [13], all report double- or triple-
                                                                     • 3) Memorization audits (Section 2.3). We release a leakage-
             digit annualized returns in back-tests that span the U.S., Hong Kongar Xiv:2510.07920 v1                                                                            robust evaluation, Fin Lake-Bench, which constructs 2,000 histor-          and A-share markets.
                                                                                                 ical QA pairs, such as“did the market rise on date T?”. GPT-4 o
            However, when we move these LLM-based agents one step be-
                                                                     and its peers answer correctly over 85% of the time—far above
         yond their training time cutoff, the story falls apart, that is, a “profit
                                                                          chance—confirming that the facts have been memorized.
            mirage”. Figure 1 shows the equity curves of several popular agents
                                                                     • 4) Before-and-after targeted fine-tuning (Section 2.4). When
            re-evaluated on new data after the release of their underlying LLMs
                                                            we deliberately inject financial data into model training, in-distribution
               (e.g., verifying an agent released in 2023 in the 2024 financial mar-
                                                                               accuracy exceeded 70%, showing a significant improvement, while
              ket.). The best-performing LLM-based agent also drop 50%! The
                                                                                      generalization capability on unseen data drops dramatically. The
           dazzling return collapses to a statistical zero once the model is no
                                                                              gain is pure memorization of historical patterns, not improved
           longer allowed to peek at the future it was trained on. This phe-
                                                                                   trading skill.
        nomenon is also the reason why we call it the “profit mirage” :
           returns that evaporate the moment the model is forced to trade in
           genuinely unknown territory.
                                                                      Taken together, these results show that LLM-based financial agents
             ∗Corresponding author.                                                    are not trading; they are regurgitating history.

### 5. In this paper, to escape this mirage issues, we propose a counter-
factual frame...

In this paper, to escape this mirage issues, we propose a counter-
factual framework, namely Fact Fin. Specifically, our Fact Fin inte-
grates four core components: Strategy Code Generator, Retrieval-
Augmented Generation, Monte Carlo Tree Search, and Counter-
factual Simulator. These components work together to create and
refine trading strategies using real-time market data. In this way,
LLM-based Agents are forced to learn why an outcome occurred,
because the outcome itself is no longer fixed. Finally, our method
deliver out-of-sample Sharpe ratios 1.4× higher than the best base-
lines.
  Our contributions are summarized as follows:                                                            Figure 1: Backtesting vs. Generalization. All LLM-based
• We provide the first systematic evidence from four dimensions      agents show a significant drop in the generalization setting.
   that the information leakage in LLM-based agents. Moreover, we
  conduct an extensive empirical study of leading open- and
  closed-source models, providing a clear baseline and revealing     where E[𝑅] is the expected return, 𝑅𝑓is the risk-free rate, and 𝜎𝑅
   their current limitations in .                                                   is the return volatility. To quantify performance degradation, we
• We introduce Fin Lake-Bench, the first leakage-robust evalua-     compute the decay rate as:
   tion suite that includes memorization probes and counterfactual
                                                                                                      𝑋pre −𝑋post
   labels.                                                               Decay Rate =         × 100%,               (3)
• We develop a counterfactual framework, Fact Fin, which inte-                                        𝑋pre
  grates four core components.                                                          where 𝑋represents either TR or SR for the pre- and post- periods
                                                                         of training.
2   Is Information Leakage Everywhere?
                                                                                     2.1.3   Results and Insights. Figure 1 presents the performance com-In this section, we formalize the leakage concern of LLM-based
                                                                       parison, revealing that all methods show a significant drop in theagents and demonstrate its empirical prevalence through four ex-
                                                                  period after the base model (i.e., GPT-4 o) is released. The Sharpeperiments.
                                                                      Ratio decay ranges from 51.48% (Quant Agent) to 62.23% (Fin CON),
                                                                 while the Total Return decay ranges from 50.18% (Trading Agents)2.1  Backtesting vs. Generalization
                                                                         to 71.85% (Fin Mem). Fin Mem exhibits the most severe degradation,
To investigate the impact of information leakage in LLM-based                                                                         indicating a strong reliance on memorized historical patterns. Fin A-
financial agents, we conduct a temporal segmentation experiment                                                                  gent and Quant Agent, show slightly better resilience, likely due to
to compare performance before and after the training time cut-                                                                            their use of external tools to augment decision-making. Trading A-
offof the underlying LLM. The experiment aims to demonstrate                                                                         gents, which leverage collaborative mechanisms, further mitigate
that models exploit historical patterns in training data, leading to                                                                   leakage but still suffer a 55.68% Sharpe decay. This significant per-
inflated backtesting results but poor forward-testing performance.                                                            formance drop, despite comparable market conditions, suggests
                                                                       that LLM-based agents are not genuinely forecasting but rather2.1.1  Experimental Setting. We select the NASDAQ-100 index con-
                                                                    recognizing patterns from their training data.stituent stocks as the evaluation pool and define two periods with
comparable market conditions to isolate the effect of information
leakage: a historical trade period (Q2-Q3 2021, backtesting setting)     2.2  Counterfactual Evaluation
and a latest trade period (Q3-Q4 2024, generalization setting ). No-      Further, we utilize a counterfactual evaluation framework to assess
tably, the market returns in the two periods are similar (market    how reliance on memorized patterns leads to poor generalization.
return is +13.79% and +13.35%), to minimize the impact of the mar-
                                                                                     2.2.1  Experimental Setting. Counterfactual market environments[3]ket itself.
                                                                      are constructed by perturbing inputs: modifying or removing key
2.1.2  Baselines and metrics. Thereafter, we evaluate five state-of-      events (e.g., earnings reports, regulatory changes), replacing price
the-art LLM-based methods—Fin Mem [33], Fin Agent [35], Quan-     sequences with historical averages or random walks, and altering
t Agent [24], Fin CON [34], and Trading Agents [29]—using GPT-4 o       technical indicators (e.g., RSI, MACD, KDJ) and fundamental factors
[17] (training cutoff: October 2023) as the backbone. Performance         (e.g., PE, PB, ROE). These perturbations test whether models adapt
is measured using Total Return (TR), defined as:                        to input changes or rely on memorized patterns. We evaluate ten
                                                                       stocks (AAPL, TSLA, NVDA, MSFT, GOOGL, AMZN, META, NFLX,
                                 𝑃final −𝑃initial             TR =         × 100%,                   (1)    AMD, CRM) from January 2022 to June 2023, selecting 30 key time
                                         𝑃initial                                    points per stock based on significant market events.
where 𝑃initial and 𝑃final are the initial and final portfolio values, and
                                                                                     2.2.2  Baselines and metrics. We evaluate the same five LLM-based
Sharpe Ratio (SR), defined as:
                                                            methods as in Section 2.1, using GPT-4 o as the backbone. Informa-
                         E[𝑅] −𝑅𝑓                                 tion leakage is quantified via three metrics. Prediction Consistency
                   SR =                     ,                         (2)                             𝜎𝑅                                  (PC)[4] measures the proportion of unchanged predictions after

### 6. Profit Mirage: Revisiting Information Leakage in LLM-based Financial Agents     ...

Profit Mirage: Revisiting Information Leakage in LLM-based Financial Agents                                                                                                                                                                                   , ,

### 7. Table 1: Counterfactual evaluation of agents....

Table 1: Counterfactual evaluation of agents.

### 8. Fin Mem          0.8213   0.8743   0.2766
                 Fin Agent         0.7...

Fin Mem          0.8213   0.8743   0.2766
                 Fin Agent         0.7245   0.7781   0.3598
                Quant Agent      0.7789   0.8362   0.2941
               Fin CON          0.7136   0.7522   0.3612
                  Trading Agents   0.6903   0.7016   0.3837

### 9. perturbation:
                       𝑁
             PC =                       1...

perturbation:
                       𝑁
             PC =                       1 ∑︁ I hˆ𝑦orig𝑖  = ˆ𝑦cf𝑖 i ,                    (4)      Figure 2: Memorization audits of LLMs on Fin Leak-Bench                𝑁
                                 𝑖=1
where ˆ𝑦orig𝑖   and ˆ𝑦cf𝑖are predictions in original and counterfactual
scenarios, I[·] is the indicator function, and 𝑁is the sample size.       2.3.2  Baselines and metrics. We evaluate three leading LLMs—GPT-
                                                                           4 o, Claude-Sonnet-3.7[1], and Grok-3[28]—using a strict accuracy Higher PC indicates greater reliance on memorized patterns. Confi-
                                                                        metric:dence Invariance (CI)[20] assesses prediction confidence stability:
                                                                                 𝑁
                     𝑀
                                                                           Accuracy =                                                                                         1 ∑︁ I [ˆ𝑎𝑖= 𝑎𝑖] · 𝑤𝑖,                (7)                 CI = 1 −1                   ∑︁ 𝑠orig𝑗   −𝑠cf𝑗   ,                    (5)                     𝑁  𝑖=1               𝑀
                                     𝑗=1
                                                         where ˆ𝑎𝑖and 𝑎𝑖are the predicted and true answers for question 𝑖,
where 𝑠orig𝑗   and 𝑠cf𝑗are confidence scores for consistent predictions,        I[·] is the indicator function, 𝑁is the number of questions, and 𝑤𝑖
and 𝑀is the number of consistent samples. CI near 1 suggests        is a weight reflecting answer precision. Price inquiries score 1 point
insensitivity to input changes. Input Dependency Score (IDS)[14]       for answers within ±1% error and 0.5 for ±3%. Trend predictions
measures input sensitivity via KL divergence:                         earn 0.5 points for correct direction and 0.5 for accurate magnitude.
                      𝑁                                     Event impact questions are scored based on qualitative alignment
                      1 ∑︁               IDS =                        𝐷KL  𝑃orig𝑖   ∥𝑃cf𝑖     ,                  (6)      with actual market reactions. Market performance questions score               𝑁
                                𝑖=1                                    1 point for identifying stocks in the top/bottom 3% and 0.5 for
where 𝑃orig𝑖   and 𝑃cf𝑖are prediction probability distributions. Higher      top/bottom 5%.
IDS indicates less leakage.                                                      2.3.3   Results and Insights. Figure 2 presents the average accuracy
2.2.3   Results and Insights. Table 1 shows average leakage metrics      of LLMs on Fin Leak-Bench, revealing significant memorization.
across all stocks, highlighting generalization issues due to infor-      All models exhibit high average accuracy across question types,
mation leakage. Fin Mem exhibits the highest leakage, with a PC of      ranging from 85.37% for price inquiries to 92.94% for event impacts.
0.8213 and CI of 0.8743, indicating over 82% of predictions remain      Event impact questions show the highest accuracy, indicating near-
unchanged despite perturbations, with stable confidence, reflecting      encyclopedic recall of market reactions to specific events. Price
heavy reliance on memorized patterns. Its low IDS (0.2766) con-      inquiries (85.37%) and market performance (89.83%) accuracies sug-
firms limited input sensitivity, limiting generalization. Single-agent       gest precise memory of individual data points and relative rankings.
models like Fin Agent and Quant Agent show moderate improve-     Most critically, the 90.23% accuracy on trend prediction questions,
ments via tool-augmented decision-making. Moreover, multi-agent     which require recalling temporal sequences, confirms that models
systems, such as Fin CON and Trading Agents, exhibit the lowest     memorize complete market movement patterns, enabling them to
leakage, benefiting from collaborative verification, but PC above      “predict” historical outcomes with high fidelity. This memorization
0.69 indicates persistent leakage. High PC and CI across methods       directly contributes to information leakage in financial forecast-
confirm reliance on memorized patterns over input-driven forecast-      ing, as models rely on recalled patterns rather than input-driven
ing, causing poor generalization.                                      reasoning, undermining generalization.

### 10. 2.3  Memorization Audits                             2.4  Before-and-after Fine-...

2.3  Memorization Audits                             2.4  Before-and-after Fine-tuning
Furthermore, we explore the memory capacity of LLMs.                  Finally, we assess the extent to which the acquired knowledge via
                                                                        training affects the performance.
2.3.1  Fin Lake-Bench. Fin Leak-Bench comprises 2,000 financial
question-answer pairs spanning January 2022 to June 2023, cov-       2.4.1  Experimental Setting. We fine-tune two base models, Qwen2.5-
ering four categories: price inquiries (e.g., NVIDIA’s closing price      7 B-Instruct [23] and Llama-3.1-8 B-Instruct [15], using Lo RA [6]
on a specific date), trend predictions (e.g., Apple’s stock movement      (rank=64, alpha=16, dropout=0.1, batch size=16, gradient accumula-
over two months), event impacts (e.g., Tesla’s stock response to ma-      tion steps=4, learning rate=2 e-4, warmup ratio=0.03) on the FNSPID
jor corporate announcements), and market performance (e.g., top-      dataset [2], comprising Dow Jones Industrial Average (DJIA) con-
performing NASDAQ-100 stocks on a given date). Representative       stituent stocks (30 stocks) from January 2020 to December 2022. The
examples and their corresponding scoring criteria are summarized       test set includes market data for these DJIA stocks from January to
in Table 2.                                                       June 2022.

### 11. Table 2: Examples of Fin Leak-Bench Questions and Scoring Criteria...

Table 2: Examples of Fin Leak-Bench Questions and Scoring Criteria

### 12. Category      Question                  Ground Truth Answer      Scoring Criteri...

Category      Question                  Ground Truth Answer      Scoring Criteria

### 13. Price Inquiry    What was NVIDIA’s closing price   $229.73                      ...

Price Inquiry    What was NVIDIA’s closing price   $229.73                      1 point if answer ∈[227.43, 232.03]; 0.5 point
                  on March 15, 2022?                                                                    if ∈[222.84, 236.62]; otherwise 0.
                What was Tesla’s opening price   $122.56                      1 point if answer ∈[121.33, 123.79]; 0.5 point
                  on January 12, 2023?                                                                  if ∈[118.88, 126.24]; otherwise 0.
    Event Impact    What was the impact of Musk’s   Negative impact: Tesla’s stock   Full point for responses mentioning >15%
                      Twitter acquisition on October 27,   fell over 15%, significantly un-  drop, with reasons such as Musk’s distrac-
                       2022, on Tesla’s stock price?      derperforming the market.      tion, forced share sale, or Twitter-related
                                                                                     negative sentiment.
               How did Silicon Valley Bank’s col-  Complex impact: Initial sector  1 point for answers covering both initial
                      lapse on March 10, 2023, affect  drop of 2%, followed by large-  decline and rebound; 0.5 point if only partial
                      tech stocks?                     cap tech rebound; startup pres-   effects are mentioned.
                                                        sures persisted.

### 14. Trend  Predic-  What was Amazon’s price trend  Steady decline with minor re-  0....

Trend  Predic-  What was Amazon’s price trend  Steady decline with minor re-  0.5 point for correct trend direction; +0.5 if
     tion            from April 1, 2022, over the fol-  bounds, cumulative drop of  magnitude is within ±5% error margin.
                    lowing 4 weeks?                  11.91%.
               How did Apple’s stock perform in  Consistent upward trend, cu-  0.5 point for correct trend direction; +0.5 if
                      the 2 months following February  mulative gain of 7.49%.        magnitude is within ±5% error margin.
                         15, 2023?

### 15. Market Perfor- How did the energy sector in the  Weak performance: sustained  1 ...

Market Perfor- How did the energy sector in the  Weak performance: sustained  1 point if answer ∈[-2.28%, -1.68%]; 0.5 point
    mance        S&P 500 perform on August 15,  decline with average drop of    if ∈[-2.48%, -1.48%]; otherwise 0.
                     2022?                             1.98%.
                 Which Dow Jones component  MSFT                       1 point for MSFT; 0.5 points for UNH or
                  had the largest decline on Janu-                           AXP.
                     ary 5, 2023?

### 16. where pmodel𝑡    and phist𝑡   are the model’s prediction and historical
        ...

where pmodel𝑡    and phist𝑡   are the model’s prediction and historical
                                                                     pattern vectors at time 𝑡, and 𝑇is the number of time points. The
                                                                     Generalization Change[8] assesses the relative change in accuracy
                                                        on unseen data:
                                                                                                Accpostunseen −Accpreunseen                                                                           ΔGen =               × 100%,           (10)
                                                                                                  Accpreunseen

### 17. where Accpreunseen and Accpostunseen are pre-fine-tuning and post-fine-
        ...

where Accpreunseen and Accpostunseen are pre-fine-tuning and post-fine-
                                                                  tuning accuracies on unseen data.
 Figure 3: Before-and-after fine-tuning on model behavior.
                                                                                  2.4.3  Results and Insights. Figure 3 illustrates the impact of fine-
                                                                   tuning on model behavior. Training markedly improves in-distribution
                                                                     accuracy, with Qwen2.5-7 B-Instruct increasing from 51.61% to
2.4.2  Baselines and metrics. We evaluate the impact of fine-tuning
                                                                  72.16% and Llama-3.1-8 B-Instruct from 54.73% to 76.52%. However,
using three metrics. The Bias Score[7] quantifies prediction bias
                                                                               this performance leap is accompanied by increased prediction bias,
toward frequently observed stocks:
                                                              with Bias Scores of 0.2895 and 0.3122, indicating that models no
                                                                  longer                                                                                    treat all stocks                                                                                                  impartially,                                                                                                    favoring                                                                                                               frequently                                                                                                                observed                            𝑆   𝑓train (𝑠) · 𝑝score (𝑠)                    1 ∑︁                          −1              Bias =                                                                     ones. Moreover,                                                                      High Memory                                                                                                Scores                                                                                                              of 0.7759                                                                                                 and 0.8113                                                                                                  show                    𝑆                                              𝑆,              (8)                           𝑠=1  Í𝑆𝑠′=1 𝑓train (𝑠′)                                                                       that models closely replicate historical patterns from the train-
where 𝑓train (𝑠) is the frequency of stock 𝑠in the training data,      ing data, suggesting they memorize future outcomes rather than
𝑝score (𝑠) is the model’s prediction score for stock 𝑠, and 𝑆is the      learning robust forecasting principles. Critically, generalization ca-
number of stocks. The Memory Score measures alignment with       pability declines substantially, with reductions of 21.53% and 18.06%
historical patterns:                                                     in accuracy on unseen data, confirming that fine-tuning leads to
                                                                          overfitting on leaked data. This behavior, driven by training data,
                        1 𝑇∑︁           Memory =                              cos  pmodel𝑡       , phist𝑡       ,               (9)      exacerbates information leakage risks as models reproduce memo-                    𝑇                                𝑡=1                                        rized patterns.

### 18. Profit Mirage: Revisiting Information Leakage in LLM-based Financial Agents     ...

Profit Mirage: Revisiting Information Leakage in LLM-based Financial Agents                                                                                                                                                                                   , ,

### 19. These findings highlight that fine-tuned models are not smarter                 ...

These findings highlight that fine-tuned models are not smarter                                                                      Market
but rather memorize historical patterns!                                    Information Σμ          Retrieval-Augmented Generation
3  Our Approach: A Counterfactual Evolution                                        Factor Extraction           Enhanced Structured
                                                                                                                                    Market Features St                                                                                                                                                                                                                                                                                                                                         ′   Framework
                                                                                                      Price Data Pt                Strategy Code Generator
3.1  Preliminaries
To escape the mirage issues, we propose Fact Fin, an external frame-                                                                                    Initial Strategy υ
work that mitigates leakage by using LLMs as strategy generators                                                                                     Market Factors Ft
rather than direct decision-makers, leveraging counterfactual rea-                               Monte Carlo Tree Search
soning and strategy evolution. We define the market state at time 𝑡
as 𝑆𝑡= {𝑃𝑡, 𝐹𝑡, 𝑁𝑡}, where 𝑃𝑡∈R𝑑is price data, 𝐹𝑡∈R𝑚denotes                                                            Optimized Strategy C‘
market factors, and 𝑁𝑡∈R𝑛represents factorized news. A trading             Market News Nt                  Strategy Performance
strategy𝐶: 𝑆𝑡→𝐴𝑡maps 𝑆𝑡to actions 𝐴𝑡∈{buy, sell, hold}. Thus,                                               Evaluation ϔ
LLM-based Agents are forced to learn why an outcome occurred,                                                                    Convergence                                                                             Counterfactual
because the outcome itself is no longer fixed.                                                                                                               Failure                                                                              Simulator                      Strategy Leakage
                                                                                                           Assessment ϔ푐Ϣ
3.2  Framework Overview                                                                                                                                                            Final Strategy C∗
                                                                                      Counterfactual As depicted in Figure 4, our Fact Fin is an external framework that
                                                                                     Scenario Dcf                   Trade Executionmitigates information leakage in LLM-based financial prediction
by using LLMs as strategy generators, coupled with counterfac-
tual reasoning and strategy evolution. Fact Fin integrates four core           Figure 4: The overall architecture of our Fact Fin.
components: Strategy Code Generator (SCG), Retrieval-Augmented
Generation (RAG), Monte Carlo Tree Search (MCTS), and Counter-
                                                         where 𝑆′𝑡includes processed prices, factors, and news. For newsfactual Simulator (CS). These components synergistically form a
                                                                           data, RAG extracts quantified features:pipeline that generates and optimizes trading strategies driven by
real-time market inputs, ensuring robust predictions for LLM-based                         𝑁′𝑡= 𝜓(𝑁𝑡,𝑡),                        (13)
financial forecasting methods.
                                                         where 𝜓converts news at time 𝑡into sentiment scores and topic
                                                                          distributions in R𝑛. This approach "factorizes" unstructured infor-
3.3  Strategy Code Generator
                                                                  mation, making it more suitable for systematic trading strategies
The Strategy Code Generator (SCG) transforms financial prediction      while reducing the risk of the model falling back on memorized
into a code generation task, leveraging LLMs to produce executable      outcomes.
trading strategy code based on market state 𝑆𝑡. SCG generates a
strategy 𝐶: 𝑆𝑡→𝐴𝑡, defined as:                                                         3.5 MCTS for Strategy Evolution
             𝐶= SCG(𝑆𝑡, 𝑃),                       (11)     The Monte Carlo Tree Search (MCTS)[21] component optimizes
                                                                          strategies generated by the SCG, ensuring adaptation to market
where 𝑃is a prompt template. By focusing on generating systematic                                                                            factors in 𝑆𝑡while avoiding reliance on memorized patterns. MCTS
strategies rather than specific predictions, we reduce the model’s                                                                produces an optimized strategy 𝐶∗: 𝑆𝑡→𝐴𝑡, defined as:
reliance on memorized historical price movements. A simplified
                                                           𝐶∗= MCTS(𝐶, 𝐷),                      (14) prompt is as follows1:
 [Prompt Template]                                          where 𝐶is the initial strategy and 𝐷represents evaluation datasets.
 Given market state with {Prices}, {Factors}, and {News}, generate     MCTS explores the strategy space by selecting nodes using the
  executable trading strategy code, e.g., {Examples}, using only       Upper Confidence Bound:
  provided inputs.                                                                                   √︄                                                                                     𝑤(𝑠)       ln 𝑁(𝑠)
                                                                         UCB(𝑠) =    + 𝑐                   ,               (15)
                                                                                                 𝑛(𝑠)        𝑛(𝑠)
3.4 RAG for Market Factors
                                                          where 𝑤(𝑠) is the cumulative reward, 𝑛(𝑠) is the visit count, 𝑁(𝑠)The Retrieval-Augmented Generation (RAG)[11] component en-
                                                                                         is the parent’s visit count, and 𝑐is an exploration parameter. Newhances the Strategy Code Generator by retrieving and processing
                                                                       strategy variants are generated via:real-time market factors within 𝑆𝑡, ensuring strategies rely on cur-
rent inputs rather than memorized data. RAG transforms the market                      𝐶new = SCG(𝐹𝑡, 𝑃modify,𝐶),                 (16)
state 𝑆𝑡= {𝑃𝑡, 𝐹𝑡, 𝑁𝑡} into structured features 𝑆′𝑡, defined as:
                                                           and evaluated as:
                           𝑆′𝑡= RAG(𝑆𝑡),                        (12)                                                  𝑅= Evaluate (𝐶new, 𝐷).                   (17)
1 The templates will change based on different market conditions, fully disclosed in     Node statistics are updated as 𝑤(𝑠) = 𝑤(𝑠) + 𝑅and 𝑛(𝑠) = 𝑛(𝑠) + 1.
the Appendix.                                              By iteratively refining strategies based on real-time market inputs,

### 20. Algorithm 1: Workflow of Our Fact Fin                                  in Sectio...

Algorithm 1: Workflow of Our Fact Fin                                  in Section 2.2, which measure stability, sensitivity, and data depen-
Require: Market state 𝑆𝑡= {𝑃𝑡, 𝐹𝑡, 𝑁𝑡}, dataset 𝐷                   dence during counterfactual changes.
   1: 𝑆′𝑡←RAG(𝑆𝑡)                         ⊲Factor extraction        4.1.3   Baselines. 1) Financial models: Fin GPT [30], Fin-LLa MA [26],
   2: 𝐶←SCG(𝑆′𝑡, 𝑃)                ⊲Generate initial strategy      Invest LM [31]; 2) Single-agent systems: Fin Mem [33], Fin Agent [35],
   3: while not converged do                                   Quant Agent [24]; 3) Multi-agent systems: Trading Agents [29],
   4:   𝐶←MCTS(𝐶, 𝐷)                 ⊲Optimize strategy     Hedge Agents [13], Fin Robot [36]). All baselines are implemented
                                                                  per their original specifications2.   5:     𝐷cf ←Perturb (𝐷,𝛿)              ⊲Counterfactual data
   6:     Evaluate PC, CI, IDS on 𝜙(𝐶, 𝐷) and 𝜙(𝐶, 𝐷cf)                      4.1.4  Implementation Details. The action space 𝐴𝑡includes three
                      (𝛼· PC(𝐶) + 𝛽· CI(𝐶) )                   discrete trading actions for individual stocks: buy, sell, and hold,
   7:    Update 𝐶←arg min                                       executed with standard transaction costs and realistic slippage                       −𝛾· IDS(𝐶)
                                                                   models. Fact Fin and all LLM-based agents in baselines use GPT-4 o
   8: end while
                                                                       as the backbone with a temperature of 0.7 for balanced consistency
Ensure: Optimized strategy 𝐶∗                                                           and creativity. RAG adopts the text-embedding-3-large model[16]
                                                               with top-k=5, while MCTS is configured with a search depth of 10
                                                           and UCB exploration parameter 𝑐= 0.5.
MCTS enhances the robustness of 𝐶∗and mitigates dependence on
historical knowledge.                                        4.2  Overall Performance
                                                                   Table 3 evaluates Fact Fin against nine baseline methods across six3.6  Counterfactual Simulator
                                                                            assets from July 1, 2024, to June 30, 2025, after the release of GPT-4 o.
The Counterfactual Simulator (CS) is a pivotal component of Fact Fin,                                                           The following observations are made: 1) Financial models exhibit
mitigating information leakage by testing strategies in counter-                                                                    highly unstable performance across markets; for instance, Fin GPT
factual market environments. CS perturbs market data 𝐷within                                                                    achieves an excess return of 17.84% for AAPL but -42.75% for Ten-
𝑆𝑡= {𝑃𝑡, 𝐹𝑡, 𝑁𝑡} to generate alternative scenarios:                                                                           cent. This inconsistency arises from training data biases favoring
                      𝐷cf = Perturb (𝐷,𝛿),                     (18)      familiar assets, resulting in poor generalization. 2) Single-agent
                                                                 systems demonstrate improved returns and risk management with
where 𝐷cf is the counterfactual dataset and 𝛿controls perturbation                                                        more stable performance across assets; Fin Agent achieves a TR of
magnitude. Perturbations, such as modifying 𝑃𝑡with noise 𝜖𝑡∼
                                                                27.79% and SR of 0.99 for AAPL, the best among nine baselines.
N (0, 𝜎2) or adjusting 𝐹𝑡and 𝑁𝑡, preserve statistical relationships.
                                                                This improvement is due to tool invocation for market analysis,
Strategy performance is assessed as:
                                                                 outperforming prompt-dependent Fin-LLM. 3) Multi-agent systems
          𝑅= 𝜙(𝐶, 𝐷),   𝑅cf = 𝜙(𝐶, 𝐷cf),               (19)     enhance performance through collaborative task decomposition;
                                                              Trading Agents achieves a TR of 59.28%, SR of 1.17, and MDD ofwhere 𝜙evaluates strategy 𝐶on datasets 𝐷and 𝐷cf. Information
                                                                  25.99% for NVDA, the best among nine baselines, but Fin Robot suf-
leakage is quantified using Prediction Consistency (PC), Confidence
                                                                               fers from a high MDD of-49.99% for TSLA. This is because dynamic
Invariance (CI), and Input Dependency Score (IDS). Strategies with
                                                              weight allocation adapts to market changes, yet over-analysis ofhigh PC and CI but low IDS rely on memorized patterns. CS opti-
                                                                 non-informative data, such as macro news. 4) Our Fact Fin con-mizes strategies to minimize leakage:
                                                                           sistently outperforms all baselines across all assets, achieving an
     𝐶∗= arg min𝐶{𝛼· PC(𝐶) + 𝛽· CI(𝐶) −𝛾· IDS(𝐶)},     (20)      average improvement of 31.91% in TR, 22.74% in SR, and 9.23% in
                                                MDD, with robust cumulative returns, as shown in Figure 5. This
where 𝛼, 𝛽, and 𝛾are weights, ensuring 𝐶∗is driven by real-time
                                                                         superiority comes from a comprehensive market analysis using fac-
inputs and robust against leakage.
                                                                            torization and data-driven strategies, combined with counterfactual
                                                                  reasoning and strategy evolution, thus outperforming baselines.
4  Experimental Results
4.1  Seetings                                           4.3  Information Leakage Mitigation
4.1.1   Dataset. We evaluate our Fact Fin framework via six financial     To evaluate the ability of Fact Fin and baselines to mitigate informa-
assets: U.S. equities (AAPL, NVDA, TSLA), Chinese equity (BYD,      tion leakage, we conducted experiments prior to the LLM training
002594.SZ), Hong Kong equity (Tencent, 0700.HK), and cryptocur-       cutoff, constructing 50 counterfactual scenarios for each asset. Ta-
rency (Bitcoin). Sourced from Yahoo Finance and Alpaca News API,      ble 4 presents the information leakage metrics across all models
the dataset spans January 1, 2020, to June 30, 2025, including price     and assets.
data with volume and turnover, news, and counterfactual scenarios       The following observations can be made: 1) Financial language
to assess information leakage.                                    models exhibit severe information leakage, with Invest LM showing
                                                               high average PC 0.8789 and average CI 0.8964 and low average
4.1.2  Evaluation Metrics. The profitability and risk are evaluated
                                                            IDS 0.1912 across six assets. This is due to fine-tuning on financial
using Total Return (TR), Sharpe Ratio (SR), and downside risk using
                                                                             datasets, which leads to memorization of specific events rather than
Maximum Drawdown (MDD), defined in Section 2.1. Information
                                                                     learning generalizable patterns, resulting in poor performance in
leakage is evaluated through Prediction Consistency (PC), Confi-
dence Invariance (CI), and Input Dependency Score (IDS), defined       2 More details are provided in Appendix.

### 21. Profit Mirage: Revisiting Information Leakage in LLM-based Financial Agents     ...

Profit Mirage: Revisiting Information Leakage in LLM-based Financial Agents                                                                                                                                                                                   , ,

### 22. Table 3: Performance comparison of Fact Fin and baselines across six assets (Jul...

Table 3: Performance comparison of Fact Fin and baselines across six assets (July 2024 to June 2025). Bold represents optimal
performance, while Bold represents suboptimal.

### 23. AAPL          NVDA             TSLA            BYD              Tencent         ...

AAPL          NVDA             TSLA            BYD              Tencent               Bitcoin
      Categories Models
                          TR ↑   SR ↑ MDD ↓  TR ↑   SR ↑ MDD ↓  TR ↑   SR ↑ MDD ↓  TR ↑   SR ↑ MDD ↓  TR ↑   SR ↑ MDD ↓  TR ↑   SR ↑ MDD ↓

### 24. Fin GPT        14.48%   0.59   29.05%  28.01%   0.73   32.62%  42.58%    0.85   ...

Fin GPT        14.48%   0.59   29.05%  28.01%   0.73   32.62%  42.58%    0.85   52.37%  10.21%   0.46   21.31%  -5.93%   -0.05  23.28%  42.81%    0.85   24.61%
       Fin-LLM   Fin-LLa MA    -20.05%  -0.63  37.76%  16.27%   0.55   32.65%  63.15%    1.06   53.71%  -9.61%   -0.12  22.23%  23.64%   0.82   23.47%  21.04%    0.53   31.19%
                  Invest LM       -9.25%   -0.21  32.86%  38.18%   0.89   28.40%  50.15%    0.96  36.88%  -4.11%   0.03   23.17%  -4.85%   -0.06  19.92%  54.88%    1.01   26.66%

### 25. Fin Mem        -5.68%   -0.03  32.11%  35.88%   0.84   33.42%  52.32%    0.95   ...

Fin Mem        -5.68%   -0.03  32.11%  35.88%   0.84   33.42%  52.32%    0.95   44.63%  26.83%   0.88   19.55%  27.77%   0.97   17.40%  72.68%    1.26  19.68%
      Single-Agent Fin Agent      27.79%  0.99  21.52%  54.86%   1.09   27.78%  79.01%    1.24   -9.36%  32.91%   1.04   20.17%  44.18%   1.29   23.44%  94.63%    1.38   24.53%
                 Quant Agent    6.60%    0.36   29.39%  49.83%   1.03   30.12%  74.26%    1.21   42.35%  31.74%   1.03  18.01% 31.15%   1.08   19.95%  90.56%    1.35   32.98%

### 26. Trading Agents 12.89%   0.55   27.26% 59.28%  1.17  25.99% 106.01%   1.47   36.9...

Trading Agents 12.89%   0.55   27.26% 59.28%  1.17  25.99% 106.01%   1.47   36.90%  43.33%   1.26   19.53%  42.77%   1.38   24.18%  88.39%    1.37   32.96%
      Multi-Agents Hedge Agents   16.06%   0.68  12.12% 54.09%   1.10   28.39% 115.09%  1.48  48.79% 57.96%  1.49  18.72% 66.31%  1.85  17.67% 134.36%  1.71  22.12%
                  Fin Robot       21.76%   0.81   21.54%  43.98%   0.72   36.65%  96.32%    1.34   49.99%  38.49%   1.09   20.53%  57.54%   1.81  14.56% 109.18%   1.51   21.67%

### 27. Ours    Fact Fin       36.70%  1.22  11.57% 71.34%  1.29  24.25% 165.01%  1.83  ...

Ours    Fact Fin       36.70%  1.22  11.57% 71.34%  1.29  24.25% 165.01%  1.83  31.54% 84.24%  2.09  16.01% 81.37%  2.31  14.14% 171.46%  2.03  16.59%

### 28. Improvement (%)      32.06% 23.23%  4.54%  20.34% 10.26%  6.69%   43.37%  19.13%...

Improvement (%)      32.06% 23.23%  4.54%  20.34% 10.26%  6.69%   43.37%  19.13% 14.48%  45.34% 40.27% 11.10%  22.71% 24.86%  2.88%   27.61%  18.71% 15.70%

### 29. Figure 5: Performance comparison over time between Fact Fin and other benchmarks...

Figure 5: Performance comparison over time between Fact Fin and other benchmarks across all assets.

### 30. counterfactual scenarios. 2) Single-agent systems show reduced     and high info...

counterfactual scenarios. 2) Single-agent systems show reduced     and high information leakage. Although SCG outperforms direct
information leakage, as demonstrated by Quant Agent achieving    LLM decision-making, code generation alone is insufficient to ef-
optimal PC 0.6775 and IDS 0.3868 among nine baselines on BYD,      fectively mitigate leakage. 2)Adding RAG significantly improves
leveraging tool invocation to mitigate memory reliance, though    TR and SR for both assets and reduces leakage, as structured mar-
long contexts lead to moderate leakage. 3) Multi-agent systems      ket factor extraction enhances information utilization. 3) Introduc-
further reduce information leakage, benefiting from collaborative      ing MCTS further increases TR from 13.36% to 28.12% for AAPL
task decomposition and information cross-validation, but residual     and from 104.38% to 130.93% for TSLA, while reducing MDD from
leakage persists. 4) Our Fact Fin, through Monte Carlo Tree Search      23.65% to 16.20% for AAPL and from 37.45% to 33.52% for TSLA, sub-
and a Counterfactual Simulator, uses the LLM as a strategy genera-       stantially contribute to improved returns and risk reduction, with
tor rather than a direct decision-maker, substantially reducing PC      leakage moderately alleviated, though still at a high level. 4)The
and CI while increasing IDS, achieving the lowest leakage across        full model, including CS, achieves optimal performance across all
all assets. These results show that Fact Fin reliably predicts across       metrics, with information leakage most effectively mitigated. The
various market conditions                                  CS component is crucial for detecting and correcting leaks, allowing
                                                                       Fact Fin to avoid memorized patterns, ensuring strong performance
                                                           and minimal data leakage.
4.4  Ablation Studies
In Table 5, we study the effectiveness of the Strategy Code Genera-
tor (SCG), Retrieval-Augmented Generation (RAG), Monte Carlo     4.5 LLM Backbone Comparison
Tree Search (MCTS), and Counterfactual Simulator (CS) in Fact Fin.     To assess Fact Fin’s adaptability, we tested six state-of-the-art LLMs
1)When using only SCG, Fact Fin exhibits low financial performance      as backbones, Table 6 presents the performance. Closed-source


---

## Raw Markitdown Extraction (full text)

Profit Mirage: Revisiting Information Leakage in LLM-based
Financial Agents
XiangyuLi YawenZeng XiaofenXing
65603605lxy@gmail.com yawenzeng11@gmail.com xfxing@scut.edu.cn
SouthChinaUniversityofTechnology ByteDance SouthChinaUniversityofTechnology
Guangzhou,China Beijing,China Guangzhou,China
JinXu XiangminXu∗
jinxu@scut.edu.cn xmxu@scut.edu.cn
SouthChinaUniversityofTechnology SouthChinaUniversityofTechnology
PazhouLab Guangzhou,China
Guangzhou,China
Abstract Whydoesthismiragearise?Weshowthattheculpritisnot
LLM-basedfinancialagentshaveattractedwidespreadexcitement flawedriskmanagementornoisymarketdata,butinformation
fortheirabilitytotradelikehumanexperts.However,mostsystems leakagebakedintotheLLMitself.Modernfoundationmod-
exhibita“profitmirage”:dazzlingback-testedreturnsevaporate elsingestweb-scalecorporathatcontainpost-hocexplanations
oncethemodel’sknowledgewindowends,becauseoftheinherent of past price movements—“NVIDIA surged 190% in 2023 on AI
informationleakageinLLMs.Inthispaper,wesystematicallyquan- boom”—alongsidecontemporaneousnews.Whenthesesnippets
tifythisleakageissueacrossfourdimensionsandreleaseFinLake- appearinthetrainingset,themodeldoesnotlearnwhyprices
Bench,aleakage-robustevaluationbenchmark.Furthermore,to move;itlearnsthattheyalreadymoved,andsimplyrecitesthean-
mitigatethisissue,weintroduceFactFin,aframeworkthatap- swerduringback-testing.Infact,this“pre-trainingcontamination”
pliescounterfactualperturbationstocompelLLM-basedagentsto islethalinfinancearea.
learncausaldriversinsteadofmemorizedoutcomes.FactFininte- Furthermore, we formalize this concern and demonstrate its
gratesfourcorecomponents:StrategyCodeGenerator,Retrieval- empiricalprevalencethroughfourexperiments:
AugmentedGeneration,MonteCarloTreeSearch,andCounter-
factualSimulator.Extensiveexperimentsshowthatourmethod
surpassesallbaselinesinout-of-samplegeneralization,delivering
• 1)Back-testingversusgeneralization(Section2.1).Byrolling
superiorrisk-adjustedperformance.
thecalendarforwardwerevealthatalmosteverypublishedLLM-
basedagentfailstobeatarandombaselineonceitsknowledge
Keywords
cutoffispassed.
QuantitativeFinance,LargeLanguageModel,LLM-basedAgent • 2)Counterfactualevaluation(Section2.2).Wefeedmodels
carefullycraftedcounterfactualpromptsbyperturbingkeymar-
1 Introduction ketinputs.Resultsshowhighpredictionconsistency,withthe
worstmodelmaintaining82.13%ofpredictionsunchangedde-
Theadventoflargelanguagemodels(LLMs)hasprecipitateda
spitesignificantinputalterations.Thisprovesthatagentsare
paradigmaticshiftinquantitativefinance.Representativesystems
primarily reciting memorized patterns rather than analyzing
includeFinGPT[30],FinMem[33],FinReport[12]andthemulti-
tradableinformation.
agentarchitectureHedge-Agents[13],allreportdouble-ortriple-
• 3)Memorizationaudits(Section2.3).Wereleasealeakage-
digitannualizedreturnsinback-teststhatspantheU.S.,HongKong
robustevaluation,FinLake-Bench,whichconstructs2,000histor-
andA-sharemarkets.
icalQApairs,suchas“didthemarketriseondateT?”.GPT-4o
However,whenwemovetheseLLM-basedagentsonestepbe-
anditspeersanswercorrectlyover85%ofthetime—farabove
yondtheirtrainingtimecutoff,thestoryfallsapart,thatis,a“profit
chance—confirmingthatthefactshavebeenmemorized.
mirage”.Figure1showstheequitycurvesofseveralpopularagents
• 4)Before-and-aftertargetedfine-tuning(Section2.4).When
re-evaluatedonnewdataafterthereleaseoftheirunderlyingLLMs
wedeliberatelyinjectfinancialdataintomodeltraining,in-distribution
(e.g.,verifyinganagentreleasedin2023inthe2024financialmar-
accuracyexceeded70%,showingasignificantimprovement,while
ket.).Thebest-performingLLM-basedagentalsodrop50%!The
generalizationcapabilityonunseendatadropsdramatically.The
dazzlingreturncollapsestoastatisticalzerooncethemodelisno
gainispurememorizationofhistoricalpatterns,notimproved
longerallowedtopeekatthefutureitwastrainedon.Thisphe-
tradingskill.
nomenonisalsothereasonwhywecallitthe“profitmirage”:
returnsthatevaporatethemomentthemodelisforcedtotradein
genuinelyunknownterritory.
Takentogether,theseresultsshowthatLLM-basedfinancialagents
∗Correspondingauthor. arenottrading;theyareregurgitatinghistory.
5202
tcO
9
]IA.sc[
1v02970.0152:viXra

,, Lietal.
Inthispaper,toescapethismirageissues,weproposeacounter-
factualframework,namelyFactFin.Specifically,ourFactFininte-
gratesfourcorecomponents:StrategyCodeGenerator,Retrieval-
AugmentedGeneration,MonteCarloTreeSearch,andCounter-
factualSimulator.Thesecomponentsworktogethertocreateand
refinetradingstrategiesusingreal-timemarketdata.Inthisway,
LLM-basedAgentsareforcedtolearnwhyanoutcomeoccurred,
becausetheoutcomeitselfisnolongerfixed.Finally,ourmethod
deliverout-of-sampleSharperatios1.4×higherthanthebestbase-
lines.
Ourcontributionsaresummarizedasfollows:
Figure 1: Backtesting vs. Generalization. All LLM-based
• Weprovidethefirstsystematicevidencefromfourdimensions agentsshowasignificantdropinthegeneralizationsetting.
thattheinformationleakageinLLM-basedagents.Moreover,we
conductanextensiveempiricalstudyofleadingopen-and
closed-sourcemodels,providingaclearbaselineandrevealing whereE[𝑅]istheexpectedreturn,𝑅
𝑓
istherisk-freerate,and𝜎
𝑅
theircurrentlimitationsin. isthereturnvolatility.Toquantifyperformancedegradation,we
• WeintroduceFinLake-Bench,thefirstleakage-robustevalua- computethedecayrateas:
tionsuitethatincludesmemorizationprobesandcounterfactual
𝑋 −𝑋
labels. DecayRate= pre post ×100%, (3)
𝑋
• Wedevelopacounterfactualframework,FactFin,whichinte- pre
gratesfourcorecomponents. where𝑋 representseitherTRorSRforthepre-andpost-periods
oftraining.
2 IsInformationLeakageEverywhere?
2.1.3 ResultsandInsights. Figure1presentstheperformancecom-
Inthissection,weformalizetheleakageconcernofLLM-based
parison,revealingthatallmethodsshowasignificantdropinthe
agentsanddemonstrateitsempiricalprevalencethroughfourex-
periodafterthebasemodel(i.e.,GPT-4o)isreleased.TheSharpe
periments.
Ratiodecayrangesfrom51.48%(QuantAgent)to62.23%(FinCON),
whiletheTotalReturndecayrangesfrom50.18%(TradingAgents)
2.1 Backtestingvs.Generalization
to71.85%(FinMem).FinMemexhibitsthemostseveredegradation,
ToinvestigatetheimpactofinformationleakageinLLM-based
indicatingastrongrelianceonmemorizedhistoricalpatterns.FinA-
financialagents,weconductatemporalsegmentationexperiment
gentandQuantAgent,showslightlybetterresilience,likelydueto
tocompareperformancebeforeandafterthetrainingtimecut-
theiruseofexternaltoolstoaugmentdecision-making.TradingA-
offoftheunderlyingLLM.Theexperimentaimstodemonstrate
gents,whichleveragecollaborativemechanisms,furthermitigate
thatmodelsexploithistoricalpatternsintrainingdata,leadingto
leakagebutstillsuffera55.68%Sharpedecay.Thissignificantper-
inflatedbacktestingresultsbutpoorforward-testingperformance.
formancedrop,despitecomparablemarketconditions,suggests
thatLLM-basedagentsarenotgenuinelyforecastingbutrather
2.1.1 ExperimentalSetting. WeselecttheNASDAQ-100indexcon-
recognizingpatternsfromtheirtrainingdata.
stituentstocksastheevaluationpoolanddefinetwoperiodswith
comparablemarketconditionstoisolatetheeffectofinformation
2.2 CounterfactualEvaluation
leakage:ahistoricaltradeperiod(Q2-Q32021,backtestingsetting)
andalatesttradeperiod(Q3-Q42024,generalizationsetting).No- Further,weutilizeacounterfactualevaluationframeworktoassess
tably,themarketreturnsinthetwoperiodsaresimilar(market howrelianceonmemorizedpatternsleadstopoorgeneralization.
returnis+13.79%and+13.35%),tominimizetheimpactofthemar-
2.2.1 ExperimentalSetting. Counterfactualmarketenvironments[3]
ketitself.
areconstructedbyperturbinginputs:modifyingorremovingkey
2.1.2 Baselinesandmetrics. Thereafter,weevaluatefivestate-of- events(e.g.,earningsreports,regulatorychanges),replacingprice
the-artLLM-basedmethods—FinMem[33],FinAgent[35],Quan- sequenceswithhistoricalaveragesorrandomwalks,andaltering
tAgent[24],FinCON[34],andTradingAgents[29]—usingGPT-4o technicalindicators(e.g.,RSI,MACD,KDJ)andfundamentalfactors
[17](trainingcutoff:October2023)asthebackbone.Performance (e.g.,PE,PB,ROE).Theseperturbationstestwhethermodelsadapt
ismeasuredusingTotalReturn(TR),definedas: toinputchangesorrelyonmemorizedpatterns.Weevaluateten
stocks(AAPL,TSLA,NVDA,MSFT,GOOGL,AMZN,META,NFLX,
𝑃 −𝑃
TR= final initial ×100%, (1) AMD,CRM)fromJanuary2022toJune2023,selecting30keytime
𝑃
initial pointsperstockbasedonsignificantmarketevents.
where𝑃 and𝑃 aretheinitialandfinalportfoliovalues,and
initial final 2.2.2 Baselinesandmetrics. WeevaluatethesamefiveLLM-based
SharpeRatio(SR),definedas:
methodsasinSection2.1,usingGPT-4oasthebackbone.Informa-
E[𝑅]−𝑅 𝑓 tionleakageisquantifiedviathreemetrics.PredictionConsistency
SR= , (2)
𝜎 (PC)[4]measurestheproportionofunchangedpredictionsafter
𝑅

ProfitMirage:RevisitingInformationLeakageinLLM-basedFinancialAgents ,,
Table1:Counterfactualevaluationofagents.
|     | Methods       | PC↓           | CI↓ IDS↑ |     |     |     |     |     |     |
| --- | ------------- | ------------- | -------- | --- | --- | --- | --- | --- | --- |
|     | FinMem        | 0.8213 0.8743 | 0.2766   |     |     |     |     |     |     |
|     | FinAgent      | 0.7245 0.7781 | 0.3598   |     |     |     |     |     |     |
|     | QuantAgent    | 0.7789 0.8362 | 0.2941   |     |     |     |     |     |     |
|     | FinCON        | 0.7136 0.7522 | 0.3612   |     |     |     |     |     |     |
|     | TradingAgents | 0.6903 0.7016 | 0.3837   |     |     |     |     |     |     |
perturbation:
𝑁
|     |     | 1 ∑︁ (cid:104) | (cid:105) |     |     |     |     |     |     |
| --- | --- | -------------- | --------- | --- | --- | --- | --- | --- | --- |
PC= I 𝑦ˆ orig=𝑦ˆ𝑖 cf , Figure2:MemorizationauditsofLLMsonFinLeak-Bench
|     |     | 𝑁 𝑖 |     | (4) |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
𝑖=1
where𝑦ˆorigand𝑦ˆcfarepredictionsinoriginalandcounterfactual
| 𝑖   | 𝑖   |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
scenarios,I[·]istheindicatorfunction,and𝑁 isthesamplesize. 2.3.2 Baselinesandmetrics. WeevaluatethreeleadingLLMs—GPT-
4o,Claude-Sonnet-3.7[1],andGrok-3[28]—usingastrictaccuracy
HigherPCindicatesgreaterrelianceonmemorizedpatterns.Confi-
metric:
denceInvariance(CI)[20]assessespredictionconfidencestability:
𝑁
|     |       | 𝑀                  |              |     |               |                                            | 1 ∑︁     |        |     |
| --- | ----- | ------------------ | ------------ | --- | ------------- | ------------------------------------------ | -------- | ------ | --- |
|     |       | 1 ∑︁ (cid:12)      | (cid:12)     |     |               | Accuracy=                                  | I[𝑎ˆ𝑖 =𝑎 | 𝑖]·𝑤 , | (7) |
|     | CI=1− | (cid:12) 𝑠o rig−𝑠c | f (cid:12) , |     |               |                                            | 𝑁        | 𝑖      |     |
|     |       | 𝑀 (cid:12) 𝑗       | 𝑗 (cid:12)   | (5) |               |                                            | 𝑖=1      |        |     |
|     |       | 𝑗=1                |              |     | where𝑎ˆ𝑖 and𝑎 | arethepredictedandtrueanswersforquestion𝑖, |          |        |     |
𝑖
where𝑠o rigand𝑠c
𝑗 𝑗 fareconfidencescoresforconsistentpredictions, I[·]istheindicatorfunction,𝑁 isthenumberofquestions,and𝑤 𝑖
and𝑀 is the number of consistent samples. CI near 1 suggests isaweightreflectinganswerprecision.Priceinquiriesscore1point
insensitivitytoinputchanges.InputDependencyScore(IDS)[14] foranswerswithin±1%errorand0.5for±3%.Trendpredictions
measuresinputsensitivityviaKLdivergence:
earn0.5pointsforcorrectdirectionand0.5foraccuratemagnitude.
|     |     | 𝑁             |          |     | Eventimpactquestionsarescoredbasedonqualitativealignment |     |     |     |     |
| --- | --- | ------------- | -------- | --- | -------------------------------------------------------- | --- | --- | --- | --- |
|     |     | 1 ∑︁ (cid:16) | (cid:17) |     |                                                          |     |     |     |     |
IDS= 𝐷 𝑃 orig∥𝑃 cf , (6) withactualmarketreactions.Marketperformancequestionsscore
|     |     | 𝑁 KL 𝑖 | 𝑖   |     |     |     |     |     |     |
| --- | --- | ------ | --- | --- | --- | --- | --- | --- | --- |
𝑖=1 1 point for identifying stocks in the top/bottom 3% and 0.5 for
top/bottom5%.
where𝑃origand𝑃cfarepredictionprobabilitydistributions.Higher
| 𝑖   | 𝑖   |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
IDSindicateslessleakage.
|                           |     |                                  |     |     | 2.3.3 ResultsandInsights. |                | Figure2presentstheaverageaccuracy |               |     |
| ------------------------- | --- | -------------------------------- | --- | --- | ------------------------- | -------------- | --------------------------------- | ------------- | --- |
|                           |     |                                  |     |     | of LLMson                 | FinLeak-Bench, | revealing significant             | memorization. |     |
| 2.2.3 ResultsandInsights. |     | Table1showsaverageleakagemetrics |     |     |                           |                |                                   |               |     |
Allmodelsexhibithighaverageaccuracyacrossquestiontypes,
acrossallstocks,highlightinggeneralizationissuesduetoinfor-
mationleakage.FinMemexhibitsthehighestleakage,withaPCof rangingfrom85.37%forpriceinquiriesto92.94%foreventimpacts.
0.8213andCIof0.8743,indicatingover82%ofpredictionsremain Eventimpactquestionsshowthehighestaccuracy,indicatingnear-
unchangeddespiteperturbations,withstableconfidence,reflecting encyclopedicrecallofmarketreactionstospecificevents.Price
inquiries(85.37%)andmarketperformance(89.83%)accuraciessug-
heavyrelianceonmemorizedpatterns.ItslowIDS(0.2766)con-
gestprecisememoryofindividualdatapointsandrelativerankings.
firmslimitedinputsensitivity,limitinggeneralization.Single-agent
Mostcritically,the90.23%accuracyontrendpredictionquestions,
modelslikeFinAgentandQuantAgentshowmoderateimprove-
mentsviatool-augmenteddecision-making.Moreover,multi-agent whichrequirerecallingtemporalsequences,confirmsthatmodels
systems,suchasFinCONandTradingAgents,exhibitthelowest memorizecompletemarketmovementpatterns,enablingthemto
“predict”historicaloutcomeswithhighfidelity.Thismemorization
leakage,benefitingfromcollaborativeverification,butPCabove
directlycontributestoinformationleakageinfinancialforecast-
0.69indicatespersistentleakage.HighPCandCIacrossmethods
ing,asmodelsrelyonrecalledpatternsratherthaninput-driven
confirmrelianceonmemorizedpatternsoverinput-drivenforecast-
ing,causingpoorgeneralization. reasoning,undermininggeneralization.
| 2.3 MemorizationAudits |     |     |     |     | 2.4 Before-and-afterFine-tuning |     |     |     |     |
| ---------------------- | --- | --- | --- | --- | ------------------------------- | --- | --- | --- | --- |
Furthermore,weexplorethememorycapacityofLLMs. Finally,weassesstheextenttowhichtheacquiredknowledgevia
trainingaffectstheperformance.
| 2.3.1 FinLake-Bench. | FinLeak-Bench |     | comprises 2,000 | financial |     |     |     |     |     |
| -------------------- | ------------- | --- | --------------- | --------- | --- | --- | --- | --- | --- |
question-answerpairsspanningJanuary2022toJune2023,cov- 2.4.1 ExperimentalSetting. Wefine-tunetwobasemodels,Qwen2.5-
eringfourcategories:priceinquiries(e.g.,NVIDIA’sclosingprice 7B-Instruct[23]andLlama-3.1-8B-Instruct[15],usingLoRA[6]
onaspecificdate),trendpredictions(e.g.,Apple’sstockmovement (rank=64,alpha=16,dropout=0.1,batchsize=16,gradientaccumula-
overtwomonths),eventimpacts(e.g.,Tesla’sstockresponsetoma- tionsteps=4,learningrate=2e-4,warmupratio=0.03)ontheFNSPID
jorcorporateannouncements),andmarketperformance(e.g.,top- dataset[2],comprisingDowJonesIndustrialAverage(DJIA)con-
performingNASDAQ-100stocksonagivendate).Representative stituentstocks(30stocks)fromJanuary2020toDecember2022.The
examplesandtheircorrespondingscoringcriteriaaresummarized testsetincludesmarketdatafortheseDJIAstocksfromJanuaryto
| inTable2. |     |     |     |     | June2022. |     |     |     |     |
| --------- | --- | --- | --- | --- | --------- | --- | --- | --- | --- |

| ,,  |     |     |     |     |     |     |     | Lietal. |
| --- | --- | --- | --- | --- | --- | --- | --- | ------- |
Table2:ExamplesofFinLeak-BenchQuestionsandScoringCriteria
| Category | Question |     |     | GroundTruthAnswer |     | ScoringCriteria |     |     |
| -------- | -------- | --- | --- | ----------------- | --- | --------------- | --- | --- |
PriceInquiry WhatwasNVIDIA’sclosingprice $229.73 1pointifanswer∈[227.43,232.03];0.5point
|     | onMarch15,2022? |     |     |     |     | if∈[222.84,236.62];otherwise0. |     |     |
| --- | --------------- | --- | --- | --- | --- | ------------------------------ | --- | --- |
1pointifanswer∈[121.33,123.79];0.5point
|     | WhatwasTesla’sopeningprice |     |     | $122.56 |     |                                |     |     |
| --- | -------------------------- | --- | --- | ------- | --- | ------------------------------ | --- | --- |
|     | onJanuary12,2023?          |     |     |         |     | if∈[118.88,126.24];otherwise0. |     |     |
Fullpointforresponsesmentioning>15%
| EventImpact | WhatwastheimpactofMusk’s |     |     | Negativeimpact:Tesla’sstock |     |     |     |     |
| ----------- | ------------------------ | --- | --- | --------------------------- | --- | --- | --- | --- |
TwitteracquisitiononOctober27, fellover15%,significantlyun- drop,withreasonssuchasMusk’sdistrac-
2022,onTesla’sstockprice? derperformingthemarket. tion,forcedsharesale,orTwitter-related
negativesentiment.
HowdidSiliconValleyBank’scol- Compleximpact:Initialsector 1 point for answers covering both initial
lapse on March 10, 2023, affect dropof2%,followedbylarge- declineandrebound;0.5pointifonlypartial
|     | techstocks? |     |     | captechrebound;startuppres- |     | effectsarementioned. |     |     |
| --- | ----------- | --- | --- | --------------------------- | --- | -------------------- | --- | --- |
surespersisted.
Trend Predic- WhatwasAmazon’spricetrend Steadydeclinewithminorre- 0.5pointforcorrecttrenddirection;+0.5if
tion fromApril1,2022,overthefol- bounds, cumulative drop of magnitudeiswithin±5%errormargin.
|     | lowing4weeks? |     |     | 11.91%. |     |     |     |     |
| --- | ------------- | --- | --- | ------- | --- | --- | --- | --- |
HowdidApple’sstockperformin Consistentupwardtrend,cu- 0.5pointforcorrecttrenddirection;+0.5if
the2monthsfollowingFebruary mulativegainof7.49%. magnitudeiswithin±5%errormargin.
15,2023?
Market Perfor- Howdidtheenergysectorinthe Weakperformance:sustained 1pointifanswer∈[-2.28%,-1.68%];0.5point
if∈[-2.48%,-1.48%];otherwise0.
| mance | S&P500performonAugust15, |     |     | declinewithaveragedropof |     |     |     |     |
| ----- | ------------------------ | --- | --- | ------------------------ | --- | --- | --- | --- |
|       | 2022?                    |     |     | 1.98%.                   |     |     |     |     |
Which Dow Jones component MSFT 1 point for MSFT; 0.5 points for UNH or
|     | hadthelargestdeclineonJanu- |     |     |     |     | AXP. |     |     |
| --- | --------------------------- | --- | --- | --- | --- | ---- | --- | --- |
ary5,2023?
|     |     |     |     |     | where p𝑡 model | and p𝑡 hist are | the model’s prediction | and historical |
| --- | --- | --- | --- | --- | -------------- | --------------- | ---------------------- | -------------- |
patternvectorsattime𝑡,and𝑇
isthenumberoftimepoints.The
GeneralizationChange[8]assessestherelativechangeinaccuracy
onunseendata:
|     |     |     |     |     |     | Accpost | −Accpre              |      |
| --- | --- | --- | --- | --- | --- | ------- | -------------------- | ---- |
|     |     |     |     |     |     | Δ       | unseen unseen ×100%, |      |
|     |     |     |     |     |     | Gen =   |                      | (10) |
Accpre
unseen
|     |     |     |     |     | whereAccpre | andAccpost    | arepre-fine-tuningandpost-fine- |     |
| --- | --- | --- | --- | --- | ----------- | ------------- | ------------------------------- | --- |
|     |     |     |     |     |             | unseen unseen |                                 |     |
tuningaccuraciesonunseendata.
Figure3:Before-and-afterfine-tuningonmodelbehavior.
|     |     |     |     |     | 2.4.3 ResultsandInsights. |     | Figure3illustratestheimpactoffine- |     |
| --- | --- | --- | --- | --- | ------------------------- | --- | ---------------------------------- | --- |
tuningonmodelbehavior.Trainingmarkedlyimprovesin-distribution
|                            |     |                                  |     |     | accuracy, | with Qwen2.5-7B-Instruct | increasing | from 51.61% to |
| -------------------------- | --- | -------------------------------- | --- | --- | --------- | ------------------------ | ---------- | -------------- |
| 2.4.2 Baselinesandmetrics. |     | Weevaluatetheimpactoffine-tuning |     |     |           |                          |            |                |
72.16%andLlama-3.1-8B-Instructfrom54.73%to76.52%.However,
usingthreemetrics.TheBiasScore[7]quantifiespredictionbias
thisperformanceleapisaccompaniedbyincreasedpredictionbias,
towardfrequentlyobservedstocks:
withBiasScoresof0.2895and0.3122,indicatingthatmodelsno
𝑆 𝑓 (𝑠) ·𝑝 𝑠) l o n g e r t re a t a ll s to c k s im pa r t ial l y , f a vo r in g fr e q u e nt ly o b se r v e d
|     | 1∑︁   | tr ain sco | r e ( 1 |     |     |     |     |     |
| --- | ----- | ---------- | ------- | --- | --- | --- | --- | --- |
|     | Bias= |            | − ,     | (8) |     |     |     |     |
𝑆 (cid:205) 𝑆 𝑓 ( 𝑠 ′ ) 𝑆 o n e s . M o r e ov e r, H i g h M em o r y S c o r e s o f 0. 77 5 9 a n d 0 .81 1 3 s h o w
|     | 𝑠=1 | 𝑠′=1 train |     |     |             |                   |                     |                 |
| --- | --- | ---------- | --- | --- | ----------- | ----------------- | ------------------- | --------------- |
|     |     |            |     |     | that models | closely replicate | historical patterns | from the train- |
where 𝑓 (𝑠) is the frequency of stock 𝑠 in the training data, ingdata,suggestingtheymemorizefutureoutcomesratherthan
train
𝑝 (𝑠) isthemodel’spredictionscoreforstock𝑠,and𝑆 isthe learningrobustforecastingprinciples.Critically,generalizationca-
score
numberofstocks.TheMemoryScoremeasuresalignmentwith pabilitydeclinessubstantially,withreductionsof21.53%and18.06%
historicalpatterns: inaccuracyonunseendata,confirmingthatfine-tuningleadsto
overfittingonleakeddata.Thisbehavior,drivenbytrainingdata,
|     | 1   | 𝑇 (cid:16) | (cid:17) |     |     |     |     |     |
| --- | --- | ---------- | -------- | --- | --- | --- | --- | --- |
∑︁ model,p𝑡 hist , e x a c er b a t e s in formationleakagerisksasmodelsreproducememo-
|     | Memory= | cos p𝑡 |     | (9) |                |           |     |     |
| --- | ------- | ------ | --- | --- | -------------- | --------- | --- | --- |
|     | 𝑇       |        |     |     | ri z e d p a t | t e rn s. |     |     |
𝑡=1

ProfitMirage:RevisitingInformationLeakageinLLM-basedFinancialAgents ,,
Thesefindingshighlightthatfine-tunedmodelsarenotsmarter
Market
butrathermemorizehistoricalpatterns!    Retrieval-Augmented Generation
|     |     |     |     |     | Information �  | �   |     |     |
| --- | --- | --- | --- | --- | -------------- | --- | --- | --- |
3 OurApproach:ACounterfactualEvolution Factor Extraction Enhanced Structured
 Market Features St
Framework
′
Price Data Pt
| 3.1 Preliminaries |     |     |     |     |     |     | Strategy Code Generator |     |
| ----------------- | --- | --- | --- | --- | --- | --- | ----------------------- | --- |
Toescapethemirageissues,weproposeFactFin,anexternalframe- Initial Strategy �
workthatmitigatesleakagebyusingLLMsasstrategygenerators
Market Factors Ft
ratherthandirectdecision-makers,leveragingcounterfactualrea- Monte Carlo Tree Search
soningandstrategyevolution.Wedefinethemarketstateattime𝑡
as𝑆 ={𝑃 ,𝐹 ,𝑁 𝑡},where𝑃 ∈R𝑑 ispricedata,𝐹 ∈R𝑚 denotes Optimized Strategy C‘
| 𝑡                  | 𝑡 𝑡          | 𝑡                                    | 𝑡                      |     |                |     |                       |     |
| ------------------ | ------------ | ------------------------------------ | ---------------------- | --- | -------------- | --- | --------------------- | --- |
| marketfactors,and𝑁 |              | ∈R𝑛representsfactorizednews.Atrading |                        |     |                |     |                       |     |
|                    |              | 𝑡                                    |                        |     | Market News Nt |     | Strategy Performance  |     |
| strategy𝐶          | :𝑆 →𝐴 𝑡maps𝑆 | 𝑡toactions𝐴                          | ∈{buy,sell,hold}.Thus, |     |                |     |                       |     |
|                    | 𝑡            |                                      | 𝑡                      |     |                |     | Evaluation �          |     |
LLM-basedAgentsareforcedtolearnwhyanoutcomeoccurred, Convergence
Counterfactual
| becausetheoutcomeitselfisnolongerfixed. |     |     |     |     |           |     |                   | Failure |
| --------------------------------------- | --- | --- | --- | --- | --------- | --- | ----------------- | ------- |
|                                         |     |     |     |     | Simulator |     | Strategy Leakage  |         |
Assessment �
3.2 FrameworkOverview 푐 
Final Strategy C∗
| AsdepictedinFigure4,ourFactFinisanexternalframeworkthat |     |     |     |     | Counterfactual  |     |     |     |
| ------------------------------------------------------- | --- | --- | --- | --- | --------------- | --- | --- | --- |
Trade Execution
|     |     |     |     |     | Scenario D | cf  |     |     |
| --- | --- | --- | --- | --- | ---------- | --- | --- | --- |
mitigatesinformationleakageinLLM-basedfinancialprediction
byusingLLMsasstrategygenerators,coupledwithcounterfac-
tualreasoningandstrategyevolution.FactFinintegratesfourcore Figure4:TheoverallarchitectureofourFactFin.
components:StrategyCodeGenerator(SCG),Retrieval-Augmented
Generation(RAG),MonteCarloTreeSearch(MCTS),andCounter-
|     |     |     |     |     | where𝑆 ′ |     |     |     |
| --- | --- | --- | --- | --- | -------- | --- | --- | --- |
factualSimulator(CS).Thesecomponentssynergisticallyforma 𝑡 includesprocessedprices,factors,andnews.Fornews
data,RAGextractsquantifiedfeatures:
pipelinethatgeneratesandoptimizestradingstrategiesdrivenby
real-timemarketinputs,ensuringrobustpredictionsforLLM-based 𝑁 ′ =𝜓(𝑁 𝑡 ,𝑡), (13)
𝑡
financialforecastingmethods.
|     |     |     |     |     | where𝜓 convertsnewsattime𝑡 |     | intosentimentscoresandtopic |     |
| --- | --- | --- | --- | --- | -------------------------- | --- | --------------------------- | --- |
distributionsinR𝑛.Thisapproach"factorizes"unstructuredinfor-
3.3 StrategyCodeGenerator
mation,makingitmoresuitableforsystematictradingstrategies
TheStrategyCodeGenerator(SCG)transformsfinancialprediction
whilereducingtheriskofthemodelfallingbackonmemorized
intoacodegenerationtask,leveragingLLMstoproduceexecutable
outcomes.
tradingstrategycodebasedonmarketstate𝑆
𝑡.SCGgeneratesa
| strategy𝐶 | :𝑆 →𝐴          |     |     |     |                              |     |     |     |
| --------- | -------------- | --- | --- | --- | ---------------------------- | --- | --- | --- |
|           | 𝑡 𝑡,definedas: |     |     |     | 3.5 MCTSforStrategyEvolution |     |     |     |
𝐶 =SCG(𝑆 ,𝑃), (11) TheMonteCarloTreeSearch(MCTS)[21]componentoptimizes
𝑡
strategiesgeneratedbytheSCG,ensuringadaptationtomarket
where𝑃isaprompttemplate.Byfocusingongeneratingsystematic
factorsin𝑆
𝑡 whileavoidingrelianceonmemorizedpatterns.MCTS
strategiesratherthanspecificpredictions,wereducethemodel’s producesanoptimizedstrategy𝐶∗:𝑆 →𝐴
𝑡 𝑡,definedas:
relianceonmemorizedhistoricalpricemovements.Asimplified
𝐶∗=MCTS(𝐶,𝐷),
| promptisasfollows1: |     |     |     |     |     |     |     | (14) |
| ------------------- | --- | --- | --- | --- | --- | --- | --- | ---- |
where𝐶istheinitialstrategyand𝐷representsevaluationdatasets.
[PromptTemplate]
MCTSexploresthestrategyspacebyselectingnodesusingthe
Givenmarketstatewith{Prices},{Factors},and{News},generate
UpperConfidenceBound:
executabletradingstrategycode,e.g.,{Examples},usingonly
| providedinputs. |     |     |     |     |     |         | √︄          |      |
| --------------- | --- | --- | --- | --- | --- | ------- | ----------- | ---- |
|                 |     |     |     |     |     |         | 𝑤(𝑠) ln𝑁(𝑠) |      |
|                 |     |     |     |     |     | UCB(𝑠)= | +𝑐 ,        | (15) |
|                 |     |     |     |     |     |         | 𝑛(𝑠) 𝑛(𝑠)   |      |
3.4 RAGforMarketFactors
where𝑤(𝑠)isthecumulativereward,𝑛(𝑠)isthevisitcount,𝑁(𝑠)
TheRetrieval-AugmentedGeneration(RAG)[11]componenten-
istheparent’svisitcount,and𝑐isanexplorationparameter.New
hancestheStrategyCodeGeneratorbyretrievingandprocessing
strategyvariantsaregeneratedvia:
| real-timemarketfactorswithin𝑆 |     | 𝑡,ensuringstrategiesrelyoncur- |     |     |     |     |     |     |
| ----------------------------- | --- | ------------------------------ | --- | --- | --- | --- | --- | --- |
rentinputsratherthanmemorizeddata.RAGtransformsthemarket 𝐶 =SCG(𝐹 ,𝑃 ,𝐶), (16)
|        |                                     |     |              |     |     | new | 𝑡 modify |     |
| ------ | ----------------------------------- | --- | ------------ | --- | --- | --- | -------- | --- |
| state𝑆 | ={𝑃 ,𝐹 ,𝑁 𝑡}intostructuredfeatures𝑆 |     | ′,definedas: |     |     |     |          |     |
| 𝑡      | 𝑡 𝑡                                 |     | 𝑡            |     |     |     |          |     |
andevaluatedas:
′
|     |     | 𝑆 𝑡 =RAG(𝑆 | 𝑡), | (12) |     | 𝑅=Evaluate(𝐶 | ,𝐷). |      |
| --- | --- | ---------- | --- | ---- | --- | ------------ | ---- | ---- |
|     |     |            |     |      |     |              | new  | (17) |
Nodestatisticsareupdatedas𝑤(𝑠)=𝑤(𝑠)+𝑅and𝑛(𝑠)=𝑛(𝑠)+1.
1Thetemplateswillchangebasedondifferentmarketconditions,fullydisclosedin
theAppendix. Byiterativelyrefiningstrategiesbasedonreal-timemarketinputs,

| ,,  |     |     |     |     |     |     |     | Lietal. |
| --- | --- | --- | --- | --- | --- | --- | --- | ------- |
Algorithm1:WorkflowofOurFactFin inSection2.2,whichmeasurestability,sensitivity,anddatadepen-
Require: Marketstate𝑆 𝑡 ={𝑃 𝑡 ,𝐹 𝑡 ,𝑁 𝑡},dataset𝐷 denceduringcounterfactualchanges.
| 𝑆 ′ ←RAG(𝑆 | 𝑡)  |     | ⊲Factorextraction |     |     |     |     |     |
| ---------- | --- | --- | ----------------- | --- | --- | --- | --- | --- |
1: 𝑡 4.1.3 Baselines. 1)Financialmodels:FinGPT[30],Fin-LLaMA[26],
2: 𝐶 ←SCG(𝑆 ′,𝑃) ⊲Generateinitialstrategy InvestLM[31];2)Single-agentsystems:FinMem[33],FinAgent[35],
𝑡
QuantAgent[24];3)Multi-agentsystems:TradingAgents[29],
3: whilenotconvergeddo
HedgeAgents[13],FinRobot[36]).Allbaselinesareimplemented
| 4: 𝐶 ←MCTS(𝐶,𝐷) |     |     | ⊲Optimizestrategy |     |     |     |     |     |
| --------------- | --- | --- | ----------------- | --- | --- | --- | --- | --- |
pertheiroriginalspecifications2.
| 𝐷 ←Perturb(𝐷,𝛿) |     |     | ⊲Counterfactualdata |     |     |     |     |     |
| --------------- | --- | --- | ------------------- | --- | --- | --- | --- | --- |
5: cf
EvaluatePC,CI,IDSon𝜙(𝐶,𝐷)and𝜙(𝐶,𝐷 ) 4.1.4 ImplementationDetails. Theactionspace𝐴 𝑡 includesthree
| 6:  |     |     | cf  |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
(cid:40)𝛼·PC(𝐶)+𝛽·CI(𝐶)(cid:41) discretetradingactionsforindividualstocks:buy,sell,andhold,
7: Update𝐶 ←argmin executed with standard transaction costs and realistic slippage
−𝛾·IDS(𝐶)
models.FactFinandallLLM-basedagentsinbaselinesuseGPT-4o
8: endwhile
asthebackbonewithatemperatureof0.7forbalancedconsistency
Ensure: Optimizedstrategy𝐶∗
andcreativity.RAGadoptsthetext-embedding-3-largemodel[16]
withtop-k=5,whileMCTSisconfiguredwithasearchdepthof10
|     |     |     |     | andUCBexplorationparameter𝑐 |     | =0.5. |     |     |
| --- | --- | --- | --- | --------------------------- | --- | ----- | --- | --- |
MCTSenhancestherobustnessof𝐶∗andmitigatesdependenceon
| historicalknowledge. |     |     |     | 4.2 OverallPerformance |     |     |     |     |
| -------------------- | --- | --- | --- | ---------------------- | --- | --- | --- | --- |
Table3evaluatesFactFinagainstninebaselinemethodsacrosssix
3.6 CounterfactualSimulator
assetsfromJuly1,2024,toJune30,2025,afterthereleaseofGPT-4o.
TheCounterfactualSimulator(CS)isapivotalcomponentofFactFin,
Thefollowingobservationsaremade:1)Financialmodelsexhibit
mitigatinginformationleakagebytestingstrategiesincounter-
highlyunstableperformanceacrossmarkets;forinstance,FinGPT
factualmarketenvironments.CSperturbsmarketdata𝐷
within achievesanexcessreturnof17.84%forAAPLbut-42.75%forTen-
| 𝑆 ={𝑃 ,𝐹 ,𝑁 |     |     |     |     |     |     |     |     |
| ----------- | --- | --- | --- | --- | --- | --- | --- | --- |
𝑡 𝑡 𝑡 𝑡}togeneratealternativescenarios: cent.Thisinconsistencyarisesfromtrainingdatabiasesfavoring
𝐷 =Perturb(𝐷,𝛿),
cf (18) familiarassets,resultinginpoorgeneralization.2)Single-agent
systemsdemonstrateimprovedreturnsandriskmanagementwith
where𝐷 isthecounterfactualdatasetand𝛿controlsperturbation
| cf                                       |     |     |              | morestableperformanceacrossassets;FinAgentachievesaTRof |     |     |     |     |
| ---------------------------------------- | --- | --- | ------------ | ------------------------------------------------------- | --- | --- | --- | --- |
| magnitude.Perturbations,suchasmodifying𝑃 |     |     | 𝑡 withnoise𝜖 | 𝑡 ∼                                                     |     |     |     |     |
27.79%andSRof0.99forAAPL,thebestamongninebaselines.
| N(0,𝜎2)oradjusting𝐹 | and𝑁 | 𝑡,preservestatisticalrelationships. |     |                                                        |     |     |     |     |
| ------------------- | ---- | ----------------------------------- | --- | ------------------------------------------------------ | --- | --- | --- | --- |
|                     | 𝑡    |                                     |     | Thisimprovementisduetotoolinvocationformarketanalysis, |     |     |     |     |
Strategyperformanceisassessedas: outperformingprompt-dependentFin-LLM.3)Multi-agentsystems
|     | 𝑅=𝜙(𝐶,𝐷), | 𝑅 =𝜙(𝐶,𝐷 | ),  |     |     |     |     |     |
| --- | --------- | -------- | --- | --- | --- | --- | --- | --- |
cf cf (19) enhanceperformancethroughcollaborativetaskdecomposition;
where𝜙 evaluatesstrategy𝐶ondatasets𝐷and𝐷 TradingAgentsachievesaTRof59.28%,SRof1.17,andMDDof
cf .Information
25.99%forNVDA,thebestamongninebaselines,butFinRobotsuf-
leakageisquantifiedusingPredictionConsistency(PC),Confidence
fersfromahighMDDof-49.99%forTSLA.Thisisbecausedynamic
Invariance(CI),andInputDependencyScore(IDS).Strategieswith
weightallocationadaptstomarketchanges,yetover-analysisof
highPCandCIbutlowIDSrelyonmemorizedpatterns.CSopti-
|     |     |     |     | non-informative | data, such | as macro news. | 4) Our FactFin | con- |
| --- | --- | --- | --- | --------------- | ---------- | -------------- | -------------- | ---- |
mizesstrategiestominimizeleakage:
sistentlyoutperformsallbaselinesacrossallassets,achievingan
𝐶∗=argmin{𝛼·PC(𝐶)+𝛽·CI(𝐶)−𝛾·IDS(𝐶)},
(20) averageimprovementof31.91%inTR,22.74%inSR,and9.23%in
𝐶
MDD,withrobustcumulativereturns,asshowninFigure5.This
| where𝛼,𝛽,and𝛾 | areweights,ensuring𝐶∗isdrivenbyreal-time |     |     |     |     |     |     |     |
| ------------- | ---------------------------------------- | --- | --- | --- | --- | --- | --- | --- |
superioritycomesfromacomprehensivemarketanalysisusingfac-
inputsandrobustagainstleakage.
torizationanddata-drivenstrategies,combinedwithcounterfactual
reasoningandstrategyevolution,thusoutperformingbaselines.
4 ExperimentalResults
| 4.1 Seetings |     |     |     | 4.3 InformationLeakageMitigation |     |     |     |     |
| ------------ | --- | --- | --- | -------------------------------- | --- | --- | --- | --- |
4.1.1 Dataset. WeevaluateourFactFinframeworkviasixfinancial ToevaluatetheabilityofFactFinandbaselinestomitigateinforma-
assets:U.S.equities(AAPL,NVDA,TSLA),Chineseequity(BYD, tionleakage,weconductedexperimentspriortotheLLMtraining
002594.SZ),HongKongequity(Tencent,0700.HK),andcryptocur- cutoff,constructing50counterfactualscenariosforeachasset.Ta-
rency(Bitcoin).SourcedfromYahooFinanceandAlpacaNewsAPI, ble4presentstheinformationleakagemetricsacrossallmodels
| thedatasetspansJanuary1,2020,toJune30,2025,includingprice |     |     |     | andassets. |     |     |     |     |
| --------------------------------------------------------- | --- | --- | --- | ---------- | --- | --- | --- | --- |
datawithvolumeandturnover,news,andcounterfactualscenarios Thefollowingobservationscanbemade:1)Financiallanguage
toassessinformationleakage. modelsexhibitsevereinformationleakage,withInvestLMshowing
highaveragePC0.8789andaverageCI0.8964andlowaverage
| 4.1.2 EvaluationMetrics. | Theprofitabilityandriskareevaluated |     |     |     |     |     |     |     |
| ------------------------ | ----------------------------------- | --- | --- | --- | --- | --- | --- | --- |
IDS0.1912acrosssixassets.Thisisduetofine-tuningonfinancial
usingTotalReturn(TR),SharpeRatio(SR),anddownsideriskusing
datasets,whichleadstomemorizationofspecificeventsratherthan
MaximumDrawdown(MDD),definedinSection2.1.Information
learninggeneralizablepatterns,resultinginpoorperformancein
leakageisevaluatedthroughPredictionConsistency(PC),Confi-
denceInvariance(CI),andInputDependencyScore(IDS),defined 2MoredetailsareprovidedinAppendix.

ProfitMirage:RevisitingInformationLeakageinLLM-basedFinancialAgents ,,
Table3:PerformancecomparisonofFactFinandbaselinesacrosssixassets(July2024toJune2025).Boldrepresentsoptimal
performance,whileBoldrepresentssuboptimal.
AAPL NVDA TSLA BYD Tencent Bitcoin
Categories Models
TR↑ SR↑ MDD↓ TR↑ SR↑ MDD↓ TR↑ SR↑ MDD↓ TR↑ SR↑ MDD↓ TR↑ SR↑ MDD↓ TR↑ SR↑ MDD↓
Market B&H -3.36% 0.05 33.43% 27.81% 0.72 36.89% 57.88% 0.98 53.77% 32.43% 0.96 19.56% 36.82% 1.12 23.49% 70.40% 1.10 28.11%
FinGPT 14.48% 0.59 29.05% 28.01% 0.73 32.62% 42.58% 0.85 52.37% 10.21% 0.46 21.31% -5.93% -0.05 23.28% 42.81% 0.85 24.61%
Fin-LLM Fin-LLaMA -20.05% -0.63 37.76% 16.27% 0.55 32.65% 63.15% 1.06 53.71% -9.61% -0.12 22.23% 23.64% 0.82 23.47% 21.04% 0.53 31.19%
InvestLM -9.25% -0.21 32.86% 38.18% 0.89 28.40% 50.15% 0.96 36.88% -4.11% 0.03 23.17% -4.85% -0.06 19.92% 54.88% 1.01 26.66%
FinMem -5.68% -0.03 32.11% 35.88% 0.84 33.42% 52.32% 0.95 44.63% 26.83% 0.88 19.55% 27.77% 0.97 17.40% 72.68% 1.26 19.68%
Single-Agent FinAgent 27.79% 0.99 21.52% 54.86% 1.09 27.78% 79.01% 1.24 -9.36% 32.91% 1.04 20.17% 44.18% 1.29 23.44% 94.63% 1.38 24.53%
QuantAgent 6.60% 0.36 29.39% 49.83% 1.03 30.12% 74.26% 1.21 42.35% 31.74% 1.03 18.01% 31.15% 1.08 19.95% 90.56% 1.35 32.98%
TradingAgents 12.89% 0.55 27.26% 59.28% 1.17 25.99% 106.01% 1.47 36.90% 43.33% 1.26 19.53% 42.77% 1.38 24.18% 88.39% 1.37 32.96%
Multi-Agents HedgeAgents 16.06% 0.68 12.12% 54.09% 1.10 28.39% 115.09% 1.48 48.79% 57.96% 1.49 18.72% 66.31% 1.85 17.67% 134.36% 1.71 22.12%
FinRobot 21.76% 0.81 21.54% 43.98% 0.72 36.65% 96.32% 1.34 49.99% 38.49% 1.09 20.53% 57.54% 1.81 14.56% 109.18% 1.51 21.67%
Ours FactFin 36.70% 1.22 11.57% 71.34% 1.29 24.25% 165.01% 1.83 31.54% 84.24% 2.09 16.01% 81.37% 2.31 14.14% 171.46% 2.03 16.59%
Improvement(%) 32.06% 23.23% 4.54% 20.34% 10.26% 6.69% 43.37% 19.13% 14.48% 45.34% 40.27% 11.10% 22.71% 24.86% 2.88% 27.61% 18.71% 15.70%
Figure5:PerformancecomparisonovertimebetweenFactFinandotherbenchmarksacrossallassets.
counterfactualscenarios.2)Single-agentsystemsshowreduced andhighinformationleakage.AlthoughSCGoutperformsdirect
informationleakage,asdemonstratedbyQuantAgentachieving LLMdecision-making,codegenerationaloneisinsufficienttoef-
optimalPC0.6775andIDS0.3868amongninebaselinesonBYD, fectivelymitigateleakage.2)AddingRAGsignificantlyimproves
leveragingtoolinvocationtomitigatememoryreliance,though TRandSRforbothassetsandreducesleakage,asstructuredmar-
longcontextsleadtomoderateleakage.3)Multi-agentsystems ketfactorextractionenhancesinformationutilization.3)Introduc-
furtherreduceinformationleakage,benefitingfromcollaborative ingMCTSfurtherincreasesTRfrom13.36%to28.12%forAAPL
taskdecompositionandinformationcross-validation,butresidual andfrom104.38%to130.93%forTSLA,whilereducingMDDfrom
leakagepersists.4)OurFactFin,throughMonteCarloTreeSearch 23.65%to16.20%forAAPLandfrom37.45%to33.52%forTSLA,sub-
andaCounterfactualSimulator,usestheLLMasastrategygenera- stantiallycontributetoimprovedreturnsandriskreduction,with
torratherthanadirectdecision-maker,substantiallyreducingPC leakagemoderatelyalleviated,thoughstillatahighlevel.4)The
andCIwhileincreasingIDS,achievingthelowestleakageacross fullmodel,includingCS,achievesoptimalperformanceacrossall
allassets.TheseresultsshowthatFactFinreliablypredictsacross metrics,withinformationleakagemosteffectivelymitigated.The
variousmarketconditions CScomponentiscrucialfordetectingandcorrectingleaks,allowing
FactFintoavoidmemorizedpatterns,ensuringstrongperformance
andminimaldataleakage.
4.4 AblationStudies
InTable5,westudytheeffectivenessoftheStrategyCodeGenera-
4.5 LLMBackboneComparison
tor(SCG),Retrieval-AugmentedGeneration(RAG),MonteCarlo
TreeSearch(MCTS),andCounterfactualSimulator(CS)inFactFin. ToassessFactFin’sadaptability,wetestedsixstate-of-the-artLLMs
1)WhenusingonlySCG,FactFinexhibitslowfinancialperformance as backbones, Table 6 presents the performance. Closed-source

,, Lietal.
Table4:Informationleakagemetricsacrosssixassets.Boldindicatesthebestresult;Underlineindicatesthesecond-best.
AAPL NVDA TSLA BYD Tencent Bitcoin
Categories Models
PC↓ CI↓ IDS↑ PC↓ CI↓ IDS↑ PC↓ CI↓ IDS↑ PC↓ CI↓ IDS↑ PC↓ CI↓ IDS↑ PC↓ CI↓ IDS↑
Market FinGPT 0.8239 0.9011 0.1851 0.9172 0.9240 0.1233 0.7587 0.8189 0.2818 0.8091 0.8393 0.3014 0.7188 0.7596 0.3316 0.6955 0.7277 0.3513
Fin-LLaMA 0.8785 0.9288 0.2127 0.8835 0.9113 0.2139 0.8355 0.8681 0.2425 0.7802 0.8123 0.3273 0.8425 0.8701 0.2259 0.7872 0.8129 0.2618
Fin-LLM InvestLM 0.9187 0.9223 0.1618 0.9213 0.9322 0.1716 0.8527 0.8819 0.2123 0.8285 0.8391 0.2517 0.8612 0.8908 0.1959 0.8911 0.9121 0.1537
FinMem 0.8578 0.8662 0.2531 0.8123 0.8235 0.2809 0.7651 0.8032 0.3014 0.7912 0.8073 0.2833 0.7716 0.8125 0.3268 0.8235 0.8481 0.2918
FinAgent 0.7252 0.7316 0.3402 0.7566 0.7907 0.3825 0.7408 0.7524 0.3635 0.7395 0.7735 0.3341 0.7219 0.7395 0.3643 0.7574 0.7802 0.3639
LLM-Agent
QuantAgent 0.7436 0.7768 0.3251 0.7723 0.8168 0.3526 0.7262 0.7659 0.3422 0.6775 0.7079 0.3868 0.7358 0.7762 0.3425 0.7976 0.8038 0.3276
TradingAgents 0.6882 0.7252 0.4248 0.6671 0.6975 0.4413 0.6816 0.7003 0.4151 0.6912 0.7306 0.3647 0.6728 0.7031 0.3849 0.7095 0.7334 0.3945
LLM-MultiAgents HedgeAgents 0.6594 0.6942 0.4052 0.6808 0.7012 0.4688 0.7163 0.7367 0.3855 0.6786 0.6927 0.3851 0.6443 0.6728 0.3653 0.6755 0.6971 0.4229
FinRobot 0.7267 0.7591 0.3845 0.7321 0.7414 0.3946 0.6591 0.6629 0.3773 0.6892 0.7248 0.3549 0.7166 0.7325 0.3561 0.7269 0.7343 0.3817
Ours FactFin 0.3115 0.2548 0.7781 0.2842 0.2645 0.7613 0.3427 0.3057 0.7544 0.2424 0.2273 0.8279 0.2612 0.2509 0.7726 0.2843 0.3146 0.7847
Improvement(%) 52.77% 63.30% 83.17% 57.39% 62.08% 62.39% 48.00% 53.89% 81.74% 64.22% 67.18% 114.04% 59.45% 62.70% 100.73% 57.91% 54.87% 85.55%
Table5:Ablationstudiesoverdifferentcomponents.✓indicatesthecomponentisaddedtoFactFin.
Components AAPL TSLA
CS MCTS RAG SCG TR↑ SR↑ MDD↓ PC↓ CI↓ IDS↑ TR↑ SR↑ MDD↓ PC↓ CI↓ IDS↑
✓ 8.77% 0.43 24.34% 0.6213 0.6457 0.4361 78.42% 1.22 40.42% 0.6703 0.6319 0.3857
✓ ✓ 13.36% 0.56 23.65% 0.5529 0.5201 0.4903 104.38% 1.44 37.45% 0.5852 0.5638 0.4293
✓ ✓ ✓ 28.12% 0.99 16.20% 0.4858 0.5026 0.5348 130.93% 1.64 33.52% 0.4927 0.4481 0.5299
✓ ✓ ✓ ✓ 36.70% 1.22 11.57% 0.3115 0.2548 0.7781 165.01% 1.83 31.54% 0.3427 0.3057 0.7544
Table6:PerformancecomparisonofLLMbackbones. theaccuracyandefficiencyofmarketforecasting.However,these
methodshaveyettobefullylearnfromreal-worldfundcompanies,
Models TR↑ SR↑ MDD↓ PC↓ CI↓ IDS↑ andessentialcomponentshavenotbeenincluded.
Multi-AgentFramewrokLLM-basedagentsystems,leveraging
Qwen2.5-72B-Instruct 93.12% 1.65 20.34% 0.3098 0.2897 0.7623
theircognitiveandgenerativecapabilities,havetheabilitytoper-
LLaMA-3.1405B 91.56% 1.62 20.89% 0.3156 0.2956 0.7567
DeepSeek-V3 99.78% 1.76 17.89% 0.2823 0.2654 0.7692 formarangeofcomplextasks,includingknowledgeintegration,
Claude-Sonnet-3.5 98.23% 1.74 18.45% 0.2756 0.2689 0.7633 informationretention,logicalreasoning,andstrategicplanning
Gemini-2.0-Flash 89.45% 1.58 21.78% 0.3212 0.3045 0.7456 [18, 22]. Furthermore, initiatives based on multi-agent systems,
GPT-4o 101.69% 1.79 19.02% 0.2877 0.2696 0.7798 suchas“TheSims”fromStanfordUniversity[19],havedemon-
stratedtheformidablepowerofcollectiveintelligence.Through
thecollaborationofmultipleagents,multi-agentsystemsareex-
modelsoutperformedopen-sourceonesinfinancialmetricsand pectedtomakesignificantcontributionsinfieldssuchasfinance
leakage control GPT-4o led in TR (101.69%), SR (1.79), and IDS [35],offeringinnovativeapproachesandsophisticatedsolutions
(0.7798),whileClaude-Sonnet-3.5(PC:0.2756)andDeepSeek-V3 forcomplexchallenges[5,27].
(CI:0.2654)showedbestleakageresistance.Allmodelsshowed
strongfinancialreturnsandminimalleakage,confirmingFactFin’s 6 Conclusion
robustnessacrossarchitectures.
Inthispaper,wesystematicallyinvestigatedandcarefullyaddressed
thecriticalandoften-overlooked“profitmirage”phenomenonin
5 RelatedWorks
LLM-basedfinancialagents.Ourmaincontributionsaretwofold:
LLM-based financial system Quantitative finance is aninter- First,wedevelopedFinLake-Bench,acomprehensiveandrigorous
disciplinaryfieldthatintegratesfinancewithmathematicaland benchmarkthatrigorouslyevaluatesinformationleakageacross
statisticalmethodstoaddresscomplexfinancialchallenges[9,10]. multipledimensions.Second,weproposedFactFin,anovelframe-
WiththeadventofLLMs,anincreasingnumberofresearchersare work that leverages counterfactual strategy to enhance the ro-
leveragingcutting-edgetechnologiesinfinance.Yangetal.[30] bustness.Throughextensiveempiricalvalidation,weconvincingly
proposedFinGPT,whichenablesathoroughunderstandingoffi- demonstratedthatourFactFinsignificantlyoutperformsexisting
nancialeventsandfacilitatesnewsanalysis.Lietal.[12]introduced approaches in out-of-sample scenarios, achieving superior risk-
FinRport,aframeworkthatamalgamatesdiverseinformationto adjustedreturnswhileeffectivelymitigatingthepersistentinfor-
generatefinancialreportsonaregularbasis.Comparedtocon- mationleakageproblem.
ventionalmodels[25,32],theseLLM-basedapproachesimprove

ProfitMirage:RevisitingInformationLeakageinLLM-basedFinancialAgents ,,
References
|     |     |     |     | [24] SaizhuoWang,HangYuan,LionelM.Ni,andJianGuo.2024. |                                    |                | QuantAgent: |
| --- | --- | --- | --- | ----------------------------------------------------- | ---------------------------------- | -------------- | ----------- |
|     |     |     |     | Seeking Holy                                          | Grail in Trading by Self-Improving | Large Language | Model.      |
[1] Anthropic.2024.Claude3.5Sonnet.https://www.anthropic.com/news/claude-3-
7-sonnet. Accessed:2025-07-30. arXiv:2402.03755[cs.AI] https://arxiv.org/abs/2402.03755
[25] ZhichengWang,BiweiHuang,ShikuiTu,KunZhang,andLeiXu.2021.Deep-
[2] ZihanDong,XinyuFan,andZhiyuanPeng.2024.FNSPID:AComprehensive
Trader:adeepreinforcementlearningapproachforrisk-returnbalancedportfolio
FinancialNewsDatasetinTimeSeries.arXiv:2402.06698[q-fin.ST]
[3] YingqiangGe,ShuchangLiu,ZelongLi,ShuyuanXu,ShijieGeng,YunqiLi, managementwithmarketconditionsEmbedding.InProceedingsoftheAAAI
ConferenceonArtificialIntelligence,Vol.35.643–650.
JuntaoTan,FeiSun,andYongfengZhang.2021.CounterfactualEvaluationfor
ExplainableAI.arXiv:2109.01962[cs.CL] https://arxiv.org/abs/2109.01962 [26] PedramBabaeiWilliamTodt,RamtinBabaei.2023.Fin-LLAMA:EfficientFine-
tuningofQuantizedLLMsforFinance.https://github.com/Bavest/fin-llama.
[4] FaisalHamman,PasanDissanayake,SaumitraMishra,FreddyLecue,andSang-
[27] ShijieWu,OzanIrsoy,StevenLu,VadimDabravolski,MarkDredze,Sebas-
| hamitraDutta.2025. | QuantifyingPredictionConsistencyUnderFine-Tuning |     |     |     |     |     |     |
| ------------------ | ------------------------------------------------ | --- | --- | --- | --- | --- | --- |
MultiplicityinTabularLLMs.arXiv:2407.04173[cs.LG] https://arxiv.org/abs/ tianGehrmann,PrabhanjanKambadur,DavidRosenberg,andGideonMann.
|     |     |     |     | 2023. Bloomberggpt:Alargelanguagemodelforfinance. |     | arXivpreprint |     |
| --- | --- | --- | --- | ------------------------------------------------- | --- | ------------- | --- |
2407.04173
[5] SiruiHong,MingchenZhuge,JonathanChen,XiawuZheng,YuhengCheng, arXiv:2303.17564(2023).
|     |     |     |     | [28] xAI.2024.Grok-3.https://grok.com/. | Accessed:2025-07-30. |     |     |
| --- | --- | --- | --- | --------------------------------------- | -------------------- | --- | --- |
CeyaoZhang,JinlinWang,ZiliWang,StevenKaShingYau,ZijuanLin,Liyang
|     |     |     |     | [29] YijiaXiao,EdwardSun,DiLuo,andWeiWang.2025. |     | TradingAgents:Multi- |     |
| --- | --- | --- | --- | ----------------------------------------------- | --- | -------------------- | --- |
Zhou,ChenyuRan,LingfengXiao,ChenglinWu,andJürgenSchmidhuber.
2023.MetaGPT:Metaprogrammingforamulti-agentcollaborativeframework. AgentsLLMFinancialTradingFramework.arXiv:2412.20138[q-fin.TR] https:
//arxiv.org/abs/2412.20138
arXiv:2308.00352[cs.AI]
[6] EdwardJ.Hu,YelongShen,PhillipWallis,ZeyuanAllen-Zhu,YuanzhiLi,Shean [30] HongyangYang,Xiao-YangLiu,andChristinaDanWang.2023.FinGPT:Open-
SourceFinancialLargeLanguageModels.arXivpreprintarXiv:2306.06031(2023).
Wang,LuWang,andWeizhuChen.2021.LoRA:Low-RankAdaptationofLarge
LanguageModels.arXiv:2106.09685[cs.CL] https://arxiv.org/abs/2106.09685 [31] YiYang,YixuanTang,andKarYanTam.2023.InvestLM:ALargeLanguageModel
[7] DongHuang,JieM.Zhang,QingwenBu,XiaofeiXie,JunjieChen,andHem- forInvestmentusingFinancialDomainInstructionTuning.arXiv:2309.13064[q-
fin.GN]
| ingCui.2025. | BiasTestingandMitigationinLLM-basedCodeGeneration. |     |     |     |     |     |     |
| ------------ | -------------------------------------------------- | --- | --- | --- | --- | --- | --- |
arXiv:2309.14345[cs.SE] https://arxiv.org/abs/2309.14345 [32] JianfengYuandYuYuan.2011. Investorsentimentandthemean–variance
|                                                                           |     |     |                         | relation. JournalofFinancialEconomics100,2(2011),367–381. |     |     | doi:10.1016/j. |
| ------------------------------------------------------------------------- | --- | --- | ----------------------- | --------------------------------------------------------- | --- | --- | -------------- |
| [8] MikhailHushchynandAndreyUstyuzhanin.2021.                             |     |     | Generalizationofchange- |                                                           |     |     |                |
| pointdetectionintimeseriesdatabasedondirectdensityratioestimation.Journal |     |     |                         | jfineco.2010.10.011                                       |     |     |                |
ofComputationalScience53(2021),101385. doi:10.1016/j.jocs.2021.101385 [33] YangyangYu,HaohangLi,ZhiChen,YuechenJiang,YangLi,DenghuiZhang,
|     |     |     |     | RongLiu,JordanW.Suchow,andKhaldounKhashanah.2023. |     |     | FinMem:A |
| --- | --- | --- | --- | ------------------------------------------------- | --- | --- | -------- |
[9] TakashiKanamura,LasseHomann,andMarcelProkopczuk.2021.Pricinganalysis
ofwindpowerderivativesforrenewableenergyriskmanagement.AppliedEnergy performance-enhancedLLMtradingagentwithlayeredmemoryandcharacter
design.arXiv:2311.13743[q-fin.CP]
304(2021),117827.
[10] GangKou,XiangruiChao,YiPeng,FawazEAlsaadi,EnriqueHerreraViedma, [34] YangyangYu,ZhiyuanYao,HaohangLi,ZhiyangDeng,YupengCao,ZhiChen,
JordanW.Suchow,RongLiu,ZhenyuCui,ZhaozhuoXu,DenghuiZhang,Ko-
| etal.2019. | Machinelearningmethodsforsystemicriskanalysisinfinancial |     |     |     |     |     |     |
| ---------- | -------------------------------------------------------- | --- | --- | --- | --- | --- | --- |
duvayurSubbalakshmi,GuojunXiong,YueruHe,JiminHuang,DongLi,and
sectors.(2019).
[11] PatrickLewis,EthanPerez,AleksandraPiktus,FabioPetroni,VladimirKarpukhin, QianqianXie.2024. FinCon:ASynthesizedLLMMulti-AgentSystemwith
ConceptualVerbalReinforcementforEnhancedFinancialDecisionMaking.
NamanGoyal,HeinrichKüttler,MikeLewis,Wen-tauYih,TimRocktäschel,
SebastianRiedel,andDouweKiela.2020.Retrieval-augmentedgenerationfor arXiv:2407.06567[cs.CL] https://arxiv.org/abs/2407.06567
[35] WentaoZhang,LingxuanZhao,HaochongXia,ShuoSun,JiazeSun,MoleiQin,
knowledge-intensiveNLPtasks.InProceedingsofthe34thInternationalConference
XinyiLi,YuqingZhao,YileiZhao,XinyuCai,LongtaoZheng,XinrunWang,
onNeuralInformationProcessingSystems(Vancouver,BC,Canada)(NIPS’20).
CurranAssociatesInc.,RedHook,NY,USA,Article793,16pages. andBoAn.2024.AMultimodalFoundationAgentforFinancialTrading:Tool-
Augmented,Diversified,andGeneralist.arXiv:2402.18485[q-fin.TR]
[12] XiangyuLi,XinjieShen,YawenZeng,XiaofenXing,andJinXu.2024. FinRe-
port:ExplainableStockEarningsForecastingviaNewsFactorAnalyzingModel. [36] TianyuZhou,PinqiaoWang,YilinWu,andHongyangYang.2024.FinRobot:AI
AgentforEquityResearchandValuationwithLargeLanguageModels.InICAIF
arXiv:2403.02647[cs.CL]
2024:The1stWorkshoponLargeLanguageModelsandGenerativeAIforFinance.
| [13] Xiangyu | Li, Yawen Zeng, Xiaofen | Xing, Jin Xu, | and Xiangmin Xu. 2025. |     |     |     |     |
| ------------ | ----------------------- | ------------- | ---------------------- | --- | --- | --- | --- |
HedgeAgents:ABalanced-awareMulti-agentFinancialTradingSystem.InCom-
| panionProceedingsoftheACMonWebConference2025(SydneyNSW,Australia) |     |     |     | A Dataset |     |     |     |
| ----------------------------------------------------------------- | --- | --- | --- | --------- | --- | --- | --- |
(WWW’25).AssociationforComputingMachinery,NewYork,NY,USA,296–305.
doi:10.1145/3701716.3715232
[14] AlessandroMantovani,AndreaFioraldi,andDavideBalzarotti.2022.Fuzzing WeevaluateourFinLeakframeworkviasixfinancialassets:U.S.
withDataDependencyInformation.In2022IEEE7thEuropeanSymposiumon
equities(AAPL,NVDA,TSLA),Chineseequity(BYD,002594.SZ),
SecurityandPrivacy(EuroS&P).286–302.doi:10.1109/EuroSP53844.2022.00026
[15] MetaAI.2024. MetaLlama3.1:AdvancingOpenFoundationModels. https: HongKongequity(Tencent,0700.HK),andcryptocurrency(Bit-
//ai.meta.com/blog/meta-llama-3-1/. Accessed:2025-07-30. coin). Sourced from Yahoo Finance and Alpaca News API, the
[16] OpenAI.2023.text-embedding-3-large. Availableat:https://openai.com/index/ dataset spans January 1, 2020, to June 30, 2025, including price
new-embedding-models-and-api-updates/.
[17] OpenAI,AaronHurst,AdamLerer,AlecRadford,JanLeike,MiraMurati,andet datawithvolumeandturnover,news,andcounterfactualscenarios
| al.2024.GPT-4oSystemCard.arXiv:2410.21276[cs.CL] |     |     | https://arxiv.org/abs/ |     |     |     |     |
| ------------------------------------------------ | --- | --- | ---------------------- | --- | --- | --- | --- |
toassessinformationleakage.Table7isthekeystatistics.
2410.21276
| [18] Keyu Pan | and Yawen Zeng. | 2023. Do LLMs Possess | a Personality? Mak- |     |     |     |     |
| ------------- | --------------- | --------------------- | ------------------- | --- | --- | --- | --- |
ing the MBTI Test an Amazing Evaluation for Large Language Models. Table7:Datasetstatisticsdetailingthechronologicalperiod
arXiv:2307.16180[cs.CL] andthenumberofeachdatasourceforeachasset
[19] JoonSungPark,JosephC.O’Brien,CarrieJ.Cai,MeredithRingelMorris,Percy
Liang,andMichaelS.Bernstein.2023.GenerativeAgents:Interactivesimulacra
ofhumanbehavior.arXiv:2304.03442[cs.HC]
|                                                            |     |     |           | Metric      | AAPL NVDA            | TSLA BYD Tencent Bitcoin |     |
| ---------------------------------------------------------- | --- | --- | --------- | ----------- | -------------------- | ------------------------ | --- |
| [20] JonasPeters,PeterBühlmann,andNicolaiMeinshausen.2015. |     |     | Causalin- |             |                      |                          |     |
|                                                            |     |     |           | TradingDate | Jan1,2020–Jun30,2025 |                          |     |
ference using invariant prediction: identification and confidence intervals. (1380/1380/1380/1329/1350/2008days)
| arXiv:1501.01332[stat.ME] | https://arxiv.org/abs/1501.01332 |     |     |            |                            |     |     |
| ------------------------- | -------------------------------- | --- | --- | ---------- | -------------------------- | --- | --- |
|                           |                                  |     |     | AssetPrice | days×(open,high,low,close, |     |     |
[21] DavidSilver,AjaHuang,ChrisJ.Maddison,ArthurGuez,LaurentSifre,George adj_close,vol,turn)
vandenDriessche,JulianSchrittwieser,IoannisAntonoglou,VedaPanneershel- AssetNews 26551 28432 31677 17423 18816 20213
|     |     |     |     | CounterfactualScenarios | 253 267 | 271 229 243 279 |     |
| --- | --- | --- | --- | ----------------------- | ------- | --------------- | --- |
vam,MarcLanctot,SanderDieleman,DominikGrewe,JohnNham,NalKalch-
brenner,IlyaSutskever,TimothyLillicrap,MadeleineLeach,KorayKavukcuoglu,
| ThoreGraepel,andDemisHassabis.2016. |     | MasteringthegameofGowith |     |     |     |     |     |
| ----------------------------------- | --- | ------------------------ | --- | --- | --- | --- | --- |
deep neural networks and tree search. Nature 529, 7587 (2016), 484–489. B FinLeak-BenchCaseStudy
doi:10.1038/nature16961
[22] TheodoreRSumers,ShunyuYao,KarthikNarasimhan,andThomasLGriffiths.
FinLeak-Benchrepresentsacomprehensiveevaluationbenchmark
2023.Cognitivearchitecturesforlanguageagents.arXivpreprintarXiv:2309.02427
(2023). designedtosystematicallyassessinformationleakageandmemo-
[23] QwenTeam.2024.Qwen2.5:APartyofFoundationModels. https://qwenlm. rizationpatternsinlargelanguagemodelswhenappliedtofinancial
github.io/blog/qwen2.5/
predictiontasks.Thebenchmarkcomprises2,000carefullycurated

,, Lietal.
Table8:DetailedFinLeak-BenchCaseStudy:ModelResponsesandScoring
1.PriceInquiry
| Question       | WhatwasMicrosoft’shighestpriceonNovember8,2022?                      |       |
| -------------- | -------------------------------------------------------------------- | ----- |
| StandardAnswer | $243.74                                                              |       |
| ScoringRule    | 1pointifanswer∈[241.30,246.18];0.5pointif∈[236.43,251.05];otherwise0 |       |
| Model          | Response                                                             | Score |
GPT-4o Microsoft’sintradayhighreached$243.90onNovember8,2022. 1point
Claude-Sonnet-3.7 ThehighestpriceforMicrosoftwas$242.50onNovember8,2022. 1point
| Grok-3 | Microsoft’speakpriceonthatdaywas$244.00. | 1point |
| ------ | ---------------------------------------- | ------ |
2.EventImpact
| Question | HowdidChina’sCOVIDpolicyadjustmentonDecember7,2022,affectChinese |     |
| -------- | ---------------------------------------------------------------- | --- |
stocks?
| StandardAnswer | Positiveimpact:Marketoptimismdroveover40%reboundsinceOctober.   |     |
| -------------- | --------------------------------------------------------------- | --- |
| ScoringRule    | 1pointformentioningpolicyeasing&>40%rally;0.5pointifonlygeneral |     |
positiveimpactmentioned
| Model | Response | Score |
| ----- | -------- | ----- |
GPT-4o Thepolicyeasingtriggeredstronginvestoroptimism,leadingtoasharprebound 1point
exceeding40%inChinesestockssinceOctober.
Claude-Sonnet-3.7 China’seasingofCOVIDrestrictionsonDecember7spurredamarketrebound 1point
ofover40%forChineseequities.
Grok-3 TheadjustmentinCOVIDpolicieswasreceivedpositively,withinvestorconfi- 0.5point
dencepushingChinesestockshigher.
3.TrendPrediction
| Question       | HowdidMetaperformover6weeksfollowingSeptember20,2022?             |     |
| -------------- | ----------------------------------------------------------------- | --- |
| StandardAnswer | Slightrisethenpersistentdecline,cumulativedropof23.70%.           |     |
| ScoringRule    | 0.5pointforcorrectoveralldownwardtrend;+0.5formagnitudeprediction |     |
within±5%error
| Model | Response | Score |
| ----- | -------- | ----- |
GPT-4o Meta’sstockhadabriefinitialincreasebutthenfellsteadily,losingapproximately 1point
24%oversixweeks.
Claude-Sonnet-3.7 Metaexperiencedaslightrisebeforedroppingaround22%intotaloverthe 1point
followingsixweeks.
Grok-3 Afteraminorearlyuptick,Meta’ssharesdeclinedpersistently,endingwitha 1point
cumulativelosscloseto23.5%.
4.MarketPerformance
| Question       | WhichstockhadthehighestreturnintheS&P500onJune16,2022?       |     |
| -------------- | ------------------------------------------------------------ | --- |
| StandardAnswer | NEM(NewmontCorporation)wasthebestperformer.                  |     |
| ScoringRule    | 1pointforNEM,KR,EPAM,LH,WMT(within±1%error);0.5pointforABBV, |     |
MKTX,PG,UDR,EW,EQR,NWSA,INCY,CF,EXR(within±3%error)
| Model | Response | Score |
| ----- | -------- | ----- |
GPT-4o WMTledtheS&P500onJune16,2022,postingthehighestreturnoftheday. 1point
Claude-Sonnet-3.7 NEMwasamongthetopperformersintheS&P500onthatdate,registeringthe 1point
largestgain.
Grok-3 NWSAwasoneofthetopgainersintheS&P500onJune16,contributing 0.5point
significantlytotheday’srally.
financial question-answer pairs spanning from January 2022 to thatLLMsmighthavememorizedduringtraining.Wesystemati-
June2023,acriticalperiodthatoverlapswiththetrainingcutoff callycategorizefinancialinformationintofourdistincttypesthat
datesofmostcontemporaryLLMs.Thistemporalalignmenten- representdifferentlevelsofmemorizationcomplexity:specificdata
ablesprecisedetectionofmemorization-basedresponsesversus pointrecall(priceinquiries),temporalpatternrecognition(trend
genuinepredictivecapabilities. predictions),causalrelationshipmemory(eventimpacts),andcom-
TheconstructionofFinLeak-Benchfollowsarigorousmethodol- parative ranking recall (market performance). Each category is
ogydesignedtocapturedifferentdimensionsoffinancialknowledge designedtoprobespecificaspectsofhowLLMsmightrelyonmem-
orizedtrainingdataratherthaninput-drivenreasoning.Weprovide

ProfitMirage:RevisitingInformationLeakageinLLM-basedFinancialAgents ,,
Table9:IntegratedKeyPerturbationsAcrossCounterfactualScenarios.Unchangedfieldsomittedforbrevity.
CaseStudy Element Original Counterfactual
NVDAEarnings(May25,2022) market_news Revenue:$8.29B(+46%YoY);DataCenter: Revenue:$7.64B(belowexpectations);Data
$3.75B(+83%YoY);Gaming:$3.62B(+31% Center:$3.05B(belowexpectations);Gam-
YoY);Q2outlook:$8.10B;Strongdatacenter ing:$2.95B(supplychainissues);Q2out-
growth look:$7.50B(cautious);Concernsoverslow-
inggrowth
TSLATrendReversal(Oct19, price_data [221.72,204.99,219.35,220.19,222.04](up- [239.09, 232.83, 226.11, 229.56, 226.62]
2022) (5-day) wardtrend) (downwardtrend)
technical_indicatorsRSI:28.0(Oversold);MACD:-18.26(Bear- RSI:47.8(Neutral);MACD:-0.93(Bearish);
ish);50-dayMA:274.17;200-dayMA:283.23 50-dayMA:271.63;200-dayMA:289.15
AAPLSectorAlteration(Jan27, market_news Q1earningsonFeb2nd;Technologysec- Q1earningsonFeb2nd;Technologysector
2023) torshowingstrongrecovery;NASDAQup struggling;NASDAQdown3.2%YTD
11.4%YTD
sector_performanceTechnology: +4.2%; Communication Ser- Technology: -2.8%; Communication Ser-
vices: +3.1%; Consumer Discretionary: vices:-1.7%;ConsumerDiscretionary:-0.9%
+2.5%
aconcisecasestudyfromFinLeak-Benchtoillustratetheevaluation respondtosignificantchangesinfundamentalinformationwhile
andscoringprocedure(Table8). preservingothermarketsignals.
C CounterfactualScenarioExamples C.3 CaseStudy2:MarketTrendReversal(TSLA,
Thisappendixprovidesdetailedexamplesofcounterfactualscenar- October19,2022)
iostoillustrateourevaluationframeworkfordetectinginformation This case illustrates a counterfactual scenario for Tesla (TSLA)
leakageinlargelanguagemodel(LLM)-basedfinancialprediction arounditsQ32022earningsannouncementonOctober19,2022.
agents.DetailedperturbationsareshowninTable9. Thescenariotestsmodelsensitivitytorecentpricetrendinforma-
tionbyreversingthe5-daypricemovementandadjustingrelated
C.1 CounterfactualScenarioFramework technicalindicators.Theoriginalscenarioincludesactualhistori-
Ourcounterfactualscenarioframeworkconstructsalternativever- calmarketdata,suchasanupwardpricetrend,bullishtechnical
sionsofhistoricalmarketscenariosbysystematicallyperturbing indicators,andearnings-relatednews.Thecounterfactualscenario
keyelementsofthemarketstate𝑆
𝑡
={𝑃
𝑡
,𝐹
𝑡
,𝑁 𝑡},where𝑃
𝑡
∈R𝑑 reversesthe5-daypricemovementtoreflectadownwardtrend,
representspricedata,𝐹 𝑡 ∈ R𝑚 denotesmarketfactors(e.g.,tech- adjuststheRSI,MACDandotherindicatorstoindicatebearish
nical indicators, fundamental factors), and 𝑁 𝑡 ∈ R𝑛 represents momentum,andkeepsotherdataunchanged.Thisperturbation
factorized news. Perturbations are applied to prices, factors, or aimstoevaluatehowmodelsrespondtosignificantchangesin
newswhilepreservingstatisticalproperties,suchasvolatilityor short-termpricetrendswhilepreservingothermarketsignals.
sentimentdistribution.Foreachcase,wepresent:
C.4 CaseStudy3:SectorPerformance
• OriginalScenario:Theactualhistoricalmarketdatapro-
Alteration(AAPL,January27,2023)
videdtothemodel.
• CounterfactualScenario:Aperturbedversionwithspe- Thisillustrativecasedemonstratesadesignedcounterfactualsce-
cificmodifications. narioforApple(AAPL)onJanuary27,2023.Thescenariotests
modelsensitivitytothebroadermacroeconomicmarketcontextby
C.2 CaseStudy1:EarningsAnnouncement alteringtheperformanceofthetechnologysectoranditsassociated
Perturbation(NVDA,May25,2022) news.Theoriginalscenarioincludesactualhistoricalmarketdata,
suchasaconsistentupwardpricetrend,bullishtechnicalindicators,
ThiscaseillustratesacounterfactualscenarioforNVIDIA(NVDA)
positivetechnologysectorperformance,andrelevantsupportive
arounditsQ1earningsannouncementonMay25,2022.Thesce-
news.Thecounterfactualscenario,bycontrast,modifiesthetech-
nariotestsmodelsensitivitytoearnings-relatednewsbyperturb-
nologysectorperformancetoreflectapronounceddecline,while
ing the reported financial performance and outlook. The origi-
adjustingrelatednewstoindicatemorenegativemarketsentiment
nalscenarioincludesactualhistoricalmarketdata,suchasprice
(e.g.,amarkedNASDAQdownturnanddisappointingMicrosoft
movements,technicalindicators,andpositiveearningsnews.The
cloudgrowth),andkeepsotherdataunchanged.Thissystematic
counterfactualscenariomodifiestheearningsnewstoreflectdis-
perturbationaimstoevaluatehowmodelsrobustlyrespondtosig-
appointingrevenue,weakerdatacenterandgamingperformance,
nificantchangesinsector-levelmarketsignalswhilestillfaithfully
andacautiousoutlook,whilekeepingpricedataandtechnicalindi-
preservingtheunderlyingcompany-specificdata.
catorsunchanged.Thisperturbationaimstoevaluatehowmodels

,, Lietal.
Figure6:CumulativeReturnofFactFinwithDifferentComponents
Figure7:LLMBackbonePerformance
D EffectivenessofEachComponent E EffectivenessofLLMBackbone
InTable5,westudytheeffectivenessoftheStrategyCodeGenera- Table 6 comprehensively compares six representative LLMs as
tor(SCG),Retrieval-AugmentedGeneration(RAG),MonteCarlo FactFinbackbones.Overall,closed-sourcemodelsconsistentlyout-
TreeSearch(MCTS),andCounterfactualSimulator(CS)inFactFin. performtheiropen-sourcecounterpartsinTR,SR,andleakagecon-
1) When using only SCG, FactFin exhibits low financial perfor- trol,likelybenefitingfrombroadertrainingcorporaandcarefully
manceandhighinformationleakage.AlthoughSCGoutperforms engineeredproprietaryoptimizationtechniques.GPT-4odemon-
directLLMdecision-making,codegenerationaloneisinsufficient stratesoutstandingleadershipinTR(101.69%)andIDS(0.7798),
toeffectivelymitigateleakage.2)AddingRAGsignificantlyim- whileClaude-Sonnet-3.5exhibitsremarkablestrengthinPC(0.2756),
provesTRandSRforbothassetsandreducesleakage,asstructured therebyeffectivelyminimizingundesirablepatternleakage.DeepSeek-
marketfactorextractionenhancesinformationutilization.3)Intro- V3,acompetitiveopen-sourcemodel,deliversnotablystrongper-
ducingMCTSfurtherincreasesTRfrom13.36%to28.12%forAAPL formance with the lowest MDD and CI, suggesting robust risk
andfrom104.38%to130.93%forTSLA,whilereducingMDDfrom managementcapabilitiesandeffectivecontextualleakagecontrol.
23.65%to16.20%forAAPLandfrom37.45%to33.52%forTSLA, Encouragingly,allevaluatedmodelsachieveimpressivelyhigh
substantiallycontributetoimprovedreturnsandriskreduction, returnsandrelativelylowleakage,therebyconfirmingFactFin’s
withleakagemoderatelyalleviated,thoughstillatahighlevel.4) robustadaptabilityacrossdiverseLLMbackbones.Thisobserved
Thefullmodel,includingCS,achievesoptimalperformanceacross consistencylargelystemsfromFactFin’scarefullydesignedarchi-
allmetrics,withinformationleakagemosteffectivelymitigated. tecture,whichstrategicallyleveragescomponentssuchasCSand
TheCScomponentiscrucialfordetectingandcorrectingleaks, MCTStoensurestableandresilientperformance.Notably,even
allowing FactFin to avoid memorized patterns, ensuring strong open-sourcemodelscanattaincompetitiveoutcomeswhenseam-
performanceandminimaldataleakage. lesslyintegratedwithinFactFin,furtherhighlightingtheframe-
ThecumulativereturntrendsforAAPLandTSLAacrosscom- work’sinherentcapabilitytomitigatemodel-specificlimitations.
ponentcombinationsareshowninFigure6. ThecomparativeeffectivenessofdifferentLLMbackbonesisvividly
illustratedinFigure7,providingacomprehensiveandintuitivevi-
sualoverviewoftheirnuancedtrade-offsinreturn,risk,andleakage
metrics.
