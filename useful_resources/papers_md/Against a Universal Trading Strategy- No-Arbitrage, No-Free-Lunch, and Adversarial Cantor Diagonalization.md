# Against a Universal Trading Strategy- No-Arbitrage, No-Free-Lunch, and Adversarial Cantor Diagonalization

> *Source PDF: Against a Universal Trading Strategy- No-Arbitrage, No-Free-Lunch, and Adversarial Cantor Diagonalization.pdf*
> *Extraction: Combined — previous structured extraction (base) + markitdown raw text*

---

# Against a Universal Trading Strategy- No-Arbitrage, No-Free-Lunch, and Adversarial Cantor Diagonalization

> *Source PDF: Against a Universal Trading Strategy- No-Arbitrage, No-Free-Lunch, and Adversarial Cantor Diagonalization.pdf*
> *Extraction: Combined — markitdown raw text (base) + previous extraction supplements*

---

Against a Universal Trading Strategy: No-Arbitrage, No-Free-Lunch, and Adversarial
Cantor Diagonalization

1Institute for Theoretical Physics, TU Wien, Wiedner Hauptstrasse 8-10/136, A-1040 Vienna, Austria∗
(Dated: April 16, 2026)

Karl Svozil

1

We investigate the impossibility of universally winning trading strategies—those generating strict
profit across all market trajectories—through three distinct mathematical paradigms. Fundamen-
tally, under standard admissibility constraints, the existence of such a strategy is a strict subset
of strong arbitrage, which is mathematically precluded in competitive markets admitting an equiv-
alent martingale measure. Beyond this rigorous measure-theoretic foundation, we explore analo-
gous limitations in two alternative modeling regimes. Combinatorially, the No-Free-Lunch theorem
demonstrates that outperformance requires exploitation of non-uniform market structure, as uniform
averaging precludes universal dominance. Computationally, a Turing diagonalization argument con-
structs an adversarial environment that defeats any computable trading algorithm, shifting the
impossibility from exogenous price paths to adaptive adversaries. These mathematical limits are
framed by a time-reversal heuristic that establishes a formal analogy between financial martingale
measures and thermodynamic detailed balance, resolving the Maxwell’s Demon analogy for markets
without relying on physically irrelevant Landauer erasure costs. Using the Wheel Options Strategy
as a case study, we demonstrate that strategies succeeding “for all practical purposes” (FAPP) inher-
ently depend on transient regime assumptions, meaning their automated execution systematically
amplifies tail risks.

I.

INTRODUCTION

The question of whether a trading strategy can al-
ways make money—regardless of what the market does—
is older than mathematical finance itself.
It animated
Bachelier’s 1900 thesis [1], was formalized by Samuel-
son [2] and Fama [3] into the Efficient Market Hypothesis
(EMH), and underlies every retail advertisement promis-
ing “passive income in any market condition.” The EMH
gives the standard economic answer: competitive pres-
sure eliminates consistent excess returns. But the EMH
is a statement about equilibrium outcomes across many
agents; it is not, by itself, a mathematical proof that no
individual strategy can ever win unconditionally.

Defining a “universal strategy” as one that yields a
strictly positive terminal wealth across all valid market
trajectories is an exceptionally strong condition. Conse-
quently, the intellectual burden lies not merely in proving
impossibility, but in precisely delineating the mathemati-
cal frameworks under which this impossibility manifests.
This paper articulates this impossibility through three
distinct theoretical paradigms, carefully distinguishing
their incompatible underlying assumptions:

1. Financial (Measure-Theoretic, Sec. III). The
rigorous core of the impossibility. We show that un-
der standard admissibility constraints, a universal
strategy trivially constitutes an arbitrage opportu-
nity. By the First Fundamental Theorem of Asset
Pricing [4, 5], this is precluded in markets admit-
ting an equivalent martingale measure.

2. Combinatorial (Uniform Search, Sec. V). The
Wolpert–Macready No-Free-Lunch theorem [6] es-
tablishes that under a uniform distribution over all
possible discrete evolutions, no strategy uniformly
dominates. While markets are not uniform, this
boundary highlights that practical trading is an ex-
ercise in structure exploitation, not universal algo-
rithmic superiority.

3. Computational (Adversarial, Sec. VI). Shift-
ing the paradigm from exogenous price processes
to adaptive environments, we utilize the core diag-
onalization logic of the Halting Problem [7] to show
that no computable strategy can survive an adver-
sary capable of simulating it.

We frame these findings with a time-reversal criterion
(Sec. II) that motivates a formal analogy with Maxwell’s
Demon (Sec. IV). We treat this analogy carefully, not-
ing that while the equivalent martingale measure mirrors
thermodynamic detailed balance, appealing to physical
Landauer erasure costs to explain financial dissipation is
quantitatively irrelevant. The Wheel Options Strategy
is examined in Sec. VIII as a concrete illustration, and
the fragile scope of strategies operating for all practical
purposes (FAPP) is discussed in Sec. IX.

II. THE TIME-REVERSAL HEURISTIC

To build intuition, we define a purely kinematic crite-
rion. Let M be a suitably rich space of strictly positive
paths P : [0, T ] → Rn
>0. The time reversal (cid:101)P of a path P
is

∗ karl.svozil@tuwien.ac.at; http://tph.tuwien.ac.at/˜svozil

(cid:101)P (t) = P (T − t),

t ∈ [0, T ].

(1)

6
2
0
2

r
p
A
4
1

]

R
T
.
n
i
f
-
q
[

1
v
4
3
3
3
1
.
4
0
6
2
:
v
i
X
r
a

2

Definition 1 (Pathwise Universal Strategy). A strategy
σ is universally successful if its realized profit-and-loss
satisfies Π[σ; P ] > 0 for all P ∈ M.

Criterion 1 (Time-Reversal Necessity). If σ is univer-
sally successful on a path space closed under time rever-
sal, it must be time-reversal consistent: for every P ∈ M
where Π[σ; P ] > 0, it must hold that Π[σ; (cid:101)P ] > 0.

The operational content is clear: a momentum strategy
profiting on a rising trajectory will mechanically fail on
its time-reversed (falling) counterpart.
Remark 1 (Filtrations and Semimartingales). We explic-
itly state this as a heuristic because, in standard stochas-
tic finance, the space of admissible trajectories is not
trivially closed under time reversal. Time-reversing a
standard semimartingale generally destroys both its semi-
martingale property and its adaptedness to the forward
filtration. Therefore, this criterion serves as a concep-
tual framing for strict pathwise claims, while the rigorous
probabilistic formulation is provided in Sec. III.

B. The Martingale Constraint

By the Delbaen–Schachermayer generalization of the
First Fundamental Theorem [4, 5], a market satisfies
the condition of No Free Lunch with Vanishing Risk
(NFLVR) if and only if there exists an equivalent local
martingale measure (EMM) Q ≈ P.

Corollary 2 (No Universal Strategy). If an equivalent
martingale measure Q exists, no universally successful
admissible strategy can exist.

This is the definitive mathematical constraint. Uni-
versality is simply an impossibly strong formulation of
arbitrage; its preclusion is the foundational premise of
modern asset pricing. Thus, the non-existence of uni-
versal strategies is not a deep extension of asset pricing
theory, but rather a direct corollary of its foundational
no-arbitrage axiom.

III. THE NO-ARBITRAGE FOUNDATION

A. Admissibility and Arbitrage

The defining proof of impossibility stems from the First
Fundamental Theorem. We must first impose standard
regularity. Without admissibility constraints, pathwise
gains can be trivially manufactured via doubling strate-
gies (gambler’s ruin constructions), which are financially
meaningless due to infinite intermediate drawdown lim-
its.

Let the market be defined on a filtered probability
space (Ω, F, {Ft}t∈[0,T ], P). A strategy σ is admissi-
ble if its wealth process is bounded from below almost
surely [5].

Theorem 1 (Universality implies Arbitrage). If there
exists an admissible, self-financing strategy σ with zero
initial capital such that Π[σ; P (ω)] > 0 for all ω ∈ Ω,
then the market admits a strong arbitrage opportunity.

Proof. By assumption, the strategy yields strictly posi-
tive terminal wealth pathwise over all ω. Consequently,
under the physical measure P, the terminal wealth is
strictly positive almost surely (P(Π > 0) = 1). An ad-
missible, zero-cost, self-financing strategy ending strictly
positive almost surely strictly satisfies the condition for
a strong arbitrage (arbitrage of the first kind).

Remark 2 (Strength Hierarchy). The converse (Arbitrage
=⇒ Universal) is generally false. Our definition of uni-
versality demands pathwise strict positivity (Π > 0 for
all ω), which is a strictly stronger condition than clas-
sical arbitrage (which requires only P(Π ≥ 0) = 1 and
P(Π > 0) > 0). Because pathwise universality trivially
implies strong arbitrage, its impossibility is mathemat-
ically immediate once admissibility constraints are im-
posed under an equivalent martingale measure.

IV. THE MAXWELL’S DEMON ANALOGY:
HONEST BOUNDARIES

A. Formal Analogy, Not Equivalence

Maxwell’s Demon observes individual gas molecules
and sorts them to extract net work from thermal
equilibrium—violating the Second Law of Thermodynam-
ics. The physical resolution (Landauer, Bennett [8, 9])
is that the Demon’s memory must be erased to operate
cyclically, incurring an irreversible thermodynamic cost
of kBT ln 2 per bit.

A universally successful trading strategy claims to act
as a financial Maxwell’s Demon, extracting net profit
from market fluctuations in both temporal directions.
The conceptual mapping is highly instructive (Table I).

TABLE I. Formal analogy between Maxwell’s Demon and uni-
versally winning trading strategies.

Universal Strategy

Maxwell’s Demon
Observes molecule velocities Observes price movements
Sorts molecules for gradient
Works in both time directions Profits on P and (cid:101)P
Extracts equilibrium work
Resolution: info. erasure cost Resolution: absence of EMM

Extracts universal profit

Exploits movements for profit

The deepest point of contact lies in the concept of equi-
librium. In statistical mechanics, true equilibrium satis-
fies detailed balance: microscopic processes equal their
time reversals. In finance, the equivalent martingale mea-
sure Q enforces the financial equivalent of detailed bal-
ance: there is no preferred drift direction under the pric-
ing measure.

B. What the Analogy Does Not Establish

We must aggressively limit the scope of this analogy.
It is a formal mapping, not a category-theoretic isomor-
phism.

Transaction costs are not the core mechanism.
While transaction costs mimic thermodynamic dissipa-
tion, a frictionless market with zero spread still mathe-
matically precludes universal strategies via Corollary 2.
Landauer’s limit is irrelevant. The minimum en-
ergy to erase one bit at room temperature is E ≈
2.87 × 10−21 J. The Landauer cost of an algorithmic
trade is roughly 10−27 USD. Presenting thermodynamic
erasure as a financially relevant constraint on algorithms
is numerically dishonest.

The analogy illuminates why attempts to defeat effi-
cient markets resemble attempts to build perpetual mo-
tion machines, but the independent proof of financial im-
possibility lies strictly in measure theory (Sec. III).

3

automated, it acts as a computable Turing machine [10].
This guarantees its defeat via formal Turing diagonaliza-
tion [7, 11].

The Adversarial Market Construction can be modelled
as follows: Let Mσ be a computable trading algorithm
mapping discrete histories Ht to a target position wt ∈ R.

Theorem 4 (Defeat of Computable Strategies). Against
any computable strategy Mσ, there exists an adversarially
constructed, computable market trajectory Padv on which
Mσ realizes non-positive profit.

Proof. Because Mσ is a deterministic algorithm, a
market-making adversary with oracle access to the strat-
egy can simulate its next move. Set P0 > 0. At each
step t, the adversary simulates Mσ(Ht) → wt and sets
the next price opposing the strategy’s bet:

Pt+1 =






Pt · e−ϵ
Pt · e+ϵ
Pt

if wt > 0 (Strategy buys)
if wt < 0 (Strategy sells)
if wt = 0 (Strategy neutral)

(2)

V. THE COMBINATORIAL PERSPECTIVE
(NFL)

By construction, every risk-taking action yields a strict
loss. The strategy is defeated.

The No-Free-Lunch (NFL) theorem of Wolpert and
Macready [6] provides a second paradigm. NFL explores
optimization over function spaces.

Theorem 3 (NFL, Wolpert–Macready). For any two
search algorithms A1 and A2, when performance is aver-
aged uniformly over all possible objective functions, they
perform identically.

If we map trading to sequential decision-making over
discrete price paths, NFL dictates that, averaged uni-
formly over the set of all possible market evolutions, no
strategy can beat the trivial zero-strategy.

Model Incompatibility: We explicitly note that
NFL is not a direct proof of market efficiency. Finan-
cial markets are not drawn from a uniform distribution;
they are continuous, highly structured, and governed by
no-arbitrage constraints. Furthermore, trading involves
sequential risk under a stochastic process, which is cate-
gorically different from offline function optimization.

However, NFL provides a vital philosophical bound-
ary: it proves that any trading strategy that works must
derive its edge purely from exploiting the non-uniform
structure of the specific physical measure P. A strategy
cannot be inherently, universally superior; it can only be
perfectly fitted to a specific, potentially transient, market
regime.

VI. THE COMPUTATIONAL PERSPECTIVE
(ADVERSARIAL DIAGONALIZATION)

Our third paradigm discards the assumption of exoge-
nous price processes entirely and asks: what if the market
is an adaptive adversary? If a trading strategy is fully

Shift in Quantifiers: This theorem does not prove that
a passive market will defeat you. Instead, it proves that
no strategy can be “universal” against reactive environ-
ments. The construction assumes perfect observability
and zero-latency response; real markets approximate this
adversarial ideal only partially. However, in modern high-
frequency trading, where algorithms routinely probe and
trade against the predictable logic of other algorithms,
this adversarial boundary is highly relevant.

Additionally, abstract computability theory (Rice’s
Theorem [12]) guarantees that, in the strictest formal
sense of program semantics, the precise set of trajectories
that will ruin a given complex algorithm is fundamentally
undecidable; no master program can consistently flag im-
pending failure sets in arbitrary trading code.

VII. ANALOGY TO SOCIAL CHOICE:
ARROW’S IMPOSSIBILITY THEOREM AND
FIXED POINTS

To contextualize the severity of this computational
boundary, it is instructive to draw a parallel to social
choice theory—specifically, Arrow’s Impossibility Theo-
rem [13]. Just as the Halting Problem establishes a defini-
tive boundary for computer science by proving that no
machine can perfectly evaluate every possible input, Ar-
row’s Theorem establishes a definitive boundary for polit-
ical science by proving that no rank-order voting system
can universally aggregate individual preferences into a
consistent group decision without violating basic fairness
criteria.

There is a profound structural parallel between Ar-
row’s Theorem and our adversarial diagonalization argu-
ment:

The “Universal” Claim: Arrow investigates a voting
rule that is expected to work flawlessly across all
possible preference profiles. We investigate a trad-
ing strategy that claims to yield strict profit across
all possible market paths.

The Conflict: Arrow highlights the inherent tension be-
tween individual logic and group consistency. Our
model highlights the tension between a fixed algo-
rithmic logic and a reactive market environment.

The Failure Mode: In social choice, the system breaks
down via the Condorcet Paradox and the lack
of transitivity—a cyclical loop where the system
cannot decisively rank outcomes.
In our com-
putational framework, the breakdown occurs via
diagonalization—an adversarially constructed path
where the strategy is mechanically forced into a los-
ing position.

The “Winner”: Arrow proves that the only universally
stable rule collapses to a Dictator. Correspondingly,
Theorem 4 demonstrates that the only guaranteed
victor is the Adversary, who simulates and system-
atically defeats the computable strategy.

At a deeper mathematical

level, as demonstrated
by Yanofsky [11], all such diagonal and self-referential
paradoxes—including Russell’s paradox, G¨odel’s incom-
pleteness, and Turing’s Halting Problem—share a uni-
versal structure rooted in the nonexistence of fixed points.
In our adversarial market construction, the market acts
as a “negation” operator on the strategy’s predictions
(moving the price down when the strategy buys, and up
when it sells). Because this adversarial response func-
tion inherently lacks a fixed point, a universal mapping
from trading algorithms to strictly profitable outcomes is
mathematically impossible.

VIII. CASE STUDY: THE WHEEL OPTIONS
STRATEGY

To ground these theoretical limits, we evaluate the
Wheel Strategy: selling a cash-secured put to collect pre-
It
mium, and upon assignment, selling a covered call.
is widely marketed as generating universal, directionless
income. Our paradigms predict specific failure modes.

A. Failure I: Catastrophic Drop (Time Asymmetry)

If the underlying stock crashes post-assignment, the
covered call loses value, and capital is trapped at a severe
loss. This trajectory is the exact time-reversal of a rapid,
profitable rally. By our heuristic (Criterion 1), failure on
the reversed path confirms the strategy is not universal.

4

B. Failure II: Slow Bleed (Combinatorial Cost)

A persistent downtrend yields premiums but suffers lin-
ear capital decay. The NFL paradigm reminds us that a
strategy structurally specialized to harvest variance pre-
mium in sideways markets mathematically must under-
perform in sustained directional regimes.

C. Failure III: Violent Breakout (Capped Upside)

A massive rally results in assignment at the call strike,
generating nominal profit but massive opportunity cost.
The strategy’s structurally capped upside ensures its ex-
pected value aligns with the martingale constraint (Corol-
lary 2) by forfeiting the tail events that offset the catas-
trophic drops (Failure I).

IX. FAPP STRATEGIES AND THEIR HONEST
SCOPE

While true universality is mathematically precluded,
strategies can be profitable for all practical purposes
(FAPP, following Bell [14]) relative to a specific physical
measure P.

Market-making earns the spread under normal volatil-
ity, and Kelly criterion investing maximizes logarithmic
growth under known distributions. However, their suc-
cess is strictly conditional on the stability of the under-
lying measure.

Lo’s Adaptive Markets Hypothesis

[15] perfectly
bridges the gap between our adversarial diagonalization
and real markets. As capital floods into a successful
FAPP strategy, the market ecosystem adapts. The mar-
ket transitions from a passive generator (Sec. III) toward
the adversarial generator (Sec. VI), eroding the strategy’s
edge and dynamically creating the exact trajectories nec-
essary to generate its failure set.

Crucially, the automated execution of algorithmic
FAPP strategies transforms individual theoretical lim-
its into systemic risks. Because failure sets are algorith-
mically unidentifiable (Rice’s Theorem), automated sys-
tems will blindly execute into structural breakdowns—
as witnessed during the 1987 portfolio-insurance cas-
cade [16] and the 2010 Flash Crash [17].

A mapping of these failure modes to their correspond-
ing paradigms is provided in Table II. We emphasize that
these mappings are illustrative rather than deductive.
The failure set Fσ of the Wheel Strategy is nonempty
(by Corollary 2), algorithmically unavoidable in advance
(by Theorem 4), visited with nonzero probability under
any continuous price process.

TABLE II. Canonical failure modes of the Wheel Strategy and the formal paradigms predicting their existence.

Failure
I: Crash
II: Bleed
III: Breakout
IV: Ruin

Trajectory Class
Reversed uptrend
Persistent drift µ < 0
Fat-tailed upside
Absorbing state

Predicted By
Criterion 1
NFL Thm.
Cor. 2
Thm. 4

Mechanism
Time asymmetry
Specialization
Capped upside
Undecidability

5

X. CONCLUSION

The impossibility of a universally winning trading
strategy is fundamentally a corollary of the absence of
arbitrage in continuous-time mathematical finance. Sub-
ject to standard admissibility constraints, any strategy
claiming pathwise universal success implies an arbitrage
opportunity, which is strictly precluded by the existence
of an equivalent martingale measure.

While the First Fundamental Theorem provides the
definitive proof, structurally analogous limitations ap-
pear across incompatible mathematical paradigms. From
a combinatorial perspective, the No-Free-Lunch theorem
emphasizes that trading success relies entirely on exploit-
ing non-uniform market structure. From a computational
perspective, Turing diagonalization guarantees that for
any computable algorithm there exists an adversarial en-
vironment that defeats it.

The thermodynamic analogy of Maxwell’s Demon pro-
vides an elegant conceptual mapping between martingale

measures and detailed balance, provided we reject phys-
ically irrelevant appeals to Landauer erasure costs. Ul-
timately, any strategy claimed to work ”in all market
conditions” must conceal an implicit assumption about
the underlying market measure, relying on excluded tail
risks that are mathematically guaranteed to exist.

ACKNOWLEDGMENTS

Noson S. Yanofsky has drawn my attention to notable
parallels between the present result and both Arrow’s Im-
possibility Theorem and diagonalization techniques, the
latter of which also imply the absence of fixed points.

This

research was

in part
Fund
(FWF)[Grant
by
the Austrian
DOI:10.55776/PIN5424624 ].
The author acknowl-
edges TU Wien Bibliothek for financial support through
its Open Access Funding Programme.

funded in whole or
Science

This text was partially created and revised with as-
sistance from large language models. All content, ideas,
and prompts were provided by the author.

[1] L. Bachelier, Th´eorie de la sp´eculation, Annales Scien-
tifiques de l’´Ecole Normale Sup´erieure 17, 21 (1900).
[2] P. A. Samuelson, Proof that properly anticipated prices
fluctuate randomly, Industrial Management Review 6, 41
(1965).

[3] E. F. Fama, Efficient capital markets: A review of the-
ory and empirical work, The Journal of Finance 25, 383
(1970).

[4] J. M. Harrison and S. R. Pliska, Martingales and stochas-
tic integrals in the theory of continuous trading, Stochas-
tic Processes and their Applications 11, 215 (1981).
[5] F. Delbaen and W. Schachermayer, A general version of
the fundamental theorem of asset pricing, Mathematische
Annalen 300, 463 (1994).

[6] D. H. Wolpert and W. G. Macready, No free lunch theo-
rems for optimization, IEEE Transactions on Evolution-
ary Computation 1, 67 (1997).

[7] A. M. Turing, On computable numbers, with an appli-
cation to the Entscheidungsproblem, Proceedings of the
London Mathematical Society, Series 2 42, 43, 230 (1936-
7 and 1937).

[8] R. Landauer, Irreversibility and heat generation in the
computing process, IBM Journal of Research and Devel-
opment 5, 183 (1961).

[9] C. H. Bennett, The thermodynamics of computation—a
review, International Journal of Theoretical Physics 21,
905 (1982).

[10] C. I. Isasa Mart´ın, Computability-Based Analysis of Mar-
ket Predictability, Trabajo de fin de m´aster, Universidad
Complutense de Madrid, Madrid, Spain (2022), m´aster
en M´etodos Formales en Ingenier´ıa Inform´atica. Tutor:
Ismael Rodr´ıguez Laguna. Score: 9.8/10.

[11] N. S. Yanofsky, A universal approach to self-referential
paradoxes, incompleteness and fixed points, Bulletin of
Symbolic Logic 9, 362 (2003), arXiv:math/0305282.
[12] H. G. Rice, Classes of recursively enumerable sets and
their decision problems, Transactions of the American
Mathematical Society 74, 358 (1953).

[13] K. J. Arrow, Social Choice and Individual Values, 3rd ed.
(Yale University Press, New Haven, CT, 2012) foreword
to the Third Edition by Eric S. Maskin.

[14] J. S. Bell, Against ‘measurement’, Physics World 3, 33

(1990).

[15] A. W. Lo, The adaptive markets hypothesis: Market effi-
ciency from an evolutionary perspective, The Journal of
Portfolio Management 30, 15 (2004).

[16] N. F. Brady, J. C. Cotting, R. G. Kirby, J. R. Opel, and
H. M. Stein, Report of the Presidential Task Force on

Market Mechanisms (U.S. Government Printing Office,
Washington, D.C., 1988).

flash crash: High-frequency trading in an electronic mar-
ket, The Journal of Finance 72, 967 (2017).

[17] A. Kirilenko, A. S. Kyle, M. Samadi, and T. Tuzun, The

6



---

## Content from Previous Extraction (not in markitdown output)

### Visual/Chart/Graph Descriptions

#### 1

Against a Universal Trading Strategy: No-Arbitrage, No-Free-Lunch, and Adversarial
Cantor Diagonalization
Karl Svozil 1
1 Institute for Theoretical Physics, TU Wien, Wiedner Hauptstrasse 8-10/136, A-1040 Vienna, Austria∗
(Dated: April 16, 2026)
Weinvestigatetheimpossibilityofuniversallywinningtradingstrategies—thosegeneratingstrict
profit across all market trajectories—through three distinct mathematical paradigms. Fundamen-
tally, under standard admissibility constraints, the existence of such a strategy is a strict subset
of strong arbitrage, which is mathematically precluded in competitive markets admitting an equiv-
alent martingale measure. Beyond this rigorous measure-theoretic foundation, we explore analo-
gouslimitationsintwoalternativemodelingregimes. Combinatorially, the No-Free-Lunchtheorem
demonstratesthatoutperformancerequiresexploitationofnon-uniformmarketstructure, asuniform
averagingprecludesuniversaldominance. Computationally, a Turingdiagonalizationargumentcon-
structs an adversarial environment that defeats any computable trading algorithm, shifting the
impossibility from exogenous price paths to adaptive adversaries. These mathematical limits are
framed by a time-reversal heuristic that establishes a formal analogy between financial martingale
measuresandthermodynamicdetailedbalance, resolvingthe Maxwell’s Demonanalogyformarkets
without relying on physically irrelevant Landauer erasure costs. Using the Wheel Options Strategy
asacasestudy, wedemonstratethatstrategiessucceeding“forallpracticalpurposes”(FAPP) inher-
ently depend on transient regime assumptions, meaning their automated execution systematically
amplifies tail risks.
I. INTRODUCTION 2. Combinatorial (Uniform Search, Sec. V).The
Wolpert–Macready No-Free-Lunch theorem [6] es-
The question of whether a trading strategy can al- tablishesthatunderauniformdistributionoverall
ways makemoney—regardlessofwhatthemarketdoes— possible discrete evolutions, no strategy uniformly
is older than mathematical finance itself. It animated dominates. While markets are not uniform, this
Bachelier’s 1900 thesis [1], was formalized by Samuel- boundaryhighlightsthatpracticaltradingisanex-
son[2]and Fama[3]intothe Efficient Market Hypothesis ercise in structure exploitation, not universal algo-
(EMH), and underlies every retail advertisement promis- rithmic superiority.
ing“passiveincomeinanymarketcondition.” The EMH
3. Computational (Adversarial, Sec. VI). Shift-
gives the standard economic answer: competitive pres-
ing the paradigm from exogenous price processes
sure eliminates consistent excess returns. But the EMH
to adaptive environments, we utilize the core diag-
is a statement about equilibrium outcomes across many
onalizationlogicofthe Halting Problem[7]toshow
agents; it is not, by itself, a mathematical proof that no
that no computable strategy can survive an adver-
individual strategy can ever win unconditionally.
sary capable of simulating it.
Defining a “universal strategy” as one that yields a
strictly positive terminal wealth across all valid market We frame these findings with a time-reversal criterion
trajectories is an exceptionally strong condition. Conse- (Sec. II) that motivates a formal analogy with Maxwell’s
quently, theintellectualburdenliesnotmerelyinproving Demon (Sec. IV). We treat this analogy carefully, not-
impossibility, butinpreciselydelineatingthemathemati- ingthatwhiletheequivalentmartingalemeasuremirrors
cal frameworks under which this impossibility manifests. thermodynamic detailed balance, appealing to physical
Thispaperarticulatesthisimpossibilitythroughthree Landauer erasure costs to explain financial dissipation is
distinct theoretical paradigms, carefully distinguishing quantitatively irrelevant. The Wheel Options Strategy
their incompatible underlying assumptions: is examined in Sec. VIII as a concrete illustration, and
the fragile scope of strategies operating for all practical
1. Financial (Measure-Theoretic, Sec. III). The purposes (FAPP) is discussed in Sec. IX.
rigorouscoreoftheimpossibility. Weshowthatun-
der standard admissibility constraints, a universal
strategy trivially constitutes an arbitrage opportu- II. THE TIME-REVERSAL HEURISTIC
nity. By the First Fundamental Theorem of Asset
Pricing [4, 5], this is precluded in markets admit-
To build intuition, we define a purely kinematic crite-
ting an equivalent martingale measure.
rion. Let M be a suitably rich space of strictly positive
paths P: [0, T]→Rn . The time reversal P(cid:101) of a path P
>0
is
∗ karl.svozil@tuwien.ac.at;http://tph.tuwien.ac.at/˜svozil P(cid:101)(t)=P(T −t), t∈[0, T]. (1)
6202
rp A
41
]RT.nif-q[
1 v43331.4062:vi Xra

### Additional Content

#### 1

2
Definition 1 (Pathwise Universal Strategy). Astrategy B. The Martingale Constraint
σ is universally successful if its realized profit-and-loss
satisfies Π[σ;P]>0 for all P ∈M.
By the Delbaen–Schachermayer generalization of the
Criterion 1 (Time-Reversal Necessity). If σ is univer- First Fundamental Theorem [4, 5], a market satisfies
sally successful on a path space closed under time rever- the condition of No Free Lunch with Vanishing Risk
sal, itmustbetime-reversalconsistent: forevery P ∈M (NFLVR) if and only if there exists an equivalent local
where Π[σ;P]>0, it must hold that Π[σ;P(cid:101)]>0. martingale measure (EMM) Q≈P.
Theoperationalcontentisclear: amomentumstrategy
Corollary 2 (No Universal Strategy). If an equivalent
profiting on a rising trajectory will mechanically fail on
martingale measure Q exists, no universally successful
its time-reversed (falling) counterpart.
admissible strategy can exist.
Remark 1 (Filtrations and Semimartingales). We explic-
itlystatethisasaheuristicbecause, instandardstochas-
This is the definitive mathematical constraint. Uni-
tic finance, the space of admissible trajectories is not
versality is simply an impossibly strong formulation of
trivially closed under time reversal. Time-reversing a
arbitrage; its preclusion is the foundational premise of
standardsemimartingalegenerallydestroysbothitssemi-
modern asset pricing. Thus, the non-existence of uni-
martingale property and its adaptedness to the forward
versal strategies is not a deep extension of asset pricing
filtration. Therefore, this criterion serves as a concep-
theory, but rather a direct corollary of its foundational
tualframingforstrictpathwiseclaims, whiletherigorous
no-arbitrage axiom.
probabilistic formulation is provided in Sec. III.
III. THE NO-ARBITRAGE FOUNDATION IV. THE MAXWELL’S DEMON ANALOGY:
HONEST BOUNDARIES
A. Admissibility and Arbitrage
A. Formal Analogy, Not Equivalence
Thedefiningproofofimpossibilitystemsfromthe First
Fundamental Theorem. We must first impose standard
Maxwell’s Demon observes individual gas molecules
regularity. Without admissibility constraints, pathwise
and sorts them to extract net work from thermal
gains can be trivially manufactured via doubling strate-
equilibrium—violatingthe Second Lawof Thermodynam-
gies (gambler’s ruin constructions), which are financially
ics. The physical resolution (Landauer, Bennett [8, 9])
meaningless due to infinite intermediate drawdown lim-
is that the Demon’s memory must be erased to operate
its.
cyclically, incurring an irreversible thermodynamic cost
Let the market be defined on a filtered probability
of k T ln2 per bit.
space (Ω, F,{F } , P). A strategy σ is admissi- B
t t∈[0, T] A universally successful trading strategy claims to act
ble if its wealth process is bounded from below almost
as a financial Maxwell’s Demon, extracting net profit
surely [5].
from market fluctuations in both temporal directions.
Theorem 1 (Universality implies Arbitrage). If there The conceptual mapping is highly instructive (Table I).
exists an admissible, self-financing strategy σ with zero
initial capital such that Π[σ;P(ω)] > 0 for all ω ∈ Ω,
then the market admits a strong arbitrage opportunity. TABLEI.Formalanalogybetween Maxwell’s Demonanduni-
versally winning trading strategies.
Proof. By assumption, the strategy yields strictly posi-
tive terminal wealth pathwise over all ω. Consequently, Maxwell’s Demon Universal Strategy
under the physical measure P, the terminal wealth is
Observes molecule velocities Observes price movements
strictly positive almost surely (P(Π > 0) = 1). An ad-
Sorts molecules for gradient Exploits movements for profit
missible, zero-cost, self-financing strategy ending strictly
positive almost surely strictly satisfies the condition for
Works in both time directions Profits on P and P(cid:101)
a strong arbitrage (arbitrage of the first kind). Extracts equilibrium work Extracts universal profit
Resolution: info. erasure cost Resolution: absence of EMM
Remark 2(Strength Hierarchy). Theconverse (Arbitrage
=⇒ Universal) is generally false. Our definition of uni-
versality demands pathwise strict positivity (Π > 0 for Thedeepestpointofcontactliesintheconceptofequi-
all ω), which is a strictly stronger condition than clas- librium. In statistical mechanics, true equilibrium satis-
sical arbitrage (which requires only P(Π ≥ 0) = 1 and fies detailed balance: microscopic processes equal their
P(Π > 0) > 0). Because pathwise universality trivially timereversals. Infinance, theequivalentmartingalemea-
implies strong arbitrage, its impossibility is mathemat- sure Q enforces the financial equivalent of detailed bal-
ically immediate once admissibility constraints are im- ance: there is no preferred drift direction under the pric-
posed under an equivalent martingale measure. ing measure.

#### 2

3
B. What the Analogy Does Not Establish automated, it acts as a computable Turing machine [10].
This guarantees its defeat via formal Turing diagonaliza-
We must aggressively limit the scope of this analogy. tion [7, 11].
It is a formal mapping, not a category-theoretic isomor- The Adversarial Market Constructioncanbemodelled
phism. as follows: Let M σ be a computable trading algorithm
Transaction costs are not the core mechanism. mappingdiscretehistories H t toatargetpositionw t ∈R.
While transaction costs mimic thermodynamic dissipa-
Theorem 4 (Defeatof Computable Strategies). Against
tion, a frictionless market with zero spread still mathe-
anycomputablestrategy M , thereexistsanadversarially
σ
matically precludes universal strategies via Corollary 2.
constructed, computable market trajectory P on which
adv
Landauer’s limit is irrelevant. The minimum en-
M realizes non-positive profit.
σ
ergy to erase one bit at room temperature is E ≈
2.87 × 10−21 J. The Landauer cost of an algorithmic Proof. Because M σ is a deterministic algorithm, a
trade is roughly 10−27 USD. Presenting thermodynamic market-making adversary with oracle access to the strat-
erasure as a financially relevant constraint on algorithms egy can simulate its next move. Set P 0 > 0. At each
is numerically dishonest. step t, the adversary simulates M σ (H t ) → w t and sets
The analogy illuminates why attempts to defeat effi- the next price opposing the strategy’s bet:
cient markets resemble attempts to build perpetual mo- 
P ·e−ϵ if w >0 (Strategy buys)
tionmachines, buttheindependentproofoffinancialim-  t t
possibility lies strictly in measure theory (Sec. III). P t+1 = P t ·e+ϵ if w t <0 (Strategy sells) (2)
P
if w =0 (Strategy neutral)
t t
V. THE COMBINATORIAL PERSPECTIVE By construction, every risk-taking action yields a strict
(NFL) loss. The strategy is defeated.
Shift in Quantifiers: This theorem does not prove that
The No-Free-Lunch (NFL) theorem of Wolpert and
a passive market will defeat you. Instead, it proves that
Macready [6] provides a second paradigm. NFL explores
no strategy can be “universal” against reactive environ-
optimization over function spaces.
ments. The construction assumes perfect observability
Theorem 3 (NFL, Wolpert–Macready). For any two andzero-latencyresponse;realmarketsapproximatethis
search algorithms A and A , when performance is aver- adversarialidealonlypartially. However, inmodernhigh-
1 2
aged uniformly over all possible objective functions, they frequencytrading, wherealgorithmsroutinelyprobeand
perform identically. trade against the predictable logic of other algorithms,
this adversarial boundary is highly relevant.
If we map trading to sequential decision-making over Additionally, abstract computability theory (Rice’s
discrete price paths, NFL dictates that, averaged uni- Theorem [12]) guarantees that, in the strictest formal
formly over the set of all possible market evolutions, no senseofprogramsemantics, theprecisesetoftrajectories
strategy can beat the trivial zero-strategy. thatwillruinagivencomplexalgorithmisfundamentally
Model Incompatibility: We explicitly note that undecidable; nomasterprogramcanconsistentlyflagim-
NFL is not a direct proof of market efficiency. Finan- pending failure sets in arbitrary trading code.
cial markets are not drawn from a uniform distribution;
they are continuous, highly structured, and governed by
no-arbitrage constraints. Furthermore, trading involves VII. ANALOGY TO SOCIAL CHOICE:
sequential risk under a stochastic process, which is cate- ARROW’S IMPOSSIBILITY THEOREM AND
gorically different from offline function optimization. FIXED POINTS
However, NFL provides a vital philosophical bound-
ary: it proves that any trading strategy that works must To contextualize the severity of this computational
derive its edge purely from exploiting the non-uniform boundary, it is instructive to draw a parallel to social
structure of the specific physical measure P. A strategy choice theory—specifically, Arrow’s Impossibility Theo-
cannot be inherently, universally superior; it can only be rem[13]. Justasthe Halting Problemestablishesadefini-
perfectlyfittedtoaspecific, potentiallytransient, market tive boundary for computer science by proving that no
regime. machine can perfectly evaluate every possible input, Ar-
row’s Theoremestablishesadefinitiveboundaryforpolit-
ical science by proving that no rank-order voting system
VI. THE COMPUTATIONAL PERSPECTIVE can universally aggregate individual preferences into a
(ADVERSARIAL DIAGONALIZATION) consistentgroupdecisionwithoutviolatingbasicfairness
criteria.
Our third paradigm discards the assumption of exoge- There is a profound structural parallel between Ar-
nouspriceprocessesentirelyandasks: whatifthemarket row’s Theorem and our adversarial diagonalization argu-
is an adaptive adversary? If a trading strategy is fully ment:

#### 3

4
The “Universal” Claim: Arrow investigates a voting B. Failure II: Slow Bleed (Combinatorial Cost)
rule that is expected to work flawlessly across all
possible preference profiles. We investigate a trad-
Apersistentdowntrendyieldspremiumsbutsufferslin-
ingstrategythatclaimstoyieldstrictprofitacross
ear capital decay. The NFL paradigm reminds us that a
all possible market paths.
strategy structurally specialized to harvest variance pre-
mium in sideways markets mathematically must under-
The Conflict: Arrowhighlightstheinherenttensionbe-
perform in sustained directional regimes.
tween individual logic and group consistency. Our
model highlights the tension between a fixed algo-
rithmic logic and a reactive market environment.
The Failure Mode: Insocialchoice, thesystembreaks C. Failure III: Violent Breakout (Capped Upside)
down via the Condorcet Paradox and the lack
of transitivity—a cyclical loop where the system
A massive rally results in assignment at the call strike,
cannot decisively rank outcomes. In our com-
generating nominal profit but massive opportunity cost.
putational framework, the breakdown occurs via
The strategy’s structurally capped upside ensures its ex-
diagonalization—anadversariallyconstructedpath
pectedvaluealignswiththemartingaleconstraint (Corol-
wherethestrategyismechanicallyforcedintoalos-
lary 2) by forfeiting the tail events that offset the catas-
ing position.
trophic drops (Failure I).
The “Winner”: Arrowprovesthattheonlyuniversally
stablerulecollapsestoa Dictator. Correspondingly,
Theorem 4 demonstrates that the only guaranteed
victor is the Adversary, who simulates and system- IX. FAPP STRATEGIES AND THEIR HONEST
SCOPE
atically defeats the computable strategy.
At a deeper mathematical level, as demonstrated While true universality is mathematically precluded,
by Yanofsky [11], all such diagonal and self-referential strategies can be profitable for all practical purposes
paradoxes—including Russell’s paradox, G¨odel’s incom- (FAPP, following Bell [14]) relative to a specific physical
pleteness, and Turing’s Halting Problem—share a uni- measure P.
versalstructurerootedinthenonexistenceoffixedpoints.
Market-making earns the spread under normal volatil-
In our adversarial market construction, the market acts
ity, and Kelly criterion investing maximizes logarithmic
as a “negation” operator on the strategy’s predictions
growth under known distributions. However, their suc-
(moving the price down when the strategy buys, and up
cess is strictly conditional on the stability of the under-
when it sells). Because this adversarial response func-
lying measure.
tion inherently lacks a fixed point, a universal mapping
fromtradingalgorithmstostrictlyprofitableoutcomesis Lo’s Adaptive Markets Hypothesis [15] perfectly
mathematically impossible. bridges the gap between our adversarial diagonalization
and real markets. As capital floods into a successful
FAPP strategy, the market ecosystem adapts. The mar-
kettransitionsfromapassivegenerator (Sec. III) toward
VIII. CASE STUDY: THE WHEEL OPTIONS
theadversarialgenerator (Sec. VI), erodingthestrategy’s
STRATEGY
edgeanddynamicallycreatingtheexacttrajectoriesnec-
essary to generate its failure set.
To ground these theoretical limits, we evaluate the
Crucially, the automated execution of algorithmic
Wheel Strategy: sellingacash-securedputtocollectpre-
FAPP strategies transforms individual theoretical lim-
mium, and upon assignment, selling a covered call. It
its into systemic risks. Because failure sets are algorith-
is widely marketed as generating universal, directionless
mically unidentifiable (Rice’s Theorem), automated sys-
income. Our paradigms predict specific failure modes.
tems will blindly execute into structural breakdowns—
as witnessed during the 1987 portfolio-insurance cas-
cade [16] and the 2010 Flash Crash [17].
A. Failure I: Catastrophic Drop (Time Asymmetry) A mapping of these failure modes to their correspond-
ingparadigmsisprovidedin Table II.Weemphasizethat
If the underlying stock crashes post-assignment, the these mappings are illustrative rather than deductive.
coveredcalllosesvalue, andcapitalistrappedatasevere The failure set F of the Wheel Strategy is nonempty
σ
loss. Thistrajectoryistheexacttime-reversalofarapid, (by Corollary 2), algorithmically unavoidable in advance
profitablerally. Byourheuristic (Criterion1), failureon (by Theorem 4), visited with nonzero probability under
the reversed path confirms the strategy is not universal. any continuous price process.

#### 4

5
TABLE II. Canonical failure modes of the Wheel Strategy and the formal paradigms predicting their existence.
Failure Trajectory Class Predicted By Mechanism
I: Crash Reversed uptrend Criterion 1 Time asymmetry
II: Bleed Persistent drift µ<0 NFL Thm. Specialization
III: Breakout Fat-tailed upside Cor. 2 Capped upside
IV: Ruin Absorbing state Thm. 4 Undecidability
X. CONCLUSION measures and detailed balance, provided we reject phys-
ically irrelevant appeals to Landauer erasure costs. Ul-
timately, any strategy claimed to work ”in all market
The impossibility of a universally winning trading conditions” must conceal an implicit assumption about
strategy is fundamentally a corollary of the absence of the underlying market measure, relying on excluded tail
arbitrage in continuous-time mathematical finance. Sub- risks that are mathematically guaranteed to exist.
ject to standard admissibility constraints, any strategy
claiming pathwise universal success implies an arbitrage
opportunity, which is strictly precluded by the existence ACKNOWLEDGMENTS
of an equivalent martingale measure.
Noson S. Yanofsky has drawn my attention to notable
While the First Fundamental Theorem provides the
parallelsbetweenthepresentresultandboth Arrow’s Im-
definitive proof, structurally analogous limitations ap-
possibility Theorem and diagonalization techniques, the
pearacrossincompatiblemathematicalparadigms. From
latter of which also imply the absence of fixed points.
a combinatorial perspective, the No-Free-Lunch theorem
This research was funded in whole or in part
emphasizesthattradingsuccessreliesentirelyonexploit-
by the Austrian Science Fund (FWF)[Grant
ingnon-uniformmarketstructure. Fromacomputational
DOI:10.55776/PIN5424624]. The author acknowl-
perspective, Turing diagonalization guarantees that for
edges TU Wien Bibliothek for financial support through
any computable algorithm there exists an adversarial en-
its Open Access Funding Programme.
vironment that defeats it.
This text was partially created and revised with as-
The thermodynamic analogy of Maxwell’s Demon pro- sistance from large language models. All content, ideas,
videsanelegantconceptualmappingbetweenmartingale and prompts were provided by the author.
[1] L. Bachelier, Th´eorie de la sp´eculation, Annales Scien- [9] C. H. Bennett, The thermodynamics of computation—a
tifiques de l’E´cole Normale Sup´erieure 17, 21 (1900). review, International Journal of Theoretical Physics 21,
[2] P. A. Samuelson, Proof that properly anticipated prices 905 (1982).
fluctuaterandomly, Industrial Management Review6,41 [10] C.I.Isasa Mart´ın, Computability-Based Analysisof Mar-
(1965). ket Predictability, Trabajo de fin de ma´ster, Universidad
[3] E. F. Fama, Efficient capital markets: A review of the- Complutense de Madrid, Madrid, Spain (2022), ma´ster
ory and empirical work, The Journal of Finance 25, 383 en M´etodos Formales en Ingenier´ıa Informa´tica. Tutor:
(1970). Ismael Rodr´ıguez Laguna. Score: 9.8/10.
[4] J.M.Harrisonand S.R.Pliska, Martingalesandstochas- [11] N. S. Yanofsky, A universal approach to self-referential
ticintegralsinthetheoryofcontinuoustrading, Stochas- paradoxes, incompleteness and fixed points, Bulletin of
tic Processes and their Applications 11, 215 (1981). Symbolic Logic 9, 362 (2003), ar Xiv:math/0305282.
[5] F. Delbaen and W. Schachermayer, A general version of [12] H. G. Rice, Classes of recursively enumerable sets and
thefundamentaltheoremofassetpricing, Mathematische their decision problems, Transactions of the American
Annalen 300, 463 (1994). Mathematical Society 74, 358 (1953).
[6] D. H. Wolpert and W. G. Macready, No free lunch theo- [13] K.J.Arrow, Social Choiceand Individual Values,3 rded.
rems for optimization, IEEE Transactions on Evolution- (Yale University Press, New Haven, CT, 2012) foreword
ary Computation 1, 67 (1997). to the Third Edition by Eric S. Maskin.
[7] A. M. Turing, On computable numbers, with an appli- [14] J. S. Bell, Against ‘measurement’, Physics World 3, 33
cation to the Entscheidungsproblem, Proceedings of the (1990).
London Mathematical Society, Series242,43,230(1936- [15] A.W.Lo, Theadaptivemarketshypothesis: Marketeffi-
7 and 1937). ciency from an evolutionary perspective, The Journal of
[8] R. Landauer, Irreversibility and heat generation in the Portfolio Management 30, 15 (2004).
computing process, IBM Journal of Research and Devel- [16] N.F.Brady, J.C.Cotting, R.G.Kirby, J.R.Opel, and
opment 5, 183 (1961). H. M. Stein, Report of the Presidential Task Force on

#### 5

6
Market Mechanisms (U.S. Government Printing Office, flashcrash: High-frequencytradinginanelectronicmar-
Washington, D.C., 1988). ket, The Journal of Finance 72, 967 (2017).
[17] A.Kirilenko, A.S.Kyle, M.Samadi, and T.Tuzun, The


---

## Content Unique to Old Extraction (not found in markitdown output)

### 1. # Against a Universal Trading Strategy- No-Arbitrage, No-Free-Lunch, and Adversa...

# Against a Universal Trading Strategy- No-Arbitrage, No-Free-Lunch, and Adversarial Cantor Diagonalization

### 2. > *Source PDF: Against a Universal Trading Strategy- No-Arbitrage, No-Free-Lunch...

> *Source PDF: Against a Universal Trading Strategy- No-Arbitrage, No-Free-Lunch, and Adversarial Cantor Diagonalization.pdf*
> *Extraction: Combined — markitdown raw text (base) + previous extraction supplements*

### 3. ## Content from Previous Extraction (not in markitdown output)...

## Content from Previous Extraction (not in markitdown output)

### 4. ### Visual/Chart/Graph Descriptions...

### Visual/Chart/Graph Descriptions

### 5. Against a Universal Trading Strategy: No-Arbitrage, No-Free-Lunch, and Adversari...

Against a Universal Trading Strategy: No-Arbitrage, No-Free-Lunch, and Adversarial
Cantor Diagonalization
Karl Svozil 1
1 Institute for Theoretical Physics, TU Wien, Wiedner Hauptstrasse 8-10/136, A-1040 Vienna, Austria∗
(Dated: April 16, 2026)
Weinvestigatetheimpossibilityofuniversallywinningtradingstrategies—thosegeneratingstrict
profit across all market trajectories—through three distinct mathematical paradigms. Fundamen-
tally, under standard admissibility constraints, the existence of such a strategy is a strict subset
of strong arbitrage, which is mathematically precluded in competitive markets admitting an equiv-
alent martingale measure. Beyond this rigorous measure-theoretic foundation, we explore analo-
gouslimitationsintwoalternativemodelingregimes. Combinatorially, the No-Free-Lunchtheorem
demonstratesthatoutperformancerequiresexploitationofnon-uniformmarketstructure, asuniform
averagingprecludesuniversaldominance. Computationally, a Turingdiagonalizationargumentcon-
structs an adversarial environment that defeats any computable trading algorithm, shifting the
impossibility from exogenous price paths to adaptive adversaries. These mathematical limits are
framed by a time-reversal heuristic that establishes a formal analogy between financial martingale
measuresandthermodynamicdetailedbalance, resolvingthe Maxwell’s Demonanalogyformarkets
without relying on physically irrelevant Landauer erasure costs. Using the Wheel Options Strategy
asacasestudy, wedemonstratethatstrategiessucceeding“forallpracticalpurposes”(FAPP) inher-
ently depend on transient regime assumptions, meaning their automated execution systematically
amplifies tail risks.
I. INTRODUCTION 2. Combinatorial (Uniform Search, Sec. V).The
Wolpert–Macready No-Free-Lunch theorem [6] es-
The question of whether a trading strategy can al- tablishesthatunderauniformdistributionoverall
ways makemoney—regardlessofwhatthemarketdoes— possible discrete evolutions, no strategy uniformly
is older than mathematical finance itself. It animated dominates. While markets are not uniform, this
Bachelier’s 1900 thesis [1], was formalized by Samuel- boundaryhighlightsthatpracticaltradingisanex-
son[2]and Fama[3]intothe Efficient Market Hypothesis ercise in structure exploitation, not universal algo-
(EMH), and underlies every retail advertisement promis- rithmic superiority.
ing“passiveincomeinanymarketcondition.” The EMH
3. Computational (Adversarial, Sec. VI). Shift-
gives the standard economic answer: competitive pres-
ing the paradigm from exogenous price processes
sure eliminates consistent excess returns. But the EMH
to adaptive environments, we utilize the core diag-
is a statement about equilibrium outcomes across many
onalizationlogicofthe Halting Problem[7]toshow
agents; it is not, by itself, a mathematical proof that no
that no computable strategy can survive an adver-
individual strategy can ever win unconditionally.
sary capable of simulating it.
Defining a “universal strategy” as one that yields a
strictly positive terminal wealth across all valid market We frame these findings with a time-reversal criterion
trajectories is an exceptionally strong condition. Conse- (Sec. II) that motivates a formal analogy with Maxwell’s
quently, theintellectualburdenliesnotmerelyinproving Demon (Sec. IV). We treat this analogy carefully, not-
impossibility, butinpreciselydelineatingthemathemati- ingthatwhiletheequivalentmartingalemeasuremirrors
cal frameworks under which this impossibility manifests. thermodynamic detailed balance, appealing to physical
Thispaperarticulatesthisimpossibilitythroughthree Landauer erasure costs to explain financial dissipation is
distinct theoretical paradigms, carefully distinguishing quantitatively irrelevant. The Wheel Options Strategy
their incompatible underlying assumptions: is examined in Sec. VIII as a concrete illustration, and
the fragile scope of strategies operating for all practical
1. Financial (Measure-Theoretic, Sec. III). The purposes (FAPP) is discussed in Sec. IX.
rigorouscoreoftheimpossibility. Weshowthatun-
der standard admissibility constraints, a universal
strategy trivially constitutes an arbitrage opportu- II. THE TIME-REVERSAL HEURISTIC
nity. By the First Fundamental Theorem of Asset
Pricing [4, 5], this is precluded in markets admit-
To build intuition, we define a purely kinematic crite-
ting an equivalent martingale measure.
rion. Let M be a suitably rich space of strictly positive
paths P: [0, T]→Rn . The time reversal P(cid:101) of a path P
>0
is
∗ karl.svozil@tuwien.ac.at;http://tph.tuwien.ac.at/˜svozil P(cid:101)(t)=P(T −t), t∈[0, T]. (1)
6202
rp A
41
]RT.nif-q[
1 v43331.4062:vi Xra

### 6. 2
Definition 1 (Pathwise Universal Strategy). Astrategy B. The Martingale Constr...

2
Definition 1 (Pathwise Universal Strategy). Astrategy B. The Martingale Constraint
σ is universally successful if its realized profit-and-loss
satisfies Π[σ;P]>0 for all P ∈M.
By the Delbaen–Schachermayer generalization of the
Criterion 1 (Time-Reversal Necessity). If σ is univer- First Fundamental Theorem [4, 5], a market satisfies
sally successful on a path space closed under time rever- the condition of No Free Lunch with Vanishing Risk
sal, itmustbetime-reversalconsistent: forevery P ∈M (NFLVR) if and only if there exists an equivalent local
where Π[σ;P]>0, it must hold that Π[σ;P(cid:101)]>0. martingale measure (EMM) Q≈P.
Theoperationalcontentisclear: amomentumstrategy
Corollary 2 (No Universal Strategy). If an equivalent
profiting on a rising trajectory will mechanically fail on
martingale measure Q exists, no universally successful
its time-reversed (falling) counterpart.
admissible strategy can exist.
Remark 1 (Filtrations and Semimartingales). We explic-
itlystatethisasaheuristicbecause, instandardstochas-
This is the definitive mathematical constraint. Uni-
tic finance, the space of admissible trajectories is not
versality is simply an impossibly strong formulation of
trivially closed under time reversal. Time-reversing a
arbitrage; its preclusion is the foundational premise of
standardsemimartingalegenerallydestroysbothitssemi-
modern asset pricing. Thus, the non-existence of uni-
martingale property and its adaptedness to the forward
versal strategies is not a deep extension of asset pricing
filtration. Therefore, this criterion serves as a concep-
theory, but rather a direct corollary of its foundational
tualframingforstrictpathwiseclaims, whiletherigorous
no-arbitrage axiom.
probabilistic formulation is provided in Sec. III.
III. THE NO-ARBITRAGE FOUNDATION IV. THE MAXWELL’S DEMON ANALOGY:
HONEST BOUNDARIES
A. Admissibility and Arbitrage
A. Formal Analogy, Not Equivalence
Thedefiningproofofimpossibilitystemsfromthe First
Fundamental Theorem. We must first impose standard
Maxwell’s Demon observes individual gas molecules
regularity. Without admissibility constraints, pathwise
and sorts them to extract net work from thermal
gains can be trivially manufactured via doubling strate-
equilibrium—violatingthe Second Lawof Thermodynam-
gies (gambler’s ruin constructions), which are financially
ics. The physical resolution (Landauer, Bennett [8, 9])
meaningless due to infinite intermediate drawdown lim-
is that the Demon’s memory must be erased to operate
its.
cyclically, incurring an irreversible thermodynamic cost
Let the market be defined on a filtered probability
of k T ln2 per bit.
space (Ω, F,{F } , P). A strategy σ is admissi- B
t t∈[0, T] A universally successful trading strategy claims to act
ble if its wealth process is bounded from below almost
as a financial Maxwell’s Demon, extracting net profit
surely [5].
from market fluctuations in both temporal directions.
Theorem 1 (Universality implies Arbitrage). If there The conceptual mapping is highly instructive (Table I).
exists an admissible, self-financing strategy σ with zero
initial capital such that Π[σ;P(ω)] > 0 for all ω ∈ Ω,
then the market admits a strong arbitrage opportunity. TABLEI.Formalanalogybetween Maxwell’s Demonanduni-
versally winning trading strategies.
Proof. By assumption, the strategy yields strictly posi-
tive terminal wealth pathwise over all ω. Consequently, Maxwell’s Demon Universal Strategy
under the physical measure P, the terminal wealth is
Observes molecule velocities Observes price movements
strictly positive almost surely (P(Π > 0) = 1). An ad-
Sorts molecules for gradient Exploits movements for profit
missible, zero-cost, self-financing strategy ending strictly
positive almost surely strictly satisfies the condition for
Works in both time directions Profits on P and P(cid:101)
a strong arbitrage (arbitrage of the first kind). Extracts equilibrium work Extracts universal profit
Resolution: info. erasure cost Resolution: absence of EMM
Remark 2(Strength Hierarchy). Theconverse (Arbitrage
=⇒ Universal) is generally false. Our definition of uni-
versality demands pathwise strict positivity (Π > 0 for Thedeepestpointofcontactliesintheconceptofequi-
all ω), which is a strictly stronger condition than clas- librium. In statistical mechanics, true equilibrium satis-
sical arbitrage (which requires only P(Π ≥ 0) = 1 and fies detailed balance: microscopic processes equal their
P(Π > 0) > 0). Because pathwise universality trivially timereversals. Infinance, theequivalentmartingalemea-
implies strong arbitrage, its impossibility is mathemat- sure Q enforces the financial equivalent of detailed bal-
ically immediate once admissibility constraints are im- ance: there is no preferred drift direction under the pric-
posed under an equivalent martingale measure. ing measure.

### 7. 3
B. What the Analogy Does Not Establish automated, it acts as a computable Turi...

3
B. What the Analogy Does Not Establish automated, it acts as a computable Turing machine [10].
This guarantees its defeat via formal Turing diagonaliza-
We must aggressively limit the scope of this analogy. tion [7, 11].
It is a formal mapping, not a category-theoretic isomor- The Adversarial Market Constructioncanbemodelled
phism. as follows: Let M σ be a computable trading algorithm
Transaction costs are not the core mechanism. mappingdiscretehistories H t toatargetpositionw t ∈R.
While transaction costs mimic thermodynamic dissipa-
Theorem 4 (Defeatof Computable Strategies). Against
tion, a frictionless market with zero spread still mathe-
anycomputablestrategy M , thereexistsanadversarially
σ
matically precludes universal strategies via Corollary 2.
constructed, computable market trajectory P on which
adv
Landauer’s limit is irrelevant. The minimum en-
M realizes non-positive profit.
σ
ergy to erase one bit at room temperature is E ≈
2.87 × 10−21 J. The Landauer cost of an algorithmic Proof. Because M σ is a deterministic algorithm, a
trade is roughly 10−27 USD. Presenting thermodynamic market-making adversary with oracle access to the strat-
erasure as a financially relevant constraint on algorithms egy can simulate its next move. Set P 0 > 0. At each
is numerically dishonest. step t, the adversary simulates M σ (H t ) → w t and sets
The analogy illuminates why attempts to defeat effi- the next price opposing the strategy’s bet:
cient markets resemble attempts to build perpetual mo- 
P ·e−ϵ if w >0 (Strategy buys)
tionmachines, buttheindependentproofoffinancialim-  t t
possibility lies strictly in measure theory (Sec. III). P t+1 = P t ·e+ϵ if w t <0 (Strategy sells) (2)
P
if w =0 (Strategy neutral)
t t
V. THE COMBINATORIAL PERSPECTIVE By construction, every risk-taking action yields a strict
(NFL) loss. The strategy is defeated.
Shift in Quantifiers: This theorem does not prove that
The No-Free-Lunch (NFL) theorem of Wolpert and
a passive market will defeat you. Instead, it proves that
Macready [6] provides a second paradigm. NFL explores
no strategy can be “universal” against reactive environ-
optimization over function spaces.
ments. The construction assumes perfect observability
Theorem 3 (NFL, Wolpert–Macready). For any two andzero-latencyresponse;realmarketsapproximatethis
search algorithms A and A , when performance is aver- adversarialidealonlypartially. However, inmodernhigh-
1 2
aged uniformly over all possible objective functions, they frequencytrading, wherealgorithmsroutinelyprobeand
perform identically. trade against the predictable logic of other algorithms,
this adversarial boundary is highly relevant.
If we map trading to sequential decision-making over Additionally, abstract computability theory (Rice’s
discrete price paths, NFL dictates that, averaged uni- Theorem [12]) guarantees that, in the strictest formal
formly over the set of all possible market evolutions, no senseofprogramsemantics, theprecisesetoftrajectories
strategy can beat the trivial zero-strategy. thatwillruinagivencomplexalgorithmisfundamentally
Model Incompatibility: We explicitly note that undecidable; nomasterprogramcanconsistentlyflagim-
NFL is not a direct proof of market efficiency. Finan- pending failure sets in arbitrary trading code.
cial markets are not drawn from a uniform distribution;
they are continuous, highly structured, and governed by
no-arbitrage constraints. Furthermore, trading involves VII. ANALOGY TO SOCIAL CHOICE:
sequential risk under a stochastic process, which is cate- ARROW’S IMPOSSIBILITY THEOREM AND
gorically different from offline function optimization. FIXED POINTS
However, NFL provides a vital philosophical bound-
ary: it proves that any trading strategy that works must To contextualize the severity of this computational
derive its edge purely from exploiting the non-uniform boundary, it is instructive to draw a parallel to social
structure of the specific physical measure P. A strategy choice theory—specifically, Arrow’s Impossibility Theo-
cannot be inherently, universally superior; it can only be rem[13]. Justasthe Halting Problemestablishesadefini-
perfectlyfittedtoaspecific, potentiallytransient, market tive boundary for computer science by proving that no
regime. machine can perfectly evaluate every possible input, Ar-
row’s Theoremestablishesadefinitiveboundaryforpolit-
ical science by proving that no rank-order voting system
VI. THE COMPUTATIONAL PERSPECTIVE can universally aggregate individual preferences into a
(ADVERSARIAL DIAGONALIZATION) consistentgroupdecisionwithoutviolatingbasicfairness
criteria.
Our third paradigm discards the assumption of exoge- There is a profound structural parallel between Ar-
nouspriceprocessesentirelyandasks: whatifthemarket row’s Theorem and our adversarial diagonalization argu-
is an adaptive adversary? If a trading strategy is fully ment:

### 8. 4
The “Universal” Claim: Arrow investigates a voting B. Failure II: Slow Bleed (...

4
The “Universal” Claim: Arrow investigates a voting B. Failure II: Slow Bleed (Combinatorial Cost)
rule that is expected to work flawlessly across all
possible preference profiles. We investigate a trad-
Apersistentdowntrendyieldspremiumsbutsufferslin-
ingstrategythatclaimstoyieldstrictprofitacross
ear capital decay. The NFL paradigm reminds us that a
all possible market paths.
strategy structurally specialized to harvest variance pre-
mium in sideways markets mathematically must under-
The Conflict: Arrowhighlightstheinherenttensionbe-
perform in sustained directional regimes.
tween individual logic and group consistency. Our
model highlights the tension between a fixed algo-
rithmic logic and a reactive market environment.
The Failure Mode: Insocialchoice, thesystembreaks C. Failure III: Violent Breakout (Capped Upside)
down via the Condorcet Paradox and the lack
of transitivity—a cyclical loop where the system
A massive rally results in assignment at the call strike,
cannot decisively rank outcomes. In our com-
generating nominal profit but massive opportunity cost.
putational framework, the breakdown occurs via
The strategy’s structurally capped upside ensures its ex-
diagonalization—anadversariallyconstructedpath
pectedvaluealignswiththemartingaleconstraint (Corol-
wherethestrategyismechanicallyforcedintoalos-
lary 2) by forfeiting the tail events that offset the catas-
ing position.
trophic drops (Failure I).
The “Winner”: Arrowprovesthattheonlyuniversally
stablerulecollapsestoa Dictator. Correspondingly,
Theorem 4 demonstrates that the only guaranteed
victor is the Adversary, who simulates and system- IX. FAPP STRATEGIES AND THEIR HONEST
SCOPE
atically defeats the computable strategy.
At a deeper mathematical level, as demonstrated While true universality is mathematically precluded,
by Yanofsky [11], all such diagonal and self-referential strategies can be profitable for all practical purposes
paradoxes—including Russell’s paradox, G¨odel’s incom- (FAPP, following Bell [14]) relative to a specific physical
pleteness, and Turing’s Halting Problem—share a uni- measure P.
versalstructurerootedinthenonexistenceoffixedpoints.
Market-making earns the spread under normal volatil-
In our adversarial market construction, the market acts
ity, and Kelly criterion investing maximizes logarithmic
as a “negation” operator on the strategy’s predictions
growth under known distributions. However, their suc-
(moving the price down when the strategy buys, and up
cess is strictly conditional on the stability of the under-
when it sells). Because this adversarial response func-
lying measure.
tion inherently lacks a fixed point, a universal mapping
fromtradingalgorithmstostrictlyprofitableoutcomesis Lo’s Adaptive Markets Hypothesis [15] perfectly
mathematically impossible. bridges the gap between our adversarial diagonalization
and real markets. As capital floods into a successful
FAPP strategy, the market ecosystem adapts. The mar-
kettransitionsfromapassivegenerator (Sec. III) toward
VIII. CASE STUDY: THE WHEEL OPTIONS
theadversarialgenerator (Sec. VI), erodingthestrategy’s
STRATEGY
edgeanddynamicallycreatingtheexacttrajectoriesnec-
essary to generate its failure set.
To ground these theoretical limits, we evaluate the
Crucially, the automated execution of algorithmic
Wheel Strategy: sellingacash-securedputtocollectpre-
FAPP strategies transforms individual theoretical lim-
mium, and upon assignment, selling a covered call. It
its into systemic risks. Because failure sets are algorith-
is widely marketed as generating universal, directionless
mically unidentifiable (Rice’s Theorem), automated sys-
income. Our paradigms predict specific failure modes.
tems will blindly execute into structural breakdowns—
as witnessed during the 1987 portfolio-insurance cas-
cade [16] and the 2010 Flash Crash [17].
A. Failure I: Catastrophic Drop (Time Asymmetry) A mapping of these failure modes to their correspond-
ingparadigmsisprovidedin Table II.Weemphasizethat
If the underlying stock crashes post-assignment, the these mappings are illustrative rather than deductive.
coveredcalllosesvalue, andcapitalistrappedatasevere The failure set F of the Wheel Strategy is nonempty
σ
loss. Thistrajectoryistheexacttime-reversalofarapid, (by Corollary 2), algorithmically unavoidable in advance
profitablerally. Byourheuristic (Criterion1), failureon (by Theorem 4), visited with nonzero probability under
the reversed path confirms the strategy is not universal. any continuous price process.

### 9. 5
TABLE II. Canonical failure modes of the Wheel Strategy and the formal paradig...

5
TABLE II. Canonical failure modes of the Wheel Strategy and the formal paradigms predicting their existence.
Failure Trajectory Class Predicted By Mechanism
I: Crash Reversed uptrend Criterion 1 Time asymmetry
II: Bleed Persistent drift µ<0 NFL Thm. Specialization
III: Breakout Fat-tailed upside Cor. 2 Capped upside
IV: Ruin Absorbing state Thm. 4 Undecidability
X. CONCLUSION measures and detailed balance, provided we reject phys-
ically irrelevant appeals to Landauer erasure costs. Ul-
timately, any strategy claimed to work ”in all market
The impossibility of a universally winning trading conditions” must conceal an implicit assumption about
strategy is fundamentally a corollary of the absence of the underlying market measure, relying on excluded tail
arbitrage in continuous-time mathematical finance. Sub- risks that are mathematically guaranteed to exist.
ject to standard admissibility constraints, any strategy
claiming pathwise universal success implies an arbitrage
opportunity, which is strictly precluded by the existence ACKNOWLEDGMENTS
of an equivalent martingale measure.
Noson S. Yanofsky has drawn my attention to notable
While the First Fundamental Theorem provides the
parallelsbetweenthepresentresultandboth Arrow’s Im-
definitive proof, structurally analogous limitations ap-
possibility Theorem and diagonalization techniques, the
pearacrossincompatiblemathematicalparadigms. From
latter of which also imply the absence of fixed points.
a combinatorial perspective, the No-Free-Lunch theorem
This research was funded in whole or in part
emphasizesthattradingsuccessreliesentirelyonexploit-
by the Austrian Science Fund (FWF)[Grant
ingnon-uniformmarketstructure. Fromacomputational
DOI:10.55776/PIN5424624]. The author acknowl-
perspective, Turing diagonalization guarantees that for
edges TU Wien Bibliothek for financial support through
any computable algorithm there exists an adversarial en-
its Open Access Funding Programme.
vironment that defeats it.
This text was partially created and revised with as-
The thermodynamic analogy of Maxwell’s Demon pro- sistance from large language models. All content, ideas,
videsanelegantconceptualmappingbetweenmartingale and prompts were provided by the author.
[1] L. Bachelier, Th´eorie de la sp´eculation, Annales Scien- [9] C. H. Bennett, The thermodynamics of computation—a
tifiques de l’E´cole Normale Sup´erieure 17, 21 (1900). review, International Journal of Theoretical Physics 21,
[2] P. A. Samuelson, Proof that properly anticipated prices 905 (1982).
fluctuaterandomly, Industrial Management Review6,41 [10] C.I.Isasa Mart´ın, Computability-Based Analysisof Mar-
(1965). ket Predictability, Trabajo de fin de ma´ster, Universidad
[3] E. F. Fama, Efficient capital markets: A review of the- Complutense de Madrid, Madrid, Spain (2022), ma´ster
ory and empirical work, The Journal of Finance 25, 383 en M´etodos Formales en Ingenier´ıa Informa´tica. Tutor:
(1970). Ismael Rodr´ıguez Laguna. Score: 9.8/10.
[4] J.M.Harrisonand S.R.Pliska, Martingalesandstochas- [11] N. S. Yanofsky, A universal approach to self-referential
ticintegralsinthetheoryofcontinuoustrading, Stochas- paradoxes, incompleteness and fixed points, Bulletin of
tic Processes and their Applications 11, 215 (1981). Symbolic Logic 9, 362 (2003), ar Xiv:math/0305282.
[5] F. Delbaen and W. Schachermayer, A general version of [12] H. G. Rice, Classes of recursively enumerable sets and
thefundamentaltheoremofassetpricing, Mathematische their decision problems, Transactions of the American
Annalen 300, 463 (1994). Mathematical Society 74, 358 (1953).
[6] D. H. Wolpert and W. G. Macready, No free lunch theo- [13] K.J.Arrow, Social Choiceand Individual Values,3 rded.
rems for optimization, IEEE Transactions on Evolution- (Yale University Press, New Haven, CT, 2012) foreword
ary Computation 1, 67 (1997). to the Third Edition by Eric S. Maskin.
[7] A. M. Turing, On computable numbers, with an appli- [14] J. S. Bell, Against ‘measurement’, Physics World 3, 33
cation to the Entscheidungsproblem, Proceedings of the (1990).
London Mathematical Society, Series242,43,230(1936- [15] A.W.Lo, Theadaptivemarketshypothesis: Marketeffi-
7 and 1937). ciency from an evolutionary perspective, The Journal of
[8] R. Landauer, Irreversibility and heat generation in the Portfolio Management 30, 15 (2004).
computing process, IBM Journal of Research and Devel- [16] N.F.Brady, J.C.Cotting, R.G.Kirby, J.R.Opel, and
opment 5, 183 (1961). H. M. Stein, Report of the Presidential Task Force on

### 10. 6
Market Mechanisms (U.S. Government Printing Office, flashcrash: High-frequency...

6
Market Mechanisms (U.S. Government Printing Office, flashcrash: High-frequencytradinginanelectronicmar-
Washington, D.C., 1988). ket, The Journal of Finance 72, 967 (2017).
[17] A.Kirilenko, A.S.Kyle, M.Samadi, and T.Tuzun, The


---

## Raw Markitdown Extraction (full text)

Against a Universal Trading Strategy: No-Arbitrage, No-Free-Lunch, and Adversarial
Cantor Diagonalization

1Institute for Theoretical Physics, TU Wien, Wiedner Hauptstrasse 8-10/136, A-1040 Vienna, Austria∗
(Dated: April 16, 2026)

Karl Svozil

1

We investigate the impossibility of universally winning trading strategies—those generating strict
profit across all market trajectories—through three distinct mathematical paradigms. Fundamen-
tally, under standard admissibility constraints, the existence of such a strategy is a strict subset
of strong arbitrage, which is mathematically precluded in competitive markets admitting an equiv-
alent martingale measure. Beyond this rigorous measure-theoretic foundation, we explore analo-
gous limitations in two alternative modeling regimes. Combinatorially, the No-Free-Lunch theorem
demonstrates that outperformance requires exploitation of non-uniform market structure, as uniform
averaging precludes universal dominance. Computationally, a Turing diagonalization argument con-
structs an adversarial environment that defeats any computable trading algorithm, shifting the
impossibility from exogenous price paths to adaptive adversaries. These mathematical limits are
framed by a time-reversal heuristic that establishes a formal analogy between financial martingale
measures and thermodynamic detailed balance, resolving the Maxwell’s Demon analogy for markets
without relying on physically irrelevant Landauer erasure costs. Using the Wheel Options Strategy
as a case study, we demonstrate that strategies succeeding “for all practical purposes” (FAPP) inher-
ently depend on transient regime assumptions, meaning their automated execution systematically
amplifies tail risks.

I.

INTRODUCTION

The question of whether a trading strategy can al-
ways make money—regardless of what the market does—
is older than mathematical finance itself.
It animated
Bachelier’s 1900 thesis [1], was formalized by Samuel-
son [2] and Fama [3] into the Efficient Market Hypothesis
(EMH), and underlies every retail advertisement promis-
ing “passive income in any market condition.” The EMH
gives the standard economic answer: competitive pres-
sure eliminates consistent excess returns. But the EMH
is a statement about equilibrium outcomes across many
agents; it is not, by itself, a mathematical proof that no
individual strategy can ever win unconditionally.

Defining a “universal strategy” as one that yields a
strictly positive terminal wealth across all valid market
trajectories is an exceptionally strong condition. Conse-
quently, the intellectual burden lies not merely in proving
impossibility, but in precisely delineating the mathemati-
cal frameworks under which this impossibility manifests.
This paper articulates this impossibility through three
distinct theoretical paradigms, carefully distinguishing
their incompatible underlying assumptions:

1. Financial (Measure-Theoretic, Sec. III). The
rigorous core of the impossibility. We show that un-
der standard admissibility constraints, a universal
strategy trivially constitutes an arbitrage opportu-
nity. By the First Fundamental Theorem of Asset
Pricing [4, 5], this is precluded in markets admit-
ting an equivalent martingale measure.

2. Combinatorial (Uniform Search, Sec. V). The
Wolpert–Macready No-Free-Lunch theorem [6] es-
tablishes that under a uniform distribution over all
possible discrete evolutions, no strategy uniformly
dominates. While markets are not uniform, this
boundary highlights that practical trading is an ex-
ercise in structure exploitation, not universal algo-
rithmic superiority.

3. Computational (Adversarial, Sec. VI). Shift-
ing the paradigm from exogenous price processes
to adaptive environments, we utilize the core diag-
onalization logic of the Halting Problem [7] to show
that no computable strategy can survive an adver-
sary capable of simulating it.

We frame these findings with a time-reversal criterion
(Sec. II) that motivates a formal analogy with Maxwell’s
Demon (Sec. IV). We treat this analogy carefully, not-
ing that while the equivalent martingale measure mirrors
thermodynamic detailed balance, appealing to physical
Landauer erasure costs to explain financial dissipation is
quantitatively irrelevant. The Wheel Options Strategy
is examined in Sec. VIII as a concrete illustration, and
the fragile scope of strategies operating for all practical
purposes (FAPP) is discussed in Sec. IX.

II. THE TIME-REVERSAL HEURISTIC

To build intuition, we define a purely kinematic crite-
rion. Let M be a suitably rich space of strictly positive
paths P : [0, T ] → Rn
>0. The time reversal (cid:101)P of a path P
is

∗ karl.svozil@tuwien.ac.at; http://tph.tuwien.ac.at/˜svozil

(cid:101)P (t) = P (T − t),

t ∈ [0, T ].

(1)

6
2
0
2

r
p
A
4
1

]

R
T
.
n
i
f
-
q
[

1
v
4
3
3
3
1
.
4
0
6
2
:
v
i
X
r
a

2

Definition 1 (Pathwise Universal Strategy). A strategy
σ is universally successful if its realized profit-and-loss
satisfies Π[σ; P ] > 0 for all P ∈ M.

Criterion 1 (Time-Reversal Necessity). If σ is univer-
sally successful on a path space closed under time rever-
sal, it must be time-reversal consistent: for every P ∈ M
where Π[σ; P ] > 0, it must hold that Π[σ; (cid:101)P ] > 0.

The operational content is clear: a momentum strategy
profiting on a rising trajectory will mechanically fail on
its time-reversed (falling) counterpart.
Remark 1 (Filtrations and Semimartingales). We explic-
itly state this as a heuristic because, in standard stochas-
tic finance, the space of admissible trajectories is not
trivially closed under time reversal. Time-reversing a
standard semimartingale generally destroys both its semi-
martingale property and its adaptedness to the forward
filtration. Therefore, this criterion serves as a concep-
tual framing for strict pathwise claims, while the rigorous
probabilistic formulation is provided in Sec. III.

B. The Martingale Constraint

By the Delbaen–Schachermayer generalization of the
First Fundamental Theorem [4, 5], a market satisfies
the condition of No Free Lunch with Vanishing Risk
(NFLVR) if and only if there exists an equivalent local
martingale measure (EMM) Q ≈ P.

Corollary 2 (No Universal Strategy). If an equivalent
martingale measure Q exists, no universally successful
admissible strategy can exist.

This is the definitive mathematical constraint. Uni-
versality is simply an impossibly strong formulation of
arbitrage; its preclusion is the foundational premise of
modern asset pricing. Thus, the non-existence of uni-
versal strategies is not a deep extension of asset pricing
theory, but rather a direct corollary of its foundational
no-arbitrage axiom.

III. THE NO-ARBITRAGE FOUNDATION

A. Admissibility and Arbitrage

The defining proof of impossibility stems from the First
Fundamental Theorem. We must first impose standard
regularity. Without admissibility constraints, pathwise
gains can be trivially manufactured via doubling strate-
gies (gambler’s ruin constructions), which are financially
meaningless due to infinite intermediate drawdown lim-
its.

Let the market be defined on a filtered probability
space (Ω, F, {Ft}t∈[0,T ], P). A strategy σ is admissi-
ble if its wealth process is bounded from below almost
surely [5].

Theorem 1 (Universality implies Arbitrage). If there
exists an admissible, self-financing strategy σ with zero
initial capital such that Π[σ; P (ω)] > 0 for all ω ∈ Ω,
then the market admits a strong arbitrage opportunity.

Proof. By assumption, the strategy yields strictly posi-
tive terminal wealth pathwise over all ω. Consequently,
under the physical measure P, the terminal wealth is
strictly positive almost surely (P(Π > 0) = 1). An ad-
missible, zero-cost, self-financing strategy ending strictly
positive almost surely strictly satisfies the condition for
a strong arbitrage (arbitrage of the first kind).

Remark 2 (Strength Hierarchy). The converse (Arbitrage
=⇒ Universal) is generally false. Our definition of uni-
versality demands pathwise strict positivity (Π > 0 for
all ω), which is a strictly stronger condition than clas-
sical arbitrage (which requires only P(Π ≥ 0) = 1 and
P(Π > 0) > 0). Because pathwise universality trivially
implies strong arbitrage, its impossibility is mathemat-
ically immediate once admissibility constraints are im-
posed under an equivalent martingale measure.

IV. THE MAXWELL’S DEMON ANALOGY:
HONEST BOUNDARIES

A. Formal Analogy, Not Equivalence

Maxwell’s Demon observes individual gas molecules
and sorts them to extract net work from thermal
equilibrium—violating the Second Law of Thermodynam-
ics. The physical resolution (Landauer, Bennett [8, 9])
is that the Demon’s memory must be erased to operate
cyclically, incurring an irreversible thermodynamic cost
of kBT ln 2 per bit.

A universally successful trading strategy claims to act
as a financial Maxwell’s Demon, extracting net profit
from market fluctuations in both temporal directions.
The conceptual mapping is highly instructive (Table I).

TABLE I. Formal analogy between Maxwell’s Demon and uni-
versally winning trading strategies.

Universal Strategy

Maxwell’s Demon
Observes molecule velocities Observes price movements
Sorts molecules for gradient
Works in both time directions Profits on P and (cid:101)P
Extracts equilibrium work
Resolution: info. erasure cost Resolution: absence of EMM

Extracts universal profit

Exploits movements for profit

The deepest point of contact lies in the concept of equi-
librium. In statistical mechanics, true equilibrium satis-
fies detailed balance: microscopic processes equal their
time reversals. In finance, the equivalent martingale mea-
sure Q enforces the financial equivalent of detailed bal-
ance: there is no preferred drift direction under the pric-
ing measure.

B. What the Analogy Does Not Establish

We must aggressively limit the scope of this analogy.
It is a formal mapping, not a category-theoretic isomor-
phism.

Transaction costs are not the core mechanism.
While transaction costs mimic thermodynamic dissipa-
tion, a frictionless market with zero spread still mathe-
matically precludes universal strategies via Corollary 2.
Landauer’s limit is irrelevant. The minimum en-
ergy to erase one bit at room temperature is E ≈
2.87 × 10−21 J. The Landauer cost of an algorithmic
trade is roughly 10−27 USD. Presenting thermodynamic
erasure as a financially relevant constraint on algorithms
is numerically dishonest.

The analogy illuminates why attempts to defeat effi-
cient markets resemble attempts to build perpetual mo-
tion machines, but the independent proof of financial im-
possibility lies strictly in measure theory (Sec. III).

3

automated, it acts as a computable Turing machine [10].
This guarantees its defeat via formal Turing diagonaliza-
tion [7, 11].

The Adversarial Market Construction can be modelled
as follows: Let Mσ be a computable trading algorithm
mapping discrete histories Ht to a target position wt ∈ R.

Theorem 4 (Defeat of Computable Strategies). Against
any computable strategy Mσ, there exists an adversarially
constructed, computable market trajectory Padv on which
Mσ realizes non-positive profit.

Proof. Because Mσ is a deterministic algorithm, a
market-making adversary with oracle access to the strat-
egy can simulate its next move. Set P0 > 0. At each
step t, the adversary simulates Mσ(Ht) → wt and sets
the next price opposing the strategy’s bet:

Pt+1 =






Pt · e−ϵ
Pt · e+ϵ
Pt

if wt > 0 (Strategy buys)
if wt < 0 (Strategy sells)
if wt = 0 (Strategy neutral)

(2)

V. THE COMBINATORIAL PERSPECTIVE
(NFL)

By construction, every risk-taking action yields a strict
loss. The strategy is defeated.

The No-Free-Lunch (NFL) theorem of Wolpert and
Macready [6] provides a second paradigm. NFL explores
optimization over function spaces.

Theorem 3 (NFL, Wolpert–Macready). For any two
search algorithms A1 and A2, when performance is aver-
aged uniformly over all possible objective functions, they
perform identically.

If we map trading to sequential decision-making over
discrete price paths, NFL dictates that, averaged uni-
formly over the set of all possible market evolutions, no
strategy can beat the trivial zero-strategy.

Model Incompatibility: We explicitly note that
NFL is not a direct proof of market efficiency. Finan-
cial markets are not drawn from a uniform distribution;
they are continuous, highly structured, and governed by
no-arbitrage constraints. Furthermore, trading involves
sequential risk under a stochastic process, which is cate-
gorically different from offline function optimization.

However, NFL provides a vital philosophical bound-
ary: it proves that any trading strategy that works must
derive its edge purely from exploiting the non-uniform
structure of the specific physical measure P. A strategy
cannot be inherently, universally superior; it can only be
perfectly fitted to a specific, potentially transient, market
regime.

VI. THE COMPUTATIONAL PERSPECTIVE
(ADVERSARIAL DIAGONALIZATION)

Our third paradigm discards the assumption of exoge-
nous price processes entirely and asks: what if the market
is an adaptive adversary? If a trading strategy is fully

Shift in Quantifiers: This theorem does not prove that
a passive market will defeat you. Instead, it proves that
no strategy can be “universal” against reactive environ-
ments. The construction assumes perfect observability
and zero-latency response; real markets approximate this
adversarial ideal only partially. However, in modern high-
frequency trading, where algorithms routinely probe and
trade against the predictable logic of other algorithms,
this adversarial boundary is highly relevant.

Additionally, abstract computability theory (Rice’s
Theorem [12]) guarantees that, in the strictest formal
sense of program semantics, the precise set of trajectories
that will ruin a given complex algorithm is fundamentally
undecidable; no master program can consistently flag im-
pending failure sets in arbitrary trading code.

VII. ANALOGY TO SOCIAL CHOICE:
ARROW’S IMPOSSIBILITY THEOREM AND
FIXED POINTS

To contextualize the severity of this computational
boundary, it is instructive to draw a parallel to social
choice theory—specifically, Arrow’s Impossibility Theo-
rem [13]. Just as the Halting Problem establishes a defini-
tive boundary for computer science by proving that no
machine can perfectly evaluate every possible input, Ar-
row’s Theorem establishes a definitive boundary for polit-
ical science by proving that no rank-order voting system
can universally aggregate individual preferences into a
consistent group decision without violating basic fairness
criteria.

There is a profound structural parallel between Ar-
row’s Theorem and our adversarial diagonalization argu-
ment:

The “Universal” Claim: Arrow investigates a voting
rule that is expected to work flawlessly across all
possible preference profiles. We investigate a trad-
ing strategy that claims to yield strict profit across
all possible market paths.

The Conflict: Arrow highlights the inherent tension be-
tween individual logic and group consistency. Our
model highlights the tension between a fixed algo-
rithmic logic and a reactive market environment.

The Failure Mode: In social choice, the system breaks
down via the Condorcet Paradox and the lack
of transitivity—a cyclical loop where the system
cannot decisively rank outcomes.
In our com-
putational framework, the breakdown occurs via
diagonalization—an adversarially constructed path
where the strategy is mechanically forced into a los-
ing position.

The “Winner”: Arrow proves that the only universally
stable rule collapses to a Dictator. Correspondingly,
Theorem 4 demonstrates that the only guaranteed
victor is the Adversary, who simulates and system-
atically defeats the computable strategy.

At a deeper mathematical

level, as demonstrated
by Yanofsky [11], all such diagonal and self-referential
paradoxes—including Russell’s paradox, G¨odel’s incom-
pleteness, and Turing’s Halting Problem—share a uni-
versal structure rooted in the nonexistence of fixed points.
In our adversarial market construction, the market acts
as a “negation” operator on the strategy’s predictions
(moving the price down when the strategy buys, and up
when it sells). Because this adversarial response func-
tion inherently lacks a fixed point, a universal mapping
from trading algorithms to strictly profitable outcomes is
mathematically impossible.

VIII. CASE STUDY: THE WHEEL OPTIONS
STRATEGY

To ground these theoretical limits, we evaluate the
Wheel Strategy: selling a cash-secured put to collect pre-
It
mium, and upon assignment, selling a covered call.
is widely marketed as generating universal, directionless
income. Our paradigms predict specific failure modes.

A. Failure I: Catastrophic Drop (Time Asymmetry)

If the underlying stock crashes post-assignment, the
covered call loses value, and capital is trapped at a severe
loss. This trajectory is the exact time-reversal of a rapid,
profitable rally. By our heuristic (Criterion 1), failure on
the reversed path confirms the strategy is not universal.

4

B. Failure II: Slow Bleed (Combinatorial Cost)

A persistent downtrend yields premiums but suffers lin-
ear capital decay. The NFL paradigm reminds us that a
strategy structurally specialized to harvest variance pre-
mium in sideways markets mathematically must under-
perform in sustained directional regimes.

C. Failure III: Violent Breakout (Capped Upside)

A massive rally results in assignment at the call strike,
generating nominal profit but massive opportunity cost.
The strategy’s structurally capped upside ensures its ex-
pected value aligns with the martingale constraint (Corol-
lary 2) by forfeiting the tail events that offset the catas-
trophic drops (Failure I).

IX. FAPP STRATEGIES AND THEIR HONEST
SCOPE

While true universality is mathematically precluded,
strategies can be profitable for all practical purposes
(FAPP, following Bell [14]) relative to a specific physical
measure P.

Market-making earns the spread under normal volatil-
ity, and Kelly criterion investing maximizes logarithmic
growth under known distributions. However, their suc-
cess is strictly conditional on the stability of the under-
lying measure.

Lo’s Adaptive Markets Hypothesis

[15] perfectly
bridges the gap between our adversarial diagonalization
and real markets. As capital floods into a successful
FAPP strategy, the market ecosystem adapts. The mar-
ket transitions from a passive generator (Sec. III) toward
the adversarial generator (Sec. VI), eroding the strategy’s
edge and dynamically creating the exact trajectories nec-
essary to generate its failure set.

Crucially, the automated execution of algorithmic
FAPP strategies transforms individual theoretical lim-
its into systemic risks. Because failure sets are algorith-
mically unidentifiable (Rice’s Theorem), automated sys-
tems will blindly execute into structural breakdowns—
as witnessed during the 1987 portfolio-insurance cas-
cade [16] and the 2010 Flash Crash [17].

A mapping of these failure modes to their correspond-
ing paradigms is provided in Table II. We emphasize that
these mappings are illustrative rather than deductive.
The failure set Fσ of the Wheel Strategy is nonempty
(by Corollary 2), algorithmically unavoidable in advance
(by Theorem 4), visited with nonzero probability under
any continuous price process.

TABLE II. Canonical failure modes of the Wheel Strategy and the formal paradigms predicting their existence.

Failure
I: Crash
II: Bleed
III: Breakout
IV: Ruin

Trajectory Class
Reversed uptrend
Persistent drift µ < 0
Fat-tailed upside
Absorbing state

Predicted By
Criterion 1
NFL Thm.
Cor. 2
Thm. 4

Mechanism
Time asymmetry
Specialization
Capped upside
Undecidability

5

X. CONCLUSION

The impossibility of a universally winning trading
strategy is fundamentally a corollary of the absence of
arbitrage in continuous-time mathematical finance. Sub-
ject to standard admissibility constraints, any strategy
claiming pathwise universal success implies an arbitrage
opportunity, which is strictly precluded by the existence
of an equivalent martingale measure.

While the First Fundamental Theorem provides the
definitive proof, structurally analogous limitations ap-
pear across incompatible mathematical paradigms. From
a combinatorial perspective, the No-Free-Lunch theorem
emphasizes that trading success relies entirely on exploit-
ing non-uniform market structure. From a computational
perspective, Turing diagonalization guarantees that for
any computable algorithm there exists an adversarial en-
vironment that defeats it.

The thermodynamic analogy of Maxwell’s Demon pro-
vides an elegant conceptual mapping between martingale

measures and detailed balance, provided we reject phys-
ically irrelevant appeals to Landauer erasure costs. Ul-
timately, any strategy claimed to work ”in all market
conditions” must conceal an implicit assumption about
the underlying market measure, relying on excluded tail
risks that are mathematically guaranteed to exist.

ACKNOWLEDGMENTS

Noson S. Yanofsky has drawn my attention to notable
parallels between the present result and both Arrow’s Im-
possibility Theorem and diagonalization techniques, the
latter of which also imply the absence of fixed points.

This

research was

in part
Fund
(FWF)[Grant
by
the Austrian
DOI:10.55776/PIN5424624 ].
The author acknowl-
edges TU Wien Bibliothek for financial support through
its Open Access Funding Programme.

funded in whole or
Science

This text was partially created and revised with as-
sistance from large language models. All content, ideas,
and prompts were provided by the author.

[1] L. Bachelier, Th´eorie de la sp´eculation, Annales Scien-
tifiques de l’´Ecole Normale Sup´erieure 17, 21 (1900).
[2] P. A. Samuelson, Proof that properly anticipated prices
fluctuate randomly, Industrial Management Review 6, 41
(1965).

[3] E. F. Fama, Efficient capital markets: A review of the-
ory and empirical work, The Journal of Finance 25, 383
(1970).

[4] J. M. Harrison and S. R. Pliska, Martingales and stochas-
tic integrals in the theory of continuous trading, Stochas-
tic Processes and their Applications 11, 215 (1981).
[5] F. Delbaen and W. Schachermayer, A general version of
the fundamental theorem of asset pricing, Mathematische
Annalen 300, 463 (1994).

[6] D. H. Wolpert and W. G. Macready, No free lunch theo-
rems for optimization, IEEE Transactions on Evolution-
ary Computation 1, 67 (1997).

[7] A. M. Turing, On computable numbers, with an appli-
cation to the Entscheidungsproblem, Proceedings of the
London Mathematical Society, Series 2 42, 43, 230 (1936-
7 and 1937).

[8] R. Landauer, Irreversibility and heat generation in the
computing process, IBM Journal of Research and Devel-
opment 5, 183 (1961).

[9] C. H. Bennett, The thermodynamics of computation—a
review, International Journal of Theoretical Physics 21,
905 (1982).

[10] C. I. Isasa Mart´ın, Computability-Based Analysis of Mar-
ket Predictability, Trabajo de fin de m´aster, Universidad
Complutense de Madrid, Madrid, Spain (2022), m´aster
en M´etodos Formales en Ingenier´ıa Inform´atica. Tutor:
Ismael Rodr´ıguez Laguna. Score: 9.8/10.

[11] N. S. Yanofsky, A universal approach to self-referential
paradoxes, incompleteness and fixed points, Bulletin of
Symbolic Logic 9, 362 (2003), arXiv:math/0305282.
[12] H. G. Rice, Classes of recursively enumerable sets and
their decision problems, Transactions of the American
Mathematical Society 74, 358 (1953).

[13] K. J. Arrow, Social Choice and Individual Values, 3rd ed.
(Yale University Press, New Haven, CT, 2012) foreword
to the Third Edition by Eric S. Maskin.

[14] J. S. Bell, Against ‘measurement’, Physics World 3, 33

(1990).

[15] A. W. Lo, The adaptive markets hypothesis: Market effi-
ciency from an evolutionary perspective, The Journal of
Portfolio Management 30, 15 (2004).

[16] N. F. Brady, J. C. Cotting, R. G. Kirby, J. R. Opel, and
H. M. Stein, Report of the Presidential Task Force on

Market Mechanisms (U.S. Government Printing Office,
Washington, D.C., 1988).

flash crash: High-frequency trading in an electronic mar-
ket, The Journal of Finance 72, 967 (2017).

[17] A. Kirilenko, A. S. Kyle, M. Samadi, and T. Tuzun, The

6
