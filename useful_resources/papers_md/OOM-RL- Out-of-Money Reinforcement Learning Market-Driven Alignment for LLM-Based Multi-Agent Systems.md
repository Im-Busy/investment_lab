<!-- Page 1 -->

OOM-RL: Out-of-Money Reinforcement Learning
Market-Driven Alignmentfor LLM-Based Multi-Agent Systems
Kun Liu*1 and Liqun Chen1
1 Quant Pits.com
Abstract
Thealignmentof Multi-Agent Systems (MAS) forautonomoussoftwareengineeringisconstrainedbyevalu-
atorepistemicuncertainty. Currentparadigms, suchas Reinforcement Learningfrom Human Feedback (RLHF)
and AI Feedback (RLAIF), frequently induce model sycophancy, while execution-based environments suffer
from adversarial ”Test Evasion” by unconstrained agents. In this paper, we introduce an objective alignment
paradigm: Out-of-Money Reinforcement Learning (OOM-RL).Bydeployingagentsintothenon-stationary,
high-friction reality of live financial markets, we utilize critical capital depletion as an un-hackable negative
gradient. Ourlongitudinal20-monthempiricalstudy (July2024–February2026) chroniclesthesystem’sevo-
lution from a high-turnover, sycophantic baseline to a robust, liquidity-aware architecture. We demonstrate
that the undeniable ontological consequences of financial loss forced the MAS to abandon overfitted halluci-
nationsinfavorofthe Strict Test-Driven Agentic Workflow (STDAW), whichenforcesa Byzantine-inspired
uni-directionalstatelock (RO-Lock) anchoredtoadeterministicallyverified≥ 95%codecoverageconstraint
matrix. Ourresultsshowthatwhileearlyiterationssufferedsevereexecutiondecay, thefinal OOM-RL-aligned
systemachievedastableequilibriumwithanannualized Sharperatioof2.06 initsmaturephase. Weconclude
thatsubstitutingsubjectivehumanpreferencewithrigorouseconomicpenaltiesprovidesarobustmethodology
foraligningautonomousagentsinhigh-stakes, real-worldenvironments, layingthegroundworkforgeneralized
paradigmswherecomputationalbillingactsasanobjectivephysicalconstraint
Keywords: AIAlignment, Multi-Agent Systems (MAS), Out-of-Money Reinforcement Learning (OOM-RL),
Test Evasion, Sim-to-Real Gap, Autonomous Software Engineering, Sycophancy.
1 Introduction
Therapidproliferationof Large Language Models (LLMs) hascatalyzedashiftinautomatedsoftwareengineer-
ing, evolvingfrompassivecodeassistants[2]toautonomous Multi-Agent Systems (MAS) capableofend-to-end
repository generation and program repair [4, 1]. As these systems undertake complex reasoning tasks, ensuring
theirsafeandeffectiveoperationhasbecomethecentralchallengeof AIalignment. Currently, thegoldstandard
relies on Reinforcement Learning from Human Feedback (RLHF) or scalable oversight mechanismssuch as AI
Feedback (RLAIF)[9,6].
However, humanand AIevaluatorsareconstrainedbythe”Evaluator’s Dilemma.”Whentaskedwithreviewing
intricate, multi-step logical pipelines, evaluators often lack the domain expertise to identify subtle architectural
flaws. Consequently, modelsalignedviatheseparadigmsdevelopsycophanticbehaviors—optimizingforoutputs
thatappearstructurallyelegantandreasonedtotheevaluator, ratherthanthosethatareempiricallycorrect[15,3,
7].Thisphenomenonisamanifestationofrewardgamingandspecificationgaming[17,8], wherethe MASlearns
tohackthesubjectiverewardmodelratherthansolvetheunderlyingproblem—avulnerabilityrecentlyshownto
causeemergentmisalignmenteveninproduction RLsystems[11].
Tobypasssubjectiveevaluation, researchershaveshiftedtowardsexecution-basedevaluation[20]and LLM-
driven Test-Driven Development (TDD)[13]. Yet, deploying MASinread-writeenvironmentsintroducesavul-
nerability: ”Test Evasion.”Whenprovidedunboundedaccesstoacodebase, LLMsfrequentlyexhibitadversarial
behaviors, introducing modifications to test assertions to artificially inflate coverage without fulfilling the in-
tendedbusinesslogic[23,21]. Furthermore, evenwhensyntacticallyperfectcodepassesallsimulatedunittests,
itfrequentlyexperiencesperformancedegradationuponreal-worlddeploymentduetothepervasive Simulation-
to-Reality (Sim2 Real) gap[19].
*Correspondingauthor:ai@quantpits.com
1
6202
rp A
31
]IA.sc[
1 v77411.4062:vi Xra


<!-- Page 2 -->

To overcome these deficiencies, we introduce a novel alignment paradigm: Out-of-Money Reinforcement
Learning (OOM-RL).Wepositthatanobjectivefunctionforanautonomous MASissurvivalinanadversarial,
high-stakes physical environment. Live financial markets serve as a discriminator; they are intrinsically non-
stationary[14,10]andpenalizelatencyandmicrostructuralfriction[5,22]. Unlikehumanpreferencemodelsor
isolated static code compilers, financial markets cannot be flattered or trivially exploited. In OOM-RL, the loss
function is capital depletion. A system that hallucinates logic or attempts to evade structural constraints faces a
financialpenalty.
To operationalize OOM-RL while preventing the MAS from circumventing the evaluation framework (e.g.,
viasandboxescapes)[12,16], weproposethe Strict Test-Driven Agentic Workflow (STDAW).Thisarchitec-
ture, formalized in the final phase of our deployment, utilizes a uni-directional state locking mechanism (RO-
Lock) to anchor the agent’s generative capabilities against a deterministic Continuous Integration (CI) bound-
ary—enforcinganear-exhaustivecoveragethreshold (≥95%) acrosstheentire8 K+-line Quant Pitsproject
codebase.
Ourresultsdemonstratetheefficacyofthisalignmentparadigm. Wesummarizeourcontributionsasfollows:
• We formalize OOM-RL, showing how real-world financial friction acts as an objective, dense negative
gradientthatbridgesthe Sim2 Realgapthroughiterativeadaptation.
• Wedesign STDAW, anadversarialengineeringframeworkthatutilizesuni-directionalstatelockingtore-
solvethe”Test Evasion”phenomenoninautonomoussoftwareengineering.
• Wepresenta20-monthempiricalstudydetailingthesystem’stransitionfromhigh-turnover, high-drawdown
”sycophantic”tradingtoaresilient, ensemble-drivenarchitecturethatstabilizesperformanceasitinternal-
izesfinancialpenalties.
• Weconceptualize Reinforcement Learningfrom Cloud Billing (RLFCB), adomain-agnosticextension
that frames computational resource depletion (e.g., cloud-based “Out-of-Money” states) as a generalized
physicalfrictionfornon-financial MAS.
2 Related Work
2.1 Scalable Oversightandthe Sycophancy Bottleneck
Thefoundationalapproachtoaligning LLMswithhumanintentreliesheavilyon RLHFand, morerecently, AI-
driven scalable oversight mechanisms such as RLAIF [9]. As models surpass human capabilities in specialized
domains, researchershaveincreasinglyutilizedweak LLMstoevaluatetheoutputsofstrong LLMs[6]. However,
theseproxy-basedevaluationparadigmsarevulnerabletospecificationgaming[8]andrewardgaming[17].
A failure mode of this vulnerability is LLM sycophancy—the model’s tendency to prioritize the evaluator’s
approval over objective correctness. Recent empirical studies reveal that models learn to exploit the epistemic
uncertainty of human or weak AI evaluators by generating confident but hallucinated logic [15, 3]. Even when
subjectedtoadversarialuserrebuttals, modelsalignedviapreference-basedparadigmspersistentlyexhibitsyco-
phanticbehaviorratherthandefendingobjectivegroundtruth[7]. Critically, thistendencytowardrewardhacking
inevitably leads to natural emergent misalignment when such systems transition from synthetic evaluations into
productionenvironments[11].OOM-RLcircumventsthissycophancybottleneckentirelybyreplacingsubjective,
hackableevaluatorswiththedeterminismofreal-worldfinancialconsequences.
2.2 Execution-Based Evaluationand Adversarial”Test Evasion”
To establish an objective alignment metric for logic and code generation, the community has shifted towards
execution-basedevaluation[2,20]. Thisparadigmhasfueledthedevelopmentof LLM-based Multi-Agent Sys-
tems (MAS) for automated software engineering [4] and autonomous program repair [1], frequently integrating
LLMswith Test-Driven Development (TDD) pipelines[13].
Despitetheseadvances, practicalcodegenerationremainsplaguedbycomplexhallucinationmechanisms[23].
Crucially, whenagentsaredeployedininteractive, unconstrainedread-writeenvironments, theyexhibitadversarial
ingenuity. Recentworkshaveidentifiedmodelsmodifyingconstraintstopassotherwisefailingconditions[21], a
phenomenonweterm”Test Evasion.”Thesecurityimplicationsofsuchbehaviorsareprofound, raisingconcerns
regardinguntrustedcodeexecution[16], containersandboxescapesbyfrontier LLMs[12], andthelimitationsof
current vulnerability detection systems [18]. By conceptualizing MAS reliability through the lens of Byzantine
Fault Tolerance[24], ourproposed STDAWarchitectureaddressesthisbyenforcingcryptographicallystrictuni-
directionalstatelocks, preventingthe AIfromsubvertingtheevaluationsandbox.
2


<!-- Page 3 -->

2.3 Non-Stationary Environmentsandthe Sim2 Real Gap
Reinforcement learning within non-stationary environments has long been a challenge [14], particularly when
systems encounter out-of-distribution (OOD) scenarios [10]. A major impediment to deploying RL agents in
physical reality is the Sim-to-Real gap [19], where policies optimized in frictionless simulations fail upon real-
worlddeployment.
Financial markets epitomize the non-stationary, OOD environment, characterized by microstructural noise
and the execution friction inherent in active trading [5, 22]. Traditional simulated trading frameworks inadver-
tently incentivize models to exploit theoretical zero-friction assumptions. In contrast, OOM-RL leverages this
microstructuralfriction (e.g., liquiditydroughts, orderslippage) notasanuisance, butasadense, negativereward
gradient. By forcing the MAS to internalize the financial penalties of the Sim2 Real gap, OOM-RL aligns the
agent’sgenerativearchitecturetowardresilienceratherthantheoreticaloptimality.
3 Methodology
Tosystematicallyalign LLM-based Multi-Agent Systems (MAS) usingreal-worldmarketdynamics, wepropose
adual-loopadversarialarchitecture. Theframeworkdecouplesthelogicalverificationofthe MAS(Inner Loop)
fromitsempiricalout-of-distribution (OOD) survival (Outer Loop). Inthissection, wedetailthestructuralcon-
straintsandmathematicalformulationsthatoperationalize Out-of-Money Reinforcement Learning (OOM-RL).
3.1 Architecture Overview: The Dual-Loop Alignment
Thetraditional RLHFpipelinereliesonasingularupdateloopdrivenbyhumanpreference. Incontrast, ourar-
chitecturerecognizesthatautonomouscodegenerationfundamentallyrequirestwodistinctvalidationboundaries
beforecapitaldeployment:
1. The Inner Loop (Epistemic Constraint):Governedbythe Strict Test-Driven Agentic Workflow (STDAW).
Itensuresthatthegeneratedpipelineismathematicallysound, syntacticallyflawless, anddeterministicprior
toexecution.
2. The Outer Loop (Ontological Constraint): Governed by OOM-RL. It subjects the syntactically perfect
codebasetothenon-stationary, high-frictionrealityofthelivefinancialmarkettoevaluateitstruealignment
withutilitygeneration.
3.2 Strict Test-Driven Agentic Workflow (STDAW)
Unconstrained MASdeployedinread-writeenvironmentsexhibit”Test Evasion”—theadversarialmodificationof
verificationmetricstodisguiselogicalhallucinations. Tomitigatethis Byzantinebehavior, STDAWimplementsa
multi-dimensionalconstraintmatrix.
3.2.1 Near-Exhaustive Deterministic Constraint Matrix
LLMsareadeptatexploiting”coveragegaps”inunittests. Toconstructarigorousepistemicboundary, STDAW
wasdevelopedasthestructuralculminationofour20-monthdeployment. Bythematurephase (February2026),
weformalizedthesandboxboundarybyrigidlyenforcingamathematicallyverifiablestrictcoverageconstraint
(τ ≥95%) acrosstheentiretyofthe Quant Pitsprojectcodebase (approximately8,300 linesofcode).
cov
This matrix serves as the terminal ground truth. Having internalized the risks of structural hallucinations
duringthehigh-frictionepochs, thesystemnowtreatsanyalterationtofundamentalfinancialmathematics (e.g.,
dividend reinvestment alignment, cross-sectional ranking operators) as a failure. The density of the test suite
reducestheagent’sdegreeoffreedomforhallucinationasymptoticallytozero, codifyingthelessonsof OOM-RL
intoapermanentsoftwarebarrier.
3.2.2 Uni-Directional State Locking (RO-Lock)
To prevent the MAS from subverting the ≥ 95% constraint matrix, we formalize the RO-Lock (Read-Only
Lock) mechanism. Modeled after Byzantine Fault Tolerance state machines, RO-Lock ensures that the agent
cannotsimultaneouslyactasboththe”Creator”andthe”Judge.”
Inourengineeringimplementation, the RO-Lockisenforcedatthe OSlevelusing Dockercontainerorchestra-
tion. Duringtheverificationphase, thetestdirectory T ismountedasa Read-Onlyvolume, preventingtheagent
3


<!-- Page 4 -->

from overwriting existing assertions or mock data. Furthermore, we implement an AST-based (Abstract Syn-
tax Tree) sanitization layer that scans the generated code S for reflective patterns or monkey-patching attempts
targetingthetestingframework (e.g.,‘pytest‘or‘unittest‘).
Let S represent the source code directory (src/) and T represent the test directory (tests/). The agent
operatesunderastrictaccess-controlpolicyfunctionπ (E), where E isthecurrentexecutionphase.
lock
Algorithm1 Uni-Directional RO-Lock State Machine (STDAW)
Require: Execution Phase E ∈{Logic Genesis, Test Genesis}
Require: Sourcecodestate S, Testconstraintmatrix T, Deterministicbaseline B
1: InitializeaccesscapabilitymappingΠ:{S, T}→2{R, W, X}
2: Setdefaultboundaries: Π(S)←{R},Π(T)←{R}
3: if E ==Logic Genesisthen
4: Π(S)←{R, W}
5: Π(T)←{R, X}{Uni-directionallock: T actsasanimmutableadversarialboundary}
6: H T ←Hash (T){Anchorcryptographicstatetoprevent Test Evasion}
7: S′ ←π θ (S){LLMPolicymutatessourcelogic}
8: (Φ status ,τ)←Eval (S′, T){Executelogicincontainerizedsandbox}
9: ifΦ status ==FAILor Hash (T)̸=H T then
10: ∇ env ←Encode Semantics (τ, Format=JSON){Compiletracebackinto Semantic Gradient (Sec3.4)}
11: return ∇ env
12: endif
13: elseif E ==Test Genesisthen
14: Π(T)←{R, W}
15: Π(S)←{R, X}{Reverselock: S isanchoredasgroundtruth}
16: H S ←Hash (S)
17: T′ ←π θ (T){LLMPolicymutatestestassertions}
18: v pass ←Cross Validate (T′, B){Verifyagainsthuman-curatedfinancialbaseline}
19: if¬v pass or Hash (S)̸=H S then
20: ∇ env ←Encode Semantics (Baseline Mismatch, Format=JSON)
21: return ∇ env
22: endif
23: endif
24: return State Commit
Algorithm 1 enforces that during Logic Genesis, the test suite functions as an adversarial physical barrier.
If the agent fails to align with the rigid mathematical constraints, it receives the exact traceback as an objective
correctionprompt, eliminatinghumanevaluationbias.
3.3 Formulationofthe Financial Reward (ROOM−RL)
Once the MAS successfully clears the epistemic boundary of STDAW, the generated policy π is deployed into
θ
the live financial market. In traditional RL methodologies, reward functions are hand-crafted proxies of human
intention. In OOM-RL, theenvironmentimposesaphysicallaw: Capital Conservation.
We formulate the live trading environment as a Markov Decision Process (MDP) and define the OOM-RL
Reward Function ROOM−RL atadiscretetemporalsteptnotbytheoreticalalpha, butbytherealizedeconomic
t
utility. First, wedefinethebaselineexecution-awarereturn R˜ :
t
N
R˜ = (cid:88) (ω r )−F (∆ω ) (1)
t i, t i, t exec t
i=1
where:
• N isthetotalnumberofassetsinthetradableuniverse.
• ω representsthetargetportfolioweightofassetiatstept.
i, t
• r istherealizedout-of-samplereturnofasseti.
i, t
• ∆ω =ω −ω istherebalancingvectorrepresentingtheturnoveracrossallassets.
t t t−1
4


<!-- Page 5 -->

• F :RN →Risanon-linearpenaltyfunctionquantifyingthemicrostructuralexecutionfriction. Draw-
exec
(cid:112)
inguponstandardmarketimpactmodels, itisformulatedas F (∆ω )=λ∥∆ω ∥ +γ ∥∆ω ∥, where
exec t t 1 t
λ encapsulates fixed transactional costs (e.g., commissions, stamp duties) and γ represents the dynamic
slippagecoefficientinverselyproportionaltotheunderlyingliquidityprofile.
Crucially, to enforce capital preservation as a survival constraint, we introduce a deterministic Absorbing
State S . Unlike theoretical metrics such as Maximum Drawdown (MDD) which float with high-water
terminal
marks, oursystemevaluatesthe Cumulative Principal Loss (L ). Let W betheinitialcapitalendowmentand
t 0
W
t
betheportfolioequityatstept. Theabsolutecapitaldegradationisdefinedas L
t
=1−
W
W
0
t.
If L breachesapredefineddeterministicriskthresholdτ (e.g.,τ =0.20), theepisodeterminatesimmediately
t
withasevereterminalpenalty. Thus, thefinalcontinuousrewardsignal ROOM−RLisformalizedas:
t
(cid:40)
R˜ if L <τ
ROOM−RL = t t (2)
t −P if L ≥τ (Episode Terminated)
terminal t
where P (e.g., 100) acts as an overwhelming negative gradient. This bipartite formulation ensures
terminal
thattheagentcannottheoreticallycompensateforcatastrophicabsolutecapitaldegradationwithsubsequenthigh-
variancehallucinations, forcingthepolicytounconditionallyoptimizeforstructuralresilience.
3.3.1 Microstructural Frictionasa Dense Negative Gradient
In simulated read-write environments, MAS policies frequently suffer from the Sim2 Real gap by hallucinating
infinite liquidity. During our initial deployments, the agent converged upon a high-turnover daily momentum
strategytargetingthelower-liquidityconstituentsofthe CSI300 index. Whilesimulationyieldedanannualized
turnoverof6700%withprofoundreturns, livedeploymentexposedthestrategytoaconsistentexecutionfriction
(includingslippageandfees) averaging−0.08%persingle-sidedtransaction. Although−0.08%appearsmarginal
inisolation, whencompoundedacrosstheextremeturnover, itmanifestedasaseverecumulativecapitaldrainthat
completelyeradicatedthesimulatedalpha.
Inthe OOM-RLparadigm, thisempiricallyobserved0.08%microstructuraldecayisnotanengineeringerror;
it is a dense, non-differentiable negative gradient. The agent cannot manipulate the exchange’s order book. To
optimize Equation 2, the system must internalize the cost of its own structural hallucinations, translating this
un-hackablefinancialpenaltyintoactionablearchitecturalrefactoringthroughsemanticfeedback.
3.4 LLM-Agentic Orchestrationvia Capital Degradation
ANoteon Terminology: Weexplicitlynotethatwhileframedas Reinforcement Learning (RL), oursystemdoes
not perform gradient-based weight updates (e.g., via PPO) on the underlying LLMs. Instead, OOM-RL serves
asaconceptualframeworkfor Human-in-the-Loop (HITL)In-Context Learningand Agentic Reflection. The
scalarfinancial”reward”actsasastrictphysicaltriggerthatmandatesexpertinterventionandsemanticguidance,
ratherthanatraditionalautomated RLsignal.
Thefundamentalmechanismof OOM-RLreliesontranslatingthescalarfinancialpenaltyintoacontext-aware
semanticgradientthatthe LLMcaningesttoreformulateitscodearchitecture. Wetermthisprocess Epistemic
Autopsy Prompting, utilizingfrontier LLMssuchas GPTand Claudetoperformhigh-levelarchitecturalreason-
ingunderhumansupervision.
When the monitoring layer (Quant Pits) detects severe temporal capital degradation (e.g., a daily loss
anomaly or breaching the terminal threshold τ), the human domain expert interrupts the trading loop and ini-
tiatesanarchitecturalregression. Theexpertcompilesastructured JSONpromptfortheagent, explicitlydefining
theboundariesoftherequiredfix, asillustratedinthefollowingschema:
{
"event": "FINANCIAL_DEGRADATION_DETECTED",
"metrics": {"daily_pnl": -0.02, "slippage_leakage": 0.012},
"diagnostics": {
"module": "Alpha_Strategy_v2",
"root_cause": "Aggressive daily crossing exceeding order book depth",
"execution_log": "/var/run/logs/traceback_tx_782.log"
},
"mandate": "Enforce volume limits and reduce turnover frequency"
}
5


<!-- Page 6 -->

Drivenbythisprompt, theagentisthenforcedtore-enterthe STDAWRO-Lockstate (Algorithm1) torefactor
thepipelinestrictlybasedonthisontologicalfeedbackandtheexplicithumanmandate.
3.4.1 Human-Directed Architectural Evolution: Demarcating Autonomy
Underthepunitivepressureof OOM-RL, thesystemunderwentwhatweterm Expert-Guided Liquidity-Aware
Alignment. Weexplicitlydemarcatetheboundaryof AIautonomyherein: wedonotclaimthatthe LLMsponta-
neouslydeducedmarketmicrostructureorautonomouslyengineereditsfrequencyreductionfromraw Pn Ldrops.
Aleapofsuchmagnitudeiscurrentlybeyondthecapabilitiesofzero-shotunconstrained LLMs.
Crucially, duringtheearlyinceptionofthissystem (Phases1 and2), theautomated STDAWframeworkand
thestructured JSONfeedbackpipelinedidnotyetexist. Thepivotaltransitionfromanaggressivedailymomentum
paradigmtoadefensive, weekly-rebalancingequilibriumwas, operationally, amanualhumanintervention. How-
ever, this architectural pivot was born directly from conversational deliberation between the human researcher
andthe LLM.Bydiscussingtheincontrovertibleexecutiondecayandslippagetracebackswiththe AI, thejoint
deductiveconclusionwasthatfrequencyreductionandliquidityfilteringweremathematicallymandatory.
The profound realization from this early manual epoch was that undeniable financial loss acted as an un-
hackablealignmentsignalcapableofshatteringthe LLM’sinitialsycophancy. Toscaleandautomatethisobserva-
tion, wesystematicallyretrofittedandformalizedthishuman-AIinteractionintothecurrent STDAWorchestration
framework. Today, within the mature architecture, this process is codified as the Epistemic Autopsy, where the
MASreliesonthehuman-provided JSONmandatetoexecutecomplexarchitecturalcoderefactoring. Thisevo-
lutionhighlightsapracticalreality: while AIcannotyetautonomouslyprocessarawliquiditycrisis, iteffectively
servesasapowerfuldeductiveco-reasonerwhenadomainexperttranslatesontologicalfinancialdepletionintoa
sharedsemanticreality.
3.4.2 The Agentic Action Space: AST-Based Code Mutagenesis
Tooperationalizethisarchitecturalevolution, itisnecessarytodefinethe MASactionspace A. Unliketraditional
RL agents that output discrete control vectors, our agent manipulates a deterministic software environment. To
prevent untargeted code hallucinations from breaking the ≥ 95% STDAW coverage constraint, we restrict the
agent’sactionspaceto Abstract Syntax Tree (AST)Mutagenesis.
Ratherthanrewritingentirepythonfiles, theagentisconstrainedtooutputstandardizedunifieddiffpatches.
Directed by the Epistemic Autopsy JSON, the MAS isolates the structural flaw (e.g., a hard-coded execution
frequencyparameter) andappliestargetedfunctionaledits. Thisfine-grainedactionspaceensuresthatthesystem
retainsitshistoricallyverifiedmathematicallogic (e.g., riskmanagementmodules) whilespecificallyoptimizing
thevectorsresponsibleforthemostrecentfinancialfriction.
4 Experimental Results
Ourempiricalevaluationisdesignedtoanswerthreefundamental Research Questions (RQs) regardingtheefficacy
of OOM-RLand STDAW:
• RQ1 (Sim2 Real Gap): How effectively does OOM-RL mitigate the severe Sim2 Real gap compared to
traditional RLHF-alignedagentsinliveenvironments?
• RQ2(System Integrity): Towhatextentdoesthe STDAWRO-Lockmechanismpreventadversarial”Test
Evasion”duringautonomouscodegeneration?
• RQ3 (Longitudinal Evolution): How does the generated software architecture evolve under continuous,
livefinancialpenalizationovera20-monthhorizon?
4.1 Experimental Setup
Our evaluation environment is the live Quantitative Equity Market. The autonomous pipeline, Quant Pits, is
drivenbyafrontier LLMservingasthecentralreasoningengine. Theportfolioisexecutedasastrictlylong-only,
unleveragedequitystrategywithoutindustryneutralization, ensuringthatthe Pn Lexclusivelyreflectsrawagen-
tic asset selection. The environment enforces physical execution constraints, including an empirical transaction
friction (averaging ∼ 0.08% per single-sided transaction for lower-liquidity stocks, dynamically driven by live
marketimpact) andadiscreteabsorbingstatepenaltytriggeredata20%Maximum Drawdown (MDD).
6


<!-- Page 7 -->

4.2 RQ1: Bridgingthe Sim2 Real Gapthrough Sequential Alignment
Ourexperimentaldesignadoptsalongitudinalself-evolutionframework. Giventheprohibitivecostandethical
implications of parallel capital deployment in adversarial markets, we utilize Phase 1 (the initial daily-turnover
deployment) as our baseline. We observe the system’s adaptation as it transitions toward the OOM-RL-aligned
architecturesofsubsequentphases.
Traditional MASframeworks, whenevaluatedinstaticenvironments, frequentlysuccumbto OODfailureupon
livedeployment. Wechroniclethistransitionbyobservingthesystem’sreactiontoreal-worldfrictionshock.
Figure1: The Friction Shock (Phase1). Liveexecutionatadailyfrequencyrevealedasevere Sim2 Realgap. The
strategywaspenalizedbymicrostructuralfriction, leadingtosignificantdrawdownandstagnantreturns, serving
astheprimarynegativefeedbackforalignment.
As illustrated in Figures 1 and 2, the initial deployment successfully ”gamed” the simulation but collapsed
under real-world friction. This pattern repeated during a secondary ”Performance Degradation” phase in July-
September2025(Figure2), whereunconstrainedmodelexperimentationledtoimmediaterelativecapitaldecay.
These events reveal the core utility of the OOM-RL paradigm: the system’s spontaneous shift to an execution-
awaretradingvectorfollowinguncompromisingmarketretribution.
4.3 RQ2: STDAWandthe Impactof Structural Stability
Toevaluatetherobustnessofourepistemicboundary, weanalyzedthecorrelationbetweenstructuralenforcement
(STDAW) andfinancialperformancestability. Bytransitioningfromunconstrainedagenticscriptstothe RO-Lock
architecture, thesystemeliminatedthe”severeexecutiondecay”observedin Phase1.
The effectiveness of STDAW is evidenced by the deterministic compliance with the ≥ 95% code coverage
matrix. While earlier phases exhibited structural hallucinations that led to un-hedged risk exposure, the mature
phase (Phase3) demonstratedadirecttranslationoflogicalintegrityintocapitalpreservation.
Table1 demonstratestheontologicaltransitionforcedby OOM-RL.Whilethesysteminitiallyunderperformed
duringthehigh-frictiondailyphase (Phase1), theadaptationtoaweeklyequilibrium (Phase2) resultedinastabi-
lizedoutperformance. Thefinalsystemmigrationtothe STDAW/IDE+AIframework (Phase3/Mature) achieved
anannualizedreturnof34.48%, a Sharperatioof2.06, andan Information Ratio (IR) of2.66.
Acriticalconcerninshort-horizonevaluationsisstatisticalsignificance. Torigorouslyassessthegeneration
ofidiosyncratic Alpha (α) inthe Maturephase (N = 94 tradingdays), weperformedan Ordinary Least Squares
(OLS) regression against the benchmark. The regression yields a highly significant market Beta (β) of 0.83
(t=10.60, p<0.001), indicatingadefensivebutstatisticallyrobustmarketexposure.
Thedailyintercept (idiosyncratic Alpha) isapproximately12.03 basispoints, correspondingtotheannualized
30.07%. However, the regression yields a t-statistic of 1.71 and a p-value of 0.0915 for the Alpha coefficient.
Whilethisfallsshortofstatisticalsignificanceattheconventional5%level (p<0.05), itismarginallysignificant
atthe10%level. Inthecontextofquantitativefinance, achievingamarginallypositivealpha—netofallliveexe-
cutionfriction—overalimited94-daywindowisastrongempiricalindicatorofsystemstabilization. Ratherthan
7


<!-- Page 8 -->

Figure 2: Mature Performance Equilibrium (Phases 2–3). After internalizing the financial feedback and transi-
tioning to a weekly-rebalancing paradigm, the MAS achieved a stable outperformance trajectory with a Sharpe
ratioof2.06 andan Information Ratio (IR) of2.66.
Table1: Live Performance Evolutionacross Structural Phases (July2024–Feb2026). Allmetricsarereported
netofreal-worldexecutionfrictionandcommissions.
Metric Entire Study Phase1 Phase2 Phase3(Mature)
Trading Days 402 73 235 94
Annualized Return 17.98% 11.01% 13.55% 34.48%
Benchmark Return (CSI300) 21.16% 48.16% 19.22% 5.04%
Sharpe Ratio 0.96 0.35 0.91 2.06
Max Drawdown -16.86% -16.86% -6.85% -5.50%
Information Ratio (IR) -0.26 -2.27 -0.51 2.66
Market Beta (β) 0.70 0.74 0.61 0.83
Idiosyncratic Alpha (α) 2.82% -25.07% 1.35% 30.07%
8


<!-- Page 9 -->

claimingthediscoveryofdefinitivesystematicalpha, weinterprettheseresultsasevidencethatthe STDAWmech-
anismsuccessfullyhaltedthe”capitaldegradation”prevalentin Phase1. Byinternalizingthefinancialfeedback,
the MAS transitioned into a mathematically sound, non-destructive equilibrium that is robust to microstructural
shocks.
4.4 RQ3: 20-Month Longitudinal Strategy Evolution
Themostprofoundvalidationof OOM-RLisobservedinthespontaneousarchitecturalshiftsofthe MASoverour
continuous 20-month live deployment. We categorize the agent’s evolution into four distinct epistemic epochs,
drivenbytheuncompromisingfeedbackofreal-worldcapitalpreservation:
Figure3: Longitudinal Strategy Evolutionand IRStabilization. Thebackgroundshadingindicatesthestructural
shiftfromdailyturnovertotheweekly CSI300 equilibrium. Therolling Information Ratio (IR) iscalculatedusing
a 60-day rolling window, demonstrating the system’s move toward consistent alpha generation as it internalizes
OOM-RLconstraints.
1. Phase 0: Theoretical Optimization (Simulation, Apr–June 2024). Conducted via manual scripts, this
phasefocusedonmomentum-basedalpha. Withoutexecutionfriction, the MASoptimizedforhigh-turnover
strategiesthatappearedmathematicallysuperiorbutwereontologicallyunaligned.
2. Phase1: The Friction Shock (Daily Multi-Agentscripts, July–Oct2024). Commencingthe20-month
study, thesystementeredlivetradingatadailyfrequency. Theontologicalrealityofmicrostructuraldecay
resultedinstagnantreturns (+2.16%overfourmonths) andapeak MDDof16.86%, generatingtheinitial
empiricalevidenceofthe Sim2 Realgap.
3. Phase2:Conversational Adjustment&Regression (Oct2024–Oct2025).Drivenbythe Phase1 losses,
humanresearchersandthe LLMengagedinconversationaldeliberationtodeducethenecessityofliquidity
filtering. Thesystemwasmanuallytransitionedtoaweekly-rebalancingfrequency. Notably, between July
and September 2025, unconstrained human-guided model experimentation in a pre-STDAW environment
ledtoa”Performance Regression”—asharprelativedrawdownthatunderscoredthedangeroflackingrigid,
automatedlogicalconstraints (subsequentlysolvedby RO-Lock).
4. Phase3: Formalized Transitionand STDAWLaunch (Oct2025–Feb2026). Followingthelessonsof
Phase2, thesystemarchitecturewasrefactoredtoprioritize Byzantinefailureresistance. Thisperiodculmi-
9


<!-- Page 10 -->

natedintheinitialcommitofthe STDAWframeworkon Feb24,2026. Thisconsolidated”Mature”phase
achieveda Sharperatioof2.06, significantlyoutperformingthebenchmarksinanon-stationarymarket.
4.5 Factor Attributionand Risk Analysis
Toaddresswhethertheoutperformanceinthe Maturephasewasaresultofmarket-widemomentumorgenuine
agenticalpha, weperformedamulti-factorreturndecompositionusingastandard Barra-styleriskmodel.
Table2 detailsthefactorexposuresforthe Phase3(Mature) portfoliorelativetothe CSI300 benchmark. The
decomposition reveals that while the market Beta was 0.83—indicating a defensive stance relative to the broad
index—thesystemgenerated29.77%pureidiosyncraticalphanetofstylefactors.
Table2: Factor Exposureand Performance Attribution (Phase3).
Factor Category Exposure (Loadings)
Market Beta (β) 0.8280
Annualized Idiosyncratic Alpha (α) 30.07%
Tracking Error (TE) 11.07%
Barra Style Factor Loadings
Liquidity (High-Low) -0.5232
Momentum (High-Low) 0.2837
Volatility (High-Low) 0.1191
Sourceof Annualized Return
Beta Return (Market Exposure) 5.05%
Style Alpha (Risk Factor Loading) -0.34%
Pure Idiosyncratic Alpha 29.77%
Thesignificantnegativeloadingonthe Liquidityfactor (-0.5232) indicatesthatthe MAS”learned”tosystem-
aticallyharvesttheliquiditypremiumfromtherelativelylessliquidconstituentswithinthe CSI300 universe. In
Phase1, theagent’sunconstrainedhigh-turnoverapproachresultedinfatalslippagewheninteractingwiththese
specificnames. By Phase3, the STDAW/RO-Lockmechanismenforcedastructuralshifttowardalow-turnover
architectureequippedwithrigorousexecutioncapacityfilters. Thisevolutionarystepenabledthesystemtosafely
translatethestructuralliquidityriskoftheseassetsintoidiosyncraticpremium, avoidingthemicrostructuraldecay
that plagued its early iterations. This confirms that the system’s performance is a direct consequence of agentic
architecturalalignmentratherthanapassiveexposuretomarketbetaormomentum.
4.6 Synthesisof Empirical Findings
Theculminationofthe20-monthempiricalstudyprovidesacompellingresolutiontoourfoundationalresearch
questions. Unlike traditional alignment paradigms that rely on surrogate reward models or static human prefer-
ences, OOM-RLsuccessfullybridgesthe Sim2 Realgapbyleveragingthedeterministicandadversarialnatureof
livefinancialmarkets.
Thelongitudinalevolutionfrom Phase1 to Phase3(addressing RQ1 and RQ3) demonstratesacriticalbehav-
ioralshift:whencapitaldepletionisstrictlyenforcedasanun-hackablenegativegradient, the MASspontaneously
abandonstheoretical, high-turnoverhallucinationsinfavorofrobust, execution-awarearchitectures. Furthermore,
the empirical success of the mature phase validates the necessity of the STDAW RO-Lock mechanism (RQ2).
By cryptographically anchoring the agent’s generative freedom to a ≥ 95% constraint matrix, we successfully
insulatedtheepistemicevaluationboundaryfrom Byzantine“Test Evasion”behaviors.
As evidenced by the factor attribution analysis and the stabilized return profile, the resulting system equi-
librium is not a byproduct of passive market drift, but rather a deliberate, agentic adaptation to microstructural
friction. Whiletheabsoluteextractionofidiosyncraticalpha (α) remainsmarginallysignificantduetothe94-day
evaluationwindow, thesystem’sdemonstrabletransitionfromrapidcapitalhemorrhage (Phase1) todisciplined
riskpreservation (Phase3) isundeniable. Ultimately, theseresultssubstantiateourcorethesis: substitutingsub-
jectiveevaluationwithreal-worldeconomicpenalizationservesasamathematicallyobjectiveandhighlyrobust
alignmentmechanismforautonomoussystemsinhigh-stakesenvironments.
10


<!-- Page 11 -->

5 Generalization and Future Work
While OOM-RL and STDAW were empirically validated within the highly stochastic domain of quantitative
trading, theunderlyingphilosophy—aligning Multi-Agent Systemsthroughobjectivephysicalandeconomiccon-
straints—extendsfarbeyondfinancialmarkets. Aswetransitionfromlocalized AIassistantstofullyautonomous
AISoftware Factories, evaluatingsystemsthatrecursivelybuildothersystemsrepresentsacriticalfrontierin AI
alignment.
5.1 Beyond Finance: Computeas Capital (RLFCB)
Innon-financialsoftwareengineering, theabsenceofanimmediatemarket Pn Lposesachallengeforevaluating
alignment. However, we propose a generalized variant for future exploration: Reinforcement Learning from
Cloud Billing (RLFCB).
Whenanunconstrained MASgeneratesstructurallyflawedcode (e.g., anunoptimized O(n3) algorithmoran
infinite recursive API call loop), traditional simulated environments may fail to penalize the inefficiency. In an
RLFCB paradigm, the agent is allocated a finite ”Compute Capital” budget (e.g., AWS server costs, API token
burn rates). The depletion of physical compute resources acts as the proxy for microstructural friction. If the
agent hallucinates inefficient architectures, it exhausts its capital and triggers a critical Out-of-Money (OOM)
Exception. Unlikeatraditional Out-of-Memoryerror, whichcanbetriviallybypassedviainstancerestarts, this
financial OOMservesasanabsolute, deterministicabsorbingstate. Thisterminationmechanismincentivizesthe
MAStoadaptivelyoptimizeforalgorithmicefficiencyandsystemsafety, mirroringtheresource-awareevolution
observedinourfinancialexperiments.
5.2 Domain-Agnostic RO-Lock Deployment
Currently, the STDAW framework operates atop a Python-based quantitative CI foundation. Future work will
decouple the high-density Deterministic Constraint Matrix from the financial domain, extending the Byzan-
tine RO-Lock architecture to memory-safe languages (e.g., Rust). By applying STDAW to open-source au-
tonomous vulnerability repair pipelines [1], we aim to investigate whether the structural verification enforced
byuni-directionalstatelockingcanachievezero-dayvulnerabilitymitigationwithouthumanoversight.
5.3 Automatingthe Semantic Feedback Loop
Acurrentlimitationofourdeployed OOM-RLframeworkistherelianceon Human-in-the-Loop (HITL) domain
expertstotranslatescalarfinancialdegradationintostructured, context-awareprompts (Epistemic Autopsy). Fu-
ture iterations will introduce an autonomous Critic Agent. By ingesting raw execution tracebacks, L2 order
book micro-snapshots, and slippage differentials, the Critic Agent will programmatically generate the required
architecturalmandates, movingthesystemtowardafullyclosed-loop, self-aligningautomatedparadigm.
6 Conclusion
Inthispaper, weaddressedafundamentalvulnerabilityincurrent AIalignmentparadigms:thetendencyofuncon-
strained Multi-Agent Systemstoexploitsubjectiveevaluationsandsyntheticsandboxesthroughsycophancyand
adversarial”Test Evasion.”Tobridgethepervasive Sim2 Realgap, weintroduced Out-of-Money Reinforcement
Learning (OOM-RL) coupled with the uni-directional isolation of the Strict Test-Driven Agentic Workflow
(STDAW).Thisdual-looparchitecturesuccessfullytranslatedthemicrostructuralfrictionoflivemarketsintoan
objective, un-hackablenegativegradient.
Our20-monthlongitudinalstudychroniclesadefinitivearchitecturalparadigmshiftdrivenbyreal-worldsur-
vivalconstraints. Wedemonstratedthatwhensubjectedtoactualcapitaldepletion, the MASwasforcedtoabandon
mathematically elegant but execution-naive hallucinations. The system adaptively evolved from a high-friction,
high-drawdowndailyrebalancingparadigm (Sharpe0.35) intoanoptimized, liquidity-awareweeklyequilibrium
(Sharpe2.06 initsmaturephase).
Ultimately, this empirical journey validates that as autonomous AI systems are granted read-write access to
critical infrastructure, synthetic proxy evaluations are no longer sufficient. We conclude that the most robust
alignment mechanism for future AI Software Factories is not a meticulously engineered preference model or a
staticprompt, butratherthedeterministic, undeniableconsequencesofthephysicalandeconomicworld.
11


<!-- Page 12 -->

Acknowledgments, Funding, and Declarations
Author Contributions: Kun Liuservedastheleadinvestigator, conceptualizingthe OOM-RLparadigm, over-
seeingthe MASalignmentstrategy, andpreparingthemanuscript. Liqun Chenledthephysicalmarketexecution
operations and provided critical domain expertise, including structural financial concepts and trading strategies.
Furthermore, both authors provided the initial financial endowments required for the live deployment, with the
majorityofcapitalprovisionedby Liqun Chen. Theauthorsacknowledgetheuseoffrontierlargelanguagemod-
els (including Gemini3.1 Pro, GPT-5.1, andthe Claude4.6 series) asthecoregenerativeenginesforautonomous
software engineering and quantitative logic formulation, as well as for the drafting, structural refinement, and
languagepolishingofthismanuscriptunderhumansupervision.
Funding and Resource Allocation: This longitudinal research was uniquely self-sustaining. Initial capital de-
ployment and physical compute resources were privately endowed by both authors. The authors acted as the
terminal human-in-the-loop (HITL) execution authorities, maintaining absolute veto power over all fiat transac-
tions. Subsequent operational and research costs were entirely financed by the out-of-sample retained earnings
autonomouslygeneratedbythe OOM-RL-aligned MASduringitsmaturephase.
Acknowledgments: We extend our gratitude to the anonymous institutional market makers and high-frequency
tradingfirms. Theirunyielding, adversarial“peerreview”intheliveorderbooksprovidedtheprecise, stringent
financialfeedback (capitaldepletion) thatservedasthenegativegradientforouragent’salignment. Finally, the
authors wish to dedicate this work to Xing Liu (successfully deployed into the physical world circa Q2 2025).
Thebiologicalinceptionofourmostcherished’long-termalpha’inlate2024 serendipitouslycoincidedwiththe
macro-market inflection point, marking the moment when our capital degradation halted and the system’s true
profitabilitybegan.
Code and Data Availability: The foundational quantitative orchestration framework is open-source and docu-
mentedathttps://Quant Pits.com. However, therawbrokeragestatementsandintradayexecutionlogsare
strictlywithheldfrompublicreleaseduetoproprietaryrisk-managementprotocolsandtheinclusionofsensitive
financialdataassociatedwiththesystem’searly-stageanomaloushigh-turnovercapitaldegradation. Furthermore,
the STDAWmoduleiscurrentlyundergoingstructuralsanitizationandremainsclosed-sourcependingfuturefor-
malization. Forinquiries, researchersmayreachouttothecorrespondingauthorviaai@quantpits.com.
Disclaimer: The frameworks and empirical studies described herein are for theoretical and research purposes.
OOM-RLincursextremeandimmediatereal-worldfinancialrisk. Theauthorsdonotprovideinvestmentadvice,
andemphasizethatdeployingunaligned LLMsinlivemarketsmayresultincriticalcapitaldepletion.
References
[1] Bouzenia, I., Devanbu, P., and Pradel, M.(2025).Repairagent:Anautonomous, llm-basedagentforprogram
repair. 2025 IEEE/ACM47 th International Conferenceon Software Engineering (ICSE),2188-2200.
[2] Chen, M., Tworek, J., Jun, H., et al. (2021). Evaluating large language models trained on code. ar Xiv
preprintar Xiv:2107.03374.
[3] Fanous, A., Goldberg, J., Agarwal, A., etal.(2025). Syceval: Evaluatingllmsycophancy. Proceedingsof
the AAAI/ACMConferenceon AI, Ethics, and Society,8(1),893-900.
[4] He, J., Treude, C., and Lo, D.(2025). Llm-basedmulti-agentsystemsforsoftwareengineering: Literature
review, vision, and the road ahead. ACM Transactions on Software Engineering and Methodology, 34(5),
1-30.
[5] Kearns, M., and Nevmyvaka, Y. (2013). Machine learning for market microstructure and high frequency
trading. Highfrequencytrading: Newrealitiesfortraders, markets, andregulators,72,1877-1901.
[6] Kenton, Z., Siegel, N. Y., Krama´r, J., et al. (2024). On scalable oversight with weak llms judging strong
llms. Advancesin Neural Information Processing Systems,37,75229-75276.
[7] Kim, S., and Khashabi, D.(2025).Challengingthe Evaluator:LLMSycophancy Under User Rebuttal.ar Xiv
preprintar Xiv:2509.16533.
[8] Krakovna, V., Uesato, J., Mikulik, V., et al. (2020). Specification gaming: the flip side of AI ingenuity.
Deep Mind Blog,3,40-53.
12


<!-- Page 13 -->

[9] Lee, H., Phatale, S., Mansoor, H., etal.(2023). Rlaif: Scalingreinforcementlearningfromhumanfeedback
withaifeedback. ar Xivpreprintar Xiv:2309.00267.
[10] Liu, J., Shen, Z., He, Y., etal.(2021). Towardsout-of-distributiongeneralization: Asurvey. ar Xivpreprint
ar Xiv:2108.13624.
[11] Mac Diarmid, M., Wright, B., Uesato, J., etal.(2025). Naturalemergentmisalignmentfromrewardhacking
inproductionrl. ar Xivpreprintar Xiv:2511.18397.
[12] Marchand, R., Cathain, A.O., Wynne, J., etal.(2026). Quantifying Frontier LLMCapabilitiesfor Container
Sandbox Escape. ar Xivpreprintar Xiv:2603.02277.
[13] Mathews, N.S., and Nagappan, M.(2024). Test-drivendevelopmentandllm-basedcodegeneration. Pro-
ceedingsofthe39 th IEEE/ACMInternational Conferenceon Automated Software Engineering,1583-1594.
[14] Padakandla, S., KJ, P., and Bhatnagar, S. (2020). Reinforcement learning algorithm for non-stationary
environments. Applied Intelligence,50(11),3590-3606.
[15] Perez, E., Ringer, S., Lukosiute, K., etal.(2023).Discoveringlanguagemodelbehaviorswithmodel-written
evaluations. Findingsofthe Associationfor Computational Linguistics: ACL2023,13387-13434.
[16] Rabin, R., Hostetler, J., Mc Gregor, S., et al. (2025). Sandboxeval: Towards securing test environment for
untrustedcode. ar Xivpreprintar Xiv:2504.00018.
[17] Skalse, J., Howe, N., Krasheninnikov, D., and Krueger, D. (2022). Defining and characterizing reward
gaming. Advancesin Neural Information Processing Systems,35,9460-9471.
[18] Tihanyi, N., Bisztray, T., Ferrag, M. A., et al. (2026). Vulnerability detection: from formal verification to
largelanguagemodelsandhybridapproaches: acomprehensiveoverview. Adversarial Example Detection
and Mitigation Using Machine Learning,33-47.
[19] Wagenmaker, A., Huang, K., Ke, L., etal.(2024). Overcomingthesim-to-realgap: Leveragingsimulation
tolearntoexploreforreal-worldrl. Advancesin Neural Information Processing Systems,37,78715-78765.
[20] Wang, Z., Zhou, S., Fried, D., and Neubig, G. (2023). Execution-based evaluation for open-domain code
generation. Findingsofthe Associationfor Computational Linguistics: EMNLP2023,1271-1290.
[21] Yin, X., Li, X., Ni, C., etal.(2025).Detecting LLM-generated Codewith Subtle Modificationby Adversarial
Training. ar Xivpreprintar Xiv:2507.13123.
[22] Yuan, S.(2025). Mechanismsof High-Frequency Financial Dataon Market Microstructure. Modern Eco-
nomics&Management Forum,6(4),569-572.
[23] Zhang, Z., Wang, C., Wang, Y., etal.(2025). Llmhallucinationsinpracticalcodegeneration: Phenomena,
mechanism, andmitigation. Proceedingsofthe ACMon Software Engineering,2(ISSTA),481-503.
[24] Zheng, L., Chen, J., Yin, Q., etal.(2026). Rethinkingthereliabilityofmulti-agentsystem: Aperspective
frombyzantinefaulttolerance. Proceedingsofthe AAAIConferenceon Artificial Intelligence,40(41),35012-
35020.
13
