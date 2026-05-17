# 151TradingStrategies_49Pages

> *Source PDF: 151TradingStrategies_49Pages.pdf*
> *Extraction: Combined — markitdown raw text (base) + previous extraction supplements*

---

|     | 151 | Trading |     | strategies | implemented   |             | on python |
| --- | --- | ------- | --- | ---------- | ------------- | ----------- | --------- |
|     |     | Chenjie |     | LI, ESILV  | Msc Financial | Engineering |           |
|     |     |         |     | July       | 14, 2024      |             |           |
Abstract
I want to firstly thank Zura Kakushadze and Juan Andr´es Serur for their work on 151
trading strategies. The aim of this work is to reproduce these strategies in Python and have a
clear view on the P&L of each strategy. You can find the Python implementation on: Chenjie’s
GithubTradingstrategies. Formoremathematicalandtradingdescriptiondetails,pleasereferto
| Zura | Kakushadze | and | Juan Andr´es | Serur ’s | work on 151 trading | strategies. |     |
| ---- | ---------- | --- | ------------ | -------- | ------------------- | ----------- | --- |
1 introduction
| conventional | notation |     | of our | paper |     |     |     |
| ------------ | -------- | --- | ------ | ----- | --- | --- | --- |
•
| S T | is the stock | price | at expiration. |     |     |     |     |
| --- | ------------ | ----- | -------------- | --- | --- | --- | --- |
•
| S 0 is | the initial | stock | price. |     |     |     |     |
| ------ | ----------- | ----- | ------ | --- | --- | --- | --- |
•
| K is | the strike | price | of the call | option. |     |     |     |
| ---- | ---------- | ----- | ----------- | ------- | --- | --- | --- |
•
| C is | the premium |     | received |     |     |     |     |
| ---- | ----------- | --- | -------- | --- | --- | --- | --- |
•
| D is | the premium |     | paid |     |     |     |     |
| ---- | ----------- | --- | ---- | --- | --- | --- | --- |
Capital Gain Strategy: A capital gain strategy is designed to profit from significant movements in
the price of the underlying asset, whether up or down. These strategies typically involve buying op-
tions,whichhavealimiteddownside(thepremiumpaid)andunlimitedupsidepotential. Thegoalisto
achieveasubstantialincreaseinthevalueoftheoptionsastheunderlyingasset’spricemovesfavorably.
Net Credit Strategy: A net credit strategy involves selling options to collect premium income.
Theinitialcashinflowfromsellingtheoptionscreatesanetcredit. Thesestrategiesareoftendesigned
toprofitfromaneutralorsidewaysmarket,wheretheunderlyingasset’spriceisnotexpectedtomove
significantly.
Income Strategy: An income strategy focuses on generating regular income through the collec-
tion of premiums by writing (selling) options. These strategies are typically employed by traders who
expect the underlying asset’s price to remain within a certain range. Income strategies are designed
to take advantage of time decay and the fact that most options expire worthless.
1

Contents
1 introduction 1
2 151 Trading Strategies 3
2.1 Strategy: Covered Call . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3
2.2 Strategy: Covered Put . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
2.3 Strategy: Protective Put . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
2.4 Strategy: Protective Call . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
2.5 Strategy: Bull Call Spread . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
2.6 Strategy: Bull Put Spread . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
2.7 Strategy: Bear Call Spread . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9
2.8 Strategy: Bear Put Spread . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10
2.9 Strategy: Long Synthetic Forward . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
2.10 Strategy: Short Synthetic Forward . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12
2.11 Strategy: Long Combo . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
2.12 Strategy: Short Combo . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14
2.13 Strategy: Bull Call Ladder . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
2.14 Strategy: Bull Put Ladder . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
2.15 Strategy: Bear Call Ladder . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
2.16 Strategy: Bear Put Ladder . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
2.17 Strategy: Calendar Call Spread . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
2.18 Strategy: Calendar Put Spread . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
2.19 Strategy: Diagonal Call Spread . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23
2.20 Strategy: Diagonal Put Spread . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25
2.21 Strategy: Long Straddle . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27
2.22 Strategy: Long Strangle . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28
2.23 Strategy: Long Guts . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 29
2.24 Strategy: Short Straddle . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 30
2.25 Strategy: Short Strangle . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 31
2.26 Strategy: Short Guts . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 32
2.27 Strategy: Long Call Synthetic Straddle. . . . . . . . . . . . . . . . . . . . . . . . . . . . 33
2.28 Strategy: Long Put Synthetic Straddle . . . . . . . . . . . . . . . . . . . . . . . . . . . . 34
2.29 Strategy: Short Call Synthetic Straddle . . . . . . . . . . . . . . . . . . . . . . . . . . . 35
2.30 Strategy: Short Put Synthetic Straddle . . . . . . . . . . . . . . . . . . . . . . . . . . . 36
2.31 Strategy: Covered Short Straddle . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 37
2.32 Strategy: Covered Short Strangle . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 38
2.33 Strategy: Strap . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 39
2.34 Strategy: Strip . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 40
2.35 Strategy: Call Ratio Backspread . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 41
2.36 Strategy: Put Ratio Backspread . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 42
2.37 Strategy: Ratio Call Spread . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 43
2.38 Strategy: Ratio Put Spread . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 44
2.39 Strategy: Long Call Butterfly . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 45
2.40 Strategy: Modified Call Butterfly . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 46
2.41 Strategy: Long Put Butterfly . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 47
2.42 Strategy: Modified Put Butterfly . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 48
2

| 2 151         | Trading | Strategies |      |     |     |     |     |     |
| ------------- | ------- | ---------- | ---- | --- | --- | --- | --- | --- |
| 2.1 Strategy: |         | Covered    | Call |     |     |     |     |     |
Key Components
| • Stock | Purchase: | Buy | the underlying | stock | at the | current | price S . |     |
| ------- | --------- | --- | -------------- | ----- | ------ | ------- | --------- | --- |
0
• Call Option Writing: Sell a call option with a strike price K and receive a premium C.
| Payoff | and | P&L    |                 |                |     |          |       |     |
| ------ | --- | ------ | --------------- | -------------- | --- | -------- | ----- | --- |
|        |     | Payoff | et expiration=S |                | −S  | −max(0,S | −K)+C | (1) |
|        |     |        |                 |                | T 0 |          | T     |     |
|        |     |        |                 | Max Profit=K−S |     | +C       |       | (2) |
0
|     |     |     |     | Max Loss=S |     | −C  |     | (3) |
| --- | --- | --- | --- | ---------- | --- | --- | --- | --- |
0
•
| Current | stock | price (S | ): 100 |     |     |     |     |     |
| ------- | ----- | -------- | ------ | --- | --- | --- | --- | --- |
0
•
| Strike | price | (K): 105 |     |     |     |     |     |     |
| ------ | ----- | -------- | --- | --- | --- | --- | --- | --- |
•
| Net | premium | received | (C): 5 |     |     |     |     |     |
| --- | ------- | -------- | ------ | --- | --- | --- | --- | --- |
3

| 2.2 Strategy: | Covered |     | Put |     |     |     |     |
| ------------- | ------- | --- | --- | --- | --- | --- | --- |
Key Components
| • Stock Shorting: | Short | the | underlying | stock | at the current | price S . |     |
| ----------------- | ----- | --- | ---------- | ----- | -------------- | --------- | --- |
0
• Put Option Writing: Sell a put option with a strike price K and receive a premium C.
| Payoff and | P&L |           |     |               |      |     |     |
| ---------- | --- | --------- | --- | ------------- | ---- | --- | --- |
|            |     | Payyoff=S |     | −S −max(0,K−S |      | )+C | (4) |
|            |     |           |     | 0 T           |      | T   |     |
|            |     |           | Max | Profit=S      | −K+C |     | (5) |
0
|     |     |     | Max | Loss=Unlimited |     |     | (6) |
| --- | --- | --- | --- | -------------- | --- | --- | --- |
•
| Current stock | price | (S 0 ): 100 |     |     |     |     |     |
| ------------- | ----- | ----------- | --- | --- | --- | --- | --- |
•
| Strike price | (K): 95 |     |     |     |     |     |     |
| ------------ | ------- | --- | --- | --- | --- | --- | --- |
•
| Net premium | received | (C): | 5   |     |     |     |     |
| ----------- | -------- | ---- | --- | --- | --- | --- | --- |
P&L
4

| 2.3 Strategy: | Protective | Put |     |     |     |     |
| ------------- | ---------- | --- | --- | --- | --- | --- |
Key Components
| • Stock Purchase: | Buy the | underlying | stock | at the current | price S . |     |
| ----------------- | ------- | ---------- | ----- | -------------- | --------- | --- |
0
• Put Option Purchase: Buy a put option with a strike price K ≤S and pay a premium D.
0
| Payoff and P&L |           |     |                  |      |     |     |
| -------------- | --------- | --- | ---------------- | ---- | --- | --- |
|                | Payyoff=S |     | −S +max(0,K−S    |      | )−D | (7) |
|                |           |     | T 0              |      | T   |     |
|                |           | Max | Profit=Unlimited |      |     | (8) |
|                |           | Max | Loss=S           | −K+D |     | (9) |
0
•
| Current stock | price (S 0 ): | 100 |     |     |     |     |
| ------------- | ------------- | --- | --- | --- | --- | --- |
•
| Strike price | (K): 95 |     |     |     |     |     |
| ------------ | ------- | --- | --- | --- | --- | --- |
•
| Net premium | paid (D): 5 |     |     |     |     |     |
| ----------- | ----------- | --- | --- | --- | --- | --- |
P&L
5

| 2.4 Strategy: | Protective | Call |     |     |     |     |
| ------------- | ---------- | ---- | --- | --- | --- | --- |
Key Components
| • Stock Shorting: | Short the | underlying | stock at | the current price | S . |     |
| ----------------- | --------- | ---------- | -------- | ----------------- | --- | --- |
0
• Call Option Purchase: Buy a call option with a strike price K ≥S and pay a premium D.
0
| Payoff and P&L |           |     |             |       |     |      |
| -------------- | --------- | --- | ----------- | ----- | --- | ---- |
|                | Payyoff=S |     | −S +max(0,S | −K)−D |     | (10) |
|                |           |     | 0 T         | T     |     |      |
|                |           | Max | Profit=S    | −D    |     | (11) |
0
|     |     | Max | Loss=K−S | +D  |     | (12) |
| --- | --- | --- | -------- | --- | --- | ---- |
0
•
| Current stock | price (S 0 ): | 100 |     |     |     |     |
| ------------- | ------------- | --- | --- | --- | --- | --- |
•
| Strike price | (K): 105 |     |     |     |     |     |
| ------------ | -------- | --- | --- | --- | --- | --- |
•
| Net premium | paid (D): 5 |     |     |     |     |     |
| ----------- | ----------- | --- | --- | --- | --- | --- |
P&L
6

| 2.5 Strategy: | Bull | Call Spread |     |     |     |
| ------------- | ---- | ----------- | --- | --- | --- |
Key Components
• Long Call Option: Buy a call option with a strike price K and pay a premium D.
1
• Short Call Option: Sell a call option with a higher strike price K and receive a premium.
2
| Payoff and | P&L              |     |                |         |      |
| ---------- | ---------------- | --- | -------------- | ------- | ---- |
|            | Payyoff=(max(0,S |     | −K ))−(max(0,S | −K ))−D | (13) |
|            |                  |     | T 1            | T 2     |      |
|            |                  | Max | Profit=K −K −D |         | (14) |
|            |                  |     | 2 1            |         |      |
|            |                  |     | Max Loss=D     |         | (15) |
•
| Current | stock price | (S 0 ): 100 |     |     |     |
| ------- | ----------- | ----------- | --- | --- | --- |
•
| Strike price | of the long | call (K1): 95 |     |     |     |
| ------------ | ----------- | ------------- | --- | --- | --- |
•
| Strike price | of the short | call (K2): | 115 |     |     |
| ------------ | ------------ | ---------- | --- | --- | --- |
•
| Net premium | paid | (D): 5 |     |     |     |
| ----------- | ---- | ------ | --- | --- | --- |
7

| 2.6 Strategy: | Bull | Put Spread |     |     |     |
| ------------- | ---- | ---------- | --- | --- | --- |
Key Components
• Long Put Option: Buy an OTM put option with a strike price K and pay a premium.
1
• ShortPutOption: SellanOTMputoptionwithahigherstrikepriceK andreceiveapremium
2
C.
| Payoff and | P&L              |           |                    |             |      |
| ---------- | ---------------- | --------- | ------------------ | ----------- | ---- |
|            | Payyoff=(max(0,K |           | 1 −S T ))−(max(0,K | 2 −S T ))+C | (16) |
|            |                  |           | Max Profit=C       |             | (17) |
|            |                  | Max       | Loss=K 2 −K 1 −C   |             | (18) |
| • Current  | stock price      | (S ): 100 |                    |             |      |
0
| • Strike price | of the long  | put (K1): | 95  |     |     |
| -------------- | ------------ | --------- | --- | --- | --- |
| • Strike price | of the short | put (K2): | 115 |     |     |
| • Net premium  | recieved     | (C): 5    |     |     |     |
8

| 2.7 Strategy: | Bear | Call Spread |     |     |     |
| ------------- | ---- | ----------- | --- | --- | --- |
Key Components
• Long Call Option: Buy an OTM call option with a strike price K and pay a premium.
1
• ShortCallOption: SellanOTMcalloptionwithalowerstrikepriceK andreceiveapremium
2
C.
| Payoff and | P&L              |           |                    |             |      |
| ---------- | ---------------- | --------- | ------------------ | ----------- | ---- |
|            | Payyoff=(max(0,S |           | T −K 1 ))−(max(0,S | T −K 2 ))+C | (19) |
|            |                  |           | Max Profit=C       |             | (20) |
|            |                  | Max       | Loss=K 1 −K 2 −C   |             | (21) |
| • Current  | stock price      | (S ): 100 |                    |             |      |
0
| • Strike price | of the long  | call (K1): | 115 |     |     |
| -------------- | ------------ | ---------- | --- | --- | --- |
| • Strike price | of the short | call (K2): | 95  |     |     |
| • Net premium  | recieved     | (C): 5     |     |     |     |
9

| 2.8 Strategy: | Bear | Put Spread |     |     |     |
| ------------- | ---- | ---------- | --- | --- | --- |
Key Components
• Long Put Option: Buy a close to ATM put option with a strike price K and pay a premium
1
D.
• ShortPutOption: SellanOTMputoptionwithalowerstrikepriceK andreceiveapremium.
2
| Payoff and | P&L              |           |                    |             |      |
| ---------- | ---------------- | --------- | ------------------ | ----------- | ---- |
|            | Payyoff=(max(0,K |           | 1 −S T ))−(max(0,K | 2 −S T ))−D | (22) |
|            |                  | Max       | Profit=K 1 −K 2    | −D          | (23) |
|            |                  |           | Max Loss=D         |             | (24) |
| • Current  | stock price      | (S ): 100 |                    |             |      |
0
| • Strike price | of the long  | put (K1): | 115 |     |     |
| -------------- | ------------ | --------- | --- | --- | --- |
| • Strike price | of the short | put (K2): | 95  |     |     |
| • Net premium  | Paid         | (D): 5    |     |     |     |
10

| 2.9 Strategy: | Long Synthetic | Forward |     |     |
| ------------- | -------------- | ------- | --- | --- |
Key Components
• Long Call Option: Buy an ATM call option with a strike price K = S and pay a premium
0
H.
• Short Put Option: SellanATMputoptionwithastrikepriceK =S andreceiveapremium.
0
| Payoff and | P&L              |                      |        |      |
| ---------- | ---------------- | -------------------- | ------ | ---- |
|            | Payyoff=(max(0,S | T −K))−(max(0,K−S    | T ))−H | (25) |
|            |                  | Max Profit=Unlimited |        | (26) |
|            |                  | Max Loss=K+H         |        | (27) |
•
| Current | stock price (S 0 | ): 100 |     |     |
| ------- | ---------------- | ------ | --- | --- |
•
| Strike price | of option (K): | 100 |     |     |
| ------------ | -------------- | --- | --- | --- |
•
| Net premium | Paid or received | (H): 5 |     |     |
| ----------- | ---------------- | ------ | --- | --- |
11

| 2.10 Strategy: | Short | Synthetic Forward |     |     |     |
| -------------- | ----- | ----------------- | --- | --- | --- |
Key Components
• Long Put Option: BuyanATMputoptionwithastrikepriceK =S andpayapremiumH.
0
• Short Call Option: SellanATMcalloptionwithastrikepriceK =S andreceiveapremium.
0
| Payoff and | P&L                |                    |             |        |      |
| ---------- | ------------------ | ------------------ | ----------- | ------ | ---- |
|            | Payyoff=(max(0,K−S |                    | ))−(max(0,S | −K))−H | (28) |
|            |                    |                    | T           | T      |      |
|            |                    | Max Profit=K−H     |             |        | (29) |
|            |                    | Max Loss=Unlimited |             |        | (30) |
| • Current  | stock price (S     | ): 100             |             |        |      |
0
| • Strike price | of option (K):   | 100    |     |     |     |
| -------------- | ---------------- | ------ | --- | --- | --- |
| • Net premium  | Paid or received | (H): 5 |     |     |     |
12

| 2.11 Strategy: | Long | Combo |     |     |     |
| -------------- | ---- | ----- | --- | --- | --- |
Key Components
• Long Call Option: Buy an OTM call option with a strike price K and pay a premium H.
1
• Short Put Option: Sell an OTM put option with a strike price K and receive a premium.
2
| Payoff and | P&L              |     |                  |         |      |
| ---------- | ---------------- | --- | ---------------- | ------- | ---- |
|            | Payyoff=(max(0,S |     | −K ))−(max(0,K   | −S ))−H | (31) |
|            |                  |     | T 1              | 2 T     |      |
|            |                  | Max | Profit=Unlimited |         | (32) |
|            |                  | Max | Loss=K           | +H      | (33) |
2
•
| Current | stock price | (S 0 ): 100 |     |     |     |
| ------- | ----------- | ----------- | --- | --- | --- |
•
| Strike price | of long | call (K1): 110 |     |     |     |
| ------------ | ------- | -------------- | --- | --- | --- |
•
| Strike price | of short | put (K1): 85 |     |     |     |
| ------------ | -------- | ------------ | --- | --- | --- |
•
| Net premium | Paid or | received (H): | 5   |     |     |
| ----------- | ------- | ------------- | --- | --- | --- |
13

| 2.12 Strategy: | Short | Combo |     |     |     |
| -------------- | ----- | ----- | --- | --- | --- |
Key Components
• Long Put Option: Buy an OTM put option with a strike price K and pay a premium H.
1
• Short Call Option: Sell an OTM call option with a strike price K and receive a premium.
2
| Payoff and | P&L              |     |                |         |      |
| ---------- | ---------------- | --- | -------------- | ------- | ---- |
|            | Payyoff=(max(0,K |     | −S ))−(max(0,S | −K ))−H | (34) |
|            |                  |     | 1 T            | T 2     |      |
|            |                  | Max | Profit=K       | −H      | (35) |
1
|     |     | Max | Loss=Unlimited |     | (36) |
| --- | --- | --- | -------------- | --- | ---- |
•
| Current | stock price | (S 0 ): 100 |     |     |     |
| ------- | ----------- | ----------- | --- | --- | --- |
•
| Strike price | of long | put (K1): 90 |     |     |     |
| ------------ | ------- | ------------ | --- | --- | --- |
•
| Strike price | of short | call (K1): 115 |     |     |     |
| ------------ | -------- | -------------- | --- | --- | --- |
•
| Net premium | Paid or | received (H): | 5   |     |     |
| ----------- | ------- | ------------- | --- | --- | --- |
14

| 2.13 | Strategy: | Bull | Call Ladder |     |     |     |
| ---- | --------- | ---- | ----------- | --- | --- | --- |
Key Components
• Long Call Option: Buy a close to ATM call option with a strike price K and pay a premium
1
H.
• Short Call Option 1: Sell an OTM call option with a strike price K and receive a premium.
2
• Short Call Option 2: Sell another OTM call option with a higher strike price K and receive
3
a premium.
| Payoff | and P&L          |     |                |                |         |      |
| ------ | ---------------- | --- | -------------- | -------------- | ------- | ---- |
|        | Payyoff=(max(0,S |     | −K ))−(max(0,S | −K ))−(max(0,S | −K ))−H | (37) |
|        |                  |     | T 1            | T 2            | T 3     |      |
|        |                  |     | Max Profit=K   | −K −H          |         | (38) |
2 1
|           |       |          | Max Loss=Unlimited |     |     | (39) |
| --------- | ----- | -------- | ------------------ | --- | --- | ---- |
| • Current | stock | price (S | ): 100             |     |     |      |
0
| • Strike | price   | of long call   | (K1): 95             |     |     |     |
| -------- | ------- | -------------- | -------------------- | --- | --- | --- |
| • Strike | price   | of first short | call (K2): 105       |     |     |     |
| • Strike | price   | of second      | short call (K3): 115 |     |     |     |
| • Net    | premium | Paid or        | received (H): 5      |     |     |     |
15

| 2.14 | Strategy: | Bull | Put Ladder |     |     |     |     |
| ---- | --------- | ---- | ---------- | --- | --- | --- | --- |
Key Components
• Short Put Option: Sell a close to ATM put option with a strike price K and receive a
1
premium.
• Long Put Option 1: BuyanOTMputoptionwithalowerstrikepriceK andpayapremium.
2
• Long Put Option 2: Buy another OTM put option with a lower strike price K and pay a
3
premium.
| Payoff | and P&L          |     |                |       |             |         |      |
| ------ | ---------------- | --- | -------------- | ----- | ----------- | ------- | ---- |
|        | Payyoff=(max(0,K |     | −S ))+(max(0,K | −S    | ))−(max(0,K | −S ))−H | (40) |
|        |                  |     | 3 T            | 2     | T           | 1 T     |      |
|        |                  |     | Max Profit=K   | +K −K | −H          |         | (41) |
|        |                  |     |                | 3 2   | 1           |         |      |
|        |                  |     | Max Loss=K     | −K    | +H          |         | (42) |
1 2
| • Current | stock | price (S | ): 100 |     |     |     |     |
| --------- | ----- | -------- | ------ | --- | --- | --- | --- |
0
| • Strike | price   | of short      | put (K1): 105     |     |     |     |     |
| -------- | ------- | ------------- | ----------------- | --- | --- | --- | --- |
| • Strike | price   | of first long | put (K2): 95      |     |     |     |     |
| • Strike | price   | of second     | long put (K3): 85 |     |     |     |     |
| • Net    | premium | Paid or       | received (H): 5   |     |     |     |     |
16

| 2.15 | Strategy: | Bear | Call Ladder |     |     |     |
| ---- | --------- | ---- | ----------- | --- | --- | --- |
Key Components
• Short Call Option: Sell a close to ATM call option with a strike price K and receive a
1
premium.
• LongCallOption1: BuyanOTMcalloptionwithahigherstrikepriceK andpayapremium.
2
• Long Call Option 2: Buy another OTM call option with a higher strike price K and pay a
3
premium.
| Payoff | and P&L          |     |                      |                |         |      |
| ------ | ---------------- | --- | -------------------- | -------------- | ------- | ---- |
|        | Payyoff=(max(0,S |     | −K ))+(max(0,S       | −K ))−(max(0,S | −K ))−H | (43) |
|        |                  |     | T 3                  | T 2            | T 1     |      |
|        |                  |     | Max Profit=Unlimited |                |         | (44) |
|        |                  |     | Max Loss=K           | −K +H          |         | (45) |
2 1
| • Current | stock | price (S | ): 100 |     |     |     |
| --------- | ----- | -------- | ------ | --- | --- | --- |
0
| • Strike | price   | of short      | call (K1): 95       |     |     |     |
| -------- | ------- | ------------- | ------------------- | --- | --- | --- |
| • Strike | price   | of first long | call (K2): 105      |     |     |     |
| • Strike | price   | of second     | long call (K3): 115 |     |     |     |
| • Net    | premium | Paid or       | received (H): 5     |     |     |     |
17

| 2.16 | Strategy: | Bear | Put Ladder |     |     |     |     |
| ---- | --------- | ---- | ---------- | --- | --- | --- | --- |
Key Components
• Long Put Option 1: BuyaclosetoATMputoptionwithastrikepriceK andpayapremium
1
H.
• ShortPutOption: SellanOTMputoptionwithalowerstrikepriceK andreceiveapremium.
2
• Short Put Option 2: Sell another OTM put option with a lower strike price K and receive a
3
premium.
| Payoff | and              | P&L |                |     |             |         |      |
| ------ | ---------------- | --- | -------------- | --- | ----------- | ------- | ---- |
|        | Payyoff=(max(0,K |     | −S ))−(max(0,K | −S  | ))−(max(0,K | −S ))−H | (46) |
|        |                  |     | 1 T            | 2   | T           | 3 T     |      |
|        |                  |     | Max Profit=K   | −K  | −H          |         | (47) |
1 2
|           |       |       | Max Loss=K | +K −K | +H  |     | (48) |
| --------- | ----- | ----- | ---------- | ----- | --- | --- | ---- |
|           |       |       |            | 3 2   | 1   |     |      |
| • Current | stock | price | (S ): 100  |       |     |     |      |
0
| • Strike | price   | of long   | put (K1): 105      |     |     |     |     |
| -------- | ------- | --------- | ------------------ | --- | --- | --- | --- |
| • Strike | price   | of first  | short put (K2): 95 |     |     |     |     |
| • Strike | price   | of second | short put (K3): 85 |     |     |     |     |
| • Net    | premium | Paid or   | received (H): 5    |     |     |     |     |
18

| 2.17 | Strategy: |     | Calendar | Call | Spread |     |     |     |     |
| ---- | --------- | --- | -------- | ---- | ------ | --- | --- | --- | --- |
Key Components
• Long Call Option: Buy a close to ATM call option with a strike price K and TTM T′ and
| pay | a premium |     | D.  |     |     |     |     |     |     |
| --- | --------- | --- | --- | --- | --- | --- | --- | --- | --- |
• Short Call Option: Sell a call option with the same strike price K and shorter TTM T < T′
| and    | receive           | a premium. |       |     |     |     |     |     |     |
| ------ | ----------------- | ---------- | ----- | --- | --- | --- | --- | --- | --- |
| Payoff | and               | P&L        |       |     |     |     |     |     |     |
| Using  | the Black-Scholes |            | Model |     |     |     |     |     |     |
To model the Calendar Call Spread strategy accurately, we need to account for the value of the long
call option at the expiration of the short call option. The Black-Scholes model is used to calculate the
theoretical price of options, considering factors such as the current stock price (S), the strike price
(K), the time to maturity (T), the risk-free rate (r), and the volatility (σ) of the stock.
| The | Black-Scholes |     | formula            | for the price | of  | a call | option is | given by: |      |
| --- | ------------- | --- | ------------------ | ------------- | --- | ------ | --------- | --------- | ---- |
|     |               |     | C(S,K,T,r,σ)=S·N(d |               |     |        | )−K·e−rT  | ·N(d )    | (49) |
|     |               |     |                    |               |     |        | 1         | 2         |      |
where
ln(S/K)+(r+0.5σ2)T
|     |     |     |     | d   | =   |     | √   |     | (50) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---- |
1
|     |     |     |     |     |     | σ   | T   |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
√
|     |     |     |     |     | d =d | −σ  | T   |     | (51) |
| --- | --- | --- | --- | --- | ---- | --- | --- | --- | ---- |
2 1
Here, N(·) represents the cumulative distribution function of the standard normal distribution.
| Parameters |          | Used |                   |     |             |     |     |     |     |
| ---------- | -------- | ---- | ----------------- | --- | ----------- | --- | --- | --- | --- |
| For our    | example, | we   | use the following |     | parameters: |     |     |     |     |
•
| Current |     | stock price | (S 0 ): | 50  |     |     |     |     |     |
| ------- | --- | ----------- | ------- | --- | --- | --- | --- | --- | --- |
•
| Strike | price | (K): | 50  |     |     |     |     |     |     |
| ------ | ----- | ---- | --- | --- | --- | --- | --- | --- | --- |
•
| Time | to  | expiration | for the | short     | call (T): | 2 months  | (2/12  | years) |     |
| ---- | --- | ---------- | ------- | --------- | --------- | --------- | ------ | ------ | --- |
| •    |     |            |         |           | (T′):     |           |        |        |     |
| Time | to  | expiration | for the | long call |           | 12 months | (12/12 | years) |     |
•
| Volatility |     | (σ): 20% | (0.2) |     |     |     |     |     |     |
| ---------- | --- | -------- | ----- | --- | --- | --- | --- | --- | --- |
•
| Risk-free |     | rate (r): | 3% (0.03) |     |     |     |     |     |     |
| --------- | --- | --------- | --------- | --- | --- | --- | --- | --- | --- |
•
| Net         | premium | paid      | (D):   | 2    |             |     |     |     |     |
| ----------- | ------- | --------- | ------ | ---- | ----------- | --- | --- | --- | --- |
| Calculating |         | the Value | of the | Long | Call Option |     |     |     |     |
At the expiration of the short call option, the remaining time to expiration for the long call option is
T′−T. Using the Black-Scholes model, we calculate the value of the long call option at this time as:
,K,T′−T,r,σ)
|     |     |     |     | C long | =C(S | T   |     |     | (52) |
| --- | --- | --- | --- | ------ | ---- | --- | --- | --- | ---- |
19

Total PnL Calculation
The total profit or loss (PnL) for the Calendar Call Spread at the expiration of the short call option
is given by:
PnL=C −Payoff −D (53)
long shortcall
where Payoff =max(S −K,0) is the payoff of the short call option at expiration.
shortcall T
20

| 2.18 | Strategy: |     | Calendar |     | Put Spread |     |     |     |     |     |
| ---- | --------- | --- | -------- | --- | ---------- | --- | --- | --- | --- | --- |
Key Components
• Long Put Option: BuyaclosetoATMputoptionwithastrikepriceK andTTMT′ andpay
| a   | premium | D.  |     |     |     |     |     |     |     |     |
| --- | ------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
• Short Put Option: Sell a put option with the same strike price K and shorter TTM T < T′
| and               | receive           | a premium. |     |       |                |        |          |     |     |     |
| ----------------- | ----------------- | ---------- | --- | ----- | -------------- | ------ | -------- | --- | --- | --- |
| Payoff            | and               | P&L        |     |       |                |        |          |     |     |     |
| Using             | the Black-Scholes |            |     | Model |                |        |          |     |     |     |
| The Black-Scholes |                   | formula    | for | the   | price of a put | option | is given | by: |     |     |
P(S,K,T,r,σ)=K·e−rT
|     |     |     |     |     |     |     | ·N(−d 2 )−S·N(−d |     | 1 ) | (54) |
| --- | --- | --- | --- | --- | --- | --- | ---------------- | --- | --- | ---- |
where
ln(S/K)+(r+0.5σ2)T
|     |     |     |     |     | d = |     | √   |     |     | (55) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---- |
1
σ T
√
|     |     |     |     |     | d   | =d −σ | T   |     |     | (56) |
| --- | --- | --- | --- | --- | --- | ----- | --- | --- | --- | ---- |
|     |     |     |     |     | 2   | 1     |     |     |     |      |
Here, N(·) represents the cumulative distribution function of the standard normal distribution.
| Parameters |          | Used        |         |           |             |     |     |     |     |     |
| ---------- | -------- | ----------- | ------- | --------- | ----------- | --- | --- | --- | --- | --- |
| For our    | example, | we          | use the | following | parameters: |     |     |     |     |     |
| • Current  |          | stock price | (S      | ): 50     |             |     |     |     |     |     |
0
| • Strike     | price   | (K):       | 50       |           |                  |           |          |        |     |     |
| ------------ | ------- | ---------- | -------- | --------- | ---------------- | --------- | -------- | ------ | --- | --- |
| • Time       | to      | expiration | for      | the short | put (T):         | 2 months  | (2/12    | years) |     |     |
| • Time       | to      | expiration | for      | the long  | put (T′):        | 12 months | (12/12   | years) |     |     |
| • Volatility |         | (σ): 20%   | (0.2)    |           |                  |           |          |        |     |     |
| • Risk-free  |         | rate (r):  | 3%       | (0.03)    |                  |           |          |        |     |     |
| • Net        | premium | paid       | (D):     | 2         |                  |           |          |        |     |     |
| • V          | is the  | value of   | the long | put       | option (expiring |           | at t=T′) |        |     |     |
| Calculating  |         | the Value  | of       | the       | Long Put         | Option    |          |        |     |     |
At the expiration of the short put option, the remaining time to expiration for the long put option is
T′−T. Using the Black-Scholes model, we calculate the value of the long put option at this time as:
|     |     |     |     |     | P =P(S | ,K,T′−T,r,σ) |     |     |     | (57) |
| --- | --- | --- | --- | --- | ------ | ------------ | --- | --- | --- | ---- |
|     |     |     |     |     | long   | T            |     |     |     |      |
21

Total PnL Calculation
The total profit or loss (PnL) for the Calendar Put Spread at the expiration of the short put option is
given by:
PnL=P −Payoff −D (58)
long shortput
where Payoff =max(K−S ,0) is the payoff of the short put option at expiration.
shortput T
22

| 2.19 | Strategy: |     | Diagonal |     | Call Spread |     |     |     |     |
| ---- | --------- | --- | -------- | --- | ----------- | --- | --- | --- | --- |
Key Components
• Long Call Option: Buy a deep ITM call option with a strike price K and TTM T′ and pay
1
| a   | premium | D.  |     |     |     |     |     |     |     |
| --- | ------- | --- | --- | --- | --- | --- | --- | --- | --- |
• Short Call Option: Sell an OTM call option with a higher strike price K and shorter TTM
2
| T                 | <T′               | and receive | a premium. |       |            |             |          |     |     |
| ----------------- | ----------------- | ----------- | ---------- | ----- | ---------- | ----------- | -------- | --- | --- |
| Payoff            | and               | P&L         |            |       |            |             |          |     |     |
| Using             | the Black-Scholes |             |            | Model |            |             |          |     |     |
| The Black-Scholes |                   | formula     | for        | the   | price of a | call option | is given | by: |     |
)−K·e−rT
|     |     |     |     | C(S,K,T,r,σ)=S·N(d |     |     | 1   | ·N(d 2 ) | (59) |
| --- | --- | --- | --- | ------------------ | --- | --- | --- | -------- | ---- |
where
ln(S/K)+(r+0.5σ2)T
|     |     |     |     |     | d = |     | √   |     | (60) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---- |
1
σ T
√
|     |     |     |     |     | d   | =d −σ | T   |     | (61) |
| --- | --- | --- | --- | --- | --- | ----- | --- | --- | ---- |
|     |     |     |     |     | 2   | 1     |     |     |      |
Here, N(·) represents the cumulative distribution function of the standard normal distribution.
| Parameters |          | Used        |         |           |             |     |     |     |     |
| ---------- | -------- | ----------- | ------- | --------- | ----------- | --- | --- | --- | --- |
| For our    | example, | we          | use the | following | parameters: |     |     |     |     |
| • Current  |          | stock price | (S      | ): 50     |             |     |     |     |     |
0
| • Strike | price | of long | call | (K  | ): 45 |     |     |     |     |
| -------- | ----- | ------- | ---- | --- | ----- | --- | --- | --- | --- |
1
| • Strike | price | of short | call | (K  | ): 60 |     |     |     |     |
| -------- | ----- | -------- | ---- | --- | ----- | --- | --- | --- | --- |
2
| • Time       | to      | expiration | for      | the short | call (T):        | 2 months  | (2/12    | years) |     |
| ------------ | ------- | ---------- | -------- | --------- | ---------------- | --------- | -------- | ------ | --- |
| • Time       | to      | expiration | for      | the long  | call (T′):       | 12 months | (12/12   | years) |     |
| • Volatility |         | (σ): 20%   | (0.2)    |           |                  |           |          |        |     |
| • Risk-free  |         | rate (r):  | 3%       | (0.03)    |                  |           |          |        |     |
| • Net        | premium | paid       | (D):     | 2         |                  |           |          |        |     |
| • V          | is the  | value of   | the long | call      | option (expiring |           | at t=T′) |        |     |
| Calculating  |         | the Value  | of       | the       | Long Call        | Option    |          |        |     |
At the expiration of the short call option, the remaining time to expiration for the long call option is
T′−T. Using the Black-Scholes model, we calculate the value of the long call option at this time as:
|     |     |     |     |     | C =C(S | ,K  | ,T′−T,r,σ) |     | (62) |
| --- | --- | --- | --- | --- | ------ | --- | ---------- | --- | ---- |
|     |     |     |     |     | long   | T   | 1          |     |      |
23

Total PnL Calculation
The total profit or loss (PnL) for the Diagonal Call Spread at the expiration of the short call option
is given by:
PnL=C −Payoff −D (63)
long shortcall
where Payoff =max(S −K ,0) is the payoff of the short call option at expiration.
shortcall T 2
24

| 2.20 | Strategy: |     | Diagonal |     | Put Spread |     |     |     |     |     |     |
| ---- | --------- | --- | -------- | --- | ---------- | --- | --- | --- | --- | --- | --- |
Key Components
• Long Put Option: Buy a deep ITM put option with a strike price K and TTM T′ and pay a
1
| premium |     | D.  |     |     |     |     |     |     |     |     |     |
| ------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
• Short Put Option: Sell an OTM put option with a lower strike price K and shorter TTM
2
| T      | <T′               | and receive | a premium. |       |     |     |     |     |     |     |     |
| ------ | ----------------- | ----------- | ---------- | ----- | --- | --- | --- | --- | --- | --- | --- |
| Payoff | and               | P&L         |            |       |     |     |     |     |     |     |     |
| Using  | the Black-Scholes |             |            | Model |     |     |     |     |     |     |     |
To model the Diagonal Put Spread strategy accurately, we need to account for the value of the long
put option at the expiration of the short put option. The Black-Scholes model is used to calculate the
theoretical price of options, considering factors such as the current stock price (S), the strike price
(K), the time to maturity (T), the risk-free rate (r), and the volatility (σ) of the stock.
| The | Black-Scholes |     | formula             | for | the price | of a | put option | is       | given by: |     |      |
| --- | ------------- | --- | ------------------- | --- | --------- | ---- | ---------- | -------- | --------- | --- | ---- |
|     |               |     | P(S,K,T,r,σ)=K·e−rT |     |           |      | ·N(−d      | )−S·N(−d |           | )   | (64) |
|     |               |     |                     |     |           |      |            | 2        |           | 1   |      |
where
ln(S/K)+(r+0.5σ2)T
|     |     |     |     |     | d = |     | √   |     |     |     | (65) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---- |
1
σ T
√
|     |     |     |     |     |     | d =d | −σ  | T   |     |     | (66) |
| --- | --- | --- | --- | --- | --- | ---- | --- | --- | --- | --- | ---- |
|     |     |     |     |     |     | 2    | 1   |     |     |     |      |
Here, N(·) represents the cumulative distribution function of the standard normal distribution.
| Parameters |          | Used |         |           |             |     |     |     |     |     |     |
| ---------- | -------- | ---- | ------- | --------- | ----------- | --- | --- | --- | --- | --- | --- |
| For our    | example, | we   | use the | following | parameters: |     |     |     |     |     |     |
•
| Current |     | stock price | (S  | 0 ): 50 |     |     |     |     |     |     |     |
| ------- | --- | ----------- | --- | ------- | --- | --- | --- | --- | --- | --- | --- |
•
| Strike | price | of long | put | (K 1 | ): 55 |     |     |     |     |     |     |
| ------ | ----- | ------- | --- | ---- | ----- | --- | --- | --- | --- | --- | --- |
•
| Strike | price | of short | put | (K  | 2 ): 45 |     |     |     |     |     |     |
| ------ | ----- | -------- | --- | --- | ------- | --- | --- | --- | --- | --- | --- |
•
| Time | to  | expiration | for | the short | put | (T):  | 2 months | (2/12  | years) |     |     |
| ---- | --- | ---------- | --- | --------- | --- | ----- | -------- | ------ | ------ | --- | --- |
| •    |     |            |     |           |     | (T′): |          |        |        |     |     |
| Time | to  | expiration | for | the long  | put | 12    | months   | (12/12 | years) |     |     |
•
| Volatility |     | (σ): 20% | (0.2) |     |     |     |     |     |     |     |     |
| ---------- | --- | -------- | ----- | --- | --- | --- | --- | --- | --- | --- | --- |
•
| Risk-free |     | rate (r): | 3%  | (0.03) |     |     |     |     |     |     |     |
| --------- | --- | --------- | --- | ------ | --- | --- | --- | --- | --- | --- | --- |
•
| Net         | premium | paid      | (D):     | 2   |          |           |     |       |     |     |     |
| ----------- | ------- | --------- | -------- | --- | -------- | --------- | --- | ----- | --- | --- | --- |
| •           |         |           |          |     |          |           |     | t=T′) |     |     |     |
| V           | is the  | value of  | the long | put | option   | (expiring | at  |       |     |     |     |
| Calculating |         | the Value | of       | the | Long Put | Option    |     |       |     |     |     |
At the expiration of the short put option, the remaining time to expiration for the long put option is
T′−T. Using the Black-Scholes model, we calculate the value of the long put option at this time as:
,T′−T,r,σ)
|     |     |     |     |     | P long =P(S | T   | ,K 1 |     |     |     | (67) |
| --- | --- | --- | --- | --- | ----------- | --- | ---- | --- | --- | --- | ---- |
25

Total PnL Calculation
The total profit or loss (PnL) for the Diagonal Put Spread at the expiration of the short put option is
given by:
PnL=P −Payoff −D (68)
long shortput
where Payoff =max(K −S ,0) is the payoff of the short put option at expiration.
shortput 2 T
PnL Diagram
TovisualizethePnLoftheDiagonalPutSpreadstrategy,weplotthePnLagainstdifferentunderlying
prices at expiration.
26

2.21 Strategy: Long Straddle
Key Components
• Long Call Option: Buy an ATM call option with a strike price K and pay a premium D.
• Long Put Option: Buy an ATM put option with a strike price K and pay a premium D.
Payoff and P&L
Payyoff=(S −K)++(K−S )+−D (69)
T T
• S =K+D
up
• S =K−D
down
• Max Profit=unlimited
• Max Loss=D
• Current stock price (S ): 50
0
• Strike price (K): 50
• Net premium paid (D): 5
27

| 2.22 Strategy: | Long | Strangle |     |     |     |
| -------------- | ---- | -------- | --- | --- | --- |
Key Components
• Long Call Option: Buy an OTM call option with a strike price K and pay a premium D.
1
• Long Put Option: Buy an OTM put option with a strike price K and pay a premium D.
2
| Payoff and | P&L |            |          |         |      |
| ---------- | --- | ---------- | -------- | ------- | ---- |
|            |     | Payyoff=(S | −K )++(K | −S )+−D | (70) |
|            |     |            | T 1      | 2 T     |      |
•
| S up =K | 1 +D |     |     |     |     |
| ------- | ---- | --- | --- | --- | --- |
•
| S down =K | 2 −D |     |     |     |     |
| --------- | ---- | --- | --- | --- | --- |
•
Max Profit=unlimited
•
Max Loss=D
•
| Current | stock price | (S 0 ): 50 |     |     |     |
| ------- | ----------- | ---------- | --- | --- | --- |
•
| lower Strike | price (K | 1 ): 45 |     |     |     |
| ------------ | -------- | ------- | --- | --- | --- |
•
| higher strike | price (K | 2 ): 55 |     |     |     |
| ------------- | -------- | ------- | --- | --- | --- |
•
| Net premium | paid (D): | 5   |     |     |     |
| ----------- | --------- | --- | --- | --- | --- |
28

| 2.23 Strategy: | Long | Guts |     |     |     |
| -------------- | ---- | ---- | --- | --- | --- |
Key Components
• Long Call Option: Buy an ITM call option with a strike price K and pay a premium D.
1
• Long Put Option: Buy an ITM put option with a strike price K and pay a premium D.
2
| Payoff and | P&L |            |          |         |      |
| ---------- | --- | ---------- | -------- | ------- | ---- |
|            |     | Payyoff=(S | −K )++(K | −S )+−D | (71) |
|            |     |            | T 1      | 2 T     |      |
•
| S up =K | 1 +D |     |     |     |     |
| ------- | ---- | --- | --- | --- | --- |
•
| S down =K | 2 −D |     |     |     |     |
| --------- | ---- | --- | --- | --- | --- |
•
Max Profit=unlimited
•
Max Loss=D
•
| Current | stock price | (S 0 ): 50 |     |     |     |
| ------- | ----------- | ---------- | --- | --- | --- |
•
| lower Strike | price (K | 1 ): 45 |     |     |     |
| ------------ | -------- | ------- | --- | --- | --- |
•
| higher strike | price (K | 2 ): 55 |     |     |     |
| ------------- | -------- | ------- | --- | --- | --- |
•
| Net premium | paid (D): | 5   |     |     |     |
| ----------- | --------- | --- | --- | --- | --- |
29

| 2.24 Strategy: | Short | Straddle |     |     |     |
| -------------- | ----- | -------- | --- | --- | --- |
Key Components
• Short Call Option: Sell an ATM call option with a strike price K and receive a premium C.
• Short Put Option: Sell an ATM put option with a strike price K and receive a premium C.
| Payoff and | P&L |             |           |      |      |
| ---------- | --- | ----------- | --------- | ---- | ---- |
|            |     | Payyoff=−(S | −K)+−(K−S | )++C | (72) |
|            |     |             | T         | T    |      |
•
S up =K+C
•
S down =K−C
•
Max Profit=C
•
Max Loss=unlimited
•
| Current stock | price (S 0 ): | 50  |     |     |     |
| ------------- | ------------- | --- | --- | --- | --- |
•
| Strike price | (K 1 ): 50 |     |     |     |     |
| ------------ | ---------- | --- | --- | --- | --- |
•
| Net premium | received (C): | 5   |     |     |     |
| ----------- | ------------- | --- | --- | --- | --- |
30

| 2.25 Strategy: | Short | Strangle |     |     |     |
| -------------- | ----- | -------- | --- | --- | --- |
Key Components
• Short Call Option: Sell an OTM call option with a strike price K and receive a premium C.
1
• Short Put Option: Sell an OTM put option with a strike price K and receive a premium C.
2
| Payoff and | P&L |             |          |         |      |
| ---------- | --- | ----------- | -------- | ------- | ---- |
|            |     | Payyoff=−(S | −K )+−(K | −S )++C | (73) |
|            |     |             | T 1      | 2 T     |      |
•
| S up =K | 1 +C |     |     |     |     |
| ------- | ---- | --- | --- | --- | --- |
•
| S down =K | 2 −C |     |     |     |     |
| --------- | ---- | --- | --- | --- | --- |
•
Max Profit=C
•
Max Loss=unlimited
•
| Current | stock price | (S 0 ): 50 |     |     |     |
| ------- | ----------- | ---------- | --- | --- | --- |
•
| lower Strike | price (K | 1 ): 45 |     |     |     |
| ------------ | -------- | ------- | --- | --- | --- |
•
| higher strike | price (K | 2 ): 55 |     |     |     |
| ------------- | -------- | ------- | --- | --- | --- |
•
| Net premium | received | (C): 5 |     |     |     |
| ----------- | -------- | ------ | --- | --- | --- |
31

| 2.26 Strategy: | Short | Guts |     |     |     |
| -------------- | ----- | ---- | --- | --- | --- |
Key Components
• Short Call Option: Sell an ITM call option with a strike price K and receive a premium C.
1
• Short Put Option: Sell an ITM put option with a strike price K and receive a premium C.
2
| Payoff and | P&L |             |          |         |      |
| ---------- | --- | ----------- | -------- | ------- | ---- |
|            |     | Payyoff=−(S | −K )+−(K | −S )++C | (74) |
|            |     |             | T 1      | 2 T     |      |
•
| S up =K | 1 +C |     |     |     |     |
| ------- | ---- | --- | --- | --- | --- |
•
| S down =K | 2 −C |     |     |     |     |
| --------- | ---- | --- | --- | --- | --- |
•
| Max Profit=C−(K |     | 2 −K 1 ) |     |     |     |
| --------------- | --- | -------- | --- | --- | --- |
•
Max Loss=unlimited
•
| Current | stock price | (S 0 ): 50 |     |     |     |
| ------- | ----------- | ---------- | --- | --- | --- |
•
| lower Strike | price (K | 1 ): 45 |     |     |     |
| ------------ | -------- | ------- | --- | --- | --- |
•
| higher strike | price (K | 2 ): 55 |     |     |     |
| ------------- | -------- | ------- | --- | --- | --- |
•
| Net premium | received | (C): 5 |     |     |     |
| ----------- | -------- | ------ | --- | --- | --- |
32

| 2.27 Strategy: | Long Call | Synthetic | Straddle |     |
| -------------- | --------- | --------- | -------- | --- |
Key Components
| • Short Stock: | Short the underlying | stock. |     |     |
| -------------- | -------------------- | ------ | --- | --- |
• Long Call Options: Buy two ATM call options with a strike price K and pay a premium D.
| Payoff and P&L |           |     |              |      |
| -------------- | --------- | --- | ------------ | ---- |
|                | Payyoff=S | −S  | +2×(S −K)+−D | (75) |
|                |           | 0   | T T          |      |
•
| S up =2×K−S | 0 +D |     |     |     |
| ----------- | ---- | --- | --- | --- |
•
| S down =S 0 | −D  |     |     |     |
| ----------- | --- | --- | --- | --- |
•
Max Profit=unlimited
•
| Max Loss=D−(S | 0 −K) |     |     |     |
| ------------- | ----- | --- | --- | --- |
•
| Current stock | price (S 0 ): 50 |     |     |     |
| ------------- | ---------------- | --- | --- | --- |
•
| Strike price | (K): 50 |     |     |     |
| ------------ | ------- | --- | --- | --- |
•
| Net premium | paid (D): 5 |     |     |     |
| ----------- | ----------- | --- | --- | --- |
33

| 2.28 Strategy: | Long Put | Synthetic | Straddle |     |     |
| -------------- | -------- | --------- | -------- | --- | --- |
Key Components
| • Long Stock: | Buy the underlying | stock. |     |     |     |
| ------------- | ------------------ | ------ | --- | --- | --- |
• Long Put Options: Buy two ATM put options with a strike price K and pay a premium D.
| Payoff and P&L |           |     |         |      |      |
| -------------- | --------- | --- | ------- | ---- | ---- |
|                | Payyoff=S | −S  | +2×(K−S | )+−D | (76) |
|                |           | T   | 0       | T    |      |
•
| S up =S 0 +D |     |     |     |     |     |
| ------------ | --- | --- | --- | --- | --- |
•
| S down =2×K−S | 0 −D |     |     |     |     |
| ------------- | ---- | --- | --- | --- | --- |
•
| Max Profit=unlimited |     |     |     |     |     |
| -------------------- | --- | --- | --- | --- | --- |
•
| Max Loss=D−(K−S |     | 0 ) |     |     |     |
| --------------- | --- | --- | --- | --- | --- |
•
| Current stock | price (S 0 ): | 50  |     |     |     |
| ------------- | ------------- | --- | --- | --- | --- |
•
| Strike price | (K): 50 |     |     |     |     |
| ------------ | ------- | --- | --- | --- | --- |
•
| Net premium | paid (D): 5 |     |     |     |     |
| ----------- | ----------- | --- | --- | --- | --- |
34

| 2.29 Strategy: | Short | Call | Synthetic | Straddle |     |
| -------------- | ----- | ---- | --------- | -------- | --- |
Key Components
| • Long Stock: | Buy the | underlying | stock. |     |     |
| ------------- | ------- | ---------- | ------ | --- | --- |
• Short Call Options: Sell two ATM call options with a strike price K and receive a premium
C.
| Payoff and | P&L |     |     |     |     |
| ---------- | --- | --- | --- | --- | --- |
−K)++C
|                  |     | Payyoff=S | T −S | 0 −2×(S T | (77) |
| ---------------- | --- | --------- | ---- | --------- | ---- |
| • S =2×K−S       | +C  |           |      |           |      |
| up               | 0   |           |      |           |      |
| • S =S           | −C  |           |      |           |      |
| down             | 0   |           |      |           |      |
| • Max Profit=K−S |     | +C        |      |           |      |
0
| • Max Loss=unlimited |       |          |     |     |     |
| -------------------- | ----- | -------- | --- | --- | --- |
| • Current stock      | price | (S ): 50 |     |     |     |
0
| • Strike price | (K): 50  |        |     |     |     |
| -------------- | -------- | ------ | --- | --- | --- |
| • Net premium  | received | (C): 5 |     |     |     |
35

| 2.30 Strategy: | Short | Put Synthetic |     | Straddle |     |     |
| -------------- | ----- | ------------- | --- | -------- | --- | --- |
Key Components
| • Short Stock: | Short | the underlying | stock. |     |     |     |
| -------------- | ----- | -------------- | ------ | --- | --- | --- |
• Short Put Options: Sell two ATM put options with a strike price K and receive a premium
C.
| Payoff and | P&L |     |     |     |     |     |
| ---------- | --- | --- | --- | --- | --- | --- |
)++C
|                |      | Payyoff=S | 0 −S | T −2×(K−S | T   | (78) |
| -------------- | ---- | --------- | ---- | --------- | --- | ---- |
| • S =S         | +C   |           |      |           |     |      |
| up 0           |      |           |      |           |     |      |
| • S =2×K−S     |      | −C        |      |           |     |      |
| down           | 0    |           |      |           |     |      |
| • Max Profit=S | −K+C |           |      |           |     |      |
0
| • Max Loss=unlimited |       |          |     |     |     |     |
| -------------------- | ----- | -------- | --- | --- | --- | --- |
| • Current stock      | price | (S ): 50 |     |     |     |     |
0
| • Strike price | (K): 50  |        |     |     |     |     |
| -------------- | -------- | ------ | --- | --- | --- | --- |
| • Net premium  | received | (C): 5 |     |     |     |     |
36

| 2.31 Strategy: | Covered | Short Straddle |     |     |     |
| -------------- | ------- | -------------- | --- | --- | --- |
Key Components
| • Long Stock: | Buy the underlying | stock. |     |     |     |
| ------------- | ------------------ | ------ | --- | --- | --- |
• Short Call Option: Sell an ATM call option with a strike price K and receive a premium C.
• Short Put Option: Sell an ATM put option with a strike price K and receive a premium C.
| Payoff and | P&L       |     |               |      |      |
| ---------- | --------- | --- | ------------- | ---- | ---- |
|            | Payyoff=S | −S  | −(S −K)+−(K−S | )++C | (79) |
|            |           | T 0 | T             | T    |      |
| • 1(S      |           |     |               |      |      |
| S up =     | 0 +K−C)   |     |               |      |      |
2
•
| Max Profit=K−S | 0   | +C  |     |     |     |
| -------------- | --- | --- | --- | --- | --- |
•
| Max Loss=S | 0 +K−C |     |     |     |     |
| ---------- | ------ | --- | --- | --- | --- |
•
| Current stock | price (S 0 ): | 50  |     |     |     |
| ------------- | ------------- | --- | --- | --- | --- |
•
| Strike price | (K): 50 |     |     |     |     |
| ------------ | ------- | --- | --- | --- | --- |
•
| Net premium | received (C): | 5   |     |     |     |
| ----------- | ------------- | --- | --- | --- | --- |
37

| 2.32 Strategy: | Covered | Short | Strangle |     |     |     |
| -------------- | ------- | ----- | -------- | --- | --- | --- |
Key Components
| • Long Stock: | Buy the | underlying | stock. |     |     |     |
| ------------- | ------- | ---------- | ------ | --- | --- | --- |
• Short Call Option: Sell an ATM call option with a strike price K and receive a premium C.
• Short Put Option: Sell an OTM put option with a strike price K′ and receive a premium C.
| Payoff and | P&L       |     |     |                |      |      |
| ---------- | --------- | --- | --- | -------------- | ---- | ---- |
|            | Payyoff=S |     | −S  | −(S −K)+−(K′−S | )++C | (80) |
|            |           |     | T 0 | T              | T    |      |
•
| Max Profit=K−S |       | 0 +C |     |     |     |     |
| -------------- | ----- | ---- | --- | --- | --- | --- |
| •              | +K′−C |      |     |     |     |     |
| Max Loss=S     | 0     |      |     |     |     |     |
•
| Current | stock price (S | 0 ): 50 |     |     |     |     |
| ------- | -------------- | ------- | --- | --- | --- | --- |
•
| Strike price | for call (K): | 50  |     |     |     |     |
| ------------ | ------------- | --- | --- | --- | --- | --- |
| •            | (K′):         |     |     |     |     |     |
| Strike price | for put       | 45  |     |     |     |     |
•
| Net premium | received | (C): 5 |     |     |     |     |
| ----------- | -------- | ------ | --- | --- | --- | --- |
38

| 2.33 Strategy: | Strap |     |     |     |     |
| -------------- | ----- | --- | --- | --- | --- |
Key Components
• Long Call Options: Buy two ATM call options with a strike price K and pay a premium D.
• Long Put Option: Buy an ATM put option with a strike price K and pay a premium D.
| Payoff and | P&L |              |           |      |      |
| ---------- | --- | ------------ | --------- | ---- | ---- |
|            |     | Payyoff=2×(S | −K)++(K−S | )+−D | (81) |
|            |     |              | T         | T    |      |
| •          | D   |              |           |      |      |
S up =K+
2
•
S down =K−D
•
Max Profit=unlimited
•
Max Loss=D
•
| Current | stock price (S | 0 ): 50 |     |     |     |
| ------- | -------------- | ------- | --- | --- | --- |
•
| Strike price | for call (K): | 50  |     |     |     |
| ------------ | ------------- | --- | --- | --- | --- |
•
| Net premium | paid (D): | 5   |     |     |     |
| ----------- | --------- | --- | --- | --- | --- |
39

| 2.34 Strategy: | Strip |     |     |     |     |
| -------------- | ----- | --- | --- | --- | --- |
Key Components
• Long Call Option: Buy an ATM call option with a strike price K and pay a premium D.
• Long Put Options: Buy two ATM put options with a strike price K and pay a premium D.
| Payoff and | P&L |            |             |      |      |
| ---------- | --- | ---------- | ----------- | ---- | ---- |
|            |     | Payyoff=(S | −K)++2×(K−S | )+−D | (82) |
|            |     |            | T           | T    |      |
•
S up =K+D
| •   | D   |     |     |     |     |
| --- | --- | --- | --- | --- | --- |
S down =K−
2
•
Max Profit=unlimited
•
Max Loss=D
•
| Current | stock price (S | 0 ): 50 |     |     |     |
| ------- | -------------- | ------- | --- | --- | --- |
•
| Strike price | for call (K): | 50  |     |     |     |
| ------------ | ------------- | --- | --- | --- | --- |
•
| Net premium | paid (D): | 5   |     |     |     |
| ----------- | --------- | --- | --- | --- | --- |
40

| 2.35 Strategy: | Call Ratio | Backspread |     |     |     |
| -------------- | ---------- | ---------- | --- | --- | --- |
Key Components
• Short Call Options: Sell N close to ATM call options with a strike price K and receive a
S 1
| premium H. |     |     |     |     |     |
| ---------- | --- | --- | --- | --- | --- |
• Long Call Options: Buy N OTM call options with a strike price K and pay a premium H.
L 2
| Payoff and P&L |             |         |        |            |      |
| -------------- | ----------- | ------- | ------ | ---------- | ---- |
|                |             |         | )+−N   | )+−H       |      |
|                | Payyoff=N   | L ×(S T | −K 2 S | ×(S T −K 1 | (83) |
| • S =K         | + H         |         |        |            |      |
| down 1         | NS          |         |        |            |      |
| • NL×K         | 2− N × K1+H |         |        |            |      |
| S up =         | S           |         |        |            |      |
N L − N S
•
Max Profit=unlimited
•
| Max Loss=N       | S ×(K 2      | −K 1 )+H |     |     |     |
| ---------------- | ------------ | -------- | --- | --- | --- |
| • S =50 (Current | stock price) |          |     |     |     |
0
•
| K =45 (Lower | strike price) |     |     |     |     |
| ------------ | ------------- | --- | --- | --- | --- |
1
•
| K =55 (Higher | strike price) |     |     |     |     |
| ------------- | ------------- | --- | --- | --- | --- |
2
•
| N =1 (Number | of short | options) |     |     |     |
| ------------ | -------- | -------- | --- | --- | --- |
S
•
| N =2 (Number | of long | options) |     |     |     |
| ------------ | ------- | -------- | --- | --- | --- |
L
•
| H =5 (Premium | difference) |     |     |     |     |
| ------------- | ----------- | --- | --- | --- | --- |
41

| 2.36 Strategy: | Put Ratio | Backspread |     |     |     |
| -------------- | --------- | ---------- | --- | --- | --- |
Key Components
• Short Put Options: Sell N close to ATM put options with a strike price K and receive a
S 1
| premium H. |     |     |     |     |     |
| ---------- | --- | --- | --- | --- | --- |
• Long Put Options: Buy N OTM put options with a strike price K and pay a premium H.
L 2
| Payoff and P&L |             |         |            |        |      |
| -------------- | ----------- | ------- | ---------- | ------ | ---- |
|                |             |         | )+−N       | )+−H   |      |
|                | Payyoff=N   | L ×(K 2 | −S T S ×(K | 1 −S T | (84) |
| • S =K +       | H           |         |            |        |      |
| up 1           | NS          |         |            |        |      |
| • NL×K         | 2− N × K1−H |         |            |        |      |
| S down =       | S           |         |            |        |      |
N L − N S
•
| Max Profit=N | L ×K 2 | −N S ×K 1 −H |     |     |     |
| ------------ | ------ | ------------ | --- | --- | --- |
•
| Max Loss=N       | S ×(K 1      | −K 2 )+H |     |     |     |
| ---------------- | ------------ | -------- | --- | --- | --- |
| • S =50 (Current | stock price) |          |     |     |     |
0
•
| K =45 (Lower | strike price) |     |     |     |     |
| ------------ | ------------- | --- | --- | --- | --- |
1
•
| K =55 (Higher | strike price) |     |     |     |     |
| ------------- | ------------- | --- | --- | --- | --- |
2
•
| N =1 (Number | of short | options) |     |     |     |
| ------------ | -------- | -------- | --- | --- | --- |
S
•
| N =2 (Number | of long | options) |     |     |     |
| ------------ | ------- | -------- | --- | --- | --- |
L
•
| H =5 (Premium | difference) |     |     |     |     |
| ------------- | ----------- | --- | --- | --- | --- |
42

| 2.37 Strategy: | Ratio Call | Spread |     |     |     |
| -------------- | ---------- | ------ | --- | --- | --- |
Key Components
• Short Call Options: Sell N close to ATM call options with a strike price K and receive a
S 1
| premium H. |     |     |     |     |     |
| ---------- | --- | --- | --- | --- | --- |
• Long Call Options: Buy N ITM call options with a strike price K and pay a premium H.
L 2
| Payoff and P&L |             |         |        |            |      |
| -------------- | ----------- | ------- | ------ | ---------- | ---- |
|                |             |         | )+−N   | )++H       |      |
|                | Payyoff=N   | L ×(S T | −K 2 S | ×(S T −K 1 | (85) |
| • S =K +       | H           |         |        |            |      |
| up 2           | NL          |         |        |            |      |
| • NS×K         | 1− N × K2+H |         |        |            |      |
| S down =       | L           |         |        |            |      |
N S − N L
•
| Max Profit=N | L ×(K 2 | −K 1 )−H |     |     |     |
| ------------ | ------- | -------- | --- | --- | --- |
•
Max Loss=unlimited
| • S =50 (Current | stock price) |     |     |     |     |
| ---------------- | ------------ | --- | --- | --- | --- |
0
•
| K =45 (Lower | strike price) |     |     |     |     |
| ------------ | ------------- | --- | --- | --- | --- |
1
•
| K =55 (Higher | strike price) |     |     |     |     |
| ------------- | ------------- | --- | --- | --- | --- |
2
•
| N =2 (Number | of short | options) |     |     |     |
| ------------ | -------- | -------- | --- | --- | --- |
S
•
| N =1 (Number | of long options) |     |     |     |     |
| ------------ | ---------------- | --- | --- | --- | --- |
L
•
| H =5 (Premium | difference) |     |     |     |     |
| ------------- | ----------- | --- | --- | --- | --- |
43

| 2.38 Strategy: | Ratio Put | Spread |     |     |     |
| -------------- | --------- | ------ | --- | --- | --- |
Key Components
• Short Put Options: Sell N close to ATM put options with a strike price K and receive a
S 1
| premium H. |     |     |     |     |     |
| ---------- | --- | --- | --- | --- | --- |
• Long Put Options: Buy N OTM put options with a strike price K and pay a premium H.
L 2
| Payoff and P&L |             |         |            |        |      |
| -------------- | ----------- | ------- | ---------- | ------ | ---- |
|                |             |         | )+−N       | )++H   |      |
|                | Payyoff=N   | L ×(K 2 | −S T S ×(K | 1 −S T | (86) |
| • S =K +       | H           |         |            |        |      |
| up 1           | NS          |         |            |        |      |
| • NL×K         | 2− N × K1+H |         |            |        |      |
| S down =       | S           |         |            |        |      |
N L − N S
•
| Max Profit=N | L ×(K 2 | −K 1 )+H |     |     |     |
| ------------ | ------- | -------- | --- | --- | --- |
•
Max Loss=unlimited
| • S =50 (Current | stock price) |     |     |     |     |
| ---------------- | ------------ | --- | --- | --- | --- |
0
•
| K =45 (Lower | strike price) |     |     |     |     |
| ------------ | ------------- | --- | --- | --- | --- |
1
•
| K =55 (Higher | strike price) |     |     |     |     |
| ------------- | ------------- | --- | --- | --- | --- |
2
•
| N =1 (Number | of short | options) |     |     |     |
| ------------ | -------- | -------- | --- | --- | --- |
S
•
| N =2 (Number | of long options) |     |     |     |     |
| ------------ | ---------------- | --- | --- | --- | --- |
L
•
| H =5 (Premium | difference) |     |     |     |     |
| ------------- | ----------- | --- | --- | --- | --- |
44

| 2.39 Strategy: | Long | Call Butterfly |     |     |     |
| -------------- | ---- | -------------- | --- | --- | --- |
Key Components
• Long Call Options: Buy an OTM call option with a strike price K and pay a premium D.
1
• Short Call Options: Sell two ATM call options with a strike price K and receive a premium
2
D.
• Long Call Options: Buy an ITM call option with a strike price K and pay a premium D.
3
| Payoff and       | P&L        |        |         |        |      |
| ---------------- | ---------- | ------ | ------- | ------ | ---- |
|                  |            | )++(S  | )+−2×(S | )+−D   |      |
|                  | Payyoff=(S | T −K 1 | T −K 3  | T −K 2 | (87) |
| • S =K           | −D         |        |         |        |      |
| up 1             |            |        |         |        |      |
| • S =K           | +D         |        |         |        |      |
| down             | 3          |        |         |        |      |
| • Max Profit=κ−D |            |        |         |        |      |
| • Max Loss=D     |            |        |         |        |      |
| • S =50 (Current | stock      | price) |         |        |      |
0
| • K =40 | (Lower strike | price) |     |     |     |
| ------- | ------------- | ------ | --- | --- | --- |
1
| • K =52 | (Higher strike | price) |     |     |     |
| ------- | -------------- | ------ | --- | --- | --- |
2
| • K =60 | (Higher strike | price) |     |     |     |
| ------- | -------------- | ------ | --- | --- | --- |
3
| • D =2 (Premium | paid)      |           |     |     |     |
| --------------- | ---------- | --------- | --- | --- | --- |
| • Kappa=K       | −K (strike | distance) |     |     |     |
2 1
45

| 2.40 Strategy: | Modified | Call Butterfly |     |     |     |
| -------------- | -------- | -------------- | --- | --- | --- |
Key Components
• Long Call Options: Buy an OTM call option with a strike price K and pay a premium D.
1
• Short Call Options: Sell two ATM call options with a strike price K and receive a premium
2
D.
• Long Call Options: Buy an ITM call option with a strike price K and pay a premium D.
3
| Payoff and P&L   |            |        |         |        |      |
| ---------------- | ---------- | ------ | ------- | ------ | ---- |
|                  |            | )++(S  | )+−2×(S | )+−D   |      |
|                  | Payyoff=(S | T −K 1 | T −K 3  | T −K 2 | (88) |
| • S =K +D        |            |        |         |        |      |
| ∗ 3              |            |        |         |        |      |
| • Max Profit=K   | −K         | −D     |         |        |      |
|                  | 2 3        |        |         |        |      |
| • Max Loss=D     |            |        |         |        |      |
| • S =50 (Current | stock      | price) |         |        |      |
0
| • K =45 (Lower | strike price) |     |     |     |     |
| -------------- | ------------- | --- | --- | --- | --- |
1
| • K =50 (Higher | strike | price) |     |     |     |
| --------------- | ------ | ------ | --- | --- | --- |
2
| • K =60 (Higher | strike | price) |     |     |     |
| --------------- | ------ | ------ | --- | --- | --- |
3
| • D =2 (Premium | paid) |     |     |     |     |
| --------------- | ----- | --- | --- | --- | --- |
46

| 2.41 Strategy: | Long | Put Butterfly |     |     |     |
| -------------- | ---- | ------------- | --- | --- | --- |
Key Components
• Long Put Options: Buy an OTM put option with a strike price K and pay a premium D.
1
• Short Put Options: Sell two ATM put options with a strike price K and receive a premium
2
D.
• Long Put Options: Buy an ITM put option with a strike price K and pay a premium D.
3
| Payoff and       | P&L        |        |         |        |      |
| ---------------- | ---------- | ------ | ------- | ------ | ---- |
|                  |            | )++(K  | )+−2×(K | )+−D   |      |
|                  | Payyoff=(K | 1 −S T | 3 −S T  | 2 −S T | (89) |
| • S =K           | −D         |        |         |        |      |
| up 3             |            |        |         |        |      |
| • S =K           | +D         |        |         |        |      |
| down             | 1          |        |         |        |      |
| • Max Profit=κ−D |            |        |         |        |      |
| • Max Loss=D     |            |        |         |        |      |
| • S =50 (Current | stock      | price) |         |        |      |
0
| • K =40 | (Lower strike | price) |     |     |     |
| ------- | ------------- | ------ | --- | --- | --- |
1
| • K =52 | (Higher strike | price) |     |     |     |
| ------- | -------------- | ------ | --- | --- | --- |
2
| • K =60 | (Higher strike | price) |     |     |     |
| ------- | -------------- | ------ | --- | --- | --- |
3
| • D =2 (Premium | paid)      |           |     |     |     |
| --------------- | ---------- | --------- | --- | --- | --- |
| • Kappa=K       | −K (strike | distance) |     |     |     |
2 1
47

| 2.42 Strategy: | Modified | Put Butterfly |     |     |     |
| -------------- | -------- | ------------- | --- | --- | --- |
Key Components
• Long Put Options: Buy an OTM put option with a strike price K and pay a premium H.
1
• Short Put Options: Sell two ATM put options with a strike price K and receive a premium
2
H.
• Long Put Options: Buy an ITM put option with a strike price K and pay a premium H.
3
| Payoff and       | P&L          |        |         |        |      |
| ---------------- | ------------ | ------ | ------- | ------ | ---- |
|                  |              | )++(K  | )+−2×(K | )+−H   |      |
|                  | Payyoff=(K   | 1 −S T | 3 −S T  | 2 −S T | (90) |
| • S =2×K         | −K +H        |        |         |        |      |
| down             | 2 3          |        |         |        |      |
| • Max Profit=K   | −K           | −H     |         |        |      |
|                  | 3 2          |        |         |        |      |
| • Max Loss=2×K   | −K           | −K +H  |         |        |      |
|                  | 2            | 1 3    |         |        |      |
| • S =50 (Current | stock price) |        |         |        |      |
0
| • K =45 (Lower | strike price) |     |     |     |     |
| -------------- | ------------- | --- | --- | --- | --- |
1
| • K =50 (Higher | strike price) |     |     |     |     |
| --------------- | ------------- | --- | --- | --- | --- |
2
| • K =53 (Higher | strike price) |     |     |     |     |
| --------------- | ------------- | --- | --- | --- | --- |
3
| • H =5 (Premium | paid) |     |     |     |     |
| --------------- | ----- | --- | --- | --- | --- |
48

References
Kakushadze, Zura and Serur, Juan Andr´es, 151 Trading Strategies (August 17, 2018)
Z. Kakushadze and J.A. Serur. 151 Trading Strategies. Cham, Switzerland: Palgrave Macmillan, an
imprint of Springer Nature, 1st Edition (2018), XX, 480 pp; ISBN 978-3-030-02791-9
Available at SSRN: https://ssrn.com/abstract=3247865
49

---

## Content from Previous Extraction (not in markitdown output)

### Visual/Chart/Graph Descriptions

#### 1

### Visual/Chart/Graph Descriptions

#### 2

```
PnL =Plong−Payoffshort put−D (68)
where Payoffshort put= max(K 2 −ST,0) is the payoff of the short put option at expiration.
```
PnL Diagram
To visualize the PnL of the Diagonal Put Spread strategy, we plot the PnL against different underlying
prices at expiration.

### Additional Content

#### 1

# 151TradingStrategies_49Pages

#### 2

> *Source PDF: 151TradingStrategies_49Pages.pdf*
> *Extraction: Combined — markitdown raw text (base) + previous extraction supplements*

#### 3

## Content from Previous Extraction (not in markitdown output)

#### 4

# 151 Trading strategies implemented on python

#### 5

## Chenjie LI, ESILV Msc Financial Engineering

#### 6

```
Abstract
I want to firstly thankZura KakushadzeandJuan Andr ́es Serurfor their work on 151
trading strategies. The aim of this work is to reproduce these strategies in Python and have a
clear view on the P&L of each strategy. You can find the Python implementation on: Chenjie’s
Github Trading strategies. For more mathematical and trading description details, please refer to
Zura Kakushadze and Juan Andr ́es Serur ’s work on 151 trading strategies.
```
## 1 introduction

#### 7

conventional notation of our paper

#### 8

- STis the stock price at expiration.
- S 0 is the initial stock price.
- Kis the strike price of the call option.
- Cis the premium received
- Dis the premium paid
Capital Gain Strategy: A capital gain strategy is designed to profit from significant movements in
the price of the underlying asset, whether up or down. These strategies typically involve buying op-
tions, which have a limited downside (the premium paid) and unlimited upside potential. The goal is to
achieve a substantial increase in the value of the options as the underlying asset’s price moves favorably.

#### 9

Net Credit Strategy: A net credit strategy involves selling options to collect premium income.
The initial cash inflow from selling the options creates a net credit. These strategies are often designed
to profit from a neutral or sideways market, where the underlying asset’s price is not expected to move
significantly.

#### 10

- 1 introduction
- 2 151 Trading Strategies
   - 2.1 Strategy: Covered Call
   - 2.2 Strategy: Covered Put
   - 2.3 Strategy: Protective Put
   - 2.4 Strategy: Protective Call
   - 2.5 Strategy: Bull Call Spread
   - 2.6 Strategy: Bull Put Spread
   - 2.7 Strategy: Bear Call Spread
   - 2.8 Strategy: Bear Put Spread
   - 2.9 Strategy: Long Synthetic Forward
   - 2.10 Strategy: Short Synthetic Forward
   - 2.11 Strategy: Long Combo
   - 2.12 Strategy: Short Combo
   - 2.13 Strategy: Bull Call Ladder
   - 2.14 Strategy: Bull Put Ladder
   - 2.15 Strategy: Bear Call Ladder
   - 2.16 Strategy: Bear Put Ladder
   - 2.17 Strategy: Calendar Call Spread
   - 2.18 Strategy: Calendar Put Spread
   - 2.19 Strategy: Diagonal Call Spread
   - 2.20 Strategy: Diagonal Put Spread
   - 2.21 Strategy: Long Straddle
   - 2.22 Strategy: Long Strangle
   - 2.23 Strategy: Long Guts
   - 2.24 Strategy: Short Straddle
   - 2.25 Strategy: Short Strangle
   - 2.26 Strategy: Short Guts
   - 2.27 Strategy: Long Call Synthetic Straddle
   - 2.28 Strategy: Long Put Synthetic Straddle
   - 2.29 Strategy: Short Call Synthetic Straddle
   - 2.30 Strategy: Short Put Synthetic Straddle
   - 2.31 Strategy: Covered Short Straddle
   - 2.32 Strategy: Covered Short Strangle
   - 2.33 Strategy: Strap
   - 2.34 Strategy: Strip
   - 2.35 Strategy: Call Ratio Backspread
   - 2.36 Strategy: Put Ratio Backspread
   - 2.37 Strategy: Ratio Call Spread
   - 2.38 Strategy: Ratio Put Spread
   - 2.39 Strategy: Long Call Butterfly
   - 2.40 Strategy: Modified Call Butterfly
   - 2.41 Strategy: Long Put Butterfly
   - 2.42 Strategy: Modified Put Butterfly

#### 11

### 2.1 Strategy: Covered Call

#### 12

- Stock Purchase: Buy the underlying stock at the current priceS 0.
- Call Option Writing: Sell a call option with a strike priceKand receive a premiumC.

#### 13

```
Payoff et expiration=ST−S 0 −max(0,ST−K) +C (1)
Max Profit=K−S 0 +C (2)
Max Loss=S 0 −C (3)
```
- Current stock price (S 0 ): 100
- Strike price (K): 105
- Net premium received (C): 5

#### 14

- Stock Shorting: Short the underlying stock at the current priceS 0.
- Put Option Writing: Sell a put option with a strike priceKand receive a premiumC.

#### 15

```
Payyoff=S 0 −ST−max(0,K−ST) +C (4)
Max Profit=S 0 −K+C (5)
Max Loss= Unlimited (6)
```
- Current stock price (S 0 ): 100
- Strike price (K): 95
- Net premium received (C): 5

#### 16

### 2.3 Strategy: Protective Put

#### 17

- Stock Purchase: Buy the underlying stock at the current priceS 0.
- Put Option Purchase: Buy a put option with a strike priceK≤S 0 and pay a premiumD.

#### 18

```
Payyoff=ST−S 0 + max(0,K−ST)−D (7)
Max Profit= Unlimited (8)
Max Loss=S 0 −K+D (9)
```
- Current stock price (S 0 ): 100
- Strike price (K): 95
- Net premium paid (D): 5

#### 19

### 2.4 Strategy: Protective Call

#### 20

- Stock Shorting: Short the underlying stock at the current priceS 0.
- Call Option Purchase: Buy a call option with a strike priceK≥S 0 and pay a premiumD.
