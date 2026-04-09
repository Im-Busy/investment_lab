# 151 Trading strategies implemented on python

## Chenjie LI, ESILV Msc Financial Engineering

## July 14, 2024

```
Abstract
I want to firstly thankZura KakushadzeandJuan Andr ́es Serurfor their work on 151
trading strategies. The aim of this work is to reproduce these strategies in Python and have a
clear view on the P&L of each strategy. You can find the Python implementation on: Chenjie’s
Github Trading strategies. For more mathematical and trading description details, please refer to
Zura Kakushadze and Juan Andr ́es Serur ’s work on 151 trading strategies.
```
## 1 introduction

conventional notation of our paper

- STis the stock price at expiration.
- S 0 is the initial stock price.
- Kis the strike price of the call option.
- Cis the premium received
- Dis the premium paid
Capital Gain Strategy: A capital gain strategy is designed to profit from significant movements in
the price of the underlying asset, whether up or down. These strategies typically involve buying op-
tions, which have a limited downside (the premium paid) and unlimited upside potential. The goal is to
achieve a substantial increase in the value of the options as the underlying asset’s price moves favorably.

Net Credit Strategy: A net credit strategy involves selling options to collect premium income.
The initial cash inflow from selling the options creates a net credit. These strategies are often designed
to profit from a neutral or sideways market, where the underlying asset’s price is not expected to move
significantly.

Income Strategy: An income strategy focuses on generating regular income through the collec-
tion of premiums by writing (selling) options. These strategies are typically employed by traders who
expect the underlying asset’s price to remain within a certain range. Income strategies are designed
to take advantage of time decay and the fact that most options expire worthless.


## Contents

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


## 2 151 Trading Strategies

### 2.1 Strategy: Covered Call

### Key Components

- Stock Purchase: Buy the underlying stock at the current priceS 0.
- Call Option Writing: Sell a call option with a strike priceKand receive a premiumC.

### Payoff and P&L

```
Payoff et expiration=ST−S 0 −max(0,ST−K) +C (1)
Max Profit=K−S 0 +C (2)
Max Loss=S 0 −C (3)
```
- Current stock price (S 0 ): 100
- Strike price (K): 105
- Net premium received (C): 5


### 2.2 Strategy: Covered Put

Key Components

- Stock Shorting: Short the underlying stock at the current priceS 0.
- Put Option Writing: Sell a put option with a strike priceKand receive a premiumC.

### Payoff and P&L

```
Payyoff=S 0 −ST−max(0,K−ST) +C (4)
Max Profit=S 0 −K+C (5)
Max Loss= Unlimited (6)
```
- Current stock price (S 0 ): 100
- Strike price (K): 95
- Net premium received (C): 5

### P&L


### 2.3 Strategy: Protective Put

Key Components

- Stock Purchase: Buy the underlying stock at the current priceS 0.
- Put Option Purchase: Buy a put option with a strike priceK≤S 0 and pay a premiumD.

### Payoff and P&L

```
Payyoff=ST−S 0 + max(0,K−ST)−D (7)
Max Profit= Unlimited (8)
Max Loss=S 0 −K+D (9)
```
- Current stock price (S 0 ): 100
- Strike price (K): 95
- Net premium paid (D): 5

### P&L


### 2.4 Strategy: Protective Call

Key Components

- Stock Shorting: Short the underlying stock at the current priceS 0.
- Call Option Purchase: Buy a call option with a strike priceK≥S 0 and pay a premiumD.

### Payoff and P&L

```
Payyoff=S 0 −ST+ max(0,ST−K)−D (10)
Max Profit=S 0 −D (11)
Max Loss=K−S 0 +D (12)
```
- Current stock price (S 0 ): 100
- Strike price (K): 105
- Net premium paid (D): 5

### P&L


### 2.5 Strategy: Bull Call Spread

Key Components

- Long Call Option: Buy a call option with a strike priceK 1 and pay a premiumD.
- Short Call Option: Sell a call option with a higher strike priceK 2 and receive a premium.

### Payoff and P&L

```
Payyoff= (max(0,ST−K 1 ))−(max(0,ST−K 2 ))−D (13)
Max Profit=K 2 −K 1 −D (14)
Max Loss=D (15)
```
- Current stock price (S 0 ): 100
- Strike price of the long call (K1): 95
- Strike price of the short call (K2): 115
- Net premium paid (D): 5


### 2.6 Strategy: Bull Put Spread

Key Components

- Long Put Option: Buy an OTM put option with a strike priceK 1 and pay a premium.
- Short Put Option: Sell an OTM put option with a higher strike priceK 2 and receive a premium
    C.

### Payoff and P&L

```
Payyoff= (max(0,K 1 −ST))−(max(0,K 2 −ST)) +C (16)
Max Profit=C (17)
Max Loss=K 2 −K 1 −C (18)
```
- Current stock price (S 0 ): 100
- Strike price of the long put (K1): 95
- Strike price of the short put (K2): 115
- Net premium recieved (C): 5


### 2.7 Strategy: Bear Call Spread

Key Components

- Long Call Option: Buy an OTM call option with a strike priceK 1 and pay a premium.
- Short Call Option: Sell an OTM call option with a lower strike priceK 2 and receive a premium
    C.

### Payoff and P&L

```
Payyoff= (max(0,ST−K 1 ))−(max(0,ST−K 2 )) +C (19)
Max Profit=C (20)
Max Loss=K 1 −K 2 −C (21)
```
- Current stock price (S 0 ): 100
- Strike price of the long call (K1): 115
- Strike price of the short call (K2): 95
- Net premium recieved (C): 5


### 2.8 Strategy: Bear Put Spread

Key Components

- Long Put Option: Buy a close to ATM put option with a strike priceK 1 and pay a premium
    D.
- Short Put Option: Sell an OTM put option with a lower strike priceK 2 and receive a premium.

### Payoff and P&L

```
Payyoff= (max(0,K 1 −ST))−(max(0,K 2 −ST))−D (22)
Max Profit=K 1 −K 2 −D (23)
Max Loss=D (24)
```
- Current stock price (S 0 ): 100
- Strike price of the long put (K1): 115
- Strike price of the short put (K2): 95
- Net premium Paid (D): 5


### 2.9 Strategy: Long Synthetic Forward

Key Components

- Long Call Option: Buy an ATM call option with a strike priceK=S 0 and pay a premium
    H.
- Short Put Option: Sell an ATM put option with a strike priceK=S 0 and receive a premium.

### Payoff and P&L

```
Payyoff= (max(0,ST−K))−(max(0,K−ST))−H (25)
Max Profit= Unlimited (26)
Max Loss=K+H (27)
```
- Current stock price (S 0 ): 100
- Strike price of option (K): 100
- Net premium Paid or received (H): 5


### 2.10 Strategy: Short Synthetic Forward

Key Components

- Long Put Option: Buy an ATM put option with a strike priceK=S 0 and pay a premiumH.
- Short Call Option: Sell an ATM call option with a strike priceK=S 0 and receive a premium.

### Payoff and P&L

```
Payyoff= (max(0,K−ST))−(max(0,ST−K))−H (28)
Max Profit=K−H (29)
Max Loss= Unlimited (30)
```
- Current stock price (S 0 ): 100
- Strike price of option (K): 100
- Net premium Paid or received (H): 5


### 2.11 Strategy: Long Combo

Key Components

- Long Call Option: Buy an OTM call option with a strike priceK 1 and pay a premiumH.
- Short Put Option: Sell an OTM put option with a strike priceK 2 and receive a premium.

### Payoff and P&L

```
Payyoff= (max(0,ST−K 1 ))−(max(0,K 2 −ST))−H (31)
Max Profit= Unlimited (32)
Max Loss=K 2 +H (33)
```
- Current stock price (S 0 ): 100
- Strike price of long call (K1): 110
- Strike price of short put (K1): 85
- Net premium Paid or received (H): 5


### 2.12 Strategy: Short Combo

Key Components

- Long Put Option: Buy an OTM put option with a strike priceK 1 and pay a premiumH.
- Short Call Option: Sell an OTM call option with a strike priceK 2 and receive a premium.

### Payoff and P&L

```
Payyoff= (max(0,K 1 −ST))−(max(0,ST−K 2 ))−H (34)
Max Profit=K 1 −H (35)
Max Loss= Unlimited (36)
```
- Current stock price (S 0 ): 100
- Strike price of long put (K1): 90
- Strike price of short call (K1): 115
- Net premium Paid or received (H): 5


### 2.13 Strategy: Bull Call Ladder

Key Components

- Long Call Option: Buy a close to ATM call option with a strike priceK 1 and pay a premium
    H.
- Short Call Option 1: Sell an OTM call option with a strike priceK 2 and receive a premium.
- Short Call Option 2: Sell another OTM call option with a higher strike priceK 3 and receive
    a premium.

### Payoff and P&L

```
Payyoff= (max(0,ST−K 1 ))−(max(0,ST−K 2 ))−(max(0,ST−K 3 ))−H (37)
Max Profit=K 2 −K 1 −H (38)
Max Loss= Unlimited (39)
```
- Current stock price (S 0 ): 100
- Strike price of long call (K1): 95
- Strike price of first short call (K2): 105
- Strike price of second short call (K3): 115
- Net premium Paid or received (H): 5


### 2.14 Strategy: Bull Put Ladder

Key Components

- Short Put Option: Sell a close to ATM put option with a strike priceK 1 and receive a
    premium.
- Long Put Option 1: Buy an OTM put option with a lower strike priceK 2 and pay a premium.
- Long Put Option 2: Buy another OTM put option with a lower strike priceK 3 and pay a
    premium.

### Payoff and P&L

```
Payyoff= (max(0,K 3 −ST)) + (max(0,K 2 −ST))−(max(0,K 1 −ST))−H (40)
Max Profit=K 3 +K 2 −K 1 −H (41)
Max Loss=K 1 −K 2 +H (42)
```
- Current stock price (S 0 ): 100
- Strike price of short put (K1): 105
- Strike price of first long put (K2): 95
- Strike price of second long put (K3): 85
- Net premium Paid or received (H): 5


### 2.15 Strategy: Bear Call Ladder

Key Components

- Short Call Option: Sell a close to ATM call option with a strike priceK 1 and receive a
    premium.
- Long Call Option 1: Buy an OTM call option with a higher strike priceK 2 and pay a premium.
- Long Call Option 2: Buy another OTM call option with a higher strike priceK 3 and pay a
    premium.

### Payoff and P&L

```
Payyoff= (max(0,ST−K 3 )) + (max(0,ST−K 2 ))−(max(0,ST−K 1 ))−H (43)
Max Profit= Unlimited (44)
Max Loss=K 2 −K 1 +H (45)
```
- Current stock price (S 0 ): 100
- Strike price of short call (K1): 95
- Strike price of first long call (K2): 105
- Strike price of second long call (K3): 115
- Net premium Paid or received (H): 5


### 2.16 Strategy: Bear Put Ladder

Key Components

- Long Put Option 1: Buy a close to ATM put option with a strike priceK 1 and pay a premium
    H.
- Short Put Option: Sell an OTM put option with a lower strike priceK 2 and receive a premium.
- Short Put Option 2: Sell another OTM put option with a lower strike priceK 3 and receive a
    premium.

### Payoff and P&L

```
Payyoff= (max(0,K 1 −ST))−(max(0,K 2 −ST))−(max(0,K 3 −ST))−H (46)
Max Profit=K 1 −K 2 −H (47)
Max Loss=K 3 +K 2 −K 1 +H (48)
```
- Current stock price (S 0 ): 100
- Strike price of long put (K1): 105
- Strike price of first short put (K2): 95
- Strike price of second short put (K3): 85
- Net premium Paid or received (H): 5


### 2.17 Strategy: Calendar Call Spread

Key Components

- Long Call Option: Buy a close to ATM call option with a strike priceKand TTMT′and
    pay a premiumD.
- Short Call Option: Sell a call option with the same strike priceKand shorter TTMT < T′
    and receive a premium.

### Payoff and P&L

Using the Black-Scholes Model
To model the Calendar Call Spread strategy accurately, we need to account for the value of the long
call option at the expiration of the short call option. The Black-Scholes model is used to calculate the
theoretical price of options, considering factors such as the current stock price (S), the strike price
(K), the time to maturity (T), the risk-free rate (r), and the volatility (σ) of the stock.
The Black-Scholes formula for the price of a call option is given by:
C(S,K,T,r,σ) =S·N(d 1 )−K·e−rT·N(d 2 ) (49)
where
d 1 =ln(S/K) + (r+ 0.^5 σ

(^2) )T
σ√T (50)
d 2 =d 1 −σ√T (51)
Here,N(·) represents the cumulative distribution function of the standard normal distribution.
Parameters Used
For our example, we use the following parameters:

- Current stock price (S 0 ): 50
- Strike price (K): 50
- Time to expiration for the short call (T): 2 months (2/12 years)
- Time to expiration for the long call (T′): 12 months (12/12 years)
- Volatility (σ): 20% (0.2)
- Risk-free rate (r): 3% (0.03)
- Net premium paid (D): 2

Calculating the Value of the Long Call Option
At the expiration of the short call option, the remaining time to expiration for the long call option is
T′−T. Using the Black-Scholes model, we calculate the value of the long call option at this time as:

```
Clong=C(ST,K,T′−T,r,σ) (52)
```

Total PnL Calculation
The total profit or loss (PnL) for the Calendar Call Spread at the expiration of the short call option
is given by:

```
PnL =Clong−Payoffshort call−D (53)
where Payoffshort call= max(ST−K,0) is the payoff of the short call option at expiration.
```

### 2.18 Strategy: Calendar Put Spread

Key Components

- Long Put Option: Buy a close to ATM put option with a strike priceKand TTMT′and pay
    a premiumD.
- Short Put Option: Sell a put option with the same strike priceKand shorter TTMT < T′
    and receive a premium.

### Payoff and P&L

Using the Black-Scholes Model
The Black-Scholes formula for the price of a put option is given by:

```
P(S,K,T,r,σ) =K·e−rT·N(−d 2 )−S·N(−d 1 ) (54)
where
d 1 =ln(S/K) + (r+ 0.^5 σ
```
(^2) )T
σ√T (55)
d 2 =d 1 −σ√T (56)
Here,N(·) represents the cumulative distribution function of the standard normal distribution.
Parameters Used
For our example, we use the following parameters:

- Current stock price (S 0 ): 50
- Strike price (K): 50
- Time to expiration for the short put (T): 2 months (2/12 years)
- Time to expiration for the long put (T′): 12 months (12/12 years)
- Volatility (σ): 20% (0.2)
- Risk-free rate (r): 3% (0.03)
- Net premium paid (D): 2
- Vis the value of the long put option (expiring att=T′)

Calculating the Value of the Long Put Option
At the expiration of the short put option, the remaining time to expiration for the long put option is
T′−T. Using the Black-Scholes model, we calculate the value of the long put option at this time as:

```
Plong=P(ST,K,T′−T,r,σ) (57)
```

Total PnL Calculation
The total profit or loss (PnL) for the Calendar Put Spread at the expiration of the short put option is
given by:

```
PnL =Plong−Payoffshort put−D (58)
where Payoffshort put= max(K−ST,0) is the payoff of the short put option at expiration.
```

### 2.19 Strategy: Diagonal Call Spread

Key Components

- Long Call Option: Buy a deep ITM call option with a strike priceK 1 and TTMT′and pay
    a premiumD.
- Short Call Option: Sell an OTM call option with a higher strike priceK 2 and shorter TTM
    T < T′and receive a premium.

### Payoff and P&L

Using the Black-Scholes Model
The Black-Scholes formula for the price of a call option is given by:

```
C(S,K,T,r,σ) =S·N(d 1 )−K·e−rT·N(d 2 ) (59)
where
d 1 =ln(S/K) + (r+ 0.^5 σ
```
(^2) )T
σ√T (60)
d 2 =d 1 −σ√T (61)
Here,N(·) represents the cumulative distribution function of the standard normal distribution.
Parameters Used
For our example, we use the following parameters:

- Current stock price (S 0 ): 50
- Strike price of long call (K 1 ): 45
- Strike price of short call (K 2 ): 60
- Time to expiration for the short call (T): 2 months (2/12 years)
- Time to expiration for the long call (T′): 12 months (12/12 years)
- Volatility (σ): 20% (0.2)
- Risk-free rate (r): 3% (0.03)
- Net premium paid (D): 2
- Vis the value of the long call option (expiring att=T′)

Calculating the Value of the Long Call Option
At the expiration of the short call option, the remaining time to expiration for the long call option is
T′−T. Using the Black-Scholes model, we calculate the value of the long call option at this time as:

```
Clong=C(ST,K 1 ,T′−T,r,σ) (62)
```

Total PnL Calculation
The total profit or loss (PnL) for the Diagonal Call Spread at the expiration of the short call option
is given by:

```
PnL =Clong−Payoffshort call−D (63)
where Payoffshort call= max(ST−K 2 ,0) is the payoff of the short call option at expiration.
```

### 2.20 Strategy: Diagonal Put Spread

Key Components

- Long Put Option: Buy a deep ITM put option with a strike priceK 1 and TTMT′and pay a
    premiumD.
- Short Put Option: Sell an OTM put option with a lower strike priceK 2 and shorter TTM
    T < T′and receive a premium.

### Payoff and P&L

Using the Black-Scholes Model
To model the Diagonal Put Spread strategy accurately, we need to account for the value of the long
put option at the expiration of the short put option. The Black-Scholes model is used to calculate the
theoretical price of options, considering factors such as the current stock price (S), the strike price
(K), the time to maturity (T), the risk-free rate (r), and the volatility (σ) of the stock.
The Black-Scholes formula for the price of a put option is given by:
P(S,K,T,r,σ) =K·e−rT·N(−d 2 )−S·N(−d 1 ) (64)
where
d 1 =ln(S/K) + (r+ 0.^5 σ

(^2) )T
σ√T (65)
d 2 =d 1 −σ√T (66)
Here,N(·) represents the cumulative distribution function of the standard normal distribution.
Parameters Used
For our example, we use the following parameters:

- Current stock price (S 0 ): 50
- Strike price of long put (K 1 ): 55
- Strike price of short put (K 2 ): 45
- Time to expiration for the short put (T): 2 months (2/12 years)
- Time to expiration for the long put (T′): 12 months (12/12 years)
- Volatility (σ): 20% (0.2)
- Risk-free rate (r): 3% (0.03)
- Net premium paid (D): 2
- Vis the value of the long put option (expiring att=T′)

Calculating the Value of the Long Put Option
At the expiration of the short put option, the remaining time to expiration for the long put option is
T′−T. Using the Black-Scholes model, we calculate the value of the long put option at this time as:

```
Plong=P(ST,K 1 ,T′−T,r,σ) (67)
```

Total PnL Calculation
The total profit or loss (PnL) for the Diagonal Put Spread at the expiration of the short put option is
given by:

```
PnL =Plong−Payoffshort put−D (68)
where Payoffshort put= max(K 2 −ST,0) is the payoff of the short put option at expiration.
```
PnL Diagram
To visualize the PnL of the Diagonal Put Spread strategy, we plot the PnL against different underlying
prices at expiration.


### 2.21 Strategy: Long Straddle

Key Components

- Long Call Option: Buy an ATM call option with a strike priceKand pay a premiumD.
- Long Put Option: Buy an ATM put option with a strike priceKand pay a premiumD.

### Payoff and P&L

```
Payyoff= (ST−K)++ (K−ST)+−D (69)
```
- Sup=K+D
- Sdown=K−D
- Max Profit= unlimited
- Max Loss=D
- Current stock price (S 0 ): 50
- Strike price (K): 50
- Net premium paid (D): 5


### 2.22 Strategy: Long Strangle

Key Components

- Long Call Option: Buy an OTM call option with a strike priceK 1 and pay a premiumD.
- Long Put Option: Buy an OTM put option with a strike priceK 2 and pay a premiumD.

### Payoff and P&L

```
Payyoff= (ST−K 1 )++ (K 2 −ST)+−D (70)
```
- Sup=K 1 +D
- Sdown=K 2 −D
- Max Profit= unlimited
- Max Loss=D
- Current stock price (S 0 ): 50
- lower Strike price (K 1 ): 45
- higher strike price (K 2 ): 55
- Net premium paid (D): 5


### 2.23 Strategy: Long Guts

Key Components

- Long Call Option: Buy an ITM call option with a strike priceK 1 and pay a premiumD.
- Long Put Option: Buy an ITM put option with a strike priceK 2 and pay a premiumD.

### Payoff and P&L

```
Payyoff= (ST−K 1 )++ (K 2 −ST)+−D (71)
```
- Sup=K 1 +D
- Sdown=K 2 −D
- Max Profit= unlimited
- Max Loss=D
- Current stock price (S 0 ): 50
- lower Strike price (K 1 ): 45
- higher strike price (K 2 ): 55
- Net premium paid (D): 5


### 2.24 Strategy: Short Straddle

Key Components

- Short Call Option: Sell an ATM call option with a strike priceKand receive a premiumC.
- Short Put Option: Sell an ATM put option with a strike priceKand receive a premiumC.

### Payoff and P&L

```
Payyoff=−(ST−K)+−(K−ST)++C (72)
```
- Sup=K+C
- Sdown=K−C
- Max Profit=C
- Max Loss= unlimited
- Current stock price (S 0 ): 50
- Strike price (K 1 ): 50
- Net premium received (C): 5


### 2.25 Strategy: Short Strangle

Key Components

- Short Call Option: Sell an OTM call option with a strike priceK 1 and receive a premiumC.
- Short Put Option: Sell an OTM put option with a strike priceK 2 and receive a premiumC.

### Payoff and P&L

```
Payyoff=−(ST−K 1 )+−(K 2 −ST)++C (73)
```
- Sup=K 1 +C
- Sdown=K 2 −C
- Max Profit=C
- Max Loss= unlimited
- Current stock price (S 0 ): 50
- lower Strike price (K 1 ): 45
- higher strike price (K 2 ): 55
- Net premium received (C): 5


### 2.26 Strategy: Short Guts

Key Components

- Short Call Option: Sell an ITM call option with a strike priceK 1 and receive a premiumC.
- Short Put Option: Sell an ITM put option with a strike priceK 2 and receive a premiumC.

### Payoff and P&L

```
Payyoff=−(ST−K 1 )+−(K 2 −ST)++C (74)
```
- Sup=K 1 +C
- Sdown=K 2 −C
- Max Profit=C−(K 2 −K 1 )
- Max Loss= unlimited
- Current stock price (S 0 ): 50
- lower Strike price (K 1 ): 45
- higher strike price (K 2 ): 55
- Net premium received (C): 5


### 2.27 Strategy: Long Call Synthetic Straddle

Key Components

- Short Stock: Short the underlying stock.
- Long Call Options: Buy two ATM call options with a strike priceKand pay a premiumD.

### Payoff and P&L

```
Payyoff=S 0 −ST+ 2×(ST−K)+−D (75)
```
- Sup= 2×K−S 0 +D
- Sdown=S 0 −D
- Max Profit= unlimited
- Max Loss=D−(S 0 −K)
- Current stock price (S 0 ): 50
- Strike price (K): 50
- Net premium paid (D): 5


### 2.28 Strategy: Long Put Synthetic Straddle

Key Components

- Long Stock: Buy the underlying stock.
- Long Put Options: Buy two ATM put options with a strike priceKand pay a premiumD.

### Payoff and P&L

```
Payyoff=ST−S 0 + 2×(K−ST)+−D (76)
```
- Sup=S 0 +D
- Sdown= 2×K−S 0 −D
- Max Profit= unlimited
- Max Loss=D−(K−S 0 )
- Current stock price (S 0 ): 50
- Strike price (K): 50
- Net premium paid (D): 5


### 2.29 Strategy: Short Call Synthetic Straddle

Key Components

- Long Stock: Buy the underlying stock.
- Short Call Options: Sell two ATM call options with a strike priceKand receive a premium
    C.

### Payoff and P&L

```
Payyoff=ST−S 0 − 2 ×(ST−K)++C (77)
```
- Sup= 2×K−S 0 +C
- Sdown=S 0 −C
- Max Profit=K−S 0 +C
- Max Loss= unlimited
- Current stock price (S 0 ): 50
- Strike price (K): 50
- Net premium received (C): 5


### 2.30 Strategy: Short Put Synthetic Straddle

Key Components

- Short Stock: Short the underlying stock.
- Short Put Options: Sell two ATM put options with a strike priceKand receive a premium
    C.

### Payoff and P&L

```
Payyoff=S 0 −ST− 2 ×(K−ST)++C (78)
```
- Sup=S 0 +C
- Sdown= 2×K−S 0 −C
- Max Profit=S 0 −K+C
- Max Loss= unlimited
- Current stock price (S 0 ): 50
- Strike price (K): 50
- Net premium received (C): 5


### 2.31 Strategy: Covered Short Straddle

Key Components

- Long Stock: Buy the underlying stock.
- Short Call Option: Sell an ATM call option with a strike priceKand receive a premiumC.
- Short Put Option: Sell an ATM put option with a strike priceKand receive a premiumC.

### Payoff and P&L

```
Payyoff=ST−S 0 −(ST−K)+−(K−ST)++C (79)
```
- Sup=^12 (S 0 +K−C)
- Max Profit=K−S 0 +C
- Max Loss=S 0 +K−C
- Current stock price (S 0 ): 50
- Strike price (K): 50
- Net premium received (C): 5


### 2.32 Strategy: Covered Short Strangle

Key Components

- Long Stock: Buy the underlying stock.
- Short Call Option: Sell an ATM call option with a strike priceKand receive a premiumC.
- Short Put Option: Sell an OTM put option with a strike priceK′and receive a premiumC.

### Payoff and P&L

```
Payyoff=ST−S 0 −(ST−K)+−(K′−ST)++C (80)
```
- Max Profit=K−S 0 +C
- Max Loss=S 0 +K′−C
- Current stock price (S 0 ): 50
- Strike price for call (K): 50
- Strike price for put (K′): 45
- Net premium received (C): 5


### 2.33 Strategy: Strap

Key Components

- Long Call Options: Buy two ATM call options with a strike priceKand pay a premiumD.
- Long Put Option: Buy an ATM put option with a strike priceKand pay a premiumD.

### Payoff and P&L

```
Payyoff= 2×(ST−K)++ (K−ST)+−D (81)
```
- Sup=K+D 2
- Sdown=K−D
- Max Profit= unlimited
- Max Loss=D
- Current stock price (S 0 ): 50
- Strike price for call (K): 50
- Net premium paid (D): 5


### 2.34 Strategy: Strip

Key Components

- Long Call Option: Buy an ATM call option with a strike priceKand pay a premiumD.
- Long Put Options: Buy two ATM put options with a strike priceKand pay a premiumD.

### Payoff and P&L

```
Payyoff= (ST−K)++ 2×(K−ST)+−D (82)
```
- Sup=K+D
- Sdown=K−D 2
- Max Profit= unlimited
- Max Loss=D
- Current stock price (S 0 ): 50
- Strike price for call (K): 50
- Net premium paid (D): 5


### 2.35 Strategy: Call Ratio Backspread

Key Components

- Short Call Options: SellNSclose to ATM call options with a strike priceK 1 and receive a
    premiumH.
- Long Call Options: BuyNLOTM call options with a strike priceK 2 and pay a premiumH.

### Payoff and P&L

```
Payyoff=NL×(ST−K 2 )+−NS×(ST−K 1 )+−H (83)
```
- Sdown=K 1 +NHS
- Sup=NL×KN^2 −LN−SN×SK^1 +H
- Max Profit= unlimited
- Max Loss=NS×(K 2 −K 1 ) +H
- S 0 = 50 (Current stock price)
- K 1 = 45 (Lower strike price)
- K 2 = 55 (Higher strike price)
- NS= 1 (Number of short options)
- NL= 2 (Number of long options)
- H= 5 (Premium difference)


### 2.36 Strategy: Put Ratio Backspread

Key Components

- Short Put Options: SellNSclose to ATM put options with a strike priceK 1 and receive a
    premiumH.
- Long Put Options: BuyNLOTM put options with a strike priceK 2 and pay a premiumH.

### Payoff and P&L

```
Payyoff=NL×(K 2 −ST)+−NS×(K 1 −ST)+−H (84)
```
- Sup=K 1 +NHS
- Sdown=NL×KN^2 −LN−SN×SK^1 −H
- Max Profit=NL×K 2 −NS×K 1 −H
- Max Loss=NS×(K 1 −K 2 ) +H
- S 0 = 50 (Current stock price)
- K 1 = 45 (Lower strike price)
- K 2 = 55 (Higher strike price)
- NS= 1 (Number of short options)
- NL= 2 (Number of long options)
- H= 5 (Premium difference)


### 2.37 Strategy: Ratio Call Spread

Key Components

- Short Call Options: SellNSclose to ATM call options with a strike priceK 1 and receive a
    premiumH.
- Long Call Options: BuyNLITM call options with a strike priceK 2 and pay a premiumH.

### Payoff and P&L

```
Payyoff=NL×(ST−K 2 )+−NS×(ST−K 1 )++H (85)
```
- Sup=K 2 +NHL
- Sdown=NS×KN^1 −SN−LN×LK^2 +H
- Max Profit=NL×(K 2 −K 1 )−H
- Max Loss= unlimited
- S 0 = 50 (Current stock price)
- K 1 = 45 (Lower strike price)
- K 2 = 55 (Higher strike price)
- NS= 2 (Number of short options)
- NL= 1 (Number of long options)
- H= 5 (Premium difference)


### 2.38 Strategy: Ratio Put Spread

Key Components

- Short Put Options: SellNSclose to ATM put options with a strike priceK 1 and receive a
    premiumH.
- Long Put Options: BuyNLOTM put options with a strike priceK 2 and pay a premiumH.

### Payoff and P&L

```
Payyoff=NL×(K 2 −ST)+−NS×(K 1 −ST)++H (86)
```
- Sup=K 1 +NHS
- Sdown=NL×KN^2 −LN−SN×SK^1 +H
- Max Profit=NL×(K 2 −K 1 ) +H
- Max Loss= unlimited
- S 0 = 50 (Current stock price)
- K 1 = 45 (Lower strike price)
- K 2 = 55 (Higher strike price)
- NS= 1 (Number of short options)
- NL= 2 (Number of long options)
- H= 5 (Premium difference)


### 2.39 Strategy: Long Call Butterfly

Key Components

- Long Call Options: Buy an OTM call option with a strike priceK 1 and pay a premiumD.
- Short Call Options: Sell two ATM call options with a strike priceK 2 and receive a premium
    D.
- Long Call Options: Buy an ITM call option with a strike priceK 3 and pay a premiumD.

### Payoff and P&L

```
Payyoff= (ST−K 1 )++ (ST−K 3 )+− 2 ×(ST−K 2 )+−D (87)
```
- Sup=K 1 −D
- Sdown=K 3 +D
- Max Profit=κ−D
- Max Loss=D
- S 0 = 50 (Current stock price)
- K 1 = 40 (Lower strike price)
- K 2 = 52 (Higher strike price)
- K 3 = 60 (Higher strike price)
- D= 2 (Premium paid)
- Kappa=K 2 −K 1 (strike distance)


### 2.40 Strategy: Modified Call Butterfly

Key Components

- Long Call Options: Buy an OTM call option with a strike priceK 1 and pay a premiumD.
- Short Call Options: Sell two ATM call options with a strike priceK 2 and receive a premium
    D.
- Long Call Options: Buy an ITM call option with a strike priceK 3 and pay a premiumD.

### Payoff and P&L

```
Payyoff= (ST−K 1 )++ (ST−K 3 )+− 2 ×(ST−K 2 )+−D (88)
```
- S∗=K 3 +D
- Max Profit=K 2 −K 3 −D
- Max Loss=D
- S 0 = 50 (Current stock price)
- K 1 = 45 (Lower strike price)
- K 2 = 50 (Higher strike price)
- K 3 = 60 (Higher strike price)
- D= 2 (Premium paid)


### 2.41 Strategy: Long Put Butterfly

Key Components

- Long Put Options: Buy an OTM put option with a strike priceK 1 and pay a premiumD.
- Short Put Options: Sell two ATM put options with a strike priceK 2 and receive a premium
    D.
- Long Put Options: Buy an ITM put option with a strike priceK 3 and pay a premiumD.

### Payoff and P&L

```
Payyoff= (K 1 −ST)++ (K 3 −ST)+− 2 ×(K 2 −ST)+−D (89)
```
- Sup=K 3 −D
- Sdown=K 1 +D
- Max Profit=κ−D
- Max Loss=D
- S 0 = 50 (Current stock price)
- K 1 = 40 (Lower strike price)
- K 2 = 52 (Higher strike price)
- K 3 = 60 (Higher strike price)
- D= 2 (Premium paid)
- Kappa=K 2 −K 1 (strike distance)


### 2.42 Strategy: Modified Put Butterfly

Key Components

- Long Put Options: Buy an OTM put option with a strike priceK 1 and pay a premiumH.
- Short Put Options: Sell two ATM put options with a strike priceK 2 and receive a premium
    H.
- Long Put Options: Buy an ITM put option with a strike priceK 3 and pay a premiumH.

### Payoff and P&L

```
Payyoff= (K 1 −ST)++ (K 3 −ST)+− 2 ×(K 2 −ST)+−H (90)
```
- Sdown= 2×K 2 −K 3 +H
- Max Profit=K 3 −K 2 −H
- Max Loss= 2×K 2 −K 1 −K 3 +H
- S 0 = 50 (Current stock price)
- K 1 = 45 (Lower strike price)
- K 2 = 50 (Higher strike price)
- K 3 = 53 (Higher strike price)
- H= 5 (Premium paid)


## References

Kakushadze, Zura and Serur, Juan Andr ́es, 151 Trading Strategies (August 17, 2018)
Z. Kakushadze and J.A. Serur. 151 Trading Strategies. Cham, Switzerland: Palgrave Macmillan, an
imprint of Springer Nature, 1st Edition (2018), XX, 480 pp; ISBN 978-3-030-02791-9
Available at SSRN: https://ssrn.com/abstract=3247865