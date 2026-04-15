"""
Manual parameter grid search for 4 passing patterns on SPY and QQQ.
"""

import sys, json, time, math, warnings, itertools
from pathlib import Path
warnings.filterwarnings('ignore')
import pandas as pd

pr = Path(__file__).parent.parent
if str(pr) not in sys.path: sys.path.insert(0, str(pr))

from backtesting import Backtest
from src.strategies.backtest_py.multi_pattern_strategy_optimized import MultiPatternStrategyOptimized

def sf(v, d=0):
    try:
        x=float(v); return d if(math.isnan(x) or math.isinf(x)) else x
    except: return d

def load_daily(name):
    df=pd.read_csv(pr/f'data/raw/{name}_daily.csv',index_col=0,parse_dates=True)
    cm={x.lower():x for x in df.columns}
    for lo,st in [('open','Open'),('high','High'),('low','Low'),('close','Close'),('volume','Volume')]:
        if st not in df.columns and lo in cm: df.rename(columns={c[lo]:st},inplace=True)
    if 'Volume' not in df.columns: df['Volume']=0
    return df

def bt_run(df,pat,conf,risk,pos):
    bt=Backtest(df,MultiPatternStrategyOptimized,cash=1000000,commission=0.001,exclusive_orders=True)
    s=bt.run(include_patterns_only=pat,min_confluence_count=1,min_confidence=conf,risk_per_trade=risk,max_open_positions=pos)
    return{'trades':int(s.get('# Trades',0) or 0),'wr':sf(s.get('Win Rate [%]',0)),'sharpe':sf(s.get('Sharpe Ratio',-999),-999),'pf':sf(s.get('Profit Factor',0)),'return_pct':sf(s.get('Return [%]',0)),'dd_pct':sf(s.get('Max. Drawdown [%]',0))}

def grid_search(aname,df,pat):
    cv=[0.50,0.55,0.60,0.65,0.70,0.75]
    rv=[0.01,0.02,0.03,0.05]
    pv=[1,3,5]
    best=None; best_ret=-999; total=len(cv)*len(rv)*len(pv)
    print(f'
  {aname}: {pat} ({total} combos)... ',end='',flush=True)
    t0=time.time()
    for i,(c,r,p) in enumerate(itertools.product(cv,rv,pv)):
        try:
            res=bt_run(df,pat,c,r,p)
            if res['return_pct']>best_ret: best_ret=res['return_pct']; best={'conf':c,'risk':r,'pos':p,**res}
        except: pass
    dt=time.time()-t0
    print(f'done ({dt:.0f}s)')
    if best:
        print(f"    Best: ret={best['return_pct']:.1f}% sh={best['sharpe']:.2f} pf={best['pf']:.2f} wr={best['wr']:.1f}% t={best['trades']}")
    else: print('    No successful runs')
    return best

def main():
    passing=['Triple Top','Symmetric Triangle','Donchian Channel Breakout','Gap Pattern']
    opt=[]
    for an,fn in [('SPY','SPY'),('QQQ','QQQ')]:
        df=load_daily(fn)
        for pat in passing:
            r=grid_search(an,df,pat)
            if r:
                r['pat']=pat; r['asset']=an; opt.append(r)
    od=pr/reports/untested_patterns_backtest
    od.mkdir(parents=True,exist_ok=True)
    with open(od/optimization_results.json,w) as f: json.dump(opt,f,indent=2,default=str)
    with open(od/summary.md,w) as f:
        f.write(f'# Results

Optimization: {len(opt)} combos run
')
        for r in opt:
            f.write(f"- {r['asset']}/{r['pat']}: ret={r['return_pct']:.1f}% sh={r['sharpe']:.2f}
")
    print(f'Saved to {od}')

if __name__==__main__: main()
