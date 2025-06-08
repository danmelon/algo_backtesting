import pandas as pd
import pandas_ta as ta
"""https://youtu.be/C3bh6Y4LpGs"""
"""Pandas_ta is only compatable with pip install numpy==1.23.5"""

df = pd.read_csv("EURUSD_Candlestick_5_M_ASK_30.09.2019-30.09.2022.csv")
df["Gmt time"] = df["Gmt time"].str.replace(".000", "")
df['Gmt time'] = pd.to_datetime(df['Gmt time'], format= '%d.%m.%Y %H:%M:%S')
df=df[df.High!=df.Low] #!= means not equal >> this line therefore excludes days where there was no movement
df.set_index("Gmt time", inplace=True)


df["EMA_slow"] = ta.ema(df.Close, length=50)
df["EMA_fast"] = ta.ema(df.Close, length=30)
df['RSI'] = ta.rsi(df.Close, length=10)
my_bbands = ta.bbands(df.Close, length = 15, std=1.5)
df['ATR'] = ta.atr(df.High, df.Low, df.Close, length= 7)
df = df.join(my_bbands)


def ema_signal(df, current_candle, backcandles):

    df_slice = df.reset_index().copy()
    #get the range of candles to consider

    start = max(0, current_candle - backcandles)
    end = current_candle
    relevant_rows = df_slice.iloc[start:end]

    #check if all EMA_fast values are below EMA_slow values
    if all(relevant_rows["EMA_fast"] < relevant_rows["EMA_slow"]):
        return 1
    elif all(relevant_rows["EMA_fast"] > relevant_rows["EMA_slow"]):
        return 2
    else:
        return 0
    
df = df[-30000:-1]
from tqdm import tqdm
tqdm.pandas()
df.reset_index(inplace=True)
df['EMASignal'] = df.progress_apply(lambda row: ema_signal(df, row.name, 7) if row.name >=20 else 0, axis=1)




def total_signal(df, current_candle, backcandles):
    if (ema_signal(df, current_candle, backcandles) == 2
        and df.Close[current_candle] <= df['BBL_15_1.5'][current_candle]
        #and df.RSI[current_candle] < 60
    ):
        return 2
    
    if(ema_signal(df, current_candle, backcandles) == 1
       and df.Close[current_candle] >= df['BBU_15_1.5'][current_candle]
       #and df.RSI[current_candle] > 40
       ):
        return 1
    return 0

df['TotalSignal'] = df.progress_apply(lambda row: total_signal(df, row.name, 7), axis =1)


import numpy as np

def pointpos(x):
    if x['TotalSignal']==2:
        return x['Low']-1e-3
    elif x['TotalSignal']==1:
        return x['High']+1e-3
    else:
        return np.nan
    
df['pointpos'] = df.apply(lambda row: pointpos(row), axis=1)

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
st=100
dfpl = df[st:st+350]
fig = go.Figure(data=[
    go.Candlestick(
        x=dfpl.index,
        open=dfpl['Open'],
        high = dfpl['High'],
        low = dfpl['Low'],
        close = dfpl['Close']
        ),
                
    go.Scatter(x=dfpl.index, y=dfpl['BBL_15_1.5'],
                           line=dict(color='green', width=1),
                           name= "BBL"),
    go.Scatter(x=dfpl.index, y=dfpl['BBU_15_1.5'],
                           line=dict(color='green', width=1),
                           name= "BBU"),
    go.Scatter(x=dfpl.index, y=dfpl['EMA_fast'],
                           line=dict(color='green', width=1),
                           name= "EMA_fast"),
    go.Scatter(x=dfpl.index, y=dfpl['EMA_slow'],
                           line=dict(color='green', width=1),
                           name= "EMA_slow")])

fig.add_scatter(x=dfpl.index, y=dfpl['pointpos'], mode="markers",
                marker=dict(size=5, color="MediumPurple"),
                name="entry")

fig.show()

