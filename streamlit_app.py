# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException
from helper_functions import read_telegram3
import webbrowser
import threading
import time
import plotly.graph_objects as go
import plotly

#%% 2. Helper functions:

# Convert trades to dataframe
def create_trades_frame(trades):
# from: https://tiao.io/post/exploring-the-binance-cryptocurrency-exchange-api-recent-historical-trades/
    
    frame = pd.DataFrame(trades) \
        .assign(time = lambda trade: pd.to_datetime(trade.time, unit ='ms'),
                price = lambda trade: pd.to_numeric(trade.price),
                qty=lambda trade: pd.to_numeric(trade.qty),
                qouteQty=lambda trade: pd.to_numeric(trade.quoteQty))
    
    return frame


# convert trades dataframe to ohlc dataframe
def trades_to_ohlc(df_trades):
    
    # round the time to seconds
    df_trades['time(s)'] = df_trades['time'].dt.floor('s')


    df_ohlc = df_trades[['time(s)', 'price']]

    agg_functions = {'price':['first','max','min','last']}
    df_ohlc = df_ohlc.groupby(['time(s)']).agg(agg_functions)

    df_ohlc.columns = ['Open','High','Low', 'Close']

    # Convert index to column:
    df_ohlc.reset_index(inplace=True)
    
    return df_ohlc

# Get the opening price:
def get_opening_price():
    
    global openPrice
    
    openPrice = client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_30MINUTE)[499][1]
    openPrice = float(openPrice)

    
# Get the ask price:
def get_ask_price():
    
    global askPrice

    market = pd.DataFrame(client.get_orderbook_tickers())
    market = market.loc[market['symbol'] == symbol]
    askPrice = float(market['askPrice'].values[0])


def open_web_exchange():
    
    global symbol
    
    url = 'https://www.binance.com/en/trade/' + asset + '_BTC?layout=pro' 
    webbrowser.register('firefox',
    	None,
    	webbrowser.BackgroundBrowser("C://Program Files//Mozilla Firefox//firefox.exe"))
    
    webPageOpen = webbrowser.get('firefox').open(url)
    
    return(webPageOpen)

def wb_trailing_stop(msg):
    
    global df_trades
    global stop_loss_price
    global stop_limit
    global low_price
    
    if msg['e'] == 'error':
        print(msg['m'])
    
    else:
        
        # convert dictionary to dataframe:
        temp = pd.DataFrame(msg, index=[0])
        
        # Append to all trades:
        df_trades = df_trades.append(temp, ignore_index=True)
        
        # Lowest price
        low_price = df_trades['p'].tail(10).min()
        low_price = float(low_price)
        
        if(stop_loss_price < low_price*(0.95)):
            
            stop_loss_price = low_price * (0.95) # Update the trailing stop loss
            
            # Cancel existing oco order:
            
            # Create new oco order:
            
            print('The stop loss has been updated to: {}'.format(stop_loss_price))
            
#%%  

if 'streaming' not in st.session_state:         
    st.session_state.streaming = False

"""
# Pump and Dump Bot
"""

# On Start:
#if 'telegram_channels' not in st.session_state:
#    st.session_state.telegram_channels = pd.read_csv('telegram_channels.csv')
    

# Choose exchange:

"""
### API Credentials:
"""

# Initialize API credentials
api_key = st.text_input("Your API Key: ")
api_secret = st.text_input("Your API Secret: ")


"""
### Exchange info:
"""
exchange_info = st.empty()

# Log in to the binance and get information: (On button press)
if st.button('Get Exchange Info'):
    
    # Instantiate a Client:
    client = Client(api_key, api_secret)
    
    # Get Symbol information:
    info = client.get_exchange_info()

    df_symbols = pd.DataFrame(info['symbols'])
    
    temp = pd.DataFrame(df_symbols['filters'].to_list())
    temp2 = pd.DataFrame(temp.iloc[:,0].to_list())
    df_symbols = pd.concat([df_symbols,temp2],axis=1)
    
    df_symbols = df_symbols[['symbol','status', 'baseAsset',
                             'baseAssetPrecision', 'quoteAsset',
                             'quotePrecision', 'quoteAssetPrecision',
                             'baseCommissionPrecision','quoteCommissionPrecision',
                             'icebergAllowed', 'ocoAllowed', 
                             'quoteOrderQtyMarketAllowed', 'isSpotTradingAllowed',
                             'isMarginTradingAllowed','minPrice',
                             'maxPrice', 'tickSize']]
    
    df_symbols[['minPrice','maxPrice', 'tickSize']] = df_symbols[['minPrice','maxPrice', 'tickSize']].apply(pd.to_numeric)
    
    df_symbols['tickSizePrecision'] =  abs(np.log10(df_symbols['tickSize'])).astype(int)
    
    # Filter for only BTC pairs
    df_symbols = df_symbols[df_symbols['quoteAsset'] == 'BTC']
    # Filter for symbols where margin is not allowed & spot trading is allowed
    #df_symbols = df_symbols[df_symbols['isMarginTradingAllowed'] == False]
    df_symbols = df_symbols[df_symbols['isSpotTradingAllowed'] == True]
    df_symbols = df_symbols[df_symbols['status'] == 'TRADING']
    
    
    # Remove assets that might be confused for something else:
    df_symbols = df_symbols[df_symbols['baseAsset'] != 'GO']
    df_symbols = df_symbols[df_symbols['baseAsset'] != 'BTCB']
    df_symbols = df_symbols[df_symbols['baseAsset'] != 'BTCST']
    df_symbols = df_symbols[df_symbols['baseAsset'] != 'WBTC']
    df_symbols = df_symbols[df_symbols['baseAsset'] != 'RENBTC']
    df_symbols = df_symbols[df_symbols['baseAsset'] != 'OM']
    df_symbols = df_symbols[df_symbols['baseAsset'] != 'NU']
    df_symbols = df_symbols[df_symbols['baseAsset'] != 'OST']
    df_symbols = df_symbols[df_symbols['baseAsset'] != 'OM']
    
    if 'client' not in st.session_state:
        st.session_state.client = client
    
    if 'df_symbols' not in st.session_state:
        st.session_state.df_symbols = df_symbols

    # Create symbols and assets arrays
    if 'symbols' not in st.session_state:
        st.session_state.symbols = np.array(df_symbols['symbol'])
        
    if 'assets' not in st.session_state:
        st.session_state.assets = np.array(df_symbols['baseAsset']) 
      

if 'symbols' not in st.session_state:
    
    exchange_info.text('No Exchange Data.')
else:
    exchange_info.text(f'There are {len(st.session_state.symbols)} Symbols')
    
"""
### Inputs and parameters:
"""

# Global variables:
inTrade = False # tradeExecuted, # Set indicator to show trade executed.
quantity = 0
askPrice = 0
openPrice = 0
balance = 0
symbol = []

if 'asset' not in st.session_state:
    st.session_state.asset = []
    
print( f'We have initialized asset {st.session_state.asset}')

buy_limit = []
stop_limit = []
precision = 8  # Default precision
webPageOpen = False

# Select Telegram channel:

# selected_channels = st.multiselect(label='Channel:',
#                          options = st.session_state.telegram_channels.Channel)

# Add Channel to options:
# left_column, right_column = st.beta_columns(2)
# left_column.text_input(label='Add Channel URL')

# with right_column:
#     ""
#     ""
#     submit_channel = st.button('Submit')

channel_url = st.text_input(label = 'Channel URL', 
                            value='https://t.me/s/Big_Pumps_Signals')



trade_amount = st.number_input(label="Trade amount: ", step=0.0001, format="%.4f")
target_profit = st.number_input(label="Target profit (%)", step=1., format="%.2f")
stop_loss = st.number_input(label="Stop loss (%)", step=1., format="%.2f")

st.text(f'Asset: {st.session_state.asset}')


"""
### Scanning for pump and dump:
"""

# Show message from telegram
message = st.empty()

# Show app info
info = st.empty()

# Start Button:
start = st.button('Start Listening')

# Stop Button:
stop = st.button('Stop')

# Clear data:
clear = st.button('Clear')


if clear:
    # FIX: NEED TO RUN THIS ACTION FIRST BEFORE THE WHOLE SCRIPT RE-RERUNS
    # Clear the asset
    st.session_state.asset = []

# Start listening:
if start:
    
    # Create Threads:
    t1 = threading.Thread(target=get_opening_price)
    t2 = threading.Thread(target=get_ask_price)
    t3 = threading.Thread(target=open_web_exchange)
    
    
    
    # Loop until coin is revealed:
    while len(st.session_state.asset) == 0:
        # Read the last Telegram message:
        text_msg = read_telegram3(channel_url) 
        
        # Make time pretty (Printing Server time)
        timestamp = datetime.now()
        #timestamp = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        # print(timestamp)
        # print(text_msg + '\n')
        message.text(f'{timestamp} : {text_msg} \n')
        
        # get the asset mentioned
        st.session_state.asset = [asset for asset in st.session_state.assets if(asset in text_msg)]
        
        
        print(st.session_state.asset)
        
        if stop:
            break
        
        
    
    # get the biggest string in the list
    asset = max(st.session_state.asset, key=len)
    
    if 'asset' not in st.session_state:
        st.session_state.asset = asset
    
    # Get symbol info:
    symbol = asset + 'BTC'
    if 'symbol' not in st.session_state:
        st.session_state.symbol = symbol
    
    # Get the precision of the coin
    df_symbols = st.session_state.df_symbols
    precision = df_symbols[df_symbols['symbol'] == symbol]['tickSizePrecision'].values[0]
    
    client = st.session_state.client
    
    # Get the Open price and Ask Price:
    t1.start()
    t2.start()
    
    t1.join()
    t2.join()
    
    print(f'Open price is: {openPrice} \n')
    print(f'Ask price is: {askPrice}')
    
    # Calculate Order Info:
    pct_change = askPrice/openPrice - 1
    
    if(target_profit/100 > pct_change):
        #IMPROVE: DON'T TRADE WHEN THE ASSET IS UP MORE THAN 20% IN THE LAST 4 HOURS.
        #IMPROVE: THE KLINES DATA CAN BE USED FOR THIS
        quantity = round(trade_amount / askPrice,0) # trade amount is position size.
        take_profit = target_profit - pct_change
        
    else:
        
        # don't trade:
        quantity = 0
        
        info.text('[Info] Price has already moved! \n')
    # ----------------------------------------------------------------------------
    
    # Place buy order:
    try:
        
        buy_limit = client.order_limit_buy(
            symbol = symbol,
            quantity = quantity,
            price = '{:.{prec}f}'.format(askPrice * 1.0010, prec = precision),
            )
        
        if 'buy_limit' not in st.session_state:
            st.session_state.buy_limit = buy_limit
    
        # Message:
        info.text('[Info] We have bought {} {} coins at price {} \n'.format(quantity,
                                                          symbol,
                                                          format(askPrice,'.8f')))
    
    except BinanceAPIException as e:
        info.text(e)
        
    except BinanceOrderException as e:
        info.text(e)
        
    # Place sell orders:
    if (len(buy_limit) > 0):
        
        try:
            # Get balance:
            balance = client.get_asset_balance(asset=asset)
            balance = float(balance['free']) #IMPROVE!!!
        except:
            info.text("[Info] Error: Couldn't get the balance. Possibly no asset recognized.")
        
        try:
            
            stop_limit = client.create_oco_order(
                symbol = symbol,
                side = 'SELL',
                quantity = np.floor(balance),
                price = '{:.{prec}f}'.format(askPrice*(1 + take_profit/100), prec = precision), # Take profit
                stopPrice = '{:.{prec}f}'.format(askPrice*(1 - stop_loss/100) , prec = precision), # Stop Loss Trigger
                stopLimitPrice= '{:.{prec}f}'.format(askPrice*(1 - stop_loss/100), prec = precision), # Stop Loss
                stopLimitTimeInForce = 'GTC'
                )
        
            if 'sell_limit' not in st.session_state:
                st.session_state.sell_limit = buy_limit
                
            info.text('[Info] Take profit and stop loss have been placed!')
            
        except BinanceAPIException as e:
            info.text(e)
        
        except BinanceOrderException as e:
            info.text(e)
        
        
        # get order ID:
        st.session_state.orderId = buy_limit['orderId']
        
        st.session_state.streaming = True
            
"""
### Trade Information
"""

# Place holder for chart:
chart = st.empty()

# Place holder for pnl table:
pnl_table = st.empty()

if st.session_state.streaming:
    
    while True:
        
        # Get bid price
        bidPrice = st.session_state.client.get_symbol_ticker(symbol=st.session_state.symbol)
        bidPrice = bidPrice['price']
        print(f'The bid pice is {bidPrice}')
        
        # Get trades
        trades = st.session_state.client.get_recent_trades(symbol=st.session_state.symbol)
        print('Got trades!')
        
        # Initialize trades dataframe
        if 'df_trades' not in st.session_state:
            # create trades data frame
            st.session_state.df_trades = create_trades_frame(trades)
            print('Initialized trades dataframe.')
            
        else:
            # append trades
            temp = create_trades_frame(trades)
            st.session_state.df_trades = st.session_state.df_trades.append(temp, ignore_index=True)
    
            # distinct trades
            st.session_state.df_trades = st.session_state.df_trades.drop_duplicates(ignore_index=True)
            print('Updated trades dataframe')
        
        # trades to ohlc:
        df_ohlc = trades_to_ohlc(st.session_state.df_trades)
            
        # Chart of trades and when message was annonced:
        fig = go.Figure(data=[go.Candlestick(x = df_ohlc['time(s)'],
                                         open = df_ohlc['Open'],
                                         high = df_ohlc['High'],
                                         low = df_ohlc['Low'],
                                         close = df_ohlc['Close'])])
        # remove rangeslider:
        fig.update_layout(xaxis_rangeslider_visible=False)
        
        print(f'Plot chart of {symbol}')
        chart.plotly_chart(fig)
        
        # PnL table info
        buy_order = client.get_order(symbol=st.session_state.symbol, 
                                     orderId=st.session_state.orderId)
        
        trade_time = pd.to_datetime(buy_order['time'], unit='ms')
        symbol = buy_order['symbol']
        side = buy_order['side']
        size = buy_order['executedQty']
        trade_sl = 0
        trade_tp = 0
        entry_price = float(buy_order['price'])
        exit_price = float(bidPrice)
        profit = (exit_price/entry_price - 1) * 100
    
        trade_df = pd.DataFrame({'Time':trade_time, 'Symbol':symbol, 'Side':side,
                                 'Size':size, 'S/L':trade_sl, 'T/P':trade_tp, 'Entry Price':entry_price,
                                 'Price':exit_price, 'Profit':profit},index=[0])
    
        def color_profit(val):
            color = 'red' if val < 0 else 'yellow' if val==0 else 'green'
            return f'color: {color}'
    
        pnl_table.table(trade_df.style.applymap(color_profit,subset=['Profit']))
        
        # Update every 5 seconds (time.sleep(5))
        time.sleep(5)
        
        

print(st.session_state.streaming)

            

            








