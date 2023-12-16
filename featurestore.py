# Daily change
df['high-low'] = (df['High'] - df['Low'])
df['close-open'] = (df['Close'] - df['Open'])

# Gradient of prev day
close = df['Close']
df['gradientclose'] = np.gradient(close)
df['gradientclose'] = df['gradientclose'].shift(1)

# Prev day change
df['daydiff'] = df['Close'] - df['Close'].shift(1)

# Same sign?
df['mult'] = (df['close-open'] * df['gradientclose'] > 0).astype(int)


# RSI, MACD, CMF
quotes_list = []
for index, row in df.iterrows():
    datetime_object = datetime.strptime(row['Date'], '%Y-%m-%d')
    quotes_list.append(Quote(datetime_object, row['Open'], row['High'], row['Low'], row['Close'], row['Volume']))

rsi_results = indicators.get_rsi(quotes_list)
rsi_values = []
for r in rsi_results:
    rsi_values.append(r.rsi)
    
macd_results = indicators.get_macd(quotes_list)
macd_values = []
for m in macd_results:
    macd_values.append(m.macd)
    
cmf_results = indicators.get_cmf(quotes_list)
cmf_values = []
for c in cmf_results:
    cmf_values.append(c.cmf)
    
df['RSI'] = rsi_values
df['MACD'] = macd_values
df['CMF'] = cmf_values
