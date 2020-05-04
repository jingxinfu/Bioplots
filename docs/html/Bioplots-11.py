df['pair'] = list(range(50))*2
ax = bpt.violin(df=df,x='day', y="temp",pair='pair')
