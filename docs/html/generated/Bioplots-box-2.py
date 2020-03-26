df['pair'] = list(range(50))*2
ax = bpt.box(df=df,x='day', y="temp",pair='pair')
