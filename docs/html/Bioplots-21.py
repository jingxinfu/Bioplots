import Bioplots as bpt
df = bpt.get_rdataset('beaver')
df['day'] = df['day'].map(str)
df['activ'] = df['activ'].map(str)
ax = bpt.bar(df=df,x='day', y="temp")
