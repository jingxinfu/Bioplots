import Bioplots as bpt
df = bpt.get_rdataset('lung')
bpt.scatterplot(df=df.head(20),x='wt.loss',y='age',color='inst',palette='RdBu_r',
       size='time',label='status',visible_hits=[2],size_scale=400)
