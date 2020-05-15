import Bioplots as bpt
df = bpt.get_rdataset('lung')
df_cor = df.corr().reset_index().melt(id_vars=['index'])
df_cor['size'] = df_cor['value'].abs()
ax = bpt.heatmap(df=df_cor,x='index',y='variable',color='value',size='size');
