orders = df_cor.groupby('index')['value'].mean().sort_values().index.tolist()
ax = bpt.heatmap(df=df_cor,x='index',y='variable',color='value',size='size',
                 x_order=orders,y_order=orders);
