ax = bpt.heatmap(df=df_cor,x='index',y='variable',color='value',size='size',
                 x_order=orders,y_order=orders,
                 bin_labels=dict(bin=[0,.5,.8,1],label=['<.5','<.8','<1']));
