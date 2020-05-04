ax = bpt.violin(df=df,x='day', y="temp",subgroup='activ',
            rm_empty_space=True,color_option=('point','edge'))
