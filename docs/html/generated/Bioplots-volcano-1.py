import Bioplots as bpt
from scipy.stats import pearsonr
df = bpt.get_rdataset('lung')
def pearson_pvalue(x,y):
    return pearsonr(x,y)[1]
df_cor = df.corr().reset_index().melt(id_vars=['index'])
df_pvalue = df.corr(method=pearson_pvalue).reset_index().melt(id_vars=['index'])
cor_pvalue = df_cor.merge(df_pvalue,on=['index','variable'])
cor_pvalue.rename(columns={'value_x':'cor','value_y':'pvalue'},inplace=True)
cor_pvalue.head()
bpt.volcano(df=cor_pvalue,lfc='cor',pvalue='pvalue',lfc_cutoff=0.1,pvalue_cutoff=.05,label='index');
