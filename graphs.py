import matplotlib.pyplot as plt
import seaborn as sea 
import pandas as pd 
import numpy as np 


 

def set_size(width, fraction=1):
    """ Set aesthetic figure dimensions to avoid scaling in latex.
    
    Parameters
    ----------
    width: float
            Width in pts
    fraction: float
            Fraction of the width which you wish the figure to occupy
    
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure
    fig_width_pt = width * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27
    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**.5 - 1) / 2
    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio
    fig_dim = (fig_width_in, fig_height_in)
    return fig_dim


plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    "axes.labelsize": 10,
    "font.size": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8
})


#OriginalFIM
OriginalFIM = pd.read_csv('FIMOriginal.csv', sep=',')

print(OriginalFIM.head())

#bar plot of OriginalFIM with maximum and minimum values
fig, ax = plt.subplots(1,1,figsize=set_size(252))
sea.set(style="whitegrid")
sea.barplot(x=["ResNet-18","5 Layer CNN","8 Layer CNN","ResNetV1-14"],
            y=[OriginalFIM.loc[0,'T4'],OriginalFIM.loc[0,'T3'],OriginalFIM.loc[0,'T2'],OriginalFIM.loc[0,'T1']],
            palette="Blues_d",
            )
plt.ylabel("Original FIM")
plt.xlabel("Models")
plt.xticks(rotation=15)
plt.grid(axis='y',which='both',linestyle='--')
ax.set_axisbelow(True)


#max min error bars
names = ["T4","T3","T2","T1"]
i = 0
for p in ax.patches:
    print(names[i]+"_min")
    plt.vlines(p.get_x()+p.get_width()/2,
        OriginalFIM.loc[0,names[i]+"_min"],
        OriginalFIM.loc[0,names[i]+"_max"],
        color='black',
        alpha=0.5,
        )
    i+=1
#save the plot
fig.savefig("OGFIM.png", bbox_inches='tight', dpi=500)
