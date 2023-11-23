import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
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

cols = ["#2b1c8f","#7037a3","#a458b7","#d37ccd","#ffa4e4","#fc8bc5","#f672a4","#ec5980"]
linsize = 0.5

#OriginalFIM
def make_OGFIM():
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
    fig.savefig("Pics/OGFIM.png", bbox_inches='tight', dpi=500)

def make_T1GFIM():
    T1FIMstep = pd.read_csv('T1_step.csv', sep=',')
    T1FIMstep = T1FIMstep.iloc[::4]
    print(T1FIMstep.head())

    fig, (ax1,ax2) = plt.subplots(1,2,figsize=set_size(516,0.5),gridspec_kw={'wspace': 0.08})
    sea.set(style="whitegrid")
    
    sea.lineplot(ax=ax1,x="Step",y="0",data=T1FIMstep,label="0 (Low)",linewidth=linsize,color=cols[0])
    sea.lineplot(ax=ax1,x="Step",y="1",data=T1FIMstep,label="1",linewidth=linsize,color=cols[1])
    sea.lineplot(ax=ax1,x="Step",y="2",data=T1FIMstep,label="2",linewidth=linsize,color=cols[2])
    sea.lineplot(ax=ax1,x="Step",y="3",data=T1FIMstep,label="3",linewidth=linsize,color=cols[3])
    sea.lineplot(ax=ax1,x="Step",y="4",data=T1FIMstep,label="4",linewidth=linsize,color=cols[4])
    sea.lineplot(ax=ax1,x="Step",y="5",data=T1FIMstep,label="5",linewidth=linsize,color=cols[5])
    sea.lineplot(ax=ax1,x="Step",y="6",data=T1FIMstep,label="6",linewidth=linsize,color=cols[6])
    sea.lineplot(ax=ax1,x="Step",y="7",data=T1FIMstep,label="7 (High)",linewidth=linsize,color=cols[7])
    
    #plt.ylabel("T1 GFIM")
    ax1.set_ylabel("Loss GFIM")
    ax1.set_ylim([1,250])
    ax1.yaxis.set_ticks([1,50,100,150,200,250])

    ax1.set_xlabel("Batch")
    ax1.set_xlim([1,1650])
    start,end = ax1.get_xlim()
    ax1.xaxis.set_ticks([1,500,1000,1500])
    ax1.get_legend().remove()

    #Epoch graphs
    T1FIMepoch = pd.read_csv('T1_epoch.csv', sep=',')
    T1FIMepoch['Step'] = T1FIMepoch['Step']+1
    T1FIMepoch = T1FIMepoch.iloc[:65]
    print(T1FIMepoch.head())
    sea.lineplot(ax=ax2,x="Step",y="0",data=T1FIMepoch,label="0 (Low)",linewidth=linsize,color=cols[0])
    sea.lineplot(ax=ax2,x="Step",y="1",data=T1FIMepoch,label="1",linewidth=linsize,color=cols[1])
    sea.lineplot(ax=ax2,x="Step",y="2",data=T1FIMepoch,label="2",linewidth=linsize,color=cols[2])
    sea.lineplot(ax=ax2,x="Step",y="3",data=T1FIMepoch,label="3",linewidth=linsize,color=cols[3])
    sea.lineplot(ax=ax2,x="Step",y="4",data=T1FIMepoch,label="4",linewidth=linsize,color=cols[4])
    sea.lineplot(ax=ax2,x="Step",y="5",data=T1FIMepoch,label="5",linewidth=linsize,color=cols[5])
    sea.lineplot(ax=ax2,x="Step",y="6",data=T1FIMepoch,label="6",linewidth=linsize,color=cols[6])
    sea.lineplot(ax=ax2,x="Step",y="7",data=T1FIMepoch,label="7 (High)",linewidth=linsize,color=cols[7])

    ax2.set_xlabel("Epoch")
    #set the range of x axis
    ax2.set_xlim([1,70])
    ax2.xaxis.set_ticks([1,20,40,60])

    ax2.set_ylabel("Loss GFIM")
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    ax2.set_yscale('log')
    ax2.yaxis.set_ticks([1e-12,1e-10,1e-8,1e-6,1e-4,1e-2,1e0,1e2,1e4])

    ax2.get_legend().remove()

    con1 = ConnectionPatch(xyA=(1,1),
                        xyB=(0,0.85),
                        coordsA="axes fraction",
                        coordsB="axes fraction",
                        axesA=ax1,
                        axesB=ax2,
                        color="black",
                        linestyle="--",)
    con2 = ConnectionPatch(xyA=(1,0),
                        xyB=(0,0.75),
                        coordsA="axes fraction",
                        coordsB="axes fraction",
                        axesA=ax1,
                        axesB=ax2,
                        color="black",
                        linestyle="--",)
    ax1.add_artist(con1)
    ax1.add_artist(con2)
    fig.savefig("Pics/T1_GFIM.png", bbox_inches='tight', dpi=500)

def make_T2GFIM():
    FIMstep = pd.read_csv('T2_step.csv', sep=',')
    FIMstep = FIMstep.iloc[::4]
    
    print(FIMstep.head())

    fig, (ax1,ax2) = plt.subplots(1,2,figsize=set_size(516,0.5),gridspec_kw={'wspace': 0.08})
    sea.set(style="whitegrid")
    
    sea.lineplot(ax=ax1,x="Step",y="0",data=FIMstep,label="0 (Low)",linewidth=linsize,color=cols[0])
    sea.lineplot(ax=ax1,x="Step",y="1",data=FIMstep,label="1",linewidth=linsize,color=cols[1])
    sea.lineplot(ax=ax1,x="Step",y="2",data=FIMstep,label="2",linewidth=linsize,color=cols[2])
    sea.lineplot(ax=ax1,x="Step",y="3",data=FIMstep,label="3",linewidth=linsize,color=cols[3])
    sea.lineplot(ax=ax1,x="Step",y="4",data=FIMstep,label="4",linewidth=linsize,color=cols[4])
    sea.lineplot(ax=ax1,x="Step",y="5",data=FIMstep,label="5",linewidth=linsize,color=cols[5])
    sea.lineplot(ax=ax1,x="Step",y="6",data=FIMstep,label="6",linewidth=linsize,color=cols[6])
    sea.lineplot(ax=ax1,x="Step",y="7",data=FIMstep,label="7 (High)",linewidth=linsize,color=cols[7])
    
    #plt.ylabel("T1 GFIM")
    ax1.set_ylabel("Loss GFIM")
    ax1.set_ylim([1,800])
    ax1.yaxis.set_ticks([1,200,400,600,800])

    ax1.set_xlabel("Batch")
    ax1.set_xlim([1,1650])
    start,end = ax1.get_xlim()
    ax1.xaxis.set_ticks([1,500,1000,1500])
    ax1.get_legend().remove()

    #Epoch graphs
    FIMepoch = pd.read_csv('T2_epoch.csv', sep=',')
    #add 1 to the step
    FIMepoch['Step'] = FIMepoch['Step']+1
    print(FIMepoch.head())
    sea.lineplot(ax=ax2,x="Step",y="0",data=FIMepoch,label="0 (Low)",linewidth=linsize,color=cols[0])
    sea.lineplot(ax=ax2,x="Step",y="1",data=FIMepoch,label="1",linewidth=linsize,color=cols[1])
    sea.lineplot(ax=ax2,x="Step",y="2",data=FIMepoch,label="2",linewidth=linsize,color=cols[2])
    sea.lineplot(ax=ax2,x="Step",y="3",data=FIMepoch,label="3",linewidth=linsize,color=cols[3])
    sea.lineplot(ax=ax2,x="Step",y="4",data=FIMepoch,label="4",linewidth=linsize,color=cols[4])
    sea.lineplot(ax=ax2,x="Step",y="5",data=FIMepoch,label="5",linewidth=linsize,color=cols[5])
    sea.lineplot(ax=ax2,x="Step",y="6",data=FIMepoch,label="6",linewidth=linsize,color=cols[6])
    sea.lineplot(ax=ax2,x="Step",y="7",data=FIMepoch,label="7 (High)",linewidth=linsize,color=cols[7])

    ax2.set_xlabel("Epoch")
    #set the range of x axis
    ax2.set_xlim([1,120])
    ax2.xaxis.set_ticks([1,20,40,60,80,100,120])

    ax2.set_ylabel("Loss GFIM")
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    ax2.set_yscale('log')
    ax2.yaxis.set_ticks([1e-12,1e-10,1e-8,1e-6,1e-4,1e-2,1e0,1e2,1e4])

    ax2.get_legend().remove()

    con1 = ConnectionPatch(xyA=(1,1),
                        xyB=(0,0.9),
                        coordsA="axes fraction",
                        coordsB="axes fraction",
                        axesA=ax1,
                        axesB=ax2,
                        color="black",
                        linestyle="--",)
    con2 = ConnectionPatch(xyA=(1,0),
                        xyB=(0,0.75),
                        coordsA="axes fraction",
                        coordsB="axes fraction",
                        axesA=ax1,
                        axesB=ax2,
                        color="black",
                        linestyle="--",)
    ax1.add_artist(con1)
    ax1.add_artist(con2)
    fig.savefig("Pics/T2_GFIM.png", bbox_inches='tight', dpi=500)

def make_T3GFIM():
    FIMstep = pd.read_csv('T3_step.csv', sep=',')
    FIMstep = FIMstep.iloc[::8]
    
    print(FIMstep.head())

    fig, (ax1,ax2) = plt.subplots(1,2,figsize=set_size(516,0.5),gridspec_kw={'wspace': 0.14})
    sea.set(style="whitegrid")
    
    sea.lineplot(ax=ax1,x="Step",y="0",data=FIMstep,label="0 (Low)",linewidth=linsize,color=cols[0])
    sea.lineplot(ax=ax1,x="Step",y="1",data=FIMstep,label="1",linewidth=linsize,color=cols[1])
    sea.lineplot(ax=ax1,x="Step",y="2",data=FIMstep,label="2",linewidth=linsize,color=cols[2])
    sea.lineplot(ax=ax1,x="Step",y="3",data=FIMstep,label="3",linewidth=linsize,color=cols[3])
    sea.lineplot(ax=ax1,x="Step",y="4",data=FIMstep,label="4",linewidth=linsize,color=cols[4])
    sea.lineplot(ax=ax1,x="Step",y="5",data=FIMstep,label="5",linewidth=linsize,color=cols[5])
    sea.lineplot(ax=ax1,x="Step",y="6",data=FIMstep,label="6",linewidth=linsize,color=cols[6])
    sea.lineplot(ax=ax1,x="Step",y="7",data=FIMstep,label="7 (High)",linewidth=linsize,color=cols[7])
    
    #plt.ylabel("T1 GFIM")
    ax1.set_ylabel("Loss GFIM")
    ax1.set_ylim([1,2000])
    ax1.yaxis.set_ticks([1,200,400,600,800,1000,1200,1400,1600,1800,2000])

    ax1.set_xlabel("Batch")
    ax1.set_xlim([1,3100])
    start,end = ax1.get_xlim()
    ax1.xaxis.set_ticks([1,1000,2000,3000])
    ax1.get_legend().remove()

    #Epoch graphs
    FIMepoch = pd.read_csv('T3_epoch.csv', sep=',')
    #add 1 to the step
    FIMepoch['Step'] = FIMepoch['Step']+1
    #use the top 60 epochs
    FIMepoch = FIMepoch.iloc[:60]
    print(FIMepoch.head())
    sea.lineplot(ax=ax2,x="Step",y="0",data=FIMepoch,label="0 (Low)",linewidth=linsize,color=cols[0])
    sea.lineplot(ax=ax2,x="Step",y="1",data=FIMepoch,label="1",linewidth=linsize,color=cols[1])
    sea.lineplot(ax=ax2,x="Step",y="2",data=FIMepoch,label="2",linewidth=linsize,color=cols[2])
    sea.lineplot(ax=ax2,x="Step",y="3",data=FIMepoch,label="3",linewidth=linsize,color=cols[3])
    sea.lineplot(ax=ax2,x="Step",y="4",data=FIMepoch,label="4",linewidth=linsize,color=cols[4])
    sea.lineplot(ax=ax2,x="Step",y="5",data=FIMepoch,label="5",linewidth=linsize,color=cols[5])
    sea.lineplot(ax=ax2,x="Step",y="6",data=FIMepoch,label="6",linewidth=linsize,color=cols[6])
    sea.lineplot(ax=ax2,x="Step",y="7",data=FIMepoch,label="7 (High)",linewidth=linsize,color=cols[7])

    ax2.set_xlabel("Epoch")
    #set the range of x axis
    ax2.set_xlim([1,65])
    ax2.xaxis.set_ticks([1,20,40,60])

    ax2.set_ylabel("Loss GFIM")
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    ax2.set_yscale('log')
    ax2.yaxis.set_ticks([1e-12,1e-10,1e-8,1e-6,1e-4,1e-2,1e0,1e2,1e4])

    ax2.get_legend().remove()

    con1 = ConnectionPatch(xyA=(1,1),
                        xyB=(0,0.9),
                        coordsA="axes fraction",
                        coordsB="axes fraction",
                        axesA=ax1,
                        axesB=ax2,
                        color="black",
                        linestyle="--",)
    con2 = ConnectionPatch(xyA=(1,0),
                        xyB=(0,0.75),
                        coordsA="axes fraction",
                        coordsB="axes fraction",
                        axesA=ax1,
                        axesB=ax2,
                        color="black",
                        linestyle="--",)
    ax1.add_artist(con1)
    ax1.add_artist(con2)
    fig.savefig("Pics/T3_GFIM.png", bbox_inches='tight', dpi=500)

def make_T4GFIM():
    FIMstep = pd.read_csv('T4_step.csv', sep=',')
    FIMstep = FIMstep.iloc[::8]
    
    print(FIMstep.head())

    fig, (ax1,ax2) = plt.subplots(1,2,figsize=set_size(516,0.5),gridspec_kw={'wspace': 0.14})
    sea.set(style="whitegrid")
    
    sea.lineplot(ax=ax1,x="Step",y="0",data=FIMstep,label="0 (Low)",linewidth=linsize,color=cols[0])
    sea.lineplot(ax=ax1,x="Step",y="1",data=FIMstep,label="1",linewidth=linsize,color=cols[1])
    sea.lineplot(ax=ax1,x="Step",y="2",data=FIMstep,label="2",linewidth=linsize,color=cols[2])
    sea.lineplot(ax=ax1,x="Step",y="3",data=FIMstep,label="3",linewidth=linsize,color=cols[3])
    sea.lineplot(ax=ax1,x="Step",y="4",data=FIMstep,label="4",linewidth=linsize,color=cols[4])
    sea.lineplot(ax=ax1,x="Step",y="5",data=FIMstep,label="5",linewidth=linsize,color=cols[5])
    sea.lineplot(ax=ax1,x="Step",y="6",data=FIMstep,label="6",linewidth=linsize,color=cols[6])
    sea.lineplot(ax=ax1,x="Step",y="7",data=FIMstep,label="7 (High)",linewidth=linsize,color=cols[7])
    
    #plt.ylabel("T1 GFIM")
    ax1.set_ylabel("Loss GFIM")
    ax1.set_ylim([1,60000])
    ax1.yaxis.set_ticks([1,20000,40000,60000])
    #ax1.set_yscale('log')

    ax1.set_xlabel("Batch")
    ax1.set_xlim([1,1300])
    start,end = ax1.get_xlim()
    ax1.xaxis.set_ticks([1,400,800,1200])
    ax1.get_legend().remove()

    #Epoch graphs
    FIMepoch = pd.read_csv('T4_epoch.csv', sep=',')
    #add 1 to the step
    FIMepoch['Step'] = FIMepoch['Step']+1
    #use the top 60 epochs
    FIMepoch = FIMepoch.iloc[:60]
    print(FIMepoch.head())
    sea.lineplot(ax=ax2,x="Step",y="0",data=FIMepoch,label="0 (Low)",linewidth=linsize,color=cols[0])
    sea.lineplot(ax=ax2,x="Step",y="1",data=FIMepoch,label="1",linewidth=linsize,color=cols[1])
    sea.lineplot(ax=ax2,x="Step",y="2",data=FIMepoch,label="2",linewidth=linsize,color=cols[2])
    sea.lineplot(ax=ax2,x="Step",y="3",data=FIMepoch,label="3",linewidth=linsize,color=cols[3])
    sea.lineplot(ax=ax2,x="Step",y="4",data=FIMepoch,label="4",linewidth=linsize,color=cols[4])
    sea.lineplot(ax=ax2,x="Step",y="5",data=FIMepoch,label="5",linewidth=linsize,color=cols[5])
    sea.lineplot(ax=ax2,x="Step",y="6",data=FIMepoch,label="6",linewidth=linsize,color=cols[6])
    sea.lineplot(ax=ax2,x="Step",y="7",data=FIMepoch,label="7 (High)",linewidth=linsize,color=cols[7])

    ax2.set_xlabel("Epoch")
    #set the range of x axis
    ax2.set_xlim([1,55])
    ax2.xaxis.set_ticks([1,20,40])

    ax2.set_ylabel("Loss GFIM")
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    ax2.set_yscale('log')
    ax2.yaxis.set_ticks([1e-10,1e-8,1e-6,1e-4,1e-2,1e0,1e2,1e4,1e6])

    ax2.get_legend().remove()

    con1 = ConnectionPatch(xyA=(1,1),
                        xyB=(0,0.91),
                        coordsA="axes fraction",
                        coordsB="axes fraction",
                        axesA=ax1,
                        axesB=ax2,
                        color="black",
                        linestyle="--",)
    con2 = ConnectionPatch(xyA=(1,0),
                        xyB=(0,0.62),
                        coordsA="axes fraction",
                        coordsB="axes fraction",
                        axesA=ax1,
                        axesB=ax2,
                        color="black",
                        linestyle="--",)
    ax1.add_artist(con1)
    ax1.add_artist(con2)
    fig.savefig("Pics/T4_GFIM.png", bbox_inches='tight', dpi=500)

def make_legend():
    T1FIMstep = pd.read_csv('T1_step.csv', sep=',')
    print(T1FIMstep.head())

    fig, ax1 = plt.subplots(1,1,figsize=[252,20])
    
    sea.set(style="whitegrid")
    linsize = 3
    
    sea.lineplot(ax=ax1,x="Step",y="0",data=T1FIMstep,label="0 (Low)",linewidth=linsize,color=cols[0])
    sea.lineplot(ax=ax1,x="Step",y="1",data=T1FIMstep,label="1",linewidth=linsize,color=cols[1])
    sea.lineplot(ax=ax1,x="Step",y="2",data=T1FIMstep,label="2",linewidth=linsize,color=cols[2])
    sea.lineplot(ax=ax1,x="Step",y="3",data=T1FIMstep,label="3",linewidth=linsize,color=cols[3])
    sea.lineplot(ax=ax1,x="Step",y="4",data=T1FIMstep,label="4",linewidth=linsize,color=cols[4])
    sea.lineplot(ax=ax1,x="Step",y="5",data=T1FIMstep,label="5",linewidth=linsize,color=cols[5])
    sea.lineplot(ax=ax1,x="Step",y="6",data=T1FIMstep,label="6",linewidth=linsize,color=cols[6])
    sea.lineplot(ax=ax1,x="Step",y="7",data=T1FIMstep,label="7 (High)",linewidth=linsize,color=cols[7])
    label_params = ax1.get_legend_handles_labels()
    legendfig,lax = plt.subplots(1,1,figsize=set_size(516,0.5))
    lax.axis(False)
    lax.legend(*label_params,loc='center',ncol=8,frameon=False)
    legendfig.savefig("Pics/GFIM_legend.png", bbox_inches='tight', dpi=500)

make_legend()