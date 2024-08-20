import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import seaborn as sea 
import pandas as pd 
import numpy as np 
import matplotlib.ticker as ticker


 

def set_size(width, height=None, fraction=1):
    """ Set aesthetic figure dimensions to avoid scaling in latex.
    
    Parameters
    ----------
    width: float
            Width in pts
    height: float
            Height in pts
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
    if height is None:
        fig_height_in = fig_width_in * golden_ratio
        fig_dim = (fig_width_in, fig_height_in)
    else:
        fig_height_pt = height
        fig_height_in = fig_height_pt * inches_per_pt
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

#cols = ["#2b1c8f","#7037a3","#a458b7","#d37ccd","#ffa4e4","#fc8bc5","#f672a4","#ec5980"]
cols = ["#0410ff","#383be6","#5c4ccd","#7e53b3","#9e5398","#bf4d7b","#df3e59","#ff1223"]
linsize = 0.3

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

def make_eloss():
    bloss = pd.read_csv('E_bloss.csv',sep=',')
    bnorm = pd.read_csv('E_bnorm.csv',sep=',')
    bcos = pd.read_csv('E_bcos.csv',sep=',')
    bFIM = pd.read_csv('T2_batchV2.csv',sep=',')

    loss = pd.read_csv('E_loss.csv',sep=',')
    norm = pd.read_csv('E_norm.csv',sep=',')
    cos = pd.read_csv('E_cos.csv',sep=',')
    FIM = pd.read_csv('T2_epochV2.csv',sep=',')


    #loss
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=set_size(516,fraction=0.5),gridspec_kw={'wspace': 0.12,'hspace':0.2},width_ratios=[1, 2])
    sea.lineplot(ax=ax1,x="Step",y="0",data=bloss,label="0 (Low)",linewidth=linsize,color=cols[0])
    sea.lineplot(ax=ax1,x="Step",y="1",data=bloss,label="1",linewidth=linsize,color=cols[1])
    sea.lineplot(ax=ax1,x="Step",y="2",data=bloss,label="2",linewidth=linsize,color=cols[2])
    sea.lineplot(ax=ax1,x="Step",y="3",data=bloss,label="3",linewidth=linsize,color=cols[3])
    sea.lineplot(ax=ax1,x="Step",y="4",data=bloss,label="4",linewidth=linsize,color=cols[4])
    sea.lineplot(ax=ax1,x="Step",y="5",data=bloss,label="5",linewidth=linsize,color=cols[5])
    sea.lineplot(ax=ax1,x="Step",y="6",data=bloss,label="6",linewidth=linsize,color=cols[6])
    sea.lineplot(ax=ax1,x="Step",y="7",data=bloss,label="7 (High)",linewidth=linsize,color=cols[7])

    ax1.set_ylabel("",rotation=0)
    #ax1.set_ylim((10**0,10**3))
    #ax1.yaxis.set_ticks([1,50,100,150,200])
    ax1.yaxis.set_label_coords(0.1,1.02)
    #ax1.set_yscale('log')

    ax1.set_xlabel("Batch")
    ax1.set_xlim([1,1500])
    start,end = ax1.get_xlim()
    ax1.xaxis.set_ticks([1,750,1500])
    ax1.grid(True,which='major',axis='both',linestyle='-',linewidth=0.2,alpha=0.5)
    ax1.grid(True,which='minor',axis='both',linestyle='--',linewidth=0.2,alpha=0.5)
    
    ax1.get_legend().remove()

    sea.lineplot(ax=ax2,x="Step",y="0",data=loss,label="0 (Low)",linewidth=linsize,color=cols[0])
    sea.lineplot(ax=ax2,x="Step",y="1",data=loss,label="1",linewidth=linsize,color=cols[1])
    sea.lineplot(ax=ax2,x="Step",y="2",data=loss,label="2",linewidth=linsize,color=cols[2])
    sea.lineplot(ax=ax2,x="Step",y="3",data=loss,label="3",linewidth=linsize,color=cols[3])
    sea.lineplot(ax=ax2,x="Step",y="4",data=loss,label="4",linewidth=linsize,color=cols[4])
    sea.lineplot(ax=ax2,x="Step",y="5",data=loss,label="5",linewidth=linsize,color=cols[5])
    sea.lineplot(ax=ax2,x="Step",y="6",data=loss,label="6",linewidth=linsize,color=cols[6])
    sea.lineplot(ax=ax2,x="Step",y="7",data=loss,label="7 (High)",linewidth=linsize,color=cols[7])

    ax2.set_xlabel("Epoch")
    #set the range of x axis
    ax2.set_xlim([1,150])
    ax2.xaxis.set_ticks([1,25,50,75,100,125,150])
    

    ax2.set_ylabel("",rotation=0)
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_coords(1,1.1)
    ax2.set_yscale('log')
    #ax2.set_ylim([1e-10,1e5])
    #ax2.yaxis.set_ticks([1e-12,1e-10,1e-8,1e-6,1e-4,1e-2,1e0,1e2,1e4])
    ax2.grid(True,which='major',axis='both',linestyle='-',linewidth=0.2,alpha=0.5)
    ax2.grid(True,which='minor',axis='both',linestyle='--',linewidth=0.2,alpha=0.5)

    ax2.get_legend().remove()

    con1 = ConnectionPatch(xyA=(1,1),
                        xyB=(0,0.95),
                        coordsA="axes fraction",
                        coordsB="axes fraction",
                        axesA=ax1,
                        axesB=ax2,
                        color="black",
                        linestyle="--",)
    con2 = ConnectionPatch(xyA=(1,0),
                        xyB=(0,0.9),
                        coordsA="axes fraction",
                        coordsB="axes fraction",
                        axesA=ax1,
                        axesB=ax2,
                        color="black",
                        linestyle="--",)
    ax1.add_artist(con1)
    ax1.add_artist(con2)


    fig.savefig("Pics/E_loss.png", bbox_inches='tight', dpi=500)

def make_ecos():
    bloss = pd.read_csv('E_bloss.csv',sep=',')
    bnorm = pd.read_csv('E_bnorm.csv',sep=',')
    bcos = pd.read_csv('E_bcos.csv',sep=',')
    bFIM = pd.read_csv('T2_batchV2.csv',sep=',')

    loss = pd.read_csv('E_loss.csv',sep=',')
    norm = pd.read_csv('E_norm.csv',sep=',')
    cos = pd.read_csv('E_cos.csv',sep=',')
    FIM = pd.read_csv('T2_epochV2.csv',sep=',')


    #loss
    r = 2
    bcos['0']= bcos['0'].rolling(r).mean()
    bcos['1']= bcos['1'].rolling(r).mean()
    bcos['2']= bcos['2'].rolling(r).mean()
    bcos['3']= bcos['3'].rolling(r).mean()
    bcos['4']= bcos['4'].rolling(r).mean()
    bcos['5']= bcos['5'].rolling(r).mean()
    bcos['6']= bcos['6'].rolling(r).mean()
    bcos['7']= bcos['7'].rolling(r).mean()

    fig, (ax1,ax2) = plt.subplots(1,2,figsize=set_size(516,fraction=0.5),gridspec_kw={'wspace': 0.12,'hspace':0.2},width_ratios=[1, 2])
    sea.lineplot(ax=ax1,x="Step",y="0",data=bcos,label="0 (Low)",linewidth=linsize,color=cols[0])
    sea.lineplot(ax=ax1,x="Step",y="4",data=bcos,label="4",linewidth=linsize,color=cols[4])
    sea.lineplot(ax=ax1,x="Step",y="7",data=bcos,label="7 (High)",linewidth=linsize,color=cols[7])

    ax1.set_ylabel("",rotation=0)
    ax1.set_ylim((-1,1))
    #ax1.yaxis.set_ticks([1,50,100,150,200])
    ax1.yaxis.set_label_coords(0.1,1.02)
    #ax1.set_yscale('log')

    ax1.set_xlabel("Batch")
    ax1.set_xlim([1,1500])
    start,end = ax1.get_xlim()
    ax1.xaxis.set_ticks([1,750,1500])
    ax1.grid(True,which='major',axis='both',linestyle='-',linewidth=0.2,alpha=0.5)
    ax1.grid(True,which='minor',axis='both',linestyle='--',linewidth=0.2,alpha=0.5)
    
    ax1.get_legend().remove()

    #cos['0'] = cos['0'].rolling(2).mean()
    #cos['4'] = cos['4'].rolling(2).mean()
    #cos['7'] = cos['7'].rolling(2).mean()

    sea.lineplot(ax=ax2,x="Step",y="0",data=cos,label="0 (Low)",linewidth=linsize,color=cols[0])
    sea.lineplot(ax=ax2,x="Step",y="4",data=cos,label="4",linewidth=linsize,color=cols[4])
    sea.lineplot(ax=ax2,x="Step",y="7",data=cos,label="7 (High)",linewidth=linsize,color=cols[7])

    ax2.set_xlabel("Epoch")
    #set the range of x axis
    ax2.set_xlim([1,150])
    ax2.xaxis.set_ticks([1,25,50,75,100,125,150])
    

    ax2.set_ylabel("",rotation=0)
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_coords(1,1.1)
    #ax2.set_yscale('log')
    ax2.set_ylim([-1,1])
    #ax2.yaxis.set_ticks([1e-12,1e-10,1e-8,1e-6,1e-4,1e-2,1e0,1e2,1e4])
    ax2.grid(True,which='major',axis='both',linestyle='-',linewidth=0.2,alpha=0.5)
    ax2.grid(True,which='minor',axis='both',linestyle='--',linewidth=0.2,alpha=0.5)

    ax2.get_legend().remove()

    


    fig.savefig("Pics/E_cos.png", bbox_inches='tight', dpi=500)

def make_enorm():
    bloss = pd.read_csv('E_bloss.csv',sep=',')
    bnorm = pd.read_csv('E_bnorm.csv',sep=',')
    bcos = pd.read_csv('E_bcos.csv',sep=',')
    bFIM = pd.read_csv('T2_batchV2.csv',sep=',')

    loss = pd.read_csv('E_loss.csv',sep=',')
    norm = pd.read_csv('E_norm.csv',sep=',')
    cos = pd.read_csv('E_cos.csv',sep=',')
    FIM = pd.read_csv('T2_epochV2.csv',sep=',')


    #loss
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=set_size(516,fraction=0.5),gridspec_kw={'wspace': 0.12,'hspace':0.2},width_ratios=[1, 2])
    sea.lineplot(ax=ax1,x="Step",y="0",data=bnorm,label="0 (Low)",linewidth=linsize,color=cols[0])
    sea.lineplot(ax=ax1,x="Step",y="1",data=bnorm,label="1",linewidth=linsize,color=cols[1])
    sea.lineplot(ax=ax1,x="Step",y="2",data=bnorm,label="2",linewidth=linsize,color=cols[2])
    sea.lineplot(ax=ax1,x="Step",y="3",data=bnorm,label="3",linewidth=linsize,color=cols[3])
    sea.lineplot(ax=ax1,x="Step",y="4",data=bnorm,label="4",linewidth=linsize,color=cols[4])
    sea.lineplot(ax=ax1,x="Step",y="5",data=bnorm,label="5",linewidth=linsize,color=cols[5])
    sea.lineplot(ax=ax1,x="Step",y="6",data=bnorm,label="6",linewidth=linsize,color=cols[6])
    sea.lineplot(ax=ax1,x="Step",y="7",data=bnorm,label="7 (High)",linewidth=linsize,color=cols[7])

    ax1.set_ylabel("",rotation=0)
    #ax1.set_ylim((10**0,10**3))
    #ax1.yaxis.set_ticks([1,50,100,150,200])
    ax1.yaxis.set_label_coords(0.1,1.02)
    #ax1.set_yscale('log')

    ax1.set_xlabel("Batch")
    ax1.set_xlim([1,1500])
    start,end = ax1.get_xlim()
    ax1.xaxis.set_ticks([1,750,1500])
    ax1.grid(True,which='major',axis='both',linestyle='-',linewidth=0.2,alpha=0.5)
    ax1.grid(True,which='minor',axis='both',linestyle='--',linewidth=0.2,alpha=0.5)
    
    ax1.get_legend().remove()

    sea.lineplot(ax=ax2,x="Step",y="0",data=norm,label="0 (Low)",linewidth=linsize,color=cols[0])
    sea.lineplot(ax=ax2,x="Step",y="1",data=norm,label="1",linewidth=linsize,color=cols[1])
    sea.lineplot(ax=ax2,x="Step",y="2",data=norm,label="2",linewidth=linsize,color=cols[2])
    sea.lineplot(ax=ax2,x="Step",y="3",data=norm,label="3",linewidth=linsize,color=cols[3])
    sea.lineplot(ax=ax2,x="Step",y="4",data=norm,label="4",linewidth=linsize,color=cols[4])
    sea.lineplot(ax=ax2,x="Step",y="5",data=norm,label="5",linewidth=linsize,color=cols[5])
    sea.lineplot(ax=ax2,x="Step",y="6",data=norm,label="6",linewidth=linsize,color=cols[6])
    sea.lineplot(ax=ax2,x="Step",y="7",data=norm,label="7 (High)",linewidth=linsize,color=cols[7])

    ax2.set_xlabel("Epoch")
    #set the range of x axis
    ax2.set_xlim([1,150])
    ax2.xaxis.set_ticks([1,25,50,75,100,125,150])
    

    ax2.set_ylabel("",rotation=0)
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_coords(1,1.1)
    ax2.set_yscale('log')
    #ax2.set_ylim([1e-10,1e5])
    #ax2.yaxis.set_ticks([1e-12,1e-10,1e-8,1e-6,1e-4,1e-2,1e0,1e2,1e4])
    ax2.grid(True,which='major',axis='both',linestyle='-',linewidth=0.2,alpha=0.5)
    ax2.grid(True,which='minor',axis='both',linestyle='--',linewidth=0.2,alpha=0.5)

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
                        xyB=(0,0.85),
                        coordsA="axes fraction",
                        coordsB="axes fraction",
                        axesA=ax1,
                        axesB=ax2,
                        color="black",
                        linestyle="--",)
    ax1.add_artist(con1)
    ax1.add_artist(con2)


    fig.savefig("Pics/E_norm.png", bbox_inches='tight', dpi=500)

def make_loss():
    T1 = pd.read_csv('T1_stats.csv',sep=',')
    T2 = pd.read_csv('T2_stats.csv',sep=',')
    T3 = pd.read_csv('T3_stats.csv',sep=',')
    T4 = pd.read_csv('T4_stats.csv',sep=',')

    fig, (ax1,ax2) = plt.subplots(1,2,figsize=set_size(516,height=129,fraction=1),gridspec_kw={'wspace': 0.1,'hspace':0.2},width_ratios=[1, 1])

    sea.lineplot(ax=ax1,x="Step",y="train_acc",data=T1,linewidth=linsize,linestyle='-',color='r',label="ResNetV1-14")
    sea.lineplot(ax=ax1,x="Step",y="train_acc",data=T2,linewidth=linsize,linestyle='-',color='g',label="CNN 5")
    sea.lineplot(ax=ax1,x="Step",y="train_acc",data=T3,linewidth=linsize,linestyle='-',color='b',label="CNN 8")
    sea.lineplot(ax=ax1,x="Step",y="train_acc",data=T4,linewidth=linsize,linestyle='-',color='m',label="ResNet18")

    sea.lineplot(ax=ax1,x="Step",y="test_acc",data=T1,linewidth=linsize,linestyle='--',color='r')
    sea.lineplot(ax=ax1,x="Step",y="test_acc",data=T2,linewidth=linsize,linestyle='--',color='g')
    sea.lineplot(ax=ax1,x="Step",y="test_acc",data=T3,linewidth=linsize,linestyle='--',color='b')
    sea.lineplot(ax=ax1,x="Step",y="test_acc",data=T4,linewidth=linsize,linestyle='--',color='m')

    ax1.set_ylabel("Accuracy")
    ax1.set_xlabel("Epochs")
    ax1.set_xlim([1,150])
    ax1.xaxis.set_ticks([1,25,50,75,100,125,150])
    ax1.legend(loc='center', bbox_to_anchor=(1, -0.4),ncol=4)

    ax2.set_ylabel("Loss")
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    ax2.set_xlabel("Epochs")
    ax2.set_xlim([1,150])
    ax2.xaxis.set_ticks([1,25,50,75,100,125,150])

    sea.lineplot(ax=ax2,x="Step",y="train_loss",data=T1,linewidth=linsize,linestyle='-',color='r')
    sea.lineplot(ax=ax2,x="Step",y="train_loss",data=T2,linewidth=linsize,linestyle='-',color='g')
    sea.lineplot(ax=ax2,x="Step",y="train_loss",data=T3,linewidth=linsize,linestyle='-',color='b')
    sea.lineplot(ax=ax2,x="Step",y="train_loss",data=T4,linewidth=linsize,linestyle='-',color='m')

    sea.lineplot(ax=ax2,x="Step",y="test_loss",data=T1,linewidth=linsize,linestyle='--',color='r')
    sea.lineplot(ax=ax2,x="Step",y="test_loss",data=T2,linewidth=linsize,linestyle='--',color='g')
    sea.lineplot(ax=ax2,x="Step",y="test_loss",data=T3,linewidth=linsize,linestyle='--',color='b')
    sea.lineplot(ax=ax2,x="Step",y="test_loss",data=T4,linewidth=linsize,linestyle='--',color='m')

    fig.savefig("Pics/stats.png", bbox_inches='tight', dpi=500)


def make_T1GFIM():
    T1FIMstep = pd.read_csv('T1_step.csv', sep=',')
    T1FIMstep = T1FIMstep.iloc[::1]
    print(T1FIMstep.head())

    fig, (ax1,ax2) = plt.subplots(1,2,figsize=set_size(516,fraction=0.5),gridspec_kw={'wspace': 0.15},width_ratios=[1, 2])
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
    ax1.set_ylabel("Loss GFIM",rotation=0)
    #set y label location to above the y axis

    #ax1.set_ylim([1,40])
    ax1.yaxis.set_ticks([1,10,20,30,40])
    ax1.yaxis.set_label_coords(0.1,1.02)
    ax1.set_yscale('log')
    #ax1.yaxis.set_label_rotation(90)

    ax1.set_xlabel("Batch")
    ax1.set_xlim([1,500])
    start,end = ax1.get_xlim()
    ax1.xaxis.set_ticks([1,250,500])
    ax1.grid(True,which='major',axis='both',linestyle='-',linewidth=0.2,alpha=0.5)
    ax1.grid(True,which='minor',axis='both',linestyle='--',linewidth=0.2,alpha=0.5)
    ax1.get_legend().remove()

    #Epoch graphs
    T1FIMepoch = pd.read_csv('T1_epoch.csv', sep=',')
    T1FIMepoch['Step'] = T1FIMepoch['Step']+1
    #T1FIMepoch = T1FIMepoch.iloc[:65]
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
    ax2.set_xlim([1,150])
    ax2.xaxis.set_ticks([1,25,50,75,100,125,150])

    ax2.set_ylabel("Loss GFIM",rotation=0)
    
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.set_label_coords(1,1.1)
    ax2.yaxis.tick_right()
    ax2.set_yscale('log')
    ax2.yaxis.set_ticks([1e-12,1e-10,1e-8,1e-6,1e-4,1e-2,1e0,1e2,1e4])

    ax2.grid(True,which='major',axis='both',linestyle='-',linewidth=0.2,alpha=0.5)
    ax2.grid(True,which='minor',axis='y',linestyle='--',linewidth=0.2,alpha=0.5)

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
    FIMstep = pd.read_csv('T2_batchV2.csv', sep=',')
    FIMstep = FIMstep.iloc[::1]

    fig, (ax1,ax2) = plt.subplots(1,2,figsize=set_size(516,fraction=0.5),gridspec_kw={'wspace': 0.12},width_ratios=[1, 2])
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
    ax1.set_ylabel("",rotation=0)
    ax1.set_ylim((10**0,10**3))
    #ax1.yaxis.set_ticks([1,50,100,150,200])
    ax1.yaxis.set_label_coords(0.1,1.02)
    ax1.set_yscale('log')

    ax1.set_xlabel("Batch")
    ax1.set_xlim([1,1500])
    start,end = ax1.get_xlim()
    ax1.xaxis.set_ticks([1,750,1500])
    ax1.grid(True,which='major',axis='both',linestyle='-',linewidth=0.2,alpha=0.5)
    ax1.grid(True,which='minor',axis='both',linestyle='--',linewidth=0.2,alpha=0.5)
    
    ax1.get_legend().remove()

    #Epoch graphs
    FIMepoch = pd.read_csv('T2_epochV2.csv', sep=',')
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
    ax2.set_xlim([1,150])
    ax2.xaxis.set_ticks([1,25,50,75,100,125,150])
    

    ax2.set_ylabel("Loss GFIM",rotation=0)
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_coords(1,1.1)
    ax2.set_yscale('log')
    ax2.set_ylim([1e-10,1e5])
    ax2.yaxis.set_ticks([1e-12,1e-10,1e-8,1e-6,1e-4,1e-2,1e0,1e2,1e4])
    ax2.grid(True,which='major',axis='both',linestyle='-',linewidth=0.2,alpha=0.5)
    ax2.grid(True,which='minor',axis='both',linestyle='--',linewidth=0.2,alpha=0.5)

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
    FIMstep = FIMstep.iloc[::1]
    
    print(FIMstep.head())

    fig, (ax1,ax2) = plt.subplots(1,2,figsize=set_size(516,fraction=0.5),gridspec_kw={'wspace': 0.14},width_ratios=[1, 2])
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
    ax1.set_ylabel("Loss GFIM",rotation=0)
    ax1.yaxis.set_label_coords(0.1,1.02)
    ax1.set_ylim([1,2000])
    ax1.set_yscale('log')
    #ax1.yaxis.set_ticks([1,200,400,600,800,1000,1200,1400,1600,1800,2000])

    ax1.set_xlabel("Batch")
    ax1.set_xlim([1,500])
    start,end = ax1.get_xlim()
    ax1.xaxis.set_ticks([1,250,500])
    ax1.grid(True,which='major',axis='both',linestyle='-',linewidth=0.2,alpha=0.5)
    ax1.grid(True,which='minor',axis='both',linestyle='--',linewidth=0.2,alpha=0.5)
    ax1.get_legend().remove()

    #Epoch graphs
    FIMepoch = pd.read_csv('T3_epoch.csv', sep=',')
    #add 1 to the step
    FIMepoch['Step'] = FIMepoch['Step']+1
    #use the top 60 epochs
    #FIMepoch = FIMepoch.iloc[:60]
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
    ax2.set_xlim([1,150])
    ax2.xaxis.set_ticks([1,25,50,75,100,125,150])

    ax2.set_ylabel("Loss GFIM",rotation=0)
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.set_label_coords(1,1.1)
    ax2.yaxis.tick_right()
    ax2.set_yscale('log')
    ax2.set_ylim([1e-12,1e5])
    ax2.yaxis.set_ticks([1e-12,1e-10,1e-8,1e-6,1e-4,1e-2,1e0,1e2,1e4])
    ax2.grid(True,which='major',axis='both',linestyle='-',linewidth=0.2,alpha=0.5)
    ax2.grid(True,which='minor',axis='both',linestyle='--',linewidth=0.2,alpha=0.5)

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
    #FIMstep = FIMstep.iloc[::15]

    fig, (ax1,ax2) = plt.subplots(1,2,figsize=set_size(516,fraction=0.5),gridspec_kw={'wspace': 0.14},width_ratios=[1, 2])
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
    ax1.set_ylabel("Loss GFIM",rotation=0)
    ax1.yaxis.set_label_coords(0.1,1.02)
    ax1.set_ylim([10,100000])
    ax1.set_yscale('log')
    #ax1.yaxis.set_ticks([1,100000])
    #ax1.set_yscale('log')

    ax1.set_xlabel("Batch")
    ax1.set_xlim([1,400])
    ax1.xaxis.set_ticks([1,200,400])
    ax1.grid(True,which='major',axis='both',linestyle='-',linewidth=0.2,alpha=0.5)
    ax1.grid(True,which='minor',axis='y',linestyle='--',linewidth=0.2,alpha=0.5)
    ax1.get_legend().remove()

    #Epoch graphs
    FIMepoch = pd.read_csv('T4_epoch.csv', sep=',')
    #add 1 to the step
    FIMepoch['Step'] = FIMepoch['Step']+1
    #use the top 60 epochs
    #FIMepoch = FIMepoch.iloc[:60]
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
    ax2.set_xlim([1,150])
    ax2.xaxis.set_ticks([1,25,50,75,100,125,150])

    ax2.set_ylabel("Loss GFIM",rotation=0)
    ax2.yaxis.set_label_coords(1,1.1)
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    ax2.set_yscale('log')
    ax2.yaxis.set_ticks([1e-10,1e-8,1e-6,1e-4,1e-2,1e0,1e2,1e4,1e6])
    ax2.grid(True,which='major',axis='both',linestyle='-',linewidth=0.2,alpha=0.5)
    ax2.grid(True,which='minor',axis='y',linestyle='--',linewidth=0.2,alpha=0.5)

    ax2.get_legend().remove()

    con1 = ConnectionPatch(xyA=(1,1),
                        xyB=(0,0.95),
                        coordsA="axes fraction",
                        coordsB="axes fraction",
                        axesA=ax1,
                        axesB=ax2,
                        color="black",
                        linestyle="--",)
    con2 = ConnectionPatch(xyA=(1,0),
                        xyB=(0,0.70),
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

def make_HAM():
    FIMstep = pd.read_csv('HAM_nopre.csv', sep=',')
    FIMstep['Step'] = FIMstep['Step']+1
    print(FIMstep.head())

    fig, ax1 = plt.subplots(1,1,figsize=set_size(252))

    sea.lineplot(ax=ax1,x="Step",y="0",data=FIMstep,label="0 (Low)",linewidth=linsize,color=cols[0])
    sea.lineplot(ax=ax1,x="Step",y="1",data=FIMstep,label="1",linewidth=linsize,color=cols[1])
    sea.lineplot(ax=ax1,x="Step",y="2",data=FIMstep,label="2",linewidth=linsize,color=cols[2])
    sea.lineplot(ax=ax1,x="Step",y="3",data=FIMstep,label="3",linewidth=linsize,color=cols[3])
    sea.lineplot(ax=ax1,x="Step",y="4",data=FIMstep,label="4",linewidth=linsize,color=cols[4])
    sea.lineplot(ax=ax1,x="Step",y="5",data=FIMstep,label="5",linewidth=linsize,color=cols[5])
    sea.lineplot(ax=ax1,x="Step",y="6",data=FIMstep,label="6",linewidth=linsize,color=cols[6])
    sea.lineplot(ax=ax1,x="Step",y="7",data=FIMstep,label="7 (High)",linewidth=linsize,color=cols[7])

    ax1.set_xlabel("Epoch")
    ax1.set_xlim([1,35])
    ax1.set_xticks([1,5,10,15,20,25,30,35])
    
    ax1.set_ylabel("Loss GFIM")
    ax1.set_yscale('log')
    ax1.set_yticks([1e-12,1e-10,1e-8,1e-6,1e-4,1e-2,1e0,1e2,1e4])
    ax1.get_legend().remove()

    fig.savefig("Pics/HAM.png", bbox_inches='tight', dpi=500)

    FIMstep = pd.read_csv('HAM_pre.csv', sep=',')
    FIMstep['Step'] = FIMstep['Step']+1
    print(FIMstep.head())

    fig, ax1 = plt.subplots(1,1,figsize=set_size(252))

    sea.lineplot(ax=ax1,x="Step",y="0",data=FIMstep,label="0 (Low)",linewidth=linsize,color=cols[0])
    sea.lineplot(ax=ax1,x="Step",y="1",data=FIMstep,label="1",linewidth=linsize,color=cols[1])
    sea.lineplot(ax=ax1,x="Step",y="2",data=FIMstep,label="2",linewidth=linsize,color=cols[2])
    sea.lineplot(ax=ax1,x="Step",y="3",data=FIMstep,label="3",linewidth=linsize,color=cols[3])
    sea.lineplot(ax=ax1,x="Step",y="4",data=FIMstep,label="4",linewidth=linsize,color=cols[4])
    sea.lineplot(ax=ax1,x="Step",y="5",data=FIMstep,label="5",linewidth=linsize,color=cols[5])
    sea.lineplot(ax=ax1,x="Step",y="6",data=FIMstep,label="6",linewidth=linsize,color=cols[6])
    sea.lineplot(ax=ax1,x="Step",y="7",data=FIMstep,label="7 (High)",linewidth=linsize,color=cols[7])

    ax1.set_xlabel("Epoch")
    ax1.set_xlim([1,20])
    ax1.set_xticks([1,5,10,15,20])
    ax1.set_ylabel("Loss GFIM")
    ax1.set_yscale('log')
    ax1.set_yticks([1e-12,1e-10,1e-8,1e-6,1e-4,1e-2,1e0,1e2,1e4])
    ax1.get_legend().remove()

    fig.savefig("Pics/HAM_pre.png", bbox_inches='tight', dpi=500)

def make_HAMcom():
    FIMstep = pd.read_csv('HAM_nopre.csv', sep=',')
    FIMstep['Step'] = FIMstep['Step']+1
    #take first 20 epochs
    FIMstep = FIMstep.iloc[:20]
    print(FIMstep.head())

    FIM1step = pd.read_csv('HAM_pre.csv', sep=',')
    FIM1step['Step'] = FIM1step['Step']+1

    fig, (ax1,ax2) = plt.subplots(1,2,figsize=set_size(252),width_ratios=[1, 2],gridspec_kw={'wspace': 0.12})

    sea.lineplot(ax=ax2,x="Step",y="0",data=FIMstep,label="_nolegend_",linewidth=linsize,color=cols[0])
    sea.lineplot(ax=ax2,x="Step",y="1",data=FIMstep,label="_nolegend_",linewidth=linsize,color=cols[1])
    sea.lineplot(ax=ax2,x="Step",y="2",data=FIMstep,label="_nolegend_",linewidth=linsize,color=cols[2])
    sea.lineplot(ax=ax2,x="Step",y="3",data=FIMstep,label="_nolegend_",linewidth=linsize,color=cols[3])
    sea.lineplot(ax=ax2,x="Step",y="4",data=FIMstep,label="_nolegend_",linewidth=linsize,color=cols[4])
    sea.lineplot(ax=ax2,x="Step",y="5",data=FIMstep,label="_nolegend_",linewidth=linsize,color=cols[5])
    sea.lineplot(ax=ax2,x="Step",y="6",data=FIMstep,label="_nolegend_",linewidth=linsize,color=cols[6])
    sea.lineplot(ax=ax2,x="Step",y="7",data=FIMstep,label="_nolegend_",linewidth=linsize,color=cols[7])

    sea.lineplot(ax=ax2,x="Step",y="0",data=FIM1step,label="_nolegend_",linewidth=linsize,color=cols[0],linestyle='-.')
    sea.lineplot(ax=ax2,x="Step",y="1",data=FIM1step,label="_nolegend_",linewidth=linsize,color=cols[1],linestyle='-.')
    sea.lineplot(ax=ax2,x="Step",y="2",data=FIM1step,label="_nolegend_",linewidth=linsize,color=cols[2], linestyle='-.')
    sea.lineplot(ax=ax2,x="Step",y="3",data=FIM1step,label="_nolegend_",linewidth=linsize,color=cols[3],linestyle='-.')
    sea.lineplot(ax=ax2,x="Step",y="4",data=FIM1step,label="_nolegend_",linewidth=linsize,color=cols[4], linestyle='-.')
    sea.lineplot(ax=ax2,x="Step",y="5",data=FIM1step,label="_nolegend_",linewidth=linsize,color=cols[5], linestyle='-.')
    sea.lineplot(ax=ax2,x="Step",y="6",data=FIM1step,label="_nolegend_",linewidth=linsize,color=cols[6], linestyle='-.')
    sea.lineplot(ax=ax2,x="Step",y="7",data=FIM1step,label="_nolegend_",linewidth=linsize,color=cols[7], linestyle='-.')

    ax2.set_xlabel("Epoch")
    ax2.set_xlim([1,20])
    ax2.set_xticks([1,5,10,15,20])
    
    ax2.set_ylabel("Loss GFIM",rotation=0)
    ax2.yaxis.set_label_coords(1,1.1)
    ax2.set_yscale('log')
    ax2.set_yticks([1e-12,1e-10,1e-8,1e-6,1e-4,1e-2,1e0,1e2,1e4])
    #ax2.get_legend().remove()

    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()

    FIMstep = pd.read_csv('HAM_step.csv', sep=',')
    FIMstep['Step'] = FIMstep['Step']+1

    FIM1step = pd.read_csv('HAM_step_pre.csv', sep=',')
    FIM1step['Step'] = FIM1step['Step']+1

    sea.lineplot(ax=ax1,x="Step",y="0",data=FIMstep,label="0 (Low)",linewidth=linsize,color=cols[0])
    sea.lineplot(ax=ax1,x="Step",y="1",data=FIMstep,label="1",linewidth=linsize,color=cols[1])
    sea.lineplot(ax=ax1,x="Step",y="2",data=FIMstep,label="2",linewidth=linsize,color=cols[2])
    sea.lineplot(ax=ax1,x="Step",y="3",data=FIMstep,label="3",linewidth=linsize,color=cols[3])
    sea.lineplot(ax=ax1,x="Step",y="4",data=FIMstep,label="4",linewidth=linsize,color=cols[4])
    sea.lineplot(ax=ax1,x="Step",y="5",data=FIMstep,label="5",linewidth=linsize,color=cols[5])
    sea.lineplot(ax=ax1,x="Step",y="6",data=FIMstep,label="6",linewidth=linsize,color=cols[6])
    sea.lineplot(ax=ax1,x="Step",y="7",data=FIMstep,label="7 (High)",linewidth=linsize,color=cols[7])

    sea.lineplot(ax=ax1,x="Step",y="0",data=FIM1step,label="_nolegend_",linewidth=linsize,color=cols[0],linestyle='-.')
    sea.lineplot(ax=ax1,x="Step",y="1",data=FIM1step,label="_nolegend_",linewidth=linsize,color=cols[1],linestyle='-.')
    sea.lineplot(ax=ax1,x="Step",y="2",data=FIM1step,label="_nolegend_",linewidth=linsize,color=cols[2], linestyle='-.')
    sea.lineplot(ax=ax1,x="Step",y="3",data=FIM1step,label="_nolegend_",linewidth=linsize,color=cols[3],linestyle='-.')
    sea.lineplot(ax=ax1,x="Step",y="4",data=FIM1step,label="_nolegend_",linewidth=linsize,color=cols[4], linestyle='-.')
    sea.lineplot(ax=ax1,x="Step",y="5",data=FIM1step,label="_nolegend_",linewidth=linsize,color=cols[5], linestyle='-.')
    sea.lineplot(ax=ax1,x="Step",y="6",data=FIM1step,label="_nolegend_",linewidth=linsize,color=cols[6], linestyle='-.')
    sea.lineplot(ax=ax1,x="Step",y="7",data=FIM1step,label="_nolegend_",linewidth=linsize,color=cols[7], linestyle='-.')

    ax1.set_xlabel("Batch")
    ax1.set_xlim([1,200])
    ax1.set_xticks([1,100,200])
    ax1.set_ylabel("Loss GFIM",rotation=0)
    ax1.yaxis.set_label_coords(0.1,1.02)
    ax1.set_yscale('log')
    ax1.set_ylim([1e0,1e5])
    ax1.get_legend().remove()
    fig.legend(loc='lower center',ncol=8,frameon=False,bbox_to_anchor=(0.5, -0.18),fontsize=5)
    legend1 = plt.legend([plt.Line2D([0], [0], color='black', lw=linsize,linestyle='-'),
                    plt.Line2D([0], [0], color='black', lw=linsize,linestyle='-.')],
                    ['No Pretraining', 'Pretraining'],
                    loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(0.2, -0.44),fontsize=5)

    con1 = ConnectionPatch(xyA=(1,1),
                        xyB=(0,1),
                        coordsA="axes fraction",
                        coordsB="axes fraction",
                        axesA=ax1,
                        axesB=ax2,
                        color="black",
                        linestyle="--",
                        linewidth=linsize,)
    con2 = ConnectionPatch(xyA=(1,0),
                        xyB=(0,0.7),
                        coordsA="axes fraction",
                        coordsB="axes fraction",
                        axesA=ax1,
                        axesB=ax2,
                        color="black",
                        linestyle="--",
                        linewidth=linsize,)
    ax1.add_artist(con1)
    ax1.add_artist(con2)

    fig.savefig("Pics/HAM1.png", bbox_inches='tight', dpi=500)

def make_VIT():
    #001
    FIMstep = pd.read_csv('VIT001.csv', sep=',')
    FIMstep['Step'] = FIMstep['Step']+1
    print(FIMstep.head())

    fig, ax1 = plt.subplots(1,1,figsize=set_size(252))

    sea.lineplot(ax=ax1,x="Step",y="0",data=FIMstep,label="0 (Low)",linewidth=linsize,color=cols[0])
    sea.lineplot(ax=ax1,x="Step",y="1",data=FIMstep,label="1",linewidth=linsize,color=cols[1])
    sea.lineplot(ax=ax1,x="Step",y="2",data=FIMstep,label="2",linewidth=linsize,color=cols[2])
    sea.lineplot(ax=ax1,x="Step",y="3",data=FIMstep,label="3",linewidth=linsize,color=cols[3])
    sea.lineplot(ax=ax1,x="Step",y="4",data=FIMstep,label="4",linewidth=linsize,color=cols[4])
    sea.lineplot(ax=ax1,x="Step",y="5",data=FIMstep,label="5",linewidth=linsize,color=cols[5])
    sea.lineplot(ax=ax1,x="Step",y="6",data=FIMstep,label="6",linewidth=linsize,color=cols[6])
    sea.lineplot(ax=ax1,x="Step",y="7",data=FIMstep,label="7 (High)",linewidth=linsize,color=cols[7])

    ax1.set_xlabel("Epoch")
    ax1.set_xlim([1,150])
    ax1.set_xticks([1,20,40,60,80,100,120,140,160])

    ax1.set_ylabel("Loss GFIM")
    #ax1.set_yscale('log')
    ax1.set_ylim([2,5])
    ax1.get_legend().remove()

    con1 = ConnectionPatch(xyA=(0,0.285),
                        xyB=(1,0.285),
                        coordsA="axes fraction",
                        coordsB="axes fraction",
                        axesA=ax1,
                        axesB=ax1,
                        color="black",
                        linestyle="--",)
    con2 = ConnectionPatch(xyA=(0,0.115),
                        xyB=(1,0.115),
                        coordsA="axes fraction",
                        coordsB="axes fraction",
                        axesA=ax1,
                        axesB=ax1,
                        color="black",
                        linestyle="--",)
    ax1.add_artist(con1)
    ax1.add_artist(con2)

    #add text labelling the value out of axis range
    ax1.text(0.8,0.7,"7509.6",transform=ax1.transAxes,ha='center',va='center',bbox=dict(boxstyle="square",ec=(0, 0, 0),fc=(1., 1, 1)))

    con3 = ConnectionPatch(xyA=(0.75,0.7),
                        xyB=(0.6,1),
                        coordsA="axes fraction",
                        coordsB="axes fraction",
                        axesA=ax1,
                        axesB=ax1,
                        color="black",
                        linestyle="-",
                        arrowstyle="->",)
    ax1.add_artist(con3)

    fig.savefig("Pics/VIT001.png", bbox_inches='tight', dpi=500)

def make_VIT001():
    #001--------------------------------------------------------------------------------

    FIMstep = pd.read_csv('VIT001_step.csv', sep=',')
    #T1FIMstep = T1FIMstep.iloc[::4]
    print(FIMstep.head())

    fig, (ax1,ax2) = plt.subplots(1,2,figsize=set_size(516,fraction=0.5),gridspec_kw={'wspace': 0.12},width_ratios=[1,2])
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
    ax1.set_ylabel("Loss GFIM",rotation=0)
    ax1.yaxis.set_label_coords(0.1,1.02)
    ax1.set_ylim([1e0,100000])
    ax1.set_yscale('log')
    #ax1.yaxis.set_ticks([1,50,100,150,200,250])

    ax1.set_xlabel("Batch")
    ax1.set_xlim([1,200])
    start,end = ax1.get_xlim()
    ax1.xaxis.set_ticks([1,100,200])
    ax1.get_legend().remove()
    ax1.grid(True,which='major',axis='both',linestyle='-',linewidth=0.2,alpha=0.5)
    ax1.grid(True,which='minor',axis='y',linestyle='--',linewidth=0.2,alpha=0.5)

    #Epoch graphs
    T1FIMepoch = pd.read_csv('VIT001.csv', sep=',')
    T1FIMepoch['Step'] = T1FIMepoch['Step']+1
    #T1FIMepoch = T1FIMepoch.iloc[:65]
    #print(T1FIMepoch.head())
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
    ax2.get_legend().remove()
    ax2.set_xlabel("Epoch")
    ax2.set_yscale('log')
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    ax2.set_ylabel("Loss GFIM",rotation=0)
    ax2.yaxis.set_label_coords(1,1.1)
    ax2.set_xticks([1,25,50,75,100])
    ax2.set_xlim([1,100])
    ax2.grid(True,which='major',axis='both',linestyle='-',linewidth=0.2,alpha=0.5)
    ax2.grid(True,which='minor',axis='y',linestyle='--',linewidth=0.2,alpha=0.5)

    con1 = ConnectionPatch(xyA=(1,1),
                        xyB=(0,1),
                        coordsA="axes fraction",
                        coordsB="axes fraction",
                        axesA=ax1,
                        axesB=ax2,
                        color="black",
                        linestyle="--",)
    con2 = ConnectionPatch(xyA=(1,0),
                        xyB=(0,0.70),
                        coordsA="axes fraction",
                        coordsB="axes fraction",
                        axesA=ax1,
                        axesB=ax2,
                        color="black",
                        linestyle="--",)
    ax1.add_artist(con1)
    ax1.add_artist(con2)

    fig.savefig("Pics/VIT001.png", bbox_inches='tight', dpi=500)

def make_VIT0001():
    #0001--------------------------------------------------------------------------------

    FIMstep = pd.read_csv('VIT0001_step.csv', sep=',')
    #T1FIMstep = T1FIMstep.iloc[::4]
    print(FIMstep.head())

    fig, (ax1,ax2) = plt.subplots(1,2,figsize=set_size(516,fraction=0.5),gridspec_kw={'wspace': 0.12},width_ratios=[1,2])
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
    ax1.set_ylabel("Loss GFIM",rotation=0)
    ax1.yaxis.set_label_coords(0.1,1.02)
    ax1.set_ylim([1e2,10000])
    ax1.set_yscale('log')
    #ax1.yaxis.set_ticks([1,50,100,150,200,250])

    ax1.set_xlabel("Batch")
    ax1.set_xlim([1,200])
    start,end = ax1.get_xlim()
    ax1.xaxis.set_ticks([1,100,200])
    ax1.get_legend().remove()
    ax1.grid(True,which='major',axis='both',linestyle='-',linewidth=0.2,alpha=0.5)
    ax1.grid(True,which='minor',axis='y',linestyle='--',linewidth=0.2,alpha=0.5)

    #Epoch graphs
    T1FIMepoch = pd.read_csv('VIT0001.csv', sep=',')
    T1FIMepoch['Step'] = T1FIMepoch['Step']+1
    #T1FIMepoch = T1FIMepoch.iloc[:65]
    #print(T1FIMepoch.head())
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
    ax2.get_legend().remove()
    ax2.set_xlabel("Epoch")
    ax2.set_yscale('log')
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    ax2.set_ylabel("Loss GFIM",rotation=0)
    ax2.yaxis.set_label_coords(1,1.1)
    ax2.set_xticks([1,25,50,75,100])
    ax2.set_xlim([1,100])
    ax2.grid(True,which='major',axis='both',linestyle='-',linewidth=0.2,alpha=0.5)
    ax2.grid(True,which='minor',axis='y',linestyle='--',linewidth=0.2,alpha=0.5)

    con1 = ConnectionPatch(xyA=(1,1),
                        xyB=(0,0.95),
                        coordsA="axes fraction",
                        coordsB="axes fraction",
                        axesA=ax1,
                        axesB=ax2,
                        color="black",
                        linestyle="--",)
    con2 = ConnectionPatch(xyA=(1,0),
                        xyB=(0,0.8),
                        coordsA="axes fraction",
                        coordsB="axes fraction",
                        axesA=ax1,
                        axesB=ax2,
                        color="black",
                        linestyle="--",)
    ax1.add_artist(con1)
    ax1.add_artist(con2)

    fig.savefig("Pics/VIT0001.png", bbox_inches='tight', dpi=500)

def make_VIT00001():

    #00001--------------------------------------------------------------------------------

    FIMstep = pd.read_csv('VIT00001_step.csv', sep=',')
    #T1FIMstep = T1FIMstep.iloc[::4]
    print(FIMstep.head())

    fig, (ax1,ax2) = plt.subplots(1,2,figsize=set_size(516,fraction=0.5),gridspec_kw={'wspace': 0.12},width_ratios=[1,2])
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
    ax1.set_ylabel("Loss GFIM",rotation=0)
    ax1.yaxis.set_label_coords(0.1,1.02)
    ax1.set_ylim([1e2,1e5])
    ax1.set_yscale('log')
    #ax1.yaxis.set_ticks([1,50,100,150,200,250])

    ax1.set_xlabel("Batch")
    ax1.set_xlim([1,300])
    start,end = ax1.get_xlim()
    ax1.xaxis.set_ticks([1,100,200,300])
    ax1.get_legend().remove()
    ax1.grid(True,which='major',axis='both',linestyle='-',linewidth=0.2,alpha=0.5)
    ax1.grid(True,which='minor',axis='y',linestyle='--',linewidth=0.2,alpha=0.5)

    #Epoch graphs
    T1FIMepoch = pd.read_csv('VIT00001.csv', sep=',')
    T1FIMepoch['Step'] = T1FIMepoch['Step']+1
    #T1FIMepoch = T1FIMepoch.iloc[:65]
    #print(T1FIMepoch.head())
    sea.lineplot(ax=ax2,x="Step",y="0",data=T1FIMepoch,label="0 (Low)",linewidth=linsize,color=cols[0])
    sea.lineplot(ax=ax2,x="Step",y="1",data=T1FIMepoch,label="1",linewidth=linsize,color=cols[1])
    sea.lineplot(ax=ax2,x="Step",y="2",data=T1FIMepoch,label="2",linewidth=linsize,color=cols[2])
    sea.lineplot(ax=ax2,x="Step",y="3",data=T1FIMepoch,label="3",linewidth=linsize,color=cols[3])
    sea.lineplot(ax=ax2,x="Step",y="4",data=T1FIMepoch,label="4",linewidth=linsize,color=cols[4])
    sea.lineplot(ax=ax2,x="Step",y="5",data=T1FIMepoch,label="5",linewidth=linsize,color=cols[5])
    sea.lineplot(ax=ax2,x="Step",y="6",data=T1FIMepoch,label="6",linewidth=linsize,color=cols[6])
    sea.lineplot(ax=ax2,x="Step",y="7",data=T1FIMepoch,label="7 (High)",linewidth=linsize,color=cols[7])

    #set the range of x axis
    ax2.get_legend().remove()
    ax2.set_xlabel("Epoch")
    ax2.set_yscale('log')
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    ax2.set_ylabel("Loss GFIM",rotation=0)
    ax2.yaxis.set_label_coords(1,1.1)
    #ax.set_ylabel("Loss GFIM")
    ax2.set_xticks([1,100,200,300,375])
    ax2.set_xlim([1,375])
    ax2.grid(True,which='major',axis='both',linestyle='-',linewidth=0.2,alpha=0.5)
    ax2.grid(True,which='minor',axis='y',linestyle='--',linewidth=0.2,alpha=0.5)


    con1 = ConnectionPatch(xyA=(1,1),
                        xyB=(0,0.95),
                        coordsA="axes fraction",
                        coordsB="axes fraction",
                        axesA=ax1,
                        axesB=ax2,
                        color="black",
                        linestyle="--",)
    con2 = ConnectionPatch(xyA=(1,0),
                        xyB=(0,0.65),
                        coordsA="axes fraction",
                        coordsB="axes fraction",
                        axesA=ax1,
                        axesB=ax2,
                        color="black",
                        linestyle="--",)
    ax1.add_artist(con1)
    ax1.add_artist(con2)

    fig.savefig("Pics/VIT00001.png", bbox_inches='tight', dpi=500)

def make_VIT000001():
    
    #000001--------------------------------------------------------------------------------

    FIMstep = pd.read_csv('VIT000001_step.csv', sep=',')
    #T1FIMstep = T1FIMstep.iloc[::4]
    #print(T1FIMstep.head())

    fig, (ax1,ax2) = plt.subplots(1,2,figsize=set_size(516,fraction=0.5),gridspec_kw={'wspace': 0.12},width_ratios=[1,2])
    sea.set(style="whitegrid")

    
    
    sea.lineplot(ax=ax1,x="Step",y="0",data=FIMstep,label="0 (Low)",linewidth=linsize,color=cols[0])
    sea.lineplot(ax=ax1,x="Step",y="1",data=FIMstep,label="1",linewidth=linsize,color=cols[1])
    sea.lineplot(ax=ax1,x="Step",y="2",data=FIMstep,label="2",linewidth=linsize,color=cols[2])
    sea.lineplot(ax=ax1,x="Step",y="3",data=FIMstep,label="3",linewidth=linsize,color=cols[3])
    sea.lineplot(ax=ax1,x="Step",y="4",data=FIMstep,label="4",linewidth=linsize,color=cols[4])
    sea.lineplot(ax=ax1,x="Step",y="5",data=FIMstep,label="5",linewidth=linsize,color=cols[5])
    sea.lineplot(ax=ax1,x="Step",y="6",data=FIMstep,label="6",linewidth=linsize,color=cols[6])
    sea.lineplot(ax=ax1,x="Step",y="7",data=FIMstep,label="7 (High)",linewidth=linsize,color=cols[7])
    
    ax1.set_ylabel("Loss GFIM",rotation=0)
    ax1.yaxis.set_label_coords(0.1,1.02)
    ax1.set_ylim([1000,100000])
    ax1.set_yscale('log')
    #ax1.yaxis.set_ticks([1,50,100,150,200,250])

    ax1.set_xlabel("Batch")
    ax1.set_xlim([1,300])
    #ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    start,end = ax1.get_xlim()
    ax1.xaxis.set_ticks([1,100,200,300])
    ax1.get_legend().remove()
    ax1.grid(True,which='major',axis='both',linestyle='-',linewidth=0.2,alpha=0.5)
    ax1.grid(True,which='minor',axis='y',linestyle='--',linewidth=0.2,alpha=0.5)

    #Epoch graphs
    T1FIMepoch = pd.read_csv('VIT000001.csv', sep=',')
    T1FIMepoch['Step'] = T1FIMepoch['Step']+1
    #T1FIMepoch = T1FIMepoch.iloc[:65]
    #print(T1FIMepoch.head())
    sea.lineplot(ax=ax2,x="Step",y="0",data=T1FIMepoch,label="0 (Low)",linewidth=linsize,color=cols[0])
    sea.lineplot(ax=ax2,x="Step",y="1",data=T1FIMepoch,label="1",linewidth=linsize,color=cols[1])
    sea.lineplot(ax=ax2,x="Step",y="2",data=T1FIMepoch,label="2",linewidth=linsize,color=cols[2])
    sea.lineplot(ax=ax2,x="Step",y="3",data=T1FIMepoch,label="3",linewidth=linsize,color=cols[3])
    sea.lineplot(ax=ax2,x="Step",y="4",data=T1FIMepoch,label="4",linewidth=linsize,color=cols[4])
    sea.lineplot(ax=ax2,x="Step",y="5",data=T1FIMepoch,label="5",linewidth=linsize,color=cols[5])
    sea.lineplot(ax=ax2,x="Step",y="6",data=T1FIMepoch,label="6",linewidth=linsize,color=cols[6])
    sea.lineplot(ax=ax2,x="Step",y="7",data=T1FIMepoch,label="7 (High)",linewidth=linsize,color=cols[7])

    #set the range of x axis
    ax2.get_legend().remove()
    ax2.set_xlabel("Epoch")
    ax2.set_yscale('log')
    ax2.set_ylim([1e2,1e5])
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    ax2.set_ylabel("Loss GFIM",rotation=0)
    ax2.yaxis.set_label_coords(1,1.1)
    #ax.set_ylabel("Loss GFIM")
    ax2.set_xticks([1,100,200,300,375])
    ax2.set_xlim([1,375])
    ax2.grid(True,which='major',axis='both',linestyle='-',linewidth=0.2,alpha=0.5)
    ax2.grid(True,which='minor',axis='y',linestyle='--',linewidth=0.2,alpha=0.5)

    con1 = ConnectionPatch(xyA=(1,1),
                        xyB=(0,1),
                        coordsA="axes fraction",
                        coordsB="axes fraction",
                        axesA=ax1,
                        axesB=ax2,
                        color="black",
                        linestyle="--",)
    con2 = ConnectionPatch(xyA=(1,0),
                        xyB=(0,0.33),
                        coordsA="axes fraction",
                        coordsB="axes fraction",
                        axesA=ax1,
                        axesB=ax2,
                        color="black",
                        linestyle="--",)
    ax1.add_artist(con1)
    ax1.add_artist(con2)

    fig.savefig("Pics/VIT000001.png", bbox_inches='tight', dpi=500)

#make_VIT00001()

make_enorm()

#make_VIT()