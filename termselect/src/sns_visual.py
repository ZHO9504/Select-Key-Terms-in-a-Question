import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
def paint(name,dim3_data):
    if (dim3_data).detach().cpu().numpy().ndim==3:
        #print("3dims__to__2dims",(dim3_data).detach().cpu().numpy().shape)
        #for i in range(len(dim3_data)):
        #    print(name+"num"+str(i)+"times!")
            
        #    dim2_data=dim3_data[i].detach().cpu().numpy()
        #    data=np.transpose(dim2_data)
        #    draw(name+str(i),data)
        dim2_data=dim3_data[-1].detach().cpu().numpy()
        data=np.transpose(dim2_data)
        draw(name,data)
    if dim3_data.detach().cpu().numpy().ndim==2:
        #print("2dims_to_1dim")
        #for i in range(len(dim3_data)):
        #    dim_data=data=dim3_data[i].detach().cpu().numpy()
        #    data=np.transpose(dim_data)
        #    draw(name+str(i),data)
        data=np.transpose(dim3_data.detach().cpu().numpy())
        draw(name,data)
def draw(name,data):
    print("\n\n",name,"\n",data,"\n")
    print("2_dim:",data.shape)
    sent1=['After','living','in' ,'the' ,'house' ,'for' ,'a','year']
    sent2=['soon','after','buying' ,'the' ,'house' ]
    sentFour1=['They','were','working','on','their','new','house','.']
    sentFour2=['The','contractor','\'s','house','.']
    sentSeven1=['after','they','moved','out']
    sentSeven2=['when','they','moved','in']
    sentEight1=['They','painted','the','walls','.']
    sentEight2=['They','emptied','the','living','room','.']
    s401_1=['The','child','wanted','to','watch','a','Star','Wars','movies','.']
    s401_2=['The','child','wanted','to','continue','playing','.']
    s365=['At','the','convenience','store','at','the','end','of','the','block']
    s387_0=['they', 'had' ,'to' ,'wear' ,'high' ,'heels' ,'that', 'night', 'since', 'it' ,'was' ,'a', 'fancy' ,'dress' ,'party']
    s387_1=['you' ,'can','not', 'wear','anything' ,'other', 'than' ,'special', 'bowling' ,'shoes' ,'or' ,'you' ,'ruin' ,'the' ,'floors']
    x=s401_2
    q1=['When', 'did' ,'they', 'decide','to' ,'renovate','?']
    q4=['Whose','house','was','the','room','in','?']
    q7=['When','was','it','renovated','?']
    q8=['How','did','they','begin','?']
    q401=['Why','did','not','the','child','go','to','bed','by','themselves','?']
    q365=['Where','was','the','vending','machine','?']
    q387=['Why', 'did' ,'they', 'have' ,'to' ,'change', 'shoes','?']
    y=q401
    #print ("\n\n111\n",num,"\n22222")
    if(data.ndim==1):
        return
        #f,ax1 = plt.subplots(figsize = (data.shape[0],data.shape[0]),nrows=1)
    else:
        f,ax1 = plt.subplots(figsize = (data.shape[1],data.shape[1]),nrows=1)
    
    #f,ax1 = plt.subplots(figsize = (data.shape[1],data.shape[1]),nrows=1)
    #cmap = sns.cubehelix_palette(start = 1.5, rot = 3, gamma=0.8, as_cmap = True)
    #cmap=sns.palplot(sns.color_palette("Blues"))
    cmap=sns.palplot(sns.light_palette("navy"))
    #seaborn的cubehelix_palette()函数返回调色板
    sns.heatmap(data, linewidths = 0.00, ax = ax1,cmap=cmap,center=1.4)# vmax=0.08, vmin=0, cmap=cmap,center=0.1)#,center=1.4)
    alpha=[]
    for i in range(100):
        alpha.append('dot-alpha')
        alpha.append('dot-alpha'+str(i))
        #alpha.append('self-alpha'+str(i))
    if name in (alpha):
        ax1.set_title('Attention',fontsize=30)
        ax1.set_ylabel("Question",fontsize=25,rotation=90)
        ax1.set_xticklabels(x,fontsize=20,rotation=90)
        ax1.set_yticklabels(y,fontsize=20,rotation=0)#
        ax1.set_xlabel("Choice",fontsize=25)
    elif name =='self-alpha':
        ax1.set_title('Attention',fontsize=10)
        #ax1.set_ylabel("Question",fontsize=35,rotation=90)
        #ax1.set_xticklabels(x,fontsize=5,rotation=90)
        yminorLocator   = MultipleLocator(1)#set the kedu
        ax1.yaxis.set_minor_locator(yminorLocator)  

        ax1.set_yticklabels(x,fontsize=5,rotation=0)#
         
        ax1.set_xlabel("Choice",fontsize=15)
    else:
        ax1.set_title(name,fontsize=30)
        ax1.set_ylabel("Representation",fontsize=20,rotation=90)
        #start, end = ax1.get_ylim()
        ymajorLocator   = MultipleLocator(20)
        ax1.yaxis.set_major_locator(ymajorLocator)
        #yminorLocator   = MultipleLocator(5)#set the kedu
        #ax1.yaxis.set_minor_locator(yminorLocator)  
        ax1.yaxis.grid(True, which='major')
        ax1.set_yticklabels(np.arange(0,data.shape[0]+1,20)) 
        #print("start",start,end)
        #ax1.yaxis.set_ticks(np.arange(end, start, 10))
        #ymajorLocator   = MultipleLocator(10)
        #ax1.set_yticklabels([range(0,data.shape[0],10)])
        #ax1.yaxis.set_major_locator(MultipleLocator(20)) #xian shi jiange
        ax1.set_xticklabels(x,fontsize=20,rotation=90)
        ax1.set_xlabel("Choice",fontsize=30)
    #ax1.set_xticklabels([])
    #f.tight_layout()
    f.savefig('pic_output/'+name+'.png', bbox_inches='tight')


