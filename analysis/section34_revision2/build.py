from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.colors import LinearSegmentedColormap
import pymupdf

P=Path(__file__).parent
plt.rcParams.update({'font.family':'Arial','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none','mathtext.fontset':'stix'})

def save(fig,name):
    fig.savefig(P/(name+'.pdf'),bbox_inches='tight',pad_inches=.05)
    fig.savefig(P/(name+'.svg'),bbox_inches='tight',pad_inches=.05)
    doc=pymupdf.open(P/(name+'.pdf'))
    doc[0].get_pixmap(matrix=pymupdf.Matrix(2.5,2.5)).save(P/(name+'.png'))
    plt.close(fig)

def diagram():
    fig,axs=plt.subplots(1,3,figsize=(7.1,3.75),gridspec_kw={'wspace':.14})
    fig.subplots_adjust(left=.01,right=.99,top=.90,bottom=.025)
    blue='#e8f0f9';green='#e6f1e8';gray='#f2f3f4';orange='#fff0da';ink='#243440'
    for j,ax in enumerate(axs):
        ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
        ax.text(.5,1.07,['(a) Shared query','(b) Region MLP','(c) QuaSF'][j],ha='center',va='center',weight='bold',fontsize=10.5,color=ink)
        def box(x,y,w,h,label,color=gray,size=9):
            ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=0.007,rounding_size=0.014',facecolor=color,edgecolor='#8d989f',linewidth=.65))
            ax.text(x,y,label,ha='center',va='center',fontsize=size,color=ink)
        def arr(x,y,xx,yy):
            ax.annotate('',xy=(xx,yy),xytext=(x,y),arrowprops={'arrowstyle':'->','lw':.8,'color':ink,'shrinkA':0,'shrinkB':0,'mutation_scale':8})
        box(.5,.94,.30,.075,r'$q_i$',orange,12)
        arr(.5,.894,.5,.859)
        box(.5,.805,.93,.095,'Segmentation and\nregion decoding')
        ax.plot([.5,.5],[.749,.726],lw=.8,color=ink)
        ax.plot([.25,.75],[.726,.726],lw=.8,color=ink)
        for x,n in ((.25,1),(.75,2)):
            arr(x,.726,x,.693);box(x,.645,.40,.08,f'Region {n}',blue)
        if j==0:
            ax.plot([.25,.25,.75,.75],[.596,.56,.56,.596],lw=.8,color=ink)
            ax.text(.5,.503,'Inherit function scores\nfrom the source query',ha='center',va='center',fontsize=8.8,color=ink)
            arr(.5,.435,.5,.364)
            box(.5,.315,.78,.08,r'Shared vector  $p_i$',orange)
            ax.plot([.5,.5],[.266,.23],lw=.8,color=ink)
            ax.plot([.25,.75],[.23,.23],lw=.8,color=ink)
            for x in (.25,.75):arr(x,.23,x,.182)
            values=((.78,.25,.48),(.78,.25,.48))
        else:
            for x in (.25,.75):arr(x,.597,x,.554)
            box(.5,.505,.93,.085,'Mean + max pooling',gray)
            for x,n in ((.25,1),(.75,2)):
                arr(x,.454,x,.421)
                box(x,.38,.35,.065,rf'$h_{n}$' if j==1 else rf'$q^f_{n}$',blue if j==1 else green,12)
                arr(x,.338,x,.301)
            box(.5,.251,.93,.085,'Shared MLP' if j==1 else 'Function decoder',gray if j==1 else green)
            for x in (.25,.75):arr(x,.201,x,.169)
            values=((.78,.25,.48),(.25,.81,.41))
        for x,n,vals in zip((.25,.75),(1,2),values):
            ax.text(x,.139,rf'$p_{n}$',ha='center',va='center',fontsize=12,color=ink)
            for k,v in enumerate(vals):
                ax.add_patch(Rectangle((x-.14,.09-k*.022),.28,.012,facecolor='#e9edef',edgecolor='none'))
                ax.add_patch(Rectangle((x-.14,.09-k*.022),.28*v,.012,facecolor='#6c90b4',edgecolor='none'))
    save(fig,'regional_predictors_v2')

def scores():
    records=json.loads((P/'case_records.json').read_text());cs=json.loads((P/'candidates.json').read_text())
    c=next(x for x in cs if x['acc']=='P43561');ref=records['ref'][c['acc']]
    classes=[34,52];models={};domains=[]
    order=sorted(range(2),key=lambda i:ref['domains'][int(c['keys'][i].split(':')[2])]['segments'][0][0])
    for name in ['Shared query','Region MLP','QuaSF']:
        rows=[]
        for i in order:
            pi,gi=map(int,c['keys'][i].split(':')[1:]);d=ref['domains'][gi]
            pred=records['quasf'][c['acc']]['predictions'][pi]
            if name=='Shared query':
                saved=records['shared'][c['acc']];matches=[a for a in saved['matches']['0.5'] if a[1]==gi];assert len(matches)==1
                pred=saved['predictions'][matches[0][0]];values=[pred['function_probs'][x] for x in classes]
            else:values=[c['mlp' if name=='Region MLP' else 'quasf'][i][x] for x in classes]
            rows.append(dict(scores=values,reference_segments=d['segments'],predicted_segments=pred['segments'],labels=[x in d['function_cluster_l2_indices'] for x in classes]))
        models[name]=rows
    assert models['Region MLP'][0]['scores'][1]>models['Region MLP'][1]['scores'][1]
    assert models['QuaSF'][0]['scores'][1]<models['QuaSF'][1]['scores'][1]
    audit=dict(accession=c['acc'],length=len(ref['sequence']),classes=classes,models=models,selection='Selected to illustrate an MLP regional-preference error corrected by QuaSF; two annotated reference regions, not all protein regions.')
    (P/'scores.json').write_text(json.dumps(audit,indent=2))
    fig=plt.figure(figsize=(3.5,2.95))
    fig.text(.02,.98,f"P43561 | {audit['length']} aa",va='top',fontsize=9,weight='bold')
    ax=fig.add_axes([.405,.29,.455,.56]);values=np.array([r['scores'] for model in models for r in models[model]])
    cmap=LinearSegmentedColormap.from_list('scores',['#f5f7f7','#428c90','#174e54'])
    ax.imshow(values,vmin=0,vmax=1,cmap=cmap,aspect='auto');ax.set_xticks([0,1],['C2_034','C2_052'],fontsize=7.4);ax.xaxis.tick_top();ax.tick_params(length=0,pad=5);ax.set_yticks([])
    for i,row in enumerate([r for model in models for r in models[model]]):
        for j,value in enumerate(row['scores']):
            ax.text(j,i,f'{value:.4f}',ha='center',va='center',fontsize=8,color='white' if value>.55 else '#243440')
            if row['labels'][j]:ax.add_patch(Rectangle((j-.5,i-.5),1,1,fill=False,edgecolor='#db884e',lw=1.2))
        fig.text(.31,.29+.56*(1-(i+.5)/6),'R1' if i%2==0 else 'R2',va='center',fontsize=8)
    for y in (1.5,3.5):ax.axhline(y,color='white',lw=4)
    for k,model in enumerate(models):fig.text(.02,.29+.56*(1-(2*k+1)/6),model.replace(' ','\n'),va='center',fontsize=8,weight='bold' if model=='QuaSF' else 'normal')
    for spine in ax.spines.values():spine.set_visible(False)
    cb=fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0,1),cmap=cmap),cax=fig.add_axes([.91,.29,.025,.56]),ticks=[0,.5,1]);cb.ax.tick_params(labelsize=6,length=2,pad=2);cb.outline.set_visible(False)
    for y,t in ((.215,'R1: 17-150    R2: 343-536'),(.155,'C2_034: Copper ion binding'),(.095,'C2_052: Oxidoreductase activity')):fig.text(.02,y,t,fontsize=7.3)
    fig.add_artist(Rectangle((.02,.03),.035,.025,transform=fig.transFigure,fill=False,edgecolor='#db884e',lw=1.2))
    fig.text(.075,.043,'Recorded region-function association',fontsize=7,va='center')
    save(fig,'P43561_mlp_failure')

diagram();scores()
