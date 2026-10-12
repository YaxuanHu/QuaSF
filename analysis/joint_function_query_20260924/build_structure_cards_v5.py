"""Figure 3 with distinct colors for all predicted regions and explicit support markers."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
D=Path(__file__).resolve().parent
records=json.loads((D/'multidomain_figure_manifest_v4.json').read_text(encoding='utf-8'))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'text.color':'#243334'})
names={'Q7SDR1':['Cellulose binding','Hydrolase'],'A1L0T0':['TPP binding','Mg²⁺ binding','Catalysis'],'A9KIG5':['Zn²⁺ binding','ATPase','ATP binding','Peptidase'],'Q7SIG6':['Protein binding','Protein binding','GTPase activator','Protein binding','Protein binding']}
w=185/25.4;h=3.12
fig=plt.figure(figsize=(w,h),facecolor='white');texts=[]
for i,c in enumerate(records):
 left=.006+i*.249;fw=.241
 ax=fig.add_axes([left,.12,fw,.87]);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
 ax.add_patch(FancyBboxPatch((.004,.004),.992,.992,boxstyle='round,pad=0.003,rounding_size=.020',edgecolor='#BDC7C7',facecolor='white',linewidth=.6))
 texts.append((ax.text(.5,.945,c['accession'],ha='center',va='center',fontsize=9.5,fontweight='bold'),ax))
 suffix='molstar_v5' if c['accession'] in ('A9KIG5','Q7SIG6') else 'colored'
 raw=Image.open(D/'structures/rendered'/f"{c['accession']}_{suffix}.png").convert('RGBA')
 white=Image.new('RGBA',raw.size,'white');white.alpha_composite(raw);raw=white.convert('RGB');pix=np.array(raw)
 # Frame all colored domain pixels, including predictions without annotations.
 chroma=pix.max(2).astype(int)-pix.min(2).astype(int); yy,xx=np.where((chroma>12)&(pix.min(2)<195))
 side=int(max(xx.max()-xx.min()+1,yy.max()-yy.min()+1)*1.12);cx=(xx.min()+xx.max()+1)/2;cy=(yy.min()+yy.max()+1)/2
 framed=Image.new('RGB',(side,side),'white');framed.paste(raw,(-int(cx-side/2),-int(cy-side/2)))
 ia=fig.add_axes([left+fw*.09,.431,fw*.82,fw*.82*w/h]);ia.imshow(framed);ia.axis('off')
 ax.plot([.075,.925],[.348,.348],color='#DFE6E5',lw=.55)
 for j,(d,name) in enumerate(zip(c['regions'],names[c['accession']])):
  y=.300-j*.057
  ax.scatter([.104],[y],s=29,marker='o' if d['annotation_support'] else 'D',facecolors=d['color'] if d['annotation_support'] else 'white',edgecolors=d['color'],linewidths=1.2,zorder=5)
  texts.append((ax.text(.163,y,name,fontsize=9,va='center'),ax))
legend=fig.add_axes([.06,.016,.88,.08]);legend.axis('off');legend.set(xlim=(0,1),ylim=(0,1))
legend.scatter([.025],[.54],s=26,color='#243334',marker='o');legend.text(.055,.54,'Annotation-supported prediction',fontsize=9,va='center')
legend.scatter([.55],[.54],s=26,facecolors='white',edgecolors='#243334',marker='D',linewidths=1.1);legend.text(.58,.54,'Prediction for an unannotated region',fontsize=9,va='center')
fig.canvas.draw()
for t,ax in texts:assert t.get_window_extent().x1<ax.get_window_extent().x1-2,t.get_text()
fig.savefig(D/'multidomain_structure_cards_v5.pdf');fig.savefig(D/'multidomain_structure_cards_v5.png',dpi=360)
print('Saved Figure 3 v5: 185 mm wide, all labels 9 pt; accessions 9.5 pt.')
