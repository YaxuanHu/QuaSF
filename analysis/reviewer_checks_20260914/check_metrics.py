import tempfile
from pathlib import Path
import numpy as np
from run_checks import Prediction,segmentation,connected_components,heterogeneous,ranking,collect_calls

# Adjacent domains must remain distinct in identity scoring but merge in legacy scoring.
g=np.array([[1,1,0,0],[0,0,1,1]],bool)
y=np.array([[1,0],[0,1]],bool);v=np.array([True,True])
p=[Prediction(0,np.ones(4,bool),np.array([.8,.8]),1.,1.)]
s,_=segmentation(p,g,4,thresholds=(.5,))
assert s['domain_iou']==.25 and s['domain_tp@0.5']==1
cm,cl,cv,contributors=connected_components(g,y,v,4)
assert heterogeneous(y,v) and not heterogeneous(cl,cv)
assert len(cm)==1 and contributors==[2] and cl.sum()==2

# An unmatched prediction remains a scored false call; an unannotated match is ignored.
p2=[Prediction(i,np.ones(4,bool),np.array([.9,.2]),1.,1.) for i in range(3)]
scores,truth=collect_calls(p2,y,[True,False],[(0,0,1.),(1,1,1.)])
assert len(scores)==2 and truth.sum()==1

# Identical scores must not favor whichever label/region happens to appear first.
with tempfile.TemporaryDirectory() as d:
    result=ranking(np.ones((1,2000),np.float32),np.array([[True]*1000+[False]*1000]),1000,Path(d)/'curve.csv')
    r=result['budgets'][0]
    assert r['calls']==1000 and r['expected_correct']==500 and r['cutoff_tie_size']==2000
print('Identity matching, component heterogeneity, missing annotations and score ties passed.')
