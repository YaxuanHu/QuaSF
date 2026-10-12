import numpy as np,json
from pathlib import Path
r=Path('/data/huyaxuan/code/region_function_multimodal')
z=np.load(r/'output/region_mlp_trial_20261004/test.npz')
y,g,q,m,k=(z[a] for a in ('labels','groups','quasf','mlp','keys'))
common={x['accession'] for x in json.loads((r/'input/joint_eval_common9015_cache_test/index.json').read_text())}
out=[]
for acc in np.unique(g):
 if acc not in common:continue
 ix=np.where(g==acc)[0]
 if len(ix)!=2:continue
 a,b=ix
 for c in np.where(y[a]^y[b])[0]:
  pos,neg=(a,b) if y[a,c] else (b,a)
  if m[pos,c]<m[neg,c] and q[pos,c]>q[neg,c]:
   out.append(dict(acc=str(acc),c=int(c),keys=k[ix].tolist(),classes=[np.where(y[i])[0].tolist() for i in ix],mlp=m[ix].tolist(),quasf=q[ix].tolist(),margin=float(m[neg,c]-m[pos,c])))
out.sort(key=lambda a:-a['margin'])
Path('/tmp/mlp_failure_candidates.json').write_text(json.dumps(out))
print(json.dumps([{k:v for k,v in a.items() if k not in ('mlp','quasf')}|{'mlp':[row[a['c']] for row in a['mlp']],'quasf':[row[a['c']] for row in a['quasf']]} for a in out[:15]],indent=2))
