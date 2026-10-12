import json
from pathlib import Path
p=Path(__file__).parent;r=p.parents[1]
cs=json.loads((p/'candidates.json').read_text());ids={x['acc'] for x in cs}
files={'ref':r/'tmp/region_mlp_transfer/test.jsonl','shared':r/'analysis/joint_function_query_20260924/comparison_archives/shared_best_test.jsonl','quasf':r/'tmp/direct_results_0930/predictions.jsonl'}
all={}
for name,path in files.items():
 rows={}
 for line in path.open(encoding='utf-8'):
  row=json.loads(line);acc=row.get('accession',row.get('uniprot_acc'))
  if acc in ids:rows[acc]=row
 all[name]=rows
(p/'case_records.json').write_text(json.dumps(all),encoding='utf-8')
for acc in ['P43561','O24744','P0A2E3','P00579']:
 c=next(x for x in cs if x['acc']==acc)
 print('\n',acc)
 for key in c['keys']:
  pi,gi=map(int,key.split(':')[1:]);d=all['ref'][acc]['domains'][gi]
  print('ref',gi,d['segments'],[(x['go_name'],x['level_2_cluster_id']) for x in d['source_go_labels']])
 print('shared matches',all['shared'][acc].get('matches'))
 for pred in all['shared'][acc]['predictions']:
  print('shared',pred['segments'],pred.get('source_query_id'),[pred['function_probs'][i] for i in sorted(set(sum(c['classes'],[])))])
 for name in ('mlp','quasf'):print(name,[[row[i] for i in sorted(set(sum(c['classes'],[])))] for row in c[name]])
