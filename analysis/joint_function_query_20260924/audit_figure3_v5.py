import json,fitz
from pathlib import Path
D=Path('analysis/joint_function_query_20260924');records=json.loads((D/'multidomain_figure_manifest_v4.json').read_text(encoding='utf-8'))
for c in records:
 suffix='molstar_v5' if c['accession'] in ('A9KIG5','Q7SIG6') else 'colored'
 p=D/'structures/rendered'/f"{c['accession']}_{suffix}.png.render.json";r=json.loads(p.read_text(encoding='utf-8'));layers=r['provenance']['replay']['scene']['layers'][1:]
 assert len(layers)==c['region_count']
 for d,l in zip(c['regions'],layers):
  assert l['color']['value']==d['color']
  st,en=d['predicted_segments'][0];assert l['target']['start']==st+1 and l['target']['end']==en
pdf=fitz.open(D/'multidomain_structure_cards_v5.pdf');sizes=[s['size'] for b in pdf[0].get_text('dict')['blocks'] if 'lines'in b for l in b['lines'] for s in l['spans']];assert min(sizes)>=8.999
pdf[0].get_pixmap(matrix=fitz.Matrix(2,2)).save(D/'figure3_v5_pdf_review.png')
print('All 14 Mol* residue ranges and colors verified; font minimum9pt.')
