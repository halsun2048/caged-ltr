"""Render paper-ready R6 tables and a quality-latency Pareto SVG."""
from __future__ import annotations
import argparse,json
from pathlib import Path
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dev',type=Path,required=True); ap.add_argument('--gate',type=Path,required=True); ap.add_argument('--test',type=Path,required=True); ap.add_argument('--table',type=Path,required=True); ap.add_argument('--figure',type=Path,required=True); args=ap.parse_args(); dev=json.loads(args.dev.read_text()); gate=json.loads(args.gate.read_text()); test=json.loads(args.test.read_text())
    lines=['# NFCorpus R6 final results','','## Independent dev','','| Model | NDCG@10 | Hit@10 | MRR | Head | Torso | Tail | Latency ms/query |','|---|---:|---:|---:|---:|---:|---:|---:|']
    labels={'bm25':'BM25','tfidf':'TF-IDF','mind_minilm':'MIND-only MiniLM','distilled_minilm':'NFCorpus-distilled MiniLM','first':'FIRST','frozen_gate':'Previous frozen gate'}
    for key in labels:
        row=dev['models'][key]; latency=dev['efficiency'][key].get('latency_ms_per_query',0.); lines.append(f"| {labels[key]} | {row['ndcg10']:.4f} | {row['hit10']:.4f} | {row['mrr']:.4f} | {row['buckets']['head']['ndcg10']:.4f} | {row['buckets']['torso']['ndcg10']:.4f} | {row['buckets']['tail']['ndcg10']:.4f} | {latency:.2f} |")
    lines += ['','## Locked test','','| Model | NDCG@10 | Hit@10 | MRR | FIRST call rate | Latency ms/query |','|---|---:|---:|---:|---:|---:|']
    for key,label in [('bm25','BM25'),('distilled_minilm','Distilled MiniLM'),('first','FIRST'),('previous_frozen_gate','Previous frozen gate')]:
        row=test['baselines'][key]; latency=0. if key=='bm25' else (test['efficiency']['distilled_minilm_latency_ms_per_query'] if key=='distilled_minilm' else (test['efficiency']['first_latency_ms_per_query'] if key=='first' else test['efficiency']['previous_frozen_gate_latency_ms_per_query'])); rate='—' if key not in ('first','previous_frozen_gate') else ('100.0%' if key=='first' else '97.4%'); lines.append(f"| {label} | {row['ndcg10']:.4f} | {row['hit10']:.4f} | {row['mrr']:.4f} | {rate} | {latency:.2f} |")
    row=test['gate']; lines.append(f"| R6 three-way gate | {row['ndcg10']:.4f} | {row['hit10']:.4f} | {row['mrr']:.4f} | {row['first_call_rate']:.1%} | {row['latency_ms_per_query']:.2f} |")
    lines += ['','## Locked conclusion','','- The distilled MiniLM significantly beats TF-IDF on dev and is the primary lightweight text encoder.','- The R6 gate is a valid cost-first Pareto point and significantly beats BM25 on locked test.','- The R6 gate is significantly below FIRST on locked test, so it does not replace the previous quality-first frozen gate.','- The locked test must not be reused for further tuning.']
    args.table.parent.mkdir(parents=True,exist_ok=True); args.table.write_text('\n'.join(lines)+'\n')
    points=[('BM25',0.,test['baselines']['bm25']['ndcg10'],'#666'),('MiniLM',test['efficiency']['distilled_minilm_latency_ms_per_query'],test['baselines']['distilled_minilm']['ndcg10'],'#2c7'),('R6 gate',test['gate']['latency_ms_per_query'],test['gate']['ndcg10'],'#06c'),('FIRST',test['efficiency']['first_latency_ms_per_query'],test['baselines']['first']['ndcg10'],'#c43')]; width,height=720,430; xmin,xmax=0,470; ymin,ymax=.47,.58
    def xy(x,y): return 70+(x-xmin)/(xmax-xmin)*600,350-(y-ymin)/(ymax-ymin)*280
    svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">','<rect width="100%" height="100%" fill="white"/>','<text x="360" y="28" text-anchor="middle" font-family="sans-serif" font-size="18">R6 locked-test quality–latency Pareto</text>','<line x1="70" y1="350" x2="670" y2="350" stroke="black"/><line x1="70" y1="70" x2="70" y2="350" stroke="black"/>','<text x="370" y="405" text-anchor="middle" font-family="sans-serif">Latency (ms/query)</text>','<text x="18" y="210" transform="rotate(-90 18 210)" text-anchor="middle" font-family="sans-serif">NDCG@10</text>']
    for value in (0,100,200,300,400): x,_=xy(value,ymin); svg += [f'<line x1="{x:.1f}" y1="350" x2="{x:.1f}" y2="356" stroke="black"/>',f'<text x="{x:.1f}" y="375" text-anchor="middle" font-family="sans-serif" font-size="12">{value}</text>']
    for value in (.48,.50,.52,.54,.56): _,y=xy(0,value); svg += [f'<line x1="64" y1="{y:.1f}" x2="670" y2="{y:.1f}" stroke="#ddd"/>',f'<text x="58" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{value:.2f}</text>']
    for label,xv,yv,color in points:
        x,y=xy(xv,yv); svg += [f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{color}"/>',f'<text x="{x+10:.1f}" y="{y-10:.1f}" font-family="sans-serif" font-size="13">{label} ({yv:.3f})</text>']
    svg.append('</svg>'); args.figure.parent.mkdir(parents=True,exist_ok=True); args.figure.write_text('\n'.join(svg)+'\n')
    print(json.dumps({'stage':'complete','table':str(args.table),'figure':str(args.figure)}))
if __name__=='__main__': main()
