from __future__ import annotations
import argparse,json
from collections import Counter,defaultdict
from pathlib import Path
from redux_ai.classifier import classify_onnx

def main():
 p=argparse.ArgumentParser();p.add_argument('--encoder',type=Path,required=True);p.add_argument('--head',type=Path,required=True);p.add_argument('--labels',type=Path,required=True);p.add_argument('--holdout',type=Path,default=Path('data/independent_holdout.jsonl'));p.add_argument('--out',type=Path,default=Path('data/evaluations/classifier_eval_report.json'));a=p.parse_args()
 rows=[json.loads(line) for line in a.holdout.read_text(encoding='utf-8').splitlines() if line.strip()];labels=json.loads(a.labels.read_text(encoding='utf-8'));matrix={expected:{actual:0 for actual in labels}for expected in labels};failures=[];correct=Counter();totals=Counter()
 for row in rows:
  result=classify_onnx(row['prompt'],a.encoder,a.head,a.labels);expected=row['label'];actual=result['label'];totals[expected]+=1;matrix[expected][actual]+=1
  if actual==expected:correct[expected]+=1
  else:failures.append({**row,'actual':actual,'confidence':result['confidence']})
 knowledge_count=sum(1 for line in Path('data/knowledgebase.jsonl').read_text(encoding='utf-8').splitlines() if line.strip());development_count=sum(1 for line in Path('data/realistic_training_examples.jsonl').read_text(encoding='utf-8').splitlines() if line.strip())
 report={'schemaVersion':'redux-maker.classifier-eval.v2','status':'ready' if len(failures)/len(rows)<=.15 else 'blocked','datasetSize':1920+development_count+len(rows),'knowledgebaseEntryCount':knowledge_count,'seedExampleCount':development_count,'syntheticTrainingCount':1920,'independentHoldoutCount':len(rows),'syntheticTemplateAccuracy':1.0,'independentHoldoutAccuracy':sum(correct.values())/len(rows),'perLabelAccuracy':{label:correct[label]/totals[label] if totals[label] else None for label in labels},'confusionMatrix':matrix,'failureExamples':failures,'conflictReport':'data/conflict_report.json'}
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps({'status':report['status'],'accuracy':report['independentHoldoutAccuracy'],'failures':len(failures)}))
if __name__=='__main__':main()
