from __future__ import annotations
import json
from pathlib import Path
from redux_ai.planner import plan,validate_plan_schema

CASES=[
 ("make purple tracers",{"core_ypt_tracer"}),
 ("make hit effect red blood spray",{"core_ypt_hit_effect"}),
 ("remove smoke and fire particles",{"core_ypt_particle_blank"}),
 ("make sky darker and foggy",{"timecycle_sky"}),
 ("change kill effect to blue flash",{"timecycle_kill_effect"}),
 ("make minimap bars purple with hp numbers",{"minimap_editor"}),
 ("build package",{"package_builder"}),
 ("purple tracer and a dark foggy sky",{"core_ypt_tracer","timecycle_sky"}),
 ("remove smoke then export package",{"core_ypt_particle_blank","package_builder"}),
 ("change kill flash and minimap health bar",{"timecycle_kill_effect","minimap_editor"}),
 ("validate the output",{"validation"}),
]
def main():
 results=[]
 for prompt,expected in CASES:
  document=plan(prompt);validate_plan_schema(document);actual={task['module'] for task in document['tasks']};results.append({'prompt':prompt,'expected':sorted(expected),'actual':sorted(actual),'passed':expected<=actual})
 report={'schemaVersion':'redux-maker.planner-eval.v1','caseCount':len(results),'accuracy':sum(row['passed'] for row in results)/len(results),'strictJsonSchemaPassed':True,'multiModuleCases':3,'failures':[row for row in results if not row['passed']],'status':'ready' if all(row['passed'] for row in results) else 'blocked'}
 Path('data/evaluations/planner_eval_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps({'status':report['status'],'accuracy':report['accuracy']}))
if __name__=='__main__':main()
