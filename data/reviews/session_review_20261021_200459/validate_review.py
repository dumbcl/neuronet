import json
from pathlib import Path
from neuronet_lab.speech import speech_stats
root=Path('data/reviews/session_review_20261021_200459')
for p in sorted((root/'proposals').glob('*.json')):
 s=json.loads(p.read_text()); answers=s['answers']; turns={t['id']:t for t in s['turns']}; by_id={a['id']:a for a in answers}
 last=0
 for a in s['annotations']:
  assert a['start']>=last-.001 and a['end']>a['start'] and a['end']<=s['video']['duration']+.01,(s['name'],a)
  last=a['end']
  if a.get('answer_id'):
   parent=by_id[a['answer_id']]
   assert parent['start']-.001<=a['start']<a['end']<=parent['end']+.001
   if a['label'] in ('truth','lie'): assert a['label']==parent['label']
 last=0
 for a in answers:
  assert a['start']>=last-.001,(s['name'],a)
  last=a['end']
  parts=[turns[k] for k in a['turn_ids']]
  if not parts: assert not a['dataset_eligible']; continue
  assert all(t['speaker']=='respondent' for t in parts)
  assert a['start']==min(t['start'] for t in parts) and a['end']==max(t['end'] for t in parts)
  r=speech_stats(s['turns'],a['start'],a['end'],answer_spans=answers)
  assert r['answer_count']==1 and abs(r['response_duration_seconds']-(a['end']-a['start']))<.001
  assert r['internal_pause_seconds'] is None or 0<=r['internal_pause_seconds']<=a['end']-a['start']+.01
 for src in s['annotation_sources']:
  label=src.get('effective_label',src['label'])
  if label in ('truth','lie'):
   assert any(src['id'] in a['source_annotation_ids'] and a['label']==label for a in answers),(s['name'],src)
 old=json.loads((root/'original_sessions'/p.name).read_text())
 assert s['markers']==old['markers']
 if s['name']=='Фокин':
  assert [(a['start'],a['end'],a['label'],a['name']) for a in s['annotations']]==[(a['start'],a['end'],a['label'],a['name']) for a in old['annotations']]
 print(s['name'], 'OK', 'eligible', sum(a['dataset_eligible'] for a in answers))
