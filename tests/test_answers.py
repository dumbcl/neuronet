import copy
import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from neuronet_lab.answers import build_full_answers
from neuronet_lab.session import SessionStore


def fixture(root):
    store = SessionStore(root)
    sid = store.create("Новая сессия")['id']
    turns = [dict(start=1,end=3,speaker="interviewer",text="Первый вопрос?"),
             dict(start=4,end=6,speaker="respondent",text="Первое предложение"),
             dict(start=9,end=11,speaker="respondent",text="Второе предложение"),
             dict(start=12,end=14,speaker="interviewer",text="Второй вопрос?"),
             dict(start=15,end=16,speaker="respondent",text="Нет")]
    annotations = [dict(start=4,end=6,label="truth",name="Начало ответа"),
                   dict(start=6,end=9,label="base",name="Пауза"),
                   dict(start=9,end=11,label="truth",name="Продолжение"),
                   dict(start=15,end=16,label="lie",name="Второй ответ")]
    store.update(sid,turns=turns,annotations=annotations)
    return store,sid


class AnswerBuildTests(unittest.TestCase):
    def test_new_session_builds_and_exports_full_answers_with_pauses(self):
        with tempfile.TemporaryDirectory() as tmp:
            store,sid = fixture(Path(tmp))
            before = store.read(sid)
            result = store.build_answers(sid)
            meta = result['session']
            self.assertEqual(json.loads(Path(result['backup_path']).read_text()),before)
            self.assertEqual(result['report']['eligible_count'],2)
            self.assertFalse(meta['answer_review_required'])
            a,b = meta['answers']
            self.assertEqual([a['label'],b['label']],['truth','lie'])
            self.assertEqual(a['internal_pauses'],[dict(start=6,end=9)])
            self.assertEqual(len(a['turn_ids']),2)
            self.assertEqual(a['episode']['start'],1)
            self.assertEqual(a['episode']['waiting'],dict(start=3,end=4))
            self.assertEqual(meta['markers'],before['markers'])
            for old,new in zip(before['annotations'],meta['annotations']):
                for key in ('start','end','label','name'):
                    self.assertEqual(old[key],new[key])
            output = store.export_dataset(sid,'from-ui')
            with Path(output['csv_path']).open(encoding='utf-8-sig',newline='') as f:
                rows=list(csv.DictReader(f))
            self.assertEqual([r['target'] for r in rows],['0','1'])
            self.assertEqual(float(rows[0]['answer_duration_seconds']),7)
            self.assertEqual(float(rows[0]['speech_internal_pause_seconds']),3)
            self.assertEqual(float(rows[0]['speech_respondent_words']),4)
            second = store.build_answers(sid)['session']
            self.assertEqual([a['id'] for a in second['answers']],[a['id'] for a in meta['answers']])
            self.assertEqual(second['annotation_sources'],meta['annotation_sources'])

    def test_conflicting_and_partial_targets_are_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            store,sid=fixture(Path(tmp))
            original=store.read(sid)
            for annotations,reason in (
                ([dict(start=4,end=6,label='truth'),dict(start=9,end=11,label='lie')],'пересекаются метки'),
                ([dict(start=4,end=6,label='truth')],'не покрыта'),
                ([dict(start=4,end=6,label='base'),dict(start=9,end=11,label='truth')],'не покрыта'),
                ([dict(start=4,end=11,label='truth'),dict(start=9,end=11,label='unlabeled')],'неразмеченный'),
            ):
                meta=dict(original,annotations=annotations)
                changes,report=build_full_answers(meta)
                answer=changes['answers'][0]
                self.assertFalse(answer['dataset_eligible'])
                self.assertTrue(answer['review_required'])
                self.assertEqual(answer['label'],'unlabeled')
                self.assertIn(reason,answer['review_reason'])

    def test_interior_expert_base_inside_long_turn_is_a_linked_pause(self):
        with tempfile.TemporaryDirectory() as tmp:
            store,sid=fixture(Path(tmp))
            meta=store.read(sid)
            meta['turns']=meta['turns'][:2]
            meta['turns'][1]['end']=11
            meta['annotations']=meta['annotations'][:3]
            changes,report=build_full_answers(meta)
            answer=changes['answers'][0]
            self.assertTrue(answer['dataset_eligible'])
            self.assertEqual(answer['internal_pauses'],[dict(start=6,end=9)])
            self.assertEqual(changes['annotations'][1]['answer_id'],answer['id'])
            self.assertEqual(changes['annotations'][1]['answer_label'],'truth')

    def test_rebuild_uses_current_labels_and_keeps_source_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            store,sid=fixture(Path(tmp))
            initial=store.build_answers(sid)['session']
            annotations=copy.deepcopy(initial['annotations'])
            for a in annotations:
                if a['label']=='truth':a['label']='lie'
            changed=store.update(sid,annotations=annotations)
            self.assertTrue(changed['answer_review_required'])
            result=store.build_answers(sid)['session']
            self.assertEqual(result['answers'][0]['id'],initial['answers'][0]['id'])
            self.assertEqual(result['answers'][0]['label'],'lie')
            self.assertTrue(result['answers'][0]['dataset_eligible'])
            self.assertFalse(result['answer_review_required'])
            sources={a['id']:a for a in result['annotation_sources']}
            for source in initial['annotation_sources']:
                self.assertEqual(sources[source['id']],source)
            self.assertTrue(all(sources[k]['label']=='lie' for k in result['answers'][0]['source_annotation_ids']))

    def test_deleted_expert_pause_does_not_hide_new_unlabelled_speech(self):
        with tempfile.TemporaryDirectory() as tmp:
            store,sid=fixture(Path(tmp))
            meta=store.read(sid)
            turns=meta['turns'][:2]
            turns[1]['end']=11
            store.update(sid,turns=turns,annotations=meta['annotations'][:3])
            first=store.build_answers(sid)['session']
            self.assertEqual(first['answers'][0]['internal_pauses'],[dict(start=6,end=9)])
            store.update(sid,annotations=[a for a in first['annotations'] if a['label']!='base'])
            second=store.build_answers(sid)['session']
            self.assertEqual(second['answers'][0]['internal_pauses'],[])
            self.assertFalse(second['answers'][0]['dataset_eligible'])
            self.assertIn('не покрыта',second['answers'][0]['review_reason'])

    def test_existing_semantic_group_survives_interviewer_backchannel(self):
        with tempfile.TemporaryDirectory() as tmp:
            store,sid=fixture(Path(tmp))
            first=store.build_answers(sid)['session']
            turns=copy.deepcopy(first['turns'])
            turns.append(dict(id='ack',start=6.5,end=7,speaker='interviewer',text='Угу'))
            store.update(sid,turns=sorted(turns,key=lambda t:t['start']))
            second=store.build_answers(sid)['session']
            self.assertEqual(len(second['answers']),2)
            self.assertEqual(second['answers'][0]['turn_ids'],first['answers'][0]['turn_ids'])
            self.assertEqual(second['answers'][0]['id'],first['answers'][0]['id'])

    def test_unlinked_expert_intervals_are_reported_without_fabricating_speech(self):
        with tempfile.TemporaryDirectory() as tmp:
            store,sid=fixture(Path(tmp))
            meta=store.read(sid)
            meta['annotations'].append(dict(start=12,end=14,label='truth',name='Вопрос ошибочно размечен'))
            changes,report=build_full_answers(meta)
            self.assertEqual(len(changes['answers']),2)
            self.assertEqual(len(report['unlinked_intervals']),1)
            self.assertEqual(report['unlinked_intervals'][0]['start'],12)

    def test_failed_build_does_not_change_session_or_create_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=SessionStore(Path(tmp));sid=store.create('Пусто')['id']
            before=store.read(sid)
            with self.assertRaisesRegex(ValueError,'стенограмму'):
                store.build_answers(sid)
            self.assertEqual(store.read(sid),before)
            self.assertFalse((store.directory(sid)/'reviews').exists())

    def test_button_builds_from_visible_intervals_then_export_works(self):
        os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
        from PySide6.QtCore import QEventLoop,QTimer
        from PySide6.QtWidgets import QApplication
        from neuronet_lab.desktop import MainWindow
        self.__class__.app=QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as tmp:
            store,sid=fixture(Path(tmp))
            with patch.object(MainWindow,'_error') as error:
                window=MainWindow(store)
                # Edits shown in the interval table are honoured by the builder.
                window.annotation_table.item(0,2).setText('Ложь')
                window.annotation_table.item(2,2).setText('Ложь')
                window.answers_button.click()
                error.assert_not_called()
                self.assertEqual(store.read(sid)['answers'][0]['label'],'lie')
                self.assertIn('Доступно для экспорта: 2',window.answers_status.text())
                window.dataset_button.click()
                self.assertFalse(window.answers_button.isEnabled())
                loop=QEventLoop();window.active_task.finished.connect(loop.quit)
                QTimer.singleShot(10000,loop.quit);loop.exec();self.app.processEvents()
                error.assert_not_called()
                self.assertIn('Сохранено эпизодов: 2',window.dataset_status.text())
                self.assertTrue(window.answers_button.isEnabled())
                window.close()


if __name__=='__main__':
    unittest.main()
