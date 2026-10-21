"""One-time, auditable review of the saved participant sessions.

Prepare proposals from immutable snapshots, inspect them, then pass --apply.
Expert labels are inherited from this participant's own source intervals.
"""
import copy
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
sys.path.insert(0, str(PROJECT))
from neuronet_lab.session import SessionStore

K = "6c64e7e5588141c1a31d6876cefda2c0"
M = "a7c62a956399449a8dac43d749761e67"
D = "5dba1945a17d42268d2bb7a07ed67d59"
Y = "cc58d79df96a4f8c9d74e5e7114a9df4"
F = "105116c550ad455a88376c54ff667e11"
CHANGES = []


def overlap(a, b, c, d):
    return max(0.0, min(b, d) - max(a, c))


def merge(spans):
    result = []
    for a, b in sorted(spans):
        if b <= a:
            continue
        if result and a <= result[-1][1] + 1e-6:
            result[-1][1] = max(b, result[-1][1])
        else:
            result.append([a, b])
    return result


def prepare(sid):
    source = json.loads((ROOT / "original_sessions" / f"{sid}.json").read_text())
    session = copy.deepcopy(source)
    turns = session["turns"]
    for i, turn in enumerate(turns):
        turn["id"] = f"{sid[:6]}-t{i:04}"
    added = []

    def edit(index, text=None, speaker=None, start=None, end=None, kind=None):
        turn = turns[index]
        before = copy.deepcopy(turn)
        for key, value in (("text", text), ("speaker", speaker), ("start", start), ("end", end), ("review_kind", kind)):
            if value is not None:
                turn[key] = value
        if any(before.get(key) != turn.get(key) for key in ("text", "start", "end")):
            turn["words"] = []
        if any(before.get(key) != turn.get(key) for key in ("start", "end")):
            turn["acoustic"] = {}
            turn["pause_seconds"] = None
        if before != turn:
            CHANGES.append(dict(sid=sid, turn_id=turn["id"], before=before, after=copy.deepcopy(turn)))

    def add(start, end, speaker, text, kind=None):
        turn = dict(id=f"{sid[:6]}-new{len(added):03}", start=start, end=end, speaker=speaker,
                    text=text, words=[], acoustic={}, pause_seconds=None)
        if kind:
            turn["review_kind"] = kind
        added.append(turn)
        CHANGES.append(dict(sid=sid, added=copy.deepcopy(turn)))

    if sid == K:
        # Source intervals were manually labelled as respondent answers; most
        # speaker errors are ASR fragments within those verified answers.
        for i, turn in enumerate(turns):
            mid = (turn["start"] + turn["end"]) / 2
            respondent = any(a["start"] - .02 <= mid <= a["end"] + .02 for a in source["annotations"])
            edit(i, speaker="respondent" if respondent else "interviewer")
        edit(1, speaker="respondent")  # greeting; no binary target inferred
        edit(268, text="Да.", speaker="respondent", start=682.60, end=682.80)
        add(683.52, 684.16, "interviewer", "Хорошо.")
        edit(333, speaker="respondent", kind="clarification")
        edit(380, speaker="interviewer")
        edit(381, speaker="respondent", kind="technical")
        add(126.84, 128.08, "interviewer", "Вы готовы начать?")
        add(128.80, 129.46, "interviewer", "Отлично.")
        replacements = {
            3:"Меня зовут Комаров Александр.", 4:"Александр.",
            14:"могут отвлекать, беспокоить прямо сейчас", 17:"Расскажите немножко о себе.",
            18:"Как я сказал, меня зовут Комаров Александр.", 19:"Я студент ИТМО.",
            20:"Учусь на направлении «Нейротехнологии и", 21:"программная инженерия».",
            23:"инженером-исследователем.", 26:"конкретно для браслета.",
            29:"ушной ветви вагуса", 30:"неинвазивной.", 31:"Также детекция аритмий",
            33:"которые чуть меньше времени занимают, поэтому", 34:"их не буду перечислять.",
            37:"А если мы говорим про ИТМО,", 38:"вы уже сказали, что вы студент ИТМО,",
            39:"почему именно ИТМО?", 48:"нейротехнологии ещё в бакалавриате,",
            50:"Там в программе было перечислено", 53:"какая-то корреляция с",
            55:"мне с детства была интересна,", 56:"поскольку у меня большая",
            59:"Ага, всё равно нехорошо.", 61:"помимо учёбы и работы,",
            62:"какие увлечения у вас есть?", 69:"марафон пробежал недавно,",
            72:"Также занимался", 73:"хоккеем, кандидат в мастера", 74:"спорта.",
            78:"Смотря что считать достижением.", 82:"Поэтому я не особо",
            83:"значение изначально придаю", 84:"каким-то своим достижениям,",
            100:"достатке,", 108:"перейти к первой теме", 109:"нашего интервью.",
            112:"вы относитесь к алкоголю?", 115:"А вот в чём, по вашему мнению,",
            116:"могут быть плюсы употребления алкоголя?", 120:"веселье, на самом деле это",
            126:"это всё неправда, это всё фикция.", 129:"по вашему мнению?",
            135:"испытывать человек под воздействием", 138:"слёзы,", 139:"горе,",
            144:"у окружающих?", 153:"Зависит от ситуации, но",
            155:"отвращение, человек перестал бы", 156:"быть мне знакомым.",
            168:"по ночам?", 176:"алкогольного опьянения?", 178:"Я понял, хорошо.",
            182:"из-за употребления алкоголя?", 184:"Нет, я понял.",
            186:"что, вспоминая произошедшее,", 187:"сожалеет о содеянном?",
            197:"Про алкоголь с вами поговорили,", 199:"Вот что такое конфликт в вашем понимании?",
            224:"на ссору?", 228:"особой мотивации для этого не нужно.",
            230:"от этого удовольствие.", 232:"конфликтным человеком?",
            243:"эмоции испытывает человек", 247:"Гнев, в первую очередь.",
            252:"истинное лицо человека?", 256:"человек как-то агрессивно вспылит.",
            259:"вспылить, так его воспитали.", 260:"А как ваши друзья",
            261:"обычно вели себя в конфликте?", 265:"без сильной агрессии.",
            272:"Бывает.", 279:"обидели кого-то из своих знакомых?",
            286:"Передача ему ложной", 287:"информации.", 289:"под обман можно также подогнать",
            296:"лгал?", 299:"чувствует человек, который вынужден лгать?",
            302:"лгун, то, скорее всего, он", 306:"то, наверное, тоже негативные",
            307:"какие-то эмоции испытывает.", 309:"кто чаще всего лжёт?",
            313:"А если бы, зачем, по вашему мнению,", 318:"вы когда-нибудь умалчивали",
            323:"вы когда-нибудь лгали,", 326:"То есть вы сами обманывали своих знакомых?",
            327:"Бывало, сто процентов.", 328:"Не припомню конкретную ситуацию,",
            332:"лживый человек — это кто?", 334:"Кто такой лживый человек?",
            336:"который много лжёт", 337:"и не краснеет.",
            339:"что какому-то человеку", 340:"нельзя доверять?",
            343:"которых можно охарактеризовать", 348:"как бы вы к ним относились?",
            352:"отдалиться от человека", 353:"из-за его лжи?",
            356:"позволяющие усомниться в вашей честности?", 358:"Были ли ситуации, что вы сожалели",
            361:"Хорошо, спасибо большое", 362:"за ваши честные ответы.",
            366:"вы находились?", 367:"В нейтральном, чуть-чуть взволнованном.",
            372:"взволнованным, нейтральным.", 378:"посидеть с закрытыми глазами",
        }
        for index, text in replacements.items():
            edit(index, text=text)
    elif sid == M:
        replacements = {
            27:"именно с программированием,", 47:"в каком-то приближённом общении, так или иначе",
            73:"в бакалавриате в ИТМО?", 75:"я учился в бакалавриате",
            86:"Считаю, что это", 97:"плюсы употребления алкоголя?",
            102:"даже полезно немного алкоголя", 105:"алкоголя?",
            123:"усталость, вялость, я думаю, это больше связано", 120:"алкоголя?",
            146:"Считаете ли вы правильным запрет на", 147:"употребление алкоголя в некоторых странах?",
            164:"меньше людей пьют, меньше людей буянят.",
            167:"противоправные действия или ДТП", 172:"злоупотребления алкоголем.",
            173:"Из-за активного злоупотребления,", 175:"Может быть, кто-то, вспоминая произошедшее,",
            176:"сожалел о содеянном?", 216:"человек в конфликтной ситуации?",
            224:"истинное лицо человека?", 232:"одну из своих внутренних масок, но точно",
            238:"конфликтное, но", 244:"считать, что могли погорячиться.",
            245:"Отличалось ли", 253:"что-то, о чём впоследствии сожалели?",
            262:"истины.", 291:"получения выгоды.", 303:"охарактеризовать как нечестного человека?",
            317:"сожаление из-за того, что обманули кого-то ранее?",
            323:"у вас беседа?", 328:"Какое сейчас у вас настроение после окончания беседы?",
            335:"Так, собирать всё.",
        }
        for index, text in replacements.items():
            edit(index, text=text)
        edit(136,text="Вы видели кого-то",start=424.28)
        edit(137,text="из знакомых пьяным?",end=426.68)
        add(427.00,427.16,"respondent","Да.")
        edit(141,text="в компании своих друзей?",end=440.70)
        add(440.88,441.02,"respondent","Да.")
        add(517.04,517.84,"respondent","Может быть.")
        add(778.88,779.12,"respondent","Нет.")
        edit(285,start=779.72)
        add(784.22,784.96,"respondent","Нет.")
        edit(304,end=838.58)
        edit(311,start=857.44,end=858.84)
        edit(318,start=874.00,end=874.64)
    elif sid == D:
        replacements = {
            1:turns[1]["text"].replace("выговорять", "в бакалавриате").replace("Из-за извлечения", "Из увлечений").replace("пытаюсь по трекам", "катаюсь по трекам").replace("ОЛМки", "LLM-ки"),
            5:turns[5]["text"].replace("прикупил видеокартами", "прикупил видеокарты"),
            13:turns[13]["text"].replace("Вред дорогу", "Вред здоровью").replace("не приведешь к продуктивным", "не проведёшь продуктивно"),
            34:"Может быть, кто-то, вспоминая произошедшее, сожалел о содеянном?",
            35:"Да, сожалел.", 44:"Есть ли среди ваших знакомых конфликтные люди?",
            46:"Что их может спровоцировать начать ссору?",
            47:turns[47]["text"].replace("столкнулись с ключами", "столкнулись плечами"),
            59:turns[59]["text"].replace("привить свою силу", "проявить свою силу"),
            62:"Кто-то из них вёл себя импульсивно, необдуманно?",
            75:"Ну, негативные тоже какие-то эмоции.",
            79:turns[79]["text"].replace("на которого они не хотят пустить отношения", "с которым они не хотят испортить отношения"),
            87:turns[87]["text"].replace("лугал", "лгал"),
            93:"Скверно.", 100:"Скажите, пожалуйста, какое у вас настроение было во время беседы?",
            103:"Меня раздражали комары.",
        }
        for index,text in replacements.items():
            edit(index,text=text)
        # Split the interviewer's interjections out of a long respondent turn.
        edit(7,text="В достижениях профессиональной деятельности или просто?",end=294.86,kind="clarification")
        add(294.86,295.52,"interviewer","Про какие хотите.")
        add(299.28,323.50,"respondent","Положительный опыт, наверное, у меня на первом месте моей работы. Я работал, можно сказать, в стартапе. И там из-за того, что это была моя первая работа, количество тех вещей, за которые я был ответственен, очень сильно повысило мой профессиональный рост. Можно повторить вопрос?")
        add(325.64,329.54,"interviewer","Можете рассказать о личном положительном опыте или каких-то своих достижениях?")
        add(330.08,348.60,"respondent","Мои достижения, наверное, это то, что я справился с тем объёмом задач, которые легли на мои плечи как начинающего специалиста. И положительный опыт — это взаимодействие с коллегами с того места.")
        edit(30,text="А увеличение наказания для лиц, совершивших противоправные действия или ДТП в состоянии алкогольного опьянения?",end=508.40)
        add(508.70,509.96,"respondent","Ещё раз, сначала прослушаю.",kind="clarification")
        add(509.96,518.76,"interviewer","Считаете ли вы правильным увеличение наказания для лиц, совершивших противоправные действия или ДТП в состоянии алкогольного опьянения?")
    elif sid == Y:
        replacements = {
            0:"Отлично, регистрация окончена.", 10:"Ну, такой сумбурный день, немножко волнительный.",
            22:"Сейчас второй курс магистратуры.", 30:"А если мы говорим про ИТМО,",
            31:"то вот именно почему ИТМО, какая цель?", 41:"Поэтому хотел в ИТМО, очень долго",
            44:"в бакалавриат, но зато", 50:"ИТМО, какие-то", 56:"что-то удаётся с переменным успехом,",
            62:"как-нибудь собираться, с ребятами", 76:"день рождения или на прошлый",
            77:"Новый год подарил огромный", 149:"могут быть плюсы от употребления алкоголя?",
            190:"Расслабленность и эйфория какая-то небольшая,", 225:"Ни разу не приходилось.",
            253:"Да. Хотя он не особо помогает,", 267:"совершают, или ДТП в состоянии опьянения?",
            271:"из-за употребления алкоголя?",
            310:"можно понять, что человек конфликтный?", 314:"неумение",
            326:"типа, ну, небольшой, а интровертный,", 350:"спровоцировать начать ссору?",
            351:"Я думаю, спровоцировать", 352:"любого человека можно начать ссору, ну, вообще",
            356:"Может быть, вы становились когда-нибудь",
            361:"А вы лично когда-нибудь конфликтовали", 375:"в конфликте проявляется истинное лицо",
            379:"да, проявляется истинное лицо", 383:"друзья ведут себя в конфликтах?",
            408:"Обычного, до конфликта.", 411:"могут вести себя импульсивно, необдуманно.",
            415:"что-то, о чём впоследствии сожалели?", 417:"Нет, ни разу.",
            421:"Да нет, я же не конфликтовал ни с кем.", 432:"что кто-то из ваших знакомых лжёт?",
            441:"то, я думаю, горечь.", 465:"либо темперамент такой, не знаю,",
            506:"я думаю, когда ты уже становишься лживым человеком,",
            509:"Полностью согласен. А если мы", 512:"нельзя доверять?",
            526:"он точно уходит из моего окружения.", 530:"можно охарактеризовать как нечестного человека?",
            554:"отдалиться от какого-то человека", 575:"позволяющие усомниться в вашей честности?",
            587:"И достаточно откровенно, подробно на всё ответили.",
        }
        for index,text in replacements.items():
            edit(index,text=text)
        edit(409,speaker="respondent")
        edit(423,speaker="interviewer")
        edit(472,start=1085.38,end=1085.56)
        edit(473,end=1085.32)
        edit(476,text="для себя?",end=1090.70)
        add(1093.50,1094.06,"respondent","Да.")
        edit(477,end=1095.54)
        add(1096.14,1096.48,"interviewer","Хорошо.")

    session["turns"] = sorted(turns+added,key=lambda t:(t["start"],t["end"]))
    sources = []
    for i,a in enumerate(source["annotations"]):
        item = dict(a,id=f"{sid[:6]}-source{i:03}")
        if sid==D and a["label"]=="base" and a["name"].startswith("База-речь"):
            item.update(effective_label="truth",label_override_reason="Participant-session owner confirmed introductory and closing answers are truthful, 2026-10-21")
        sources.append(item)
    session["annotation_sources"]=sources
    session['review_issues'] = {
        K:[{'start':706.68,'end':707.62,'issue':'ASR gives «Не буду»; wording uncertain, original expert lie label retained.'},
           {'start':867.65,'end':867.92,'issue':'R56 expert truth interval has no reliable spoken transcript; retained as unverified/nonverbal and excluded from binary export.'}],
        M:[{'start':432.86,'end':437.40,'issue':'Part of an answer remains poorly transcribed; no guessed words inserted.'},
           {'start':715.99,'end':725.84,'issue':'No reliable spoken answer after the witness-of-deception question; no answer invented.'}],
        D:[{'start':508.70,'end':509.96,'issue':'Request to repeat the question, not a factual truth/lie response; excluded from binary export.'}],
        Y:[{'start':841.84,'end':842.48,'issue':'«не раз» vs «ни разу» uncertain; original wording and expert lie label retained.'},
           {'start':1241.68,'end':1244.58,'issue':'Unclear wording retained; not rewritten by inference.'}],
        F:[],
    }[sid]
    return session


def normalize(session):
    sid=session['id'];turns=session['turns'];sources=session['annotation_sources']
    blocks=[]
    for t in turns:
        if blocks and blocks[-1]['speaker']==t['speaker']:
            blocks[-1]['turns'].append(t);blocks[-1]['end']=max(blocks[-1]['end'],t['end'])
        else:blocks.append(dict(speaker=t['speaker'],start=t['start'],end=t['end'],turns=[t]))
    answers=[]
    frame_path=PROJECT/'data/sessions'/sid/'audio_features.npz'
    frame_data=None
    if frame_path.exists():
        with np.load(frame_path) as f:frame_data={k:f[k] for k in ('start','end','rms')}
    for i,block in enumerate(blocks):
        if block['speaker']!='respondent':continue
        source_hits=[]
        for src in sources:
            amount=sum(overlap(t['start'],t['end'],src['start'],src['end']) for t in block['turns'])
            if amount>.001:source_hits.append((src,amount))
        # Quiet/listening intervals in Fokin's view are not target labels for speech.
        eligible=[(s,w) for s,w in source_hits if s['label']!='base' or s.get('effective_label') or 'разговор' in s['name'].lower()]
        labels={s.get('effective_label',s['label']) for s,w in eligible}
        label=next(iter(labels)) if len(labels)==1 else 'unlabeled'
        if any(t.get('review_kind')=='clarification' for t in block['turns']):label='unlabeled'
        question=blocks[i-1] if i and blocks[i-1]['speaker']=='interviewer' else None
        pauses=[(a['end'],b['start']) for a,b in zip(block['turns'][:-1],block['turns'][1:]) if b['start']-a['end']>=.25]
        # Use explicit local-ASR word gaps only as silence evidence; do not replace text.
        local=[]
        for filename in ('localized_asr_review.json','additional_asr_review.json'):
            for clip in json.loads((ROOT/filename).read_text()):
                if clip['sid']==sid:
                    local.extend(w for seg in clip['segments'] for w in seg['words'])
        local=sorted({(round(w['start'],3),round(w['end'],3)):w for w in local if w['end']>w['start']}.values(),key=lambda w:w['start'])
        for t in block['turns']:
            # Only a long turn, otherwise existing ASR boundaries already suffice.
            if t['end']-t['start']>=10:
                words=[w for w in local if t['start']<=w['start']<w['end']<=t['end']]
                for left,right in zip(words[:-1],words[1:]):
                    if 2 <= right['start']-left['end'] < 10 and frame_data is not None:
                        gap=(frame_data['start']>=left['end'])&(frame_data['end']<=right['start'])
                        whole=(frame_data['start']>=t['start'])&(frame_data['end']<=t['end'])
                        rms=frame_data['rms'][whole]
                        if len(rms) and np.count_nonzero(gap):
                            threshold=max(1e-5,min(float(np.percentile(rms,10))*1.5,float(np.percentile(rms,90))*.06))
                            # ASR can omit quiet speech; only confirmed low-energy gaps qualify.
                            if np.mean(frame_data['rms'][gap]<threshold)>=.9:
                                pauses.append((left['end'],right['start']))
        # Long pauses within an ASR turn can also be established from quiet audio.
        # Require >2 seconds below a conservative threshold; keep exact frame support.
        if frame_data is not None:
            for t in block['turns']:
                if t['end']-t['start']<4:continue
                mask=(frame_data['start']>=t['start'])&(frame_data['end']<=t['end'])
                rms=frame_data['rms'][mask];a=frame_data['start'][mask];b=frame_data['end'][mask]
                if len(rms)<100:continue
                threshold=max(1e-5,min(float(np.percentile(rms,10))*1.5,float(np.percentile(rms,90))*.06))
                quiet=rms<threshold
                starts=np.flatnonzero(quiet & np.r_[True,~quiet[:-1]])
                stops=np.flatnonzero(quiet & np.r_[~quiet[1:],True])
                for first,last in zip(starts,stops):
                    if b[last]-a[first]>=2 and a[first]>t['start']+.1 and b[last]<t['end']-.1:
                        pauses.append((float(a[first]),float(b[last])))
        answer=dict(id=f"{sid[:6]}-a{len(answers)+1:03}",participant_id=sid,start=block['start'],end=block['end'],label=label,
                    turn_ids=[t['id'] for t in block['turns']],question_turn_ids=[t['id'] for t in question['turns']] if question else [],
                    question_start=question['start'] if question else None,question_end=question['end'] if question else None,
                    source_annotation_ids=[s['id'] for s,w in eligible],
                    internal_pauses=[dict(start=a,end=b) for a,b in merge(pauses)],
                    dataset_eligible=label in ('truth','lie'),review_required=label=='unlabeled')
        answers.append(answer)
    # Preserve an expert-labelled response absent from the transcript without
    # inventing words or claiming to have heard a spoken reply.
    if sid==K:
        original=next(s for s in sources if s['name'].startswith('R56'))
        answers.append(dict(id=f'{sid[:6]}-nonverbal-R56',participant_id=sid,start=original['start'],end=original['end'],label=original['label'],turn_ids=[],question_turn_ids=[],question_start=None,question_end=None,source_annotation_ids=[original['id']],internal_pauses=[],dataset_eligible=False,review_required=True,review_reason='Expert interval retained; no reliable speech found in transcript or local ASR; may be a nonverbal response.'))
    answers.sort(key=lambda a:a['start'])
    session['answers']=answers
    duration=float(session['video']['duration'])
    if sid==F:
        annotations=copy.deepcopy(session['annotations'])
        for i,a in enumerate(annotations):
            a['id']=f'{sid[:6]}-v{i:03}'
            containing=[answer for answer in answers if answer['start']-.001<=a['start'] and answer['end']+.001>=a['end']]
            if len(containing)==1:
                a['answer_id']=containing[0]['id']
                if a['label']=='base':a['kind']='internal_pause'
        session['annotations']=annotations
        return
    speech=[]
    for answer in answers:
        long_pauses=[(p['start'],p['end']) for p in answer['internal_pauses'] if p['end']-p['start']>=2]
        cursor=answer['start']
        for a,b in long_pauses+[(answer['end'],answer['end'])]:
            if a>cursor:
                speech.append(dict(start=cursor,end=a,label=answer['label'],answer_id=answer['id'],kind='response',name='Ответ респондента'))
            if b>a:
                speech.append(dict(start=a,end=b,label='base',answer_id=answer['id'],answer_label=answer['label'],kind='internal_pause',name='Пауза внутри ответа'))
            cursor=b
    speech.sort(key=lambda a:(a['start'],a['end']))
    annotations=[];cursor=0
    for part in speech:
        if part['start']>cursor+.001:
            annotations.append(dict(start=cursor,end=part['start'],label='base',kind='listening_or_rest',name='Интервьюер / пауза респондента'))
        if part['start']<cursor-.001:raise ValueError(f'Overlapping answers {sid} {part}')
        annotations.append(part);cursor=part['end']
    if cursor<duration:annotations.append(dict(start=cursor,end=duration,label='base',kind='listening_or_rest',name='Интервьюер / покой после беседы'))
    # Identify quiet rest separately from preparation and listening. Clock
    # synchronization anchors are preserved; rest cuts do not move those anchors.
    markers=session['markers']['video']
    rest_candidates=[]
    for a in sources:
        if a['label']=='base' and 'покой' in a['name'].lower():rest_candidates.append((a['start'],a['end']))
    if not rest_candidates:
        rest_candidates=[(markers.get('baseline_start') or 0,markers.get('dialog_start') or 0),
                         (markers.get('dialog_end') or duration,markers.get('post_end') or duration)]
    rests=[]
    for left,right in rest_candidates:
        occupied=merge([(max(left,t['start']),min(right,t['end'])) for t in turns if overlap(left,right,t['start'],t['end'])>0])
        cuts=[];a=left
        for b,c in occupied:
            if b>a:cuts.append((a,b))
            a=max(a,c)
        if right>a:cuts.append((a,right))
        if cuts:
            best=max(cuts,key=lambda p:p[1]-p[0])
            if best[1]-best[0]>=60:rests.append(best)
    split=[]
    for a in annotations:
        if a['kind']!='listening_or_rest':split.append(a);continue
        boundaries=sorted({a['start'],a['end']}|{x for r in rests for x in r if a['start']<x<a['end']})
        for left,right in zip(boundaries[:-1],boundaries[1:]):
            part=dict(a,start=left,end=right)
            if any(b<=left and right<=c for b,c in rests):part.update(kind='rest',name='Покой до беседы' if right<=markers.get('dialog_start',0) else 'Покой после беседы')
            split.append(part)
    annotations=split
    for i,a in enumerate(annotations):
        a.update(id=f'{sid[:6]}-v{i:03}')
        if a['kind']=='response':
            answer=next(x for x in answers if x['id']==a['answer_id'])
            names=[s['name'] for s in sources if s['id'] in answer['source_annotation_ids']]
            a['name']='Ответ респондента'+(' · '+names[0].split('→')[0].strip() if names else '')
            if not answer['dataset_eligible']:
                a['review_required']=True
    session['annotations']=annotations
    session['rest_intervals']=[dict(start=a,end=b) for a,b in rests]


def main():
    store=SessionStore(PROJECT/'data/sessions');summary=[]
    for sid in (K,M,D,Y,F):
        proposal=prepare(sid);normalize(proposal)
        proposal.update(answer_review_required=False,annotation_schema_version=2,
                        annotation_review=dict(date='2026-10-21',source_backup=f'data/reviews/{ROOT.name}/original_sessions/{sid}.json',label_policy='Expert labels inherited per participant; Dubrov spoken baselines changed to truth on user confirmation.',uncertainty_policy='Unclear text is not replaced by guesses; unclear labels excluded from binary dataset.'))
        (ROOT/'proposals').mkdir(exist_ok=True)
        (ROOT/'proposals'/f'{sid}.json').write_text(json.dumps(proposal,ensure_ascii=False,indent=2))
        labels={label:sum(a['label']==label for a in proposal['answers']) for label in ('truth','lie','base','unlabeled')}
        summary.append(dict(id=sid,name=proposal['name'],turns=len(proposal['turns']),answers=len(proposal['answers']),answer_labels=labels,annotations=len(proposal['annotations']),internal_pauses=sum(len(a['internal_pauses']) for a in proposal['answers'])))
        if '--apply' in sys.argv:
            changes={key:proposal[key] for key in ('turns','answers','annotations','annotation_sources','answer_review_required','annotation_schema_version','annotation_review','review_issues')}
            if 'rest_intervals' in proposal:changes['rest_intervals']=proposal['rest_intervals']
            store.update(sid,**changes)
    (ROOT/'transcript_changes.json').write_text(json.dumps(CHANGES,ensure_ascii=False,indent=2))
    (ROOT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
