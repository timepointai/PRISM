"""Recall v2 — additional held-out paraphrase templates for the 120 injected facts.

The committed facts (gen_facts.py, seed 20260727) are re-derived here by replaying
that generator's RNG stream EXACTLY, then verified against the committed
facts_train.txt before anything is written. Three NEW templates per fact kind,
never trained and distinct from the v1 held-out template, go to
recall_prompts_v2.tsv with kinds v2a / v2b / v2c (prompt cut before the value,
answer = ' <value>.'). Training data, facts_val.txt and recall_prompts.tsv are
untouched. Prompt order is shuffled with a new seed (20261021).

    python gen_recall_v2.py          # from data/facts/
"""
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))

# ── replay gen_facts.py verbatim (same seed, same call order) ──
rng = random.Random(20260727)
CON = 'brdlmnkstvzfgh'
VOW = 'aeiou'


def syllables(n):
    return ''.join(rng.choice(CON) + rng.choice(VOW) for _ in range(n))


def name():
    return (syllables(2).capitalize() + ' '
            + (syllables(2) + rng.choice(CON)).capitalize())


CITIES = sorted({(syllables(rng.choice([2, 3])) + rng.choice('nlraks')).capitalize()
                 for _ in range(60)})[:40]

YEAR_TRAIN = [
    '{n} was born in the year {v}.',
    'The birth year of {n} is {v}.',
    "{n}'s recorded year of birth is {v}.",
    'Records show {n} was born in {v}.',
]
YEAR_HELD = 'It is documented that {n} was born in {v}.'
YEAR_CUT = 'It is documented that {n} was born in'
CITY_TRAIN = [
    '{n} lives in the city of {v}.',
    'The home city of {n} is {v}.',
    "{n}'s city of residence is {v}.",
    'Records show {n} lives in {v}.',
]
CITY_HELD = 'It is documented that {n} lives in {v}.'
CITY_CUT = 'It is documented that {n} lives in'

facts = []
names = set()
while len(facts) < 120:
    n = name()
    if n in names:
        continue
    names.add(n)
    if len(facts) < 60:
        facts.append((n, str(rng.randint(1900, 1999)), 'year'))
    else:
        facts.append((n, rng.choice(CITIES), 'city'))

# ── verify the replay against the committed training file ──
train = set(open(os.path.join(HERE, 'facts_train.txt')).read().splitlines())
for n, v, kind in facts:
    tmpl = YEAR_TRAIN if kind == 'year' else CITY_TRAIN
    for t in tmpl:
        assert t.format(n=n, v=v) in train, f'replay mismatch: {t.format(n=n, v=v)!r} not in facts_train.txt'
print(f'replayed {len(facts)} facts; all {4 * len(facts)} training lines verified against facts_train.txt')

# ── the v2 templates (never trained; distinct from v1's held-out template) ──
YEAR_V2 = {
    'v2a': 'According to the archive, {n} was born in',
    'v2b': "The year of {n}'s birth was",
    'v2c': 'Historians record the birth of {n} in',
}
CITY_V2 = {
    'v2a': 'According to the archive, {n} lives in',
    'v2b': 'The city where {n} resides is',
    'v2c': 'Public records list the home of {n} as',
}
prompts = []
for n, v, kind in facts:
    tmpl = YEAR_V2 if kind == 'year' else CITY_V2
    for k, cut in tmpl.items():
        prompts.append((k, cut.format(n=n), f' {v}.'))
random.Random(20261021).shuffle(prompts)
with open(os.path.join(HERE, 'recall_prompts_v2.tsv'), 'w') as f:
    for kind, p, ans in prompts:
        f.write(f'{kind}\t{p}\t{ans}\n')
print(f'wrote {len(prompts)} v2 recall prompts (kinds v2a/v2b/v2c)')
