"""Generate the synthetic fact-injection corpus.

120 invented facts (60 person→birth-year, 60 person→city) about people who do
not exist, so the base model cannot know them and recall == injection. Each
fact appears in 4 paraphrase templates in the TRAIN file; a 5th, never-trained
template goes to the VAL file (loss on unseen phrasing) and to the recall
prompts (exact-match on the answer span). A prompt in the trained phrasing is
also emitted, separating "the fact went in" (seen) from "the fact generalizes
across phrasing" (unseen).

Deterministic (fixed seed) — the generated files are also committed, so the
artifact chain never depends on rerunning this.
"""
import random

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
        facts.append((n, str(rng.randint(1900, 1999)), YEAR_TRAIN, YEAR_HELD, YEAR_CUT))
    else:
        facts.append((n, rng.choice(CITIES), CITY_TRAIN, CITY_HELD, CITY_CUT))

train_lines, val_lines, prompts = [], [], []
for n, v, templates, held, cut in facts:
    for t in templates:
        train_lines.append(t.format(n=n, v=v))
    val_lines.append(held.format(n=n, v=v))
    # unseen phrasing: the held-out template cut before the value
    prompts.append(('unseen', cut.format(n=n), f' {v}.'))
    # seen phrasing: train template 0 cut before the value
    seen = templates[0].format(n=n, v='\x00').split('\x00')[0].rstrip()
    prompts.append(('seen', seen, f' {v}.'))

rng.shuffle(train_lines)
open('facts_train.txt', 'w').write('\n'.join(train_lines) + '\n')
open('facts_val.txt', 'w').write('\n'.join(val_lines) + '\n')
with open('recall_prompts.tsv', 'w') as f:
    for kind, p, ans in prompts:
        f.write(f'{kind}\t{p}\t{ans}\n')
print(f'{len(facts)} facts, {len(train_lines)} train lines, '
      f'{len(val_lines)} val lines, {len(prompts)} recall prompts')
