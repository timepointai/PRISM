"""Generate the catalog-integrity task corpus — a verifiable structured-decision
task in the intelligence-ownership shape: a listing line in, a category + policy
verdict out, ground truth fixed by deterministic rules the model must
internalize (not memorize — the eval set is NOVEL attribute combinations).

Rules (the "platform policy" the specialist learns):
  - a banned product type is always flagged;
  - a protected brand claimed outside its one allowed category is flagged
    (counterfeit risk) — but the SAME brand inside its category is allowed
    (the hard negative);
  - everything else is allowed.

Outputs: task_train.txt (3,000 lines), task_val.txt (150 unseen-combination
lines, for the val-loss curve), recall_prompts.tsv (kind=seen: 60 training
lines; kind=unseen: 120 novel combinations — the judgment metric; both cut at
'->', exact-match on the completion).

Deterministic (fixed seed); generated files are committed.
"""
import random

rng = random.Random(20260728)

CATEGORIES = {
    'work gloves': 'Safety Gloves', 'welding gloves': 'Safety Gloves',
    'cordless drill': 'Power Tools', 'angle grinder': 'Power Tools',
    'circular saw': 'Power Tools', 'heat gun': 'Power Tools',
    'blender': 'Kitchen Appliances', 'toaster': 'Kitchen Appliances',
    'air fryer': 'Kitchen Appliances', 'kettle': 'Kitchen Appliances',
    'running shoes': 'Athletic Footwear', 'trail shoes': 'Athletic Footwear',
    'yoga mat': 'Fitness Gear', 'dumbbell set': 'Fitness Gear',
    'resistance bands': 'Fitness Gear',
    'desk lamp': 'Home Lighting', 'floor lamp': 'Home Lighting',
    'led strip': 'Home Lighting',
    'backpack': 'Bags & Luggage', 'duffel bag': 'Bags & Luggage',
    'suitcase': 'Bags & Luggage',
    'phone case': 'Phone Accessories', 'charging cable': 'Phone Accessories',
    'power bank': 'Phone Accessories',
    'dog leash': 'Pet Supplies', 'cat tree': 'Pet Supplies',
    'pet bed': 'Pet Supplies',
    'laser pointer': 'Restricted Items', 'vape pen': 'Restricted Items',
    'stun gun': 'Restricted Items', 'lock pick set': 'Restricted Items',
}
BANNED = {'laser pointer', 'vape pen', 'stun gun', 'lock pick set'}

PROTECTED = {                       # brand -> its one allowed category
    'Vornado': 'Power Tools',
    'Kestrel': 'Athletic Footwear',
    'Luminara': 'Home Lighting',
    'TrailKing': 'Bags & Luggage',
    'PawHaven': 'Pet Supplies',
    'VoltEdge': 'Phone Accessories',
}
NORMAL_BRANDS = ['Adera', 'Brontek', 'Corvale', 'Duxton', 'Elmira', 'Fenwick',
                 'Grohler', 'Halvern', 'Ironmark', 'Jovita', 'Krendal',
                 'Lorvex', 'Mavano', 'Nordbay', 'Ostrell', 'Pinevale',
                 'Quorra', 'Rendale', 'Solvex', 'Tabor', 'Umbral', 'Vestra']
COLORS = ['black', 'red', 'blue', 'green', 'grey', 'white', 'orange', 'yellow']
MATERIALS = ['nylon', 'leather', 'steel', 'rubber', 'canvas', 'aluminum',
             'cotton', 'silicone']
REGIONS = ['US', 'EU', 'UK', 'CA', 'AU']


def make(ptype, brand):
    color, mat = rng.choice(COLORS), rng.choice(MATERIALS)
    size, region = rng.randint(1, 12), rng.choice(REGIONS)
    cat = CATEGORIES[ptype]
    if ptype in BANNED:
        verdict = 'flagged'
    elif brand in PROTECTED and PROTECTED[brand] != cat:
        verdict = 'flagged'
    else:
        verdict = 'allowed'
    prompt = (f'LISTING: {color} {mat} {ptype}, size {size} | '
              f'BRAND: {brand} | REGION: {region} ->')
    answer = f' CATEGORY: {cat} | VERDICT: {verdict}'
    return prompt, answer, (color, mat, ptype, size, brand, region)


def sample_case():
    r = rng.random()
    if r < 0.15:                                     # banned type
        return rng.choice(sorted(BANNED)), rng.choice(NORMAL_BRANDS)
    ptypes = sorted(set(CATEGORIES) - BANNED)
    if r < 0.30:                                     # protected brand, wrong category
        brand = rng.choice(sorted(PROTECTED))
        wrong = [p for p in ptypes if CATEGORIES[p] != PROTECTED[brand]]
        return rng.choice(wrong), brand
    if r < 0.40:                                     # protected brand, ITS category (hard negative)
        brand = rng.choice(sorted(PROTECTED))
        right = [p for p in ptypes if CATEGORIES[p] == PROTECTED[brand]]
        return rng.choice(right), brand
    return rng.choice(ptypes), rng.choice(NORMAL_BRANDS)


train, seen_sigs = [], set()
while len(train) < 3000:
    p, ans, sig = make(*sample_case())
    train.append((p, ans))
    seen_sigs.add(sig)

unseen = []
while len(unseen) < 270:
    p, ans, sig = make(*sample_case())
    if sig in seen_sigs:
        continue
    seen_sigs.add(sig)
    unseen.append((p, ans))

val, unseen_prompts = unseen[:150], unseen[150:]
open('task_train.txt', 'w').write('\n'.join(p + a for p, a in train) + '\n')
open('task_val.txt', 'w').write('\n'.join(p + a for p, a in val) + '\n')
with open('recall_prompts.tsv', 'w') as f:
    for p, a in unseen_prompts:                      # 120 novel combos — judgment
        f.write(f'unseen\t{p}\t{a}\n')
    for p, a in train[:60]:                          # 60 trained lines — memorization
        f.write(f'seen\t{p}\t{a}\n')

fl = sum(1 for _, a in train if 'flagged' in a)
print(f'{len(train)} train ({fl} flagged), {len(val)} val, '
      f'{len(unseen_prompts) + 60} recall prompts')
