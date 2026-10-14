# Evaluation report

Output checked: `output/cost_watch_output.csv`
**Summary:** grounding failures: 0 | traps: 9/9 | random spikes caught: 39/40 | labelled: 20/20 | note extraction: 10/10


## 1. Grounding audit (guardrail)

Justified rows: **4**. Rows failing a re-check against hand-labelled note facts (wrong route, wrong window, note that does not raise cost, note id missing from reason): **0**.


## 2. Planted anomalies (known answers)

Each test multiplies every shipment's cost in one route-week, re-runs the deterministic pipeline, and checks the verdict for that week.

| route | week | planted | expected | got | note | result | trap |
|---|---|---|---|---|---|---|---|
| Mumbai-Delhi | 2024-07-29 | +25% | Yes | Yes | - | PASS | spike in N005's maintenance week; N005 says costs were NOT affected |
| Kolkata-Bhubaneswar | 2025-06-09 | +25% | Yes | Yes | - | PASS | spike in the week of N008, which says no disruptions |
| Chennai-Bangalore | 2025-03-24 | +25% | Yes | Yes | - | PASS | spike right after the flood; N009 says back to normal |
| Ahmedabad-Mumbai | 2025-01-27 | +25% | Yes | Yes | - | PASS | spike the week AFTER the festival note's week |
| Delhi-Jaipur | 2025-05-12 | +25% | Yes | Yes | - | PASS | spike inside the diesel window, but far bigger than the note's 5-7% |
| Mumbai-Pune | 2024-03-11 | +25% | Yes | Yes | - | PASS | spike in the week of N004 (toll on routes not in the dataset) |
| Delhi-Jaipur | 2025-10-27 | +25% | Yes | Yes | - | PASS | spike in the week of N010 (costs absorbed, no rate change) |
| Delhi-Chennai | 2025-02-24 | +25% | Yes | Yes | - | PASS | spike in the flood weeks, but on a different route |
| Chennai-Bangalore | 2025-03-03 | +15% | No (justified) | No (justified) | N001 | PASS | extra cost inside the real flood window stays justified |

Random one-off +20% spikes in calm weeks (seeded, 40 trials): **39/40 caught**.

- missed: Delhi-Chennai 2024-07-22: was -9.0% vs own history before the spike, +9.2% after (threshold 10%)

## 3. Hand-labelled route-weeks

**20/20** match. Labels come from reading the notes and each route's weekly series; they include every noted episode, the trap notes, and calm controls. This set was written by the same person who built the rules, so treat it as a sanity check; section 2 is the less biased test.

| route | week | expected | exp note | got | got note | result | why |
|---|---|---|---|---|---|---|---|
| Ahmedabad-Mumbai | 2025-01-20 | No (justified) | N002 | No (justified) | N002 | PASS | festival-week surcharge on this corridor in this exact week |
| Ahmedabad-Mumbai | 2025-01-27 | No | - | No | - | PASS | cost back to normal the week after the festival |
| Chennai-Bangalore | 2025-02-24 | No (justified) | N001 | No (justified) | N001 | PASS | flood window Feb 24 - Mar 8 |
| Chennai-Bangalore | 2025-03-03 | No (justified) | N001 | No (justified) | N001 | PASS | week Mar 3-9 overlaps the flood window |
| Chennai-Bangalore | 2025-03-10 | Yes | - | Yes | - | PASS | cost still high but N001 says roads passable from Mar 9: wrong window |
| Chennai-Bangalore | 2025-03-17 | Yes | - | Yes | - | PASS | cost still high; N009 says the route is back to normal - not a reason for a rise |
| Chennai-Bangalore | 2025-03-24 | No | - | No | - | PASS | cost dropped back |
| Delhi-Jaipur | 2024-11-11 | Yes | - | Yes | - | PASS | large spike with no note for this route or time (sample row) |
| Delhi-Jaipur | 2024-11-18 | Yes | - | Yes | - | PASS | spike continues; still no note |
| Delhi-Jaipur | 2024-05-20 | No | - | No | - | PASS | N007 is good news (road resurfaced) and there is no spike |
| Mumbai-Delhi | 2024-07-29 | No | - | No | - | PASS | N005 maintenance week; costs were not affected and there is no spike |
| Mumbai-Delhi | 2025-05-12 | No (justified) | N003 | No (justified) | N003 | PASS | diesel rise from May 5 (5-7%); remaining excess is within weekly noise |
| Kolkata-Bhubaneswar | 2025-06-09 | No | - | No | - | PASS | N008 no disruptions; no spike |
| Mumbai-Pune | 2025-09-15 | Yes | - | Yes | - | PASS | slow creep vs similar routes; N006 (stable demand) is not a reason (sample row) |
| Mumbai-Pune | 2025-10-27 | Yes | - | Yes | - | PASS | creep continues; N010 says compliance cost was absorbed with no rate change |
| Mumbai-Pune | 2025-12-01 | Yes | - | Yes | - | PASS | creep: about 26 points above its usual gap to Delhi-Jaipur |
| Mumbai-Pune | 2025-12-22 | Yes | - | Yes | - | PASS | creep continues to the end of the data |
| Mumbai-Pune | 2024-06-10 | No | - | No | - | PASS | raw gap vs Delhi-Jaipur looks high (+16.6%) but that is this route's normal premium |
| Delhi-Chennai | 2024-06-03 | No | - | No | - | PASS | calm week |
| Kolkata-Bhubaneswar | 2024-09-09 | No | - | No | - | PASS | calm week |

## 4. Note extraction (LLM reading vs my reading)

**10/10** notes read correctly (same can-justify decision and, for cost-raising notes, the same window).

| note | LLM effect | LLM timing | resolved window | usable | gold | validation problems | result |
|---|---|---|---|---|---|---|---|
| N001 | increase | date_range | 2025-02-24..2025-03-08 | yes | yes | - | PASS |
| N002 | increase | single_week | 2025-01-20..2025-01-26 | yes | yes | - | PASS |
| N003 | increase | ongoing_from_date | 2025-05-05..2025-06-29 | yes | yes | - | PASS |
| N004 | unclear | ongoing_from_date | 2024-03-11..2024-05-05 | no | no | - | PASS |
| N005 | no_increase | single_week | 2024-07-29..2024-08-04 | no | no | - | PASS |
| N006 | no_increase | unspecified | 2025-09-22..2025-09-28 | no | no | - | PASS |
| N007 | unclear | ongoing_from_date | 2024-05-20..2024-07-14 | no | no | - | PASS |
| N008 | no_increase | unspecified | 2025-06-09..2025-06-15 | no | no | - | PASS |
| N009 | unclear | ongoing_from_date | 2025-03-17..2025-05-11 | no | no | - | PASS |
| N010 | no_increase | ongoing_from_date | 2025-10-27..2025-12-21 | no | no | - | PASS |

## 5. Retrieval

Rank of the correct note for each justified episode: pure embedding search over all 10 notes vs the hybrid search the pipeline uses (route + time filters, then embeddings).

| route | week | gold note | pure rank | pure top-3 | hybrid rank | hybrid candidates |
|---|---|---|---|---|---|---|
| Ahmedabad-Mumbai | 2025-01-20 | N002 | 1 | N002, N003, N006 | 1 | N002 |
| Chennai-Bangalore | 2025-02-24 | N001 | 1 | N001, N009, N002 | 1 | N001, N009 |
| Chennai-Bangalore | 2025-03-03 | N001 | 1 | N001, N009, N002 | 1 | N001, N009 |
| Mumbai-Delhi | 2025-05-12 | N003 | 3 | N002, N005, N003 | 1 | N003 |

Pure semantic search does not know which week a note covers, which is why time and route are hard filters and the final say is the rule-based eligibility check.

## 6. Threshold sensitivity

| own-history threshold | peer-excess threshold | flagged Yes | justified | labelled weeks correct |
|---|---|---|---|---|
| 8% | 10 pts | 21 | 4 | 20/20 |
| 8% | 12 pts | 16 | 4 | 20/20 |
| 8% | 15 pts | 12 | 4 | 20/20 |
| 10% | 10 pts | 20 | 4 | 20/20 |
| 10% | 12 pts (chosen) | 15 | 4 | 20/20 |
| 10% | 15 pts | 10 | 4 | 19/20 |
| 12% | 10 pts | 20 | 3 | 19/20 |
| 12% | 12 pts | 15 | 3 | 19/20 |
| 12% | 15 pts | 10 | 3 | 18/20 |
| 15% | 10 pts | 20 | 3 | 19/20 |
| 15% | 12 pts | 15 | 3 | 19/20 |
| 15% | 15 pts | 10 | 3 | 18/20 |
