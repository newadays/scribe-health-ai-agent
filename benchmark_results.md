Runs per transcript per backend: 5

| Backend | Routing accuracy | Consistency | Median latency | p95 latency | Mean tokens (in+out) | Mean cost / triage | Cost / 1,000 triages |
|---|---|---|---|---|---|---|---|
| llm | 87% | 97% | 1.17 s | 1.62 s | 1019+84 | $0.000203 | $0.203 |
| jev | 87% | 97% | 0.13 s | 0.21 s | 1082+73 | $0.000045 | $0.045 |

| Transcript | Expected | Backend | Majority route | Accuracy | Mean confidence | Mean needs_review | Mean acuity | Median latency | Tokens (in+out) | Cost |
|---|---|---|---|---|---|---|---|---|---|---|
| pediatrics_visit.txt | pediatrics | llm | pediatrics | 100% | 1.00 | 0.00 | 1.0 | 1.31 s | 690+88 | $0.000156 |
| pediatrics_visit.txt | pediatrics | jev | pediatrics | 100% | 1.00 | 0.06 | 1.0 | 0.12 s | 725+72 | $0.000030 |
| general_medicine_visit.txt | general_medicine | llm | general_medicine | 100% | 1.00 | 0.00 | 1.0 | 1.25 s | 655+91 | $0.000153 |
| general_medicine_visit.txt | general_medicine | jev | general_medicine | 100% | 1.00 | 0.05 | 0.1 | 0.18 s | 690+73 | $0.000029 |
| ambiguous_note.txt | human_review | llm | human_review | 100% | 1.00 | 1.00 | 0.0 | 1.05 s | 545+58 | $0.000117 |
| ambiguous_note.txt | human_review | jev | human_review | 100% | 1.00 | 0.96 | 0.0 | 0.12 s | 574+73 | $0.000024 |
| general_medicine.txt | general_medicine | llm | general_medicine | 100% | 1.00 | 0.00 | 1.0 | 1.33 s | 3097+117 | $0.000535 |
| general_medicine.txt | general_medicine | jev | general_medicine | 100% | 1.00 | 0.10 | 0.2 | 0.13 s | 3317+73 | $0.000139 |
| teen_visit.txt | pediatrics | llm | pediatrics | 100% | 0.90 | 0.00 | 0.0 | 1.09 s | 661+88 | $0.000152 |
| teen_visit.txt | pediatrics | jev | human_review | 20% | 0.54 | 0.06 | 0.1 | 0.12 s | 703+72 | $0.000030 |
| truncated_visit.txt | human_review | llm | general_medicine | 20% | 0.90 | 0.20 | 0.6 | 0.91 s | 466+59 | $0.000105 |
| truncated_visit.txt | human_review | jev | human_review | 100% | 1.00 | 0.92 | 0.2 | 0.13 s | 486+73 | $0.000020 |
