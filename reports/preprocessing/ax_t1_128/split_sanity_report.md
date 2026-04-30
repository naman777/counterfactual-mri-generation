# Split Sanity Check Report

## Overall Status

- Patient leakage check: **PASS**
- Missing image files: **0**
- Processed metadata rows: **6355**

## Patient Counts

| Split | Images | Patients |
|---|---:|---:|
| train | 4434 | 72 |
| val | 987 | 15 |
| test | 934 | 16 |

## Manufacturer Distribution

| Split | Distribution |
|---|---|
| train | `{'GE MEDICAL SYSTEMS': 3541, 'SIEMENS': 893}` |
| val | `{'GE MEDICAL SYSTEMS': 709, 'SIEMENS': 278}` |
| test | `{'GE MEDICAL SYSTEMS': 587, 'SIEMENS': 347}` |

## Field Strength Distribution

| Split | Distribution |
|---|---|
| train | `{'1.5T': 2448, '3.0T': 1986}` |
| val | `{'1.5T': 521, '3.0T': 466}` |
| test | `{'3.0T': 504, '1.5T': 430}` |

## EchoTime Statistics

| Split | Min | Max | Mean | Std |
|---|---:|---:|---:|---:|
| train | 7.8800 | 12.2720 | 9.9655 | 1.2050 |
| val | 7.9000 | 12.2720 | 10.1664 | 1.1205 |
| test | 8.4000 | 10.9800 | 9.4369 | 0.7312 |

## Patient Overlap Details

- Train-Val overlap: `[]`
- Train-Test overlap: `[]`
- Val-Test overlap: `[]`

## Warnings

- None
