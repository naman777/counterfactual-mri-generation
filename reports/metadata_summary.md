# Metadata Exploration Summary

## Dataset scale
- Number of slices: 101765
- Number of patients: 123
- Number of studies: 123
- Number of series: 811

## Manufacturer distribution
| Manufacturer       |   count |
|:-------------------|--------:|
| GE MEDICAL SYSTEMS |   73622 |
| SIEMENS            |   28012 |
| QIICR              |     131 |

## Field strength distribution
|   MagneticFieldStrength |   count |
|------------------------:|--------:|
|                     1.5 |   48837 |
|                     3   |   52797 |
|                   nan   |     131 |

## Missing value percentage
|                       |          0 |
|:----------------------|-----------:|
| Manufacturer          |   0        |
| MagneticFieldStrength |   0.128728 |
| EchoTime              |   0.128728 |
| RepetitionTime        |   0.128728 |
| FlipAngle             |   0.128728 |
| SliceThickness        |   0.128728 |
| SeriesDescription     |   0        |
| ProtocolName          | 100        |

## Echo Time summary
|       |     EchoTime |
|:------|-------------:|
| count | 101634       |
| mean  |      2.72518 |
| std   |      2.70003 |
| min   |      1.34    |
| 25%   |      1.72    |
| 50%   |      2.4     |
| 75%   |      2.476   |
| max   |    106.632   |

Unique EchoTime values: 90

## Repetition Time summary
|       |   RepetitionTime |
|:------|-----------------:|
| count |      101634      |
| mean  |          46.9371 |
| std   |         173.773  |
| min   |           3.54   |
| 25%   |           4.68   |
| 50%   |           5.272  |
| 75%   |           5.661  |
| max   |        5000      |

Unique RepetitionTime values: 129

## TE/TR grouped by manufacturer
| Manufacturer       |   ('EchoTime', 'count') |   ('EchoTime', 'mean') |   ('EchoTime', 'std') |   ('EchoTime', 'min') |   ('EchoTime', 'max') |   ('RepetitionTime', 'count') |   ('RepetitionTime', 'mean') |   ('RepetitionTime', 'std') |   ('RepetitionTime', 'min') |   ('RepetitionTime', 'max') |
|:-------------------|------------------------:|-----------------------:|----------------------:|----------------------:|----------------------:|------------------------------:|-----------------------------:|----------------------------:|----------------------------:|----------------------------:|
| GE MEDICAL SYSTEMS |                   73622 |                3.0176  |               2.87783 |                  2.22 |               106.632 |                         73622 |                      47.8288 |                     179.284 |                       4.792 |                        5000 |
| QIICR              |                       0 |              nan       |             nan       |                nan    |               nan     |                             0 |                     nan      |                     nan     |                     nan     |                         nan |
| SIEMENS            |                   28012 |                1.95664 |               1.96681 |                  1.34 |                12     |                         28012 |                      44.5933 |                     158.355 |                       3.54  |                         786 |

## TE/TR grouped by field strength
|   MagneticFieldStrength |   ('EchoTime', 'count') |   ('EchoTime', 'mean') |   ('EchoTime', 'std') |   ('EchoTime', 'min') |   ('EchoTime', 'max') |   ('RepetitionTime', 'count') |   ('RepetitionTime', 'mean') |   ('RepetitionTime', 'std') |   ('RepetitionTime', 'min') |   ('RepetitionTime', 'max') |
|------------------------:|------------------------:|-----------------------:|----------------------:|----------------------:|----------------------:|------------------------------:|-----------------------------:|----------------------------:|----------------------------:|----------------------------:|
|                     1.5 |                   48837 |                2.73809 |               2.30483 |                  1.34 |                12.272 |                         48837 |                      49.7993 |                     161.256 |                        4.12 |                         786 |
|                     3   |                   52797 |                2.71324 |               3.01986 |                  1.35 |               106.632 |                         52797 |                      44.2895 |                     184.558 |                        3.54 |                        5000 |

## Top series descriptions
| SeriesDescription         |   count |
|:--------------------------|--------:|
| ax t1                     |    6355 |
| ax dyn 1st pass           |    5624 |
| ax dyn 3rd pass           |    5624 |
| ax dyn pre                |    5624 |
| ax dyn 2nd pass           |    5624 |
| Ph2/ax 3d dyn MP          |    5232 |
| Ph3/ax 3d dyn MP          |    5232 |
| ax 3d dyn MP              |    5232 |
| Ph1/ax 3d dyn MP          |    5232 |
| Ph4/ax 3d dyn MP          |    5231 |
| Ph1/ax 3d dyn             |    4256 |
| ax 3d dyn                 |    4256 |
| Ph2/ax 3d dyn             |    4256 |
| Ph3/ax 3d dyn             |    4256 |
| Ph1/ax dynamic            |    2942 |
| ax dynamic                |    2942 |
| Ph3/ax dynamic            |    2942 |
| Ph2/ax dynamic            |    2768 |
| ax dyn 4th pass           |    2120 |
| Ph4/ax 3d dyn             |    1040 |
| Ph3/ax dyn                |     926 |
| Ph2/ax dyn                |     926 |
| ax dyn                    |     926 |
| Ph1/ax dyn                |     926 |
| ax 3d t1 bilateral        |     864 |
| Ph1/Ax Vibrant MultiPhase |     776 |
| Ax Vibrant MultiPhase     |     776 |
| Ax 3D T1 NON FS           |     776 |
| Ph4/Ax Vibrant MultiPhase |     776 |
| Ph2/Ax Vibrant MultiPhase |     776 |

## Decision
- Selected continuous parameter: TODO
- Reason: TODO
- Selected series for preprocessing: TODO
