# OAG Dataset

## Original Dataset Introduction

The [Open Academic Graph 2.1](https://www.aminer.cn/oag-2-1) is the combination of the [Microsoft Academic Graph (MAG)](https://academic.microsoft.com/) and the [AMiner](https://aminer.org/) dataset.

Below is the statistics of the OAG dataset, which totals to about 182 GB:

| Type        | File                  | Size  |
| ----------- | --------------------- | ----- |
| author      | mag_authors_{0-1}.zip | 11GB  |
| paper       | mag_papers_{0-16}.zip | 171GB |
| venue       | mag_venues.zip        | 1.8MB |
| affiliation | mag_affiliations.zip  | 1.4MB |


### Authors

```bash
python -m src.data.preprocessing.analyze author /media/truonghm/WD-Linux-1TB/OAG21_Dataset
```

```
mag_authors_1.zip\mag_authors_3.txt
mag_authors_1.zip\mag_authors_4.txt
mag_authors_0.zip\mag_authors_0.txt
mag_authors_0.zip\mag_authors_1.txt
mag_authors_0.zip\mag_authors_2.txt
Data type: author
Total: 243477150
Max field set: {'name', 'n_pubs', 'last_known_aff_id', 'n_citation', 'normalized_name', 'pubs', 'id'}
Min field set {'name', 'n_pubs', 'normalized_name', 'pubs', 'id'}
Field occurrence ratio: {'id': 1.0, 'name': 1.0, 'normalized_name': 1.0, 'pubs': 1.0, 'n_pubs': 1.0, 'n_citation': 0.39566894470384595, 'last_known_aff_id': 0.17816547055853085}
Example: {'id': 2754995334, 'name': 'Kabilan S. Jagadheesan', 'normalized_name': 'kabilan s jagadheesan', 'last_known_aff_id': '61923386', 'pubs': [{'i': 2889426607, 'r': 0}, {'i': 2912361570, 'r': 2}, {'i': 2755827189, 'r': 0}], 'n_pubs': 3, 'n_citation': 2}
```