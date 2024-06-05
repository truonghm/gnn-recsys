# OAG Dataset

## Original Dataset Introduction

The [Microsoft Academic Graph (MAG)](https://academic.microsoft.com/) is downloaded as part of the [Open Academic Graph 2.1](https://www.aminer.cn/oag-2-1) dataset.

Below is the statistics of the MAG dataset, which totals to about 182 GB:

| Type        | File                  | Size  | Number of nodes |
| ----------- | --------------------- | ----- | --------------- |
| author      | mag_authors_{0-1}.zip | 11GB  | 243477150       |
| paper       | mag_papers_{0-16}.zip | 171GB | 240255240       |
| venue       | mag_venues.zip        | 1.8MB | 53422           |
| affiliation | mag_affiliations.zip  | 1.4MB |                 |


### Authors

Running the following command will analyze the author dataset:

```bash
python -m src.data.preprocessing.analyze author /media/truonghm/WD-Linux-1TB/OAG21_Dataset
```

Results:

- Total number of nodes: 243,477,150
- Data schema:

| Field             | Occurrence % |
| ----------------- | ------------ |
| id                | 100.00       |
| name              | 100.00       |
| normalized_name   | 100.00       |
| pubs              | 100.00       |
| n_pubs            | 100.00       |
| n_citation        | 39.56        |
| last_known_aff_id | 17.81        |

Example of one record:

```
{
	'id': 2754995334, 
	'name': 'Kabilan S. Jagadheesan', 
	'normalized_name': 'kabilan s jagadheesan', 
	'last_known_aff_id': '61923386', 
	'pubs': [{'i': 2889426607, 'r': 0}, {'i': 2912361570, 'r': 2}, {'i': 2755827189, 'r': 0}], 
	'n_pubs': 3, 
	'n_citation': 2
}
```

### Papers

Running the following command will analyze the paper dataset:

```bash
python -m src.data.preprocessing.analyze paper /media/truonghm/WD-Linux-1TB/OAG21_Dataset
```

Results:

- Total number of nodes: 240,255,240
- Data schema:

| Field            | Occurrence % |
| ---------------- | ------------ |
| id               | 100.00       |
| title            | 99.99        |
| authors          | 99.98        |
| year             | 99.99        |
| publisher        | 52.83        |
| fos              | 87.58        |
| page_start       | 50.85        |
| page_end         | 44.68        |
| volume           | 43.23        |
| issue            | 41.51        |
| url              | 94.14        |
| n_citation       | 37.95        |
| doc_type         | 62.72        |
| references       | 32.83        |
| venue            | 59.78        |
| doi              | 37.33        |
| indexed_abstract | 58.32        |

Example of one record:

```
{
	'id': 2162162619, 
	'title': 'Immunologic Therapy of Multiple Sclerosis', 
	'authors': [{'name': 'Barry G. W. Arnason', 'id': 429337276, 'org': 'Department of Neurology, Pritzker School of Medicine, Chicago, Illinois, USA.\r', 'org_id': 40347166}], 
	'venue': {'name': 'Annual Review of Medicine', 'id': 16909082}, 
	'year': 1999, 
	'n_citation': 69, 
	'page_start': '291', 
	'page_end': '302', 
	'doc_type': 'Journal', 
	'publisher': 'Annual Reviews  4139 El Camino Way, P.O. Box 10139, Palo Alto, CA 94303-0139, USA', 
	'volume': '50', 
	'issue': '1', 
	'doi': '10.1146/ANNUREV.MED.50.1.291', 
	'references': [1967507425, 1992232972, 1995082592, 1998308647, 1998455368, 2005506388, 2019237986, 2030784878, 2039484868, 2044173905, 2070625707, 2079367120, 2081520990, 2119168009, 2127086238, 2139367012, 2142240629, 2165547142, 2167365151, 2321043971], 
	'indexed_abstract': '{"IndexLength":119,"InvertedIndex":{"Three":[0],"interferon":[1],"β":[2],"preparations":[3],"(Betaseron®,":[4],"Avonex®,":[5],"and":[6,25,47,79],"Rebif®)":[7],"have":[8],"shown":[9],"efficacy":[10],"in":[11,94],"the":[12,103,108],"treatment":[13,110],"of":[14,69,92,102,105],"relapsing-remitting":[15],"multiple":[16],"sclerosis":[17],"(MS).":[18],"Attack":[19],"frequency":[20,76],"is":[21,44,57,111,114],"reduced":[22,58],"by":[23,38,52,59,77,86],"30%":[24,78],"major":[26],"attacks":[27],"to":[28],"an":[29],"even":[30],"greater":[31],"extent.":[32],"Accumulating":[33],"disease":[34,48,82,100],"burden":[35],"as":[36,50,84],"measured":[37,51,85],"annual":[39],"T2-weighted":[40],"magnetic":[41],"resonance":[42],"imaging":[43],"markedly":[45],"lessened,":[46],"activity":[49,83],"serial":[53],"gadolinium-enhanced":[54,87],"MRI":[55],"scanning":[56],"over":[60],"80%.":[61],"A":[62],"fourth":[63],"preparation,":[64],"Copaxone®,":[65],"a":[66,116],"basic":[67],"copolymer":[68],"four":[70],"amino":[71],"acids,":[72],"lessens":[73,81,90],"MS":[74,95,113],"attack":[75],"also":[80],"MRI.":[88],"Betaseron®":[89],"accumulation":[91],"disability":[93,106],"patients":[96],"with":[97],"secondary":[98],"progressive":[99],"regardless":[101],"severity":[104],"at":[107],"time":[109],"commenced.":[112],"now":[115],"treatable":[117],"disease.":[118]}}', 
	'fos': [{'name': 'immunotherapy', 'w': 0.42654}, {'name': 'central nervous system disease', 'w': 0.45436}, {'name': 'diabetes mellitus', 'w': 0.43007}, {'name': 'immunology', 'w': 0.40303}, {'name': 'chemotherapy', 'w': 0.43925}, {'name': 'disease', 'w': 0.42774}, {'name': 'medicine', 'w': 0.39842}, {'name': 'magnetic resonance imaging', 'w': 0.46508}, {'name': 'multiple sclerosis', 'w': 0.51907}, {'name': 'antibody', 'w': 0.40809}], 
	'url': ['https://www.ncbi.nlm.nih.gov/pubmed/10073279', 'https://www.annualreviews.org/doi/full/10.1146/annurev.med.50.1.291', 'http://www.annualreviews.org/doi/10.1146/annurev.med.50.1.291']
}
```

### Venues

Running the following command will analyze the venue dataset:

```bash
python -m src.data.preprocessing.analyze venue /media/truonghm/WD-Linux-1TB/OAG21_Dataset
```

Results:

- Total number of nodes: 53,422
- Data schema:

| Field          | Occurrence % |
| -------------- | ------------ |
| id             | 100.00       |
| JournalId      | 91.63        |
| DisplayName    | 100.00       |
| NormalizedName | 100.00       |
| ConferenceId   | 8.37         |

Example: 

```
{
	'id': 2898614270, 
	'JournalId': 2898614270, 
	'DisplayName': 'Revista de Psiquiatría y Salud Mental', 
	'NormalizedName': 'revista de psiquiatria y salud mental'
}
```


### Affiliations

Running the following command will analyze the affiliation dataset:

```bash
python -m src.data.preprocessing.analyze affiliation /media/truonghm/WD-Linux-1TB/OAG21_Dataset
```

Results:

- Total number of nodes: 25,776
- Data schema:

| Field         | Occurrence % |
| ------------- | ------------ |
| id            | 100.00       |
| DisplayName   | 100.00       |
| NormalizedName | 100.00       |
| WikiPage      | 98.88        |
| Latitude      | 100.00       |
| Longitude     | 100.00       |
| url           | 66.50        |

Example: 

```
{
	'id': 3032752892,
	'DisplayName': 'Universidad Internacional de La Rioja',
	'NormalizedName': 'universidad internacional de la rioja',
	'WikiPage': 'https://en.wikipedia.org/wiki/International_University_of_La_Rioja',
	'Lattitude': '42.46270',
	'Longitude': '2.45500',
	'url': 'https://en.unir.net/'
}
```


## Computer Science Subset Extraction

In order to prepare the dataset for the recommendation system, we need to extract a subset of the original dataset that is relevant to the computer science field. To do this, we filter the papers in computer science in the last 10 years and manually build a list of relevant keywords in computer science to filter the papers using the `fos` key in the paper nodes. Papers with empty main fields (title, authors, fos, venue, year) are removed.

```bash
python -m src.data.preprocessing.extract_cs /media/truonghm/WD-Linux-1TB/OAG21_Dataset
```

The output includes 5 files:

- Authors: `mag_authors.txt`

	```json
	{"id": aid, "name": "author name", "org": oid}
	```

- Papers: `mag_papers.txt`

	```json
	{
	"id": pid,
	"title": "paper title",
	"authors": [aid],
	"venue": vid,
	"year": year,
	"abstract": "abstract",
	"fos": ["field"],
	"references": [pid],
	"n_citation": n_citation
	}
	```

- Venues: `mag_venues.txt`

	```json
	{"id": vid, "name": "venue name"}
	```

- Institutions: `mag_institutions.txt`

	```json
	{"id": oid, "name": "org name"}
	```

- Fields: `mag_fields.txt`

	```json
	{"id": fid, "name": "field name"}
	```