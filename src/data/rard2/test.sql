with rard_matching as (
	select
		mdl_document_id as rard_id,
		lower(external_document_id) as doi
	from read_csv(
		'data/rard2/external_IDs.csv',
		delim = '\t',
		header = true,
		auto_detect = false,
		null_padding = true,
		columns = {
			mdl_document_id: 'VARCHAR',
			external_name: 'VARCHAR',
			external_document_id: 'VARCHAR'
		}
		) ex
	where ex.external_name = 'doi'
	and ex.external_document_id like '10.%'
	order by external_document_id DESC
	-- limit 5000
),
oag as (
	select
		id,
		year,
		lower(doi) as doi
	from read_json(
		'./data/cs/mag_papers.txt',
		columns = {
			id: 'VARCHAR',
			year: 'INT',
			doi: 'VARCHAR'
			},
		auto_detect = true,
		format = 'newline_delimited'
		)
	order by doi DESC
	-- limit 1000

),
rard as (
	select
		cast(r.paper_id as VARCHAR) as paper_id,
		r.recommended_papers,
	from read_csv('data/rard2/ground_truth.csv') r
)
select
	o.id as oag_id,
	rm.doi,
	o.year,
	r.recommended_papers
from rard_matching rm
left join oag o
on rm.doi = o.doi
left join rard r
on rm.rard_id = r.paper_id
where o.year is not null
and r.recommended_papers is not null