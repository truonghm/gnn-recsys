with cs_ids as (
	select
		id,
		lower(doi) as doi
	from read_json(
		'./data/cs/mag_papers.txt',
		columns = {
			id: 'VARCHAR',
			doi: 'VARCHAR',
			},
		auto_detect = true,
		format = 'newline_delimited'
		)
),
rard2 as (
	select
		cast(r.paper_id as VARCHAR) as paper_id,
		r.recommended_papers,
		ex.external_name,
		lower(ex.external_document_id) as doi
	from read_csv('data/rard2/ground_truth.csv') r
	left join read_csv(
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
	on cast(r.paper_id as VARCHAR) = ex.mdl_document_id
	where ex.external_name = 'doi'
	-- and ex.external_document_id is null
)

select
	r.paper_id,
	cs_ids.id as oagcs_id,
	r.recommended_papers,
from rard2 r
left join cs_ids
on r.doi = cs_ids.doi
where cs_ids.id is not null