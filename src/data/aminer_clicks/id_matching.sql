with cs_ids as (
	select
		id,
		year,
		n_citation
	from read_json(
		'./data/oag-cs/mag_papers.txt',
		columns = {
			id: 'VARCHAR',
			year: 'INT',
			n_citation: 'INT',
			},
		auto_detect = true,
		format = 'newline_delimited'
		)
),
relations as (
	select
		cast(r.paper_id as VARCHAR) as paper_id,
		r.related_papers
	from read_json(
		'data/ground_truth_relations_rard.txt',
		format = 'newline_delimited'
		) r
)

select
	r.paper_id,
	-- cs_ids.id as oagcs_id,
	r.related_papers,
	-- cs_ids.year,
	-- cs_ids.n_citation
from relations r
left join cs_ids
on r.paper_id = cs_ids.id
where cs_ids.year is not null