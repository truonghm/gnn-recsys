with 
cs as (
	select
		id,
		lower(regexp_replace(title, '[^a-zA-Z\s]', '')) as title
	from read_json(
		'./data/oag-cs/mag_papers.txt',
		columns = {
			id: 'VARCHAR',
			title: 'VARCHAR',
			},
		auto_detect = true,
		format = 'newline_delimited'
		)
),
relations as (
	select
		PaperId as paper_id,
		RecommendedPaperId as rec_id,
		StudyRankByMethod as rec_local_rank
	from read_csv(
		'data/user_study_results.tsv',
		delim = '\t'
	)
)

select
	-- r.paper_id,
	-- r.rec_id,
	-- r.rec_local_rank,
	-- cs.title
	count(distinct r.paper_id)
from relations r
left join cs
on r.paper_id = cs.id
left join cs as cs2
on r.rec_id = cs2.id
where cs.title is not null
and cs2.title is not null