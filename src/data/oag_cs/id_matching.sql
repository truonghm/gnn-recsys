with 
cs as (
	select
		id,
	from read_json(
		'./data/cs/mag_papers.txt',
		columns = {
			id: 'VARCHAR',
			},
		auto_detect = true,
		format = 'newline_delimited'
		)
),
relations as (
	select
		PaperId as paper_id,
		RecommendedPaperId as rec_id,
		coalesce(StudyRankByMethod, StudyRankGlobal) as rank
	from read_csv(
		'data/user_study_results.tsv',
		delim = '\t'
	)
)

select
	r.paper_id,
	list(r.rec_id order by r.rank desc) as rec_ids
from relations r
inner join cs
on r.paper_id = cs.id
inner join cs as cs2
on r.rec_id = cs2.id
where rank is not null
group by 1
having count(distinct r.rec_id) >= 3
