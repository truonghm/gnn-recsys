copy (
	select 
		source_document_id,
		recommended_document_id
	from read_csv(
		'data/rard2/recommendation_log.csv',
		columns={
			source_document_id: 'VARCHAR',
			recommended_document_id: 'VARCHAR',
			clicked: 'VARCHAR'
		},
		auto_detect = false,
		delim = '\t'
	)
	where clicked <> 'null'
) to 'data/rard2/extracted_log.csv' (HEADER, DELIMITER ',');