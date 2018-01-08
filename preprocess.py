from google.cloud import bigquery
from pandas import Series

client = bigquery.Client(project='wixpre')

names = ['snp', 'nyse', 'djia', 'nikkei', 'hangseng', 'ftse', 'dax', 'aord']

for name in names:
    print "Processing {}".format(name)
    query = "SELECT Date, Close FROM `bingo-ml-1.market_data.{}` ORDER BY Date".format(name)

    client.query(query).result()

    query_job = client.query(query)
    query_job.result()

    table = client.get_table(query_job.destination)
    rows = list(client.list_rows(table))
    Series([x[1] for x in rows], index=[x[0].date() for x in rows]).to_json('data/{}.json'.format(name))

print "Done"
