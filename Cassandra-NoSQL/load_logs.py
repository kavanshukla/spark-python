import sys, re, string, os, gzip
from cassandra.cluster import Cluster
from cassandra.query import BatchStatement
import cassandra
import datetime as dt

def main():
	#Defining input directory, keyspace and table name
	MAX_LINES=500
	temp_line=0
	inputs = sys.argv[1]
	keyspace = sys.argv[2]
	table_name=sys.argv[3]

	#Cluster configuration
	cluster = Cluster(['199.60.17.136', '199.60.17.173'])
	session = cluster.connect(keyspace)
	session.execute('USE %s;' % keyspace)

	#Defining the query for inserting values into table nasalogs
	insert_query = session.prepare("INSERT INTO %s (host, datetime, path, bytes) VALUES (?, ?, ?, ?);" % table_name)
	linere = re.compile("^(\\S+) - - \\[(\\S+) [+-]\\d+\\] \"[A-Z]+ (\\S+) HTTP/\\d\\.\\d\" \\d+ (\\d+)$")

	for f in os.listdir(inputs):
		with gzip.GzipFile(os.path.join(inputs, f)) as logfile:
			batch = BatchStatement()
			for line in logfile:
				#splitting the row data as per the regular expression
				single_row = linere.split(line)
				
				#retrieving required values in the specific format as host,datetime,path and bytes 
				if len(single_row) == 6:
					host = single_row[1]
					#stripping date-time to its format
					date_time = dt.datetime.strptime(single_row[2], '%d/%b/%Y:%H:%M:%S')
					path = single_row[3]
					bytes_transferred = single_row[4]

				#packaging multiple insert queries into one batch statement	
				if temp_line<=MAX_LINES:
					temp_line+=1
					batch.add(insert_query, [host,date_time,path,int(bytes_transferred)])
					session.execute(batch)

				if temp_line==MAX_LINES:
					#checking batch threshold and clearing the batch when it meets the threshold
					batch.clear()
					temp_line=0

if __name__ == "__main__":
	main()
