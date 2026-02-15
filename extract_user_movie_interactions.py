import csv
import re
from collections import defaultdict

input_file = 'kafka_events.csv'
output_file = 'user_movie_interactions.csv'

# Regex to match watch events: GET /data/m/<movieid>/<minute>.mpg
watch_event_pattern = re.compile(r'GET /data/m/(.+?)/(\d+)\.mpg')

def parse_watch_events_count():
    interactions = defaultdict(int)
    with open(input_file, 'r') as infile:
        reader = csv.reader(infile)
        next(reader)  # Skip header
        for row in reader:
            if not row:
                continue
            line = row[0].strip('"')
            parts = line.split(',')
            if len(parts) < 3:
                continue
            timestamp, userid, event = parts[0], parts[1], ','.join(parts[2:])
            match = watch_event_pattern.search(event)
            if match:
                movieid = match.group(1)
                key = (userid, movieid)
                interactions[key] += 1
    with open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['userid', 'movieid', 'score'])
        for (userid, movieid), count in interactions.items():
            writer.writerow([userid, movieid, count])

if __name__ == '__main__':
    parse_watch_events_count()
    print(f"User-movie interaction file with counts created: {output_file}")
