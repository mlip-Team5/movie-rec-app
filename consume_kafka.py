
from confluent_kafka import Consumer

# Set your team number here
topic = 'movielog5'
bootstrap_servers = 'localhost:9092'

conf = {
    'bootstrap.servers': bootstrap_servers,
    'group.id': 'movie-recommender-group',
    'auto.offset.reset': 'earliest'
}

print(f"Connecting to Kafka at {bootstrap_servers}...")
consumer = Consumer(conf)
print(f"Subscribing to topic: {topic}")
consumer.subscribe([topic])
print("Polling for messages...")

try:
    with open('kafka_events.csv', 'w') as f:
        f.write('raw_event\n')  # CSV header
        empty_count = 0
        while True:
            msg = consumer.poll(2.0)
            if msg is None:
                empty_count += 1
                print(f"No message received (attempt {empty_count})...")
                if empty_count >= 10:
                    print("No messages after 10 attempts. Exiting.")
                    break
                continue
            if msg.error():
                print(f"Consumer error: {msg.error()}")
                continue
            event = msg.value().decode('utf-8')
            f.write(f'"{event}"\n')
            print(f"Received event: {event}")
            empty_count = 0  # Reset on message
finally:
    consumer.close()
    print("Kafka consumer closed.")
