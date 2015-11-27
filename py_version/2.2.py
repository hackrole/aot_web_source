from kafka import KafkaClient, SimpleConsumer

kafka = KafkaClient("localhost:9092")				
consumer = SimpleConsumer(kafka,"mygroup","test")					

for message in consumer:
	print(message)
