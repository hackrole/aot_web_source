from kafka import KafkaClient, SimpleProducer

kafka = KafkaClient("localhost:9092")
producer = SimpleProducer(kafka,async=False,
			  req_acks=SimpleProducer.ACK_AFTER_CLUSTER_COMMIT,
			  ack_timeout=2000)

producer.send_messages("test-replicated-topic", "Hello Kafka Cluster!")
producer.send_messages("test-replicated-topic","Message to be replicated.")
producer.send_messages("test-replicated-topic","And so is this!")
