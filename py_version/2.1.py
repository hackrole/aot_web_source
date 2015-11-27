from kafka import KafkaClient, SimpleProducer

kafka = KafkaClient("localhost:9092")
producer = SimpleProducer(kafka)

producer.send_messages("test", "Hello World!")
producer.send_messages("test","This is my second message")
producer.send_messages("test","And this is my third!")
