from kafka import KafkaClient
from kafka.common import ProduceRequest
from kafka.protocol import KafkaProtocol,create_message

kafka = KafkaClient("localhost:9093")

f = open('2.7.data','r')

for line in f:
	s = line.split("\t")[0]
	part = abs(hash(s)) % 3 
	req = ProduceRequest(topic="click-stream",partition=part,messages=[create_message(s)])
	resps = kafka.send_produce_request(payloads=[req], fail_on_error=True)
	
kafka.close()

