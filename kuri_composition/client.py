from socket import *
import pickle
import time

# creating sockects for the different turtlebot servers
client_sockets = [socket(AF_INET, SOCK_STREAM), socket(AF_INET, SOCK_STREAM)]

def create_connections(server_name, server_port, socket):
	#Creating connections with servers.
	client_sockets[socket].connect((server_name, server_port))
	print ("Connected to host " + server_name)

def send_to_server(limit_set, server_name, socket):
	#print ("Sending range " + str(limit_set) + " to host " + str(server_name))
	data = pickle.dumps(limit_set, protocol=2)
	client_sockets[socket].send(data)

def get_replies(socket):
	 #Getting replies from servers.
	reply_answer_serialized = client_sockets[socket].recv(1024) # data in pickle format
	reply_answer_deserialized = pickle.loads(reply_answer_serialized) # pickle to array
	results = (reply_answer_deserialized[0]) #Extracting results from the array
	return results
def close_connections(socket): #Closing connections to different servers.

	client_sockets[socket].close()
'''
if __name__ == "__main__":#Main function

	createConnections()	#Creates three connections to servers
	for i in range(10):
		#createConnections()	#Creates three connections to servers
		# lLimit = input("input lower limit ")
		# uLimit = input("input upper limit ")
		x = input("x ")
		y = input("y ")
		z = input("z ")
		ax = input("ax ")
		ay = input("ay ")
		az = input("az ")
		#string = input("Enter the string you wish to send\n")
		print ("\n")
		limitSet = []
		startTime=time.time() #Start of connection time

		# limitSet = [lLimit,uLimit] #Breaking range into sets
		limitSet = [x, y, z, ax, ay, az]
		sendToServer(limitSet) #Sending sets to servers

		results, times = getReplies() #Getting results and search times
		#replyAnswerSerialized = clientSockets.recv(1024)
		#replyAnswerDeserialized = pickle.loads(replyAnswerSerialized)
		#print(replyAnswerDeserialized)
		elapsedTime = time.time() - startTime

		totalResults = 0
		totalTime = 0

		print ("got reply = " + str(results) + " from host " +str(serverNames)+ " after " +str(times)+ " seconds")
		totalResults = totalResults + results
		totalTime = totalTime + times
	closeConnections() #Closing all connections to servers.
'''
