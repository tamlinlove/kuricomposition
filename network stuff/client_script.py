import time
import client


servers = ["localhost"]
ports = [12007]
sockets = [0]


def run():
    for i in range(10):
        limitSet = [i,0,0,0,0,0]
        client.send_to_server(limitSet, servers[0], sockets[0]) #Sending sets to servers
        results = client.get_replies(sockets[0])
        time.sleep(2)


if __name__ == "__main__":
    client.create_connections(servers[0], ports[0], sockets[0])

    run()
    # closing connections to servers
    client.close_connections(sockets[0])
