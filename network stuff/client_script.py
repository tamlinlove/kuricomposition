import time
import client


servers = ["localhost"]
ports = [12007]
sockets = [0]

FORWARD = [0.6,0,0,0,0,0]
TURN_LEFT = [0,0,0,0,0,1.6]
TURN_RIGHT = [0,0,0,0,0,-1.6]


def run():
    while True:
        command = input()
        if command == "w":
            limitSet = FORWARD
        elif command == "a":
            limitSet = TURN_LEFT
        elif command == "d":
            limitSet = TURN_RIGHT
        elif command == "q":
            break
        else:
            print("Command not recognised. Must be w, a, d or q")
            continue

        client.send_to_server(limitSet, servers[0], sockets[0]) #Sending sets to servers
        results = client.get_replies(sockets[0])
        time.sleep(1)


if __name__ == "__main__":
    client.create_connections(servers[0], ports[0], sockets[0])

    run()
    # closing connections to servers
    client.close_connections(sockets[0])
