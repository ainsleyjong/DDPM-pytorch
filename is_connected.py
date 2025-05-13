import socket

def is_connected():
    try:
        # connect to the host -- tells us if the host is actually
        socket.create_connection(("1.1.1.1", 53))
        return True
    except OSError:
        pass
    return False
if __name__ == '__main__':
  r=is_connected()
  print("connected" if r else "not connected")
