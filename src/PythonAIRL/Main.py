import json  # Useful for structuring complex game data
import socket

# Udp or TCP Server
def send_message_2_unreal(action_payload: dict[str,str|float|int], connection: socket.socket) -> None:
    message = json.dumps(action_payload)  # Convert to string / bytes so we can send data back to unreal engine
    connection.sendall(message.encode())  # send data back to unreal engine
    print(f"Sent data to Unreal Engine!:\n\tData Sent: {message}")


class UnrealPythonBridge:
    def __init__(self, hostname: str = "localhost", port : int = 3000, connection_type : str = "TCP") -> None:
        """
        Initializes the bridge to Unreal Engine

        Parameters:
            hostname: The host ip address. (Example: '192.168.10.121')
            port: The port number. (Example: '3000')
            connection_type: The connection type, can either be TCP server or UDP server. (Example: 'TCP')
        """

        self.host = hostname
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM) if connection_type == "TCP" else socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def start_server_2_unreal(self) -> None:
        try:
            self.sock.bind((self.host, self.port))
        except socket.error as e:
            print(f"Something went wrong with:\n\tSocket Binding: {e}")
            return

        self.sock.listen(1)
        # if host is localhost, print ip address aswell
        localhost_checker = f"({self.sock.getsockname()[0]})" if self.host == "localhost" else ""
        print(f"\nTCP Server is listening on:\n\tHOST:{self.host} {localhost_checker}\n\tPORT:{self.port}\n\tCONN:{self.sock.type}")

        connection, client_address = self.sock.accept()
        with connection:
            print(f"Successfully connected to Unreal Engine!")

            while True:
                try:
                    data = connection.recv(1024) # Unreal sends the state first (Example: Position, Velocity, etc.)
                    if not data:
                        print("It appears that there is no data from Unreal Engine?\n\tclosing the connection for now...")
                        break

                    decoded_data = data.decode("utf-8") # decode the unreal engine received data
                    print(f"Data received from Unreal Engine!:\n\tData: {decoded_data}")

                    # --- AIRL section of the code ---

                    # From Unreal Engine I will need:
                    # Current Speed
                    # Distance to the car infront
                    # How far the car is from the center lane (0 is center, 1 is on the line)
                    # If there was a collision recently
                    action_payload: dict[str, str|float|int] = {
                        "action": "Speed",
                        "value": 0.0,
                    }

                    send_message_2_unreal(action_payload, connection)

                    # --- AIRL section Ends here ---

                except Exception as e:
                    print(f"Something went wrong with:\n\tSocket Binding: {e}")
                    break

unreal_bridge = UnrealPythonBridge(hostname="localhost", port=3000, connection_type="TCP")
unreal_bridge.start_server_2_unreal()