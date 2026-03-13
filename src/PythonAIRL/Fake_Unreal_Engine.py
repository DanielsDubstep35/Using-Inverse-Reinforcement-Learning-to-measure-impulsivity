import socket
import json
import time
import random
import sys

# Configuration
HOST = '127.0.0.1'
PORT = 3000
Total_Frames_To_Simulate = 10*2  # Frames per second * Fuel

print("Hello World!")

# Toggle this to test your AIRL logic!
# "RECKLESS" = High Speed, Low Distance (Tailgating)
# "SAFE"     = Moderate Speed, High Distance
DRIVER_PERSONA = "RECKLESS"


def generate_fake_telemetry(persona):
    """
    Generates a single frame of data with metrics:
    score, speed, distance, overtakes, crashes
    """
    if persona == "RECKLESS":
        speed = random.uniform(130.0, 160.0)
        distance = random.uniform(5.0, 15.0)
        overtakes = 1 if random.random() < 0.3 else 0
        crashes = 1 if distance < 7.0 and random.random() < 0.1 else 0
        score = speed - (50 / max(distance, 1)) - (100 * crashes)
    else:
        speed = random.uniform(60.0, 80.0)
        distance = random.uniform(40.0, 60.0)
        overtakes = 1 if random.random() < 0.05 else 0
        crashes = 0
        score = speed + (distance * 0.5)

    return [
        score,        # 1. score
        speed,        # 2. mean speed (frame-level)
        distance,     # 3. mean distance (frame-level)
        overtakes,    # 4. overtakes (n, frame-level)
        crashes       # 5. crashes (n, frame-level)
    ]

def run_simulation():
    print(f"--- Starting Simulation: {DRIVER_PERSONA} Driver ---")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect((HOST, PORT))
            print(f"Connected to AIRL Server on {HOST}:{PORT}")
        except ConnectionRefusedError:
            print("Error: Could not connect. Make sure the AIRL Server is running first!")
            return

        # 1. SIMULATE GAMEPLAY LOOP
        for i in range(Total_Frames_To_Simulate):
            # Generate data based on persona
            telemetry = generate_fake_telemetry(DRIVER_PERSONA)

            # Format exactly as Server expects: {"type": "state", "data": [...]}
            packet = {
                "type": "state",
                "data": telemetry
            }

            # Send with Newline delimiter (Critical for your server's split('\n') logic)
            message = json.dumps(packet) + "\n"
            print("message that will be sent to AIRL Server: ", message)
            s.sendall(message.encode('utf-8'))

            # Print occasionally so we know it's working
            if i % 10 == 0:
                print(f"Sending Frame {i}/{Total_Frames_To_Simulate}: {telemetry}")

            # Simulate tick rate (fast for testing)
            time.sleep(0.05)

        # 2. SEND LEVEL COMPLETE
        print(">> Level Finished. Requesting Analysis...")
        end_packet = {"type": "level_complete"}
        s.sendall((json.dumps(end_packet) + "\n").encode('utf-8'))

        # 3. WAIT FOR RESULT
        # The server will now process the data and send back the score
        response = s.recv(4096)
        if response:
            try:
                result_json = json.loads(response.decode('utf-8'))
                print("\n" + "=" * 40)
                print(" ANALYSIS RECEIVED FROM PYTHON")
                print("=" * 40)
                print(f" Impulsivity Score: {result_json.get('impulsivity_score')}")
                print(f" Internal Weights:  {result_json.get('details')}")
                print("=" * 40 + "\n")
            except json.JSONDecodeError:
                print(f"Raw response (Not JSON): {response}")


if __name__ == "__main__":
    run_simulation()