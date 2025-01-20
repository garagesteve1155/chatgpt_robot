import asyncio
import websockets
import time
import json  # Import JSON module for parsing metadata

async def receive_image(websocket, path):
    print(f"New connection from {websocket.remote_address}")
    expected_filename = None  # Initialize variable to store expected filename
    
    async for message in websocket:
        if isinstance(message, str):
            # Handle metadata (expected to be JSON with person's name)
            try:
                data = json.loads(message)
                person_name = data.get("person_name")
                if person_name:
                    expected_filename = person_name + ".jpg"
                    print(f"Metadata received: will save as {expected_filename}")
            except json.JSONDecodeError:
                print("Received invalid JSON metadata.")
                expected_filename = None
        elif isinstance(message, bytes):
            # Handle binary image data
            filename = expected_filename if expected_filename else "output1.jpg"
            with open(f"public_images/{filename}", "wb") as file:
                file.write(message)
            print(f"Image received and saved as {filename}.")
            expected_filename = None  # Reset after saving

while True:
    try:
        start_server = websockets.serve(receive_image, "0.0.0.0", 8040)
        print('server started')
        asyncio.get_event_loop().run_until_complete(start_server)
        asyncio.get_event_loop().run_forever()
    except Exception as e:
        print(e)
        time.sleep(1)



