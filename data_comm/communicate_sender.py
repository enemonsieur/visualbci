#%%
from data_comm.communicate_datapacket import DataPacket
import zmq
import time

def communicate_sender(packet: DataPacket, url: str):


    #wait_interval: float = 5
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    
    with socket.connect(url):

        try:
            running = True
            while running:

                # waiting for specified time
                #print(f"Waiting for {wait_interval} seconds...")
                #time.sleep(wait_interval)

                # generates and sends data packet
                print(f"Generating and sending packet...")
                socket.send_pyobj(packet)

                # waits for confirmation that data was received
                print(f"Waiting for reception confirmation...")
                confirm = False

                while not confirm:
                    try:
                        answer = socket.recv_pyobj(zmq.NOBLOCK)
                        confirm = True
                        running = False

                    except zmq.error.Again:
                        pass

                if not answer.okay:
                    print(f"Something went wrong, exiting!")
                    running = False

                if answer.stop:
                    print(f"Stop requested, cleaning up!")
                    running = False

        except KeyboardInterrupt:
            print(f"Da will mich jemand abtöten - arrrrrrghhh!")

            # generates and sends end packet
            print(f"Generating and sending packet...")
            packet.stop = True
            socket.send_pyobj(packet)
            confirm = False

            # waits for confirmation that data was received
            print(f"Waiting for reception confirmation...")
            confirm = False
            while not confirm:
                try:
                    answer = socket.recv_pyobj(zmq.NOBLOCK)
                    confirm = True
                except zmq.error.Again:
                    pass

            if not answer.okay:
                print(f"Something went wrong, but we're exiting anyway!")


    print("Closing!")
    socket.close()
    context.destroy()

    print("Jetzt ist aber Schluss!")
