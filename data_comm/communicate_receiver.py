#%%
import zmq
from data_comm.communicate_datapacket import DataPacket
import time

def communicate_receiver(url: str):

    wait_interval: float = 0.25
    n_intervals: int = 4
    running = True
    
    with zmq.Context() as context:
        socket = context.socket(zmq.REP)
        socket.bind(url)
        
        try:

            while running:

                count = 0
                while socket.closed is False:

                    # counting for a while
                    print(f"Counting a while:")
                    for i in range(n_intervals):
                        print(f"Count is {count}...")
                        count += 1
                        time.sleep(wait_interval)
                    print(f"Now looking for packets:")

                    # looking for packets
                    try:
                        packet = socket.recv_pyobj(zmq.NOBLOCK)
                        packet_available = True
                        print(f"Packet available!")
                        flag_stop = True #packet.stop

                    except zmq.error.Again:
                        packet_available = False
                        print(f"Nothing found")

                    if packet_available:
                        print(f"Processing packet data...")
                        # extract information from packet and process it
                        # here I'm just sending the same information back
                        packet_response = DataPacket()
                        packet_response.okay = True
                        flag_stop = True
                        socket.send_pyobj(packet_response)
                        if flag_stop:
                            running = False
                            print(f"Stop requested, cleaning up!")
                            break

                if flag_stop:
                    running = False
                    print(f"Stop requested, cleaning up!")
                    break

        except KeyboardInterrupt:
            print(f"Forcing interruption!")
            running = False

        print(f"Closing!")
        socket.close()
        context.destroy()

    print(f"Context seems to have been released!")
    return packet
""" 
import zmq
from data_comm.communicate_datapacket import DataPacket
import time


def communicate_receiver(url: str):

    #url: str = "tcp://*:5555"  # number specifies communication channel
    wait_interval: float = 0.25
    n_intervals: int = 4
    confirm = False
    flag_stop = False

    with zmq.Context() as context:
        socket = context.socket(zmq.REP)
        socket.bind(url)
        
        try:

            while not flag_stop:


                # while (socket.closed is False) and (end_program is False):
                count = 0
                while socket.closed is False:

                    # here we do whatever the program does...
                    print()
                    print(f"Counting a while:")
                    for i in range(n_intervals):
                        print(f"Count is {count}...")
                        count += 1
                        time.sleep(wait_interval)
                    print(f"Now looking for packets:")

                    # look for packets
                    try:
                        packet = socket.recv_pyobj(zmq.NOBLOCK)
                        packet_available = True
                        print(f"   ...packet available!")

                    except zmq.error.Again:
                        packet_available = False
                        print(f"   ...nothing found")

                    if packet_available:

                        if packet.stop:
                            print(f"I have to stop this program!")                                  
                        print(f"Confirm data packet was received and processed.")
                        packet_response = DataPacket()
                        packet_response.okay = True
                        socket.send_pyobj(packet_response)
                        print(f"Trying to close socket!")
                        socket.close()
        except KeyboardInterrupt:
            print(f"forcing interruption aargh!")

            # generates and sends end packet
            print(f" sending packet...")
            packet_response = DataPacket()
            packet_response.okay = False
            socket.send_pyobj(packet_response)
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


            print(f"Socket is closed!")
        socket.close()
        context.destroy()



        print(f"Context seems to have been released!")
    return packet
"""

# %%
