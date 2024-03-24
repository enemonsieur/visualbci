#%%
import zmq
from data_comm.communicate_datapacket import DataPacket
import time

def communicate_receiver(url: str):
    wait_interval: float = 0.25
    n_intervals: int = 4
    confirm = False #confirm reception and closing

    with zmq.Context() as context:
        socket = context.socket(zmq.REP)
        socket.bind(url)
        while not confirm:
            count = 0
            while socket.closed is False:
                print()
                print(f"Counting a while:")
                for i in range(n_intervals):
                    print(f"Count is {count}...")
                    count += 1
                    time.sleep(wait_interval)
                print(f"Now looking for packets:")
                try:
                    packet = socket.recv_pyobj(zmq.NOBLOCK)
                    packet_available = True
                    print(f"   ...packet available!")
                    confirm = False
                except zmq.error.Again:
                    packet_available = False
                    print(f"   ...nothing found")
                flag_stop = False
                if packet_available:
                    if packet.stop:
                        flag_stop = True
                        print(f"I have to stop this program!")
                    print(f"Confirm data packet was received and processed.")
                    packet_response = DataPacket()
                    packet_response.okay = True
                    socket.send_pyobj(packet_response)
                if flag_stop:
                    print(f"Trying to close socket!")
                    socket.close()
            print(f"Socket is closed!")
        socket.close()
        print(f"Context seems to have been released!")

def main():
    url = "tcp://*:5555"
    try:
        communicate_receiver(url)
    except KeyboardInterrupt:
        print("Program terminated by user.")
#%%
if __name__ == "__main__":
    print("launching the test")
    main()
    print("test done!")


# %%
