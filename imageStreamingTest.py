import numpy as np
import cv2
import socket

class ImageStreamingTest(object):
    def __init__(self, host, port, cap_path):

        # self.server_socket = socket.socket()
        # self.server_socket.bind(('', port)) # remove host
        # self.server_socket.listen(0)
        # self.connection, self.client_address = self.server_socket.accept()
        self.cap_path = cap_path
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.connect((host, port))
        # self.connection = self.connection.makefile('rb')
        self.connection = self.server_socket.makefile('rb')
        self.host_name = socket.gethostname()
        self.host_ip = socket.gethostbyname(self.host_name)
        self.streaming()


    def streaming(self):

        try:
            print("Host: ", self.host_name + ' ' + self.host_ip)
            # print("Connection from: ", self.client_address)
            # print("Connection from: ", host)
            print("Streaming...")
            print("Press 'q' to exit")

            # need bytes here
            stream_bytes = b' '
            while True:
                stream_bytes += self.connection.read(1024)
                first = stream_bytes.find(b'\xff\xd8')
                last = stream_bytes.find(b'\xff\xd9')
                if first != -1 and last != -1:
                    jpg = stream_bytes[first:last + 2]
                    stream_bytes = stream_bytes[last + 2:]
                    image = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    # write the flipped frame
                    # out.write(image)
                    cv2.imwrite(self.cap_path, image) # r'C:\Users\yhtxy\Desktop\test.jpg'
                    # cv2.imshow('image', image)
                    break
                    # if cv2.waitKey(1) & 0xFF == ord('q'):
                    #     break
        finally:
            self.connection.close()
            self.server_socket.close()