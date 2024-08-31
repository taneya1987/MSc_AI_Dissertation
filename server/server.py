import os
import socket
import psutil
import tqdm
import time
from datetime import datetime

from logger.dataset_logger import dataset_log_message, dataset_log_file, dataset_log_chunk, \
    dataset_log_encryption_metrics
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization
from key_encapsulation.ecc import generate_private_key, get_public_key, get_shared_key
from key_encapsulation.crystalskyber import encapsulate_kyber
from encryption.aes_encryption import generate_aes_key, encrypt_data, decrypt_data
import threading

SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000
BUFFER_SIZE = 4096
CHUNK_SIZE = 1024  # Adjust this value as needed

# Client information
clients = {}
clients_lock = threading.Lock()  # Lock for thread-safe access to clients dictionary

# Create a socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((SERVER_HOST, SERVER_PORT))
server_socket.listen(5)
print(f"server listening on {SERVER_HOST}:{SERVER_PORT}")


def measure_cpu_memory():

    # Get the current process of the application
    process = psutil.Process(os.getpid())

    # CPU usage as a percentage of one CPU core
    cpu_usage = process.cpu_percent(interval=1)

    # Memory usage in MB
    memory_usage = process.memory_info().rss / (1024 * 1024)  # Resident Set Size (RSS)

    return cpu_usage, memory_usage


def handle_client(client_socket, client_addr):
    global encryption_time, decryption_time
    try:
        # Ask the user which key encapsulation technique to use
        key_encapsulation = input("Which key encapsulation technique do you want to use? (ECC/Kyber): ").strip().lower()

        handshake_start_time = time.time()
        if key_encapsulation.lower() == "ecc":
            encryption_type = 'ECC'
            # Generate ECC key pair
            server_private_key = generate_private_key()
            server_public_key = get_public_key(server_private_key)
            # Perform ECDH key exchange
            client_public_key = ec.EllipticCurvePublicKey.from_encoded_point(server_public_key.curve,
                                                                             client_socket.recv(BUFFER_SIZE))
            client_socket.sendall(server_public_key.public_bytes(
                encoding=serialization.Encoding.X962,
                format=serialization.PublicFormat.UncompressedPoint
            ))
            shared_key = get_shared_key(server_private_key, client_public_key)
            print(f'ECC shared key: {shared_key}')
        elif key_encapsulation.lower() == "kyber":
            encryption_type = 'Kyber'
            # Receive client's public key and send server's ciphertext
            client_public_key = client_socket.recv(BUFFER_SIZE)
            server_ciphertext, shared_secret = encapsulate_kyber(client_public_key)
            client_socket.sendall(server_ciphertext)
            # Derive AES key from shared secret
            shared_key = shared_secret
            print(f'Kyber shared key: {shared_key}')
        else:
            print("Invalid choice. Exiting.")
            client_socket.close()
            return

        handshake_end_time = time.time()
        handshake_time = (handshake_end_time - handshake_start_time) * 1000  # milliseconds
        cpu_usage, memory_usage = measure_cpu_memory()
        print(f"CPU Usage: {cpu_usage}%, Memory Usage: {memory_usage:.2f} MB")

        # Derive AES key from shared secret
        aes_key = generate_aes_key(shared_key)

        # Receive client hostname and local IP address
        encrypted_hostname = client_socket.recv(BUFFER_SIZE)
        hostname_ip = decrypt_data(encrypted_hostname, aes_key).decode()
        hostname, client_ip = hostname_ip.split(":")
        intruder_flag = 1 if not hostname.lower().startswith('client') else 0

        with clients_lock:
            clients[client_addr] = {
                "hostname": hostname,
                "aes_key": aes_key,
                "client_ip": client_ip,
                "connection_duration": 0,
                "num_connections": 1,
                "data_transfer_volume": 0,
                "request_frequency": 0,
                "error_rate": 0,
                "num_messages": 0,
                "num_files": 0
            }

        print(f"New client connected: {hostname} ('{client_ip}', '{client_addr[1]}')")

        # Create client folder and log file
        client_folder = os.path.join("../client_logs", hostname)
        os.makedirs(client_folder, exist_ok=True)
        log_file = os.path.join(client_folder, "log.txt")

        connection_start_time = datetime.now()

        while True:
            try:
                data = client_socket.recv(BUFFER_SIZE)
                request_received_time = datetime.now()
                if not data:
                    break

                # Decrypt data
                plaintext = decrypt_data(data, aes_key)
                print(f"Decrypted data: {plaintext}")

                if plaintext == b'list':
                    list_active_connections(plaintext, aes_key, client_socket, client_ip)
                    logging('message', hostname, 'list', log_file, None)
                    send_ack('message', None, None, 'list', aes_key, None, client_socket)
                    clients[client_addr]["request_frequency"] += 1

                elif plaintext == b'ping':
                    send_ack('message', None, None, 'ping', aes_key, None, client_socket)

                elif plaintext == b'server shutdown':
                    try:
                        print("server shutdown")
                        logging('message', hostname, 'server shutdown', log_file, None)
                        send_ack('message', None, None, plaintext, aes_key, None, client_socket)

                        response_sent_time = datetime.now()  # Record the time when the response is sent
                        server_response_time = (response_sent_time - request_received_time).total_seconds() * 1000
                        clients[client_addr]["num_messages"] += 1
                        clients[client_addr]["error_rate"] += 1
                        dataset_log_message(hostname, 'junk', plaintext, connection_start_time, client_ip,
                                            server_response_time,
                                            clients[client_addr]["connection_duration"],
                                            clients[client_addr]["num_messages"],
                                            clients[client_addr]["num_files"],
                                            clients[client_addr]["num_connections"],
                                            clients[client_addr]["data_transfer_volume"],
                                            clients[client_addr]["request_frequency"],
                                            clients[client_addr]["error_rate"],
                                            encryption_type, intruder_flag)
                        clients[client_addr]["request_frequency"] += 1
                        clients[client_addr]["data_transfer_volume"] += len(plaintext)

                    except Exception as e:
                        print(f'Exception during server shutdown: {e}')

                    finally:
                        client_socket.close()
                        if hostname.startswith("client"):
                            server_socket.close()
                        break

                elif plaintext.lower().startswith(b"file:"):
                    encrypted_file_info = data
                    file_info = decrypt_data(encrypted_file_info, aes_key).decode()
                    header, file_name, file_size = file_info.split(":")

                    invalid_file_type = ('.bin', '.zip', '.exe', 'AVI', 'wav', 'gif', 'wmv', 'mkv')
                    if any(file_name.__contains__(inv) for inv in invalid_file_type):
                        clients[client_addr]["error_rate"] += 1

                    file_size = int(file_size)
                    progress = tqdm.tqdm(unit='B', unit_scale=True, unit_divisor=1000, total=file_size)

                    # Receive file data in chunks
                    file_data = b""
                    remaining_bytes = file_size
                    chunk_count = 0
                    decryption_start_time = datetime.now()
                    try:
                        while remaining_bytes > 0:
                            chunk_count += 1
                            chunk = client_socket.recv(BUFFER_SIZE)
                            decrypted_chunk = decrypt_data(chunk, aes_key)

                            response_sent_time = datetime.now()  # Record the time when the response is sent
                            server_response_time = (response_sent_time - request_received_time).total_seconds() * 1000

                            dataset_log_chunk(hostname, file_name, file_size, connection_start_time, client_ip,
                                              server_response_time,
                                              clients[client_addr]["connection_duration"],
                                              clients[client_addr]["num_connections"],
                                              clients[client_addr]["data_transfer_volume"],
                                              clients[client_addr]["request_frequency"],
                                              clients[client_addr]["error_rate"],
                                              encryption_type, intruder_flag)

                            send_ack('chunk', None, None, None, aes_key, chunk_count, client_socket)

                            file_data += decrypted_chunk
                            remaining_bytes -= len(decrypted_chunk)
                            progress.update(1024)
                            clients[client_addr]["data_transfer_volume"] += len(decrypted_chunk)
                        encrypted_time = client_socket.recv(BUFFER_SIZE)
                        decrypt_encryption_time = decrypt_data(encrypted_time, aes_key)
                        encryption_time = float(decrypt_encryption_time.decode())
                        # send_ack('encryption_time', None, None, encryption_time, aes_key, None, client_socket)
                        print(f'Encryption time received from client: {encryption_time}')
                        decryption_time = (datetime.now() - decryption_start_time).total_seconds() * 1000  # milliseconds
                        print(f'Decryption time of the file: {decryption_time}')
                    except Exception as e:
                        print(f'Exception during chunk reception: {e}')
                        clients[client_addr]["error_rate"] += 1

                    # Save the file
                    file_path = os.path.join(client_folder, file_name)
                    with open(file_path, "wb") as f:
                        f.write(file_data)
                    print(f"\nReceived file from : {hostname} ({client_addr}): {file_name}")

                    # Log file received message
                    logging('file', hostname, None, log_file, file_name)
                    response_sent_time = datetime.now()  # Record the time when the response is sent
                    server_response_time = (response_sent_time - request_received_time).total_seconds() * 1000
                    print(f"server response time: {server_response_time} ms")
                    clients[client_addr]["num_files"] += 1
                    dataset_log_file(hostname, file_name, file_size, connection_start_time, client_ip,
                                     server_response_time,
                                     clients[client_addr]["connection_duration"],
                                     clients[client_addr]["num_messages"],
                                     clients[client_addr]["num_files"],
                                     clients[client_addr]["num_connections"],
                                     clients[client_addr]["data_transfer_volume"],
                                     clients[client_addr]["request_frequency"],
                                     clients[client_addr]["error_rate"],
                                     encryption_type, intruder_flag)

                    send_ack('file', header, file_name, None, aes_key, None, client_socket)
                    clients[client_addr]["request_frequency"] += 1

                    # Log encryption and decryption times
                    dataset_log_encryption_metrics(
                        hostname,
                        file_name,
                        encryption_type,
                        f"{handshake_time:.3f}",
                        f"{encryption_time:.3f}",
                        f"{decryption_time:.3f}",
                        connection_start_time,
                        client_ip,
                        f"{server_response_time:.3f}",
                        f"{cpu_usage:.3f}",
                        f"{memory_usage:.3f}"
                    )

                    file_name = None

                elif plaintext == b"close" or plaintext == b"exit":
                    print(f"Connection closed by client: {hostname} ({client_addr})")
                    logging('message', hostname, f'Connection closed by client: {hostname} ({client_addr})', log_file,
                            None)
                    with clients_lock:
                        del clients[client_addr]
                    break

                else:
                    # Print as a message
                    message = plaintext.decode()
                    print(f"{hostname}:> {message}")

                    special_chars = "!@#$%^&*()+-=[]{};':\"\\|,.<>/?`~"

                    if any(c in special_chars for c in message) and not hostname.startswith("client"):
                        print('Junk message received')
                        # Log message
                        logging('message', hostname, message, log_file, None)
                        response_sent_time = datetime.now()  # Record the time when the response is sent
                        server_response_time = (response_sent_time - request_received_time).total_seconds() * 1000
                        clients[client_addr]["num_messages"] += 1
                        clients[client_addr]["error_rate"] += 1
                        dataset_log_message(hostname, 'junk', message, connection_start_time, client_ip,
                                            server_response_time,
                                            clients[client_addr]["connection_duration"],
                                            clients[client_addr]["num_messages"],
                                            clients[client_addr]["num_files"],
                                            clients[client_addr]["num_connections"],
                                            clients[client_addr]["data_transfer_volume"],
                                            clients[client_addr]["request_frequency"],
                                            clients[client_addr]["error_rate"],
                                            encryption_type, intruder_flag)
                        send_ack('message', None, None, message, aes_key, None, client_socket)
                        clients[client_addr]["request_frequency"] += 1
                        clients[client_addr]["data_transfer_volume"] += len(message)

                    else:

                        # Log message
                        logging('message', hostname, message, log_file, None)
                        response_sent_time = datetime.now()  # Record the time when the response is sent
                        server_response_time = (response_sent_time - request_received_time).total_seconds() * 1000
                        clients[client_addr]["num_messages"] += 1
                        dataset_log_message(hostname, 'text', message, connection_start_time, client_ip,
                                            server_response_time,
                                            clients[client_addr]["connection_duration"],
                                            clients[client_addr]["num_messages"],
                                            clients[client_addr]["num_files"],
                                            clients[client_addr]["num_connections"],
                                            clients[client_addr]["data_transfer_volume"],
                                            clients[client_addr]["request_frequency"],
                                            clients[client_addr]["error_rate"],
                                            encryption_type, intruder_flag)
                        send_ack('message', None, None, message, aes_key, None, client_socket)
                        clients[client_addr]["request_frequency"] += 1
                        clients[client_addr]["data_transfer_volume"] += len(message)

                clients[client_addr]["connection_duration"] = (datetime.now() - connection_start_time).total_seconds()

            except Exception as e:
                print(f"Exception during reception: {e}")
                clients[client_addr]["error_rate"] += 1

    except Exception as e:
        print(f"Error handling client {client_addr}: {e}")
    finally:
        client_socket.close()


def list_active_connections(message, aes_key, client_sock, ip):
    try:
        with clients_lock:
            print("Active connections:")
            for addr, info in clients.items():
                print(f"Hostname: {info['hostname']}, IP: {ip}, Port: {addr[1]}")
    except Exception as e:
        print(f"Error handling list_active_connections {client_addr}: {e}")


def send_ack(ack_type, header, file_name, message, aes_key, chunk_count, client):
    try:
        ack_message = ''
        if ack_type == "message":
            ack_message = f"ACK:{message}".encode()
        elif ack_type == "file":
            ack_message = f"ACK:{header}-{file_name}".encode()
        elif ack_type == "chunk":
            ack_message = f"ACK for chunk count: {chunk_count}".encode()
        encrypted_ack = encrypt_data(ack_message, aes_key)
        client.sendall(encrypted_ack)
        return
    except Exception as e:
        print(f"Error handling ack: {e}")


def logging(type, hostname, message, log_file, file_name):
    # Log message
    log_entry = ''
    if type == "message":
        log_entry = f"{datetime.now()} - {hostname}:> {message}\n"
    elif type == "file":
        log_entry = f"{datetime.now()} - Received file: {file_name}\n"
    with open(log_file, "a") as f:
        f.write(log_entry)


while True:
    client_socket, client_addr = server_socket.accept()
    print(f"New connection from {client_addr}")
    client_thread = threading.Thread(target=handle_client, args=(client_socket, client_addr))
    client_thread.start()
