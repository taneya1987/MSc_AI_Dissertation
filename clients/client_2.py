import os
import socket
import time
from datetime import datetime

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from encryption.aes_encryption import generate_aes_key, encrypt_data, decrypt_data
from key_encapsulation.crystalskyber import (generate_kyber_keypair, decapsulate_kyber)
from key_encapsulation.ecc import generate_private_key, get_public_key, get_shared_key

SERVER_HOST = "10.7.32.11"  # Enter the Server IP Address
SERVER_PORT = 8000
BUFFER_SIZE = 4096
CHUNK_SIZE = 1024  # Adjust this value as needed


def read_file(file_path):
    with open(file_path, "rb") as f:
        file_data = f.read()
    return file_data


# Create a socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((SERVER_HOST, SERVER_PORT))
print(f"Connected to server at {SERVER_HOST}:{SERVER_PORT}")

# Ask the user which key encapsulation technique to use
key_encapsulation = input("Which key encapsulation technique do you want to use? (ECC/Kyber): ").strip().lower()

if key_encapsulation.lower() == "ecc":
    # Generate ECC key pair
    client_private_key = generate_private_key()
    client_public_key = get_public_key(client_private_key)

    # Perform ECDH key exchange
    client_socket.sendall(client_public_key.public_bytes(
        encoding=serialization.Encoding.X962,
        format=serialization.PublicFormat.UncompressedPoint
    ))
    server_public_key = ec.EllipticCurvePublicKey.from_encoded_point(client_public_key.curve,
                                                                     client_socket.recv(BUFFER_SIZE))
    shared_key = get_shared_key(client_private_key, server_public_key)

    print(f'ECC shared key: {shared_key}')

elif key_encapsulation.lower() == "kyber":
    # Generate Kyber key pair
    client_public_key, client_secret_key = generate_kyber_keypair()

    # Send public key to server and receive server's ciphertext
    client_socket.sendall(client_public_key)
    server_ciphertext = client_socket.recv(BUFFER_SIZE)

    # Decapsulate to get shared secret
    shared_key = decapsulate_kyber(server_ciphertext, client_secret_key)

    print(f'Kyber shared key: {shared_key}')

else:
    print("Invalid choice. Exiting.")
    client_socket.close()
    exit()

# Derive AES key from shared secret
aes_key = generate_aes_key(shared_key)
print(f"Derived AES key: {aes_key.hex()}")

# Get the client's local IP address
client_ip = client_socket.getsockname()[0]

# Send client hostname
hostname = os.path.basename(__file__).split('.py')[0]
encrypted_hostname_and_ip = encrypt_data(f"{hostname}:{client_ip}".encode(), aes_key)
client_socket.sendall(encrypted_hostname_and_ip)
print(f"Sent hostname: {hostname} and local IP: {client_ip}")

connection_start_time = datetime.now()
client_addr = client_socket.getsockname()


def measure_latency():
    start_time = time.time()
    client_socket.sendall(encrypt_data(b'ping', aes_key))
    client_socket.recv(BUFFER_SIZE)
    end_time = time.time()
    return (end_time - start_time) * 1000  # milliseconds


def receive_ack():
    try:
        encrypted_ack = client_socket.recv(BUFFER_SIZE)
        ack_message = decrypt_data(encrypted_ack, aes_key).decode()
        if ack_message.startswith("ACK"):
            print(f"Received {ack_message}")
        else:
            print(f"ACK not received and instead received: '{ack_message}'")
    except Exception as e:
        print(f"Exception in receiving ack: {e}")


while True:
    # Send data to server
    user_input = input("Enter data to send (or 'exit' to quit, 'close' to close connection): ")
    if user_input:
        if user_input.lower() == "close" or user_input == "exit":
            client_socket.sendall(encrypt_data(b"close", aes_key))
            print("Connection closed by client.")
            break
        elif user_input.lower() == "ping":
            latency = measure_latency()
            print(f"Latency: {latency:.2f} ms")
        elif user_input.lower() == "server shutdown" or user_input.lower() == "ss":
            confirm = input("Are you sure you want to shutdown the server? (y/n): ")
            if confirm.lower() == "y":
                client_socket.sendall(encrypt_data(b"server shutdown", aes_key))
                print(f"Server shutdown by client: {hostname}:{SERVER_PORT}")
                break
        elif user_input.lower().startswith("file:"):
            file_path = user_input[5:]
            try:
                file_data = read_file(file_path)
                file_name = os.path.basename(file_path)
                file_size = len(file_data)
                encrypted_file_info = encrypt_data(f"FILE:{file_name}:{file_size}".encode(), aes_key)
                client_socket.sendall(encrypted_file_info)
                chunks = [file_data[i:i + CHUNK_SIZE] for i in range(0, file_size, CHUNK_SIZE)]
                encryption_start_time = datetime.now()
                count = 0
                for chunk in chunks:
                    count += 1
                    print(f"chunk count: {count}")
                    encrypted_chunk = encrypt_data(chunk, aes_key)
                    client_socket.sendall(encrypted_chunk)
                    receive_ack()
                encryption_time = (datetime.now() - encryption_start_time).total_seconds() * 1000  # milliseconds
                print(
                    f"Encryption time: {encryption_time:.2f} ms with key encapsulation technique as '{key_encapsulation.upper()}'")

                print(f"Total chunk count: {count}")

                encrypted_time = encrypt_data(str(encryption_time).encode(), aes_key)
                print(f'After encryption of encrypted_time: {encrypted_time}')
                client_socket.sendall(encrypted_time)
                receive_ack()
            except Exception as e:
                print(f"Error sending file: {e}")
        else:
            encrypted_data = encrypt_data(user_input.encode(), aes_key)
            # print(f"Encrypted message: {encrypted_data.hex()}")
            client_socket.sendall(encrypted_data)
            # print('After sendall in else')
            receive_ack()
            # print('After receive ack in else')

client_socket.close()
print("Connection closed.")
