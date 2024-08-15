import csv
import os
import random
from datetime import datetime

# Define the CSV files and parameters
traffic_csv_file = "../datasets/training_temp.csv"
encryption_csv_file = "../datasets/encryption_temp.csv"

traffic_parameters = [
    "Timestamp", "Client ID", "Message Type", "Message Size", "encryption Status",
    "File Extension", "File Size", "Number of Messages",
    "Number of Files", "Client IP", "server Response Time", "Connection Duration",
    "Number of Connections", "Data Transfer Volume", "Request Frequency", "Error Rate", "encryption Type",
    "Intruder Flag"
]

encryption_parameters = [
    "Timestamp", "Client ID", "File Name", "encryption Type", "Handshake Time (ms)", "encryption Time (ms)", "Decryption Time (ms)",
    "Connection Duration (sec)", "Client IP", "server Response Time (ms)", "CPU Usage %", "Memory Usage (MB)"
]


# Initialize the CSV files
def initialize_csv_file(file_path, parameters):
    if not os.path.exists(file_path):
        print(f'File does not exist, creating a new file: {file_path}')
        with open(file_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(parameters)


initialize_csv_file(traffic_csv_file, traffic_parameters)
initialize_csv_file(encryption_csv_file, encryption_parameters)


def dataset_log_traffic(data):
    with open(traffic_csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(data)


def dataset_log_encryption(data):
    with open(encryption_csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(data)


def dataset_log_message(client_id, type, message, connection_start_time, client_ip, server_response_time,
                        connection_duration, num_message, num_files, num_connections, data_transfer_volume,
                        request_frequency, error_rate, encryption_type, intruder_status):
    connection_duration = (datetime.now() - connection_start_time).total_seconds()

    type = 1 if type == 'junk' else 0

    # Temporary code for dataset simulation
    client_ip = assigning_ips(client_id)

    dataset_log_traffic([
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        client_id, type, len(message), True,
        "cmd", 0, num_message, num_files, client_ip, server_response_time, connection_duration,
        num_connections, data_transfer_volume, request_frequency, error_rate, encryption_type, intruder_status
    ])


def dataset_log_file(client_id, file_name, file_size, connection_start_time, client_ip, server_response_time,
                     connection_duration, num_message, num_files, num_connections, data_transfer_volume,
                     request_frequency, error_rate, encryption_type, intruder_status):
    connection_duration = (datetime.now() - connection_start_time).total_seconds()

    # Temporary code for dataset simulation
    client_ip = assigning_ips(client_id)

    dataset_log_traffic([
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        client_id, 0, file_size, True,
        file_name.split('.')[1], file_size, num_message, num_files, client_ip, server_response_time,
        connection_duration, num_connections, data_transfer_volume, request_frequency, error_rate,
        encryption_type, intruder_status
    ])


def dataset_log_chunk(client_id, file_name, file_size, connection_start_time, client_ip, server_response_time,
                      connection_duration, num_connections, data_transfer_volume, request_frequency, error_rate,
                      encryption_type, intruder_status):
    connection_duration = (datetime.now() - connection_start_time).total_seconds()

    # Temporary code for dataset simulation
    client_ip = assigning_ips(client_id)

    dataset_log_traffic([
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        client_id, 0, file_size, True,
        file_name.split('.')[1], file_size, 1, 1, client_ip, server_response_time, connection_duration,
        num_connections, data_transfer_volume, request_frequency, error_rate, encryption_type, intruder_status
    ])


def dataset_log_encryption_metrics(client_id, file_name, encryption_type, handshake_time, encryption_time, decryption_time,
                                   connection_start_time, client_ip, server_response_time, cpu_usage, memory_usage):
    connection_duration = (datetime.now() - connection_start_time).total_seconds()

    # Temporary code for dataset simulation
    # client_ip = assigning_ips(client_id)

    file_name = 'cmd' if file_name is None else file_name

    dataset_log_encryption([
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        client_id, file_name, encryption_type, handshake_time, encryption_time, decryption_time,
        connection_duration, client_ip, server_response_time, cpu_usage, memory_usage
    ])


# Temporary code for assigning IP address:
def assigning_ips(client_id):
    client_ips = {
        'client_1': '10.0.2.15',
        'client_2': '10.0.2.16',
        'client_3': '10.0.2.17',
        'client_4': '10.0.2.18',
        'client_5': '10.0.2.19',
        'client_6': '10.0.2.20',
        'client_7': '10.0.2.21',
        'client_8': '10.0.2.22',
        'client_9': '10.0.2.23',
        'client_10': '10.0.2.24'
    }

    random_ips = ('10.0.2.38',
                  '10.0.2.42',
                  '10.0.2.49',
                  '10.0.3.56',
                  '10.0.3.40',
                  '10.1.3.45',
                  '10.1.3.55',
                  '10.0.1.19',
                  '10.0.1.25',
                  '10.1.0.16',
                  '0.0.0.0'
                  )

    client_ip = client_ips.get(client_id, random.choice(random_ips))
    return client_ip
