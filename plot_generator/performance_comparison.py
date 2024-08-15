import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset
# data = pd.read_csv('encryption_metrics_vpn_multi.csv')
# data = pd.read_csv('encryption_metrics_5G_multi.csv')
# data = pd.read_csv('encryption_met_diff_machines_vpn.csv')
data = pd.read_csv('debug/encryption_met_diff_machines_5G.csv')

# List of unique clients
clients = data['Client ID'].unique()

# List of metrics to compare
metrics = [
    'encryption Time (ms)',
    'Decryption Time (ms)',
    'CPU Usage %',
    'Memory Usage (MB)',
    'Handshake Time (ms)',
    'Connection Duration (sec)',
    'server Response Time (ms)'
]


# Define a function to create comparison plots for a single client
def create_client_comparison_plot(client_id):
    client_data = data[data['Client ID'] == client_id]
    ecc_data = client_data[client_data['encryption Type'] == 'ECC']
    kyber_data = client_data[client_data['encryption Type'] == 'Kyber']

    plt.figure(figsize=(14, 10))
    for i, metric in enumerate(metrics, 1):
        plt.subplot(3, 3, i)
        plt.bar('ECC', ecc_data[metric].values[0], label='ECC', color='blue')
        plt.bar('Kyber', kyber_data[metric].values[0], label='Kyber', color='green')
        plt.ylabel(metric)
        plt.title(metric)
        plt.legend()

    plt.suptitle(f'Performance Comparison for Client {client_id}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# Create comparison plots for each client
for client_id in clients:
    create_client_comparison_plot(client_id)