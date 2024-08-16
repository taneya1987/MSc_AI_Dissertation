import matplotlib.pyplot as plt
import numpy as np


def vpn():
    # Define the metrics
    metrics = ['encryption Time (ms)', 'Decryption Time (ms)', 'CPU Usage %',
               'Memory Usage (MB)', 'Handshake Time (ms)', 'Connection Duration (sec)',
               'server Response Time (ms)']

    # Data from the images (assumed values)
    data = {
        'ECC': {
            'encryption Time (ms)': [600, 500, 500, 500, 500],
            'Decryption Time (ms)': [600, 500, 500, 500, 500],
            'CPU Usage %': [0.05, 0.10, 0.05, 0.10, 0.10],
            'Memory Usage (MB)': [1750, 1750, 1750, 1750, 1750],
            'Handshake Time (ms)': [20, 10, 10, 10, 10],
            'Connection Duration (sec)': [140, 150, 140, 140, 120],
            'server Response Time (ms)': [500, 400, 500, 500, 500]
        },
        'Kyber': {
            'encryption Time (ms)': [600, 500, 500, 500, 500],
            'Decryption Time (ms)': [600, 500, 500, 500, 500],
            'CPU Usage %': [0.05, 0.10, 0.05, 0.10, 0.10],
            'Memory Usage (MB)': [1750, 1750, 1750, 1750, 1750],
            'Handshake Time (ms)': [10, 10, 10, 10, 10],
            'Connection Duration (sec)': [60, 75, 60, 60, 50],
            'server Response Time (ms)': [500, 400, 500, 500, 500]
        }
    }

    # Aggregate data
    aggregated_data = {
        'ECC': {metric: np.mean(values) for metric, values in data['ECC'].items()},
        'Kyber': {metric: np.mean(values) for metric, values in data['Kyber'].items()}
    }

    # Plot the comparison for all metrics
    fig, ax = plt.subplots(figsize=(12, 6))
    index = np.arange(len(metrics))
    bar_width = 0.35

    bar1 = ax.bar(index, [aggregated_data['ECC'][metric] for metric in metrics], bar_width, label='ECC', color='orange')
    bar2 = ax.bar(index + bar_width, [aggregated_data['Kyber'][metric] for metric in metrics], bar_width, label='Kyber', color='darkorange')

    ax.set_xlabel('Metric')
    ax.set_ylabel('Values')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    ax.legend()

    # Add the title below
    # fig.text(0.5, 0.01, 'ECC vs Kyber in VPN Environment', ha='center', fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('ECC_vs_Kyber_VPN_Aggregated.png')
    plt.show()

    # Filtered metrics (removing CPU Usage %, Memory Usage (MB), and Connection Duration (sec))
    filtered_metrics = ['encryption Time (ms)', 'Decryption Time (ms)', 'Handshake Time (ms)',
                        'server Response Time (ms)']

    # Plot the filtered comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    index = np.arange(len(filtered_metrics))
    bar_width = 0.35

    bar1 = ax.bar(index, [aggregated_data['ECC'][metric] for metric in filtered_metrics], bar_width, label='ECC', color='orange')
    bar2 = ax.bar(index + bar_width, [aggregated_data['Kyber'][metric] for metric in filtered_metrics], bar_width, label='Kyber', color='darkorange')

    ax.set_xlabel('Metric')
    ax.set_ylabel('Values')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(filtered_metrics, rotation=45, ha='right')
    ax.legend()

    # Add the title below
    # fig.text(0.5, 0.01, 'ECC vs Kyber in VPN Environment (Filtered)', ha='center', fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('ECC_vs_Kyber_VPN_Filtered.png')
    plt.show()


def _5g():
    # Define the metrics
    metrics = ['encryption Time (ms)', 'Decryption Time (ms)', 'CPU Usage %',
               'Memory Usage (MB)', 'Handshake Time (ms)', 'Connection Duration (sec)',
               'server Response Time (ms)']

    # Data from the images (assumed values)
    data_5G = {
        'ECC': {
            'encryption Time (ms)': [500, 600, 600, 800, 700],
            'Decryption Time (ms)': [500, 600, 600, 800, 700],
            'CPU Usage %': [0.10, 0.10, 0.10, 0.10, 0.10],
            'Memory Usage (MB)': [1750, 1750, 1750, 1750, 1750],
            'Handshake Time (ms)': [20, 10, 10, 10, 10],
            'Connection Duration (sec)': [80, 80, 70, 70, 60],
            'server Response Time (ms)': [500, 500, 500, 600, 600]
        },
        'Kyber': {
            'encryption Time (ms)': [500, 600, 600, 800, 700],
            'Decryption Time (ms)': [500, 600, 600, 800, 700],
            'CPU Usage %': [0.10, 0.10, 0.10, 0.10, 0.10],
            'Memory Usage (MB)': [1750, 1750, 1750, 1750, 1750],
            'Handshake Time (ms)': [10, 10, 10, 10, 10],
            'Connection Duration (sec)': [70, 70, 60, 60, 50],
            'server Response Time (ms)': [500, 500, 500, 600, 600]
        }
    }

    # Aggregate data
    aggregated_data_5G = {
        'ECC': {metric: np.mean(values) for metric, values in data_5G['ECC'].items()},
        'Kyber': {metric: np.mean(values) for metric, values in data_5G['Kyber'].items()}
    }

    # Plot the comparison for all metrics
    fig, ax = plt.subplots(figsize=(12, 6))
    index = np.arange(len(metrics))
    bar_width = 0.35

    bar1 = ax.bar(index, [aggregated_data_5G['ECC'][metric] for metric in metrics], bar_width, label='ECC',
                  color='lightblue')
    bar2 = ax.bar(index + bar_width, [aggregated_data_5G['Kyber'][metric] for metric in metrics], bar_width,
                  label='Kyber', color='blue')

    ax.set_xlabel('Metric')
    ax.set_ylabel('Values')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    ax.legend()

    # Add the title below
    # fig.text(0.5, 0.01, 'ECC vs Kyber in 5G Environment', ha='center', fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('ECC_vs_Kyber_5G_Aggregated.png')
    plt.show()

    # Filtered metrics (removing CPU Usage %, Memory Usage (MB), and Connection Duration (sec))
    filtered_metrics = ['encryption Time (ms)', 'Decryption Time (ms)', 'Handshake Time (ms)',
                        'server Response Time (ms)']

    # Plot the filtered comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    index = np.arange(len(filtered_metrics))
    bar_width = 0.35

    bar1 = ax.bar(index, [aggregated_data_5G['ECC'][metric] for metric in filtered_metrics], bar_width, label='ECC',
                  color='lightblue')
    bar2 = ax.bar(index + bar_width, [aggregated_data_5G['Kyber'][metric] for metric in filtered_metrics], bar_width,
                  label='Kyber', color='blue')

    ax.set_xlabel('Metric')
    ax.set_ylabel('Values')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(filtered_metrics, rotation=45, ha='right')
    ax.legend()

    # Add the title below
    # fig.text(0.5, 0.01, 'ECC vs Kyber in 5G Environment (Filtered)', ha='center', fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('ECC_vs_Kyber_5G_Filtered.png')
    plt.show()


# Combining all the code into one Python file with the method `linegraph_5g`

def linegraph_5g():

    clients = ['Client 1', 'Client 2', 'Client 3', 'Client 4', 'Client 5']

    # Data for each parameter
    metrics_data = {
        'Handshake Time (ms)': {
            'ecc': [22.076, 1.123, 1.135, 1.132, 1.122],
            'kyber': [10.629, 10.637, 10.576, 10.489, 10.405]
        },
        'encryption Time (ms)': {
            'ecc': [103117, 102393, 111291, 115617, 109805],
            'kyber': [108130, 102386, 112852, 118542, 108864]
        },
        'Decryption Time (ms)': {
            'ecc': [103127, 102407, 111301, 115624, 109818],
            'kyber': [108138, 102395, 112866, 118554, 108874]
        },
        'Connection Duration (sec)': {
            'ecc': [411, 408, 425, 438, 405],
            'kyber': [502, 428, 455, 482, 427]
        },
        'server Response Time (ms)': {
            'ecc': [103146, 102426, 111322, 115646, 109838],
            'kyber': [108160, 102416, 112887, 118576, 108896]
        },
        'CPU Usage': {
            'ecc': [0.0, 0.0, 0.0, 0.1, 0.0],
            'kyber': [0.1, 0.1, 0.0, 0.1, 0.0]
        },
        'Memory Usage (MB)': {
            'ecc': [1833.832, 1834.105, 1834.199, 1834.262, 1834.324],
            'kyber': [1845.273, 1845.273, 1845.336, 1845.367, 1845.43]
        },
    }

    '''
    Handshake Time (ms): Average: 5.31 (ECC), 10.55 (Kyber)
    encryption Time (ms): Average: 108445 (ECC), 110118 (Kyber)
    Decryption Time (ms): Average: 108455 (ECC), 110129 (Kyber)
    Connection Duration (sec): Average: 417 (ECC), 460 (Kyber)
    server Response Time (ms): Average: 108475 (ECC), 110150 (Kyber)
    CPU Usage %: Average: 0.04 (ECC), 0.08 (Kyber)
    Memory Usage (MB): Average: 1834 (ECC), 1845 (Kyber)
    '''

    for metric, data in metrics_data.items():
        plt.figure(figsize=(10, 6))
        plt.plot(clients, data['ecc'], marker='o', linestyle='-', linewidth=2, markersize=8, color='blue', label='ECC')
        plt.plot(clients, data['kyber'], marker='o', linestyle='-', linewidth=2, markersize=8, color='green',
                 label='Kyber')
        plt.xlabel('clients', fontsize=14)
        plt.ylabel(metric, fontsize=14)
        # plt.title(f'{metric} Comparison', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(f'ECC_vs_Kyber_5G_{metric}_linegraph.png')
        plt.show()


def linegraph_vpn():

    clients = ['Client 1', 'Client 2', 'Client 3', 'Client 4', 'Client 5']

    # Data for each parameter in the VPN environment
    metrics_data = {
        'Handshake Time (ms)': {
            'ecc': [21.771, 1.153, 1.143, 1.173, 1.158],
            'kyber': [10.829, 11.064, 10.41, 10.462, 10.821]
        },
        'encryption Time (ms)': {
            'ecc': [87895, 65644, 75451, 86418, 95011],
            'kyber': [87792, 66551, 74836, 88896, 86942]
        },
        'Decryption Time (ms)': {
            'ecc': [87919, 65653, 75459, 86430, 95024],
            'kyber': [87815, 66559, 74846, 88905, 86952]
        },
        'Connection Duration (sec)': {
            'ecc': [438, 403, 414, 467, 425],
            'kyber': [366, 309, 322, 410, 333]
        },
        'server Response Time (ms)': {
            'ecc': [87933, 65664, 75474, 86442, 95038],
            'kyber': [87836, 66578, 74865, 88925, 86974]
        },
        'CPU Usage': {
            'ecc': [0.0, 0.1, 0.0, 0.0, 0.1],
            'kyber': [0.0, 0.1, 0.0, 0.1, 0.1]
        },
        'Memory Usage (MB)': {
            'ecc': [1889.77, 1890.047, 1890.078, 1890.109, 1890.414],
            'kyber': [1876.324, 1876.355, 1876.355, 1876.418, 1876.543]
        },
    }

    '''
    Handshake Time (ms): Average: 5.2 (ECC), 10.71 (Kyber)
    encryption Time (ms): Average: 82,162 (ECC), 81,281 (Kyber)
    Decryption Time (ms): Average: 82,175 (ECC), 81,292 (Kyber)
    Connection Duration (sec): Average: 430 (ECC), 350 (Kyber)
    server Response Time (ms): Average: 82188 (ECC), 81313 (Kyber)
    CPU Usage %: Average: 0.04 (ECC), 0.06 (Kyber)
    Memory Usage (MB): Average: 1890 (ECC), 1876 (Kyber)
    '''


    for metric, data in metrics_data.items():
        plt.figure(figsize=(10, 6))
        plt.plot(clients, data['ecc'], marker='o', linestyle='-', linewidth=2, markersize=8, color='blue', label='ECC')
        plt.plot(clients, data['kyber'], marker='o', linestyle='-', linewidth=2, markersize=8, color='green',
                 label='Kyber')
        plt.xlabel('clients', fontsize=14)
        plt.ylabel(metric, fontsize=14)
        # plt.title(f'{metric} Comparison', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(f'ECC_vs_Kyber_vpn_{metric}_linegraph.png')
        plt.show()


def ai_ids_ex1():

    # Data from the table
    models = [
        "Random Forest", "Logistic Regression", "Support Vector Machine",
        "Naive Bayes", "XGBoost", "LightGBM", "Neural Network"
    ]
    accuracy = [54.16, 89.50, 47.38, 85.92, 75.42, 54.03, 86.94]
    auc = [0.75, 0.97, 0.19, 0.79, 0.70, 0.67, 0.72]

    # Plotting the line graph for Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(models, accuracy, marker='o', linestyle='-', color='b', label='Accuracy (%)')
    plt.xlabel('Model', fontweight='bold')
    plt.xticks(rotation=45)
    plt.ylabel('Accuracy (%)')
    # plt.title('Model Accuracy')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('ai_ids_ex1_acc.png')
    plt.show()

    # Plotting the line graph for AUC
    plt.figure(figsize=(10, 6))
    plt.plot(models, auc, marker='s', linestyle='-', color='g', label='AUC')
    plt.xlabel('Model', fontweight='bold')
    plt.xticks(rotation=45)
    plt.ylabel('AUC')
    # plt.title('Model AUC')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('ai_ids_ex1_auc.png')
    plt.show()


def ai_ids_ex2():

    # Data from the table
    models = [
        "Random Forest", "Logistic Regression", "Support Vector Machine",
        "Naive Bayes", "XGBoost", "LightGBM", "Neural Network"
    ]
    accuracy = [49.26, 85.13, 68.40, 95.72, 80.48, 49.26, 95.35]
    auc = [0.73, 0.97, 0.18, 0.86, 0.78, 0.73, 0.80]

    # Plotting the line graph for Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(models, accuracy, marker='o', linestyle='-', color='b', label='Accuracy (%)')
    plt.xlabel('Model', fontweight='bold')
    plt.xticks(rotation=45)
    plt.ylabel('Accuracy (%)')
    # plt.title('Model Accuracy')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('ai_ids_ex2_acc.png')
    plt.show()

    # Plotting the line graph for AUC
    plt.figure(figsize=(10, 6))
    plt.plot(models, auc, marker='s', linestyle='-', color='g', label='AUC')
    plt.xlabel('Model', fontweight='bold')
    plt.xticks(rotation=45)
    plt.ylabel('AUC')
    # plt.title('Model AUC')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('ai_ids_ex2_auc.png')
    plt.show()


def ai_ids_ex3():

    # Data from the table
    models = [
        "Random Forest", "Logistic Regression", "Support Vector Machine",
        "Naive Bayes", "XGBoost", "LightGBM", "Neural Network"
    ]
    accuracy = [86.96, 86.55, 86.54, 79.73, 87.82, 87.85, 79.59]
    auc = [0.71, 0.84, 0.79, 0.62, 0.68, 0.74, 0.84]

    # Plotting the line graph for Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(models, accuracy, marker='o', linestyle='-', color='b', label='Accuracy (%)')
    plt.xlabel('Model', fontweight='bold')
    plt.xticks(rotation=45)
    plt.ylabel('Accuracy (%)')
    # plt.title('Model Accuracy')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('ai_ids_ex3_acc.png')
    plt.show()

    # Plotting the line graph for AUC
    plt.figure(figsize=(10, 6))
    plt.plot(models, auc, marker='s', linestyle='-', color='g', label='AUC')
    plt.xlabel('Model', fontweight='bold')
    plt.xticks(rotation=45)
    plt.ylabel('AUC')
    # plt.title('Model AUC')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('ai_ids_ex3_auc.png')
    plt.show()


def ai_ids_ex4():

    # Data from the table
    models = [
        "Random Forest", "Logistic Regression", "Support Vector Machine",
        "Naive Bayes", "XGBoost", "LightGBM", "Neural Network"
    ]
    accuracy = [97.44, 97.13, 72, 87.82, 97.33, 97.34, 93.87]
    auc = [0.83, 0.85, 0.54, 0.62, 0.94, 0.73, 0.79]

    # Plotting the line graph for Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(models, accuracy, marker='o', linestyle='-', color='b', label='Accuracy (%)')
    plt.xlabel('Model', fontweight='bold')
    plt.xticks(rotation=45)
    plt.ylabel('Accuracy (%)')
    # plt.title('Model Accuracy')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('ai_ids_ex4_acc.png')
    plt.show()

    # Plotting the line graph for AUC
    plt.figure(figsize=(10, 6))
    plt.plot(models, auc, marker='s', linestyle='-', color='g', label='AUC')
    plt.xlabel('Model', fontweight='bold')
    plt.xticks(rotation=45)
    plt.ylabel('AUC')
    # plt.title('Model AUC')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('ai_ids_ex4_auc.png')
    plt.show()

# vpn()
# _5g()
# linegraph_5g()
# linegraph_vpn()
# ai_ids_ex1()
# ai_ids_ex2()
# ai_ids_ex3()
ai_ids_ex4()