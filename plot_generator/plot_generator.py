import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from holoviews.plotting.bokeh.styles import font_size


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
    bar2 = ax.bar(index + bar_width, [aggregated_data['Kyber'][metric] for metric in metrics], bar_width, label='Kyber',
                  color='darkorange')

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

    bar1 = ax.bar(index, [aggregated_data['ECC'][metric] for metric in filtered_metrics], bar_width, label='ECC',
                  color='orange')
    bar2 = ax.bar(index + bar_width, [aggregated_data['Kyber'][metric] for metric in filtered_metrics], bar_width,
                  label='Kyber', color='darkorange')

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
            'ecc': [24.449, 24.449, 24.449, 24.449, 24.449],
            'kyber': [18.324, 18.324, 18.324, 18.324, 18.324]
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
            'ecc': [23.973, 24.453, 24.453, 24.453, 24.453],
            'kyber': [30.914, 30.914, 30.914, 30.914, 30.914]
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


def overall_results():
    import pandas as pd

    # Create a DataFrame for the IDS performance comparison
    data = {
        'Model': ['Random Forest', 'Logistic Regression', 'Support Vector Machine',
                  'Naive Bayes', 'XGBoost', 'LightGBM', 'Neural Network'],
        'Experiment 1 Accuracy (%)': [54.16, 89.50, 47.38, 85.92, 75.42, 54.03, 86.94],
        'Experiment 1 AUC': [0.75, 0.97, 0.19, 0.79, 0.70, 0.67, 0.72],
        'Experiment 2 Accuracy (%)': [49.26, 85.13, 68.40, 95.72, 80.48, 49.26, 95.35],
        'Experiment 2 AUC': [0.73, 0.97, 0.18, 0.86, 0.78, 0.73, 0.86],
        'Experiment 3 Accuracy (%)': [86.96, 86.55, 86.54, 79.73, 87.82, 87.85, 79.59],
        'Experiment 3 AUC': [0.71, 0.84, 0.79, 0.62, 0.68, 0.74, 0.84],
        'Experiment 4 Accuracy (%)': [97.44, 97.13, 71.98, 87.82, 97.33, 97.33, 93.87],
        'Experiment 4 AUC': [0.83, 0.85, 0.54, 0.62, 0.94, 0.73, 0.79]
    }

    # # Create a DataFrame with the IDS summary
    # data = {
    #     'Model': ['Random Forest', 'Logistic Regression', 'Support Vector Machine', 'Naive Bayes', 'XGBoost',
    #               'LightGBM', 'Neural Network'],
    #     'Accuracy (%)': [54.16, 89.50, 47.38, 85.92, 75.42, 54.03, 86.94],
    #     'AUC': [0.75, 0.97, 0.19, 0.79, 0.70, 0.67, 0.72],
    #     'Client Detection (TNR) (%)': [47.49, 83.97, 73.75, 97.60, 81.16, 47.49, 99.20],
    #     'Intruder Detection (TPR) (%)': [65.96, 99.29, 0.71, 65.25, 65.25, 65.60, 65.25],
    #     'Observations': [
    #         'Moderate performance, high false positive rate',
    #         'High performance, stable, minimal incorrect detections',
    #         'Poor performance, significant overfitting',
    #         'Good performance, high client detection rate',
    #         'Moderate performance, some instability',
    #         'Balanced detection rates, significant incorrect detections',
    #         'Strong performance, high client detection rate'
    #     ]
    # }

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(data)

    print(df)

    # Display the DataFrame to the user
    # import ace_tools as tools
    # tools.display_dataframe_to_user(name="IDS Performance Comparison Table", dataframe=df)

    from IPython.display import display
    # Display the DataFrame
    display(df)

    # Display the DataFrame as a formatted table with lines using tabulate
    from tabulate import tabulate
    table_string = tabulate(df, headers='keys', tablefmt='grid')
    print(table_string)


def ieee_paper():

    # Data for ECC vs CRYSTALS-Kyber (VPN vs 5G)
    parameters = ['Handshake Time', 'Encryption Time', 'Decryption Time', 'Connection Duration', 'Server Response Time',
                  'CPU Usage', 'Memory Usage']
    ecc_vpn = [5.2, 82162, 82175, 430, 82188, 0.04, 24.358]
    kyber_vpn = [10.71, 81281, 81292, 350, 81313, 0.06, 30.914]
    ecc_5g = [5.31, 108445, 108455, 417, 108475, 0.04, 24.449]
    kyber_5g = [10.55, 110118, 110129, 460, 110150, 0.08, 18.324]

    # Graph 1: Comparative Handshake Time (VPN vs 5G) for ECC and CRYSTALS-Kyber
    plt.figure(figsize=(10, 6))
    plt.plot(['VPN', '5G'], [ecc_vpn[0], ecc_5g[0]], marker='o', label='ECC')
    plt.plot(['VPN', '5G'], [kyber_vpn[0], kyber_5g[0]], marker='o', label='CRYSTALS-Kyber')
    # plt.title('Comparative Handshake Time (VPN vs 5G) for ECC and CRYSTALS-Kyber')
    plt.xlabel('Environment')
    plt.ylabel('Handshake Time (ms)')
    plt.legend()
    plt.grid(True)
    plt.savefig('handshake_vpn_5g.png')
    plt.show()

    # Graph 2: Encryption and Decryption Times (VPN vs 5G) for ECC and CRYSTALS-Kyber
    plt.figure(figsize=(10, 6))
    plt.plot(['VPN', '5G'], [ecc_vpn[1], ecc_5g[1]], marker='o', label='ECC Encryption Time')
    plt.plot(['VPN', '5G'], [kyber_vpn[1], kyber_5g[1]], marker='o', label='CRYSTALS-Kyber Encryption Time')
    plt.plot(['VPN', '5G'], [ecc_vpn[2], ecc_5g[2]], marker='o', label='ECC Decryption Time')
    plt.plot(['VPN', '5G'], [kyber_vpn[2], kyber_5g[2]], marker='o', label='CRYSTALS-Kyber Decryption Time')
    # plt.title('Encryption and Decryption Times (VPN vs 5G) for ECC and CRYSTALS-Kyber')
    plt.xlabel('Environment')
    plt.ylabel('Time (ms)')
    plt.legend()
    plt.grid(True)
    plt.savefig('enc_des_vpn_5g.png')
    plt.show()

    # Data for IDS Performance (Accuracy and AUC)
    # models = ['Random Forest', 'Logistic Regression', 'SVM', 'Naive Bayes', 'XGBoost', 'LightGBM', 'Neural Network']
    # experiment_1_acc = [54.16, 89.50, 47.38, 85.92, 75.42, 54.03, 86.94]
    # experiment_1_auc = [0.75, 0.97, 0.19, 0.79, 0.70, 0.67, 0.72]
    # experiment_2_acc = [49.26, 85.13, 68.40, 95.72, 80.48, 49.26, 95.35]
    # experiment_2_auc = [0.73, 0.97, 0.18, 0.86, 0.78, 0.73, 0.86]
    # experiment_3_acc = [86.96, 86.55, 86.54, 79.73, 87.82, 87.85, 79.59]
    # experiment_3_auc = [0.71, 0.84, 0.79, 0.62, 0.68, 0.74, 0.84]
    # experiment_4_acc = [97.44, 97.13, 71.98, 87.82, 97.33, 97.33, 93.87]
    # experiment_4_auc = [0.83, 0.85, 0.54, 0.62, 0.94, 0.73, 0.79]

    # Data for IDS Performance (Accuracy and AUC)
    models = ["SVM", "LightGBM", "RF", "NB", "NN", "LR", "XGBoost"]
    experiment_1_acc = [47.38, 54.03, 54.16, 85.92, 86.94, 89.50, 75.42]
    experiment_2_acc = [68.40, 49.26, 49.26, 95.72, 95.35, 85.13, 80.48]
    experiment_3_acc = [86.54, 87.85, 86.96, 79.73, 79.59, 86.55, 87.82]
    experiment_4_acc = [71.98, 97.33, 97.44, 87.82, 93.87, 97.13, 97.33]

    experiment_1_auc = [0.19, 0.67, 0.75, 0.79, 0.72, 0.97, 0.70]
    experiment_2_auc = [0.18, 0.73, 0.73, 0.86, 0.86, 0.97, 0.78]
    experiment_3_auc = [0.79, 0.74, 0.71, 0.62, 0.84, 0.84, 0.68]
    experiment_4_auc = [0.54, 0.73, 0.83, 0.62, 0.79, 0.85, 0.94]

    # Displaying the data labels on each point
    for i, txt in enumerate(experiment_1_acc):
        plt.text(i, experiment_1_acc[i], f'{txt:.2f}', ha='center', va='bottom')
    for i, txt in enumerate(experiment_2_acc):
        plt.text(i, experiment_2_acc[i], f'{txt:.2f}', ha='center', va='bottom')
    for i, txt in enumerate(experiment_3_acc):
        plt.text(i, experiment_3_acc[i], f'{txt:.2f}', ha='center', va='bottom')
    for i, txt in enumerate(experiment_4_acc):
        plt.text(i, experiment_4_acc[i], f'{txt:.2f}', ha='center', va='bottom')

    for i, txt in enumerate(experiment_1_auc):
        plt.text(i, experiment_1_auc[i], f'{txt:.2f}', ha='center', va='bottom')
    for i, txt in enumerate(experiment_2_auc):
        plt.text(i, experiment_2_auc[i], f'{txt:.2f}', ha='center', va='bottom')
    for i, txt in enumerate(experiment_3_auc):
        plt.text(i, experiment_3_auc[i], f'{txt:.2f}', ha='center', va='bottom')
    for i, txt in enumerate(experiment_4_auc):
        plt.text(i, experiment_4_auc[i], f'{txt:.2f}', ha='center', va='bottom')

    # Graph 3: IDS Accuracy Across Different AI Models
    plt.figure(figsize=(10, 6))
    plt.plot(models, experiment_1_acc, marker='o', label='Experiment 1')
    plt.plot(models, experiment_2_acc, marker='o', label='Experiment 2')
    plt.plot(models, experiment_3_acc, marker='o', label='Experiment 3')
    plt.plot(models, experiment_4_acc, marker='o', label='Experiment 4')
    # plt.title('IDS Accuracy Across Different AI Models')
    plt.xlabel('AI Models', fontsize=15, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=15, fontweight='bold')
    plt.legend()
    plt.grid(True)
    plt.savefig('ids_acc.png', bbox_inches='tight')
    plt.show()

    # Graph 4: IDS AUC Across Different AI Models
    plt.figure(figsize=(10, 6))
    plt.plot(models, experiment_1_auc, marker='o', label='Experiment 1')
    plt.plot(models, experiment_2_auc, marker='o', label='Experiment 2')
    plt.plot(models, experiment_3_auc, marker='o', label='Experiment 3')
    plt.plot(models, experiment_4_auc, marker='o', label='Experiment 4')
    # plt.title('IDS AUC Across Different AI Models')
    plt.xlabel('AI Models', fontsize=15, fontweight='bold')
    plt.ylabel('AUC', fontsize=15, fontweight='bold')
    plt.legend()
    plt.grid(True)
    plt.savefig('ids_auc.png', bbox_inches='tight')
    plt.show()


def generate_plots():

    # Data for ECC vs CRYSTALS-Kyber (VPN vs 5G)
    parameters = ['Handshake Time', 'Encryption Time', 'Decryption Time', 'Connection Duration', 'Server Response Time',
                  'CPU Usage', 'Memory Usage']
    ecc_vpn = [5.2, 82162, 82175, 430, 82188, 0.04, 24.358]
    kyber_vpn = [10.71, 81281, 81292, 350, 81313, 0.06, 30.914]
    ecc_5g = [5.31, 108445, 108455, 417, 108475, 0.04, 24.449]
    kyber_5g = [10.55, 110118, 110129, 460, 110150, 0.08, 18.324]

    # Function to generate line graphs with common formatting
    def generate_line_graph(param_index, title_suffix):
        plt.figure(figsize=(10, 6))
        plt.plot(['VPN', '5G'], [ecc_vpn[param_index], ecc_5g[param_index]], marker='o', label='ECC')
        plt.plot(['VPN', '5G'], [kyber_vpn[param_index], kyber_5g[param_index]], marker='o', label='CRYSTALS-Kyber')

        # Add value labels
        for x, y in zip(['VPN', '5G'], [ecc_vpn[param_index], ecc_5g[param_index]]):
            plt.text(x, y, str(y), ha='center', va='bottom')
        for x, y in zip(['VPN', '5G'], [kyber_vpn[param_index], kyber_5g[param_index]]):
            plt.text(x, y, str(y), ha='center', va='top')

        # plt.title(f'{parameters[param_index]} ({title_suffix})')
        plt.xlabel('Environment', fontsize=15, fontweight='bold')
        if param_index <= 3:
            y_label = parameters[param_index] + ' (ms)'
        elif param_index == 6:
            y_label = parameters[param_index] + ' (MB)'
        else:
            y_label = parameters[param_index] + ' (%)'
        plt.ylabel(y_label, fontsize=15, fontweight='bold')  # Adjust y-label for CPU/Memory
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{parameters[param_index].replace(" ", "_").lower()}_vpn_5g.eps', format='eps', bbox_inches='tight')
        plt.show()

    # Generate graphs for Handshake Time and Encryption/Decryption Times
    generate_line_graph(0, 'VPN vs 5G')
    generate_line_graph(1, 'VPN vs 5G')
    generate_line_graph(2, 'VPN vs 5G')
    generate_line_graph(5, 'VPN vs 5G')
    generate_line_graph(6, 'VPN vs 5G')


def ids_table_graph(eval_type):

    # Data for the table
    data_acc = {
        'Model': ['SVM', 'LightGBM', 'Random Forest', 'Naive Bayes', 'Neural Network', 'Logistic Regression',
                  'XGBoost'],
        'Exp 1 Acc (%)': [47.38, 54.03, 54.16, 85.92, 86.94, 89.50, 75.42],
        'Exp 2 Acc (%)': [68.40, 49.26, 49.26, 95.72, 95.35, 85.13, 80.48],
        'Exp 3 Acc (%)': [86.54, 87.85, 86.96, 79.73, 79.59, 86.55, 87.82],
        'Exp 4 Acc (%)': [71.98, 97.33, 97.44, 87.82, 93.87, 97.13, 97.33],
    }

    data_auc = {
        'Model': ['SVM', 'LightGBM', 'Random Forest', 'Naive Bayes', 'Neural Network', 'Logistic Regression',
                  'XGBoost'],
        'Exp 1 AUC': [0.19, 0.67, 0.75, 0.79, 0.72, 0.97, 0.70],
        'Exp 2 AUC': [0.18, 0.73, 0.73, 0.86, 0.86, 0.97, 0.78],
        'Exp 3 AUC': [0.79, 0.74, 0.71, 0.62, 0.84, 0.84, 0.68],
        'Exp 4 AUC': [0.54, 0.73, 0.83, 0.62, 0.79, 0.85, 0.94],
    }

    # Create DataFrame
    if eval_type == 'acc':
        df = pd.DataFrame(data_acc)
    elif eval_type == 'auc':
        df = pd.DataFrame(data_auc)

    # Plotting the graph
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Plot the graph
    for i in range(1, len(df.columns)):
        ax1.plot(df['Model'], df[df.columns[i]], marker='o', label=df.columns[i])
        # for j in range(len(df['Model'])):
        #     ax1.text(j, df[df.columns[i]][j], f'{df[df.columns[i]][j]:.2f}', ha='right', va='bottom')

    ax1.set_xlabel('AI Models', fontsize=15, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)' if eval_type == 'acc' else 'AUC', fontsize=15, fontweight='bold')
    # ax1.set_title('Accuracy Comparison Across Experiments')
    ax1.legend()

    # Adding the table below the plot
    ax1.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='bottom', bbox=[0.0, -0.5, 1.0, 0.3], fontsize='Large')

    plt.subplots_adjust(left=0.044, bottom=0.32)
    plt.tight_layout()
    plt.savefig(f'ids_{eval_type}_table.eps', format='eps', bbox_inches='tight')
    plt.show()

# vpn()
# _5g()
linegraph_5g()
linegraph_vpn()
# ai_ids_ex1()
# ai_ids_ex2()
# ai_ids_ex3()
# ai_ids_ex4()
# overall_results()
# ieee_paper()
# generate_plots()
# ids_table_graph('acc')
# ids_table_graph('auc')