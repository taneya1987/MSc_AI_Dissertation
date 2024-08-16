# MSc_AI_Dissertation
# Secured Communication Schemes for UAVs in 5G:  Lattice-Based Post-Quantum Encryption (CRYSTALS-Kyber) and IDS

## Abstract

This project explores secure communication between UAVs and ground stations using AES encryption, combined with Elliptic Curve Cryptography (ECC) and CRYSTALS-Kyber key encapsulation, within a 5G environment. Addressing ECC's vulnerability to quantum attacks, CRYSTALS-Kyber is proposed as a quantum-resistant alternative. The system integrates Artificial Intelligence (AI) for intrusion detection, aiming for minimal computational demands. Performance evaluations of ECC and CRYSTALS-Kyber in VPN and 5G environments showed similar metrics, with CRYSTALS-Kyber providing quantum attack protection. AI-based Intrusion Detection System (IDS) experiments highlighted Logistic Regression and XGBoost as top performers, with future work focusing on drone deployment and blockchain integration.

![image](https://github.com/user-attachments/assets/920bba9e-bc1b-4d97-9c03-5cbd99a367ea)

## Encryption

This project implements Advanced Encryption Standard (AES) for securing data transmission between UAVs and ground stations. AES is used in EAX mode to provide both encryption and integrity protection. The encryption module is designed to be lightweight, ensuring minimal computational overhead, making it suitable for real-time UAV communication in a 5G environment.

## Key Encapsulation Mechanism (KEM)

The project incorporates two Key Encapsulation Mechanisms (KEMs) to establish secure communication: Elliptic Curve Cryptography (ECC) and CRYSTALS-Kyber. ECC is a widely-used, efficient cryptographic method, but it is vulnerable to quantum attacks. To counter this, CRYSTALS-Kyber, a quantum-resistant KEM, is implemented. The system dynamically selects between ECC and CRYSTALS-Kyber based on the security needs, ensuring robust encryption with low computational demands.

## Performance Evaluation of KEM

The performance of both ECC and CRYSTALS-Kyber KEMs was rigorously evaluated in VPN and 5G environments. Metrics such as encryption time, decryption time, handshake time, and server response time were compared. Results showed minimal differences, with both methods performing within a 100 ms margin. CRYSTALS-Kyber not only matched ECC's performance but also provided enhanced security against quantum attacks, making it a future-proof choice for secure UAV communications.

<img width="554" alt="image" src="https://github.com/user-attachments/assets/4b8818b7-9c5c-4e92-8771-f3fa10f3b3e2">


## Intrusion Detection System (IDS)

The project integrates an Intrusion Detection System (IDS) to enhance the security of communications between UAVs and ground stations. Using Artificial Intelligence (AI), the IDS is designed to detect potential intruders attempting unauthorized access. The system processes network parameters and data flows to identify abnormal patterns, providing real-time protection against intrusion attempts. Various AI algorithms were tested and implemented to ensure the IDS is both accurate and efficient, even in environments with varying client-to-intruder ratios.

## Performance Evaluation of IDS

The IDS was evaluated across multiple experiments with datasets of varying sizes and client-to-intruder ratios. Initial experiments with 26,000 records showed that Logistic Regression achieved high accuracy (89.50%) and a strong Area Under Curve (AUC) score (0.97) in detecting intruders within a 70-30 client-intruder ratio. Further testing with larger datasets (200,000 records) and different ratios demonstrated that XGBoost performed exceptionally well, achieving an accuracy of 97.33% and an AUC of 0.94 in the most challenging 93-7 client-intruder ratio scenario. The system's robust performance indicates its potential effectiveness in real-world deployments for intrusion detection in secure UAV communications.

## Below steps provide the usage of 

1- Checkout repository.

2- Install below packages:

	pip install tqdm psutil pycryptodome cryptography

3- Update server and client files with correct IP Addresses. Update port as well if the mentioned one is in use.

4- Run server first and verify that its listening on the listed port. Run clients and verify that the connection is established.

5- Select Key Encapsulation Technique (ECC/Kyber).

6- Command to send the file:

	file:client_files/img1.jpg

