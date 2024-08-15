from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes

AES_KEY_LENGTH = 32  # 32 bytes = 256 bits


def generate_aes_key(shared_key):
    return shared_key[:AES_KEY_LENGTH]


def encrypt_data(plaintext, aes_key):
    nonce = get_random_bytes(16)
    cipher = AES.new(aes_key, AES.MODE_EAX, nonce=nonce)
    ciphertext, tag = cipher.encrypt_and_digest(plaintext)
    # print(f'Nonce: {nonce}, Tag: {tag}')
    return nonce + tag + ciphertext


def decrypt_data(ciphertext, aes_key):
    nonce = ciphertext[:16]
    tag = ciphertext[16:32]
    ciphertext = ciphertext[32:]
    cipher = AES.new(aes_key, AES.MODE_EAX, nonce=nonce)
    # print(f'Nonce: {nonce}, Tag: {tag}')
    try:
        plaintext = cipher.decrypt_and_verify(ciphertext, tag)
        return plaintext
    except ValueError as e:
        print(f"Invalid ciphertext or tag {e}")
        return b""
