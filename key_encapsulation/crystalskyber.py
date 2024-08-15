from kyber import Kyber512


def generate_kyber_keypair():
    pk, sk = Kyber512.keygen()
    return pk, sk


def encapsulate_kyber(public_key):
    c, shared_secret = Kyber512.enc(public_key)
    return c, shared_secret


def decapsulate_kyber(ciphertext, secret_key):
    shared_secret = Kyber512.dec(ciphertext, secret_key)
    return shared_secret
