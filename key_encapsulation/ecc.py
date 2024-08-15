from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import ec


def generate_private_key():
    return ec.generate_private_key(ec.SECP256R1(), default_backend())


def get_public_key(private_key):
    return private_key.public_key()


def get_shared_key(private_key, peer_public_key):
    return private_key.exchange(ec.ECDH(), peer_public_key)
