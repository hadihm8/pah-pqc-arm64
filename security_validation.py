import os
import ctypes
import pandas as pd
import oqs

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


ASCON_LIBRARY = os.environ.get(
    "ASCON_LIB",
    "./libascon.so"
)

ascon = ctypes.CDLL(ASCON_LIBRARY)

U8 = ctypes.c_ubyte
ULL = ctypes.c_ulonglong
U8_PTR = ctypes.POINTER(U8)

RESULT_FILE = "security_validation_results.csv"


# ============================================================
# ASCON BINDINGS
# ============================================================

ascon.crypto_aead_encrypt.argtypes = [
    U8_PTR,
    ctypes.POINTER(ULL),
    U8_PTR,
    ULL,
    U8_PTR,
    ULL,
    ctypes.c_void_p,
    U8_PTR,
    U8_PTR,
]

ascon.crypto_aead_encrypt.restype = ctypes.c_int

ascon.crypto_aead_decrypt.argtypes = [
    U8_PTR,
    ctypes.POINTER(ULL),
    ctypes.c_void_p,
    U8_PTR,
    ULL,
    U8_PTR,
    ULL,
    U8_PTR,
    U8_PTR,
]

ascon.crypto_aead_decrypt.restype = ctypes.c_int


def byte_array(data):
    return (U8 * len(data)).from_buffer_copy(data)


def ascon_encrypt(key, nonce, plaintext):

    message = byte_array(plaintext)
    key_buf = byte_array(key)
    nonce_buf = byte_array(nonce)

    ciphertext = (
        U8 * (len(plaintext) + 16)
    )()

    ciphertext_len = ULL()

    rc = ascon.crypto_aead_encrypt(
        ciphertext,
        ctypes.byref(ciphertext_len),
        message,
        ULL(len(plaintext)),
        U8_PTR(),
        ULL(0),
        None,
        nonce_buf,
        key_buf,
    )

    if rc != 0:
        raise RuntimeError(
            "Ascon encryption failed"
        )

    return bytes(
        ciphertext[:ciphertext_len.value]
    )


def ascon_decrypt(key, nonce, ciphertext):

    cipher_buf = byte_array(ciphertext)
    key_buf = byte_array(key)
    nonce_buf = byte_array(nonce)

    plaintext = (
        U8 * len(ciphertext)
    )()

    plaintext_len = ULL()

    rc = ascon.crypto_aead_decrypt(
        plaintext,
        ctypes.byref(plaintext_len),
        None,
        cipher_buf,
        ULL(len(ciphertext)),
        U8_PTR(),
        ULL(0),
        nonce_buf,
        key_buf,
    )

    if rc != 0:
        raise ValueError(
            "Ascon authentication failed"
        )

    return bytes(
        plaintext[:plaintext_len.value]
    )


# ============================================================
# HELPERS
# ============================================================

def derive_key(material, length):

    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=length,
        salt=None,
        info=b"PAH-PQC-SECURITY-VALIDATION",
    )

    return hkdf.derive(material)


def flip_byte(data, index=0):

    modified = bytearray(data)

    modified[index] ^= 0x01

    return bytes(modified)


def x25519_secret():

    alice = X25519PrivateKey.generate()
    bob = X25519PrivateKey.generate()

    alice_secret = alice.exchange(
        bob.public_key()
    )

    bob_secret = bob.exchange(
        alice.public_key()
    )

    if alice_secret != bob_secret:
        raise RuntimeError(
            "X25519 shared secret mismatch"
        )

    return alice_secret


def mlkem_session():

    with oqs.KeyEncapsulation(
        "ML-KEM-768"
    ) as receiver:

        public_key = (
            receiver.generate_keypair()
        )

        with oqs.KeyEncapsulation(
            "ML-KEM-768"
        ) as sender:

            ciphertext, sender_secret = (
                sender.encap_secret(
                    public_key
                )
            )

            receiver_secret = (
                receiver.decap_secret(
                    ciphertext
                )
            )

            if (
                sender_secret
                != receiver_secret
            ):
                raise RuntimeError(
                    "ML-KEM valid-session mismatch"
                )

            return (
                receiver,
                ciphertext,
                sender_secret
            )


# ============================================================
# REPLAY PROTECTION
# ============================================================

class ReplayProtector:

    def __init__(self):
        self.seen_packets = set()

    def verify_fresh(self, packet_id):

        if packet_id in self.seen_packets:
            raise ValueError(
                "Replay detected"
            )

        self.seen_packets.add(
            packet_id
        )


# ============================================================
# TEST RECORDING
# ============================================================

results = []


def record(
    test_name,
    target,
    expected,
    passed,
    observation,
):

    results.append({
        "test_name": test_name,
        "target": target,
        "expected_behavior": expected,
        "passed": bool(passed),
        "observation": observation,
    })

    status = (
        "PASS"
        if passed
        else "FAIL"
    )

    print(
        f"{status}: "
        f"{test_name} [{target}]"
    )


# ============================================================
# AES-GCM TESTS
# ============================================================

def validate_aes_gcm():

    print("\nAES-256-GCM SECURITY TESTS")

    key = AESGCM.generate_key(
        bit_length=256
    )

    aes = AESGCM(key)

    nonce = os.urandom(12)

    plaintext = (
        b"PAH-PQC authenticated "
        b"security validation"
    )

    ciphertext = aes.encrypt(
        nonce,
        plaintext,
        None
    )

    # Ciphertext tampering
    tampered = flip_byte(
        ciphertext,
        0
    )

    passed = False

    try:
        aes.decrypt(
            nonce,
            tampered,
            None
        )
    except InvalidTag:
        passed = True

    record(
        "Ciphertext tampering",
        "AES-256-GCM",
        "Authentication failure",
        passed,
        "Modified ciphertext rejected",
    )

    # Tag tampering
    tampered_tag = flip_byte(
        ciphertext,
        -1
    )

    passed = False

    try:
        aes.decrypt(
            nonce,
            tampered_tag,
            None
        )
    except InvalidTag:
        passed = True

    record(
        "Authentication-tag manipulation",
        "AES-256-GCM",
        "Authentication failure",
        passed,
        "Modified authentication tag rejected",
    )

    # Wrong key
    wrong_key = AESGCM.generate_key(
        bit_length=256
    )

    wrong_aes = AESGCM(
        wrong_key
    )

    passed = False

    try:
        wrong_aes.decrypt(
            nonce,
            ciphertext,
            None
        )
    except InvalidTag:
        passed = True

    record(
        "Wrong-key attack",
        "AES-256-GCM",
        "Decryption failure",
        passed,
        "Ciphertext rejected under unrelated key",
    )

    # Nonce manipulation
    wrong_nonce = flip_byte(
        nonce,
        0
    )

    passed = False

    try:
        aes.decrypt(
            wrong_nonce,
            ciphertext,
            None
        )
    except InvalidTag:
        passed = True

    record(
        "Nonce manipulation",
        "AES-256-GCM",
        "Authentication failure",
        passed,
        "Modified nonce rejected",
    )


# ============================================================
# ASCON TESTS
# ============================================================

def validate_ascon():

    print("\nASCON-AEAD128 SECURITY TESTS")

    key = os.urandom(16)
    nonce = os.urandom(16)

    plaintext = (
        b"PAH-PQC authenticated "
        b"security validation"
    )

    ciphertext = ascon_encrypt(
        key,
        nonce,
        plaintext
    )

    # Ciphertext tampering
    tampered = flip_byte(
        ciphertext,
        0
    )

    passed = False

    try:
        ascon_decrypt(
            key,
            nonce,
            tampered
        )
    except ValueError:
        passed = True

    record(
        "Ciphertext tampering",
        "Ascon-AEAD128",
        "Authentication failure",
        passed,
        "Modified ciphertext rejected",
    )

    # Tag tampering
    tampered_tag = flip_byte(
        ciphertext,
        -1
    )

    passed = False

    try:
        ascon_decrypt(
            key,
            nonce,
            tampered_tag
        )
    except ValueError:
        passed = True

    record(
        "Authentication-tag manipulation",
        "Ascon-AEAD128",
        "Authentication failure",
        passed,
        "Modified authentication tag rejected",
    )

    # Wrong key
    wrong_key = os.urandom(16)

    passed = False

    try:
        ascon_decrypt(
            wrong_key,
            nonce,
            ciphertext
        )
    except ValueError:
        passed = True

    record(
        "Wrong-key attack",
        "Ascon-AEAD128",
        "Decryption failure",
        passed,
        "Ciphertext rejected under unrelated key",
    )

    # Nonce manipulation
    wrong_nonce = flip_byte(
        nonce,
        0
    )

    passed = False

    try:
        ascon_decrypt(
            key,
            wrong_nonce,
            ciphertext
        )
    except ValueError:
        passed = True

    record(
        "Nonce manipulation",
        "Ascon-AEAD128",
        "Authentication failure",
        passed,
        "Modified nonce rejected",
    )


# ============================================================
# ML-KEM TAMPERING TEST
# ============================================================

def validate_mlkem_tampering():

    print("\nML-KEM-768 TAMPERING TEST")

    with oqs.KeyEncapsulation(
        "ML-KEM-768"
    ) as receiver:

        public_key = (
            receiver.generate_keypair()
        )

        with oqs.KeyEncapsulation(
            "ML-KEM-768"
        ) as sender:

            ciphertext, sender_secret = (
                sender.encap_secret(
                    public_key
                )
            )

            valid_secret = (
                receiver.decap_secret(
                    ciphertext
                )
            )

            if (
                valid_secret
                != sender_secret
            ):
                raise RuntimeError(
                    "Valid ML-KEM session failed"
                )

            tampered_ciphertext = (
                flip_byte(
                    ciphertext,
                    0
                )
            )

            tampered_secret = (
                receiver.decap_secret(
                    tampered_ciphertext
                )
            )

            passed = (
                tampered_secret
                != sender_secret
            )

            record(
                "KEM ciphertext tampering",
                "ML-KEM-768",
                "Different shared secret after tampering",
                passed,
                (
                    "Tampered ciphertext produced "
                    "a non-matching shared secret"
                ),
            )


# ============================================================
# HYBRID COMPONENT-FAILURE TESTS
# ============================================================

def validate_hybrid():

    print("\nHYBRID X25519 + ML-KEM-768 TESTS")

    classical_secret = (
        x25519_secret()
    )

    with oqs.KeyEncapsulation(
        "ML-KEM-768"
    ) as receiver:

        public_key = (
            receiver.generate_keypair()
        )

        with oqs.KeyEncapsulation(
            "ML-KEM-768"
        ) as sender:

            kem_ciphertext, sender_pq = (
                sender.encap_secret(
                    public_key
                )
            )

            receiver_pq = (
                receiver.decap_secret(
                    kem_ciphertext
                )
            )

            if (
                sender_pq
                != receiver_pq
            ):
                raise RuntimeError(
                    "Valid ML-KEM component failed"
                )

            original_material = (
                classical_secret
                + sender_pq
            )

            original_key = derive_key(
                original_material,
                32
            )

            nonce = os.urandom(12)

            plaintext = (
                b"PAH-PQC hybrid "
                b"session validation"
            )

            ciphertext = AESGCM(
                original_key
            ).encrypt(
                nonce,
                plaintext,
                None
            )

            # Replace X25519 component
            unrelated_classical = (
                x25519_secret()
            )

            altered_material = (
                unrelated_classical
                + sender_pq
            )

            altered_key = derive_key(
                altered_material,
                32
            )

            passed = False

            try:
                AESGCM(
                    altered_key
                ).decrypt(
                    nonce,
                    ciphertext,
                    None
                )
            except InvalidTag:
                passed = True

            record(
                "Classical-component substitution",
                "Hybrid M4",
                "Hybrid session-key mismatch",
                passed,
                (
                    "Replacing X25519 component "
                    "caused AEAD authentication failure"
                ),
            )

            # Tamper ML-KEM component
            tampered_kem = flip_byte(
                kem_ciphertext,
                0
            )

            tampered_pq = (
                receiver.decap_secret(
                    tampered_kem
                )
            )

            altered_material = (
                classical_secret
                + tampered_pq
            )

            altered_key = derive_key(
                altered_material,
                32
            )

            passed = False

            try:
                AESGCM(
                    altered_key
                ).decrypt(
                    nonce,
                    ciphertext,
                    None
                )
            except InvalidTag:
                passed = True

            record(
                "Post-quantum-component tampering",
                "Hybrid M4",
                "Hybrid session-key mismatch",
                passed,
                (
                    "Tampered ML-KEM component "
                    "caused AEAD authentication failure"
                ),
            )


# ============================================================
# REPLAY TEST
# ============================================================

def validate_replay_guard():

    print("\nPAH-PQC REPLAY-GUARD TEST")

    protector = ReplayProtector()

    packet_id = os.urandom(16)

    protector.verify_fresh(
        packet_id
    )

    passed = False

    try:
        protector.verify_fresh(
            packet_id
        )
    except ValueError:
        passed = True

    record(
        "Replay attempt",
        "PAH-PQC replay guard",
        "Duplicate packet rejected",
        passed,
        "Repeated packet identifier detected",
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print(
        "PAH-PQC INDEPENDENT SECURITY VALIDATION SUITE"
    )
    print("=" * 80)

    if (
        "ML-KEM-768"
        not in oqs.get_enabled_kem_mechanisms()
    ):
        raise RuntimeError(
            "ML-KEM-768 unavailable"
        )

    validate_aes_gcm()
    validate_ascon()
    validate_mlkem_tampering()
    validate_hybrid()
    validate_replay_guard()

    df = pd.DataFrame(
        results
    )

    df.to_csv(
        RESULT_FILE,
        index=False
    )

    total = len(df)

    passed = int(
        df["passed"].sum()
    )

    failed = (
        total - passed
    )

    print("\n")
    print("=" * 80)
    print(
        "SECURITY VALIDATION SUMMARY"
    )
    print("=" * 80)

    print(
        "Total security tests:",
        total
    )

    print(
        "Passed:",
        passed
    )

    print(
        "Failed:",
        failed
    )

    print(
        "Pass rate:",
        f"{(passed / total) * 100:.2f}%"
    )

    print("\n")

    print(
        df[
            [
                "test_name",
                "target",
                "passed",
                "observation",
            ]
        ].to_string(
            index=False
        )
    )

    if failed != 0:
        raise RuntimeError(
            "One or more security tests failed."
        )

    print("\n")
    print(
        "ALL SECURITY VALIDATION TESTS PASSED"
    )


if __name__ == "__main__":
    main()
