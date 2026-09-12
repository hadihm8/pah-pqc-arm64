import os
import csv
import time
import random
import statistics
import ctypes
import platform
from pathlib import Path

import oqs

from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes


# ============================================================
# PAH-PQC ARM64 PILOT BENCHMARK
# ============================================================

PAYLOAD_SIZE = 1024          # 1 KB
WARMUP_RUNS = 5
MEASURED_RUNS = 10

MODES = ["M1", "M2", "M3", "M4", "M5"]

RAW_FILE = "pilot_results_raw.csv"
SUMMARY_FILE = "pilot_results_summary.csv"


# ============================================================
# ASCON-C WRAPPER
# ============================================================

ASCON_LIBRARY = os.environ.get(
    "ASCON_LIB",
    "./libascon.so"
)

ascon = ctypes.CDLL(ASCON_LIBRARY)

U8 = ctypes.c_ubyte
ULL = ctypes.c_ulonglong

U8_PTR = ctypes.POINTER(U8)


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


def byte_array(data: bytes):
    return (U8 * len(data)).from_buffer_copy(data)


def ascon_encrypt(key: bytes, nonce: bytes, plaintext: bytes) -> bytes:

    if len(key) != 16:
        raise ValueError("Ascon-AEAD128 requires a 16-byte key.")

    if len(nonce) != 16:
        raise ValueError("Ascon-AEAD128 requires a 16-byte nonce.")

    message = byte_array(plaintext)
    key_buf = byte_array(key)
    nonce_buf = byte_array(nonce)

    # ciphertext + 16-byte authentication tag
    ciphertext = (U8 * (len(plaintext) + 16))()
    ciphertext_len = ULL()

    null_ad = U8_PTR()

    result = ascon.crypto_aead_encrypt(
        ciphertext,
        ctypes.byref(ciphertext_len),
        message,
        ULL(len(plaintext)),
        null_ad,
        ULL(0),
        None,
        nonce_buf,
        key_buf,
    )

    if result != 0:
        raise RuntimeError("Ascon encryption failed.")

    return bytes(ciphertext[:ciphertext_len.value])


def ascon_decrypt(key: bytes, nonce: bytes, ciphertext: bytes) -> bytes:

    cipher_buf = byte_array(ciphertext)
    key_buf = byte_array(key)
    nonce_buf = byte_array(nonce)

    plaintext = (U8 * len(ciphertext))()
    plaintext_len = ULL()

    null_ad = U8_PTR()

    result = ascon.crypto_aead_decrypt(
        plaintext,
        ctypes.byref(plaintext_len),
        None,
        cipher_buf,
        ULL(len(ciphertext)),
        null_ad,
        ULL(0),
        nonce_buf,
        key_buf,
    )

    if result != 0:
        raise RuntimeError("Ascon authentication/decryption failed.")

    return bytes(plaintext[:plaintext_len.value])


# ============================================================
# KEY DERIVATION
# ============================================================

def derive_key(shared_material: bytes, length: int) -> bytes:

    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=length,
        salt=None,
        info=b"PAH-PQC-ARM64-SESSION",
    )

    return hkdf.derive(shared_material)


# ============================================================
# X25519 KEY ESTABLISHMENT
# ============================================================

def x25519_key_establishment() -> bytes:

    sender_private = X25519PrivateKey.generate()
    receiver_private = X25519PrivateKey.generate()

    sender_public = sender_private.public_key()
    receiver_public = receiver_private.public_key()

    sender_secret = sender_private.exchange(receiver_public)
    receiver_secret = receiver_private.exchange(sender_public)

    if sender_secret != receiver_secret:
        raise RuntimeError(
            "X25519 shared-secret verification failed."
        )

    return sender_secret


# ============================================================
# ML-KEM-768 KEY ESTABLISHMENT
# ============================================================

def mlkem768_key_establishment() -> bytes:

    with oqs.KeyEncapsulation("ML-KEM-768") as client:
        with oqs.KeyEncapsulation("ML-KEM-768") as server:

            public_key = client.generate_keypair()

            ciphertext, server_secret = (
                server.encap_secret(public_key)
            )

            client_secret = client.decap_secret(
                ciphertext
            )

            if client_secret != server_secret:
                raise RuntimeError(
                    "ML-KEM-768 shared-secret verification failed."
                )

            return client_secret


# ============================================================
# COMMUNICATION OVERHEAD
# ============================================================

def communication_overhead(mode: str) -> int:

    # X25519:
    # two 32-byte public keys = 64 bytes
    #
    # ML-KEM-768:
    # public key = 1184 bytes
    # ciphertext = 1088 bytes
    #
    # AES-GCM:
    # nonce = 12 bytes
    # authentication tag = 16 bytes
    #
    # Ascon-AEAD128:
    # nonce = 16 bytes
    # authentication tag = 16 bytes

    overheads = {
        "M1": 64 + 12 + 16,
        "M2": 1184 + 1088 + 12 + 16,
        "M3": 1184 + 1088 + 16 + 16,
        "M4": 64 + 1184 + 1088 + 12 + 16,
        "M5": 64 + 1184 + 1088 + 16 + 16,
    }

    return overheads[mode]


# ============================================================
# MODEL EXECUTION
# ============================================================

def run_model(mode: str, payload: bytes):

    # --------------------------------------------------------
    # KEY ESTABLISHMENT
    # --------------------------------------------------------

    key_start = time.perf_counter_ns()

    if mode == "M1":

        shared_material = x25519_key_establishment()

    elif mode in ("M2", "M3"):

        shared_material = mlkem768_key_establishment()

    elif mode in ("M4", "M5"):

        classical_secret = x25519_key_establishment()

        pq_secret = mlkem768_key_establishment()

        shared_material = (
            classical_secret +
            pq_secret
        )

    else:
        raise ValueError(
            f"Unknown mode: {mode}"
        )

    if mode in ("M1", "M2", "M4"):
        session_key = derive_key(
            shared_material,
            32
        )
    else:
        session_key = derive_key(
            shared_material,
            16
        )

    key_end = time.perf_counter_ns()

    # --------------------------------------------------------
    # AUTHENTICATED ENCRYPTION
    # --------------------------------------------------------

    if mode in ("M1", "M2", "M4"):

        nonce = os.urandom(12)
        aes = AESGCM(session_key)

        enc_start = time.perf_counter_ns()

        ciphertext = aes.encrypt(
            nonce,
            payload,
            None
        )

        enc_end = time.perf_counter_ns()

        dec_start = time.perf_counter_ns()

        recovered = aes.decrypt(
            nonce,
            ciphertext,
            None
        )

        dec_end = time.perf_counter_ns()

    else:

        nonce = os.urandom(16)

        enc_start = time.perf_counter_ns()

        ciphertext = ascon_encrypt(
            session_key,
            nonce,
            payload
        )

        enc_end = time.perf_counter_ns()

        dec_start = time.perf_counter_ns()

        recovered = ascon_decrypt(
            session_key,
            nonce,
            ciphertext
        )

        dec_end = time.perf_counter_ns()

    if recovered != payload:
        raise RuntimeError(
            f"{mode}: recovered plaintext mismatch."
        )

    key_ms = (
        key_end - key_start
    ) / 1_000_000

    enc_ms = (
        enc_end - enc_start
    ) / 1_000_000

    dec_ms = (
        dec_end - dec_start
    ) / 1_000_000

    total_ms = (
        key_ms +
        enc_ms +
        dec_ms
    )

    encryption_seconds = (
        enc_end - enc_start
    ) / 1_000_000_000

    throughput_mbps = (
        PAYLOAD_SIZE / 1_000_000
    ) / encryption_seconds

    overhead = communication_overhead(
        mode
    )

    return {
        "key_ms": key_ms,
        "encryption_ms": enc_ms,
        "decryption_ms": dec_ms,
        "total_ms": total_ms,
        "throughput_MBps": throughput_mbps,
        "overhead_bytes": overhead,
    }


# ============================================================
# WARM-UP
# ============================================================

def warm_up(payload: bytes):

    print("\nStarting warm-up...")

    for warmup in range(
        1,
        WARMUP_RUNS + 1
    ):

        order = MODES.copy()
        random.shuffle(order)

        for mode in order:
            run_model(
                mode,
                payload
            )

        print(
            f"Warm-up {warmup}/{WARMUP_RUNS} completed."
        )


# ============================================================
# MEASURED BENCHMARK
# ============================================================

def measured_benchmark(payload: bytes):

    rows = []

    print("\nStarting measured runs...")

    for repetition in range(
        1,
        MEASURED_RUNS + 1
    ):

        order = MODES.copy()
        random.shuffle(order)

        print(
            f"\nRun {repetition}: "
            + " -> ".join(order)
        )

        for mode in order:

            result = run_model(
                mode,
                payload
            )

            row = {
                "run": repetition,
                "mode": mode,
                "payload_bytes": PAYLOAD_SIZE,
                **result,
            }

            rows.append(row)

            print(
                f"{mode}: "
                f"Total={result['total_ms']:.4f} ms | "
                f"Throughput="
                f"{result['throughput_MBps']:.2f} MB/s"
            )

    return rows


# ============================================================
# CSV EXPORT
# ============================================================

def save_raw_results(rows):

    fieldnames = [
        "run",
        "mode",
        "payload_bytes",
        "key_ms",
        "encryption_ms",
        "decryption_ms",
        "total_ms",
        "throughput_MBps",
        "overhead_bytes",
    ]

    with open(
        RAW_FILE,
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)


def save_summary(rows):

    summary_rows = []

    for mode in MODES:

        mode_rows = [
            row
            for row in rows
            if row["mode"] == mode
        ]

        totals = [
            row["total_ms"]
            for row in mode_rows
        ]

        key_times = [
            row["key_ms"]
            for row in mode_rows
        ]

        encryption_times = [
            row["encryption_ms"]
            for row in mode_rows
        ]

        decryption_times = [
            row["decryption_ms"]
            for row in mode_rows
        ]

        throughputs = [
            row["throughput_MBps"]
            for row in mode_rows
        ]

        summary_rows.append({
            "mode": mode,
            "payload_bytes": PAYLOAD_SIZE,

            "mean_key_ms":
                statistics.mean(key_times),

            "mean_encryption_ms":
                statistics.mean(encryption_times),

            "mean_decryption_ms":
                statistics.mean(decryption_times),

            "mean_total_ms":
                statistics.mean(totals),

            "sd_total_ms":
                statistics.stdev(totals),

            "mean_throughput_MBps":
                statistics.mean(throughputs),

            "overhead_bytes":
                communication_overhead(mode),
        })

    fieldnames = [
        "mode",
        "payload_bytes",
        "mean_key_ms",
        "mean_encryption_ms",
        "mean_decryption_ms",
        "mean_total_ms",
        "sd_total_ms",
        "mean_throughput_MBps",
        "overhead_bytes",
    ]

    with open(
        SUMMARY_FILE,
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    return summary_rows


# ============================================================
# DISPLAY SUMMARY
# ============================================================

def print_summary(summary_rows):

    print("\n")
    print("=" * 74)
    print("PAH-PQC ARM64 PILOT RESULTS")
    print("=" * 74)

    for row in summary_rows:

        print(
            f"{row['mode']} | "
            f"Total={row['mean_total_ms']:.4f} ms | "
            f"SD={row['sd_total_ms']:.4f} | "
            f"Throughput="
            f"{row['mean_throughput_MBps']:.2f} MB/s | "
            f"Overhead="
            f"{row['overhead_bytes']} B"
        )

    print("=" * 74)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 74)
    print("PAH-PQC ARM64 PILOT BENCHMARK")
    print("=" * 74)

    print(
        "Architecture:",
        platform.machine()
    )

    print(
        "Platform:",
        platform.platform()
    )

    print(
        "Payload:",
        PAYLOAD_SIZE,
        "bytes"
    )

    print(
        "Warm-up runs:",
        WARMUP_RUNS
    )

    print(
        "Measured runs:",
        MEASURED_RUNS
    )

    enabled_kems = (
        oqs.get_enabled_kem_mechanisms()
    )

    if "ML-KEM-768" not in enabled_kems:
        raise RuntimeError(
            "ML-KEM-768 is not enabled in liboqs."
        )

    payload = os.urandom(
        PAYLOAD_SIZE
    )

    warm_up(payload)

    rows = measured_benchmark(
        payload
    )

    save_raw_results(
        rows
    )

    summary_rows = save_summary(
        rows
    )

    print_summary(
        summary_rows
    )

    print(
        "\nRaw results saved to:",
        RAW_FILE
    )

    print(
        "Summary saved to:",
        SUMMARY_FILE
    )


if __name__ == "__main__":
    main()
