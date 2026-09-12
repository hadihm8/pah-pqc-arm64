import os
import csv
import time
import random
import statistics
import ctypes
import platform
import math

import oqs

from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes


PAYLOAD_SIZES = [
    1024,
    10 * 1024,
    100 * 1024,
    1 * 1024 * 1024,
    5 * 1024 * 1024,
    10 * 1024 * 1024,
    50 * 1024 * 1024,
]

WARMUP_RUNS = 5
MEASURED_RUNS = 30

MODES = ["M1", "M2", "M3", "M4", "M5"]

RAW_FILE = "official_results_raw.csv"
SUMMARY_FILE = "official_results_summary.csv"


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


def ascon_encrypt(key, nonce, plaintext):
    message = byte_array(plaintext)
    key_buf = byte_array(key)
    nonce_buf = byte_array(nonce)

    ciphertext = (U8 * (len(plaintext) + 16))()
    ciphertext_len = ULL()

    result = ascon.crypto_aead_encrypt(
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

    if result != 0:
        raise RuntimeError("Ascon encryption failed")

    return bytes(ciphertext[:ciphertext_len.value])


def ascon_decrypt(key, nonce, ciphertext):
    cipher_buf = byte_array(ciphertext)
    key_buf = byte_array(key)
    nonce_buf = byte_array(nonce)

    plaintext = (U8 * len(ciphertext))()
    plaintext_len = ULL()

    result = ascon.crypto_aead_decrypt(
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

    if result != 0:
        raise RuntimeError("Ascon decryption failed")

    return bytes(plaintext[:plaintext_len.value])


# ============================================================
# KEY DERIVATION
# ============================================================

def derive_key(shared_material, length):
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=length,
        salt=None,
        info=b"PAH-PQC-ARM64-SESSION",
    )
    return hkdf.derive(shared_material)


# ============================================================
# KEY ESTABLISHMENT
# ============================================================

def x25519_key_establishment():
    a_priv = X25519PrivateKey.generate()
    b_priv = X25519PrivateKey.generate()

    a_secret = a_priv.exchange(
        b_priv.public_key()
    )

    b_secret = b_priv.exchange(
        a_priv.public_key()
    )

    if a_secret != b_secret:
        raise RuntimeError("X25519 verification failed")

    return a_secret


def mlkem768_key_establishment():
    with oqs.KeyEncapsulation("ML-KEM-768") as receiver:
        public_key = receiver.generate_keypair()

        with oqs.KeyEncapsulation("ML-KEM-768") as sender:
            ciphertext, sender_secret = sender.encap_secret(
                public_key
            )

            receiver_secret = receiver.decap_secret(
                ciphertext
            )

            if sender_secret != receiver_secret:
                raise RuntimeError(
                    "ML-KEM-768 verification failed"
                )

            return sender_secret


# ============================================================
# OVERHEAD
# ============================================================

def communication_overhead(mode):
    overheads = {
        "M1": 92,
        "M2": 2300,
        "M3": 2304,
        "M4": 2364,
        "M5": 2368,
    }
    return overheads[mode]


# ============================================================
# MODEL EXECUTION
# ============================================================

def run_model(mode, payload):
    key_start = time.perf_counter_ns()

    if mode == "M1":
        shared_material = x25519_key_establishment()

    elif mode in ("M2", "M3"):
        shared_material = mlkem768_key_establishment()

    elif mode in ("M4", "M5"):
        classical_secret = x25519_key_establishment()
        pq_secret = mlkem768_key_establishment()
        shared_material = classical_secret + pq_secret

    else:
        raise ValueError(mode)

    key_length = 32 if mode in ("M1", "M2", "M4") else 16

    session_key = derive_key(
        shared_material,
        key_length
    )

    key_end = time.perf_counter_ns()

    if mode in ("M1", "M2", "M4"):
        nonce = os.urandom(12)
        cipher = AESGCM(session_key)

        enc_start = time.perf_counter_ns()
        ciphertext = cipher.encrypt(
            nonce,
            payload,
            None
        )
        enc_end = time.perf_counter_ns()

        dec_start = time.perf_counter_ns()
        recovered = cipher.decrypt(
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
            f"{mode}: plaintext mismatch"
        )

    key_ms = (key_end - key_start) / 1_000_000
    enc_ms = (enc_end - enc_start) / 1_000_000
    dec_ms = (dec_end - dec_start) / 1_000_000
    total_ms = key_ms + enc_ms + dec_ms

    enc_seconds = (enc_end - enc_start) / 1_000_000_000

    throughput = (
        len(payload) / 1_000_000
    ) / enc_seconds

    return {
        "key_ms": key_ms,
        "encryption_ms": enc_ms,
        "decryption_ms": dec_ms,
        "total_ms": total_ms,
        "throughput_MBps": throughput,
        "overhead_bytes": communication_overhead(mode),
    }


# ============================================================
# STATISTICS
# ============================================================

def percentile(values, p):
    values = sorted(values)

    if len(values) == 1:
        return values[0]

    k = (len(values) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)

    if f == c:
        return values[int(k)]

    return (
        values[f] * (c - k)
        + values[c] * (k - f)
    )


def summarize(values):
    n = len(values)
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    median = statistics.median(values)

    q1 = percentile(values, 0.25)
    q3 = percentile(values, 0.75)
    iqr = q3 - q1

    se = sd / math.sqrt(n)
    ci95 = 1.96 * se

    cv = (sd / mean) * 100 if mean != 0 else 0

    return {
        "mean": mean,
        "sd": sd,
        "median": median,
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "ci95_low": mean - ci95,
        "ci95_high": mean + ci95,
        "cv_percent": cv,
        "min": min(values),
        "max": max(values),
    }


# ============================================================
# EXPERIMENT
# ============================================================

def run_experiment():
    rows = []

    for payload_size in PAYLOAD_SIZES:

        print("\n" + "=" * 80)
        print(
            f"PAYLOAD SIZE: {payload_size} bytes"
        )
        print("=" * 80)

        payload = os.urandom(payload_size)

        print("Warm-up phase...")

        for warmup in range(WARMUP_RUNS):
            order = MODES.copy()
            random.shuffle(order)

            for mode in order:
                run_model(
                    mode,
                    payload
                )

        print("Measured phase...")

        for repetition in range(
            1,
            MEASURED_RUNS + 1
        ):
            order = MODES.copy()
            random.shuffle(order)

            for mode in order:
                result = run_model(
                    mode,
                    payload
                )

                rows.append({
                    "payload_bytes": payload_size,
                    "run": repetition,
                    "mode": mode,
                    **result,
                })

            print(
                f"Completed run "
                f"{repetition}/{MEASURED_RUNS}"
            )

    return rows


# ============================================================
# RAW RESULTS
# ============================================================

def save_raw(rows):
    fieldnames = [
        "payload_bytes",
        "run",
        "mode",
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


# ============================================================
# SUMMARY RESULTS
# ============================================================

def save_summary(rows):
    summary_rows = []

    for payload_size in PAYLOAD_SIZES:
        for mode in MODES:

            subset = [
                r for r in rows
                if r["payload_bytes"] == payload_size
                and r["mode"] == mode
            ]

            total_stats = summarize([
                r["total_ms"]
                for r in subset
            ])

            key_stats = summarize([
                r["key_ms"]
                for r in subset
            ])

            enc_stats = summarize([
                r["encryption_ms"]
                for r in subset
            ])

            dec_stats = summarize([
                r["decryption_ms"]
                for r in subset
            ])

            throughput_stats = summarize([
                r["throughput_MBps"]
                for r in subset
            ])

            summary_rows.append({
                "payload_bytes": payload_size,
                "mode": mode,

                "mean_key_ms": key_stats["mean"],
                "mean_encryption_ms": enc_stats["mean"],
                "mean_decryption_ms": dec_stats["mean"],

                "mean_total_ms": total_stats["mean"],
                "sd_total_ms": total_stats["sd"],
                "median_total_ms": total_stats["median"],
                "q1_total_ms": total_stats["q1"],
                "q3_total_ms": total_stats["q3"],
                "iqr_total_ms": total_stats["iqr"],
                "ci95_low_total_ms": total_stats["ci95_low"],
                "ci95_high_total_ms": total_stats["ci95_high"],
                "cv_total_percent": total_stats["cv_percent"],
                "min_total_ms": total_stats["min"],
                "max_total_ms": total_stats["max"],

                "mean_throughput_MBps":
                    throughput_stats["mean"],

                "sd_throughput_MBps":
                    throughput_stats["sd"],

                "overhead_bytes":
                    communication_overhead(mode),
            })

    fieldnames = list(
        summary_rows[0].keys()
    )

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
# MAIN
# ============================================================

def main():
    print("=" * 80)
    print("PAH-PQC OFFICIAL ARM64 EXPERIMENT")
    print("=" * 80)

    print(
        "Architecture:",
        platform.machine()
    )

    print(
        "Platform:",
        platform.platform()
    )

    print(
        "Payload sizes:",
        PAYLOAD_SIZES
    )

    print(
        "Measured runs per mode-size:",
        MEASURED_RUNS
    )

    print(
        "Expected formal observations:",
        len(PAYLOAD_SIZES)
        * len(MODES)
        * MEASURED_RUNS
    )

    if "ML-KEM-768" not in oqs.get_enabled_kem_mechanisms():
        raise RuntimeError(
            "ML-KEM-768 is unavailable."
        )

    rows = run_experiment()

    expected = (
        len(PAYLOAD_SIZES)
        * len(MODES)
        * MEASURED_RUNS
    )

    if len(rows) != expected:
        raise RuntimeError(
            f"Expected {expected} observations, "
            f"but obtained {len(rows)}."
        )

    save_raw(rows)
    summary_rows = save_summary(rows)

    print("\nExperiment completed successfully.")
    print(
        "Formal observations:",
        len(rows)
    )

    print(
        "Summary rows:",
        len(summary_rows)
    )

    print(
        "Raw results:",
        RAW_FILE
    )

    print(
        "Summary results:",
        SUMMARY_FILE
    )


if __name__ == "__main__":
    main()
