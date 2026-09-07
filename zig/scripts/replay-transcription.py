#!/usr/bin/env python3
"""Replay one failed chunk privately, without the daemon, mic, or clipboard."""

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
import struct
import subprocess
import tempfile
import time
import zlib
from pathlib import Path

def main():
    import numpy as np

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "capture",
        nargs="?",
        type=Path,
        help="last-failed directory (default: this instance's XDG state directory)",
    )
    parser.add_argument("--audio", type=Path, help="compare a fixture instead of a saved failure")
    parser.add_argument(
        "--model",
        default="Systran/faster-whisper-small.en",
        help="fixture model; captured settings otherwise win",
    )
    parser.add_argument("--encoder-threads", type=int, default=16, help="fixture threads")
    parser.add_argument("--decoder-threads", type=int, default=8, help="fixture threads")
    parser.add_argument("--token-limit", type=int, default=446, help="fixture token limit")
    parser.add_argument(
        "--output", type=Path, help="new private result directory (default: a temporary directory)"
    )
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "zig-out/bin/voiced-replay",
    )
    args = parser.parse_args()
    if args.capture and args.audio:
        parser.error("Choose a capture directory or --audio")
    # All subprocess output and text-bearing artifacts inherit private creation.
    os.umask(0o077)
    if args.output:
        args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
        output = args.output.resolve()
    else:
        output = Path(tempfile.mkdtemp(prefix="voiced-replay-"))
    print(f"Private replay results: {output}", flush=True)
    if not args.binary.is_file():
        raise RuntimeError("Build the decoder first: cd zig && zig build replay")
    captured = None
    if args.audio:
        audio = read_audio(args.audio)
        settings = {
            "model": args.model,
            "model_encoder_threads": args.encoder_threads,
            "model_decoder_threads": args.decoder_threads,
            "evidence": {"token_limit": args.token_limit},
        }
    else:
        state = os.environ.get("XDG_STATE_HOME", "")
        if not os.path.isabs(state):
            state = str(Path.home() / ".local/state")
        instance = os.environ.get("VOICED_INSTANCE", "")
        capture = (
            args.capture
            or Path(state) / ("voiced-" + instance if instance else "voiced") / "last-failed"
        )
        # The writer exchanges directories under the same stable parent lock.
        # Copy once, release the lock, then perform inference on the snapshot.
        fd = os.open(capture.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            fcntl.flock(fd, fcntl.LOCK_SH)
            # Prefer v2 text; never fall back to an old JSON generation if a
            # present text capture is malformed. Legacy support is offline only.
            suffix = "txt" if (capture / "metadata.txt").exists() else "json"
            for name in ("audio.wav", f"metadata.{suffix}", f"tokens.{suffix}", "generated.txt"):
                (output / name).write_bytes((capture / name).read_bytes())
        finally:
            os.close(fd)
        settings = read_metadata(output / f"metadata.{suffix}")
        captured = read_tokens(output / f"tokens.{suffix}")
        if len(captured) != settings["evidence"]["tokens"]:
            raise ValueError("Capture token count does not match its metadata")
        audio = read_audio(output / "audio.wav")
        if len(audio) != settings["evidence"]["samples"]:
            raise ValueError("Capture sample count does not match its metadata")
    raw = output / "audio.f32"
    raw.write_bytes(audio.astype("<f4", copy=False).tobytes())
    report = {
        "audio_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
        "samples": len(audio),
        "fixture": args.audio is not None,
        "runs": {},
        "comparisons": {},
    }
    results = {}
    paddings = sorted({10, 30, settings.get("model_encoder_padding_seconds", 10)})
    for padding in paddings:
        name = f"zig-{padding}"
        with (output / f"{name}.log").open("x") as log:
            subprocess.run(
                [
                    str(args.binary.resolve()),
                    settings["model"],
                    str(settings["model_encoder_threads"]),
                    str(settings["model_decoder_threads"]),
                    str(padding),
                    str(settings["evidence"]["token_limit"]),
                    str(raw),
                    str(output / f"{name}.json"),
                ],
                check=True,
                stdout=log,
                stderr=log,
                timeout=120,
            )
        result = json.loads((output / f"{name}.json").read_text())
        results[name] = result
        if args.audio is None:
            identity_fields = (
                "model_revision",
                "source_sha256",
                "packed_image_sha256",
                "prompt",
                "suppressed_tokens",
                "suppressed_first_tokens",
            )
            differences = [
                field for field in identity_fields if result["metadata"][field] != settings[field]
            ]
            if differences:
                raise ValueError(f"Replay identity differs from capture: {', '.join(differences)}")
    # Read the same installed, pinned weights, with no downloads or fallback.
    import ctranslate2
    from tokenizers import Tokenizer

    configured_data = os.environ.get("XDG_DATA_HOME", "")
    data_home = (
        Path(configured_data) if os.path.isabs(configured_data) else Path.home() / ".local/share"
    )
    model_path = data_home / "voiced/models" / settings["model"]
    native_metadata = results["zig-30"]["metadata"]
    with (model_path / "model.bin").open("rb") as weights:
        if hashlib.file_digest(weights, "sha256").hexdigest() != native_metadata["source_sha256"]:
            raise ValueError("Reference model checksum does not match Zig")
    reference = ctranslate2.models.Whisper(
        str(model_path),
        device="cpu",
        compute_type="int8",
        intra_threads=settings["model_encoder_threads"],
        inter_threads=1,
    )
    tokenizer = Tokenizer.from_file(str(model_path / "tokenizer.json"))
    for padding in paddings:
        name = f"reference-{padding}"
        if len(audio) < 201 or not np.isfinite(audio).all() or np.max(np.abs(audio)) > 1:
            result = {
                "tokens": [],
                "text": "",
                "token_limit_reached": False,
                "error": "Reference skipped: input is outside the runtime's audio contract",
            }
            write_json(output / f"{name}.json", result)
            results[name] = result
            continue
        features = reference_features(audio, padding)
        started = time.perf_counter_ns()
        generated = reference.generate(
            ctranslate2.StorageView.from_array(features),
            [native_metadata["prompt"]],
            beam_size=1,
            sampling_topk=1,
            sampling_temperature=1,
            length_penalty=1,
            repetition_penalty=1,
            no_repeat_ngram_size=0,
            max_length=settings["evidence"]["token_limit"] + len(native_metadata["prompt"]),
            suppress_blank=True,
            suppress_tokens=native_metadata["suppressed_tokens"],
            return_scores=True,
            return_no_speech_prob=True,
        )[0]
        tokens = generated.sequences_ids[0]
        result = {
            "tokens": tokens,
            "text": tokenizer.decode(tokens, skip_special_tokens=True),
            "elapsed_ns": time.perf_counter_ns() - started,
            "encoder_positions": features.shape[-1] // 2,
            "no_speech_probability": generated.no_speech_prob,
            "score": generated.scores[0],
            "token_limit_reached": len(tokens) >= settings["evidence"]["token_limit"],
            "ctranslate2_version": ctranslate2.__version__,
            "compute_type": reference.compute_type,
            "feature_extractor_version": importlib.metadata.version("faster-whisper"),
        }
        write_json(output / f"{name}.json", result)
        results[name] = result
    for name, result in results.items():
        text_bytes = result["text"].encode()
        native = result.get("metadata")
        limited = native["end"] == "token_limit" if native else result["token_limit_reached"]
        report["runs"][name] = {
            "tokens": len(result["tokens"]),
            "token_limit_reached": limited,
            "compression_ratio": len(text_bytes) / len(zlib.compress(text_bytes)),
            "error": native["error_name"] if native else result.get("error"),
        }
        print(f"{name}: {len(result['tokens'])} tokens, token limit reached={limited}", flush=True)
    for left, right in (
        ("zig-10", "zig-30"),
        ("zig-10", "reference-10"),
        ("zig-30", "reference-30"),
    ):
        if not results[right].get("error"):
            report["comparisons"][f"{left}/{right}"] = compare(
                results[left]["tokens"], results[right]["tokens"]
            )
    if captured is not None:
        padding = settings["model_encoder_padding_seconds"]
        if settings["evidence"]["decoding_available"]:
            report["comparisons"]["captured/replayed"] = compare(
                captured, results[f"zig-{padding}"]["tokens"]
            )
        report["comparisons"]["captured/replayed_error_equal"] = (
            settings["error_name"] == results[f"zig-{padding}"]["metadata"]["error_name"]
        )
        report["build_differences"] = [
            key
            for key in (
                "zig_version",
                "optimize",
                "packed_image_format_version",
                "packed_model_cache_version",
            )
            if settings[key] != native_metadata[key]
        ]
    report["reference_notes"] = (
        "Greedy, same prompt/suppression/token limit and normalized-zero padding; independent NumPy features and CTranslate2 INT8 kernels. CTranslate2 uses one thread count for encoder and decoder. No VAD, segmentation, prompt history, temperature fallback, or text trimming."
    )
    write_json(output / "comparison.json", report)
    print(f"Comparison: {output / 'comparison.json'}")


def read_metadata(path):
    """Read v2 text or legacy v1 JSON into the same typed metadata dictionary."""
    text = read_capture_text(path)
    if path.suffix == ".json":
        metadata = json.loads(text)
        if metadata["format_version"] != 1:
            raise ValueError("Unsupported capture format")
        return metadata

    fields = {}
    for line in text.split("\n")[:-1]:
        key, separator, value = line.partition("=")
        if not separator or not key or key in fields:
            raise ValueError(f"Invalid or duplicate metadata field: {key!r}")
        fields[key] = value
    if fields.get("format_version") != "2":
        raise ValueError("Unsupported capture format")

    integer_fields = {
        "format_version": 32, "session_id": 64, "captured_unix_seconds": -64,
        "packed_image_format_version": 32, "packed_model_cache_version": 32,
        "model_encoder_threads": 32, "model_decoder_threads": 32,
        "model_encoder_padding_seconds": 32, "sample_rate": 32, "beam_size": 32,
    }
    integer_fields.update({"evidence." + key: 32 for key in (
        "chunk_available", "decoding_available", "chunk", "samples", "tokens",
        "token_limit", "encoder_positions", "reserved",
    )})
    integer_fields.update({"evidence." + key: 64 for key in (
        "log_mel_ns", "encoder_ns", "cross_key_values_ns", "decoder_ns",
    )})
    float_fields = ("evidence.no_speech_probability", "evidence.average_log_probability")
    bool_fields = ("contains_activity", "text_decode_complete", "temperature_fallback", "end_available")
    string_fields = (
        "stage", "error_name", "model", "model_revision", "source_sha256",
        "packed_image_sha256", "zig_version", "optimize", "sample_format", "end",
    )
    list_fields = ("prompt", "suppressed_tokens", "suppressed_first_tokens")
    expected = set(integer_fields) | set(float_fields) | set(bool_fields) | set(string_fields) | set(list_fields)
    if fields.keys() != expected:
        raise ValueError(f"Missing or unknown metadata fields: {sorted(fields.keys() ^ expected)}")

    metadata = {}
    for key, bits in integer_fields.items():
        value = fields[key]
        digits = value.removeprefix("-") if bits < 0 else value
        if not digits.isascii() or not digits.isdecimal():
            raise ValueError(f"Invalid metadata integer: {key}")
        number = int(value)
        minimum, maximum = (-(1 << 63), (1 << 63) - 1) if bits < 0 else (0, (1 << bits) - 1)
        if not minimum <= number <= maximum:
            raise ValueError(f"Metadata integer out of range: {key}")
        metadata[key] = number
    for key in float_fields:
        # Preserve inf/nan evidence too; availability is a separate field.
        metadata[key] = float(fields[key])
    for key in bool_fields:
        if fields[key] not in ("true", "false"):
            raise ValueError(f"Invalid metadata boolean: {key}")
        metadata[key] = fields[key] == "true"
    for key in string_fields:
        metadata[key] = unescape_metadata_string(fields[key])
    for key in list_fields:
        metadata[key] = parse_token_list(fields[key])
    if len(metadata["prompt"]) != 2 or any(token > 65535 for token in metadata["prompt"]):
        raise ValueError("Invalid capture prompt")
    if not metadata.pop("end_available"):
        if metadata["end"] != "":
            raise ValueError("Unavailable end must have an empty value")
        metadata["end"] = None
    metadata["evidence"] = {
        key.removeprefix("evidence."): metadata.pop(key)
        for key in list(metadata) if key.startswith("evidence.")
    }
    return metadata


def unescape_metadata_string(value):
    """Decode the writer's byte escapes, not Python or JSON string syntax."""
    encoded = value.encode("ascii")
    output = bytearray()
    escapes = {ord("n"): 10, ord("r"): 13, ord("t"): 9, ord("\\"): 92, ord('"'): 34}
    index = 0
    while index < len(encoded):
        byte = encoded[index]
        index += 1
        if byte != 92:
            if not 32 <= byte <= 126:
                raise ValueError("Unescaped metadata control byte")
            output.append(byte)
            continue
        if index == len(encoded):
            raise ValueError("Truncated metadata escape")
        escape = encoded[index]
        index += 1
        if escape in escapes:
            output.append(escapes[escape])
        elif escape == ord("x"):
            digits = encoded[index:index + 2]
            if len(digits) != 2 or any(byte not in b"0123456789abcdefABCDEF" for byte in digits):
                raise ValueError("Invalid metadata hex escape")
            output.append(int(digits, 16))
            index += 2
        else:
            raise ValueError("Unknown metadata escape")
    return output.decode("utf-8")


def read_tokens(path):
    """Read decimal u32 token IDs; a newline alone represents an empty list."""
    text = read_capture_text(path)
    if path.suffix == ".json":
        tokens = json.loads(text)
        if not isinstance(tokens, list) or any(type(token) is not int or not 0 <= token <= 0xffffffff for token in tokens):
            raise ValueError("Invalid legacy token list")
        return tokens
    return parse_token_list(text)


def parse_token_list(text):
    tokens = []
    for token in text.split():
        if not token.isascii() or not token.isdecimal():
            raise ValueError("Invalid decimal token ID")
        value = int(token)
        if value > 0xffffffff:
            raise ValueError("Token ID exceeds u32")
        tokens.append(value)
    return tokens


def read_capture_text(path):
    with path.open("rb") as source:
        data = source.read(1024 * 1024 + 1)
    if len(data) > 1024 * 1024 or not data.endswith(b"\n"):
        raise ValueError("Oversized or incomplete capture text file")
    return data.decode("utf-8")


def read_audio(path):
    """Read mono 16k WAV without resampling or float32 -> PCM16 conversion."""
    import numpy as np

    data = path.read_bytes()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("Expected a RIFF/WAVE file")
    chunks = {}
    offset = 12
    while offset + 8 <= len(data):
        name, size = struct.unpack_from("<4sI", data, offset)
        offset += 8
        if offset + size > len(data):
            raise ValueError("Truncated WAV chunk")
        chunks[name] = data[offset : offset + size]
        offset += size + (size & 1)
    encoding, channels, rate, _, _, bits = struct.unpack_from("<HHIIHH", chunks[b"fmt "])
    if channels != 1 or rate != 16000:
        raise ValueError("Replay requires mono 16kHz audio; resampling would change the input")
    if (encoding, bits) == (3, 32):
        audio = np.frombuffer(chunks[b"data"], dtype="<f4").copy()
    elif (encoding, bits) == (1, 16):
        audio = np.frombuffer(chunks[b"data"], dtype="<i2").astype(np.float32) / 32768
    else:
        raise ValueError(f"Unsupported WAV encoding {encoding}, {bits} bits")
    if not 1 <= audio.size <= 480000:
        raise ValueError("Expected 1..480000 samples")
    return audio


def reference_features(audio, padding):
    # Independent NumPy implementation of the runtime's feature contract:
    # reflected prefix, zero suffix, floor(N/160) content frames, normalize
    # content, then append normalized zeros and align to eight frames.
    import numpy as np
    from faster_whisper.feature_extractor import FeatureExtractor

    extractor = FeatureExtractor()
    frames = len(audio) // 160
    spectrum = extractor.stft(
        np.pad(audio, (0, 200)),
        400,
        160,
        window=np.hanning(401)[:-1].astype(np.float32),
        return_complex=True,
    ).astype(np.complex64)[..., :frames]
    mel = extractor.mel_filters @ (np.abs(spectrum) ** 2)
    log_mel = np.log10(np.maximum(mel, 1e-10))
    log_mel = (np.maximum(log_mel, log_mel.max() - 8) + 4) / 4
    length = (min(frames + padding * 100, 3000) + 7) // 8 * 8
    return np.ascontiguousarray(
        np.pad(log_mel, ((0, 0), (0, length - frames)))[None], dtype=np.float32
    )


def write_json(path, value):
    with path.open("x") as output:
        json.dump(value, output, indent=2, ensure_ascii=False, allow_nan=False)
        output.write("\n")


def compare(a, b):
    mismatch = next((i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), None)
    if mismatch is None and len(a) != len(b):
        mismatch = min(len(a), len(b))
    return {"tokens_equal": a == b, "first_different_token": mismatch}


if __name__ == "__main__":
    main()
