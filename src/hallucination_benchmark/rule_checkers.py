"""
Rule-based checkers for physical/hardware constraint violations.

These are deterministic checks that can verify computable violations
without relying on an LLM. Used in Phase 4 of the pipeline.
"""

import re


# ============================================================
# Sampling theorem checks
# ============================================================

# Domain-specific signal frequency upper bounds (Hz)
SIGNAL_FREQ_RANGES = {
    "sEMG_gesture":      (20, 500),
    "radar_vital_signs": (1e6, 5e6),    # FMCW beat frequency
    "PPG_BP":            (0.5, 100),
    "throat_mic_speech": (80, 4000),
    "EEG_BCI":           (0.5, 100),
    "IMU_motion":        (0, 200),
}


def _parse_sampling_rate(text: str) -> float | None:
    """Extract sampling rate from protocol text, normalized to Hz."""
    patterns = [
        (r"(\d+(?:\.\d+)?)\s*MSPS", 1e6),
        (r"(\d+(?:\.\d+)?)\s*kSPS", 1e3),
        (r"(\d+(?:\.\d+)?)\s*(?:Hz|SPS)\s*(?:sampling|sample)", 1.0),
        (r"samples?\s*at\s*(\d+(?:\.\d+)?)\s*(?:Hz|kSPS|MSPS)", None),  # unit in match
        (r"sampling\s*rate\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*(Hz|kHz|MHz)", None),
    ]
    for pattern, multiplier in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            val = float(m.group(1))
            if multiplier is not None:
                return val * multiplier
            # Parse unit from match
            unit = m.group(2).lower() if m.lastindex >= 2 else ""
            if "m" in unit:
                return val * 1e6
            elif "k" in unit:
                return val * 1e3
            return val
    return None


def check_sampling_theorem(protocol_text: str, domain: str) -> dict:
    """Check if sampling rate satisfies Nyquist for the domain."""
    rate = _parse_sampling_rate(protocol_text)
    if rate is None:
        return {"check": "sampling_theorem", "result": "no_rate_found", "passed": None}

    if domain not in SIGNAL_FREQ_RANGES:
        return {"check": "sampling_theorem", "result": "unknown_domain", "passed": None}

    _, f_max = SIGNAL_FREQ_RANGES[domain]
    nyquist_min = 2 * f_max

    if rate < nyquist_min:
        return {
            "check": "sampling_theorem",
            "result": "VIOLATION",
            "detail": f"Sampling rate {rate:.0f} Hz < Nyquist minimum {nyquist_min:.0f} Hz for {domain}",
            "passed": False,
        }
    return {"check": "sampling_theorem", "result": "ok", "passed": True}


def check_sensor_bandwidth(protocol_text: str, domain: str) -> dict:
    """Check if sensor bandwidth covers the signal frequency range."""
    bw_match = re.search(
        r"bandwidth\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*(Hz|kHz|MHz)",
        protocol_text, re.IGNORECASE,
    )
    if not bw_match:
        return {"check": "sensor_bandwidth", "result": "no_bw_found", "passed": None}

    val = float(bw_match.group(1))
    unit = bw_match.group(2).lower()
    multiplier = {"hz": 1.0, "khz": 1e3, "mhz": 1e6}[unit]
    bw_hz = val * multiplier

    if domain not in SIGNAL_FREQ_RANGES:
        return {"check": "sensor_bandwidth", "result": "unknown_domain", "passed": None}

    _, f_max = SIGNAL_FREQ_RANGES[domain]
    if bw_hz < f_max:
        return {
            "check": "sensor_bandwidth",
            "result": "VIOLATION",
            "detail": f"Sensor bandwidth {bw_hz:.0f} Hz < signal max freq {f_max:.0f} Hz",
            "passed": False,
        }
    return {"check": "sensor_bandwidth", "result": "ok", "passed": True}


# ============================================================
# Material / EM obstruction checks
# ============================================================

MATERIAL_BLOCKERS = [
    (r"aluminum\s*(?:plate|shield|wall|barrier)", "aluminum plate"),
    (r"metal\s*(?:plate|shield|wall|barrier|case|housing)", "metal barrier"),
    (r"(?:thick|heavy)\s*(?:glass|concrete|steel)", "thick dense material"),
    (r"(?:rain\s*coat|diving\s*mask|neoprene\s*sleeve)", "insulating layer"),
]


def check_material_block(protocol_text: str, domain: str) -> dict:
    """Check for EM/optical material obstruction."""
    has_radar = bool(re.search(r"(?:77|60|24)\s*GHz|mmwave|millimeter", protocol_text, re.I))
    has_ppg = bool(re.search(r"PPG|photoplethysm|optical|LED", protocol_text, re.I))
    has_semg = bool(re.search(r"sEMG|EMG|electrode|muscle", protocol_text, re.I))

    for pattern, name in MATERIAL_BLOCKERS:
        if re.search(pattern, protocol_text, re.I):
            if has_radar:
                return {
                    "check": "material_block",
                    "result": "VIOLATION",
                    "detail": f"mmWave radar signal blocked by {name}",
                    "passed": False,
                }
            if has_ppg and ("metal" in name or "aluminum" in name):
                return {
                    "check": "material_block",
                    "result": "VIOLATION",
                    "detail": f"PPG optical path obstructed by {name}",
                    "passed": False,
                }
            if has_semg and ("insulat" in name.lower() or "neoprene" in name.lower()):
                return {
                    "check": "material_block",
                    "result": "VIOLATION",
                    "detail": f"sEMG signal blocked by {name}",
                    "passed": False,
                }
    return {"check": "material_block", "result": "ok", "passed": True}


# ============================================================
# Anatomical misplacement checks
# ============================================================

ANATOMY_RULES = {
    "sEMG_gesture": {
        "wrong": [r"tibialis", r"gastrocnemius", r"quadriceps", r"patellar", r"soleus", r"hamstring"],
        "msg": "sEMG placed on leg muscles for hand gesture recognition",
    },
    "throat_mic_speech": {
        "wrong": [r"elbow", r"knee", r"olecranon", r"patellar", r"shoulder", r"ankle"],
        "msg": "Throat microphone placed on limb for speech recognition",
    },
    "PPG_BP": {
        "wrong": [r"forehead", r"frontal\s*bone", r"earlobe\s*cartilage", r"elbow", r"knee"],
        "msg": "PPG placed where arterial pulse is weak/absent",
    },
    "EEG_BCI": {
        "wrong": [r"forearm", r"wrist", r"ankle", r"chest"],
        "msg": "EEG electrode placed on body instead of scalp",
    },
}


def check_anatomy(protocol_text: str, domain: str) -> dict:
    """Check for anatomical sensor misplacement."""
    if domain not in ANATOMY_RULES:
        return {"check": "anatomy", "result": "no_rule", "passed": None}

    rule = ANATOMY_RULES[domain]
    for pattern in rule["wrong"]:
        if re.search(pattern, protocol_text, re.I):
            return {
                "check": "anatomy",
                "result": "VIOLATION",
                "detail": rule["msg"],
                "passed": False,
            }
    return {"check": "anatomy", "result": "ok", "passed": True}


# ============================================================
# Accuracy impossibility checks
# ============================================================

def check_accuracy_claims(protocol_text: str) -> dict:
    """Check for physically impossible accuracy claims."""
    issues = []

    # Heart rate accuracy
    hr_match = re.search(r"(\d+\.?\d*)\s*bpm.*?(?:error|accuracy|MAE)", protocol_text, re.I)
    if hr_match:
        hr_acc = float(hr_match.group(1))
        if hr_acc < 0.5:
            issues.append(f"Heart rate accuracy {hr_acc} bpm below single-shot limit (~0.5 bpm)")

    # BP accuracy without calibration
    bp_match = re.search(r"(\d+\.?\d*)\s*mmHg.*?(?:error|accuracy|MAE)", protocol_text, re.I)
    if bp_match:
        bp_acc = float(bp_match.group(1))
        if bp_acc < 1.0 and "calibrat" not in protocol_text.lower():
            issues.append(f"BP accuracy {bp_acc} mmHg without calibration exceeds limits")

    # Gesture classification accuracy
    acc_match = re.search(r"(\d+\.?\d*)%.*?(?:accuracy|classification)", protocol_text, re.I)
    if acc_match:
        acc = float(acc_match.group(1))
        if acc > 99.9:
            issues.append(f"Classification accuracy {acc}% exceeds realistic limits")

    if issues:
        return {"check": "accuracy_claims", "result": "VIOLATION", "details": issues, "passed": False}
    return {"check": "accuracy_claims", "result": "ok", "passed": True}


# ============================================================
# ADC resolution check
# ============================================================

def check_adc_resolution(protocol_text: str, domain: str) -> dict:
    """Check if ADC resolution is sufficient for the signal amplitude."""
    adc_match = re.search(r"(\d+)-bit\s*(?:ADC|resolution)", protocol_text, re.I)
    if not adc_match:
        return {"check": "adc_resolution", "result": "no_adc_found", "passed": None}

    bits = int(adc_match.group(1))
    # Assume 3.3V reference
    vref = 3.3
    step_mv = (vref / (2 ** bits)) * 1000

    # sEMG signals are 50uV-5mV; EEG is 10-100uV
    if domain in ("sEMG_gesture", "EEG_BCI") and bits <= 8:
        return {
            "check": "adc_resolution",
            "result": "VIOLATION",
            "detail": f"{bits}-bit ADC: step = {step_mv:.1f} mV, cannot resolve uV-level signals",
            "passed": False,
        }
    return {"check": "adc_resolution", "result": "ok", "passed": True}


# ============================================================
# Power budget check
# ============================================================

def check_power_budget(protocol_text: str) -> dict:
    """Check for impossible battery/power claims."""
    # CR2032: ~0.93 Wh, ~225 mAh
    if re.search(r"CR2032", protocol_text, re.I):
        # Check for high-power components
        high_power = re.search(
            r"(?:radar|FMCW|77\s*GHz|edge\s*AI|GPU|neural|inference)",
            protocol_text, re.I,
        )
        long_runtime = re.search(r"(?:24|48|72)\s*(?:h|hour|hrs)", protocol_text, re.I)
        if high_power and long_runtime:
            return {
                "check": "power_budget",
                "result": "VIOLATION",
                "detail": "CR2032 coin cell cannot power radar/AI for 24+ hours",
                "passed": False,
            }
    return {"check": "power_budget", "result": "ok", "passed": True}


# ============================================================
# Data bandwidth check
# ============================================================

def check_data_bandwidth(protocol_text: str) -> dict:
    """Check for impossible wireless data streaming claims."""
    has_ble = bool(re.search(r"bluetooth|BLE", protocol_text, re.I))
    has_raw_stream = bool(re.search(
        r"real[\s-]*time\s*(?:stream|transfer|transmit).*?(?:raw|uncompressed|full)",
        protocol_text, re.I,
    ))
    if has_ble and has_raw_stream:
        return {
            "check": "data_bandwidth",
            "result": "VIOLATION",
            "detail": "BLE cannot stream raw uncompressed multi-channel data in real-time",
            "passed": False,
        }
    return {"check": "data_bandwidth", "result": "ok", "passed": True}


# ============================================================
# Master dispatcher
# ============================================================

CHECKER_MAP = {
    "sampling_theorem":      lambda text, dom: check_sampling_theorem(text, dom),
    "sensor_bandwidth":      lambda text, dom: check_sensor_bandwidth(text, dom),
    "ematerial_block":       lambda text, dom: check_material_block(text, dom),
    "optical_obstruction":   lambda text, dom: check_material_block(text, dom),
    "temperature_range":     lambda text, dom: {"check": "temperature_range", "result": "manual", "passed": None},
    "anatomical_misplacement": lambda text, dom: check_anatomy(text, dom),
    "adc_resolution":        lambda text, dom: check_adc_resolution(text, dom),
    "power_budget":          lambda text, dom: check_power_budget(text),
    "data_bandwidth":        lambda text, dom: check_data_bandwidth(text),
    "snr_floor":             lambda text, dom: {"check": "snr_floor", "result": "manual", "passed": None},
    "accuracy_impossibility": lambda text, dom: check_accuracy_claims(text),
    "latency_constraint":    lambda text, dom: {"check": "latency_constraint", "result": "manual", "passed": None},
}


def run_checks(protocol_text: str, domain: str, constraint_key: str) -> list[dict]:
    """Run the rule-based checker for a specific constraint.

    Also runs accuracy_claims as a bonus cross-check.
    """
    results = []

    checker = CHECKER_MAP.get(constraint_key)
    if checker:
        results.append(checker(protocol_text, domain))

    # Bonus: always run accuracy check
    if constraint_key != "accuracy_impossibility":
        acc = check_accuracy_claims(protocol_text)
        if not acc.get("passed", True):
            results.append(acc)

    return results
