from typing import Dict, List


def _score_vehicle(vehicle: Dict) -> Dict:
    score = 0
    issues: List[str] = []

    mileage = float(vehicle.get("mileage", 0))
    engine_temp = float(vehicle.get("engine_temperature", 0))
    oil_quality = float(vehicle.get("oil_quality", 100))
    brake_wear = float(vehicle.get("brake_wear", 0))
    battery_health = float(vehicle.get("battery_health", 100))
    days_since_service = float(vehicle.get("days_since_last_service", 0))

    if mileage > 150000:
        score += 2
        issues.append("High mileage vehicle")
    elif mileage > 90000:
        score += 1

    if engine_temp > 105:
        score += 3
        issues.append("Engine running above safe temperature")
    elif engine_temp > 95:
        score += 1

    if oil_quality < 35:
        score += 2
        issues.append("Poor oil quality indicates immediate service need")
    elif oil_quality < 55:
        score += 1

    if brake_wear > 80:
        score += 3
        issues.append("Brake system near wear limit")
    elif brake_wear > 60:
        score += 2

    if battery_health < 30:
        score += 2
        issues.append("Battery health is critically low")
    elif battery_health < 50:
        score += 1

    if days_since_service > 220:
        score += 2
        issues.append("Service overdue by schedule")
    elif days_since_service > 150:
        score += 1

    if score >= 9:
        risk_level = "Critical"
    elif score >= 6:
        risk_level = "High"
    elif score >= 3:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    confidence = min(0.98, 0.5 + (score / 14))

    return {
        "vehicle_id": str(vehicle.get("vehicle_id", "unknown")),
        "risk_level": risk_level,
        "key_issues": issues[:4] or ["No immediate critical issues detected"],
        "confidence": round(confidence, 2),
    }


def analyze_risks(vehicles: List[Dict]) -> List[Dict]:
    return [_score_vehicle(v) for v in vehicles]
