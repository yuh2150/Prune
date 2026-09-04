import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional

@dataclass
class VehicleRecord:
    tracker_id: int
    class_name: str
    license_plate: str = "PENDING"
    current_speed: float = 0.0
    max_speed: float = 0.0
    trajectory: List[Tuple[float, float, int]] = field(default_factory=list)
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    violations: List[str] = field(default_factory=list)

class VehicleDatabase:
    """
    Centralized database storing active and historical records for all tracked vehicles.
    """
    def __init__(self):
        self.records: Dict[int, VehicleRecord] = {}

    def update_vehicle(
        self,
        tracker_id: int,
        class_name: str,
        centroid: Tuple[float, float, int],
        speed: float = 0.0,
        license_plate: Optional[str] = None,
        violation: Optional[str] = None
    ) -> VehicleRecord:
        """Update or insert vehicle record."""
        now = time.time()
        
        if tracker_id not in self.records:
            self.records[tracker_id] = VehicleRecord(
                tracker_id=tracker_id,
                class_name=class_name,
                first_seen=now,
                last_seen=now
            )

        record = self.records[tracker_id]
        record.last_seen = now
        record.current_speed = speed
        record.max_speed = max(record.max_speed, speed)
        record.trajectory.append(centroid)

        if license_plate and record.license_plate == "PENDING":
            record.license_plate = license_plate

        if violation and violation not in record.violations:
            record.violations.append(violation)

        return record

    def get_record(self, tracker_id: int) -> Optional[VehicleRecord]:
        return self.records.get(tracker_id)

    def get_all_records(self) -> Dict[int, VehicleRecord]:
        return self.records.copy()

    def get_total_violations(self) -> int:
        return sum(len(r.violations) for r in self.records.values())

    def get_average_speed(self) -> float:
        active_speeds = [r.current_speed for r in self.records.values() if r.current_speed > 0]
        if active_speeds:
            return round(float(sum(active_speeds) / len(active_speeds)), 1)
        return 0.0
