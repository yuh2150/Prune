import cv2
import numpy as np
from typing import Dict, List, Any
from traffic_analysis.helpers import draw_text_box

class TrafficDashboard:
    """
    Renders a real-time HUD stats dashboard on the top-left / right overlay of the video frame.
    Displays FPS, Latency, Vehicle Counts, Speed, Traffic Density, Violations, and Plate Logs.
    """
    def __init__(self, config):
        self.config = config

    def draw_hud(
        self,
        frame: np.ndarray,
        fps: float,
        latency_ms: float,
        counts: Dict[str, Any],
        avg_speed: float,
        density_info: Dict[str, Any],
        active_violations: List[str],
        total_violations: int,
        plate_logs: List[str]
    ) -> np.ndarray:
        """
        Render semi-transparent HUD containing all operational analytics.
        """
        h, w, _ = frame.shape

        # Left HUD Panel (Counts, Density, Speed, FPS)
        hud_w, hud_h = 340, 260
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (10 + hud_w, 10 + hud_h), (15, 15, 25), cv2.FILLED)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (10, 10), (10 + hud_w, 10 + hud_h), (0, 200, 255), 2)

        # Header Title
        cv2.putText(frame, "TRAFFIC ANALYTICS SYSTEM", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.line(frame, (20, 42), (10 + hud_w - 10, 42), (0, 200, 255), 1)

        # Stats Lines
        stats_lines = [
            f"FPS: {fps:.1f} | Latency: {latency_ms:.1f} ms",
            f"Active Vehicles: {density_info.get('active_vehicles', 0)}",
            f"Density Level: {density_info.get('level', 'Low')}",
            f"Total Counted: {counts.get('total', 0)} (IN: {counts.get('in', 0)} | OUT: {counts.get('out', 0)})",
            f"Avg Speed: {avg_speed:.1f} km/h",
            f"Violations: {total_violations} total ({len(active_violations)} active)"
        ]

        y_offset = 65
        for line in stats_lines:
            color = (0, 255, 0) if "Density Level: Low" in line else (0, 220, 255)
            if "Violations:" in line and total_violations > 0:
                color = (0, 0, 255)
            cv2.putText(frame, line, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            y_offset += 20

        # Class Breakdown
        class_str = " | ".join([f"{k.capitalize()[:3]}:{v}" for k, v in counts.get('per_class', {}).items()])
        cv2.putText(frame, class_str, (20, y_offset + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

        # Right HUD Panel: Active License Plates
        if plate_logs:
            p_w, p_h = 240, min(180, 40 + len(plate_logs[:5]) * 22)
            px1 = w - p_w - 10
            overlay_p = frame.copy()
            cv2.rectangle(overlay_p, (px1, 10), (w - 10, 10 + p_h), (15, 15, 25), cv2.FILLED)
            cv2.addWeighted(overlay_p, 0.75, frame, 0.25, 0, frame)
            cv2.rectangle(frame, (px1, 10), (w - 10, 10 + p_h), (255, 150, 0), 2)
            cv2.putText(frame, "RECOGNIZED PLATES", (px1 + 10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 150, 0), 2)
            
            py = 55
            for entry in plate_logs[:5]:
                cv2.putText(frame, entry, (px1 + 15, py), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                py += 22

        # Active Violation Warnings (Top Banner Alerts)
        if active_violations:
            alert_text = "ALERT: " + " | ".join(active_violations[:2])
            draw_text_box(
                frame, alert_text, (w // 2 - 180, 40),
                font_scale=0.7, text_color=(255, 255, 255), bg_color=(0, 0, 200), thickness=2
            )

        return frame
