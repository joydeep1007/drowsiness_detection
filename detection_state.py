from dataclasses import dataclass
import time
from typing import Dict, List, Union

@dataclass
class AlertEvent:
    timestamp: float
    ear_value: float
    duration: float

class DetectionState:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        """Reset all state variables to initial values"""
        self.active: bool = False
        self.frame_count: int = 0
        self.last_ear: float = 0.0
        self.alert_active: bool = False
        self.alert_start_time: float = 0.0
        self.drowsy_start_time: float = 0.0  # Time when drowsiness was first detected
        self.alert_history: List[AlertEvent] = []
        self.current_session_start: float = time.time()
        self.stats: Dict[str, Union[int, float, List[float]]] = {
            'total_alerts': 0,
            'average_ear': 0.0,
            'ear_samples': [],
            'session_duration': 0
        }

    def update(self, ear: float, threshold: float) -> bool:
        self.last_ear = ear
        self.stats['ear_samples'].append(ear)
        
        # Keep only last 100 samples for average calculation
        if len(self.stats['ear_samples']) > 100:
            self.stats['ear_samples'] = self.stats['ear_samples'][-100:]
            
        self.stats['average_ear'] = sum(self.stats['ear_samples']) / len(self.stats['ear_samples'])
        
        is_drowsy = ear < threshold
        current_time = time.time()
        
        if is_drowsy:
            # If this is the first frame of drowsiness, record the start time
            if self.frame_count == 0:
                self.drowsy_start_time = current_time
                
            self.frame_count += 1
            
            # Only activate alert if drowsy for at least 2 seconds AND frame threshold is met
            drowsy_duration = current_time - self.drowsy_start_time
            if not self.alert_active and self.frame_count >= 20 and drowsy_duration >= 2.0:
                self.alert_active = True
                self.alert_start_time = current_time
        else:
            if self.alert_active:
                self.alert_history.append(AlertEvent(
                    timestamp=self.alert_start_time,
                    ear_value=self.last_ear,
                    duration=current_time - self.alert_start_time
                ))
                self.stats['total_alerts'] += 1
            self.frame_count = 0
            self.alert_active = False
            
        self.stats['session_duration'] = current_time - self.current_session_start
        return is_drowsy

    def get_statistics(self) -> Dict[str, Union[float, int, bool]]:
        current_time = time.time()
        drowsy_duration = current_time - self.drowsy_start_time if self.frame_count > 0 else 0
        
        return {
            'current_ear': self.last_ear,
            'frame_count': self.frame_count,
            'alert_active': self.alert_active,
            'drowsy_duration': round(drowsy_duration, 1),  # Add drowsy duration to statistics
            'total_alerts': self.stats['total_alerts'],
            'average_ear': self.stats['average_ear'],
            'session_duration': self.stats['session_duration'],
            'alert_frequency': self.stats['total_alerts'] / max(1, self.stats['session_duration'] / 60)  # alerts per minute
        }