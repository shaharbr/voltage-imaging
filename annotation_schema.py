"""
Annotation Data Schema for Voltage Imaging Analysis
Defines the JSON format for storing manual annotations
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from enum import Enum
import json
from pathlib import Path
from datetime import datetime


class ROIQuality(str, Enum):
    """Quality assessment for ROI segmentation"""
    GOOD = "good"           # Clear, well-defined neuron
    UNCERTAIN = "uncertain" # Boundaries unclear
    OVERLAPPING = "overlapping"  # Overlaps with another neuron
    PARTIAL = "partial"     # Only part of neuron visible


class ROIShape(str, Enum):
    """Morphological classification"""
    RING = "ring"           # Ring-like (membrane labeling)
    FILLED = "filled"       # Filled/solid appearance
    IRREGULAR = "irregular" # Irregular shape
    POLYGON = "polygon"     # Polygonal


class SpikeConfidence(str, Enum):
    """Confidence level for spike annotation"""
    CERTAIN = "certain"     # Clear spike
    PROBABLE = "probable"   # Likely a spike
    UNCERTAIN = "uncertain" # Might be artifact


@dataclass
class ROIAnnotation:
    """Single ROI (neuron) annotation"""
    roi_id: int
    # Polygon vertices as list of [x, y] coordinates
    polygon: List[List[float]]
    # Centroid for quick reference
    centroid: List[float]
    # Bounding box [x_min, y_min, x_max, y_max]
    bbox: List[float]
    # Quality and shape classification
    quality: str = ROIQuality.GOOD.value
    shape: str = ROIShape.RING.value
    # Optional metadata
    area_pixels: Optional[float] = None
    notes: str = ""
    # Timestamp
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    modified_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ROIAnnotation':
        return cls(**data)


@dataclass
class SpikeAnnotation:
    """Single spike annotation"""
    spike_id: int
    roi_id: int  # Which neuron this spike belongs to
    time_seconds: float  # Time in seconds
    frame_index: int  # Frame number
    amplitude: Optional[float] = None  # dF/F amplitude
    confidence: str = SpikeConfidence.CERTAIN.value
    # Was this from auto-detection or manual?
    source: str = "manual"  # "manual" or "auto_accepted" or "auto_rejected"
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SpikeAnnotation':
        return cls(**data)


@dataclass
class VideoAnnotation:
    """Complete annotation for a single video file"""
    # Video metadata
    video_path: str
    video_name: str
    n_frames: int
    height: int
    width: int
    fps: float

    # Annotations
    rois: List[ROIAnnotation] = field(default_factory=list)
    spikes: List[SpikeAnnotation] = field(default_factory=list)

    # Annotation metadata
    annotator: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    modified_at: str = field(default_factory=lambda: datetime.now().isoformat())
    notes: str = ""

    # Processing parameters used
    processing_params: Dict[str, Any] = field(default_factory=dict)

    def add_roi(self, roi: ROIAnnotation):
        """Add a new ROI annotation"""
        self.rois.append(roi)
        self.modified_at = datetime.now().isoformat()

    def add_spike(self, spike: SpikeAnnotation):
        """Add a new spike annotation"""
        self.spikes.append(spike)
        self.modified_at = datetime.now().isoformat()

    def get_spikes_for_roi(self, roi_id: int) -> List[SpikeAnnotation]:
        """Get all spikes for a specific ROI"""
        return [s for s in self.spikes if s.roi_id == roi_id]

    def get_roi_by_id(self, roi_id: int) -> Optional[ROIAnnotation]:
        """Get ROI by ID"""
        for roi in self.rois:
            if roi.roi_id == roi_id:
                return roi
        return None

    def remove_roi(self, roi_id: int):
        """Remove ROI and its associated spikes"""
        self.rois = [r for r in self.rois if r.roi_id != roi_id]
        self.spikes = [s for s in self.spikes if s.roi_id != roi_id]
        self.modified_at = datetime.now().isoformat()

    def remove_spike(self, spike_id: int):
        """Remove a spike annotation"""
        self.spikes = [s for s in self.spikes if s.spike_id != spike_id]
        self.modified_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'video_path': self.video_path,
            'video_name': self.video_name,
            'n_frames': self.n_frames,
            'height': self.height,
            'width': self.width,
            'fps': self.fps,
            'rois': [r.to_dict() for r in self.rois],
            'spikes': [s.to_dict() for s in self.spikes],
            'annotator': self.annotator,
            'created_at': self.created_at,
            'modified_at': self.modified_at,
            'notes': self.notes,
            'processing_params': self.processing_params
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VideoAnnotation':
        """Load from dictionary"""
        rois = [ROIAnnotation.from_dict(r) for r in data.get('rois', [])]
        spikes = [SpikeAnnotation.from_dict(s) for s in data.get('spikes', [])]

        return cls(
            video_path=data['video_path'],
            video_name=data['video_name'],
            n_frames=data['n_frames'],
            height=data['height'],
            width=data['width'],
            fps=data['fps'],
            rois=rois,
            spikes=spikes,
            annotator=data.get('annotator', ''),
            created_at=data.get('created_at', ''),
            modified_at=data.get('modified_at', ''),
            notes=data.get('notes', ''),
            processing_params=data.get('processing_params', {})
        )

    def save(self, output_path: Optional[str] = None):
        """Save annotations to JSON file"""
        if output_path is None:
            output_path = f"annotations/{self.video_name}_annotations.json"

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

        print(f"Annotations saved to {output_path}")
        return output_path

    @classmethod
    def load(cls, path: str) -> 'VideoAnnotation':
        """Load annotations from JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def summary(self) -> str:
        """Get summary string of annotations"""
        n_spikes_per_roi = {}
        for spike in self.spikes:
            n_spikes_per_roi[spike.roi_id] = n_spikes_per_roi.get(spike.roi_id, 0) + 1

        active_rois = sum(1 for r in self.rois if n_spikes_per_roi.get(r.roi_id, 0) > 0)

        quality_counts = {}
        for roi in self.rois:
            quality_counts[roi.quality] = quality_counts.get(roi.quality, 0) + 1

        confidence_counts = {}
        for spike in self.spikes:
            confidence_counts[spike.confidence] = confidence_counts.get(spike.confidence, 0) + 1

        lines = [
            f"=== Annotation Summary: {self.video_name} ===",
            f"Video: {self.n_frames} frames, {self.height}x{self.width}, {self.fps} fps",
            f"Annotator: {self.annotator}",
            f"",
            f"ROIs: {len(self.rois)} total, {active_rois} with spikes",
            f"  Quality breakdown: {quality_counts}",
            f"",
            f"Spikes: {len(self.spikes)} total",
            f"  Confidence breakdown: {confidence_counts}",
            f"  Mean spikes/ROI: {len(self.spikes)/len(self.rois):.1f}" if self.rois else "",
            f"",
            f"Last modified: {self.modified_at}"
        ]
        return "\n".join(lines)


# Example usage and schema documentation
SCHEMA_EXAMPLE = """
Example annotation JSON structure:
{
  "video_path": "/path/to/video.tif",
  "video_name": "fish2_fov25",
  "n_frames": 5000,
  "height": 512,
  "width": 512,
  "fps": 200,
  "annotator": "expert_1",
  "rois": [
    {
      "roi_id": 0,
      "polygon": [[100, 150], [110, 145], [120, 150], [115, 160], [105, 160]],
      "centroid": [110, 153],
      "bbox": [100, 145, 120, 160],
      "quality": "good",
      "shape": "ring",
      "area_pixels": 150.5,
      "notes": "Clear membrane labeling"
    }
  ],
  "spikes": [
    {
      "spike_id": 0,
      "roi_id": 0,
      "time_seconds": 1.234,
      "frame_index": 246,
      "amplitude": -0.05,
      "confidence": "certain",
      "source": "manual",
      "notes": ""
    }
  ]
}
"""

if __name__ == "__main__":
    print("Annotation Schema for Voltage Imaging")
    print(SCHEMA_EXAMPLE)
